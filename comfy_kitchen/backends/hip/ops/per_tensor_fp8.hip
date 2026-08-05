// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Per-tensor FP8 quantize/dequantize. The OCP encode/decode lives in fp8_utils.h
// so this kernel and stochastic_round_fp8 cannot drift apart.
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "../fp8_utils.h"

namespace comfy::hip_backend {

constexpr int kThreads = 256;

// Elements per thread on the vectorized path: enough for a lane to move a whole
// 16-byte fp8 chunk, so a wave writes 512 contiguous bytes rather than 32.
constexpr int kVecElems = 16;

// Register staging for one thread's run. The alignment is what lets the accesses
// widen to dwordx4.
template <typename T>
struct alignas(16) VecChunk {
    T v[kVecElems];
};

template<typename T>
__forceinline__ __device__ void store_value(void* dst, int64_t idx, float value) {
    reinterpret_cast<T*>(dst)[idx] = static_cast<T>(value);
}

template<>
__forceinline__ __device__ void store_value<__half>(void* dst, int64_t idx, float value) {
    reinterpret_cast<__half*>(dst)[idx] = __float2half(value);
}

template<typename T>
__forceinline__ __device__ T from_float_as(float value) {
    return static_cast<T>(value);
}

template<>
__forceinline__ __device__ __half from_float_as<__half>(float value) {
    return __float2half(value);
}

template<typename InputType>
__global__ void quantize_per_tensor_fp8_kernel(
    const InputType* input,
    const float* scale,
    uint8_t* output,
    int64_t numel,
    int output_dtype_code) {

    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= numel) {
        return;
    }
    const float scaled = to_float(input[idx]) / scale[0];
    output[idx] = pack_fp8(scaled, output_dtype_code);
}

template<typename OutputType>
__global__ void dequantize_per_tensor_fp8_kernel(
    const uint8_t* input,
    const float* scale,
    void* output,
    int64_t numel,
    int input_dtype_code) {

    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= numel) {
        return;
    }
    store_value<OutputType>(output, idx, unpack_fp8(input[idx], input_dtype_code) * scale[0]);
}

// Vectorized counterparts: kVecElems consecutive elements per thread, moved as
// whole 16-byte chunks. The per-element arithmetic is identical to the scalar
// kernels above, so both paths give bit-for-bit the same result.
template<typename InputType>
__global__ void quantize_per_tensor_fp8_vec_kernel(
    const InputType* __restrict__ input,
    const float* __restrict__ scale,
    uint8_t* __restrict__ output,
    int64_t numel,
    int output_dtype_code) {

    const int64_t i0 = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * kVecElems;
    if (i0 >= numel) {
        return;
    }
    const float s = scale[0];

    if (i0 + kVecElems <= numel) {
        const VecChunk<InputType> in = *reinterpret_cast<const VecChunk<InputType>*>(input + i0);
        VecChunk<uint8_t> out;
        #pragma unroll
        for (int i = 0; i < kVecElems; ++i) {
            out.v[i] = pack_fp8(to_float(in.v[i]) / s, output_dtype_code);
        }
        *reinterpret_cast<VecChunk<uint8_t>*>(output + i0) = out;
    } else {
        for (int64_t i = i0; i < numel; ++i) {
            output[i] = pack_fp8(to_float(input[i]) / s, output_dtype_code);
        }
    }
}

template<typename OutputType>
__global__ void dequantize_per_tensor_fp8_vec_kernel(
    const uint8_t* __restrict__ input,
    const float* __restrict__ scale,
    void* __restrict__ output,
    int64_t numel,
    int input_dtype_code) {

    const int64_t i0 = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * kVecElems;
    if (i0 >= numel) {
        return;
    }
    const float s = scale[0];
    OutputType* out_typed = reinterpret_cast<OutputType*>(output);

    if (i0 + kVecElems <= numel) {
        const VecChunk<uint8_t> in = *reinterpret_cast<const VecChunk<uint8_t>*>(input + i0);
        VecChunk<OutputType> out;
        #pragma unroll
        for (int i = 0; i < kVecElems; ++i) {
            out.v[i] = from_float_as<OutputType>(unpack_fp8(in.v[i], input_dtype_code) * s);
        }
        *reinterpret_cast<VecChunk<OutputType>*>(out_typed + i0) = out;
    } else {
        for (int64_t i = i0; i < numel; ++i) {
            store_value<OutputType>(output, i, unpack_fp8(input[i], input_dtype_code) * s);
        }
    }
}

// Gates the vectorized path: a contiguous tensor can still be a mid-buffer view,
// whose base carries the offset and is not 16-byte aligned.
__forceinline__ static bool vec_aligned(const void* p) {
    return (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}

} // namespace comfy::hip_backend

extern "C" {

void launch_quantize_per_tensor_fp8_kernel(
    const void* input,
    const void* scale,
    void* output,
    int64_t numel,
    int input_dtype_code,
    int output_dtype_code,
    hipStream_t stream) {

    if (numel == 0) {
        return;  // a zero-block launch is hipErrorInvalidConfiguration
    }

    using namespace comfy::hip_backend;

    const bool vec = vec_aligned(input) && vec_aligned(output);
    const int64_t per_thread = vec ? kVecElems : 1;
    const int blocks =
        static_cast<int>((numel + per_thread * kThreads - 1) / (per_thread * kThreads));

#define COMFY_LAUNCH_QUANT(T)                                                                     \
    do {                                                                                          \
        if (vec) {                                                                                \
            quantize_per_tensor_fp8_vec_kernel<T><<<blocks, kThreads, 0, stream>>>(               \
                static_cast<const T*>(input), static_cast<const float*>(scale),                   \
                static_cast<uint8_t*>(output), numel, output_dtype_code);                         \
        } else {                                                                                  \
            quantize_per_tensor_fp8_kernel<T><<<blocks, kThreads, 0, stream>>>(                   \
                static_cast<const T*>(input), static_cast<const float*>(scale),                   \
                static_cast<uint8_t*>(output), numel, output_dtype_code);                         \
        }                                                                                         \
    } while (0)

    if (input_dtype_code == 0) {
        COMFY_LAUNCH_QUANT(float);
    } else if (input_dtype_code == 1) {
        COMFY_LAUNCH_QUANT(__half);
    } else if (input_dtype_code == 2) {
        COMFY_LAUNCH_QUANT(__bf16);
    } else {
        throw std::runtime_error("Unsupported input dtype for quantize_per_tensor_fp8");
    }
#undef COMFY_LAUNCH_QUANT
}

void launch_dequantize_per_tensor_fp8_kernel(
    const void* input,
    const void* scale,
    void* output,
    int64_t numel,
    int input_dtype_code,
    int output_dtype_code,
    hipStream_t stream) {

    if (numel == 0) {
        return;  // a zero-block launch is hipErrorInvalidConfiguration
    }

    using namespace comfy::hip_backend;

    const bool vec = vec_aligned(input) && vec_aligned(output);
    const int64_t per_thread = vec ? kVecElems : 1;
    const int blocks =
        static_cast<int>((numel + per_thread * kThreads - 1) / (per_thread * kThreads));

#define COMFY_LAUNCH_DEQUANT(T)                                                                   \
    do {                                                                                          \
        if (vec) {                                                                                \
            dequantize_per_tensor_fp8_vec_kernel<T><<<blocks, kThreads, 0, stream>>>(             \
                static_cast<const uint8_t*>(input), static_cast<const float*>(scale), output,     \
                numel, input_dtype_code);                                                         \
        } else {                                                                                  \
            dequantize_per_tensor_fp8_kernel<T><<<blocks, kThreads, 0, stream>>>(                 \
                static_cast<const uint8_t*>(input), static_cast<const float*>(scale), output,     \
                numel, input_dtype_code);                                                         \
        }                                                                                         \
    } while (0)

    if (output_dtype_code == 0) {
        COMFY_LAUNCH_DEQUANT(float);
    } else if (output_dtype_code == 1) {
        COMFY_LAUNCH_DEQUANT(__half);
    } else if (output_dtype_code == 2) {
        COMFY_LAUNCH_DEQUANT(__bf16);
    } else {
        throw std::runtime_error("Unsupported output dtype for dequantize_per_tensor_fp8");
    }
#undef COMFY_LAUNCH_DEQUANT
}

} // extern "C"
