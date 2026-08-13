// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// INT8 quantization kernels for the WMMA int8 GEMM.
#include <stdexcept>

#include "../hadamard.h"

namespace comfy::hip_backend {

constexpr float kInt8Max = 127.0f;

__forceinline__ __device__ int8_t quant_round(float v, float inv) {
    // rintf is round-half-to-even, matching torch.round.
    int q = static_cast<int>(rintf(v * inv));
    q = q < -128 ? -128 : (q > 127 ? 127 : q);
    return static_cast<int8_t>(q);
}

// One block per row: block-reduce the row absmax, then quantize the row.
__global__ __launch_bounds__(256) void quantize_int8_rowwise_kernel(
    const void* __restrict__ x, int in_code, int8_t* __restrict__ q,
    float* __restrict__ scales, int M, int K) {

    __shared__ float red[256];
    const int row = blockIdx.x;
    const int t = threadIdx.x;

    float lmax = 0.0f;
    for (int j = t; j < K; j += 256)
        lmax = fmaxf(lmax, fabsf(load_in(x, static_cast<int64_t>(row) * K + j, in_code)));

    red[t] = lmax;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (t < s) red[t] = fmaxf(red[t], red[t + s]);
        __syncthreads();
    }

    const float scale = fmaxf(red[0] / kInt8Max, 1e-30f);
    if (t == 0) scales[row] = scale;
    const float inv = 1.0f / scale;

    for (int j = t; j < K; j += 256) {
        const float v = load_in(x, static_cast<int64_t>(row) * K + j, in_code);
        q[static_cast<int64_t>(row) * K + j] = quant_round(v, inv);
    }
}

__global__ void absmax_kernel(const void* __restrict__ x, int in_code,
                              unsigned int* __restrict__ out, int64_t numel) {
    __shared__ float red[256];
    const int t = threadIdx.x;

    float lmax = 0.0f;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + t; i < numel;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x)
        lmax = fmaxf(lmax, fabsf(load_in(x, i, in_code)));

    red[t] = lmax;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (t < s) red[t] = fmaxf(red[t], red[t + s]);
        __syncthreads();
    }
    // Non-negative floats compare identically to their bit patterns as unsigned.
    if (t == 0) atomicMax(out, __float_as_uint(red[0]));
}

__global__ void scale_from_absmax_kernel(const unsigned int* __restrict__ absmax,
                                         float* __restrict__ scale) {
    if (threadIdx.x == 0) *scale = fmaxf(__uint_as_float(*absmax) / kInt8Max, 1e-30f);
}

__global__ void quantize_int8_tensorwise_kernel(
    const void* __restrict__ x, int in_code, int8_t* __restrict__ q,
    const float* __restrict__ scale, int64_t numel) {

    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    q[i] = quant_round(load_in(x, i, in_code), 1.0f / scale[0]);
}

template <typename T>
__forceinline__ __device__ void store_dequant(T* output, int64_t i, float value) {
    output[i] = static_cast<T>(value);
}

template <>
__forceinline__ __device__ void store_dequant<__half>(
    __half* output, int64_t i, float value) {
    output[i] = __float2half(value);
}

template <typename OutputT>
__global__ void dequantize_int8_simple_kernel(
    const int8_t* __restrict__ q, const float* __restrict__ scales,
    OutputT* __restrict__ output, int64_t numel, int64_t inner_dim,
    int scale_mode) {

    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < numel; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t scale_idx = 0;
        if (scale_mode == 1) {
            scale_idx = i;
        } else if (scale_mode == 2) {
            scale_idx = i / inner_dim;
        }
        store_dequant(output, i, static_cast<float>(q[i]) * scales[scale_idx]);
    }
}

// One block per row, rotating up to 256 values at a time. Unlike the fused
// quantizer, this does not stage the whole row, so K is not limited by the
// device's per-block LDS capacity.
template <typename OutputT>
__global__ __launch_bounds__(256) void dequantize_int8_convrot_weight_kernel(
    const int8_t* __restrict__ q, const float* __restrict__ scales,
    OutputT* __restrict__ output, int K, int scale_size, int G) {

    const float h4[4][4] = {
        {1, 1, 1, -1},
        {1, 1, -1, 1},
        {1, -1, 1, 1},
        {-1, 1, 1, 1},
    };
    __shared__ float values[256];

    const int row = blockIdx.x;
    const int t = threadIdx.x;
    const int groups_per_pass = 256 / G;
    const int local_group = t / G;
    const int element = t % G;
    const int local_base = local_group * G;
    const int group_count = K / G;
    const float scale = scales[scale_size == 1 ? 0 : row];
    const float norm = rsqrtf(static_cast<float>(G));

    int stage_count = 0;
    while ((1 << (2 * stage_count)) < G) {
        ++stage_count;
    }

    for (int group_base = 0; group_base < group_count;
         group_base += groups_per_pass) {
        const int group = group_base + local_group;
        const bool active = group < group_count;
        const int64_t col = static_cast<int64_t>(group) * G + element;
        values[t] = active
            ? static_cast<float>(q[static_cast<int64_t>(row) * K + col]) * scale
            : 0.0f;
        __syncthreads();

        for (int stage = 0; stage < stage_count; ++stage) {
            const int stride = 1 << (2 * stage);
            const int digit = (element / stride) & 3;
            const int base = local_base + element - digit * stride;
            const float v0 = values[base];
            const float v1 = values[base + stride];
            const float v2 = values[base + 2 * stride];
            const float v3 = values[base + 3 * stride];
            const float transformed =
                h4[digit][0] * v0 + h4[digit][1] * v1 +
                h4[digit][2] * v2 + h4[digit][3] * v3;
            __syncthreads();
            values[t] = transformed;
            __syncthreads();
        }

        if (active) {
            store_dequant(
                output, static_cast<int64_t>(row) * K + col, values[t] * norm);
        }
        __syncthreads();
    }
}

}  // namespace comfy::hip_backend

namespace comfy::hip_backend {

template <int ACT>
struct LaunchConvrotQuantGlobalManaged {
    const void* x;
    int in_dtype;
    int8_t* qout;
    float* scaleout;
    int M;
    int K;
    hipStream_t stream;
    void* spill_rotated;
    void* spill_partials;

    template <typename RowT>
    void operator()() const {
        launch_convrot_quant_global<RowT, ACT>(
            x, in_dtype, qout, scaleout, M, K,
            static_cast<RowT*>(spill_rotated),
            static_cast<float*>(spill_partials), stream);
    }
};

template <int ACT>
void launch_convrot_quant_global_managed(
    const void* x, int in_dtype, int8_t* qout, float* scaleout, int M, int K, hipStream_t stream,
    void* spill_rotated, void* spill_partials) {
    if (spill_rotated == nullptr || spill_partials == nullptr) {
        throw std::runtime_error("convrot global spill requires caller-provided workspace buffers");
    }

    dispatch_convrot_row_type(
        in_dtype,
        LaunchConvrotQuantGlobalManaged<ACT>{
            x,
            in_dtype,
            qout,
            scaleout,
            M,
            K,
            stream,
            spill_rotated,
            spill_partials});
}

template void launch_convrot_quant_global_managed<kActNone>(
    const void*, int, int8_t*, float*, int, int, hipStream_t, void*, void*);
template void launch_convrot_quant_global_managed<kActGeluTanh>(
    const void*, int, int8_t*, float*, int, int, hipStream_t, void*, void*);
template void launch_convrot_quant_global_managed<kActSwiGLU>(
    const void*, int, int8_t*, float*, int, int, hipStream_t, void*, void*);

}  // namespace comfy::hip_backend

extern "C" void launch_quantize_int8_rowwise_kernel(
    const void* x, int in_code, void* q, void* scales, int M, int K, hipStream_t stream) {

    using namespace comfy::hip_backend;
    if (M == 0) {
        return;  // a zero-block launch is hipErrorInvalidConfiguration
    }
    quantize_int8_rowwise_kernel<<<M, 256, 0, stream>>>(
        x, in_code, static_cast<int8_t*>(q), static_cast<float*>(scales), M, K);
}

// Fused ConvRot rotation + rowwise int8 quantize. act_code: 0 none, 1 gelu tanh,
// 2 swiglu. K is the activated (written) width; swiglu reads a row twice as wide.
extern "C" void launch_quantize_int8_convrot_kernel(
    const void* x, int in_code, void* q, void* scales, void* spill_rotated, void* spill_partials,
    int M, int K, int group_size, int act_code, hipStream_t stream) {

    using namespace comfy::hip_backend;
    check_convrot_group_size(group_size);
    check_convrot_k(K, group_size, in_code, /*int8_quant=*/true);
    check_convrot_act(act_code);
    if (M == 0) {
        return;  // a zero-block launch is hipErrorInvalidConfiguration
    }
    if (act_code == kActGeluTanh) {
        launch_convrot_quant<false, kActGeluTanh>(
            x, in_code, static_cast<int8_t*>(q), static_cast<float*>(scales),
            M, K, group_size, stream, spill_rotated, spill_partials);
    } else if (act_code == kActSwiGLU) {
        launch_convrot_quant<false, kActSwiGLU>(
            x, in_code, static_cast<int8_t*>(q), static_cast<float*>(scales),
            M, K, group_size, stream, spill_rotated, spill_partials);
    } else {
        launch_convrot_quant<false, kActNone>(
            x, in_code, static_cast<int8_t*>(q), static_cast<float*>(scales),
            M, K, group_size, stream, spill_rotated, spill_partials);
    }
}

// scratch is a 4-byte device buffer, pre-zeroed by the caller.
extern "C" void launch_quantize_int8_tensorwise_kernel(
    const void* x, int in_code, void* q, void* scale, void* scratch,
    int64_t numel, hipStream_t stream) {

    using namespace comfy::hip_backend;
    const int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    const int rblocks = blocks < 1024 ? blocks : 1024;

    // An empty input has no absmax and nothing to quantize, but the scale is still
    // an output, so only the zero-block launches are skipped. scratch is zeroed,
    // which the scale kernel clamps.
    if (numel > 0) {
        absmax_kernel<<<rblocks, threads, 0, stream>>>(
            x, in_code, static_cast<unsigned int*>(scratch), numel);
    }
    scale_from_absmax_kernel<<<1, 1, 0, stream>>>(
        static_cast<const unsigned int*>(scratch), static_cast<float*>(scale));
    if (numel > 0) {
        quantize_int8_tensorwise_kernel<<<blocks, threads, 0, stream>>>(
            x, in_code, static_cast<int8_t*>(q), static_cast<const float*>(scale), numel);
    }
}

extern "C" void launch_dequantize_int8_simple_kernel(
    const void* q, const void* scales, void* output, int64_t numel,
    int64_t inner_dim, int scale_mode, int output_dtype_code, hipStream_t stream) {

    using namespace comfy::hip_backend;
    if (numel == 0) {
        return;
    }
    if (inner_dim <= 0) {
        throw std::runtime_error("dequantize_int8_simple: inner_dim must be positive");
    }
    if (scale_mode < 0 || scale_mode > 2) {
        throw std::runtime_error("dequantize_int8_simple: invalid scale mode");
    }

    constexpr int threads = 256;
    const int64_t required_blocks = (numel + threads - 1) / threads;
    const int blocks = static_cast<int>(required_blocks < 4096 ? required_blocks : 4096);

#define CK_LAUNCH_DEQUANT_SIMPLE(OutputT)                                                \
    dequantize_int8_simple_kernel<OutputT><<<blocks, threads, 0, stream>>>(              \
        static_cast<const int8_t*>(q), static_cast<const float*>(scales),                \
        static_cast<OutputT*>(output), numel, inner_dim, scale_mode)

    if (output_dtype_code == 0) {
        CK_LAUNCH_DEQUANT_SIMPLE(float);
    } else if (output_dtype_code == 1) {
        CK_LAUNCH_DEQUANT_SIMPLE(__half);
    } else if (output_dtype_code == 2) {
        CK_LAUNCH_DEQUANT_SIMPLE(__bf16);
    } else {
        throw std::runtime_error("dequantize_int8_simple: unsupported output dtype code");
    }
#undef CK_LAUNCH_DEQUANT_SIMPLE
}

extern "C" void launch_dequantize_int8_convrot_weight_kernel(
    const void* q, const void* scales, void* output, int M, int K,
    int scale_size, int group_size, int output_dtype_code, hipStream_t stream) {

    using namespace comfy::hip_backend;
    check_convrot_group_size(group_size);
    if (K % group_size != 0) {
        throw std::runtime_error(
            "dequantize_int8_convrot_weight: K=" + std::to_string(K) +
            " is not divisible by group_size=" + std::to_string(group_size));
    }
    if (scale_size != 1 && scale_size != M) {
        throw std::runtime_error(
            "dequantize_int8_convrot_weight: scale must be scalar or per-row");
    }
    if (M == 0 || K == 0) {
        return;
    }

#define CK_LAUNCH_DEQUANT_CONVROT(OutputT)                                               \
    dequantize_int8_convrot_weight_kernel<OutputT><<<M, 256, 0, stream>>>(               \
        static_cast<const int8_t*>(q), static_cast<const float*>(scales),                \
        static_cast<OutputT*>(output), K, scale_size, group_size)

    if (output_dtype_code == 0) {
        CK_LAUNCH_DEQUANT_CONVROT(float);
    } else if (output_dtype_code == 1) {
        CK_LAUNCH_DEQUANT_CONVROT(__half);
    } else if (output_dtype_code == 2) {
        CK_LAUNCH_DEQUANT_CONVROT(__bf16);
    } else {
        throw std::runtime_error(
            "dequantize_int8_convrot_weight: unsupported output dtype code");
    }
#undef CK_LAUNCH_DEQUANT_CONVROT
}
