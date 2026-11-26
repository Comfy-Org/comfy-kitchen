/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "utils.cuh"
#include "float_utils.cuh"
#include "dtype_dispatch.cuh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace comfy {

constexpr int kQMaxKernelThreads = 128;
constexpr int kE4M3Alignment = 16;
constexpr int kStochasticWidth = 16;

namespace {

__forceinline__ __device__ float clamp(float val, float min, float max) {
    return fminf(max, fmaxf(min, val));
}

template<typename InputType, typename OutputType>
__global__ void quantize_fp8_tensor_kernel(
    const InputType *src, 
    OutputType *dst, 
    const float *scale_f, 
    const uint32_t size) {
    
    constexpr float kFP8Max = FP8LimitsTrait<OutputType>::max;
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    idx *= kE4M3Alignment;
    if (idx >= size) {
        return;
    }
    constexpr int n_src_load = sizeof(float4) / sizeof(InputType); // 8
    constexpr int n_loads = kE4M3Alignment / n_src_load; // 2

    union {
        float4 f4;
        OutputType f8[kE4M3Alignment];
    } f4_e4m3;

#pragma unroll
    for(int i = 0; i < n_loads; i++) {
        float4 _src_f4 = *reinterpret_cast<const float4*>(src + idx + i * n_src_load);
        InputType *_src_ptr = reinterpret_cast<InputType*>(&_src_f4);
#pragma unroll
        for(int j = 0; j < n_src_load; j++) {
            float scaled_val = static_cast<float>(_src_ptr[j]) / *scale_f;
            scaled_val = clamp(scaled_val, -kFP8Max, kFP8Max);
            f4_e4m3.f8[i*n_src_load + j] = static_cast<OutputType>(scaled_val);
        }
    }
    *reinterpret_cast<float4*>(dst + idx) = f4_e4m3.f4;
}

template<typename InputType, typename OutputType>
__global__ void dequantize_fp8_tensor_kernel(
    const InputType *src,
    OutputType *dst,
    const float *scale_f,
    const uint32_t size) {

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr uint32_t n_src_load = sizeof(float2) / sizeof(InputType); // 8

    idx *= n_src_load;
    if (idx >= size) {
        return;
    }

    const InputType *vals = load_f8x8(src + idx);

    // For 16-bit output types (FP16/BF16), use vectorized float4 store (8 values = 16 bytes)
    // For 32-bit output types (FP32), use two float4 stores (8 values = 32 bytes)
    if constexpr (sizeof(OutputType) == 2) {
        union {
            float4 f4;
            OutputType f16[n_src_load];
        } f4_f16;

#pragma unroll
        for (int i = 0; i < n_src_load; i++) {
            f4_f16.f16[i] = static_cast<OutputType>(static_cast<float>(vals[i]) * *scale_f);
        }

        *reinterpret_cast<float4*>(dst + idx) = f4_f16.f4;
    } else {
        // FP32 output: store 8 float values as 2 float4
        union {
            float4 f4[2];
            float f32[n_src_load];
        } f4_f32;

#pragma unroll
        for (int i = 0; i < n_src_load; i++) {
            f4_f32.f32[i] = static_cast<float>(vals[i]) * *scale_f;
        }

        reinterpret_cast<float4*>(dst + idx)[0] = f4_f32.f4[0];
        reinterpret_cast<float4*>(dst + idx)[1] = f4_f32.f4[1];
    }
}

template<typename OutputType>
__device__ __forceinline__ unsigned stochastic_round_fp8_pair(unsigned bits, unsigned noise) {
    constexpr bool e4m3 = std::is_same_v<OutputType, __nv_fp8_e4m3>;
    const __half2 maximum = __float2half2_rn(FP8LimitsTrait<OutputType>::max);
    union { unsigned bits; __half2 halves; } value, magnitude;
    value.bits = bits;
    // Adding +0 clears source -0; clamping first preserves finite saturation and NaN -> -max.
    value.halves = __hmin2(maximum, __hmax2(__hadd2(value.halves, __float2half2_rn(0.0f)),
                                         __hneg2(maximum)));
    magnitude.bits = value.bits & 0x7fff7fff;
    unsigned rounded;
    if constexpr (e4m3) {
        union { unsigned bits; __half2 halves; } normal, subnormal;
        normal.halves = __hmax2(magnitude.halves, __float2half2_rn(0.015625f));
        // Keep carries and borrows within each lane, then discard seven mantissa bits.
        normal.bits = (((normal.bits | 0x80008000) - 0x20002000
                       + ((noise >> 1) & 0x007f007f)) >> 7) & 0x00ff00ff;
        // Below E4M3's normal range, floor(x * 2^17) retains every relevant RNG bit.
        // The two power-of-two multiplies are exact for those FP16 values.
        subnormal.halves = __hmul2(__hmul2(__hmin2(magnitude.halves, __float2half2_rn(0.25f)),
                                         __float2half2_rn(32768.0f)), __float2half2_rn(4.0f));
        const unsigned lo = __half2uint_rz(__low2half(subnormal.halves));
        const unsigned hi = __half2uint_rz(__high2half(subnormal.halves));
        subnormal.bits = ((__byte_perm(lo, hi, 0x5410) + noise) >> 8) & 0x00ff00ff;
        // These nonnegative integer codes are also ordered as FP16 subnormals.
        normal.halves = __hmin2(normal.halves, subnormal.halves);
        rounded = normal.bits;
    } else {
        // E5M2 shares FP16's exponent bias, so both lanes round with one addition.
        rounded = ((magnitude.bits + noise) >> 8) & 0x00ff00ff;
    }
    rounded |= (value.bits >> 8) & 0x00800080;
    return rounded;
}

template<typename InputType, typename OutputType>
__global__ void stochastic_round_fp8_kernel(
    uint8_t *rng_and_dst,
    const InputType *src,
    const int64_t size,
    const int width) {

    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x * width;
    for (int64_t base = static_cast<int64_t>(blockIdx.x) * blockDim.x * width;
         base < size; base += stride) {
        const int64_t idx = base + static_cast<int64_t>(threadIdx.x) * width;
        if (width == kStochasticWidth && base + blockDim.x * kStochasticWidth <= size) {
            struct alignas(16) Input { InputType values[kStochasticWidth]; };
            struct alignas(16) Bytes { unsigned values[kStochasticWidth / 4]; };
            const Input input = *reinterpret_cast<const Input*>(src + idx);
            Bytes bytes = *reinterpret_cast<const Bytes*>(rng_and_dst + idx);
#pragma unroll
            for (int i = 0; i < kStochasticWidth / 4; ++i) {
                unsigned rounded[2];
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    // Preserve FP16 materialization for every input dtype.
                    union { __half2 halves; unsigned bits; } pair;
                    if constexpr (std::is_same_v<InputType, __half>)
                        pair.halves = __halves2half2(input.values[4*i+2*j], input.values[4*i+2*j+1]);
                    else
                        pair.halves = __floats2half2_rn(static_cast<float>(input.values[4*i+2*j]),
                                                      static_cast<float>(input.values[4*i+2*j+1]));
                    rounded[j] = stochastic_round_fp8_pair<OutputType>(
                        pair.bits, __byte_perm(bytes.values[i], 0, j == 0 ? 0x4140 : 0x4342));
                }
                bytes.values[i] = __byte_perm(rounded[0], rounded[1], 0x6420);
            }
            *reinterpret_cast<Bytes*>(rng_and_dst + idx) = bytes;
        } else {
            for (int i = 0; i < width; ++i) {
                const int64_t j = base + threadIdx.x + static_cast<int64_t>(i) * blockDim.x;
                if (j < size) {
                    const unsigned bits = __half_as_ushort(__float2half(static_cast<float>(src[j])));
                    rng_and_dst[j] = stochastic_round_fp8_pair<OutputType>(bits, rng_and_dst[j]);
                }
            }
        }
    }
}

} // anonymous namespace

} // namespace comfy

// C interface for DLPack bindings
extern "C" {

void launch_quantize_fp8_kernel(
    const void* input, 
    void* output, 
    const void* scale, 
    int64_t numel,
    int input_dtype_code, 
    int output_dtype_code,
    cudaStream_t stream) {
  
    if (numel == 0) {
        return;
    }

    const float* scale_f = static_cast<const float*>(scale);
    
    constexpr int vals_per_thread = comfy::kE4M3Alignment;
    constexpr int vals_per_block = vals_per_thread * comfy::kQMaxKernelThreads;
    const int blocks = static_cast<int>((numel + vals_per_block - 1) / vals_per_block);

    // Dispatch based on input and output dtypes
    // Input dtype codes: 0=float32, 1=float16, 2=bfloat16
    // Output dtype codes: 5=float8_e4m3fn, 6=float8_e5m2
    
    DISPATCH_INPUT_FP8_OUTPUT_DTYPES(
        input_dtype_code, output_dtype_code, 
        InputType, OutputType, [&] {
            comfy::quantize_fp8_tensor_kernel<InputType, OutputType>
                <<<blocks, comfy::kQMaxKernelThreads, 0, stream>>>(
                    static_cast<const InputType*>(input),
                    static_cast<OutputType*>(output),
                    scale_f,
                    numel);
        });
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    }
}

void launch_dequantize_fp8_kernel(
    const void* input,
    void* output,
    const void* scale,
    int64_t numel,
    int input_dtype_code,
    int output_dtype_code,
    cudaStream_t stream) {

    if (numel == 0) {
        return;
    }

    const float* scale_f = static_cast<const float*>(scale);

    constexpr int vals_per_thread = 8;
    constexpr int vals_per_block = vals_per_thread * comfy::kQMaxKernelThreads;
    const int blocks = static_cast<int>((numel + vals_per_block - 1) / vals_per_block);

    // Dispatch based on input and output dtypes
    // Input dtype codes: 5=float8_e4m3fn, 6=float8_e5m2
    // Output dtype codes: 0=float32, 1=float16, 2=bfloat16
    DISPATCH_FP8_INPUT_FP_OUTPUT_DTYPES(
        input_dtype_code, output_dtype_code,
        InputType, OutputType, [&] {
            comfy::dequantize_fp8_tensor_kernel<InputType, OutputType>
                <<<blocks, comfy::kQMaxKernelThreads, 0, stream>>>(
                    static_cast<const InputType*>(input),
                    static_cast<OutputType*>(output),
                    scale_f,
                    numel);
        });

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    }
}

void launch_stochastic_round_fp8_kernel(
    void* rng_and_output,
    const void* input,
    int64_t numel,
    int rng_dtype_code,
    int input_dtype_code,
    int output_dtype_code,
    cudaStream_t stream) {

    if (numel == 0) {
        return;
    }

    if (rng_dtype_code != 3) {
        throw std::runtime_error("stochastic_round_fp8 requires uint8 RNG storage");
    }

    const int vals_per_thread = numel >= comfy::kStochasticWidth * comfy::kQMaxKernelThreads &&
        ((reinterpret_cast<uintptr_t>(input) | reinterpret_cast<uintptr_t>(rng_and_output)) & 15) == 0
        ? comfy::kStochasticWidth : 1;
    const int vals_per_block = vals_per_thread * comfy::kQMaxKernelThreads;
    const int max_blocks = vals_per_thread == comfy::kStochasticWidth &&
        output_dtype_code == comfy::DTYPE_CODE_FLOAT8_E4M3FN ? 16384 : std::numeric_limits<int>::max();
    const int blocks = static_cast<int>(std::min<int64_t>(
        (numel + vals_per_block - 1) / vals_per_block, max_blocks));

    DISPATCH_INPUT_FP8_OUTPUT_DTYPES(
        input_dtype_code, output_dtype_code,
        InputType, OutputType, [&] {
            comfy::stochastic_round_fp8_kernel<InputType, OutputType>
                <<<blocks, comfy::kQMaxKernelThreads, 0, stream>>>(
                    static_cast<uint8_t*>(rng_and_output),
                    static_cast<const InputType*>(input),
                    numel, vals_per_thread);
        });

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    }
}

} // extern "C"
