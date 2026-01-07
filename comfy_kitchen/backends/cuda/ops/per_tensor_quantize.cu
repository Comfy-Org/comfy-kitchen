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

#include <limits>
#include <stdexcept>
#include <string>

namespace comfy {

constexpr int kQMaxKernelThreads = 128;
constexpr int kE4M3Alignment = 16;

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

} // extern "C"
