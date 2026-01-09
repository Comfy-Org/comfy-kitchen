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

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdexcept>


namespace comfy {

constexpr unsigned int kMXFP8ValsPerThread = 8;
constexpr unsigned int kMXFP8BlockSize = 32; // Always 32 for MXFP8
constexpr unsigned int kMXFP8ThreadsPerGroup = kMXFP8BlockSize / kMXFP8ValsPerThread;  // 4 threads per group

namespace {

template <
    typename IType,
    typename OType=__nv_fp8_e4m3,
    bool Misaligned = false>
__global__ void quantize_mxfp8_kernel(
    const IType* __restrict__ input,
    OType* __restrict__ output,
    uint8_t* __restrict__ block_scales,  // E8M0 stored as uint8
    const size_t num_cols,
    const size_t num_rows,
    const size_t orig_rows,
    const size_t orig_cols) {

    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t global_elem_idx = static_cast<size_t>(idx) * kMXFP8ValsPerThread;
    
    // Early exit for threads beyond data (reduces divergence)
    if (global_elem_idx >= num_rows * num_cols) return;

    // Storage for values
    IType vals[kMXFP8ValsPerThread];
    
    if constexpr (Misaligned) {
        // Misaligned path: load values one at a time with 2D bounds checking
#pragma unroll
        for (int i = 0; i < kMXFP8ValsPerThread; i++) {
            const size_t elem_idx = global_elem_idx + i;
            const size_t row = elem_idx / num_cols;
            const size_t col = elem_idx % num_cols;
            
            if (row < orig_rows && col < orig_cols) {
                vals[i] = input[row * orig_cols + col];
            } else {
                vals[i] = static_cast<IType>(0.0f);
            }
        }
    } else {
        // Aligned path: 128-bit vectorized load (8 half values)
        float4 loaded = *reinterpret_cast<const float4*>(input + global_elem_idx);
        *reinterpret_cast<float4*>(vals) = loaded;
    }

    // Compute local absmax
    IType absmax = __habs(vals[0]);
#pragma unroll
    for (int i = 1; i < kMXFP8ValsPerThread; i++) {
        absmax = __hmax(absmax, __habs(vals[i]));
    }

    // XOR shuffle reduction across kMXFP8ThreadsPerGroup threads
    // All threads end up with the same max value - no broadcast needed
    constexpr unsigned int mask = 0xffffffff;
    
#pragma unroll
    for (int offset = kMXFP8ThreadsPerGroup / 2; offset >= 1; offset /= 2) {
        IType other = __shfl_xor_sync(mask, absmax, offset);
        absmax = __hmax(absmax, other);
    }
    
    // Compute E8M0 scale: smallest power-of-2 such that absmax / scale <= FP8_MAX
    // Use CUDA intrinsic for E8M0 conversion with round toward +infinity
    constexpr float fp8_max = 448.0f;  // FP8 E4M3 max
    float absmax_f = static_cast<float>(absmax);
    
    float ratio = absmax_f / fp8_max;
    // Round toward +infinity to get the ceiling power-of-2
    // For zero/near-zero, __NV_SATFINITE saturates to minimum E8M0 (2^-127)
    __nv_fp8_storage_t e8m0_val = __nv_cvt_float_to_e8m0(ratio, __NV_SATFINITE, cudaRoundPosInf);
    
    // Store E8M0 scale in swizzled layout (only group leaders)
    if ((threadIdx.x % kMXFP8ThreadsPerGroup) == 0) {
        // Calculate which block this thread group is processing
        const size_t block_linear_idx = global_elem_idx / kMXFP8BlockSize;
        
        // Convert linear block index to 2D coordinates
        const size_t num_blocks_per_row = num_cols / kMXFP8BlockSize;
        const size_t row_idx = block_linear_idx / num_blocks_per_row;
        const size_t col_idx = block_linear_idx % num_blocks_per_row;
        
        // Use swizzled layout function
        size_t scale_offset = scale_factor_swizzled_offset(
            row_idx, col_idx, num_blocks_per_row);
        
        block_scales[scale_offset] = e8m0_val;
    }

    // Compute encode_scale = 1/scale = 2^(127 - e8m0_val) via bit manipulation
    // E8M0 stores scale = 2^(e8m0_val - 127), so encode_scale = 2^(127 - e8m0_val)
    // IEEE 754 float32: exponent field = (127 - e8m0_val) + 127 = 254 - e8m0_val
    uint32_t encode_bits = static_cast<uint32_t>(254 - e8m0_val) << 23;
    float encode_scale = __uint_as_float(encode_bits);
    
    OType vals_output[kMXFP8ValsPerThread];
#pragma unroll
    for (int i = 0; i < kMXFP8ValsPerThread; i++) {
        float val_scaled = static_cast<float>(vals[i]) * encode_scale;
        // Clamp to FP8 range
        val_scaled = fminf(fmaxf(val_scaled, -fp8_max), fp8_max);
        vals_output[i] = static_cast<OType>(val_scaled);
    }

    // Store output - FP8 is 1 byte per value, so kMXFP8ValsPerThread=8 uses float2 (8 bytes)
    *reinterpret_cast<float2*>(output + global_elem_idx) = *reinterpret_cast<float2*>(vals_output);
}

} // namespace

} // namespace comfy

// C interface for DLPack bindings
extern "C" {

void launch_quantize_mxfp8_kernel(
    const void* input,
    void* output,
    void* block_scales,
    int64_t num_rows,
    int64_t num_cols,
    int64_t orig_rows,
    int64_t orig_cols,
    int input_dtype_code,
    cudaStream_t stream) {
    
    if (num_rows == 0 || num_cols == 0) {
        return;
    }
    
    // Check that dimensions are divisible by block size (32)
    if (num_rows % comfy::kMXFP8BlockSize != 0 || num_cols % comfy::kMXFP8BlockSize != 0) {
        throw std::runtime_error("num_rows and num_cols must be divisible by 32 for MXFP8 block quantization");
    }
    
    // Check if input is misaligned
    const bool misaligned = (orig_rows != num_rows) || (orig_cols != num_cols);
    
    const int64_t numel = num_rows * num_cols;
    
    // Each thread processes kMXFP8ValsPerThread values
    constexpr int threads_per_block = 128;
    const int64_t total_threads_needed = numel / comfy::kMXFP8ValsPerThread;
    const int blocks = static_cast<int>((total_threads_needed + threads_per_block - 1) / threads_per_block);
    
    // Dispatch based on input dtype
    DISPATCH_HALF_DTYPE(input_dtype_code, InputType, [&] {
        if (misaligned) {
            comfy::quantize_mxfp8_kernel<InputType, __nv_fp8_e4m3, true>
                <<<blocks, threads_per_block, 0, stream>>>(
                    static_cast<const InputType*>(input),
                    static_cast<__nv_fp8_e4m3*>(output),
                    static_cast<uint8_t*>(block_scales),
                    num_cols,
                    num_rows,
                    orig_rows,
                    orig_cols);
        } else {
            comfy::quantize_mxfp8_kernel<InputType, __nv_fp8_e4m3, false>
                <<<blocks, threads_per_block, 0, stream>>>(
                    static_cast<const InputType*>(input),
                    static_cast<__nv_fp8_e4m3*>(output),
                    static_cast<uint8_t*>(block_scales),
                    num_cols,
                    num_rows,
                    orig_rows,
                    orig_cols);
        }
    });
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    }
}

} // extern "C"

