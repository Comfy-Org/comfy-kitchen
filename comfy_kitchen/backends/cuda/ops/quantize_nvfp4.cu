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
#include <cuda_runtime.h>
#include <cfloat>
#include <cuda/barrier>
#include <stdexcept>


namespace comfy {

constexpr unsigned int kValsPerThread = 4; // Can only be 2 or 4
constexpr unsigned int kBlockSize = 16; // Always 16 for NVFP4
constexpr unsigned int kThreadsPerGroup = kBlockSize / kValsPerThread;  // 4 threads per group (with kValsPerThread=4)

namespace {

/*
 * FP4 E2M1 Block Quantization with E4M3 Scales
 *  https://docs.nvidia.com/cuda/cublas/index.html#d-block-quantization
 * 
 * Algorithm:
 *   1. Load kValsPerThread values per thread and compute local absmax
 *   2. Shuffle-down reduction to get single absmax per block of kBlockSize values
 *   3. Compute decode_scale = absmax / FP4_MAX / global_decode_scale
 *   4. Clamp decode_scale to E4M3 range and store as FP8 E4M3 in swizzled layout
 *   5. Compute encode_scale = 1.0 / (decode_scale * global_decode_scale) = FP4_MAX / absmax
 *   6. Encode values: multiply by encode_scale and convert to FP4 E2M1
 * 
 * Block Scale Layout (swizzled):
 *   - One scale per block of kBlockSize (16) elements
 *   - Uses swizzled layout: 128-row base blocks, 4-column groups
 *   - Layout: (rb_cnt, cbg_cnt, 32, 4, 4) reshaped to (M_padded, N_blocks_padded)
 *   - M_padded = RoundUp(M, 128), N_blocks_padded = RoundUp(N/16, 4)
 *   - Only group leader (lane_id % kThreadsPerGroup == 0) writes scale
 * 
 * With kValsPerThread=4, kBlockSize=16:
 *   - 4 threads per group, 8 groups per warp
 *   - 2 shuffle iterations (log2(4))
 */
template <
    typename IType,
    typename OType=__nv_fp4x2_e2m1,
    typename ScaleType=__nv_fp8_e4m3,
    bool Misaligned = false>
__global__ void quantize_nvfp4_kernel(
    const IType* const input,
    const float* global_scale, // absmax(input) / (6.0 * 488.0)
    OType* const output,
    ScaleType* const block_scales,
    const size_t num_cols,
    const size_t num_rows,
    const size_t orig_rows,
    const size_t orig_cols,
    const float epsilon) {

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Storage for values - loaded differently based on Misaligned template
    IType vals[kValsPerThread];
    
    if constexpr (Misaligned) {
        // Misaligned path: load kValsPerThread values one at a time with 2D bounds checking
        // 
        // elem_idx is in PADDED coordinate space (num_rows x num_cols layout)
        // but input tensor memory is in ORIGINAL coordinate space (orig_rows x orig_cols layout)
        const size_t base_elem_idx = static_cast<size_t>(idx) * kValsPerThread;
        
#pragma unroll
        for (int i = 0; i < kValsPerThread; i++) {
            const size_t elem_idx = base_elem_idx + i;
            const size_t row = elem_idx / num_cols;
            const size_t col = elem_idx % num_cols;
            
            if (row < orig_rows && col < orig_cols) {
                vals[i] = input[row * orig_cols + col];
            } else {
                vals[i] = static_cast<IType>(0.0f);
            }
        }
    } else {
        // Aligned path: vectorized load of kValsPerThread values
        const IType *vals_ptr = (kValsPerThread == 2) 
            ? load_f16x2(input + kValsPerThread * idx)
            : load_f16x4(input + kValsPerThread * idx);
        
#pragma unroll
        for (int i = 0; i < kValsPerThread; i++) {
            vals[i] = vals_ptr[i];
        }
    }

    // Compute local absmax (shared by both paths)
    IType absmax = __habs(vals[0]);
#pragma unroll
    for (int i = 1; i < kValsPerThread; i++) {
        absmax = __hmax(absmax, __habs(vals[i]));
    }

    // Shuffle down reduction across kThreadsPerGroup threads to get absmax of kBlockSize values
    // Each thread has kValsPerThread values, so kThreadsPerGroup threads cover kBlockSize values
    unsigned int mask = 0xffffffff;
#pragma unroll
    for (int offset = kThreadsPerGroup / 2; offset >= 1; offset /= 2) {
        IType other = __shfl_down_sync(mask, absmax, offset, kThreadsPerGroup);
        absmax = __hmax(absmax, other);
    }
    
    // Broadcast from group leader to all threads in the group
    unsigned int lane_id = threadIdx.x & 0x1f;  // Lane ID within warp (0-31)
    unsigned int group_leader = (lane_id / kThreadsPerGroup) * kThreadsPerGroup;
    absmax = __shfl_sync(mask, absmax, group_leader, comfy::kThreadsPerWarp);
    
    // Compute block scale per NVIDIA spec: decode_scale = absmax / FP4_MAX
    float decode_scale = static_cast<float>(absmax) / FP4LimitsTrait<__nv_fp4x2_storage_t>::max;
    
    // Apply global scale (divide by global_decode_scale, following the transpose kernel pattern)
    // global_scale[0] is the decode scale, so we use its reciprocal for encoding
    const float global_decode_scale = global_scale[0];
    decode_scale = decode_scale / global_decode_scale;  // This is: (absmax/6.0) / global_decode_scale
    
    decode_scale = fminf(decode_scale, FP8LimitsTrait<__nv_fp8_e4m3>::max);
    ScaleType scale_fp8 = static_cast<ScaleType>(decode_scale);
    
    // Cast back to float to match reference implementation (ComputeEncodeScaleFP4)
    // This ensures FP8 quantization error is included in encode_scale computation
    float decode_scale_fp8 = static_cast<float>(scale_fp8);
    
    // Store block scale using swizzled layout
    // Only the first thread in each group (group leader) writes the scale
    // Use threadIdx.x to match reference implementation pattern
    const bool is_group_leader = (threadIdx.x % kThreadsPerGroup) == 0;
    const size_t global_elem_idx = static_cast<size_t>(idx) * kValsPerThread;
    
    if (is_group_leader) {
        // Calculate which block this thread group is processing
        const size_t block_linear_idx = global_elem_idx / 16;  // Linear block index (use literal 16)
        
        // Convert linear block index to 2D coordinates (row, col) based on num_cols
        const size_t num_blocks_per_row = num_cols / 16;  // Use literal 16
        const size_t row_idx = block_linear_idx / num_blocks_per_row;  // Row in the matrix
        const size_t col_idx = block_linear_idx % num_blocks_per_row;  // Column block index
        
        // Use swizzled layout function (matches transpose kernel with swizzled_scale=True)
        size_t scale_offset = scale_factor_swizzled_offset(
            row_idx, col_idx, num_blocks_per_row);
        
        block_scales[scale_offset] = scale_fp8;
    }

    // Quantize kValsPerThread values: compute encode scale and pack into FP4
    // Each OType (__nv_fp4x2_e2m1) stores 2 FP4 values
    // With kValsPerThread=2, each thread outputs exactly 1 packed value
    
    // Compute encode scale using the FP8-quantized decode_scale (matches ComputeEncodeScaleFP4)
    // encode_scale = 1.0 / (decode_scale_fp8 * global_decode_scale)
    float encode_scale = fminf(
        1.0f / (decode_scale_fp8 * global_decode_scale),
        3.402823466e+38f);  // FLT_MAX to avoid overflow
    
    // Apply encode scale to map to FP4 range (multiply by encode_scale = 6.0 / absmax
    float vals_encoded[kValsPerThread];
    for (int i = 0; i < kValsPerThread; i++) {
        vals_encoded[i] = static_cast<float>(vals[i]) * encode_scale;
    }

    // Store quantized values to output using vectorized stores
    // Both aligned and misaligned paths use the same store since each thread processes kValsPerThread values
    if (kValsPerThread == 2) {
        store_fp4x2(output, idx, vals_encoded[0], vals_encoded[1]);
    }
    else if (kValsPerThread == 4) {
        store_fp4x4(output, idx, vals_encoded[0], vals_encoded[1], vals_encoded[2], vals_encoded[3]);
    }
}

/*
 * FP4 E2M1 Block Dequantization with E4M3 Scales
 * 
 * Algorithm (reverse of quantization):
 *   1. Load kValsPerThread FP4 packed values per thread
 *   2. Compute which block this thread belongs to
 *   3. Read the corresponding block scale from swizzled layout
 *   4. Compute decode_scale = block_scale_fp8 * global_decode_scale
 *   5. Unpack FP4 values to float and multiply by decode_scale
 *   6. Convert to output type (FP16 or BF16)
 * 
 * Block Scale Layout (swizzled):
 *   - One scale per block of kBlockSize (16) elements
 *   - Uses swizzled layout matching quantization kernel
 */
template <
    typename IType=__nv_fp4x2_e2m1,
    typename OType,
    typename ScaleType=__nv_fp8_e4m3>
__global__ void dequantize_nvfp4_kernel(
    const IType* const input,
    const float* global_scale,
    const ScaleType* const block_scales,
    OType* const output,
    const size_t num_cols,
    const size_t num_rows) {

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate which block this thread group is processing
    const size_t global_elem_idx = static_cast<size_t>(idx) * kValsPerThread;
    const size_t block_linear_idx = global_elem_idx / 16;
    
    // Convert linear block index to 2D coordinates
    const size_t num_blocks_per_row = num_cols / 16;
    const size_t row_idx = block_linear_idx / num_blocks_per_row;
    const size_t col_idx = block_linear_idx % num_blocks_per_row;
    
    // Read block scale from swizzled layout
    size_t scale_offset = scale_factor_swizzled_offset(
        row_idx, col_idx, num_blocks_per_row);
    ScaleType block_scale_fp8 = block_scales[scale_offset];
    
    // Compute decode scale: decode_scale = block_scale_fp8 * global_decode_scale
    const float global_decode_scale = global_scale[0];
    float decode_scale = static_cast<float>(block_scale_fp8) * global_decode_scale;
    
    // Read packed FP4 values
    // With kValsPerThread=4, we read 2 packed values (each IType contains 2 FP4 values)
    const IType* input_ptr = input + idx * (kValsPerThread / 2);
    
    // Unpack and dequantize
    OType vals_output[kValsPerThread];
    
    if (kValsPerThread == 2) {
        // Read 1 packed value (2 FP4 values)
        __half2_raw unpacked_raw = __nv_cvt_fp4x2_to_halfraw2(
            *reinterpret_cast<const __nv_fp4x2_storage_t*>(input_ptr),
            __NV_E2M1);
        float2 unpacked_f2 = __half22float2(*reinterpret_cast<__half2*>(&unpacked_raw));
        // Note: order is swapped because store_fp4x2 packs as float2{val1, val0}
        vals_output[0] = static_cast<OType>(unpacked_f2.y * decode_scale);
        vals_output[1] = static_cast<OType>(unpacked_f2.x * decode_scale);
    }
    else if (kValsPerThread == 4) {
        // Read 2 packed values (4 FP4 values total)
        union {
            uint16_t u16;
            __nv_fp4x2_storage_t fp4x2[2];
        } packed;
        packed.u16 = *reinterpret_cast<const uint16_t*>(input_ptr);
        
        __half2_raw unpacked_raw0 = __nv_cvt_fp4x2_to_halfraw2(packed.fp4x2[0], __NV_E2M1);
        __half2_raw unpacked_raw1 = __nv_cvt_fp4x2_to_halfraw2(packed.fp4x2[1], __NV_E2M1);
        float2 unpacked_f2_0 = __half22float2(*reinterpret_cast<__half2*>(&unpacked_raw0));
        float2 unpacked_f2_1 = __half22float2(*reinterpret_cast<__half2*>(&unpacked_raw1));
        
        // Note: order is swapped because store_fp4x4 packs values with reversed order
        vals_output[0] = static_cast<OType>(unpacked_f2_0.y * decode_scale);
        vals_output[1] = static_cast<OType>(unpacked_f2_0.x * decode_scale);
        vals_output[2] = static_cast<OType>(unpacked_f2_1.y * decode_scale);
        vals_output[3] = static_cast<OType>(unpacked_f2_1.x * decode_scale);
    }
    
    // Store output using vectorized writes
    // For 16-bit types: kValsPerThread=4 uses float2 (8 bytes), kValsPerThread=2 uses float (4 bytes)
    // For 32-bit types: kValsPerThread=4 uses float4 (16 bytes), kValsPerThread=2 uses float2 (8 bytes)
    OType* output_ptr = output + kValsPerThread * idx;
    if constexpr (sizeof(OType) == 2) {
        // FP16/BF16 output
        if (kValsPerThread == 2) {
            *reinterpret_cast<float*>(output_ptr) = *reinterpret_cast<float*>(vals_output);
        }
        else if (kValsPerThread == 4) {
            *reinterpret_cast<float2*>(output_ptr) = *reinterpret_cast<float2*>(vals_output);
        }
    } else {
        // FP32 output
        if (kValsPerThread == 2) {
            *reinterpret_cast<float2*>(output_ptr) = *reinterpret_cast<float2*>(vals_output);
        }
        else if (kValsPerThread == 4) {
            *reinterpret_cast<float4*>(output_ptr) = *reinterpret_cast<float4*>(vals_output);
        }
    }
}

} // namespace

} // namespace comfy

// C interface for DLPack bindings
extern "C" {

void launch_quantize_nvfp4_kernel(
    const void* input,
    const void* global_scale,
    void* output,
    void* block_scales,
    int64_t num_rows,
    int64_t num_cols,
    int64_t orig_rows,
    int64_t orig_cols,
    float epsilon,
    int input_dtype_code,
    cudaStream_t stream) {
    
    if (num_rows == 0 || num_cols == 0) {
        return;
    }
    
    // Check that dimensions are divisible by block size (16)
    if (num_rows % comfy::kBlockSize != 0 || num_cols % comfy::kBlockSize != 0) {
        throw std::runtime_error("num_rows and num_cols must be divisible by 16 for FP4 block quantization");
    }
    
    // Check if input is misaligned (original dimensions differ from padded dimensions)
    const bool misaligned = (orig_rows != num_rows) || (orig_cols != num_cols);
    
    const int64_t numel = num_rows * num_cols;
    const float* scale_f = static_cast<const float*>(global_scale);
    
    // Each thread processes kValsPerThread values (same for both aligned and misaligned paths)
    constexpr int threads_per_block = 128;  // CUDA block size
    const int64_t total_threads_needed = numel / comfy::kValsPerThread;
    const int blocks = static_cast<int>((total_threads_needed + threads_per_block - 1) / threads_per_block);
    
    // Dispatch based on input dtype (only FP16/BF16 supported)
    // Input dtype codes: 1=float16, 2=bfloat16
    DISPATCH_HALF_DTYPE(input_dtype_code, InputType, [&] {
        if (misaligned) {
            // Misaligned path: loads values one at a time with bounds checking
            comfy::quantize_nvfp4_kernel<InputType, __nv_fp4x2_e2m1, __nv_fp8_e4m3, true>
                <<<blocks, threads_per_block, 0, stream>>>(
                    static_cast<const InputType*>(input),
                    scale_f,
                    static_cast<__nv_fp4x2_e2m1*>(output),
                    static_cast<__nv_fp8_e4m3*>(block_scales),
                    num_cols,
                    num_rows,
                    orig_rows,
                    orig_cols,
                    epsilon);
        } else {
            // Aligned path: uses vectorized loads
            comfy::quantize_nvfp4_kernel<InputType, __nv_fp4x2_e2m1, __nv_fp8_e4m3, false>
                <<<blocks, threads_per_block, 0, stream>>>(
                    static_cast<const InputType*>(input),
                    scale_f,
                    static_cast<__nv_fp4x2_e2m1*>(output),
                    static_cast<__nv_fp8_e4m3*>(block_scales),
                    num_cols,
                    num_rows,
                    orig_rows,
                    orig_cols,
                    epsilon);
        }
    });
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    }
}

void launch_dequantize_nvfp4_kernel(
    const void* input,
    const void* global_scale,
    const void* block_scales,
    void* output,
    int64_t num_rows,
    int64_t num_cols,
    int output_dtype_code,
    cudaStream_t stream) {

    if (num_rows == 0 || num_cols == 0) {
        return;
    }
    
    // Check that dimensions are divisible by block size (16)
    if (num_rows % comfy::kBlockSize != 0 || num_cols % comfy::kBlockSize != 0) {
        throw std::runtime_error("num_rows and num_cols must be divisible by 16 for FP4 block dequantization");
    }
    
    const int64_t numel = num_rows * num_cols;
    const float* scale_f = static_cast<const float*>(global_scale);
    
    // Each thread processes kValsPerThread values
    constexpr int threads_per_block = 128;
    const int64_t total_threads_needed = numel / comfy::kValsPerThread;
    const int blocks = static_cast<int>((total_threads_needed + threads_per_block - 1) / threads_per_block);
    
    // Dispatch based on output dtype
    // Output dtype codes: 0=float32, 1=float16, 2=bfloat16
    DISPATCH_FP_DTYPE(output_dtype_code, OutputType, [&] {
        comfy::dequantize_nvfp4_kernel<__nv_fp4x2_e2m1, OutputType, __nv_fp8_e4m3>
            <<<blocks, threads_per_block, 0, stream>>>(
                static_cast<const __nv_fp4x2_e2m1*>(input),
                scale_f,
                static_cast<const __nv_fp8_e4m3*>(block_scales),
                static_cast<OutputType*>(output),
                num_cols,
                num_rows);
    });
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    }
}

} // extern "C"