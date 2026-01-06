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
#include "dtype_dispatch.cuh"

#include <stdexcept>

namespace comfy {

namespace {

constexpr int kBlockSize = 128;

template <typename InputType>
__global__ void apply_rope_kernel(
    const InputType* xq,
    const InputType* xk,
    const float* freqs,
    InputType* xq_out,
    InputType* xk_out,
    int64_t batch,
    int64_t dim1,
    int64_t dim2,
    int64_t head_dim,
    int64_t freqs_batch,
    int64_t freqs_dim1,
    int64_t freqs_dim2,
    int64_t stride_x_batch,
    int64_t stride_x_dim1,
    int64_t stride_x_dim2,
    int64_t stride_x_dim,
    int64_t stride_freqs_batch,
    int64_t stride_freqs_dim1,
    int64_t stride_freqs_dim2,
    int64_t stride_freqs_dim,
    int64_t stride_freqs_rot,
    int64_t stride_freqs_pair) {
    
    // Each thread processes 2 pairs (4 elements) for better memory coalescing
    const int64_t n_pairs = head_dim / 2;
    const int64_t pairs_per_thread = 2;
    const int64_t n_pair_groups = (n_pairs + pairs_per_thread - 1) / pairs_per_thread;
    const int64_t total_pair_groups = batch * dim1 * dim2 * n_pair_groups;
    
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pair_groups) return;
    
    // Decompose linear index into (batch_idx, dim1_idx, dim2_idx, pair_group_idx)
    const int64_t pair_group_idx = idx % n_pair_groups;
    int64_t temp = idx / n_pair_groups;
    const int64_t dim2_idx = temp % dim2;
    temp = temp / dim2;
    const int64_t dim1_idx = temp % dim1;
    const int64_t batch_idx = temp / dim1;
    
    // Calculate actual pair indices for this thread
    const int64_t pair_idx_0 = pair_group_idx * pairs_per_thread;
    const int64_t pair_idx_1 = pair_idx_0 + 1;
    const bool process_pair_1 = pair_idx_1 < n_pairs;
    
    // Handle broadcasting for freqs_cis (all spatial dimensions)
    const int64_t freqs_batch_idx = (freqs_batch == 1) ? 0 : batch_idx;
    const int64_t freqs_dim1_idx = (freqs_dim1 == 1) ? 0 : dim1_idx;
    const int64_t freqs_dim2_idx = (freqs_dim2 == 1) ? 0 : dim2_idx;
    
    // Helper lambda to load a single 2x2 rotation matrix with adaptive vectorization
    auto load_freqs_matrix = [&](int64_t pair_idx, float& f00, float& f01, float& f10, float& f11) {
        const int64_t freqs_base = freqs_batch_idx * stride_freqs_batch + 
                                   freqs_dim1_idx * stride_freqs_dim1 + 
                                   freqs_dim2_idx * stride_freqs_dim2 + 
                                   pair_idx * stride_freqs_dim;
        
        if (stride_freqs_pair == 1 && stride_freqs_rot == 2) {
            // Fully contiguous 2x2 matrix - single float4 load
            const float4 mat = *reinterpret_cast<const float4*>(&freqs[freqs_base]);
            f00 = mat.x;
            f01 = mat.y;
            f10 = mat.z;
            f11 = mat.w;
        } else if (stride_freqs_pair == 1) {
            // Rows are contiguous - two float2 loads
            const float2 row0 = *reinterpret_cast<const float2*>(&freqs[freqs_base + 0 * stride_freqs_rot]);
            const float2 row1 = *reinterpret_cast<const float2*>(&freqs[freqs_base + 1 * stride_freqs_rot]);
            f00 = row0.x;
            f01 = row0.y;
            f10 = row1.x;
            f11 = row1.y;
        } else if (stride_freqs_rot == 1) {
            // Columns are contiguous - two float2 loads
            const float2 col0 = *reinterpret_cast<const float2*>(&freqs[freqs_base + 0 * stride_freqs_pair]);
            const float2 col1 = *reinterpret_cast<const float2*>(&freqs[freqs_base + 1 * stride_freqs_pair]);
            f00 = col0.x;
            f10 = col0.y;
            f01 = col1.x;
            f11 = col1.y;
        } else {
            // Fully strided - four scalar loads
            f00 = freqs[freqs_base + 0 * stride_freqs_rot + 0 * stride_freqs_pair];
            f01 = freqs[freqs_base + 0 * stride_freqs_rot + 1 * stride_freqs_pair];
            f10 = freqs[freqs_base + 1 * stride_freqs_rot + 0 * stride_freqs_pair];
            f11 = freqs[freqs_base + 1 * stride_freqs_rot + 1 * stride_freqs_pair];
        }
    };
    
    // Load rotation matrices for both pairs
    float freqs_00_0, freqs_01_0, freqs_10_0, freqs_11_0;
    float freqs_00_1, freqs_01_1, freqs_10_1, freqs_11_1;
    
    load_freqs_matrix(pair_idx_0, freqs_00_0, freqs_01_0, freqs_10_0, freqs_11_0);
    
    if (process_pair_1) {
        load_freqs_matrix(pair_idx_1, freqs_00_1, freqs_01_1, freqs_10_1, freqs_11_1);
    }
    
    // Helper lambda to calculate base offset for x tensors
    auto calc_offset = [&](int64_t elem_offset) {
        return batch_idx * stride_x_batch + 
               dim1_idx * stride_x_dim1 + 
               dim2_idx * stride_x_dim2 + 
               elem_offset;
    };
    
    // Helper lambda to load 2 pairs (4 elements) - adapts to contiguous vs strided
    auto load_pairs = [&](const InputType* x_in, float& x_0, float& x_1, float& x_2, float& x_3) {
        if (stride_x_dim == 1) {
            // Vectorized load for contiguous tensors (4x fp16/bf16 = 64-bit as float2)
            union { float2 vec; InputType elems[4]; } data;
            data.vec = *reinterpret_cast<const float2*>(&x_in[calc_offset(pair_idx_0 * 2)]);
            x_0 = static_cast<float>(data.elems[0]);
            x_1 = static_cast<float>(data.elems[1]);
            x_2 = static_cast<float>(data.elems[2]);
            x_3 = static_cast<float>(data.elems[3]);
        } else {
            // Scalar load for strided tensors
            const int64_t off0 = calc_offset(pair_idx_0 * 2 * stride_x_dim);
            x_0 = static_cast<float>(x_in[off0]);
            x_1 = static_cast<float>(x_in[off0 + stride_x_dim]);
            if (process_pair_1) {
                x_2 = static_cast<float>(x_in[off0 + 2 * stride_x_dim]);
                x_3 = static_cast<float>(x_in[off0 + 3 * stride_x_dim]);
            }
        }
    };
    
    // Helper lambda to store 2 pairs (4 elements) - adapts to contiguous vs strided
    auto store_pairs = [&](InputType* x_out, float x_0, float x_1, float x_2, float x_3) {
        if (stride_x_dim == 1) {
            // Vectorized store for contiguous tensors (4x fp16/bf16 = 64-bit as float2)
            union { float2 vec; InputType elems[4]; } data;
            data.elems[0] = static_cast<InputType>(x_0);
            data.elems[1] = static_cast<InputType>(x_1);
            data.elems[2] = static_cast<InputType>(x_2);
            data.elems[3] = static_cast<InputType>(x_3);
            *reinterpret_cast<float2*>(&x_out[calc_offset(pair_idx_0 * 2)]) = data.vec;
        } else {
            // Scalar store for strided tensors
            const int64_t off0 = calc_offset(pair_idx_0 * 2 * stride_x_dim);
            x_out[off0] = static_cast<InputType>(x_0);
            x_out[off0 + stride_x_dim] = static_cast<InputType>(x_1);
            if (process_pair_1) {
                x_out[off0 + 2 * stride_x_dim] = static_cast<InputType>(x_2);
                x_out[off0 + 3 * stride_x_dim] = static_cast<InputType>(x_3);
            }
        }
    };
    
    // Process xq (always) - load 2 pairs (4 elements)
    float x_0, x_1, x_2, x_3;
    load_pairs(xq, x_0, x_1, x_2, x_3);
    
    // Apply 2D rotation to first pair
    float out_0 = freqs_00_0 * x_0 + freqs_01_0 * x_1;
    float out_1 = freqs_10_0 * x_0 + freqs_11_0 * x_1;
    
    // Apply 2D rotation to second pair (if processing)
    float out_2 = 0.0f, out_3 = 0.0f;
    if (process_pair_1) {
        out_2 = freqs_00_1 * x_2 + freqs_01_1 * x_3;
        out_3 = freqs_10_1 * x_2 + freqs_11_1 * x_3;
    }
    
    store_pairs(xq_out, out_0, out_1, out_2, out_3);
    
    // Process xk if provided (reuse loaded freqs)
    if (xk != nullptr) {
        load_pairs(xk, x_0, x_1, x_2, x_3);
        
        // Apply same rotation to first pair
        out_0 = freqs_00_0 * x_0 + freqs_01_0 * x_1;
        out_1 = freqs_10_0 * x_0 + freqs_11_0 * x_1;
        
        // Apply rotation to second pair (if processing)
        if (process_pair_1) {
            out_2 = freqs_00_1 * x_2 + freqs_01_1 * x_3;
            out_3 = freqs_10_1 * x_2 + freqs_11_1 * x_3;
        }
        
        store_pairs(xk_out, out_0, out_1, out_2, out_3);
    }
}

template <typename InputType>
void apply_rope_launcher(
    const InputType* xq,
    const InputType* xk,
    const float* freqs,
    InputType* xq_out,
    InputType* xk_out,
    int64_t batch,
    int64_t dim1,
    int64_t dim2,
    int64_t head_dim,
    int64_t freqs_batch,
    int64_t freqs_dim1,
    int64_t freqs_dim2,
    int64_t stride_x_batch,
    int64_t stride_x_dim1,
    int64_t stride_x_dim2,
    int64_t stride_x_dim,
    int64_t stride_freqs_batch,
    int64_t stride_freqs_dim1,
    int64_t stride_freqs_dim2,
    int64_t stride_freqs_dim,
    int64_t stride_freqs_rot,
    int64_t stride_freqs_pair,
    cudaStream_t stream) {
    
    const int64_t n_pairs = head_dim / 2;
    const int64_t pairs_per_thread = 2;
    const int64_t n_pair_groups = (n_pairs + pairs_per_thread - 1) / pairs_per_thread;
    const int64_t total_pair_groups = batch * dim1 * dim2 * n_pair_groups;
    
    if (total_pair_groups == 0) {
        return;
    }
    
    const int block_size = kBlockSize;
    const int64_t num_blocks = (total_pair_groups + block_size - 1) / block_size;
    
    // Single unified kernel handles both contiguous and strided tensors
    apply_rope_kernel<InputType><<<num_blocks, block_size, 0, stream>>>(
        xq, xk, freqs, xq_out, xk_out, batch, dim1, dim2, head_dim,
        freqs_batch, freqs_dim1, freqs_dim2,
        stride_x_batch, stride_x_dim1, stride_x_dim2, stride_x_dim,
        stride_freqs_batch, stride_freqs_dim1, stride_freqs_dim2, stride_freqs_dim,
        stride_freqs_rot, stride_freqs_pair
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    }
}

} // anonymous namespace

} // namespace comfy

// C interface for DLPack bindings
extern "C" {

void launch_apply_rope_kernel(
    const void* xq,
    const void* xk,
    const void* freqs,
    void* xq_out,
    void* xk_out,
    int64_t batch,
    int64_t dim1,
    int64_t dim2,
    int64_t head_dim,
    int64_t freqs_batch,
    int64_t freqs_dim1,
    int64_t freqs_dim2,
    int64_t stride_x_batch,
    int64_t stride_x_dim1,
    int64_t stride_x_dim2,
    int64_t stride_x_dim,
    int64_t stride_freqs_batch,
    int64_t stride_freqs_dim1,
    int64_t stride_freqs_dim2,
    int64_t stride_freqs_dim,
    int64_t stride_freqs_rot,
    int64_t stride_freqs_pair,
    int input_dtype_code,
    cudaStream_t stream) {
    
    // Dispatch based on input dtype code (only FP16/BF16 supported)
    // dtype codes: 1=float16, 2=bfloat16
    DISPATCH_HALF_DTYPE(input_dtype_code, InputType, [&] {
        comfy::apply_rope_launcher<InputType>(
            static_cast<const InputType*>(xq),
            xk ? static_cast<const InputType*>(xk) : nullptr,
            static_cast<const float*>(freqs),
            static_cast<InputType*>(xq_out),
            xk_out ? static_cast<InputType*>(xk_out) : nullptr,
            batch, dim1, dim2, head_dim,
            freqs_batch, freqs_dim1, freqs_dim2,
            stride_x_batch, stride_x_dim1, stride_x_dim2, stride_x_dim,
            stride_freqs_batch, stride_freqs_dim1, stride_freqs_dim2, stride_freqs_dim,
            stride_freqs_rot, stride_freqs_pair,
            stream
        );
    });
}

} // extern "C"

