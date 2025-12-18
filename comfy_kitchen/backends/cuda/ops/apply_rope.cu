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
#include <string>

namespace comfy {

namespace {

constexpr int kBlockSize = 128;

// Helper to convert to float32 for computation
template<typename T>
__forceinline__ __device__ float to_float(T val) {
    if constexpr (std::is_same_v<T, half>) {
        return __half2float(val);
    } else if constexpr (std::is_same_v<T, nv_bfloat16>) {
        return __bfloat162float(val);
    }
    return 0.0f;
}

// Helper to convert from float32 back to output type
template<typename T>
__forceinline__ __device__ T from_float(float val) {
    if constexpr (std::is_same_v<T, half>) {
        return __float2half(val);
    } else if constexpr (std::is_same_v<T, nv_bfloat16>) {
        return __float2bfloat16(val);
    }
    return T(0);
}

// Vectorized version - uses union trick for guaranteed aligned loads/stores
template <typename InputType>
__global__ void apply_rope_kernel_vectorized(
    const InputType* x,
    const float* freqs,
    InputType* x_out,
    int64_t batch,
    int64_t n_heads,
    int64_t seq_len,
    int64_t head_dim,
    int64_t freqs_batch,
    int64_t freqs_heads,
    int64_t stride_x_batch,
    int64_t stride_x_heads,
    int64_t stride_x_seq,
    int64_t stride_x_dim,
    int64_t stride_freqs_batch,
    int64_t stride_freqs_heads,
    int64_t stride_freqs_seq,
    int64_t stride_freqs_dim,
    int64_t stride_freqs_rot,
    int64_t stride_freqs_pair) {
    
    // Each thread processes 1 pair (2 elements) using vectorized load/store via union
    const int64_t n_pairs = head_dim / 2;
    const int64_t total_pairs = batch * n_heads * seq_len * n_pairs;
    
    // Global thread index
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pairs) {
        return;
    }
    
    // Decompose linear index into (batch_idx, head_idx, seq_idx, pair_idx)
    const int64_t pair_idx = idx % n_pairs;
    int64_t temp = idx / n_pairs;
    const int64_t seq_idx = temp % seq_len;
    temp = temp / seq_len;
    const int64_t head_idx = temp % n_heads;
    const int64_t batch_idx = temp / n_heads;
    
    // Calculate base offset for this (batch, head, seq) location
    const int64_t x_base = batch_idx * stride_x_batch + 
                           head_idx * stride_x_heads + 
                           seq_idx * stride_x_seq + 
                           pair_idx * 2;
    
    // Union for vectorized load/store (2 fp16/bf16 = 32 bits = uint32_t)
    union {
        uint32_t vec;  // 32-bit vectorized load/store for one pair
        InputType elems[2];
    } x_data, x_out_data;
    
    // Vectorized load using reinterpret_cast
    x_data.vec = *reinterpret_cast<const uint32_t*>(&x[x_base]);
    
    // Convert to float32 for computation
    const float x_0 = to_float(x_data.elems[0]);
    const float x_1 = to_float(x_data.elems[1]);
    
    // Handle broadcasting for freqs_cis
    const int64_t freqs_batch_idx = (freqs_batch == 1) ? 0 : batch_idx;
    const int64_t freqs_head_idx = (freqs_heads == 1) ? 0 : head_idx;
    
    // Calculate base offset for freqs_cis
    const int64_t freqs_base = freqs_batch_idx * stride_freqs_batch + 
                               freqs_head_idx * stride_freqs_heads + 
                               seq_idx * stride_freqs_seq + 
                               pair_idx * stride_freqs_dim;
    
    // Load rotation matrix elements using read-only cache
    const float freqs_00 = __ldg(&freqs[freqs_base + 0 * stride_freqs_rot + 0 * stride_freqs_pair]);
    const float freqs_01 = __ldg(&freqs[freqs_base + 0 * stride_freqs_rot + 1 * stride_freqs_pair]);
    const float freqs_10 = __ldg(&freqs[freqs_base + 1 * stride_freqs_rot + 0 * stride_freqs_pair]);
    const float freqs_11 = __ldg(&freqs[freqs_base + 1 * stride_freqs_rot + 1 * stride_freqs_pair]);
    
    // Apply 2D rotation in float32
    const float x_out_0 = freqs_00 * x_0 + freqs_01 * x_1;
    const float x_out_1 = freqs_10 * x_0 + freqs_11 * x_1;
    
    // Convert back to input type and store in union
    x_out_data.elems[0] = from_float<InputType>(x_out_0);
    x_out_data.elems[1] = from_float<InputType>(x_out_1);
    
    // Vectorized store using reinterpret_cast
    *reinterpret_cast<uint32_t*>(&x_out[x_base]) = x_out_data.vec;
}

// Fallback scalar version for edge cases
template <typename InputType>
__global__ void apply_rope_kernel(
    const InputType* x,
    const float* freqs,
    InputType* x_out,
    int64_t batch,
    int64_t n_heads,
    int64_t seq_len,
    int64_t head_dim,
    int64_t freqs_batch,
    int64_t freqs_heads,
    int64_t stride_x_batch,
    int64_t stride_x_heads,
    int64_t stride_x_seq,
    int64_t stride_x_dim,
    int64_t stride_freqs_batch,
    int64_t stride_freqs_heads,
    int64_t stride_freqs_seq,
    int64_t stride_freqs_dim,
    int64_t stride_freqs_rot,
    int64_t stride_freqs_pair) {
    
    // Calculate total number of pairs to process
    const int64_t n_pairs = head_dim / 2;
    const int64_t total_elements = batch * n_heads * seq_len * n_pairs;
    
    // Global thread index
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) {
        return;
    }
    
    // Decompose linear index into (batch_idx, head_idx, seq_idx, pair_idx)
    const int64_t pair_idx = idx % n_pairs;
    int64_t temp = idx / n_pairs;
    const int64_t seq_idx = temp % seq_len;
    temp = temp / seq_len;
    const int64_t head_idx = temp % n_heads;
    const int64_t batch_idx = temp / n_heads;
    
    // Calculate indices for the two elements in each pair
    const int64_t dim_idx_0 = pair_idx * 2;
    const int64_t dim_idx_1 = pair_idx * 2 + 1;
    
    // Calculate offsets for x and x_out
    const int64_t x_offset_0 = batch_idx * stride_x_batch + 
                               head_idx * stride_x_heads + 
                               seq_idx * stride_x_seq + 
                               dim_idx_0 * stride_x_dim;
    const int64_t x_offset_1 = batch_idx * stride_x_batch + 
                               head_idx * stride_x_heads + 
                               seq_idx * stride_x_seq + 
                               dim_idx_1 * stride_x_dim;
    
    // Handle broadcasting for freqs_cis (batch=1 and/or heads=1)
    const int64_t freqs_batch_idx = (freqs_batch == 1) ? 0 : batch_idx;
    const int64_t freqs_head_idx = (freqs_heads == 1) ? 0 : head_idx;
    
    // Calculate base offset for freqs_cis
    // freqs_cis shape: (batch, heads, seq_len, head_dim//2, 2, 2)
    const int64_t freqs_base = freqs_batch_idx * stride_freqs_batch + 
                               freqs_head_idx * stride_freqs_heads + 
                               seq_idx * stride_freqs_seq + 
                               pair_idx * stride_freqs_dim;
    
    // Load rotation matrix elements using read-only cache (__ldg)
    const float freqs_00 = __ldg(&freqs[freqs_base + 0 * stride_freqs_rot + 0 * stride_freqs_pair]);
    const float freqs_01 = __ldg(&freqs[freqs_base + 0 * stride_freqs_rot + 1 * stride_freqs_pair]);
    const float freqs_10 = __ldg(&freqs[freqs_base + 1 * stride_freqs_rot + 0 * stride_freqs_pair]);
    const float freqs_11 = __ldg(&freqs[freqs_base + 1 * stride_freqs_rot + 1 * stride_freqs_pair]);
    
    // Load input values and convert to float32
    const float x_0 = to_float(x[x_offset_0]);
    const float x_1 = to_float(x[x_offset_1]);
    
    // Apply 2D rotation in float32
    const float x_out_0 = freqs_00 * x_0 + freqs_01 * x_1;
    const float x_out_1 = freqs_10 * x_0 + freqs_11 * x_1;
    
    // Convert back to input type and store
    x_out[x_offset_0] = from_float<InputType>(x_out_0);
    x_out[x_offset_1] = from_float<InputType>(x_out_1);
}

template <typename InputType>
void apply_rope_launcher(
    const InputType* x,
    const float* freqs,
    InputType* x_out,
    int64_t batch,
    int64_t n_heads,
    int64_t seq_len,
    int64_t head_dim,
    int64_t freqs_batch,
    int64_t freqs_heads,
    int64_t stride_x_batch,
    int64_t stride_x_heads,
    int64_t stride_x_seq,
    int64_t stride_x_dim,
    int64_t stride_freqs_batch,
    int64_t stride_freqs_heads,
    int64_t stride_freqs_seq,
    int64_t stride_freqs_dim,
    int64_t stride_freqs_rot,
    int64_t stride_freqs_pair,
    cudaStream_t stream) {
    
    const int64_t n_pairs = head_dim / 2;
    const int64_t total_elements = batch * n_heads * seq_len * n_pairs;
    
    if (total_elements == 0) {
        return;
    }
    
    // Adaptively choose block size based on problem characteristics
    const int block_size = kBlockSize;
    
    // Launch with adaptive block size
    const int64_t num_blocks = (total_elements + block_size - 1) / block_size;
    
    if (stride_x_dim == 1) {
        // Use vectorized kernel for contiguous tensors
        apply_rope_kernel_vectorized<InputType><<<num_blocks, block_size, 0, stream>>>(
            x, freqs, x_out, batch, n_heads, seq_len, head_dim,
            freqs_batch, freqs_heads,
            stride_x_batch, stride_x_heads, stride_x_seq, stride_x_dim,
            stride_freqs_batch, stride_freqs_heads, stride_freqs_seq, stride_freqs_dim,
            stride_freqs_rot, stride_freqs_pair
        );
    } else {
        // Fallback to scalar kernel for non-contiguous tensors
        apply_rope_kernel<InputType><<<num_blocks, block_size, 0, stream>>>(
            x, freqs, x_out, batch, n_heads, seq_len, head_dim,
            freqs_batch, freqs_heads,
            stride_x_batch, stride_x_heads, stride_x_seq, stride_x_dim,
            stride_freqs_batch, stride_freqs_heads, stride_freqs_seq, stride_freqs_dim,
            stride_freqs_rot, stride_freqs_pair
        );
    }
    
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
    const void* x,
    const void* freqs,
    void* x_out,
    int64_t batch,
    int64_t n_heads,
    int64_t seq_len,
    int64_t head_dim,
    int64_t freqs_batch,
    int64_t freqs_heads,
    int64_t stride_x_batch,
    int64_t stride_x_heads,
    int64_t stride_x_seq,
    int64_t stride_x_dim,
    int64_t stride_freqs_batch,
    int64_t stride_freqs_heads,
    int64_t stride_freqs_seq,
    int64_t stride_freqs_dim,
    int64_t stride_freqs_rot,
    int64_t stride_freqs_pair,
    int input_dtype_code,
    cudaStream_t stream) {
    
    // Dispatch based on input dtype code (only FP16/BF16 supported)
    // dtype codes: 1=float16, 2=bfloat16
    DISPATCH_HALF_DTYPE(input_dtype_code, InputType, [&] {
        comfy::apply_rope_launcher<InputType>(
            static_cast<const InputType*>(x),
            static_cast<const float*>(freqs),
            static_cast<InputType*>(x_out),
            batch, n_heads, seq_len, head_dim,
            freqs_batch, freqs_heads,
            stride_x_batch, stride_x_heads, stride_x_seq, stride_x_dim,
            stride_freqs_batch, stride_freqs_heads, stride_freqs_seq, stride_freqs_dim,
            stride_freqs_rot, stride_freqs_pair,
            stream
        );
    });
}

} // extern "C"

