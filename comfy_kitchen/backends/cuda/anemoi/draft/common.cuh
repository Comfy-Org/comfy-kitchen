/*
 * Copyright 2026 Mixed Attention Project Contributors
 * SPDX-License-Identifier: Apache-2.0
 * Ported from mixed_precision_attention e318368c15a2962885df2117c35710e86837d5d1.
 * Shared FP16 DraftMap normalization and GQA GEMM arithmetic. Architecture
 * wrappers retain their own metadata validation and cuBLAS handle policies.
 */
#pragma once
#include "../cublas_runtime.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
#include <cstdint>

namespace {
constexpr int kSoftmaxThreads = 128;
constexpr int kWarpSize = 32;
constexpr int kSoftmaxWarps = kSoftmaxThreads / kWarpSize;
constexpr int64_t kMaxGridX = 2147483647LL;
constexpr cublasComputeType_t kDraftComputeType = CUBLAS_COMPUTE_32F;
constexpr cublasGemmAlgo_t kDraftAlgorithm = CUBLAS_GEMM_DEFAULT_TENSOR_OP;

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
  }
  return value;
}
__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}
template <bool IsMax>
__device__ __forceinline__ float block_reduce(
    float value, float* __restrict__ warp_scratch) {
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  value = IsMax ? warp_max(value) : warp_sum(value);
  if (lane == 0) {
    warp_scratch[warp] = value;
  }
  __syncthreads();
  if (warp == 0) {
    value = lane < kSoftmaxWarps
        ? warp_scratch[lane] : (IsMax ? -CUDART_INF_F : 0.0f);
    value = IsMax ? warp_max(value) : warp_sum(value);
    if (lane == 0) {
      warp_scratch[0] = value;
    }
  }
  __syncthreads();
  return warp_scratch[0];
}

__global__ __launch_bounds__(kSoftmaxThreads) void row_softmax_fp16_kernel(
    half* logits_probability, int64_t row_count, int64_t columns) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= row_count) {
    return;
  }
  const int64_t row_offset = row * columns;
  __shared__ float warp_scratch[kSoftmaxWarps];
  float local_max = -CUDART_INF_F;
  for (int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
    local_max = fmaxf(local_max, __half2float(logits_probability[row_offset + column]));
  }
  const float row_max = block_reduce<true>(local_max, warp_scratch);
  float local_sum = 0.0f;
  for (int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
    local_sum += expf(__half2float(logits_probability[row_offset + column]) - row_max);
  }
  const float inverse_sum = 1.0f / block_reduce<false>(local_sum, warp_scratch);
  for (int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
    const float normalized = expf(
        __half2float(logits_probability[row_offset + column]) - row_max) * inverse_sum;
    logits_probability[row_offset + column] = __float2half_rn(normalized);
  }
}

__global__ __launch_bounds__(kSoftmaxThreads)
void row_softmax_fusion_fp16_kernel(
    half* mean_logits_probability, const half* __restrict__ max_logits,
    int64_t row_count, int64_t columns, float maxpool_weight) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= row_count) return;
  const int64_t row_offset = row * columns;
  __shared__ float mean_scratch[kSoftmaxWarps];
  __shared__ float max_scratch[kSoftmaxWarps];
  float local_mean_max = -CUDART_INF_F;
  float local_max_max = -CUDART_INF_F;
  for (int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
    local_mean_max = fmaxf(local_mean_max,
        __half2float(mean_logits_probability[row_offset + column]));
    local_max_max = fmaxf(local_max_max, __half2float(max_logits[row_offset + column]));
  }
  const float mean_row_max = block_reduce<true>(local_mean_max, mean_scratch);
  const float max_row_max = block_reduce<true>(local_max_max, max_scratch);
  float local_mean_sum = 0.0f;
  float local_max_sum = 0.0f;
  for (int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
    local_mean_sum += expf(
        __half2float(mean_logits_probability[row_offset + column]) - mean_row_max);
    local_max_sum += expf(__half2float(max_logits[row_offset + column]) - max_row_max);
  }
  const float mean_inverse_sum = 1.0f / block_reduce<false>(local_mean_sum, mean_scratch);
  const float max_inverse_sum = 1.0f / block_reduce<false>(local_max_sum, max_scratch);
  const float mean_weight = 1.0f - maxpool_weight;
  for (int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
    const float mean_probability = expf(
        __half2float(mean_logits_probability[row_offset + column]) - mean_row_max) * mean_inverse_sum;
    const float max_probability = expf(
        __half2float(max_logits[row_offset + column]) - max_row_max) * max_inverse_sum;
    mean_logits_probability[row_offset + column] = __float2half_rn(
        mean_weight * mean_probability + maxpool_weight * max_probability);
  }
}

// Geometry/products are checked by each caller before entering this arithmetic
// body. Return the first cuBLAS error to that caller's existing error handler.
inline cublasStatus_t draft_gemm(
    const void* q_pool, const void* k_pool, void* logits,
    int64_t batch_size, int64_t q_heads, int64_t kv_heads,
    int64_t query_rows, int64_t key_rows, int64_t head_dim,
    cublasHandle_t handle) {
  const int64_t queries_per_kv = q_heads / kv_heads;
  const int64_t batch_kv_heads = batch_size * kv_heads;
  const int64_t q_operand_stride = query_rows * head_dim;
  const int64_t k_operand_stride = key_rows * head_dim;
  const int64_t output_stride = query_rows * key_rows;
  const int64_t grouped_q_operand_stride = queries_per_kv * q_operand_stride;
  const int64_t grouped_output_stride = queries_per_kv * output_stride;
  const float alpha = 1.0f / std::sqrt(static_cast<float>(head_dim));
  constexpr float beta = 0.0f;
  const half* q_ptr = static_cast<const half*>(q_pool);
  const half* k_ptr = static_cast<const half*>(k_pool);
  half* logits_ptr = static_cast<half*>(logits);

  // Row-major Q*K^T is emitted through the equivalent column-major K*Q^T.
  // Choose the cheaper affine GQA batching axis exactly as the donor does.
  if (queries_per_kv < batch_kv_heads) {
    for (int64_t query_in_group = 0; query_in_group < queries_per_kv; ++query_in_group) {
      const auto status = comfy::anemoi_cublas().gemm_strided_batched_ex(
          handle, CUBLAS_OP_T, CUBLAS_OP_N,
          static_cast<int>(key_rows), static_cast<int>(query_rows), static_cast<int>(head_dim),
          &alpha, k_ptr, CUDA_R_16F, static_cast<int>(head_dim), k_operand_stride,
          q_ptr + query_in_group * q_operand_stride, CUDA_R_16F,
          static_cast<int>(head_dim), grouped_q_operand_stride, &beta,
          logits_ptr + query_in_group * output_stride, CUDA_R_16F,
          static_cast<int>(key_rows), grouped_output_stride,
          static_cast<int>(batch_kv_heads), kDraftComputeType, kDraftAlgorithm);
      if (status != CUBLAS_STATUS_SUCCESS) return status;
    }
    return CUBLAS_STATUS_SUCCESS;
  }
  for (int64_t batch = 0; batch < batch_size; ++batch) {
    for (int64_t kv_head = 0; kv_head < kv_heads; ++kv_head) {
      const int64_t first_q_head = kv_head * queries_per_kv;
      const half* q_group = q_ptr + (batch * q_heads + first_q_head) * q_operand_stride;
      const half* k_head = k_ptr + (batch * kv_heads + kv_head) * k_operand_stride;
      half* logits_group = logits_ptr + (batch * q_heads + first_q_head) * output_stride;
      const auto status = comfy::anemoi_cublas().gemm_strided_batched_ex(
          handle, CUBLAS_OP_T, CUBLAS_OP_N,
          static_cast<int>(key_rows), static_cast<int>(query_rows), static_cast<int>(head_dim),
          &alpha, k_head, CUDA_R_16F, static_cast<int>(head_dim), 0,
          q_group, CUDA_R_16F, static_cast<int>(head_dim), q_operand_stride, &beta,
          logits_group, CUDA_R_16F, static_cast<int>(key_rows), output_stride,
          static_cast<int>(queries_per_kv), kDraftComputeType, kDraftAlgorithm);
      if (status != CUBLAS_STATUS_SUCCESS) return status;
    }
  }
  return CUBLAS_STATUS_SUCCESS;
}
} // namespace
