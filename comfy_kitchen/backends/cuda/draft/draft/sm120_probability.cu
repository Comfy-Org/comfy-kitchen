/*
 * Copyright 2026 Mixed Attention Project Contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Ported from mixed_precision_attention commit
 * e318368c15a2962885df2117c35710e86837d5d1.  FP16 pooled operands feed a
 * cuBLAS GEMM with FP32 accumulation; row softmax keeps max, sum, and exp in
 * FP32 and rounds only the final probability to FP16. K-tail reuses the same
 * GEMM contract with one mean and one or two extreme descriptors per K64 block.
 */

#include "draft_probability_launch.h"
#include "common.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {

constexpr int kKTailThreads = 256;
constexpr int kKTailWarps = kKTailThreads / kWarpSize;
constexpr int kKTailTokens = 64;
constexpr int kKTailHeadDim = 128;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

int64_t checked_positive_product(int64_t lhs, int64_t rhs, const char* name) {
  if (lhs <= 0 || rhs <= 0 || lhs > std::numeric_limits<int64_t>::max() / rhs)
    throw std::runtime_error(std::string(name) + " requires positive factors and an int64 result");
  return lhs * rhs;
}

void check_cublas(cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string(operation) + " failed with cuBLAS status " +
                             std::to_string(static_cast<int>(status)));
}

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

// cuBLAS owns its opaque allocations. No tensor/data workspace is allocated here.
// Each host thread owns an independent handle for each current CUDA device.
struct DraftHandle {
  const comfy::DraftCublasFunctions& api;
  cublasHandle_t value = nullptr;
  int device;
  explicit DraftHandle(int device_id)
      : api(comfy::draft_cublas()), device(device_id) {
    check_cublas(api.create(&value), "Draft cublasCreate");
  }
  ~DraftHandle() noexcept {
    // Runtime teardown may already have begun at thread/process exit. Destructors
    // cannot report errors; all operational CUDA/cuBLAS calls below are checked.
    int previous = -1;
    if (cudaGetDevice(&previous) != cudaSuccess) return;
    if (previous != device && cudaSetDevice(device) != cudaSuccess) return;
    (void)api.destroy(value);
    if (previous != device) (void)cudaSetDevice(previous);
  }
  DraftHandle(const DraftHandle&) = delete;
  DraftHandle& operator=(const DraftHandle&) = delete;
};

cublasHandle_t draft_handle(cudaStream_t stream) {
  int device = 0;
  check_cuda(cudaGetDevice(&device), "Draft cudaGetDevice");
  static thread_local std::unordered_map<int, std::unique_ptr<DraftHandle>> handles;
  auto found = handles.find(device);
  if (found == handles.end()) {
    found = handles.emplace(device, std::make_unique<DraftHandle>(device)).first;
  }
  const auto handle = found->second->value;
  const auto& api = found->second->api;
  check_cublas(api.set_stream(handle, stream), "Draft cublasSetStream");
  check_cublas(api.set_pointer_mode(handle, CUBLAS_POINTER_MODE_HOST),
               "Draft cublasSetPointerMode");
  check_cublas(api.set_math_mode(handle, CUBLAS_DEFAULT_MATH), "Draft cublasSetMathMode");
  return handle;
}

template <int TailCount, int HeadDim>
__global__ __launch_bounds__(kKTailThreads) void k_tail_descriptor_kernel(
    const half* __restrict__ k_pool,
    const half* __restrict__ packed_k,
    const int32_t* __restrict__ valid_counts,
    half* __restrict__ descriptors,
    int64_t blocks,
    int64_t prefix_blocks) {
  const int64_t head_block = static_cast<int64_t>(blockIdx.x);
  const int64_t logical_block = head_block % blocks;
  const int64_t packed_blocks = prefix_blocks + blocks;
  const int64_t packed_block =
      (head_block / blocks) * packed_blocks + prefix_blocks + logical_block;
  const int64_t mean_base = head_block * HeadDim;
  const int64_t packed_base = packed_block * kKTailTokens * HeadDim;
  const int64_t descriptor_base =
      head_block * (TailCount + 1) * HeadDim;
  const int count = valid_counts[logical_block];
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  __shared__ float distances[kKTailTokens];
  __shared__ int extremes[2];

#pragma unroll
  for (int round = 0; round < kKTailTokens / kKTailWarps; ++round) {
    const int token = round * kKTailWarps + warp;
    float distance = 0.0f;
    if (token < count) {
#pragma unroll
      for (int element = 0; element < HeadDim / kWarpSize; ++element) {
        const int channel = lane * (HeadDim / kWarpSize) + element;
        const float delta =
            __half2float(packed_k[packed_base + token * HeadDim + channel]) -
            __half2float(k_pool[mean_base + channel]);
        distance += delta * delta;
      }
    }
    distance = warp_sum(distance);
    if (lane == 0 && token < count) {
      distances[token] = distance;
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    float first_distance = -CUDART_INF_F;
    int first = 0;
    float second_distance = -CUDART_INF_F;
    int second = 0;
    for (int token = 0; token < count; ++token) {
      const float distance = distances[token];
      if (distance > first_distance) {
        if constexpr (TailCount == 2) {
          second_distance = first_distance;
          second = first;
        }
        first_distance = distance;
        first = token;
      } else if constexpr (TailCount == 2) {
        if (distance > second_distance) {
          second_distance = distance;
          second = token;
        }
      }
    }
    extremes[0] = first;
    if constexpr (TailCount == 2) {
      extremes[1] = count > 1 ? second : first;
    }
  }
  __syncthreads();

  if (threadIdx.x < HeadDim) {
    const int channel = threadIdx.x;
    descriptors[descriptor_base + channel] = k_pool[mean_base + channel];
    descriptors[descriptor_base + HeadDim + channel] =
        packed_k[packed_base + extremes[0] * HeadDim + channel];
    if constexpr (TailCount == 2) {
      descriptors[descriptor_base + 2 * HeadDim + channel] =
          packed_k[packed_base + extremes[1] * HeadDim + channel];
    }
  }
}

template <int TailCount>
__device__ __forceinline__ float k_tail_log_n(
    const half* __restrict__ expanded_logits,
    int64_t row_offset,
    int64_t key_block,
    const int32_t* __restrict__ valid_counts) {
  const int count = valid_counts[key_block];
  const int64_t offset = row_offset + key_block * (TailCount + 1);
  const float mean = __half2float(expanded_logits[offset]);
  if (count <= 1) {
    return mean;
  }
  const float first = __half2float(expanded_logits[offset + 1]);
  if constexpr (TailCount == 1) {
    const float bulk =
        (static_cast<float>(count) * mean - first) /
        static_cast<float>(count - 1);
    const float weighted_bulk = logf(static_cast<float>(count - 1)) + bulk;
    const float maximum = fmaxf(weighted_bulk, first);
    return maximum +
        logf(expf(weighted_bulk - maximum) + expf(first - maximum));
  } else {
    const float second = __half2float(expanded_logits[offset + 2]);
    if (count == 2) {
      const float maximum = fmaxf(first, second);
      return maximum + logf(expf(first - maximum) + expf(second - maximum));
    }
    const float bulk =
        (static_cast<float>(count) * mean - first - second) /
        static_cast<float>(count - 2);
    const float weighted_bulk = logf(static_cast<float>(count - 2)) + bulk;
    const float maximum = fmaxf(weighted_bulk, fmaxf(first, second));
    return maximum + logf(
        expf(weighted_bulk - maximum) + expf(first - maximum) +
        expf(second - maximum));
  }
}

template <int TailCount>
__global__ __launch_bounds__(kSoftmaxThreads)
void k_tail_probability_kernel(
    const half* __restrict__ expanded_logits,
    const int32_t* __restrict__ valid_counts,
    half* __restrict__ probability,
    int64_t blocks) {
  const int64_t output_row = static_cast<int64_t>(blockIdx.x);
  const int64_t logits_row_offset =
      output_row * blocks * (TailCount + 1);
  const int64_t probability_row_offset = output_row * blocks;
  __shared__ float warp_scratch[kSoftmaxWarps];

  float local_max = -CUDART_INF_F;
  for (int64_t key_block = threadIdx.x; key_block < blocks;
       key_block += blockDim.x) {
    local_max = fmaxf(
        local_max,
        k_tail_log_n<TailCount>(
            expanded_logits, logits_row_offset, key_block, valid_counts));
  }
  const float row_max = block_reduce<true>(local_max, warp_scratch);

  float local_sum = 0.0f;
  for (int64_t key_block = threadIdx.x; key_block < blocks;
       key_block += blockDim.x) {
    const float log_n = k_tail_log_n<TailCount>(
        expanded_logits, logits_row_offset, key_block, valid_counts);
    local_sum += expf(log_n - row_max);
  }
  const float inverse_sum =
      1.0f / block_reduce<false>(local_sum, warp_scratch);

  for (int64_t key_block = threadIdx.x; key_block < blocks;
       key_block += blockDim.x) {
    const float log_n = k_tail_log_n<TailCount>(
        expanded_logits, logits_row_offset, key_block, valid_counts);
    probability[probability_row_offset + key_block] =
        __float2half_rn(expf(log_n - row_max) * inverse_sum);
  }
}

// Check both element and byte extents, including every GEMM batch stride,
// before enqueuing any work (especially before the K-tail descriptor kernel).
void check_half_extent(int64_t heads, int64_t rows, int64_t columns, const char* name) {
  checked_positive_product(checked_positive_product(
      checked_positive_product(heads, rows, name), columns, name), sizeof(half), name);
}

void validate_gemm(int64_t B, int64_t Hq, int64_t Hkv,
                   int64_t query_rows, int64_t key_rows, int64_t D) {
  require(B > 0 && Hq > 0 && Hkv > 0 && query_rows > 0 && key_rows > 0,
          "Draft dimensions must be positive");
  require(Hq % Hkv == 0, "Draft Hq must be divisible by Hkv");
  require(D == 64 || D == 128, "Draft head dimension must be 64 or 128");
  const int64_t q_heads = checked_positive_product(B, Hq, "Draft B*Hq");
  const int64_t kv_heads = checked_positive_product(B, Hkv, "Draft B*Hkv");
  const int64_t int_max = std::numeric_limits<int>::max();
  require(query_rows <= int_max && key_rows <= int_max &&
              Hq / Hkv <= int_max && kv_heads <= int_max,
          "Draft GEMM dimensions/batch exceed cuBLAS int range");
  check_half_extent(q_heads, query_rows, D, "Draft Q bytes");
  check_half_extent(kv_heads, key_rows, D, "Draft K bytes");
  check_half_extent(q_heads, query_rows, key_rows, "Draft logits bytes");
}

void launch_draft_gemm(
    const void* q_pool, const void* k_pool, void* logits,
    int64_t batch_size, int64_t q_heads, int64_t kv_heads,
    int64_t query_rows, int64_t key_rows, int64_t head_dim,
    cublasHandle_t handle) {
  // validate_gemm has already bounded all products and narrowing conversions.
  check_cublas(draft_gemm(q_pool, k_pool, logits, batch_size, q_heads, kv_heads,
      query_rows, key_rows, head_dim, handle), "Draft cublasGemmStridedBatchedEx");
}

template <int TailCount, int HeadDim>
void launch_k_tail_impl(
    const void* q, const void* k, const void* packed_k,
    const int32_t* valid_counts, void* out, void* descriptors,
    void* expanded_logits, int64_t B, int64_t Hq, int64_t Hkv,
    int64_t R, int64_t prefix, int64_t descriptor_blocks,
    int64_t probability_rows, cublasHandle_t handle, cudaStream_t stream) {
  k_tail_descriptor_kernel<TailCount, HeadDim><<<
      static_cast<unsigned int>(descriptor_blocks), kKTailThreads, 0, stream>>>(
      static_cast<const half*>(k), static_cast<const half*>(packed_k),
      valid_counts, static_cast<half*>(descriptors), R, prefix);
  check_cuda(cudaGetLastError(), "K-tail descriptor kernel launch");
  launch_draft_gemm(q, descriptors, expanded_logits, B, Hq, Hkv,
                    R, R * (TailCount + 1), HeadDim, handle);
  k_tail_probability_kernel<TailCount><<<
      static_cast<unsigned int>(probability_rows), kSoftmaxThreads, 0, stream>>>(
      static_cast<const half*>(expanded_logits), valid_counts,
      static_cast<half*>(out), R);
  check_cuda(cudaGetLastError(), "K-tail probability kernel launch");
}

}  // namespace

namespace draft_sm120 {

void launch_draft_probability(
    const void* q, const void* k, const void* q_max, const void* k_max,
    void* out, void* max_logits, int64_t B, int64_t Hq, int64_t Hkv,
    int64_t R, int64_t D, double weight, cudaStream_t stream) {
  require(std::isfinite(weight) && weight >= 0.0 && weight <= 1.0,
          "Draft weight must be finite and in [0,1]");
  require(q && k && out, "Draft Q/K/output pointers must be non-null");
  require(weight == 0.0 || (q_max && k_max), "Draft max pools are required for nonzero weight");
  require(weight == 0.0 || weight == 1.0 || max_logits,
          "Draft max_logits workspace is required for mixed weight");
  validate_gemm(B, Hq, Hkv, R, R, D);
  const int64_t rows = checked_positive_product(
      checked_positive_product(B, Hq, "Draft B*Hq"), R, "Draft softmax rows");
  require(rows <= kMaxGridX, "Draft softmax grid.x exceeds CUDA limit");
  const auto handle = draft_handle(stream);
  if (weight == 0.0 || weight == 1.0) {
    launch_draft_gemm(weight == 0.0 ? q : q_max, weight == 0.0 ? k : k_max,
                      out, B, Hq, Hkv, R, R, D, handle);
    row_softmax_fp16_kernel<<<static_cast<unsigned int>(rows), kSoftmaxThreads, 0, stream>>>(
        static_cast<half*>(out), rows, R);
    check_cuda(cudaGetLastError(), "Draft softmax kernel launch");
    return;
  }
  launch_draft_gemm(q, k, out, B, Hq, Hkv, R, R, D, handle);
  launch_draft_gemm(q_max, k_max, max_logits, B, Hq, Hkv, R, R, D, handle);
  row_softmax_fusion_fp16_kernel<<<
      static_cast<unsigned int>(rows), kSoftmaxThreads, 0, stream>>>(
      static_cast<half*>(out), static_cast<const half*>(max_logits), rows, R,
      static_cast<float>(weight));
  check_cuda(cudaGetLastError(), "Draft fusion softmax kernel launch");
}

void launch_k_tail_probability(
    const void* q, const void* k, const void* packed_k,
    const int32_t* valid_counts, void* out, void* descriptors,
    void* expanded_logits, int64_t B, int64_t Hq, int64_t Hkv,
    int64_t R, int64_t prefix, int tail, int64_t head_dim, cudaStream_t stream) {
  require(tail == 1 || tail == 2, "K-tail tail must be 1 or 2");
  require(q && k && packed_k && valid_counts && out && descriptors && expanded_logits,
          "K-tail input/output/workspace pointers must be non-null");
  require(head_dim == 64 || head_dim == 128,
          "K-tail requires matching head dimensions in {64,128}");
  const int64_t expanded_rows = checked_positive_product(R, tail + 1, "K-tail expanded rows");
  validate_gemm(B, Hq, Hkv, R, expanded_rows, head_dim);
  require(prefix >= 0 && prefix <= std::numeric_limits<int64_t>::max() - R,
          "K-tail prefix must be nonnegative and prefix+R must fit int64");
  const int64_t kv_heads = checked_positive_product(B, Hkv, "K-tail B*Hkv");
  const int64_t q_heads = checked_positive_product(B, Hq, "K-tail B*Hq");
  const int64_t descriptor_blocks = checked_positive_product(kv_heads, R, "K-tail descriptor blocks");
  const int64_t probability_rows = checked_positive_product(q_heads, R, "K-tail probability rows");
  require(descriptor_blocks <= kMaxGridX && probability_rows <= kMaxGridX,
          "K-tail grid.x exceeds CUDA limit");
  const int64_t packed_tokens = checked_positive_product(prefix + R, kKTailTokens, "K-tail packed tokens");
  check_half_extent(kv_heads, packed_tokens, head_dim, "K-tail packed K bytes");
  check_half_extent(kv_heads, R, head_dim, "K-tail pooled K bytes");
  check_half_extent(q_heads, R, R, "K-tail probability bytes");
  checked_positive_product(R, sizeof(int32_t), "K-tail valid_counts bytes");
  const auto handle = draft_handle(stream);
  const auto launch = [&](auto head_dim_tag) {
    constexpr int HeadDim = decltype(head_dim_tag)::value;
    if (tail == 1) {
      launch_k_tail_impl<1, HeadDim>(q, k, packed_k, valid_counts, out, descriptors,
                           expanded_logits, B, Hq, Hkv, R, prefix,
                           descriptor_blocks, probability_rows, handle, stream);
    } else {
      launch_k_tail_impl<2, HeadDim>(q, k, packed_k, valid_counts, out, descriptors,
                           expanded_logits, B, Hq, Hkv, R, prefix,
                           descriptor_blocks, probability_rows, handle, stream);
    }
  };
  if (head_dim == 64) {
    launch(std::integral_constant<int, 64>{});
  } else {
    launch(std::integral_constant<int, 128>{});
  }
}

}  // namespace draft_sm120
