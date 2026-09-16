/*
 * Copyright (c) 2024 by SageAttention team.
 * Copyright (c) 2025 by SpargeAttn team.
 * Copyright (c) 2026 mixed-attention project contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Project-owned current-stream derivative of SpargeAttention's
 * TransposePadPermuteKernel and MeanScaleKernel.  The numerical layout and
 * scale_max=2.25 behavior are retained. The project adds an explicit
 * all-zero-channel contract and a single typed extension entry point.
 */

// Native host adaptation of donor 4e85afba741bdeaf2d9486cab19cb76d3e7985a4.
#include "native_tensor.h"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <tuple>

#include "../attention/ada/primitives/cp_async.cuh"
#include "../attention/ada/primitives/numeric_conversion.cuh"
#include "../attention/ada/primitives/reduction_utils.cuh"
#include "api.h"

namespace mpa::attention {
namespace {

constexpr uint32_t kVPad = 128;
constexpr uint32_t kTransposeCtaTokens = 64;
constexpr uint32_t kQuantThreads = 256;
constexpr float kFp8PvScaleMax = 2.25f;
constexpr uint32_t kH3Heads = 14;
constexpr uint32_t kH3HeadDim = 128;
constexpr uint32_t kH3PhysicalStages = 666;
constexpr uint32_t kH3VirtualTokens = 42'624;
constexpr uint32_t kH3PackedVChannelTile = 32;
constexpr uint32_t kH3PackedVStageStripes = 4;
constexpr uint32_t kH3PackedVSharedPitch = 72;

template <uint32_t HeadDim>
__global__ void transpose_pad_permute_v_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    uint32_t num_tokens,
    uint32_t num_heads,
    uint32_t padded_tokens) {
  constexpr uint32_t pack_size = 8;
  constexpr uint32_t threads_per_token = HeadDim / pack_size;
  constexpr uint32_t threads_per_cta_token_pack =
      kTransposeCtaTokens / pack_size;

  const uint32_t token_block = blockIdx.x;
  const uint32_t head = blockIdx.y;
  const uint32_t batch = blockIdx.z;
  const uint32_t thread = threadIdx.x;
  const uint32_t source_token =
      token_block * kTransposeCtaTokens + thread / threads_per_token;

  const half* input_ptr =
      input + ((batch * num_heads + head) * num_tokens + source_token) *
                  HeadDim +
      (thread % threads_per_token) * pack_size;
  half* output_ptr =
      output + (batch * num_heads + head) * HeadDim * padded_tokens +
      (thread / threads_per_cta_token_pack) * padded_tokens +
      token_block * kTransposeCtaTokens +
      (thread % threads_per_cta_token_pack) * pack_size;

  __shared__ half shared_load[kTransposeCtaTokens][HeadDim];
  __shared__ half shared_store[HeadDim][kTransposeCtaTokens];

  // Sage/Sparge FP8-MMA token permutation:
  // 0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15 within each group of 16.
  const uint32_t natural_row = thread / threads_per_token;
  const uint32_t row_base = (natural_row / 16) * 16;
  const uint32_t row_mod = natural_row % 16;
  const uint32_t permuted_row =
      row_base + (row_mod / 8) * 2 + ((row_mod / 2) % 4) * 4 +
      row_mod % 2;

  cp_async::pred_load_128b<
      cp_async::PrefetchMode::kNoPrefetch,
      cp_async::SharedMemFillMode::kFillZero>(
      shared_load[permuted_row] +
          (thread % threads_per_token) * pack_size,
      input_ptr,
      source_token < num_tokens);
  cp_async::commit_group();
  cp_async::wait_group<0>();
  __syncthreads();

  const uint32_t smem_row = thread % kTransposeCtaTokens;
  const uint32_t smem_col = thread / kTransposeCtaTokens;
  constexpr uint32_t smem_col_stride = HeadDim / 8;
#pragma unroll
  for (uint32_t index = 0; index < 8; ++index) {
    shared_store[smem_col + index * smem_col_stride][smem_row] =
        shared_load[smem_row][smem_col + index * smem_col_stride];
  }
  __syncthreads();

  *reinterpret_cast<float4*>(output_ptr) =
      *reinterpret_cast<const float4*>(
          &shared_store[thread / threads_per_cta_token_pack]
                       [(thread % threads_per_cta_token_pack) * pack_size]);
}

__global__ void quantize_v_per_channel_kernel(
    const half* __restrict__ input,
    __nv_fp8_e4m3* __restrict__ output,
    float* __restrict__ scale,
    uint32_t num_tokens,
    uint32_t num_heads,
    uint32_t head_dim,
    uint32_t padded_tokens) {
  constexpr uint32_t pack_size = 8;
  const uint32_t head = blockIdx.x;
  const uint32_t batch = blockIdx.y;
  const uint32_t channel = blockIdx.z;
  const uint32_t thread = threadIdx.x;
  constexpr uint32_t gmem_stride = kQuantThreads * pack_size;

  const half* input_base =
      input + ((batch * num_heads + head) * head_dim + channel) *
                  padded_tokens +
      thread * pack_size;
  __nv_fp8_e4m3* output_base =
      output + ((batch * num_heads + head) * head_dim + channel) *
                   padded_tokens +
      thread * pack_size;

  const uint32_t reduction_tokens = ((num_tokens + 15) / 16) * 16;
  const uint32_t reduction_iters =
      reduction_tokens / gmem_stride +
      ((reduction_tokens % gmem_stride) > thread * pack_size);

  float max_value = -1000000.0f;
  float min_value = 1000000.0f;
  half values[pack_size];
  for (uint32_t iteration = 0; iteration < reduction_iters; ++iteration) {
    *reinterpret_cast<float4*>(values) =
        *reinterpret_cast<const float4*>(
            input_base + iteration * gmem_stride);
#pragma unroll
    for (uint32_t element = 0; element < pack_size; ++element) {
      const float value = __half2float(values[element]);
      max_value = fmaxf(max_value, value);
      min_value = fminf(min_value, value);
    }
  }

  const float block_max = vllm::blockReduceMax(max_value);
  const float block_min = vllm::blockReduceMin(min_value);
  __shared__ float shared_reciprocal;
  if (thread == 0) {
    const float amax = fmaxf(fabsf(block_max), fabsf(block_min));
    // Do not divide by zero or propagate NaNs for an all-zero channel.
    shared_reciprocal = amax == 0.0f ? 0.0f : kFp8PvScaleMax / amax;
    scale[(batch * num_heads + head) * head_dim + channel] =
        amax / kFp8PvScaleMax;
  }
  __syncthreads();

  const float reciprocal = shared_reciprocal;
  const uint32_t output_iters =
      padded_tokens / gmem_stride +
      ((padded_tokens % gmem_stride) > thread * pack_size);
  for (uint32_t iteration = 0; iteration < output_iters; ++iteration) {
    *reinterpret_cast<float4*>(values) =
        *reinterpret_cast<const float4*>(
            input_base + iteration * gmem_stride);
    float converted[pack_size];
#pragma unroll
    for (uint32_t element = 0; element < pack_size; ++element) {
      converted[element] = reciprocal == 0.0f
          ? 0.0f
          : __half2float(values[element]) * reciprocal;
    }
    uint32_t fp8_words[2];
    floatx4_to_e4m3x4(fp8_words, converted, converted + 2);
    floatx4_to_e4m3x4(fp8_words + 1, converted + 4, converted + 6);
    *reinterpret_cast<uint2*>(
        output_base + iteration * gmem_stride) =
        *reinterpret_cast<const uint2*>(fp8_words);
  }
}

__global__ void quantize_v_per_channel_from_stage_amax_kernel(
    const half* __restrict__ input,
    const float* __restrict__ stage_amax,
    __nv_fp8_e4m3* __restrict__ output,
    float* __restrict__ scale,
    uint32_t physical_stages,
    uint32_t num_heads,
    uint32_t head_dim,
    uint32_t padded_tokens) {
  constexpr uint32_t pack_size = 8;
  constexpr uint32_t gmem_stride = kQuantThreads * pack_size;
  constexpr float kAllNanAmax = 1000000.0f;
  const uint32_t head = blockIdx.x;
  const uint32_t batch = blockIdx.y;
  const uint32_t channel = blockIdx.z;
  const uint32_t thread = threadIdx.x;

  float thread_amax = -1.0f;
  const int64_t partial_base =
      (static_cast<int64_t>(batch) * num_heads + head) * physical_stages *
          head_dim +
      channel;
  for (uint32_t stage = thread; stage < physical_stages;
       stage += kQuantThreads) {
    thread_amax = fmaxf(
        thread_amax,
        stage_amax[partial_base + static_cast<int64_t>(stage) * head_dim]);
  }

  const float reduced_amax = vllm::blockReduceMax(thread_amax);
  __shared__ float shared_reciprocal;
  if (thread == 0) {
    // -1 is the producer's all-NaN sentinel.  The legacy two-pass kernel
    // starts max/min at +/-1e6, so an input containing no numeric value must
    // recover an amax of exactly 1e6.  Ragged zero padding contributes a
    // numeric zero in the producer and therefore remains an all-zero channel.
    const float amax = reduced_amax < 0.0f ? kAllNanAmax : reduced_amax;
    shared_reciprocal = amax == 0.0f ? 0.0f : kFp8PvScaleMax / amax;
    scale[(batch * num_heads + head) * head_dim + channel] =
        amax / kFp8PvScaleMax;
  }
  __syncthreads();

  const half* input_base =
      input + ((batch * num_heads + head) * head_dim + channel) *
                  padded_tokens +
      thread * pack_size;
  __nv_fp8_e4m3* output_base =
      output + ((batch * num_heads + head) * head_dim + channel) *
                   padded_tokens +
      thread * pack_size;
  const float reciprocal = shared_reciprocal;
  const uint32_t output_iters =
      padded_tokens / gmem_stride +
      ((padded_tokens % gmem_stride) > thread * pack_size);
  half values[pack_size];
  for (uint32_t iteration = 0; iteration < output_iters; ++iteration) {
    *reinterpret_cast<float4*>(values) =
        *reinterpret_cast<const float4*>(
            input_base + iteration * gmem_stride);
    float converted[pack_size];
#pragma unroll
    for (uint32_t element = 0; element < pack_size; ++element) {
      converted[element] = reciprocal == 0.0f
          ? 0.0f
          : __half2float(values[element]) * reciprocal;
    }
    uint32_t fp8_words[2];
    floatx4_to_e4m3x4(fp8_words, converted, converted + 2);
    floatx4_to_e4m3x4(fp8_words + 1, converted + 4, converted + 6);
    *reinterpret_cast<uint2*>(
        output_base + iteration * gmem_stride) =
        *reinterpret_cast<const uint2*>(fp8_words);
  }
}

// Package-private released-H3 production consumer.  Each CTA owns a
// 32-channel tile and one of four disjoint physical-stage stripes.  All CTAs
// reduce the complete, identical stage-amax domain into private shared
// reciprocals; only
// stripe zero publishes v_scale, so quantization has no cross-CTA dependency.
__global__ void quantize_packed_v_fp8_h3_vpartials_kernel(
    const half* __restrict__ packed_value,
    const float* __restrict__ stage_amax,
    __nv_fp8_e4m3* __restrict__ output,
    float* __restrict__ scale) {
  constexpr uint32_t pack_size = 8;
  constexpr uint32_t channel_vectors = kH3PackedVChannelTile / pack_size;
  constexpr uint32_t reduction_lanes =
      kQuantThreads / kH3PackedVChannelTile;
  constexpr float kAllNanAmax = 1000000.0f;
  static_assert(channel_vectors * kTransposeCtaTokens == kQuantThreads);
  static_assert(reduction_lanes * kH3PackedVChannelTile == kQuantThreads);
  static_assert(kH3HeadDim % kH3PackedVChannelTile == 0);
  static_assert(kH3VirtualTokens == kH3PhysicalStages * kTransposeCtaTokens);
  static_assert(kH3PackedVSharedPitch >= kTransposeCtaTokens);
  static_assert(
      (kH3PackedVSharedPitch * sizeof(half)) % alignof(uint4) == 0);

  union __align__(16) SharedWorkspace {
    float reduction[kQuantThreads];
    half transposed[kH3PackedVChannelTile][kH3PackedVSharedPitch];
  };
  __shared__ SharedWorkspace workspace;
  __shared__ float shared_reciprocal[kH3PackedVChannelTile];

  const uint32_t thread = threadIdx.x;
  const uint32_t combined_tile = blockIdx.x;
  const uint32_t channel_tile =
      combined_tile / kH3PackedVStageStripes;
  const uint32_t stage_stripe =
      combined_tile % kH3PackedVStageStripes;
  const uint32_t head = blockIdx.y;
  const uint32_t batch = blockIdx.z;
  const uint32_t channel_base =
      channel_tile * kH3PackedVChannelTile;

  // A warp reads one physical-stage row and all 32 adjacent channels, so the
  // duplicated four-stripe reduction remains fully coalesced.  The stripe id
  // is deliberately absent: all CTAs derive the same reciprocal bits locally.
  const uint32_t local_channel = thread % kH3PackedVChannelTile;
  const uint32_t reduction_lane = thread / kH3PackedVChannelTile;
  const int64_t partial_base =
      (static_cast<int64_t>(batch) * kH3Heads + head) *
          kH3PhysicalStages * kH3HeadDim +
      channel_base + local_channel;
  float thread_amax = -1.0f;
  for (uint32_t stage = reduction_lane; stage < kH3PhysicalStages;
       stage += reduction_lanes) {
    thread_amax = fmaxf(
        thread_amax,
        stage_amax[partial_base + static_cast<int64_t>(stage) * kH3HeadDim]);
  }
  workspace.reduction[thread] = thread_amax;
  __syncthreads();

  if (thread < kH3PackedVChannelTile) {
    float reduced_amax = -1.0f;
#pragma unroll
    for (uint32_t lane = 0; lane < reduction_lanes; ++lane) {
      reduced_amax = fmaxf(
          reduced_amax,
          workspace.reduction[lane * kH3PackedVChannelTile + thread]);
    }
    const float amax = reduced_amax < 0.0f ? kAllNanAmax : reduced_amax;
    shared_reciprocal[thread] =
        amax == 0.0f ? 0.0f : kFp8PvScaleMax / amax;
    if (stage_stripe == 0) {
      scale[(static_cast<int64_t>(batch) * kH3Heads + head) *
                kH3HeadDim +
            channel_base + thread] = amax / kFp8PvScaleMax;
    }
  }
  __syncthreads();

  // packed V is [B,H,T,D].  Four adjacent 128-bit threads own one natural
  // token and all 32 tile channels.  Store directly into channel-major shared
  // rows at the exact Sage/Sparge 16-token permutation used by the producer.
  const uint32_t natural_token = thread / channel_vectors;
  const uint32_t channel_vector = thread % channel_vectors;
  const uint32_t row_base = (natural_token / 16) * 16;
  const uint32_t row_mod = natural_token % 16;
  const uint32_t permuted_row =
      row_base + (row_mod / 8) * 2 + ((row_mod / 2) % 4) * 4 +
      row_mod % 2;
  const uint32_t output_channel = thread / (kTransposeCtaTokens / pack_size);
  const uint32_t output_token =
      (thread % (kTransposeCtaTokens / pack_size)) * pack_size;

#pragma unroll 1
  for (uint32_t stage = stage_stripe; stage < kH3PhysicalStages;
       stage += kH3PackedVStageStripes) {
    half values[pack_size];
    const int64_t input_offset =
        ((static_cast<int64_t>(batch) * kH3Heads + head) *
             kH3VirtualTokens +
         stage * kTransposeCtaTokens + natural_token) *
            kH3HeadDim +
        channel_base + channel_vector * pack_size;
    *reinterpret_cast<uint4*>(values) =
        *reinterpret_cast<const uint4*>(packed_value + input_offset);
#pragma unroll
    for (uint32_t element = 0; element < pack_size; ++element) {
      workspace.transposed[channel_vector * pack_size + element]
                            [permuted_row] = values[element];
    }
    __syncthreads();

    *reinterpret_cast<uint4*>(values) = *reinterpret_cast<const uint4*>(
        workspace.transposed[output_channel] + output_token);
    const float reciprocal = shared_reciprocal[output_channel];
    float converted[pack_size];
#pragma unroll
    for (uint32_t element = 0; element < pack_size; ++element) {
      converted[element] = reciprocal == 0.0f
          ? 0.0f
          : __half2float(values[element]) * reciprocal;
    }
    uint32_t fp8_words[2];
    floatx4_to_e4m3x4(fp8_words, converted, converted + 2);
    floatx4_to_e4m3x4(fp8_words + 1, converted + 4, converted + 6);
    const int64_t output_offset =
        ((static_cast<int64_t>(batch) * kH3Heads + head) * kH3HeadDim +
         channel_base + output_channel) *
            kH3VirtualTokens +
        stage * kTransposeCtaTokens + output_token;
    *reinterpret_cast<uint2*>(output + output_offset) =
        *reinterpret_cast<const uint2*>(fp8_words);
    __syncthreads();
  }
}


template <uint32_t HeadDim>
void launch_transpose_pad_permute(
    const half* input,
    half* output,
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t num_tokens,
    uint32_t padded_tokens,
    cudaStream_t stream) {
  static_assert(kTransposeCtaTokens * HeadDim <= 8192);
  const dim3 grid(
      padded_tokens / kTransposeCtaTokens, num_heads, batch_size);
  const dim3 block(kTransposeCtaTokens * (HeadDim / 8));
  transpose_pad_permute_v_kernel<HeadDim><<<grid, block, 0, stream>>>(
      input, output, num_tokens, num_heads, padded_tokens);
}

}  // namespace
}  // namespace mpa::attention

std::tuple<draft_native::Tensor, draft_native::Tensor> preprocess_v_fp8(
    draft_native::Tensor value) {
  DRAFT_CHECK(value.is_cuda(), "value must be a CUDA tensor");
  DRAFT_CHECK(value.is_contiguous(), "value must be contiguous HND");
  DRAFT_CHECK(value.scalar_type() == draft_native::ScalarType::Half,
              "value must be FP16");
  DRAFT_CHECK(value.dim() == 4, "value must have shape [B,Hkv,K,D]");
  DRAFT_CHECK(
      value.size(0) > 0 && value.size(1) > 0 && value.size(2) > 0,
      "value batch, head, and token dimensions must be positive");
  DRAFT_CHECK(value.size(3) == 64 || value.size(3) == 128,
              "value head dimension must be 64 or 128");

  draft_native::cuda::CUDAGuard device_guard(value.device());
  const int64_t batch_size = value.size(0);
  const int64_t num_heads = value.size(1);
  const int64_t num_tokens = value.size(2);
  const int64_t head_dim = value.size(3);
  const int64_t padded_tokens =
      ((num_tokens + mpa::attention::kVPad - 1) /
       mpa::attention::kVPad) *
      mpa::attention::kVPad;

  auto permuted = draft_native::empty(
      {batch_size, num_heads, head_dim, padded_tokens}, value.options());
  auto value_fp8 = draft_native::empty(
      {batch_size, num_heads, head_dim, padded_tokens},
      value.options().dtype(draft_native::ScalarType::Float8_e4m3fn));
  auto value_scale = draft_native::empty(
      {batch_size, num_heads, head_dim},
      value.options().dtype(draft_native::ScalarType::Float));

  cudaStream_t stream = draft_native::getCurrentCUDAStream();
  const auto* input =
      reinterpret_cast<const half*>(value.data_ptr<draft_native::Half>());
  auto* intermediate =
      reinterpret_cast<half*>(permuted.data_ptr<draft_native::Half>());
  if (head_dim == 64) {
    mpa::attention::launch_transpose_pad_permute<64>(
        input, intermediate, static_cast<uint32_t>(batch_size),
        static_cast<uint32_t>(num_heads), static_cast<uint32_t>(num_tokens),
        static_cast<uint32_t>(padded_tokens), stream);
  } else {
    mpa::attention::launch_transpose_pad_permute<128>(
        input, intermediate, static_cast<uint32_t>(batch_size),
        static_cast<uint32_t>(num_heads), static_cast<uint32_t>(num_tokens),
        static_cast<uint32_t>(padded_tokens), stream);
  }
  DRAFT_CUDA_KERNEL_LAUNCH_CHECK();

  const dim3 quant_grid(num_heads, batch_size, head_dim);
  mpa::attention::quantize_v_per_channel_kernel<<<
      quant_grid, mpa::attention::kQuantThreads, 0, stream>>>(
      intermediate,
      reinterpret_cast<__nv_fp8_e4m3*>(value_fp8.data_ptr()),
      value_scale.data_ptr<float>(), static_cast<uint32_t>(num_tokens),
      static_cast<uint32_t>(num_heads), static_cast<uint32_t>(head_dim),
      static_cast<uint32_t>(padded_tokens));
  DRAFT_CUDA_KERNEL_LAUNCH_CHECK();
  return {value_fp8, value_scale};
}

std::tuple<draft_native::Tensor, draft_native::Tensor> quantize_permuted_v_fp8(
    draft_native::Tensor permuted_value) {
  DRAFT_CHECK(
      permuted_value.is_cuda(), "permuted_value must be a CUDA tensor");
  DRAFT_CHECK(
      permuted_value.is_contiguous(), "permuted_value must be contiguous");
  DRAFT_CHECK(
      permuted_value.scalar_type() == draft_native::ScalarType::Half,
      "permuted_value must be FP16");
  DRAFT_CHECK(
      permuted_value.dim() == 4,
      "permuted_value must have shape [B,Hkv,D,K]");
  DRAFT_CHECK(
      permuted_value.size(0) > 0 && permuted_value.size(1) > 0 &&
          permuted_value.size(3) > 0,
      "permuted_value batch, head, and token dimensions must be positive");
  DRAFT_CHECK(
      permuted_value.size(2) == 64 || permuted_value.size(2) == 128,
      "permuted_value head dimension must be 64 or 128");
  DRAFT_CHECK(
      permuted_value.size(3) % mpa::attention::kVPad == 0,
      "permuted_value token dimension must be a multiple of 128");

  draft_native::cuda::CUDAGuard device_guard(permuted_value.device());
  const int64_t batch_size = permuted_value.size(0);
  const int64_t num_heads = permuted_value.size(1);
  const int64_t head_dim = permuted_value.size(2);
  const int64_t padded_tokens = permuted_value.size(3);
  auto value_fp8 = draft_native::empty(
      permuted_value.sizes(),
      permuted_value.options().dtype(draft_native::ScalarType::Float8_e4m3fn));
  auto value_scale = draft_native::empty(
      {batch_size, num_heads, head_dim},
      permuted_value.options().dtype(draft_native::ScalarType::Float));

  const cudaStream_t stream =
      draft_native::getCurrentCUDAStream();
  const dim3 quant_grid(num_heads, batch_size, head_dim);
  mpa::attention::quantize_v_per_channel_kernel<<<
      quant_grid, mpa::attention::kQuantThreads, 0, stream>>>(
      reinterpret_cast<const half*>(permuted_value.data_ptr<draft_native::Half>()),
      reinterpret_cast<__nv_fp8_e4m3*>(value_fp8.data_ptr()),
      value_scale.data_ptr<float>(),
      static_cast<uint32_t>(padded_tokens),
      static_cast<uint32_t>(num_heads),
      static_cast<uint32_t>(head_dim),
      static_cast<uint32_t>(padded_tokens));
  DRAFT_CUDA_KERNEL_LAUNCH_CHECK();
  return {value_fp8, value_scale};
}

std::tuple<draft_native::Tensor, draft_native::Tensor>
quantize_permuted_v_fp8_h3_vpartials(
    draft_native::Tensor permuted_value,
    draft_native::Tensor value_stage_amax) {
  constexpr int64_t kH3Batch = 1;
  constexpr int64_t kH3Heads = 14;
  constexpr int64_t kH3HeadDim = 128;
  constexpr int64_t kH3PhysicalStages = 666;
  constexpr int64_t kH3VirtualTokens = 42'624;
  DRAFT_CHECK(
      permuted_value.is_cuda() && value_stage_amax.is_cuda(),
      "private H3 V quantization operands must be CUDA tensors");
  DRAFT_CHECK(
      permuted_value.device() == value_stage_amax.device(),
      "private H3 V quantization operands must share one CUDA device");
  DRAFT_CHECK(
      permuted_value.is_contiguous() &&
          permuted_value.scalar_type() == draft_native::ScalarType::Half &&
          permuted_value.sizes() == draft_native::IntArrayRef(
              {kH3Batch, kH3Heads, kH3HeadDim, kH3VirtualTokens}),
      "private H3 permuted V must be contiguous FP16 "
      "[1,14,128,42624]");
  DRAFT_CHECK(
      value_stage_amax.is_contiguous() &&
          value_stage_amax.scalar_type() == draft_native::ScalarType::Float &&
          value_stage_amax.sizes() == draft_native::IntArrayRef(
              {kH3Batch, kH3Heads, kH3PhysicalStages, kH3HeadDim}),
      "private H3 V stage amax must be contiguous FP32 "
      "[1,14,666,128]");

  draft_native::cuda::CUDAGuard device_guard(permuted_value.device());
  const cudaDeviceProp* properties =
      draft_native::cuda::getCurrentDeviceProperties();
  DRAFT_CHECK(
      draft_native::ada_serves_device(properties),
      "private H3 V-partial quantization serves SM89 to SM119");
  auto value_fp8 = draft_native::empty(
      permuted_value.sizes(),
      permuted_value.options().dtype(draft_native::ScalarType::Float8_e4m3fn));
  auto value_scale = draft_native::empty(
      {kH3Batch, kH3Heads, kH3HeadDim},
      permuted_value.options().dtype(draft_native::ScalarType::Float));
  const cudaStream_t stream =
      draft_native::getCurrentCUDAStream();
  const dim3 quant_grid(kH3Heads, kH3Batch, kH3HeadDim);
  mpa::attention::quantize_v_per_channel_from_stage_amax_kernel<<<
      quant_grid, mpa::attention::kQuantThreads, 0, stream>>>(
      reinterpret_cast<const half*>(permuted_value.data_ptr<draft_native::Half>()),
      value_stage_amax.data_ptr<float>(),
      reinterpret_cast<__nv_fp8_e4m3*>(value_fp8.data_ptr()),
      value_scale.data_ptr<float>(),
      static_cast<uint32_t>(kH3PhysicalStages),
      static_cast<uint32_t>(kH3Heads),
      static_cast<uint32_t>(kH3HeadDim),
      static_cast<uint32_t>(kH3VirtualTokens));
  DRAFT_CUDA_KERNEL_LAUNCH_CHECK();
  return {value_fp8, value_scale};
}

std::tuple<draft_native::Tensor, draft_native::Tensor>
quantize_packed_v_fp8_h3_vpartials(
    draft_native::Tensor packed_value,
    draft_native::Tensor value_stage_amax) {
  constexpr int64_t kH3Batch = 1;
  constexpr int64_t kH3Heads = 14;
  constexpr int64_t kH3HeadDim = 128;
  constexpr int64_t kH3PhysicalStages = 666;
  constexpr int64_t kH3VirtualTokens = 42'624;
  constexpr int64_t kChannelTile = 32;
  constexpr int64_t kStageStripes = 4;
  DRAFT_CHECK(
      packed_value.is_cuda() && value_stage_amax.is_cuda(),
      "private packed-H3 V quantization operands must be CUDA tensors");
  DRAFT_CHECK(
      packed_value.device() == value_stage_amax.device(),
      "private packed-H3 V quantization operands must share one CUDA device");
  DRAFT_CHECK(
      packed_value.is_contiguous() &&
          packed_value.scalar_type() == draft_native::ScalarType::Half &&
          packed_value.sizes() == draft_native::IntArrayRef(
              {kH3Batch, kH3Heads, kH3VirtualTokens, kH3HeadDim}),
      "private packed-H3 V must be contiguous FP16 [1,14,42624,128]");
  DRAFT_CHECK(
      value_stage_amax.is_contiguous() &&
          value_stage_amax.scalar_type() == draft_native::ScalarType::Float &&
          value_stage_amax.sizes() == draft_native::IntArrayRef(
              {kH3Batch, kH3Heads, kH3PhysicalStages, kH3HeadDim}),
      "private packed-H3 V stage amax must be contiguous FP32 "
      "[1,14,666,128]");

  draft_native::cuda::CUDAGuard device_guard(packed_value.device());
  const cudaDeviceProp* properties =
      draft_native::cuda::getCurrentDeviceProperties();
  DRAFT_CHECK(
      draft_native::ada_serves_device(properties),
      "private packed-H3 V-partial quantization serves SM89 to SM119");
  auto value_fp8 = draft_native::empty(
      {kH3Batch, kH3Heads, kH3HeadDim, kH3VirtualTokens},
      packed_value.options().dtype(draft_native::ScalarType::Float8_e4m3fn));
  auto value_scale = draft_native::empty(
      {kH3Batch, kH3Heads, kH3HeadDim},
      packed_value.options().dtype(draft_native::ScalarType::Float));
  const cudaStream_t stream =
      draft_native::getCurrentCUDAStream();
  const dim3 quant_grid(
      kH3HeadDim / kChannelTile * kStageStripes,
      kH3Heads,
      kH3Batch);
  mpa::attention::quantize_packed_v_fp8_h3_vpartials_kernel<<<
      quant_grid, mpa::attention::kQuantThreads, 0, stream>>>(
      reinterpret_cast<const half*>(packed_value.data_ptr<draft_native::Half>()),
      value_stage_amax.data_ptr<float>(),
      reinterpret_cast<__nv_fp8_e4m3*>(value_fp8.data_ptr()),
      value_scale.data_ptr<float>());
  DRAFT_CUDA_KERNEL_LAUNCH_CHECK();
  return {value_fp8, value_scale};
}
