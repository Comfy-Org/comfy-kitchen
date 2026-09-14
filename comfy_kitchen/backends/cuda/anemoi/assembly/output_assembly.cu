/*
 * Copyright 2026 Mixed Attention Project Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * Fast, allocation-inclusive final output assembly for the Anemoi integration.
 * This translation unit deliberately inherits the attention extension's
 * --use_fast_math policy.  Numerical acceptance is therefore defined against
 * the eager construction by end-to-end error metrics, not bitwise identity.
 */


#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "assembly_route_draft_native.cuh"

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 8;
constexpr int kThreads = kWarpSize * kWarpsPerBlock;
// This caps queued grid work per SM; it is not a claim that 16 CTAs can be
// simultaneously resident.
constexpr int kQueuedBlocksPerSm = 16;

inline int64_t checked_nonnegative_product(
    int64_t lhs, int64_t rhs, const char* description) {
  ANEMOI_CHECK(lhs >= 0 && rhs >= 0, description, " factors must be nonnegative");
  if (lhs == 0 || rhs == 0) {
    return 0;
  }
  ANEMOI_CHECK(
      lhs <= std::numeric_limits<int64_t>::max() / rhs,
      description, " exceeds int64 range");
  return lhs * rhs;
}

inline void check_cuda_contiguous(
    const anemoi_native::Tensor& tensor, const char* name) {
  ANEMOI_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  ANEMOI_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

inline void check_bf16_rank4(
    const anemoi_native::Tensor& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  ANEMOI_CHECK(
      tensor.scalar_type() == anemoi_native::ScalarType::BFloat16,
      name, " must have dtype torch.bfloat16");
  ANEMOI_CHECK(tensor.dim() == 4, name, " must be rank-four");
}

inline void check_fp32_bhs(
    const anemoi_native::Tensor& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  ANEMOI_CHECK(
      tensor.scalar_type() == anemoi_native::ScalarType::Float,
      name, " must have dtype torch.float32");
  ANEMOI_CHECK(tensor.dim() == 3, name, " must have shape [B,H,S_video]");
}

__device__ __forceinline__ float stable_logaddexp(float left, float right) {
  // This is the same stable construction used by torch.logaddexp.  The equal
  // infinity guard prevents inf-inf from turning two identical infinities into
  // NaN.  Device exp/log1p follow this extension's accepted fast-math policy.
  if (isinf(left) && left == right) {
    return left;
  }
  const float maximum = left > right ? left : right;
  return maximum + log1pf(expf(-fabsf(left - right)));
}

__global__ __launch_bounds__(kThreads) void assemble_video_text_output_kernel(
    const __nv_bfloat16* __restrict__ video_output_bhsd,
    const float* __restrict__ video_lse_bhs,
    const __nv_bfloat16* __restrict__ visual_text_output_bshd,
    const float* __restrict__ visual_text_lse_bhs,
    const __nv_bfloat16* __restrict__ text_output_bshd,
    const bool* __restrict__ text_mask,
    __nv_bfloat16* __restrict__ output_bshd,
    int64_t visual_length,
    int64_t text_length,
    int64_t heads,
    int64_t dimension,
    bool has_text_mask,
    int64_t text_mask_batch_stride,
    int64_t text_mask_sequence_stride,
    int64_t output_rows) {
  const int64_t sequence_length = visual_length + text_length;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  for (int64_t row =
           static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
       row < output_rows;
       row += static_cast<int64_t>(gridDim.x) * kWarpsPerBlock) {
    const int64_t head = row % heads;
    const int64_t sequence = (row / heads) % sequence_length;
    const int64_t batch = row / (heads * sequence_length);
    const int64_t output_offset = row * dimension;

    if (sequence < visual_length) {
      float video_weight = 0.0f;
      float text_weight = 0.0f;
      if (lane == 0) {
        const int64_t lse_offset =
            (batch * heads + head) * visual_length + sequence;
        const float video_lse = video_lse_bhs[lse_offset];
        const float text_lse = visual_text_lse_bhs[lse_offset];
        const float merged_lse = stable_logaddexp(video_lse, text_lse);
        video_weight = expf(video_lse - merged_lse);
        text_weight = expf(text_lse - merged_lse);
      }
      video_weight = __shfl_sync(0xffffffffu, video_weight, 0);
      text_weight = __shfl_sync(0xffffffffu, text_weight, 0);

      const int64_t video_offset =
          ((batch * heads + head) * visual_length + sequence) * dimension;
      const int64_t dense_offset =
          ((batch * visual_length + sequence) * heads + head) * dimension;
      for (int64_t column = lane; column < dimension;
           column += kWarpSize) {
        const float video_value =
            __bfloat162float(video_output_bhsd[video_offset + column]);
        const float text_value =
            __bfloat162float(visual_text_output_bshd[dense_offset + column]);
        const float video_product = __fmul_rn(video_value, video_weight);
        const float text_product = __fmul_rn(text_value, text_weight);
        output_bshd[output_offset + column] =
            __float2bfloat16_rn(__fadd_rn(video_product, text_product));
      }
    } else {
      const int64_t text_sequence = sequence - visual_length;
      const bool valid = !has_text_mask ||
          text_mask[
              batch * text_mask_batch_stride +
              text_sequence * text_mask_sequence_stride];
      const int64_t text_offset =
          ((batch * text_length + text_sequence) * heads + head) * dimension;
      for (int64_t column = lane; column < dimension;
           column += kWarpSize) {
        // __float2bfloat16_rn(0.0f) is an explicit +0, even when the source
        // contains -0 or NaN at a masked text position.
        output_bshd[output_offset + column] = valid
            ? text_output_bshd[text_offset + column]
            : __float2bfloat16_rn(0.0f);
      }
    }
  }
}

// Fused output boundary. The visual-query softmax has already traversed both
// video and text K/V inside one CTA, so there is no second visual partition or
// LSE state to merge here.  This kernel only converts the native BHSD visual
// prefix to Anemoi's BSHD layout and appends the existing text-query output.
__global__ __launch_bounds__(kThreads)
void assemble_fused_visual_text_output_kernel(
    const __nv_bfloat16* __restrict__ visual_output_bhsd,
    const __nv_bfloat16* __restrict__ text_output_bshd,
    const bool* __restrict__ text_mask,
    __nv_bfloat16* __restrict__ output_bshd,
    int64_t visual_length,
    int64_t text_length,
    int64_t heads,
    int64_t dimension,
    bool has_text_mask,
    int64_t text_mask_batch_stride,
    int64_t text_mask_sequence_stride,
    int64_t output_rows) {
  const int64_t sequence_length = visual_length + text_length;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  for (int64_t row =
           static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
       row < output_rows;
       row += static_cast<int64_t>(gridDim.x) * kWarpsPerBlock) {
    const int64_t head = row % heads;
    const int64_t sequence = (row / heads) % sequence_length;
    const int64_t batch = row / (heads * sequence_length);
    const int64_t output_offset = row * dimension;

    if (sequence < visual_length) {
      const int64_t visual_offset =
          ((batch * heads + head) * visual_length + sequence) * dimension;
      for (int64_t column = lane; column < dimension;
           column += kWarpSize) {
        output_bshd[output_offset + column] =
            visual_output_bhsd[visual_offset + column];
      }
    } else {
      const int64_t text_sequence = sequence - visual_length;
      const bool valid = !has_text_mask ||
          text_mask[
              batch * text_mask_batch_stride +
              text_sequence * text_mask_sequence_stride];
      const int64_t text_offset =
          ((batch * text_length + text_sequence) * heads + head) * dimension;
      for (int64_t column = lane; column < dimension;
           column += kWarpSize) {
        output_bshd[output_offset + column] = valid
            ? text_output_bshd[text_offset + column]
            : __float2bfloat16_rn(0.0f);
      }
    }
  }
}

template <typename PrefixT, typename OutputT, bool RouteAware>
__global__ __launch_bounds__(kThreads) void assemble_h3_k64_output_kernel(
    const PrefixT* __restrict__ prefix_output_bhsd,
    const half* __restrict__ video_output_bhsd,
    const int64_t* __restrict__ video_inverse_indices,
    OutputT* __restrict__ output_bshd,
    int64_t prefix_batch_stride,
    int64_t prefix_head_stride,
    int64_t prefix_sequence_stride,
    int64_t prefix_tokens,
    int64_t video_tokens,
    int64_t video_capacity,
    int64_t heads,
    int64_t dimension,
    int64_t output_rows,
    const int* __restrict__ low_counts,
    const int* __restrict__ middle_counts,
    const int* __restrict__ high_counts,
    int64_t route_rows,
    int query_block_shift) {
  const int64_t sequence_tokens = prefix_tokens + video_tokens;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  for (int64_t row =
           static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
       row < output_rows;
       row += static_cast<int64_t>(gridDim.x) * kWarpsPerBlock) {
    const int64_t head = row % heads;
    const int64_t sequence = (row / heads) % sequence_tokens;
    const int64_t batch = row / (heads * sequence_tokens);
    const int64_t output_offset = row * dimension;
    if (sequence < prefix_tokens) {
      const int64_t source_offset =
          batch * prefix_batch_stride + head * prefix_head_stride +
          sequence * prefix_sequence_stride;
      for (int64_t column = lane; column < dimension; column += kWarpSize) {
        if constexpr (std::is_same_v<PrefixT, OutputT>) {
          output_bshd[output_offset + column] =
              prefix_output_bhsd[source_offset + column];
        } else if constexpr (std::is_same_v<OutputT, half>) {
          output_bshd[output_offset + column] = __float2half_rn(
              __bfloat162float(prefix_output_bhsd[source_offset + column]));
        } else {
          output_bshd[output_offset + column] = __float2bfloat16_rn(
              __half2float(prefix_output_bhsd[source_offset + column]));
        }
      }
    } else {
      const int64_t video_token = sequence - prefix_tokens;
      const int64_t physical_token = video_inverse_indices[video_token];
      const int64_t source_offset =
          ((batch * heads + head) * video_capacity + physical_token) *
          dimension;
      int empty = 0;
      if constexpr (RouteAware) {
        if (lane == 0) {
          const int64_t route_offset =
              (batch * heads + head) * route_rows +
              (physical_token >> query_block_shift);
          empty = low_counts[route_offset] == 0 &&
              middle_counts[route_offset] == 0 &&
              (high_counts == nullptr || high_counts[route_offset] == 0);
        }
        empty = __shfl_sync(0xffffffffu, empty, 0);
      }
      if (empty) {
        for (int64_t column = lane; column < dimension; column += kWarpSize) {
          if constexpr (std::is_same_v<OutputT, half>) {
            output_bshd[output_offset + column] = __float2half_rn(0.0f);
          } else {
            output_bshd[output_offset + column] = __float2bfloat16_rn(0.0f);
          }
        }
      } else {
        for (int64_t column = lane; column < dimension; column += kWarpSize) {
          if constexpr (std::is_same_v<OutputT, half>) {
            output_bshd[output_offset + column] =
                video_output_bhsd[source_offset + column];
          } else {
            output_bshd[output_offset + column] = __float2bfloat16_rn(
                __half2float(video_output_bhsd[source_offset + column]));
          }
        }
      }
    }
  }
}

template <typename PrefixT, typename OutputT>
void launch_h3_k64_output_assembly(
    const PrefixT* prefix_output_bhsd,
    const half* video_output_bhsd,
    const int64_t* video_inverse_indices,
    OutputT* output_bshd,
    int64_t prefix_batch_stride,
    int64_t prefix_head_stride,
    int64_t prefix_sequence_stride,
    int64_t prefix_tokens,
    int64_t video_tokens,
    int64_t video_capacity,
    int64_t heads,
    int64_t dimension,
    int64_t output_rows,
    const int* low_counts,
    const int* middle_counts,
    const int* high_counts,
    int64_t route_rows,
    int query_block_shift,
    bool route_aware,
    int64_t grid_x,
    cudaStream_t stream) {
  if (route_aware) {
    assemble_h3_k64_output_kernel<PrefixT, OutputT, true>
        <<<grid_x, kThreads, 0, stream>>>(
        prefix_output_bhsd, video_output_bhsd, video_inverse_indices,
        output_bshd, prefix_batch_stride, prefix_head_stride,
        prefix_sequence_stride, prefix_tokens, video_tokens, video_capacity,
        heads, dimension, output_rows, low_counts, middle_counts, high_counts,
        route_rows, query_block_shift);
  } else {
    assemble_h3_k64_output_kernel<PrefixT, OutputT, false>
        <<<grid_x, kThreads, 0, stream>>>(
        prefix_output_bhsd, video_output_bhsd, video_inverse_indices,
        output_bshd, prefix_batch_stride, prefix_head_stride,
        prefix_sequence_stride, prefix_tokens, video_tokens, video_capacity,
        heads, dimension, output_rows, nullptr, nullptr, nullptr, 0, 0);
  }
}

}  // namespace

anemoi_native::Tensor assemble_h3_k64_output(
    anemoi_native::Tensor prefix_output_bhsd,
    anemoi_native::Tensor video_output_bhsd_fp16,
    anemoi_native::Tensor video_inverse_indices,
    std::optional<anemoi_native::ScalarType> output_dtype,
    std::optional<anemoi_native::Tensor> low_counts,
    std::optional<anemoi_native::Tensor> middle_counts,
    std::optional<anemoi_native::Tensor> high_counts,
    int64_t query_block_size) {
  ANEMOI_CHECK(
      prefix_output_bhsd.is_cuda(),
      "prefix_output_bhsd must be a defined CUDA tensor");
  check_cuda_contiguous(video_output_bhsd_fp16, "video_output_bhsd_fp16");
  check_cuda_contiguous(video_inverse_indices, "video_inverse_indices");
  ANEMOI_CHECK(
      prefix_output_bhsd.scalar_type() == anemoi_native::ScalarType::Half ||
          prefix_output_bhsd.scalar_type() == anemoi_native::ScalarType::BFloat16,
      "prefix_output_bhsd must be FP16 or BF16");
  const anemoi_native::ScalarType resolved_output_dtype =
      output_dtype.value_or(prefix_output_bhsd.scalar_type());
  ANEMOI_CHECK(
      resolved_output_dtype == anemoi_native::ScalarType::Half ||
          resolved_output_dtype == anemoi_native::ScalarType::BFloat16,
      "H3 output dtype must be FP16 or BF16");
  ANEMOI_CHECK(
      video_output_bhsd_fp16.scalar_type() == anemoi_native::ScalarType::Half,
      "video_output_bhsd_fp16 must be FP16");
  ANEMOI_CHECK(
      video_inverse_indices.scalar_type() == anemoi_native::ScalarType::Long &&
          video_inverse_indices.dim() == 1,
      "video_inverse_indices must be int64 [video_tokens]");
  ANEMOI_CHECK(
      prefix_output_bhsd.device() == video_output_bhsd_fp16.device() &&
          prefix_output_bhsd.device() == video_inverse_indices.device(),
      "H3 output assembly tensors must share one CUDA device");
  ANEMOI_CHECK(
      prefix_output_bhsd.dim() == 4 && video_output_bhsd_fp16.dim() == 4,
      "H3 output operands must be rank-four BHSD tensors");
  ANEMOI_CHECK(
      prefix_output_bhsd.stride(0) > 0 &&
          prefix_output_bhsd.stride(1) > 0 &&
          prefix_output_bhsd.stride(2) > 0 &&
          prefix_output_bhsd.stride(3) == 1,
      "prefix_output_bhsd must have positive B/H/S strides and contiguous D");
  const int64_t batch = prefix_output_bhsd.size(0);
  const int64_t heads = prefix_output_bhsd.size(1);
  const int64_t prefix_tokens = prefix_output_bhsd.size(2);
  const int64_t dimension = prefix_output_bhsd.size(3);
  const int64_t video_tokens = video_inverse_indices.numel();
  const int64_t video_capacity = video_output_bhsd_fp16.size(2);
  ANEMOI_CHECK(
      batch > 0 && heads > 0 && prefix_tokens >= 0 && dimension > 0 &&
          video_tokens > 0 && video_capacity >= video_tokens,
      "H3 output assembly dimensions must be positive and capacity-valid");
  ANEMOI_CHECK(
      video_output_bhsd_fp16.size(0) == batch &&
          video_output_bhsd_fp16.size(1) == heads &&
          video_output_bhsd_fp16.size(3) == dimension,
      "video_output_bhsd_fp16 must share B/H/D with the prefix output");

  const bool has_low_counts = low_counts.has_value() && low_counts->defined();
  const bool has_middle_counts =
      middle_counts.has_value() && middle_counts->defined();
  const bool has_high_counts =
      high_counts.has_value() && high_counts->defined() && high_counts->numel() != 0;
  const bool route_aware = has_low_counts || has_middle_counts ||
      (high_counts.has_value() && high_counts->defined()) ||
      query_block_size != 0;
  int64_t route_rows = 0;
  int query_block_shift = 0;
  const int* low_counts_ptr = nullptr;
  const int* middle_counts_ptr = nullptr;
  const int* high_counts_ptr = nullptr;
  if (route_aware) {
    ANEMOI_CHECK(
        has_low_counts && has_middle_counts,
        "route-aware H3 assembly requires low and middle counts");
    ANEMOI_CHECK(
        query_block_size == 64 || query_block_size == 128,
        "route-aware H3 assembly requires query_block_size 64 or 128");
    const auto check_route_counts = [&](const anemoi_native::Tensor& counts,
                                        const char* name) {
      check_cuda_contiguous(counts, name);
      ANEMOI_CHECK(
          counts.scalar_type() == anemoi_native::ScalarType::Int && counts.dim() == 3 &&
              counts.size(0) == batch && counts.size(1) == heads &&
              counts.device() == prefix_output_bhsd.device(),
          name, " must be contiguous CUDA int32 [B,H,R] on the output device");
    };
    check_route_counts(*low_counts, "low_counts");
    check_route_counts(*middle_counts, "middle_counts");
    ANEMOI_CHECK(
        assembly_route_draft_native::shape(*middle_counts) == assembly_route_draft_native::shape(*low_counts),
        "low_counts and middle_counts must have matching [B,H,R] shapes");
    route_rows = low_counts->size(2);
    ANEMOI_CHECK(route_rows > 0, "route count rows must be positive");
    ANEMOI_CHECK(
        video_capacity == checked_nonnegative_product(
            route_rows, query_block_size, "route-aware video capacity"),
        "video capacity must equal route rows times query_block_size");
    if (high_counts.has_value() && high_counts->defined()) {
      if (has_high_counts) {
        check_route_counts(*high_counts, "high_counts");
        ANEMOI_CHECK(
            assembly_route_draft_native::shape(*high_counts) == assembly_route_draft_native::shape(*low_counts),
            "high_counts must match low_counts [B,H,R]");
        high_counts_ptr = high_counts->data_ptr<int>();
      } else {
        check_cuda_contiguous(*high_counts, "high_counts");
        ANEMOI_CHECK(
            high_counts->scalar_type() == anemoi_native::ScalarType::Int &&
                high_counts->device() == prefix_output_bhsd.device(),
            "empty high_counts must be CUDA int32 on the output device");
      }
    }
    low_counts_ptr = low_counts->data_ptr<int>();
    middle_counts_ptr = middle_counts->data_ptr<int>();
    query_block_shift = query_block_size == 64 ? 6 : 7;
  }

  assembly_route_draft_native::DeviceGuard device_guard(prefix_output_bhsd.device());
  const auto device_properties = assembly_route_draft_native::device_properties(prefix_output_bhsd.device());
  const cudaDeviceProp* properties = &device_properties;
  ANEMOI_CHECK(
      assembly_route_draft_native::shared_device(properties),
      "H3 K64 output assembly requires sm89 or sm120, found sm_",
      properties->major, properties->minor);
  ANEMOI_CHECK(
      prefix_tokens <= std::numeric_limits<int64_t>::max() - video_tokens,
      "H3 K64 output sequence length overflows int64");
  const int64_t sequence_tokens = prefix_tokens + video_tokens;
  const int64_t batch_sequence = checked_nonnegative_product(
      batch, sequence_tokens, "H3 K64 output batch-sequence rows");
  const int64_t output_rows = checked_nonnegative_product(
      batch_sequence, heads, "H3 K64 output BSH rows");
  checked_nonnegative_product(
      output_rows, dimension, "H3 K64 output elements");
  auto output = anemoi_native::empty(
      {batch, sequence_tokens, heads, dimension},
      prefix_output_bhsd.options().dtype(resolved_output_dtype));
  const int64_t blocks_needed =
      (output_rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
  const int64_t queued_grid_cap = checked_nonnegative_product(
      properties->multiProcessorCount, kQueuedBlocksPerSm,
      "H3 K64 output assembly SM grid cap");
  const int64_t grid_x = std::min(blocks_needed, queued_grid_cap);
  const cudaStream_t stream =
      anemoi_native::cuda::getCurrentCUDAStream();
  if (prefix_output_bhsd.scalar_type() == anemoi_native::ScalarType::Half &&
      resolved_output_dtype == anemoi_native::ScalarType::Half) {
    launch_h3_k64_output_assembly<half, half>(
        reinterpret_cast<const half*>(prefix_output_bhsd.data_ptr<half>()),
        reinterpret_cast<const half*>(video_output_bhsd_fp16.data_ptr<half>()),
        video_inverse_indices.data_ptr<int64_t>(),
        reinterpret_cast<half*>(output.data_ptr<half>()),
        prefix_output_bhsd.stride(0), prefix_output_bhsd.stride(1),
        prefix_output_bhsd.stride(2), prefix_tokens, video_tokens,
        video_capacity, heads, dimension, output_rows, low_counts_ptr,
        middle_counts_ptr, high_counts_ptr, route_rows, query_block_shift,
        route_aware, grid_x, stream);
  } else if (prefix_output_bhsd.scalar_type() == anemoi_native::ScalarType::Half) {
    launch_h3_k64_output_assembly<half, __nv_bfloat16>(
        reinterpret_cast<const half*>(prefix_output_bhsd.data_ptr<half>()),
        reinterpret_cast<const half*>(video_output_bhsd_fp16.data_ptr<half>()),
        video_inverse_indices.data_ptr<int64_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<__nv_bfloat16>()),
        prefix_output_bhsd.stride(0), prefix_output_bhsd.stride(1),
        prefix_output_bhsd.stride(2), prefix_tokens, video_tokens,
        video_capacity, heads, dimension, output_rows, low_counts_ptr,
        middle_counts_ptr, high_counts_ptr, route_rows, query_block_shift,
        route_aware, grid_x, stream);
  } else if (resolved_output_dtype == anemoi_native::ScalarType::BFloat16) {
    launch_h3_k64_output_assembly<__nv_bfloat16, __nv_bfloat16>(
        reinterpret_cast<const __nv_bfloat16*>(
            prefix_output_bhsd.data_ptr<__nv_bfloat16>()),
        reinterpret_cast<const half*>(video_output_bhsd_fp16.data_ptr<half>()),
        video_inverse_indices.data_ptr<int64_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<__nv_bfloat16>()),
        prefix_output_bhsd.stride(0), prefix_output_bhsd.stride(1),
        prefix_output_bhsd.stride(2), prefix_tokens, video_tokens,
        video_capacity, heads, dimension, output_rows, low_counts_ptr,
        middle_counts_ptr, high_counts_ptr, route_rows, query_block_shift,
        route_aware, grid_x, stream);
  } else {
    launch_h3_k64_output_assembly<__nv_bfloat16, half>(
        reinterpret_cast<const __nv_bfloat16*>(
            prefix_output_bhsd.data_ptr<__nv_bfloat16>()),
        reinterpret_cast<const half*>(video_output_bhsd_fp16.data_ptr<half>()),
        video_inverse_indices.data_ptr<int64_t>(),
        reinterpret_cast<half*>(output.data_ptr<half>()),
        prefix_output_bhsd.stride(0), prefix_output_bhsd.stride(1),
        prefix_output_bhsd.stride(2), prefix_tokens, video_tokens,
        video_capacity, heads, dimension, output_rows, low_counts_ptr,
        middle_counts_ptr, high_counts_ptr, route_rows, query_block_shift,
        route_aware, grid_x, stream);
  }
  ANEMOI_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

anemoi_native::Tensor assemble_video_text_output(
    anemoi_native::Tensor video_output_bhsd,
    anemoi_native::Tensor video_lse_bhs,
    anemoi_native::Tensor visual_text_output_bshd,
    anemoi_native::Tensor visual_text_lse_bhs,
    anemoi_native::Tensor text_output_bshd,
    anemoi_native::Tensor text_mask) {
  check_bf16_rank4(video_output_bhsd, "video_output_bhsd");
  check_fp32_bhs(video_lse_bhs, "video_lse_bhs");
  check_bf16_rank4(visual_text_output_bshd, "visual_text_output_bshd");
  check_fp32_bhs(visual_text_lse_bhs, "visual_text_lse_bhs");
  check_bf16_rank4(text_output_bshd, "text_output_bshd");
  ANEMOI_CHECK(text_mask.is_cuda(), "text_mask must be a CUDA tensor");
  ANEMOI_CHECK(
      text_mask.scalar_type() == anemoi_native::ScalarType::Bool,
      "text_mask must have dtype torch.bool");

  const auto device = video_output_bhsd.device();
  const std::array<std::pair<const anemoi_native::Tensor*, const char*>, 5>
      same_device_tensors{{
          {&video_lse_bhs, "video_lse_bhs"},
          {&visual_text_output_bshd, "visual_text_output_bshd"},
          {&visual_text_lse_bhs, "visual_text_lse_bhs"},
          {&text_output_bshd, "text_output_bshd"},
          {&text_mask, "text_mask"},
      }};
  for (const auto& item : same_device_tensors) {
    ANEMOI_CHECK(item.first->device() == device, item.second, " must share one device");
  }

  const int64_t batch_size = video_output_bhsd.size(0);
  const int64_t heads = video_output_bhsd.size(1);
  const int64_t visual_length = video_output_bhsd.size(2);
  const int64_t dimension = video_output_bhsd.size(3);
  ANEMOI_CHECK(
      batch_size > 0 && heads > 0 && visual_length > 0 && dimension > 0,
      "video_output_bhsd dimensions must be positive");
  ANEMOI_CHECK(
      assembly_route_draft_native::shape(video_lse_bhs) ==
          std::vector<int64_t>({batch_size, heads, visual_length}),
      "video_lse_bhs must have shape [B,H,S_video]");
  ANEMOI_CHECK(
      assembly_route_draft_native::shape(visual_text_output_bshd) ==
          std::vector<int64_t>({batch_size, visual_length, heads, dimension}),
      "visual_text_output_bshd must have shape [B,S_video,H,D]");
  ANEMOI_CHECK(
      assembly_route_draft_native::shape(visual_text_lse_bhs) == assembly_route_draft_native::shape(video_lse_bhs),
      "visual_text_lse_bhs must have shape [B,H,S_video]");
  ANEMOI_CHECK(
      text_output_bshd.size(0) == batch_size &&
          text_output_bshd.size(2) == heads &&
          text_output_bshd.size(3) == dimension,
      "text_output_bshd must have shape [B,S_text,H,D]");
  const int64_t text_length = text_output_bshd.size(1);
  const bool has_text_mask = text_mask.numel() != 0;
  if (has_text_mask) {
    ANEMOI_CHECK(
        text_mask.dim() == 2 && text_mask.size(0) == batch_size &&
            text_mask.size(1) == text_length,
        "nonempty text_mask must have shape [B,S_text]");
    ANEMOI_CHECK(
        text_mask.stride(0) >= 0 && text_mask.stride(1) >= 0,
        "text_mask strides must be nonnegative");
  } else {
    ANEMOI_CHECK(
        text_mask.dim() == 1 && text_mask.size(0) == 0,
        "empty text_mask sentinel must have shape [0]");
  }

  ANEMOI_CHECK(
      visual_length <= std::numeric_limits<int64_t>::max() - text_length,
      "S_video+S_text exceeds int64 range");
  const int64_t sequence_length = visual_length + text_length;
  const int64_t batch_sequence = checked_nonnegative_product(
      batch_size, sequence_length, "output B*S");
  const int64_t output_rows = checked_nonnegative_product(
      batch_sequence, heads, "output B*S*H");
  checked_nonnegative_product(output_rows, dimension, "output elements");

  assembly_route_draft_native::DeviceGuard device_guard(device);
  const auto device_properties = assembly_route_draft_native::device_properties(video_output_bhsd.device());
  const cudaDeviceProp* properties = &device_properties;
  ANEMOI_CHECK(
      assembly_route_draft_native::shared_device(properties),
      "anemoi.layers.attention.mpa._cuda_attention output assembly requires compute capability 8.9 or 12.0, found sm_",
      properties->major, properties->minor);

  auto output = anemoi_native::empty(
      {batch_size, sequence_length, heads, dimension},
      video_output_bhsd.options());
  const int64_t blocks_needed =
      output_rows / kWarpsPerBlock +
      static_cast<int64_t>(output_rows % kWarpsPerBlock != 0);
  const int64_t queued_grid_cap = checked_nonnegative_product(
      properties->multiProcessorCount, kQueuedBlocksPerSm,
      "output assembly SM grid cap");
  const int64_t grid_x = std::min(blocks_needed, queued_grid_cap);
  const cudaStream_t stream =
      anemoi_native::cuda::getCurrentCUDAStream();
  assemble_video_text_output_kernel<<<
      static_cast<unsigned int>(grid_x), kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(
          video_output_bhsd.data_ptr<__nv_bfloat16>()),
      video_lse_bhs.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(
          visual_text_output_bshd.data_ptr<__nv_bfloat16>()),
      visual_text_lse_bhs.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(
          text_output_bshd.data_ptr<__nv_bfloat16>()),
      text_mask.data_ptr<bool>(),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr<__nv_bfloat16>()),
      visual_length, text_length, heads, dimension,
      has_text_mask,
      has_text_mask ? text_mask.stride(0) : 0,
      has_text_mask ? text_mask.stride(1) : 0,
      output_rows);
  ANEMOI_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

anemoi_native::Tensor assemble_fused_visual_text_output(
    anemoi_native::Tensor visual_output_bhsd,
    anemoi_native::Tensor text_output_bshd,
    anemoi_native::Tensor text_mask) {
  check_bf16_rank4(visual_output_bhsd, "visual_output_bhsd");
  check_bf16_rank4(text_output_bshd, "text_output_bshd");
  ANEMOI_CHECK(text_mask.is_cuda(), "text_mask must be a CUDA tensor");
  ANEMOI_CHECK(
      text_mask.scalar_type() == anemoi_native::ScalarType::Bool,
      "text_mask must have dtype torch.bool");

  const auto device = visual_output_bhsd.device();
  ANEMOI_CHECK(
      text_output_bshd.device() == device,
      "text_output_bshd must share the visual output device");
  ANEMOI_CHECK(
      text_mask.device() == device,
      "text_mask must share the visual output device");

  const int64_t batch_size = visual_output_bhsd.size(0);
  const int64_t heads = visual_output_bhsd.size(1);
  const int64_t visual_length = visual_output_bhsd.size(2);
  const int64_t dimension = visual_output_bhsd.size(3);
  ANEMOI_CHECK(
      batch_size > 0 && heads > 0 && visual_length > 0 && dimension > 0,
      "visual_output_bhsd dimensions must be positive");
  ANEMOI_CHECK(
      text_output_bshd.size(0) == batch_size &&
          text_output_bshd.size(2) == heads &&
          text_output_bshd.size(3) == dimension,
      "text_output_bshd must have shape [B,S_text,H,D]");
  const int64_t text_length = text_output_bshd.size(1);
  const bool has_text_mask = text_mask.numel() != 0;
  if (has_text_mask) {
    ANEMOI_CHECK(
        text_mask.dim() == 2 && text_mask.size(0) == batch_size &&
            text_mask.size(1) == text_length,
        "nonempty text_mask must have shape [B,S_text]");
    ANEMOI_CHECK(
        text_mask.stride(0) >= 0 && text_mask.stride(1) >= 0,
        "text_mask strides must be nonnegative");
  } else {
    ANEMOI_CHECK(
        text_mask.dim() == 1 && text_mask.size(0) == 0,
        "empty text_mask sentinel must have shape [0]");
  }

  ANEMOI_CHECK(
      visual_length <= std::numeric_limits<int64_t>::max() - text_length,
      "S_video+S_text exceeds int64 range");
  const int64_t sequence_length = visual_length + text_length;
  const int64_t batch_sequence = checked_nonnegative_product(
      batch_size, sequence_length, "output B*S");
  const int64_t output_rows = checked_nonnegative_product(
      batch_sequence, heads, "output B*S*H");
  checked_nonnegative_product(output_rows, dimension, "output elements");

  assembly_route_draft_native::DeviceGuard device_guard(device);
  const auto device_properties = assembly_route_draft_native::device_properties(visual_output_bhsd.device());
  const cudaDeviceProp* properties = &device_properties;
  ANEMOI_CHECK(
      assembly_route_draft_native::shared_device(properties),
      "anemoi.layers.attention.mpa._cuda_attention fused output assembly requires compute capability 8.9 or 12.0, found sm_",
      properties->major, properties->minor);

  auto output = anemoi_native::empty(
      {batch_size, sequence_length, heads, dimension},
      visual_output_bhsd.options());
  const int64_t blocks_needed =
      output_rows / kWarpsPerBlock +
      static_cast<int64_t>(output_rows % kWarpsPerBlock != 0);
  const int64_t queued_grid_cap = checked_nonnegative_product(
      properties->multiProcessorCount, kQueuedBlocksPerSm,
      "fused output assembly SM grid cap");
  const int64_t grid_x = std::min(blocks_needed, queued_grid_cap);
  const cudaStream_t stream =
      anemoi_native::cuda::getCurrentCUDAStream();
  assemble_fused_visual_text_output_kernel<<<
      static_cast<unsigned int>(grid_x), kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(
          visual_output_bhsd.data_ptr<__nv_bfloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(
          text_output_bshd.data_ptr<__nv_bfloat16>()),
      text_mask.data_ptr<bool>(),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr<__nv_bfloat16>()),
      visual_length, text_length, heads, dimension,
      has_text_mask,
      has_text_mask ? text_mask.stride(0) : 0,
      has_text_mask ? text_mask.stride(1) : 0,
      output_rows);
  ANEMOI_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
