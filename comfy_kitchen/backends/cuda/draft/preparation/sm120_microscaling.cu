/* Project-owned Q64/Q128 NVFP4 preparation for already-packed FP16 QKV.
 * CUDA device code retained from the nested Anemoi repository at
 * 4e85afba741bdeaf2d9486cab19cb76d3e7985a4, updated through
 * 1765573bea31159adba32165770d19171d5153bd (D64/D128 shared phase and
 * preparation specializations); allocation-free host port.
 * Donor: csrc/attention/cuda/sm120/q128_microscaling_preparation.cu.
 */
#include "preparation.h"
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <math_constants.h>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace draft::sm120::preparation {
namespace {
void check_launch() {
  const auto status = cudaGetLastError();
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("SM120 preparation launch: ") + cudaGetErrorString(status));
}
/*
 * Copyright (c) 2024 by SageAttention team.
 *
 * Inspired by CUTLASS, https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/numeric_conversion.h
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
 */
__device__ __forceinline__ void floatx4_to_e4m3x4(uint32_t *dest, float *source0, float *source1)
{
  asm volatile( \
      "{\n" \
      ".reg .b16 lo;\n" \
      ".reg .b16 hi;\n" \
      "cvt.rn.satfinite.e4m3x2.f32   lo, %2, %1;\n" \
      "cvt.rn.satfinite.e4m3x2.f32   hi, %4, %3;\n" \
      "mov.b32 %0, {lo, hi};\n" \
      "}" \
      : "=r"(dest[0]) : "f"(source0[0]), "f"(source0[1]), "f"(source1[0]), "f"(source1[1]));
}
__device__ __forceinline__ float subgroup_max8(float value) {
#pragma unroll
  for (int delta = 4; delta > 0; delta >>= 1) {
    value = fmaxf(
        value, __shfl_down_sync(0xffffffffU, value, delta, 8));
  }
  return __shfl_sync(0xffffffffU, value, 0, 8);
}

__device__ __forceinline__ float subgroup_max32(float value) {
#pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1) {
    value = fmaxf(value, __shfl_down_sync(0xffffffffU, value, delta));
  }
  return __shfl_sync(0xffffffffU, value, 0);
}

__device__ __forceinline__ uint8_t encode_e2m1_pair(
    float low, float high, float dequant) {
  if (dequant == 0.0f) return 0U;
  return __nv_cvt_float2_to_fp4x2(
      make_float2(__fdiv_rn(low, dequant), __fdiv_rn(high, dequant)),
      __NV_E2M1, cudaRoundNearest);
}

__device__ __forceinline__ float decode_e4m3(uint8_t bits) {
  __nv_fp8_e4m3 value;
  value.__x = bits;
  return static_cast<float>(value);
}

__device__ __forceinline__ uint8_t nv_scale_bits(
    float amax, float global_scale) {
  return __nv_cvt_float_to_fp8(
      __fdiv_rn(__fdiv_rn(amax, 6.0f), global_scale),
      __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ uint8_t encode_e8m0(float amax) {
  if (amax == 0.0f) return 0U;
  int exponent = ilogbf(amax) - 8;
  exponent += amax > ldexpf(448.0f, exponent);
  exponent = max(-127, min(127, exponent));
  return static_cast<uint8_t>(exponent + 127);
}

__device__ __forceinline__ uint8_t encode_e4m3(float value, uint8_t scale) {
  if (scale == 0U) return 0U;
  const float dequant = exp2f(static_cast<int>(scale) - 127);
  return __nv_cvt_float_to_fp8(
      __fdiv_rn(value, dequant), __NV_SATFINITE, __NV_E4M3);
}

template <int HeadDim>
__global__ void prepare_mxfp8_qk_kernel(
    const half* input, uint8_t* data, uint8_t* scales, int64_t rows) {
  const int64_t row = blockIdx.x;
  const int channel = threadIdx.x;
  if (row >= rows) return;
  const float value = __half2float(input[row * HeadDim + channel]);
  const uint8_t scale = encode_e8m0(subgroup_max32(fabsf(value)));
  if ((channel & 31) == 0) {
    scales[row * (HeadDim / 32) + channel / 32] = scale;
  }
  data[row * HeadDim + channel] = encode_e4m3(value, scale);
}

template <int HeadDim>
__global__ void prepare_mxfp8_v_kernel(
    const half* value,
    uint8_t* data,
    uint8_t* scales,
    int64_t heads,
    int64_t tokens) {
  const int64_t stage = blockIdx.x;
  const int64_t head = blockIdx.y;
  const int64_t batch = blockIdx.z;
  const int channel = threadIdx.x;
  constexpr int stage_tokens = 64;
  const int64_t bh = batch * heads + head;
  const half* source = value + bh * tokens * HeadDim;
  uint8_t* destination = data + (bh * HeadDim + channel) * tokens;
  uint8_t* scale_record = scales +
      (bh * (tokens / stage_tokens) + stage) * (HeadDim * 2);
#pragma unroll
  for (int group = 0; group < 2; ++group) {
    half values[32];
    float amax = 0.0f;
#pragma unroll
    for (int token = 0; token < 32; ++token) {
      const int row = stage * stage_tokens + group * 32 + token;
      values[token] = source[row * HeadDim + channel];
      amax = fmaxf(amax, fabsf(__half2float(values[token])));
    }
    const uint8_t scale = encode_e8m0(amax);
    const int d16 = channel / 16;
    const int within16 = channel % 16;
    const int scale_index =
        d16 * 32 + (within16 % 8) * 4 + (within16 / 8) * 2 + group;
    scale_record[scale_index] = scale;
#pragma unroll
    for (int stored = 0; stored < 32; ++stored) {
      const int local16 = stored % 16;
      const int natural = (stored / 16) * 16 + (local16 / 4) * 2 +
          ((local16 % 4) / 2) * 8 + local16 % 2;
      destination[stage * stage_tokens + group * 32 + stored] =
          encode_e4m3(__half2float(values[natural]), scale);
    }
  }
}

// Localized from SageAttention3 d1a57a5 scaled_fp4_quant_permute.  The
// involutive K32 permutation makes QK C fragments directly usable as PV A
// fragments; Q remains natural because Draft loads it straight into registers.
template <bool PermuteTokens, int HeadDim>
__global__ void prepare_nvfp4_qk_kernel(
    const half* input,
    uint8_t* data,
    uint8_t* scales,
    const float* global_scale,
    int64_t rows) {
  const int64_t row = blockIdx.x;
  const int channel = threadIdx.x * 2;
  if (row >= rows) return;
  int64_t source_row = row;
  if constexpr (PermuteTokens) {
    const int64_t local = row & 31;
    source_row = row - local + (local / 8) * 2 +
        ((local % 8) / 2) * 8 + local % 2;
  }
  const half2 values = *reinterpret_cast<const half2*>(
      input + source_row * HeadDim + channel);
  const float2 pair = __half22float2(values);
  const float amax = subgroup_max8(fmaxf(fabsf(pair.x), fabsf(pair.y)));
  const uint8_t scale = nv_scale_bits(amax, global_scale[0]);
  if ((threadIdx.x & 7) == 0) {
    scales[row * (HeadDim / 16) + channel / 16] = scale;
  }
  const float dequant = decode_e4m3(scale) * global_scale[0];
  data[row * (HeadDim / 2) + threadIdx.x] =
      encode_e2m1_pair(pair.x, pair.y, dequant);
}

template <int HeadDim>
__global__ void prepare_nvfp4_v_kernel(
    const half* value,
    uint8_t* data,
    uint8_t* scales,
    const float* global_scale,
    int64_t heads,
    int64_t tokens) {
  const int64_t stage = blockIdx.x;
  const int64_t head = blockIdx.y;
  const int64_t batch = blockIdx.z;
  const int channel = threadIdx.x;
  constexpr int stage_tokens = 64;
  constexpr int group_tokens = 16;
  const int64_t bh = batch * heads + head;
  const half* source = value + bh * tokens * HeadDim;
  uint8_t* destination =
      data + (bh * HeadDim + channel) * (tokens / 2);
  uint8_t* scale_record = scales +
      (bh * (tokens / stage_tokens) + stage) * (HeadDim * 4) +
      channel * 4;
#pragma unroll
  for (int group = 0; group < 4; ++group) {
    half values[group_tokens];
    float amax = 0.0f;
#pragma unroll
    for (int token = 0; token < group_tokens; ++token) {
      const int64_t row = stage * stage_tokens + group * group_tokens + token;
      values[token] = source[row * HeadDim + channel];
      amax = fmaxf(
          amax, fabsf(__half2float(values[token])));
    }
    const uint8_t scale = nv_scale_bits(amax, global_scale[0]);
    scale_record[group] = scale;
    const float dequant = decode_e4m3(scale) * global_scale[0];
#pragma unroll
    for (int pair = 0; pair < group_tokens / 2; ++pair) {
      const float value0 = __half2float(values[pair * 2]);
      const float value1 = __half2float(values[pair * 2 + 1]);
      destination[stage * (stage_tokens / 2) +
                  group * (group_tokens / 2) + pair] =
          encode_e2m1_pair(value0, value1, dequant);
    }
  }
}

template <typename InputT>
__device__ __forceinline__ half load_narrowed_half(
    const InputT* input,
    int64_t offset);

template <>
__device__ __forceinline__ half load_narrowed_half<half>(
    const half* input,
    int64_t offset) {
  return input[offset];
}

template <>
__device__ __forceinline__ half load_narrowed_half<nv_bfloat16>(
    const nv_bfloat16* input,
    int64_t offset) {
  return __float2half_rn(__bfloat162float(input[offset]));
}

template <int Group>
__device__ __forceinline__ float subgroup_max(float value) {
  static_assert(Group == 16 || Group == 32);
#pragma unroll
  for (int delta = Group / 2; delta > 0; delta >>= 1) {
    value = fmaxf(
        value, __shfl_down_sync(0xffffffffU, value, delta, Group));
  }
  return __shfl_sync(0xffffffffU, value, 0, Group);
}

template <typename InputT, int QueryBlock, bool HasMaxPool, int HeadDim>
__global__ void prepare_h3_qk_microscaling_kernel(
    const InputT* __restrict__ query,
    const InputT* __restrict__ key,
    const int64_t* __restrict__ video_token_indices,
    const bool* __restrict__ video_slot_valid,
    const int32_t* __restrict__ video_valid_counts,
    const float* __restrict__ q_global_scale,
    const float* __restrict__ k_global_scale,
    half* __restrict__ q_pool,
    half* __restrict__ k_pool,
    half* __restrict__ q_max_pool,
    half* __restrict__ k_max_pool,
    half* __restrict__ packed_q,
    half* __restrict__ packed_k,
    uint8_t* __restrict__ q4,
    uint8_t* __restrict__ q4_scale,
    uint8_t* __restrict__ k4,
    uint8_t* __restrict__ k4_scale,
    uint8_t* __restrict__ q8,
    uint8_t* __restrict__ q8_scale,
    uint8_t* __restrict__ k8,
    uint8_t* __restrict__ k8_scale,
    int8_t* __restrict__ q_int8,
    float* __restrict__ q_int8_scale,
    int8_t* __restrict__ k_int8,
    float* __restrict__ k_int8_scale,
    int8_t* __restrict__ prefix_q_int8,
    float* __restrict__ prefix_q_int8_scale,
    bool has_nvfp4,
    bool has_int8,
    bool has_mxfp8,
    int64_t heads,
    int64_t video_blocks,
    int64_t prefix_tokens,
    int64_t prefix_blocks,
    int64_t video_physical_tokens,
    int64_t key_physical_tokens,
    int64_t q_stride_batch,
    int64_t q_stride_head,
    int64_t q_stride_token,
    int64_t k_stride_batch,
    int64_t k_stride_head,
    int64_t k_stride_token) {
  static_assert(QueryBlock == 64 || QueryBlock == 128);
  __shared__ int64_t staged_token_indices[128];
  __shared__ bool staged_slot_valid[128];
  __shared__ float int8_warp_amax[2][HeadDim / 32];
  __shared__ float int8_scale[2];

  const int64_t task = blockIdx.x;
  const int64_t batch = blockIdx.y;
  const int64_t task_group = task / heads;
  const int64_t head = task - task_group * heads;
  const int64_t prefix_query_blocks =
      (prefix_tokens + QueryBlock - 1) / QueryBlock;
  const int64_t prefix_query_capacity = prefix_query_blocks * QueryBlock;
  const bool is_video_query = task_group < video_blocks;
  const bool is_video_key =
      task_group >= video_blocks && task_group < 2 * video_blocks;
  const bool is_prefix_key =
      task_group >= 2 * video_blocks &&
      task_group < 2 * video_blocks + prefix_blocks;
  const bool is_prefix_query =
      task_group >= 2 * video_blocks + prefix_blocks;
  const bool is_query = is_video_query || is_prefix_query;
  const bool is_prefix = is_prefix_key || is_prefix_query;
  const int64_t logical_block = is_video_query
      ? task_group
      : (is_video_key ? task_group - video_blocks : -1);
  const int64_t prefix_block = is_prefix
      ? task_group - 2 * video_blocks -
          (is_prefix_query ? prefix_blocks : 0)
      : -1;
  const int task_tokens = is_prefix_key ? 64 : QueryBlock;
  const int64_t metadata_base = logical_block * QueryBlock;
  if (!is_prefix && threadIdx.x < task_tokens) {
    staged_token_indices[threadIdx.x] =
        video_token_indices[metadata_base + threadIdx.x];
    staged_slot_valid[threadIdx.x] =
        video_slot_valid[metadata_base + threadIdx.x];
  }
  __syncthreads();

  const int channel = threadIdx.x;
  const int nv_lane = channel & 15;
  const int mx_lane = channel & 31;
  const InputT* input = is_query ? query : key;
  const int64_t stride_batch = is_query ? q_stride_batch : k_stride_batch;
  const int64_t stride_head = is_query ? q_stride_head : k_stride_head;
  const int64_t stride_token = is_query ? q_stride_token : k_stride_token;
  half* packed = is_query ? packed_q : packed_k;
  half* pool = is_query ? q_pool : k_pool;
  half* max_pool = is_query ? q_max_pool : k_max_pool;
  uint8_t* nv_data = is_query ? q4 : k4;
  uint8_t* nv_scales = is_query ? q4_scale : k4_scale;
  uint8_t* mx_data = is_query ? q8 : k8;
  uint8_t* mx_scales = is_query ? q8_scale : k8_scale;
  int8_t* int8_data = is_prefix_query
      ? prefix_q_int8
      : (is_query ? q_int8 : k_int8);
  float* int8_scales = is_prefix_query
      ? prefix_q_int8_scale
      : (is_query ? q_int8_scale : k_int8_scale);
  const float tensor_scale = has_nvfp4
      ? (is_query ? q_global_scale[0] : k_global_scale[0])
      : 1.0f;
  const int64_t output_tokens = is_prefix_query
      ? prefix_query_capacity
      : (is_query ? video_physical_tokens : key_physical_tokens);
  float pool_sum = 0.0f;
  float pool_max = -CUDART_INF_F;
  float int8_amax[2] = {-1.0f, -1.0f};

  // Q128 metadata still needs 128 threads when HeadDim is 64.
  if (HeadDim == 128 || channel < HeadDim) {
#pragma unroll 1
    for (int token = 0; token < task_tokens; ++token) {
      const bool token_valid = is_prefix
          ? prefix_block * task_tokens + token < prefix_tokens
          : staged_slot_valid[token];
      const int64_t raw_token = is_prefix
          ? prefix_block * task_tokens + token
          : prefix_tokens + staged_token_indices[token];
      half narrowed = __float2half_rn(0.0f);
      if (token_valid) {
        const int64_t raw_offset = batch * stride_batch + head * stride_head +
            raw_token * stride_token + channel;
        narrowed = load_narrowed_half(input, raw_offset);
      }
      if (!is_prefix) {
        pool_sum += __half2float(narrowed);
        if constexpr (HasMaxPool) {
          if (token_valid) {
            pool_max = fmaxf(pool_max, __half2float(narrowed));
          }
        }
      }

      const int64_t natural_row = is_prefix_query
          ? prefix_block * QueryBlock + token
          : (is_video_query
                ? logical_block * QueryBlock + token
                : (is_video_key
                      ? prefix_blocks * 64 + logical_block * QueryBlock + token
                      : prefix_block * 64 + token));
      const int64_t natural_element =
          ((batch * heads + head) * output_tokens + natural_row) * HeadDim +
          channel;
      if (!is_prefix_query) packed[natural_element] = narrowed;
      const float value = __half2float(narrowed);

      if (has_int8) {
        const int group = !is_query && !is_prefix && QueryBlock == 128
            ? token / 64
            : 0;
        int8_amax[group] = fmaxf(int8_amax[group], fabsf(value));
      }

      if (has_nvfp4 && !is_prefix_query) {
        const float amax = subgroup_max<16>(fabsf(value));
        const uint8_t scale = nv_scale_bits(amax, tensor_scale);
        int64_t destination_row = natural_row;
        if (!is_query) {
          const int64_t local = natural_row & 31;
          destination_row = natural_row - local + (local / 8) * 2 +
              ((local % 8) / 2) * 8 + local % 2;
        }
        if (nv_lane == 0) {
          nv_scales[
              ((batch * heads + head) * output_tokens + destination_row) * (HeadDim / 16) +
              channel / 16] = scale;
        }
        const float dequant = decode_e4m3(scale) * tensor_scale;
        const float next_value = __shfl_down_sync(
            0xffffffffU, value, 1, 16);
        if ((nv_lane & 1) == 0) {
          nv_data[
              ((batch * heads + head) * output_tokens + destination_row) * (HeadDim / 2) +
              channel / 2] = encode_e2m1_pair(value, next_value, dequant);
        }
      }

      if (has_mxfp8 && !is_prefix_query) {
        const uint8_t scale = encode_e8m0(
            subgroup_max<32>(fabsf(value)));
        if (mx_lane == 0) {
          mx_scales[
              ((batch * heads + head) * output_tokens + natural_row) * (HeadDim / 32) +
              channel / 32] = scale;
        }
        mx_data[natural_element] = encode_e4m3(value, scale);
      }
    }
  }

  if (has_int8) {
    const int groups = !is_query && !is_prefix && QueryBlock == 128 ? 2 : 1;
    const int warp = channel / 32;
#pragma unroll
    for (int group = 0; group < 2; ++group) {
      if (group < groups && (HeadDim == 128 || channel < HeadDim)) {
        const float warp_amax = subgroup_max<32>(int8_amax[group]);
        if (mx_lane == 0) int8_warp_amax[group][warp] = warp_amax;
      }
    }
    __syncthreads();
    if (channel < groups) {
      float amax = int8_warp_amax[channel][0];
#pragma unroll
      for (int warp_index = 1; warp_index < HeadDim / 32; ++warp_index) {
        amax = fmaxf(amax, int8_warp_amax[channel][warp_index]);
      }
      const float scale = amax / 127.0f + 1.0e-7f;
      int8_scale[channel] = scale;
      const int64_t scale_block = is_prefix_query
          ? prefix_block
          : (is_query
                ? logical_block
                : (is_prefix
                      ? prefix_block
                      : prefix_blocks +
                          logical_block * (QueryBlock / 64) + channel));
      const int64_t scale_blocks = is_prefix_query
          ? prefix_query_blocks
          : (is_query ? video_blocks : key_physical_tokens / 64);
      int8_scales[(batch * heads + head) * scale_blocks + scale_block] = scale;
    }
    __syncthreads();

    if (HeadDim == 128 || channel < HeadDim) {
#pragma unroll 1
      for (int token = 0; token < task_tokens; ++token) {
        const bool token_valid = is_prefix
            ? prefix_block * task_tokens + token < prefix_tokens
            : staged_slot_valid[token];
        const int64_t raw_token = is_prefix
            ? prefix_block * task_tokens + token
            : prefix_tokens + staged_token_indices[token];
        half narrowed = __float2half_rn(0.0f);
        if (token_valid) {
          const int64_t raw_offset = batch * stride_batch + head * stride_head +
              raw_token * stride_token + channel;
          narrowed = load_narrowed_half(input, raw_offset);
        }
        const int64_t natural_row = is_prefix_query
            ? prefix_block * QueryBlock + token
            : (is_video_query
                  ? logical_block * QueryBlock + token
                  : (is_video_key
                        ? prefix_blocks * 64 + logical_block * QueryBlock + token
                        : prefix_block * 64 + token));
        const int64_t natural_element =
            ((batch * heads + head) * output_tokens + natural_row) * HeadDim +
            channel;
        const int group = !is_query && !is_prefix && QueryBlock == 128
            ? token / 64
            : 0;
        float quantized = __half2float(narrowed) / int8_scale[group];
        quantized += quantized >= 0.0f ? 0.5f : -0.5f;
        int8_data[natural_element] =
            static_cast<int8_t>(__float2int_rz(quantized));
      }
    }
  }

  if (!is_prefix && (HeadDim == 128 || channel < HeadDim)) {
    const int32_t valid_count = video_valid_counts[logical_block];
    const int64_t pool_offset =
        ((batch * heads + head) * video_blocks + logical_block) * HeadDim +
        channel;
    pool[pool_offset] = __float2half_rn(
        pool_sum / static_cast<float>(valid_count));
    if constexpr (HasMaxPool) {
      max_pool[pool_offset] = __float2half_rn(pool_max);
    }
  }
}

template <typename InputT, int HeadDim>
__global__ void prepare_h3_v_microscaling_kernel(
    const InputT* __restrict__ value,
    const int64_t* __restrict__ video_token_indices,
    const bool* __restrict__ video_slot_valid,
    const float* __restrict__ v_global_scale,
    half* __restrict__ packed_v,
    uint8_t* __restrict__ v4,
    uint8_t* __restrict__ v4_scale,
    uint8_t* __restrict__ v8,
    uint8_t* __restrict__ v8_scale,
    float* __restrict__ int8_v_stage_amax,
    bool has_nvfp4,
    bool has_int8,
    bool has_mxfp8,
    int64_t heads,
    int64_t prefix_tokens,
    int64_t prefix_blocks,
    int64_t key_physical_tokens,
    int64_t stride_batch,
    int64_t stride_head,
    int64_t stride_token) {
  constexpr int value_stage_tokens = 32;
  constexpr int value_stages = 2;
  constexpr int shared_stride = HeadDim + 1;
  __shared__ half value_tile[value_stage_tokens * shared_stride];
  __shared__ int64_t staged_token_indices[64];
  __shared__ bool staged_slot_valid[64];

  const int64_t physical_stage = blockIdx.x;
  const int64_t head = blockIdx.y;
  const int64_t batch = blockIdx.z;
  const bool is_prefix = physical_stage < prefix_blocks;
  const int64_t video_stage = physical_stage - prefix_blocks;
  if (!is_prefix && threadIdx.x < 64) {
    staged_token_indices[threadIdx.x] =
        video_token_indices[video_stage * 64 + threadIdx.x];
    staged_slot_valid[threadIdx.x] =
        video_slot_valid[video_stage * 64 + threadIdx.x];
  }
  __syncthreads();

  const int channel = threadIdx.x;
  const int lane = channel & 31;
  const int warp = channel >> 5;
  constexpr int warps = HeadDim / 32;
  const float tensor_scale = has_nvfp4 ? v_global_scale[0] : 1.0f;
  const int64_t head_batch = batch * heads + head;
  float int8_amax = -1.0f;

#pragma unroll
  for (int stage = 0; stage < value_stages; ++stage) {
    const int stage_token_base = stage * value_stage_tokens;
#pragma unroll 4
    for (int token_in_stage = 0;
         token_in_stage < value_stage_tokens;
         ++token_in_stage) {
      const int local_token = stage_token_base + token_in_stage;
      const bool token_valid = is_prefix
          ? physical_stage * 64 + local_token < prefix_tokens
          : staged_slot_valid[local_token];
      const int64_t raw_token = is_prefix
          ? physical_stage * 64 + local_token
          : prefix_tokens + staged_token_indices[local_token];
      half narrowed = __float2half_rn(0.0f);
      if (token_valid) {
        narrowed = load_narrowed_half(
            value,
            batch * stride_batch + head * stride_head +
                raw_token * stride_token + channel);
      }
      value_tile[token_in_stage * shared_stride + channel] = narrowed;
      if (has_int8) {
        int8_amax = fmaxf(int8_amax, fabsf(__half2float(narrowed)));
      }
      const int64_t output_token = physical_stage * 64 + local_token;
      packed_v[(head_batch * key_physical_tokens + output_token) * HeadDim +
               channel] = narrowed;
    }
    __syncthreads();

    if (has_nvfp4) {
      constexpr int Fp4Group = 16;
      constexpr int subgroups_per_warp = 2;
      const int subgroup = lane / Fp4Group;
      const int fp4_lane = lane & (Fp4Group - 1);
      constexpr int fp4_groups_per_stage = value_stage_tokens / Fp4Group;
      constexpr int total_groups = HeadDim * fp4_groups_per_stage;
      for (int group = warp * subgroups_per_warp + subgroup;
           group < total_groups;
           group += warps * subgroups_per_warp) {
        const int v_channel = group / fp4_groups_per_stage;
        const int token_group_in_stage = group % fp4_groups_per_stage;
        const int token_in_stage =
            token_group_in_stage * Fp4Group + fp4_lane;
        const float v_value = __half2float(
            value_tile[token_in_stage * shared_stride + v_channel]);
        const uint8_t scale = nv_scale_bits(
            subgroup_max<16>(fabsf(v_value)), tensor_scale);
        const int token_chunk = stage * fp4_groups_per_stage +
            token_group_in_stage;
        if (fp4_lane == 0) {
          v4_scale[(head_batch * (key_physical_tokens / 64) + physical_stage) *
                       (HeadDim * 4) +
                   v_channel * 4 + token_chunk] = scale;
        }
        const float dequant = decode_e4m3(scale) * tensor_scale;
        const float next_value = __shfl_down_sync(
            0xffffffffU, v_value, 1, 16);
        if ((fp4_lane & 1) == 0) {
          const int64_t output_token =
              physical_stage * 64 + stage_token_base + token_in_stage;
          v4[(head_batch * HeadDim + v_channel) *
                 (key_physical_tokens / 2) +
             output_token / 2] =
              encode_e2m1_pair(v_value, next_value, dequant);
        }
      }
    }

    if (has_mxfp8) {
      for (int group = warp; group < HeadDim; group += warps) {
        const int v_channel = group;
        const int local16 = lane & 15;
        const int natural = (lane / 16) * 16 + (local16 / 4) * 2 +
            ((local16 % 4) / 2) * 8 + local16 % 2;
        const float v_value = __half2float(
            value_tile[natural * shared_stride + v_channel]);
        const uint8_t scale = encode_e8m0(
            subgroup_max<32>(fabsf(v_value)));
        const int output_tile = v_channel / 8;
        const int lane_group = v_channel & 7;
        const int packed_row = (output_tile / 2) * 8 + lane_group;
        const int packed_col = (output_tile & 1) * 2 + stage;
        if (lane == 0) {
          v8_scale[(head_batch * (key_physical_tokens / 64) + physical_stage) *
                       (HeadDim * 2) +
                   packed_row * 4 + packed_col] = scale;
        }
        const int64_t output_token =
            physical_stage * 64 + stage_token_base + lane;
        v8[(head_batch * HeadDim + v_channel) * key_physical_tokens +
           output_token] = encode_e4m3(v_value, scale);
      }
    }

    if (stage + 1 < value_stages) {
      __syncthreads();
    }
  }

  if (has_int8) {
    int8_v_stage_amax[
        (head_batch * (key_physical_tokens / 64) + physical_stage) * HeadDim +
        channel] = int8_amax;
  }
}

// Historical fused-INT preparation keeps V's global per-channel scale in a
// second launch.  The donor producer supplies one exact K64-stage amax while
// this consumer performs the retained Sage/Sparge token permutation and FP8
// conversion without the old standalone transpose pass.
template <int HeadDim>
__global__ void prepare_h3_int8_v_from_partials_kernel(
    const half* __restrict__ packed_value,
    const float* __restrict__ stage_amax,
    __nv_fp8_e4m3* __restrict__ output,
    float* __restrict__ scale,
    int64_t heads,
    int64_t physical_stages,
    int64_t key_physical_tokens,
    int64_t padded_key_tokens) {
  constexpr int StageTokens = 64;
  constexpr int Threads = 256;
  constexpr int ChannelTile = 32;
  constexpr int PackSize = 8;
  constexpr int ChannelVectors = ChannelTile / PackSize;
  constexpr int ReductionLanes = Threads / ChannelTile;
  constexpr int StageStripes = 4;
  constexpr int SharedPitch = 72;
  constexpr float ScaleMax = 2.25f;
  constexpr float AllNanAmax = 1000000.0f;

  union __align__(16) SharedWorkspace {
    float reduction[Threads];
    half transposed[ChannelTile][SharedPitch];
  };
  __shared__ SharedWorkspace workspace;
  __shared__ float reciprocal[ChannelTile];

  const int thread = threadIdx.x;
  const int channel_tile = blockIdx.x / StageStripes;
  const int stage_stripe = blockIdx.x % StageStripes;
  const int head = blockIdx.y;
  const int batch = blockIdx.z;
  const int channel_base = channel_tile * ChannelTile;
  const int local_channel = thread % ChannelTile;
  const int reduction_lane = thread / ChannelTile;
  const int64_t head_batch = batch * heads + head;

  float thread_amax = -1.0f;
  const int64_t partial_base =
      head_batch * physical_stages * HeadDim + channel_base + local_channel;
  for (int64_t stage = reduction_lane; stage < physical_stages;
       stage += ReductionLanes) {
    thread_amax = fmaxf(
        thread_amax,
        stage_amax[partial_base + stage * HeadDim]);
  }
  workspace.reduction[thread] = thread_amax;
  __syncthreads();

  if (thread < ChannelTile) {
    float reduced_amax = -1.0f;
#pragma unroll
    for (int lane = 0; lane < ReductionLanes; ++lane) {
      reduced_amax = fmaxf(
          reduced_amax,
          workspace.reduction[lane * ChannelTile + thread]);
    }
    const float amax = reduced_amax < 0.0f ? AllNanAmax : reduced_amax;
    reciprocal[thread] = amax == 0.0f ? 0.0f : ScaleMax / amax;
    if (stage_stripe == 0) {
      scale[head_batch * HeadDim + channel_base + thread] = amax / ScaleMax;
    }
  }
  __syncthreads();

  const int natural_token = thread / ChannelVectors;
  const int channel_vector = thread % ChannelVectors;
  const int row_base = (natural_token / 16) * 16;
  const int row_mod = natural_token % 16;
  const int permuted_row =
      row_base + (row_mod / 8) * 2 + ((row_mod / 2) % 4) * 4 + row_mod % 2;
  const int output_channel = thread / (StageTokens / PackSize);
  const int output_token = (thread % (StageTokens / PackSize)) * PackSize;

#pragma unroll 1
  for (int64_t stage = stage_stripe; stage < padded_key_tokens / StageTokens;
       stage += StageStripes) {
    half values[PackSize];
    if (stage < physical_stages) {
      const int64_t input_offset =
          (head_batch * key_physical_tokens + stage * StageTokens +
           natural_token) * HeadDim + channel_base + channel_vector * PackSize;
      *reinterpret_cast<uint4*>(values) =
          *reinterpret_cast<const uint4*>(packed_value + input_offset);
    } else {
#pragma unroll
      for (int element = 0; element < PackSize; ++element) values[element] = 0;
    }
#pragma unroll
    for (int element = 0; element < PackSize; ++element) {
      workspace.transposed[channel_vector * PackSize + element][permuted_row] =
          values[element];
    }
    __syncthreads();

    *reinterpret_cast<uint4*>(values) = *reinterpret_cast<const uint4*>(
        workspace.transposed[output_channel] + output_token);
    const float multiplier = reciprocal[output_channel];
    float converted[PackSize];
#pragma unroll
    for (int element = 0; element < PackSize; ++element) {
      converted[element] = multiplier == 0.0f
          ? 0.0f
          : __half2float(values[element]) * multiplier;
    }
    uint32_t fp8_words[2];
    floatx4_to_e4m3x4(fp8_words, converted, converted + 2);
    floatx4_to_e4m3x4(fp8_words + 1, converted + 4, converted + 6);
    const int64_t output_offset =
        (head_batch * HeadDim + channel_base + output_channel) *
            padded_key_tokens +
        stage * StageTokens + output_token;
    *reinterpret_cast<uint2*>(output + output_offset) =
        *reinterpret_cast<const uint2*>(fp8_words);
    __syncthreads();
  }
}
} // namespace

void prepare_mxfp8(TensorView query, TensorView key, TensorView value,
                   const SixOutputs& outputs, cudaStream_t stream) {
  const auto& [q8, q8_scale, k8, k8_scale, v8, v8_scale] = outputs;
  const int64_t head_dim = query.size(3);
  const int64_t q_rows = query.numel() / head_dim;
  const int64_t k_rows = key.numel() / head_dim;
  const auto launch = [&](auto head_dim_tag) {
    constexpr int HeadDim = decltype(head_dim_tag)::value;
    prepare_mxfp8_qk_kernel<HeadDim><<<q_rows, HeadDim, 0, stream>>>(
        reinterpret_cast<const half*>(query.data_ptr<half>()),
        q8.data_ptr<uint8_t>(), q8_scale.data_ptr<uint8_t>(), q_rows);
    prepare_mxfp8_qk_kernel<HeadDim><<<k_rows, HeadDim, 0, stream>>>(
        reinterpret_cast<const half*>(key.data_ptr<half>()),
        k8.data_ptr<uint8_t>(), k8_scale.data_ptr<uint8_t>(), k_rows);
    const dim3 v_grid(
        static_cast<uint32_t>(value.size(2) / 64),
        static_cast<uint32_t>(value.size(1)),
        static_cast<uint32_t>(value.size(0)));
    prepare_mxfp8_v_kernel<HeadDim><<<v_grid, HeadDim, 0, stream>>>(
        reinterpret_cast<const half*>(value.data_ptr<half>()),
        v8.data_ptr<uint8_t>(), v8_scale.data_ptr<uint8_t>(),
        value.size(1), value.size(2));
  };
  if (head_dim == 64) {
    launch(std::integral_constant<int, 64>{});
  } else {
    launch(std::integral_constant<int, 128>{});
  }
  check_launch();
  check_launch();
}

template <uint32_t QueryBlock>
void prepare_nvfp4(TensorView query, TensorView key, TensorView value,
                  TensorView q_global_scale, TensorView k_global_scale,
                  TensorView v_global_scale, const SixOutputs& outputs,
                  cudaStream_t stream) {
  static_assert(QueryBlock == 64 || QueryBlock == 128);
  const auto& [q4, q4_scale, k4, k4_scale, v4, v4_scale] = outputs;
  const int64_t head_dim = query.size(3);
  const int64_t q_rows = query.numel() / head_dim;
  const int64_t k_rows = key.numel() / head_dim;
  const auto launch = [&](auto head_dim_tag) {
    constexpr int HeadDim = decltype(head_dim_tag)::value;
    prepare_nvfp4_qk_kernel<false, HeadDim><<<q_rows, HeadDim / 2, 0, stream>>>(
        reinterpret_cast<const half*>(query.data_ptr<half>()),
        q4.data_ptr<uint8_t>(), q4_scale.data_ptr<uint8_t>(),
        q_global_scale.data_ptr<float>(), q_rows);
    prepare_nvfp4_qk_kernel<true, HeadDim><<<k_rows, HeadDim / 2, 0, stream>>>(
        reinterpret_cast<const half*>(key.data_ptr<half>()),
        k4.data_ptr<uint8_t>(), k4_scale.data_ptr<uint8_t>(),
        k_global_scale.data_ptr<float>(), k_rows);
    const dim3 v_grid(
        static_cast<uint32_t>(value.size(2) / 64),
        static_cast<uint32_t>(value.size(1)),
        static_cast<uint32_t>(value.size(0)));
    prepare_nvfp4_v_kernel<HeadDim><<<v_grid, HeadDim, 0, stream>>>(
        reinterpret_cast<const half*>(value.data_ptr<half>()),
        v4.data_ptr<uint8_t>(), v4_scale.data_ptr<uint8_t>(),
        v_global_scale.data_ptr<float>(), value.size(1), value.size(2));
  };
  if (head_dim == 64) {
    launch(std::integral_constant<int, 64>{});
  } else {
    launch(std::integral_constant<int, 128>{});
  }
  check_launch();
}

void prepare_q64_nvfp4(TensorView query, TensorView key, TensorView value,
                        TensorView q_global_scale, TensorView k_global_scale,
                        TensorView v_global_scale, const SixOutputs& outputs,
                        cudaStream_t stream) {
  prepare_nvfp4<64>(query, key, value, q_global_scale, k_global_scale,
                       v_global_scale, outputs, stream);
}

void prepare_q128_nvfp4(TensorView query, TensorView key, TensorView value,
                        TensorView q_global_scale, TensorView k_global_scale,
                        TensorView v_global_scale, const SixOutputs& outputs,
                        cudaStream_t stream) {
  prepare_nvfp4<128>(query, key, value, q_global_scale, k_global_scale,
                       v_global_scale, outputs, stream);
}

void prepare_h3_sm120_operands(TensorView query, TensorView key, TensorView value,
    TensorView video_token_indices, TensorView video_slot_valid,
    TensorView video_valid_counts, H3Options options,
    TensorView q_global_scale, TensorView k_global_scale, TensorView v_global_scale,
    const H3Outputs& outputs, TensorView workspace, cudaStream_t stream) {
  const auto prefix_tokens = options.prefix_tokens;
  const auto query_block_size = options.query_block_size;
  const auto has_nvfp4 = options.has_nvfp4;
  const auto has_int8 = options.has_int8;
  const auto has_mxfp8 = options.has_mxfp8;
  const auto has_prefix_query_int8 = options.has_prefix_query_int8;
  const auto has_maxpool = options.has_maxpool;
  // Donor always materializes narrowed packed FP16 operands (has_fp16 ignored).
  const auto q_pool = outputs[0], k_pool = outputs[1];
  const auto packed_q = outputs[2], packed_k = outputs[3], packed_v = outputs[4];
  const auto q4 = outputs[5], q4_scale = outputs[6];
  const auto k4 = outputs[7], k4_scale = outputs[8];
  const auto v4 = outputs[9], v4_scale = outputs[10];
  const auto q8 = outputs[11], q8_scale = outputs[12];
  const auto k8 = outputs[13], k8_scale = outputs[14];
  const auto v8 = outputs[15], v8_scale = outputs[16];
  const auto q_int8 = outputs[17], q_int8_scale = outputs[18];
  const auto k_int8 = outputs[19], k_int8_scale = outputs[20];
  const auto v_int8 = outputs[21], v_int8_scale = outputs[22];
  const auto prefix_q_int8 = outputs[23], prefix_q_int8_scale = outputs[24];
  const auto q_max_pool = outputs[25], k_max_pool = outputs[26];
  const auto v_int8_stage_amax = workspace;
  const int64_t video_physical_tokens = video_token_indices.numel();
  const int64_t video_blocks = video_physical_tokens / query_block_size;
  const int64_t prefix_blocks = (prefix_tokens + 63) / 64;
  const int64_t prefix_query_blocks = (prefix_tokens + query_block_size - 1) / query_block_size;
  const int64_t key_physical_tokens = prefix_blocks * 64 + video_physical_tokens;
  const int64_t padded_key_tokens = ((key_physical_tokens + 127) / 128) * 128;
  const float* q_scale = q_global_scale.data_ptr<float>();
  const float* k_scale = k_global_scale.data_ptr<float>();
  const float* v_scale = v_global_scale.data_ptr<float>();
  const int64_t qk_tasks = query.size(1) * (
      2 * video_blocks + prefix_blocks +
      (has_prefix_query_int8 ? prefix_query_blocks : 0));
  const auto launch = [&](auto head_dim_tag, auto query_block_tag, auto* input_tag, auto max_tag) {
    constexpr int HeadDim = decltype(head_dim_tag)::value;
    constexpr int QueryBlock = decltype(query_block_tag)::value;
    using InputT = std::remove_pointer_t<decltype(input_tag)>;
    constexpr bool HasMaxPool = decltype(max_tag)::value;
    const dim3 qk_grid(
        static_cast<uint32_t>(qk_tasks),
        static_cast<uint32_t>(query.size(0)));
    prepare_h3_qk_microscaling_kernel<InputT, QueryBlock, HasMaxPool, HeadDim><<<
        qk_grid, 128, 0, stream>>>(
        reinterpret_cast<const InputT*>(query.data_ptr()),
        reinterpret_cast<const InputT*>(key.data_ptr()),
        video_token_indices.data_ptr<int64_t>(),
        video_slot_valid.data_ptr<bool>(),
        video_valid_counts.data_ptr<int32_t>(),
        q_scale,
        k_scale,
        reinterpret_cast<half*>(q_pool.data_ptr<half>()),
        reinterpret_cast<half*>(k_pool.data_ptr<half>()),
        HasMaxPool
            ? reinterpret_cast<half*>(q_max_pool.data_ptr<half>())
            : nullptr,
        HasMaxPool
            ? reinterpret_cast<half*>(k_max_pool.data_ptr<half>())
            : nullptr,
        reinterpret_cast<half*>(packed_q.data_ptr<half>()),
        reinterpret_cast<half*>(packed_k.data_ptr<half>()),
        has_nvfp4 ? q4.data_ptr<uint8_t>() : nullptr,
        has_nvfp4 ? q4_scale.data_ptr<uint8_t>() : nullptr,
        has_nvfp4 ? k4.data_ptr<uint8_t>() : nullptr,
        has_nvfp4 ? k4_scale.data_ptr<uint8_t>() : nullptr,
        has_mxfp8 ? q8.data_ptr<uint8_t>() : nullptr,
        has_mxfp8 ? q8_scale.data_ptr<uint8_t>() : nullptr,
        has_mxfp8 ? k8.data_ptr<uint8_t>() : nullptr,
        has_mxfp8 ? k8_scale.data_ptr<uint8_t>() : nullptr,
        has_int8 ? q_int8.data_ptr<int8_t>() : nullptr,
        has_int8 ? q_int8_scale.data_ptr<float>() : nullptr,
        has_int8 ? k_int8.data_ptr<int8_t>() : nullptr,
        has_int8 ? k_int8_scale.data_ptr<float>() : nullptr,
        has_prefix_query_int8 ? prefix_q_int8.data_ptr<int8_t>() : nullptr,
        has_prefix_query_int8
            ? prefix_q_int8_scale.data_ptr<float>()
            : nullptr,
        has_nvfp4,
        has_int8,
        has_mxfp8,
        query.size(1),
        video_blocks,
        prefix_tokens,
        prefix_blocks,
        video_physical_tokens,
        key_physical_tokens,
        query.stride(0), query.stride(1), query.stride(2),
        key.stride(0), key.stride(1), key.stride(2));
    check_launch();

    const dim3 v_grid(
        static_cast<uint32_t>(key_physical_tokens / 64),
        static_cast<uint32_t>(value.size(1)),
        static_cast<uint32_t>(value.size(0)));
    prepare_h3_v_microscaling_kernel<InputT, HeadDim><<<v_grid, HeadDim, 0, stream>>>(
        reinterpret_cast<const InputT*>(value.data_ptr()),
        video_token_indices.data_ptr<int64_t>(),
        video_slot_valid.data_ptr<bool>(),
        v_scale,
        reinterpret_cast<half*>(packed_v.data_ptr<half>()),
        has_nvfp4 ? v4.data_ptr<uint8_t>() : nullptr,
        has_nvfp4 ? v4_scale.data_ptr<uint8_t>() : nullptr,
        has_mxfp8 ? v8.data_ptr<uint8_t>() : nullptr,
        has_mxfp8 ? v8_scale.data_ptr<uint8_t>() : nullptr,
        has_int8 ? v_int8_stage_amax.data_ptr<float>() : nullptr,
        has_nvfp4,
        has_int8,
        has_mxfp8,
        value.size(1),
        prefix_tokens,
        prefix_blocks,
        key_physical_tokens,
        value.stride(0), value.stride(1), value.stride(2));
    check_launch();

    if (has_int8) {
      constexpr int stage_stripes = 4;
      constexpr int channel_tiles = HeadDim / 32;
      const dim3 int8_v_grid(
          channel_tiles * stage_stripes,
          static_cast<uint32_t>(value.size(1)),
          static_cast<uint32_t>(value.size(0)));
      prepare_h3_int8_v_from_partials_kernel<HeadDim><<<
          int8_v_grid, 256, 0, stream>>>(
          reinterpret_cast<const half*>(packed_v.data_ptr<half>()),
          v_int8_stage_amax.data_ptr<float>(),
          reinterpret_cast<__nv_fp8_e4m3*>(v_int8.data_ptr()),
          v_int8_scale.data_ptr<float>(),
          value.size(1),
          key_physical_tokens / 64,
          key_physical_tokens,
          padded_key_tokens);
      check_launch();
    }
  };
  const auto dispatch_input = [&](auto head_dim_tag, auto* input_tag) {
    if (query_block_size == 64) {
      if (has_maxpool) {
        launch(head_dim_tag, std::integral_constant<int, 64>{}, input_tag, std::true_type{});
      } else {
        launch(head_dim_tag, std::integral_constant<int, 64>{}, input_tag, std::false_type{});
      }
    } else {
      if (has_maxpool) {
        launch(head_dim_tag, std::integral_constant<int, 128>{}, input_tag, std::true_type{});
      } else {
        launch(head_dim_tag, std::integral_constant<int, 128>{}, input_tag, std::false_type{});
      }
    }
  };
  const auto dispatch_head_dim = [&](auto head_dim_tag) {
    if (query.dtype == DType::F16) {
      dispatch_input(head_dim_tag, static_cast<half*>(nullptr));
    } else {
      dispatch_input(head_dim_tag, static_cast<nv_bfloat16*>(nullptr));
    }
  };
  if (query.size(3) == 64) {
    dispatch_head_dim(std::integral_constant<int, 64>{});
  } else {
    dispatch_head_dim(std::integral_constant<int, 128>{});
  }

}
} // namespace draft::sm120::preparation
