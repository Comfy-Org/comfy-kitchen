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

/*
 * Unified apply-RoPE and RMSNorm+RoPE kernel family. All logical dimensions
 * are addressed through element strides; the contiguous-head specialization
 * contains no runtime layout branch.
 */
#include "dtype_dispatch.cuh"
#include "rope_device.cuh"
#include "tensor.h"
#include "utils.cuh"

#include <cstdint>
#include <type_traits>

namespace comfy {
namespace {

constexpr int kWarpsPerBlock = 4;
constexpr int kThreads = kWarpsPerBlock * kThreadsPerWarp;

using TensorArg1 = tensor::TensorArg<1>;
using TensorArg4 = tensor::TensorArg<4>;
using TensorArg6 = tensor::TensorArg<6>;

template <typename InputType, typename FreqsType, typename ScaleType,
          bool HasRms, bool SplitHalf, bool HasK, bool InPlace, bool ContigHead>
__global__ __launch_bounds__(kThreads) void rope_kernel(
    TensorArg4 q_arg, TensorArg4 k_arg, TensorArg6 freqs_arg,
    TensorArg1 q_scale_arg, TensorArg1 k_scale_arg, TensorArg4 q_out_arg,
    TensorArg4 k_out_arg, int rot_dim, float epsilon) {
  using ComputeType = std::conditional_t<HasRms, float, FreqsType>;

  const auto *q = static_cast<const InputType *>(q_arg.data);
  const auto *k = static_cast<const InputType *>(k_arg.data);
  const auto *freqs = static_cast<const FreqsType *>(freqs_arg.data);
  const auto *q_scale = static_cast<const ScaleType *>(q_scale_arg.data);
  const auto *k_scale = static_cast<const ScaleType *>(k_scale_arg.data);
  auto *q_out = static_cast<InputType *>(q_out_arg.data);
  auto *k_out = static_cast<InputType *>(k_out_arg.data);

  const int64_t dim1 = q_arg.meta.sizes[1];
  const int64_t dim2 = q_arg.meta.sizes[2];
  const int head_dim = static_cast<int>(q_arg.meta.sizes[3]);

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  const int64_t rows =
      q_arg.meta.sizes[0] * q_arg.meta.sizes[1] * q_arg.meta.sizes[2];
  if (row >= rows) {
    return;
  }

  const int64_t i2 = row % dim2;
  const int64_t tmp = row / dim2;
  const int64_t i1 = tmp % dim1;
  const int64_t i0 = tmp / dim1;
  const int64_t q_base =
      i0 * q_arg.meta.strides[0] + i1 * q_arg.meta.strides[1] +
      i2 * q_arg.meta.strides[2];
  const int64_t q_out_base =
      InPlace ? q_base
              : i0 * q_out_arg.meta.strides[0] +
                    i1 * q_out_arg.meta.strides[1] +
                    i2 * q_out_arg.meta.strides[2];
  int64_t k_base = 0;
  int64_t k_out_base = 0;
  if constexpr (HasK) {
    k_base = i0 * k_arg.meta.strides[0] + i1 * k_arg.meta.strides[1] +
             i2 * k_arg.meta.strides[2];
    k_out_base =
        InPlace ? k_base
                : i0 * k_out_arg.meta.strides[0] +
                      i1 * k_out_arg.meta.strides[1] +
                      i2 * k_out_arg.meta.strides[2];
  }

  float q_rrms = 1.0f;
  float k_rrms = 1.0f;
  if constexpr (HasRms) {
    const float q_sum =
        rope::rms_sum<InputType, ContigHead>(q + q_base, head_dim, q_arg.meta.strides[3], lane);
    q_rrms = rsqrtf(q_sum / static_cast<float>(head_dim) + epsilon);
    if constexpr (HasK) {
      const float k_sum = rope::rms_sum<InputType, ContigHead>(
          k + k_base, head_dim, k_arg.meta.strides[3], lane);
      k_rrms = rsqrtf(k_sum / static_cast<float>(head_dim) + epsilon);
    }
  }

  const int64_t freq_row =
      (freqs_arg.meta.sizes[0] == 1 ? 0 : i0) * freqs_arg.meta.strides[0] +
      (freqs_arg.meta.sizes[1] == 1 ? 0 : i1) * freqs_arg.meta.strides[1] +
      (freqs_arg.meta.sizes[2] == 1 ? 0 : i2) * freqs_arg.meta.strides[2];
  // Rotation covers the first rot_dim dims (split-half pairs (i, i + rot_dim/2));
  // the RMS reduction above always spans the full head_dim.
  const int pairs = rot_dim / 2;
  constexpr int kPairsPerLane = SplitHalf && ContigHead ? 2 : 1;

  for (int pair_base = lane * kPairsPerLane; pair_base < pairs;
       pair_base += kThreadsPerWarp * kPairsPerLane) {
    InputType q0_raw[kPairsPerLane], q1_raw[kPairsPerLane];
    InputType k0_raw[kPairsPerLane], k1_raw[kPairsPerLane];

    if constexpr (SplitHalf && ContigHead) {
      const auto q_lo =
          *reinterpret_cast<const rope::Pair<InputType> *>(q + q_base + pair_base);
      const auto q_hi = *reinterpret_cast<const rope::Pair<InputType> *>(
          q + q_base + pairs + pair_base);
      q0_raw[0] = q_lo.x;
      q0_raw[1] = q_lo.y;
      q1_raw[0] = q_hi.x;
      q1_raw[1] = q_hi.y;
      if constexpr (HasK) {
        const auto k_lo = *reinterpret_cast<const rope::Pair<InputType> *>(
            k + k_base + pair_base);
        const auto k_hi = *reinterpret_cast<const rope::Pair<InputType> *>(
            k + k_base + pairs + pair_base);
        k0_raw[0] = k_lo.x;
        k0_raw[1] = k_lo.y;
        k1_raw[0] = k_hi.x;
        k1_raw[1] = k_hi.y;
      }
    } else {
      rope::load_head_pair<InputType, SplitHalf, ContigHead>(
          q + q_base, pair_base, pairs, q_arg.meta.strides[3], q0_raw[0], q1_raw[0]);
      if constexpr (HasK) {
        rope::load_head_pair<InputType, SplitHalf, ContigHead>(
            k + k_base, pair_base, pairs, k_arg.meta.strides[3], k0_raw[0], k1_raw[0]);
      }
    }

    InputType qo0_raw[kPairsPerLane], qo1_raw[kPairsPerLane];
    InputType ko0_raw[kPairsPerLane], ko1_raw[kPairsPerLane];
#pragma unroll
    for (int p = 0; p < kPairsPerLane; ++p) {
      const int pair = pair_base + p;
      ComputeType q0 = static_cast<ComputeType>(q0_raw[p]);
      ComputeType q1 = static_cast<ComputeType>(q1_raw[p]);
      const int first = SplitHalf ? pair : pair * 2;
      const int second = SplitHalf ? pair + pairs : first + 1;
      if constexpr (HasRms) {
        q0 = static_cast<float>(static_cast<InputType>(
            static_cast<float>(q0) * q_rrms *
            static_cast<float>(
                q_scale[static_cast<int64_t>(first) * q_scale_arg.meta.strides[0]])));
        q1 = static_cast<float>(static_cast<InputType>(
            static_cast<float>(q1) * q_rrms *
            static_cast<float>(
                q_scale[static_cast<int64_t>(second) * q_scale_arg.meta.strides[0]])));
      }

      FreqsType f00_raw, f01_raw, f10_raw, f11_raw;
      rope::load_rotation(
          freqs,
          freq_row + static_cast<int64_t>(pair) * freqs_arg.meta.strides[3],
          freqs_arg.meta.strides[4], freqs_arg.meta.strides[5], f00_raw,
          f01_raw, f10_raw, f11_raw);
      const ComputeType f00 = static_cast<ComputeType>(f00_raw);
      const ComputeType f01 = static_cast<ComputeType>(f01_raw);
      const ComputeType f10 = static_cast<ComputeType>(f10_raw);
      const ComputeType f11 = static_cast<ComputeType>(f11_raw);
      ComputeType qo0, qo1;
      rope::rotate(q0, q1, f00, f01, f10, f11, qo0, qo1);
      qo0_raw[p] = static_cast<InputType>(qo0);
      qo1_raw[p] = static_cast<InputType>(qo1);

      if constexpr (HasK) {
        ComputeType k0 = static_cast<ComputeType>(k0_raw[p]);
        ComputeType k1 = static_cast<ComputeType>(k1_raw[p]);
        if constexpr (HasRms) {
          k0 = static_cast<float>(static_cast<InputType>(
              static_cast<float>(k0) * k_rrms *
              static_cast<float>(
                  k_scale[static_cast<int64_t>(first) * k_scale_arg.meta.strides[0]])));
          k1 = static_cast<float>(static_cast<InputType>(
              static_cast<float>(k1) * k_rrms *
              static_cast<float>(
                  k_scale[static_cast<int64_t>(second) * k_scale_arg.meta.strides[0]])));
        }
        ComputeType ko0, ko1;
        rope::rotate(k0, k1, f00, f01, f10, f11, ko0, ko1);
        ko0_raw[p] = static_cast<InputType>(ko0);
        ko1_raw[p] = static_cast<InputType>(ko1);
      }
    }

    if constexpr (SplitHalf && ContigHead) {
      *reinterpret_cast<rope::Pair<InputType> *>(
          q_out + q_out_base + pair_base) = {qo0_raw[0], qo0_raw[1]};
      *reinterpret_cast<rope::Pair<InputType> *>(
          q_out + q_out_base + pairs + pair_base) = {qo1_raw[0], qo1_raw[1]};
      if constexpr (HasK) {
        *reinterpret_cast<rope::Pair<InputType> *>(
            k_out + k_out_base + pair_base) = {ko0_raw[0], ko0_raw[1]};
        *reinterpret_cast<rope::Pair<InputType> *>(
            k_out + k_out_base + pairs + pair_base) = {ko1_raw[0], ko1_raw[1]};
      }
    } else {
      rope::store_head_pair<InputType, SplitHalf, ContigHead>(
          q_out + q_out_base, pair_base, pairs,
          InPlace ? q_arg.meta.strides[3] : q_out_arg.meta.strides[3],
          qo0_raw[0], qo1_raw[0]);
      if constexpr (HasK) {
        rope::store_head_pair<InputType, SplitHalf, ContigHead>(
            k_out + k_out_base, pair_base, pairs,
            InPlace ? k_arg.meta.strides[3] : k_out_arg.meta.strides[3], ko0_raw[0], ko1_raw[0]);
      }
    }
  }

  // Norm-only tail: dims beyond rot_dim are normalized and scaled but never
  // rotated. Empty in the common rot_dim == head_dim case.
  const int64_t q_out_stride =
      InPlace ? q_arg.meta.strides[3] : q_out_arg.meta.strides[3];
  const int64_t k_out_stride =
      InPlace ? k_arg.meta.strides[3] : k_out_arg.meta.strides[3];
  for (int d = rot_dim + lane; d < head_dim; d += kThreadsPerWarp) {
    InputType qv = q[q_base + static_cast<int64_t>(d) * q_arg.meta.strides[3]];
    if constexpr (HasRms) {
      qv = static_cast<InputType>(
          static_cast<float>(qv) * q_rrms *
          static_cast<float>(
              q_scale[static_cast<int64_t>(d) * q_scale_arg.meta.strides[0]]));
    }
    q_out[q_out_base + static_cast<int64_t>(d) * q_out_stride] = qv;
    if constexpr (HasK) {
      InputType kv = k[k_base + static_cast<int64_t>(d) * k_arg.meta.strides[3]];
      if constexpr (HasRms) {
        kv = static_cast<InputType>(
            static_cast<float>(kv) * k_rrms *
            static_cast<float>(
                k_scale[static_cast<int64_t>(d) * k_scale_arg.meta.strides[0]]));
      }
      k_out[k_out_base + static_cast<int64_t>(d) * k_out_stride] = kv;
    }
  }
}

template <typename T>
bool pair_aligned(const T *ptr, int64_t s0, int64_t s1, int64_t s2) {
  return reinterpret_cast<uintptr_t>(ptr) % alignof(rope::Pair<T>) == 0 &&
         s0 % 2 == 0 && s1 % 2 == 0 && s2 % 2 == 0;
}

template <typename InputType, typename FreqsType, typename ScaleType,
          bool HasRms, bool SplitHalf, bool HasK, bool InPlace, bool ContigHead>
void launch_config(
    TensorArg4 q, TensorArg4 k, TensorArg6 freqs, TensorArg1 q_scale,
    TensorArg1 k_scale, TensorArg4 q_out, TensorArg4 k_out, int rot_dim,
    float epsilon, cudaStream_t stream) {
  const int64_t rows =
      q.meta.sizes[0] * q.meta.sizes[1] * q.meta.sizes[2];
  if (rows == 0) {
    return;
  }
  const int blocks = static_cast<int>((rows + kWarpsPerBlock - 1) /
                                      kWarpsPerBlock);
  rope_kernel<InputType, FreqsType, ScaleType, HasRms, SplitHalf, HasK, InPlace,
              ContigHead><<<blocks, kThreads, 0, stream>>>(
      q, k, freqs, q_scale, k_scale, q_out, k_out, rot_dim, epsilon);
}

template <typename InputType, typename FreqsType, typename ScaleType,
          bool HasRms>
void rope_launcher(
    TensorArg4 q, TensorArg4 k, TensorArg6 freqs, TensorArg1 q_scale,
    TensorArg1 k_scale, TensorArg4 q_out, TensorArg4 k_out, int rot_dim,
    float epsilon, bool has_k, bool split_half, cudaStream_t stream) {
  const bool inplace =
      q.data == q_out.data && (!has_k || k.data == k_out.data);
  bool contig = q.meta.strides[3] == 1 && q_out.meta.strides[3] == 1 &&
                pair_aligned(static_cast<const InputType *>(q.data),
                             q.meta.strides[0], q.meta.strides[1],
                             q.meta.strides[2]) &&
                pair_aligned(static_cast<InputType *>(q_out.data),
                             q_out.meta.strides[0],
                             q_out.meta.strides[1], q_out.meta.strides[2]);
  if (has_k) {
    contig = contig && k.meta.strides[3] == 1 &&
             k_out.meta.strides[3] == 1 &&
             pair_aligned(static_cast<const InputType *>(k.data),
                          k.meta.strides[0], k.meta.strides[1], k.meta.strides[2]) &&
             pair_aligned(static_cast<InputType *>(k_out.data),
                          k_out.meta.strides[0],
                          k_out.meta.strides[1], k_out.meta.strides[2]);
  }
  contig = contig &&
           (!split_half || (q.meta.sizes[3] % 4 == 0 && rot_dim % 4 == 0));

#define LAUNCH(HAS_K, SPLIT, INPLACE, CONTIG)                                  \
  launch_config<InputType, FreqsType, ScaleType, HasRms, SPLIT, HAS_K,         \
                INPLACE, CONTIG>(q, k, freqs, q_scale, k_scale, q_out, k_out,   \
                                 rot_dim, epsilon, stream)
#define DISPATCH_LAYOUT(HAS_K, SPLIT)                                           \
  if (inplace) {                                                                \
    if (contig) LAUNCH(HAS_K, SPLIT, true, true);                              \
    else LAUNCH(HAS_K, SPLIT, true, false);                                    \
  } else {                                                                      \
    if (contig) LAUNCH(HAS_K, SPLIT, false, true);                             \
    else LAUNCH(HAS_K, SPLIT, false, false);                                   \
  }
  if (has_k) {
    if (split_half) {
      DISPATCH_LAYOUT(true, true)
    } else {
      DISPATCH_LAYOUT(true, false)
    }
  } else if (split_half) {
    DISPATCH_LAYOUT(false, true)
  } else {
    DISPATCH_LAYOUT(false, false)
  }
#undef DISPATCH_LAYOUT
#undef LAUNCH
  CUDA_CHECK(cudaGetLastError());
}

} // namespace
} // namespace comfy

extern "C" void launch_apply_rope_kernel(
    comfy::tensor::TensorArg<4> q, comfy::tensor::TensorArg<4> k,
    comfy::tensor::TensorArg<6> freqs, comfy::tensor::TensorArg<4> q_out,
    comfy::tensor::TensorArg<4> k_out, bool has_k, bool split_half,
    cudaStream_t stream) {
  DISPATCH_HALF_INPUT_FP_FREQS_DTYPES(
      static_cast<int>(q.meta.dtype), static_cast<int>(freqs.meta.dtype),
      InputType, FreqsType, [&] {
    comfy::rope_launcher<InputType, FreqsType, float, false>(
        q, k, freqs, {}, {}, q_out, k_out,
        static_cast<int>(q.meta.sizes[3]), 0.0f, has_k, split_half, stream);
  });
}

extern "C" void launch_rms_rope_kernel(
    comfy::tensor::TensorArg<4> q, comfy::tensor::TensorArg<4> k,
    comfy::tensor::TensorArg<6> freqs, comfy::tensor::TensorArg<1> q_scale,
    comfy::tensor::TensorArg<1> k_scale, comfy::tensor::TensorArg<4> q_out,
    comfy::tensor::TensorArg<4> k_out, int64_t rot_dim, float epsilon,
    bool has_k, bool split_half, cudaStream_t stream) {
  DISPATCH_HALF_DTYPE(static_cast<int>(q.meta.dtype), InputType, [&] {
    DISPATCH_FP_DTYPE(static_cast<int>(freqs.meta.dtype), FreqsType, [&] {
      DISPATCH_FP_DTYPE(static_cast<int>(q_scale.meta.dtype), ScaleType, [&] {
        comfy::rope_launcher<InputType, FreqsType, ScaleType, true>(
            q, k, freqs, q_scale, k_scale, q_out, k_out,
            static_cast<int>(rot_dim), epsilon, has_k, split_half, stream);
      });
    });
  });
}
