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
#include "utils.cuh"

#include <cstdint>
#include <type_traits>

namespace comfy {
namespace {

constexpr int kWarpsPerBlock = 4;
constexpr int kThreads = kWarpsPerBlock * kThreadsPerWarp;

template <typename InputType, typename FreqsType, typename ScaleType,
          bool HasRms, bool SplitHalf, bool HasK, bool InPlace, bool ContigHead>
__global__ __launch_bounds__(kThreads) void rope_kernel(
    const InputType *q, const InputType *k,
    const FreqsType *__restrict__ freqs,
    const ScaleType *__restrict__ q_scale,
    const ScaleType *__restrict__ k_scale, InputType *q_out,
    InputType *k_out, int64_t batch, int64_t dim1, int64_t dim2,
    int head_dim, int rot_dim, int64_t freqs_batch, int64_t freqs_dim1,
    int64_t freqs_dim2, int64_t q_s0, int64_t q_s1, int64_t q_s2,
    int64_t q_s3, int64_t k_s0, int64_t k_s1, int64_t k_s2, int64_t k_s3,
    int64_t qo_s0, int64_t qo_s1, int64_t qo_s2, int64_t qo_s3,
    int64_t ko_s0, int64_t ko_s1, int64_t ko_s2, int64_t ko_s3,
    int64_t f_s0, int64_t f_s1, int64_t f_s2, int64_t f_s3, int64_t f_s4,
    int64_t f_s5, int64_t qs_stride, int64_t ks_stride, float epsilon) {
  using ComputeType =
      std::conditional_t<HasRms, float, FreqsType>;

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  const int64_t rows = batch * dim1 * dim2;
  if (row >= rows) {
    return;
  }

  const int64_t i2 = row % dim2;
  const int64_t tmp = row / dim2;
  const int64_t i1 = tmp % dim1;
  const int64_t i0 = tmp / dim1;
  const int64_t q_base = i0 * q_s0 + i1 * q_s1 + i2 * q_s2;
  const int64_t qo_base =
      InPlace ? q_base : i0 * qo_s0 + i1 * qo_s1 + i2 * qo_s2;
  int64_t k_base = 0;
  int64_t ko_base = 0;
  if constexpr (HasK) {
    k_base = i0 * k_s0 + i1 * k_s1 + i2 * k_s2;
    ko_base =
        InPlace ? k_base : i0 * ko_s0 + i1 * ko_s1 + i2 * ko_s2;
  }

  float q_rrms = 1.0f;
  float k_rrms = 1.0f;
  if constexpr (HasRms) {
    const float q_sum =
        rope::rms_sum<InputType, ContigHead>(q + q_base, head_dim, q_s3, lane);
    q_rrms = rsqrtf(q_sum / static_cast<float>(head_dim) + epsilon);
    if constexpr (HasK) {
      const float k_sum = rope::rms_sum<InputType, ContigHead>(
          k + k_base, head_dim, k_s3, lane);
      k_rrms = rsqrtf(k_sum / static_cast<float>(head_dim) + epsilon);
    }
  }

  const int64_t fi0 = freqs_batch == 1 ? 0 : i0;
  const int64_t fi1 = freqs_dim1 == 1 ? 0 : i1;
  const int64_t fi2 = freqs_dim2 == 1 ? 0 : i2;
  const int64_t freq_row = fi0 * f_s0 + fi1 * f_s1 + fi2 * f_s2;
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
          q + q_base, pair_base, pairs, q_s3, q0_raw[0], q1_raw[0]);
      if constexpr (HasK) {
        rope::load_head_pair<InputType, SplitHalf, ContigHead>(
            k + k_base, pair_base, pairs, k_s3, k0_raw[0], k1_raw[0]);
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
                q_scale[static_cast<int64_t>(first) * qs_stride])));
        q1 = static_cast<float>(static_cast<InputType>(
            static_cast<float>(q1) * q_rrms *
            static_cast<float>(
                q_scale[static_cast<int64_t>(second) * qs_stride])));
      }

      FreqsType f00_raw, f01_raw, f10_raw, f11_raw;
      rope::load_rotation(freqs, freq_row + static_cast<int64_t>(pair) * f_s3,
                          f_s4, f_s5, f00_raw, f01_raw, f10_raw, f11_raw);
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
                  k_scale[static_cast<int64_t>(first) * ks_stride])));
          k1 = static_cast<float>(static_cast<InputType>(
              static_cast<float>(k1) * k_rrms *
              static_cast<float>(
                  k_scale[static_cast<int64_t>(second) * ks_stride])));
        }
        ComputeType ko0, ko1;
        rope::rotate(k0, k1, f00, f01, f10, f11, ko0, ko1);
        ko0_raw[p] = static_cast<InputType>(ko0);
        ko1_raw[p] = static_cast<InputType>(ko1);
      }
    }

    if constexpr (SplitHalf && ContigHead) {
      *reinterpret_cast<rope::Pair<InputType> *>(q_out + qo_base + pair_base) =
          {qo0_raw[0], qo0_raw[1]};
      *reinterpret_cast<rope::Pair<InputType> *>(
          q_out + qo_base + pairs + pair_base) = {qo1_raw[0], qo1_raw[1]};
      if constexpr (HasK) {
        *reinterpret_cast<rope::Pair<InputType> *>(
            k_out + ko_base + pair_base) = {ko0_raw[0], ko0_raw[1]};
        *reinterpret_cast<rope::Pair<InputType> *>(
            k_out + ko_base + pairs + pair_base) = {ko1_raw[0], ko1_raw[1]};
      }
    } else {
      rope::store_head_pair<InputType, SplitHalf, ContigHead>(
          q_out + qo_base, pair_base, pairs, InPlace ? q_s3 : qo_s3,
          qo0_raw[0], qo1_raw[0]);
      if constexpr (HasK) {
        rope::store_head_pair<InputType, SplitHalf, ContigHead>(
            k_out + ko_base, pair_base, pairs, InPlace ? k_s3 : ko_s3,
            ko0_raw[0], ko1_raw[0]);
      }
    }
  }

  // Norm-only tail: dims beyond rot_dim are normalized and scaled but never
  // rotated. Empty in the common rot_dim == head_dim case.
  const int64_t qo_s3_eff = InPlace ? q_s3 : qo_s3;
  const int64_t ko_s3_eff = InPlace ? k_s3 : ko_s3;
  for (int d = rot_dim + lane; d < head_dim; d += kThreadsPerWarp) {
    InputType qv = q[q_base + static_cast<int64_t>(d) * q_s3];
    if constexpr (HasRms) {
      qv = static_cast<InputType>(
          static_cast<float>(qv) * q_rrms *
          static_cast<float>(q_scale[static_cast<int64_t>(d) * qs_stride]));
    }
    q_out[qo_base + static_cast<int64_t>(d) * qo_s3_eff] = qv;
    if constexpr (HasK) {
      InputType kv = k[k_base + static_cast<int64_t>(d) * k_s3];
      if constexpr (HasRms) {
        kv = static_cast<InputType>(
            static_cast<float>(kv) * k_rrms *
            static_cast<float>(k_scale[static_cast<int64_t>(d) * ks_stride]));
      }
      k_out[ko_base + static_cast<int64_t>(d) * ko_s3_eff] = kv;
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
    const InputType *q, const InputType *k, const FreqsType *freqs,
    const ScaleType *q_scale, const ScaleType *k_scale, InputType *q_out,
    InputType *k_out, int64_t batch, int64_t dim1, int64_t dim2, int head_dim,
    int rot_dim,
    int64_t freqs_batch, int64_t freqs_dim1, int64_t freqs_dim2, int64_t q_s0,
    int64_t q_s1, int64_t q_s2, int64_t q_s3, int64_t k_s0, int64_t k_s1,
    int64_t k_s2, int64_t k_s3, int64_t qo_s0, int64_t qo_s1, int64_t qo_s2,
    int64_t qo_s3, int64_t ko_s0, int64_t ko_s1, int64_t ko_s2, int64_t ko_s3,
    int64_t f_s0, int64_t f_s1, int64_t f_s2, int64_t f_s3, int64_t f_s4,
    int64_t f_s5, int64_t qs_stride, int64_t ks_stride, float epsilon,
    cudaStream_t stream) {
  const int64_t rows = batch * dim1 * dim2;
  if (rows == 0) {
    return;
  }
  const int blocks = static_cast<int>((rows + kWarpsPerBlock - 1) /
                                      kWarpsPerBlock);
  rope_kernel<InputType, FreqsType, ScaleType, HasRms, SplitHalf, HasK, InPlace,
              ContigHead><<<blocks, kThreads, 0, stream>>>(
      q, k, freqs, q_scale, k_scale, q_out, k_out, batch, dim1, dim2, head_dim,
      rot_dim,
      freqs_batch, freqs_dim1, freqs_dim2, q_s0, q_s1, q_s2, q_s3, k_s0, k_s1,
      k_s2, k_s3, qo_s0, qo_s1, qo_s2, qo_s3, ko_s0, ko_s1, ko_s2, ko_s3,
      f_s0, f_s1, f_s2, f_s3, f_s4, f_s5, qs_stride, ks_stride, epsilon);
}

template <typename InputType, typename FreqsType, typename ScaleType,
          bool HasRms>
void rope_launcher(
    const InputType *q, const InputType *k, const FreqsType *freqs,
    const ScaleType *q_scale, const ScaleType *k_scale, InputType *q_out,
    InputType *k_out, int64_t batch, int64_t dim1, int64_t dim2, int head_dim,
    int rot_dim,
    int64_t freqs_batch, int64_t freqs_dim1, int64_t freqs_dim2, int64_t q_s0,
    int64_t q_s1, int64_t q_s2, int64_t q_s3, int64_t k_s0, int64_t k_s1,
    int64_t k_s2, int64_t k_s3, int64_t qo_s0, int64_t qo_s1, int64_t qo_s2,
    int64_t qo_s3, int64_t ko_s0, int64_t ko_s1, int64_t ko_s2, int64_t ko_s3,
    int64_t f_s0, int64_t f_s1, int64_t f_s2, int64_t f_s3, int64_t f_s4,
    int64_t f_s5, int64_t qs_stride, int64_t ks_stride, float epsilon,
    bool has_k, bool split_half, cudaStream_t stream) {
  const bool inplace = q == q_out && (!has_k || k == k_out);
  bool contig = q_s3 == 1 && qo_s3 == 1 &&
                pair_aligned(q, q_s0, q_s1, q_s2) &&
                pair_aligned(q_out, qo_s0, qo_s1, qo_s2);
  if (has_k) {
    contig = contig && k_s3 == 1 && ko_s3 == 1 &&
             pair_aligned(k, k_s0, k_s1, k_s2) &&
             pair_aligned(k_out, ko_s0, ko_s1, ko_s2);
  }
  // Split-half packs adjacent values independently in each half. Both half
  // starts must therefore be pair-aligned (the rotated prefix for partial
  // rotary).
  contig = contig && (!split_half || (head_dim % 4 == 0 && rot_dim % 4 == 0));

#define LAUNCH(HAS_K, SPLIT, INPLACE, CONTIG)                                  \
  launch_config<InputType, FreqsType, ScaleType, HasRms, SPLIT, HAS_K,         \
                INPLACE, CONTIG>(                                               \
      q, k, freqs, q_scale, k_scale, q_out, k_out, batch, dim1, dim2, head_dim, \
      rot_dim,                                                                  \
      freqs_batch, freqs_dim1, freqs_dim2, q_s0, q_s1, q_s2, q_s3, k_s0, k_s1, \
      k_s2, k_s3, qo_s0, qo_s1, qo_s2, qo_s3, ko_s0, ko_s1, ko_s2, ko_s3,      \
      f_s0, f_s1, f_s2, f_s3, f_s4, f_s5, qs_stride, ks_stride, epsilon, stream)
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
    const void *q, const void *k, const void *freqs, void *q_out, void *k_out,
    int64_t batch, int64_t dim1, int64_t dim2, int64_t head_dim,
    int64_t freqs_batch, int64_t freqs_dim1, int64_t freqs_dim2,
    int64_t q_s0, int64_t q_s1, int64_t q_s2, int64_t q_s3, int64_t k_s0,
    int64_t k_s1, int64_t k_s2, int64_t k_s3, int64_t qo_s0, int64_t qo_s1,
    int64_t qo_s2, int64_t qo_s3, int64_t ko_s0, int64_t ko_s1, int64_t ko_s2,
    int64_t ko_s3, int64_t f_s0, int64_t f_s1, int64_t f_s2, int64_t f_s3,
    int64_t f_s4, int64_t f_s5, int input_dtype_code, int freqs_dtype_code,
    bool has_k, bool split_half, cudaStream_t stream) {
  DISPATCH_HALF_INPUT_FP_FREQS_DTYPES(input_dtype_code, freqs_dtype_code,
                                      InputType, FreqsType, [&] {
    comfy::rope_launcher<InputType, FreqsType, float, false>(
        static_cast<const InputType *>(q), static_cast<const InputType *>(k),
        static_cast<const FreqsType *>(freqs), nullptr, nullptr,
        static_cast<InputType *>(q_out), static_cast<InputType *>(k_out), batch,
        dim1, dim2, static_cast<int>(head_dim), static_cast<int>(head_dim),
        freqs_batch, freqs_dim1,
        freqs_dim2, q_s0, q_s1, q_s2, q_s3, k_s0, k_s1, k_s2, k_s3, qo_s0,
        qo_s1, qo_s2, qo_s3, ko_s0, ko_s1, ko_s2, ko_s3, f_s0, f_s1, f_s2,
        f_s3, f_s4, f_s5, 0, 0, 0.0f, has_k, split_half, stream);
  });
}

extern "C" void launch_rms_rope_kernel(
    const void *q, const void *k, const void *freqs, const void *q_scale,
    const void *k_scale, void *q_out, void *k_out, int64_t batch, int64_t dim1,
    int64_t dim2, int64_t head_dim, int64_t rot_dim,
    int64_t freqs_batch, int64_t freqs_dim1,
    int64_t freqs_dim2, int64_t q_s0, int64_t q_s1, int64_t q_s2, int64_t q_s3,
    int64_t k_s0, int64_t k_s1, int64_t k_s2, int64_t k_s3, int64_t qo_s0,
    int64_t qo_s1, int64_t qo_s2, int64_t qo_s3, int64_t ko_s0, int64_t ko_s1,
    int64_t ko_s2, int64_t ko_s3, int64_t f_s0, int64_t f_s1, int64_t f_s2,
    int64_t f_s3, int64_t f_s4, int64_t f_s5, int64_t qs_stride,
    int64_t ks_stride, float epsilon, int input_dtype_code,
    int freqs_dtype_code, int scale_dtype_code, bool has_k, bool split_half,
    cudaStream_t stream) {
  DISPATCH_HALF_DTYPE(input_dtype_code, InputType, [&] {
    DISPATCH_FP_DTYPE(freqs_dtype_code, FreqsType, [&] {
      DISPATCH_FP_DTYPE(scale_dtype_code, ScaleType, [&] {
        comfy::rope_launcher<InputType, FreqsType, ScaleType, true>(
            static_cast<const InputType *>(q), static_cast<const InputType *>(k),
            static_cast<const FreqsType *>(freqs),
            static_cast<const ScaleType *>(q_scale),
            static_cast<const ScaleType *>(k_scale),
            static_cast<InputType *>(q_out), static_cast<InputType *>(k_out),
            batch, dim1, dim2, static_cast<int>(head_dim),
            static_cast<int>(rot_dim), freqs_batch,
            freqs_dim1, freqs_dim2, q_s0, q_s1, q_s2, q_s3, k_s0, k_s1, k_s2,
            k_s3, qo_s0, qo_s1, qo_s2, qo_s3, ko_s0, ko_s1, ko_s2, ko_s3,
            f_s0, f_s1, f_s2, f_s3, f_s4, f_s5, qs_stride, ks_stride, epsilon,
            has_k, split_half, stream);
      });
    });
  });
}
