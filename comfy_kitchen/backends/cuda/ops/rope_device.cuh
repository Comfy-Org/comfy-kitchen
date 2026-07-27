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
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace comfy::rope {

template <typename T> struct alignas(sizeof(T) * 2) Pair {
  T x;
  T y;
};

template <typename T, bool SplitHalf, bool ContigHead>
__device__ __forceinline__ void load_head_pair(const T *row, int pair,
                                               int pairs, int64_t stride,
                                               T &x0, T &x1) {
  if constexpr (SplitHalf) {
    x0 = row[static_cast<int64_t>(pair) * stride];
    x1 = row[static_cast<int64_t>(pair + pairs) * stride];
  } else if constexpr (ContigHead) {
    const Pair<T> value =
        *reinterpret_cast<const Pair<T> *>(row + static_cast<int64_t>(pair) * 2);
    x0 = value.x;
    x1 = value.y;
  } else {
    const int64_t first = static_cast<int64_t>(pair) * 2 * stride;
    x0 = row[first];
    x1 = row[first + stride];
  }
}

template <typename T, bool SplitHalf, bool ContigHead>
__device__ __forceinline__ void store_head_pair(T *row, int pair, int pairs,
                                                int64_t stride, T x0, T x1) {
  if constexpr (SplitHalf) {
    row[static_cast<int64_t>(pair) * stride] = x0;
    row[static_cast<int64_t>(pair + pairs) * stride] = x1;
  } else if constexpr (ContigHead) {
    Pair<T> value{x0, x1};
    *reinterpret_cast<Pair<T> *>(row + static_cast<int64_t>(pair) * 2) = value;
  } else {
    const int64_t first = static_cast<int64_t>(pair) * 2 * stride;
    row[first] = x0;
    row[first + stride] = x1;
  }
}

template <typename T, bool ContigHead>
__device__ __forceinline__ float rms_sum(const T *row, int head_dim,
                                         int64_t stride, int lane) {
  float sum = 0.0f;
  for (int element = lane; element < head_dim; element += 32) {
    const float value =
        static_cast<float>(row[ContigHead ? element
                                         : static_cast<int64_t>(element) * stride]);
    sum = fmaf(value, value, sum);
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
  }
  return __shfl_sync(0xffffffffu, sum, 0);
}

template <typename F>
__device__ __forceinline__ void load_rotation(const F *freqs, int64_t base,
                                              int64_t stride_rot,
                                              int64_t stride_component, F &f00,
                                              F &f01, F &f10, F &f11) {
  f00 = freqs[base];
  f01 = freqs[base + stride_component];
  f10 = freqs[base + stride_rot];
  f11 = freqs[base + stride_rot + stride_component];
}

template <typename C>
__device__ __forceinline__ void rotate(C x0, C x1, C f00, C f01, C f10,
                                       C f11, C &y0, C &y1) {
  y0 = f00 * x0 + f01 * x1;
  y1 = f10 * x0 + f11 * x1;
}

} // namespace comfy::rope
