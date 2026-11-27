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
#include "dtype_dispatch.cuh"
#include "float_utils.cuh"
#include "utils.cuh"
#include <stdexcept>

namespace comfy {
namespace {
template <typename Input, int V, bool Masked>
__global__ void
quantize_mxfp8_kernel(const Input *input, uint8_t *output, uint8_t *scales,
                      uint64_t reciprocal, uint64_t runtime_columns,
                      uint64_t original_rows, uint64_t original_columns) {
  constexpr int threads_per_group = 32 / V;
  const uint64_t columns = runtime_columns;
  const uint64_t chunk = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  {
    uint64_t indexed_col = 0;
    const uint64_t indexed_row =
        quantization::row_index(chunk * V, columns, reciprocal, &indexed_col);
    const auto values = quantization::load_group<Input, V, Masked>(
        input, chunk, columns, original_rows, original_columns, indexed_row,
        indexed_col);
    float maximum = quantization::group_absmax(values);
    if constexpr (threads_per_group > 1) {
#pragma unroll
      for (int offset = threads_per_group / 2; offset; offset /= 2)
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, offset,
                                                  threads_per_group));
      maximum = __shfl_sync(0xffffffff, maximum, 0, threads_per_group);
    }
    const uint8_t scale = __nv_cvt_float_to_e8m0(
        __fdividef(maximum, 448.0f), __NV_SATFINITE, cudaRoundPosInf);
    const uint64_t row = indexed_row;
    const uint64_t col = indexed_col / 32;
    if (threadIdx.x % threads_per_group == 0)
      scales[scale_factor_swizzled_offset(row, col, columns / 32)] = scale;
    const float encode = __uint_as_float(uint32_t(254 - scale) << 23);
    union {
      uint4 vectors[(V + 15) / 16];
      uint64_t word;
      uint16_t pairs[V / 2];
      uint32_t words[V / 4];
    } packed;
#pragma unroll
    for (int i = 0; i < V / 2; ++i) {
      const float a = float(values.values[2 * i]) * encode;
      const float b = float(values.values[2 * i + 1]) * encode;
      packed.pairs[i] = __nv_cvt_float2_to_fp8x2(make_float2(a, b),
                                                 __NV_SATFINITE, __NV_E4M3);
    }
    // SATFINITE clips the range; repair FP8 NaNs to preserve fmin/fmax's -448.
#pragma unroll
    for (int i = 0; i < V / 4; ++i) {
      const uint32_t word = packed.words[i];
      const uint32_t high = (word & 0x7f7f7f7fu) + 0x01010101u;
      uint32_t mask;
      asm("prmt.b32 %0, %1, 0, 0xba98;" : "=r"(mask) : "r"(high));
      packed.words[i] = (word & ~mask) | (0xfefefefeu & mask);
    }
    if constexpr (V == 8)
      reinterpret_cast<uint64_t *>(output)[chunk] = packed.word;
    else {
#pragma unroll
      for (int i = 0; i < V / 16; ++i)
        reinterpret_cast<uint4 *>(output + uint64_t(chunk) * V)[i] =
            packed.vectors[i];
    }
  }
}

template <typename Input, int V>
void launch_mxfp8_quantization(const void *input, void *output, void *scales,
                               int64_t rows, int64_t columns,
                               int64_t original_rows, int64_t original_columns,
                               int threads, cudaStream_t stream) {
  const int64_t groups = rows * columns / 32;
  const int64_t chunks = groups * (32 / V);
  threads = quantization::block_size(chunks, threads);
  const int blocks = static_cast<int>(chunks / threads);
  const bool masked = rows != original_rows || columns != original_columns ||
                      !optimized::aligned_to(input, 16);
  const uint64_t reciprocal = ~uint64_t(0) / columns;
#define LAUNCH(MASKED)                                                         \
  quantize_mxfp8_kernel<Input, V, MASKED><<<blocks, threads, 0, stream>>>(  \
      static_cast<const Input *>(input), static_cast<uint8_t *>(output),       \
      static_cast<uint8_t *>(scales), reciprocal, columns, original_rows,      \
      original_columns)
  if (masked) {
    LAUNCH(true);
  } else {
    LAUNCH(false);
  }
#undef LAUNCH
}

} // namespace
} // namespace comfy

extern "C" void launch_quantize_mxfp8_kernel(const void *input, void *output,
                                             void *scales, int64_t rows,
                                             int64_t columns,
                                             int64_t original_rows,
                                             int64_t original_columns,
                                             int dtype, cudaStream_t stream) {
  if (!rows || !columns)
    return;
  if (rows % 32 || columns % 32)
    throw std::runtime_error("num_rows and num_cols must be divisible by 32 "
                             "for MXFP8 block quantization");
  const int threads = rows * columns <= 67108864 ? 128 : 256;
  DISPATCH_HALF_DTYPE(dtype, Input, [&] {
    comfy::launch_mxfp8_quantization<Input, 16>(
        input, output, scales, rows, columns, original_rows, original_columns,
        threads, stream);
  });
  CUDA_CHECK(cudaGetLastError());
}
