/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "sage_attention/cp_async.cuh"
#include "sage_attention/mma.cuh"

namespace comfy::sol_attention {

inline constexpr int WARP_SIZE = 32;

__device__ __host__ constexpr int cdiv(int value, int divisor) {
  return (value + divisor - 1) / divisor;
}

// 128-byte XOR swizzle shared by the BF16 QK and PV fragments.
template <int STRIDE>
__device__ __forceinline__ uint32_t swizzle(uint32_t index) {
  if constexpr (STRIDE == 16)
    return index;
  const uint32_t row = (index / STRIDE) % 8;
  const uint32_t bits = row / max(64 / STRIDE, 1);
  return index ^ (bits << 4);
}

template <int HEIGHT, int WIDTH, int THREADS,
          cp_async::PrefetchMode PREFETCH =
              cp_async::PrefetchMode::kNoPrefetch>
__device__ __forceinline__ void
copy_bf16_tile(uint32_t destination, const nv_bfloat16 *source,
               int64_t source_stride, int thread) {
  constexpr int ELEMENTS_PER_COPY = 16 / sizeof(nv_bfloat16);
  constexpr int ITERATIONS =
      HEIGHT * WIDTH / (THREADS * ELEMENTS_PER_COPY);
#pragma unroll
  for (int iteration = 0; iteration < ITERATIONS; ++iteration) {
    const int index = (iteration * THREADS + thread) * ELEMENTS_PER_COPY;
    const int row = index / WIDTH;
    const int column = index % WIDTH;
    const uint32_t address =
        swizzle<WIDTH * sizeof(nv_bfloat16)>(
            destination +
            (row * WIDTH + column) * sizeof(nv_bfloat16));
    auto *shared = reinterpret_cast<uint4 *>(__cvta_shared_to_generic(address));
    const auto *global =
        reinterpret_cast<const uint4 *>(source + row * source_stride + column);
    cp_async::load_128b<PREFETCH>(shared, global);
  }
}

template <int HEIGHT, int WIDTH, int THREADS>
__device__ __forceinline__ void
copy_bf16_tile_masked(uint32_t destination, const nv_bfloat16 *source,
                      int64_t source_stride, int valid_rows, int thread) {
  constexpr int ELEMENTS_PER_COPY = 16 / sizeof(nv_bfloat16);
  constexpr int ITERATIONS =
      HEIGHT * WIDTH / (THREADS * ELEMENTS_PER_COPY);
#pragma unroll
  for (int iteration = 0; iteration < ITERATIONS; ++iteration) {
    const int index = (iteration * THREADS + thread) * ELEMENTS_PER_COPY;
    const int row = index / WIDTH;
    const int column = index % WIDTH;
    const uint32_t address =
        swizzle<WIDTH * sizeof(nv_bfloat16)>(
            destination +
            (row * WIDTH + column) * sizeof(nv_bfloat16));
    auto *shared = reinterpret_cast<uint4 *>(__cvta_shared_to_generic(address));
    const auto *global =
        reinterpret_cast<const uint4 *>(source + row * source_stride + column);
    cp_async::pred_load_128b<cp_async::PrefetchMode::kNoPrefetch,
                            cp_async::SharedMemFillMode::kFillZero>(
        shared, global, row < valid_rows);
  }
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t *registers,
                                            uint32_t address) {
  mma::ldmatrix_m8n8x4(
      registers,
      static_cast<nv_bfloat16 *>(__cvta_shared_to_generic(address)));
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t *registers,
                                                  uint32_t address) {
  mma::ldmatrix_m8n8x4_trans(
      registers,
      static_cast<nv_bfloat16 *>(__cvta_shared_to_generic(address)));
}

__device__ __forceinline__ void mma_bf16(uint32_t *a, uint32_t *b,
                                         float *accumulator) {
  mma::MmaTraits<nv_bfloat16>::mma(accumulator, a, b);
}

} // namespace comfy::sol_attention
