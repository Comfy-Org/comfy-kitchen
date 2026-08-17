/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace comfy::attention {

template <typename T> struct MmaTraits;

template <> struct MmaTraits<__half> {
  static __device__ __forceinline__ void mma(float *d, const uint32_t *a,
                                              const uint32_t *b) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
          "r"(b[1]));
#endif
  }

  static __device__ __forceinline__ uint32_t pack(float lo, float hi) {
    __half2 packed = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<uint32_t *>(&packed);
  }
};

template <> struct MmaTraits<__nv_bfloat16> {
  static __device__ __forceinline__ void mma(float *d, const uint32_t *a,
                                              const uint32_t *b) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
          "r"(b[1]));
#endif
  }

  static __device__ __forceinline__ uint32_t pack(float lo, float hi) {
    __nv_bfloat162 packed = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t *>(&packed);
  }
};

__device__ __forceinline__ void ldmatrix_x4(uint32_t *registers,
                                            uint32_t address) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 750
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
      : "=r"(registers[0]), "=r"(registers[1]), "=r"(registers[2]),
        "=r"(registers[3])
      : "r"(address));
#endif
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t *registers,
                                                  uint32_t address) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 750
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
      : "=r"(registers[0]), "=r"(registers[1]), "=r"(registers[2]),
        "=r"(registers[3])
      : "r"(address));
#endif
}

} // namespace comfy::attention
