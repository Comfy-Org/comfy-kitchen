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
#ifndef COMFY_FLOAT_UTILS_CUH_
#define COMFY_FLOAT_UTILS_CUH_

#include <cstdint>
#include <type_traits>

#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#if CUDA_VERSION >= 12080
#include <cuda_fp4.h>
#endif

namespace comfy {

__device__ __forceinline__ float warp_reduce_fmax(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
  return v;
}

__device__ __forceinline__ int8_t float_to_int8_rn(float value) {
  int32_t quantized;
  asm volatile("cvt.rni.sat.s8.f32 %0, %1;"
               : "=r"(quantized)
               : "f"(value));
  return static_cast<int8_t>(quantized);
}

__device__ __forceinline__ int8_t quant_int8_rcp(float v,
                                                 float inv_scale) {
  return float_to_int8_rn(v * inv_scale);
}

__device__ __forceinline__ int8_t quant_int8(float v, float scale) {
  return quant_int8_rcp(v, 1.f / scale);
}

// ptr must be 4-byte aligned. Callers must preserve that alignment at every
// row offset, for example by requiring the row stride to be a multiple of 4.
__device__ __forceinline__ void store4_i8(int8_t *ptr, int8_t a, int8_t b,
                                          int8_t c, int8_t d) {
  *reinterpret_cast<int32_t *>(ptr) =
      (uint32_t)(uint8_t)a | ((uint32_t)(uint8_t)b << 8) |
      ((uint32_t)(uint8_t)c << 16) | ((uint32_t)(uint8_t)d << 24);
}

// fp32 -> output element type conversion for kernel epilogues (round to
// nearest even for the half types).
template <typename T> __device__ __forceinline__ T from_float(float v);
template <> __device__ __forceinline__ float from_float<float>(float v) { return v; }
template <> __device__ __forceinline__ __half from_float<__half>(float v) { return __float2half(v); }
template <> __device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

// FP8 type traits for max values
template <typename T>
struct FP8LimitsTrait;

template <>
struct FP8LimitsTrait<__nv_fp8_e4m3> {
  static constexpr float max = 448.0f;
  static constexpr float max_inverse = 1.0 / max;
};

template <>
struct FP8LimitsTrait<__nv_fp8_e5m2> {
  static constexpr float max = 57344.0f;
  static constexpr float max_inverse = 1.0 / max;
};

#if CUDA_VERSION >= 12080
// FP4 type traits
template <typename T>
struct FP4LimitsTrait;

template <>
struct FP4LimitsTrait<__nv_fp4x2_storage_t> {
  static constexpr float max = 6.0f;
  static constexpr float max_inverse = 1.0 / max;
};

// Vectorized half-precision loads
#pragma nv_diag_suppress 1056
template<typename IType>
__forceinline__ __device__ const IType* load_f16x2(const IType* val) {
    float vals = *reinterpret_cast<const float*>(val);
    return reinterpret_cast<const IType*>(&vals);
}
template<typename IType>
__forceinline__ __device__ const IType* load_f16x4(const IType* val) {
    float2 vals = *reinterpret_cast<const float2*>(val);
    return reinterpret_cast<const IType*>(&vals);
}
template<typename IType>
    __forceinline__ __device__ const IType* load_f16x8(const IType* val) {
    float4 vals = *reinterpret_cast<const float4*>(val);
    return reinterpret_cast<const IType*>(&vals);
}
template<typename IType>
    __forceinline__ __device__ const IType* load_f8x4(const IType* val) {
    float vals = *reinterpret_cast<const float*>(val);
    return reinterpret_cast<const IType*>(&vals);
}
template<typename IType>
    __forceinline__ __device__ const IType* load_f8x8(const IType* val) {
    float4 vals = *reinterpret_cast<const float4*>(val);
    return reinterpret_cast<const IType*>(&vals);
}
#pragma nv_diag_default 1056

// Store 2 FP4 values (1 __nv_fp4x2)
// hi_first=true: val0 in high nibble, val1 in low nibble (default, matches cuBLAS convention)
// hi_first=false: val0 in low nibble, val1 in high nibble
template<typename OType, bool hi_first = true>
__forceinline__ __device__ void store_fp4x2(OType* output, size_t idx, float val0, float val1) {
    float2 args = hi_first ? float2{val1, val0} : float2{val0, val1};
    *reinterpret_cast<__nv_fp4x2_storage_t*>(&output[idx]) =
        __nv_cvt_float2_to_fp4x2(args, __NV_E2M1, cudaRoundNearest);
}

// Store 4 FP4 values (2 __nv_fp4x2) using single store
template<typename OType, bool hi_first = true>
__forceinline__ __device__ void store_fp4x4(OType* output, size_t idx, float val0, float val1, float val2, float val3) {
    union {
        uint16_t u16;
        __nv_fp4x2_storage_t fp4x2[2];
    } packed;

    float2 args0 = hi_first ? float2{val1, val0} : float2{val0, val1};
    float2 args1 = hi_first ? float2{val3, val2} : float2{val2, val3};
    packed.fp4x2[0] = __nv_cvt_float2_to_fp4x2(args0, __NV_E2M1, cudaRoundNearest);
    packed.fp4x2[1] = __nv_cvt_float2_to_fp4x2(args1, __NV_E2M1, cudaRoundNearest);

    *reinterpret_cast<uint16_t*>(&output[2*idx]) = packed.u16;
}

// cuBLAS swizzled scale factor layout offset calculation
__device__ __forceinline__ size_t
scale_factor_swizzled_offset(size_t row_idx, size_t col_idx, uint32_t col_length) {
  constexpr uint32_t kTotalRowsPerBaseBlock = 128;
  constexpr uint32_t kRowsPerBaseBlockCol = 32;
  constexpr uint32_t kColsPerBaseBlockCol = 4;

  const size_t rb = row_idx / kTotalRowsPerBaseBlock;
  const size_t rem = row_idx % kTotalRowsPerBaseBlock;
  const size_t d4 = rem / kRowsPerBaseBlockCol;
  const size_t d3 = rem % kRowsPerBaseBlockCol;
  const size_t cbg = col_idx / kColsPerBaseBlockCol;
  const size_t d5 = col_idx % kColsPerBaseBlockCol;

  const size_t cbg_cnt = (col_length + kColsPerBaseBlockCol - 1) / kColsPerBaseBlockCol;
  return ((rb * cbg_cnt + cbg) * kRowsPerBaseBlockCol + d3) * 16 + d4 * kColsPerBaseBlockCol + d5;
}

#endif // CUDA_VERSION >= 12080

namespace quantization {
// Vector and masked loads share one register representation on every GPU.
template <typename T, int N> union Group {
  float4 vectors[(N * sizeof(T) + 15) / 16];
  uint64_t words[(N * sizeof(T) + 7) / 8];
  T values[N];
};

template <typename T, int N, bool Masked>
__device__ __forceinline__ Group<T, N>
load_group(const T *input, uint64_t group, uint64_t columns,
           uint64_t original_rows, uint64_t original_columns) {
  Group<T, N> result;
  if constexpr (Masked) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      const uint64_t element = group * N + i;
      const uint64_t row = element / columns, col = element % columns;
      result.values[i] = row < original_rows && col < original_columns
                             ? input[row * original_columns + col]
                             : T(0.0f);
    }
  } else if constexpr (N * sizeof(T) == 8) {
    result.words[0] = *reinterpret_cast<const uint64_t *>(input + group * N);
  } else {
#pragma unroll
    for (int i = 0; i < N / 8; ++i)
      result.vectors[i] =
          reinterpret_cast<const float4 *>(input + group * N)[i];
  }
  return result;
}

template <typename T, int N>
__device__ __forceinline__ float group_absmax(const Group<T, N> &group) {
  using Pair =
      std::conditional_t<std::is_same_v<T, __half>, __half2, __nv_bfloat162>;
  const auto *pairs = reinterpret_cast<const Pair *>(group.values);
  Pair maximum = __habs2(pairs[0]);
#pragma unroll
  for (int i = 1; i < N / 2; ++i)
    maximum = __hmax2(maximum, __habs2(pairs[i]));
  if constexpr (std::is_same_v<T, __half>)
    return __half2float(__hmax(__low2half(maximum), __high2half(maximum)));
  else
    return __bfloat162float(
        __hmax(__low2bfloat16(maximum), __high2bfloat16(maximum)));
}

// Every API dimension is a multiple of 16 or 32. Adjust the block size to
// divide the work exactly, including tiny shapes, so kernels need no tail loop.
inline int block_size(int64_t chunks, int preferred) {
  while (chunks % preferred)
    preferred /= 2;
  return preferred;
}
} // namespace quantization

} // namespace comfy

#endif // COMFY_FLOAT_UTILS_CUH_
