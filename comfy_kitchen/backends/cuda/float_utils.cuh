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

#include <cuda_fp8.h>
#if CUDA_VERSION >= 12080
#include <cuda_fp4.h>
#endif

namespace comfy {

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
template<typename OType>
__forceinline__ __device__ void store_fp4x2(OType* output, size_t idx, float val0, float val1) {
    *reinterpret_cast<__nv_fp4x2_storage_t*>(&output[idx]) =
        __nv_cvt_float2_to_fp4x2(float2{val1, val0}, __NV_E2M1, cudaRoundNearest);
}

// Store 4 FP4 values (2 __nv_fp4x2) using single store
template<typename OType>
__forceinline__ __device__ void store_fp4x4(OType* output, size_t idx, float val0, float val1, float val2, float val3) {
    union {
        uint16_t u16;
        __nv_fp4x2_storage_t fp4x2[2];
    } packed;

    packed.fp4x2[0] = __nv_cvt_float2_to_fp4x2(float2{val1, val0}, __NV_E2M1, cudaRoundNearest);
    packed.fp4x2[1] = __nv_cvt_float2_to_fp4x2(float2{val3, val2}, __NV_E2M1, cudaRoundNearest);

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

} // namespace comfy

#endif // COMFY_FLOAT_UTILS_CUH_

