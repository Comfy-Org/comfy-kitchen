// SPDX-License-Identifier: Apache-2.0
// Vendored unmodified from SageAttention
// (https://github.com/thu-ml/SageAttention) commit
// d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5.

/*
 * Copyright (c) 2024 by SageAttention team.
 *
 * This file is based on code from Flashinfer,
 * https://github.com/flashinfer-ai/flashinfer/blob/v0.1.5/include/flashinfer/math.cuh
 * Copyright (c) 2023 by FlashInfer team.
 * Small modifications made by SageAttention team, 2024 (e.g., renamed
 * namespace).
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

#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#ifndef USHORT_TYPE
#define USHORT_TYPE
typedef unsigned short ushort;
#endif

namespace math {

// log2(e)
constexpr float log2e = 1.44269504088896340736f;
constexpr float log2e_recp = 1.0f / log2e;

__forceinline__ __device__ half2 uint32_as_half2(uint32_t x) {
  return *(half2 *)&x;
}

__forceinline__ __device__ uint32_t half2_as_uint32(half2 x) {
  return *(uint32_t *)&x;
}

/*!
 * \brief Wrapper of PTX ex2.approx instruction, which computes 2^x
 * \param x input
 */
__forceinline__ __device__ float ptx_exp2(float x) {
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

/*!
 * \brief Exact int32 -> FP32 for -2^22 <= x <= 2^22 (an S8xS8 dot over <= 256
 * products stays within the range).
 *
 * Below sm_89 the conversion is an I2F on the XU pipe, which the softmax's ex2
 * already saturates, so it is done with integer/FP32 arithmetic instead:
 * 0x4B400000 is the bit pattern of 1.5 * 2^23; inside [2^23, 2^24) the FP32 ulp
 * is exactly 1, so adding x to the pattern adds exactly x and subtracting
 * 1.5 * 2^23 (one exact FADD) recovers float(x). Measured on an A6000 (sm_80
 * cubin): int8 attention -16%, output bit-identical. From sm_89 on, ptxas emits
 * the packed I2FP on the FMA pipe, which is already cheaper than two ALU/FMA
 * instructions (measured +4% attention time with the arithmetic form on an RTX
 * 5000 Ada), so the plain conversion is kept there.
 */
__forceinline__ __device__ float i32_to_f32_exact(int32_t x) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 890
  return __int_as_float(x + 0x4B400000) - 12582912.0f;
#else
  return __int2float_rn(x);
#endif
}

/*!
 * \brief Wrapper of PTX lg2.approx instruction, which computes log2(x)
 * \param x input
 */
__forceinline__ __device__ float ptx_log2(float x) {
  float y;
  asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

/*!
 * \brief Wrapper of PTX ex2.approx.f16x2 instruction, which computes 2^x
 * \param x input
 */
__forceinline__ __device__ half2 ptx_exp2(half2 x) {
  uint32_t y_u32;
  uint32_t x_u32 = half2_as_uint32(x);
  asm volatile("ex2.approx.f16x2 %0, %1;" : "=r"(y_u32) : "r"(x_u32));
  return uint32_as_half2(y_u32);
}

/*!
 * \brief Wrapper of PTX ex2.approx.f16 instruction, which computes 2^x
 * \param x input
 */
__forceinline__ __device__ half ptx_exp2(half x) {
  ushort y_u16;
  asm volatile("ex2.approx.f16 %0, %1;"
               : "=h"(y_u16)
               : "h"(__half_as_ushort(x)));
  return __ushort_as_half(y_u16);
}

/*!
 * \brief Wrapper of PTX rcp.approx instruction, which computes 1/x
 * \param x input
 */
__forceinline__ __device__ float ptx_rcp(float x) {
  float y;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

/*!
 * \brief Wrapper of PTX shfl.sync.bfly instruction, which performs a butterfly
 * shuffle between threads in a warp. \param x The value in the source lane
 * \param lane_mask The mask to perform thread index xor with: y[i] <- x[i ^
 * delta]
 */
__forceinline__ __device__ float shfl_xor_sync(float x, int lane_mask) {
  float y;
  asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;"
               : "=f"(y)
               : "f"(x), "r"(lane_mask));
  return y;
}

/*!
 * \brief Wrapper of PTX shfl.sync.bfly instruction on half2, which performs a
 * butterfly shuffle between threads in a warp. \param x The value in the source
 * lane \param lane_mask The mask to perform thread index xor with: y[i] <- x[i
 * ^ lane_mask]
 */
__forceinline__ __device__ half2 shfl_xor_sync(half2 x, int lane_mask) {
  return __shfl_xor_sync(0xffffffff, x, lane_mask);
}

/*!
 * \brief Wrapper of PTX rsqrt approximation instruction, which computes
 * 1/sqrt(x) \param x input
 */
__forceinline__ __device__ float rsqrt(float x) {
  float y;
  asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

/*!
 * \brief Wrapper of PTX tanh.approx.f32 instruction, which computes tanh(x)
 * \param x input
 */
__forceinline__ __device__ float tanh(float x) {
  float y;
  asm volatile("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

/*!
 * \brief Wrapper of PTX tanh.approx.f16x2 instruction, which computes tanh(x)
 * \param x input
 */
__forceinline__ __device__ half2 tanh(half2 x) {
  uint32_t y_u32;
  uint32_t x_u32 = half2_as_uint32(x);
  asm volatile("tanh.approx.f16x2 %0, %1;" : "=r"(y_u32) : "r"(x_u32));
  return uint32_as_half2(y_u32);
}

/*!
 * \brief Wrapper of PTX tanh.approx.f16 instruction, which computes tanh(x)
 * \param x input
 */
__forceinline__ __device__ half tanh(half x) {
  ushort y_u16;
  asm volatile("tanh.approx.f16 %0, %1;"
               : "=h"(y_u16)
               : "h"(__half_as_ushort(x)));
  return __ushort_as_half(y_u16);
}

} // namespace math

