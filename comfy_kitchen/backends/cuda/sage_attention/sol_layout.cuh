/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
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

// Shared layout contract for the CUDA Sol-Attn kernels. The producer
// (preprocess) and consumers (route, exact) must agree exactly on permutations
// and swizzles; a drift is invisible to either side's own test.

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "mma.cuh"

// The MMA/cp.async forms are sm_80+ but the build also targets sm_75, so device
// bodies compile out below sm_80 (as ops/na3d.cu does); dispatch constraints
// pin sol_attn to 8.0+, so the stubs are unreachable at runtime.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
#define SOL_SM80 1
#else
#define SOL_SM80 0
#endif

namespace sol {

constexpr int HEAD_DIM = 128;   // the only head_dim these kernels handle
constexpr int BLOCK    = 64;    // Sol-Attn's routing granularity, in tokens
constexpr float NEG    = -3.0e38f;   // finite, so NEG - NEG == 0 (unlike -inf)

// ---------------------------------------------------------------------------
// Permutations. Both are applied by the preprocess and assumed by the kernels.
// ---------------------------------------------------------------------------

// Contraction-axis permutation: makes the two MMA operand words (16 B apart)
// adjacent, so one 8-byte load fetches both. Free -- permuting both sides of a
// contraction identically leaves the product unchanged.
//   applied to: Q's d axis, K's d axis, V^T's key axis, pooled keys' d axis
__host__ __device__ __forceinline__ int perm_d(int d) {
    const int kc = d >> 5, rem = d & 31, h = rem >> 4, r2 = rem & 15;
    return kc * 32 + 8 * (r2 >> 2) + 4 * h + (r2 & 3);
}

// Key relabelling inside a 64-key block: gives each lane the 4 consecutive keys
// the INT8 PV A operand wants without cross-lane shuffles -- free, because
// attention is permutation-invariant over keys.
//   applied to: K's row axis and its per-token scales, within each block
// NOT applied to V^T: the PV B operand wants consecutive *logical* keys.
__host__ __device__ __forceinline__ int perm_key(int p) {
    return 16 * (p >> 4) + 4 * ((p & 7) >> 1) + 2 * ((p >> 3) & 1) + (p & 1);
}

// ---------------------------------------------------------------------------
// Shared-memory swizzles. Padding would cost 6 KB = one block of occupancy.
// Verify any change by enumerating both 16-lane LDS.64 phases against 32 banks.
// ---------------------------------------------------------------------------

// K tile, 64 rows x 128 B. `c16 ^ (r & 7)` is conflict-free for 32-bit reads but
// COLLIDES once reads are 64-bit; this form does not.
__device__ __forceinline__ int swz_k(int row) { return (row & 3) * 2; }

// V^T tile, 128 rows x 64 B -- only 4 granules, so two swizzle bits cannot
// separate 8 g-values: rows g and g+4 collide under the naive `c16 ^ (C & 3)`.
// Folding in C>>2 separates them.
__device__ __forceinline__ int swz_v(int col) { return ((col >> 2) ^ col) & 3; }

// ---------------------------------------------------------------------------
// MMA wrappers. sm_120 is issue-rate bound and f32-accumulate forms issue at
// half rate, so INT8 m16n8k32 (4096 MACs, full rate) beats every alternative.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void mma_s8(int32_t* d, const uint32_t* a, const uint32_t* b) {
#if SOL_SM80
    asm volatile("mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

// P is non-negative, so it rides the u8 side of a u8 x s8 MMA: 255 levels
// instead of s8's 127, for free.
__device__ __forceinline__ void mma_u8s8(int32_t* d, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.satfinite.s32.u8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void mma_bf16(float* d, const uint32_t* a, const uint32_t* b) {
#if SOL_SM80
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

__device__ __forceinline__ uint32_t pack_bf2(float lo, float hi) {
    __nv_bfloat162 p = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&p);
}

// ---------------------------------------------------------------------------
// cp.async. Pipeline depth 2 is the measured optimum: occupancy beats depth.
// The driver reserves ~1 KB smem per block on top of the request.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
#if SOL_SM80
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src));
#endif
}
// .ca keeps the line in L1. Only worthwhile where the source is reused (the
// pooled arrays in routing); the exact pass streams far past L1 and pays there.
__device__ __forceinline__ void cp_async16_ca(void* dst, const void* src) {
#if SOL_SM80
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                 :: "r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src));
#endif
}
__device__ __forceinline__ void cp_commit() {
#if SOL_SM80
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}
template <int N> __device__ __forceinline__ void cp_wait() {
#if SOL_SM80
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}

}  // namespace sol
