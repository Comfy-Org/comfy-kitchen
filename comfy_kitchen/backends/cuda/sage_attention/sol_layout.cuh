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

// Device bodies compile out below sm_80; dispatch pins sol_attn to sm_80+.
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

// Contraction-axis permutation: makes each lane's two MMA operand words one
// 8-byte load. Applied to Q/K/pooled-K d axes and V^T's key axis.
__host__ __device__ __forceinline__ int perm_d(int d) {
    const int kc = d >> 5, rem = d & 31, h = rem >> 4, r2 = rem & 15;
    return kc * 32 + 8 * (r2 >> 2) + 4 * h + (r2 & 3);
}

// Key relabelling per 64-block so the INT8 PV A operand needs no shuffles.
// Applied to K rows + their scales; NOT to V^T (wants logical key order).
__host__ __device__ __forceinline__ int perm_key(int p) {
    return 16 * (p >> 4) + 4 * ((p & 7) >> 1) + 2 * ((p >> 3) & 1) + (p & 1);
}

// ---------------------------------------------------------------------------
// Shared-memory swizzles (padding would cost a block of occupancy). Verify any
// change by enumerating both 16-lane LDS.64 phases against 32 banks.
// ---------------------------------------------------------------------------

// K tile, 64 x 128 B. The naive c16 ^ (r & 7) collides for 64-bit reads.
__device__ __forceinline__ int swz_k(int row) { return (row & 3) * 2; }

// V^T tile, 128 x 64 B. Naive c16 ^ (C & 3) collides rows g and g+4.
__device__ __forceinline__ int swz_v(int col) { return ((col >> 2) ^ col) & 3; }

// Fused per-head RMSNorm + split-half RoPE on a staged bf16 tile, in place.
// Matches ops/rms_rope.cu bit-for-bit, including its bf16 rounding between
// norm and rotation. One warp per token; lane owns channels 4*lane..+3.
__device__ __forceinline__ void norm_rope_rows(
    __nv_bfloat16* tile, int ld, int len, const float* __restrict__ fab_t0,
    const __nv_bfloat16* __restrict__ w, float eps, int rot)
{
    // fab [T, rot, 2] is per-channel: out[c] = f.x*n[c] + f.y*n[partner(c)]
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    const int nw = (int)(blockDim.x >> 5), c0 = lane * 4;
    float wreg[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) wreg[i] = __bfloat162float(w[c0 + i]);
    for (int t = wp; t < len; t += nw) {
        __nv_bfloat16* row = tile + t * ld;
        float x[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) x[i] = __bfloat162float(row[c0 + i]);
        float ss = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3];
        #pragma unroll
        for (int off = 16; off; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
        const float rrms = rsqrtf(ss / (float)HEAD_DIM + eps);
        float n[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            n[i] = __bfloat162float(__float2bfloat16(x[i] * rrms * wreg[i]));
        // Explicit source lane: a shfl_xor butterfly is only correct for
        // power-of-two offsets, and rot/8 need not be one (H3 rot=96 -> 12).
        const int poff = rot >> 3;
        const int src = (c0 < (rot >> 1)) ? lane + poff
                        : (c0 < rot ? lane - poff : lane);
        float p[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) p[i] = __shfl_sync(0xffffffffu, n[i], src);
        float out[4];
        if (c0 < rot) {
            const float* fr = fab_t0 + (int64_t)t * (rot * 2) + c0 * 2;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 f = *reinterpret_cast<const float2*>(fr + i * 2);
                out[i] = f.x * n[i] + f.y * p[i];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 4; ++i) out[i] = n[i];
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) row[c0 + i] = __float2bfloat16(out[i]);
    }
}

// ---------------------------------------------------------------------------
// MMA wrappers. INT8 m16n8k32 issues at full rate on sm_120; f32-accumulate
// forms issue at half rate.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void mma_s8(int32_t* d, const uint32_t* a, const uint32_t* b) {
#if SOL_SM80
    asm volatile("mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

// P is non-negative: the u8 side gives it 255 levels instead of 127.
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
// ---------------------------------------------------------------------------

__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
#if SOL_SM80
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src));
#endif
}
// .ca keeps the line in L1 -- only for reused sources (routing's pooled arrays).
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
