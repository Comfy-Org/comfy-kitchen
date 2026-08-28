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

// Shared layout contract for the CUDA Sol-Attn kernels. Both producers of the
// workspace carriers (preprocess, chunked producer) and the consumers (route,
// exact) must agree exactly on permutations and swizzles; a drift is invisible
// to either side's own test. The per-tile quantization helpers below are the
// single definition of how Q/K rows and query centroids become int8.

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

// ---------------------------------------------------------------------------
// Tile quantization, shared by the preprocess and the chunked producer. All of
// it assumes a 128-thread CTA working on one staged 64 x 128 bf16 tile.
// ---------------------------------------------------------------------------

// Row stride of a staged tile, in bf16; x2 bytes stays a multiple of 16 for
// the uint4 staging stores.
constexpr int LD_TILE = HEAD_DIM + 8;

__device__ __forceinline__ int8_t q8(float x, float inv) {
    return (int8_t)max(-127, min(127, __float2int_rn(x * inv)));
}

// Block-wide reductions over 128 threads (one per channel); every thread gets
// the result, and `s` is free for reuse on return.
__device__ __forceinline__ float block_max128(float x, float* s) {
    const int d = threadIdx.x;
    s[d] = x;
    __syncthreads();
    for (int w = 64; w; w >>= 1) {
        if (d < w) s[d] = fmaxf(s[d], s[d + w]);
        __syncthreads();
    }
    const float r = s[0];
    __syncthreads();
    return r;
}
__device__ __forceinline__ float block_sum128(float x, float* s) {
    const int d = threadIdx.x;
    s[d] = x;
    __syncthreads();
    for (int w = 64; w; w >>= 1) {
        if (d < w) s[d] += s[d + w];
        __syncthreads();
    }
    const float r = s[0];
    __syncthreads();
    return r;
}

// Stage rows 0..len-1 of a (token, channel) bf16 source into the tile, zero
// past len. Row t is read from src + t * stride (16 B loads, so the last dim
// must be contiguous and 16 B aligned).
__device__ __forceinline__ void stage_tile64(
    __nv_bfloat16* tile, const __nv_bfloat16* __restrict__ src, int64_t stride, int len)
{
    for (int idx = threadIdx.x; idx < BLOCK * (HEAD_DIM / 8); idx += HEAD_DIM) {
        const int t = idx / (HEAD_DIM / 8), c8 = (idx % (HEAD_DIM / 8)) * 8;
        uint4 val = make_uint4(0u, 0u, 0u, 0u);
        if (t < len)
            val = *reinterpret_cast<const uint4*>(src + (int64_t)t * stride + c8);
        *reinterpret_cast<uint4*>(tile + t * LD_TILE + c8) = val;
    }
}

// Per-token absmax scale + perm_d'd int8 row, one thread per token. qiP / qs
// point at (this tile's first token, this head): row t lands at +t*H*HEAD_DIM
// and +t*H. The permuted row is built in registers and stored as uint4s;
// writing qiP[perm_d(d)] directly costs 128 scattered byte stores per token.
__device__ __forceinline__ void quant_q_rows(
    const __nv_bfloat16* tile, int len, int8_t* __restrict__ qiP, float* __restrict__ qs, int H)
{
    for (int t = threadIdx.x; t < len; t += HEAD_DIM) {
        const __nv_bfloat16* row = tile + t * LD_TILE;
        float a = 0.f;
        for (int d = 0; d < HEAD_DIM; ++d) a = fmaxf(a, fabsf(__bfloat162float(row[d])));
        const float sc = fmaxf(a / 127.0f, 1e-8f);
        qs[(size_t)t * H] = sc;
        const float inv = 1.f / sc;
        int8_t out[HEAD_DIM];
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) out[perm_d(d)] = q8(__bfloat162float(row[d]), inv);
        int8_t* dst = qiP + (size_t)t * H * HEAD_DIM;
        #pragma unroll
        for (int c = 0; c < HEAD_DIM; c += 16)
            *reinterpret_cast<uint4*>(dst + c) = *reinterpret_cast<const uint4*>(out + c);
    }
}

// Query-block centroid (channel mean over the live rows), quantized like a
// pseudo-row with the pooled keys' perm_d so their dot needs no unpermute.
// One thread per channel; returns this thread's channel mean. `sred` holds
// bytes on return -- the caller must sync before reusing it.
__device__ __forceinline__ float centroid_quant(
    const __nv_bfloat16* tile, int len, float* sred,
    int8_t* __restrict__ cen8, float* __restrict__ cens)
{
    const int d = threadIdx.x;
    float c = 0.f;
    for (int t = 0; t < len; ++t) c += __bfloat162float(tile[t * LD_TILE + d]);
    c /= (float)len;
    const float csc = fmaxf(block_max128(fabsf(c), sred) / 127.0f, 1e-8f);
    char* s8 = reinterpret_cast<char*>(sred);
    s8[perm_d(d)] = (char)q8(c, 1.f / csc);
    __syncthreads();
    if (d < HEAD_DIM / 16)
        reinterpret_cast<uint4*>(cen8)[d] = reinterpret_cast<const uint4*>(s8)[d];
    if (d == 0) *cens = csc;
    return c;
}

// Centred per-key scale + perm_d'd int8 row, one thread per destination row p,
// which takes SOURCE row perm_key(p) (the smem read absorbs the relabelling).
// kmean is this head's [HEAD_DIM] centering vector; kbias (log2 units, or
// null) is indexed by source row -- only the exact branch reads it, so biased
// blocks must be sink-routed. kiP / ksb point at destination row 0 of the
// block; dead rows get a zero scale, NEG bias and zero bytes.
__device__ __forceinline__ void quant_k_rows(
    const __nv_bfloat16* tile, int len, const float* __restrict__ kmean,
    const float* __restrict__ kbias, int8_t* __restrict__ kiP, float2* __restrict__ ksb)
{
    for (int p = threadIdx.x; p < BLOCK; p += HEAD_DIM) {
        const int s = perm_key(p);
        const bool live = s < len;
        const __nv_bfloat16* row = tile + s * LD_TILE;
        float a = 0.f;
        for (int d = 0; d < HEAD_DIM; ++d)
            a = fmaxf(a, fabsf(__bfloat162float(row[d]) - kmean[d]));
        const float sc = fmaxf(a / 127.0f, 1e-8f);
        const float bias = (kbias && live) ? kbias[s] : 0.f;
        ksb[p] = make_float2(live ? sc : 0.f, live ? bias : NEG);
        const float inv = 1.f / sc;
        int8_t out[HEAD_DIM];
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d)
            out[perm_d(d)] = live ? q8(__bfloat162float(row[d]) - kmean[d], inv) : (int8_t)0;
        #pragma unroll
        for (int c = 0; c < HEAD_DIM; c += 16)
            *reinterpret_cast<uint4*>(kiP + (size_t)p * HEAD_DIM + c) =
                *reinterpret_cast<const uint4*>(out + c);
    }
}

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
#if SOL_SM80
    asm volatile("mma.sync.aligned.m16n8k32.row.col.satfinite.s32.u8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
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
