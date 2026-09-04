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

// Layout contract shared by every Sol-Attn kernel: permutations, smem
// swizzles, tile quantization, MMA / cp.async wrappers. Both carrier producers
// (preprocess, chunked producer) and both consumers (route, exact) rely on it.

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
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
constexpr int NTOK_MAX = 256;        // token routing: largest token budget
constexpr int TOK_HIST_BINS = 128;   // token routing: histogram bins per centroid
constexpr int TOK_GROUP = 2;         // token routing: query blocks per centroid

// Contraction-axis permutation: each lane's two MMA operand words become one
// 8-byte load. Applied to Q/K/pooled-K d axes and V^T's key axis.
__host__ __device__ __forceinline__ int perm_d(int d) {
    const int kc = d >> 5, rem = d & 31, h = rem >> 4, r2 = rem & 15;
    return kc * 32 + 8 * (r2 >> 2) + 4 * h + (r2 & 3);
}

// Key relabelling per 64-block so the INT8 PV A operand needs no shuffles.
// Applied to K rows + scales; NOT to V^T.
__host__ __device__ __forceinline__ int perm_key(int p) {
    return 16 * (p >> 4) + 4 * ((p & 7) >> 1) + 2 * ((p >> 3) & 1) + (p & 1);
}
// Which stored row holds source token s of its block (perm_key's inverse).
__host__ __device__ __forceinline__ int perm_key_inv(int s) {
    return 16 * (s >> 4) + 8 * ((s >> 1) & 1) + 4 * ((s >> 3) & 1) + 2 * ((s >> 2) & 1) + (s & 1);
}

// Smem XOR swizzles (K tile 64 x 128 B, V^T tile 128 x 64 B). Verify any
// change by enumerating both 16-lane LDS.64 phases against 32 banks.
__device__ __forceinline__ int swz_k(int row) { return (row & 3) * 2; }
__device__ __forceinline__ int swz_v(int col) { return ((col >> 2) ^ col) & 3; }

// Valid tokens at the FRONT of 64-block n: the caller's per-block table
// (zero-padded tiles, clamped to [1, rows that exist]) or the plain ragged
// tail. Rows past it are dead.
__device__ __forceinline__ int block_len_of(const int32_t* blen, int n, int T) {
    const int rem = min(BLOCK, T - n * BLOCK);
    return blen ? min(rem, max(1, blen[n])) : rem;
}

// ---- Tile quantization: one 128-thread CTA per staged 64 x 128 16-bit tile ----

constexpr int LD_TILE = HEAD_DIM + 8;   // 16-bit row stride; keeps uint4 stores aligned

__device__ __forceinline__ int8_t q8(float x, float inv) {
    return (int8_t)max(-127, min(127, __float2int_rn(x * inv)));
}

// 128-thread reductions, one thread per channel; every thread gets the result
// and `s` is free on return.
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

// q/k/v/out element type: bf16 or fp16. Everything after the load (tiles,
// scales, int8 carriers) is the same; only the loads and the final store
// convert.
enum SolElem : int { SOL_BF16 = 0, SOL_FP16 = 1 };
__device__ __forceinline__ float to_f32(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float to_f32(__half x) { return __half2float(x); }
template <typename T> __device__ __forceinline__ T from_f32(float x);
template <> __device__ __forceinline__ __nv_bfloat16 from_f32<__nv_bfloat16>(float x) { return __float2bfloat16(x); }
template <> __device__ __forceinline__ __half from_f32<__half>(float x) { return __float2half(x); }

// Stage rows 0..len-1 (row t at src + t*stride, 16 B loads) into the tile,
// zero past len.
template <typename T>
__device__ __forceinline__ void stage_tile64(
    T* tile, const T* __restrict__ src, int64_t stride, int len)
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
// point at (this tile's first token, this head). Rows [len, nrows) exist in
// memory but are dead (zero-padded tiles) and get zeros; rows past nrows do
// not exist. The permuted row is built in registers: scattered byte stores
// cost 128 per token.
template <typename T>
__device__ __forceinline__ void quant_q_rows(
    const T* tile, int len, int nrows,
    int8_t* __restrict__ qiP, float* __restrict__ qs, int H)
{
    for (int t = threadIdx.x; t < nrows; t += HEAD_DIM) {
        const T* row = tile + t * LD_TILE;   // zero-staged past len
        const bool live = t < len;
        float a = 0.f;
        #pragma unroll 8   // unbounded, nvcc hoists all 128 loads: 168 regs in the producer
        for (int d = 0; d < HEAD_DIM; ++d) a = fmaxf(a, fabsf(to_f32(row[d])));
        const float sc = live ? fmaxf(a / 127.0f, 1e-8f) : 0.f;   // dead rows: deterministic zeros
        qs[(size_t)t * H] = sc;
        const float inv = live ? 1.f / sc : 0.f;
        __align__(16) int8_t out[HEAD_DIM];
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) out[perm_d(d)] = q8(to_f32(row[d]), inv);
        int8_t* dst = qiP + (size_t)t * H * HEAD_DIM;
        #pragma unroll
        for (int c = 0; c < HEAD_DIM; c += 16)
            *reinterpret_cast<uint4*>(dst + c) = *reinterpret_cast<const uint4*>(out + c);
    }
}

// Query-block centroid, quantized like a pseudo-row with the pooled keys'
// perm_d. One thread per channel; returns this thread's channel mean. `sred`
// holds bytes on return -- sync before reusing it.
template <typename T>
__device__ __forceinline__ float centroid_quant(
    const T* tile, int len, float* sred,
    int8_t* __restrict__ cen8, float* __restrict__ cens)
{
    const int d = threadIdx.x;
    float c = 0.f;
    for (int t = 0; t < len; ++t) c += to_f32(tile[t * LD_TILE + d]);
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

// Centred per-key scale + perm_d'd int8 row; destination row p takes SOURCE
// row perm_key(p). kbias (log2 units, or null) is indexed by source row and
// only the exact branch reads it, so biased blocks must be sink-routed. Dead
// rows get a zero scale, NEG bias and zero bytes.
template <typename T>
__device__ __forceinline__ void quant_k_rows(
    const T* tile, int len, const float* __restrict__ kmean,
    const float* __restrict__ kbias, int8_t* __restrict__ kiP, float2* __restrict__ ksb)
{
    for (int p = threadIdx.x; p < BLOCK; p += HEAD_DIM) {
        const int s = perm_key(p);
        const bool live = s < len;
        const T* row = tile + s * LD_TILE;
        float a = 0.f;
        for (int d = 0; d < HEAD_DIM; ++d)
            a = fmaxf(a, fabsf(to_f32(row[d]) - kmean[d]));
        const float sc = fmaxf(a / 127.0f, 1e-8f);
        const float bias = (kbias && live) ? kbias[s] : 0.f;
        ksb[p] = make_float2(live ? sc : 0.f, live ? bias : NEG);
        const float inv = 1.f / sc;
        __align__(16) int8_t out[HEAD_DIM];
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d)
            out[perm_d(d)] = live ? q8(to_f32(row[d]) - kmean[d], inv) : (int8_t)0;
        #pragma unroll
        for (int c = 0; c < HEAD_DIM; c += 16)
            *reinterpret_cast<uint4*>(kiP + (size_t)p * HEAD_DIM + c) =
                *reinterpret_cast<const uint4*>(out + c);
    }
}

// Fused per-head RMSNorm + split-half RoPE in place; matches ops/rms_rope.cu
// bit-for-bit (incl. its bf16 rounding between norm and rotation). One warp
// per token, lane owns channels 4*lane..+3.
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
        // Explicit source lane: shfl_xor is only correct for power-of-two
        // offsets and rot/8 need not be one (H3 rot=96 -> 12).
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

// ---- MMA wrappers (int8 m16n8k32 issues at full rate on sm_120) ----

__device__ __forceinline__ void mma_s8(int32_t* d, const uint32_t* a, const uint32_t* b) {
#if SOL_SM80
    asm volatile("mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

// u8 P: 255 levels instead of 127.
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

// ---- Shared 64-key tile body (exact kernel and token passes) ----
// Staged K tile (64 x 128 int8) and V^T tile (128 x 64 int8), XOR-swizzled 16-byte
// chunks. One warp owns 16 rows: g = lane >> 2 picks rows g and g + 8, qd = lane & 3
// the column pair, so every per-row array below is [2].

constexpr int TILE_KC  = HEAD_DIM / 32;   // int8 k-chunks of S = Q.K^T
constexpr int TILE_NKT = BLOCK / 8;       // score n8 tiles
constexpr int TILE_NT  = HEAD_DIM / 8;    // output n8 tiles
constexpr int TILE_PKC = BLOCK / 32;      // int8 k-chunks of O += P.V
constexpr int TILE_LDK = HEAD_DIM;        // K tile row stride, bytes
constexpr int TILE_LDV = BLOCK;           // V^T tile row stride, bytes
constexpr float TILE_MASKED = -1.0e37f;   // scores at or below this are masked (NEG lands here)

// S = Q.K^T: the warp's 16 rows against the tile's 64 keys, int32 accumulators.
__device__ __forceinline__ void tile_qk(const int8_t* sK_tile, const uint32_t (&qa)[TILE_KC][4],
                                        int g, int qd, int32_t (&s_acc)[TILE_NKT][4]) {
    #pragma unroll
    for (int nt = 0; nt < TILE_NKT; ++nt) {
        s_acc[nt][0] = 0; s_acc[nt][1] = 0; s_acc[nt][2] = 0; s_acc[nt][3] = 0;
        const int R = nt * 8 + g;
        const int8_t* krow = sK_tile + R * TILE_LDK + ((qd & 1) << 3);
        const int swk = swz_k(R), qhi = qd >> 1;
        #pragma unroll
        for (int kc = 0; kc < TILE_KC; ++kc) {
            const uint2 kb = *reinterpret_cast<const uint2*>(krow + (((kc * 2 + qhi) ^ swk) << 4));
            uint32_t kbf[2] = {kb.x, kb.y};
            mma_s8(s_acc[nt], qa[kc], kbf);
        }
    }
}

// Scores in log2 units: (int dot) * qsc[row] * ks + bias, with the tile's 64
// (ks, bias) pairs at kb_src. FOLD_MAX takes the row max in the same loop (the
// exact kernel's shape; splitting it costs registers); otherwise call tile_rowmax.
template <bool FOLD_MAX>
__device__ __forceinline__ void tile_scores(const int32_t (&s_acc)[TILE_NKT][4], const float2* kb_src,
                                            const float (&qsc)[2], int qd, float (&p_val)[TILE_NKT][4],
                                            float (&bmax)[2]) {
    #pragma unroll
    for (int nt = 0; nt < TILE_NKT; ++nt) {
        const int c0 = nt * 8 + qd * 2;
        const float4 kb4 = *reinterpret_cast<const float4*>(kb_src + c0);
        const float k0s = kb4.x, m0 = kb4.y, k1s = kb4.z, m1 = kb4.w;
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int row = e >> 1;
            const float s = (e & 1) ? fmaf((float)s_acc[nt][e], qsc[row] * k1s, m1)
                                    : fmaf((float)s_acc[nt][e], qsc[row] * k0s, m0);
            p_val[nt][e] = s;
            if (FOLD_MAX) bmax[row] = fmaxf(bmax[row], s);
        }
    }
}

// Row max over the tile's scores (<= TILE_MASKED masked), and whether any is live.
__device__ __forceinline__ void tile_rowmax(const float (&p_val)[TILE_NKT][4], float (&bmax)[2], bool (&has)[2]) {
    #pragma unroll
    for (int nt = 0; nt < TILE_NKT; ++nt) {
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int row = e >> 1;
            if (p_val[nt][e] > TILE_MASKED) { has[row] = true; bmax[row] = fmaxf(bmax[row], p_val[nt][e]); }
        }
    }
}

// Online softmax over the tile's scores (<= TILE_MASKED masked) and O += P.V.
// P is u8 against the block max bmax; l sums the packed bytes so numerator and
// denominator quantize identically. (m_r, l_r, c_r) is the row state, c_r the
// scale of o_acc and l_r. GUARD: a row with no live score keeps its state
// (alpha 1, P 0); the exact kernel only sees live tiles and skips it.
template <bool GUARD>
__device__ __forceinline__ void tile_softmax_pv(float (&p_val)[TILE_NKT][4], float (&bmax)[2], bool (&has)[2],
                                                const int8_t* sVt_tile, int g, int qd,
                                                float (&m_r)[2], float (&l_r)[2], float (&c_r)[2],
                                                float (&o_acc)[TILE_NT][4]) {
    #pragma unroll
    for (int off = 1; off <= 2; off <<= 1) {
        bmax[0] = fmaxf(bmax[0], __shfl_xor_sync(0xffffffffu, bmax[0], off));
        bmax[1] = fmaxf(bmax[1], __shfl_xor_sync(0xffffffffu, bmax[1], off));
        if (GUARD) {
            has[0] |= __shfl_xor_sync(0xffffffffu, (int)has[0], off);
            has[1] |= __shfl_xor_sync(0xffffffffu, (int)has[1], off);
        }
    }
    if (GUARD) {
        if (!has[0]) bmax[0] = c_r[0];
        if (!has[1]) bmax[1] = c_r[1];
    }
    const float alpha0 = exp2f(c_r[0] - bmax[0]);
    const float alpha1 = exp2f(c_r[1] - bmax[1]);
    c_r[0] = bmax[0]; c_r[1] = bmax[1];
    m_r[0] = fmaxf(m_r[0], bmax[0]);
    m_r[1] = fmaxf(m_r[1], bmax[1]);

    // u8 P scale folded into the exponent (+log2 255); l carries it too
    const float m_off[2] = {GUARD && !has[0] ? 3.0e38f : bmax[0] - 7.99435344f,
                            GUARD && !has[1] ? 3.0e38f : bmax[1] - 7.99435344f};
    #pragma unroll
    for (int nt = 0; nt < TILE_NKT; ++nt) {
        #pragma unroll
        for (int e = 0; e < 4; ++e)
            p_val[nt][e] = exp2f(p_val[nt][e] - m_off[e >> 1]);
    }

    // free repack (see the header comment): n-tiles (4kk, 4kk+1) -> keys 32kk+4q..+3
    uint32_t pa[TILE_PKC][4];
    #pragma unroll
    for (int kk = 0; kk < TILE_PKC; ++kk) {
        const int b0 = 4 * kk, b1 = b0 + 1, b2 = b0 + 2, b3 = b0 + 3;
        pa[kk][0] = mma::pack_u8x4(p_val[b0][0], p_val[b0][1], p_val[b1][0], p_val[b1][1]);
        pa[kk][1] = mma::pack_u8x4(p_val[b0][2], p_val[b0][3], p_val[b1][2], p_val[b1][3]);
        pa[kk][2] = mma::pack_u8x4(p_val[b2][0], p_val[b2][1], p_val[b3][0], p_val[b3][1]);
        pa[kk][3] = mma::pack_u8x4(p_val[b2][2], p_val[b2][3], p_val[b3][2], p_val[b3][3]);
    }
    uint32_t li[2] = {0, 0};
    #pragma unroll
    for (int kk = 0; kk < TILE_PKC; ++kk) {
        li[0] = __dp4a(pa[kk][0], 0x01010101u, li[0]);
        li[0] = __dp4a(pa[kk][2], 0x01010101u, li[0]);
        li[1] = __dp4a(pa[kk][1], 0x01010101u, li[1]);
        li[1] = __dp4a(pa[kk][3], 0x01010101u, li[1]);
    }
    #pragma unroll
    for (int off = 1; off <= 2; off <<= 1) {
        li[0] += __shfl_xor_sync(0xffffffffu, li[0], off);
        li[1] += __shfl_xor_sync(0xffffffffu, li[1], off);
    }
    l_r[0] = l_r[0] * alpha0 + (float)li[0];
    l_r[1] = l_r[1] * alpha1 + (float)li[1];

    #pragma unroll
    for (int nt = 0; nt < TILE_NT; ++nt) {
        int32_t d[4] = {0, 0, 0, 0};
        const int C = nt * 8 + g;
        const int8_t* vcol = sVt_tile + C * TILE_LDV + ((qd & 1) << 3);
        const int swv = swz_v(C), qhi2 = qd >> 1;
        #pragma unroll
        for (int kk = 0; kk < TILE_PKC; ++kk) {
            const uint2 vb = *reinterpret_cast<const uint2*>(vcol + (((kk * 2 + qhi2) ^ swv) << 4));
            uint32_t vbf[2] = {vb.x, vb.y};
            mma_u8s8(d, pa[kk], vbf);
        }
        o_acc[nt][0] = fmaf(o_acc[nt][0], alpha0, (float)d[0]);
        o_acc[nt][1] = fmaf(o_acc[nt][1], alpha0, (float)d[1]);
        o_acc[nt][2] = fmaf(o_acc[nt][2], alpha1, (float)d[2]);
        o_acc[nt][3] = fmaf(o_acc[nt][3], alpha1, (float)d[3]);
    }
}

__device__ __forceinline__ uint32_t pack_bf2(float lo, float hi) {
    __nv_bfloat162 p = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&p);
}

// ---- cp.async ----

__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
#if SOL_SM80
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src));
#endif
}
// .ca: L1-cached, for reused sources (routing's pooled arrays).
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
