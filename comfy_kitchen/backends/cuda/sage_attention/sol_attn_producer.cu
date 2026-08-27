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

// Sol-Attn chunked QKV producer: consumes a token-major slice of the fused
// qkv projection output ([M, 3*H*HD] bf16, B=1) and emits the attention
// carriers for those tokens directly -- RMSNorm + split-half RoPE applied
// once per token, never written back, so the full bf16 Q/K/V round trip
// (and its residency) disappears.
//
// K centering and V channel scaling need global statistics, so the caller
// provides LAST STEP's kmean / V scale: both are range optimisations, not
// correctness requirements (the per-token K scale absorbs any centering
// vector; the V scale carries a clip margin). Fresh statistics for the next
// step come out of `finish` (kmean from the pooled sums) and the vamax
// atomics here.
//
// Emits, per chunk: qiP/qs, kiP/ksb (perm_key + perm_d), vTi (transposed
// int8), vcT (block value sums), ksumP (block K sums, post-rope), cen8/cens
// (quantized query-block centroids). finish (sol_attn.cu core) turns the
// pooled sums into kciP/kcs/threshold and runs route + exact.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "sol_layout.cuh"

namespace {
using namespace sol;

constexpr int HD = HEAD_DIM, BLK = BLOCK;
constexpr int LDT = HD + 8;

__device__ __forceinline__ int8_t q8p(float x, float inv) {
    return (int8_t)max(-127, min(127, __float2int_rn(x * inv)));
}

// One CTA per (64-token block, head); q, k, v processed sequentially through
// one staged tile. qkv rows are 3*H*HD bf16; q at column h*HD, k at
// (H + h)*HD, v at (2H + h)*HD.
__global__ void sol_producer_kernel(
    const __nv_bfloat16* __restrict__ qkv,   // [M, 3*H*HD], chunk at token t0
    const float* __restrict__ fab,           // [T, rot, 2] packed rope coeffs
    const __nv_bfloat16* __restrict__ qw, const __nv_bfloat16* __restrict__ kw,
    const float* __restrict__ kmean,         // [H, HD] stale (may be zeros)
    const float* __restrict__ vscale,        // [H, HD] stale V scale (may be ~0 -> margin)
    int8_t* __restrict__ qiP, float* __restrict__ qs,
    int8_t* __restrict__ kiP, float2* __restrict__ ksb,
    int8_t* __restrict__ vTi, __nv_bfloat16* __restrict__ vcT,
    float* __restrict__ ksumP,               // [H, NPAD, HD] block K sums (post-rope)
    int8_t* __restrict__ cen8, float* __restrict__ cens,
    float* __restrict__ vamax_next,          // [H, HD] atomicMax accumulator
    float rope_eps, int rot,
    int t0, int M, int T, int Tp, int H, int NPAD, int NQ)
{
#if SOL_SM80
    __shared__ __nv_bfloat16 sT[BLK * LDT];
    __shared__ __align__(16) float sred[HD];
    const int blk_local = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
    const int tb0 = t0 + blk_local * BLK;              // absolute token start
    const int len = min(BLK, min(M - blk_local * BLK, T - tb0));
    if (len <= 0) return;
    const int nblk = tb0 / BLK;                        // global 64-block index
    const int64_t row_stride = (int64_t)3 * H * HD;

    // ---------------- Q phase ----------------
    for (int idx = tid; idx < BLK * (HD / 8); idx += HD) {
        const int t = idx / (HD / 8), c8 = (idx % (HD / 8)) * 8;
        uint4 val = make_uint4(0u, 0u, 0u, 0u);
        if (t < len)
            val = *reinterpret_cast<const uint4*>(
                qkv + (int64_t)(blk_local * BLK + t) * row_stride + (int64_t)h * HD + c8);
        *reinterpret_cast<uint4*>(sT + t * LDT + c8) = val;
    }
    __syncthreads();
    norm_rope_rows(sT, LDT, len, fab + (int64_t)tb0 * (rot * 2), qw, rope_eps, rot);
    __syncthreads();

    for (int t = tid; t < len; t += HD) {
        float a = 0.f;
        for (int d = 0; d < HD; ++d) a = fmaxf(a, fabsf(__bfloat162float(sT[t * LDT + d])));
        const float sc = fmaxf(a / 127.0f, 1e-8f);
        qs[((size_t)(tb0 + t)) * H + h] = sc;
        const size_t base = ((size_t)(tb0 + t) * H + h) * HD;
        const float inv = 1.f / sc;
        int8_t out[HD];
        #pragma unroll
        for (int d = 0; d < HD; ++d)
            out[perm_d(d)] = q8p(__bfloat162float(sT[t * LDT + d]), inv);
        #pragma unroll
        for (int c = 0; c < HD; c += 16)
            *reinterpret_cast<uint4*>(qiP + base + c) = *reinterpret_cast<const uint4*>(out + c);
    }
    __syncthreads();

    // query-block centroid, quantized in place (all-local quantities)
    {
        const int d = tid;
        float c = 0.f;
        for (int t = 0; t < len; ++t) c += __bfloat162float(sT[t * LDT + d]);
        c /= (float)len;
        sred[d] = fabsf(c);
        __syncthreads();
        for (int w = 64; w; w >>= 1) {
            if (d < w) sred[d] = fmaxf(sred[d], sred[d + w]);
            __syncthreads();
        }
        const float csc = fmaxf(sred[0] / 127.0f, 1e-8f);
        __syncthreads();
        char* s8 = reinterpret_cast<char*>(sred);
        s8[perm_d(d)] = (char)q8p(c, 1.f / csc);
        __syncthreads();
        const size_t cbase = ((size_t)h * NQ + nblk) * HD;
        if (d < HD / 16)
            reinterpret_cast<uint4*>(cen8 + cbase)[d] =
                reinterpret_cast<const uint4*>(s8)[d];
        if (d == 0) cens[(size_t)h * NQ + nblk] = csc;
    }
    __syncthreads();

    // ---------------- K phase ----------------
    for (int idx = tid; idx < BLK * (HD / 8); idx += HD) {
        const int t = idx / (HD / 8), c8 = (idx % (HD / 8)) * 8;
        uint4 val = make_uint4(0u, 0u, 0u, 0u);
        if (t < len)
            val = *reinterpret_cast<const uint4*>(
                qkv + (int64_t)(blk_local * BLK + t) * row_stride + (int64_t)(H + h) * HD + c8);
        *reinterpret_cast<uint4*>(sT + t * LDT + c8) = val;
    }
    __syncthreads();
    norm_rope_rows(sT, LDT, len, fab + (int64_t)tb0 * (rot * 2), kw, rope_eps, rot);
    __syncthreads();

    // block K sums (post-rope, uncentered) for the pooled route tensors
    {
        const int d = tid;
        float sk = 0.f;
        for (int t = 0; t < len; ++t) sk += __bfloat162float(sT[t * LDT + d]);
        ksumP[((size_t)h * NPAD + nblk) * HD + d] = sk;
    }

    for (int p = tid; p < BLK; p += HD) {
        const int s = perm_key(p);
        const bool live = s < len;
        const size_t dst = (size_t)h * Tp + nblk * BLK + p;
        float a = 0.f;
        for (int d = 0; d < HD; ++d)
            a = fmaxf(a, fabsf(__bfloat162float(sT[s * LDT + d]) - kmean[(size_t)h * HD + d]));
        const float sc = fmaxf(a / 127.0f, 1e-8f);
        ksb[dst] = make_float2(live ? sc : 0.f, live ? 0.f : NEG);
        const float inv = 1.f / sc;
        int8_t out[HD];
        #pragma unroll
        for (int d = 0; d < HD; ++d) {
            const float x = __bfloat162float(sT[s * LDT + d]) - kmean[(size_t)h * HD + d];
            out[perm_d(d)] = live ? q8p(x, inv) : (int8_t)0;
        }
        #pragma unroll
        for (int c = 0; c < HD; c += 16)
            *reinterpret_cast<uint4*>(kiP + dst * HD + c) = *reinterpret_cast<const uint4*>(out + c);
    }
    __syncthreads();

    // ---------------- V phase ----------------
    for (int idx = tid; idx < BLK * (HD / 8); idx += HD) {
        const int t = idx / (HD / 8), c8 = (idx % (HD / 8)) * 8;
        uint4 val = make_uint4(0u, 0u, 0u, 0u);
        if (t < len)
            val = *reinterpret_cast<const uint4*>(
                qkv + (int64_t)(blk_local * BLK + t) * row_stride + (int64_t)(2 * H + h) * HD + c8);
        *reinterpret_cast<uint4*>(sT + t * LDT + c8) = val;
    }
    __syncthreads();
    {
        const int d = tid;
        const float vsc = vscale[(size_t)h * HD + d];
        const float inv = 1.f / vsc;
        float sv = 0.f, av = 0.f;
        // vTi: raw channel rows, KEY axis takes perm_d per 64-block (the
        // exact kernel's PV repack depends on it -- see sol_attn_vtranspose.cu).
        int8_t col[BLK];
        for (int t = 0; t < BLK; ++t) {
            const float x = (t < len) ? __bfloat162float(sT[t * LDT + d]) : 0.f;
            sv += x; av = fmaxf(av, fabsf(x));
            col[perm_d(t)] = q8p(x, inv);
        }
        const size_t vbase = ((size_t)h * HD + d) * Tp + nblk * BLK;
        #pragma unroll
        for (int c = 0; c < BLK; c += 16)
            *reinterpret_cast<uint4*>(vTi + vbase + c) = *reinterpret_cast<const uint4*>(col + c);
        vcT[((size_t)h * HD + d) * NPAD + nblk] = __float2bfloat16(sv);
        atomicMax(reinterpret_cast<unsigned int*>(&vamax_next[(size_t)h * HD + d]),
                  __float_as_uint(av));
    }
#endif  // SOL_SM80
}

}  // namespace

extern "C" void launch_sol_producer(
    const void* qkv, const void* fab, const void* qw, const void* kw,
    const void* kmean, const void* vscale,
    void* qiP, void* qs, void* kiP, void* ksb, void* vTi, void* vcT,
    void* ksumP, void* cen8, void* cens, void* vamax_next,
    float rope_eps, int rot,
    int t0, int M, int T, int Tp, int H, int NPAD, int NQ,
    cudaStream_t stream)
{
    const int nblocks = (M + BLK - 1) / BLK;
    sol_producer_kernel<<<dim3(nblocks, H), HD, 0, stream>>>(
        (const __nv_bfloat16*)qkv, (const float*)fab,
        (const __nv_bfloat16*)qw, (const __nv_bfloat16*)kw,
        (const float*)kmean, (const float*)vscale,
        (int8_t*)qiP, (float*)qs, (int8_t*)kiP, (float2*)ksb,
        (int8_t*)vTi, (__nv_bfloat16*)vcT, (float*)ksumP,
        (int8_t*)cen8, (float*)cens, (float*)vamax_next,
        rope_eps, rot, t0, M, T, Tp, H, NPAD, NQ);
}
