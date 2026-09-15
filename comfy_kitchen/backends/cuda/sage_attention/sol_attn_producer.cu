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

// Sol-Attn chunked QKV producer: a token-major slice of the fused qkv
// projection ([M, 3*H*HD] bf16, B=1) -> the workspace carriers for those
// tokens, with RMSNorm + RoPE applied in-tile, so full bf16 Q/K/V never exist.
//
// K centering and V scaling use LAST step's kmean / V scale (range
// optimisations only: the per-token K scale absorbs any centering vector, the
// V scale carries a clip margin). Next-step statistics come from the pooled
// K sums (launch_sol_finish) and the vamax atomics here.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "sol_layout.cuh"

namespace {
using namespace sol;

constexpr int HD = HEAD_DIM, BLK = BLOCK;

// One CTA per (64-token block, head); q, k, v pass through one staged tile.
// qkv row = [q | k | v], each H*HD wide.
__global__ void sol_producer_kernel(
    const __nv_bfloat16* __restrict__ qkv,   // [M, 3*H*HD], chunk at token t0
    const float* __restrict__ fab,           // [T, rot, 2] packed rope coeffs
    const __nv_bfloat16* __restrict__ qw, const __nv_bfloat16* __restrict__ kw,
    const float* __restrict__ kmean,         // [H, HD] stale (may be zeros)
    const float* __restrict__ vscale,        // [H, HD] stale V scale (may be ~0 -> margin)
    int8_t* __restrict__ qiP, float* __restrict__ qs,
    int8_t* __restrict__ kiP, float2* __restrict__ ksb,
    int8_t* __restrict__ vTi, int8_t* __restrict__ vRow, __nv_bfloat16* __restrict__ vcT,
    float* __restrict__ ksumP,               // [H, NPAD, HD] block K sums (post-rope)
    int8_t* __restrict__ cen8, float* __restrict__ cens,
    float* __restrict__ qmean,               // [H, NPAD, HD] f32 block means (post-rope)
    float* __restrict__ vamax_next,          // [H, HD] atomicMax accumulator
    const int32_t* __restrict__ blen,        // [NTB] valid tokens per block, or null
    float rope_eps, int rot,
    int t0, int M, int T, int Tp, int H, int NPAD, int NQ)
{
    __shared__ __align__(16) __nv_bfloat16 sT[BLK * LD_TILE];
    __shared__ __align__(16) float sred[HD];
    const int blk_local = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
    const int tb0 = t0 + blk_local * BLK;              // absolute token start
    if (tb0 >= T) return;
    const int nblk = tb0 / BLK;                        // global 64-block index
    const int nrows = min(BLK, min(M - blk_local * BLK, T - tb0));   // rows that exist
    const int len = min(block_len_of(blen, nblk, T), nrows);         // rows that are live
    if (len <= 0) return;
    const int64_t row_stride = (int64_t)3 * H * HD;
    const __nv_bfloat16* rows = qkv + (int64_t)(blk_local * BLK) * row_stride;
    const float* fab_t0 = fab + (int64_t)tb0 * (rot * 2);

    // ---------------- Q phase ----------------
    stage_tile64(sT, rows + (int64_t)h * HD, row_stride, len);
    __syncthreads();
    norm_rope_rows(sT, LD_TILE, len, fab_t0, qw, rope_eps, rot);
    __syncthreads();
    quant_q_rows(sT, len, nrows, qiP + ((size_t)tb0 * H + h) * HD, qs + (size_t)tb0 * H + h, H);
    __syncthreads();
    const size_t qrow = (size_t)h * NQ + nblk;
    const float c = centroid_quant(sT, len, sred, cen8 + qrow * HD, cens + qrow);
    qmean[((size_t)h * NPAD + nblk) * HD + tid] = c;
    __syncthreads();

    // ---------------- K phase ----------------
    stage_tile64(sT, rows + (int64_t)(H + h) * HD, row_stride, len);
    __syncthreads();
    norm_rope_rows(sT, LD_TILE, len, fab_t0, kw, rope_eps, rot);
    __syncthreads();
    {
        // block K sums (post-rope, uncentered) for the pooled tensors
        float sk = 0.f;
        for (int t = 0; t < len; ++t) sk += __bfloat162float(sT[t * LD_TILE + tid]);
        ksumP[((size_t)h * NPAD + nblk) * HD + tid] = sk;
    }
    const size_t dst0 = (size_t)h * Tp + nblk * BLK;
    quant_k_rows(sT, len, kmean + (size_t)h * HD, nullptr, kiP + dst0 * HD, ksb + dst0);
    __syncthreads();

    // ---------------- V phase ----------------
    stage_tile64(sT, rows + (int64_t)(2 * H + h) * HD, row_stride, len);
    __syncthreads();
    {
        const int d = tid;
        const float inv = 1.f / vscale[(size_t)h * HD + d];
        float sv = 0.f, av = 0.f;
        // vTi: raw channel rows, perm_d on the KEY axis per 64-block
        __align__(16) int8_t col[BLK];
        for (int t = 0; t < BLK; ++t) {
            const float x = (t < len) ? __bfloat162float(sT[t * LD_TILE + d]) : 0.f;
            sv += x; av = fmaxf(av, fabsf(x));
            col[perm_d(t)] = q8(x, inv);
            if (vRow) vRow[((size_t)h * Tp + nblk * BLK + t) * HD + d] = col[perm_d(t)];   // row-major copy (token routing)
        }
        const size_t vbase = ((size_t)h * HD + d) * Tp + nblk * BLK;
        #pragma unroll
        for (int c = 0; c < BLK; c += 16)
            *reinterpret_cast<uint4*>(vTi + vbase + c) = *reinterpret_cast<const uint4*>(col + c);
        vcT[((size_t)h * HD + d) * NPAD + nblk] = __float2bfloat16(sv);
        atomicMax(reinterpret_cast<unsigned int*>(&vamax_next[(size_t)h * HD + d]),
                  __float_as_uint(av));
    }
}

}  // namespace

void launch_sol_producer(
    const void* qkv, const void* fab, const void* qw, const void* kw,
    const void* kmean, const void* vscale,
    void* qiP, void* qs, void* kiP, void* ksb, void* vTi, void* vRow, void* vcT,
    void* ksumP, void* cen8, void* cens, void* qmean, void* vamax_next,
    const void* blen, float rope_eps, int rot,
    int t0, int M, int T, int Tp, int H, int NPAD, int NQ,
    cudaStream_t stream)
{
    const int nblocks = (M + BLK - 1) / BLK;
    sol_producer_kernel<<<dim3(nblocks, H), HD, 0, stream>>>(
        (const __nv_bfloat16*)qkv, (const float*)fab,
        (const __nv_bfloat16*)qw, (const __nv_bfloat16*)kw,
        (const float*)kmean, (const float*)vscale,
        (int8_t*)qiP, (float*)qs, (int8_t*)kiP, (float2*)ksb,
        (int8_t*)vTi, (int8_t*)vRow, (__nv_bfloat16*)vcT, (float*)ksumP,
        (int8_t*)cen8, (float*)cens, (float*)qmean, (float*)vamax_next,
        (const int32_t*)blen, rope_eps, rot, t0, M, T, Tp, H, NPAD, NQ);
}
