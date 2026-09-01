// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Sol-Attn chunked QKV producer: a token-major slice of the fused qkv projection
// ([M, 3*H*HD] bf16, B=1) -> the workspace carriers for those tokens, with RMSNorm
// and RoPE applied in-tile, so full bf16 Q/K/V never exist.
//
// K centering and V scaling use LAST step's kmean / V scale (range optimisations
// only: the per-token K scale absorbs any centering vector, the V scale carries a
// clip margin). Next-step statistics come from the pooled K sums
// (launch_sol_finish) and the vamax atomics here.
#include <hip/hip_runtime.h>

#include <cstdint>

#include "sol_common.h"

namespace comfy::hip_backend::sol {
namespace {

constexpr int HD = kHeadDim, BLK = kBlock;

// One workgroup per (64-token block, head); q, k, v pass through one staged tile.
// qkv row = [q | k | v], each H*HD wide.
__global__ __launch_bounds__(HD) void sol_producer_kernel(
    const __bf16* __restrict__ qkv,  // [M, 3*H*HD], chunk at token t0
    const float* __restrict__ fab,   // [T, rot, 2] packed rope coeffs
    const __bf16* __restrict__ qw, const __bf16* __restrict__ kw,
    const float* __restrict__ kmean,   // [H, HD] stale (may be zeros)
    const float* __restrict__ vscale,  // [H, HD] stale V scale
    int8_t* __restrict__ qi, float* __restrict__ qs, int8_t* __restrict__ ki,
    float2* __restrict__ ksb, int8_t* __restrict__ vT, __bf16* __restrict__ vcT,
    float* __restrict__ ksum,  // [H, NPAD, HD] block K sums (post-rope)
    int8_t* __restrict__ cen8, float* __restrict__ cens,
    float* __restrict__ qmean,       // [H, NPAD, HD] f32 block means (post-rope)
    float* __restrict__ vamax_next,  // [H, HD] atomicMax accumulator
    const int32_t* __restrict__ blen, float rope_eps, int rot, int t0, int M, int T, int Tp, int H,
    int NPAD, int NQ) {
    __shared__ __attribute__((aligned(16))) __bf16 sT[BLK * kLdTile];
    __shared__ __attribute__((aligned(16))) float sred[HD];
    const int blk_local = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
    const int tb0 = t0 + blk_local * BLK;  // absolute token start
    if (tb0 >= T) return;
    const int nblk = tb0 / BLK;  // global 64-block index
    const int nrows = imin(BLK, imin(M - blk_local * BLK, T - tb0));   // rows that exist
    const int len = imin(block_len_of(blen, nblk, T), nrows);          // rows that are live
    if (len <= 0) return;
    const int64_t row_stride = static_cast<int64_t>(3) * H * HD;
    const __bf16* rows = qkv + static_cast<int64_t>(blk_local * BLK) * row_stride;
    const float* fab_t0 = fab + static_cast<int64_t>(tb0) * (rot * 2);

    // ---------------- Q phase ----------------
    stage_tile64(sT, rows + static_cast<int64_t>(h) * HD, row_stride, len);
    __syncthreads();
    norm_rope_rows(sT, kLdTile, len, fab_t0, qw, rope_eps, rot);
    __syncthreads();
    quant_q_rows(sT, len, nrows, qi + (static_cast<size_t>(tb0) * H + h) * HD,
                 qs + static_cast<size_t>(tb0) * H + h, H);
    __syncthreads();
    const size_t qrow = static_cast<size_t>(h) * NQ + nblk;
    const float c = centroid_quant(sT, len, sred, cen8 + qrow * HD, cens + qrow);
    qmean[(static_cast<size_t>(h) * NPAD + nblk) * HD + tid] = c;
    __syncthreads();

    // ---------------- K phase ----------------
    stage_tile64(sT, rows + static_cast<int64_t>(H + h) * HD, row_stride, len);
    __syncthreads();
    norm_rope_rows(sT, kLdTile, len, fab_t0, kw, rope_eps, rot);
    __syncthreads();
    {
        // block K sums (post-rope, uncentered) for the pooled tensors
        float sk = 0.f;
        for (int t = 0; t < len; ++t) sk += static_cast<float>(sT[t * kLdTile + tid]);
        ksum[(static_cast<size_t>(h) * NPAD + nblk) * HD + tid] = sk;
    }
    const size_t dst0 = static_cast<size_t>(h) * Tp + nblk * BLK;
    quant_k_rows(sT, len, kmean + static_cast<size_t>(h) * HD, nullptr, ki + dst0 * HD,
                 ksb + dst0);
    __syncthreads();

    // ---------------- V phase ----------------
    stage_tile64(sT, rows + static_cast<int64_t>(2 * H + h) * HD, row_stride, len);
    __syncthreads();
    {
        const int d = tid;
        const float inv = 1.f / vscale[static_cast<size_t>(h) * HD + d];
        float sv = 0.f, av = 0.f;
        // vT: raw channel rows, keys in natural order
        __attribute__((aligned(16))) int8_t col[16];
        const size_t vbase = (static_cast<size_t>(h) * HD + d) * Tp + nblk * BLK;
#pragma unroll
        for (int c0 = 0; c0 < BLK; c0 += 16) {
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const int t = c0 + j;
                const float x = (t < len) ? static_cast<float>(sT[t * kLdTile + d]) : 0.f;
                sv += x;
                av = fmaxf(av, fabsf(x));
                col[j] = q8(x, inv);
            }
            *reinterpret_cast<uint4*>(vT + vbase + c0) = *reinterpret_cast<const uint4*>(col);
        }
        vcT[(static_cast<size_t>(h) * HD + d) * NPAD + nblk] = static_cast<__bf16>(sv);
        atomicMax(reinterpret_cast<unsigned int*>(&vamax_next[static_cast<size_t>(h) * HD + d]),
                  __float_as_uint(av));
    }
}

}  // namespace
}  // namespace comfy::hip_backend::sol

using namespace comfy::hip_backend::sol;

void launch_sol_producer(const void* qkv, const void* fab, const void* qw, const void* kw,
                         const void* kmean, const void* vscale, void* qi, void* qs, void* ki,
                         void* ksb, void* vT, void* vcT, void* ksum, void* cen8, void* cens,
                         void* qmean, void* vamax_next, const void* blen, float rope_eps, int rot,
                         int t0, int M, int T, int Tp, int H, int NPAD, int NQ,
                         hipStream_t stream) {
    const int nblocks = (M + BLK - 1) / BLK;
    if (nblocks <= 0) return;
    sol_producer_kernel<<<dim3(nblocks, H), HD, 0, stream>>>(
        static_cast<const __bf16*>(qkv), static_cast<const float*>(fab),
        static_cast<const __bf16*>(qw), static_cast<const __bf16*>(kw),
        static_cast<const float*>(kmean), static_cast<const float*>(vscale),
        static_cast<int8_t*>(qi), static_cast<float*>(qs), static_cast<int8_t*>(ki),
        static_cast<float2*>(ksb), static_cast<int8_t*>(vT), static_cast<__bf16*>(vcT),
        static_cast<float*>(ksum), static_cast<int8_t*>(cen8), static_cast<float*>(cens),
        static_cast<float*>(qmean), static_cast<float*>(vamax_next),
        static_cast<const int32_t*>(blen), rope_eps, rot, t0, M, T, Tp, H, NPAD, NQ);
}
