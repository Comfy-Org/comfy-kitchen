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

// Sol-Attn routing + approximate pass, CUDA.
//
// The routing decision has always been a centroid quantity: the column sum it
// thresholds is sum_i q_i . kc = len * (centroid(Q_block) . kc). This kernel
// extends that to the approximate branch's VALUES: all rows of a query block
// share their centroid's tail. Measured on the eager reference the change
// costs -0.0005 cosine, flat across tau and length; what it buys is this pass
// shrinking from [T rows x N pooled blocks] to [N x N] -- 64x less math -- and
// the handover state (o_part, m_part, l_part) shrinking 64x, which also
// retires the trick of aliasing o_part onto the caller's output that the
// per-row state needed.
//
// One warp per 64-token query block; the routing mask itself stays per-64.
// Lanes cover 32 pooled blocks per chunk for the scores and the decision (one
// int8 dp4a dot each, the old kernel's precision), and 4 channels each for the
// tail accumulation. Pooled tiles are staged to shared memory once per CTA and
// shared by all 8 warps.
//
// Emits per (batch, head, query block):
//   blk_idx / blk_cnt        the routed list the exact kernel walks
//   o_part, m_part, l_part   ONE online-softmax state per query block, in the
//                            exact kernel's units: o pre-divided by the
//                            per-channel V scale and, like l, pre-multiplied
//                            by 255 (the u8 P scale) (keeps the scale out of its inner loop).

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "sol_layout.cuh"

namespace {
using namespace sol;

constexpr int HD  = HEAD_DIM;
constexpr int WPB = 8;               // warps (= query blocks) per CTA
constexpr int NTHREADS = WPB * 32;
constexpr int CH  = 32;              // pooled blocks staged per chunk
// sVc row stride in halves. The tail reads sVc[(4*lane+i)*LDV2 + jj], so the
// bank a lane lands on advances by (2*LDV2 mod 32) per lane: 40 puts every
// lane on one of TWO banks (16-way conflict, measured as the kernel's
// bottleneck); 34 spreads them over 8 (4-way), the best an even stride can do.
// 34 halves is only 4-byte aligned, so staging uses uint32, not uint4.
constexpr int LDV2 = CH + 2;

// cen8:[B*H*NQ,HD] int8 perm_d   cens:[B*H*NQ] f32
// kciP:[B*H,NPAD,HD] int8 perm_d (centred)   kcs:[B*H,NPAD] f32
// vcT:[B*H,HD,NPAD] bf16   vsc:[B*H,HD] f32   threshold:[B*H,NQ] f32
__global__ void __launch_bounds__(NTHREADS) sol_route_kernel(
    const int8_t* __restrict__ cen8, const float* __restrict__ cens,
    const int8_t* __restrict__ kciP, const float* __restrict__ kcs,
    const __nv_bfloat16* __restrict__ vcT, const float* __restrict__ vsc,
    const float* __restrict__ threshold,
    uint16_t* __restrict__ blk_idx, int32_t* __restrict__ blk_cnt,
    __nv_bfloat16* __restrict__ o_part, float* __restrict__ m_part,
    float* __restrict__ l_part,
    int T, int H, int NTB, int NPAD, int NQ, int max_blk,
    int sink_s, int sink_e, int sink_qs, int sink_qe, float scale_log2)
{
#if SOL_SM80
    // No smem staging for the pooled keys: each lane dots its own 128 B row,
    // and any row-major smem tile puts all 32 lanes on the same bank for every
    // operand load (32-way conflict, measured). The rows come from global
    // instead -- a 4 KB chunk reused by all 8 warps, so it lives in L1.
    __shared__ float  sKs[CH];
    __shared__ __nv_bfloat16 sVc[HD * LDV2];

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int bh = blockIdx.y;
    const int qb = blockIdx.x * WPB + warp;
    const bool live = qb < NQ;
    const size_t qs_ = (size_t)bh * NQ + (live ? qb : 0);

    // The whole centroid lives in registers: each lane dots all 128 dims
    // against its own staged pooled key. perm_d cancels between the two.
    uint32_t cq[HD / 4];
    float cqs = 0.f, thr = 0.f;
    bool q_in_sink = false;
    if (live) {
        const uint4* crow = reinterpret_cast<const uint4*>(cen8 + qs_ * HD);
        #pragma unroll
        for (int i = 0; i < HD / 16; ++i) {
            const uint4 w4 = crow[i];
            cq[i * 4 + 0] = w4.x; cq[i * 4 + 1] = w4.y;
            cq[i * 4 + 2] = w4.z; cq[i * 4 + 3] = w4.w;
        }
        cqs = cens[qs_] * scale_log2;
        thr = threshold[qs_];
        q_in_sink = (qb >= sink_qs) && (qb < sink_qe);
    }
    const int tail_len = T - (NTB - 1) * BLOCK;

    float o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f;
    float m_r = NEG, l_r = 0.f;
    int cnt = 0;

    for (int c0 = 0; c0 < NTB; c0 += CH) {
        __syncthreads();
        // NPAD is a multiple of 64, so a 32-wide chunk never reads past it and
        // the copies are unconditional; padded blocks are masked as invalid.
        if (tid < CH) sKs[tid] = kcs[(size_t)bh * NPAD + c0 + tid];
        for (int idx = tid; idx < HD * (CH / 2); idx += NTHREADS) {
            const int d = idx / (CH / 2), part = (idx % (CH / 2)) * 2;
            *reinterpret_cast<uint32_t*>(sVc + d * LDV2 + part) =
                *reinterpret_cast<const uint32_t*>(
                    vcT + ((size_t)bh * HD + d) * NPAD + c0 + part);
        }
        __syncthreads();
        if (!live) continue;

        // --- score: lane owns pooled block j ---
        const int j = c0 + lane;
        const bool valid = j < NTB;
        int32_t acc = 0;
        const uint4* krow = reinterpret_cast<const uint4*>(
            kciP + ((size_t)bh * NPAD + c0 + lane) * HD);
        #pragma unroll
        for (int i = 0; i < HD / 16; ++i) {
            const uint4 kw = krow[i];
            acc = __dp4a((int)cq[i * 4 + 0], (int)kw.x, acc);
            acc = __dp4a((int)cq[i * 4 + 1], (int)kw.y, acc);
            acc = __dp4a((int)cq[i * 4 + 2], (int)kw.z, acc);
            acc = __dp4a((int)cq[i * 4 + 3], (int)kw.w, acc);
        }
        const float s = valid ? (float)acc * cqs * sKs[lane] : NEG;

        // --- routing decision, in block order so the list stays sorted ---
        const bool sink_kv = (j >= sink_s) && (j < sink_e);
        const bool diag = (j >= qb - 1) && (j <= qb + 1);
        const bool routed = ((s > thr) || diag || sink_kv) && valid;
        const bool exact = q_in_sink ? valid : routed;
        const uint32_t m = __ballot_sync(0xffffffffu, exact);
        const int rank = __popc(m & ((1u << lane) - 1u));
        // The slot must be bounded (a sink_q block routes everything); a
        // truncated block falls through to the tail below, so its mass is
        // approximated rather than deleted.
        const bool kept = exact && (cnt + rank) < max_blk;
        if (kept)
            blk_idx[qs_ * max_blk + cnt + rank] = (uint16_t)j;
        cnt = min(cnt + __popc(m), max_blk);

        // --- tail: everything valid and not kept ---
        const bool tail = valid && !kept;
        const float st = tail ? s : NEG;
        float mn = fmaxf(m_r, st);
        #pragma unroll
        for (int off = 16; off; off >>= 1)
            mn = fmaxf(mn, __shfl_xor_sync(0xffffffffu, mn, off));
        const float alpha = exp2f(m_r - mn);
        m_r = mn;
        const float p = tail ? exp2f(st - mn) : 0.f;
        // vc is a SUM over its block, so l carries the block length.
        float ladd = p * ((j == NTB - 1) ? (float)tail_len : (float)BLOCK);
        #pragma unroll
        for (int off = 16; off; off >>= 1)
            ladd += __shfl_xor_sync(0xffffffffu, ladd, off);
        l_r = l_r * alpha + ladd;

        o0 *= alpha; o1 *= alpha; o2 *= alpha; o3 *= alpha;
        if (__ballot_sync(0xffffffffu, p > 0.f)) {
            const int d0 = lane * 4;
            // Pairs of adjacent jj share a 4-byte word, so half2 reads halve
            // the shared-memory transactions on the hot path.
            #pragma unroll 4
            for (int jj = 0; jj < CH; jj += 2) {
                const float pa = __shfl_sync(0xffffffffu, p, jj);
                const float pb = __shfl_sync(0xffffffffu, p, jj + 1);
                if (pa == 0.f && pb == 0.f) continue;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const __nv_bfloat162 vv =
                        *reinterpret_cast<const __nv_bfloat162*>(
                            sVc + (d0 + i) * LDV2 + jj);
                    float oi = (i == 0) ? o0 : (i == 1) ? o1 : (i == 2) ? o2 : o3;
                    oi = fmaf(pa, __bfloat162float(vv.x), oi);
                    oi = fmaf(pb, __bfloat162float(vv.y), oi);
                    if (i == 0) o0 = oi; else if (i == 1) o1 = oi;
                    else if (i == 2) o2 = oi; else o3 = oi;
                }
            }
        }
    }

    if (!live) return;
    if (lane == 0) {
        blk_cnt[qs_] = cnt;
        m_part[qs_] = m_r;
        l_part[qs_] = l_r * 255.0f;
    }
    // Hand over in the exact kernel's units (see its epilogue).
    __nv_bfloat16* orow = o_part + qs_ * HD;
    const float* vrow = vsc + (size_t)bh * HD;
    const int d0 = lane * 4;
    orow[d0 + 0] = __float2bfloat16(o0 * (255.0f / vrow[d0 + 0]));
    orow[d0 + 1] = __float2bfloat16(o1 * (255.0f / vrow[d0 + 1]));
    orow[d0 + 2] = __float2bfloat16(o2 * (255.0f / vrow[d0 + 2]));
    orow[d0 + 3] = __float2bfloat16(o3 * (255.0f / vrow[d0 + 3]));
#endif  // SOL_SM80
}

}  // namespace

// ---------------------------------------------------------------------------
// Per-row tail (the pre-centroid behaviour), kept selectable behind
// centroid_tail=false so the two can be A/B'd on real workloads without a
// rebuild. Emits per-ROW (o_part, m_part, l_part); o_part aliases the caller's
// output, as before. ~2.6 ms slower per call at T=37k/H=56.
// ---------------------------------------------------------------------------
namespace perrow {
using namespace sol;

constexpr int HD = HEAD_DIM, BQ = BLOCK;   // from the layout contract
// BN is this kernel's staging tile, not a contract constant. 64 is the measured
// optimum (32 costs 38%, 96 costs 8% via occupancy, 128 exceeds 48 KB smem).
constexpr int BN = 64;                     // pooled blocks staged per pass
constexpr int NWARP = BQ / 16, NTHREADS = NWARP * 32;
constexpr int KC  = HD / 32;    // int8 k-chunks for scores = Q . Kc^T
constexpr int NKT = BN / 8;     // score n8 tiles
constexpr int NT  = HD / 8;     // output n8 tiles
constexpr int PKC = BN / 16;    // bf16 k-chunks for O += P . Vc
constexpr int LDK = HD;         // 128 B, XOR-swizzled
constexpr int LDV = BN + 8;     // 72 halves = 144 B; bank = (4C + kk*8 + q) % 32

// qi:[B,T,H,D] int8 (d-axis permuted)  qs:[B,T,H] f32  -- T, not Tp: see below
// kciP:[B*H,NPAD,D] int8 (d-axis permuted)  kcs:[B*H,NPAD] f32
// vcT:[B*H,D,NPAD] bf16
__global__ void __launch_bounds__(NTHREADS) sol_route_perrow_kernel(
    const int8_t* __restrict__ qi, const float* __restrict__ qs,
    const int8_t* __restrict__ kciP, const float* __restrict__ kcs,
    const __nv_bfloat16* __restrict__ vcT,
    const float* __restrict__ vsc,
    const float* __restrict__ threshold,
    uint16_t* __restrict__ blk_idx, int32_t* __restrict__ blk_cnt,
    __nv_bfloat16* __restrict__ o_part, float* __restrict__ m_part,
    float* __restrict__ l_part,
    int T, int H, int NTB, int NPAD, int max_blk,
    int sink_s, int sink_e, int sink_qs, int sink_qe, float scale_log2)
{
#if SOL_SM80
    __shared__ int8_t sKc[BN * LDK];
    __shared__ __nv_bfloat16 sVcT[HD * LDV];
    __shared__ float sCol[NWARP][BN];
    __shared__ uint32_t sMask[BN / 32];

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, qd = lane & 3;
    const int q_block = blockIdx.x, bh = blockIdx.y;
    const int batch = bh / H, head = bh % H;
    // Indexed by T, not Tp, matching the exact kernel -- that agreement is what
    // lets the caller's `out` alias o_part.
    const int64_t bh_base = (int64_t)batch * T * H * HD + (int64_t)head * HD;
    const int64_t bh_s    = (int64_t)batch * T * H + head;

    const int q_row0 = q_block * BQ + warp * 16 + g;
    // Rows past T-1 clamp to T-1 below. Harmless per-row, but the routing
    // column sum reduces ACROSS rows and divides by the true block length, so
    // dead rows must weigh zero or a ragged tail over-counts the last row.
    const float w_row0 = (q_row0 < T) ? 1.f : 0.f;
    const float w_row1 = (q_row0 + 8 < T) ? 1.f : 0.f;
    int cnt = 0;   // warp 0 only: routed blocks emitted so far, kept in a register

    uint32_t qa[KC][4];
    float qsc[2];
    {
        const int r0 = min(q_row0, T - 1), r1 = min(q_row0 + 8, T - 1);
        const int8_t* p0 = qi + bh_base + (int64_t)r0 * H * HD;
        const int8_t* p1 = qi + bh_base + (int64_t)r1 * H * HD;
        #pragma unroll
        for (int kc = 0; kc < KC; ++kc) {
            const int c0 = kc * 32 + qd * 8;
            const uint2 a0 = *reinterpret_cast<const uint2*>(p0 + c0);
            const uint2 a1 = *reinterpret_cast<const uint2*>(p1 + c0);
            qa[kc][0] = a0.x; qa[kc][2] = a0.y;
            qa[kc][1] = a1.x; qa[kc][3] = a1.y;
        }
        qsc[0] = qs[bh_s + (int64_t)r0 * H] * scale_log2;
        qsc[1] = qs[bh_s + (int64_t)r1 * H] * scale_log2;
    }

    const float thr = threshold[(bh * gridDim.x + q_block)];
    const float q_len = (float)min(BQ, T - q_block * BQ);
    const int tail_len = T - (NTB - 1) * 64;
    const bool q_in_sink = (q_block >= sink_qs) && (q_block < sink_qe);

    float o_acc[NT][4];
    #pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        o_acc[nt][0] = 0.f; o_acc[nt][1] = 0.f; o_acc[nt][2] = 0.f; o_acc[nt][3] = 0.f;
    }
    float m_r[2] = {NEG, NEG}, l_r[2] = {0.f, 0.f};

    for (int gs = 0; gs < NTB; gs += BN) {
        __syncthreads();
        // Staging is 61% of this kernel's runtime; cp.async is worth 1.25x even
        // single-buffered (double buffering costs more occupancy than it gains).
        // The copies are unconditional because NPAD rounds NTB up to a multiple
        // of BN, so gs + p tops out at exactly NPAD - 1.
        for (int idx = tid; idx < BN * (HD / 16); idx += NTHREADS) {
            const int p = idx / (HD / 16), c16 = idx % (HD / 16);
            cp_async16_ca(sKc + p * LDK + ((c16 ^ swz_k(p)) << 4),
                          kciP + ((int64_t)bh * NPAD + gs + p) * HD + c16 * 16);
        }
        for (int idx = tid; idx < HD * (BN / 8); idx += NTHREADS) {
            const int c = idx / (BN / 8), part = idx % (BN / 8);
            cp_async16_ca(sVcT + c * LDV + part * 8,
                          vcT + ((int64_t)bh * HD + c) * NPAD + gs + part * 8);
        }
        cp_commit();
        cp_wait<0>();
        __syncthreads();

        // --- scores, INT8 ---
        int32_t s_acc[NKT][4];
        #pragma unroll
        for (int nt = 0; nt < NKT; ++nt) {
            s_acc[nt][0] = 0; s_acc[nt][1] = 0; s_acc[nt][2] = 0; s_acc[nt][3] = 0;
            const int R = nt * 8 + g;
            const int8_t* krow = sKc + R * LDK + ((qd & 1) << 3);
            const int swk = swz_k(R), qhi = qd >> 1;
            #pragma unroll
            for (int kc = 0; kc < KC; ++kc) {
                const uint2 kb = *reinterpret_cast<const uint2*>(
                    krow + (((kc * 2 + qhi) ^ swk) << 4));
                uint32_t kbf[2] = {kb.x, kb.y};
                mma_s8(s_acc[nt], qa[kc], kbf);
            }
        }
        float sc[NKT][4];
        #pragma unroll
        for (int nt = 0; nt < NKT; ++nt) {
            const int c0 = nt * 8 + qd * 2;
            const float ks0 = kcs[(int64_t)bh * NPAD + gs + c0];
            const float ks1 = kcs[(int64_t)bh * NPAD + gs + c0 + 1];
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int row = e >> 1;
                sc[nt][e] = (float)s_acc[nt][e] * qsc[row] * ((e & 1) ? ks1 : ks0);
            }
        }

        // --- column sums: reduce over m (rows). Lanes sharing q differ by 4. ---
        #pragma unroll
        for (int nt = 0; nt < NKT; ++nt) {
            // column nt*8 + 2q (and +1), rows g and g+8; dead rows contribute 0
            float a = sc[nt][0] * w_row0 + sc[nt][2] * w_row1;
            float b = sc[nt][1] * w_row0 + sc[nt][3] * w_row1;
            #pragma unroll
            for (int off = 4; off <= 16; off <<= 1) {
                a += __shfl_xor_sync(0xffffffffu, a, off);
                b += __shfl_xor_sync(0xffffffffu, b, off);
            }
            if (g == 0) {                      // one lane per q writes the pair
                sCol[warp][nt * 8 + qd * 2]     = a;
                sCol[warp][nt * 8 + qd * 2 + 1] = b;
            }
        }
        __syncthreads();

        // --- routing decision, one warp, in block order so the list stays sorted ---
        if (warp == 0) {
            for (int base = 0; base < BN; base += 32) {
                const int c = base + lane;
                const int b = gs + c;
                float colsum = 0.f;
                #pragma unroll
                for (int w = 0; w < NWARP; ++w) colsum += sCol[w][c];
                const bool valid = b < NTB;
                const bool sink_kv = (b >= sink_s) && (b < sink_e);
                const bool routed = ((colsum / q_len > thr) || (abs(q_block - b) <= 1)
                                     || sink_kv) && valid;
                const bool exact = (q_in_sink ? valid : routed);
                const uint32_t m = __ballot_sync(0xffffffffu, exact);
                // compact in order: this lane's rank among set bits below it.
                // The slot MUST be bounded -- a sink_q query block routes every
                // block, so an unbounded write runs into the next block's region.
                const int rank = __popc(m & ((1u << lane) - 1u));
                const bool kept = exact && (cnt + rank) < max_blk;
                if (kept)
                    blk_idx[((int64_t)(bh * gridDim.x + q_block)) * max_blk + cnt + rank] =
                        (uint16_t)b;   // block ids, not tokens: < 65536 for T < 4.2M
                // Ballot `kept`, NOT `exact`: sMask must name the blocks the
                // exact kernel will really walk. Gating on `exact` drops a
                // truncated block from BOTH branches, deleting its softmax mass;
                // falling back to the pooled term is what every non-routed
                // block already does.
                const uint32_t mk = __ballot_sync(0xffffffffu, kept);
                if (lane == 0) sMask[base >> 5] = mk;
                cnt = min(cnt + __popc(m), max_blk);   // uniform across the warp
            }
        }
        __syncthreads();

        // --- approximate branch over the blocks that were NOT routed ---
        float m_new[2] = {m_r[0], m_r[1]};
        float pv[NKT][4];
        #pragma unroll
        for (int nt = 0; nt < NKT; ++nt) {
            const int c0 = nt * 8 + qd * 2;
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int c = c0 + (e & 1);
                const int b = gs + c;
                const bool ex = (sMask[c >> 5] >> (c & 31)) & 1u;
                const bool ap = (b < NTB) && !ex;
                const float s = ap ? sc[nt][e] : NEG;
                pv[nt][e] = s;
                m_new[e >> 1] = fmaxf(m_new[e >> 1], s);
            }
        }
        #pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            m_new[0] = fmaxf(m_new[0], __shfl_xor_sync(0xffffffffu, m_new[0], off));
            m_new[1] = fmaxf(m_new[1], __shfl_xor_sync(0xffffffffu, m_new[1], off));
        }
        const float alpha0 = exp2f(m_r[0] - m_new[0]);
        const float alpha1 = exp2f(m_r[1] - m_new[1]);
        m_r[0] = m_new[0]; m_r[1] = m_new[1];

        // A pooled block stands for BLOCK_SIZE real tokens, so l is weighted by
        // the block's length (the final block may be short).
        float l_add[2] = {0.f, 0.f};
        #pragma unroll
        for (int nt = 0; nt < NKT; ++nt) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int row = e >> 1;
                const int b = gs + nt * 8 + qd * 2 + (e & 1);
                const float p = (pv[nt][e] <= NEG) ? 0.f : exp2f(pv[nt][e] - m_new[row]);
                pv[nt][e] = p;
                l_add[row] += p * ((b == NTB - 1) ? (float)tail_len : 64.f);
            }
        }
        #pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            l_add[0] += __shfl_xor_sync(0xffffffffu, l_add[0], off);
            l_add[1] += __shfl_xor_sync(0xffffffffu, l_add[1], off);
        }
        l_r[0] = l_r[0] * alpha0 + l_add[0];
        l_r[1] = l_r[1] * alpha1 + l_add[1];

        uint32_t pa[PKC][4];
        #pragma unroll
        for (int kk = 0; kk < PKC; ++kk) {
            pa[kk][0] = pack_bf2(pv[2 * kk][0],     pv[2 * kk][1]);
            pa[kk][1] = pack_bf2(pv[2 * kk][2],     pv[2 * kk][3]);
            pa[kk][2] = pack_bf2(pv[2 * kk + 1][0], pv[2 * kk + 1][1]);
            pa[kk][3] = pack_bf2(pv[2 * kk + 1][2], pv[2 * kk + 1][3]);
        }
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            o_acc[nt][0] *= alpha0; o_acc[nt][1] *= alpha0;
            o_acc[nt][2] *= alpha1; o_acc[nt][3] *= alpha1;
            const __nv_bfloat16* vcol = sVcT + (nt * 8 + g) * LDV;
            #pragma unroll
            for (int kk = 0; kk < PKC; ++kk) {
                uint32_t vb[2];
                vb[0] = *reinterpret_cast<const uint32_t*>(vcol + kk * 16 + qd * 2);
                vb[1] = *reinterpret_cast<const uint32_t*>(vcol + kk * 16 + qd * 2 + 8);
                mma_bf16(o_acc[nt], pa[kk], vb);
            }
        }
    }

    if (tid == 0) blk_cnt[bh * gridDim.x + q_block] = cnt;

    #pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
        const int r = q_row0 + rr * 8;
        if (r >= T) continue;
        // Hand over in the EXACT kernel's units (its epilogue applies
        // (1/l) * vsc to a 127-scaled accumulator): pre-divide by vsc and
        // pre-multiply by 127 here, once per output element.
        __nv_bfloat16* orow = o_part + bh_base + (int64_t)r * H * HD;
        const float* vsrow = vsc + (int64_t)bh * HD;
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const int c = nt * 8 + qd * 2;
            orow[c]     = __float2bfloat16(o_acc[nt][rr * 2]     * (255.0f / vsrow[c]));
            orow[c + 1] = __float2bfloat16(o_acc[nt][rr * 2 + 1] * (255.0f / vsrow[c + 1]));
        }
        if (qd == 0) {
            m_part[bh_s + (int64_t)r * H] = m_r[rr];
            l_part[bh_s + (int64_t)r * H] = l_r[rr] * 255.0f;
        }
    }
#endif  // SOL_SM80 (INT8/BF16 mma + cp.async; dispatch constraints require sm80+)
}

}  // namespace perrow

extern "C" void launch_sol_route_perrow(
    const void* qi, const void* qs, const void* kciP, const void* kcs,
    const void* vcT, const void* vsc, const void* threshold,
    void* blk_idx, void* blk_cnt, void* o_part, void* m_part, void* l_part,
    // NQ is the query-block count and NTB the key-block count; they coincide
    // only because this is self-attention, so they stay separate parameters.
    int B, int T, int H, int NTB, int NPAD, int NQ, int max_blk,
    int sink_s, int sink_e, int sink_qs, int sink_qe, float scale_log2,
    cudaStream_t stream)
{
    dim3 grid(NQ, B * H);   // one CTA per (query block, head), 4 warps
    perrow::sol_route_perrow_kernel<<<grid, perrow::NTHREADS, 0, stream>>>(
        (const int8_t*)qi, (const float*)qs, (const int8_t*)kciP, (const float*)kcs,
        (const __nv_bfloat16*)vcT, (const float*)vsc, (const float*)threshold,
        (uint16_t*)blk_idx, (int32_t*)blk_cnt, (__nv_bfloat16*)o_part,
        (float*)m_part, (float*)l_part,
        T, H, NTB, NPAD, max_blk, sink_s, sink_e, sink_qs, sink_qe, scale_log2);
}


extern "C" void launch_sol_route(
    const void* cen8, const void* cens, const void* kciP, const void* kcs,
    const void* vcT, const void* vsc, const void* threshold,
    void* blk_idx, void* blk_cnt, void* o_part, void* m_part, void* l_part,
    // NQ is the query-block count and NTB the key-block count; they coincide
    // only because this is self-attention, so they stay separate parameters.
    int B, int T, int H, int NTB, int NPAD, int NQ, int max_blk,
    int sink_s, int sink_e, int sink_qs, int sink_qe, float scale_log2,
    cudaStream_t stream)
{
    dim3 grid((NQ + WPB - 1) / WPB, B * H);
    sol_route_kernel<<<grid, NTHREADS, 0, stream>>>(
        (const int8_t*)cen8, (const float*)cens, (const int8_t*)kciP,
        (const float*)kcs, (const __nv_bfloat16*)vcT, (const float*)vsc,
        (const float*)threshold,
        (uint16_t*)blk_idx, (int32_t*)blk_cnt, (__nv_bfloat16*)o_part,
        (float*)m_part, (float*)l_part,
        T, H, NTB, NPAD, NQ, max_blk, sink_s, sink_e, sink_qs, sink_qe,
        scale_log2);
}
