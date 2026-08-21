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

// Sol-Attn routing + approximate pass. One warp per 64-token query block.
//
// Both the routing decision and the tail VALUES are centroid quantities (the
// thresholded column sum is len * centroid.kc), so this is an [N x N] problem,
// not [T x N]; all rows of a query block share one tail (~5e-4 cosine).
//
// Emits per (batch, head, query block):
//   blk_idx / blk_cnt        the routed list the exact kernel walks
//   o_part, m_part, l_part   ONE softmax state per query block, in the exact
//                            kernel's units: o / vsc, and o and l both x255.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "sol_layout.cuh"

// Tensor-core centroid tail. Batch 64 query centroids per CTA so proxy QK and
// pooled PV use the same MMA layouts as the per-row fallback.
namespace centroid_tc {
using namespace sol;

constexpr int HD = HEAD_DIM, BQ = BLOCK, BN = 64;
constexpr int NWARP = BQ / 16, NTHREADS = NWARP * 32;
constexpr int KC = HD / 32, NKT = BN / 8, NT = HD / 8, PKC = BN / 16;
constexpr int LDK = HD, LDV = BN + 8;

__global__ void __launch_bounds__(NTHREADS) sol_route_tc_kernel(
    const int8_t* __restrict__ cen8, const float* __restrict__ cens,
    const int8_t* __restrict__ kciP, const float* __restrict__ kcs,
    const __nv_bfloat16* __restrict__ vcT, const float* __restrict__ vsc,
    const float* __restrict__ threshold,
    uint16_t* __restrict__ blk_idx, int32_t* __restrict__ blk_cnt,
    __nv_bfloat16* __restrict__ o_part, float* __restrict__ m_part,
    float* __restrict__ l_part,
    int T, int H, int NTB, int NPAD, int NQ, int max_blk,
    int sink_s, int sink_e, int sink_qs, int sink_qe, float scale_log2) {
#if SOL_SM80
    __shared__ int8_t sKc[BN * LDK];
    __shared__ __nv_bfloat16 sVcT[HD * LDV];

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, qd = lane & 3;
    const int bh = blockIdx.y;
    const int q0 = blockIdx.x * BQ + warp * 16 + g;
    const int qr[2] = {q0, q0 + 8};
    const bool live[2] = {qr[0] < NQ, qr[1] < NQ};

    uint32_t qa[KC][4];
    float qsc[2], thr[2];
    bool q_in_sink[2];
    {
        const int r0 = min(qr[0], NQ - 1), r1 = min(qr[1], NQ - 1);
        const int8_t* p0 = cen8 + ((size_t)bh * NQ + r0) * HD;
        const int8_t* p1 = cen8 + ((size_t)bh * NQ + r1) * HD;
        #pragma unroll
        for (int kc = 0; kc < KC; ++kc) {
            const int c0 = kc * 32 + qd * 8;
            const uint2 a0 = *reinterpret_cast<const uint2*>(p0 + c0);
            const uint2 a1 = *reinterpret_cast<const uint2*>(p1 + c0);
            qa[kc][0] = a0.x; qa[kc][2] = a0.y;
            qa[kc][1] = a1.x; qa[kc][3] = a1.y;
        }
        qsc[0] = live[0] ? cens[(size_t)bh * NQ + r0] * scale_log2 : 0.f;
        qsc[1] = live[1] ? cens[(size_t)bh * NQ + r1] * scale_log2 : 0.f;
        thr[0] = live[0] ? threshold[(size_t)bh * NQ + r0] : 0.f;
        thr[1] = live[1] ? threshold[(size_t)bh * NQ + r1] : 0.f;
        q_in_sink[0] = live[0] && qr[0] >= sink_qs && qr[0] < sink_qe;
        q_in_sink[1] = live[1] && qr[1] >= sink_qs && qr[1] < sink_qe;
    }

    const int S = max(0, min(sink_e, NTB) - sink_s);
    int cnt[2] = {live[0] ? S : 0, live[1] ? S : 0};
    #pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
        if (live[rr]) {
            uint16_t* row = blk_idx + ((size_t)bh * NQ + qr[rr]) * max_blk;
            for (int i = qd; i < S; i += 4) row[i] = (uint16_t)(sink_s + i);
        }
    }

    float o_acc[NT][4];
    #pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        o_acc[nt][0] = 0.f; o_acc[nt][1] = 0.f;
        o_acc[nt][2] = 0.f; o_acc[nt][3] = 0.f;
    }
    float m_r[2] = {NEG, NEG}, l_r[2] = {0.f, 0.f};
    const int tail_len = T - (NTB - 1) * BLOCK;

    for (int gs = 0; gs < NTB; gs += BN) {
        __syncthreads();
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

        int32_t s_acc[NKT][4];
        #pragma unroll
        for (int nt = 0; nt < NKT; ++nt) {
            s_acc[nt][0] = 0; s_acc[nt][1] = 0;
            s_acc[nt][2] = 0; s_acc[nt][3] = 0;
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

        float pv[NKT][4];
        #pragma unroll
        for (int nt = 0; nt < NKT; ++nt) {
            const int c0 = nt * 8 + qd * 2;
            const float ks0 = kcs[(int64_t)bh * NPAD + gs + c0];
            const float ks1 = kcs[(int64_t)bh * NPAD + gs + c0 + 1];
            #pragma unroll
            for (int rr = 0; rr < 2; ++rr) {
                bool cand[2], pre[2], valid[2];
                float score[2];
                #pragma unroll
                for (int cc = 0; cc < 2; ++cc) {
                    const int b = gs + c0 + cc;
                    valid[cc] = live[rr] && b < NTB;
                    pre[cc] = valid[cc] && b >= sink_s && b < sink_e;
                    score[cc] = valid[cc]
                        ? (float)s_acc[nt][rr * 2 + cc] * qsc[rr] * (cc ? ks1 : ks0)
                        : NEG;
                    const bool routed = valid[cc] &&
                        ((score[cc] > thr[rr]) || abs(qr[rr] - b) <= 1);
                    cand[cc] = !pre[cc] && (q_in_sink[rr] ? valid[cc] : routed);
                }

                int prefix = (int)cand[0] + (int)cand[1];
                int x = __shfl_up_sync(0xffffffffu, prefix, 1, 4);
                if (qd >= 1) prefix += x;
                x = __shfl_up_sync(0xffffffffu, prefix, 2, 4);
                if (qd >= 2) prefix += x;
                const int before = prefix - (int)cand[0] - (int)cand[1];
                const int total = __shfl_sync(0xffffffffu, prefix, 3, 4);
                const int slot0 = cnt[rr] + before;
                const int slot1 = slot0 + (int)cand[0];
                const bool keep0 = cand[0] && slot0 < max_blk;
                const bool keep1 = cand[1] && slot1 < max_blk;
                if (live[rr]) {
                    uint16_t* row = blk_idx + ((size_t)bh * NQ + qr[rr]) * max_blk;
                    if (keep0) row[slot0] = (uint16_t)(gs + c0);
                    if (keep1) row[slot1] = (uint16_t)(gs + c0 + 1);
                }
                cnt[rr] = min(cnt[rr] + total, max_blk);
                pv[nt][rr * 2] = valid[0] && !(pre[0] || keep0) ? score[0] : NEG;
                pv[nt][rr * 2 + 1] = valid[1] && !(pre[1] || keep1) ? score[1] : NEG;
            }
        }

        float m_new[2] = {m_r[0], m_r[1]};
        #pragma unroll
        for (int nt = 0; nt < NKT; ++nt) {
            #pragma unroll
            for (int e = 0; e < 4; ++e)
                m_new[e >> 1] = fmaxf(m_new[e >> 1], pv[nt][e]);
        }
        #pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            m_new[0] = fmaxf(m_new[0], __shfl_xor_sync(0xffffffffu, m_new[0], off));
            m_new[1] = fmaxf(m_new[1], __shfl_xor_sync(0xffffffffu, m_new[1], off));
        }
        const float alpha0 = exp2f(m_r[0] - m_new[0]);
        const float alpha1 = exp2f(m_r[1] - m_new[1]);
        m_r[0] = m_new[0]; m_r[1] = m_new[1];

        float l_add[2] = {0.f, 0.f};
        #pragma unroll
        for (int nt = 0; nt < NKT; ++nt) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int rr = e >> 1;
                const int b = gs + nt * 8 + qd * 2 + (e & 1);
                const float p = pv[nt][e] <= NEG ? 0.f : exp2f(pv[nt][e] - m_new[rr]);
                pv[nt][e] = p;
                l_add[rr] += p * ((b == NTB - 1) ? (float)tail_len : (float)BLOCK);
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
            pa[kk][0] = pack_bf2(pv[2 * kk][0], pv[2 * kk][1]);
            pa[kk][1] = pack_bf2(pv[2 * kk][2], pv[2 * kk][3]);
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

    #pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
        if (!live[rr]) continue;
        const size_t qs = (size_t)bh * NQ + qr[rr];
        if (qd == 0) {
            blk_cnt[qs] = cnt[rr];
            m_part[qs] = m_r[rr];
            l_part[qs] = l_r[rr] * 255.0f;
        }
        __nv_bfloat16* orow = o_part + qs * HD;
        const float* vsrow = vsc + (size_t)bh * HD;
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const int c = nt * 8 + qd * 2;
            orow[c] = __float2bfloat16(o_acc[nt][rr * 2] * (255.0f / vsrow[c]));
            orow[c + 1] = __float2bfloat16(o_acc[nt][rr * 2 + 1] * (255.0f / vsrow[c + 1]));
        }
    }
#endif
}

}  // namespace centroid_tc

// ---------------------------------------------------------------------------
// Per-row tail (centroid_tail=false): per-ROW state, o_part aliasing `out`.
// Kept for quality A/B without a rebuild; ~2.6 ms slower at T=37k/H=56.
// ---------------------------------------------------------------------------
namespace perrow {
using namespace sol;

constexpr int HD = HEAD_DIM, BQ = BLOCK;   // from the layout contract
// Staging tile, not a contract constant; 64 is the measured optimum.
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
    // Sink blocks are pre-emitted (never truncated by the cap -- see the
    // centroid kernel for why); non-sinks compete for the remaining budget.
    const int S = max(0, min(sink_e, NTB) - sink_s);
    if (warp == 0)
        for (int i = lane; i < S; i += 32)
            blk_idx[((int64_t)(bh * gridDim.x + q_block)) * max_blk + i] =
                (uint16_t)(sink_s + i);
    int cnt = S;   // warp 0 only: routed blocks emitted so far, kept in a register

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
                const bool pre_kept = (b >= sink_s) && (b < sink_e) && valid;
                const bool routed = ((colsum / q_len > thr) || (abs(q_block - b) <= 1)) && valid;
                const bool cand = (q_in_sink ? valid : routed) && !pre_kept;
                const uint32_t m = __ballot_sync(0xffffffffu, cand);
                // compact in order: this lane's rank among set bits below it.
                // The slot MUST be bounded -- a sink_q query block routes every
                // block, so an unbounded write runs into the next block's region.
                const int rank = __popc(m & ((1u << lane) - 1u));
                const bool kept = pre_kept || (cand && (cnt + rank) < max_blk);
                if (!pre_kept && kept)
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
    dim3 grid((NQ + centroid_tc::BQ - 1) / centroid_tc::BQ, B * H);
    centroid_tc::sol_route_tc_kernel<<<grid, centroid_tc::NTHREADS, 0, stream>>>(
        (const int8_t*)cen8, (const float*)cens, (const int8_t*)kciP,
        (const float*)kcs, (const __nv_bfloat16*)vcT, (const float*)vsc,
        (const float*)threshold,
        (uint16_t*)blk_idx, (int32_t*)blk_cnt, (__nv_bfloat16*)o_part,
        (float*)m_part, (float*)l_part,
        T, H, NTB, NPAD, NQ, max_blk, sink_s, sink_e, sink_qs, sink_qe,
        scale_log2);
}
