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

// Sol-Attn routing + approximate tail. Both the routing decision and the tail
// are centroid quantities, so this is an [N x N] problem and all rows of a
// query block share one tail. Emits per (batch, head, query block):
//   blk_idx / blk_cnt        the routed list the exact kernel walks
//   o_part, m_part, l_part   one softmax state, in the exact kernel's units
//                            (o / vsc; o and l both x255)

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "sol_layout.cuh"

// 64 query centroids per CTA; proxy QK on int8 MMA, pooled PV on bf16 MMA.
namespace {
using namespace sol;

constexpr int HD = HEAD_DIM, BQ = BLOCK, BN = 64;
constexpr int NWARP = BQ / 16, NTHREADS = NWARP * 32;
constexpr int KC = HD / 32, NKT = BN / 8, NT = HD / 8, PKC = BN / 16;
constexpr int LDK = HD, LDV = BN + 8;

__global__ void __launch_bounds__(NTHREADS) sol_route_kernel(
    const int8_t* __restrict__ cen8, const float* __restrict__ cens,
    const int8_t* __restrict__ kciP, const float* __restrict__ kcs,
    const __nv_bfloat16* __restrict__ vcT, const float* __restrict__ vsc,
    const float* __restrict__ threshold,
    uint16_t* __restrict__ blk_idx, int32_t* __restrict__ blk_cnt,
    __nv_bfloat16* __restrict__ o_part, float* __restrict__ m_part,
    float* __restrict__ l_part,
    float* __restrict__ tok_ref,       // token routing: best unrouted pooled score per query block, or null
    uint32_t* __restrict__ cand_bits,  // token routing: [B*H, NQ, ceil(NTB/32)] unrouted-block bitmap, or null
    const int32_t* __restrict__ blen, int tail,
    int T, int NTB, int NPAD, int NQ,
    int sink_s, int sink_e, int sink_qs, int sink_qe, float scale_log2) {
#if SOL_SM80
    __shared__ __align__(16) int8_t sKc[BN * LDK];
    __shared__ __align__(16) __nv_bfloat16 sVcT[HD * LDV];
    __shared__ float sLen[BN];

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
            uint16_t* row = blk_idx + ((size_t)bh * NQ + qr[rr]) * NTB;
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
    float refm[2] = {NEG, NEG};
    const int W = (NTB + 31) >> 5;

    for (int gs = 0; gs < NTB; gs += BN) {
        uint32_t cbits[2][2] = {{0u, 0u}, {0u, 0u}};   // this thread's candidate flags in the 64-block group
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
        if (tid < BN) sLen[tid] = (gs + tid < NTB) ? (float)block_len_of(blen, gs + tid, T) : 0.f;
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
                        ((score[cc] >= thr[rr]) || abs(qr[rr] - b) <= 1);
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
                if (live[rr]) {
                    uint16_t* row = blk_idx + ((size_t)bh * NQ + qr[rr]) * NTB;
                    if (cand[0]) row[slot0] = (uint16_t)(gs + c0);
                    if (cand[1]) row[slot1] = (uint16_t)(gs + c0 + 1);
                }
                cnt[rr] += total;
                // token routing: a block unrouted by this row AND by its TOK_GROUP partner
                // (row q ^ 1, four lanes over) goes to the token passes instead of the
                // pooled tail; a block the partner routed stays pooled here
                bool ctok[2] = {false, false};
                if (cand_bits) {
                    const uint32_t unr = (valid[0] && !pre[0] && !cand[0] ? 1u : 0u)
                                       | (valid[1] && !pre[1] && !cand[1] ? 2u : 0u);
                    const uint32_t peer = __shfl_xor_sync(0xffffffffu, unr | (live[rr] ? 4u : 0u), 4);
                    const uint32_t both = unr & ((peer & 4u) ? peer : 3u);   // no live partner: own bits
                    #pragma unroll
                    for (int cc = 0; cc < 2; ++cc) {
                        ctok[cc] = (both >> cc) & 1u;
                        const int bit = c0 + cc;   // 0..63 within the group
                        if (ctok[cc]) { cbits[rr][bit >> 5] |= 1u << (bit & 31); refm[rr] = fmaxf(refm[rr], score[cc]); }
                    }
                }
                // exact-routed and sink blocks leave the tail (this form costs 3 fewer regs);
                // tail == 0 hands the exact kernel an empty state (softmax over routed only)
                pv[nt][rr * 2] = tail && valid[0] && !(pre[0] || cand[0] || ctok[0]) ? score[0] : NEG;
                pv[nt][rr * 2 + 1] = tail && valid[1] && !(pre[1] || cand[1] || ctok[1]) ? score[1] : NEG;
            }
        }

        if (cand_bits) {
            // the four lanes of a row hold disjoint bits: OR them and let one lane store
            #pragma unroll
            for (int rr = 0; rr < 2; ++rr) {
                #pragma unroll
                for (int wd = 0; wd < 2; ++wd) {
                    uint32_t b = cbits[rr][wd];
                    b |= __shfl_xor_sync(0xffffffffu, b, 1);
                    b |= __shfl_xor_sync(0xffffffffu, b, 2);
                    const int word = (gs >> 5) + wd;
                    if (qd == 0 && live[rr] && word < W)
                        cand_bits[((size_t)bh * NQ + qr[rr]) * W + word] = b;
                }
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
                l_add[rr] += p * sLen[b - gs];
            }
        }
        #pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            l_add[0] += __shfl_xor_sync(0xffffffffu, l_add[0], off);
            l_add[1] += __shfl_xor_sync(0xffffffffu, l_add[1], off);
        }
        l_r[0] = l_r[0] * alpha0 + l_add[0];
        l_r[1] = l_r[1] * alpha1 + l_add[1];

        if (!tail) continue;
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
    for (int off = 1; off <= 2; off <<= 1) {
        refm[0] = fmaxf(refm[0], __shfl_xor_sync(0xffffffffu, refm[0], off));
        refm[1] = fmaxf(refm[1], __shfl_xor_sync(0xffffffffu, refm[1], off));
    }
    #pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
        if (!live[rr]) continue;
        const size_t qs = (size_t)bh * NQ + qr[rr];
        if (qd == 0) {
            blk_cnt[qs] = cnt[rr];
            if (tok_ref) tok_ref[qs] = refm[rr];
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

}  // namespace

void launch_sol_route(
    const void* cen8, const void* cens, const void* kciP, const void* kcs,
    const void* vcT, const void* vsc, const void* threshold,
    void* blk_idx, void* blk_cnt, void* o_part, void* m_part, void* l_part, void* tok_ref, void* cand_bits,
    // NQ (query blocks) and NTB (key blocks) coincide today; kept separate so
    // a query prefix (LTX-2 guide attention) needs no kernel change
    const void* blen, int tail,
    int B, int T, int H, int NTB, int NPAD, int NQ,
    int sink_s, int sink_e, int sink_qs, int sink_qe, float scale_log2,
    cudaStream_t stream)
{
    dim3 grid((NQ + BQ - 1) / BQ, B * H);
    sol_route_kernel<<<grid, NTHREADS, 0, stream>>>(
        (const int8_t*)cen8, (const float*)cens, (const int8_t*)kciP,
        (const float*)kcs, (const __nv_bfloat16*)vcT, (const float*)vsc,
        (const float*)threshold,
        (uint16_t*)blk_idx, (int32_t*)blk_cnt, (__nv_bfloat16*)o_part,
        (float*)m_part, (float*)l_part, (float*)tok_ref, (uint32_t*)cand_bits, (const int32_t*)blen, tail,
        T, NTB, NPAD, NQ, sink_s, sink_e, sink_qs, sink_qe,
        scale_log2);
}
