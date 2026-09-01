// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Sol-Attn routing and approximate tail. Both the routing decision and the tail
// are centroid quantities, so this is an [N x N] problem and all rows of a query
// block share one tail. Emits per (batch, head, query block):
//   blk_idx / blk_cnt        the routed list the exact kernel walks
//   o_part, m_part, l_part   one softmax state, in the exact kernel's units
//                            (o / vsc; o and l both x255)
//
// Transposed like int8_attn.hip: the proxy scores come out of S^T = Kc @ Cen^T, so
// a lane owns one query block (column lane % 16) and eight key blocks per tile
// (rows acc_row(lane, e)). The two half-waves of a lane pair split the 64 key
// blocks of a group, which is what the routed-list compaction below folds over.
#include <hip/hip_runtime.h>

#include <cstdint>

#include "sol_common.h"

namespace comfy::hip_backend::sol {
namespace {

constexpr int HD = kHeadDim;
constexpr int BQ = 64;   // query blocks per workgroup
constexpr int BN = 64;   // key blocks per group; also the compaction mask width
constexpr int NWAVE = BQ / 16, kThreads = NWAVE * 32;
constexpr int KT = BN / 16;  // proxy-score key tiles
constexpr int DT = HD / 16;  // contraction tiles
constexpr int NT = HD / 16;  // pooled-output tiles

// Same LDS row padding as int8_attn.hip: gfx11 reads a whole 16-byte K-step per
// lane, gfx12 eight bytes.
#if defined(COMFY_MMA_GFX11)
constexpr int kLdsPad = 16;
#else
constexpr int kLdsPad = 8;
#endif
constexpr int kStrideKc = HD + kLdsPad;  // bytes
// bf16 elements, a multiple of 8 so both the staging stores and the fragment
// reads land on 16-byte boundaries
constexpr int kStrideVc = BN + 8;

__global__ __launch_bounds__(kThreads) void sol_route_kernel(
    const int8_t* __restrict__ cen8, const float* __restrict__ cens,
    const int8_t* __restrict__ kci, const float* __restrict__ kcs,
    const __bf16* __restrict__ vcT, const float* __restrict__ vsc,
    const float* __restrict__ threshold, uint16_t* __restrict__ blk_idx,
    int32_t* __restrict__ blk_cnt, __bf16* __restrict__ o_part, float* __restrict__ m_part,
    float* __restrict__ l_part, const int32_t* __restrict__ blen, int tail, int T, int NTB,
    int NPAD, int NQ, int sink_s, int sink_e, int sink_qs, int sink_qe, float scale_log2) {
    __shared__ __attribute__((aligned(16))) int8_t sKc[BN * kStrideKc];
    __shared__ __attribute__((aligned(16))) __bf16 sVcT[HD * kStrideVc];
    __shared__ float sLen[BN];
    __shared__ float sKcs[BN];

    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int r = frag_row(lane), half = lane >> 4;
    const int bh = blockIdx.y;
    const int qb = blockIdx.x * BQ + wave * 16 + r;
    const bool live = qb < NQ;
    const int qb_clamped = imin(qb, NQ - 1);

    // Centroid fragments are loop invariant and fit: 16 VGPRs on gfx12, 32 on gfx11.
    typename MmaInt8::Frag cen_frag[DT];
    {
        const int8_t* crow = cen8 + (static_cast<size_t>(bh) * NQ + qb_clamped) * HD;
#pragma unroll
        for (int dt = 0; dt < DT; ++dt) {
            cen_frag[dt] = MmaInt8::load_aligned(crow, 0, dt * 16, 0, lane);
        }
    }
    const float qsc = live ? cens[static_cast<size_t>(bh) * NQ + qb_clamped] * scale_log2 : 0.f;
    const float thr = live ? threshold[static_cast<size_t>(bh) * NQ + qb_clamped] : 0.f;
    const bool q_in_sink = live && qb >= sink_qs && qb < sink_qe;

    uint16_t* my_row = blk_idx + (static_cast<size_t>(bh) * NQ + qb_clamped) * NTB;

    // Sink blocks are always exact, so they head the list and the budget below
    // counts the others only. The half-waves split the write. Clamped low as well
    // as high: a negative start would put that many extra entries in the list, as
    // block ids wrapped through uint16_t, and overrun the row.
    const int sink_lo = imax(0, sink_s);
    const int S = imax(0, imin(sink_e, NTB) - sink_lo);
    int cnt = live ? S : 0;
    if (live) {
        for (int i = half; i < S; i += 2) my_row[i] = static_cast<uint16_t>(sink_lo + i);
    }

    // Accumulated in place rather than through a per-tile temporary: eight live
    // v8f partials on top of these push gfx11 into scratch.
    typename MmaBf16::Acc o_acc[NT];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) o_acc[nt] = MmaBf16::zero();
    float m_r = kNeg, l_r = 0.f;

    for (int gs = 0; gs < NTB; gs += BN) {
        __syncthreads();
        // 8-byte units: kStrideKc is a multiple of 8, not of 16, on gfx12.
        for (int idx = tid; idx < BN * (HD / 8); idx += kThreads) {
            const int p = idx / (HD / 8), c8 = (idx % (HD / 8)) * 8;
            *reinterpret_cast<uint2*>(sKc + p * kStrideKc + c8) = *reinterpret_cast<const uint2*>(
                kci + (static_cast<int64_t>(bh) * NPAD + gs + p) * HD + c8);
        }
        for (int idx = tid; idx < HD * (BN / 8); idx += kThreads) {
            const int c = idx / (BN / 8), part = (idx % (BN / 8)) * 8;
            *reinterpret_cast<uint4*>(sVcT + c * kStrideVc + part) = *reinterpret_cast<const uint4*>(
                vcT + (static_cast<int64_t>(bh) * HD + c) * NPAD + gs + part);
        }
        if (tid < BN) {
            const int b = gs + tid;
            sLen[tid] = (b < NTB) ? static_cast<float>(block_len_of(blen, b, T)) : 0.f;
            sKcs[tid] = kcs[static_cast<int64_t>(bh) * NPAD + b];
        }
        __syncthreads();

        typename MmaInt8::Acc s_acc[KT];
#pragma unroll
        for (int kt = 0; kt < KT; ++kt) {
            s_acc[kt] = MmaInt8::zero();
#pragma unroll
            for (int dt = 0; dt < DT; ++dt) {
                s_acc[kt] = MmaInt8::mma(
                    MmaInt8::load_aligned(sKc, kt * 16 + r, dt * 16, kStrideKc, lane),
                    cen_frag[dt], s_acc[kt]);
            }
        }

        // Routing decision, and the pooled score of every block that stays in the tail.
        float pv[KT][8];
        uint32_t mask_lo = 0u, mask_hi = 0u;
#pragma unroll
        for (int kt = 0; kt < KT; ++kt) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int slot = kt * 16 + acc_row(lane, e);
                const int b = gs + slot;
                const bool valid = live && b < NTB;
                const bool pre = valid && b >= sink_lo && b < sink_e;
                const float score =
                    valid ? static_cast<float>(s_acc[kt][e]) * qsc * sKcs[slot] : kNeg;
                const bool routed = valid && ((score >= thr) || abs(qb - b) <= 1);
                const bool cand = !pre && (q_in_sink ? valid : routed);
                if (cand) {
                    if (slot < 32) {
                        mask_lo |= 1u << slot;
                    } else {
                        mask_hi |= 1u << (slot - 32);
                    }
                }
                // exact-routed and sink blocks leave the tail; tail == 0 hands the
                // exact kernel an empty state (softmax over the routed blocks only)
                pv[kt][e] = (tail && valid && !(pre || cand)) ? score : kNeg;
            }
        }

        // The half-wave partner holds this query block's other 32 key blocks, so the
        // pair shares one mask and then splits the writes: low half emits slots
        // 0..31, high half emits 32..63 after them. Ascending, hence deterministic.
        mask_lo |= swap_half_wave_b32(mask_lo);
        mask_hi |= swap_half_wave_b32(mask_hi);
        const int n_lo = __popc(mask_lo), n_hi = __popc(mask_hi);
        if (live) {
            uint32_t m = half ? mask_hi : mask_lo;
            int at = cnt + (half ? n_lo : 0);
            const int slot0 = gs + (half ? 32 : 0);
            while (m) {
                const int bit = __builtin_ctz(m);
                m &= m - 1;
                my_row[at++] = static_cast<uint16_t>(slot0 + bit);
            }
        }
        cnt += n_lo + n_hi;

        float m_new = m_r;
#pragma unroll
        for (int kt = 0; kt < KT; ++kt) {
#pragma unroll
            for (int e = 0; e < 8; ++e) m_new = fmaxf(m_new, pv[kt][e]);
        }
        m_new = fmaxf(m_new, swap_half_wave(m_new));
        const float alpha = exp2f(m_r - m_new);
        m_r = m_new;

        float l_add = 0.f;
#pragma unroll
        for (int kt = 0; kt < KT; ++kt) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const float p = pv[kt][e] <= kNeg ? 0.f : exp2f(pv[kt][e] - m_new);
                pv[kt][e] = p;
                l_add += p * sLen[kt * 16 + acc_row(lane, e)];
            }
        }
        l_add += swap_half_wave(l_add);
        l_r = l_r * alpha + l_add;

        if (!tail) continue;
        typename MmaBf16::Frag p_frag[KT];
#pragma unroll
        for (int kt = 0; kt < KT; ++kt) p_frag[kt] = pack_prob_frag_bf16(pv[kt], lane);
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
            for (int e = 0; e < 8; ++e) o_acc[nt][e] *= alpha;
#pragma unroll
            for (int kt = 0; kt < KT; ++kt) {
                o_acc[nt] = MmaBf16::mma(
                    load_frag_16bit<MmaBf16>(sVcT + (nt * 16 + r) * kStrideVc + kt * 16, lane),
                    p_frag[kt], o_acc[nt]);
            }
        }
    }

    if (!live) return;
    const size_t qs = static_cast<size_t>(bh) * NQ + qb;
    if (half == 0) {
        blk_cnt[qs] = cnt;
        m_part[qs] = m_r;
        l_part[qs] = l_r * 255.0f;
    }
    // Handed over in the exact kernel's units: it accumulates u8 P against int8 V
    // and multiplies by vsc in its epilogue.
    __bf16* orow = o_part + qs * HD;
    const float* vsrow = vsc + static_cast<size_t>(bh) * HD;
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        float vals[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            vals[e] = o_acc[nt][e] * (255.0f / vsrow[nt * 16 + acc_row(lane, e)]);
        }
        store_o_tile<__bf16>(orow, nt * 16, vals, lane);
    }
}

}  // namespace
}  // namespace comfy::hip_backend::sol

using namespace comfy::hip_backend::sol;

void launch_sol_route(const void* cen8, const void* cens, const void* kci, const void* kcs,
                      const void* vcT, const void* vsc, const void* threshold, void* blk_idx,
                      void* blk_cnt, void* o_part, void* m_part, void* l_part,
                      // NQ (query blocks) and NTB (key blocks) coincide today; kept
                      // separate so a query prefix needs no kernel change
                      const void* blen, int tail, int B, int T, int H, int NTB, int NPAD, int NQ,
                      int sink_s, int sink_e, int sink_qs, int sink_qe, float scale_log2,
                      hipStream_t stream) {
    dim3 grid((NQ + BQ - 1) / BQ, B * H);
    sol_route_kernel<<<grid, kThreads, 0, stream>>>(
        static_cast<const int8_t*>(cen8), static_cast<const float*>(cens),
        static_cast<const int8_t*>(kci), static_cast<const float*>(kcs),
        static_cast<const __bf16*>(vcT), static_cast<const float*>(vsc),
        static_cast<const float*>(threshold), static_cast<uint16_t*>(blk_idx),
        static_cast<int32_t*>(blk_cnt), static_cast<__bf16*>(o_part), static_cast<float*>(m_part),
        static_cast<float*>(l_part), static_cast<const int32_t*>(blen), tail, T, NTB, NPAD, NQ,
        sink_s, sink_e, sink_qs, sink_qe, scale_log2);
}
