// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Sol-Attn token routing: a centroid shared by kTokGroup neighbouring query blocks
// scores every token in the blocks none of them routed. 64 centroids per
// workgroup, the tile list split over gridDim.y:
//   pass 1  per-centroid score histogram (ref = route's best unrouted score)
//   pass 2  the threshold admits whole bins until n_tok would overflow, so the
//           selected set does not depend on scheduling; the tokens above it are
//           listed for the exact kernel and the rest replace route's pooled tail
// The sort makes each list deterministic and the merge folds the splits' states
// into the exact kernel's handover buffers.
//
// Transposed like the rest of the port: S^T = K @ Cen^T, so a lane owns one
// centroid and eight tokens per tile, and the two half-waves of a lane pair split
// each 16-token tile. No perm_key here, so a listed id indexes ki / vRow directly.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>

#include "sol_common.h"

namespace comfy::hip_backend::sol {
namespace {

constexpr int HD = kHeadDim;
constexpr int BQ = 64;  // centroid rows per workgroup
constexpr int BK = kBlock;
constexpr int NWAVE = BQ / 16, kThreads = NWAVE * 32;
constexpr int DT = HD / 16;  // contraction tiles for S = K Cen^T
constexpr int KT = BK / 16;  // score token tiles
constexpr int NT = HD / 16;  // output channel tiles

#if defined(COMFY_MMA_GFX11)
constexpr int kLdsPad = 16;
#else
constexpr int kLdsPad = 8;
#endif
constexpr int kStrideK = HD + kLdsPad;
constexpr int kStrideV = BK + kLdsPad;

constexpr int NB = kTokHistBins;         // histogram bins, 16-bit counts packed in pairs
constexpr float HLO = 8.f, HW = 0.25f;   // window below ref / fine bin width, log2 units
constexpr int NB_FINE = 96;              // fine bins cover [ref - 8, ref + 16); coarse ones above
constexpr float HW_COARSE = 2.f;         // coarse bin width: 32 bins up to ref + 80
constexpr int P1_MAX_TILES = 1023;       // tiles per pass-1 split, bounded by nsplit1
// A lane counts BK tokens per tile into one bin, and the counts are packed two to a
// word, so both of these decide whether a bin can wrap without anything failing.
static_assert(P1_MAX_TILES * BK <= 0xffff, "a pass-1 split must fit a 16-bit histogram bin");
static_assert(NB % 2 == 0, "bins are packed in pairs, so NB must be even");

// bin index for a score `rel` log2 units above (ref - HLO), and the lower edge of a bin
__forceinline__ __device__ int hist_bin(float rel) {
    const float fine = static_cast<float>(NB_FINE) * HW;
    return rel < fine ? static_cast<int>(rel / HW)
                      : imin(NB - 1, NB_FINE + static_cast<int>((rel - fine) / HW_COARSE));
}
__forceinline__ __device__ float hist_edge(int b) {
    return b <= NB_FINE ? static_cast<float>(b) * HW
                        : static_cast<float>(NB_FINE) * HW +
                              static_cast<float>(b - NB_FINE) * HW_COARSE;
}

// per group of kTokGroup query blocks: centroid (quantized like the producer's),
// best unrouted pooled score, and the blocks unrouted by every member
__global__ __launch_bounds__(HD) void sol_token_group_kernel(
    const float* __restrict__ qmean,    // [B*H, NPAD, HD]
    const float* __restrict__ tok_ref,  // [B*H, NQ]
    const uint32_t* __restrict__ bits,  // [B*H, NQ, W] unrouted
    int8_t* __restrict__ gcen8, float* __restrict__ gcens, float* __restrict__ gref,
    uint32_t* __restrict__ gbits, int NQ, int NPAD, int NG, int W) {
    __shared__ __attribute__((aligned(16))) float sred[HD];
    const int grp = blockIdx.x, bh = blockIdx.y, d = threadIdx.x;
    const int qb0 = grp * kTokGroup, qb1 = imin(NQ, qb0 + kTokGroup);
    float c = 0.f;
    for (int qb = qb0; qb < qb1; ++qb) c += qmean[(static_cast<size_t>(bh) * NPAD + qb) * HD + d];
    c /= static_cast<float>(qb1 - qb0);
    const float csc = fmaxf(block_max128(fabsf(c), sred) / 127.0f, 1e-8f);
    const size_t gs = static_cast<size_t>(bh) * NG + grp;
    gcen8[gs * HD + d] = q8(c, 1.f / csc);
    if (d == 0) {
        gcens[gs] = csc;
        float r = kNeg;
        for (int qb = qb0; qb < qb1; ++qb) r = fmaxf(r, tok_ref[static_cast<size_t>(bh) * NQ + qb]);
        gref[gs] = r;
    }
    for (int w = d; w < W; w += HD) {
        uint32_t u = 0xffffffffu;
        for (int qb = qb0; qb < qb1; ++qb) u &= bits[(static_cast<size_t>(bh) * NQ + qb) * W + w];
        gbits[gs * W + w] = u;
    }
}

// per workgroup of 64 groups: the union of the rows' unrouted blocks as a tile list
__global__ __launch_bounds__(128) void sol_token_tiles_kernel(const uint32_t* __restrict__ bits,
                                                              uint16_t* __restrict__ tiles,
                                                              int32_t* __restrict__ tile_cnt,
                                                              int NG, int NTB, int W) {
    __shared__ uint32_t su[2048];  // NTB <= 65535 blocks
    const int grp = blockIdx.x, bh = blockIdx.y, g0 = grp * BQ;
    for (int w = threadIdx.x; w < W; w += blockDim.x) {
        uint32_t u = 0u;
        for (int r = 0; r < BQ && g0 + r < NG; ++r) {
            u |= bits[(static_cast<size_t>(bh) * NG + g0 + r) * W + w];
        }
        su[w] = u;
    }
    __syncthreads();
    uint16_t* out = tiles + (static_cast<size_t>(bh) * gridDim.x + grp) * NTB;
    if (threadIdx.x == 0) {  // in order, so the walk streams K/V sequentially
        int n = 0;
        for (int w = 0; w < W; ++w) {
            uint32_t u = su[w];
            while (u) {
                const int b = __builtin_ctz(u);
                u &= u - 1;
                out[n++] = static_cast<uint16_t>(w * 32 + b);
            }
        }
        tile_cnt[static_cast<size_t>(bh) * gridDim.x + grp] = n;
    }
}

template <int PASS>
__global__ __launch_bounds__(kThreads) void sol_token_kernel(
    const int8_t* __restrict__ gcen8, const float* __restrict__ gcens,
    const int8_t* __restrict__ ki, const float2* __restrict__ ksb,
    const int8_t* __restrict__ vT, const uint32_t* __restrict__ gbits,
    const uint16_t* __restrict__ tiles, const int32_t* __restrict__ tile_cnt,
    const float* __restrict__ gref, uint32_t* __restrict__ tok_hist,
    uint32_t* __restrict__ tok_idx, int32_t* __restrict__ tok_cnt,
    __bf16* __restrict__ part_o, float* __restrict__ part_m, float* __restrict__ part_l, int Tp,
    int NG, int NTB, int n_tok, int tail, int NQB, int sink_q_start, int sink_q_end,
    float scale_log2) {
    __shared__ __attribute__((aligned(16))) int8_t sK[BK * kStrideK];
    __shared__ float2 sKsb[BK];
    // pass 1: BQ x NB counts, packed 16 bits to a half word; pass 2: the V^T tile
    // followed by the per-row thresholds
    constexpr int kVWords = (HD * kStrideV + 3) / 4;
    __shared__ __attribute__((aligned(16)))
    uint32_t sAux[PASS == 1 ? BQ * NB / 2 : kVWords + BQ];
    uint32_t* hist = sAux;
    int8_t* sVt = reinterpret_cast<int8_t*>(sAux);
    float* sthr = reinterpret_cast<float*>(sAux + kVWords);

    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int r = frag_row(lane), half = lane >> 4;
    const int bh = blockIdx.z, split = blockIdx.y, nsplit = gridDim.y;
    const int W = (NTB + 31) >> 5;
    const int g0 = blockIdx.x * BQ;
    const int hrow = wave * 16 + r;   // this lane's row within the workgroup
    const int row = g0 + hrow;        // and the centroid row (a group of query blocks) it is
    const int row_clamped = imin(row, NG - 1);
    // a group is dense only when every member block is a sink_q row
    const int m0 = row * kTokGroup, m1 = imin(NQB, m0 + kTokGroup) - 1;
    const bool active = row < NG && !(m0 >= sink_q_start && m1 < sink_q_end);

    const uint16_t* my_tiles = tiles + (static_cast<size_t>(bh) * gridDim.x + blockIdx.x) * NTB;
    const int n_tiles = tile_cnt[static_cast<size_t>(bh) * gridDim.x + blockIdx.x];
    const int chunk = (n_tiles + nsplit - 1) / nsplit;
    const int kb_begin = split * chunk, kb_end = imin(n_tiles, kb_begin + chunk);

    const int64_t vT_base = static_cast<int64_t>(bh) * HD * Tp;
    const int64_t kp_base = static_cast<int64_t>(bh) * Tp;
    const int64_t kd_base = static_cast<int64_t>(bh) * Tp * HD;

    typename MmaInt8::Frag cen_frag[DT];
    {
        const int8_t* crow = gcen8 + (static_cast<size_t>(bh) * NG + row_clamped) * HD;
#pragma unroll
        for (int dt = 0; dt < DT; ++dt) {
            cen_frag[dt] = MmaInt8::load_aligned(crow, 0, dt * 16, 0, lane);
        }
    }
    const float qsc = gcens[static_cast<size_t>(bh) * NG + row_clamped] * scale_log2;
    const float ref = active ? gref[static_cast<size_t>(bh) * NG + row_clamped] : kNeg;
    const uint32_t* bmrow = gbits + (static_cast<size_t>(bh) * NG + row_clamped) * W;

    if (PASS == 1) {
        for (int i = tid; i < BQ * NB / 2; i += kThreads) hist[i] = 0u;
    }
    if (PASS == 2 && tid < BQ) {
        // threshold: admit whole bins from the top until the next would overflow n_tok
        const int gr = g0 + tid;
        const int b0 = gr * kTokGroup, b1 = imin(NQB, b0 + kTokGroup) - 1;
        float t = 3.0e38f;
        if (gr < NG && !(b0 >= sink_q_start && b1 < sink_q_end)) {
            const uint32_t* h = tok_hist + (static_cast<size_t>(bh) * NG + gr) * NB;
            uint32_t acc = 0u;
            int b = NB - 1;
            for (; b >= 0; --b) {
                acc += h[b];
                if (acc > static_cast<uint32_t>(n_tok)) break;
            }
            // if every bin fits, nothing below the window is admitted: it was never counted
            t = gref[static_cast<size_t>(bh) * NG + gr] - HLO + hist_edge(b + 1);
        }
        sthr[tid] = t;
    }
    __syncthreads();
    const float thr = PASS == 2 ? sthr[wave * 16 + r] : 3.0e38f;

    float o_acc[NT][8];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
        for (int e = 0; e < 8; ++e) o_acc[nt][e] = 0.f;
    }
    float m_r = kNeg, l_r = 0.f, c_r = kNeg;

    for (int kb = kb_begin; kb < kb_end; ++kb) {
        const int j = my_tiles[kb];
        const int64_t k0 = static_cast<int64_t>(j) * BK;

        __syncthreads();
        for (int idx = tid; idx < BK * (HD / 8); idx += kThreads) {
            const int p = idx / (HD / 8), c8 = (idx % (HD / 8)) * 8;
            *reinterpret_cast<uint2*>(sK + p * kStrideK + c8) =
                *reinterpret_cast<const uint2*>(ki + kd_base + (k0 + p) * HD + c8);
        }
        for (int idx = tid; idx < BK; idx += kThreads) sKsb[idx] = ksb[kp_base + k0 + idx];
        if (PASS == 2) {
            for (int idx = tid; idx < HD * (BK / 8); idx += kThreads) {
                const int c = idx / (BK / 8), part = (idx % (BK / 8)) * 8;
                *reinterpret_cast<uint2*>(sVt + c * kStrideV + part) =
                    *reinterpret_cast<const uint2*>(vT + vT_base + static_cast<int64_t>(c) * Tp +
                                                    k0 + part);
            }
        }
        __syncthreads();

        // Skip only when the whole wave's rows are routed here: the exchanges below
        // need every lane. Both syncs are behind us, so a wave may skip alone.
        const bool masked = !active || !((bmrow[j >> 5] >> (j & 31)) & 1u);
        if (__all(masked)) continue;

        typename MmaInt8::Acc s_acc[KT];
#pragma unroll
        for (int kt = 0; kt < KT; ++kt) {
            s_acc[kt] = MmaInt8::zero();
#pragma unroll
            for (int dt = 0; dt < DT; ++dt) {
                s_acc[kt] = MmaInt8::mma(
                    MmaInt8::load_aligned(sK, kt * 16 + r, dt * 16, kStrideK, lane), cen_frag[dt],
                    s_acc[kt]);
            }
        }

        float p_val[KT][8];
        uint32_t sel = 0u;  // PASS 2: bit kt * 8 + e is this lane's selected token
#pragma unroll
        for (int kt = 0; kt < KT; ++kt) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const float2 kb2 = sKsb[kt * 16 + acc_row(lane, e)];
                // A dead key carries a zero scale and a kNeg bias, so its score is kNeg.
                float s =
                    masked ? kNeg : fmaf(static_cast<float>(s_acc[kt][e]), qsc * kb2.x, kb2.y);
                if (s > kTileMasked) {
                    if (PASS == 1) {
                        // below the window: not counted, so never admitted
                        const float rel = s - ref + HLO;
                        if (rel >= 0.f) {
                            const int bin = hist_bin(rel);
                            const int idx = hrow * NB + bin;
                            atomicAdd(&hist[idx >> 1], (bin & 1) ? 0x10000u : 1u);
                        }
                    } else if (s >= thr) {
                        sel |= 1u << (kt * 8 + e);
                    }
                }
                p_val[kt][e] = s;
            }
        }

        if (PASS == 2 && sel) {
            // The threshold fixes the set, so only the order varies and the sort
            // below settles that. Static indexing: a computed p_val index spills.
            const uint32_t base = static_cast<uint32_t>(
                atomicAdd(&tok_cnt[static_cast<size_t>(bh) * NG + row], __popc(sel)));
#pragma unroll
            for (int kt = 0; kt < KT; ++kt) {
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const uint32_t bit = 1u << (kt * 8 + e);
                    if (sel & bit) {
                        const uint32_t slot = base + __popc(sel & (bit - 1u));
                        // Only a token that made the list leaves the tail. The
                        // threshold keeps the count inside n_tok except through the
                        // clamped top bin, and a surplus there must stay pooled
                        // rather than fall out of both branches.
                        if (slot < static_cast<uint32_t>(n_tok)) {
                            tok_idx[(static_cast<size_t>(bh) * NG + row) * n_tok + slot] =
                                static_cast<uint32_t>(j * BK + kt * 16 + acc_row(lane, e));
                            p_val[kt][e] = kNeg;
                        }
                    }
                }
            }
        }
        if (PASS == 1 || !tail) continue;

        // Online softmax over what is left, in the exact kernel's units. A lane
        // with no live score keeps its state (alpha 1, P 0).
        float bmax = m_r - 20.f;
        bool has = false;
#pragma unroll
        for (int kt = 0; kt < KT; ++kt) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                if (p_val[kt][e] > kTileMasked) {
                    has = true;
                    bmax = fmaxf(bmax, p_val[kt][e]);
                }
            }
        }
        bmax = fmaxf(bmax, swap_half_wave(bmax));
        has = has || swap_half_wave_b32(static_cast<uint32_t>(has)) != 0u;
        if (!has) bmax = c_r;
        const float alpha = exp2f(c_r - bmax);
        c_r = bmax;
        m_r = fmaxf(m_r, bmax);

        const float m_off = has ? bmax - kProbU8Offset : 3.0e38f;
        typename MmaInt8::Frag p_frag[KT];
        uint32_t li = 0u;
#pragma unroll
        for (int kt = 0; kt < KT; ++kt) {
            uint32_t packed[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                packed[e] = prob_to_u8(exp2f(p_val[kt][e] - m_off));
                li += packed[e];
            }
            p_frag[kt] = pack_prob_frag(packed, lane);
        }
        li += swap_half_wave_b32(li);
        l_r = fmaf(l_r, alpha, static_cast<float>(li));

#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            typename MmaInt8::Acc partial = MmaInt8::zero();
#pragma unroll
            for (int kt = 0; kt < KT; ++kt) {
                partial = MmaInt8::mma_ub(
                    MmaInt8::load_aligned(sVt, nt * 16 + r, kt * 16, kStrideV, lane), p_frag[kt],
                    partial);
            }
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                o_acc[nt][e] = fmaf(o_acc[nt][e], alpha, static_cast<float>(partial[e]));
            }
        }
    }

    if (PASS == 1) {
        // this split's counts join the global histogram
        __syncthreads();
        for (int i = tid; i < BQ * NB; i += kThreads) {
            const int gr = g0 + i / NB;
            const uint32_t cnt = (hist[i >> 1] >> ((i & 1) ? 16 : 0)) & 0xffffu;
            if (gr < NG && cnt) {
                atomicAdd(&tok_hist[(static_cast<size_t>(bh) * NG + gr) * NB + (i % NB)], cnt);
            }
        }
        return;
    }

    if (row >= NG) return;
    // rescaled to m_r, which is what the merge exponentiates against
    const size_t qs = (static_cast<size_t>(split) * gridDim.z + bh) * NG + row;
    const float f = m_r > kTileMasked ? exp2f(c_r - m_r) : 1.f;
    if (half == 0) {
        part_m[qs] = m_r;
        part_l[qs] = l_r * f;
    }
    __bf16* orow = part_o + qs * HD;
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        float vals[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) vals[e] = o_acc[nt][e] * f;
        store_o_tile<__bf16>(orow, nt * 16, vals, lane);
    }
}

// sort each list so the exact kernel's order is deterministic: a bitonic network
// padded with a max sentinel (n_tok need not be a power of two)
__global__ __launch_bounds__(kNTokMax) void sol_token_sort_kernel(
    uint32_t* __restrict__ tok_idx, const int32_t* __restrict__ tok_cnt, int NG, int n_tok) {
    // The sentinel below pads a ragged n_tok, but the network itself sorts only a
    // power-of-two length; a bad kNTokMax would leave the list partly ordered.
    static_assert((kNTokMax & (kNTokMax - 1)) == 0, "the bitonic network needs a power of two");
    __shared__ uint32_t s[kNTokMax];
    const size_t qs = static_cast<size_t>(blockIdx.y) * NG + blockIdx.x;
    const int n = imin(tok_cnt[qs], n_tok), t = threadIdx.x;
    uint32_t* row = tok_idx + qs * n_tok;
    s[t] = t < n ? row[t] : 0xffffffffu;
    __syncthreads();
    for (int k = 2; k <= kNTokMax; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            const int p = t ^ j;
            if (p > t) {
                const bool up = (t & k) == 0;
                const uint32_t a = s[t], b = s[p];
                if ((a > b) == up) {
                    s[t] = b;
                    s[p] = a;
                }
            }
            __syncthreads();
        }
    }
    if (t < n) row[t] = s[t];
}

// merge the split states into the exact kernel's handover buffers
__global__ __launch_bounds__(HD) void sol_token_merge_kernel(
    const __bf16* __restrict__ part_o, const float* __restrict__ part_m,
    const float* __restrict__ part_l, __bf16* __restrict__ o_part, float* __restrict__ m_part,
    float* __restrict__ l_part, int NQ, int NG, int nsplit) {
    const int qb = blockIdx.x, bh = blockIdx.y, c = threadIdx.x;
    const size_t qs = static_cast<size_t>(bh) * NQ + qb;
    const size_t qg = static_cast<size_t>(bh) * NG + qb / kTokGroup;
    const size_t stride = static_cast<size_t>(gridDim.y) * NG;
    // route's pooled tail joins as one more split
    const float rm = m_part[qs], rl = l_part[qs];
    const float ro = static_cast<float>(o_part[qs * HD + c]);
    float m = rm;
    for (int s = 0; s < nsplit; ++s) m = fmaxf(m, part_m[qg + s * stride]);
    float l = 0.f, o = 0.f;
    if (m > kTileMasked) {
        if (rm > kTileMasked) {
            const float w = exp2f(rm - m);
            l += rl * w;
            o += ro * w;
        }
        for (int s = 0; s < nsplit; ++s) {
            const size_t p = qg + s * stride;
            const float w = part_m[p] > kTileMasked ? exp2f(part_m[p] - m) : 0.f;
            l += part_l[p] * w;
            o += static_cast<float>(part_o[p * HD + c]) * w;
        }
    }
    __syncthreads();  // every thread has read route's m/l before thread 0 overwrites them
    o_part[qs * HD + c] = static_cast<__bf16>(o);
    if (c == 0) {
        m_part[qs] = m;
        l_part[qs] = l;
    }
}

}  // namespace
}  // namespace comfy::hip_backend::sol

using namespace comfy::hip_backend::sol;

// Splits of the tile list: enough workgroups that short sequences fill the GPU and
// a head's K/V stays in cache. A function of the shape only, so the plan can size
// the slots. Mirrors the CUDA sol_token_splits.
int sol_token_splits(int B, int H, int NG) {
    (void)B;
    (void)H;
    const int groups = (NG + BQ - 1) / BQ;  // workgroups of 64 rows over the NG centroid rows
    return std::max(1, std::min(8, (512 + groups - 1) / groups));
}

void launch_sol_token(const void* qmean, const void* tok_ref, const void* cand_bits, void* gcen8,
                      void* gcens, void* gref, void* gbits, const void* ki, const void* ksb,
                      const void* vT, void* tok_tiles, void* tok_tilecnt, void* tok_hist,
                      void* tok_idx, void* tok_cnt, void* part_o, void* part_m, void* part_l,
                      void* o_part, void* m_part, void* l_part, int B, int Tp, int H, int NQ,
                      int NPAD, int NTB, int n_tok, int tail, int sink_q_start, int sink_q_end,
                      float scale_log2, hipStream_t stream) {
    const int bh = B * H;
    const int W = (NTB + 31) / 32;
    const int NG = (NQ + kTokGroup - 1) / kTokGroup;
    const int nsplit = sol_token_splits(B, H, NG);
    const int nctas = (NG + BQ - 1) / BQ;
    dim3 grid(nctas, nsplit, bh);
    sol_token_group_kernel<<<dim3(NG, bh), HD, 0, stream>>>(
        static_cast<const float*>(qmean), static_cast<const float*>(tok_ref),
        static_cast<const uint32_t*>(cand_bits), static_cast<int8_t*>(gcen8),
        static_cast<float*>(gcens), static_cast<float*>(gref), static_cast<uint32_t*>(gbits), NQ,
        NPAD, NG, W);
    sol_token_tiles_kernel<<<dim3(nctas, bh), 128, 0, stream>>>(
        static_cast<const uint32_t*>(gbits), static_cast<uint16_t*>(tok_tiles),
        static_cast<int32_t*>(tok_tilecnt), NG, NTB, W);
    sol_check(hipMemsetAsync(tok_hist, 0,
                             static_cast<size_t>(bh) * NG * kTokHistBins * sizeof(uint32_t),
                             stream),
              "clearing the token histogram");
    sol_check(hipMemsetAsync(tok_cnt, 0, static_cast<size_t>(bh) * NG * sizeof(int32_t), stream),
              "clearing the token counts");
    // pass 1 has no partial state, so it may split further to keep the 16-bit bins
    // from overflowing
    const int nsplit1 = std::max(nsplit, (NTB + P1_MAX_TILES - 1) / P1_MAX_TILES);
    dim3 grid1(nctas, nsplit1, bh);
#define SOLT_ARGS                                                                                 \
    static_cast<const int8_t*>(gcen8), static_cast<const float*>(gcens),                          \
        static_cast<const int8_t*>(ki), static_cast<const float2*>(ksb),                          \
        static_cast<const int8_t*>(vT), static_cast<const uint32_t*>(gbits),                      \
        static_cast<const uint16_t*>(tok_tiles), static_cast<const int32_t*>(tok_tilecnt),        \
        static_cast<const float*>(gref), static_cast<uint32_t*>(tok_hist),                        \
        static_cast<uint32_t*>(tok_idx), static_cast<int32_t*>(tok_cnt),                          \
        static_cast<__bf16*>(part_o), static_cast<float*>(part_m), static_cast<float*>(part_l),   \
        Tp, NG, NTB, n_tok, tail, NQ, sink_q_start, sink_q_end, scale_log2
    sol_token_kernel<1><<<grid1, kThreads, 0, stream>>>(SOLT_ARGS);
    sol_token_kernel<2><<<grid, kThreads, 0, stream>>>(SOLT_ARGS);
#undef SOLT_ARGS
    sol_token_sort_kernel<<<dim3(NG, bh), kNTokMax, 0, stream>>>(
        static_cast<uint32_t*>(tok_idx), static_cast<const int32_t*>(tok_cnt), NG, n_tok);
    sol_token_merge_kernel<<<dim3(NQ, bh), HD, 0, stream>>>(
        static_cast<const __bf16*>(part_o), static_cast<const float*>(part_m),
        static_cast<const float*>(part_l), static_cast<__bf16*>(o_part),
        static_cast<float*>(m_part), static_cast<float*>(l_part), NQ, NG, nsplit);
}
