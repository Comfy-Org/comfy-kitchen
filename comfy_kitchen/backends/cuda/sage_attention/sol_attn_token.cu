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
// Sol-Attn token routing: a centroid shared by TOK_GROUP neighbouring
// query blocks scores every token in the blocks none of them routed, using the
// exact kernel's tile body. 64 centroids per CTA, the tile list split over
// gridDim.y:
//   pass 1  per-centroid score histogram (0.25 log2 bins on [ref - 8, ref + 16),
//           2.0 bins up to ref + 80; ref = route's best unrouted pooled score)
//   pass 2  the threshold admits whole bins until n_tok would overflow, so the
//           selected set does not depend on scheduling; tokens above it are
//           listed for the exact kernel, the rest become a softmax state that
//           replaces route's pooled tail
// A sort makes each list's order deterministic and a merge folds the splits'
// states into the exact kernel's handover buffers. Shared memory stays under
// 33 KB so three CTAs fit per SM.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "sol_layout.cuh"

namespace {
using namespace sol;

constexpr int HD = HEAD_DIM, BQ = BLOCK, BK = BLOCK;
constexpr int NWARP = BQ / 16, NTHREADS = NWARP * 32;
constexpr int KC = TILE_KC, NKT = TILE_NKT, NT = TILE_NT;   // tile shapes: sol_layout.cuh
constexpr int LDK = TILE_LDK, LDV = TILE_LDV;
constexpr int NSTAGE = 2;
constexpr int NB = TOK_HIST_BINS;             // histogram bins, 16-bit counts packed in pairs
constexpr float HLO = 8.f, HW = 0.25f;        // window below ref / fine bin width, log2 units
constexpr int NB_FINE = 96;                   // fine bins cover [ref - 8, ref + 16); coarse ones above
constexpr float HW_COARSE = 2.f;              // coarse bin width: 32 bins up to ref + 80
// bin index for a score `rel` log2 units above (ref - HLO), and the lower edge of a bin
__device__ __forceinline__ int hist_bin(float rel) {
    const float fine = (float)NB_FINE * HW;
    return rel < fine ? (int)(rel / HW) : min(NB - 1, NB_FINE + (int)((rel - fine) / HW_COARSE));
}
__device__ __forceinline__ float hist_edge(int b) {
    return b <= NB_FINE ? (float)b * HW : (float)NB_FINE * HW + (float)(b - NB_FINE) * HW_COARSE;
}
constexpr int P1_MAX_TILES = 1023;            // per pass-1 split: 1023 x 64 counts stay below a 16-bit bin
constexpr float MASKED = TILE_MASKED;         // scores at or below this are masked
constexpr int HIST_STRIDE = 1;                // pass 1 walks every tile; exact counts keep the cap from binding

__host__ size_t token_smem_bytes(int pass) {
    size_t n = (size_t)NSTAGE * BK * LDK;
    n += pass == 1 ? (size_t)BQ * NB * sizeof(uint16_t)
                   : (size_t)NSTAGE * HD * LDV + (size_t)BQ * sizeof(float);
    return n;
}

// per group of TOK_GROUP query blocks: centroid (quantized like prep_q's), best
// unrouted pooled score, and the blocks unrouted by every member
__global__ void sol_token_group_kernel(const float* __restrict__ qmean,        // [B*H, NPAD, HD]
                                       const float* __restrict__ tok_ref,      // [B*H, NQ]
                                       const uint32_t* __restrict__ bits,      // [B*H, NQ, W] unrouted
                                       int8_t* __restrict__ gcen8, float* __restrict__ gcens,
                                       float* __restrict__ gref, uint32_t* __restrict__ gbits,
                                       int NQ, int NPAD, int NG, int W) {
    __shared__ __align__(16) float sred[HD];
    const int grp = blockIdx.x, bh = blockIdx.y, d = threadIdx.x;
    const int qb0 = grp * TOK_GROUP, qb1 = min(NQ, qb0 + TOK_GROUP);
    float c = 0.f;
    for (int qb = qb0; qb < qb1; ++qb) c += qmean[((size_t)bh * NPAD + qb) * HD + d];
    c /= (float)(qb1 - qb0);
    const float csc = fmaxf(block_max128(fabsf(c), sred) / 127.0f, 1e-8f);
    const size_t gs = (size_t)bh * NG + grp;
    gcen8[gs * HD + perm_d(d)] = q8(c, 1.f / csc);
    if (d == 0) {
        gcens[gs] = csc;
        float r = NEG;
        for (int qb = qb0; qb < qb1; ++qb) r = fmaxf(r, tok_ref[(size_t)bh * NQ + qb]);
        gref[gs] = r;
    }
    for (int w = d; w < W; w += HD) {
        uint32_t u = 0xffffffffu;
        for (int qb = qb0; qb < qb1; ++qb) u &= bits[((size_t)bh * NQ + qb) * W + w];
        gbits[gs * W + w] = u;
    }
}

// per CTA of 64 groups: the union of the rows' unrouted blocks as a tile list
__global__ void sol_token_tiles_kernel(const uint32_t* __restrict__ bits, uint16_t* __restrict__ tiles,
                                       int32_t* __restrict__ tile_cnt, int NQ, int NTB, int W) {
    __shared__ uint32_t su[2048];   // NTB <= 65536 blocks
    __shared__ int scount;
    const int grp = blockIdx.x, bh = blockIdx.y, qb0 = grp * BQ;
    for (int w = threadIdx.x; w < W; w += blockDim.x) {
        uint32_t u = 0u;
        for (int r = 0; r < BQ && qb0 + r < NQ; ++r) u |= bits[((size_t)bh * NQ + qb0 + r) * W + w];
        su[w] = u;
    }
    if (threadIdx.x == 0) scount = 0;
    __syncthreads();
    uint16_t* out = tiles + ((size_t)bh * gridDim.x + grp) * NTB;
    if (threadIdx.x == 0) {   // in order, so the walk streams K/V sequentially
        int n = 0;
        for (int w = 0; w < W; ++w) {
            uint32_t u = su[w];
            while (u) {
                const int b = __ffs(u) - 1;
                out[n++] = (uint16_t)(w * 32 + b);
                u &= u - 1;
            }
        }
        tile_cnt[(size_t)bh * gridDim.x + grp] = n;
    }
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
#define SOL_TOKEN_BOUNDS __launch_bounds__(NTHREADS, 3)   // pass 2 sits at the exact kernel's 168 regs
#else
#define SOL_TOKEN_BOUNDS __launch_bounds__(NTHREADS)
#endif

template <int PASS>
__global__ void SOL_TOKEN_BOUNDS sol_token_kernel(
    const int8_t* __restrict__ cen8, const float* __restrict__ cens,
    const int8_t* __restrict__ kiP, const float2* __restrict__ ksb,
    const int8_t* __restrict__ vTi,
    const uint32_t* __restrict__ bits,                                    // [B*H,NQ,W] candidate bitmap
    const uint16_t* __restrict__ tiles, const int32_t* __restrict__ tile_cnt,   // per CTA candidate tile list
    const float* __restrict__ tok_ref, uint32_t* __restrict__ tok_hist,   // [B*H,NQ,NB]
    uint32_t* __restrict__ tok_idx, int32_t* __restrict__ tok_cnt,
    __nv_bfloat16* __restrict__ part_o, float* __restrict__ part_m,     // [NSPLIT,B*H,NQ,(HD)]; o in bf16 like the route handover
    float* __restrict__ part_l,
    int Tp, int NQ, int NTB, int n_tok, int tail,       // NQ: rows = groups of TOK_GROUP query blocks
    int NQB, int sink_q_start, int sink_q_end, float scale_log2)   // NQB: query blocks
{
#if SOL_SM80
    extern __shared__ __align__(16) int8_t smem[];
    const int W = (NTB + 31) >> 5;
    int8_t* sK = smem;
    int8_t* sVt = sK + NSTAGE * BK * LDK;
    uint32_t* hist = reinterpret_cast<uint32_t*>(PASS == 2 ? sVt + NSTAGE * HD * LDV : sVt);   // PASS 1: BQ x NB/2 words (two 16-bit counts); PASS 2: BQ thresholds
    float* sthr = reinterpret_cast<float*>(hist);
    // grid (groups, splits, B*H): one head's CTAs are consecutive so its K/V stays in L2
    const int bh = blockIdx.z;
    const int split = blockIdx.y, nsplit = gridDim.y;
    const uint16_t* my_tiles = tiles + ((size_t)bh * gridDim.x + blockIdx.x) * NTB;
    const int n_tiles = tile_cnt[(size_t)bh * gridDim.x + blockIdx.x];
    const int chunk = (n_tiles + nsplit - 1) / nsplit;
    const int kb_begin = split * chunk, kb_end = min(n_tiles, kb_begin + chunk);
    constexpr int STEP = PASS == 1 ? HIST_STRIDE : 1;
#define SK(b)   (sK   + (size_t)(b) * BK * LDK)
#define SVT(b)  (sVt  + (size_t)(b) * HD * LDV)

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, qd = lane & 3;
    const int qb0 = blockIdx.x * BQ;
    const int64_t vT_base = (int64_t)bh * HD * Tp;
    const int64_t kp_base = (int64_t)bh * Tp;
    const int64_t kd_base = (int64_t)bh * Tp * HD;

    const int row0 = warp * 16 + g;
    const int rows[2] = {row0, row0 + 8};
    const int qbr[2] = {qb0 + row0, qb0 + row0 + 8};
    bool active[2];   // a group is dense only when every member block is a sink_q row
    #pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
        const int m0 = qbr[rr] * TOK_GROUP, m1 = min(NQB, m0 + TOK_GROUP) - 1;
        active[rr] = qbr[rr] < NQ && !(m0 >= sink_q_start && m1 < sink_q_end);
    }

    // centroid fragments (cen8 is perm_d'd like the q rows)
    uint32_t qa[KC][4];
    float qsc[2];
    {
        const int r0 = min(qbr[0], NQ - 1), r1 = min(qbr[1], NQ - 1);
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
        qsc[0] = cens[(size_t)bh * NQ + r0] * scale_log2;
        qsc[1] = cens[(size_t)bh * NQ + r1] * scale_log2;
    }
    float ref[2], thr[2] = {3.0e38f, 3.0e38f};
    #pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
        const size_t qs = (size_t)bh * NQ + min(qbr[rr], NQ - 1);
        ref[rr] = active[rr] ? tok_ref[qs] : NEG;
    }

    if (PASS == 1) for (int i = tid; i < BQ * NB / 2; i += NTHREADS) hist[i] = 0u;
    if (PASS == 2 && tid < BQ) {
        // threshold: admit whole bins from the top until the next would overflow n_tok
        const int qb = qb0 + tid;
        float t = 3.0e38f;
        const int m0 = qb * TOK_GROUP, m1 = min(NQB, m0 + TOK_GROUP) - 1;
        if (qb < NQ && !(m0 >= sink_q_start && m1 < sink_q_end)) {
            const uint32_t* h = tok_hist + ((size_t)bh * NQ + qb) * NB;
            uint32_t acc = 0;
            int b = NB - 1;
            for (; b >= 0; --b) {
                acc += h[b] * HIST_STRIDE;
                if (acc > (uint32_t)n_tok) break;
            }
            // if every bin fits, nothing below the window is admitted: it was never counted
            t = tok_ref[(size_t)bh * NQ + qb] - HLO + hist_edge(b + 1);
        }
        sthr[tid] = t;
    }
    __syncthreads();
    if (PASS == 2) { thr[0] = sthr[rows[0]]; thr[1] = sthr[rows[1]]; }
    // candidate bitmap rows (dense rows have no candidates, so they mask out)
    const uint32_t* bmrow[2] = {bits + ((size_t)bh * NQ + min(qbr[0], NQ - 1)) * W,
                                bits + ((size_t)bh * NQ + min(qbr[1], NQ - 1)) * W};

    float o_acc[NT][4];
    #pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        o_acc[nt][0] = 0.f; o_acc[nt][1] = 0.f; o_acc[nt][2] = 0.f; o_acc[nt][3] = 0.f;
    }
    float m_r[2] = {NEG, NEG}, l_r[2] = {0.f, 0.f}, c_r[2] = {NEG, NEG};

#define SOLT_STAGE(kbi, buf)                                                   \
    {                                                                          \
        const int64_t k0_ = (int64_t)(kbi) * BK;                               \
        constexpr int VEC = HD / 16;                                           \
        for (int idx = tid; idx < BK * VEC; idx += NTHREADS) {                 \
            const int p = idx / VEC, c16 = idx % VEC;                          \
            cp_async16(SK(buf) + p * LDK + ((c16 ^ swz_k(p)) << 4),            \
                       kiP + kd_base + (k0_ + p) * HD + c16 * 16);             \
        }                                                                      \
        if (PASS == 2) {                                                       \
            for (int idx = tid; idx < HD * 4; idx += NTHREADS) {               \
                const int c = idx >> 2, part = idx & 3;                        \
                cp_async16(SVT(buf) + c * LDV + ((part ^ swz_v(c)) << 4),      \
                           vTi + vT_base + (int64_t)c * Tp + k0_ + part * 16); \
            }                                                                  \
        }                                                                      \
    }

    #pragma unroll
    for (int st = 0; st < NSTAGE - 1; ++st) {
        if (kb_begin + st * STEP < kb_end) SOLT_STAGE(my_tiles[kb_begin + st * STEP], st);
        cp_commit();
    }

    for (int kb = kb_begin, it = 0; kb < kb_end; kb += STEP, ++it) {
        const int j = my_tiles[kb];
        const int cur = it % NSTAGE;
        const int ahead = kb + STEP * (NSTAGE - 1);
        if (ahead < kb_end) SOLT_STAGE(my_tiles[ahead], (it + NSTAGE - 1) % NSTAGE);
        cp_commit();
        cp_wait<NSTAGE - 1>();
        __syncthreads();

        const bool masked[2] = {
            !active[0] || !((__ldg(bmrow[0] + (j >> 5)) >> (j & 31)) & 1u),
            !active[1] || !((__ldg(bmrow[1] + (j >> 5)) >> (j & 31)) & 1u)};
        // skip only when the whole warp's rows are routed here: the shuffles need every lane
        if (__all_sync(0xffffffffu, masked[0] && masked[1])) {
            __syncthreads();
            continue;
        }

        int32_t s_acc[NKT][4];
        tile_qk(SK(cur), qa, g, qd, s_acc);
        float p_val[NKT][4];
        float bmax[2] = {m_r[0] - 20.f, m_r[1] - 20.f};
        bool has[2] = {false, false};
        tile_scores<false>(s_acc, ksb + kp_base + (int64_t)j * BK, qsc, qd, p_val, bmax);
        uint32_t sel[2] = {0u, 0u};   // PASS 2: this lane's selected columns, bit nt*2 + (e&1)
        #pragma unroll
        for (int nt = 0; nt < NKT; ++nt) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int row = e >> 1;
                float s = p_val[nt][e];
                if (masked[row]) s = NEG;
                if (s > MASKED) {
                    if (PASS == 1) {
                        const float rel = s - ref[row] + HLO;   // below the window: not counted, not admitted
                        if (rel >= 0.f) {
                            const int bin = hist_bin(rel);
                            atomicAdd(&hist[(rows[row] * NB + bin) >> 1], (bin & 1) ? 0x10000u : 1u);
                        }
                    } else if (s >= thr[row]) {
                        sel[row] |= 1u << (nt * 2 + (e & 1));
                    }
                }
                p_val[nt][e] = s;
            }
        }
        if (PASS == 2) {
            // one slot reservation per row per tile: quad prefix, lane 0 takes the range
            #pragma unroll
            for (int row = 0; row < 2; ++row) {
                const int mine = __popc(sel[row]);
                int prefix = mine;
                int x = __shfl_up_sync(0xffffffffu, prefix, 1, 4);
                if (qd >= 1) prefix += x;
                x = __shfl_up_sync(0xffffffffu, prefix, 2, 4);
                if (qd >= 2) prefix += x;
                const int total = __shfl_sync(0xffffffffu, prefix, 3, 4);
                uint32_t base = 0u;
                if (qd == 0 && total) base = (uint32_t)atomicAdd(&tok_cnt[(size_t)bh * NQ + qbr[row]], total);
                base = __shfl_sync(0xffffffffu, base, lane & ~3);
                const uint32_t first = base + (uint32_t)(prefix - mine);
                #pragma unroll
                for (int b = 0; b < 2 * NKT; ++b) {   // static indexing keeps p_val in registers
                    if (sel[row] & (1u << b)) {
                        const uint32_t slot = first + (uint32_t)__popc(sel[row] & ((1u << b) - 1u));
                        if (slot < (uint32_t)n_tok) {
                            const int nt = b >> 1;
                            tok_idx[((size_t)bh * NQ + qbr[row]) * n_tok + slot] =
                                (uint32_t)(j * BK + perm_key(nt * 8 + qd * 2 + (b & 1)));
                            p_val[nt][row * 2 + (b & 1)] = NEG;   // exact for every row now; leaves the tail
                        }
                    }
                }
            }
        }
        if (PASS == 1 || !tail) {
            __syncthreads();
            continue;
        }
        tile_rowmax(p_val, bmax, has);   // after the masks and the listed tokens left the tail
        tile_softmax_pv<true>(p_val, bmax, has, SVT(cur), g, qd, m_r, l_r, c_r, o_acc);   // shared with the exact kernel
        __syncthreads();
    }
#undef SOLT_STAGE

    if (PASS == 1) {
        // this split's counts join the global histogram
        for (int i = tid; i < BQ * NB; i += NTHREADS) {
            const int r = i / NB, qb = qb0 + r;
            const uint32_t cnt = (hist[i >> 1] >> ((i & 1) ? 16 : 0)) & 0xffffu;
            if (qb < NQ && cnt) atomicAdd(&tok_hist[((size_t)bh * NQ + qb) * NB + (i % NB)], cnt);
        }
        return;
    }

    // this split's softmax state, in the exact kernel's units (o and l x255 / vsc)
    #pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
        if (qbr[rr] >= NQ) continue;
        const size_t qs = ((size_t)split * gridDim.z + bh) * NQ + qbr[rr];
        const float f = m_r[rr] > MASKED ? exp2f(c_r[rr] - m_r[rr]) : 1.f;
        if (qd == 0) {
            part_m[qs] = m_r[rr];
            part_l[qs] = l_r[rr] * f;
        }
        __nv_bfloat16* orow = part_o + qs * HD;
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const int c = nt * 8 + qd * 2;
            *reinterpret_cast<__nv_bfloat162*>(orow + c) =
                __floats2bfloat162_rn(o_acc[nt][rr * 2] * f, o_acc[nt][rr * 2 + 1] * f);
        }
    }
#undef SK
#undef SVT
#endif  // SOL_SM80
}

// sort each list so the exact kernel's order is deterministic: a 256-lane bitonic
// network padded with a max sentinel (n_tok need not be a power of two)
__global__ void sol_token_sort_kernel(uint32_t* __restrict__ tok_idx, const int32_t* __restrict__ tok_cnt,
                                      int NQ, int n_tok) {
    __shared__ uint32_t s[NTOK_MAX];
    const size_t qs = (size_t)blockIdx.y * NQ + blockIdx.x;
    const int n = min(tok_cnt[qs], n_tok), t = threadIdx.x;
    uint32_t* row = tok_idx + qs * n_tok;
    s[t] = t < n ? row[t] : 0xffffffffu;
    __syncthreads();
    for (int k = 2; k <= NTOK_MAX; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            const int p = t ^ j;
            if (p > t) {
                const bool up = (t & k) == 0;
                const uint32_t a = s[t], b = s[p];
                if ((a > b) == up) { s[t] = b; s[p] = a; }
            }
            __syncthreads();
        }
    }
    if (t < n) row[t] = s[t];
}

// merge the split states into the exact kernel's handover buffers
__global__ void sol_token_merge_kernel(
    const __nv_bfloat16* __restrict__ part_o, const float* __restrict__ part_m,
    const float* __restrict__ part_l,
    __nv_bfloat16* __restrict__ o_part, float* __restrict__ m_part, float* __restrict__ l_part,
    int NQ, int NG, int nsplit)
{
    const int qb = blockIdx.x, bh = blockIdx.y, c = threadIdx.x;
    const size_t qs = (size_t)bh * NQ + qb;
    const size_t qg = (size_t)bh * NG + qb / TOK_GROUP, stride = (size_t)gridDim.y * NG;
    // route's pooled tail joins as one more split
    const float rm = m_part[qs], rl = l_part[qs], ro = __bfloat162float(o_part[qs * HD + c]);
    float m = rm;
    for (int s = 0; s < nsplit; ++s) m = fmaxf(m, part_m[qg + s * stride]);
    float l = 0.f, o = 0.f;
    if (m > MASKED) {
        if (rm > MASKED) { const float w = exp2f(rm - m); l += rl * w; o += ro * w; }
        for (int s = 0; s < nsplit; ++s) {
            const size_t p = qg + s * stride;
            const float w = part_m[p] > MASKED ? exp2f(part_m[p] - m) : 0.f;
            l += part_l[p] * w;
            o += __bfloat162float(part_o[p * HD + c]) * w;
        }
    }
    __syncthreads();   // every thread has read route's m/l before thread 0 overwrites them
    o_part[qs * HD + c] = __float2bfloat16(o);
    if (c == 0) { m_part[qs] = m; l_part[qs] = l; }
}

}  // namespace

// splits of the tile list: enough CTAs that short sequences fill the GPU and a
// head's K/V stays in L2. A function of the shape only, so the plan can size the slots.
int sol_token_splits(int B, int H, int NG) {
    (void)B; (void)H;
    const int groups = (NG + BQ - 1) / BQ;   // CTAs of 64 rows over the NG centroid rows
    return max(1, min(8, (512 + groups - 1) / groups));
}

void launch_sol_token(
    const void* qmean, const void* tok_ref, const void* cand_bits,          // per query block (route / preprocess)
    void* gcen8, void* gcens, void* gref, void* gbits,                        // per group
    const void* kiP, const void* ksb, const void* vTi,
    void* tok_tiles, void* tok_tilecnt,
    void* tok_hist, void* tok_idx, void* tok_cnt, void* part_o, void* part_m, void* part_l,
    void* o_part, void* m_part, void* l_part,
    int B, int Tp, int H, int NQ, int NPAD, int NTB, int n_tok, int tail,
    int sink_q_start, int sink_q_end, float scale_log2, cudaStream_t stream)
{
    const size_t s1 = token_smem_bytes(1), s2 = token_smem_bytes(2);
    const int W = (NTB + 31) / 32;
    const int NG = (NQ + TOK_GROUP - 1) / TOK_GROUP;
    const int nsplit = sol_token_splits(B, H, NG);
    dim3 grid((NG + BQ - 1) / BQ, nsplit, B * H);
    sol_token_group_kernel<<<dim3(NG, B * H), HD, 0, stream>>>(
        (const float*)qmean, (const float*)tok_ref, (const uint32_t*)cand_bits,
        (int8_t*)gcen8, (float*)gcens, (float*)gref, (uint32_t*)gbits, NQ, NPAD, NG, W);
    // pass 1 has no partial state, so it may split further to keep the 16-bit bins from overflowing
    const int nsplit1 = max(nsplit, (NTB + P1_MAX_TILES * HIST_STRIDE - 1) / (P1_MAX_TILES * HIST_STRIDE));
    dim3 grid1(grid.x, nsplit1, B * H);
    sol_token_tiles_kernel<<<dim3(grid.x, B * H), 128, 0, stream>>>(   // (CTAs of 64 groups, B*H)
        (const uint32_t*)gbits, (uint16_t*)tok_tiles, (int32_t*)tok_tilecnt, NG, NTB, W);
    cudaMemsetAsync(tok_hist, 0, (size_t)B * H * NG * NB * sizeof(uint32_t), stream);
    cudaMemsetAsync(tok_cnt, 0, (size_t)B * H * NG * sizeof(int32_t), stream);
#define SOLT_ARGS                                                                          \
    (const int8_t*)gcen8, (const float*)gcens, (const int8_t*)kiP, (const float2*)ksb,     \
    (const int8_t*)vTi, (const uint32_t*)gbits, (const uint16_t*)tok_tiles,               \
    (const int32_t*)tok_tilecnt,                                                           \
    (const float*)gref, (uint32_t*)tok_hist, (uint32_t*)tok_idx, (int32_t*)tok_cnt,        \
    (__nv_bfloat16*)part_o, (float*)part_m, (float*)part_l,                                \
    Tp, NG, NTB, n_tok, tail, NQ, sink_q_start, sink_q_end, scale_log2
    sol_token_kernel<1><<<grid1, NTHREADS, s1, stream>>>(SOLT_ARGS);
    sol_token_kernel<2><<<grid, NTHREADS, s2, stream>>>(SOLT_ARGS);
#undef SOLT_ARGS
    sol_token_sort_kernel<<<dim3(NG, B * H), NTOK_MAX, 0, stream>>>((uint32_t*)tok_idx, (const int32_t*)tok_cnt, NG, n_tok);
    sol_token_merge_kernel<<<dim3(NQ, B * H), HD, 0, stream>>>(
        (const __nv_bfloat16*)part_o, (const float*)part_m, (const float*)part_l,
        (__nv_bfloat16*)o_part, (float*)m_part, (float*)l_part, NQ, NG, nsplit);
}
