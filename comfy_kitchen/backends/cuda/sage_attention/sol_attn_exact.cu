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

// Sol-Attn exact branch: each query block walks its routed key-block list,
// resuming the online softmax (o/m/l) from route's handover.
//
// All-INT8 MMA. The PV A operand wants 4 consecutive keys per lane where the
// score C layout gives 2; sol::perm_key (applied host-side to K and its
// scales) makes the repack free: score n-tiles (4kk, 4kk+1) hold lane q's
// logical keys 32kk+4q..+3. sK and the K scales are indexed by PHYSICAL slot,
// sVt by LOGICAL key. K scales/bias are read straight from global ((ks, bias)
// of an adjacent column pair is one aligned 16 B load). Inputs are padded to
// Tp so the cp.async copies run unconditionally. The V scale and the 1/255 P
// scale are constant across key blocks and fold into the epilogue.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "sol_layout.cuh"

namespace {
using namespace sol;

constexpr int HD = HEAD_DIM, BQ = BLOCK, BK = BLOCK;
constexpr int NWARP = BQ / 16, NTHREADS = NWARP * 32;
constexpr int KC = TILE_KC, NKT = TILE_NKT, NT = TILE_NT;   // tile shapes: sol_layout.cuh
constexpr int LDK = TILE_LDK, LDV = TILE_LDV;               // 128 B / 64 B rows, XOR-swizzled
constexpr int NSTAGE = 2;      // pipeline depth; occupancy beats depth here

// qi:  [B,T,H,D] int8
// qs:  [B,T,H] f32
// kiP: [B*H,Tp,D] int8 (perm_key + perm_d)   ksb: [B*H,Tp] float2 = (ks, bias)
// vTi: [B*H][block][D][64] int8 (vt_at; perm_d on keys, no perm_key)   vsc: [B*H,D] f32
// sm_120 fits 3 blocks/SM without spilling; sm_89 would spill, so Ada is unbounded.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
#define SOL_EXACT_BOUNDS __launch_bounds__(NTHREADS, 3)
#else
#define SOL_EXACT_BOUNDS __launch_bounds__(NTHREADS)
#endif

template <typename TOut>
__global__ void SOL_EXACT_BOUNDS sol_exact_kernel(
    const int8_t* __restrict__ qi, const float* __restrict__ qs,
    const int8_t* __restrict__ kiP, const float2* __restrict__ ksb,
    const int8_t* __restrict__ vTi, const float* __restrict__ vsc,
    const uint16_t* __restrict__ blk_idx, const int32_t* __restrict__ blk_cnt,
    const __nv_bfloat16* __restrict__ o_part, const float* __restrict__ m_part,
    const float* __restrict__ l_part,
    const int8_t* __restrict__ vRow,          // [B*H,Tp,D] int8, row-major (token tiles)
    const uint32_t* __restrict__ tok_idx, const int32_t* __restrict__ tok_cnt, int n_tok,
    TOut* __restrict__ out,
    int T, int Tp, int H, int NTB, float scale_log2)
{
#if SOL_SM80
    __shared__ __align__(16) int8_t sK[NSTAGE * BK * LDK];
    __shared__ __align__(16) int8_t sVt[NSTAGE * HD * LDV];
#define SK(b)   (sK   + (size_t)(b) * BK * LDK)
#define SVT(b)  (sVt  + (size_t)(b) * HD * LDV)

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, qd = lane & 3;
    const int q_block = blockIdx.x, bh = blockIdx.y;
    const int batch = bh / H, head = bh % H;
    const int64_t bh_base = (int64_t)batch * T * H * HD + (int64_t)head * HD;
    const int64_t bh_s    = (int64_t)batch * T * H + head;
    const int64_t vT_base = (int64_t)bh * HD * Tp;
    const int64_t vsc_base = (int64_t)bh * HD;
    const int64_t kp_base = (int64_t)bh * Tp;   // pre-permuted K is [B*H, Tp, D]
    const int64_t kd_base = (int64_t)bh * Tp * HD;

    const int q_row0 = q_block * BQ + warp * 16 + g;

    uint32_t qa[KC][4];
    float qsc[2];
    {
        const int r0 = min(q_row0, T - 1), r1 = min(q_row0 + 8, T - 1);
        const int8_t* p0 = qi + bh_base + (int64_t)r0 * H * HD;
        const int8_t* p1 = qi + bh_base + (int64_t)r1 * H * HD;
        #pragma unroll
        for (int kc = 0; kc < KC; ++kc) {
            const int c0 = kc * 32 + qd * 8;      // perm_d put a0 and a2 side by side
            const uint2 a0 = *reinterpret_cast<const uint2*>(p0 + c0);
            const uint2 a1 = *reinterpret_cast<const uint2*>(p1 + c0);
            qa[kc][0] = a0.x; qa[kc][2] = a0.y;
            qa[kc][1] = a1.x; qa[kc][3] = a1.y;
        }
        qsc[0] = qs[bh_s + (int64_t)r0 * H] * scale_log2;
        qsc[1] = qs[bh_s + (int64_t)r1 * H] * scale_log2;
    }

    const uint16_t* my_idx = blk_idx + (int64_t)(bh * gridDim.x + q_block) * NTB;
    const int n_blocks = blk_cnt[bh * gridDim.x + q_block];

    // resume route's state: one (o, m, l) per (b, h, query block)
    float o_acc[NT][4];
    float m_r[2], l_r[2], c_r[2];   // c_r: scale o_acc / l_r are carried in
    {
        const int64_t qb_s = (int64_t)bh * gridDim.x + q_block;
        const __nv_bfloat16* orow = o_part + qb_s * HD;
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const int c = nt * 8 + qd * 2;
            const float v0 = __bfloat162float(orow[c]);
            const float v1 = __bfloat162float(orow[c + 1]);
            o_acc[nt][0] = v0; o_acc[nt][2] = v0;
            o_acc[nt][1] = v1; o_acc[nt][3] = v1;
        }
        m_r[0] = m_r[1] = m_part[qb_s];
        c_r[0] = c_r[1] = m_part[qb_s];
        l_r[0] = l_r[1] = l_part[qb_s];
    }

    // Staging: thread tid copies the 16-byte chunks tid + NTHREADS * i (i < 4) of the K tile and of the V^T
    // tile. Source and destination both advance at a constant stride (the swizzles repeat every 16 K rows /
    // 32 V^T rows), so a tile costs two base pointers instead of an index computation per chunk.
    static_assert(BK * HD / 16 == 4 * NTHREADS && NTHREADS == 128, "staging strides assume 4 chunks per thread");
    const int sk_off = (tid >> 3) * LDK + (((tid & 7) ^ swz_k(tid >> 3)) << 4);   // + i * 16 * LDK
    const int sv_off = (tid >> 2) * LDV + (((tid & 3) ^ swz_v(tid >> 2)) << 4);   // + i * 32 * LDV
    const int8_t* const k_src0 = kiP + kd_base + tid * 16;                        // + block * BK * HD + i * 16 * HD
    const int8_t* const v_src0 = vTi + vT_base + tid * 16;                        // + block * HD * BK + i * 32 * BK
    // the four chunks written out: a pragma cannot sit in a macro portably
#define SOLX_STAGE_K(kbi, buf)   /* kbi is a block id */                   \
    {                                                                      \
        int8_t* kd_ = SK(buf) + sk_off;                                    \
        const int8_t* ks_ = k_src0 + (int64_t)(kbi) * BK * HD;             \
        cp_async16(kd_, ks_);                                              \
        cp_async16(kd_ + 16 * LDK, ks_ + 16 * HD);                         \
        cp_async16(kd_ + 32 * LDK, ks_ + 32 * HD);                         \
        cp_async16(kd_ + 48 * LDK, ks_ + 48 * HD);                         \
    }
#define SOLX_STAGE_V(kbi, buf)                                             \
    {                                                                      \
        int8_t* vd_ = SVT(buf) + sv_off;                                   \
        const int8_t* vs_ = v_src0 + (int64_t)(kbi) * HD * BK;             \
        cp_async16(vd_, vs_);                                              \
        cp_async16(vd_ + 32 * LDV, vs_ + 32 * BK);                         \
        cp_async16(vd_ + 64 * LDV, vs_ + 64 * BK);                         \
        cp_async16(vd_ + 96 * LDV, vs_ + 96 * BK);                         \
    }

    // Split-phase staging. A tile's K is read in its first half (scores) and its V in the second (P.V), so
    // the K buffer of tile kb is free at mid-tile: K(kb + 2) is requested there and V(kb + 1) at the top,
    // which gives both one and a half tiles to land. Commit order: K(0), V(0), K(1), then V(kb + 1), K(kb + 2).
    static_assert(NSTAGE == 2, "split-phase staging alternates two buffers");
    int kbi0 = n_blocks > 0 ? my_idx[0] : 0;   // block of tile kb
    int kbi1 = n_blocks > 1 ? my_idx[1] : 0;   // block of tile kb + 1
    if (n_blocks > 0) { SOLX_STAGE_K(kbi0, 0); }
    cp_commit();
    if (n_blocks > 0) { SOLX_STAGE_V(kbi0, 0); }
    cp_commit();
    if (n_blocks > 1) { SOLX_STAGE_K(kbi1, 1); }
    cp_commit();

    auto compute_tile = [&](int cur, const float2* kb_src) {
        int32_t s_acc[NKT][4];
        tile_qk(SK(cur), qa, g, qd, s_acc);
        float p_val[NKT][4];
        float bmax[2] = {m_r[0] - 20.f, m_r[1] - 20.f};   // floor: 2^-20 under the running max
        bool has[2] = {true, true};                         // routed blocks always hold live keys
        tile_scores<true>(s_acc, kb_src, qsc, qd, p_val, bmax);
        tile_softmax_pv<false>(p_val, bmax, has, SVT(cur), g, qd, m_r, l_r, c_r, o_acc);
    };

    for (int kb = 0; kb < n_blocks; ++kb) {
        const int cur = kb % NSTAGE;
        const int kbi2 = kb + 2 < n_blocks ? my_idx[kb + 2] : 0;   // read half a tile before its use
        cp_wait<2>();      // K(kb) has landed; V(kb) and K(kb + 1) may still be in flight
        __syncthreads();   // ... for every thread, and no one still reads the other V buffer
        int32_t s_acc[NKT][4];
        tile_qk(SK(cur), qa, g, qd, s_acc);
        if (kb + 1 < n_blocks) { SOLX_STAGE_V(kbi1, (kb + 1) % NSTAGE); }
        cp_commit();
        float p_val[NKT][4];
        float bmax[2] = {m_r[0] - 20.f, m_r[1] - 20.f};   // floor: 2^-20 under the running max
        bool has[2] = {true, true};                         // routed blocks always hold live keys
        tile_scores<true>(s_acc, ksb + kp_base + (int64_t)kbi0 * BK, qsc, qd, p_val, bmax);
        cp_wait<2>();      // V(kb) has landed; K(kb + 1) and V(kb + 1) may still be in flight
        __syncthreads();   // ... for every thread, and no one still reads this K buffer
        tile_softmax_pv<false>(p_val, bmax, has, SVT(cur), g, qd, m_r, l_r, c_r, o_acc);
        if (kb + 2 < n_blocks) { SOLX_STAGE_K(kbi2, cur); }
        cp_commit();
        kbi0 = kbi1;
        kbi1 = kbi2;
    }
#undef SOLX_STAGE_K
#undef SOLX_STAGE_V

    // token tiles: the listed tokens gathered into the block layout
    if (n_tok > 0) {
        cp_wait<0>();
        __syncthreads();
        // (scale, bias) go in the idle stage-1 V buffer: more static smem would cost a CTA/SM
        float2* sKsb = reinterpret_cast<float2*>(SVT(1));
        const int NG = (gridDim.x + TOK_GROUP - 1) / TOK_GROUP;   // one list per group
        const int64_t qs_tok = (int64_t)bh * NG + q_block / TOK_GROUP;
        const int n_sel = min(tok_cnt[qs_tok], n_tok);
        const uint32_t* my_tok = tok_idx + qs_tok * n_tok;
        constexpr int VEC = HD / 16;
        for (int ti = 0; ti < (n_sel + BK - 1) / BK; ++ti) {
            const int nvalid = min(BK, n_sel - ti * BK);
            for (int idx = tid; idx < BK * VEC; idx += NTHREADS) {
                const int p = idx / VEC, c16 = idx % VEC, jj = perm_key(p);
                int8_t* dst = SK(0) + p * LDK + ((c16 ^ swz_k(p)) << 4);
                if (jj < nvalid) {
                    const uint32_t t = my_tok[ti * BK + jj];
                    cp_async16(dst, kiP + kd_base + ((int64_t)(t >> 6) * BK + perm_key_inv(t & 63)) * HD + c16 * 16);
                } else {
                    *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
                }
            }
            if (tid < BK) {
                const int jj = perm_key(tid);
                float2 kb = make_float2(0.f, NEG);
                if (jj < nvalid) {
                    const uint32_t t = my_tok[ti * BK + jj];
                    kb = ksb[kp_base + (int64_t)(t >> 6) * BK + perm_key_inv(t & 63)];
                }
                sKsb[tid] = kb;
            }
            for (int idx = tid; idx < BK * VEC; idx += NTHREADS) {
                const int j = idx / VEC, c16 = idx % VEC, kp = perm_d(j);
                uint4 raw = make_uint4(0u, 0u, 0u, 0u);
                if (j < nvalid)
                    raw = *reinterpret_cast<const uint4*>(
                        vRow + ((int64_t)bh * Tp + my_tok[ti * BK + j]) * HD + c16 * 16);
                const uint8_t* b = reinterpret_cast<const uint8_t*>(&raw);
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const int c = c16 * 16 + i;
                    SVT(0)[c * LDV + (((kp >> 4) ^ swz_v(c)) << 4) + (kp & 15)] = (int8_t)b[i];
                }
            }
            cp_commit();
            cp_wait<0>();
            __syncthreads();
            compute_tile(0, sKsb);
            __syncthreads();
        }
    }

    // l is zero only when a row got no mass from either branch; zeros are right there
    const float inv0 = 1.f / fmaxf(l_r[0], 1e-30f);
    const float inv1 = 1.f / fmaxf(l_r[1], 1e-30f);
    #pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
        const int r = q_row0 + rr * 8;
        if (r >= T) continue;
        const float inv = rr ? inv1 : inv0;
        TOut* orow = out + bh_base + (int64_t)r * H * HD;
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const int c = nt * 8 + qd * 2;
            orow[c]     = from_f32<TOut>(o_acc[nt][rr * 2]     * inv * vsc[vsc_base + c]);
            orow[c + 1] = from_f32<TOut>(o_acc[nt][rr * 2 + 1] * inv * vsc[vsc_base + c + 1]);
        }
    }
#endif  // SOL_SM80
}

}  // namespace

void launch_sol_exact(
    const void* qi, const void* qs, const void* kiP, const void* ksb,
    const void* vTi, const void* vsc,
    const void* blk_idx, const void* blk_cnt,
    const void* o_part, const void* m_part, const void* l_part,
    const void* vRow, const void* tok_idx, const void* tok_cnt, int n_tok, void* out,
    int B, int T, int Tp, int H, int NQ, int NTB,
    float scale_log2, int elem, cudaStream_t stream)
{
    dim3 grid(NQ, B * H);
#define SOLX_LAUNCH(TOut)                                                              \
    sol_exact_kernel<TOut><<<grid, NTHREADS, 0, stream>>>(                             \
        (const int8_t*)qi, (const float*)qs, (const int8_t*)kiP, (const float2*)ksb,   \
        (const int8_t*)vTi, (const float*)vsc,                                         \
        (const uint16_t*)blk_idx, (const int32_t*)blk_cnt,                             \
        (const __nv_bfloat16*)o_part, (const float*)m_part, (const float*)l_part,     \
        (const int8_t*)vRow, (const uint32_t*)tok_idx, (const int32_t*)tok_cnt, n_tok, \
        (TOut*)out, T, Tp, H, NTB, scale_log2)
    if (elem == sol::SOL_FP16) SOLX_LAUNCH(__half);
    else SOLX_LAUNCH(__nv_bfloat16);
#undef SOLX_LAUNCH
}

