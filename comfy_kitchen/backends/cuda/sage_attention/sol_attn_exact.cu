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

// Sol-Attn exact branch. Each query block walks its routed key-block list,
// resuming the online softmax (o/m/l) from the routing pass's handover so the
// two stages compose exactly.
//
// All-INT8 MMA (QK and PV): sm_120 is issue-rate bound, so int8 m16n8k32 packs
// the most MACs into one full-rate instruction and INT8 PV halves both the PV
// MMA count and the V smem loads. The PV A operand wants 4 consecutive keys per
// lane where the score tile's C layout hands each lane 2; sol::perm_key
// relabels keys inside each 64-block (host-side, on K and its scales) so the
// repack is free -- score n-tiles (4kk, 4kk+1) give lane q logical keys
// 32kk+4q..+3. sK and the K scales are indexed by PHYSICAL slot, sVt by
// LOGICAL key. Fragment loads are 64-bit via sol::perm_d; smem uses the XOR
// swizzles derived in sol_layout.cuh instead of padding (padding cost a block
// of occupancy).
//
// K scales/bias are not staged: (ks, bias) for an adjacent column pair is 16
// contiguous bytes, one L1-resident LDG.128 per score n-tile, keeping smem at
// 32768 B = 3 blocks/SM. cp.async double-buffers the staging; inputs are
// padded to Tp so the copies run unconditionally. Per-channel V scale and the
// 1/255 P scale are constant across key blocks, so they commute with the
// online-softmax rescale and fold into the epilogue.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "sol_layout.cuh"

namespace {
using namespace sol;

// Tied to the layout contract: the fragment maps and swizzles below are
// derived from these, so they must not drift from sol_layout.cuh.
constexpr int HD = HEAD_DIM, BQ = BLOCK, BK = BLOCK;
constexpr int NWARP = BQ / 16, NTHREADS = NWARP * 32;
constexpr int KC  = HD / 32;   // int8 k-chunks for S = Q.K^T
constexpr int NKT = BK / 8;    // score n8 tiles
constexpr int NT  = HD / 8;    // output n8 tiles
constexpr int PKC = BK / 32;   // int8 k-chunks for O += P.V
constexpr int LDK = HD;        // 128 B, XOR-swizzled (see header)
constexpr int LDV = BK;        // 64 B, XOR-swizzled (see header)
#ifndef NSTAGE
#define NSTAGE 2   // measured optimum: occupancy beats depth
#endif

// qi:  [B,T,H,D] int8 (T, not Tp -- the o_part/out alias depends on it)
// qs:  [B,T,H] f32
// kiP: [B*H,Tp,D] int8 (perm_key + perm_d)   ksb: [B*H,Tp] float2 = (ks, bias)
// vTi: [B*H,D,Tp] int8 (transposed, logical key order)   vsc: [B*H,D] f32
// Occupancy bound is arch-specific: sm_120 reaches 168 registers spill-free, so
// 3 blocks/SM is free (worth 1.07x); sm_89 cannot hit that without spilling,
// and a spill in this loop is worse than one fewer block, so Ada is unbounded.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
#define SOL_EXACT_BOUNDS __launch_bounds__(NTHREADS, 3)
#else
#define SOL_EXACT_BOUNDS __launch_bounds__(NTHREADS)
#endif

__global__ void SOL_EXACT_BOUNDS sol_exact_kernel(
    const int8_t* __restrict__ qi, const float* __restrict__ qs,
    const int8_t* __restrict__ kiP, const float2* __restrict__ ksb,
    const int8_t* __restrict__ vTi, const float* __restrict__ vsc,
    const uint16_t* __restrict__ blk_idx, const int32_t* __restrict__ blk_cnt,
    const __nv_bfloat16* __restrict__ o_part, const float* __restrict__ m_part,
    const float* __restrict__ l_part,
    __nv_bfloat16* __restrict__ out,
    int T, int Tp, int H, int max_blk, float scale_log2, int centroid_tail)
{
#if SOL_SM80
    extern __shared__ __align__(16) char smem_raw[];
    int8_t* const sK   = reinterpret_cast<int8_t*>(smem_raw);
    int8_t* const sVt  = sK + (size_t)NSTAGE * BK * LDK;
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
            const int c0 = kc * 32 + qd * 8;      // pi put a0 and a2 side by side
            const uint2 a0 = *reinterpret_cast<const uint2*>(p0 + c0);
            const uint2 a1 = *reinterpret_cast<const uint2*>(p1 + c0);
            qa[kc][0] = a0.x; qa[kc][2] = a0.y;
            qa[kc][1] = a1.x; qa[kc][3] = a1.y;
        }
        qsc[0] = qs[bh_s + (int64_t)r0 * H] * scale_log2;
        qsc[1] = qs[bh_s + (int64_t)r1 * H] * scale_log2;
    }

    const uint16_t* my_idx = blk_idx + (int64_t)(bh * gridDim.x + q_block) * max_blk;
    const int n_blocks = blk_cnt[bh * gridDim.x + q_block];

    // Resume from the routing pass. centroid_tail: one state per (b, h,
    // query block), shared by all rows. Fallback: per-row state, o_part
    // aliasing `out` (each element read and rewritten by the same thread).
    float o_acc[NT][4];
    float m_r[2], l_r[2];
    if (centroid_tail) {
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
        l_r[0] = l_r[1] = l_part[qb_s];
    } else {
        #pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
            const int r = min(q_row0 + rr * 8, T - 1);
            const __nv_bfloat16* orow = o_part + bh_base + (int64_t)r * H * HD;
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const int c = nt * 8 + qd * 2;
                o_acc[nt][rr * 2]     = __bfloat162float(orow[c]);
                o_acc[nt][rr * 2 + 1] = __bfloat162float(orow[c + 1]);
            }
            m_r[rr] = m_part[bh_s + (int64_t)r * H];
            l_r[rr] = l_part[bh_s + (int64_t)r * H];
        }
    }

#define SOLX_STAGE(kbi, buf)                                               \
    {                                                                      \
        const int64_t k0_ = (int64_t)(kbi) * BK;   /* kbi is a block id */ \
        constexpr int VEC = HD / 16;                                       \
        for (int idx = tid; idx < BK * VEC; idx += NTHREADS) {             \
            const int p = idx / VEC, c16 = idx % VEC;                      \
            cp_async16(SK(buf) + p * LDK + ((c16 ^ swz_k(p)) << 4),        \
                       kiP + kd_base + (k0_ + p) * HD + c16 * 16);         \
        }                                                                  \
        for (int idx = tid; idx < HD * 4; idx += NTHREADS) {               \
            const int c = idx >> 2, part = idx & 3;                        \
            cp_async16(SVT(buf) + c * LDV + ((part ^ swz_v(c)) << 4),      \
                       vTi + vT_base + (int64_t)c * Tp + k0_ + part * 16); \
        }                                                                  \
    }

    #pragma unroll
    for (int st = 0; st < NSTAGE - 1; ++st) {
        if (st < n_blocks) {
            SOLX_STAGE(my_idx[st], st);
        }
        cp_commit();
    }

    for (int kb = 0; kb < n_blocks; ++kb) {
        const int cur = kb % NSTAGE;
        const int64_t cur_k0 = (int64_t)my_idx[kb] * BK;
        const int ahead = kb + NSTAGE - 1;
        if (ahead < n_blocks) {
            SOLX_STAGE(my_idx[ahead], ahead % NSTAGE);
        }
        cp_commit();
        cp_wait<NSTAGE - 1>();
        __syncthreads();

        int32_t s_acc[NKT][4];
        #pragma unroll
        for (int nt = 0; nt < NKT; ++nt) {
            s_acc[nt][0] = 0; s_acc[nt][1] = 0; s_acc[nt][2] = 0; s_acc[nt][3] = 0;
            const int R = nt * 8 + g;
            const int8_t* krow = SK(cur) + R * LDK + ((qd & 1) << 3);
            const int swk = swz_k(R), qhi = qd >> 1;
            #pragma unroll
            for (int kc = 0; kc < KC; ++kc) {
                const uint2 kb = *reinterpret_cast<const uint2*>(
                    krow + (((kc * 2 + qhi) ^ swk) << 4));
                uint32_t kbf[2] = {kb.x, kb.y};
                mma_s8(s_acc[nt], qa[kc], kbf);
            }
        }

        float p_val[NKT][4];
        float m_new[2] = {m_r[0], m_r[1]};
        #pragma unroll
        for (int nt = 0; nt < NKT; ++nt) {
            const int c0 = nt * 8 + qd * 2;
            // (ks,bias) for columns c0 and c0+1 = 16 contiguous bytes; c0 is
            // always even, so the float4 is 16-byte aligned.
            const float4 kb4 = *reinterpret_cast<const float4*>(ksb + kp_base + cur_k0 + c0);
            const float k0s = kb4.x, m0 = kb4.y, k1s = kb4.z, m1 = kb4.w;
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int row = e >> 1;
                const float s = (e & 1) ? fmaf((float)s_acc[nt][e], qsc[row] * k1s, m1)
                                        : fmaf((float)s_acc[nt][e], qsc[row] * k0s, m0);
                p_val[nt][e] = s;
                m_new[row] = fmaxf(m_new[row], s);
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

        // The u8 P scale folds into the exponent: exp2(x)*255 = exp2(x +
        // log2 255). l and the accumulator carry the same 255; it cancels.
        const float m_off[2] = {m_new[0] - 7.99435344f, m_new[1] - 7.99435344f};
        #pragma unroll
        for (int nt = 0; nt < NKT; ++nt) {
            #pragma unroll
            for (int e = 0; e < 4; ++e)
                p_val[nt][e] = exp2f(p_val[nt][e] - m_off[e >> 1]);
        }

        // Free repack: n-tiles (4kk, 4kk+1) give logical keys 32kk+4q..+3.
        uint32_t pa[PKC][4];
        #pragma unroll
        for (int kk = 0; kk < PKC; ++kk) {
            const int b0 = 4 * kk, b1 = b0 + 1, b2 = b0 + 2, b3 = b0 + 3;
            pa[kk][0] = mma::pack_u8x4(
                p_val[b0][0], p_val[b0][1], p_val[b1][0], p_val[b1][1]);
            pa[kk][1] = mma::pack_u8x4(
                p_val[b0][2], p_val[b0][3], p_val[b1][2], p_val[b1][3]);
            pa[kk][2] = mma::pack_u8x4(
                p_val[b2][0], p_val[b2][1], p_val[b3][0], p_val[b3][1]);
            pa[kk][3] = mma::pack_u8x4(
                p_val[b2][2], p_val[b2][3], p_val[b3][2], p_val[b3][3]);
        }

        // l sums the PACKED bytes so num and den quantize identically.
        // pa[kk][0]/[2] hold row r, [1]/[3] row r+8.
        uint32_t li[2] = {0, 0};
        #pragma unroll
        for (int kk = 0; kk < PKC; ++kk) {
            li[0] = __dp4a(pa[kk][0], 0x01010101u, li[0]);
            li[0] = __dp4a(pa[kk][2], 0x01010101u, li[0]);
            li[1] = __dp4a(pa[kk][1], 0x01010101u, li[1]);
            li[1] = __dp4a(pa[kk][3], 0x01010101u, li[1]);
        }
        #pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            li[0] += __shfl_xor_sync(0xffffffffu, li[0], off);
            li[1] += __shfl_xor_sync(0xffffffffu, li[1], off);
        }
        l_r[0] = l_r[0] * alpha0 + (float)li[0];
        l_r[1] = l_r[1] * alpha1 + (float)li[1];

        #pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            int32_t d[4] = {0, 0, 0, 0};
            const int C = nt * 8 + g;
            const int8_t* vcol = SVT(cur) + C * LDV + ((qd & 1) << 3);
            const int swv = swz_v(C), qhi2 = qd >> 1;
            #pragma unroll
            for (int kk = 0; kk < PKC; ++kk) {
                const uint2 vb = *reinterpret_cast<const uint2*>(
                    vcol + (((kk * 2 + qhi2) ^ swv) << 4));
                uint32_t vbf[2] = {vb.x, vb.y};
                mma_u8s8(d, pa[kk], vbf);
            }
            o_acc[nt][0] = fmaf(o_acc[nt][0], alpha0, (float)d[0]);
            o_acc[nt][1] = fmaf(o_acc[nt][1], alpha0, (float)d[1]);
            o_acc[nt][2] = fmaf(o_acc[nt][2], alpha1, (float)d[2]);
            o_acc[nt][3] = fmaf(o_acc[nt][3], alpha1, (float)d[3]);
        }
#if NSTAGE < 3
        // Depth 2 refills cur next iteration, so the barrier is still required.
        __syncthreads();
#endif
    }
#undef SOLX_STAGE

    // An empty routed list leaves l_r at the resumed l_part, so the division
    // still normalises the approximate branch. l is zero only when a row got
    // no mass from either branch, and zeros are the right output there.
    const float inv0 = 1.f / fmaxf(l_r[0], 1e-30f);
    const float inv1 = 1.f / fmaxf(l_r[1], 1e-30f);
    #pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
        const int r = q_row0 + rr * 8;
        if (r >= T) continue;
        const float inv = rr ? inv1 : inv0;
        __nv_bfloat16* orow = out + bh_base + (int64_t)r * H * HD;
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const int c = nt * 8 + qd * 2;
            orow[c]     = __float2bfloat16(o_acc[nt][rr * 2]     * inv * vsc[vsc_base + c]);
            orow[c + 1] = __float2bfloat16(o_acc[nt][rr * 2 + 1] * inv * vsc[vsc_base + c + 1]);
        }
    }
#endif  // SOL_SM80 (INT8 mma + cp.async; dispatch constraints require sm80+)
}

}  // namespace

extern "C" void launch_sol_exact(
    const void* qi, const void* qs, const void* kiP, const void* ksb,
    const void* vTi, const void* vsc,
    const void* blk_idx, const void* blk_cnt,
    const void* o_part, const void* m_part, const void* l_part, void* out,
    int B, int T, int Tp, int H, int NQ, int max_blk,
    float scale_log2, int centroid_tail, cudaStream_t stream)
{
    const size_t SMEM = (size_t)NSTAGE * BK * LDK + (size_t)NSTAGE * HD * LDV;
    // Per (function, device): caching behind a process-wide flag would skip
    // the opt-in on every device after the first.
    cudaFuncSetAttribute(sol_exact_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM);
    dim3 grid(NQ, B * H);
    sol_exact_kernel<<<grid, NTHREADS, SMEM, stream>>>(
        (const int8_t*)qi, (const float*)qs, (const int8_t*)kiP, (const float2*)ksb,
        (const int8_t*)vTi, (const float*)vsc,
        (const uint16_t*)blk_idx, (const int32_t*)blk_cnt,
        (const __nv_bfloat16*)o_part, (const float*)m_part, (const float*)l_part,
        (__nv_bfloat16*)out, T, Tp, H, max_blk, scale_log2, centroid_tail);
}

