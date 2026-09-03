// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Sol-Attn exact branch: each query block walks its routed key-block list,
// resuming the online softmax (o / m / l) from route's handover.
//
// All-INT8, and transposed like int8_attn.hip: S^T = K @ Q^T puts one query on a
// lane with eight keys in its accumulator, so the softmax is scalar per lane and
// P^T is already the u8 A operand of O^T = V^T @ P^T. That is what the CUDA source
// buys with sol::perm_key instead; neither the keys nor the channels are permuted
// here. K's (scale, bias) pairs are staged alongside the tile because the key index
// is half-wave uniform, so every lane of a half reads the same pair.
//
// The u8 P scale and the per-channel V scale are constant across key blocks and
// fold into the epilogue; l sums the packed bytes so numerator and denominator
// quantize identically.
#include <hip/hip_runtime.h>

#include <cstdint>

#include "sol_common.h"

namespace comfy::hip_backend::sol {
namespace {

constexpr int HD = kHeadDim;
constexpr int BQ = kBlock;  // queries per workgroup: one query block
constexpr int BK = kBlock;  // keys per routed block
constexpr int NWAVE = BQ / 16, kThreads = NWAVE * 32;
constexpr int DT = HD / 16;  // contraction tiles for S = K Q^T
constexpr int KT = BK / 16;  // score key tiles
constexpr int NT = HD / 16;  // output channel tiles

#if defined(COMFY_MMA_GFX11)
constexpr int kLdsPad = 16;
#else
constexpr int kLdsPad = 8;
#endif
constexpr int kStrideK = HD + kLdsPad;
constexpr int kStrideV = BK + kLdsPad;

// qi:  [B,T,H,D] int8                  qs:  [B,T,H] f32
// ki:  [B*H,Tp,D] int8                 ksb: [B*H,Tp] float2 = (ks, bias)
// vT:  [B*H,D,Tp] int8 (transposed)    vsc: [B*H,D] f32
template <typename TOut>
__global__ __launch_bounds__(kThreads) void sol_exact_kernel(
    const int8_t* __restrict__ qi, const float* __restrict__ qs, const int8_t* __restrict__ ki,
    const float2* __restrict__ ksb, const int8_t* __restrict__ vT, const float* __restrict__ vsc,
    const uint16_t* __restrict__ blk_idx, const int32_t* __restrict__ blk_cnt,
    const __bf16* __restrict__ o_part, const float* __restrict__ m_part,
    const float* __restrict__ l_part, TOut* __restrict__ out, int T, int Tp, int H, int NTB,
    float scale_log2) {
    __shared__ __attribute__((aligned(16))) int8_t sK[BK * kStrideK];
    __shared__ __attribute__((aligned(16))) int8_t sVt[HD * kStrideV];
    __shared__ float2 sKsb[BK];

    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int r = frag_row(lane);
    const int q_block = blockIdx.x, bh = blockIdx.y;
    const int batch = bh / H, head = bh % H;
    const int64_t bh_base = static_cast<int64_t>(batch) * T * H * HD + static_cast<int64_t>(head) * HD;
    const int64_t bh_s = static_cast<int64_t>(batch) * T * H + head;
    const int64_t vT_base = static_cast<int64_t>(bh) * HD * Tp;
    const int64_t vsc_base = static_cast<int64_t>(bh) * HD;
    const int64_t kp_base = static_cast<int64_t>(bh) * Tp;
    const int64_t kd_base = static_cast<int64_t>(bh) * Tp * HD;

    const int q_row = q_block * BQ + wave * 16 + r;
    const int q_clamped = imin(q_row, T - 1);

    typename MmaInt8::Frag q_frag[DT];
    {
        const int8_t* qrow = qi + bh_base + static_cast<int64_t>(q_clamped) * H * HD;
#pragma unroll
        for (int dt = 0; dt < DT; ++dt) q_frag[dt] = MmaInt8::load_aligned(qrow, 0, dt * 16, 0, lane);
    }
    const float qsc = qs[bh_s + static_cast<int64_t>(q_clamped) * H] * scale_log2;

    const size_t qb_s = static_cast<size_t>(bh) * gridDim.x + q_block;
    const uint16_t* my_idx = blk_idx + qb_s * NTB;
    const int n_blocks = blk_cnt[qb_s];

    // resume route's state: one (o, m, l) per (b, h, query block)
    float o_acc[NT][8];
    {
        const __bf16* orow = o_part + qb_s * HD;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) load_o_tile<__bf16>(orow, nt * 16, o_acc[nt], lane);
    }
    float m_r = m_part[qb_s], l_r = l_part[qb_s], c_r = m_r;   // c_r: scale o_acc / l_r are carried in

    for (int kb = 0; kb < n_blocks; ++kb) {
        const int64_t k0 = static_cast<int64_t>(my_idx[kb]) * BK;

        __syncthreads();
        for (int idx = tid; idx < BK * (HD / 8); idx += kThreads) {
            const int p = idx / (HD / 8), c8 = (idx % (HD / 8)) * 8;
            *reinterpret_cast<uint2*>(sK + p * kStrideK + c8) =
                *reinterpret_cast<const uint2*>(ki + kd_base + (k0 + p) * HD + c8);
        }
        for (int idx = tid; idx < HD * (BK / 8); idx += kThreads) {
            const int c = idx / (BK / 8), part = (idx % (BK / 8)) * 8;
            *reinterpret_cast<uint2*>(sVt + c * kStrideV + part) = *reinterpret_cast<const uint2*>(
                vT + vT_base + static_cast<int64_t>(c) * Tp + k0 + part);
        }
        for (int idx = tid; idx < BK; idx += kThreads) sKsb[idx] = ksb[kp_base + k0 + idx];
        __syncthreads();

        typename MmaInt8::Acc s_acc[KT];
#pragma unroll
        for (int kt = 0; kt < KT; ++kt) {
            s_acc[kt] = MmaInt8::zero();
#pragma unroll
            for (int dt = 0; dt < DT; ++dt) {
                s_acc[kt] = MmaInt8::mma(
                    MmaInt8::load_aligned(sK, kt * 16 + r, dt * 16, kStrideK, lane), q_frag[dt],
                    s_acc[kt]);
            }
        }

        float p_val[KT][8];
        // P is quantized against the block max (floored 2^-20 under the running
        // max) so every block gets the full u8 range
        float bmax = m_r - 20.f;
#pragma unroll
        for (int kt = 0; kt < KT; ++kt) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const float2 kb2 = sKsb[kt * 16 + acc_row(lane, e)];
                // A dead key carries a zero scale and a kNeg bias, so its score is kNeg.
                const float s = fmaf(static_cast<float>(s_acc[kt][e]), qsc * kb2.x, kb2.y);
                p_val[kt][e] = s;
                bmax = fmaxf(bmax, s);
            }
        }
        bmax = fmaxf(bmax, swap_half_wave(bmax));
        const float alpha = exp2f(c_r - bmax);
        c_r = bmax;
        m_r = fmaxf(m_r, bmax);

        // u8 P scale folded into the exponent (+log2 255); l carries it too
        const float m_off = bmax - kProbU8Offset;
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

    if (q_row >= T) return;
    // l is zero only when a row got no mass from either branch; zeros are right there
    const float inv = 1.f / fmaxf(l_r, 1e-30f);
    TOut* orow = out + bh_base + static_cast<int64_t>(q_row) * H * HD;
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        float vals[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            vals[e] = o_acc[nt][e] * inv * vsc[vsc_base + nt * 16 + acc_row(lane, e)];
        }
        store_o_tile<TOut>(orow, nt * 16, vals, lane);
    }
}

}  // namespace
}  // namespace comfy::hip_backend::sol

using namespace comfy::hip_backend::sol;

void launch_sol_exact(const void* qi, const void* qs, const void* ki, const void* ksb,
                      const void* vT, const void* vsc, const void* blk_idx, const void* blk_cnt,
                      const void* o_part, const void* m_part, const void* l_part, void* out, int B,
                      int T, int Tp, int H, int NQ, int NTB, float scale_log2, int elem,
                      hipStream_t stream) {
    dim3 grid(NQ, B * H);
#define SOLX_LAUNCH(TOut)                                                                    \
    sol_exact_kernel<TOut><<<grid, kThreads, 0, stream>>>(                                   \
        static_cast<const int8_t*>(qi), static_cast<const float*>(qs),                       \
        static_cast<const int8_t*>(ki), static_cast<const float2*>(ksb),                     \
        static_cast<const int8_t*>(vT), static_cast<const float*>(vsc),                      \
        static_cast<const uint16_t*>(blk_idx), static_cast<const int32_t*>(blk_cnt),         \
        static_cast<const __bf16*>(o_part), static_cast<const float*>(m_part),               \
        static_cast<const float*>(l_part), static_cast<TOut*>(out), T, Tp, H, NTB, scale_log2)
    if (elem == kSolFp16) {
        SOLX_LAUNCH(_Float16);
    } else {
        SOLX_LAUNCH(__bf16);
    }
#undef SOLX_LAUNCH
}
