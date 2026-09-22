// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// INT8-QK + BF16/F16-SV attention, following the SageAttention gfx110x design:
// Q and K are int8 (WMMA iu8 scores), V is left unquantized in its own dtype
// (transposed to [B*H*D, padded_N] by launch_sage_transpose_v) and the
// probabilities stay bf16/fp16, so the PV matmul runs on the bf16 WMMA unit and
// never round-trips through u8. That is both more accurate (no u8 probability
// clamp, no V quantization) and structurally the same kernel the reference
// sageattn library ships. The transposed S^T tiling and per-lane online softmax
// are inherited from int8_attn.hip; see its header comment for why.
#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>
#include <type_traits>

#include "../mma.h"
#include "sage_common.h"

namespace comfy::hip_backend::sage {

// Mirrors MaskMode in int8_attn.hip.
enum class MaskMode { kNone = 0, kCustom = 2 };

namespace {

constexpr int kCtaQ = 128;
#if defined(COMFY_MMA_GFX11)
constexpr int kLdsPad = 16;
#else
constexpr int kLdsPad = 8;
#endif

__forceinline__ __device__ int imin(int a, int b) { return a < b ? a : b; }
constexpr int div_ceil_host(int a, int b) { return (a + b - 1) / b; }

// Reads one mask element as raw bits (same fast-math-safe test as int8_attn.hip).
__forceinline__ __device__ bool mask_keep(const void* mask, int64_t offset, int dtype_code,
                                          float& bias) {
    bias = 0.0f;
    if (dtype_code == 3) {
        return static_cast<const uint8_t*>(mask)[offset] != 0;
    }
    if (dtype_code == 0) {
        const uint32_t bits = static_cast<const uint32_t*>(mask)[offset];
        if ((bits & 0x7F800000u) == 0x7F800000u) return false;
        bias = __uint_as_float(bits);
        return true;
    }
    const uint16_t bits = static_cast<const uint16_t*>(mask)[offset];
    if (dtype_code == 1) {
        if ((bits & 0x7C00u) == 0x7C00u) return false;
        bias = __half2float(*reinterpret_cast<const __half*>(&bits));
        return true;
    }
    if ((bits & 0x7F80u) == 0x7F80u) return false;
    bias = static_cast<float>(*reinterpret_cast<const __bf16*>(&bits));
    return true;
}

template <int HD>
struct Shape {
    static constexpr int kWarpQ = 16;
    static constexpr int kWaves = kCtaQ / kWarpQ;
    static constexpr int kThreads = kWaves * 32;
    static constexpr int kQueryTiles = kWarpQ / 16;
    static constexpr int kDimTiles = HD / 16;
};

template <int HD, int CTA_K, MaskMode kMask, typename VD, typename OutT>
__global__ __launch_bounds__(Shape<HD>::kThreads) void int8_qk_bf16sv_kernel(
    const int8_t* __restrict__ q, const int8_t* __restrict__ k, const VD* __restrict__ v,
    OutT* __restrict__ o, const float* __restrict__ q_scale, const float* __restrict__ k_scale,
    const void* __restrict__ mask, int64_t mask_stride_b, int64_t mask_stride_h,
    int64_t mask_stride_q, int64_t mask_stride_k, int mask_dtype_code, int qo_len, int kv_len,
    int qo_len_padded, int k_groups_per_head, int num_kv_groups, int64_t q_stride_b,
    int64_t q_stride_h, int64_t k_stride_b, int64_t k_stride_h, int64_t v_stride_b,
    int64_t v_stride_h, int64_t v_stride_d, int64_t o_stride_b, int64_t o_stride_h,
    int64_t o_stride_n, float sm_scale) {

    using S = Shape<HD>;
    constexpr int QT = S::kQueryTiles;
    constexpr int KT = CTA_K / 16;
    constexpr int DT = S::kDimTiles;
    constexpr int kStrideK = HD + kLdsPad;
    constexpr int kStrideV = CTA_K + kLdsPad;
    constexpr bool kCustomMask = kMask == MaskMode::kCustom;
    // The PV WMMA has to match V's dtype: fp16 V feeds the f16 matrix core,
    // bf16 V the bf16 one.
    using PV_Mma = std::conditional_t<std::is_same_v<VD, __half>, MmaF16, MmaBf16G>;

    __shared__ __attribute__((aligned(16))) int8_t smem_k[CTA_K * kStrideK];
    __shared__ __attribute__((aligned(16))) VD smem_v[HD * kStrideV];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wave = tid >> 5;
    const int r = frag_row(lane);

    const int block_q = blockIdx.x * kCtaQ;
    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    const int num_qo_heads = gridDim.y;
    const int kv_head = head / num_kv_groups;

    const int8_t* q_head = q + static_cast<int64_t>(batch) * q_stride_b +
                           static_cast<int64_t>(head) * q_stride_h;
    const int8_t* k_head = k + static_cast<int64_t>(batch) * k_stride_b +
                           static_cast<int64_t>(kv_head) * k_stride_h;
    const VD* v_head = v + static_cast<int64_t>(batch) * v_stride_b +
                       static_cast<int64_t>(kv_head) * v_stride_h;

    sm_scale *= kLog2e;

    constexpr bool kCacheQ = DT * static_cast<int>(sizeof(typename MmaInt8::Frag)) / 4 <= 32;
    typename MmaInt8::Frag q_frag[QT][kCacheQ ? DT : 1];
    const int8_t* q_rows[QT];
    float q_lane_scale[QT];
    int q_lane_idx[QT];
#pragma unroll
    for (int fq = 0; fq < QT; ++fq) {
        const int q_idx = block_q + wave * S::kWarpQ + fq * 16 + r;
        q_lane_idx[fq] = q_idx;
        q_rows[fq] = q_head + static_cast<int64_t>(imin(q_idx, qo_len - 1)) * HD;
        if constexpr (kCacheQ) {
#pragma unroll
            for (int dt = 0; dt < DT; ++dt) {
                q_frag[fq][dt] = MmaInt8::load_aligned(q_rows[fq], 0, dt * 16, 0, lane);
            }
        }
        q_lane_scale[fq] =
            q_scale[(static_cast<int64_t>(batch) * num_qo_heads + head) * qo_len_padded +
                    imin(q_idx, qo_len_padded - 1)];
    }

    float o_val[QT][DT][8];
#pragma unroll
    for (int fq = 0; fq < QT; ++fq)
#pragma unroll
        for (int dt = 0; dt < DT; ++dt)
#pragma unroll
            for (int e = 0; e < 8; ++e) o_val[fq][dt][e] = 0.0f;

    float m_run[QT], d_run[QT];
    bool row_valid[QT];
#pragma unroll
    for (int fq = 0; fq < QT; ++fq) {
        m_run[fq] = kMaskedScore;
        d_run[fq] = 1.0f;
        row_valid[fq] = !kCustomMask;
    }

    const int num_iters = (kv_len + CTA_K - 1) / CTA_K;
    constexpr bool kSplitLoop = DT <= 8;

    int interior_iters = 0;
    if constexpr (!kCustomMask && kSplitLoop) {
        if (sm_scale >= 0.0f) {
            interior_iters = kv_len / CTA_K;
        }
    }

    constexpr int kUnitsK = HD / 8;
    constexpr int kUnitsV = CTA_K / 8;  // 8 VD per 16-byte chunk

    auto run_iter = [&](auto boundary_tag, int iter) {
        constexpr bool kBoundary = decltype(boundary_tag)::value;
        const int kb0 = iter * CTA_K;

        __syncthreads();
        for (int idx = tid; idx < CTA_K * kUnitsK; idx += S::kThreads) {
            const int row = idx / kUnitsK;
            const int unit = idx % kUnitsK;
            const int key = kb0 + row;
            uint2 val = {0u, 0u};
            if (!kBoundary || key < kv_len) {
                val = *reinterpret_cast<const uint2*>(k_head + static_cast<int64_t>(key) * HD +
                                                      unit * 8);
            }
            *reinterpret_cast<uint2*>(smem_k + row * kStrideK + unit * 8) = val;
        }
        for (int idx = tid; idx < HD * kUnitsV; idx += S::kThreads) {
            const int row = idx / kUnitsV;
            const int unit = idx % kUnitsV;
            *reinterpret_cast<uint4*>(smem_v + row * kStrideV + unit * 8) =
                *reinterpret_cast<const uint4*>(v_head + static_cast<int64_t>(row) * v_stride_d +
                                                kb0 + unit * 8);
        }
        __syncthreads();

        float key_scale[KT];
#pragma unroll
        for (int fk = 0; fk < KT; ++fk) {
            key_scale[fk] =
                k_scale[(static_cast<int64_t>(batch) * (num_qo_heads / num_kv_groups) + kv_head) *
                            k_groups_per_head +
                        iter * KT + fk];
        }

#pragma unroll
        for (int fq = 0; fq < QT; ++fq) {
            typename MmaInt8::Acc s_acc[KT];
#pragma unroll
            for (int fk = 0; fk < KT; ++fk) {
                s_acc[fk] = MmaInt8::zero();
#pragma unroll
                for (int dt = 0; dt < DT; ++dt) {
                    const typename MmaInt8::Frag qf =
                        kCacheQ ? q_frag[fq][kCacheQ ? dt : 0]
                                : MmaInt8::load_aligned(q_rows[fq], 0, dt * 16, 0, lane);
                    s_acc[fk] = MmaInt8::mma(
                        MmaInt8::load_aligned(smem_k, fk * 16 + r, dt * 16, kStrideK, lane), qf,
                        s_acc[fk]);
                }
            }

            const int q_idx = q_lane_idx[fq];
            float tile_scale_k[KT];
#pragma unroll
            for (int fk = 0; fk < KT; ++fk) {
                tile_scale_k[fk] = sm_scale * q_lane_scale[fq] * key_scale[fk];
            }

            float prob[kBoundary ? KT : 1][8];
            float tile_m;

            if constexpr (kBoundary) {
                int64_t mask_row = 0;
                if constexpr (kCustomMask) {
                    mask_row = static_cast<int64_t>(batch) * mask_stride_b +
                               static_cast<int64_t>(head) * mask_stride_h +
                               static_cast<int64_t>(imin(q_idx, qo_len - 1)) * mask_stride_q;
                }
#pragma unroll
                for (int fk = 0; fk < KT; ++fk) {
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        const int key = kb0 + fk * 16 + acc_row(lane, e);
                        float score = static_cast<float>(s_acc[fk][e]) * tile_scale_k[fk];
                        bool keep = key < kv_len;
                        if constexpr (kCustomMask) {
                            keep = keep && q_idx < qo_len;
                            if (keep) {
                                float bias;
                                keep = mask_keep(
                                    mask,
                                    mask_row + static_cast<int64_t>(key) * mask_stride_k,
                                    mask_dtype_code, bias);
                                score += bias * kLog2e;
                            }
                            row_valid[fq] = row_valid[fq] || keep;
                        }
                        prob[fk][e] = keep ? score : kMaskedScore;
                    }
                }
                tile_m = prob[0][0];
#pragma unroll
                for (int fk = 0; fk < KT; ++fk)
#pragma unroll
                    for (int e = 0; e < 8; ++e) tile_m = fmaxf(tile_m, prob[fk][e]);
            } else {
                tile_m = kMaskedScore;
#pragma unroll
                for (int fk = 0; fk < KT; ++fk) {
                    int key_max = s_acc[fk][0];
#pragma unroll
                    for (int e = 1; e < 8; ++e) {
                        key_max = key_max > s_acc[fk][e] ? key_max : s_acc[fk][e];
                    }
                    tile_m = fmaxf(tile_m, static_cast<float>(key_max) * tile_scale_k[fk]);
                }
            }

            // No u8 offset: the probabilities stay in bf16/fp16.
            tile_m = fmaxf(tile_m, swap_half_wave(tile_m));

            const float m_prev = m_run[fq];
            const float smaller = exp2f(-fabsf(m_prev - tile_m));
            const float o_scale = m_prev < tile_m ? smaller : 1.0f;
            const float tile_scale = tile_m < m_prev ? smaller : 1.0f;
            m_run[fq] = fmaxf(m_prev, tile_m);
            d_run[fq] *= o_scale;

#pragma unroll
            for (int dt = 0; dt < DT; ++dt)
#pragma unroll
                for (int e = 0; e < 8; ++e) o_val[fq][dt][e] *= o_scale;

            typename PV_Mma::Frag p_frag[KT];
            float tile_sum = 0.0f;
#pragma unroll
            for (int fk = 0; fk < KT; ++fk) {
                float p_vals[8];
                const float negative_m = -tile_m;
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const float p =
                        kBoundary
                            ? exp2f(prob[kBoundary ? fk : 0][e] + negative_m)
                            : exp2f(fmaf(static_cast<float>(s_acc[fk][e]), tile_scale_k[fk],
                                         negative_m));
                    p_vals[e] = p;
                    tile_sum += p;
                }
                if constexpr (std::is_same_v<VD, __half>)
                    p_frag[fk] = pack_prob_frag_f16(p_vals, lane);
                else
                    p_frag[fk] = pack_prob_frag_bf16(p_vals, lane);
            }
            d_run[fq] += tile_sum * tile_scale;

#pragma unroll
            for (int dt = 0; dt < DT; ++dt) {
                PV_Mma::Acc partial = PV_Mma::zero();
#pragma unroll
                for (int fk = 0; fk < KT; ++fk) {
                    // Element-addressed fragment read, the same shape Sol-Attn's
                    // pooled PV uses for its bf16 WMMA. V is staged transposed
                    // in LDS each iteration (staging beats direct global reads
                    // for the 6-wave re-reads on this part).
                    partial = PV_Mma::mma(
                        load_frag_16bit<PV_Mma>(
                            reinterpret_cast<const typename PV_Mma::Elem*>(smem_v +
                                                                          (dt * 16 + r) * kStrideV +
                                                                          fk * 16),
                            lane),
                        p_frag[fk], partial);
                }
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    o_val[fq][dt][e] =
                        fmaf(static_cast<float>(partial[e]), tile_scale, o_val[fq][dt][e]);
                }
            }
        }
    };

    int iter = 0;
    for (; iter < interior_iters; ++iter) run_iter(std::false_type{}, iter);
    for (; iter < num_iters; ++iter) run_iter(std::true_type{}, iter);

#pragma unroll
    for (int fq = 0; fq < QT; ++fq) {
        d_run[fq] += swap_half_wave(d_run[fq]);
        if constexpr (kCustomMask) {
            int any = row_valid[fq] ? 1 : 0;
            any |= static_cast<int>(swap_half_wave_b32(static_cast<uint32_t>(any)));
            row_valid[fq] = any != 0;
        }
    }

#pragma unroll
    for (int fq = 0; fq < QT; ++fq) {
        const int q_idx = q_lane_idx[fq];
        if (q_idx >= qo_len) continue;
        const float inv_d = row_valid[fq] ? 1.0f / d_run[fq] : 0.0f;
        OutT* row = o + static_cast<int64_t>(batch) * o_stride_b +
                    static_cast<int64_t>(head) * o_stride_h +
                    static_cast<int64_t>(q_idx) * o_stride_n;
#pragma unroll
        for (int dt = 0; dt < DT; ++dt) {
            float vals[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int d = dt * 16 + acc_row(lane, e);
                vals[e] = o_val[fq][dt][e] * inv_d;
            }
            store_o_tile<OutT>(row, dt * 16, vals, lane);
        }
    }
}

template <int HD, int CTA_K, MaskMode kMask, typename VD, typename OutT>
void launch_one(const int8_t* q, const int8_t* k, const VD* v, OutT* o, const float* q_scale,
                const float* k_scale, const void* mask, int64_t mask_stride_b,
                int64_t mask_stride_h, int64_t mask_stride_q, int64_t mask_stride_k,
                int mask_dtype_code, int batch, int qo_len, int kv_len, int qo_len_padded,
                int num_qo_heads, int k_groups_per_head, int num_kv_groups, int64_t q_stride_b,
                int64_t q_stride_h, int64_t k_stride_b, int64_t k_stride_h, int64_t v_stride_b,
                int64_t v_stride_h, int64_t v_stride_d, int64_t o_stride_b, int64_t o_stride_h,
                int64_t o_stride_n, float sm_scale, hipStream_t stream) {

    const dim3 grid(div_ceil_host(qo_len, kCtaQ), num_qo_heads, batch);
    int8_qk_bf16sv_kernel<HD, CTA_K, kMask, VD, OutT>
        <<<grid, Shape<HD>::kThreads, 0, stream>>>(
        q, k, v, o, q_scale, k_scale, mask, mask_stride_b, mask_stride_h, mask_stride_q,
        mask_stride_k, mask_dtype_code, qo_len, kv_len, qo_len_padded, k_groups_per_head,
        num_kv_groups, q_stride_b, q_stride_h, k_stride_b, k_stride_h, v_stride_b, v_stride_h,
        v_stride_d, o_stride_b, o_stride_h, o_stride_n, sm_scale);
}

}  // namespace
}  // namespace comfy::hip_backend::sage

using namespace comfy::hip_backend::sage;

extern "C" void launch_sage_bf16sv_attn(
    const void* q, const void* k, const void* v, void* o, const void* q_scale,
    const void* k_scale, const void* mask, int64_t mask_stride_b, int64_t mask_stride_h,
    int64_t mask_stride_q, int64_t mask_stride_k, int mask_dtype_code, int cta_k, int batch,
    int qo_len, int kv_len, int qo_len_padded, int num_qo_heads, int num_kv_heads, int head_dim,
    int k_groups_per_head, int64_t q_stride_b, int64_t q_stride_h, int64_t k_stride_b,
    int64_t k_stride_h, int64_t v_stride_b, int64_t v_stride_h, int64_t v_stride_d,
    int64_t o_stride_b, int64_t o_stride_h, int64_t o_stride_n, float sm_scale,
    int output_dtype_code, int input_dtype_code, hipStream_t stream) {

    if (reinterpret_cast<uintptr_t>(q) % 16 != 0 || reinterpret_cast<uintptr_t>(k) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(v) % 16 != 0 || reinterpret_cast<uintptr_t>(o) % 16 != 0) {
        throw std::runtime_error(
            "sage bf16-SV attention: q, k, v and o base pointers must be 16-byte aligned");
    }

    const int num_kv_groups = num_qo_heads / num_kv_heads;
    const auto* qi = static_cast<const int8_t*>(q);
    const auto* ki = static_cast<const int8_t*>(k);
    const auto* qs = static_cast<const float*>(q_scale);
    const auto* ks = static_cast<const float*>(k_scale);

#define SAGE_ATTN_LAUNCH(HD, CK, MM, VD, OUT)                                                    \
    launch_one<HD, CK, MM, VD, OUT>(qi, ki, static_cast<const VD*>(v),            \
                                                   static_cast<OUT*>(o), qs, ks, mask,          \
                                                   mask_stride_b, mask_stride_h, mask_stride_q,  \
                                                   mask_stride_k, mask_dtype_code, batch,        \
                                                   qo_len, kv_len, qo_len_padded, num_qo_heads,   \
                                                   k_groups_per_head, num_kv_groups, q_stride_b,  \
                                                   q_stride_h, k_stride_b, k_stride_h,           \
                                                   v_stride_b, v_stride_h, v_stride_d,           \
                                                   o_stride_b, o_stride_h, o_stride_n, sm_scale, \
                                                   stream)

    // The wide tile is a schedule for D128 long unmasked keys, same as the
    // int8 path; D256 has no LDS room for it, and `if constexpr` keeps its
    // instantiation out of the binary there.
    const bool wide = (cta_k == 128) && (head_dim != 256);

    auto launch_hd = [&](auto hd_tag, auto ck_tag, auto vd_tag, auto out_tag) {
        constexpr int HD = decltype(hd_tag)::value;
        constexpr int CK = decltype(ck_tag)::value;
        using VD = decltype(vd_tag);
        using OUT = decltype(out_tag);
        if (mask != nullptr)
            SAGE_ATTN_LAUNCH(HD, CK, MaskMode::kCustom, VD, OUT);
        else
            SAGE_ATTN_LAUNCH(HD, CK, MaskMode::kNone, VD, OUT);
    };
    auto launch_dtype = [&](auto hd_tag, auto ck_tag) {
        if (input_dtype_code == 1) {
            if (output_dtype_code == 1)
                launch_hd(hd_tag, ck_tag, __half{}, __half{});
            else
                launch_hd(hd_tag, ck_tag, __half{}, __bf16{});
        } else {
            if (output_dtype_code == 1)
                launch_hd(hd_tag, ck_tag, __bf16{}, __half{});
            else
                launch_hd(hd_tag, ck_tag, __bf16{}, __bf16{});
        }
    };
    auto launch_ck = [&](auto hd_tag) {
        constexpr int HD = decltype(hd_tag)::value;
        if constexpr (HD != 256) {
            if (wide)
                launch_dtype(hd_tag, std::integral_constant<int, 128>{});
            else
                launch_dtype(hd_tag, std::integral_constant<int, 64>{});
        } else {
            launch_dtype(hd_tag, std::integral_constant<int, 64>{});
        }
    };

    switch (head_dim) {
        case 64: launch_ck(std::integral_constant<int, 64>{}); break;
        case 128: launch_ck(std::integral_constant<int, 128>{}); break;
        case 256: launch_ck(std::integral_constant<int, 256>{}); break;
        default:
            throw std::runtime_error("sage bf16-SV attention: unsupported head_dim " +
                                     std::to_string(head_dim));
    }
}



