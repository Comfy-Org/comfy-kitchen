// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Pure INT8 attention on RDNA matrix cores. The CUDA backend's sage_attention/
// sources are the reference for what this computes; how it computes it differs,
// because the RDNA fragment layout does.
//
// The score matrix is evaluated TRANSPOSED, S^T = K @ Q^T, and the output with
// it, O^T = V^T @ P^T. The WMMA accumulator hands a lane one column and eight
// rows, so under S^T a lane owns one query and eight keys: the online softmax
// becomes per-lane scalar work with a single shuffle to fold the half-waves, and
// the query stays on the lane holding its denominator. On gfx12 the score
// accumulator is then already the P operand of the second MMA, element for
// element, so P never goes through LDS; gfx11 interleaves the accumulator rows
// across half-waves and pays two shuffles and two byte permutes for it.
//
// Tiling: 128 queries per block, 16 per wave, 64 keys per iteration; LDS holds
// only K and V. A wider query tile, a 128-key tile, staging the tiles through
// registers across the barrier, and folding the mask into the score loop were
// all tried and are all slower or spill. Do not re-try them blind.
#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>

#include "../mma.h"
#include "sage_common.h"

namespace comfy::hip_backend::sage {

// Mirrors MaskMode in the CUDA backend's attn_utils.cuh.
enum class MaskMode { kNone = 0, kCustom = 2 };

namespace {

constexpr int kCtaQ = 128;
// gfx11 reads a whole 16-byte K-step per lane, so a 16-byte row keeps that to one
// LDS instruction; the wider pad costs a two-way bank conflict and buys back four
// times the issue slots. gfx12 reads 8 bytes, where an 8-byte pad is already one
// instruction and keeps 16 rows on 16 distinct banks.
#if defined(COMFY_MMA_GFX11)
constexpr int kLdsPad = 16;
#else
constexpr int kLdsPad = 8;
#endif

__forceinline__ __device__ int imin(int a, int b) { return a < b ? a : b; }

constexpr int div_ceil_host(int a, int b) { return (a + b - 1) / b; }

// Like CUDA's ex2.approx.ftz: probabilities this small already round to zero in
// U8. exp2f also emits a subnormal correction around every hardware exponential.
__forceinline__ __device__ float softmax_exp2(float value) {
    return __builtin_amdgcn_exp2f(value);
}

// Reserve a factor of 256 in the FP32 normalization scale. This reduces how
// often a new maximum rescales the entire running output; each tile still
// quantizes its own probabilities over the full U8 range.
constexpr float kSoftmaxHeadroom = 8.0f;

// 16 queries per wave at every head_dim. The CUDA launcher widens D64 to 32 to
// halve its shared-memory traffic; here the output accumulator is transposed, so
// a wider query tile grows the live FP32 set instead and spills on both families.
template <int HD>
struct Shape {
    static constexpr int kWarpQ = 16;
    static constexpr int kWaves = kCtaQ / kWarpQ;
    static constexpr int kThreads = kWaves * 32;
    static constexpr int kQueryTiles = kWarpQ / 16;
    static constexpr int kDimTiles = HD / 16;
};

// Reads one mask element as raw bits. -ffast-math folds isfinite() away, so the
// non-finite test that decides whether an additive bias means "drop this key"
// has to be an exponent bit test on the value as loaded, never a float compare.
// See the fast-math note in hip/mma.h's sibling kernels.
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

template <int HD, int CTA_K, MaskMode kMask, typename OutT>
__global__ __launch_bounds__(Shape<HD>::kThreads) void int8_attn_kernel(
    const int8_t* __restrict__ q, const int8_t* __restrict__ k, const int8_t* __restrict__ v,
    OutT* __restrict__ o, const float* __restrict__ q_scale, const float* __restrict__ k_scale,
    const float* __restrict__ v_scale, const void* __restrict__ mask, int64_t mask_stride_b,
    int64_t mask_stride_h, int64_t mask_stride_q, int64_t mask_stride_k, int mask_dtype_code,
    int qo_len, int kv_len, int qo_len_padded, int k_groups_per_head, int num_kv_groups,
    int64_t q_stride_b, int64_t q_stride_h, int64_t k_stride_b, int64_t k_stride_h,
    int64_t v_stride_b, int64_t v_stride_h, int64_t v_stride_d, int64_t o_stride_b,
    int64_t o_stride_h, int64_t o_stride_n, float sm_scale) {

    using S = Shape<HD>;
    constexpr int QT = S::kQueryTiles;
    constexpr int KT = CTA_K / 16;
    constexpr int DT = S::kDimTiles;
    constexpr int kStrideK = HD + kLdsPad;
    constexpr int kStrideV = CTA_K + kLdsPad;
    constexpr bool kCustomMask = kMask == MaskMode::kCustom;
#if defined(COMFY_MMA_GFX12)
    constexpr bool kBiasPV = HD == 128 && !kCustomMask;
#else
    constexpr bool kBiasPV = false;
#endif
    // Adding a signed integer x to these bits gives the exact FP32 value
    // 12582912 + x while |x| <= 2^22. A PV tile is bounded by
    // CTA_K * 128 * 255, including the most negative signed INT8 value.
    // Starting the integer MMA here lets its result feed an FP32 FMA by bitcast.
    static_assert(CTA_K * 128 * 255 < (1 << 22));
    constexpr int kIntBias = 0x4b400000;
    constexpr float kFloatBias = 12582912.0f;
    typename MmaInt8::Acc bias_acc = {
        kIntBias, kIntBias, kIntBias, kIntBias, kIntBias, kIntBias, kIntBias, kIntBias};
    // Materialize the shared MMA initializer once, outside the key loop.
    if constexpr (kBiasPV) asm volatile("" : "+v"(bias_acc));

    __shared__ __attribute__((aligned(16))) int8_t smem_k[CTA_K * kStrideK];
    __shared__ __attribute__((aligned(16))) int8_t smem_v[HD * kStrideV];

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
    const int8_t* v_head = v + static_cast<int64_t>(batch) * v_stride_b +
                           static_cast<int64_t>(kv_head) * v_stride_h;

    // Base 2 throughout: the probabilities come out of exp2, not exp.
    sm_scale *= kLog2e;

    // Q is loop invariant and worth holding in registers, but only while it fits:
    // gfx11 fragments are twice as wide, and 16 of them at D256 sit on top of an
    // output accumulator that is already half the file. Past that the wave
    // reloads Q from global, which beats spilling the accumulator.
    constexpr bool kCacheQ = DT * static_cast<int>(sizeof(typename MmaInt8::Frag)) / 4 <= 32;
    typename MmaInt8::Frag q_frag[QT][kCacheQ ? DT : 1];
    const int8_t* q_rows[QT];
    float q_lane_scale[QT];
    int q_lane_idx[QT];
#pragma unroll
    for (int fq = 0; fq < QT; ++fq) {
        // A row past the end reads a clamped one; its output is never stored and
        // its scale is finite, so nothing downstream sees it.
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

    float o_bias[QT] = {};
    float o_val[QT][DT][8];
#pragma unroll
    for (int fq = 0; fq < QT; ++fq) {
#pragma unroll
        for (int dt = 0; dt < DT; ++dt) {
#pragma unroll
            for (int e = 0; e < 8; ++e) o_val[fq][dt][e] = 0.0f;
        }
    }

    // Finite sentinels, not -inf: every masked score still goes through fma and
    // exp2 before anything looks at it. Matches the CUDA initial state, d
    // included, so a fully masked row divides by one instead of zero.
    float m_run[QT], d_run[QT];
    bool row_valid[QT];
#pragma unroll
    for (int fq = 0; fq < QT; ++fq) {
        m_run[fq] = kMaskedScore;
        d_run[fq] = 1.0f;
        row_valid[fq] = !kCustomMask;
    }

    // An interior tile has every key in bounds, so nothing needs testing per
    // element: its maximum comes off the INT32 accumulator and exp2 reads the
    // accumulator too, which keeps a tile of scaled scores out of registers. A
    // custom mask has no interior.
    const int num_iters = (kv_len + CTA_K - 1) / CTA_K;

    // D256 keeps one code path: its output accumulator is 128 live floats, and
    // emitting the body twice there spills on both families.
    constexpr bool kSplitLoop = DT <= 8;

    int interior_iters = 0;
    if constexpr (!kCustomMask && kSplitLoop) {
        // Taking the maximum in INT32 assumes the dequantization scale is
        // positive. A negative softmax scale is legal and finite, so it gives up
        // the fast path rather than reporting the wrong maximum.
        if (sm_scale >= 0.0f) {
            interior_iters = kv_len / CTA_K;
        }
    }

    constexpr int kUnitsK = HD / 16;
    constexpr int kUnitsV = CTA_K / 16;

    // `boundary` is a compile-time tag, so the interior instance carries no mask
    // test and no per-key bounds check at all.
    auto run_iter = [&](auto boundary_tag, int iter) {
        constexpr bool kBoundary = decltype(boundary_tag)::value;
        const int kb0 = iter * CTA_K;

        __syncthreads();
        // K tile: [key][d]. Load 16 bytes at a time, then split the LDS stores:
        // gfx12's row padding guarantees only eight-byte alignment. A row past
        // the end is zeroed; its score is masked below.
#pragma unroll
        for (int idx = tid; idx < CTA_K * kUnitsK; idx += S::kThreads) {
            const int row = idx / kUnitsK;
            const int unit = idx % kUnitsK;
            const int key = kb0 + row;
            uint4 val = {0u, 0u, 0u, 0u};
            if (!kBoundary || key < kv_len) {
                val = *reinterpret_cast<const uint4*>(k_head + static_cast<int64_t>(key) * HD +
                                                     unit * 16);
            }
            *reinterpret_cast<uint2*>(smem_k + row * kStrideK + unit * 16) = uint2{val.x, val.y};
            *reinterpret_cast<uint2*>(smem_k + row * kStrideK + unit * 16 + 8) = uint2{val.z, val.w};
        }
        // V tile: [d][key], already transposed and zero padded by the quantizer,
        // so a whole CTA_K tile is always in bounds.
#pragma unroll
        for (int idx = tid; idx < HD * kUnitsV; idx += S::kThreads) {
            const int row = idx / kUnitsV;
            const int unit = idx % kUnitsV;
            const uint4 val = *reinterpret_cast<const uint4*>(
                v_head + static_cast<int64_t>(row) * v_stride_d + kb0 + unit * 16);
            *reinterpret_cast<uint2*>(smem_v + row * kStrideV + unit * 16) = uint2{val.x, val.y};
            *reinterpret_cast<uint2*>(smem_v + row * kStrideV + unit * 16 + 8) = uint2{val.z, val.w};
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
                // Allow two independent matrix accumulators to overlap, while
                // limiting the number of live K fragments and their registers.
                if ((fk + 1) % 2 == 0) __builtin_amdgcn_sched_barrier(0);
            }

            const int q_idx = q_lane_idx[fq];
            float tile_scale_k[KT];
#pragma unroll
            for (int fk = 0; fk < KT; ++fk) {
                tile_scale_k[fk] = sm_scale * q_lane_scale[fq] * key_scale[fk];
            }

            // A boundary tile dequantizes into a live array, because the mask has
            // to land on the scaled score. Folding the mask into the score loop
            // instead, to retire the accumulator earlier, spills D256.
            float prob[kBoundary ? KT : 1][8];
            float tile_m;

            if constexpr (kBoundary) {
                // Hoisted: only the key term varies per element, and eight live
                // 64-bit addresses in the inner loop cost registers this kernel
                // does not have. A key-only mask needs no separate instantiation:
                // it arrives with mask_stride_q at zero, so the query term drops
                // out here at runtime and the two kernels were byte-identical.
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
                for (int fk = 0; fk < KT; ++fk) {
#pragma unroll
                    for (int e = 0; e < 8; ++e) tile_m = fmaxf(tile_m, prob[fk][e]);
                }
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

            // The tile maximum is shifted so exp2 fills the unsigned INT8 range.
            // Its gap to the running maximum rides on the accumulator side as
            // tile_scale, which underflows to zero for a fully masked tile.
            tile_m = fmaxf(tile_m, swap_half_wave(tile_m));
            tile_m -= kProbU8Offset;

            const float m_prev = m_run[fq];
            const bool update_max = tile_m > m_prev;
            const float next_m = update_max ? tile_m + kSoftmaxHeadroom : m_prev;
            if (__any(update_max)) {
                const float o_scale = softmax_exp2(m_prev - next_m);
                d_run[fq] *= o_scale;
                if constexpr (kBiasPV) o_bias[fq] *= o_scale;
#pragma unroll
                for (int dt = 0; dt < DT; ++dt) {
#pragma unroll
                    for (int e = 0; e < 8; ++e) o_val[fq][dt][e] *= o_scale;
                }
            }
            const float tile_scale = softmax_exp2(tile_m - next_m);
            m_run[fq] = next_m;

            typename MmaInt8::Frag p_frag[KT];
#if defined(COMFY_MMA_GFX12)
            uint32_t tile_sum_u32 = 0;
#else
            float tile_sum = 0.0f;
#endif
#pragma unroll
            for (int fk = 0; fk < KT; ++fk) {
#if defined(COMFY_MMA_GFX12)
                uint32_t packed_lo = 0, packed_hi = 0;
#else
                uint32_t packed[8];
#endif
                const float negative_m = -tile_m;
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const float p =
                        kBoundary
                            ? softmax_exp2(prob[kBoundary ? fk : 0][e] + negative_m)
                            : softmax_exp2(fmaf(static_cast<float>(s_acc[fk][e]),
                                               tile_scale_k[fk], negative_m));
#if defined(COMFY_MMA_GFX12)
                    // Native conversion already rounds to nearest-even and
                    // saturates to U8; a separate rint/clamp is redundant.
                    if (e < 4) {
                        packed_lo = __builtin_amdgcn_cvt_pk_u8_f32(p, e, packed_lo);
                    } else {
                        packed_hi = __builtin_amdgcn_cvt_pk_u8_f32(p, e - 4, packed_hi);
                    }
#else
                    tile_sum += p;
                    packed[e] = prob_to_u8(p);
#endif
                }
#if defined(COMFY_MMA_GFX12)
                p_frag[fk] =
                    MmaInt8::Frag{static_cast<int>(packed_lo), static_cast<int>(packed_hi)};
                // Normalize with the same quantized probabilities used by PV.
                // The integer sum is exact and needs only one FP32 conversion.
                tile_sum_u32 = __builtin_amdgcn_sad_u8(
                    packed_hi, 0, __builtin_amdgcn_sad_u8(packed_lo, 0, tile_sum_u32));
#else
                p_frag[fk] = pack_prob_frag(packed, lane);
#endif
            }
#if defined(COMFY_MMA_GFX12)
            const float tile_sum = static_cast<float>(tile_sum_u32);
#endif
            d_run[fq] += tile_sum * tile_scale;

            // The INT32 partial covers one CTA_K tile, which is what keeps the
            // running output in FP32 without a second accumulator set per tile.
#pragma unroll
            for (int dt = 0; dt < DT; ++dt) {
                typename MmaInt8::Acc partial = kBiasPV ? bias_acc : MmaInt8::zero();
#pragma unroll
                for (int fk = 0; fk < KT; ++fk) {
                    partial = MmaInt8::mma_ub(
                        MmaInt8::load_aligned(smem_v, dt * 16 + r, fk * 16, kStrideV, lane), p_frag[fk],
                        partial);
                }
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const float value = kBiasPV ? __int_as_float(partial[e])
                                               : static_cast<float>(partial[e]);
                    o_val[fq][dt][e] = fmaf(value, tile_scale, o_val[fq][dt][e]);
                }
                if ((dt + 1) % 2 == 0) __builtin_amdgcn_sched_barrier(0);
            }
            if constexpr (kBiasPV) {
                // Follow the same rescaling and FMA order as O. Cancel every
                // eight tiles so the added bias cannot grow with sequence
                // length and consume the precision of small output values.
                o_bias[fq] = fmaf(kFloatBias, tile_scale, o_bias[fq]);
                if ((iter & 7) == 7) {
#pragma unroll
                    for (int dt = 0; dt < DT; ++dt) {
#pragma unroll
                        for (int e = 0; e < 8; ++e) o_val[fq][dt][e] -= o_bias[fq];
                    }
                    o_bias[fq] = 0.0f;
                }
            }
        }
    };

    int iter = 0;
    for (; iter < interior_iters; ++iter) run_iter(std::false_type{}, iter);
    for (; iter < num_iters; ++iter) run_iter(std::true_type{}, iter);

    // The two half-waves summed different keys of the same query.
#pragma unroll
    for (int fq = 0; fq < QT; ++fq) {
        d_run[fq] += swap_half_wave(d_run[fq]);
        if constexpr (kCustomMask) {
            int any = row_valid[fq] ? 1 : 0;
            any |= static_cast<int>(swap_half_wave_b32(static_cast<uint32_t>(any)));
            row_valid[fq] = any != 0;
        }
    }

    const float* v_scale_head =
        v_scale + (static_cast<int64_t>(batch) * (num_qo_heads / num_kv_groups) + kv_head) * HD;

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
                const float value = o_val[fq][dt][e] - (kBiasPV ? o_bias[fq] : 0.0f);
                vals[e] = value * v_scale_head[d] * inv_d;
            }
            store_o_tile<OutT>(row, dt * 16, vals, lane);
        }
    }
}

template <int HD, int CTA_K, MaskMode kMask, typename OutT>
void launch_one(const int8_t* q, const int8_t* k, const int8_t* v, OutT* o, const float* q_scale,
                const float* k_scale, const float* v_scale, const void* mask,
                int64_t mask_stride_b, int64_t mask_stride_h, int64_t mask_stride_q,
                int64_t mask_stride_k, int mask_dtype_code, int batch, int qo_len, int kv_len,
                int qo_len_padded, int num_qo_heads, int k_groups_per_head, int num_kv_groups,
                int64_t q_stride_b, int64_t q_stride_h, int64_t k_stride_b, int64_t k_stride_h,
                int64_t v_stride_b, int64_t v_stride_h, int64_t v_stride_d, int64_t o_stride_b,
                int64_t o_stride_h, int64_t o_stride_n, float sm_scale, hipStream_t stream) {

    const dim3 grid(div_ceil_host(qo_len, kCtaQ), num_qo_heads, batch);
    int8_attn_kernel<HD, CTA_K, kMask, OutT><<<grid, Shape<HD>::kThreads, 0, stream>>>(
        q, k, v, o, q_scale, k_scale, v_scale, mask, mask_stride_b, mask_stride_h, mask_stride_q,
        mask_stride_k, mask_dtype_code, qo_len, kv_len, qo_len_padded, k_groups_per_head,
        num_kv_groups, q_stride_b, q_stride_h, k_stride_b, k_stride_h, v_stride_b, v_stride_h,
        v_stride_d, o_stride_b, o_stride_h, o_stride_n, sm_scale);
}

}  // namespace
}  // namespace comfy::hip_backend::sage

using namespace comfy::hip_backend::sage;

extern "C" void launch_sage_int8_attn(
    const void* q, const void* k, const void* v, void* o, const void* q_scale,
    const void* k_scale, const void* v_scale, const void* mask, int64_t mask_stride_b,
    int64_t mask_stride_h, int64_t mask_stride_q, int64_t mask_stride_k, int mask_dtype_code,
    int cta_k, int batch, int qo_len, int kv_len, int qo_len_padded, int num_qo_heads,
    int num_kv_heads, int head_dim, int k_groups_per_head, int64_t q_stride_b, int64_t q_stride_h,
    int64_t k_stride_b, int64_t k_stride_h, int64_t v_stride_b, int64_t v_stride_h,
    int64_t v_stride_d, int64_t o_stride_b, int64_t o_stride_h, int64_t o_stride_n,
    float sm_scale, int output_dtype_code, hipStream_t stream) {

    // Q, K and V are read as 16-byte fragments and O is stored 8 elements at a
    // time; every offset the kernel forms is a multiple of those widths, so the
    // base addresses are the whole of the alignment requirement. The V quantizer
    // launcher checks the same thing.
    if (reinterpret_cast<uintptr_t>(q) % 16 != 0 || reinterpret_cast<uintptr_t>(k) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(v) % 16 != 0 || reinterpret_cast<uintptr_t>(o) % 16 != 0) {
        throw std::runtime_error(
            "sage int8 attention: q, k, v and o base pointers must be 16-byte aligned");
    }

    const int num_kv_groups = num_qo_heads / num_kv_heads;
    const auto* qi = static_cast<const int8_t*>(q);
    const auto* ki = static_cast<const int8_t*>(k);
    const auto* vi = static_cast<const int8_t*>(v);
    const auto* qs = static_cast<const float*>(q_scale);
    const auto* ks = static_cast<const float*>(k_scale);
    const auto* vs = static_cast<const float*>(v_scale);

#define SAGE_ATTN_LAUNCH(HD, CK, MM, OUT)                                                        \
    launch_one<HD, CK, MM, OUT>(qi, ki, vi, static_cast<OUT*>(o), qs, ks, vs, mask,              \
                                mask_stride_b, mask_stride_h, mask_stride_q, mask_stride_k,      \
                                mask_dtype_code, batch, qo_len, kv_len, qo_len_padded,           \
                                num_qo_heads, k_groups_per_head, num_kv_groups, q_stride_b,      \
                                q_stride_h, k_stride_b, k_stride_h, v_stride_b, v_stride_h,      \
                                v_stride_d, o_stride_b, o_stride_h, o_stride_n, sm_scale, stream)

    // One key tile, 64 keys. The CUDA backend widens to 128 for long unmasked
    // keys; ported and measured, that is slower on RDNA, where V is staged
    // transposed so the wide tile doubles both LDS tiles and costs more occupancy
    // than it saves. It does not fit at D256 at all, wanting 67 KB against a
    // 64 KB workgroup limit. A caller may still ask for 128 and get the same
    // answer from the narrow tile: tile size is a schedule, not a shape.
    (void)cta_k;
#define SAGE_ATTN_CTA(HD, MM, OUT) SAGE_ATTN_LAUNCH(HD, 64, MM, OUT)

#define SAGE_ATTN_OUT(HD, MM)                                                                    \
    do {                                                                                         \
        if (output_dtype_code == 1) {                                                            \
            SAGE_ATTN_CTA(HD, MM, __half);                                                       \
        } else {                                                                                 \
            SAGE_ATTN_CTA(HD, MM, __bf16);                                                       \
        }                                                                                        \
    } while (0)

#define SAGE_ATTN_MASK(HD)                                                                       \
    do {                                                                                         \
        if (mask != nullptr) {                                                                   \
            SAGE_ATTN_OUT(HD, MaskMode::kCustom);                                                \
        } else {                                                                                 \
            SAGE_ATTN_OUT(HD, MaskMode::kNone);                                                  \
        }                                                                                        \
    } while (0)

    switch (head_dim) {
        case 64: SAGE_ATTN_MASK(64); break;
        case 128: SAGE_ATTN_MASK(128); break;
        case 256: SAGE_ATTN_MASK(256); break;
        default:
            throw std::runtime_error("sage int8 attention: unsupported head_dim " +
                                     std::to_string(head_dim));
    }

#undef SAGE_ATTN_MASK
#undef SAGE_ATTN_OUT
#undef SAGE_ATTN_CTA
#undef SAGE_ATTN_LAUNCH
}
