// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// gfx1010's fp16 FMA attention kernel over the packed INT8 layouts, dense or
// block-sparse. See int8_attn.h for the entry point.
#include "int8_attn.h"

namespace comfy::hip_backend::sage::int8_attn {
namespace {

#if defined(__gfx1010__)
#define COMFY_SAGE_FP16_SIMT 1
#endif

// gfx1010 has no matrix unit and no dot4, so the MMA policy above costs ~1.5
// instructions per MAC plus a ds_bpermute per accumulator element. This kernel
// takes the thread-level route the SIMT GEMM uses instead: K and V are staged in
// LDS as f16 and every MAC is one v_fma_mix_f32 (f16 x f16 -> f32 accumulate).
// That is exact here: int8 values are exact in f16, their products fit 22 bits,
// and every integer partial sum stays below 2^24.
//
// Four lanes own two queries (16 per wave), each lane keeping a quarter of both
// queries' Q (every fourth 8-wide chunk) and O (every fourth dim), so every K and V
// word a lane reads from LDS feeds two queries. A lane pair per query re-read K and V
// for each one and was LDS-bound: on gfx1010 at H3's shape (56 heads, D128, S 4K-16K)
// this layout took 2.2-3.0 TFLOPS to 5.4-5.6. The four lanes' partial scores are
// reduce-scattered with two DPP steps, so each lane owns four keys of a 16-key step and
// runs the softmax (and its quarter-rate exponentials) for those alone; the packed
// probabilities are broadcast back for PV. P stays f16 instead of the U8 the matrix
// path needs.
//
// PK=true (used for D=128) swaps the fma_mix chains for v_pk_fma_f16, ~1.8x the MAC
// rate on gfx1010, at the cost of inexact f16 accumulation. Q is pre-scaled by 1/16
// (exact) so a 16-pair QK chain cannot overflow (|partial| <= 16 * 8 * 127 = 16256), two
// lanes' chains are added in f16 before the flush to f32, PV chains are flushed every
// 16 keys, and P is shifted by 2^-4 (it cancels in
// O / d) to keep the PV chains below 65504. Measured end-to-end error stays within
// ~3% of the exact path, far below the int8 quantization error.
#if defined(COMFY_SAGE_FP16_SIMT)
using half2_t = _Float16 __attribute__((ext_vector_type(2)));
struct __attribute__((aligned(16))) Half8 { half2_t v[4]; };

// Four int8 to fp16 with two byte permutes and two packed subtracts, no conversions.
// fp16 0x6400 is 1024, where the ulp is exactly 1, so writing a byte u into the low
// mantissa bits (0x64uu) gives the fp16 value 1024 + u. Flipping the sign bit first
// turns int8 b into u = b + 128, and subtracting 1152 = 1024 + 128 leaves b exactly.
__forceinline__ __device__ void i8x4_to_half2(uint32_t w, half2_t& lo, half2_t& hi) {
    // (u8 + 128) planted in the mantissa of 1024 is exactly 1024 + u8 + 128 - 128.
    w ^= 0x80808080u;
    const half2_t bias = {static_cast<_Float16>(1152.0f), static_cast<_Float16>(1152.0f)};
    lo = __builtin_bit_cast(half2_t, __byte_perm(w, 0x64646464u, 0x5140u)) - bias;
    hi = __builtin_bit_cast(half2_t, __byte_perm(w, 0x64646464u, 0x5342u)) - bias;
}

__forceinline__ __device__ void i8x16_to_half8x2(const uint4& raw, Half8& a, Half8& b) {
    i8x4_to_half2(raw.x, a.v[0], a.v[1]);
    i8x4_to_half2(raw.y, a.v[2], a.v[3]);
    i8x4_to_half2(raw.z, b.v[0], b.v[1]);
    i8x4_to_half2(raw.w, b.v[2], b.v[3]);
}

// DPP quad_perm controls: lane l takes lane l ^ 1, lane l ^ 2, or quad lane n.
constexpr int kSwapPair = 0xB1;
constexpr int kSwapPairOfPairs = 0x4E;
constexpr int kQuadLane0 = 0x00, kQuadLane1 = 0x55, kQuadLane2 = 0xAA, kQuadLane3 = 0xFF;

template <int kCtrl>
__forceinline__ __device__ uint32_t dpp_u32(uint32_t v) {
    return static_cast<uint32_t>(__builtin_amdgcn_mov_dpp(static_cast<int>(v), kCtrl, 0xF, 0xF, false));
}

__forceinline__ __device__ float swap_pair(float v) {
    return __uint_as_float(dpp_u32<kSwapPair>(__float_as_uint(v)));
}

__forceinline__ __device__ float swap_pair_of_pairs(float v) {
    return __uint_as_float(dpp_u32<kSwapPairOfPairs>(__float_as_uint(v)));
}

__forceinline__ __device__ float fma_h(_Float16 a, _Float16 b, float c) {
    return __builtin_fmaf(static_cast<float>(a), static_cast<float>(b), c);
}
#endif

// kWaves waves of 16 queries make a block. With a lut ([B, H_q, query blocks, topk]
// key-tile indices, as block-sparse attention selects them), a block attends to only
// those 64-key tiles of K, in the order given; without one, to all of K.
template <int HD, bool PK, int kWaves, typename OutT>
__global__ __launch_bounds__(kWaves * 32) void int8_attn_fp16_kernel(
    const int8_t* __restrict__ q, const int8_t* __restrict__ k, const int8_t* __restrict__ v,
    OutT* __restrict__ o, const float* __restrict__ q_scale, const float* __restrict__ k_scale,
    const float* __restrict__ v_scale, const int* __restrict__ lut, int topk, int qo_len,
    int kv_len, int qo_len_padded, int k_groups_per_head, int num_kv_groups, int64_t q_stride_b,
    int64_t q_stride_h, int64_t k_stride_b, int64_t k_stride_h, int64_t v_stride_b,
    int64_t v_stride_h, int64_t v_stride_d, int64_t o_stride_b, int64_t o_stride_h,
    int64_t o_stride_n, float sm_scale) {
#if defined(COMFY_SAGE_FP16_SIMT)
    constexpr int kL = 4;                    // lanes per query group
    constexpr int kR = 2;                    // queries per query group
    static_assert(32 / kL * kR == 16, "a wave holds 16 queries");
    static_assert(!PK || HD / 8 / kL <= 4, "one f16 QK chain per key (16 pairs)");
    constexpr int kThreads = kWaves * 32;
    constexpr int kBlockQ = kWaves * 16;
    constexpr int CTA_K = 64;
    constexpr int kSub = 16;                 // keys per softmax step = one k_scale group
    constexpr int kChunks = HD / 8 / kL;     // 8-wide Q/K chunks a lane owns: kL * c + part
    constexpr int kDims = HD / kL;           // O dims a lane owns: kL * t + part
    constexpr int kKStride = HD;             // halves per K row
    __shared__ __attribute__((aligned(16))) _Float16 smem_k[CTA_K * kKStride];
    // V rows are 128 B (64 keys) with their 16 B granules xor-swizzled by the row, so the
    // kL adjacent rows a group reads land in different banks. Padding the rows instead
    // made the tiles 34 KB, which fits only three 4-wave blocks in a WGP's 128 KB.
    __shared__ __attribute__((aligned(16))) _Float16 smem_v[HD * CTA_K];
    auto v_granule = [&](int row, int g) -> Half8* {
        return reinterpret_cast<Half8*>(smem_v + row * CTA_K + (g ^ (row & 7)) * 8);
    };

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wave = tid >> 5;
    const int part = lane % kL;
    const int group = 2 * (part & 1) + (part >> 1);  // key quarter this lane's softmax owns
    const int batch = blockIdx.z;
    const int head = blockIdx.y;
    const int num_qo_heads = gridDim.y;
    const int kv_head = head / num_kv_groups;
    const int q0 = blockIdx.x * kBlockQ + wave * 16 + (lane / kL) * kR;

    const int8_t* q_head = q + static_cast<int64_t>(batch) * q_stride_b +
                           static_cast<int64_t>(head) * q_stride_h;
    const int8_t* k_head = k + static_cast<int64_t>(batch) * k_stride_b +
                           static_cast<int64_t>(kv_head) * k_stride_h;
    const int8_t* v_head = v + static_cast<int64_t>(batch) * v_stride_b +
                           static_cast<int64_t>(kv_head) * v_stride_h;

    // The softmax runs in base 2 (exp2 is the hardware's native exponential), so the
    // scores' scale absorbs log2(e).
    sm_scale *= kLog2e;

    half2_t qh[kR][kChunks * 4];
    float q_lane_scale[kR];
#pragma unroll
    for (int r = 0; r < kR; ++r) {
        const int8_t* q_row = q_head + static_cast<int64_t>(imin(q0 + r, qo_len - 1)) * HD;
#pragma unroll
        for (int c = 0; c < kChunks; ++c) {
            const uint2 raw = *reinterpret_cast<const uint2*>(q_row + (kL * c + part) * 8);
            i8x4_to_half2(raw.x, qh[r][4 * c], qh[r][4 * c + 1]);
            i8x4_to_half2(raw.y, qh[r][4 * c + 2], qh[r][4 * c + 3]);
        }
        if constexpr (PK) {
            const half2_t sixteenth = {static_cast<_Float16>(0.0625f), static_cast<_Float16>(0.0625f)};
#pragma unroll
            for (int i = 0; i < kChunks * 4; ++i) qh[r][i] = qh[r][i] * sixteenth;
        }
        // The scores are integer dot products of the int8 codes; this scale (Q's
        // dequant scale, the softmax scale, and the 16 the PK path divided Q by) is
        // applied once per score, in the softmax, rather than to Q or K.
        q_lane_scale[r] = (PK ? 16.0f : 1.0f) * sm_scale *
            q_scale[(static_cast<int64_t>(batch) * num_qo_heads + head) * qo_len_padded +
                    imin(q0 + r, qo_len_padded - 1)];
    }

    float o_val[kR][kDims];
    float m_run[kR];
    float d_run[kR];
#pragma unroll
    for (int r = 0; r < kR; ++r) {
#pragma unroll
        for (int t = 0; t < kDims; ++t) o_val[r][t] = 0.0f;
        m_run[r] = kMaskedScore;
        d_run[r] = 0.0f;
    }

    const int num_iters = (kv_len + CTA_K - 1) / CTA_K;
    const float* k_scale_head =
        k_scale + (static_cast<int64_t>(batch) * (num_qo_heads / num_kv_groups) + kv_head) *
                      k_groups_per_head;

    auto run_iter = [&](auto boundary_tag, int iter) {
        constexpr bool kBoundary = decltype(boundary_tag)::value;
        const int kb0 = iter * CTA_K;

        __syncthreads();
#pragma unroll
        for (int idx = tid; idx < CTA_K * (HD / 16); idx += kThreads) {
            const int row = idx / (HD / 16);
            const int unit = idx % (HD / 16);
            const int key = kb0 + row;
            uint4 raw = {0u, 0u, 0u, 0u};
            if (!kBoundary || key < kv_len) {
                raw = *reinterpret_cast<const uint4*>(k_head + static_cast<int64_t>(key) * HD +
                                                     unit * 16);
            }
            Half8 a, b;
            i8x16_to_half8x2(raw, a, b);
            Half8* dst = reinterpret_cast<Half8*>(smem_k + row * kKStride + unit * 16);
            dst[0] = a;
            dst[1] = b;
        }
        // V arrives transposed ([dim][key]) and zero padded to a multiple of CTA_K by the
        // quantizer, so this tile needs no bounds check and a lane's PV reads 16
        // consecutive keys of one dim as two 16-byte loads.
#pragma unroll
        for (int idx = tid; idx < HD * (CTA_K / 16); idx += kThreads) {
            const int row = idx / (CTA_K / 16);
            const int unit = idx % (CTA_K / 16);
            const uint4 raw = *reinterpret_cast<const uint4*>(
                v_head + static_cast<int64_t>(row) * v_stride_d + kb0 + unit * 16);
            Half8 a, b;
            i8x16_to_half8x2(raw, a, b);
            *v_granule(row, 2 * unit) = a;
            *v_granule(row, 2 * unit + 1) = b;
        }
        __syncthreads();

#pragma unroll 1
        for (int sub = 0; sub < CTA_K / kSub; ++sub) {
            const int key0 = kb0 + sub * kSub;
            if (kBoundary && key0 >= kv_len) break;

            // Slot g4 * 4 + kk of a lane holds key (g4 ^ group) * 4 + kk, so each step of
            // the reduce-scatter below adds a register to the same register of a partner:
            // the xor-1 lane (group ^ 2) keeps the other half of the keys, the xor-2 lane
            // (group ^ 1) the other quarter. A lane is left with the full scores of keys
            // group * 4 + 0..3 of both queries and runs the softmax for those alone.
            // Groups g4 and g4 + 2 meet in the first step, so they are computed in pairs and
            // reduced at once: only one pair's partial scores is ever live. That step adds
            // the partner's half in f16 when the chains are f16 (each at most 16256, see the
            // PK note), so only the kept half is flushed to f32.
            float half_sum[kR][kSub / 2];
#pragma unroll
            for (int lo = 0; lo < 2; ++lo) {
                [[maybe_unused]] float acc[2][kR][4];
                [[maybe_unused]] half2_t pacc[2][kR][4];
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int key_row = sub * kSub + ((lo + 2 * h) ^ group) * 4;
#pragma unroll
                    for (int r = 0; r < kR; ++r)
#pragma unroll
                        for (int kk = 0; kk < 4; ++kk) {
                            acc[h][r][kk] = 0.0f;
                            pacc[h][r][kk] = half2_t{0, 0};
                        }
#pragma unroll
                    for (int c = 0; c < kChunks; ++c) {
#pragma unroll
                        for (int kk = 0; kk < 4; ++kk) {
                            const Half8 kv = *reinterpret_cast<const Half8*>(
                                smem_k + (key_row + kk) * kKStride + (kL * c + part) * 8);
#pragma unroll
                            for (int r = 0; r < kR; ++r) {
                                if constexpr (PK) {
#pragma unroll
                                    for (int e = 0; e < 4; ++e)
                                        pacc[h][r][kk] = __builtin_elementwise_fma(qh[r][4 * c + e], kv.v[e], pacc[h][r][kk]);
                                } else {
#pragma unroll
                                    for (int e = 0; e < 4; ++e) {
                                        acc[h][r][kk] = fma_h(qh[r][4 * c + e].x, kv.v[e].x, acc[h][r][kk]);
                                        acc[h][r][kk] = fma_h(qh[r][4 * c + e].y, kv.v[e].y, acc[h][r][kk]);
                                    }
                                }
                            }
                        }
                    }
                }
#pragma unroll
                for (int r = 0; r < kR; ++r)
#pragma unroll
                    for (int kk = 0; kk < 4; ++kk) {
                        // fma_h(x, 1, y) is one v_fma_mix_f32: both f16 halves are widened
                        // and added in a single instruction.
                        if constexpr (PK) {
                            const half2_t sum = pacc[0][r][kk] +
                                __builtin_bit_cast(half2_t, dpp_u32<kSwapPair>(__builtin_bit_cast(uint32_t, pacc[1][r][kk])));
                            half_sum[r][lo * 4 + kk] = fma_h(sum.x, static_cast<_Float16>(1.0f), static_cast<float>(sum.y));
                        } else {
                            half_sum[r][lo * 4 + kk] = acc[0][r][kk] + swap_pair(acc[1][r][kk]);
                        }
                    }
            }
            float s[kR][4];
#pragma unroll
            for (int r = 0; r < kR; ++r)
#pragma unroll
                for (int j = 0; j < 4; ++j) s[r][j] = half_sum[r][j] + swap_pair_of_pairs(half_sum[r][j + 4]);

            // Keep the scheduler from interleaving QK, softmax and PV: hoisting across them
            // spilled more registers (at the tile staging) and measured ~1% slower.
            __builtin_amdgcn_sched_barrier(0);
            const float k_group_scale = k_scale_head[iter * (CTA_K / kSub) + sub];
            const int key_base = key0 + group * 4;
            uint32_t own[kR][2];
#pragma unroll
            for (int r = 0; r < kR; ++r) {
                const float scale = q_lane_scale[r] * k_group_scale;
                float mx = -1.0e30f;
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    mx = fmaxf(mx, (kBoundary && key_base + kk >= kv_len) ? -1.0e30f : s[r][kk]);
                }
                mx = fmaxf(mx, swap_pair(mx));
                mx = fmaxf(mx, swap_pair_of_pairs(mx));
                const float tile_m = mx * scale;

                // Lazy rescaling (in log2 units). The running max m_run only moves when a
                // tile's max passes it by more than 8, and then to 4 below that max, so
                // most steps skip rescaling O. Probabilities may then exceed 1, but by at
                // most 2^8, which fp16 holds; with PK the extra -4 below caps them at
                // 2^4, so a 16-key PV chain stays within 16 * 16 * 128 = 32768. __any keeps
                // the branch wave-uniform: lanes that did not need it rescale by alpha = 1.
                // O and d are rescaled together, so the final O / d is unchanged.
                const bool update = tile_m > m_run[r] + 8.0f;
                if (__any(update)) {
                    const float m_next = update ? tile_m - 4.0f : m_run[r];
                    const float alpha = softmax_exp2(m_run[r] - m_next);
                    d_run[r] *= alpha;
#pragma unroll
                    for (int t = 0; t < kDims; ++t) o_val[r][t] *= alpha;
                    m_run[r] = m_next;
                }

                const float negative_m = -m_run[r] - (PK ? 4.0f : 0.0f);
                float p[4];
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    p[kk] = softmax_exp2(fmaf(s[r][kk], scale, negative_m));
                    if (kBoundary) p[kk] = key_base + kk < kv_len ? p[kk] : 0.0f;
                }
                d_run[r] += (p[0] + p[1]) + (p[2] + p[3]);  // this lane's keys; summed at the end
                // Round toward zero to f16 for PV: p <= 2^8, so no overflow, and the f16
                // rounding (2^-11 relative) is far below the int8 quantization of V.
                own[r][0] = __builtin_bit_cast(uint32_t, __builtin_amdgcn_cvt_pkrtz(p[0], p[1]));
                own[r][1] = __builtin_bit_cast(uint32_t, __builtin_amdgcn_cvt_pkrtz(p[2], p[3]));
            }

            // Every lane needs all 16 probabilities for its dims of PV: key quarter q comes
            // from the quad lane whose group is q (lanes 0, 2, 1, 3).
            half2_t ph[kR][kSub / 2];
#pragma unroll
            for (int r = 0; r < kR; ++r) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    ph[r][0 + h] = __builtin_bit_cast(half2_t, dpp_u32<kQuadLane0>(own[r][h]));
                    ph[r][2 + h] = __builtin_bit_cast(half2_t, dpp_u32<kQuadLane2>(own[r][h]));
                    ph[r][4 + h] = __builtin_bit_cast(half2_t, dpp_u32<kQuadLane1>(own[r][h]));
                    ph[r][6 + h] = __builtin_bit_cast(half2_t, dpp_u32<kQuadLane3>(own[r][h]));
                }
            }
            __builtin_amdgcn_sched_barrier(0);

#pragma unroll
            for (int t = 0; t < kDims; ++t) {
                const int row = kL * t + part;
                const Half8 v0 = *v_granule(row, 2 * sub);
                const Half8 v1 = *v_granule(row, 2 * sub + 1);
#pragma unroll
                for (int r = 0; r < kR; ++r) {
                    float a = o_val[r][t];
                    if constexpr (PK) {
                        half2_t pa = half2_t{0, 0};
#pragma unroll
                        for (int e = 0; e < 4; ++e) pa = __builtin_elementwise_fma(ph[r][e], v0.v[e], pa);
#pragma unroll
                        for (int e = 0; e < 4; ++e) pa = __builtin_elementwise_fma(ph[r][4 + e], v1.v[e], pa);
                        a = fma_h(pa.x, static_cast<_Float16>(1.0f), a);
                        a = fma_h(pa.y, static_cast<_Float16>(1.0f), a);
                    } else {
#pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            a = fma_h(ph[r][e].x, v0.v[e].x, a);
                            a = fma_h(ph[r][e].y, v0.v[e].y, a);
                        }
#pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            a = fma_h(ph[r][4 + e].x, v1.v[e].x, a);
                            a = fma_h(ph[r][4 + e].y, v1.v[e].y, a);
                        }
                    }
                    o_val[r][t] = a;
                }
            }
        }
    };

    const int interior_iters = kv_len / CTA_K;
    if (lut != nullptr) {
        const int* tiles =
            lut + ((static_cast<int64_t>(batch) * num_qo_heads + head) * gridDim.x + blockIdx.x) * topk;
#pragma unroll 1
        for (int t = 0; t < topk; ++t) {
            const int tile = tiles[t];
            // The same for the whole block, which run_iter's __syncthreads requires. A
            // tile outside K would read past V.
            if (tile < 0 || tile >= num_iters) continue;
            if (tile < interior_iters) {
                run_iter(std::false_type{}, tile);
            } else {
                run_iter(std::true_type{}, tile);
            }
        }
    } else {
        int iter = 0;
        for (; iter < interior_iters; ++iter) run_iter(std::false_type{}, iter);
        for (; iter < num_iters; ++iter) run_iter(std::true_type{}, iter);
    }

    // Each lane summed only its own keys' probabilities; every lane is still active here.
#pragma unroll
    for (int r = 0; r < kR; ++r) {
        d_run[r] += swap_pair(d_run[r]);
        d_run[r] += swap_pair_of_pairs(d_run[r]);
    }

    const float* v_scale_head =
        v_scale + (static_cast<int64_t>(batch) * (num_qo_heads / num_kv_groups) + kv_head) * HD;
#pragma unroll
    for (int r = 0; r < kR; ++r) {
        if (q0 + r >= qo_len) break;
        // d is 0 only when no key was attended (such as a block-sparse row whose tiles
        // were all skipped); its output is 0 rather than NaN.
        const float inv_d = d_run[r] > 0.0f ? 1.0f / d_run[r] : 0.0f;
        OutT* row = o + static_cast<int64_t>(batch) * o_stride_b +
                    static_cast<int64_t>(head) * o_stride_h +
                    static_cast<int64_t>(q0 + r) * o_stride_n;
#pragma unroll
        for (int t = 0; t < kDims; ++t) {
            const int d = kL * t + part;
            row[d] = static_cast<OutT>(o_val[r][t] * v_scale_head[d] * inv_d);
        }
    }
#endif
}

template <int HD, bool PK, int kWaves, typename OutT>
void launch_fp16(const Int8AttnLaunch& a, const int* lut, int topk) {
    const dim3 grid(div_ceil_host(a.qo_len, kWaves * 16), a.num_qo_heads, a.batch);
    int8_attn_fp16_kernel<HD, PK, kWaves, OutT><<<grid, kWaves * 32, 0, a.stream>>>(
        a.q, a.k, a.v, static_cast<OutT*>(a.o), a.q_scale, a.k_scale, a.v_scale, lut, topk,
        a.qo_len, a.kv_len, a.qo_len_padded, a.k_groups_per_head, a.num_kv_groups, a.q_stride_b,
        a.q_stride_h, a.k_stride_b, a.k_stride_h, a.v_stride_b, a.v_stride_h, a.v_stride_d,
        a.o_stride_b, a.o_stride_h, a.o_stride_n, a.sm_scale);
}

template <int HD, bool PK, int kWaves>
void launch_fp16_out(const Int8AttnLaunch& a, int output_dtype_code, const int* lut, int topk) {
    // Output dtype codes: 0 float32 (block-sparse only), 1 float16, 2 bfloat16.
    if (output_dtype_code == 0) {
        launch_fp16<HD, PK, kWaves, float>(a, lut, topk);
    } else if (output_dtype_code == 1) {
        launch_fp16<HD, PK, kWaves, __half>(a, lut, topk);
    } else {
        launch_fp16<HD, PK, kWaves, __bf16>(a, lut, topk);
    }
}

}  // namespace

void launch_int8_attn_fp16(const Int8AttnLaunch& a, int head_dim, int output_dtype_code,
                           const int* lut, int topk, int block_q) {
    // 16 queries a wave, so block_q fixes the waves per block.
    if (head_dim == 64) {
        block_q == 64 ? launch_fp16_out<64, false, 4>(a, output_dtype_code, lut, topk)
                      : launch_fp16_out<64, false, 8>(a, output_dtype_code, lut, topk);
    } else {
        block_q == 64 ? launch_fp16_out<128, true, 4>(a, output_dtype_code, lut, topk)
                      : launch_fp16_out<128, true, 8>(a, output_dtype_code, lut, topk);
    }
}

}  // namespace comfy::hip_backend::sage::int8_attn
