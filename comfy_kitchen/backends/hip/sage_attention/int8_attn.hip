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

#include <cfloat>
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

// vals[e] belongs to channel d_base + acc_row(lane, e). gfx12 makes those eight
// channels contiguous, so the row goes out in one 16-byte store; gfx11
// interleaves them with the partner lane's and needs eight.
template <typename OutT>
__forceinline__ __device__ void store_o_tile(OutT* __restrict__ row, int d_base,
                                             const float* vals, int lane) {
#if defined(COMFY_MMA_GFX12)
    // Aligned because the 16-byte store below reinterprets it.
    __attribute__((aligned(16))) OutT packed[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) packed[e] = static_cast<OutT>(vals[e]);
    *reinterpret_cast<uint4*>(row + d_base + 8 * (lane / 16)) =
        *reinterpret_cast<const uint4*>(packed);
#else
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        row[d_base + acc_row(lane, e)] = static_cast<OutT>(vals[e]);
    }
#endif
}

// ---------------------------------------------------------------------------
// gfx12 long-sequence specialization
//
// Keep the accepted schedule structural: BM128/BN64, eight wave32s, padded
// single-stage Q/K/V LDS, synchronous global->VGPR->LDS prefetch, native
// packed P, and FP32 output accumulation. The general kernel below retains its
// original contract.
// ---------------------------------------------------------------------------

constexpr int kLongHeadDim = 128;
constexpr int kLongBlockM = 128;
constexpr int kLongBlockN = 64;
constexpr int kLongWarps = 8;
constexpr int kLongThreads = kLongWarps * 32;
constexpr int kLongKStride = kLongHeadDim + 8;
constexpr int kLongVStride = kLongBlockN + 8;

__forceinline__ __device__ float long_fast_exp2(float x) {
    return __builtin_amdgcn_exp2f(x);
}

__forceinline__ __device__ float long_balanced_sum8(v8f value) {
    const float sum01 = value[0] + value[1];
    const float sum23 = value[2] + value[3];
    const float sum45 = value[4] + value[5];
    const float sum67 = value[6] + value[7];
    return (sum01 + sum23) + (sum45 + sum67);
}

__forceinline__ __device__ int long_fresh_lane_id() {
    int lane;
    // Rematerialize this after the long loop.  Carrying the prologue lane SSA
    // value through the prefetched K/V state makes LLVM spill epilogue indices.
    asm volatile("v_mbcnt_lo_u32_b32 %0, -1, 0" : "=v"(lane));
    return lane;
}

__forceinline__ __device__ void long_store_b128_padded(int8_t* dst, uint4 value) {
    // Eight-byte padding spreads consecutive WMMA fragment rows over banks.
    *reinterpret_cast<uint2*>(dst) = make_uint2(value.x, value.y);
    *reinterpret_cast<uint2*>(dst + 8) = make_uint2(value.z, value.w);
}

// The optimized packed contract pads K and transposed V to a whole BN64 tile.
// Exactly 256 threads own the 512 sixteen-byte chunks in each operand, so the
// bounds and loop trip count are compile-time facts.
struct LongKVTileRegs {
    uint4 k[2];
    uint4 v[2];

    __forceinline__ __device__ void load(const int8_t* __restrict__ k_head,
                                         const int8_t* __restrict__ v_head,
                                         int tile) {
        const int tid = threadIdx.x;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int chunk = tid + i * kLongThreads;
            const int krow = chunk >> 3;
            const int kd16 = (chunk & 7) << 4;
            k[i] = *reinterpret_cast<const uint4*>(
                k_head + static_cast<int64_t>(tile * kLongBlockN + krow) * kLongHeadDim + kd16);

            const int vd = chunk >> 2;
            const int vn16 = (chunk & 3) << 4;
            v[i] = *reinterpret_cast<const uint4*>(
                v_head + static_cast<int64_t>(tile) * kLongHeadDim * kLongBlockN +
                vd * kLongBlockN + vn16);
        }
    }

    __forceinline__ __device__ void store(int8_t* ks, int8_t* vs) const {
        const int tid = threadIdx.x;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int chunk = tid + i * kLongThreads;
            const int krow = chunk >> 3;
            const int kd16 = (chunk & 7) << 4;
            long_store_b128_padded(ks + krow * kLongKStride + kd16, k[i]);

            const int vd = chunk >> 2;
            const int vn16 = (chunk & 3) << 4;
            long_store_b128_padded(vs + vd * kLongVStride + vn16, v[i]);
        }
    }
};

template <bool NoQueryTail>
__forceinline__ __device__ void long_stage_q_tile(
    const int8_t* __restrict__ q_head, int8_t* __restrict__ qs,
    int query_block0, int qo_len) {
    const int tid = threadIdx.x;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int chunk = tid + i * kLongThreads;
        const int row = chunk >> 3;
        const int d16 = (chunk & 7) << 4;
        uint4 value = make_uint4(0, 0, 0, 0);
        if constexpr (NoQueryTail) {
            value = *reinterpret_cast<const uint4*>(
                q_head + static_cast<int64_t>(query_block0 + row) * kLongHeadDim + d16);
        } else if (query_block0 + row < qo_len) {
            value = *reinterpret_cast<const uint4*>(
                q_head + static_cast<int64_t>(query_block0 + row) * kLongHeadDim + d16);
        }
        long_store_b128_padded(qs + row * kLongKStride + d16, value);
    }
}

template <typename OutT, bool NoQueryTail, bool FullScoreScale>
__global__ __launch_bounds__(kLongThreads)
__attribute__((amdgpu_waves_per_eu(1, 1)))
void gfx12_int8_attention_kernel(
    const int8_t* __restrict__ q,
    const int8_t* __restrict__ k,
    const int8_t* __restrict__ v,
    OutT* __restrict__ output,
    const float* __restrict__ q_descale,
    const float* __restrict__ k_descale,
    const float* __restrict__ v_descale,
    int qo_len, int kv_len, int padded_q, int padded_k,
    int k_groups_per_head, int query_block_offset,
    int64_t output_stride_h, int64_t output_stride_n, float sm_scale) {

    __shared__ __align__(16) int8_t ks[2][kLongBlockN * kLongKStride];
    __shared__ __align__(16) int8_t vs[2][kLongHeadDim * kLongVStride];
    __shared__ __align__(16) int8_t qs[kLongBlockM * kLongKStride];

    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x >> 5;
    const int head = blockIdx.y;
    const int query_block = blockIdx.x + query_block_offset;
    const int query0 = query_block * kLongBlockM + wave * 16;
    const bool active_query_wave = NoQueryTail || query0 < qo_len;
    const int tiles = padded_k / kLongBlockN;
    const int8_t* const q_head =
        q + static_cast<int64_t>(head) * qo_len * kLongHeadDim;
    const int8_t* const k_head =
        k + static_cast<int64_t>(head) * padded_k * kLongHeadDim;
    const int8_t* const v_head =
        v + static_cast<int64_t>(head) * kLongHeadDim * padded_k;

    const int query_lane = query0 + acc_col(lane);
    const float q_scale = active_query_wave && query_lane < qo_len
        ? q_descale[head * padded_q + query_lane]
        : 0.0f;
    const float qk_scale = sm_scale * kLog2e;
    // Preserve the finite online-softmax state; this specialization changes
    // schedule and data movement, not the probability quantization algorithm.
    float row_max = kMaskedScore;
    float row_sum = 1.0f;
    v8f out_acc[8];
#pragma unroll
    for (int dt = 0; dt < 8; ++dt) {
        out_acc[dt] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
    }

    LongKVTileRegs next;
    int kv_buf = 0;
    long_stage_q_tile<NoQueryTail>(q_head, qs, query_block * kLongBlockM, qo_len);
    next.load(k_head, v_head, 0);
    next.store(ks[kv_buf], vs[kv_buf]);
    __syncthreads();

    for (int tile = 0; tile < tiles; ++tile) {
        const bool has_next = tile + 1 < tiles;
        if (has_next) {
            next.load(k_head, v_head, tile + 1);
        }

        // Tail workgroups still need every wave for cooperative K/V staging
        // and barriers.  Waves with no query rows can skip the matrix and
        // softmax work between those barriers.
        if (active_query_wave) {
            float key_scale[4];
#pragma unroll
            for (int ct = 0; ct < 4; ++ct) {
                key_scale[ct] =
                    k_descale[head * k_groups_per_head + tile * 4 + ct];
            }
            float score_scale[4];
#pragma unroll
            for (int ct = 0; ct < 4; ++ct) {
                // The full-scale variant hoists the query factor out of the
                // score loop while preserving the quality-gated association.
                if constexpr (FullScoreScale) {
                    score_scale[ct] = (q_scale * key_scale[ct]) * qk_scale;
                } else {
                    score_scale[ct] = key_scale[ct] * qk_scale;
                }
            }

            v8f score[4];
            {
                v8i score_i32[4];
#pragma unroll
                for (int ct = 0; ct < 4; ++ct) score_i32[ct] = MmaInt8::zero();
#pragma unroll
                for (int dt = 0; dt < 8; ++dt) {
                    const MmaInt8::Frag q_frag = MmaInt8::load(
                        qs, wave * 16 + frag_row(lane), dt * 16, kLongKStride, lane);
#pragma unroll
                    for (int ct = 0; ct < 4; ++ct) {
                        const MmaInt8::Frag k_frag = MmaInt8::load(
                            ks[kv_buf], ct * 16 + frag_row(lane), dt * 16,
                            kLongKStride, lane);
                        // K x Q^T leaves each score fragment in the native A
                        // operand order consumed by P x V below.
                        score_i32[ct] = MmaInt8::mma(k_frag, q_frag, score_i32[ct]);
                    }
                }
#pragma unroll
                for (int ct = 0; ct < 4; ++ct) {
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        float value;
                        if constexpr (FullScoreScale) {
                            value = static_cast<float>(score_i32[ct][e]) * score_scale[ct];
                        } else {
                            value = (static_cast<float>(score_i32[ct][e]) * q_scale) *
                                    score_scale[ct];
                        }
                        const int key = tile * kLongBlockN + ct * 16 + acc_row(lane, e);
                        if (key >= kv_len) value = kMaskedScore;
                        score[ct][e] = value;
                    }
                }
            }

            float local_max = -FLT_MAX;
#pragma unroll
            for (int ct = 0; ct < 4; ++ct) {
#pragma unroll
                for (int e = 0; e < 8; ++e) local_max = fmaxf(local_max, score[ct][e]);
            }
            float tile_max = fmaxf(local_max, __shfl_xor(local_max, 16, 32));

            // Shifting the tile maximum fills u8 [0, 255]; tile_scale carries
            // its gap to the running maximum on the accumulator side.
            tile_max -= kProbU8Offset;
            const float previous_max = row_max;
            const float smaller = long_fast_exp2(-fabsf(previous_max - tile_max));
            const float output_scale = previous_max < tile_max ? smaller : 1.0f;
            const float tile_scale = tile_max < previous_max ? smaller : 1.0f;
            row_max = fmaxf(previous_max, tile_max);
            row_sum *= output_scale;
#pragma unroll
            for (int dt = 0; dt < 8; ++dt) {
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    out_acc[dt][e] *=
                        __shfl(output_scale, acc_row(lane, e), 32);
                }
            }

            MmaInt8::Frag direct_p_frag[4];
#pragma unroll
            for (int ct = 0; ct < 4; ++ct) {
                uint32_t packed[8];
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const float probability = long_fast_exp2(score[ct][e] - tile_max);
                    score[ct][e] = probability;
                    packed[e] = prob_to_u8(probability);
                }
                direct_p_frag[ct] = pack_prob_frag(packed, lane);
            }

            // Each half-wave carries its part of the denominator through the
            // loop.  A balanced native-fragment tree shortens the dependency
            // chain while keeping the quality-gated cross-fragment ordering.
            const float tile_sum =
                (long_balanced_sum8(score[0]) + long_balanced_sum8(score[2])) +
                (long_balanced_sum8(score[1]) + long_balanced_sum8(score[3]));
            row_sum += tile_sum * tile_scale;

#pragma unroll
            for (int dt = 0; dt < 8; ++dt) {
                v8i product = MmaInt8::zero();
#pragma unroll
                for (int ct = 0; ct < 4; ++ct) {
                    const MmaInt8::Frag v_frag = MmaInt8::load(
                        vs[kv_buf], dt * 16 + frag_row(lane), ct * 16,
                        kLongVStride, lane);
                    product = MmaInt8::mma_ua(direct_p_frag[ct], v_frag, product);
                }
                const int d = dt * 16 + acc_col(lane);
                const float value_scale =
                    v_descale[(head * tiles + tile) * kLongHeadDim + d];
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const float row_tile_scale =
                        __shfl(tile_scale, acc_row(lane, e), 32);
                    out_acc[dt][e] = fmaf(
                        static_cast<float>(product[e]),
                        value_scale * row_tile_scale, out_acc[dt][e]);
                }
            }
        }

        if (has_next) {
            next.store(ks[kv_buf ^ 1], vs[kv_buf ^ 1]);
            __syncthreads();
            kv_buf ^= 1;
        }
    }

    row_sum += __shfl_xor(row_sum, 16, 32);

    if (active_query_wave) {
        const int output_lane = long_fresh_lane_id();
#pragma unroll
        for (int dt = 0; dt < 8; ++dt) {
            const int d = dt * 16 + acc_col(output_lane);
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int row = query0 + acc_row(output_lane, e);
                if constexpr (NoQueryTail) {
                    const float reciprocal = 1.0f / __shfl(
                        row_sum, acc_row(output_lane, e), 32);
                    output[static_cast<int64_t>(head) * output_stride_h +
                           static_cast<int64_t>(row) * output_stride_n + d] =
                        static_cast<OutT>(out_acc[dt][e] * reciprocal);
                } else if (row < qo_len) {
                    const float reciprocal = 1.0f / __shfl(
                        row_sum, acc_row(output_lane, e), 32);
                    output[static_cast<int64_t>(head) * output_stride_h +
                           static_cast<int64_t>(row) * output_stride_n + d] =
                        static_cast<OutT>(out_acc[dt][e] * reciprocal);
                }
            }
        }
    }
}

template <typename OutT>
void launch_gfx12_int8_attention_typed(
    const int8_t* q, const int8_t* k, const int8_t* v, OutT* output,
    const float* q_descale, const float* k_descale, const float* v_descale,
    int num_heads, int qo_len, int kv_len, int padded_q, int padded_k,
    int k_groups_per_head,
    int64_t output_stride_h, int64_t output_stride_n,
    float sm_scale, hipStream_t stream) {
    const int full_blocks = qo_len / kLongBlockM;
    if (full_blocks) {
        // Use one arithmetic association for an entirely full grid and the
        // quality-gated association shared with the masked kernel whenever a
        // query tail is present. This is a tile property, not a model shape.
        if (qo_len % kLongBlockM != 0) {
            gfx12_int8_attention_kernel<OutT, true, true>
                <<<dim3(full_blocks, num_heads), kLongThreads, 0, stream>>>(
                    q, k, v, output, q_descale, k_descale, v_descale,
                    qo_len, kv_len, padded_q, padded_k, k_groups_per_head,
                    0, output_stride_h, output_stride_n, sm_scale);
        } else {
            gfx12_int8_attention_kernel<OutT, true, false>
                <<<dim3(full_blocks, num_heads), kLongThreads, 0, stream>>>(
                    q, k, v, output, q_descale, k_descale, v_descale,
                    qo_len, kv_len, padded_q, padded_k, k_groups_per_head,
                    0, output_stride_h, output_stride_n, sm_scale);
        }
    }
    if (full_blocks * kLongBlockM != qo_len) {
        gfx12_int8_attention_kernel<OutT, false, true>
            <<<dim3(1, num_heads), kLongThreads, 0, stream>>>(
                q, k, v, output, q_descale, k_descale, v_descale,
                qo_len, kv_len, padded_q, padded_k, k_groups_per_head,
                full_blocks, output_stride_h, output_stride_n, sm_scale);
    }
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

    constexpr int kUnitsK = HD / 8;
    constexpr int kUnitsV = CTA_K / 8;

    // `boundary` is a compile-time tag, so the interior instance carries no mask
    // test and no per-key bounds check at all.
    auto run_iter = [&](auto boundary_tag, int iter) {
        constexpr bool kBoundary = decltype(boundary_tag)::value;
        const int kb0 = iter * CTA_K;

        __syncthreads();
        // K tile: [key][d]. A row past the end is zeroed; its score is masked
        // below. Staged straight to LDS: prefetching across the barrier was tried
        // and loses, because this kernel waits on LDS, not on global.
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
        // V tile: [d][key], already transposed and zero padded by the quantizer,
        // so a whole CTA_K tile is always in bounds.
        for (int idx = tid; idx < HD * kUnitsV; idx += S::kThreads) {
            const int row = idx / kUnitsV;
            const int unit = idx % kUnitsV;
            *reinterpret_cast<uint2*>(smem_v + row * kStrideV + unit * 8) =
                *reinterpret_cast<const uint2*>(v_head + static_cast<int64_t>(row) * v_stride_d +
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
            const float smaller = exp2f(-fabsf(m_prev - tile_m));
            const float o_scale = m_prev < tile_m ? smaller : 1.0f;
            const float tile_scale = tile_m < m_prev ? smaller : 1.0f;
            m_run[fq] = fmaxf(m_prev, tile_m);
            d_run[fq] *= o_scale;

#pragma unroll
            for (int dt = 0; dt < DT; ++dt) {
#pragma unroll
                for (int e = 0; e < 8; ++e) o_val[fq][dt][e] *= o_scale;
            }

            typename MmaInt8::Frag p_frag[KT];
            float tile_sum = 0.0f;
#pragma unroll
            for (int fk = 0; fk < KT; ++fk) {
                uint32_t packed[8];
                const float negative_m = -tile_m;
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const float p =
                        kBoundary
                            ? exp2f(prob[kBoundary ? fk : 0][e] + negative_m)
                            : exp2f(fmaf(static_cast<float>(s_acc[fk][e]), tile_scale_k[fk],
                                         negative_m));
                    tile_sum += p;
                    packed[e] = prob_to_u8(p);
                }
                p_frag[fk] = pack_prob_frag(packed, lane);
            }
            d_run[fq] += tile_sum * tile_scale;

            // The INT32 partial covers one CTA_K tile, which is what keeps the
            // running output in FP32 without a second accumulator set per tile.
#pragma unroll
            for (int dt = 0; dt < DT; ++dt) {
                typename MmaInt8::Acc partial = MmaInt8::zero();
#pragma unroll
                for (int fk = 0; fk < KT; ++fk) {
                    partial = MmaInt8::mma_ub(
                        MmaInt8::load_aligned(smem_v, dt * 16 + r, fk * 16, kStrideV, lane), p_frag[fk],
                        partial);
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
                vals[e] = o_val[fq][dt][e] * v_scale_head[d] * inv_d;
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

extern "C" void launch_gfx12_int8_attention(
    const void* q, const void* k, const void* v, void* o,
    const void* q_descale, const void* k_descale, const void* v_descale,
    int num_heads, int qo_len, int kv_len, int padded_q, int padded_k,
    int k_groups_per_head,
    int64_t output_stride_h, int64_t output_stride_n,
    float sm_scale, int output_dtype_code, hipStream_t stream) {
    if (num_heads <= 0 || qo_len <= 0 || kv_len <= 0) {
        throw std::runtime_error(
            "gfx12 INT8 attention requires positive heads and lengths");
    }
    if (padded_q % 128 != 0 || padded_q < qo_len ||
        padded_k % 64 != 0 || padded_k < kv_len) {
        throw std::runtime_error(
            "gfx12 INT8 attention received invalid Q or K/V padding");
    }
    if (k_groups_per_head != padded_k / 16) {
        throw std::runtime_error(
            "gfx12 INT8 attention received invalid K scale grouping");
    }
    if (reinterpret_cast<uintptr_t>(q) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(k) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(v) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(o) % 16 != 0) {
        throw std::runtime_error(
            "gfx12 INT8 attention buffers must be 16-byte aligned");
    }
    const auto* qi = static_cast<const int8_t*>(q);
    const auto* ki = static_cast<const int8_t*>(k);
    const auto* vi = static_cast<const int8_t*>(v);
    const auto* qs = static_cast<const float*>(q_descale);
    const auto* ks = static_cast<const float*>(k_descale);
    const auto* vs = static_cast<const float*>(v_descale);
    if (output_dtype_code == 1) {
        launch_gfx12_int8_attention_typed(
            qi, ki, vi, static_cast<__half*>(o), qs, ks, vs, num_heads,
            qo_len, kv_len, padded_q, padded_k, k_groups_per_head,
            output_stride_h, output_stride_n, sm_scale, stream);
    } else if (output_dtype_code == 2) {
        launch_gfx12_int8_attention_typed(
            qi, ki, vi, static_cast<__bf16*>(o), qs, ks, vs, num_heads,
            qo_len, kv_len, padded_q, padded_k, k_groups_per_head,
            output_stride_h, output_stride_n, sm_scale, stream);
    } else {
        throw std::runtime_error(
            "gfx12 INT8 attention supports float16 and bfloat16 output");
    }
}

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
