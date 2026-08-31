// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Forward-only, unmasked BF16 scaled dot-product attention for RDNA wave32
// matrix cores. The score and output products are both transposed:
//
//   S^T = K @ Q^T,  O^T = V^T @ P^T.
//
// One query column therefore stays on a lane through the online softmax, and
// the BF16 probability fragment flows directly into the second WMMA. K and a
// transposed V tile are shared by every 16-query wave in the workgroup.

#include <hip/hip_runtime.h>

#include <cfloat>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include "../mma.h"

namespace comfy::hip_backend {
namespace {

constexpr float kAttentionLog2e = 1.4426950408889634f;

inline bool use_gfx12_attention_schedule() {
    // COMFY_MMA_GFX12 is device-pass only; query the runtime arch for host dispatch.
    static const bool selected = [] {
        int device = 0;
        hipDeviceProp_t properties{};
        if (hipGetDevice(&device) == hipSuccess &&
            hipGetDeviceProperties(&properties, device) == hipSuccess &&
            (std::strcmp(properties.gcnArchName, "gfx1170") == 0 ||
             std::strncmp(properties.gcnArchName, "gfx12", 5) == 0)) {
            return true;
        }
        return false;
    }();
    return selected;
}

inline bool use_gfx1170_attention_schedule() {
    static const bool selected = [] {
        int device = 0;
        hipDeviceProp_t properties{};
        return hipGetDevice(&device) == hipSuccess &&
            hipGetDeviceProperties(&properties, device) == hipSuccess &&
            std::strcmp(properties.gcnArchName, "gfx1170") == 0;
    }();
    return selected;
}

template <int HD, int BLOCK_M, int BLOCK_N>
struct Bf16AttentionShape {
    static_assert(HD % 16 == 0);
    static_assert(BLOCK_M % 16 == 0);
    static_assert(BLOCK_N % 16 == 0);
    static constexpr int kWaves = BLOCK_M / 16;
    static constexpr int kThreads = kWaves * 32;
    static constexpr int kDimTiles = HD / 16;
    static constexpr int kKeyTiles = BLOCK_N / 16;
    static constexpr int kKStride = HD + 4;
    static constexpr int kVStride = BLOCK_N + (BLOCK_M <= 64 ? 2 : 4);
};

template <typename T>
__forceinline__ __device__ T half_wave_swap(T value) {
    return __shfl_xor(value, 16, 32);
}

__forceinline__ __device__ float sum8_balanced(const v8f& value) {
    const float sum01 = value[0] + value[1];
    const float sum23 = value[2] + value[3];
    const float sum45 = value[4] + value[5];
    const float sum67 = value[6] + value[7];
    return (sum01 + sum23) + (sum45 + sum67);
}

__forceinline__ __device__ void store_bf16x8_padded(
    __bf16* __restrict__ destination, v8bf value) {
    const uint4 words = __builtin_bit_cast(uint4, value);
    *reinterpret_cast<uint2*>(destination) = make_uint2(words.x, words.y);
    *reinterpret_cast<uint2*>(destination + 4) = make_uint2(words.z, words.w);
}

__forceinline__ __device__ void store_bf16_acc(
    __bf16* __restrict__ row, int d_base, const v8f& value, int lane) {
#if defined(COMFY_MMA_GFX12)
    __attribute__((aligned(16))) __bf16 packed[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) packed[e] = static_cast<__bf16>(value[e]);
    *reinterpret_cast<uint4*>(row + d_base + 8 * (lane / 16)) =
        *reinterpret_cast<const uint4*>(packed);
#else
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        row[d_base + acc_row(lane, e)] = static_cast<__bf16>(value[e]);
    }
#endif
}

template <int HD, int BLOCK_M, int BLOCK_N, bool DIRECT_GLOBAL,
          bool NO_KV_TAIL = false, bool STAGE_V = false>
__global__ __launch_bounds__(BLOCK_M * 2)
void bf16_attention_kernel(
    const __bf16* __restrict__ q, const __bf16* __restrict__ k,
    const __bf16* __restrict__ v, __bf16* __restrict__ output,
    int q_heads, int kv_heads, int q_len, int kv_len,
    int64_t q_stride_b, int64_t q_stride_h, int64_t q_stride_n,
    int64_t k_stride_b, int64_t k_stride_h, int64_t k_stride_n,
    int64_t v_stride_b, int64_t v_stride_h, int64_t v_stride_n,
    int64_t o_stride_b, int64_t o_stride_h, int64_t o_stride_n,
    float sm_scale) {

    using Shape = Bf16AttentionShape<HD, BLOCK_M, BLOCK_N>;
    constexpr int DT = Shape::kDimTiles;
    constexpr int KT = Shape::kKeyTiles;

    __shared__ __align__(16) __bf16 staged_k[
        DIRECT_GLOBAL ? 1 : (STAGE_V ? 1 : 2) * BLOCK_N * Shape::kKStride];
    __shared__ __align__(16) __bf16 staged_v[
        STAGE_V ? HD * Shape::kVStride : 1];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wave = tid >> 5;
    const int head = blockIdx.y;
    const int batch_index = blockIdx.z;
    const int kv_head = head / (q_heads / kv_heads);
    const int query_block = blockIdx.x * BLOCK_M;
    const int query_row = query_block + wave * 16 + frag_row(lane);
    const int query_column = query_block + wave * 16 + acc_col(lane);

    const __bf16* const q_head =
        q + static_cast<int64_t>(batch_index) * q_stride_b +
        static_cast<int64_t>(head) * q_stride_h;
    const __bf16* const k_head =
        k + static_cast<int64_t>(batch_index) * k_stride_b +
        static_cast<int64_t>(kv_head) * k_stride_h;
    const __bf16* const v_head =
        v + static_cast<int64_t>(batch_index) * v_stride_b +
        static_cast<int64_t>(kv_head) * v_stride_h;
    __bf16* const o_head =
        output + static_cast<int64_t>(batch_index) * o_stride_b +
        static_cast<int64_t>(head) * o_stride_h;

    MmaBf16::Frag q_fragment[DT];
    const int safe_query_row = query_row < q_len ? query_row : q_len - 1;
#pragma unroll
    for (int dt = 0; dt < DT; ++dt) {
        q_fragment[dt] = load_frag_16bit<MmaBf16>(
            q_head + static_cast<int64_t>(safe_query_row) * q_stride_n + dt * 16,
            lane);
    }

    MmaBf16::Acc output_acc[DT];
#pragma unroll
    for (int dt = 0; dt < DT; ++dt) output_acc[dt] = MmaBf16::zero();

    float row_max = -FLT_MAX;
    float row_sum = 1.0f;
    const float score_scale = sm_scale * kAttentionLog2e;
    const int key_tiles = NO_KV_TAIL
        ? kv_len / BLOCK_N
        : (kv_len + BLOCK_N - 1) / BLOCK_N;

    constexpr int kPackElems = 8;
    constexpr int kPacksPerRow = HD / kPackElems;
    constexpr int kPacksPerTile = BLOCK_N * kPacksPerRow;
    constexpr int kKStageItems = DIRECT_GLOBAL
        ? 1
        : (kPacksPerTile + Shape::kThreads - 1) / Shape::kThreads;
    if constexpr (!DIRECT_GLOBAL && !STAGE_V) {
#pragma unroll
        for (int item = 0; item < kKStageItems; ++item) {
            const int item_pack = tid + item * Shape::kThreads;
            if (item_pack >= kPacksPerTile) continue;
            const int item_key = item_pack / kPacksPerRow;
            const int item_d =
                (item_pack - item_key * kPacksPerRow) * kPackElems;
            v8bf k_value{};
            if constexpr (NO_KV_TAIL) {
                k_value = *reinterpret_cast<const v8bf*>(
                    k_head + static_cast<int64_t>(item_key) * k_stride_n + item_d);
            } else if (item_key < kv_len) {
                k_value = *reinterpret_cast<const v8bf*>(
                    k_head + static_cast<int64_t>(item_key) * k_stride_n + item_d);
            }
            store_bf16x8_padded(
                staged_k + item_key * Shape::kKStride + item_d, k_value);
        }
        __syncthreads();
    }

    for (int tile = 0; tile < key_tiles; ++tile) {
        const int key0 = tile * BLOCK_N;
        const bool has_next_tile = tile + 1 < key_tiles;
        const int next_buffer = (tile + 1) & 1;

        if constexpr (STAGE_V) {
            if (tile != 0) __syncthreads();
#pragma unroll
            for (int item = 0; item < kKStageItems; ++item) {
                const int item_pack = tid + item * Shape::kThreads;
                if (item_pack >= kPacksPerTile) continue;
                const int item_key = item_pack / kPacksPerRow;
                const int item_d =
                    (item_pack - item_key * kPacksPerRow) * kPackElems;
                const int key_index = key0 + item_key;
                v8bf k_value{};
                if constexpr (NO_KV_TAIL) {
                    k_value = *reinterpret_cast<const v8bf*>(
                        k_head + static_cast<int64_t>(key_index) * k_stride_n + item_d);
                } else if (key_index < kv_len) {
                    k_value = *reinterpret_cast<const v8bf*>(
                        k_head + static_cast<int64_t>(key_index) * k_stride_n + item_d);
                }
                store_bf16x8_padded(
                    staged_k + item_key * Shape::kKStride + item_d, k_value);

                v8bf v_value{};
                if constexpr (NO_KV_TAIL) {
                    v_value = *reinterpret_cast<const v8bf*>(
                        v_head + static_cast<int64_t>(key_index) * v_stride_n + item_d);
                } else if (key_index < kv_len) {
                    v_value = *reinterpret_cast<const v8bf*>(
                        v_head + static_cast<int64_t>(key_index) * v_stride_n + item_d);
                }
#pragma unroll
                for (int e = 0; e < kPackElems; ++e) {
                    staged_v[(item_d + e) * Shape::kVStride + item_key] =
                        v_value[e];
                }
            }
            __syncthreads();
        }

        const __bf16* const current_k = staged_k +
            (STAGE_V ? 0 : (tile & 1) * BLOCK_N * Shape::kKStride);

        MmaBf16::Acc score[KT];
#pragma unroll
        for (int kt = 0; kt < KT; ++kt) {
            score[kt] = MmaBf16::zero();
        }
#pragma unroll
        for (int dt = 0; dt < DT; ++dt) {
#pragma unroll
            for (int kt = 0; kt < KT; ++kt) {
                MmaBf16::Frag k_fragment{};
                if constexpr (DIRECT_GLOBAL) {
                    const int key_index = key0 + kt * 16 + frag_row(lane);
                    if (key_index < kv_len) {
                        k_fragment = load_frag_16bit<MmaBf16>(
                            k_head + static_cast<int64_t>(key_index) * k_stride_n + dt * 16,
                            lane);
                    }
                } else {
                    k_fragment = load_frag_16bit<MmaBf16>(
                        current_k + (kt * 16 + frag_row(lane)) * Shape::kKStride + dt * 16,
                        lane);
                }
                score[kt] = MmaBf16::mma(k_fragment, q_fragment[dt], score[kt]);
            }
        }

        // Prefetch the next K tile after score MMA so global latency hides
        // behind softmax prep instead of sitting in front of the MMA nest.
        v8bf next_k[kKStageItems]{};
        if constexpr (!DIRECT_GLOBAL && !STAGE_V) {
#pragma unroll
            for (int item = 0; item < kKStageItems; ++item) {
                const int item_pack = tid + item * Shape::kThreads;
                if (!has_next_tile || item_pack >= kPacksPerTile) continue;
                const int item_key = item_pack / kPacksPerRow;
                const int item_d =
                    (item_pack - item_key * kPacksPerRow) * kPackElems;
                const int next_key = key0 + BLOCK_N + item_key;
                if constexpr (NO_KV_TAIL) {
                    next_k[item] = *reinterpret_cast<const v8bf*>(
                        k_head + static_cast<int64_t>(next_key) * k_stride_n + item_d);
                } else if (next_key < kv_len) {
                    next_k[item] = *reinterpret_cast<const v8bf*>(
                        k_head + static_cast<int64_t>(next_key) * k_stride_n + item_d);
                }
            }
        }

        float local_max = -FLT_MAX;
#pragma unroll
        for (int kt = 0; kt < KT; ++kt) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                float value = score[kt][e] * score_scale;
                const int key_index = key0 + kt * 16 + acc_row(lane, e);
                if constexpr (!NO_KV_TAIL) {
                    if (key_index >= kv_len) value = -FLT_MAX;
                }
                score[kt][e] = value;
                local_max = fmaxf(local_max, value);
            }
        }
        const float tile_max = fmaxf(local_max, half_wave_swap(local_max));
        const float next_max = fmaxf(row_max, tile_max);
        const float alpha = __builtin_amdgcn_exp2f(row_max - next_max);
        row_max = next_max;
        row_sum *= alpha;
#pragma unroll
        for (int dt = 0; dt < DT; ++dt) {
#pragma unroll
            for (int e = 0; e < 8; ++e) output_acc[dt][e] *= alpha;
        }

        float tile_sum = 0.0f;
#pragma unroll
        for (int kt = 0; kt < KT; ++kt) {
            MmaBf16::Frag probability;
            v8f p;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                p[e] = __builtin_amdgcn_exp2f(score[kt][e] - next_max);
                probability[e] = static_cast<__bf16>(p[e]);
            }
            tile_sum += sum8_balanced(p);
#pragma unroll
            for (int dt = 0; dt < DT; ++dt) {
                MmaBf16::Frag v_fragment{};
                if constexpr (STAGE_V) {
                    v_fragment = load_frag_16bit<MmaBf16>(
                        staged_v + (dt * 16 + frag_row(lane)) * Shape::kVStride +
                        kt * 16,
                        lane);
                } else if constexpr (!DIRECT_GLOBAL) {
                    const int tile_key = key0 + kt * 16;
                    if (tile_key + 16 <= kv_len) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(COMFY_MMA_GFX12) && defined(__GFX12__)
                        const int source_key = ((lane >> 4) << 3) + (lane & 7);
                        const int source_half = (lane >> 3) & 1;
                        const v8bf* const source = reinterpret_cast<const v8bf*>(
                            v_head + static_cast<int64_t>(tile_key + source_key) * v_stride_n +
                            dt * 16 + source_half * 8);
                        v_fragment = __builtin_amdgcn_global_load_tr_b128_v8bf16(
                            const_cast<v8bf*>(source));
#else
                        const int d = dt * 16 + frag_row(lane);
#pragma unroll
                        for (int i = 0; i < MmaBf16::kFragElems; ++i) {
                            const int key_index =
                                tile_key + MmaBf16::frag_base(lane) + i;
                            v_fragment[i] = v_head[
                                static_cast<int64_t>(key_index) * v_stride_n + d];
                        }
#endif
                    } else if (tile_key < kv_len) {
                        const int d = dt * 16 + frag_row(lane);
#pragma unroll
                        for (int i = 0; i < MmaBf16::kFragElems; ++i) {
                            const int key_index = tile_key + MmaBf16::frag_base(lane) + i;
                            if (key_index < kv_len) {
                                v_fragment[i] = v_head[
                                    static_cast<int64_t>(key_index) * v_stride_n + d];
                            }
                        }
                    }
                } else {
                    const int d = dt * 16 + frag_row(lane);
#pragma unroll
                    for (int i = 0; i < MmaBf16::kFragElems; ++i) {
                        const int key_index =
                            key0 + kt * 16 + MmaBf16::frag_base(lane) + i;
                        if (key_index < kv_len) {
                            v_fragment[i] = v_head[
                                static_cast<int64_t>(key_index) * v_stride_n + d];
                        }
                    }
                }
                output_acc[dt] = MmaBf16::mma(
                    v_fragment, probability, output_acc[dt]);
            }
        }
        row_sum += tile_sum;

        if constexpr (!DIRECT_GLOBAL && !STAGE_V) {
#pragma unroll
            for (int item = 0; item < kKStageItems; ++item) {
                const int item_pack = tid + item * Shape::kThreads;
                if (!has_next_tile || item_pack >= kPacksPerTile) continue;
                const int item_key = item_pack / kPacksPerRow;
                const int item_d =
                    (item_pack - item_key * kPacksPerRow) * kPackElems;
                store_bf16x8_padded(
                    staged_k + next_buffer * BLOCK_N * Shape::kKStride +
                    item_key * Shape::kKStride + item_d, next_k[item]);
            }
            if (has_next_tile) __syncthreads();
        }
    }

    row_sum += half_wave_swap(row_sum);
    if (query_column < q_len) {
        const float reciprocal = 1.0f / row_sum;
        __bf16* const row = o_head + static_cast<int64_t>(query_column) * o_stride_n;
#pragma unroll
        for (int dt = 0; dt < DT; ++dt) {
            v8f normalized;
#pragma unroll
            for (int e = 0; e < 8; ++e) normalized[e] = output_acc[dt][e] * reciprocal;
            store_bf16_acc(row, dt * 16, normalized, lane);
        }
    }
}

template <int HD, int BLOCK_M, int BLOCK_N, bool DIRECT_GLOBAL,
          bool NO_KV_TAIL = false, bool STAGE_V = false>
void launch_bf16_attention_shape(
    const void* q, const void* k, const void* v, void* output,
    int batch, int q_heads, int kv_heads, int q_len, int kv_len,
    int64_t q_stride_b, int64_t q_stride_h, int64_t q_stride_n,
    int64_t k_stride_b, int64_t k_stride_h, int64_t k_stride_n,
    int64_t v_stride_b, int64_t v_stride_h, int64_t v_stride_n,
    int64_t o_stride_b, int64_t o_stride_h, int64_t o_stride_n,
    float sm_scale, hipStream_t stream) {
    const dim3 grid((q_len + BLOCK_M - 1) / BLOCK_M, q_heads, batch);
    bf16_attention_kernel<HD, BLOCK_M, BLOCK_N, DIRECT_GLOBAL, NO_KV_TAIL, STAGE_V>
        <<<grid, Bf16AttentionShape<HD, BLOCK_M, BLOCK_N>::kThreads, 0, stream>>>(
            static_cast<const __bf16*>(q), static_cast<const __bf16*>(k),
            static_cast<const __bf16*>(v), static_cast<__bf16*>(output),
            q_heads, kv_heads, q_len, kv_len,
            q_stride_b, q_stride_h, q_stride_n,
            k_stride_b, k_stride_h, k_stride_n,
            v_stride_b, v_stride_h, v_stride_n,
            o_stride_b, o_stride_h, o_stride_n, sm_scale);
}

template <int HD>
void dispatch_bf16_attention(
    const void* q, const void* k, const void* v, void* output,
    int batch, int q_heads, int kv_heads, int q_len, int kv_len,
    int64_t q_stride_b, int64_t q_stride_h, int64_t q_stride_n,
    int64_t k_stride_b, int64_t k_stride_h, int64_t k_stride_n,
    int64_t v_stride_b, int64_t v_stride_h, int64_t v_stride_n,
    int64_t o_stride_b, int64_t o_stride_h, int64_t o_stride_n,
    float sm_scale, hipStream_t stream) {
    if (q_len <= 16 && kv_len <= 16) {
        launch_bf16_attention_shape<HD, 16, 16, true>(
            q, k, v, output, batch, q_heads, kv_heads, q_len, kv_len,
            q_stride_b, q_stride_h, q_stride_n, k_stride_b, k_stride_h,
            k_stride_n, v_stride_b, v_stride_h, v_stride_n, o_stride_b,
            o_stride_h, o_stride_n, sm_scale, stream);
    } else if (q_len <= 256) {
        if (use_gfx1170_attention_schedule() && kv_len >= 128 &&
            kv_len % 32 == 0) {
            launch_bf16_attention_shape<HD, 64, 32, false, true, true>(
                q, k, v, output, batch, q_heads, kv_heads, q_len, kv_len,
                q_stride_b, q_stride_h, q_stride_n, k_stride_b, k_stride_h,
                k_stride_n, v_stride_b, v_stride_h, v_stride_n, o_stride_b,
                o_stride_h, o_stride_n, sm_scale, stream);
        } else {
            launch_bf16_attention_shape<HD, 32, 32, true>(
                q, k, v, output, batch, q_heads, kv_heads, q_len, kv_len,
                q_stride_b, q_stride_h, q_stride_n, k_stride_b, k_stride_h,
                k_stride_n, v_stride_b, v_stride_h, v_stride_n, o_stride_b,
                o_stride_h, o_stride_n, sm_scale, stream);
        }
    } else {
        if (use_gfx1170_attention_schedule() && kv_len >= 1024 &&
            kv_len % 32 == 0) {
            launch_bf16_attention_shape<HD, 512, 32, false, true, true>(
                q, k, v, output, batch, q_heads, kv_heads, q_len, kv_len,
                q_stride_b, q_stride_h, q_stride_n, k_stride_b, k_stride_h,
                k_stride_n, v_stride_b, v_stride_h, v_stride_n, o_stride_b,
                o_stride_h, o_stride_n, sm_scale, stream);
        } else if (use_gfx12_attention_schedule() && kv_len >= 1024 &&
                   kv_len % 32 == 0) {
            launch_bf16_attention_shape<HD, 128, 32, false, true>(
                q, k, v, output, batch, q_heads, kv_heads, q_len, kv_len,
                q_stride_b, q_stride_h, q_stride_n, k_stride_b, k_stride_h,
                k_stride_n, v_stride_b, v_stride_h, v_stride_n, o_stride_b,
                o_stride_h, o_stride_n, sm_scale, stream);
        } else if (kv_len % 48 == 0) {
            launch_bf16_attention_shape<HD, 192, 48, false, true>(
                q, k, v, output, batch, q_heads, kv_heads, q_len, kv_len,
                q_stride_b, q_stride_h, q_stride_n, k_stride_b, k_stride_h,
                k_stride_n, v_stride_b, v_stride_h, v_stride_n, o_stride_b,
                o_stride_h, o_stride_n, sm_scale, stream);
        } else {
            launch_bf16_attention_shape<HD, 192, 48, false, false>(
                q, k, v, output, batch, q_heads, kv_heads, q_len, kv_len,
                q_stride_b, q_stride_h, q_stride_n, k_stride_b, k_stride_h,
                k_stride_n, v_stride_b, v_stride_h, v_stride_n, o_stride_b,
                o_stride_h, o_stride_n, sm_scale, stream);
        }
    }
}

}  // namespace
}  // namespace comfy::hip_backend

extern "C" void launch_bf16_sdpa_hip(
    const void* q, const void* k, const void* v, void* output,
    int batch, int q_heads, int kv_heads, int q_len, int kv_len, int head_dim,
    int64_t q_stride_b, int64_t q_stride_h, int64_t q_stride_n,
    int64_t k_stride_b, int64_t k_stride_h, int64_t k_stride_n,
    int64_t v_stride_b, int64_t v_stride_h, int64_t v_stride_n,
    int64_t o_stride_b, int64_t o_stride_h, int64_t o_stride_n,
    float sm_scale, hipStream_t stream) {
    using namespace comfy::hip_backend;
    if (batch <= 0 || q_heads <= 0 || kv_heads <= 0 || q_len <= 0 || kv_len <= 0 ||
        q_heads % kv_heads != 0) {
        throw std::runtime_error("BF16 HIP attention received invalid extents");
    }

#define COMFY_BF16_ATTN_CASE(HD)                                                        \
    dispatch_bf16_attention<HD>(                                                       \
        q, k, v, output, batch, q_heads, kv_heads, q_len, kv_len,                     \
        q_stride_b, q_stride_h, q_stride_n, k_stride_b, k_stride_h, k_stride_n,       \
        v_stride_b, v_stride_h, v_stride_n, o_stride_b, o_stride_h, o_stride_n,       \
        sm_scale, stream)

    switch (head_dim) {
        case 128: COMFY_BF16_ATTN_CASE(128); break;
        default:
            throw std::runtime_error(
                "BF16 HIP attention supports head dimension 128, got " +
                std::to_string(head_dim));
    }
#undef COMFY_BF16_ATTN_CASE

    const hipError_t error = hipGetLastError();
    if (error != hipSuccess) {
        throw std::runtime_error(
            std::string("BF16 HIP attention launch failed: ") + hipGetErrorString(error));
    }
}
