// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Port of the SageAttention gfx110x int8-QK attention kernel
// (attn_gfx110x.cu, attn_kernel_impl_t + the wpe wrappers) into comfy_kitchen.
// Same schedule as the reference library: BLOCK_M 64/128 queries per block,
// 256 threads for BLOCK_M=128 with amdgpu_waves_per_eu(2), K tile prefetched
// into LDS, scores kept in fp32 registers, PV on the WMMA.
//
// Two PV paths, chosen by INT8_PV:
//   false: fp16 SV like the reference library (int8 QK, unquantized fp16 V,
//          fp16 probabilities, f16 WMMA PV).
//   true:  the pure-int8 path (int8 V + u8 probabilities + iu8 WMMA with an
//          unsigned B operand) for the int8 precision / speed budget.
//
// Adaptations to comfy_kitchen: HND layout only, no causal flag (masks fall
// back to the mask-capable bf16-SV kernel), and the Q scale is per query row
// (comfy's quantizer) instead of per MIN_BLK_Q rows. The int8 path folds the
// per-channel V scale into the writeback.
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

#include <cfloat>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "sage_common.h"
#include "../launchers.h"

namespace sageattn_gfx11 {

typedef int int32_t_v4 __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef __bf16 v16bf __attribute__((ext_vector_type(16)));
typedef _Float16 v16h __attribute__((ext_vector_type(16)));

// 2^x for the softmax. See hw_exp2 in ../mma.h for why the hardware
// instruction replaces exp2f here.
__device__ __forceinline__ float fast_exp2(float x) {
    return comfy::hip_backend::hw_exp2(x);
}

__device__ __forceinline__ _Float16 f16_from_bits(unsigned bits) {
    _Float16 r;
    __builtin_memcpy(&r, &bits, sizeof(_Float16));
    return r;
}

__device__ __forceinline__ float permlanex16(float src) {
    float result;
    asm volatile("v_permlanex16_b32 %0, %1, %2, %3 op_sel:[1,0]"
                 : "=v"(result)
                 : "v"(src), "s"(0x76543210), "n"(0xfedcba98));
    return result;
}

__device__ __forceinline__ unsigned permlanex16_u32(unsigned src) {
    unsigned result;
    asm volatile("v_permlanex16_b32 %0, %1, %2, %3 op_sel:[1,0]"
                 : "=v"(result)
                 : "v"(src), "s"(0x76543210), "n"(0xfedcba98));
    return result;
}

__device__ __forceinline__ int32_t_v4 load_i8_frag(const void* lds, int row, int kbyte,
                                                   int stride) {
    const char* p = static_cast<const char*>(lds) + row * stride + kbyte;
    int32_t_v4 r;
    const int* q = reinterpret_cast<const int*>(p);
    r[0] = q[0];
    r[1] = q[1];
    r[2] = q[2];
    r[3] = q[3];
    return r;
}

__device__ __forceinline__ v8i wmma_i32_iu8(int32_t_v4 a, int32_t_v4 b, v8i c) {
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, false);
}

// Signed A, unsigned B: the u8 probabilities (0-255) must not be re-read as
// signed int8, which would negate every value > 127.
__device__ __forceinline__ v8i wmma_i32_iu8_ub(int32_t_v4 a, int32_t_v4 b, v8i c) {
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, false, b, c, false);
}

__device__ __forceinline__ v8f wmma_f32_f16(v16h a, v16h b, v8f c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

__device__ __forceinline__ v8f wmma_f32_bf16(v16bf a, v16bf b, v8f c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
}

// fp16 P B-fragment for the PV: the full 16-column row (L&15) of P^T. p_vals[e]
// = P[L&15][2e+hw]; permlanex16 supplies the other parity's columns.
__device__ __forceinline__ v16h assemble_p_frag(const float p_vals[8], int hw) {
    unsigned pack[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        __half lo = __float2half_rn(p_vals[2 * k]);
        __half hi = __float2half_rn(p_vals[2 * k + 1]);
        pack[k] = (static_cast<unsigned>(__half_as_ushort(hi)) << 16) |
                  static_cast<unsigned>(__half_as_ushort(lo));
    }
    unsigned cross[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) cross[k] = permlanex16_u32(pack[k]);
    unsigned even[4], odd[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        even[k] = (hw == 0) ? pack[k] : cross[k];
        odd[k] = (hw == 0) ? cross[k] : pack[k];
    }
    v16h b;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        b[4 * k] = f16_from_bits(even[k] & 0xFFFF);
        b[4 * k + 1] = f16_from_bits(odd[k] & 0xFFFF);
        b[4 * k + 2] = f16_from_bits(even[k] >> 16);
        b[4 * k + 3] = f16_from_bits(odd[k] >> 16);
    }
    return b;
}

}  // namespace sageattn_gfx11

namespace comfy::hip_backend::sage {
using namespace sageattn_gfx11;
namespace {

constexpr int kRM = 16;
constexpr int kBK = 16;
constexpr int kMinBlkK = 16;
constexpr int kLdsPad = 16;

constexpr int64_t div_ceil_i64(int64_t a, int64_t b) { return (a + b - 1) / b; }

// Bit access for the scalar __bf16 builtin (this HIP has no __bfloat16_as_ushort).
__device__ __forceinline__ unsigned short bf16_to_ushort(__bf16 v) {
    unsigned short bits;
    __builtin_memcpy(&bits, &v, sizeof(bits));
    return bits;
}
__device__ __forceinline__ __bf16 ushort_to_bf16(unsigned short bits) {
    __bf16 v;
    __builtin_memcpy(&v, &bits, sizeof(bits));
    return v;
}

// u8 P B-fragment for the iu8 PV. gfx11's i8 WMMA B fragment layout (element e
// = key 2e + lane/16) needs the interleave the proven S^T kernel builds: 8 own
// u8 values per lane, half-wave exchange for the other parity, v_perm mix.
__device__ __forceinline__ int32_t_v4 assemble_p_frag_u8(const float p_vals[8], int hw) {
    uint32_t packed[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) packed[e] = prob_to_u8(p_vals[e]);
    const uint32_t lo = packed[0] | (packed[1] << 8) | (packed[2] << 16) | (packed[3] << 24);
    const uint32_t hi = packed[4] | (packed[5] << 8) | (packed[6] << 16) | (packed[7] << 24);
    const uint32_t xlo = permlanex16_u32(lo);
    const uint32_t xhi = permlanex16_u32(hi);
    const bool even_half = (hw == 0);
    const uint32_t even0 = even_half ? lo : xlo;  // keys 0, 2, 4, 6
    const uint32_t odd0 = even_half ? xlo : lo;   // keys 1, 3, 5, 7
    const uint32_t even1 = even_half ? hi : xhi;  // keys 8, 10, 12, 14
    const uint32_t odd1 = even_half ? xhi : hi;   // keys 9, 11, 13, 15
    int32_t_v4 f;
    f[0] = static_cast<int>(__builtin_amdgcn_perm(odd0, even0, 0x05010400u));
    f[1] = static_cast<int>(__builtin_amdgcn_perm(odd0, even0, 0x07030602u));
    f[2] = static_cast<int>(__builtin_amdgcn_perm(odd1, even1, 0x05010400u));
    f[3] = static_cast<int>(__builtin_amdgcn_perm(odd1, even1, 0x07030602u));
    return f;
}

// ---------------------------------------------------------------------------
// Direct fp16/bf16 attention (no int8 quantization): QK runs on the f16/bf16
// matrix core straight from the inputs, V on f16 WMMA from the transposed
// buffer. This is what the reference library uses for short keys where the
// int8 prepass overhead dominates (D64 kv <= 2048, D128 kv <= 256).
// ---------------------------------------------------------------------------
template <int HeadDim, int BLOCK_M, int BLOCK_N, typename QK_DTYPE, typename OUT_DTYPE = QK_DTYPE>
__device__ __forceinline__ void direct_attn_kernel_impl_t(
    const QK_DTYPE* __restrict__ q, const QK_DTYPE* __restrict__ k,
    const __half* __restrict__ v, OUT_DTYPE* __restrict__ output, int64_t batch_size,
    int64_t qo_len, int64_t kv_len, int64_t num_qo_heads, int64_t num_kv_heads,
    int64_t q_stride_b, int64_t q_stride_n, int64_t q_stride_h, int64_t k_stride_b,
    int64_t k_stride_n, int64_t k_stride_h, int64_t v_stride_b, int64_t v_stride_n,
    int64_t v_stride_h, int64_t o_stride_b, int64_t o_stride_n, int64_t o_stride_h,
    float sm_scale_log2e) {
    constexpr int WARPS = BLOCK_M / kRM;
    constexpr int THREADS = WARPS * 32;
    constexpr int DTiles = HeadDim / kBK;
    constexpr int ColTiles = BLOCK_N / kBK;
    constexpr int KStride = HeadDim + kLdsPad;
    using QK_VEC = std::conditional_t<std::is_same<QK_DTYPE, __half>::value, v16h, v16bf>;
    using QK_ELEM = std::conditional_t<std::is_same<QK_DTYPE, __half>::value, _Float16, __bf16>;

    __shared__ QK_DTYPE k_tile[BLOCK_N * KStride];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wave = tid >> 5;
    const int64_t q_base = static_cast<int64_t>(blockIdx.x) * BLOCK_M;
    const int64_t hq = blockIdx.y;
    const int64_t b = blockIdx.z;
    const int64_t hkv = num_kv_heads > 1 ? (hq / (num_qo_heads / num_kv_heads)) : 0;
    if (hq >= num_qo_heads || q_base >= qo_len) return;
    const int64_t q_start = q_base + static_cast<int64_t>(wave) * kRM;

    const int q_row = lane & 15;
    const int64_t q_idx_q = q_start + q_row;
    QK_VEC q_frag[DTiles];
#pragma unroll
    for (int dt = 0; dt < DTiles; ++dt) {
        if (q_idx_q < qo_len) {
            const int64_t q_off = b * q_stride_b + hq * q_stride_h + q_idx_q * q_stride_n +
                                  dt * kBK;
            const QK_DTYPE* src = q + q_off;
#pragma unroll
            for (int i = 0; i < 16; ++i) q_frag[dt][i] = static_cast<QK_ELEM>(src[i]);
        } else {
#pragma unroll
            for (int i = 0; i < 16; ++i) q_frag[dt][i] = 0;
        }
    }

    v8f out_acc[DTiles];
#pragma unroll
    for (int dt = 0; dt < DTiles; ++dt) out_acc[dt] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
    float row_m = -FLT_MAX * 0.5f;
    float row_l = 0.0f;

    const int64_t kv_limit = kv_len;
    const int64_t v_t_n = ((kv_len + 63) / 64) * 64;
    constexpr int KVecsPerRow = HeadDim / 8;
    constexpr int KVecsTotal = BLOCK_N * KVecsPerRow;
    constexpr int KPrefetchPerThread = (KVecsTotal + THREADS - 1) / THREADS;

    uint4 k_prefetch[KPrefetchPerThread];
    for (int i = tid; i < KVecsTotal; i += THREADS) {
        const int n = i / KVecsPerRow;
        const int d = (i - n * KVecsPerRow) * 8;
        if (n < kv_len) {
            const int64_t k_off = b * k_stride_b + hkv * k_stride_h + n * k_stride_n + d;
            *reinterpret_cast<uint4*>(&k_tile[n * KStride + d]) =
                *reinterpret_cast<const uint4*>(k + k_off);
        } else {
            *reinterpret_cast<uint4*>(&k_tile[n * KStride + d]) = make_uint4(0, 0, 0, 0);
        }
    }
    __syncthreads();

    const int hw = lane >> 4;
    const int m_row = lane & 15;

    for (int64_t kb_base = 0; kb_base < kv_limit; kb_base += BLOCK_N) {
        const int64_t next_base = kb_base + BLOCK_N;
        const bool has_next = (next_base < kv_limit);

        if (has_next) {
#pragma unroll
            for (int i = 0; i < KPrefetchPerThread; ++i) {
                const int vec = tid + i * THREADS;
                if (vec < KVecsTotal) {
                    const int n = vec / KVecsPerRow;
                    const int d = (vec - n * KVecsPerRow) * 8;
                    const int64_t k_idx = next_base + n;
                    if (k_idx < kv_len) {
                        const int64_t k_off =
                            b * k_stride_b + hkv * k_stride_h + k_idx * k_stride_n + d;
                        k_prefetch[i] = *reinterpret_cast<const uint4*>(k + k_off);
                    } else {
                        k_prefetch[i] = make_uint4(0, 0, 0, 0);
                    }
                }
            }
        }

        float score_cache[ColTiles][8];
#pragma unroll
        for (int ct = 0; ct < ColTiles; ++ct) {
            v8f score_acc = v8f{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int dt = 0; dt < DTiles; ++dt) {
                const QK_DTYPE* k_ptr = &k_tile[(ct * kBK + m_row) * KStride + dt * kBK];
                QK_VEC k_frag;
#pragma unroll
                for (int i = 0; i < 16; ++i) k_frag[i] = static_cast<QK_ELEM>(k_ptr[i]);
                if constexpr (std::is_same<QK_DTYPE, __half>::value) {
                    score_acc = wmma_f32_f16(k_frag, q_frag[dt], score_acc);
                } else {
                    score_acc = wmma_f32_bf16(k_frag, q_frag[dt], score_acc);
                }
            }
            const int64_t k_col_start = kb_base + ct * kBK;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int col = static_cast<int>(k_col_start) + 2 * e + hw;
                float s = static_cast<float>(score_acc[e]) * sm_scale_log2e;
                if (col >= kv_len) s = -FLT_MAX * 0.5f;
                score_cache[ct][e] = s;
            }
        }

        float local_mx = score_cache[0][0];
#pragma unroll
        for (int ct = 0; ct < ColTiles; ++ct) {
#pragma unroll
            for (int e = 0; e < 8; ++e) local_mx = fmaxf(local_mx, score_cache[ct][e]);
        }
        float gm = fmaxf(row_m, local_mx);
        gm = fmaxf(gm, permlanex16(gm));
        const float alpha = (row_l == 0.0f) ? 0.0f : fast_exp2(row_m - gm);
        row_m = gm;
        row_l *= alpha;
#pragma unroll
        for (int dt = 0; dt < DTiles; ++dt) out_acc[dt] *= alpha;
#pragma unroll
        for (int ct = 0; ct < ColTiles; ++ct) {
#pragma unroll
            for (int e = 0; e < 8; ++e) score_cache[ct][e] = fast_exp2(score_cache[ct][e] - gm);
        }
        float partial_sm = 0.0f;
#pragma unroll
        for (int ct = 0; ct < ColTiles; ++ct) {
#pragma unroll
            for (int e = 0; e < 8; ++e) partial_sm += score_cache[ct][e];
        }
        row_l += partial_sm + permlanex16(partial_sm);

#pragma unroll
        for (int ct = 0; ct < ColTiles; ++ct) {
            float p_vals[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) p_vals[e] = score_cache[ct][e];
            const v16h p_frag = assemble_p_frag(p_vals, hw);
#pragma unroll
            for (int dt = 0; dt < DTiles; ++dt) {
                const int64_t vt_off =
                    ((b * num_kv_heads + hkv) * HeadDim + (dt * kBK + m_row)) * v_t_n +
                    (kb_base + ct * kBK);
                const v16h v_frag_t = *reinterpret_cast<const v16h*>(v + vt_off);
                out_acc[dt] = wmma_f32_f16(v_frag_t, p_frag, out_acc[dt]);
            }
        }

        if (has_next) {
            __syncthreads();
#pragma unroll
            for (int i = 0; i < KPrefetchPerThread; ++i) {
                const int vec = tid + i * THREADS;
                if (vec < KVecsTotal) {
                    const int n = vec / KVecsPerRow;
                    const int d = (vec - n * KVecsPerRow) * 8;
                    *reinterpret_cast<uint4*>(&k_tile[n * KStride + d]) = k_prefetch[i];
                }
            }
            __syncthreads();
        }
    }

    const int64_t q_idx = q_start + m_row;
    if (q_idx < qo_len) {
        const float inv_l = (row_l > 0.0f) ? (1.0f / row_l) : 0.0f;
#pragma unroll
        for (int dt = 0; dt < DTiles; ++dt) {
            unsigned pack[4], cross[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                if constexpr (std::is_same<OUT_DTYPE, __half>::value) {
                    __half lo = __float2half_rn(out_acc[dt][2 * k] * inv_l);
                    __half hi = __float2half_rn(out_acc[dt][2 * k + 1] * inv_l);
                    pack[k] = (static_cast<unsigned>(__half_as_ushort(hi)) << 16) |
                              static_cast<unsigned>(__half_as_ushort(lo));
                } else {
                    __bf16 lo = static_cast<__bf16>(out_acc[dt][2 * k] * inv_l);
                    __bf16 hi = static_cast<__bf16>(out_acc[dt][2 * k + 1] * inv_l);
                    pack[k] = (static_cast<unsigned>(bf16_to_ushort(hi)) << 16) |
                              static_cast<unsigned>(bf16_to_ushort(lo));
                }
            }
#pragma unroll
            for (int k = 0; k < 4; ++k) cross[k] = permlanex16_u32(pack[k]);

            const int d = dt * kBK + hw * 8;
            const int64_t o_off = b * o_stride_b + hq * o_stride_h + q_idx * o_stride_n + d;
            if constexpr (std::is_same<OUT_DTYPE, __half>::value) {
                alignas(16) unsigned b16[8];
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const unsigned even = (hw == 0) ? pack[k] : cross[k];
                    const unsigned odd = (hw == 0) ? cross[k] : pack[k];
                    b16[2 * k] = (even & 0xFFFF) | ((odd & 0xFFFF) << 16);
                    b16[2 * k + 1] = (even >> 16) | (odd & 0xFFFF0000);
                }
                *reinterpret_cast<uint4*>(reinterpret_cast<__half*>(output) + o_off) =
                    *reinterpret_cast<const uint4*>(b16 + hw * 4);
            } else {
                alignas(16) __bf16 b[16];
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const unsigned even = (hw == 0) ? pack[k] : cross[k];
                    const unsigned odd = (hw == 0) ? cross[k] : pack[k];
                    const unsigned short e_lo = static_cast<unsigned short>(even & 0xFFFF);
                    const unsigned short o_lo = static_cast<unsigned short>(odd & 0xFFFF);
                    const unsigned short e_hi = static_cast<unsigned short>(even >> 16);
                    const unsigned short o_hi = static_cast<unsigned short>(odd >> 16);
                    b[4 * k + 0] = ushort_to_bf16(e_lo);
                    b[4 * k + 1] = ushort_to_bf16(o_lo);
                    b[4 * k + 2] = ushort_to_bf16(e_hi);
                    b[4 * k + 3] = ushort_to_bf16(o_hi);
                }
                *reinterpret_cast<uint4*>(reinterpret_cast<__bf16*>(output) + o_off) =
                    *reinterpret_cast<const uint4*>(b + hw * 8);
            }
        }
    }
}

template <int HeadDim, int BLOCK_M, int BLOCK_N, typename QK_DTYPE, typename OUT_DTYPE = QK_DTYPE>
__global__ __launch_bounds__(BLOCK_M / 16 * 32, 2) __attribute__((amdgpu_waves_per_eu(2, 2)))
void direct_attn_kernel_wpe2_t(
    const QK_DTYPE* __restrict__ q, const QK_DTYPE* __restrict__ k, const __half* __restrict__ v,
    OUT_DTYPE* __restrict__ output, int64_t batch_size, int64_t qo_len, int64_t kv_len,
    int64_t num_qo_heads, int64_t num_kv_heads, int64_t q_stride_b, int64_t q_stride_n,
    int64_t q_stride_h, int64_t k_stride_b, int64_t k_stride_n, int64_t k_stride_h,
    int64_t v_stride_b, int64_t v_stride_n, int64_t v_stride_h, int64_t o_stride_b,
    int64_t o_stride_n, int64_t o_stride_h, float sm_scale_log2e) {
    direct_attn_kernel_impl_t<HeadDim, BLOCK_M, BLOCK_N, QK_DTYPE, OUT_DTYPE>(
        q, k, v, output, batch_size, qo_len, kv_len, num_qo_heads, num_kv_heads, q_stride_b,
        q_stride_n, q_stride_h, k_stride_b, k_stride_n, k_stride_h, v_stride_b, v_stride_n,
        v_stride_h, o_stride_b, o_stride_n, o_stride_h, sm_scale_log2e);
}

// 128-thread twin of attn_kernel_impl_t: each wave owns TWO 16-query tiles
// (32 queries) and V is read straight from the transposed global buffer 闁?the
// reference library's D64 schedule (attn_kernel_impl_32_t).
template <int HeadDim, int BLOCK_M, int BLOCK_N, typename OUT_DTYPE, bool INT8_PV = false>
__device__ __forceinline__ void attn_kernel_impl_32_t(
    const int8_t* __restrict__ q, const int8_t* __restrict__ k,
    const std::conditional_t<INT8_PV, int8_t, __half>* __restrict__ v,
    OUT_DTYPE* __restrict__ output, const float* __restrict__ q_scale,
    const float* __restrict__ k_scale, const float* __restrict__ v_scale, int64_t batch_size,
    int64_t qo_len, int64_t kv_len, int64_t num_qo_heads, int64_t num_kv_heads,
    int64_t q_stride_b, int64_t q_stride_n, int64_t q_stride_h, int64_t k_stride_b,
    int64_t k_stride_n, int64_t k_stride_h, int64_t v_stride_b, int64_t v_stride_n,
    int64_t v_stride_h, int64_t o_stride_b, int64_t o_stride_n, int64_t o_stride_h,
    int64_t qs_stride_b, int64_t qs_stride_h, int64_t ks_stride_b, int64_t ks_stride_h,
    float sm_scale) {
    constexpr int WARPS = BLOCK_M / (2 * kRM);
    constexpr int THREADS = WARPS * 32;
    constexpr int DTiles = HeadDim / kBK;
    constexpr int ColTiles = BLOCK_N / kBK;
    constexpr int KStride = HeadDim + kLdsPad;
    using AccT = std::conditional_t<INT8_PV, v8i, v8f>;

    __shared__ int8_t k_tile[BLOCK_N * KStride];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wave = tid >> 5;
    const int64_t q_base = static_cast<int64_t>(blockIdx.x) * BLOCK_M;
    const int64_t hq = blockIdx.y;
    const int64_t b = blockIdx.z;
    const int64_t hkv = num_kv_heads > 1 ? (hq / (num_qo_heads / num_kv_heads)) : 0;
    if (hq >= num_qo_heads || q_base >= qo_len) return;
    const int64_t q_start = q_base + static_cast<int64_t>(wave) * (2 * kRM);

    const int q_row = lane & 15;
    int32_t_v4 q_frag[2][DTiles];
#pragma unroll
    for (int s = 0; s < 2; ++s) {
#pragma unroll
        for (int dt = 0; dt < DTiles; ++dt) {
            const int64_t q_idx = q_start + s * kRM + q_row;
            if (q_idx < qo_len) {
                const int64_t q_off = b * q_stride_b + hq * q_stride_h + q_idx * q_stride_n +
                                      dt * kBK;
                q_frag[s][dt] = *reinterpret_cast<const int32_t_v4*>(q + q_off);
            } else {
                q_frag[s][dt] = int32_t_v4{0, 0, 0, 0};
            }
        }
    }

    float qs_lane[2];
#pragma unroll
    for (int s = 0; s < 2; ++s) {
        const int64_t q_idx = q_start + s * kRM + q_row;
        qs_lane[s] =
            (q_idx < qo_len) ? q_scale[b * qs_stride_b + hq * qs_stride_h + q_idx] : 0.0f;
    }

    AccT out_acc[2][DTiles];
#pragma unroll
    for (int s = 0; s < 2; ++s) {
#pragma unroll
        for (int dt = 0; dt < DTiles; ++dt) {
            if constexpr (INT8_PV) {
                out_acc[s][dt] = v8i{0, 0, 0, 0, 0, 0, 0, 0};
            } else {
                out_acc[s][dt] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
            }
        }
    }

    float row_m[2] = {-FLT_MAX * 0.5f, -FLT_MAX * 0.5f};
    float row_l[2] = {0.0f, 0.0f};

    const int64_t kv_limit = kv_len;
    const int64_t v_t_n = ((kv_len + 63) / 64) * 64;
    constexpr int KVecsPerRow = HeadDim / 16;
    constexpr int KVecsTotal = BLOCK_N * KVecsPerRow;
    constexpr int KPrefetchPerThread = (KVecsTotal + THREADS - 1) / THREADS;

    uint4 k_prefetch[KPrefetchPerThread];
    for (int i = tid; i < KVecsTotal; i += THREADS) {
        const int n = i / KVecsPerRow;
        const int d = (i - n * KVecsPerRow) * 16;
        if (n < kv_len) {
            const int64_t k_off = b * k_stride_b + hkv * k_stride_h + n * k_stride_n + d;
            *reinterpret_cast<uint4*>(&k_tile[n * KStride + d]) =
                *reinterpret_cast<const uint4*>(k + k_off);
        } else {
            *reinterpret_cast<uint4*>(&k_tile[n * KStride + d]) = make_uint4(0, 0, 0, 0);
        }
    }
    __syncthreads();

    const int hw = lane >> 4;
    const int m_row = lane & 15;

    for (int64_t kb_base = 0; kb_base < kv_limit; kb_base += BLOCK_N) {
        const int64_t next_base = kb_base + BLOCK_N;
        const bool has_next = (next_base < kv_limit);

        if (has_next) {
#pragma unroll
            for (int i = 0; i < KPrefetchPerThread; ++i) {
                const int vec = tid + i * THREADS;
                if (vec < KVecsTotal) {
                    const int n = vec / KVecsPerRow;
                    const int d = (vec - n * KVecsPerRow) * 16;
                    const int64_t k_idx = next_base + n;
                    if (k_idx < kv_len) {
                        const int64_t k_off =
                            b * k_stride_b + hkv * k_stride_h + k_idx * k_stride_n + d;
                        k_prefetch[i] = *reinterpret_cast<const uint4*>(k + k_off);
                    } else {
                        k_prefetch[i] = make_uint4(0, 0, 0, 0);
                    }
                }
            }
        }

        float score_cache[2][ColTiles][8];
#pragma unroll
        for (int ct = 0; ct < ColTiles; ++ct) {
            v8i score_acc0 = v8i{0, 0, 0, 0, 0, 0, 0, 0};
            v8i score_acc1 = v8i{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int dt = 0; dt < DTiles; ++dt) {
                const int32_t_v4 k_frag =
                    load_i8_frag(k_tile + ct * kBK * KStride, m_row, dt * kBK, KStride);
                score_acc0 = wmma_i32_iu8(k_frag, q_frag[0][dt], score_acc0);
                score_acc1 = wmma_i32_iu8(k_frag, q_frag[1][dt], score_acc1);
            }
            const int64_t k_col_start = kb_base + ct * kBK;
            const int k_scale_idx = static_cast<int>(k_col_start / kMinBlkK);
            const float ks_val = k_scale[b * ks_stride_b + hkv * ks_stride_h + k_scale_idx];
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                const v8i score_acc = (s == 0) ? score_acc0 : score_acc1;
                const float score_scale = qs_lane[s] * ks_val * sm_scale;
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int col = static_cast<int>(k_col_start) + 2 * e + hw;
                    float sv = static_cast<float>(score_acc[e]) * score_scale;
                    if (col >= kv_len) sv = -FLT_MAX * 0.5f;
                    score_cache[s][ct][e] = sv;
                }
            }
        }

#pragma unroll
        for (int s = 0; s < 2; ++s) {
            float local_mx = score_cache[s][0][0];
#pragma unroll
            for (int ct = 0; ct < ColTiles; ++ct) {
#pragma unroll
                for (int e = 0; e < 8; ++e) local_mx = fmaxf(local_mx, score_cache[s][ct][e]);
            }
            float gm = fmaxf(row_m[s], local_mx);
            gm = fmaxf(gm, permlanex16(gm));
            const float alpha = (row_l[s] == 0.0f) ? 0.0f : fast_exp2(row_m[s] - gm);
            row_m[s] = gm;
            row_l[s] *= alpha;
#pragma unroll
            for (int dt = 0; dt < DTiles; ++dt) {
                if constexpr (INT8_PV) {
#pragma unroll
                    for (int e = 0; e < 8; ++e)
                        out_acc[s][dt][e] = static_cast<int>(
                            static_cast<float>(out_acc[s][dt][e]) * alpha);
                } else {
                    out_acc[s][dt] *= alpha;
                }
            }
#pragma unroll
            for (int ct = 0; ct < ColTiles; ++ct) {
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    score_cache[s][ct][e] =
                        fast_exp2(score_cache[s][ct][e] - gm + (INT8_PV ? kProbU8Offset : 0.0f));
                }
            }
            float partial_sm = 0.0f;
#pragma unroll
            for (int ct = 0; ct < ColTiles; ++ct) {
#pragma unroll
                for (int e = 0; e < 8; ++e) partial_sm += score_cache[s][ct][e];
            }
            row_l[s] += partial_sm + permlanex16(partial_sm);
        }

        // V straight from the transposed global buffer (no LDS tile). The
        // V fragments are read once per key tile and shared by both query
        // tiles (halves the global V traffic, like the reference kernel).
#pragma unroll
        for (int ct = 0; ct < ColTiles; ++ct) {
            const int64_t vt_base =
                ((b * num_kv_heads + hkv) * HeadDim) * v_t_n + (kb_base + ct * kBK);
            if constexpr (INT8_PV) {
                int32_t_v4 v_frag[DTiles];
#pragma unroll
                for (int dt = 0; dt < DTiles; ++dt) {
                    const int64_t vt_off = vt_base + (dt * kBK + m_row) * v_t_n;
                    v_frag[dt] = *reinterpret_cast<const int32_t_v4*>(v + vt_off);
                }
#pragma unroll
                for (int s = 0; s < 2; ++s) {
                    float p_vals[8];
#pragma unroll
                    for (int e = 0; e < 8; ++e) p_vals[e] = score_cache[s][ct][e];
                    const int32_t_v4 p_frag = assemble_p_frag_u8(p_vals, hw);
#pragma unroll
                    for (int dt = 0; dt < DTiles; ++dt)
                        out_acc[s][dt] = wmma_i32_iu8_ub(v_frag[dt], p_frag, out_acc[s][dt]);
                }
            } else {
                v16h v_frag[DTiles];
#pragma unroll
                for (int dt = 0; dt < DTiles; ++dt) {
                    const int64_t vt_off = vt_base + (dt * kBK + m_row) * v_t_n;
                    v_frag[dt] = *reinterpret_cast<const v16h*>(v + vt_off);
                }
#pragma unroll
                for (int s = 0; s < 2; ++s) {
                    float p_vals[8];
#pragma unroll
                    for (int e = 0; e < 8; ++e) p_vals[e] = score_cache[s][ct][e];
                    const v16h p_frag = assemble_p_frag(p_vals, hw);
#pragma unroll
                    for (int dt = 0; dt < DTiles; ++dt)
                        out_acc[s][dt] = wmma_f32_f16(v_frag[dt], p_frag, out_acc[s][dt]);
                }
            }
        }

        if (has_next) {
            __syncthreads();
#pragma unroll
            for (int i = 0; i < KPrefetchPerThread; ++i) {
                const int vec = tid + i * THREADS;
                if (vec < KVecsTotal) {
                    const int n = vec / KVecsPerRow;
                    const int d = (vec - n * KVecsPerRow) * 16;
                    *reinterpret_cast<uint4*>(&k_tile[n * KStride + d]) = k_prefetch[i];
                }
            }
            __syncthreads();
        }
    }

    const int64_t vs_base = (b * num_kv_heads + hkv) * HeadDim;
#pragma unroll
    for (int s = 0; s < 2; ++s) {
        const int64_t q_idx = q_start + s * kRM + m_row;
        if (q_idx < qo_len) {
            const float inv_l = (row_l[s] > 0.0f) ? (1.0f / row_l[s]) : 0.0f;
#pragma unroll
            for (int dt = 0; dt < DTiles; ++dt) {
                unsigned pack[4], cross[4];
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const float lo0 = static_cast<float>(out_acc[s][dt][2 * k]) * inv_l;
                    const float hi0 = static_cast<float>(out_acc[s][dt][2 * k + 1]) * inv_l;
                    float vlo = lo0, vhi = hi0;
                    if constexpr (INT8_PV) {
                        vlo = lo0 * v_scale[vs_base + dt * kBK + 4 * k + hw];
                        vhi = hi0 * v_scale[vs_base + dt * kBK + 4 * k + 2 + hw];
                    }
                    if constexpr (std::is_same<OUT_DTYPE, __half>::value) {
                        __half lo = __float2half_rn(vlo);
                        __half hi = __float2half_rn(vhi);
                        pack[k] = (static_cast<unsigned>(__half_as_ushort(hi)) << 16) |
                                  static_cast<unsigned>(__half_as_ushort(lo));
                    } else {
                        __bf16 lo = static_cast<__bf16>(vlo);
                        __bf16 hi = static_cast<__bf16>(vhi);
                        pack[k] = (static_cast<unsigned>(bf16_to_ushort(hi)) << 16) |
                                  static_cast<unsigned>(bf16_to_ushort(lo));
                    }
                }
#pragma unroll
                for (int k = 0; k < 4; ++k) cross[k] = permlanex16_u32(pack[k]);

                const int d = dt * kBK + hw * 8;
                const int64_t o_off = b * o_stride_b + hq * o_stride_h + q_idx * o_stride_n + d;
                if constexpr (std::is_same<OUT_DTYPE, __half>::value) {
                    alignas(16) unsigned b16[8];
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const unsigned even = (hw == 0) ? pack[k] : cross[k];
                        const unsigned odd = (hw == 0) ? cross[k] : pack[k];
                        b16[2 * k] = (even & 0xFFFF) | ((odd & 0xFFFF) << 16);
                        b16[2 * k + 1] = (even >> 16) | (odd & 0xFFFF0000);
                    }
                    *reinterpret_cast<uint4*>(reinterpret_cast<__half*>(output) + o_off) =
                        *reinterpret_cast<const uint4*>(b16 + hw * 4);
                } else {
                    alignas(16) __bf16 b[16];
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const unsigned even = (hw == 0) ? pack[k] : cross[k];
                        const unsigned odd = (hw == 0) ? cross[k] : pack[k];
                        const unsigned short e_lo = static_cast<unsigned short>(even & 0xFFFF);
                        const unsigned short o_lo = static_cast<unsigned short>(odd & 0xFFFF);
                        const unsigned short e_hi = static_cast<unsigned short>(even >> 16);
                        const unsigned short o_hi = static_cast<unsigned short>(odd >> 16);
                        b[4 * k + 0] = ushort_to_bf16(e_lo);
                        b[4 * k + 1] = ushort_to_bf16(o_lo);
                        b[4 * k + 2] = ushort_to_bf16(e_hi);
                        b[4 * k + 3] = ushort_to_bf16(o_hi);
                    }
                    *reinterpret_cast<uint4*>(reinterpret_cast<__bf16*>(output) + o_off) =
                        *reinterpret_cast<const uint4*>(b + hw * 8);
                }
            }
        }
    }
}

template <int HeadDim, int BLOCK_M, int BLOCK_N, typename OUT_DTYPE, bool INT8_PV = false>
__global__ __launch_bounds__(BLOCK_M / 32 * 32, 1) __attribute__((amdgpu_waves_per_eu(1, 1)))
void attn_kernel_wpe1_32_t(
    const int8_t* __restrict__ q, const int8_t* __restrict__ k,
    const std::conditional_t<INT8_PV, int8_t, __half>* __restrict__ v,
    OUT_DTYPE* __restrict__ output, const float* __restrict__ q_scale,
    const float* __restrict__ k_scale, const float* __restrict__ v_scale, int64_t batch_size,
    int64_t qo_len, int64_t kv_len, int64_t num_qo_heads, int64_t num_kv_heads,
    int64_t q_stride_b, int64_t q_stride_n, int64_t q_stride_h, int64_t k_stride_b,
    int64_t k_stride_n, int64_t k_stride_h, int64_t v_stride_b, int64_t v_stride_n,
    int64_t v_stride_h, int64_t o_stride_b, int64_t o_stride_n, int64_t o_stride_h,
    int64_t qs_stride_b, int64_t qs_stride_h, int64_t ks_stride_b, int64_t ks_stride_h,
    float sm_scale) {
    attn_kernel_impl_32_t<HeadDim, BLOCK_M, BLOCK_N, OUT_DTYPE, INT8_PV>(
        q, k, v, output, q_scale, k_scale, v_scale, batch_size, qo_len, kv_len, num_qo_heads,
        num_kv_heads, q_stride_b, q_stride_n, q_stride_h, k_stride_b, k_stride_n, k_stride_h,
        v_stride_b, v_stride_n, v_stride_h, o_stride_b, o_stride_n, o_stride_h, qs_stride_b,
        qs_stride_h, ks_stride_b, ks_stride_h, sm_scale);
}

template <int HeadDim, int BLOCK_M, int BLOCK_N, typename OUT_DTYPE, bool INT8_PV = false>
__device__ __forceinline__ void attn_kernel_impl_t(
    const int8_t* __restrict__ q, const int8_t* __restrict__ k,
    const std::conditional_t<INT8_PV, int8_t, __half>* __restrict__ v,
    OUT_DTYPE* __restrict__ output, const float* __restrict__ q_scale,
    const float* __restrict__ k_scale, const float* __restrict__ v_scale, int64_t batch_size,
    int64_t qo_len, int64_t kv_len, int64_t num_qo_heads, int64_t num_kv_heads,
    int64_t q_stride_b, int64_t q_stride_n, int64_t q_stride_h, int64_t k_stride_b,
    int64_t k_stride_n, int64_t k_stride_h, int64_t v_stride_b, int64_t v_stride_n,
    int64_t v_stride_h, int64_t o_stride_b, int64_t o_stride_n, int64_t o_stride_h,
    int64_t qs_stride_b, int64_t qs_stride_h, int64_t ks_stride_b, int64_t ks_stride_h,
    float sm_scale) {
    constexpr int WARPS = BLOCK_M / kRM;
    constexpr int THREADS = WARPS * 32;
    constexpr int DTiles = HeadDim / kBK;
    constexpr int ColTiles = BLOCK_N / kBK;
    constexpr int KStride = HeadDim + kLdsPad;
    using AccT = std::conditional_t<INT8_PV, v8i, v8f>;

    __shared__ int8_t k_tile[BLOCK_N * KStride];
    constexpr int VTileStride = INT8_PV ? (BLOCK_N + 16) : 80;
    __shared__ __attribute__((aligned(32)))
        std::conditional_t<INT8_PV, int8_t, __half> v_tile[HeadDim * VTileStride];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wave = tid >> 5;
    const int64_t q_base = static_cast<int64_t>(blockIdx.x) * BLOCK_M;
    const int64_t hq = blockIdx.y;
    const int64_t b = blockIdx.z;
    const int64_t hkv = num_kv_heads > 1 ? (hq / (num_qo_heads / num_kv_heads)) : 0;
    if (hq >= num_qo_heads || q_base >= qo_len) return;
    const int64_t q_start = q_base + static_cast<int64_t>(wave) * kRM;

    int32_t_v4 q_frag[DTiles];
    const int q_row = lane & 15;
    const int64_t q_idx_q = q_start + q_row;
#pragma unroll
    for (int dt = 0; dt < DTiles; ++dt) {
        if (q_idx_q < qo_len) {
            const int64_t q_off = b * q_stride_b + hq * q_stride_h + q_idx_q * q_stride_n +
                                  dt * kBK;
            q_frag[dt] = *reinterpret_cast<const int32_t_v4*>(q + q_off);
        } else {
            q_frag[dt] = int32_t_v4{0, 0, 0, 0};
        }
    }

    const float qs_lane =
        (q_idx_q < qo_len) ? q_scale[b * qs_stride_b + hq * qs_stride_h + q_idx_q] : 0.0f;

    AccT out_acc[DTiles];
#pragma unroll
    for (int dt = 0; dt < DTiles; ++dt) {
        if constexpr (INT8_PV) {
            out_acc[dt] = v8i{0, 0, 0, 0, 0, 0, 0, 0};
        } else {
            out_acc[dt] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
        }
    }

    float row_m = -FLT_MAX * 0.5f;
    float row_l = 0.0f;

    const int64_t kv_limit = kv_len;
    const int64_t v_t_n = ((kv_len + 63) / 64) * 64;
    constexpr int KVecsPerRow = HeadDim / 16;
    constexpr int KVecsTotal = BLOCK_N * KVecsPerRow;
    constexpr int KPrefetchPerThread = (KVecsTotal + THREADS - 1) / THREADS;

    uint4 k_prefetch[KPrefetchPerThread];
    for (int i = tid; i < KVecsTotal; i += THREADS) {
        const int n = i / KVecsPerRow;
        const int d = (i - n * KVecsPerRow) * 16;
        if (n < kv_len) {
            const int64_t k_off = b * k_stride_b + hkv * k_stride_h + n * k_stride_n + d;
            *reinterpret_cast<uint4*>(&k_tile[n * KStride + d]) =
                *reinterpret_cast<const uint4*>(k + k_off);
        } else {
            *reinterpret_cast<uint4*>(&k_tile[n * KStride + d]) = make_uint4(0, 0, 0, 0);
        }
    }
    __syncthreads();

    const int hw = lane >> 4;
    const int m_row = lane & 15;

    constexpr int VVecsPerTile = (HeadDim * BLOCK_N) / 16;

    // Loop-invariant address bases. Rebuilding these int64 expressions per
    // element left the D64 key-tile loop carrying ~556 scalar instructions (see
    // the ISA histogram in the optimization log); factoring them out lets the
    // compiler strength-reduce the per-element offsets to constant adds.
    const int64_t k_base = b * k_stride_b + hkv * k_stride_h;
    const int64_t ks_base = b * ks_stride_b + hkv * ks_stride_h;
    const int64_t v_head_base = (b * num_kv_heads + hkv) * HeadDim;
    const int64_t v_row_base = v_head_base + m_row;

    for (int64_t kb_base = 0; kb_base < kv_limit; kb_base += BLOCK_N) {
        const int64_t next_base = kb_base + BLOCK_N;
        const bool has_next = (next_base < kv_limit);

        if (has_next) {
#pragma unroll
            for (int i = 0; i < KPrefetchPerThread; ++i) {
                const int vec = tid + i * THREADS;
                if (vec < KVecsTotal) {
                    const int n = vec / KVecsPerRow;
                    const int d = (vec - n * KVecsPerRow) * 16;
                    const int64_t k_idx = next_base + n;
                    if (k_idx < kv_len) {
                        const int64_t k_off = k_base + k_idx * k_stride_n + d;
                        k_prefetch[i] = *reinterpret_cast<const uint4*>(k + k_off);
                    } else {
                        k_prefetch[i] = make_uint4(0, 0, 0, 0);
                    }
                }
            }
        }

        float score_cache[ColTiles][8];
#pragma unroll
        for (int ct = 0; ct < ColTiles; ++ct) {
            v8i score_acc = v8i{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int dt = 0; dt < DTiles; ++dt) {
                const int32_t_v4 k_frag =
                    load_i8_frag(k_tile + ct * kBK * KStride, m_row, dt * kBK, KStride);
                score_acc = wmma_i32_iu8(k_frag, q_frag[dt], score_acc);
            }
            const int64_t k_col_start = kb_base + ct * kBK;
            const int k_scale_idx = static_cast<int>(k_col_start / kMinBlkK);
            const float ks_val = k_scale[ks_base + k_scale_idx];
            const float score_scale = qs_lane * ks_val * sm_scale;
            if (k_col_start + kBK <= kv_len) {
                // Whole 16-key group is in bounds, which is every tile but the
                // last for the aligned key lengths the models use: skip the
                // per-element mask compare and select.
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    score_cache[ct][e] = static_cast<float>(score_acc[e]) * score_scale;
                }
            } else {
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int col = static_cast<int>(k_col_start) + 2 * e + hw;
                    float s = static_cast<float>(score_acc[e]) * score_scale;
                    if (col >= kv_len) s = -FLT_MAX * 0.5f;
                    score_cache[ct][e] = s;
                }
            }
        }

        if constexpr (BLOCK_M >= 128) {
#pragma unroll
            for (int i = tid; i < VVecsPerTile; i += THREADS) {
                const int d = i / (BLOCK_N / kBK);
                const int c16 = i % (BLOCK_N / kBK);
                const int64_t v_off = (v_head_base + d) * v_t_n + (kb_base + c16 * kBK);
                if constexpr (INT8_PV) {
                    *reinterpret_cast<uint4*>(v_tile + d * VTileStride + c16 * kBK) =
                        *reinterpret_cast<const uint4*>(v + v_off);
                } else {
                    const v16h src = *reinterpret_cast<const v16h*>(v + v_off);
                    *reinterpret_cast<v16h*>(reinterpret_cast<char*>(v_tile) +
                                             (d * VTileStride + c16 * kBK) * 2) = src;
                }
            }
            __syncthreads();
        }

        float local_mx = score_cache[0][0];
#pragma unroll
        for (int ct = 0; ct < ColTiles; ++ct) {
#pragma unroll
            for (int e = 0; e < 8; ++e) local_mx = fmaxf(local_mx, score_cache[ct][e]);
        }
        float gm = fmaxf(row_m, local_mx);
        gm = fmaxf(gm, permlanex16(gm));
        const float alpha = (row_l == 0.0f) ? 0.0f : fast_exp2(row_m - gm);
        row_m = gm;
        row_l *= alpha;
#pragma unroll
        for (int dt = 0; dt < DTiles; ++dt) {
            if constexpr (INT8_PV) {
#pragma unroll
                for (int e = 0; e < 8; ++e)
                    out_acc[dt][e] = static_cast<int>(static_cast<float>(out_acc[dt][e]) * alpha);
            } else {
                out_acc[dt] *= alpha;
            }
        }
        // The int8 path shifts the max inside the exponent so exp2 fills the
        // unsigned INT8 range; the running max stays in score space (offsetting
        // row_m would drift it down whenever a tile's local max trails it).
#pragma unroll
        for (int ct = 0; ct < ColTiles; ++ct) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                score_cache[ct][e] =
                    fast_exp2(score_cache[ct][e] - gm + (INT8_PV ? kProbU8Offset : 0.0f));
            }
        }

        float partial_sm = 0.0f;
#pragma unroll
        for (int ct = 0; ct < ColTiles; ++ct) {
#pragma unroll
            for (int e = 0; e < 8; ++e) partial_sm += score_cache[ct][e];
        }
        row_l += partial_sm + permlanex16(partial_sm);

        if constexpr (BLOCK_M >= 128) {
#pragma unroll
            for (int ct = 0; ct < ColTiles; ++ct) {
                float p_vals[8];
#pragma unroll
                for (int e = 0; e < 8; ++e) p_vals[e] = score_cache[ct][e];
                if constexpr (INT8_PV) {
                    const int32_t_v4 p_frag = assemble_p_frag_u8(p_vals, hw);
#pragma unroll
                    for (int dt = 0; dt < DTiles; ++dt) {
                        const int d_row = dt * kBK + m_row;
                        const int32_t_v4 v_frag =
                            load_i8_frag(v_tile, d_row, ct * kBK, VTileStride);
                        out_acc[dt] = wmma_i32_iu8_ub(v_frag, p_frag, out_acc[dt]);
                    }
                } else {
                    const v16h p_frag = assemble_p_frag(p_vals, hw);
#pragma unroll
                    for (int dt = 0; dt < DTiles; ++dt) {
                        const int d_row = dt * kBK + m_row;
                        const v16h* vp = reinterpret_cast<const v16h*>(
                            reinterpret_cast<const char*>(v_tile) +
                            (d_row * VTileStride + ct * kBK) * 2);
                        out_acc[dt] = wmma_f32_f16(*vp, p_frag, out_acc[dt]);
                    }
                }
            }
        } else {
            // V straight from the transposed global buffer. The row base is
            // hoisted out of the ct loop so the per-MMA address is a constant
            // add instead of an int64 multiply chain.
            const auto* v_row_ptr = v + v_row_base * v_t_n + kb_base;
#pragma unroll
            for (int ct = 0; ct < ColTiles; ++ct) {
                float p_vals[8];
#pragma unroll
                for (int e = 0; e < 8; ++e) p_vals[e] = score_cache[ct][e];
                if constexpr (INT8_PV) {
                    const int32_t_v4 p_frag = assemble_p_frag_u8(p_vals, hw);
#pragma unroll
                    for (int dt = 0; dt < DTiles; ++dt) {
                        const int32_t_v4 v_frag = *reinterpret_cast<const int32_t_v4*>(
                            v_row_ptr + (dt * kBK) * v_t_n + ct * kBK);
                        out_acc[dt] = wmma_i32_iu8_ub(v_frag, p_frag, out_acc[dt]);
                    }
                } else {
                    const v16h p_frag = assemble_p_frag(p_vals, hw);
#pragma unroll
                    for (int dt = 0; dt < DTiles; ++dt) {
                        const v16h v_frag_t = *reinterpret_cast<const v16h*>(
                            v_row_ptr + (dt * kBK) * v_t_n + ct * kBK);
                        out_acc[dt] = wmma_f32_f16(v_frag_t, p_frag, out_acc[dt]);
                    }
                }
            }
        }

        if (has_next) {
            __syncthreads();
#pragma unroll
            for (int i = 0; i < KPrefetchPerThread; ++i) {
                const int vec = tid + i * THREADS;
                if (vec < KVecsTotal) {
                    const int n = vec / KVecsPerRow;
                    const int d = (vec - n * KVecsPerRow) * 16;
                    *reinterpret_cast<uint4*>(&k_tile[n * KStride + d]) = k_prefetch[i];
                }
            }
            __syncthreads();
        }
    }

    const int64_t q_idx = q_start + m_row;
    if (q_idx < qo_len) {
        const float inv_l = (row_l > 0.0f) ? (1.0f / row_l) : 0.0f;
        const int64_t vs_base = (b * num_kv_heads + hkv) * HeadDim;
#pragma unroll
        for (int dt = 0; dt < DTiles; ++dt) {
            unsigned pack[4], cross[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float lo0 = static_cast<float>(out_acc[dt][2 * k]) * inv_l;
                const float hi0 = static_cast<float>(out_acc[dt][2 * k + 1]) * inv_l;
                float vlo = lo0, vhi = hi0;
                if constexpr (INT8_PV) {
                    // acc element e holds row (2e + hw) of the dt tile; fold the
                    // per-channel V scale in here.
                    vlo = lo0 * v_scale[vs_base + dt * kBK + 4 * k + hw];
                    vhi = hi0 * v_scale[vs_base + dt * kBK + 4 * k + 2 + hw];
                }
                if constexpr (std::is_same<OUT_DTYPE, __half>::value) {
                    __half lo = __float2half_rn(vlo);
                    __half hi = __float2half_rn(vhi);
                    pack[k] = (static_cast<unsigned>(__half_as_ushort(hi)) << 16) |
                              static_cast<unsigned>(__half_as_ushort(lo));
                } else {
                    __bf16 lo = static_cast<__bf16>(vlo);
                    __bf16 hi = static_cast<__bf16>(vhi);
                    pack[k] = (static_cast<unsigned>(bf16_to_ushort(hi)) << 16) |
                              static_cast<unsigned>(bf16_to_ushort(lo));
                }
            }
#pragma unroll
            for (int k = 0; k < 4; ++k) cross[k] = permlanex16_u32(pack[k]);

            const int d = dt * kBK + hw * 8;
            const int64_t o_off = b * o_stride_b + hq * o_stride_h + q_idx * o_stride_n + d;
            if constexpr (std::is_same<OUT_DTYPE, __half>::value) {
                alignas(16) unsigned b16[8];
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const unsigned even = (hw == 0) ? pack[k] : cross[k];
                    const unsigned odd = (hw == 0) ? cross[k] : pack[k];
                    b16[2 * k] = (even & 0xFFFF) | ((odd & 0xFFFF) << 16);
                    b16[2 * k + 1] = (even >> 16) | (odd & 0xFFFF0000);
                }
                *reinterpret_cast<uint4*>(reinterpret_cast<__half*>(output) + o_off) =
                    *reinterpret_cast<const uint4*>(b16 + hw * 4);
            } else {
                alignas(16) __bf16 b[16];
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const unsigned even = (hw == 0) ? pack[k] : cross[k];
                    const unsigned odd = (hw == 0) ? cross[k] : pack[k];
                    const unsigned short e_lo = static_cast<unsigned short>(even & 0xFFFF);
                    const unsigned short o_lo = static_cast<unsigned short>(odd & 0xFFFF);
                    const unsigned short e_hi = static_cast<unsigned short>(even >> 16);
                    const unsigned short o_hi = static_cast<unsigned short>(odd >> 16);
                    b[4 * k + 0] = ushort_to_bf16(e_lo);
                    b[4 * k + 1] = ushort_to_bf16(o_lo);
                    b[4 * k + 2] = ushort_to_bf16(e_hi);
                    b[4 * k + 3] = ushort_to_bf16(o_hi);
                }
                *reinterpret_cast<uint4*>(reinterpret_cast<__bf16*>(output) + o_off) =
                    *reinterpret_cast<const uint4*>(b + hw * 8);
            }
        }
    }
}

template <int HeadDim, int BLOCK_M, int BLOCK_N, typename OUT_DTYPE, bool INT8_PV = false>
__global__ __launch_bounds__(BLOCK_M / 16 * 32, 1) __attribute__((amdgpu_waves_per_eu(1, 1)))
void attn_kernel_wpe1_t(
    const int8_t* __restrict__ q, const int8_t* __restrict__ k,
    const std::conditional_t<INT8_PV, int8_t, __half>* __restrict__ v,
    OUT_DTYPE* __restrict__ output, const float* __restrict__ q_scale,
    const float* __restrict__ k_scale, const float* __restrict__ v_scale, int64_t batch_size,
    int64_t qo_len, int64_t kv_len, int64_t num_qo_heads, int64_t num_kv_heads,
    int64_t q_stride_b, int64_t q_stride_n, int64_t q_stride_h, int64_t k_stride_b,
    int64_t k_stride_n, int64_t k_stride_h, int64_t v_stride_b, int64_t v_stride_n,
    int64_t v_stride_h, int64_t o_stride_b, int64_t o_stride_n, int64_t o_stride_h,
    int64_t qs_stride_b, int64_t qs_stride_h, int64_t ks_stride_b, int64_t ks_stride_h,
    float sm_scale) {
    attn_kernel_impl_t<HeadDim, BLOCK_M, BLOCK_N, OUT_DTYPE, INT8_PV>(
        q, k, v, output, q_scale, k_scale, v_scale, batch_size, qo_len, kv_len, num_qo_heads,
        num_kv_heads, q_stride_b, q_stride_n, q_stride_h, k_stride_b, k_stride_n, k_stride_h,
        v_stride_b, v_stride_n, v_stride_h, o_stride_b, o_stride_n, o_stride_h, qs_stride_b,
        qs_stride_h, ks_stride_b, ks_stride_h, sm_scale);
}

template <int HeadDim, int BLOCK_M, int BLOCK_N, typename OUT_DTYPE, bool INT8_PV = false>
__global__ __launch_bounds__(BLOCK_M / 16 * 32, 2) __attribute__((amdgpu_waves_per_eu(2, 2)))
void attn_kernel_wpe2_t(
    const int8_t* __restrict__ q, const int8_t* __restrict__ k,
    const std::conditional_t<INT8_PV, int8_t, __half>* __restrict__ v,
    OUT_DTYPE* __restrict__ output, const float* __restrict__ q_scale,
    const float* __restrict__ k_scale, const float* __restrict__ v_scale, int64_t batch_size,
    int64_t qo_len, int64_t kv_len, int64_t num_qo_heads, int64_t num_kv_heads,
    int64_t q_stride_b, int64_t q_stride_n, int64_t q_stride_h, int64_t k_stride_b,
    int64_t k_stride_n, int64_t k_stride_h, int64_t v_stride_b, int64_t v_stride_n,
    int64_t v_stride_h, int64_t o_stride_b, int64_t o_stride_n, int64_t o_stride_h,
    int64_t qs_stride_b, int64_t qs_stride_h, int64_t ks_stride_b, int64_t ks_stride_h,
    float sm_scale) {
    attn_kernel_impl_t<HeadDim, BLOCK_M, BLOCK_N, OUT_DTYPE, INT8_PV>(
        q, k, v, output, q_scale, k_scale, v_scale, batch_size, qo_len, kv_len, num_qo_heads,
        num_kv_heads, q_stride_b, q_stride_n, q_stride_h, k_stride_b, k_stride_n, k_stride_h,
        v_stride_b, v_stride_n, v_stride_h, o_stride_b, o_stride_n, o_stride_h, qs_stride_b,
        qs_stride_h, ks_stride_b, ks_stride_h, sm_scale);
}

template <int HeadDim, int BLOCK_M, int BLOCK_N, typename OUT_DTYPE, bool INT8_PV = false>
__global__ __launch_bounds__(BLOCK_M / 16 * 32, 4) __attribute__((amdgpu_waves_per_eu(4, 4)))
void attn_kernel_wpe4_t(
    const int8_t* __restrict__ q, const int8_t* __restrict__ k,
    const std::conditional_t<INT8_PV, int8_t, __half>* __restrict__ v,
    OUT_DTYPE* __restrict__ output, const float* __restrict__ q_scale,
    const float* __restrict__ k_scale, const float* __restrict__ v_scale, int64_t batch_size,
    int64_t qo_len, int64_t kv_len, int64_t num_qo_heads, int64_t num_kv_heads,
    int64_t q_stride_b, int64_t q_stride_n, int64_t q_stride_h, int64_t k_stride_b,
    int64_t k_stride_n, int64_t k_stride_h, int64_t v_stride_b, int64_t v_stride_n,
    int64_t v_stride_h, int64_t o_stride_b, int64_t o_stride_n, int64_t o_stride_h,
    int64_t qs_stride_b, int64_t qs_stride_h, int64_t ks_stride_b, int64_t ks_stride_h,
    float sm_scale) {
    attn_kernel_impl_t<HeadDim, BLOCK_M, BLOCK_N, OUT_DTYPE, INT8_PV>(
        q, k, v, output, q_scale, k_scale, v_scale, batch_size, qo_len, kv_len, num_qo_heads,
        num_kv_heads, q_stride_b, q_stride_n, q_stride_h, k_stride_b, k_stride_n, k_stride_h,
        v_stride_b, v_stride_n, v_stride_h, o_stride_b, o_stride_n, o_stride_h, qs_stride_b,
        qs_stride_h, ks_stride_b, ks_stride_h, sm_scale);
}

}  // namespace
}  // namespace comfy::hip_backend::sage

using namespace comfy::hip_backend::sage;

extern "C" void launch_sage_direct_attn(
    const void* q, const void* k, const void* v, void* o, int64_t q_stride_b,
    int64_t q_stride_h, int64_t q_stride_n, int64_t k_stride_b, int64_t k_stride_h,
    int64_t k_stride_n, int64_t v_stride_b, int64_t v_stride_h, int64_t v_stride_d,
    int64_t o_stride_b, int64_t o_stride_h, int64_t o_stride_n, int batch, int qo_len,
    int kv_len, int num_qo_heads, int num_kv_heads, int head_dim, float sm_scale,
    int dtype_code, hipStream_t stream) {
    if (reinterpret_cast<uintptr_t>(q) % 16 != 0 || reinterpret_cast<uintptr_t>(k) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(v) % 16 != 0 || reinterpret_cast<uintptr_t>(o) % 16 != 0) {
        throw std::runtime_error(
            "sage direct attention: q, k, v and o base pointers must be 16-byte aligned");
    }
    sm_scale *= kLog2e;

#define LAUNCH_DIRECT(HD, BM, BN, DT)                                                            \
    {                                                                                           \
        const dim3 grid(div_ceil_i64(qo_len, BM), num_qo_heads, batch);                          \
        direct_attn_kernel_wpe2_t<HD, BM, BN, DT, DT>                                           \
            <<<grid, BM / 16 * 32, 0, stream>>>(                                                \
                static_cast<const DT*>(q), static_cast<const DT*>(k),                            \
                static_cast<const __half*>(v), static_cast<DT*>(o), batch, qo_len, kv_len,       \
                num_qo_heads, num_kv_heads, q_stride_b, q_stride_n, q_stride_h, k_stride_b,      \
                k_stride_n, k_stride_h, v_stride_b, v_stride_d, v_stride_h, o_stride_b,          \
                o_stride_n, o_stride_h, sm_scale);                                               \
    }

    if (dtype_code == 1) {  // fp16
        static const int ebm = []{ const char* s = getenv("SAGE_DIRECT_BM"); return s ? atoi(s) : 0; }();
        static const int ebn = []{ const char* s = getenv("SAGE_DIRECT_BN"); return s ? atoi(s) : 0; }();
        if (ebm != 0 && ebn != 0) {
            if (ebm == 128 && ebn == 64) {
                LAUNCH_DIRECT(64, 128, 64, __half);
            } else if (ebm == 128 && ebn == 32) {
                LAUNCH_DIRECT(64, 128, 32, __half);
            } else if (ebm == 64 && ebn == 32) {
                LAUNCH_DIRECT(64, 64, 32, __half);
            } else if (ebm == 64 && ebn == 128) {
                LAUNCH_DIRECT(64, 64, 128, __half);
            } else if (ebm == 128 && ebn == 128) {
                LAUNCH_DIRECT(64, 128, 128, __half);
            } else {
                LAUNCH_DIRECT(64, 64, 64, __half);
            }
        } else if (head_dim == 64) {
            if (kv_len <= 128) {
                LAUNCH_DIRECT(64, 64, 16, __half);
            } else if (qo_len == kv_len) {
                LAUNCH_DIRECT(64, 64, 64, __half);
            } else {
                LAUNCH_DIRECT(64, 128, 32, __half);
            }
        } else {
            LAUNCH_DIRECT(128, 64, 16, __half);
        }
    } else {  // bf16
        if (head_dim == 64) {
            if (kv_len <= 128) {
                LAUNCH_DIRECT(64, 64, 16, __bf16);
            } else if (qo_len == kv_len) {
                LAUNCH_DIRECT(64, 64, 64, __bf16);
            } else {
                LAUNCH_DIRECT(64, 128, 32, __bf16);
            }
        } else {
            LAUNCH_DIRECT(128, 64, 16, __bf16);
        }
    }
#undef LAUNCH_DIRECT

    const hipError_t error = hipGetLastError();
    if (error != hipSuccess) {
        throw std::runtime_error(std::string("sage direct attention launch failed: ") +
                                 hipGetErrorString(error));
    }
}

extern "C" void launch_sage_port_attn(
    const void* q, const void* k, const void* v, void* o, const void* q_scale,
    const void* k_scale, const void* v_scale, const void* mask, int64_t mask_stride_b,
    int64_t mask_stride_h, int64_t mask_stride_q, int64_t mask_stride_k, int mask_dtype_code,
    int cta_k, int batch, int qo_len, int kv_len, int qo_len_padded, int num_qo_heads,
    int num_kv_heads, int head_dim, int k_groups_per_head, int64_t q_stride_b, int64_t q_stride_h,
    int64_t k_stride_b, int64_t k_stride_h, int64_t v_stride_b, int64_t v_stride_h,
    int64_t v_stride_d, int64_t o_stride_b, int64_t o_stride_h, int64_t o_stride_n, float sm_scale,
    int output_dtype_code, int v_dtype_code, hipStream_t stream) {
    if (mask != nullptr || head_dim == 256) {
        if (v_dtype_code == 4) {
            throw std::runtime_error(
                "sage port attention: int8 V with a mask must go through the legacy kernel");
        }
        launch_sage_bf16sv_attn(
            q, k, v, o, q_scale, k_scale, mask, mask_stride_b, mask_stride_h, mask_stride_q,
            mask_stride_k, mask_dtype_code, cta_k, batch, qo_len, kv_len, qo_len_padded,
            num_qo_heads, num_kv_heads, head_dim, k_groups_per_head, q_stride_b, q_stride_h,
            k_stride_b, k_stride_h, v_stride_b, v_stride_h, v_stride_d, o_stride_b, o_stride_h,
            o_stride_n, sm_scale, output_dtype_code, 1 /* fp16 V */, stream);
        return;
    }

    if (reinterpret_cast<uintptr_t>(q) % 16 != 0 || reinterpret_cast<uintptr_t>(k) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(v) % 16 != 0 || reinterpret_cast<uintptr_t>(o) % 16 != 0) {
        throw std::runtime_error(
            "sage port attention: q, k, v and o base pointers must be 16-byte aligned");
    }

    const bool int8_pv = (v_dtype_code == 4);
    const auto* qi = static_cast<const int8_t*>(q);
    const auto* ki = static_cast<const int8_t*>(k);
    const auto* qs = static_cast<const float*>(q_scale);
    const auto* ks = static_cast<const float*>(k_scale);
    const auto* vs = static_cast<const float*>(v_scale);

    sm_scale *= kLog2e;

    const int64_t qs_stride_b = static_cast<int64_t>(num_qo_heads) * qo_len_padded;
    const int64_t qs_stride_h = qo_len_padded;
    const int64_t ks_stride_b = static_cast<int64_t>(num_kv_heads) * k_groups_per_head;
    const int64_t ks_stride_h = k_groups_per_head;

    // Shape-differentiated dispatch. Environment overrides (SAGE_PORT_WPE /
    // SAGE_PORT_BM / SAGE_PORT_BN / SAGE_PORT_T32) force one config for sweeping;
    // the defaults encode the measured best per shape class. Read per call, not
    // once per process, so a sweep can interleave configs in a single process
    // (cross-process timing on this iGPU drifts with the clock).
    const int env_wpe = [] {
        const char* s = getenv("SAGE_PORT_WPE");
        return s ? atoi(s) : 0;
    }();
    const int env_bm = [] {
        const char* s = getenv("SAGE_PORT_BM");
        return s ? atoi(s) : 0;
    }();
    const int env_t32 = [] { const char* s = getenv("SAGE_PORT_T32"); return s ? atoi(s) : -1; }();
    const bool use_t32 = (env_t32 == 1 || (env_t32 == -1 && head_dim != 64));
    const int env_bn = [] {
        const char* s = getenv("SAGE_PORT_BN");
        return s ? atoi(s) : 0;
    }();
    const bool forced = (env_wpe != 0);
    const int wpe = forced ? env_wpe : 0;  // resolved per shape below

#define LAUNCH_PORT_IMPL(HD, BM, BN, OUT, I8PV, KERNEL)                                         \
    {                                                                                           \
        const dim3 grid(div_ceil_i64(qo_len, BM), num_qo_heads, batch);                          \
        using VT = std::conditional_t<I8PV, const int8_t*, const __half*>;                       \
        const VT vtyped = static_cast<VT>(v);                                                   \
        KERNEL<HD, BM, BN, OUT, I8PV>                                                           \
            <<<grid, BM / 16 * 32, 0, stream>>>(                                                \
                qi, ki, vtyped, static_cast<OUT*>(o), qs, ks, vs, batch, qo_len, kv_len,         \
                num_qo_heads, num_kv_heads, q_stride_b, head_dim, q_stride_h, k_stride_b,        \
                head_dim, k_stride_h, v_stride_b, v_stride_d, v_stride_h, o_stride_b,            \
                o_stride_n, o_stride_h, qs_stride_b, qs_stride_h, ks_stride_b, ks_stride_h,      \
                sm_scale);                                                                       \
    }
#define LAUNCH_PORT_32(HD, BM, BN, OUT, I8PV)                                                   \
    {                                                                                           \
        const dim3 grid(div_ceil_i64(qo_len, BM), num_qo_heads, batch);                          \
        using VT = std::conditional_t<I8PV, const int8_t*, const __half*>;                       \
        const VT vtyped = static_cast<VT>(v);                                                   \
        comfy::hip_backend::sage::attn_kernel_wpe1_32_t<HD, BM, BN, OUT, I8PV>                  \
            <<<grid, BM / 32 * 32, 0, stream>>>(                                                \
                qi, ki, vtyped, static_cast<OUT*>(o), qs, ks, vs, batch, qo_len, kv_len,         \
                num_qo_heads, num_kv_heads, q_stride_b, head_dim, q_stride_h, k_stride_b,        \
                head_dim, k_stride_h, v_stride_b, v_stride_d, v_stride_h, o_stride_b,            \
                o_stride_n, o_stride_h, qs_stride_b, qs_stride_h, ks_stride_b, ks_stride_h,      \
                sm_scale);                                                                       \
    }
#define LAUNCH_PORT(HD, BM, BN, OUT, I8PV)                                                      \
    {                                                                                           \
        if (wpe == 1) {                                                                          \
            LAUNCH_PORT_IMPL(HD, BM, BN, OUT, I8PV, comfy::hip_backend::sage::attn_kernel_wpe1_t); \
        } else if (wpe == 4) {                                                                   \
            LAUNCH_PORT_IMPL(HD, BM, BN, OUT, I8PV, comfy::hip_backend::sage::attn_kernel_wpe4_t); \
        } else {                                                                                 \
            LAUNCH_PORT_IMPL(HD, BM, BN, OUT, I8PV, comfy::hip_backend::sage::attn_kernel_wpe2_t); \
        }                                                                                        \
    }

    const bool out_half = (output_dtype_code == 1);

    // Resolve (BM, BN): forced by env for sweeping, else per-shape heuristics.
    int bm_sel = 0, bn_sel = 0;
    if (forced) {
        bm_sel = env_bm ? env_bm : 64;
        bn_sel = env_bn ? env_bn : 64;
    } else {
        if (head_dim == 128) {
            if (kv_len % 128 == 0 && kv_len >= 2048) {
                bm_sel = 128;
                bn_sel = 64;
            } else {
                bm_sel = 64;
                bn_sel = 64;
            }
        } else if (qo_len == kv_len) {
            // D64 self runs the 128-thread 32_t kernel: BN 64 for long keys,
            // BN 32 for the short self shapes.
            bm_sel = 64;
            bn_sel = (kv_len >= 4096) ? 64 : 32;
        } else {
            // D64 cross: 32-key tiles on the 32_t kernel.
            bm_sel = 64;
            bn_sel = 32;
        }
    }

    auto launch_d128 = [&]<bool I8PV>(int bm, int bn) {
        if (bm == 128) {
            if (bn == 64) {
                if (out_half) {
                    LAUNCH_PORT(128, 128, 64, __half, I8PV);
                } else {
                    LAUNCH_PORT(128, 128, 64, __bf16, I8PV);
                }
            } else if (bn == 32) {
                if (out_half) {
                    LAUNCH_PORT(128, 128, 32, __half, I8PV);
                } else {
                    LAUNCH_PORT(128, 128, 32, __bf16, I8PV);
                }
            } else {
                if (out_half) {
                    LAUNCH_PORT(128, 128, 16, __half, I8PV);
                } else {
                    LAUNCH_PORT(128, 128, 16, __bf16, I8PV);
                }
            }
        } else {
            if (bn == 64) {
                if (out_half) {
                    LAUNCH_PORT(128, 64, 64, __half, I8PV);
                } else {
                    LAUNCH_PORT(128, 64, 64, __bf16, I8PV);
                }
            } else if (bn == 32) {
                if (out_half) {
                    LAUNCH_PORT(128, 64, 32, __half, I8PV);
                } else {
                    LAUNCH_PORT(128, 64, 32, __bf16, I8PV);
                }
            } else {
                if (out_half) {
                    LAUNCH_PORT(128, 64, 16, __half, I8PV);
                } else {
                    LAUNCH_PORT(128, 64, 16, __bf16, I8PV);
                }
            }
        }
    };
    auto launch_d64 = [&]<bool I8PV>(int bm, int bn) {
        if (bm == 128) {
            if (bn == 64) {
                if (out_half) {
                    if (use_t32) { LAUNCH_PORT_32(64, 128, 64, __half, I8PV); } else { LAUNCH_PORT(64, 128, 64, __half, I8PV); }
                } else {
                    LAUNCH_PORT_32(64, 128, 64, __bf16, I8PV);
                }
            } else if (bn == 32) {
                if (out_half) {
                    if (use_t32) { LAUNCH_PORT_32(64, 128, 32, __half, I8PV); } else { LAUNCH_PORT(64, 128, 32, __half, I8PV); }
                } else {
                    LAUNCH_PORT_32(64, 128, 32, __bf16, I8PV);
                }
            } else {
                if (out_half) {
                    LAUNCH_PORT_32(64, 128, 16, __half, I8PV);
                } else {
                    LAUNCH_PORT_32(64, 128, 16, __bf16, I8PV);
                }
            }
        } else {
            if (bn == 64) {
                if (out_half) {
                    if (use_t32) { LAUNCH_PORT_32(64, 64, 64, __half, I8PV); } else { LAUNCH_PORT(64, 64, 64, __half, I8PV); }
                } else {
                    LAUNCH_PORT_32(64, 64, 64, __bf16, I8PV);
                }
            } else if (bn == 32) {
                if (out_half) {
                    if (use_t32) { LAUNCH_PORT_32(64, 64, 32, __half, I8PV); } else { LAUNCH_PORT(64, 64, 32, __half, I8PV); }
                } else {
                    LAUNCH_PORT_32(64, 64, 32, __bf16, I8PV);
                }
            } else {
                if (out_half) {
                    if (use_t32) { LAUNCH_PORT_32(64, 64, 16, __half, I8PV); } else { LAUNCH_PORT(64, 64, 16, __half, I8PV); }
                } else {
                    LAUNCH_PORT_32(64, 64, 16, __bf16, I8PV);
                }
            }
        }
    };

    if (head_dim == 128) {
        if (int8_pv) { launch_d128.template operator()<true>(bm_sel, bn_sel); } else { launch_d128.template operator()<false>(bm_sel, bn_sel); }
    } else {
        if (int8_pv) { launch_d64.template operator()<true>(bm_sel, bn_sel); } else { launch_d64.template operator()<false>(bm_sel, bn_sel); }
    }
#undef LAUNCH_PORT_IMPL
#undef LAUNCH_PORT

    const hipError_t error = hipGetLastError();
    if (error != hipSuccess) {
        throw std::runtime_error(std::string("sage port attention launch failed: ") +
                                 hipGetErrorString(error));
    }
}




