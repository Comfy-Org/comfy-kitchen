// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Port of the SageAttention gfx110x **direct** (short-key, no int8 quantization)
// attention kernel -- attn_gfx110x.cu's direct_attn_kernel_impl_t and the wpe
// wrappers. QK runs on the f16/bf16 matrix core straight from the inputs, V on
// f16 WMMA from the transposed buffer, skipping the int8 prepass entirely.
//
// Adapted to comfy_kitchen: HND layout only, no causal flag, and the output is
// taken through explicit strides so the iGPU caller can hand back an NHD-packed
// buffer (upstream's layout forces a transpose and copy).
//
// This file used to also carry a port of the *ported int8* schedule
// (attn_kernel_impl_t and its 32-query twin) for gfx110x. It was removed after
// measuring it against upstream's int8_attn.hip on the 18 shapes
// benchmark_attn.py exercises, interleaved in one process with min-of-rounds
// sampling and a ~1% null control: it was 1.10x-1.40x SLOWER on every case it
// served, 0 wins / 11 losses / 7 ties overall. The direct kernel below was kept
// because it wins 0.64x on the six SDXL cross-attention cases (kv=77, where the
// int8 prepass dominates) and 1.15x-1.42x on the short-key shapes it targets.
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

// The wave32 WMMA builtins below are gfx11-only: gfx10 (no WMMA at all) and
// gfx12 (its WMMA builtins carry a _gfx12 suffix) reject them at compile time,
// which would break any multi-arch build that includes those targets. The
// runtime dispatch only ever launches this kernel on gfx1103
// (comfy_small_igpu gates every path here), so the other targets get trap
// stubs: they compile, they link, and -- reaching them at all -- fault loudly
// instead of silently accumulating zeros. The host pass compiles the real body
// regardless (device builtins are not target-checked there), keeping the
// extern "C" launchers below defined for the binding.
#if defined(COMFY_MMA_GFX11)
__device__ __forceinline__ v8f wmma_f32_f16(v16h a, v16h b, v8f c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

__device__ __forceinline__ v8f wmma_f32_bf16(v16bf a, v16bf b, v8f c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
}
#else
__device__ __forceinline__ v8f wmma_f32_f16(v16h, v16h, v8f c) {
    __builtin_trap();  // unreachable: launch_sage_direct_attn is gfx1103-only
    return c;
}
__device__ __forceinline__ v8f wmma_f32_bf16(v16bf, v16bf, v8f c) {
    __builtin_trap();
    return c;
}
#endif

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
            // Two 16-byte loads instead of 16 scalar vload1s. q_off is a
            // multiple of kBK == 16 elements (32 bytes), and the launcher
            // asserts a 16-byte-aligned base, so both halves are aligned.
            const QK_DTYPE* src = q + b * q_stride_b + hq * q_stride_h +
                                  q_idx_q * q_stride_n + dt * kBK;
            QK_VEC loaded;
            *reinterpret_cast<uint4*>(&loaded) =
                *reinterpret_cast<const uint4*>(src);
            *reinterpret_cast<uint4*>((char*)&loaded + 16) =
                *reinterpret_cast<const uint4*>(src + 8);
            q_frag[dt] = loaded;
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
    // The transposed V buffer's row stride is padded_k (v_stride_n), not a local
    // 64-rounding: the binding pads the key axis to cta_k (128 for the wide D128
    // tile), and any other stride reads shifted rows once kv_len is not a
    // multiple of 64.
    const int64_t v_t_n = v_stride_n;
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
                // Two 16-byte LDS loads: k_ptr is k_tile + (row * KStride + dt * kBK)
                // elements, and both KStride and kBK are multiples of 16 elements.
                QK_VEC k_frag;
                *reinterpret_cast<uint4*>(&k_frag) = *reinterpret_cast<const uint4*>(k_ptr);
                *reinterpret_cast<uint4*>((char*)&k_frag + 16) =
                    *reinterpret_cast<const uint4*>(k_ptr + 8);
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
        // The sweep overrides force a single fp16 config. They are read per call
        // rather than once per process so a sweep can interleave configs inside
        // one process: cross-process timing on this iGPU drifts with the clock and
        // would swamp a config comparison (same reason as SAGE_PORT_*). They are
        // only valid for head_dim == 64: a direct launch is templated on head_dim,
        // and an override set while processing a head_dim-128 input would run the
        // 64-channel kernel, write half of o and leave the rest uninitialized.
        // With head_dim == 128 they are ignored and the shape-differentiated
        // dispatch below runs instead.
        const char* s_bm = getenv("SAGE_DIRECT_BM");
        const char* s_bn = getenv("SAGE_DIRECT_BN");
        const int ebm = s_bm ? atoi(s_bm) : 0;
        const int ebn = s_bn ? atoi(s_bn) : 0;
        if (ebm != 0 && ebn != 0 && head_dim == 64) {
            if (ebm == 128 && ebn == 64) {
                LAUNCH_DIRECT(64, 128, 64, __half);
            } else if (ebm == 128 && ebn == 32) {
                LAUNCH_DIRECT(64, 128, 32, __half);
            } else if (ebm == 64 && ebn == 32) {
                LAUNCH_DIRECT(64, 64, 32, __half);
            } else if (ebm == 64 && ebn == 16) {
                LAUNCH_DIRECT(64, 64, 16, __half);
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
