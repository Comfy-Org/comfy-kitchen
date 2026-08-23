// SPDX-License-Identifier: Apache-2.0
//
// Kitchen CUDA SVDQuant W4A4: activation quantize + smooth.
//
// Signature (matches backends/eager/svdquant.py::quantize_svdquant_w4a4):
//   in : x (M, K) bf16/fp16, smooth (K,) bf16/fp16, lora_down (K, R) bf16/fp16
//   out: q_x      (M_pad, K/2) int8      packed int4 (signed or unsigned)
//        ascales  (K/G, M_pad) bf16/fp16 per-row per-group scale
//        lora_act (M_pad, R)   fp32      computed externally via cuBLAS bf16 matmul
//                                        (see comfy_kitchen/backends/cuda/__init__.py;
//                                        LoRA-down operates on pre-quantization x)
//
// Per-warp layout:
//   kMRowsPerCTA warps per CTA (default 4), 1 warp (32 threads) per M row.
//   Warp loops over K/G groups; each thread handles 2 consecutive K elements
//   per group (32 × 2 = 64 = kGroupSize).
//   Thread t writes byte q_x[m, g*32 + t] = pack(q[2t], q[2t+1]).
//
// kActUnsigned=true: scale = max/15, clamp [0, 15]  (for u4.s4 MMA downstream)
// kActUnsigned=false: scale = max/7,  clamp [-7, 7] (for s4.s4 MMA downstream)
// No shift: callers that need non-negative x (e.g., post-GELU+shift) pre-shift
// at the layer level — see comfy_kitchen/tensor/svdquant_w4a4.py::_w4a4_forward.
#include "svdquant_utils.cuh"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace {

using comfy::svdquant::kGroupSize;
using comfy::svdquant::kInt4Max;
using comfy::svdquant::pack_int4_pair;
using comfy::svdquant::warp_absmax;

template<typename InType>
__device__ __forceinline__ float to_fp32(InType v);

template<>
__device__ __forceinline__ float to_fp32<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template<>
__device__ __forceinline__ float to_fp32<__half>(__half v) { return __half2float(v); }

template<typename InType>
__device__ __forceinline__ InType fp32_to_in(float v);

template<>
__device__ __forceinline__ __nv_bfloat16 fp32_to_in<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

template<>
__device__ __forceinline__ __half fp32_to_in<__half>(float v) { return __float2half(v); }

template<typename InType, bool kActUnsigned>
__global__ void svdquant_quantize_w4a4_kernel(
    const InType* __restrict__ x,          // (M, K)
    const InType* __restrict__ smooth,     // (K,)
    const InType* __restrict__ lora_down,  // (K, R)
    int8_t* __restrict__ q_x,              // (M_pad, K/2)
    InType* __restrict__ ascales,          // (K/G, M_pad)
    float* __restrict__ lora_act,          // (M_pad, R)
    int M, int M_pad, int K, int R)
{
    // CTA layout: kMRowsPerCTA warps, each warp handles one M row. The
    // LoRA-down path is computed externally via cuBLAS bf16 matmul (see the
    // Python wrapper), so this kernel only handles smooth + per-group int4
    // quantize + pack. Multi-warp CTA still helps by amortizing kernel launch
    // overhead and improving SM occupancy relative to 1-warp CTAs.
    constexpr int kMRowsPerCTA = 4;
    const int warp_id = threadIdx.x / 32;
    const int t = threadIdx.x & 31;
    const int m = blockIdx.x * kMRowsPerCTA + warp_id;
    const bool warp_active = (m < M_pad);
    const bool in_bounds = (m < M);

    const int num_groups = K / kGroupSize;
    const int K_half = K / 2;

    // LoRA-down is computed externally (bf16 cuBLAS matmul). This kernel
    // ignores the lora_down/lora_act pointers and the rank.
    (void)lora_down;
    (void)lora_act;
    (void)R;

    if (!warp_active) return;

    for (int g = 0; g < num_groups; ++g) {
        const int k0 = g * kGroupSize + t * 2;
        const int k1 = k0 + 1;

        float x0, x1, s0, s1;
        if (in_bounds) {
            x0 = to_fp32(x[m * K + k0]);
            x1 = to_fp32(x[m * K + k1]);
        } else {
            x0 = 0.f; x1 = 0.f;
        }
        s0 = to_fp32(smooth[k0]);
        s1 = to_fp32(smooth[k1]);

        // ---- Quantize (smooth then per-group int4) ----
        // kActUnsigned only changes the quantization grid (scale=max/15, clamp [0,15]).
        // Any activation shift (e.g., GELU + 0.171875 for nunchaku unsigned path) is
        // applied by the caller at the layer level, not inside this kernel — keeps
        // kitchen ops orthogonal to model topology.
        const float inv_s0 = (fabsf(s0) > 1e-10f) ? (1.f / s0) : 0.f;
        const float inv_s1 = (fabsf(s1) > 1e-10f) ? (1.f / s1) : 0.f;
        const float y0 = x0 * inv_s0;
        const float y1 = x1 * inv_s1;

        float local = fmaxf(fabsf(y0), fabsf(y1));
        const float group_absmax = warp_absmax(local, 32);
        constexpr int kQMax = kActUnsigned ? 15 : kInt4Max;  // 15 for u4, 7 for s4
        const float scale = fmaxf(group_absmax / static_cast<float>(kQMax), 1e-10f);
        const float inv_scale = 1.f / scale;

        int q0 = __float2int_rn(y0 * inv_scale);
        int q1 = __float2int_rn(y1 * inv_scale);
        if constexpr (kActUnsigned) {
            // Clamp to [0, 15] unsigned range. Values should be non-negative after
            // shift; any residual negatives clamp to 0.
            q0 = max(0, min(15, q0));
            q1 = max(0, min(15, q1));
        } else {
            q0 = max(-kInt4Max, min(kInt4Max, q0));
            q1 = max(-kInt4Max, min(kInt4Max, q1));
        }
        q_x[m * K_half + g * (kGroupSize / 2) + t] = pack_int4_pair(q0, q1);

        if (t == 0) {
            ascales[g * M_pad + m] = fp32_to_in<InType>(in_bounds ? scale : 0.f);
        }

    }

}

} // anonymous namespace

extern "C" {

void launch_svdquant_quantize_w4a4_kernel(
    const void* x,
    const void* smooth,
    const void* lora_down,
    void* q_x,
    void* ascales,
    void* lora_act,
    int M,
    int M_pad,
    int K,
    int R,
    int input_dtype_code,
    int act_unsigned,      // 0: signed [-7,7] + scale=max/7; 1: unsigned [0,15] + scale=max/15
    cudaStream_t stream)
{
    if (K % comfy::svdquant::kGroupSize != 0) return;
    // No rank cap: LoRA-down matmul is external, Python wrapper picks cuBLAS
    // bf16 path which handles any R efficiently (benched at R={16..256}).

    constexpr int kMRowsPerCTA = 4;
    const dim3 grid((M_pad + kMRowsPerCTA - 1) / kMRowsPerCTA);
    const dim3 block(kMRowsPerCTA * 32);

    #define LAUNCH_QUANTIZE(InType, Unsigned)                                               \
        svdquant_quantize_w4a4_kernel<InType, Unsigned><<<grid, block, 0, stream>>>(        \
            reinterpret_cast<const InType*>(x),                                             \
            reinterpret_cast<const InType*>(smooth),                                        \
            reinterpret_cast<const InType*>(lora_down),                                     \
            reinterpret_cast<int8_t*>(q_x),                                                 \
            reinterpret_cast<InType*>(ascales),                                             \
            reinterpret_cast<float*>(lora_act),                                             \
            M, M_pad, K, R)

    if (input_dtype_code == 2) {
        if (act_unsigned) LAUNCH_QUANTIZE(__nv_bfloat16, true);
        else              LAUNCH_QUANTIZE(__nv_bfloat16, false);
    } else if (input_dtype_code == 1) {
        if (act_unsigned) LAUNCH_QUANTIZE(__half, true);
        else              LAUNCH_QUANTIZE(__half, false);
    }
    #undef LAUNCH_QUANTIZE
}

} // extern "C"
