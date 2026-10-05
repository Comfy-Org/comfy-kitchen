// SPDX-FileCopyrightText: Copyright (c) 2024 SageAttention team.
// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#else
#error "attn_gfx103x.cu is only intended for ROCm/HIP."
#endif


#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "../mma_gfx103x.h"
#include "sage_attn_gfx103x.h"

namespace {

constexpr int kNHD = 0;
constexpr int kHND = 1;

// Shared with the tile header, which is included at file scope and so cannot see
// this namespace's constants. Pulled in unqualified so every use below reads as one
// set rather than two names for the same number.
using sageattn_gfx10::kLog2e;
using sageattn_gfx10::MIN_BLK_K;
using sageattn_gfx10::MIN_BLK_Q;

// The two warp/block maximum reductions the Q/K quantizer needs, and nothing
// else. These came from reduction_utils.cuh, a port of vLLM's reduction set that
// arrived with the rest of the SageAttention reference code. Ten functions came in
// that way and nine were never called from anywhere in the tree; only
// blockReduceMax was, once, in quant_qk_int8_gfx103x below, and it needs
// warpReduceMax to do its work. The header went rather than being trimmed, taking
// a namespace indirection and a warp-mask macro with it.
template <typename T>
__device__ __forceinline__ T warp_reduce_max(T val) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val = max(val, __shfl_xor(val, mask, 32));
    return val;
}

// Maximum over the whole block. The lane-0 slot of each warp goes through shared
// memory, and the second pass reads those slots back with -1e20f in the lanes past
// the warp count. That keeps the reduction total-count agnostic, so blockDim.x need
// not be a multiple of 32.
template <typename T>
__device__ __forceinline__ T block_reduce_max(T val) {
    __shared__ T partial[32];
    const int lane = threadIdx.x & 0x1f;
    const int wid = threadIdx.x >> 5;
    val = warp_reduce_max(val);
    if (lane == 0) partial[wid] = val;
    __syncthreads();
    val = (threadIdx.x < (blockDim.x / 32.f)) ? partial[lane] : -1e20f;
    return warp_reduce_max(val);
}

// Every operand the attention kernel takes, gathered so the launcher can stay a
// template over the tile shape and the mask representation instead of a wall of
// positional arguments. The pointer fields are the raw device pointers; the
// stride fields are in elements, as the kernels index them.
struct AttnLaunch {
    const int8_t* q;
    const int8_t* k;
    const __half* v;
    void* out;
    const float* q_scale;
    const float* k_scale;
    const void* q_fp;
    int q_src_bf16;
    float sm_scale_log2e;
    int64_t batch, qo_len, kv_len, q_heads, kv_heads;
    int64_t q_stride_b, q_stride_n, q_stride_h;
    int64_t q_fp_stride_b, q_fp_stride_n, q_fp_stride_h;
    int64_t k_stride_b, k_stride_n, k_stride_h;
    int64_t v_stride_b, v_stride_n, v_stride_h;
    int64_t o_stride_b, o_stride_n, o_stride_h;
    int64_t qs_stride_b, qs_stride_h, ks_stride_b, ks_stride_h;
    // Mask. `mask` is null when mode is kNone; the strides are only meaningful
    // for kRaw, every prepared form being contiguous (see gfx10_mask_row).
    const void* mask;
    int mask_mode, mask_dtype;
    int64_t mask_sb, mask_sh, mask_sq, mask_sk, mask_batch, mask_heads, mask_q_tiles;
    int64_t mask_k_tiles;
    // The key mask's real row width, which the producer pads past the exact
    // tiles*65 so its vector loads stay aligned.
    int64_t mask_k_row_pitch;
};

// The launch context: the operands plus the geometry of the grid they run on.
// Bundled so the mask-mode dispatcher below can take one argument instead of six
// and stay readable at the call sites.
struct AttnLaunchCtx {
    AttnLaunch ops;
    dim3 grid;
    dim3 block;
    hipStream_t stream;
    int mask_mode;
    int mask_dtype;
    // Which stage of the kernel body to drop, bit 0 QK / 1 softmax / 2 PV /
    // 3 staging / 4 writeback. Zero on every real launch; see CK_GFX103X_DIAG in
    // gfx103x_launch_attn.
    int diag;
};

// Dispatches one kernel instantiation. MASK and MASK_DTYPE are template
// parameters rather than arguments to the kernel for the same reason HD and TM
// are: a runtime mode would put a branch and a live register on the critical
// path of every key in the inner loop, where the register budget is already the
// binding constraint on this architecture.
template <int HD, bool CAUSAL, int BN, int BM, int TM, typename ODT, bool INQ, int MASK,
          int MASK_DTYPE>
void launch_attn(const AttnLaunchCtx& c) {
    const AttnLaunch& a = c.ops;
    // The in-kernel-Q path reads Q from the fp16/bf16 source; otherwise it reads
    // the packed int8 the prepass wrote.
    const int64_t qi_b = INQ ? a.q_fp_stride_b : a.q_stride_b;
    const int64_t qi_n = INQ ? a.q_fp_stride_n : a.q_stride_n;
    const int64_t qi_h = INQ ? a.q_fp_stride_h : a.q_stride_h;
    sageattn_gfx10::attn_kernel_i8q_f16pv_tiled_pv<HD, CAUSAL, BN, BM, ODT, TM, INQ, MASK,
                                                   MASK_DTYPE>
        <<<c.grid, c.block, 0, c.stream>>>(
        a.q, a.k, a.v, static_cast<ODT*>(a.out), a.q_scale, a.k_scale, a.batch, a.qo_len,
        a.kv_len, a.q_heads, a.kv_heads, qi_b, qi_n, qi_h, a.k_stride_b, a.k_stride_n,
        a.k_stride_h, a.v_stride_b, a.v_stride_n, a.v_stride_h, a.o_stride_b, a.o_stride_n,
        a.o_stride_h, a.qs_stride_b, a.qs_stride_h, a.ks_stride_b, a.ks_stride_h, c.diag, a.q_fp,
        a.q_src_bf16, a.sm_scale_log2e, a.mask, a.mask_sb, a.mask_sh, a.mask_sq, a.mask_sk,
        a.mask_batch, a.mask_heads, a.mask_q_tiles, a.mask_k_tiles, a.mask_k_row_pitch);
}

// Resolves the runtime mask mode into the template argument the kernel takes.
// The dtype is a second dimension of the same dispatch: kRaw is the only mode
// that cares, and a Boolean mask and a float mask compile to different reads.
template <int HD, bool CAUSAL, int BN, int BM, int TM, typename ODT, bool INQ>
void launch_attn_mask(const AttnLaunchCtx& c) {
    const AttnLaunch& a = c.ops;
    switch (c.mask_mode) {
        case sageattn_gfx10::MaskMode::kNone:
            launch_attn<HD, CAUSAL, BN, BM, TM, ODT, INQ, sageattn_gfx10::MaskMode::kNone, 0>(c);
            return;
        case sageattn_gfx10::MaskMode::kPreparedKey:
            launch_attn<HD, CAUSAL, BN, BM, TM, ODT, INQ, sageattn_gfx10::MaskMode::kPreparedKey, 0>(c);
            return;
        case sageattn_gfx10::MaskMode::kPreparedDense:
            launch_attn<HD, CAUSAL, BN, BM, TM, ODT, INQ, sageattn_gfx10::MaskMode::kPreparedDense, 0>(c);
            return;
        case sageattn_gfx10::MaskMode::kPreparedDenseBool:
            launch_attn<HD, CAUSAL, BN, BM, TM, ODT, INQ,
                        sageattn_gfx10::MaskMode::kPreparedDenseBool, 0>(c);
            return;
        case sageattn_gfx10::MaskMode::kPreparedDenseBF16:
            launch_attn<HD, CAUSAL, BN, BM, TM, ODT, INQ,
                        sageattn_gfx10::MaskMode::kPreparedDenseBF16, 0>(c);
            return;
        case sageattn_gfx10::MaskMode::kPreparedDenseF16:
            launch_attn<HD, CAUSAL, BN, BM, TM, ODT, INQ,
                        sageattn_gfx10::MaskMode::kPreparedDenseF16, 0>(c);
            return;
        case sageattn_gfx10::MaskMode::kRaw:
            break;
        default:
            throw std::runtime_error("attn_gfx103x: unknown attn_mask mode");
    }
    // kRaw: the dtype selects which load the inner loop uses. 0 fp32, 1 fp16,
    // 2 bf16, 3 bool -- the same codes the main HIP backend's mask helpers use.
    switch (c.mask_dtype) {
        case 3:
            launch_attn<HD, CAUSAL, BN, BM, TM, ODT, INQ, sageattn_gfx10::MaskMode::kRaw, 3>(c);
            return;
        case 1:
            launch_attn<HD, CAUSAL, BN, BM, TM, ODT, INQ, sageattn_gfx10::MaskMode::kRaw, 1>(c);
            return;
        case 2:
            launch_attn<HD, CAUSAL, BN, BM, TM, ODT, INQ, sageattn_gfx10::MaskMode::kRaw, 2>(c);
            return;
        default:
            launch_attn<HD, CAUSAL, BN, BM, TM, ODT, INQ, sageattn_gfx10::MaskMode::kRaw, 0>(c);
            return;
    }
}

// Resolves the runtime tile shape onto the template arguments the kernel takes.
// Every level is a template parameter for the same reason: the mask lookup and
// the tile geometry sit in the innermost loop, where a runtime branch would cost
// a register and an issue slot per key. Only the mask mode, its dtype, and the
// tuning overrides stay runtime, and those are decided once per launch.
template <bool CAUSAL>
void launch_attn_cf(const AttnLaunchCtx& c, int64_t head_dim, bool inq, bool out_bf, int v10_bn,
                    int tm) {
    // The query tile BM is a template argument on the kernel for the same reason
    // BN and TM are, and it is pinned at 128: every extra value in this cross
    // product is a full set of instantiations (head dim x key tile x rows per
    // thread x output dtype x in-kernel Q x seven mask modes x causal) and the
    // kernel is already compiled for hundreds of them.
    //
    // 256 was instantiated and measured (2026-10, 6-CU gfx1035) to test the claim
    // that the L2 absorbs a head's K/V being re-read by all 32 query tiles at
    // BM=128 -- 1.5 MiB per head at S=4096 D=128. It is: B1 H16 S4096x4096 D128
    // went 40.63 -> 40.31 ms and B1 H16 S4096x512 D128 5.64 -> 5.93, i.e. inside
    // the run-to-run spread and in opposite directions. The instantiation was
    // removed again rather than left in the tree, because a knob that measures as
    // nothing is not worth doubling this kernel's compile time for.
    //
    // The tile that *is* worth its instantiations: at D128 the key tile is 16 and
    // at D64 it is 32, with 8 rows per thread everywhere. All six combinations
    // the kernel can be built with were swept in one process on the shapes above
    // (CK_GFX103X_BN x CK_GFX103X_TM): the shipped pair wins both, BN=32 costs
    // 1.47x and TM=4 1.04x at D128, and TM=2 is 1.38x.
    constexpr int BM = 128;

    // One macro per level of the dispatch so the template argument lists cannot
    // drift apart between the combinations: the geometry picks the instantiation
    // and the rows-per-thread sweep picks within it.
#define CK_GFX103X_BY_TM(HD_, BN_)                                                            \
    do {                                                                                      \
        if (out_bf) {                                                                         \
            if (inq) {                                                                        \
                if (tm <= 2)   { launch_attn_mask<HD_, CAUSAL, BN_, BM, 2, __hip_bfloat16, true>(c);  return; } \
                if (tm <= 4)   { launch_attn_mask<HD_, CAUSAL, BN_, BM, 4, __hip_bfloat16, true>(c);  return; } \
                launch_attn_mask<HD_, CAUSAL, BN_, BM, 8, __hip_bfloat16, true>(c);  return;  \
            }                                                                                 \
            if (tm <= 2)       { launch_attn_mask<HD_, CAUSAL, BN_, BM, 2, __hip_bfloat16, false>(c); return; } \
            if (tm <= 4)       { launch_attn_mask<HD_, CAUSAL, BN_, BM, 4, __hip_bfloat16, false>(c); return; } \
            launch_attn_mask<HD_, CAUSAL, BN_, BM, 8, __hip_bfloat16, false>(c);      return;  \
        }                                                                                     \
        if (inq) {                                                                            \
            if (tm <= 2)   { launch_attn_mask<HD_, CAUSAL, BN_, BM, 2, __half, true>(c);  return; }   \
            if (tm <= 4)   { launch_attn_mask<HD_, CAUSAL, BN_, BM, 4, __half, true>(c);  return; }   \
            launch_attn_mask<HD_, CAUSAL, BN_, BM, 8, __half, true>(c);                  return;   \
        }                                                                                     \
        if (tm <= 2)       { launch_attn_mask<HD_, CAUSAL, BN_, BM, 2, __half, false>(c); return; }    \
        if (tm <= 4)       { launch_attn_mask<HD_, CAUSAL, BN_, BM, 4, __half, false>(c); return; }    \
        launch_attn_mask<HD_, CAUSAL, BN_, BM, 8, __half, false>(c);                  return;    \
    } while (0)

    if (head_dim == 64) {
        // D64 keeps the wide key tile whatever the override asks for: 16 halves
        // the V tile here and measured slower.
        CK_GFX103X_BY_TM(64, 32);
        return;
    }
    // D128 tunes its key tile between 16 and 32.
    if (v10_bn >= 32) {
        CK_GFX103X_BY_TM(128, 32);
        return;
    }
    CK_GFX103X_BY_TM(128, 16);
#undef CK_GFX103X_BY_TM
}

__device__ __forceinline__ float to_float(const __half v) { return __half2float(v); }
__device__ __forceinline__ float to_float(const __hip_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float to_float(float v) { return v; }
__device__ __forceinline__ __half from_float_f16(float v) { return __float2half_rn(v); }
__device__ __forceinline__ __hip_bfloat16 from_float_bf16(float v) { return __float2bfloat16(v); }

__device__ __forceinline__ int8_t float_to_int8(float x) {
    x += (x >= 0.0f) ? 0.5f : -0.5f;
    int32_t rounded;
    asm volatile("v_cvt_i32_f32 %[dst], %[src]" : [dst] "=v"(rounded) : [src] "v"(x));
    rounded = rounded > 127 ? 127 : rounded;
    rounded = rounded < -128 ? -128 : rounded;
    return static_cast<int8_t>(rounded);
}

template <typename T, int HeadDim, int BLK, int MIN_BLK>
__global__ void quant_qk_int8_hnd_kernel(
    const T* __restrict__ input,
    int8_t* __restrict__ output,
    const T* __restrict__ key_mean,
    float* __restrict__ scale_out,
    const int64_t batch,
    const int64_t heads,
    const int64_t seq_len,
    const int scale_groups,
    const int is_q,
    const float sm_scale_log2e,
    const int64_t in_stride_b,
    const int64_t in_stride_n,
    const int64_t in_stride_h,
    const int64_t groups_per_block) {
    constexpr int Threads = 256;
    constexpr int RATIO = BLK / MIN_BLK;
    constexpr int PackElems = 8;
    __shared__ float shared_amax[RATIO];
    __shared__ uint4 shared_data[(BLK * HeadDim) / 8];

    const int head = blockIdx.y;
    const int b = blockIdx.z;
    if (b >= batch || head >= heads) return;
    const int total_blocks = static_cast<int>((seq_len + BLK - 1) / BLK);
    const float pass1_scale = is_q ? sm_scale_log2e : 1.0f;
    const float extra_scale = is_q ? sm_scale_log2e : 1.0f;

    for (int gi = 0; gi < groups_per_block; ++gi) {
        const int blk = blockIdx.x * groups_per_block + gi;
        if (blk >= total_blocks) break;
        const int64_t base_row = static_cast<int64_t>(blk) * BLK;
        const int tid = threadIdx.x;
        constexpr int Packs = (BLK * HeadDim) / 8;

        float local_amax[RATIO];
#pragma unroll
        for (int r = 0; r < RATIO; ++r) local_amax[r] = 1e-7f;
        for (int p = tid; p < Packs / 2; p += Threads) {
            const int pack = p * 2;
            const int elem_base = pack * PackElems;
            const int row = elem_base / HeadDim;
            const int d = elem_base - row * HeadDim;
            const int64_t seq = base_row + row;
            if (seq < seq_len) {
                const int64_t in_off = static_cast<int64_t>(b) * in_stride_b + seq * in_stride_n + head * in_stride_h + d;
                const uint4 raw0 = *reinterpret_cast<const uint4*>(input + in_off);
                const uint4 raw1 = *reinterpret_cast<const uint4*>(input + in_off + 8);
                shared_data[pack] = raw0;
                shared_data[pack + 1] = raw1;
                const int r = row / MIN_BLK;
                const T* v0 = reinterpret_cast<const T*>(&raw0);
                const T* v1 = reinterpret_cast<const T*>(&raw1);
                float am = local_amax[r];
                if (!is_q && key_mean != nullptr) {
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        float v = to_float(v0[i]) - to_float(key_mean[(b * heads + head) * HeadDim + d + i]);
                        am = fmaxf(am, fabsf(v * pass1_scale));
                    }
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        float v = to_float(v1[i]) - to_float(key_mean[(b * heads + head) * HeadDim + d + 8 + i]);
                        am = fmaxf(am, fabsf(v * pass1_scale));
                    }
                } else {
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        float v = to_float(v0[i]);
                        am = fmaxf(am, fabsf(v * pass1_scale));
                    }
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        float v = to_float(v1[i]);
                        am = fmaxf(am, fabsf(v * pass1_scale));
                    }
                }
                local_amax[r] = am;
            } else {
                shared_data[pack] = make_uint4(0, 0, 0, 0);
                shared_data[pack + 1] = make_uint4(0, 0, 0, 0);
            }
        }
        for (int r = 0; r < RATIO; ++r) {
            const float block_amax = block_reduce_max(local_amax[r]);
            if (tid == 0) {
                shared_amax[r] = block_amax;
                if (base_row + static_cast<int64_t>(r) * MIN_BLK < seq_len) {
                    scale_out[(static_cast<int64_t>(b) * heads + head) * scale_groups + blk * RATIO + r] =
                        block_amax / 127.0f;
                }
            }
        }
        __syncthreads();
        float inv_scale[RATIO];
#pragma unroll
        for (int r = 0; r < RATIO; ++r) inv_scale[r] = 127.0f / shared_amax[r];

        for (int p = tid; p < Packs / 2; p += Threads) {
            const int pack = p * 2;
            const int elem_base = pack * PackElems;
            const int row = elem_base / HeadDim;
            const int d = elem_base - row * HeadDim;
            const int64_t seq = base_row + row;
            if (seq < seq_len) {
                const int r = row / MIN_BLK;
                const int64_t out_off = (static_cast<int64_t>(b) * heads + head) * seq_len * HeadDim + seq * HeadDim + d;
                const uint4 raw0 = shared_data[pack];
                const uint4 raw1 = shared_data[pack + 1];
                const T* values = reinterpret_cast<const T*>(&raw0);
                const T* values1 = reinterpret_cast<const T*>(&raw1);
                char4 out0, out1, out2, out3;
                float v0 = to_float(values[0]), v1 = to_float(values[1]);
                float v2 = to_float(values[2]), v3 = to_float(values[3]);
                float v4 = to_float(values[4]), v5 = to_float(values[5]);
                float v6 = to_float(values[6]), v7 = to_float(values[7]);
                float w0 = to_float(values1[0]), w1 = to_float(values1[1]);
                float w2 = to_float(values1[2]), w3 = to_float(values1[3]);
                float w4 = to_float(values1[4]), w5 = to_float(values1[5]);
                float w6 = to_float(values1[6]), w7 = to_float(values1[7]);
                if (!is_q && key_mean != nullptr) {
                    const int64_t mean_base = (b * heads + head) * HeadDim + d;
                    v0 -= to_float(key_mean[mean_base + 0]);
                    v1 -= to_float(key_mean[mean_base + 1]);
                    v2 -= to_float(key_mean[mean_base + 2]);
                    v3 -= to_float(key_mean[mean_base + 3]);
                    v4 -= to_float(key_mean[mean_base + 4]);
                    v5 -= to_float(key_mean[mean_base + 5]);
                    v6 -= to_float(key_mean[mean_base + 6]);
                    v7 -= to_float(key_mean[mean_base + 7]);
                    w0 -= to_float(key_mean[mean_base + 8]);
                    w1 -= to_float(key_mean[mean_base + 9]);
                    w2 -= to_float(key_mean[mean_base + 10]);
                    w3 -= to_float(key_mean[mean_base + 11]);
                    w4 -= to_float(key_mean[mean_base + 12]);
                    w5 -= to_float(key_mean[mean_base + 13]);
                    w6 -= to_float(key_mean[mean_base + 14]);
                    w7 -= to_float(key_mean[mean_base + 15]);
                }
                const float iscale = inv_scale[r] * extra_scale;
                out0.x = float_to_int8(v0 * iscale);
                out0.y = float_to_int8(v1 * iscale);
                out0.z = float_to_int8(v2 * iscale);
                out0.w = float_to_int8(v3 * iscale);
                out1.x = float_to_int8(v4 * iscale);
                out1.y = float_to_int8(v5 * iscale);
                out1.z = float_to_int8(v6 * iscale);
                out1.w = float_to_int8(v7 * iscale);
                out2.x = float_to_int8(w0 * iscale);
                out2.y = float_to_int8(w1 * iscale);
                out2.z = float_to_int8(w2 * iscale);
                out2.w = float_to_int8(w3 * iscale);
                out3.x = float_to_int8(w4 * iscale);
                out3.y = float_to_int8(w5 * iscale);
                out3.z = float_to_int8(w6 * iscale);
                out3.w = float_to_int8(w7 * iscale);
                *reinterpret_cast<char4*>(output + out_off) = out0;
                *reinterpret_cast<char4*>(output + out_off + 4) = out1;
                *reinterpret_cast<char4*>(output + out_off + 8) = out2;
                *reinterpret_cast<char4*>(output + out_off + 12) = out3;
            }
        }
        __syncthreads();
    }
}

}

// Quantize Q and K. The block shape is picked from the head dim: the two wider
// pairs were reachable only through the SAGEATTN_QUANT_BLK sweep knob, which this
// tree has dropped (it read the environment on every launch, and the fork removed
// the equivalent COMFY_KITCHEN_WMMA_TILE scaffolding for the same reason), so D64
// uses 128/64 and D128 falls back to the minimum block.
namespace sageattn_gfx103x {

void gfx103x_launch_quant_qk(const QuantArgs& a, hipStream_t stream) {
    constexpr int BLK_Q64 = 128;
    constexpr int BLK_K64 = 64;
    int blk_q, blk_k;
    if (a.head_dim == 64) { blk_q = BLK_Q64; blk_k = BLK_K64; }
    else { blk_q = MIN_BLK_Q; blk_k = MIN_BLK_K; }

    dim3 block(256);
    constexpr int q_gpb = 1;  // one (q_block, head, batch) per program
    dim3 grid_q(static_cast<unsigned>((a.q_len + blk_q - 1) / blk_q),
                static_cast<unsigned>(a.q_heads), static_cast<unsigned>(a.batch));
    dim3 grid_k(static_cast<unsigned>((a.kv_len + blk_k - 1) / blk_k),
                static_cast<unsigned>(a.kv_heads), static_cast<unsigned>(a.batch));

    const float sm_scale_log2e = static_cast<float>(a.sm_scale) * kLog2e;

    #define LAUNCH_QUANT(HD, T, BQ, BK) \
        do { \
            if (!a.skip_q) { \
                quant_qk_int8_hnd_kernel<T, HD, BQ, MIN_BLK_Q><<<grid_q, block, 0, stream>>>( \
                    reinterpret_cast<const T*>(a.q), \
                    a.q_int8, \
                    nullptr, \
                    a.q_scale, \
                    a.batch, a.q_heads, a.q_len, static_cast<int>(a.q_groups), 1, sm_scale_log2e, \
                    a.q_stride_b, a.q_stride_n, a.q_stride_h, q_gpb); \
            } \
            quant_qk_int8_hnd_kernel<T, HD, BK, MIN_BLK_K><<<grid_k, block, 0, stream>>>( \
                reinterpret_cast<const T*>(a.k), \
                a.k_int8, \
                reinterpret_cast<const T*>(a.key_mean), \
                a.k_scale, \
                a.batch, a.kv_heads, a.kv_len, static_cast<int>(a.k_groups), 0, 1.0f, \
                a.k_stride_b, a.k_stride_n, a.k_stride_h, q_gpb); \
        } while(0)

    #define LAUNCH_QUANT_DISPATCH(HD, T) \
        do { \
            if (blk_q == BLK_Q64) { LAUNCH_QUANT(HD, T, 128, 64); } \
            else { LAUNCH_QUANT(HD, T, MIN_BLK_Q, MIN_BLK_K); } \
        } while(0)

    if (a.src_bf16) {
        if (a.head_dim == 64) { LAUNCH_QUANT_DISPATCH(64, __hip_bfloat16); }
        else { LAUNCH_QUANT_DISPATCH(128, __hip_bfloat16); }
    } else {
        if (a.head_dim == 64) { LAUNCH_QUANT_DISPATCH(64, __half); }
        else { LAUNCH_QUANT_DISPATCH(128, __half); }
    }
    #undef LAUNCH_QUANT_DISPATCH
    #undef LAUNCH_QUANT
}

void gfx103x_launch_attn(const AttnArgs& in, hipStream_t stream) {
    const bool inq = in.kv_len <= 1024 && in.has_q_fp;

    // Mask descriptor. Raw masks are indexed by the caller's strides; every
    // prepared form is contiguous with a pitch the tile counts fix, so the
    // kernel indexes those arithmetically (see gfx10_mask_row).
    const int gmask = in.mask_mode;
    const bool masked = gmask != 0;
    int64_t mask_sb = 0, mask_sh = 0, mask_sq = 0, mask_sk = 0;
    int64_t mask_batch = 1, mask_heads = 1, mask_q_tiles = 0, mask_k_tiles = 0;
    int64_t mask_k_row_pitch = 0;
    if (masked) {
        if (gmask == sageattn_gfx10::MaskMode::kRaw) {
            if (in.mask_ndim != 4) {
                throw std::runtime_error("attn_gfx103x: a raw attn_mask must be [B,H,Q,K]");
            }
            mask_sb = in.mask_stride[0];
            mask_sh = in.mask_stride[1];
            mask_sq = in.mask_stride[2];
            mask_sk = in.mask_stride[3];
            // The compact extents tell the kernel which batch/head indices the
            // mask actually covers; a stride-0 axis is one it does not. The
            // strides are passed through unchanged -- they already encode the
            // broadcast, and rewriting them would break the in-bounds
            // arithmetic the producer's layout depends on.
            mask_batch = (mask_sb == 0) ? 1 : in.mask_shape[0];
            mask_heads = (mask_sh == 0) ? 1 : in.mask_shape[1];
        } else if (gmask == sageattn_gfx10::MaskMode::kPreparedKey) {
            // [B, H, tiles * 65]: the row pitch is the buffer's own width, and
            // the kernel derives the tile count from kv_len. There are no query
            // or key tile extents to read -- that is why this branch does not
            // touch shape[2]/shape[3], which a 3D tensor does not have.
            if (in.mask_ndim != 3) {
                throw std::runtime_error("attn_gfx103x: a prepared key mask must be [B,H,W]");
            }
            if (!in.mask_contiguous) {
                throw std::runtime_error("attn_gfx103x: a prepared attn_mask must be contiguous");
            }
            mask_batch = in.mask_shape[0];
            mask_heads = in.mask_shape[1];
            mask_k_tiles = (in.kv_len + sageattn_gfx10::kGfx10TileK - 1) / sageattn_gfx10::kGfx10TileK;
            // The producer rounds the row up to a multiple of four so its own
            // vector loads stay aligned; the kernel has to stride by that width,
            // not by the exact tiles*65.
            const int64_t exact = mask_k_tiles * (sageattn_gfx10::kGfx10TileK + 1);
            mask_k_row_pitch = (exact + 3) / 4 * 4;
        } else {
            // Dense: [B, H, q_tiles, k_tiles, slots].
            if (in.mask_ndim != 5) {
                throw std::runtime_error("attn_gfx103x: a prepared dense mask must be 5D");
            }
            if (!in.mask_contiguous) {
                throw std::runtime_error("attn_gfx103x: a prepared attn_mask must be contiguous");
            }
            mask_batch = in.mask_shape[0];
            mask_heads = in.mask_shape[1];
            mask_q_tiles = in.mask_shape[2];
            mask_k_tiles = in.mask_shape[3];
            // The buffer has to cover every (query, key) the kernel will ask
            // for, or the lookup runs off the end of the allocation.
            const int64_t need_q_tiles = (in.qo_len + sageattn_gfx10::kGfx10TileQ - 1) /
                                         sageattn_gfx10::kGfx10TileQ;
            const int64_t need_k_tiles = (in.kv_len + sageattn_gfx10::kGfx10TileK - 1) /
                                         sageattn_gfx10::kGfx10TileK;
            if (mask_q_tiles < need_q_tiles || mask_k_tiles < need_k_tiles) {
                throw std::runtime_error(
                    "attn_gfx103x: prepared dense mask covers " +
                    std::to_string(mask_q_tiles) + "x" + std::to_string(mask_k_tiles) +
                    " tiles but the call needs " + std::to_string(need_q_tiles) + "x" +
                    std::to_string(need_k_tiles));
            }
        }
    }

    AttnLaunch a{};
    // The kernel always takes a Q pointer: on the prepass path it is the packed
    // int8, on the in-kernel path the fp16/bf16 source it quantizes itself.
    a.q = inq ? reinterpret_cast<const int8_t*>(in.q_fp) : in.q;
    a.k = in.k;
    a.v = in.v;
    a.out = in.out;
    a.q_scale = in.q_scale;
    a.k_scale = in.k_scale;
    a.q_fp = inq ? in.q_fp : a.q;
    a.q_src_bf16 = inq ? in.q_fp_bf16 : 0;
    a.sm_scale_log2e = inq ? static_cast<float>(in.sm_scale) * kLog2e : 0.0f;
    a.batch = in.batch;
    a.qo_len = in.qo_len;
    a.kv_len = in.kv_len;
    a.q_heads = in.q_heads;
    a.kv_heads = in.kv_heads;
    a.q_stride_b = in.q_stride_b;
    a.q_stride_n = in.q_stride_n;
    a.q_stride_h = in.q_stride_h;
    // The Q source is absent on the packed-Q path, so its strides are only read
    // when there is a real 4D source behind it.
    if (inq) {
        a.q_fp_stride_b = in.q_fp_stride_b;
        a.q_fp_stride_n = in.q_fp_stride_n;
        a.q_fp_stride_h = in.q_fp_stride_h;
    }
    a.k_stride_b = in.k_stride_b;
    a.k_stride_n = in.k_stride_n;
    a.k_stride_h = in.k_stride_h;
    a.v_stride_b = in.v_stride_b;
    a.v_stride_n = in.v_stride_n;
    a.v_stride_h = in.v_stride_h;
    a.o_stride_b = in.o_stride_b;
    a.o_stride_n = in.layout_is_hnd ? in.o_stride_head : in.o_stride_seq;
    a.o_stride_h = in.layout_is_hnd ? in.o_stride_seq : in.o_stride_head;
    a.qs_stride_b = in.qs_stride_b;
    a.qs_stride_h = in.qs_stride_h;
    a.ks_stride_b = in.ks_stride_b;
    a.ks_stride_h = in.ks_stride_h;
    a.mask = masked ? in.mask : nullptr;
    a.mask_mode = gmask;
    a.mask_dtype = in.mask_dtype;
    a.mask_sb = mask_sb;
    a.mask_sh = mask_sh;
    a.mask_sq = mask_sq;
    a.mask_sk = mask_sk;
    a.mask_batch = mask_batch;
    a.mask_heads = mask_heads;
    a.mask_q_tiles = mask_q_tiles;
    a.mask_k_tiles = mask_k_tiles;
    a.mask_k_row_pitch = mask_k_row_pitch;

    AttnLaunchCtx ctx{};
    ctx.ops = a;

    // CK_GFX103X_BN / CK_GFX103X_TM put them back for measurement only. A
    // rebuild-per-tile sweep cannot be trusted on this device -- the first kernel
    // after a rebuild runs on cold clocks, measured 1.9x, which is larger than
    // these effects -- so the sweep has to happen inside one process, which means
    // the knob has to be read per launch. Values outside the instantiated set are
    // ignored rather than clamped, because a sweep that silently keeps measuring
    // the default reads exactly like "the tile does not matter".
    constexpr int tm_default = 8;
    int tm = tm_default;
    if (const char* e = getenv("CK_GFX103X_TM")) {
        const int v = atoi(e);
        if (v == 2 || v == 4 || v == 8) tm = v;
    }
    int v10_bn = (in.head_dim == 64) ? 32 : 16;
    if (const char* e = getenv("CK_GFX103X_BN")) {
        const int v = atoi(e);
        if (v == 16 || v == 32) v10_bn = v;
    }

    // The diag bitfield drops one stage of the kernel body (bit 0 QK, 1 softmax,
    // 2 PV, 3 staging, 4 writeback). It exists in the ported kernel as a
    // development aid and is always 0 on this path; CK_GFX103X_DIAG exposes it so
    // the per-stage cost can be measured from outside without a rebuild.
    // It produces wrong output by construction, so nothing may default to
    // non-zero.
    //
    // What it says, on the 6-CU gfx1035 at this kernel's largest production shape
    // (B1 H16 S4096x4096 D128, the Anima self-attention -- 31% of that model's step):
    //
    //   diag  dropped        ms      share
    //   0     nothing        40.63   --
    //   1     QK             24.71   QK     39.2%
    //   2     softmax        40.42   softmax  0.5%
    //   4     PV             29.92   PV     26.4%
    //   8     K/V staging    37.80   staging  7.0%
    //   15    everything      7.10   loop, barriers, alpha/l store, writeback 17.5%
    //
    // Those two shares are the reason the obvious rewrite is not the obvious win.
    // Per thread per BN=16 key tile this kernel issues 512 v_dot4_i32_i8 for the
    // QK and 1024 v_fdot2_f32_f16 for the PV, for the same 2048 MACs each. On
    // this part sdot4 retires at one instruction per clock per lane and fdot2 at
    // two (the VALU's paired FP32 pipes), so both halves already cost the same
    // 512 cycles and neither can be shortened by moving the operands. Moving the
    // PV to INT8 -- what the WMMA path does, and what SageAttention does on
    // silicon with a matrix unit -- would need 512 v_dot4 at 1/clk, the same 512
    // cycles, plus ~3x the LDS reads (four P bytes per (row, key group) instead of
    // four halves per (row, key pair), re-read once per head-dim group) and a V
    // quantization pass. It is strictly more work here, which is why this port
    // keeps P in f16 and calls itself i8q_f16pv.
    int diag = 0;
    if (const char* e = getenv("CK_GFX103X_DIAG")) {
        diag = atoi(e);
    }

    ctx.block = dim3(128);
    ctx.grid = dim3(static_cast<unsigned>((in.qo_len + 127) / 128),
                    static_cast<unsigned>(in.q_heads), static_cast<unsigned>(in.batch));
    ctx.stream = stream;
    ctx.mask_mode = gmask;
    ctx.mask_dtype = in.mask_dtype;
    ctx.diag = diag;

    // One launcher covers every combination the old macro set spelled out. The
    // tile shape (head dim, causal, key tile, rows per thread, in-kernel Q
    // quantization) and the mask representation are all template arguments, so a
    // masked launch costs nothing the unmasked one did not already pay: the
    // lookup sits behind a constant, and the MaskMode::kNone instantiation
    // compiles to exactly the code it did before masks existed.
    //
    // Only the mask mode and its dtype are left as runtime switches, and they
    // cost one branch per launch. Everything else is a template argument because
    // a runtime tile choice would put a branch and a live register on the inner
    // loop's critical path, where the register budget is already the binding
    // constraint on this architecture.
    //
    // causal selects the triangular schedule. comfy_kitchen's int8 attention has
    // no causal mode, but the ported op exposes the flag, so the instantiation is
    // kept rather than dropped.
    if (in.causal) { launch_attn_cf<true>(ctx, in.head_dim, inq, in.out_bf16 != 0, v10_bn, tm); }
    else           { launch_attn_cf<false>(ctx, in.head_dim, inq, in.out_bf16 != 0, v10_bn, tm); }
}

}  // namespace sageattn_gfx103x
