// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Fused ConvRot rotation + rowwise quantization.
//
// The regular Hadamard-G (G in {16, 64, 256}) is kron(H4, ...) / sqrt(G) and so
// factors into log4(G) radix-4 butterfly stages over the base-4 digits of the
// index, matching the eager _rotate_activation reference.
//
// INT8 G=256: fused single-kernel path when K fits in LDS, otherwise a global
// rotate + quantize split. Legacy G=16/64 and int4 paths stage the whole row in
// LDS (one block per row). INT4 output packs two nibbles per byte (low nibble =
// even index), which is the layout the iu4 A-fragment consumes directly.
#pragma once

#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "rope_math.h"
#include "swiglu_bf16.h"

namespace comfy::hip_backend {

typedef __bf16 convrot_bf16x2 __attribute__((ext_vector_type(2)));
typedef __bf16 convrot_bf16x4 __attribute__((ext_vector_type(4)));

extern "C" __device__ float __ocml_rsqrt_f32(float);

__forceinline__ __device__ float ieee_div_f32(
    float numerator, float denominator) {
    bool denominator_scale = false;
    bool scale = false;
    const float scaled_denominator = __builtin_amdgcn_div_scalef(
        numerator, denominator, false, &denominator_scale);
    const float scaled_numerator = __builtin_amdgcn_div_scalef(
        numerator, denominator, true, &scale);
    float reciprocal = __builtin_amdgcn_rcpf(scaled_denominator);
    const float reciprocal_error =
        fmaf(-scaled_denominator, reciprocal, 1.0f);
    reciprocal = fmaf(reciprocal_error, reciprocal, reciprocal);
    float quotient = scaled_numerator * reciprocal;
    float remainder =
        fmaf(-scaled_denominator, quotient, scaled_numerator);
    quotient = fmaf(remainder, reciprocal, quotient);
    remainder = fmaf(-scaled_denominator, quotient, scaled_numerator);
    quotient = __builtin_amdgcn_div_fmasf(
        remainder, reciprocal, quotient, scale);
    return __builtin_amdgcn_div_fixupf(
        quotient, denominator, numerator);
}

// convrot_quant_kernel handles 256/G groups per pass and rotates in log4(G)
// stages, so a G outside this set either divides to a zero-width pass or is not
// a power of four. The dispatch wrappers fall back to eager before reaching here.
inline void check_convrot_group_size(int group_size) {
    if (group_size != 16 && group_size != 64 && group_size != 256) {
        throw std::runtime_error("convrot: group_size must be 16, 64 or 256");
    }
}

// Legacy convrot_quant_kernel static LDS: g[BLOCK_THREADS] + red[BLOCK_THREADS].
inline size_t convrot_static_lds_bytes(int block_threads) {
    return 2 * static_cast<size_t>(block_threads) * sizeof(float);
}

constexpr int kConvRotGroup256 = 256;

// convrot_quant_kernel stages the whole rotated row in dynamic LDS, preserving
// the input dtype's rounding contract. K is therefore bounded by both the
// workgroup budget and the input element size. Past it the wrappers fall back
// to eager instead. 0 means unknown or an invalid dtype code.
inline size_t convrot_row_element_size(int in_dtype) {
    if (in_dtype == 0) return sizeof(float);
    if (in_dtype == 1) return sizeof(__half);
    if (in_dtype == 2) return sizeof(__bf16);
    return 0;
}

inline int convrot_max_k(int in_dtype, int block_threads = 256) {
    const size_t element_size = convrot_row_element_size(in_dtype);
    if (element_size == 0 || block_threads <= 0) {
        return 0;
    }
    int device = 0;
    if (hipGetDevice(&device) != hipSuccess) {
        return 0;
    }
    int lds = 0;
    if (hipDeviceGetAttribute(&lds, hipDeviceAttributeMaxSharedMemoryPerBlock, device) !=
            hipSuccess) {
        return 0;
    }
    const size_t static_lds = convrot_static_lds_bytes(block_threads);
    if (lds <= static_cast<int>(static_lds)) {
        return 0;
    }
    return static_cast<int>((static_cast<size_t>(lds) - static_lds) / element_size);
}

inline int convrot_quant_fused_block_threads(int M, int K) {
    if (M == 1) {
        return 512;
    }
    if (K == kConvRotGroup256) {
        return 64;
    }
    if (K == 2560) {
        return 640;
    }
    if (K == 6144) {
        return 768;
    }
    // Empirical block-size tuning for K=10240 (640 over default).
    if (K == 10240) {
        return 640;
    }
    return 1024;
}

inline int convrot_device_lds_limit() {
    int device = 0;
    if (hipGetDevice(&device) != hipSuccess) {
        return 0;
    }
    int lds = 0;
    if (hipDeviceGetAttribute(&lds, hipDeviceAttributeMaxSharedMemoryPerBlock, device) !=
            hipSuccess ||
        lds <= 0) {
        return 0;
    }
    return lds;
}

inline bool convrot_fused_lds_fits(int K, int block_threads, int in_dtype) {
    if (block_threads <= 0 || (block_threads % 64) != 0) {
        return false;
    }
    const size_t row_element_size = convrot_row_element_size(in_dtype);
    if (row_element_size == 0) {
        return false;
    }
    const int lds = convrot_device_lds_limit();
    if (lds <= 0) {
        return false;
    }
    const int groups_in_flight = block_threads / 64;
    const size_t need =
        (static_cast<size_t>(block_threads / 32) + 1) * sizeof(float) +
        static_cast<size_t>(K) * row_element_size +
        static_cast<size_t>(groups_in_flight) * 2 * kConvRotGroup256 * sizeof(float);
    return need <= static_cast<size_t>(lds);
}

// Heuristic block first, then narrower blocks before global spill. Returns 0 to spill.
inline int convrot_pick_fused_block_threads(int M, int K, int in_dtype) {
    const int preferred = convrot_quant_fused_block_threads(M, K);
    if (convrot_fused_lds_fits(K, preferred, in_dtype)) {
        return preferred;
    }
    static constexpr int kFallbackBlocks[] = {768, 640, 512, 64};
    for (int block_threads : kFallbackBlocks) {
        // Once a wide row can only fit the one-wave schedule, each workgroup
        // serializes dozens of G=256 rotations. The cooperative
        // two-pass path is the scalable fallback for these long inference
        // rows; retain one-wave LDS for short-M calls where its launch and
        // workspace savings still matter.
        if (block_threads == 64 && M >= 96 && K >= 12288) {
            continue;
        }
        if (block_threads < preferred && convrot_fused_lds_fits(K, block_threads, in_dtype)) {
            return block_threads;
        }
    }
    return 0;
}

// Mirrors launch_convrot_quant() spill fallback for Python buffer allocation.
inline bool convrot_int8_needs_spill_buffers(int M, int K, int in_dtype) {
    if (M <= 0 || K <= 0 || (K % kConvRotGroup256) != 0) {
        return false;
    }
    if (convrot_row_element_size(in_dtype) == 0 || convrot_device_lds_limit() <= 0) {
        return true;
    }
    const int block = convrot_pick_fused_block_threads(M, K, in_dtype);
    return block == 0 || !convrot_fused_lds_fits(K, block, in_dtype);
}

inline void check_convrot_k(int k, int group_size, int in_dtype, bool int8_quant = false) {
    // The kernel rotates K/G whole groups but reads back all K entries of the row
    // buffer, so a partial trailing group would quantize uninitialized LDS.
    if (group_size <= 0 || k % group_size != 0) {
        throw std::runtime_error("convrot: K=" + std::to_string(k) +
                                 " is not divisible by group_size=" + std::to_string(group_size));
    }
    // INT8 G=256 uses fused LDS or global spill; int4 and G=16/64 still stage the
    // whole row in dynamic LDS and are bounded by convrot_max_k().
    if (int8_quant && group_size == 256) {
        return;
    }
    const int legacy_bt = (group_size == 256 && k >= 512) ? 1024 : 256;
    const int max_k = convrot_max_k(in_dtype, legacy_bt);
    if (max_k <= 0 || k > max_k) {
        throw std::runtime_error("convrot: K=" + std::to_string(k) +
                                 " does not fit in LDS (max " + std::to_string(max_k) + ")");
    }
}

// Runtime in_dtype (0/1/2) -> RowT template dispatch for convrot launchers.
template <typename Fn>
inline void dispatch_convrot_row_type(int in_dtype, Fn fn) {
    if (in_dtype == 0) {
        fn.template operator()<float>();
    } else if (in_dtype == 1) {
        fn.template operator()<__half>();
    } else if (in_dtype == 2) {
        fn.template operator()<__bf16>();
    } else {
        throw std::runtime_error("convrot: unsupported input dtype code");
    }
}

// Pick fused-kernel block width from convrot_quant_fused_block_threads().
template <typename Fn>
inline void dispatch_convrot_fused_block_threads(int block_threads, Fn fn) {
    if (block_threads == 64) {
        fn.template operator()<64>();
    } else if (block_threads == 512) {
        fn.template operator()<512>();
    } else if (block_threads == 640) {
        fn.template operator()<640>();
    } else if (block_threads == 768) {
        fn.template operator()<768>();
    } else {
        fn.template operator()<1024>();
    }
}

// dtype codes match comfy_kitchen.backends.eager.quantization.DTYPE_TO_CODE
__forceinline__ __device__ float load_in(const void* x, int64_t idx, int code) {
    if (code == 0) return static_cast<const float*>(x)[idx];
    if (code == 1) return __half2float(static_cast<const __half*>(x)[idx]);
    return static_cast<float>(static_cast<const __bf16*>(x)[idx]);
}

// Codes match comfy_kitchen.backends._activations.INPUT_ACT_TO_CODE. SwiGLU is
// the gated pair: the raw row is [gate | up] (2*K wide) and the activated row
// silu(gate) * up is K wide; the others are elementwise.
enum : int { kActNone = 0, kActGeluTanh = 1, kActSwiGLU = 2 };

inline void check_convrot_act(int act) {
    if (act != kActNone && act != kActGeluTanh && act != kActSwiGLU) {
        throw std::runtime_error("convrot: unsupported input activation code");
    }
}

template <int ACT>
__forceinline__ __device__ float apply_input_act(float v) {
    if constexpr (ACT == kActGeluTanh) {
        // Matches torch.nn.functional.gelu(x, approximate="tanh").
        constexpr float kBeta = 0.7978845608028654f;  // sqrt(2/pi)
        constexpr float kKappa = 0.044715f;
        return 0.5f * v * (1.0f + tanhf(kBeta * (v + kKappa * v * v * v)));
    }
    return v;
}

// One activated value: column `col` of the K-wide activated row starting at
// `in_row`. SwiGLU reads the gate at col and the up at K + col; every other
// activation reads the same K-wide row it writes.
template <int ACT>
__forceinline__ __device__ float load_input_act(
    const void* x, int64_t in_row, int col, int K, int code) {
    if constexpr (ACT == kActSwiGLU) {
        // Matches torch silu(gate) * up.
        const float gate = load_in(x, in_row + col, code);
        const float up = load_in(x, in_row + K + col, code);
        return (gate / (1.0f + expf(-gate))) * up;
    } else {
        return apply_input_act<ACT>(load_in(x, in_row + col, code));
    }
}

__forceinline__ __device__ void load_input_act4_bf16(
    const void* x, int64_t in_row, int col, float& o0, float& o1, float& o2, float& o3) {
    const __bf16* row = static_cast<const __bf16*>(x) + in_row + col;
    const uint64_t pack = *reinterpret_cast<const uint64_t*>(row);
    const __bf16* elems = reinterpret_cast<const __bf16*>(&pack);
    o0 = static_cast<float>(elems[0]);
    o1 = static_cast<float>(elems[1]);
    o2 = static_cast<float>(elems[2]);
    o3 = static_cast<float>(elems[3]);
}

template <typename RowT>
__forceinline__ __device__ RowT store_row_value(float v) {
    return static_cast<RowT>(v);
}

template <>
__forceinline__ __device__ __half store_row_value<__half>(float v) {
    return __float2half(v);
}

template <typename RowT>
__forceinline__ __device__ float load_row_value(RowT v) {
    return static_cast<float>(v);
}

template <>
__forceinline__ __device__ float load_row_value<__half>(__half v) {
    return __half2float(v);
}

constexpr int kWarpSize = 32;

__forceinline__ __device__ float warp_reduce_max(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down(v, offset));
    }
    return v;
}

template <int NUM_WARPS>
__forceinline__ __device__ float block_reduce_max(float v, float* warp_smem, float* block_smem) {
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int wid = threadIdx.x >> 5;
    v = warp_reduce_max(v);
    if (lane == 0) {
        warp_smem[wid] = v;
    }
    __syncthreads();
    if (wid == 0) {
        float total = lane < NUM_WARPS ? warp_smem[lane] : 0.0f;
        total = warp_reduce_max(total);
        if (lane == 0) {
            *block_smem = total;
        }
    }
    __syncthreads();
    return *block_smem;
}

template <int S>
__forceinline__ __device__ void convrot_fht_stage64(
    const float* __restrict__ src, float* __restrict__ dst, int lane) {
    const int base = (lane % S) + (lane / S) * (4 * S);
    const float x0 = src[base];
    const float x1 = src[base + S];
    const float x2 = src[base + 2 * S];
    const float x3 = src[base + 3 * S];
    dst[base] = 0.5f * (x0 + x1 + x2 - x3);
    dst[base + S] = 0.5f * (x0 + x1 - x2 + x3);
    dst[base + 2 * S] = 0.5f * (x0 - x1 + x2 + x3);
    dst[base + 3 * S] = 0.5f * (-x0 + x1 + x2 + x3);
}

template <int S>
__forceinline__ __device__ float convrot_fht_stage64_store_absmax(
    const float* __restrict__ src, float* __restrict__ row_buf, int lane) {
    const int base = (lane % S) + (lane / S) * (4 * S);
    const float x0 = src[base];
    const float x1 = src[base + S];
    const float x2 = src[base + 2 * S];
    const float x3 = src[base + 3 * S];
    const float y0 = 0.5f * (x0 + x1 + x2 - x3);
    const float y1 = 0.5f * (x0 + x1 - x2 + x3);
    const float y2 = 0.5f * (x0 - x1 + x2 + x3);
    const float y3 = 0.5f * (-x0 + x1 + x2 + x3);
    row_buf[base] = y0;
    row_buf[base + S] = y1;
    row_buf[base + 2 * S] = y2;
    row_buf[base + 3 * S] = y3;
    return fmaxf(fmaxf(fabsf(y0), fabsf(y1)), fmaxf(fabsf(y2), fabsf(y3)));
}

template <int S, typename RowT>
__forceinline__ __device__ float convrot_fht_stage64_store_absmax_typed(
    const float* __restrict__ src, RowT* __restrict__ output, int lane) {
    const int base = (lane % S) + (lane / S) * (4 * S);
    const float x0 = src[base];
    const float x1 = src[base + S];
    const float x2 = src[base + 2 * S];
    const float x3 = src[base + 3 * S];
    const float y0 = 0.5f * (x0 + x1 + x2 - x3);
    const float y1 = 0.5f * (x0 + x1 - x2 + x3);
    const float y2 = 0.5f * (x0 - x1 + x2 + x3);
    const float y3 = 0.5f * (-x0 + x1 + x2 + x3);
    output[base] = store_row_value<RowT>(y0);
    output[base + S] = store_row_value<RowT>(y1);
    output[base + 2 * S] = store_row_value<RowT>(y2);
    output[base + 3 * S] = store_row_value<RowT>(y3);
    const float a0 = fabsf(load_row_value(output[base]));
    const float a1 = fabsf(load_row_value(output[base + S]));
    const float a2 = fabsf(load_row_value(output[base + 2 * S]));
    const float a3 = fabsf(load_row_value(output[base + 3 * S]));
    return fmaxf(fmaxf(a0, a1), fmaxf(a2, a3));
}

template <typename RowT>
__forceinline__ __device__ float finite_absmax_for_quant(float abs_max) {
    if constexpr (std::is_same_v<RowT, __bf16>) {
        return fminf(abs_max, 3.38953139e38f);
    }
    if constexpr (std::is_same_v<RowT, __half>) {
        return fminf(abs_max, 65504.0f);
    }
    return abs_max;
}

// Match ``torch.addcmul(shift, x, 1 + scale)`` when all operands and the
// materialized result are BF16.  The add that forms the scale factor is a
// separate PyTorch operation, while addcmul evaluates the multiply-add in
// FP32 before rounding its output once to BF16.
__forceinline__ __device__ float load_affine_modulated_bf16(
    const void* __restrict__ x, const void* __restrict__ modulation_scale,
    const void* __restrict__ modulation_shift, int64_t index, int column) {
    const float factor = round_bf16(
        1.0f + static_cast<float>(
            static_cast<const __bf16*>(modulation_scale)[column]));
    const float value = static_cast<float>(static_cast<const __bf16*>(x)[index]);
    const float shift = static_cast<float>(
        static_cast<const __bf16*>(modulation_shift)[column]);
    return round_bf16(fmaf(value, factor, shift));
}

constexpr int kConvrotGlobalGroupsPerBlock = 4;
constexpr int kConvrotGlobalBlockThreads = kConvrotGlobalGroupsPerBlock * 64;
constexpr size_t kConvrotGlobalSmemBytes =
    static_cast<size_t>(kConvrotGlobalGroupsPerBlock) * 2 * kConvRotGroup256 * sizeof(float);

// Large K: rotate 8 groups/block into global memory, record per-group absmax, then
// quantize in a second kernel. Fixed 16 KiB LDS regardless of K.
template <int GROUPS_PER_BLOCK, typename RowT, int ACT>
__global__ __launch_bounds__(GROUPS_PER_BLOCK* 64) void convrot_rotate_groups64_amax_kernel(
    const void* __restrict__ x, int in_dtype, RowT* __restrict__ rotated,
    float* __restrict__ partial_absmax, int K) {
    constexpr int kGroupThreads = 64;
    extern __shared__ float smem[];

    const int sub = threadIdx.x / kGroupThreads;
    const int lane = threadIdx.x % kGroupThreads;
    const int group = static_cast<int>(blockIdx.y) * GROUPS_PER_BLOCK + sub;
    const int64_t row = blockIdx.x;
    const int n_groups = K / kConvRotGroup256;
    const bool active = group < n_groups;
    const int64_t row_offset = row * K;
    constexpr int kInWidth = (ACT == kActSwiGLU) ? 2 : 1;
    const int64_t in_row_offset = row_offset * kInWidth;
    const int group_col = group * kConvRotGroup256;

    float* buf0 = smem + sub * (2 * kConvRotGroup256);
    float* buf1 = buf0 + kConvRotGroup256;

    const int base = lane * 4;
    const int col = group_col + base;
    float xv0 = 0.0f;
    float xv1 = 0.0f;
    float xv2 = 0.0f;
    float xv3 = 0.0f;
    if (active) {
        if constexpr (ACT == kActNone) {
            if (in_dtype == 2) {
                load_input_act4_bf16(x, in_row_offset, col, xv0, xv1, xv2, xv3);
            } else {
                xv0 = load_input_act<ACT>(x, in_row_offset, col, K, in_dtype);
                xv1 = load_input_act<ACT>(x, in_row_offset, col + 1, K, in_dtype);
                xv2 = load_input_act<ACT>(x, in_row_offset, col + 2, K, in_dtype);
                xv3 = load_input_act<ACT>(x, in_row_offset, col + 3, K, in_dtype);
            }
        } else {
            xv0 = load_input_act<ACT>(x, in_row_offset, col, K, in_dtype);
            xv1 = load_input_act<ACT>(x, in_row_offset, col + 1, K, in_dtype);
            xv2 = load_input_act<ACT>(x, in_row_offset, col + 2, K, in_dtype);
            xv3 = load_input_act<ACT>(x, in_row_offset, col + 3, K, in_dtype);
        }
    }
    buf1[base] = 0.5f * (xv0 + xv1 + xv2 - xv3);
    buf1[base + 1] = 0.5f * (xv0 + xv1 - xv2 + xv3);
    buf1[base + 2] = 0.5f * (xv0 - xv1 + xv2 + xv3);
    buf1[base + 3] = 0.5f * (-xv0 + xv1 + xv2 + xv3);
    __syncthreads();

    convrot_fht_stage64<4>(buf1, buf0, lane);
    __syncthreads();
    convrot_fht_stage64<16>(buf0, buf1, lane);
    __syncthreads();

    float local_max = 0.0f;
    if (active) {
        local_max = convrot_fht_stage64_store_absmax_typed<64, RowT>(
            buf1, rotated + row_offset + group_col, lane);
    }
    buf0[lane] = local_max;
    __syncthreads();

    if (lane < 32) {
        float v = fmaxf(buf0[lane], buf0[lane + 32]);
        v = warp_reduce_max(v);
        if (lane == 0 && active) {
            partial_absmax[static_cast<int64_t>(row) * n_groups + group] = v;
        }
    }
}

// Global spill pass 2: fold per-group absmax into the row scale, then quantize the
// rotated row already in global memory (pass 1 is convrot_rotate_groups64_amax_kernel).
template <typename RowT, int BLOCK_THREADS>
__global__ __launch_bounds__(BLOCK_THREADS) void convrot_quant_from_partials_kernel(
    const RowT* __restrict__ rotated, const float* __restrict__ partial_absmax,
    int8_t* __restrict__ qout, float* __restrict__ scaleout, int K) {
    constexpr int kWarps = BLOCK_THREADS / kWarpSize;
    __shared__ float warp_smem[kWarps];
    __shared__ float block_smem;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int n_groups = K / kConvRotGroup256;
    const int64_t row_offset = static_cast<int64_t>(row) * K;
    const float* row_partials = partial_absmax + static_cast<int64_t>(row) * n_groups;

    float abs_max = 0.0f;
    for (int g = tid; g < n_groups; g += BLOCK_THREADS) {
        abs_max = fmaxf(abs_max, row_partials[g]);
    }
    abs_max = block_reduce_max<kWarps>(abs_max, warp_smem, &block_smem);
    const float rowmax = fmaxf(finite_absmax_for_quant<RowT>(abs_max), 1e-10f);
    const float scale = rowmax / 127.0f;
    const float inv = 127.0f / rowmax;
    if (tid == 0) {
        scaleout[row] = scale;
    }

    for (int col = tid; col < K; col += BLOCK_THREADS) {
        const float v = load_row_value<RowT>(rotated[row_offset + col]);
        int q = static_cast<int>(rintf(v * inv));
        q = q < -127 ? -127 : (q > 127 ? 127 : q);
        qout[row_offset + col] = static_cast<int8_t>(q);
    }
}

template <typename RowT, int ACT>
inline void launch_convrot_quant_global(
    const void* x, int in_dtype, int8_t* qout, float* scaleout, int M, int K,
    RowT* rotated, float* partial_absmax, hipStream_t stream) {
    const int n_groups = K / kConvRotGroup256;
    const int group_blocks =
        (n_groups + kConvrotGlobalGroupsPerBlock - 1) / kConvrotGlobalGroupsPerBlock;
    const dim3 rotate_grid(static_cast<unsigned int>(M), static_cast<unsigned int>(group_blocks));
    convrot_rotate_groups64_amax_kernel<kConvrotGlobalGroupsPerBlock, RowT, ACT>
        <<<rotate_grid, kConvrotGlobalBlockThreads, kConvrotGlobalSmemBytes, stream>>>(
            x, in_dtype, rotated, partial_absmax, K);

    const int quant_threads = K >= 4096 ? 512 : 256;
    if (quant_threads == 512) {
        convrot_quant_from_partials_kernel<RowT, 512>
            <<<M, 512, 0, stream>>>(rotated, partial_absmax, qout, scaleout, K);
    } else {
        convrot_quant_from_partials_kernel<RowT, 256>
            <<<M, 256, 0, stream>>>(rotated, partial_absmax, qout, scaleout, K);
    }
}

template <int ACT>
void launch_convrot_quant_global_managed(
    const void* x, int in_dtype, int8_t* qout, float* scaleout, int M, int K, hipStream_t stream,
    void* spill_rotated, void* spill_partials);

// Fused single-kernel path: FHT in LDS, vectorized loads, warp-shuffle absmax.
// One block per row; the rotated row stays in shared memory as RowT.
template <typename RowT, int BLOCK_THREADS, int ACT, bool AFFINE_MODULATE = false>
__global__ __launch_bounds__(BLOCK_THREADS) void convrot_quant_fused_kernel(
    const void* __restrict__ x, int in_dtype, int8_t* __restrict__ qout,
    float* __restrict__ scaleout, int M, int K,
    const void* __restrict__ modulation_scale = nullptr,
    const void* __restrict__ modulation_shift = nullptr) {

    static_assert(!AFFINE_MODULATE || std::is_same_v<RowT, __bf16>);
    static_assert(!AFFINE_MODULATE || ACT == kActNone);

    constexpr int kGroupThreads = 64;
    constexpr int kGroupsInFlight = BLOCK_THREADS / kGroupThreads;
    constexpr int kWarps = BLOCK_THREADS / kWarpSize;

    extern __shared__ unsigned char smem_raw[];
    RowT* row_buf = reinterpret_cast<RowT*>(smem_raw);
    float* tmp = reinterpret_cast<float*>(smem_raw + static_cast<size_t>(K) * sizeof(RowT));

    __shared__ float warp_smem[kWarps];
    __shared__ float block_smem;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int sub = tid / kGroupThreads;
    const int lane = tid % kGroupThreads;
    const int64_t row_offset = static_cast<int64_t>(row) * K;
    constexpr int kInWidth = (ACT == kActSwiGLU) ? 2 : 1;
    const int64_t in_row_offset = row_offset * kInWidth;
    const int n_groups = K / kConvRotGroup256;

    float* buf0 = tmp + sub * (2 * kConvRotGroup256);
    float* buf1 = buf0 + kConvRotGroup256;
    float abs_max = 0.0f;

    const int iters = (n_groups + kGroupsInFlight - 1) / kGroupsInFlight;
    for (int it = 0; it < iters; ++it) {
        const int group = it * kGroupsInFlight + sub;
        const bool active = group < n_groups;
        const int base = lane * 4;
        const int group_col = group * kConvRotGroup256;
        const int col = group_col + base;

        float xv0 = 0.0f;
        float xv1 = 0.0f;
        float xv2 = 0.0f;
        float xv3 = 0.0f;
        if (active) {
            if constexpr (AFFINE_MODULATE) {
                xv0 = load_affine_modulated_bf16(
                    x, modulation_scale, modulation_shift,
                    in_row_offset + col, col);
                xv1 = load_affine_modulated_bf16(
                    x, modulation_scale, modulation_shift,
                    in_row_offset + col + 1, col + 1);
                xv2 = load_affine_modulated_bf16(
                    x, modulation_scale, modulation_shift,
                    in_row_offset + col + 2, col + 2);
                xv3 = load_affine_modulated_bf16(
                    x, modulation_scale, modulation_shift,
                    in_row_offset + col + 3, col + 3);
            } else if constexpr (ACT == kActNone) {
                if (in_dtype == 2) {
                    load_input_act4_bf16(x, in_row_offset, col, xv0, xv1, xv2, xv3);
                } else {
                    xv0 = load_input_act<ACT>(x, in_row_offset, col, K, in_dtype);
                    xv1 = load_input_act<ACT>(x, in_row_offset, col + 1, K, in_dtype);
                    xv2 = load_input_act<ACT>(x, in_row_offset, col + 2, K, in_dtype);
                    xv3 = load_input_act<ACT>(x, in_row_offset, col + 3, K, in_dtype);
                }
            } else {
                xv0 = load_input_act<ACT>(x, in_row_offset, col, K, in_dtype);
                xv1 = load_input_act<ACT>(x, in_row_offset, col + 1, K, in_dtype);
                xv2 = load_input_act<ACT>(x, in_row_offset, col + 2, K, in_dtype);
                xv3 = load_input_act<ACT>(x, in_row_offset, col + 3, K, in_dtype);
            }
        }
        buf1[base] = 0.5f * (xv0 + xv1 + xv2 - xv3);
        buf1[base + 1] = 0.5f * (xv0 + xv1 - xv2 + xv3);
        buf1[base + 2] = 0.5f * (xv0 - xv1 + xv2 + xv3);
        buf1[base + 3] = 0.5f * (-xv0 + xv1 + xv2 + xv3);
        __syncthreads();

        convrot_fht_stage64<4>(buf1, buf0, lane);
        __syncthreads();
        convrot_fht_stage64<16>(buf0, buf1, lane);
        __syncthreads();

        if (active) {
            abs_max = fmaxf(
                abs_max,
                convrot_fht_stage64_store_absmax_typed<64, RowT>(
                    buf1, row_buf + group_col, lane));
        }
        __syncthreads();
    }

    abs_max = block_reduce_max<kWarps>(abs_max, warp_smem, &block_smem);
    const float rowmax = fmaxf(finite_absmax_for_quant<RowT>(abs_max), 1e-10f);
    const float scale = rowmax / 127.0f;
    const float inv = 127.0f / rowmax;
    if (tid == 0) {
        scaleout[row] = scale;
    }

    for (int col = tid; col < K; col += BLOCK_THREADS) {
        const float v = load_row_value(row_buf[col]);
        int q = static_cast<int>(rintf(v * inv));
        q = q < -127 ? -127 : (q > 127 ? 127 : q);
        qout[row_offset + col] = static_cast<int8_t>(q);
    }
}

template <typename RowT, int ACT, int BLOCK_THREADS, bool AFFINE_MODULATE = false>
inline bool launch_convrot_quant_fused_impl(
    const void* x, int in_dtype, int8_t* qout, float* scaleout, int M, int K,
    hipStream_t stream, const void* modulation_scale = nullptr,
    const void* modulation_shift = nullptr) {
    const int groups_in_flight = BLOCK_THREADS / 64;
    const size_t shmem =
        static_cast<size_t>(K) * sizeof(RowT) +
        static_cast<size_t>(groups_in_flight) * 2 * kConvRotGroup256 * sizeof(float);
    auto kernel = convrot_quant_fused_kernel<
        RowT, BLOCK_THREADS, ACT, AFFINE_MODULATE>;
    const hipError_t attr_err = hipFuncSetAttribute(
        reinterpret_cast<const void*>(kernel), hipFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shmem));
    if (attr_err != hipSuccess) {
        return false;
    }
    kernel<<<M, BLOCK_THREADS, shmem, stream>>>(
        x, in_dtype, qout, scaleout, M, K,
        modulation_scale, modulation_shift);
    return hipGetLastError() == hipSuccess;
}

template <typename RowT, int ACT>
struct LaunchConvrotQuantFusedForBlock {
    const void* x;
    int in_dtype;
    int8_t* qout;
    float* scaleout;
    int M;
    int K;
    hipStream_t stream;
    bool* launched;

    template <int BLOCK_THREADS>
    void operator()() const {
        *launched = launch_convrot_quant_fused_impl<RowT, ACT, BLOCK_THREADS>(
            x, in_dtype, qout, scaleout, M, K, stream);
    }
};

template <int BLOCK_THREADS>
inline bool launch_convrot_quant_affine_fused_impl(
    const void* x, const void* modulation_scale,
    const void* modulation_shift, int8_t* qout, float* scaleout,
    int M, int K, hipStream_t stream) {
    return launch_convrot_quant_fused_impl<
        __bf16, kActNone, BLOCK_THREADS, true>(
            x, 2, qout, scaleout, M, K, stream,
            modulation_scale, modulation_shift);
}

struct LaunchConvrotQuantAffineForBlock {
    const void* x;
    const void* modulation_scale;
    const void* modulation_shift;
    int8_t* qout;
    float* scaleout;
    int M;
    int K;
    hipStream_t stream;
    bool* launched;

    template <int BLOCK_THREADS>
    void operator()() const {
        *launched = launch_convrot_quant_affine_fused_impl<BLOCK_THREADS>(
            x, modulation_scale, modulation_shift, qout, scaleout,
            M, K, stream);
    }
};

inline bool launch_convrot_quant_affine_bf16(
    const void* x, const void* modulation_scale,
    const void* modulation_shift, int8_t* qout, float* scaleout,
    int M, int K, hipStream_t stream) {
    const int block_threads = convrot_pick_fused_block_threads(M, K, /*bf16*/ 2);
    if (block_threads == 0) {
        return false;
    }
    bool launched = false;
    dispatch_convrot_fused_block_threads(
        block_threads,
        LaunchConvrotQuantAffineForBlock{
            x, modulation_scale, modulation_shift, qout, scaleout,
            M, K, stream, &launched});
    return launched;
}

template <int ACT>
struct LaunchConvrotQuantFusedForRow {
    const void* x;
    int in_dtype;
    int8_t* qout;
    float* scaleout;
    int M;
    int K;
    hipStream_t stream;
    int block_threads;
    bool* launched;

    template <typename RowT>
    void operator()() const {
        dispatch_convrot_fused_block_threads(
            block_threads,
            LaunchConvrotQuantFusedForBlock<RowT, ACT>{
                x, in_dtype, qout, scaleout, M, K, stream, launched});
    }
};

template <typename RowT, bool PACK_INT4, int ACT, int BLOCK_THREADS>
__global__ __launch_bounds__(BLOCK_THREADS) void convrot_quant_kernel(
    const void* __restrict__ x, int in_dtype,
    int8_t* __restrict__ qout, float* __restrict__ scaleout,
    int M, int K, int G) {

    const float h4[4][4] = {{1, 1, 1, -1}, {1, 1, -1, 1}, {1, -1, 1, 1}, {-1, 1, 1, 1}};
    __shared__ float g[BLOCK_THREADS];
    __shared__ float red[BLOCK_THREADS];
    extern __shared__ unsigned char rowbuf_raw[];
    RowT* rowbuf = reinterpret_cast<RowT*>(rowbuf_raw);  // K entries: the rotated row

    const int row = blockIdx.x;
    const int t = threadIdx.x;

    int nstages = 0;
    while ((1 << (2 * nstages)) < G) nstages++;  // log4(G)

    const int gpw = BLOCK_THREADS / G;           // groups handled per pass
    const int glocal = t / G;                    // this thread's group within the pass
    const int e = t % G;                         // element within the group
    const int gbase_idx = glocal * G;
    const float norm = rsqrtf(static_cast<float>(G));
    const int ngrp = K / G;
    // SwiGLU reads a [gate | up] raw row twice as wide as the K it writes.
    constexpr int kInWidth = (ACT == kActSwiGLU) ? 2 : 1;
    const int64_t in_row = static_cast<int64_t>(row) * K * kInWidth;

    float lmax = 0.0f;
    for (int gbase = 0; gbase < ngrp; gbase += gpw) {
        const int grp = gbase + glocal;
        const bool active = grp < ngrp;
        g[t] = active ? load_input_act<ACT>(x, in_row, grp * G + e, K, in_dtype) : 0.0f;
        __syncthreads();

        for (int stage = 0; stage < nstages; ++stage) {
            const int stride = 1 << (2 * stage);
            const int ds = (e / stride) & 3;
            const int b = gbase_idx + (e - ds * stride);
            const float v0 = g[b], v1 = g[b + stride], v2 = g[b + 2 * stride], v3 = g[b + 3 * stride];
            const float nv = h4[ds][0] * v0 + h4[ds][1] * v1 + h4[ds][2] * v2 + h4[ds][3] * v3;
            __syncthreads();
            g[t] = nv;
            __syncthreads();
        }

        if (active) {
            const float tv = g[t] * norm;
            const RowT stored = store_row_value<RowT>(tv);
            rowbuf[static_cast<int64_t>(grp) * G + e] = stored;
            lmax = fmaxf(lmax, fabsf(load_row_value(stored)));
        }
        __syncthreads();
    }

    red[t] = lmax;
    __syncthreads();
    for (int s = BLOCK_THREADS / 2; s > 0; s >>= 1) {
        if (t < s) red[t] = fmaxf(red[t], red[t + s]);
        __syncthreads();
    }

    constexpr float kQMax = PACK_INT4 ? 7.0f : 127.0f;
    const float rowmax = fmaxf(red[0], 1e-10f);
    const float scale = rowmax / kQMax;
    const float inv = kQMax / rowmax;
    if (t == 0) scaleout[row] = scale;

    if constexpr (PACK_INT4) {
        const int Kp = K / 2;
        for (int jb = t; jb < Kp; jb += BLOCK_THREADS) {
            const float a = load_row_value(rowbuf[2 * jb]);
            const float b = load_row_value(rowbuf[2 * jb + 1]);
            int qa = static_cast<int>(rintf(a * inv));
            int qb = static_cast<int>(rintf(b * inv));
            qa = qa < -7 ? -7 : (qa > 7 ? 7 : qa);
            qb = qb < -7 ? -7 : (qb > 7 ? 7 : qb);
            qout[static_cast<int64_t>(row) * Kp + jb] =
                static_cast<int8_t>(((qa & 0xF) | ((qb & 0xF) << 4)) & 0xFF);
        }
    } else {
        for (int j = t; j < K; j += BLOCK_THREADS) {
            const float v = load_row_value(rowbuf[j]);
            int q = static_cast<int>(rintf(v * inv));
            q = q < -127 ? -127 : (q > 127 ? 127 : q);
            qout[static_cast<int64_t>(row) * K + j] = static_cast<int8_t>(q);
        }
    }
}

// Wide-row ConvRot-256: four groups per pass. Wave32 shuffles cover the first
// two radix-4 stages; the rest use ping-pong LDS (512 threads, two values each,
// two barriers). Bank-aligned FP32 LDS. Not an arch-specific WMMA path; keep
// separate from convrot_quant_kernel for other shapes and devices.
template <int ACT, bool TILED_QOUT = false, bool SPLIT_SWIGLU = false,
          bool MODULATE = false, bool PACK_QUANT = false,
          bool RMSNORM = false, bool FUSED_RMS_STATS = false,
          bool PACKED_ELEMENT_SCHEDULE = false>
__global__ __launch_bounds__(512) void convrot_quant_512x2_bf16_kernel(
    const void* __restrict__ x, const void* __restrict__ auxiliary, int in_dtype,
    int8_t* __restrict__ qout, float* __restrict__ scaleout,
    int M, int K, const void* __restrict__ norm_weight = nullptr,
    float rms_eps = 0.0f) {

    static_assert(!SPLIT_SWIGLU || ACT == kActSwiGLU);
    static_assert(!MODULATE || ACT == kActNone);
    static_assert(!MODULATE || !SPLIT_SWIGLU);
    static_assert(!RMSNORM || MODULATE);
    static_assert(RMSNORM == FUSED_RMS_STATS);
    static_assert(!PACKED_ELEMENT_SCHEDULE ||
                  ((ACT == kActSwiGLU && TILED_QOUT && SPLIT_SWIGLU &&
                    PACK_QUANT && !MODULATE && !RMSNORM) ||
                   (ACT == kActNone && !TILED_QOUT && !SPLIT_SWIGLU &&
                    MODULATE && PACK_QUANT) ||
                   (ACT == kActNone && !TILED_QOUT && !SPLIT_SWIGLU &&
                    !MODULATE && PACK_QUANT && !RMSNORM)));

    constexpr int kThreads = 512;
    constexpr int kGroup = 256;
    constexpr int kGroupsPerThread = 2;
    constexpr int kGroupsPerSlot = kThreads / kGroup;
    constexpr int kGroupsPerPass = kGroupsPerSlot * kGroupsPerThread;
    const float h4[4][4] = {
        {1, 1, 1, -1},
        {1, 1, -1, 1},
        {1, -1, 1, 1},
        {-1, 1, 1, 1},
    };

    __shared__ float stage[2][kThreads * kGroupsPerThread];
    __shared__ float wave_max[kThreads / 32];
    extern __shared__ unsigned char rowbuf_raw[];
    __bf16* rowbuf = reinterpret_cast<__bf16*>(rowbuf_raw);

    const int row = blockIdx.x;
    const int t = threadIdx.x;
    const int lane = t & 31;
    const int wave = t >> 5;
    const int local_group = t / kGroup;
    const int element = t % kGroup;
    const int group_count = K / kGroup;
    // Tile-major INT8 output is [M_tile, K_tile, row_in_tile, K_in_tile].
    // This removes the hot down GEMM's K-strided activation loads.
    const int64_t qout_row_base = TILED_QOUT
        ? static_cast<int64_t>(row >> 7) * K * 128 + (row & 127) * 128
        : static_cast<int64_t>(row) * K;
    constexpr int kInputWidth =
        ACT == kActSwiGLU && !SPLIT_SWIGLU ? 2 : 1;
    const int64_t input_row = static_cast<int64_t>(row) * K * kInputWidth;
    const float norm = rsqrtf(static_cast<float>(kGroup));

    float row_rstd = 0.0f;
    if constexpr (RMSNORM) {
        if constexpr (FUSED_RMS_STATS) {
            #pragma clang fp reassociate(off)
            #pragma clang fp contract(on)
            #pragma clang fp reciprocal(off)
            // Match PyTorch's vectorized K=3840 RMSNorm statistics exactly.
            // Only the first eight waves participate, reproducing its 32x8
            // workgroup and four-BF16 vector ownership.  The same loads also
            // seed the existing BF16 row buffer for the ConvRot pass.
            float sum_sq = 0.0f;
            if (t < 256) {
                const auto* input_vectors =
                    reinterpret_cast<const convrot_bf16x4*>(
                        static_cast<const __bf16*>(x) + input_row);
                auto* row_vectors =
                    reinterpret_cast<convrot_bf16x4*>(rowbuf);
                #pragma unroll
                for (int vector = t; vector < 3840 / 4; vector += 256) {
                    const convrot_bf16x4 values = input_vectors[vector];
                    row_vectors[vector] = values;
                    #pragma unroll
                    for (int element4 = 0; element4 < 4; ++element4) {
                        const float value =
                            static_cast<float>(values[element4]);
                        sum_sq += value * value;
                    }
                }
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    sum_sq += __shfl_down(sum_sq, offset, 32);
                }
            }

            #pragma unroll
            for (int offset = 4; offset > 0; offset >>= 1) {
                if (lane == 0 && wave < 8 && wave >= offset &&
                    wave < 2 * offset) {
                    wave_max[wave - offset] = sum_sq;
                }
                __syncthreads();
                if (lane == 0 && wave < offset) {
                    sum_sq += wave_max[wave];
                }
                __syncthreads();
            }
            if (t == 0) {
                const float mean_square =
                    ieee_div_f32(sum_sq, static_cast<float>(K));
                wave_max[0] = __ocml_rsqrt_f32(mean_square + rms_eps);
            }
            __syncthreads();
            row_rstd = wave_max[0];
        }
    }
    float local_max = 0.0f;
    if constexpr (PACKED_ELEMENT_SCHEDULE) {
        // Eight ConvRot-256 groups per pass: radix 0 is register-local, 1/2
        // stay in-wave, only radix 3 uses LDS. Same 8-KiB ping-pong stage
        // as the four-group schedule (one visibility + one reuse barrier).
        constexpr int kPackedElements = 4;
        constexpr int kPackedThreadsPerGroup = kGroup / kPackedElements;
        constexpr int kPackedGroupsPerPass = kThreads / kPackedThreadsPerGroup;
        float* packed_stage = &stage[0][0];
        const int packed_local_group = t / kPackedThreadsPerGroup;
        const int packed_thread = t % kPackedThreadsPerGroup;
        const int element_base = packed_thread * kPackedElements;

        for (int group_base = 0; group_base < group_count;
             group_base += kPackedGroupsPerPass) {
            const int group = group_base + packed_local_group;
            const bool active = group < group_count;
            float transformed[kPackedElements] = {0.0f, 0.0f, 0.0f, 0.0f};
            if (active) {
                const int64_t index = input_row + group * kGroup + element_base;
                float input[kPackedElements];
                if constexpr (SPLIT_SWIGLU) {
                    const convrot_bf16x4 gates =
                        *reinterpret_cast<const convrot_bf16x4*>(
                            static_cast<const __bf16*>(x) + index);
                    const convrot_bf16x4 ups =
                        *reinterpret_cast<const convrot_bf16x4*>(
                            static_cast<const __bf16*>(auxiliary) + index);
                    #pragma unroll
                    for (int element4 = 0; element4 < kPackedElements; ++element4) {
                        input[element4] = static_cast<float>(
                            swiglu_bf16_value(gates[element4], ups[element4]));
                    }
                } else if constexpr (MODULATE) {
                    const int column = group * kGroup + element_base;
                    const convrot_bf16x4 values =
                        *reinterpret_cast<const convrot_bf16x4*>(
                            (FUSED_RMS_STATS ? rowbuf : static_cast<const __bf16*>(x) + input_row)
                            + column);
                    const convrot_bf16x4 weights =
                        *reinterpret_cast<const convrot_bf16x4*>(
                            static_cast<const __bf16*>(norm_weight) + column);
                    const convrot_bf16x4 modulation =
                        *reinterpret_cast<const convrot_bf16x4*>(
                            static_cast<const __bf16*>(auxiliary) + column);
                    #pragma unroll
                    for (int element4 = 0; element4 < kPackedElements; ++element4) {
                        const float factor = round_bf16(
                            1.0f + static_cast<float>(modulation[element4]));
                        float normalized =
                            row_rstd * static_cast<float>(values[element4]);
                        asm volatile("" : "+v"(normalized));
                        normalized *= static_cast<float>(weights[element4]);
                        normalized = round_bf16(normalized);
                        input[element4] = round_bf16(normalized * factor);
                    }
                } else {
                    const convrot_bf16x4 values =
                        *reinterpret_cast<const convrot_bf16x4*>(
                            static_cast<const __bf16*>(x) + index);
                    #pragma unroll
                    for (int element4 = 0; element4 < kPackedElements; ++element4) {
                        input[element4] = static_cast<float>(values[element4]);
                    }
                }
                #pragma unroll
                for (int digit = 0; digit < 4; ++digit) {
                    transformed[digit] =
                        h4[digit][0] * input[0] + h4[digit][1] * input[1] +
                        h4[digit][2] * input[2] + h4[digit][3] * input[3];
                }
            }

            // Radix 1: four adjacent physical threads own the four inputs.
            const int digit1 = packed_thread & 3;
            const int base1 = lane - digit1;
            #pragma unroll
            for (int element4 = 0; element4 < kPackedElements; ++element4) {
                const float v0 = __shfl(transformed[element4], base1, 32);
                const float v1 = __shfl(transformed[element4], base1 + 1, 32);
                const float v2 = __shfl(transformed[element4], base1 + 2, 32);
                const float v3 = __shfl(transformed[element4], base1 + 3, 32);
                transformed[element4] =
                    h4[digit1][0] * v0 + h4[digit1][1] * v1 +
                    h4[digit1][2] * v2 + h4[digit1][3] * v3;
            }

            // Radix 2: the four source threads are still in one wave.
            const int digit2 = (packed_thread >> 2) & 3;
            const int base2 = lane - digit2 * 4;
            #pragma unroll
            for (int element4 = 0; element4 < kPackedElements; ++element4) {
                const float v0 = __shfl(transformed[element4], base2, 32);
                const float v1 = __shfl(transformed[element4], base2 + 4, 32);
                const float v2 = __shfl(transformed[element4], base2 + 8, 32);
                const float v3 = __shfl(transformed[element4], base2 + 12, 32);
                transformed[element4] =
                    h4[digit2][0] * v0 + h4[digit2][1] * v1 +
                    h4[digit2][2] * v2 + h4[digit2][3] * v3;
                packed_stage[packed_local_group * kGroup + element_base + element4] =
                    transformed[element4];
            }
            __syncthreads();

            // Radix 3 crosses the two waves assigned to this group.
            const int digit3 = (packed_thread >> 4) & 3;
            const int base3 = packed_local_group * kGroup + element_base - digit3 * 64;
            #pragma unroll
            for (int element4 = 0; element4 < kPackedElements; ++element4) {
                const float v0 = packed_stage[base3 + element4];
                const float v1 = packed_stage[base3 + 64 + element4];
                const float v2 = packed_stage[base3 + 128 + element4];
                const float v3 = packed_stage[base3 + 192 + element4];
                transformed[element4] =
                    h4[digit3][0] * v0 + h4[digit3][1] * v1 +
                    h4[digit3][2] * v2 + h4[digit3][3] * v3;
                if (active) {
                    const __bf16 stored =
                        static_cast<__bf16>(transformed[element4] * norm);
                    rowbuf[static_cast<int64_t>(group) * kGroup + element_base + element4] =
                        stored;
                    local_max = fmaxf(
                        local_max, fabsf(static_cast<float>(stored)));
                }
            }
            if (group_base + kPackedGroupsPerPass < group_count) {
                __syncthreads();
            }
        }
    } else {
    for (int group_base = 0; group_base < group_count;
         group_base += kGroupsPerPass) {
        float transformed[kGroupsPerThread];
        bool active[kGroupsPerThread];

        #pragma unroll
        for (int slot = 0; slot < kGroupsPerThread; ++slot) {
            const int group =
                group_base + local_group + slot * kGroupsPerSlot;
            active[slot] = group < group_count;
            if (!active[slot]) {
                transformed[slot] = 0.0f;
            } else if constexpr (SPLIT_SWIGLU) {
                const int64_t index =
                    input_row + group * kGroup + element;
                transformed[slot] = static_cast<float>(
                    swiglu_bf16_value(
                        static_cast<const __bf16*>(x)[index],
                        static_cast<const __bf16*>(auxiliary)[index]));
            } else if constexpr (MODULATE) {
                const int column = group * kGroup + element;
                const int64_t index = input_row + column;
                // Match `x * (1 + scale)` as two BF16 elementwise kernels:
                // round the add before the multiply, then round the product
                // before the first ConvRot butterfly consumes it.
                // `__bf16` casts carry excess precision under -ffast-math and
                // clang otherwise contracts both observable rounding points
                // into the final conversion.  Use the bitwise helper shared
                // with fused RoPE so the add and multiply each materialize
                // PyTorch's BF16 result without an intermediate tensor.
                const float factor = round_bf16(
                    1.0f + static_cast<float>(
                        static_cast<const __bf16*>(auxiliary)[column]));
                if constexpr (RMSNORM) {
                    // Match PyTorch's BF16 RMSNorm materialization exactly:
                    // gamma * (rstd * x), all in FP32, then one BF16 rounding.
                    // The empty vector-asm boundary prevents Kitchen's global
                    // -ffast-math from reassociating the two FP32 multiplies.
                    float normalized =
                        row_rstd * static_cast<float>(
                            FUSED_RMS_STATS
                                ? rowbuf[column]
                                : static_cast<const __bf16*>(x)[index]);
                    asm volatile("" : "+v"(normalized));
                    normalized *= static_cast<float>(
                        static_cast<const __bf16*>(norm_weight)[column]);
                    normalized = round_bf16(normalized);
                    transformed[slot] = round_bf16(normalized * factor);
                } else {
                    transformed[slot] = round_bf16(
                        static_cast<float>(
                            static_cast<const __bf16*>(x)[index]) * factor);
                }
            } else {
                transformed[slot] = load_input_act<ACT>(
                    x, input_row, group * kGroup + element, K, in_dtype);
            }
        }

        // Radix stages 0 and 1 never leave their aligned 16-lane subgroup.
        #pragma unroll
        for (int radix_stage = 0; radix_stage < 2; ++radix_stage) {
            const int stride = 1 << (2 * radix_stage);
            const int digit = (element / stride) & 3;
            const int base = (element & 15) - digit * stride;
            #pragma unroll
            for (int slot = 0; slot < kGroupsPerThread; ++slot) {
                const float v0 = __shfl(transformed[slot], base, 16);
                const float v1 = __shfl(transformed[slot], base + stride, 16);
                const float v2 = __shfl(transformed[slot], base + 2 * stride, 16);
                const float v3 = __shfl(transformed[slot], base + 3 * stride, 16);
                transformed[slot] =
                    h4[digit][0] * v0 + h4[digit][1] * v1 +
                    h4[digit][2] * v2 + h4[digit][3] * v3;
            }
        }

        #pragma unroll
        for (int slot = 0; slot < kGroupsPerThread; ++slot) {
            stage[0][t + slot * kThreads] = transformed[slot];
        }
        __syncthreads();

        // Radix stage 2 reads FP32 LDS and publishes stage 3's input.  Each
        // wave accesses one contiguous 32-float span per operand, so no two
        // lanes address the same 4-byte bank.
        {
            constexpr int stride = 16;
            const int digit = (element / stride) & 3;
            #pragma unroll
            for (int slot = 0; slot < kGroupsPerThread; ++slot) {
                const int group_offset =
                    (local_group + slot * kGroupsPerSlot) * kGroup;
                const int base = group_offset + element - digit * stride;
                const float v0 = stage[0][base];
                const float v1 = stage[0][base + stride];
                const float v2 = stage[0][base + 2 * stride];
                const float v3 = stage[0][base + 3 * stride];
                transformed[slot] =
                    h4[digit][0] * v0 + h4[digit][1] * v1 +
                    h4[digit][2] * v2 + h4[digit][3] * v3;
                stage[1][t + slot * kThreads] = transformed[slot];
            }
        }
        __syncthreads();

        // Final radix stage only reads stage[1].  The next pass starts by
        // writing stage[0], so no end-of-pass barrier is needed.
        {
            constexpr int stride = 64;
            const int digit = (element / stride) & 3;
            #pragma unroll
            for (int slot = 0; slot < kGroupsPerThread; ++slot) {
                const int group_offset =
                    (local_group + slot * kGroupsPerSlot) * kGroup;
                const int base = group_offset + element - digit * stride;
                const float v0 = stage[1][base];
                const float v1 = stage[1][base + stride];
                const float v2 = stage[1][base + 2 * stride];
                const float v3 = stage[1][base + 3 * stride];
                transformed[slot] =
                    h4[digit][0] * v0 + h4[digit][1] * v1 +
                    h4[digit][2] * v2 + h4[digit][3] * v3;
            }
        }

        #pragma unroll
        for (int slot = 0; slot < kGroupsPerThread; ++slot) {
            const int group =
                group_base + local_group + slot * kGroupsPerSlot;
            if (active[slot]) {
                const __bf16 stored =
                    static_cast<__bf16>(transformed[slot] * norm);
                rowbuf[static_cast<int64_t>(group) * kGroup + element] = stored;
                local_max = fmaxf(local_max, fabsf(static_cast<float>(stored)));
            }
        }
    }
    }

    // Register/wave reduction replaces the stock 256-float LDS reduction and
    // its eight workgroup barriers with two workgroup barriers total.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down(local_max, offset, 32));
    }
    if (lane == 0) wave_max[wave] = local_max;
    __syncthreads();
    if (wave == 0) {
        float value = lane < kThreads / 32 ? wave_max[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value = fmaxf(value, __shfl_down(value, offset, 32));
        }
        if (lane == 0) wave_max[0] = value;
    }
    __syncthreads();

    const float row_max = fmaxf(wave_max[0], 1.0e-10f);
    const float scale = row_max / 127.0f;
    const float inverse_scale = 127.0f / row_max;
    if (t == 0) scaleout[row] = scale;

    if constexpr (PACK_QUANT) {
        // One aligned dword LDS read covers two adjacent BF16
        // values, avoiding the two-lanes-per-bank mapping of scalar BF16 reads.
        // Even logical columns are also physically adjacent in the tiled128
        // layout, so one aligned 16-bit store preserves both output layouts.
        for (int pair = t; pair < K / 2; pair += kThreads) {
            const int column = 2 * pair;
            const convrot_bf16x2 values =
                *reinterpret_cast<const convrot_bf16x2*>(rowbuf + column);
            int q0 = static_cast<int>(
                rintf(static_cast<float>(values[0]) * inverse_scale));
            int q1 = static_cast<int>(
                rintf(static_cast<float>(values[1]) * inverse_scale));
            q0 = q0 < -127 ? -127 : (q0 > 127 ? 127 : q0);
            q1 = q1 < -127 ? -127 : (q1 > 127 ? 127 : q1);
            const int64_t qindex = TILED_QOUT
                ? qout_row_base + static_cast<int64_t>(column >> 7) * 16384 +
                      (column & 127)
                : qout_row_base + column;
            const uint16_t packed =
                static_cast<uint8_t>(static_cast<int8_t>(q0)) |
                (static_cast<uint16_t>(
                     static_cast<uint8_t>(static_cast<int8_t>(q1))) << 8);
            *reinterpret_cast<uint16_t*>(qout + qindex) = packed;
        }
    } else {
        for (int column = t; column < K; column += kThreads) {
            int quantized = static_cast<int>(
                rintf(static_cast<float>(rowbuf[column]) * inverse_scale));
            quantized = quantized < -127 ? -127 : (quantized > 127 ? 127 : quantized);
            const int64_t qindex = TILED_QOUT
                ? qout_row_base + static_cast<int64_t>(column >> 7) * 16384 +
                      (column & 127)
                : qout_row_base + column;
            qout[qindex] = static_cast<int8_t>(quantized);
        }
    }
}

inline bool convrot_wave32_512_supported() {
    int device = 0;
    hipDeviceProp_t properties{};
    return hipGetDevice(&device) == hipSuccess &&
        hipGetDeviceProperties(&properties, device) == hipSuccess &&
        properties.warpSize == 32 && properties.maxThreadsPerBlock >= 512;
}

inline bool convrot_512x2_supported(int K) {
    if (!convrot_wave32_512_supported()) {
        return false;
    }
    int device = 0;
    int lds_bytes = 0;
    if (hipGetDevice(&device) != hipSuccess ||
        hipDeviceGetAttribute(
            &lds_bytes, hipDeviceAttributeMaxSharedMemoryPerBlock, device) !=
            hipSuccess) {
        return false;
    }
    constexpr size_t kStaticBytes =
        (2 * 512 * 2 + 512 / 32) * sizeof(float);
    return static_cast<size_t>(K) * sizeof(__bf16) + kStaticBytes <=
        static_cast<size_t>(lds_bytes);
}

inline bool use_convrot_packed_quant() {
    return convrot_wave32_512_supported();
}

inline bool use_convrot_packed_elements() {
    return convrot_wave32_512_supported();
}

inline bool use_gfx12_convrot_packed_none() {
    static const bool selected = [] {
        int device = 0;
        hipDeviceProp_t properties{};
        if (hipGetDevice(&device) != hipSuccess ||
            hipGetDeviceProperties(&properties, device) != hipSuccess ||
            std::strncmp(properties.gcnArchName, "gfx12", 5) != 0) {
            return false;
        }
        return use_convrot_packed_elements();
    }();
    return selected;
}

template <int ACT>
inline void launch_convrot_quant_512x2_bf16(
    const void* x, int in_dtype, int8_t* qout, float* scaleout,
    int M, int K, hipStream_t stream) {
    // gfx12: eight-group packed schedule for plain activation ConvRot.
    if constexpr (ACT == kActNone) {
        if (use_gfx12_convrot_packed_none()) {
            convrot_quant_512x2_bf16_kernel<
                ACT, false, false, false, true, false, false, true>
                <<<M, 512, static_cast<size_t>(K) * sizeof(__bf16), stream>>>(
                    x, nullptr, in_dtype, qout, scaleout, M, K);
            return;
        }
    }
    if (use_convrot_packed_quant()) {
        convrot_quant_512x2_bf16_kernel<ACT, false, false, false, true>
            <<<M, 512, static_cast<size_t>(K) * sizeof(__bf16), stream>>>(
                x, nullptr, in_dtype, qout, scaleout, M, K);
    } else {
        convrot_quant_512x2_bf16_kernel<ACT>
            <<<M, 512, static_cast<size_t>(K) * sizeof(__bf16), stream>>>(
                x, nullptr, in_dtype, qout, scaleout, M, K);
    }
}

inline void launch_convrot_quant_512x2_bf16_swiglu_split_tiled128(
    const void* gate, const void* up, int8_t* qout, float* scaleout,
    int M, int K, hipStream_t stream) {
    if (use_convrot_packed_elements()) {
        convrot_quant_512x2_bf16_kernel<
            kActSwiGLU, true, true, false, true, false, false, true>
            <<<M, 512, static_cast<size_t>(K) * sizeof(__bf16), stream>>>(
                gate, up, 2, qout, scaleout, M, K);
    } else {
        convrot_quant_512x2_bf16_kernel<kActSwiGLU, true, true>
            <<<M, 512, static_cast<size_t>(K) * sizeof(__bf16), stream>>>(
                gate, up, 2, qout, scaleout, M, K);
    }
}

inline void launch_convrot_quant_512x2_bf16_modulated(
    const void* x, const void* modulation_scale,
    int8_t* qout, float* scaleout, int M, int K, hipStream_t stream) {
    if (use_convrot_packed_quant()) {
        convrot_quant_512x2_bf16_kernel<kActNone, false, false, true, true>
            <<<M, 512, static_cast<size_t>(K) * sizeof(__bf16), stream>>>(
                x, modulation_scale, 2, qout, scaleout, M, K);
    } else {
        convrot_quant_512x2_bf16_kernel<kActNone, false, false, true>
            <<<M, 512, static_cast<size_t>(K) * sizeof(__bf16), stream>>>(
                x, modulation_scale, 2, qout, scaleout, M, K);
    }
}

template <typename RowT, bool PACK_INT4, int ACT, int BLOCK_THREADS>
inline void launch_convrot_quant_impl(
    const void* x, int in_dtype, int8_t* qout, float* scaleout,
    int M, int K, int group_size, hipStream_t stream) {

    const size_t shmem = static_cast<size_t>(K) * convrot_row_element_size(in_dtype);
    convrot_quant_kernel<RowT, PACK_INT4, ACT, BLOCK_THREADS><<<M, BLOCK_THREADS, shmem, stream>>>(
        x, in_dtype, qout, scaleout, M, K, group_size);
}

template <bool PACK_INT4, int ACT, int BLOCK_THREADS>
struct LaunchConvrotQuantLegacyForBlock {
    const void* x;
    int in_dtype;
    int8_t* qout;
    float* scaleout;
    int M;
    int K;
    int group_size;
    hipStream_t stream;

    template <typename RowT>
    void operator()() const {
        launch_convrot_quant_impl<RowT, PACK_INT4, ACT, BLOCK_THREADS>(
            x, in_dtype, qout, scaleout, M, K, group_size, stream);
    }
};

template <bool PACK_INT4, int ACT, int BLOCK_THREADS>
inline void launch_convrot_quant_for_block(
    const void* x, int in_dtype, int8_t* qout, float* scaleout,
    int M, int K, int group_size, hipStream_t stream) {
    dispatch_convrot_row_type(
        in_dtype,
        LaunchConvrotQuantLegacyForBlock<PACK_INT4, ACT, BLOCK_THREADS>{
            x, in_dtype, qout, scaleout, M, K, group_size, stream});
}

template <bool PACK_INT4, int ACT = kActNone>
inline void launch_convrot_quant(
    const void* x, int in_dtype, int8_t* qout, float* scaleout,
    int M, int K, int group_size, hipStream_t stream,
    void* spill_rotated = nullptr, void* spill_partials = nullptr) {

    // INT8 G=256: fused when (M, K) fits in LDS, else caller-owned global spill.
    if (!PACK_INT4 && group_size == 256) {
        const int block_threads = convrot_pick_fused_block_threads(M, K, in_dtype);
        if (block_threads > 0) {
            bool launched = false;
            dispatch_convrot_row_type(
                in_dtype,
                LaunchConvrotQuantFusedForRow<ACT>{
                    x, in_dtype, qout, scaleout, M, K, stream, block_threads, &launched});
            if (launched) {
                return;
            }
        }
        if (spill_rotated == nullptr || spill_partials == nullptr) {
            throw std::runtime_error(
                "convrot global spill requires caller-provided workspace buffers");
        }
        launch_convrot_quant_global_managed<ACT>(
            x, in_dtype, qout, scaleout, M, K, stream, spill_rotated, spill_partials);
        return;
    }

    // INT4 G=256 with K>=512: 1024-thread blocks process 4 Hadamard groups/pass
    // (15->4 passes at K=3840). INT8 G=256 returns above. Smaller configs keep
    // 256 threads for occupancy on narrow K.
    const int block_threads = (group_size == 256 && K >= 512) ? 1024 : 256;
    if (block_threads == 1024) {
        launch_convrot_quant_for_block<PACK_INT4, ACT, 1024>(
            x, in_dtype, qout, scaleout, M, K, group_size, stream);
    } else {
        launch_convrot_quant_for_block<PACK_INT4, ACT, 256>(
            x, in_dtype, qout, scaleout, M, K, group_size, stream);
    }
}

}  // namespace comfy::hip_backend
