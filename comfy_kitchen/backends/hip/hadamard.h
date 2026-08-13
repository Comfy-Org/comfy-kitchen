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

#include <stdexcept>
#include <string>
#include <type_traits>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace comfy::hip_backend {

// convrot_quant_kernel handles 256/G groups per pass and rotates in log4(G)
// stages, so a G outside this set either divides to a zero-width pass or is not
// a power of four. The dispatch wrappers fall back to eager before reaching here.
inline void check_convrot_group_size(int group_size) {
    if (group_size != 16 && group_size != 64 && group_size != 256) {
        throw std::runtime_error("convrot: group_size must be 16, 64 or 256");
    }
}

// The kernel's static LDS: butterfly workspace + row absmax reduction (up to 1024 threads).
constexpr size_t kConvrotStaticLds = 2 * 1024 * sizeof(float);
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

inline int convrot_max_k(int in_dtype) {
    const size_t element_size = convrot_row_element_size(in_dtype);
    if (element_size == 0) {
        return 0;
    }
    int device = 0;
    if (hipGetDevice(&device) != hipSuccess) {
        return 0;
    }
    int lds = 0;
    if (hipDeviceGetAttribute(&lds, hipDeviceAttributeMaxSharedMemoryPerBlock, device) !=
            hipSuccess ||
        lds <= static_cast<int>(kConvrotStaticLds)) {
        return 0;
    }
    return static_cast<int>((static_cast<size_t>(lds) - kConvrotStaticLds) /
                            element_size);
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
    return 1024;
}

inline bool convrot_fused_lds_fits(int K, int block_threads) {
    if (block_threads <= 0 || (block_threads % 64) != 0) {
        return false;
    }
    int device = 0;
    int lds = 0;
    if (hipGetDevice(&device) != hipSuccess ||
        hipDeviceGetAttribute(&lds, hipDeviceAttributeMaxSharedMemoryPerBlock, device) !=
            hipSuccess ||
        lds <= 0) {
        return false;
    }
    const int groups_in_flight = block_threads / 64;
    const size_t need =
        (static_cast<size_t>(block_threads / 32) + 1) * sizeof(float) +
        (static_cast<size_t>(K) + static_cast<size_t>(groups_in_flight) * 2 * kConvRotGroup256) *
            sizeof(float);
    return need <= static_cast<size_t>(lds);
}

// Heuristic block first, then narrower blocks before global spill. Returns 0 to spill.
inline int convrot_pick_fused_block_threads(int M, int K) {
    const int preferred = convrot_quant_fused_block_threads(M, K);
    if (convrot_fused_lds_fits(K, preferred)) {
        return preferred;
    }
    static constexpr int kFallbackBlocks[] = {768, 640, 512, 64};
    for (int block_threads : kFallbackBlocks) {
        if (block_threads != preferred && convrot_fused_lds_fits(K, block_threads)) {
            return block_threads;
        }
    }
    return 0;
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
    const int max_k = convrot_max_k(in_dtype);
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
    return fmaxf(fmaxf(fabsf(y0), fabsf(y1)), fmaxf(fabsf(y2), fabsf(y3)));
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

constexpr int kConvrotGlobalGroupsPerBlock = 8;
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
    const void* x, int in_dtype, int8_t* qout, float* scaleout, int M, int K, hipStream_t stream);

// Fused single-kernel path: FHT in LDS, vectorized loads, warp-shuffle absmax.
// One block per row; the rotated row stays in shared memory as float.
template <typename RowT, int BLOCK_THREADS, int ACT>
__global__ __launch_bounds__(BLOCK_THREADS) void convrot_quant_fused_kernel(
    const void* __restrict__ x, int in_dtype, int8_t* __restrict__ qout,
    float* __restrict__ scaleout, int M, int K) {

    constexpr int kGroupThreads = 64;
    constexpr int kGroupsInFlight = BLOCK_THREADS / kGroupThreads;
    constexpr int kWarps = BLOCK_THREADS / kWarpSize;

    extern __shared__ float smem[];
    float* row_buf = smem;
    float* tmp = smem + K;

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

        if (active) {
            abs_max = fmaxf(
                abs_max, convrot_fht_stage64_store_absmax<64>(buf1, row_buf + group_col, lane));
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
        const float v = row_buf[col];
        int q = static_cast<int>(rintf(v * inv));
        q = q < -127 ? -127 : (q > 127 ? 127 : q);
        qout[row_offset + col] = static_cast<int8_t>(q);
    }
}

template <typename RowT, int ACT, int BLOCK_THREADS>
inline void launch_convrot_quant_fused_impl(
    const void* x, int in_dtype, int8_t* qout, float* scaleout, int M, int K,
    hipStream_t stream) {
    const int groups_in_flight = BLOCK_THREADS / 64;
    const size_t shmem =
        (static_cast<size_t>(K) + static_cast<size_t>(groups_in_flight) * 2 * kConvRotGroup256) *
        sizeof(float);
    auto kernel = convrot_quant_fused_kernel<RowT, BLOCK_THREADS, ACT>;
    const hipError_t attr_err = hipFuncSetAttribute(
        reinterpret_cast<const void*>(kernel), hipFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shmem));
    if (attr_err != hipSuccess) {
        throw std::runtime_error(
            std::string("convrot fused: shared memory request (") + std::to_string(shmem) +
            " bytes) failed: " + hipGetErrorString(attr_err));
    }
    kernel<<<M, BLOCK_THREADS, shmem, stream>>>(x, in_dtype, qout, scaleout, M, K);
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

    template <int BLOCK_THREADS>
    void operator()() const {
        launch_convrot_quant_fused_impl<RowT, ACT, BLOCK_THREADS>(
            x, in_dtype, qout, scaleout, M, K, stream);
    }
};

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

    template <typename RowT>
    void operator()() const {
        dispatch_convrot_fused_block_threads(
            block_threads,
            LaunchConvrotQuantFusedForBlock<RowT, ACT>{x, in_dtype, qout, scaleout, M, K, stream});
    }
};

template <int ACT>
inline void launch_convrot_quant_fused(
    const void* x, int in_dtype, int8_t* qout, float* scaleout, int M, int K,
    hipStream_t stream) {
    const int block_threads = convrot_quant_fused_block_threads(M, K);
    dispatch_convrot_row_type(
        in_dtype,
        LaunchConvrotQuantFusedForRow<ACT>{x, in_dtype, qout, scaleout, M, K, stream, block_threads});
}

template <typename RowT, bool PACK_INT4, int ACT, int BLOCK_THREADS>
__global__ __launch_bounds__(BLOCK_THREADS) void convrot_quant_kernel(
    const void* __restrict__ x, int in_dtype,
    int8_t* __restrict__ qout, float* __restrict__ scaleout,
    int M, int K, int G) {

    const float h4[4][4] = {{1, 1, 1, -1}, {1, 1, -1, 1}, {1, -1, 1, 1}, {-1, 1, 1, 1}};
    __shared__ float g[1024];
    __shared__ float red[1024];
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
    int M, int K, int group_size, hipStream_t stream) {

    // INT8 G=256: fused when (M, K) fits in LDS, else global spill.
    if (!PACK_INT4 && group_size == 256) {
        const int block_threads = convrot_pick_fused_block_threads(M, K);
        if (block_threads > 0) {
            dispatch_convrot_row_type(
                in_dtype,
                LaunchConvrotQuantFusedForRow<ACT>{
                    x, in_dtype, qout, scaleout, M, K, stream, block_threads});
        } else {
            launch_convrot_quant_global_managed<ACT>(
                x, in_dtype, qout, scaleout, M, K, stream);
        }
        return;
    }

    // G=256 with K>=512: 1024-thread blocks process 4 Hadamard groups/pass (15->4
    // passes at K=3840). Smaller configs keep 256 threads for occupancy on narrow K.
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
