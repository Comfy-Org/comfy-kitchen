/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * int4_tensorwise (W4A4 + ConvRot) fused activation quantize.
 *
 * Produces the operands the SVDQuant W4A4 MMA kernel consumes, but with the
 * int4_tensorwise semantics: optional fused ConvRot rotation (group-256 regular
 * Hadamard, FHT — same butterfly as ops/int8_linear.cu) followed by PER-ROW
 * symmetric int4 quantization (scale = absmax/7, emission [-7, 7], packed two
 * codes per byte LOW nibble first). The per-row scale is broadcast into the
 * (K/64, M_pad) per-group ascales layout expected by scaled_mm_svdquant_w4a4 —
 * a rank-1 scale is an exact special case of the per-group layout, so the
 * existing s4 MMA GEMM computes the int4_tensorwise linear exactly.
 *
 *   in : x        (M, K)       bf16/fp16/fp32
 *   out: q_x      (M_pad, K/2) int8   two signed int4 per byte, low nibble first
 *        ascales  (K/64, M_pad) same dtype as x  (per-row scale, broadcast)
 *
 * Rows in [M, M_pad) are zero-filled (q = 0, scale = 0) so the GEMM's padding
 * rows are benign; callers slice the output back to M rows.
 */

#include "utils.cuh"
#include "dtype_dispatch.cuh"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace comfy {

namespace {

constexpr int kInt4ConvRotGroup = 256;
constexpr int kInt4ScaleGroup = 64;   // ascales group expected by the svdquant MMA
constexpr float kInt4Max = 7.0f;

template<typename T>
__device__ __forceinline__ float i4_to_float(T val);
template<> __device__ __forceinline__ float i4_to_float<float>(float val) { return val; }
template<> __device__ __forceinline__ float i4_to_float<half>(half val) { return __half2float(val); }
template<> __device__ __forceinline__ float i4_to_float<nv_bfloat16>(nv_bfloat16 val) { return __bfloat162float(val); }

template<typename T>
__device__ __forceinline__ T i4_from_float(float val);
template<> __device__ __forceinline__ float i4_from_float<float>(float val) { return val; }
template<> __device__ __forceinline__ half i4_from_float<half>(float val) { return __float2half_rn(val); }
template<> __device__ __forceinline__ nv_bfloat16 i4_from_float<nv_bfloat16>(float val) { return __float2bfloat16_rn(val); }

// Emulates the eager recipe's division at the source dtype: value and scale are
// both rounded through T before the divide (matches quantize_int4_rowwise).
template<typename T>
__device__ __forceinline__ float i4_quant_div(float val, float scale) {
    const float scale_t = i4_to_float(i4_from_float<T>(scale));
    return i4_to_float(i4_from_float<T>(i4_to_float(i4_from_float<T>(val)) / scale_t));
}

__device__ __forceinline__ float i4_warp_reduce_max(float v) {
    for (int offset = kThreadsPerWarp / 2; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
    }
    return v;
}

template<int NUM_WARPS>
__device__ __forceinline__ float i4_block_reduce_max(float v, float* warp_smem, float* block_smem) {
    const int lane = threadIdx.x & (kThreadsPerWarp - 1);
    const int wid = threadIdx.x >> 5;
    v = i4_warp_reduce_max(v);
    if (lane == 0) {
        warp_smem[wid] = v;
    }
    __syncthreads();
    if (wid == 0) {
        float total = lane < NUM_WARPS ? warp_smem[lane] : 0.0f;
        total = i4_warp_reduce_max(total);
        if (lane == 0) {
            *block_smem = total;
        }
    }
    __syncthreads();
    return *block_smem;
}

__device__ __forceinline__ float i4_h4_row_dot(int d, float x0, float x1, float x2, float x3) {
    switch (d) {
        case 0:  return  x0 + x1 + x2 - x3;
        case 1:  return  x0 + x1 - x2 + x3;
        case 2:  return  x0 - x1 + x2 + x3;
        default: return -x0 + x1 + x2 + x3;
    }
}

__device__ __forceinline__ int8_t i4_pack_pair(float c0, float c1) {
    const int i0 = static_cast<int>(c0);
    const int i1 = static_cast<int>(c1);
    return static_cast<int8_t>((i0 & 0xF) | ((i1 & 0xF) << 4));
}

// One block per output row (grid = M_pad). ROTATE=true runs the group-256 FHT
// in shared memory first (structure identical to int8_linear.cu's fused
// ConvRot kernel); ROTATE=false quantizes straight from global memory.
template<typename InputType, int BLOCK_THREADS, bool ROTATE>
__global__ void quantize_int4_tensorwise_kernel(
    const InputType* __restrict__ x,
    int8_t* __restrict__ q,
    InputType* __restrict__ ascales,
    int M,
    int M_pad,
    int K,
    int n_scale_groups)
{
    constexpr int kWarps = BLOCK_THREADS / kThreadsPerWarp;
    const int row = static_cast<int>(blockIdx.x);
    const int tid = threadIdx.x;
    const int k_half = K / 2;

    if (row >= M) {
        // Padding row: benign zeros (the GEMM output rows are discarded).
        for (int p = tid; p < k_half; p += BLOCK_THREADS) {
            q[static_cast<int64_t>(row) * k_half + p] = 0;
        }
        for (int g = tid; g < n_scale_groups; g += BLOCK_THREADS) {
            ascales[static_cast<int64_t>(g) * M_pad + row] = i4_from_float<InputType>(0.0f);
        }
        return;
    }

    extern __shared__ float smem[];
    float* row_buf = smem;      // ROTATE: K floats (rotated row, in place)
    float* tmp = smem + K;      // ROTATE: kGroupsInFlight * 2 * 256 floats
    __shared__ float warp_smem[kWarps];
    __shared__ float block_smem;

    const int64_t row_offset = static_cast<int64_t>(row) * K;

    if constexpr (ROTATE) {
        if constexpr (sizeof(InputType) == 2) {
            // 16-byte vectorized loads: 8 elements per iteration per thread.
            const uint4* x4 = reinterpret_cast<const uint4*>(x + row_offset);
            const int n_vec = K / 8;
            for (int v = tid; v < n_vec; v += BLOCK_THREADS) {
                const uint4 raw = x4[v];
                const InputType* e = reinterpret_cast<const InputType*>(&raw);
                const int base = v * 8;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    row_buf[base + j] = i4_to_float(e[j]);
                }
            }
        } else {
            for (int col = tid; col < K; col += BLOCK_THREADS) {
                row_buf[col] = i4_to_float(x[row_offset + col]);
            }
        }
        __syncthreads();

        constexpr int kGroupsInFlight = BLOCK_THREADS / kInt4ConvRotGroup;
        const int n_groups = K / kInt4ConvRotGroup;
        const int sub = tid / kInt4ConvRotGroup;
        const int i = tid % kInt4ConvRotGroup;
        float* buf0 = tmp + sub * (2 * kInt4ConvRotGroup);
        float* buf1 = buf0 + kInt4ConvRotGroup;
        const int iters = (n_groups + kGroupsInFlight - 1) / kGroupsInFlight;

        for (int it = 0; it < iters; ++it) {
            const int g = it * kGroupsInFlight + sub;
            const bool active = (g < n_groups);
            float* src = active ? (row_buf + g * kInt4ConvRotGroup) : buf0;
            float* dst = active ? buf0 : buf1;
            #pragma unroll
            for (int stage = 0; stage < 4; ++stage) {
                const int s = (stage == 0) ? 1 : (stage == 1) ? 4 : (stage == 2) ? 16 : 64;
                const int d = (i / s) & 3;
                const int base = i - d * s;
                const float v = 0.5f * i4_h4_row_dot(
                    d, src[base], src[base + s], src[base + 2 * s], src[base + 3 * s]);
                dst[i] = v;
                __syncthreads();
                float* t = src; src = dst; dst = t;
            }
        }

        // Match the eager reference bit-for-bit: eager rotates via one
        // fp32-accumulated matmul whose OUTPUT is rounded to the source dtype
        // once. The FHT above is the same fp32 linear map, so a single
        // round-through-InputType here reproduces eager's rotated values.
        for (int col = tid; col < K; col += BLOCK_THREADS) {
            row_buf[col] = i4_to_float(i4_from_float<InputType>(row_buf[col]));
        }
        __syncthreads();
    }

    // Per-row absmax over the (rotated) values -> scale = absmax / 7.
    float abs_max = 0.0f;
    if constexpr (ROTATE) {
        for (int col = tid; col < K; col += BLOCK_THREADS) {
            abs_max = fmaxf(abs_max, fabsf(row_buf[col]));
        }
    } else {
        for (int col = tid; col < K; col += BLOCK_THREADS) {
            abs_max = fmaxf(abs_max, fabsf(i4_to_float(x[row_offset + col])));
        }
    }
    abs_max = i4_block_reduce_max<kWarps>(abs_max, warp_smem, &block_smem);
    const float scale = fmaxf(abs_max * (1.0f / kInt4Max), 1.0e-30f);

    for (int g = tid; g < n_scale_groups; g += BLOCK_THREADS) {
        ascales[static_cast<int64_t>(g) * M_pad + row] = i4_from_float<InputType>(scale);
    }

    // Quantize + pack two codes per byte, LOW nibble first.
    if constexpr (ROTATE) {
        // Vectorized: each thread packs 8 rotated values into 4 bytes (uchar4).
        uchar4* q4 = reinterpret_cast<uchar4*>(q + static_cast<int64_t>(row) * k_half);
        const int n_vec = k_half / 4;
        for (int v = tid; v < n_vec; v += BLOCK_THREADS) {
            const int col = v * 8;
            uchar4 packed;
            unsigned char* pb = reinterpret_cast<unsigned char*>(&packed);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float a = i4_quant_div<InputType>(row_buf[col + 2 * j], scale);
                const float b = i4_quant_div<InputType>(row_buf[col + 2 * j + 1], scale);
                const float c0 = fminf(kInt4Max, fmaxf(-kInt4Max, nearbyintf(a)));
                const float c1 = fminf(kInt4Max, fmaxf(-kInt4Max, nearbyintf(b)));
                pb[j] = static_cast<unsigned char>(i4_pack_pair(c0, c1));
            }
            q4[v] = packed;
        }
    } else {
        for (int p = tid; p < k_half; p += BLOCK_THREADS) {
            const int col = 2 * p;
            const float v0 = i4_quant_div<InputType>(i4_to_float(x[row_offset + col]), scale);
            const float v1 = i4_quant_div<InputType>(i4_to_float(x[row_offset + col + 1]), scale);
            const float c0 = fminf(kInt4Max, fmaxf(-kInt4Max, nearbyintf(v0)));
            const float c1 = fminf(kInt4Max, fmaxf(-kInt4Max, nearbyintf(v1)));
            q[static_cast<int64_t>(row) * k_half + p] = i4_pack_pair(c0, c1);
        }
    }
}

}  // namespace

}  // namespace comfy

extern "C" {

void launch_int4_tensorwise_quantize_kernel(
    const void* x,
    void* q,
    void* ascales,
    int64_t M,
    int64_t M_pad,
    int64_t K,
    int64_t n_scale_groups,
    int input_dtype_code,
    bool rotate,
    cudaStream_t stream)
{
    if (M_pad == 0 || K == 0) {
        return;
    }
    if (K % (2 * comfy::kInt4ScaleGroup) != 0) {
        throw std::runtime_error("int4_tensorwise quantize requires K divisible by 128");
    }
    if (rotate && K % comfy::kInt4ConvRotGroup != 0) {
        throw std::runtime_error("int4_tensorwise ConvRot requires K divisible by 256");
    }
    if (K > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("int4_tensorwise quantize only supports K <= INT_MAX");
    }

    DISPATCH_FP_DTYPE(input_dtype_code, InputType, [&] {
        if (rotate) {
            constexpr int kBlockThreads = 1024;
            constexpr int kGroupsInFlight = kBlockThreads / comfy::kInt4ConvRotGroup;
            auto kernel = comfy::quantize_int4_tensorwise_kernel<InputType, kBlockThreads, true>;
            const size_t smem_bytes =
                (static_cast<size_t>(K) + kGroupsInFlight * 2 * comfy::kInt4ConvRotGroup) * sizeof(float);
            cudaError_t attr_err = cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(smem_bytes));
            if (attr_err != cudaSuccess) {
                throw std::runtime_error(
                    std::string("int4_tensorwise quantize shared memory request (") +
                    std::to_string(smem_bytes) + " bytes) failed: " +
                    cudaGetErrorString(attr_err));
            }
            kernel<<<static_cast<unsigned int>(M_pad), kBlockThreads, smem_bytes, stream>>>(
                static_cast<const InputType*>(x),
                static_cast<int8_t*>(q),
                static_cast<InputType*>(ascales),
                static_cast<int>(M),
                static_cast<int>(M_pad),
                static_cast<int>(K),
                static_cast<int>(n_scale_groups));
        } else {
            constexpr int kBlockThreads = 256;
            auto kernel = comfy::quantize_int4_tensorwise_kernel<InputType, kBlockThreads, false>;
            kernel<<<static_cast<unsigned int>(M_pad), kBlockThreads, 0, stream>>>(
                static_cast<const InputType*>(x),
                static_cast<int8_t*>(q),
                static_cast<InputType*>(ascales),
                static_cast<int>(M),
                static_cast<int>(M_pad),
                static_cast<int>(K),
                static_cast<int>(n_scale_groups));
        }
    });
}

}  // extern "C"
