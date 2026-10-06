/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils.cuh"
#include "dtype_dispatch.cuh"
#include "input_act_codes.h"
#include "../prefetch_ring.h"
#include <cooperative_groups.h>

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace comfy {

namespace {

constexpr int kInt8Threads = 256;

// Prefetch ring consumer state for the M=1/2 GEMVs, per device (set by the ring
// on first use). Passed to the kernels as an argument: a __device__ global costs
// each block a dependent global load before its ring credit.
PrefetchRingState* g_int8_prefetch_ring[16] = {};

PrefetchRingState* int8_prefetch_ring_state() {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= 16) return nullptr;
    return g_int8_prefetch_ring[device];
}

template<typename T>
__device__ __forceinline__ float to_float(T val);
template<> __device__ __forceinline__ float to_float<float>(float val) { return val; }
template<> __device__ __forceinline__ float to_float<half>(half val) { return __half2float(val); }
template<> __device__ __forceinline__ float to_float<nv_bfloat16>(nv_bfloat16 val) { return __bfloat162float(val); }

template<typename T>
__device__ __forceinline__ float finite_max_for_dtype();
template<> __device__ __forceinline__ float finite_max_for_dtype<float>() { return FLT_MAX; }
template<> __device__ __forceinline__ float finite_max_for_dtype<half>() { return 65504.0f; }
template<> __device__ __forceinline__ float finite_max_for_dtype<nv_bfloat16>() { return 3.38953139e38f; }

template<typename T>
__device__ __forceinline__ float finite_absmax_for_int8_scale(float abs_max) {
    return fminf(abs_max, finite_max_for_dtype<T>());
}

template<typename T>
__device__ __forceinline__ T from_float(float val);
template<> __device__ __forceinline__ float from_float<float>(float val) { return val; }
template<> __device__ __forceinline__ half from_float<half>(float val) { return __float2half_rn(val); }
template<> __device__ __forceinline__ nv_bfloat16 from_float<nv_bfloat16>(float val) { return __float2bfloat16_rn(val); }

template<typename T>
__device__ __forceinline__ float quant_div_to_float(T val, float scale) {
    const float scale_t = to_float(from_float<T>(scale));
    return to_float(from_float<T>(to_float(val) / scale_t));
}

template<typename T>
__device__ __forceinline__ float quant_div_float_to_float(float val, float scale) {
    const float scale_t = to_float(from_float<T>(scale));
    return to_float(from_float<T>(to_float(from_float<T>(val)) / scale_t));
}

template<typename T>
__device__ __forceinline__ float stochastic_sum_to_float(float scaled, T rng) {
    return to_float(from_float<T>(scaled + to_float(rng)));
}

template<>
__device__ __forceinline__ float stochastic_sum_to_float<float>(float scaled, float rng) {
    return scaled + rng;
}

template<>
__device__ __forceinline__ float stochastic_sum_to_float<half>(float scaled, half rng) {
    return __half2float(__hadd(__float2half_rn(scaled), rng));
}

template<>
__device__ __forceinline__ float stochastic_sum_to_float<nv_bfloat16>(float scaled, nv_bfloat16 rng) {
    return __bfloat162float(__hadd(__float2bfloat16_rn(scaled), rng));
}

__device__ __forceinline__ uint32_t pcg_hash(uint32_t x) {
    const uint32_t state = x * 747796405u + 2891336453u;
    const uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

template<typename T>
__device__ __forceinline__ T stochastic_rng_value(int64_t idx, uint64_t seed) {
    const uint64_t key = static_cast<uint64_t>(idx) + seed;
    const uint32_t folded = static_cast<uint32_t>(key) ^ static_cast<uint32_t>(key >> 32);
    const float value = static_cast<float>(pcg_hash(folded) >> 8) * 0x1.0p-24f;
    return from_float<T>(value);
}

template<typename T>
__device__ __forceinline__ void store4_contiguous(T* out, int64_t idx, float x, float y, float z, float w) {
    out[idx] = from_float<T>(x);
    out[idx + 1] = from_float<T>(y);
    out[idx + 2] = from_float<T>(z);
    out[idx + 3] = from_float<T>(w);
}

template<>
__device__ __forceinline__ void store4_contiguous<float>(float* out, int64_t idx, float x, float y, float z, float w) {
    reinterpret_cast<float4*>(out)[idx / 4] = make_float4(x, y, z, w);
}

template<>
__device__ __forceinline__ void store4_contiguous<half>(half* out, int64_t idx, float x, float y, float z, float w) {
    reinterpret_cast<half2*>(out)[idx / 2] = __floats2half2_rn(x, y);
    reinterpret_cast<half2*>(out)[idx / 2 + 1] = __floats2half2_rn(z, w);
}

template<>
__device__ __forceinline__ void store4_contiguous<nv_bfloat16>(
    nv_bfloat16* out, int64_t idx, float x, float y, float z, float w)
{
    reinterpret_cast<nv_bfloat162*>(out)[idx / 2] = __floats2bfloat162_rn(x, y);
    reinterpret_cast<nv_bfloat162*>(out)[idx / 2 + 1] = __floats2bfloat162_rn(z, w);
}

__device__ __forceinline__ float warp_reduce_max(float v) {
    for (int offset = kThreadsPerWarp / 2; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
    }
    return v;
}

// Weights are read once per step: evict_first keeps the demand stream from
// displacing lines the prefetch ring already landed (see w4a8_gemm.cu).
// L2 cache hints are sm_80+; Turing takes plain read-only loads.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
__device__ __forceinline__ uint64_t l2_evict_first_policy() { return 0; }
__device__ __forceinline__ int ld_evict_first(const int* p, uint64_t) { return __ldg(p); }
__device__ __forceinline__ int4 ld_evict_first(const int4* p, uint64_t) { return __ldg(p); }
#else
__device__ __forceinline__ uint64_t l2_evict_first_policy() {
    uint64_t policy;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(policy));
    return policy;
}
__device__ __forceinline__ int ld_evict_first(const int* p, uint64_t policy) {
    int v;
    asm volatile("ld.global.nc.L2::cache_hint.b32 %0, [%1], %2;\n" : "=r"(v) : "l"(p), "l"(policy));
    return v;
}
__device__ __forceinline__ int4 ld_evict_first(const int4* p, uint64_t policy) {
    int4 v;
    asm volatile("ld.global.nc.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p), "l"(policy));
    return v;
}
#endif

__device__ __forceinline__ int warp_reduce_sum_i32(int v) {
    for (int offset = kThreadsPerWarp / 2; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__device__ __forceinline__ float warp_reduce_sum_f32(float v) {
    for (int offset = kThreadsPerWarp / 2; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

template<int NUM_WARPS>
__device__ __forceinline__ float block_reduce_sum_f32_t(float v, float* warp_smem, float* block_smem) {
    const int lane = threadIdx.x & (kThreadsPerWarp - 1);
    const int wid = threadIdx.x >> 5;
    v = warp_reduce_sum_f32(v);
    if (lane == 0) {
        warp_smem[wid] = v;
    }
    __syncthreads();
    if (wid == 0) {
        float total = lane < NUM_WARPS ? warp_smem[lane] : 0.0f;
        total = warp_reduce_sum_f32(total);
        if (lane == 0) {
            *block_smem = total;
        }
    }
    __syncthreads();
    return *block_smem;
}

template<int NUM_WARPS>
__device__ __forceinline__ float block_reduce_max_t(float v, float* warp_smem, float* block_smem);

template<int NUM_WARPS>
__device__ __forceinline__ int block_reduce_sum_i32_t(int v, int* warp_smem, int* block_smem) {
    const int lane = threadIdx.x & (kThreadsPerWarp - 1);
    const int wid = threadIdx.x >> 5;
    v = warp_reduce_sum_i32(v);
    if (lane == 0) {
        warp_smem[wid] = v;
    }
    __syncthreads();
    if (wid == 0) {
        int total = lane < NUM_WARPS ? warp_smem[lane] : 0;
        total = warp_reduce_sum_i32(total);
        if (lane == 0) {
            *block_smem = total;
        }
    }
    __syncthreads();
    return *block_smem;
}

template<typename InputType, int BLOCK_THREADS, bool STOCHASTIC>
__global__ void quantize_int8_rowwise_kernel(
    const InputType* __restrict__ x,
    int8_t* __restrict__ q,
    float* __restrict__ scales,
    int K,
    uint64_t seed)
{
    constexpr int kWarps = BLOCK_THREADS / kThreadsPerWarp;
    __shared__ float warp_smem[kWarps];
    __shared__ float block_smem;

    const int row = static_cast<int>(blockIdx.x);
    const int tid = threadIdx.x;
    const int64_t row_offset = static_cast<int64_t>(row) * K;

    float abs_max = 0.0f;
    for (int col = tid; col < K; col += blockDim.x) {
        abs_max = fmaxf(abs_max, fabsf(to_float(x[row_offset + col])));
    }

    abs_max = block_reduce_max_t<kWarps>(abs_max, warp_smem, &block_smem);
    const float scale = fmaxf(
        finite_absmax_for_int8_scale<InputType>(abs_max) * (1.0f / 127.0f),
        1.0e-30f);

    if (tid == 0) {
        scales[row] = scale;
    }

    for (int col = tid; col < K; col += blockDim.x) {
        const int64_t idx = row_offset + col;
        const float scaled = quant_div_to_float<InputType>(x[idx], scale);
        float quantized;
        if constexpr (STOCHASTIC) {
            const InputType noise = stochastic_rng_value<InputType>(idx, seed);
            quantized = floorf(stochastic_sum_to_float<InputType>(scaled, noise));
        } else {
            quantized = nearbyintf(scaled);
        }
        quantized = fminf(127.0f, fmaxf(-128.0f, quantized));
        q[idx] = static_cast<int8_t>(quantized);
    }
}

// Fused ConvRot (online Hadamard rotation) + row-wise INT8 quantization.
//
// The rotation group is fixed to 256 = 4^4 and uses the symmetric "regular"
// Hadamard block H4:
//     [  1  1  1 -1 ]
//     [  1  1 -1  1 ]
//     [  1 -1  1  1 ]
//     [ -1  1  1  1 ]
// The full normalized matrix is H256 = (1/16) * (H4 (x) H4 (x) H4 (x) H4).
// Because it is a Kronecker power, the matvec H256 @ x factors into 4 radix-4
// "butterfly" stages (strides 1, 4, 16, 64) with a 1/2 scale per stage, i.e. a
// Fast Hadamard Transform: O(256*4) work per group instead of O(256*256).
//
// The whole row is rotated in shared memory and quantized in place, so the
// rotated bf16 activation is never written to / read back from global memory
// (the unfused path's main cost).
//
// One block per row. The block holds BLOCK_THREADS / 256 groups "in flight" at
// once: for large K the row's shared-memory footprint forces a single block per
// SM, so we want a wide block (many warps) to hide latency rather than a narrow
// 256-thread block. Each thread owns local element `i = tid % 256` of group
// slot `sub = tid / 256`.
constexpr int kConvRotGroup = 256;

__device__ __forceinline__ float h4_row_dot(int d, float x0, float x1, float x2, float x3) {
    // Row d of H4 dotted with (x0, x1, x2, x3).
    switch (d) {
        case 0:  return  x0 + x1 + x2 - x3;
        case 1:  return  x0 + x1 - x2 + x3;
        case 2:  return  x0 - x1 + x2 + x3;
        default: return -x0 + x1 + x2 + x3;
    }
}

template<int NUM_WARPS>
__device__ __forceinline__ float block_reduce_max_t(float v, float* warp_smem, float* block_smem) {
    const int lane = threadIdx.x & (kThreadsPerWarp - 1);
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

template<typename InputType, int BLOCK_THREADS, bool STOCHASTIC>
__global__ void quantize_int8_rowwise_convrot_kernel(
    const InputType* __restrict__ x,
    int8_t* __restrict__ q,
    float* __restrict__ scales,
    int K,
    uint64_t seed)
{
    constexpr int kGroupsInFlight = BLOCK_THREADS / kConvRotGroup;
    constexpr int kWarps = BLOCK_THREADS / kThreadsPerWarp;

    extern __shared__ float smem[];
    float* row_buf = smem;                         // K floats: rotated row, in place
    float* tmp = smem + K;                          // kGroupsInFlight * 2 * 256 floats

    __shared__ float warp_smem[kWarps];
    __shared__ float block_smem;

    const int row = static_cast<int>(blockIdx.x);
    const int tid = threadIdx.x;
    const int64_t row_offset = static_cast<int64_t>(row) * K;

    // Load the row into shared memory as float.
    for (int col = tid; col < K; col += BLOCK_THREADS) {
        row_buf[col] = to_float(x[row_offset + col]);
    }
    __syncthreads();

    // Fast Hadamard transform, kGroupsInFlight groups at a time.
    const int n_groups = K / kConvRotGroup;
    const int sub = tid / kConvRotGroup;
    const int i = tid % kConvRotGroup;
    // Each slot gets a private double buffer so inactive lanes (when n_groups is
    // not a multiple of kGroupsInFlight) can keep hitting __syncthreads without
    // touching the live row data.
    float* buf0 = tmp + sub * (2 * kConvRotGroup);
    float* buf1 = buf0 + kConvRotGroup;
    const int iters = (n_groups + kGroupsInFlight - 1) / kGroupsInFlight;

    for (int it = 0; it < iters; ++it) {
        const int g = it * kGroupsInFlight + sub;
        const bool active = (g < n_groups);
        // active: ping-pong between the row region and buf0, ending in the row
        // region after 4 swaps. inactive: ping-pong privately in buf0/buf1.
        float* src = active ? (row_buf + g * kConvRotGroup) : buf0;
        float* dst = active ? buf0 : buf1;
        #pragma unroll
        for (int stage = 0; stage < 4; ++stage) {
            const int s = (stage == 0) ? 1 : (stage == 1) ? 4 : (stage == 2) ? 16 : 64;
            const int d = (i / s) & 3;
            const int base = i - d * s;
            const float v = 0.5f * h4_row_dot(
                d, src[base], src[base + s], src[base + 2 * s], src[base + 3 * s]);
            dst[i] = v;
            __syncthreads();
            float* t = src; src = dst; dst = t;
        }
    }

    // Row absmax over the rotated values -> per-row scale.
    float abs_max = 0.0f;
    for (int col = tid; col < K; col += BLOCK_THREADS) {
        abs_max = fmaxf(abs_max, fabsf(row_buf[col]));
    }
    abs_max = block_reduce_max_t<kWarps>(abs_max, warp_smem, &block_smem);
    const float scale = fmaxf(
        finite_absmax_for_int8_scale<InputType>(abs_max) * (1.0f / 127.0f),
        1.0e-30f);
    if (tid == 0) {
        scales[row] = scale;
    }

    for (int col = tid; col < K; col += BLOCK_THREADS) {
        const int64_t idx = row_offset + col;
        const float scaled = quant_div_float_to_float<InputType>(row_buf[col], scale);
        float quantized;
        if constexpr (STOCHASTIC) {
            const InputType noise = stochastic_rng_value<InputType>(idx, seed);
            quantized = floorf(stochastic_sum_to_float<InputType>(scaled, noise));
        } else {
            quantized = nearbyintf(scaled);
        }
        quantized = fminf(127.0f, fmaxf(-128.0f, quantized));
        q[idx] = static_cast<int8_t>(quantized);
    }
}

// Small-M decode variant: the 256-point FHT runs warp-local in registers via
// shfl_xor butterflies instead of the smem ping-pong, so the only barrier left
// is the absmax block reduce. Bit-exact with the smem kernel: h4_row_dot's four
// rows are all left-to-right sums with one operand negated (FADD of a negated
// value is bit-identical to FSUB), so the butterfly below negates the mirror
// operand branchlessly and keeps the exact add order; scale/round math is
// unchanged and only the (exact) max reduction order differs. Element map:
// e = d0 + 4*d1 + 16*d2 + 64*d3 with lane = d0 + 4*d1 + 16*(d2 & 1) and
// reg j = (d2 >> 1) + 2*d3.

__device__ __forceinline__ float h4_butterfly(int d, float m0, float m1, float m2, float m3) {
    // row d of H4 negates operand (3 - d); predicated negation avoids the
    // per-lane divergent switch while matching h4_row_dot bit-for-bit
    const int neg = 3 - d;
    const float y0 = neg == 0 ? -m0 : m0;
    const float y1 = neg == 1 ? -m1 : m1;
    const float y2 = neg == 2 ? -m2 : m2;
    const float y3 = neg == 3 ? -m3 : m3;
    return 0.5f * (((y0 + y1) + y2) + y3);
}
template<typename InputType, int BLOCK_THREADS, int MAX_GROUPS_PER_WARP>
__global__ void quantize_int8_rowwise_convrot_warp_kernel(
    const InputType* __restrict__ x,
    int8_t* __restrict__ q,
    float* __restrict__ scales,
    int K)
{
    constexpr int kWarps = BLOCK_THREADS / kThreadsPerWarp;
    __shared__ float warp_smem[kWarps];
    __shared__ float block_smem;

    const int row = static_cast<int>(blockIdx.x);
    const int lane = threadIdx.x & (kThreadsPerWarp - 1);
    const int wid = threadIdx.x >> 5;
    const int64_t row_offset = static_cast<int64_t>(row) * K;
    const int n_groups = K / kConvRotGroup;

    const int d0 = lane & 3;
    const int d1 = (lane >> 2) & 3;
    int e_of[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int d2 = ((lane >> 4) & 1) | ((j & 1) << 1);
        const int d3 = j >> 1;
        e_of[j] = d0 + 4 * d1 + 16 * d2 + 64 * d3;
    }

    float v[MAX_GROUPS_PER_WARP][8];
    int gs[MAX_GROUPS_PER_WARP];
    #pragma unroll
    for (int it = 0; it < MAX_GROUPS_PER_WARP; ++it) {
        const int g = it * kWarps + wid;
        gs[it] = g;
        if (g < n_groups) {
            const int64_t base = row_offset + static_cast<int64_t>(g) * kConvRotGroup;
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                v[it][j] = to_float(x[base + e_of[j]]);
        }
    }

    #pragma unroll
    for (int it = 0; it < MAX_GROUPS_PER_WARP; ++it) {
        if (gs[it] >= n_groups)
            continue;
        // stage 0 (digit d0, stride 1) and stage 1 (digit d1, stride 4):
        // partners live in other lanes at the same register index
        #pragma unroll
        for (int stage = 0; stage < 2; ++stage) {
            const int d = stage == 0 ? d0 : d1;
            const int sh = stage == 0 ? 1 : 4;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float s0 = v[it][j];
                const float a1 = __shfl_xor_sync(0xffffffffu, s0, sh);
                const float a2 = __shfl_xor_sync(0xffffffffu, s0, 2 * sh);
                const float a3 = __shfl_xor_sync(0xffffffffu, s0, 3 * sh);
                float m[4];
                m[d] = s0;
                m[d ^ 1] = a1;
                m[d ^ 2] = a2;
                m[d ^ 3] = a3;
                v[it][j] = h4_butterfly(d, m[0], m[1], m[2], m[3]);
            }
        }
        // stage 2 (digit d2, stride 16): partners split between lane bit 4 and
        // register bit 0, so gather via one shuffle of each register pair
        float nv[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int d2 = ((lane >> 4) & 1) | ((j & 1) << 1);
            const float s0 = v[it][j];
            const float p_lane = __shfl_xor_sync(0xffffffffu, s0, 16);
            const float s1 = v[it][j ^ 1];
            const float p_both = __shfl_xor_sync(0xffffffffu, s1, 16);
            float m[4];
            m[d2] = s0;
            m[d2 ^ 1] = p_lane;
            m[d2 ^ 2] = s1;
            m[d2 ^ 3] = p_both;
            nv[j] = h4_butterfly(d2, m[0], m[1], m[2], m[3]);
        }
        // stage 3 (digit d3, stride 64): purely register-local
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int d3 = j >> 1;
            const int j0 = j & 1;
            float m[4];
            #pragma unroll
            for (int k = 0; k < 4; ++k)
                m[k] = nv[j0 + 2 * k];
            v[it][j] = h4_butterfly(d3, m[0], m[1], m[2], m[3]);
        }
    }

    float abs_max = 0.0f;
    #pragma unroll
    for (int it = 0; it < MAX_GROUPS_PER_WARP; ++it) {
        if (gs[it] >= n_groups)
            continue;
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            abs_max = fmaxf(abs_max, fabsf(v[it][j]));
    }
    abs_max = block_reduce_max_t<kWarps>(abs_max, warp_smem, &block_smem);
    const float scale = fmaxf(
        finite_absmax_for_int8_scale<InputType>(abs_max) * (1.0f / 127.0f),
        1.0e-30f);
    if (threadIdx.x == 0) {
        scales[row] = scale;
    }

    #pragma unroll
    for (int it = 0; it < MAX_GROUPS_PER_WARP; ++it) {
        if (gs[it] >= n_groups)
            continue;
        const int64_t base = row_offset + static_cast<int64_t>(gs[it]) * kConvRotGroup;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float scaled = quant_div_float_to_float<InputType>(v[it][j], scale);
            float quantized = nearbyintf(scaled);
            quantized = fminf(127.0f, fmaxf(-128.0f, quantized));
            q[base + e_of[j]] = static_cast<int8_t>(quantized);
        }
    }
}

template<typename OutputType, typename BiasType>
__global__ void dequantize_int8_linear_kernel(
    const int32_t* __restrict__ input,
    const float* __restrict__ x_scales,
    const float* __restrict__ weight_scales,
    const BiasType* __restrict__ bias,
    OutputType* __restrict__ output,
    int64_t total,
    int N,
    int weight_scale_size,
    bool has_bias)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    const int col = static_cast<int>(idx % N);
    const int row = static_cast<int>(idx / N);
    const float weight_scale = weight_scales[weight_scale_size == 1 ? 0 : col];
    float value = static_cast<float>(input[idx]) * x_scales[row] * weight_scale;
    if (has_bias) {
        value += to_float(bias[col]);
    }
    output[idx] = from_float<OutputType>(value);
}

template<typename OutputType>
__global__ void dequantize_int8_linear_vec4_kernel(
    const int32_t* __restrict__ input,
    const float* __restrict__ x_scales,
    const float* __restrict__ weight_scales,
    OutputType* __restrict__ output,
    int64_t total_vec4,
    int N,
    int weight_scale_size)
{
    const int64_t idx4 = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx4 >= total_vec4) {
        return;
    }

    const int64_t idx = idx4 * 4;
    const int col = static_cast<int>(idx % N);
    const int row = static_cast<int>(idx / N);
    const float x_scale = x_scales[row];
    const int4 acc = reinterpret_cast<const int4*>(input)[idx4];

    if (weight_scale_size == 1) {
        const float scale = x_scale * weight_scales[0];
        store4_contiguous(
            output, idx,
            static_cast<float>(acc.x) * scale,
            static_cast<float>(acc.y) * scale,
            static_cast<float>(acc.z) * scale,
            static_cast<float>(acc.w) * scale);
        return;
    }

    const float4 ws = reinterpret_cast<const float4*>(weight_scales)[col / 4];
    store4_contiguous(
        output, idx,
        static_cast<float>(acc.x) * x_scale * ws.x,
        static_cast<float>(acc.y) * x_scale * ws.y,
        static_cast<float>(acc.z) * x_scale * ws.z,
        static_cast<float>(acc.w) * x_scale * ws.w);
}

template<int BLOCK_THREADS, typename OutputType, typename BiasType>
__global__ void int8_gemv_dequant_kernel(
    const int8_t* __restrict__ x,
    const int8_t* __restrict__ weight,
    const float* __restrict__ x_scales,
    const float* __restrict__ weight_scales,
    const BiasType* __restrict__ bias,
    OutputType* __restrict__ output,
    int N,
    int K,
    int weight_scale_size,
    bool has_bias)
{
    constexpr int kWarps = BLOCK_THREADS / kThreadsPerWarp;
    __shared__ int warp_smem[kWarps];
    __shared__ int block_smem;

    const int n = static_cast<int>(blockIdx.x);
    const int tid = threadIdx.x;
    const int8_t* __restrict__ w_row = weight + static_cast<int64_t>(n) * K;

    int acc = 0;
    const int K4 = K >> 2;
    const int* __restrict__ x4 = reinterpret_cast<const int*>(x);
    const int* __restrict__ w4 = reinterpret_cast<const int*>(w_row);
    for (int k4 = tid; k4 < K4; k4 += BLOCK_THREADS) {
        acc = __dp4a(x4[k4], w4[k4], acc);
    }
    for (int k = (K4 << 2) + tid; k < K; k += BLOCK_THREADS) {
        acc += static_cast<int>(x[k]) * static_cast<int>(w_row[k]);
    }

    acc = block_reduce_sum_i32_t<kWarps>(acc, warp_smem, &block_smem);
    if (tid == 0) {
        const float weight_scale = weight_scales[weight_scale_size == 1 ? 0 : n];
        float value = static_cast<float>(acc) * x_scales[0] * weight_scale;
        if (has_bias) {
            value += to_float(bias[n]);
        }
        output[n] = from_float<OutputType>(value);
    }
}

// One warp per output column. Vec16: 16-byte weight/activation loads (K % 16 == 0), the
// integer accumulation is order-independent so the result matches the 4-byte loop exactly.
// residual (nullable): out = residual + residual_scale * out, with the linear rounded to
// OutputType first, as the unfused addcmul sees it.
template<int WARPS_PER_BLOCK, typename OutputType, typename BiasType, bool Vec16>
__global__ void int8_gemv_dequant_warp_kernel(
    const int8_t* __restrict__ x,
    const int8_t* __restrict__ weight,
    const float* __restrict__ x_scales,
    const float* __restrict__ weight_scales,
    const BiasType* __restrict__ bias,
    OutputType* __restrict__ output,
    int N,
    int K,
    int weight_scale_size,
    bool has_bias,
    const OutputType* __restrict__ residual,
    const OutputType* __restrict__ residual_scale,
    PrefetchRingState* ring)
{
    const int lane = threadIdx.x & (kThreadsPerWarp - 1);
    const int warp = threadIdx.x >> 5;
    const int n0 = static_cast<int>(blockIdx.x) * WARPS_PER_BLOCK;
    const int n = n0 + warp;
    if (n < N) {
        const int8_t* __restrict__ w_row = weight + static_cast<int64_t>(n) * K;
        const uint64_t policy = l2_evict_first_policy();

        int acc = 0;
        if constexpr (Vec16) {
            const int4* __restrict__ x16 = reinterpret_cast<const int4*>(x);
            const int4* __restrict__ w16 = reinterpret_cast<const int4*>(w_row);
            const int K16 = K >> 4;
            for (int i = lane; i < K16; i += kThreadsPerWarp) {
                const int4 w = ld_evict_first(w16 + i, policy);
                const int4 xv = x16[i];
                acc = __dp4a(xv.x, w.x, acc);
                acc = __dp4a(xv.y, w.y, acc);
                acc = __dp4a(xv.z, w.z, acc);
                acc = __dp4a(xv.w, w.w, acc);
            }
        } else {
            const int* __restrict__ x4 = reinterpret_cast<const int*>(x);
            const int* __restrict__ w4 = reinterpret_cast<const int*>(w_row);
            for (int k4 = lane; k4 < (K >> 2); k4 += kThreadsPerWarp) {
                acc = __dp4a(x4[k4], ld_evict_first(w4 + k4, policy), acc);
            }
        }
        acc = warp_reduce_sum_i32(acc);

        if (lane == 0) {
            const float weight_scale = weight_scales[weight_scale_size == 1 ? 0 : n];
            float value = static_cast<float>(acc) * x_scales[0] * weight_scale;
            if (has_bias) {
                value += to_float(bias[n]);
            }
            if (residual != nullptr) {
                const float linear = to_float(from_float<OutputType>(value));
                value = to_float(residual[n])
                    + to_float(residual_scale[n]) * linear;
            }
            output[n] = from_float<OutputType>(value);
        }
    }
    // Ring consumption: one add per block for the rows its warps read (the
    // weight is recorded as one region; blocks walk it in address order).
    __syncthreads();
    if (threadIdx.x == 0) {
        prefetch_ring_consume_device(ring, static_cast<uint64_t>(min(WARPS_PER_BLOCK, N - n0)) * K);
    }
}

template<int WARPS, typename OutputType, typename BiasType>
__global__ void int8_gemv2_dequant_warp_kernel(
    const int8_t* __restrict__ x,
    const int8_t* __restrict__ w,
    const float* __restrict__ x_scales,
    const float* __restrict__ weight_scales,
    const BiasType* __restrict__ bias,
    OutputType* __restrict__ output,
    int n,
    int k,
    int scale_size,
    PrefetchRingState* ring)
{
    const int lane = threadIdx.x % 32;
    const int row0 = blockIdx.x * WARPS;
    const int row = row0 + threadIdx.x / 32;
    if (row < n) {
        const int4* xv0 = reinterpret_cast<const int4*>(x);
        const int4* xv1 = reinterpret_cast<const int4*>(x + k);
        const int4* wv = reinterpret_cast<const int4*>(w + static_cast<int64_t>(row) * k);
        const uint64_t policy = l2_evict_first_policy();
        int a0 = 0, a1 = 0;
        for (int i = lane; i < k / 16; i += 32) {
            const int4 weights = ld_evict_first(wv + i, policy), x0 = xv0[i], x1 = xv1[i];
            a0 = __dp4a(x0.x, weights.x, a0);
            a1 = __dp4a(x1.x, weights.x, a1);
            a0 = __dp4a(x0.y, weights.y, a0);
            a1 = __dp4a(x1.y, weights.y, a1);
            a0 = __dp4a(x0.z, weights.z, a0);
            a1 = __dp4a(x1.z, weights.z, a1);
            a0 = __dp4a(x0.w, weights.w, a0);
            a1 = __dp4a(x1.w, weights.w, a1);
        }
        a0 = warp_reduce_sum_i32(a0);
        a1 = warp_reduce_sum_i32(a1);
        if (lane == 0) {
            const float scale = weight_scales[scale_size == 1 ? 0 : row];
            const float b = bias ? to_float(bias[row]) : 0.0f;
            output[row] = from_float<OutputType>(float(a0) * x_scales[0] * scale + b);
            output[n + row] = from_float<OutputType>(float(a1) * x_scales[1] * scale + b);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        prefetch_ring_consume_device(ring, static_cast<uint64_t>(min(WARPS, n - row0)) * k);
    }
}

template<typename OutputType>
__global__ void dequantize_int8_simple_kernel(
    const int8_t* __restrict__ input,
    const float* __restrict__ scales,
    OutputType* __restrict__ output,
    int64_t total,
    int inner_dim,
    int scale_mode)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    int64_t scale_idx = 0;
    if (scale_mode == 1) {
        scale_idx = idx;
    } else if (scale_mode == 2) {
        scale_idx = idx / inner_dim;
    }
    output[idx] = from_float<OutputType>(static_cast<float>(input[idx]) * scales[scale_idx]);
}

template<typename OutputType>
__global__ void dequantize_int8_simple_vec4_kernel(
    const int8_t* __restrict__ input,
    const float* __restrict__ scales,
    OutputType* __restrict__ output,
    int64_t total_vec4,
    int inner_dim_vec4,
    int scale_mode)
{
    const int64_t idx4 = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx4 >= total_vec4) {
        return;
    }

    const char4 q4 = reinterpret_cast<const char4*>(input)[idx4];
    const float scale = scales[scale_mode == 0 ? 0 : idx4 / inner_dim_vec4];
    const int64_t idx = idx4 * 4;
    store4_contiguous(output, idx,
        static_cast<float>(q4.x) * scale,
        static_cast<float>(q4.y) * scale,
        static_cast<float>(q4.z) * scale,
        static_cast<float>(q4.w) * scale);
}

template<typename OutputType>
__global__ void dequantize_int8_rowwise_vec4_2d_kernel(
    const int8_t* __restrict__ input,
    const float* __restrict__ scales,
    OutputType* __restrict__ output,
    int rows,
    int inner_dim_vec4)
{
    const int row = static_cast<int>(blockIdx.x);
    const int col4 = static_cast<int>(blockIdx.y) * blockDim.x + threadIdx.x;
    if (row >= rows || col4 >= inner_dim_vec4) {
        return;
    }

    const int64_t idx4 = static_cast<int64_t>(row) * inner_dim_vec4 + col4;
    const char4 q4 = reinterpret_cast<const char4*>(input)[idx4];
    const float scale = scales[row];
    const int64_t idx = idx4 * 4;
    store4_contiguous(output, idx,
        static_cast<float>(q4.x) * scale,
        static_cast<float>(q4.y) * scale,
        static_cast<float>(q4.z) * scale,
        static_cast<float>(q4.w) * scale);
}

template<int BLOCK_THREADS, typename OutputType>
__global__ void dequantize_int8_convrot_kernel(
    const int8_t* __restrict__ q,
    const float* __restrict__ scales,
    OutputType* __restrict__ output,
    int K,
    int scale_size)
{
    constexpr int kGroupsInFlight = BLOCK_THREADS / kConvRotGroup;

    extern __shared__ float smem[];
    float* row_buf = smem;
    float* tmp = smem + K;

    const int row = static_cast<int>(blockIdx.x);
    const int tid = threadIdx.x;
    const int64_t row_offset = static_cast<int64_t>(row) * K;
    const float scale = scales[scale_size == 1 ? 0 : row];

    for (int col = tid; col < K; col += BLOCK_THREADS) {
        row_buf[col] = static_cast<float>(q[row_offset + col]) * scale;
    }
    __syncthreads();

    const int n_groups = K / kConvRotGroup;
    const int sub = tid / kConvRotGroup;
    const int i = tid % kConvRotGroup;
    float* buf0 = tmp + sub * (2 * kConvRotGroup);
    float* buf1 = buf0 + kConvRotGroup;
    const int iters = (n_groups + kGroupsInFlight - 1) / kGroupsInFlight;

    for (int it = 0; it < iters; ++it) {
        const int g = it * kGroupsInFlight + sub;
        const bool active = (g < n_groups);
        float* src = active ? (row_buf + g * kConvRotGroup) : buf0;
        float* dst = active ? buf0 : buf1;
        #pragma unroll
        for (int stage = 0; stage < 4; ++stage) {
            const int s = (stage == 0) ? 1 : (stage == 1) ? 4 : (stage == 2) ? 16 : 64;
            const int d = (i / s) & 3;
            const int base = i - d * s;
            const float v = 0.5f * h4_row_dot(
                d, src[base], src[base + s], src[base + 2 * s], src[base + 3 * s]);
            dst[i] = v;
            __syncthreads();
            float* t = src; src = dst; dst = t;
        }
    }

    for (int col = tid; col < K; col += BLOCK_THREADS) {
        output[row_offset + col] = from_float<OutputType>(row_buf[col]);
    }
}

template<int BLOCK_THREADS, typename OutputType>
__global__ void dequantize_int8_convrot_groups_kernel(
    const int8_t* __restrict__ q,
    const float* __restrict__ scales,
    OutputType* __restrict__ output,
    int K,
    int scale_size)
{
    constexpr int kGroupsPerBlock = BLOCK_THREADS / kConvRotGroup;
    extern __shared__ float smem[];

    const int group = static_cast<int>(blockIdx.x) * kGroupsPerBlock + threadIdx.x / kConvRotGroup;
    const int row = static_cast<int>(blockIdx.y);
    const int i = threadIdx.x % kConvRotGroup;
    const int sub = threadIdx.x / kConvRotGroup;
    const bool active = group < K / kConvRotGroup;
    const int64_t row_offset = static_cast<int64_t>(row) * K;
    const int col = group * kConvRotGroup + i;
    const float scale = scales[scale_size == 1 ? 0 : row];
    float* buf0 = smem + sub * (2 * kConvRotGroup);
    float* buf1 = buf0 + kConvRotGroup;

    buf0[i] = active ? static_cast<float>(q[row_offset + col]) * scale : 0.0f;
    __syncthreads();

    float* src = buf0;
    float* dst = buf1;
    #pragma unroll
    for (int stage = 0; stage < 4; ++stage) {
        const int s = (stage == 0) ? 1 : (stage == 1) ? 4 : (stage == 2) ? 16 : 64;
        const int d = (i / s) & 3;
        const int base = i - d * s;
        const float v = 0.5f * h4_row_dot(
            d, src[base], src[base + s], src[base + 2 * s], src[base + 3 * s]);
        dst[i] = v;
        __syncthreads();
        float* t = src; src = dst; dst = t;
    }

    if (active) {
        output[row_offset + col] = from_float<OutputType>(src[i]);
    }
}

template<int S>
__device__ __forceinline__ void convrot_fht_stage64(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int lane)
{
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

template<int S, typename OutputType>
__device__ __forceinline__ void convrot_fht_stage64_store(
    const float* __restrict__ src,
    OutputType* __restrict__ output,
    int lane)
{
    const int base = (lane % S) + (lane / S) * (4 * S);
    const float x0 = src[base];
    const float x1 = src[base + S];
    const float x2 = src[base + 2 * S];
    const float x3 = src[base + 3 * S];
    output[base] = from_float<OutputType>(0.5f * (x0 + x1 + x2 - x3));
    output[base + S] = from_float<OutputType>(0.5f * (x0 + x1 - x2 + x3));
    output[base + 2 * S] = from_float<OutputType>(0.5f * (x0 - x1 + x2 + x3));
    output[base + 3 * S] = from_float<OutputType>(0.5f * (-x0 + x1 + x2 + x3));
}

template<int S, typename OutputType>
__device__ __forceinline__ float convrot_fht_stage64_store_absmax(
    const float* __restrict__ src,
    OutputType* __restrict__ output,
    int lane)
{
    const int base = (lane % S) + (lane / S) * (4 * S);
    const float x0 = src[base];
    const float x1 = src[base + S];
    const float x2 = src[base + 2 * S];
    const float x3 = src[base + 3 * S];
    const float y0 = 0.5f * (x0 + x1 + x2 - x3);
    const float y1 = 0.5f * (x0 + x1 - x2 + x3);
    const float y2 = 0.5f * (x0 - x1 + x2 + x3);
    const float y3 = 0.5f * (-x0 + x1 + x2 + x3);
    output[base] = from_float<OutputType>(y0);
    output[base + S] = from_float<OutputType>(y1);
    output[base + 2 * S] = from_float<OutputType>(y2);
    output[base + 3 * S] = from_float<OutputType>(y3);
    return fmaxf(fmaxf(fabsf(y0), fabsf(y1)), fmaxf(fabsf(y2), fabsf(y3)));
}

template<int GROUPS_PER_BLOCK, typename OutputType>
__global__ void dequantize_int8_convrot_groups64_kernel(
    const int8_t* __restrict__ q,
    const float* __restrict__ scales,
    OutputType* __restrict__ output,
    int K,
    int scale_size)
{
    constexpr int kGroupThreads = 64;
    extern __shared__ float smem[];

    const int sub = threadIdx.x / kGroupThreads;
    const int lane = threadIdx.x % kGroupThreads;
    const int group = static_cast<int>(blockIdx.y) * GROUPS_PER_BLOCK + sub;
    const int64_t row = blockIdx.x;
    const bool active = group < K / kConvRotGroup;
    const int64_t row_offset = static_cast<int64_t>(row) * K;
    const int group_col = group * kConvRotGroup;
    const float scale = scales[scale_size == 1 ? 0 : row];

    float* buf0 = smem + sub * (2 * kConvRotGroup);
    float* buf1 = buf0 + kConvRotGroup;

    const int base = lane * 4;
    const int64_t q_offset = row_offset + group_col + base;
    const float x0 = active ? static_cast<float>(q[q_offset]) * scale : 0.0f;
    const float x1 = active ? static_cast<float>(q[q_offset + 1]) * scale : 0.0f;
    const float x2 = active ? static_cast<float>(q[q_offset + 2]) * scale : 0.0f;
    const float x3 = active ? static_cast<float>(q[q_offset + 3]) * scale : 0.0f;
    buf1[base] = 0.5f * (x0 + x1 + x2 - x3);
    buf1[base + 1] = 0.5f * (x0 + x1 - x2 + x3);
    buf1[base + 2] = 0.5f * (x0 - x1 + x2 + x3);
    buf1[base + 3] = 0.5f * (-x0 + x1 + x2 + x3);
    __syncthreads();

    convrot_fht_stage64<4>(buf1, buf0, lane);
    __syncthreads();
    convrot_fht_stage64<16>(buf0, buf1, lane);
    __syncthreads();

    if (active) {
        convrot_fht_stage64_store<64, OutputType>(buf1, output + row_offset + group_col, lane);
    }
}

// 64-wide ConvRot groups: 16 threads per group, four elements each, the same butterfly one stage
// shorter. Groups are flattened across rows so narrow rows (K = 128: two groups) still fill a block.
template<int GROUPS_PER_BLOCK, typename OutputType>
__global__ void dequantize_int8_convrot_g64_kernel(
    const int8_t* __restrict__ q,
    const float* __restrict__ scales,
    OutputType* __restrict__ output,
    int64_t num_groups,
    int K,
    int scale_size)
{
    constexpr int kGroup = 64;
    constexpr int kGroupThreads = kGroup / 4;
    __shared__ float smem[GROUPS_PER_BLOCK * 2 * kGroup];

    const int sub = threadIdx.x / kGroupThreads;
    const int lane = threadIdx.x % kGroupThreads;
    const int64_t group = static_cast<int64_t>(blockIdx.x) * GROUPS_PER_BLOCK + sub;
    const bool active = group < num_groups;
    const int groups_per_row = K / kGroup;
    const int64_t row = active ? group / groups_per_row : 0;
    const int64_t offset = row * K + (active ? group % groups_per_row : 0) * kGroup;
    const float scale = scales[scale_size == 1 ? 0 : row];

    float* buf0 = smem + sub * (2 * kGroup);
    float* buf1 = buf0 + kGroup;

    const int base = lane * 4;
    const float x0 = active ? static_cast<float>(q[offset + base]) * scale : 0.0f;
    const float x1 = active ? static_cast<float>(q[offset + base + 1]) * scale : 0.0f;
    const float x2 = active ? static_cast<float>(q[offset + base + 2]) * scale : 0.0f;
    const float x3 = active ? static_cast<float>(q[offset + base + 3]) * scale : 0.0f;
    buf1[base] = 0.5f * (x0 + x1 + x2 - x3);
    buf1[base + 1] = 0.5f * (x0 + x1 - x2 + x3);
    buf1[base + 2] = 0.5f * (x0 - x1 + x2 + x3);
    buf1[base + 3] = 0.5f * (-x0 + x1 + x2 + x3);
    __syncthreads();

    convrot_fht_stage64<4>(buf1, buf0, lane);
    __syncthreads();

    if (active) {
        convrot_fht_stage64_store<16, OutputType>(buf0, output + offset, lane);
    }
}

template<int GROUPS_PER_BLOCK, typename InputType, typename OutputType>
__global__ void rotate_int8_convrot_groups64_amax_kernel(
    const InputType* __restrict__ x,
    OutputType* __restrict__ output,
    float* __restrict__ partial_absmax,
    int K)
{
    constexpr int kGroupThreads = 64;
    extern __shared__ float smem[];

    const int sub = threadIdx.x / kGroupThreads;
    const int lane = threadIdx.x % kGroupThreads;
    const int group = static_cast<int>(blockIdx.y) * GROUPS_PER_BLOCK + sub;
    const int64_t row = blockIdx.x;
    const int n_groups = K / kConvRotGroup;
    const bool active = group < n_groups;
    const int64_t row_offset = static_cast<int64_t>(row) * K;
    const int group_col = group * kConvRotGroup;

    float* buf0 = smem + sub * (2 * kConvRotGroup);
    float* buf1 = buf0 + kConvRotGroup;

    const int base = lane * 4;
    const int64_t x_offset = row_offset + group_col + base;
    const float x0 = active ? to_float(x[x_offset]) : 0.0f;
    const float x1 = active ? to_float(x[x_offset + 1]) : 0.0f;
    const float x2 = active ? to_float(x[x_offset + 2]) : 0.0f;
    const float x3 = active ? to_float(x[x_offset + 3]) : 0.0f;
    buf1[base] = 0.5f * (x0 + x1 + x2 - x3);
    buf1[base + 1] = 0.5f * (x0 + x1 - x2 + x3);
    buf1[base + 2] = 0.5f * (x0 - x1 + x2 + x3);
    buf1[base + 3] = 0.5f * (-x0 + x1 + x2 + x3);
    __syncthreads();

    convrot_fht_stage64<4>(buf1, buf0, lane);
    __syncthreads();
    convrot_fht_stage64<16>(buf0, buf1, lane);
    __syncthreads();

    float local_max = 0.0f;
    if (active) {
        local_max = convrot_fht_stage64_store_absmax<64, OutputType>(
            buf1, output + row_offset + group_col, lane);
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

template<int GROUPS_PER_BLOCK, typename InputType, typename OutputType>
__global__ void rotate_int8_convrot_groups64_kernel(
    const InputType* __restrict__ x,
    OutputType* __restrict__ output,
    int K)
{
    constexpr int kGroupThreads = 64;
    extern __shared__ float smem[];

    const int sub = threadIdx.x / kGroupThreads;
    const int lane = threadIdx.x % kGroupThreads;
    const int group = static_cast<int>(blockIdx.y) * GROUPS_PER_BLOCK + sub;
    const int64_t row = blockIdx.x;
    const bool active = group < K / kConvRotGroup;
    const int64_t row_offset = static_cast<int64_t>(row) * K;
    const int group_col = group * kConvRotGroup;

    float* buf0 = smem + sub * (2 * kConvRotGroup);
    float* buf1 = buf0 + kConvRotGroup;

    const int base = lane * 4;
    const int64_t x_offset = row_offset + group_col + base;
    const float x0 = active ? to_float(x[x_offset]) : 0.0f;
    const float x1 = active ? to_float(x[x_offset + 1]) : 0.0f;
    const float x2 = active ? to_float(x[x_offset + 2]) : 0.0f;
    const float x3 = active ? to_float(x[x_offset + 3]) : 0.0f;
    buf1[base] = 0.5f * (x0 + x1 + x2 - x3);
    buf1[base + 1] = 0.5f * (x0 + x1 - x2 + x3);
    buf1[base + 2] = 0.5f * (x0 - x1 + x2 + x3);
    buf1[base + 3] = 0.5f * (-x0 + x1 + x2 + x3);
    __syncthreads();

    convrot_fht_stage64<4>(buf1, buf0, lane);
    __syncthreads();
    convrot_fht_stage64<16>(buf0, buf1, lane);
    __syncthreads();

    if (active) {
        convrot_fht_stage64_store<64, OutputType>(buf1, output + row_offset + group_col, lane);
    }
}

template<typename InputType, int BLOCK_THREADS, bool STOCHASTIC>
__global__ void quantize_int8_rowwise_from_partials_kernel(
    const InputType* __restrict__ x,
    const float* __restrict__ partial_absmax,
    int8_t* __restrict__ q,
    float* __restrict__ scales,
    int K,
    uint64_t seed)
{
    constexpr int kWarps = BLOCK_THREADS / kThreadsPerWarp;
    __shared__ float warp_smem[kWarps];
    __shared__ float block_smem;

    const int row = static_cast<int>(blockIdx.x);
    const int tid = threadIdx.x;
    const int n_groups = K / kConvRotGroup;
    const int64_t row_offset = static_cast<int64_t>(row) * K;
    const float* row_partials = partial_absmax + static_cast<int64_t>(row) * n_groups;

    float abs_max = 0.0f;
    for (int g = tid; g < n_groups; g += BLOCK_THREADS) {
        abs_max = fmaxf(abs_max, row_partials[g]);
    }
    abs_max = block_reduce_max_t<kWarps>(abs_max, warp_smem, &block_smem);
    const float scale = fmaxf(
        finite_absmax_for_int8_scale<InputType>(abs_max) * (1.0f / 127.0f),
        1.0e-30f);
    if (tid == 0) {
        scales[row] = scale;
    }

    for (int col = tid; col < K; col += BLOCK_THREADS) {
        const int64_t idx = row_offset + col;
        const float scaled = quant_div_to_float<InputType>(x[idx], scale);
        float quantized;
        if constexpr (STOCHASTIC) {
            const InputType noise = stochastic_rng_value<InputType>(idx, seed);
            quantized = floorf(stochastic_sum_to_float<InputType>(scaled, noise));
        } else {
            quantized = nearbyintf(scaled);
        }
        quantized = fminf(127.0f, fmaxf(-128.0f, quantized));
        q[idx] = static_cast<int8_t>(quantized);
    }
}

// Optional activation applied on the way into the quantizer, so an MLP's
// `linear(act(proj(x)))` never writes act's output to HBM just to read it
// straight back. No reduction is involved, so it is nearly free here. The
// codes live in input_act_codes.h, shared with the nanobind layer.
template<int ACT>
__device__ __forceinline__ float apply_input_act(float v) {
    if constexpr (ACT == kActGeluTanh) {
        // Matches torch.nn.functional.gelu(x, approximate="tanh").
        constexpr float kBeta = 0.7978845608028654f;   // sqrt(2/pi)
        constexpr float kKappa = 0.044715f;
        const float inner = kBeta * (v + kKappa * v * v * v);
        return 0.5f * v * (1.0f + tanhf(inner));
    }
    return v;
}

// Reads one activated value: column `col` of the K-wide activated row starting
// at `in_row`. For SwiGLU the raw row is 2*K wide with the gate in the first
// half; every other activation reads the same K-wide row it writes.
template<int ACT, typename InputType>
__device__ __forceinline__ float load_input_act(
    const InputType* __restrict__ x, int64_t in_row, int col, int K)
{
    if constexpr (ACT == kActSwiGLU) {
        // Matches torch silu(gate) * up.
        const float gate = to_float(x[in_row + col]);
        const float up = to_float(x[in_row + K + col]);
        return (gate / (1.0f + expf(-gate))) * up;
    } else {
        return apply_input_act<ACT>(to_float(x[in_row + col]));
    }
}

// One row of the ConvRot quantizer, run by a whole BLOCK_THREADS block: optional input
// activation (SwiGLU reads a [gate | up] raw row twice as wide as the K it writes;
// RmsNorm stages the raw row first for its mean of squares), 256-point FHT per group,
// row absmax -> scale, int8 store into `q_row` (global or shared). Returns the scale.
// Shared by the standalone quantizer and the GEMV prologue, so both round identically.
template<typename InputType, int BLOCK_THREADS, bool STOCHASTIC, int ACT>
__device__ __forceinline__ float convrot64_quantize_row(
    const InputType* __restrict__ x_row,
    int8_t* q_row,
    int K,
    int64_t rng_base,
    uint64_t seed,
    const InputType* __restrict__ act_weight,
    float act_eps,
    float* row_buf,
    float* tmp,
    float* warp_smem,
    float* block_smem)
{
    constexpr int kGroupThreads = 64;
    constexpr int kGroupsInFlight = BLOCK_THREADS / kGroupThreads;
    constexpr int kWarps = BLOCK_THREADS / kThreadsPerWarp;

    const int tid = threadIdx.x;
    const int sub = tid / kGroupThreads;
    const int lane = tid % kGroupThreads;
    const int n_groups = K / kConvRotGroup;

    float* buf0 = tmp + sub * (2 * kConvRotGroup);
    float* buf1 = buf0 + kConvRotGroup;
    float abs_max = 0.0f;

    // RmsNorm needs the row's mean of squares first, so the raw row is staged in
    // row_buf; the FHT loop reads its 256-wide slice and overwrites it two stages
    // later, with __syncthreads between, so nothing is clobbered early.
    float rstd = 1.0f;
    if constexpr (ACT == kActRmsNorm) {
        float sum_sq = 0.0f;
        for (int col = tid; col < K; col += BLOCK_THREADS) {
            const float v = to_float(x_row[col]);
            row_buf[col] = v;
            sum_sq += v * v;
        }
        sum_sq = block_reduce_sum_f32_t<kWarps>(sum_sq, warp_smem, block_smem);
        rstd = rsqrtf(sum_sq / static_cast<float>(K) + act_eps);
    }

    const int iters = (n_groups + kGroupsInFlight - 1) / kGroupsInFlight;
    for (int it = 0; it < iters; ++it) {
        const int group = it * kGroupsInFlight + sub;
        const bool active = group < n_groups;
        const int base = lane * 4;
        const int group_col = group * kConvRotGroup;
        const int col = group_col + base;

        float x0, x1, x2, x3;
        if constexpr (ACT == kActRmsNorm) {
            x0 = active ? row_buf[col] * rstd * to_float(act_weight[col]) : 0.0f;
            x1 = active ? row_buf[col + 1] * rstd * to_float(act_weight[col + 1]) : 0.0f;
            x2 = active ? row_buf[col + 2] * rstd * to_float(act_weight[col + 2]) : 0.0f;
            x3 = active ? row_buf[col + 3] * rstd * to_float(act_weight[col + 3]) : 0.0f;
        } else {
            x0 = active ? load_input_act<ACT>(x_row, 0, col, K) : 0.0f;
            x1 = active ? load_input_act<ACT>(x_row, 0, col + 1, K) : 0.0f;
            x2 = active ? load_input_act<ACT>(x_row, 0, col + 2, K) : 0.0f;
            x3 = active ? load_input_act<ACT>(x_row, 0, col + 3, K) : 0.0f;
        }
        buf1[base] = 0.5f * (x0 + x1 + x2 - x3);
        buf1[base + 1] = 0.5f * (x0 + x1 - x2 + x3);
        buf1[base + 2] = 0.5f * (x0 - x1 + x2 + x3);
        buf1[base + 3] = 0.5f * (-x0 + x1 + x2 + x3);
        __syncthreads();

        convrot_fht_stage64<4>(buf1, buf0, lane);
        __syncthreads();
        convrot_fht_stage64<16>(buf0, buf1, lane);
        __syncthreads();

        if (active) {
            abs_max = fmaxf(
                abs_max,
                convrot_fht_stage64_store_absmax<64, float>(buf1, row_buf + group_col, lane));
        }
        __syncthreads();
    }

    abs_max = block_reduce_max_t<kWarps>(abs_max, warp_smem, block_smem);
    const float scale = fmaxf(
        finite_absmax_for_int8_scale<InputType>(abs_max) * (1.0f / 127.0f),
        1.0e-30f);

    for (int col = tid; col < K; col += BLOCK_THREADS) {
        const float scaled = quant_div_float_to_float<InputType>(row_buf[col], scale);
        float quantized;
        if constexpr (STOCHASTIC) {
            const InputType noise = stochastic_rng_value<InputType>(rng_base + col, seed);
            quantized = floorf(stochastic_sum_to_float<InputType>(scaled, noise));
        } else {
            quantized = nearbyintf(scaled);
        }
        quantized = fminf(127.0f, fmaxf(-128.0f, quantized));
        q_row[col] = static_cast<int8_t>(quantized);
    }
    return scale;
}

template<typename InputType, int BLOCK_THREADS, bool STOCHASTIC, int ACT = kActNone>
__global__ void quantize_int8_rowwise_convrot64_kernel(
    const InputType* __restrict__ x,
    int8_t* __restrict__ q,
    float* __restrict__ scales,
    int K,
    uint64_t seed,
    const InputType* __restrict__ act_weight,
    float act_eps)
{
    constexpr int kWarps = BLOCK_THREADS / kThreadsPerWarp;
    extern __shared__ float smem[];
    __shared__ float warp_smem[kWarps];
    __shared__ float block_smem;

    const int row = static_cast<int>(blockIdx.x);
    const int64_t row_offset = static_cast<int64_t>(row) * K;
    constexpr int kInWidth = (ACT == kActSwiGLU) ? 2 : 1;
    const float scale = convrot64_quantize_row<InputType, BLOCK_THREADS, STOCHASTIC, ACT>(
        x + row_offset * kInWidth, q + row_offset, K, row_offset, seed, act_weight, act_eps,
        smem, smem + K, warp_smem, &block_smem);
    if (threadIdx.x == 0) {
        scales[row] = scale;
    }
}

// M=1 ConvRot GEMV with the activation quantizer as its prologue: every block
// re-quantizes the (L2-resident) input row into shared memory with the same
// 512-thread convrot64_quantize_row the standalone quantizer runs for one row, so
// the int8 operands and scale are identical and the separate launch disappears.
// The prologue is compute the blocks on one SM would repeat, so the launcher runs
// few blocks per SM and each walks column groups grid-stride, one warp per column
// over all of K exactly as int8_gemv_dequant_warp_kernel does. Epilogue as there.
constexpr int kFusedGemvThreads = 512;
constexpr int kFusedGemvCols = kFusedGemvThreads / kThreadsPerWarp;

template<typename InputType, typename OutputType, typename BiasType, int ACT>
__global__ void __launch_bounds__(kFusedGemvThreads) int8_gemv_convrot_fused_kernel(
    const InputType* __restrict__ x,
    const InputType* __restrict__ act_weight,
    float act_eps,
    const int8_t* __restrict__ weight,
    const float* __restrict__ weight_scales,
    const BiasType* __restrict__ bias,
    OutputType* __restrict__ output,
    int N,
    int K,
    int weight_scale_size,
    bool has_bias,
    const OutputType* __restrict__ residual,
    const OutputType* __restrict__ residual_scale,
    PrefetchRingState* ring)
{
    constexpr int kWarps = kFusedGemvThreads / kThreadsPerWarp;
    extern __shared__ float smem[];
    __shared__ float warp_smem[kWarps];
    __shared__ float block_smem;

    float* row_buf = smem;
    float* tmp = smem + K;
    int8_t* q_smem = reinterpret_cast<int8_t*>(tmp + (kFusedGemvThreads / 64) * 2 * kConvRotGroup);

    const int lane = threadIdx.x & (kThreadsPerWarp - 1);
    const int warp = threadIdx.x >> 5;
    const int K16 = K >> 4;
    const uint64_t policy = l2_evict_first_policy();

    const float x_scale = convrot64_quantize_row<InputType, kFusedGemvThreads, false, ACT>(
        x, q_smem, K, 0, 0, act_weight, act_eps, row_buf, tmp, warp_smem, &block_smem);
    __syncthreads();

    const int4* __restrict__ x16 = reinterpret_cast<const int4*>(q_smem);

    for (int n0 = static_cast<int>(blockIdx.x) * kFusedGemvCols; n0 < N; n0 += gridDim.x * kFusedGemvCols) {
        const int n = n0 + warp;
        if (n < N) {
            const int4* __restrict__ w16 = reinterpret_cast<const int4*>(weight + static_cast<int64_t>(n) * K);
            int acc = 0;
            #pragma unroll 4
            for (int i = lane; i < K16; i += kThreadsPerWarp) {
                const int4 w = ld_evict_first(w16 + i, policy);
                const int4 xv = x16[i];
                acc = __dp4a(xv.x, w.x, acc);
                acc = __dp4a(xv.y, w.y, acc);
                acc = __dp4a(xv.z, w.z, acc);
                acc = __dp4a(xv.w, w.w, acc);
            }
            acc = warp_reduce_sum_i32(acc);
            if (lane == 0) {
                const float weight_scale = weight_scales[weight_scale_size == 1 ? 0 : n];
                float value = static_cast<float>(acc) * x_scale * weight_scale;
                if (has_bias) {
                    value += to_float(bias[n]);
                }
                if (residual != nullptr) {
                    const float linear = to_float(from_float<OutputType>(value));
                    value = to_float(residual[n]) + to_float(residual_scale[n]) * linear;
                }
                output[n] = from_float<OutputType>(value);
            }
        }
        // one ring add per column group once every warp has read its row
        __syncthreads();
        if (threadIdx.x == 0) {
            prefetch_ring_consume_device(ring, static_cast<uint64_t>(min(kFusedGemvCols, N - n0)) * K);
        }
    }
}

// Decode-shape (M <= 16) ConvRot quantizer in one launch: the row is split over a
// cluster of CL blocks (8 groups of 256 per block per pass), each block rotates its
// groups into registers, the row absmax is exchanged over DSMEM, and the same
// registers are quantized. Numerically identical to the staged rotate+quantize pair
// (fp32 absmax of the unrounded rotation, bf16-rounded rotated value, same divide and
// rounding). Optional input activations are folded in and match the eager torch ops
// bit for bit: RmsNorm reproduces torch's vectorized rms_norm reduction (128 threads,
// vec4, warp shfl_down tree, 4-warp merge) and rounds gamma*(rstd*x) to the input
// dtype; SwiGLU rounds silu(gate) and silu(gate)*up to the input dtype. Requires
// clusters (sm_90+).
constexpr int kClusterQuantThreads = 512;
constexpr int kClusterQuantGroups = kClusterQuantThreads / 64;
// the attribute itself is rejected by the pre-sm_90 device passes of a multi-arch build
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
#define CLUSTER_DIMS(cl)
#else
#define CLUSTER_DIMS(cl) __cluster_dims__(cl, 1, 1)
#endif
constexpr int kClusterQuantMaxPasses = 4;

// torch's silu, x / (1 + exp(-x)), bit for bit. This file is compiled with
// --use_fast_math, which would turn expf into ex2.approx(x*log2e) and the
// division into an approximate reciprocal; torch is not. Every step below is
// pinned with a rounding-mode intrinsic so fast-math cannot rewrite it, and it
// follows the SASS nvcc emits for torch's kernel: the precise expf range
// reduction (saturating j = floor(a*log2e) via the 1.5*2^23 trick, 2^j from the
// mantissa bits, ex2.approx of the residual) with the trailing "+ 1" contracted
// into the scale multiply exactly as nvcc's default fmad does for torch. The
// final division is non-ftz so bf16 denormal inputs keep their denormal result.
// Verified exhaustively against torch over all 65536 fp16 and bf16 inputs.
__device__ __forceinline__ float torch_silu(float x)
{
    const float a = -x;
    const float t = __saturatef(__fmaf_rn(a, __uint_as_float(0x3bbb989du) /* log2e/252 */, 0.5f));
    const float u = __fmaf_rd(t, 252.0f, 12582913.0f);
    const float j = __fadd_rn(u, -12583039.0f);
    float f = __fmaf_rn(a, 1.4426950216293334961f, -j);
    f = __fmaf_rn(a, 1.925963033500011079e-08f, f);
    float r;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(f));
    const float scale = __uint_as_float(__float_as_uint(u) << 23);
    const float denom = __fmaf_rn(scale, r, 1.0f);
    float y;
    asm("div.rn.f32 %0, %1, %2;" : "=f"(y) : "f"(x), "f"(denom));
    return y;
}

template<typename InputType>
__device__ __forceinline__ float torch_rms_rstd(
    const InputType* __restrict__ row, int K, float eps, float* buf /* >= 6 floats */)
{
    // Mirrors PyTorch vectorized_layer_norm_kernel<rms_norm=true>: blockDim (32, 4).
    constexpr int kNumThreads = 128;
    constexpr int kVec = 4;
    const int tid = threadIdx.x;
    float sigma2 = 0.0f;
    if (tid < kNumThreads) {
        const int n_vec = K / kVec;
        for (int i = tid; i < n_vec; i += kNumThreads) {
            const InputType* v = row + static_cast<int64_t>(i) * kVec;
            #pragma unroll
            for (int ii = 0; ii < kVec; ++ii) {
                const float val = to_float(v[ii]);
                sigma2 = __fmaf_rn(val, val, sigma2);
            }
        }
        for (int offset = kThreadsPerWarp / 2; offset > 0; offset >>= 1) {
            sigma2 = sigma2 + __shfl_down_sync(0xffffffffu, sigma2, offset);
        }
    }
    const int lane = tid & (kThreadsPerWarp - 1);
    const int wid = tid >> 5;
    // inter-warp merge over warps 0..3 in torch's order
    for (int offset = 2; offset > 0; offset /= 2) {
        if (lane == 0 && wid >= offset && wid < 2 * offset) buf[wid - offset] = sigma2;
        __syncthreads();
        if (lane == 0 && wid < offset) sigma2 = sigma2 + buf[wid];
        __syncthreads();
    }
    // IEEE divide like torch's; --use_fast_math would make it approximate
    if (tid == 0) buf[4] = __fdiv_rn(sigma2, static_cast<float>(K));
    __syncthreads();
    return rsqrtf(buf[4] + eps);
}

// act_out (optional, [M, K]) receives the activated row as well, in the input dtype, for
// a consumer that needs act(x) itself alongside the int8 image (only with ACT != none).
template<typename InputType, int CL, int ACT>
__global__ void CLUSTER_DIMS(CL) __launch_bounds__(kClusterQuantThreads)
quantize_int8_convrot_cluster_kernel(
    const InputType* __restrict__ x,
    const InputType* __restrict__ act_weight,
    float act_eps,
    int8_t* __restrict__ q,
    float* __restrict__ scales,
    InputType* __restrict__ act_out,
    int K)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    const int rank = static_cast<int>(cluster.block_rank());
    constexpr int kWarps = kClusterQuantThreads / kThreadsPerWarp;
    constexpr int kSlots = CL * kClusterQuantGroups;

    __shared__ float fht[kClusterQuantGroups][2][kConvRotGroup];
    __shared__ float warp_smem[kWarps];
    __shared__ float block_smem;
    __shared__ float norm_buf[6];
    __shared__ float cluster_max;

    const int tid = threadIdx.x;
    const int sub = tid / 64;
    const int lane = tid % 64;
    const int row = static_cast<int>(blockIdx.x) / CL;
    const int n_groups = K / kConvRotGroup;
    const int passes = (n_groups + kSlots - 1) / kSlots;
    const int64_t row_offset = static_cast<int64_t>(row) * K;
    constexpr int kInWidth = (ACT == kActSwiGLU) ? 2 : 1;
    const InputType* in_row = x + row_offset * kInWidth;

    float rstd = 1.0f;
    if constexpr (ACT == kActRmsNorm) {
        rstd = torch_rms_rstd<InputType>(in_row, K, act_eps, norm_buf);
    }

    float y[kClusterQuantMaxPasses][4];
    float abs_max = 0.0f;
    #pragma unroll
    for (int p = 0; p < kClusterQuantMaxPasses; ++p) {
        if (p >= passes) break;
        const int group = p * kSlots + rank * kClusterQuantGroups + sub;
        const bool active = group < n_groups;
        const int col = group * kConvRotGroup + lane * 4;
        float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (active) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                if constexpr (ACT == kActRmsNorm) {
                    const float g = to_float(act_weight[col + i]);
                    v[i] = to_float(from_float<InputType>(g * (rstd * to_float(in_row[col + i]))));
                } else if constexpr (ACT == kActSwiGLU) {
                    const float gate = to_float(in_row[col + i]);
                    const float up = to_float(in_row[K + col + i]);
                    const float s = to_float(from_float<InputType>(torch_silu(gate)));
                    v[i] = to_float(from_float<InputType>(s * up));
                } else {
                    v[i] = to_float(in_row[col + i]);
                }
            }
            if constexpr (ACT != kActNone) {
                if (act_out != nullptr) {
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        act_out[row_offset + col + i] = from_float<InputType>(v[i]);
                    }
                }
            }
        }
        float* buf0 = fht[sub][0];
        float* buf1 = fht[sub][1];
        const int base = lane * 4;
        buf1[base] = 0.5f * (v[0] + v[1] + v[2] - v[3]);
        buf1[base + 1] = 0.5f * (v[0] + v[1] - v[2] + v[3]);
        buf1[base + 2] = 0.5f * (v[0] - v[1] + v[2] + v[3]);
        buf1[base + 3] = 0.5f * (-v[0] + v[1] + v[2] + v[3]);
        __syncthreads();
        convrot_fht_stage64<4>(buf1, buf0, lane);
        __syncthreads();
        convrot_fht_stage64<16>(buf0, buf1, lane);
        __syncthreads();
        {
            constexpr int S = 64;
            const int b = (lane % S) + (lane / S) * (4 * S);
            const float x0 = buf1[b], x1 = buf1[b + S], x2 = buf1[b + 2 * S], x3 = buf1[b + 3 * S];
            y[p][0] = 0.5f * (x0 + x1 + x2 - x3);
            y[p][1] = 0.5f * (x0 + x1 - x2 + x3);
            y[p][2] = 0.5f * (x0 - x1 + x2 + x3);
            y[p][3] = 0.5f * (-x0 + x1 + x2 + x3);
        }
        if (active) {
            abs_max = fmaxf(abs_max, fmaxf(fmaxf(fabsf(y[p][0]), fabsf(y[p][1])),
                                           fmaxf(fabsf(y[p][2]), fabsf(y[p][3]))));
        }
        __syncthreads();
    }

    abs_max = block_reduce_max_t<kWarps>(abs_max, warp_smem, &block_smem);
    if (tid == 0) cluster_max = abs_max;
    cluster.sync();
    float row_max = 0.0f;
    #pragma unroll
    for (int r = 0; r < CL; ++r) {
        row_max = fmaxf(row_max, cluster.map_shared_rank(&cluster_max, r)[0]);
    }
    cluster.sync();  // every peer has read cluster_max before anyone may exit

    const float scale = fmaxf(
        finite_absmax_for_int8_scale<InputType>(row_max) * (1.0f / 127.0f),
        1.0e-30f);
    if (rank == 0 && tid == 0) {
        scales[row] = scale;
    }
    #pragma unroll
    for (int p = 0; p < kClusterQuantMaxPasses; ++p) {
        if (p >= passes) break;
        const int group = p * kSlots + rank * kClusterQuantGroups + sub;
        if (group >= n_groups) continue;
        // The last FHT stage leaves lane l holding output columns l, l+64, l+128,
        // l+192 of its group (same layout as convrot_fht_stage64_store_absmax).
        int8_t* qg = q + row_offset + group * kConvRotGroup + lane;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float scaled = quant_div_float_to_float<InputType>(y[p][i], scale);
            const float quantized = fminf(127.0f, fmaxf(-128.0f, nearbyintf(scaled)));
            qg[i * 64] = static_cast<int8_t>(quantized);
        }
    }
#endif
}

} // namespace

} // namespace comfy

extern "C" {

void set_int8_prefetch_ring_state(PrefetchRingState* state) {
    int device = 0;
    if (cudaGetDevice(&device) == cudaSuccess && device >= 0 && device < 16) {
        comfy::g_int8_prefetch_ring[device] = state;
    }
}

void launch_quantize_int8_rowwise_kernel(
    const void* input,
    void* output,
    void* scales,
    int64_t num_rows,
    int64_t num_cols,
    int input_dtype_code,
    bool stochastic,
    uint64_t seed,
    cudaStream_t stream)
{
    if (num_rows == 0 || num_cols == 0) {
        return;
    }
    if (num_cols > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("quantize_int8_rowwise only supports K <= INT_MAX");
    }

    DISPATCH_FP_DTYPE(input_dtype_code, InputType, [&] {
        auto launch = [&](auto kernel, int block_threads) {
            kernel<<<static_cast<unsigned int>(num_rows), block_threads, 0, stream>>>(
                static_cast<const InputType*>(input),
                static_cast<int8_t*>(output),
                static_cast<float*>(scales),
                static_cast<int>(num_cols),
                seed);
        };

        if (num_cols >= 4096 && num_rows != 1) {
            if (stochastic) {
                launch(comfy::quantize_int8_rowwise_kernel<InputType, 512, true>, 512);
            } else {
                launch(comfy::quantize_int8_rowwise_kernel<InputType, 512, false>, 512);
            }
        } else {
            if (stochastic) {
                launch(comfy::quantize_int8_rowwise_kernel<InputType, comfy::kInt8Threads, true>,
                       comfy::kInt8Threads);
            } else {
                launch(comfy::quantize_int8_rowwise_kernel<InputType, comfy::kInt8Threads, false>,
                       comfy::kInt8Threads);
            }
        }
    });

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA INT8 rowwise quantization failed: ") + cudaGetErrorString(err));
    }
}

void launch_quantize_int8_rowwise_convrot_kernel(
    const void* input,
    void* output,
    void* scales,
    int64_t num_rows,
    int64_t num_cols,
    int group_size,
    int input_dtype_code,
    bool stochastic,
    uint64_t seed,
    cudaStream_t stream)
{
    if (num_rows == 0 || num_cols == 0) {
        return;
    }
    if (group_size != comfy::kConvRotGroup) {
        throw std::runtime_error("convrot fused kernel only supports group_size 256");
    }
    if (num_cols % comfy::kConvRotGroup != 0) {
        throw std::runtime_error("convrot fused kernel requires K divisible by 256");
    }
    if (num_cols > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("convrot fused kernel only supports K <= INT_MAX");
    }

    // Decode fast path: at small M the smem kernel's barrier chain dominates
    // its runtime (1-3 blocks in flight); the warp-shuffle variant is bit-exact
    // and latency-bound only on the absmax reduce.
    if (!stochastic && num_rows <= 8 && num_cols <= 24576) {  // 256*12 groups covers K<=24576
        // 512-thread wide blocks: a 1024-thread block caps threads at 64
        // registers and spills the register-resident groups to local memory
        DISPATCH_FP_DTYPE(input_dtype_code, InputType, [&] {
            auto launch = [&](auto kernel) {
                kernel<<<static_cast<unsigned int>(num_rows), 256, 0, stream>>>(
                    static_cast<const InputType*>(input),
                    static_cast<int8_t*>(output),
                    static_cast<float*>(scales),
                    static_cast<int>(num_cols));
            };
            if (num_cols <= 8192) {
                launch(comfy::quantize_int8_rowwise_convrot_warp_kernel<InputType, 256, 4>);
            } else {
                launch(comfy::quantize_int8_rowwise_convrot_warp_kernel<InputType, 256, 12>);
            }
        });
        cudaError_t warp_err = cudaGetLastError();
        if (warp_err != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA INT8 rowwise convrot warp quantization failed: ") + cudaGetErrorString(warp_err));
        }
        return;
    }

    // Narrow block for small K (high occupancy via many small blocks); wide
    // block for large K (single smem-bound block per SM needs many warps to
    // hide latency). 256 threads -> 1 group/iter; 1024 threads -> 4 groups/iter.
    const bool wide = num_cols > 5120;
    const int block_threads = wide ? 1024 : comfy::kInt8Threads;  // 1024 or 256
    const int groups_in_flight = block_threads / comfy::kConvRotGroup;
    const size_t smem_bytes =
        (static_cast<size_t>(num_cols) + groups_in_flight * 2 * comfy::kConvRotGroup) * sizeof(float);

    DISPATCH_FP_DTYPE(input_dtype_code, InputType, [&] {
        auto launch = [&](auto kernel) {
            cudaError_t attr_err = cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(smem_bytes));
            if (attr_err != cudaSuccess) {
                throw std::runtime_error(
                    std::string("convrot fused kernel shared memory request (") +
                    std::to_string(smem_bytes) + " bytes) failed: " +
                    cudaGetErrorString(attr_err));
            }
            kernel<<<static_cast<unsigned int>(num_rows), block_threads, smem_bytes, stream>>>(
                static_cast<const InputType*>(input),
                static_cast<int8_t*>(output),
                static_cast<float*>(scales),
                static_cast<int>(num_cols),
                seed);
        };
        if (wide) {
            if (stochastic) {
                launch(comfy::quantize_int8_rowwise_convrot_kernel<InputType, 1024, true>);
            } else {
                launch(comfy::quantize_int8_rowwise_convrot_kernel<InputType, 1024, false>);
            }
        } else {
            if (stochastic) {
                launch(comfy::quantize_int8_rowwise_convrot_kernel<InputType, 256, true>);
            } else {
                launch(comfy::quantize_int8_rowwise_convrot_kernel<InputType, 256, false>);
            }
        }
    });

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA INT8 rowwise convrot quantization failed: ") + cudaGetErrorString(err));
    }
}

void launch_rotate_int8_convrot_weight_kernel(
    const void* input,
    void* output,
    int64_t num_rows,
    int64_t num_cols,
    int group_size,
    int input_dtype_code,
    int output_dtype_code,
    cudaStream_t stream)
{
    if (num_rows == 0 || num_cols == 0) {
        return;
    }
    if (group_size != comfy::kConvRotGroup) {
        throw std::runtime_error("convrot rotate kernel only supports group_size 256");
    }
    if (num_cols % comfy::kConvRotGroup != 0) {
        throw std::runtime_error("convrot rotate kernel requires K divisible by 256");
    }
    if (num_cols > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("convrot rotate kernel only supports K <= INT_MAX");
    }

    constexpr int groups_per_block = 8;
    constexpr int block_threads = groups_per_block * 64;
    const int group_blocks =
        static_cast<int>((num_cols / comfy::kConvRotGroup + groups_per_block - 1) / groups_per_block);
    const size_t smem_bytes = groups_per_block * 2 * comfy::kConvRotGroup * sizeof(float);
    const dim3 grid(
        static_cast<unsigned int>(num_rows),
        static_cast<unsigned int>(group_blocks));

    DISPATCH_FP_DTYPE(input_dtype_code, InputType, [&] {
        DISPATCH_FP_DTYPE(output_dtype_code, OutputType, [&] {
            comfy::rotate_int8_convrot_groups64_kernel<groups_per_block, InputType, OutputType>
                <<<grid, block_threads, smem_bytes, stream>>>(
                    static_cast<const InputType*>(input),
                    static_cast<OutputType*>(output),
                    static_cast<int>(num_cols));
        });
    });

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA INT8 convrot rotation failed: ") + cudaGetErrorString(err));
    }
}

void launch_quantize_int8_convrot_staged_kernel(
    const void* input,
    void* rotated,
    void* partial_absmax,
    void* output,
    void* scales,
    int64_t num_rows,
    int64_t num_cols,
    int group_size,
    int input_dtype_code,
    int rotated_dtype_code,
    bool stochastic,
    uint64_t seed,
    cudaStream_t stream)
{
    if (num_rows == 0 || num_cols == 0) {
        return;
    }
    if (group_size != comfy::kConvRotGroup) {
        throw std::runtime_error("convrot staged quantize only supports group_size 256");
    }
    if (num_cols % comfy::kConvRotGroup != 0) {
        throw std::runtime_error("convrot staged quantize requires K divisible by 256");
    }
    if (num_cols > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("convrot staged quantize only supports K <= INT_MAX");
    }

    constexpr int groups_per_block = 8;
    constexpr int rotate_threads = groups_per_block * 64;
    const int group_blocks =
        static_cast<int>((num_cols / comfy::kConvRotGroup + groups_per_block - 1) / groups_per_block);
    const size_t smem_bytes = groups_per_block * 2 * comfy::kConvRotGroup * sizeof(float);
    const dim3 rotate_grid(
        static_cast<unsigned int>(num_rows),
        static_cast<unsigned int>(group_blocks));

    DISPATCH_FP_DTYPE(input_dtype_code, InputType, [&] {
        DISPATCH_FP_DTYPE(rotated_dtype_code, RotatedType, [&] {
            comfy::rotate_int8_convrot_groups64_amax_kernel<groups_per_block, InputType, RotatedType>
                <<<rotate_grid, rotate_threads, smem_bytes, stream>>>(
                    static_cast<const InputType*>(input),
                    static_cast<RotatedType*>(rotated),
                    static_cast<float*>(partial_absmax),
                    static_cast<int>(num_cols));
        });
    });

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA INT8 staged convrot rotation failed: ") + cudaGetErrorString(err));
    }

    const int quant_threads = num_cols >= 4096 ? 512 : comfy::kInt8Threads;
    DISPATCH_FP_DTYPE(rotated_dtype_code, RotatedType, [&] {
        auto launch = [&](auto kernel, int block_threads) {
            kernel<<<static_cast<unsigned int>(num_rows), block_threads, 0, stream>>>(
                static_cast<const RotatedType*>(rotated),
                static_cast<const float*>(partial_absmax),
                static_cast<int8_t*>(output),
                static_cast<float*>(scales),
                static_cast<int>(num_cols),
                seed);
        };

        if (quant_threads == 512) {
            if (stochastic) {
                launch(comfy::quantize_int8_rowwise_from_partials_kernel<RotatedType, 512, true>, 512);
            } else {
                launch(comfy::quantize_int8_rowwise_from_partials_kernel<RotatedType, 512, false>, 512);
            }
        } else {
            if (stochastic) {
                launch(comfy::quantize_int8_rowwise_from_partials_kernel<RotatedType, comfy::kInt8Threads, true>,
                       comfy::kInt8Threads);
            } else {
                launch(comfy::quantize_int8_rowwise_from_partials_kernel<RotatedType, comfy::kInt8Threads, false>,
                       comfy::kInt8Threads);
            }
        }
    });

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA INT8 staged convrot quantization failed: ") + cudaGetErrorString(err));
    }
}

// Single-launch replacement for the staged pair on sm_90+ (same int8/scale output,
// optional fused input activation). Returns false when the shape or device is not
// covered so the caller can fall back.
bool launch_quantize_int8_convrot_cluster_kernel(
    const void* input,
    const void* act_weight,
    float act_eps,
    int act_code,
    void* output,
    void* scales,
    void* act_out,
    int64_t num_rows,
    int64_t num_cols,
    int input_dtype_code,
    cudaStream_t stream)
{
    using namespace comfy;
    if (num_rows <= 0 || num_rows > 16 || num_cols <= 0 || num_cols % kConvRotGroup != 0
            || input_dtype_code < 0 || input_dtype_code > 2)
        return false;
    if (act_code != kActNone && act_code != kActSwiGLU && act_code != kActRmsNorm)
        return false;
    if (act_code == kActRmsNorm && act_weight == nullptr)
        return false;
    if (act_code == kActNone && act_out != nullptr)
        return false;
    const int64_t n_groups = num_cols / kConvRotGroup;
    // Smallest cluster that covers the row in one pass; otherwise the widest cluster
    // and as many passes as it takes (bounded by the register file the kernel holds).
    int cl = 8;
    for (int c = 1; c <= 8; c *= 2) {
        if (n_groups <= static_cast<int64_t>(c) * kClusterQuantGroups) { cl = c; break; }
    }
    if (n_groups > static_cast<int64_t>(cl) * kClusterQuantGroups * kClusterQuantMaxPasses)
        return false;
    int device = 0, cc_major = 0;
    if (cudaGetDevice(&device) != cudaSuccess
            || cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess
            || cc_major < 9)
        return false;

    DISPATCH_FP_DTYPE(input_dtype_code, InputType, [&] {
        auto launch = [&]<int CL, int ACT>() {
            quantize_int8_convrot_cluster_kernel<InputType, CL, ACT>
                <<<static_cast<unsigned>(num_rows * CL), kClusterQuantThreads, 0, stream>>>(
                    static_cast<const InputType*>(input),
                    static_cast<const InputType*>(act_weight),
                    act_eps,
                    static_cast<int8_t*>(output),
                    static_cast<float*>(scales),
                    static_cast<InputType*>(act_out),
                    static_cast<int>(num_cols));
        };
        auto launch_cl = [&]<int CL>() {
            switch (act_code) {
                case kActSwiGLU: launch.template operator()<CL, kActSwiGLU>(); break;
                case kActRmsNorm: launch.template operator()<CL, kActRmsNorm>(); break;
                default: launch.template operator()<CL, kActNone>(); break;
            }
        };
        switch (cl) {
            case 1: launch_cl.template operator()<1>(); break;
            case 2: launch_cl.template operator()<2>(); break;
            case 4: launch_cl.template operator()<4>(); break;
            default: launch_cl.template operator()<8>(); break;
        }
    });
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA INT8 cluster convrot quantization failed: ") + cudaGetErrorString(err));
    }
    return true;
}

void launch_quantize_int8_rowwise_convrot64_kernel(
    const void* input,
    void* output,
    void* scales,
    int64_t num_rows,
    int64_t num_cols,
    int group_size,
    int input_dtype_code,
    bool stochastic,
    int act_code,
    uint64_t seed,
    const void* act_weight,
    float act_eps,
    cudaStream_t stream)
{
    if (num_rows == 0 || num_cols == 0) {
        return;
    }
    if (group_size != comfy::kConvRotGroup) {
        throw std::runtime_error("convrot64 fused kernel only supports group_size 256");
    }
    if (num_cols % comfy::kConvRotGroup != 0) {
        throw std::runtime_error("convrot64 fused kernel requires K divisible by 256");
    }
    if (num_cols > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("convrot64 fused kernel only supports K <= INT_MAX");
    }
    if (act_code != comfy::kActNone && act_code != comfy::kActGeluTanh
        && act_code != comfy::kActSwiGLU && act_code != comfy::kActRmsNorm) {
        throw std::runtime_error("convrot64 fused kernel: unsupported input activation code");
    }
    if (act_code == comfy::kActRmsNorm && act_weight == nullptr) {
        throw std::runtime_error("convrot64 fused kernel: rms_norm activation requires a weight");
    }

    DISPATCH_FP_DTYPE(input_dtype_code, InputType, [&] {
        auto launch = [&](auto kernel, int block_threads) {
            const int groups_in_flight = block_threads / 64;
            const size_t smem_bytes =
                (static_cast<size_t>(num_cols) + groups_in_flight * 2 * comfy::kConvRotGroup) * sizeof(float);
            cudaError_t attr_err = cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(smem_bytes));
            if (attr_err != cudaSuccess) {
                throw std::runtime_error(
                    std::string("convrot64 fused kernel shared memory request (") +
                    std::to_string(smem_bytes) + " bytes) failed: " +
                    cudaGetErrorString(attr_err));
            }
            kernel<<<static_cast<unsigned int>(num_rows), block_threads, smem_bytes, stream>>>(
                static_cast<const InputType*>(input),
                static_cast<int8_t*>(output),
                static_cast<float*>(scales),
                static_cast<int>(num_cols),
                seed,
                static_cast<const InputType*>(act_weight),
                act_eps);
        };

        // one block per row: smaller blocks keep more rows in flight, wide
        // blocks only pay off for deep rows (measured on sm_120)
        const int block_threads = (num_rows == 1) ? 512
                                : (num_cols == comfy::kConvRotGroup) ? 64
                                : (num_cols <= 3072) ? 128
                                : (num_cols <= 6144) ? 256
                                : (num_cols < 12288) ? 512
                                : 1024;

        DISPATCH_BOOL(stochastic, kStoch, [&] {
            auto launch_act = [&](auto act_tag, int bt) {
                constexpr int kAct = decltype(act_tag)::value;
                switch (bt) {
                    case 64:
                        launch(comfy::quantize_int8_rowwise_convrot64_kernel<InputType, 64, kStoch, kAct>, 64);
                        break;
                    case 128:
                        launch(comfy::quantize_int8_rowwise_convrot64_kernel<InputType, 128, kStoch, kAct>, 128);
                        break;
                    case 256:
                        launch(comfy::quantize_int8_rowwise_convrot64_kernel<InputType, 256, kStoch, kAct>, 256);
                        break;
                    case 512:
                        launch(comfy::quantize_int8_rowwise_convrot64_kernel<InputType, 512, kStoch, kAct>, 512);
                        break;
                    default:
                        launch(comfy::quantize_int8_rowwise_convrot64_kernel<InputType, 1024, kStoch, kAct>, 1024);
                        break;
                }
            };
            switch (act_code) {
                case comfy::kActGeluTanh:
                    launch_act(std::integral_constant<int, comfy::kActGeluTanh>{}, block_threads);
                    break;
                case comfy::kActSwiGLU:
                    launch_act(std::integral_constant<int, comfy::kActSwiGLU>{}, block_threads);
                    break;
                case comfy::kActRmsNorm:
                    launch_act(std::integral_constant<int, comfy::kActRmsNorm>{}, block_threads);
                    break;
                default:
                    launch_act(std::integral_constant<int, comfy::kActNone>{}, block_threads);
                    break;
            }
        });
    });

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA INT8 rowwise convrot64 quantization failed: ") + cudaGetErrorString(err));
    }
}

void launch_dequantize_int8_linear_kernel(
    const void* input,
    const void* x_scales,
    const void* weight_scales,
    const void* bias,
    void* output,
    int64_t num_rows,
    int64_t num_cols,
    int64_t weight_scale_size,
    bool has_bias,
    int output_dtype_code,
    int bias_dtype_code,
    cudaStream_t stream)
{
    if (num_rows == 0 || num_cols == 0) {
        return;
    }
    if (num_cols > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("dequantize_int8_linear only supports N <= INT_MAX");
    }
    if (weight_scale_size != 1 && weight_scale_size != num_cols) {
        throw std::runtime_error("INT8 weight scale must be scalar or per-output-channel");
    }

    const int64_t total = num_rows * num_cols;
    const int blocks = static_cast<int>((total + comfy::kInt8Threads - 1) / comfy::kInt8Threads);

    DISPATCH_FP_DTYPE(output_dtype_code, OutputType, [&] {
        if (!has_bias) {
            if ((num_cols & 3) == 0) {
                constexpr int kVec4Threads = 256;
                const int64_t total_vec4 = total / 4;
                const int vec4_blocks = static_cast<int>((total_vec4 + kVec4Threads - 1) / kVec4Threads);
                comfy::dequantize_int8_linear_vec4_kernel<OutputType>
                    <<<vec4_blocks, kVec4Threads, 0, stream>>>(
                        static_cast<const int32_t*>(input),
                        static_cast<const float*>(x_scales),
                        static_cast<const float*>(weight_scales),
                        static_cast<OutputType*>(output),
                        total_vec4,
                        static_cast<int>(num_cols),
                        static_cast<int>(weight_scale_size));
            } else {
                comfy::dequantize_int8_linear_kernel<OutputType, float>
                    <<<blocks, comfy::kInt8Threads, 0, stream>>>(
                        static_cast<const int32_t*>(input),
                        static_cast<const float*>(x_scales),
                        static_cast<const float*>(weight_scales),
                        nullptr,
                        static_cast<OutputType*>(output),
                        total,
                        static_cast<int>(num_cols),
                        static_cast<int>(weight_scale_size),
                        false);
            }
            return;
        }

        DISPATCH_FP_DTYPE(bias_dtype_code, BiasType, [&] {
            comfy::dequantize_int8_linear_kernel<OutputType, BiasType>
                <<<blocks, comfy::kInt8Threads, 0, stream>>>(
                    static_cast<const int32_t*>(input),
                    static_cast<const float*>(x_scales),
                    static_cast<const float*>(weight_scales),
                    static_cast<const BiasType*>(bias),
                    static_cast<OutputType*>(output),
                    total,
                    static_cast<int>(num_cols),
                    static_cast<int>(weight_scale_size),
                    true);
        });
    });

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA INT8 linear dequantization failed: ") + cudaGetErrorString(err));
    }
}

void launch_int8_gemv_dequant_kernel(
    const void* input,
    const void* weight,
    const void* x_scales,
    const void* weight_scales,
    const void* bias,
    void* output,
    int64_t num_rows,
    int64_t num_cols,
    int64_t K,
    int64_t weight_scale_size,
    bool has_bias,
    int output_dtype_code,
    int bias_dtype_code,
    const void* residual,
    const void* residual_scale,
    cudaStream_t stream)
{
    if (num_cols == 0 || K == 0) {
        return;
    }
    if (num_cols > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        K > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("int8_gemv_dequant only supports N,K <= INT_MAX");
    }
    if (weight_scale_size != 1 && weight_scale_size != num_cols) {
        throw std::runtime_error("INT8 GEMV weight scale must be scalar or per-output-channel");
    }
    if (residual != nullptr && (num_rows != 1 || (K & 3) != 0)) {
        throw std::runtime_error("INT8 GEMV fused residual requires M == 1 and K divisible by 4");
    }
    PrefetchRingState* ring = comfy::int8_prefetch_ring_state();
    // 16-byte loads need K % 16 == 0 and 16-byte aligned activation and weight bases.
    const bool vec16 = (K & 15) == 0
        && (reinterpret_cast<uintptr_t>(input) & 15) == 0
        && (reinterpret_cast<uintptr_t>(weight) & 15) == 0;

    if (num_rows == 2) {
        if (K % 16 != 0) {
            throw std::runtime_error("INT8 two-row GEMV requires K divisible by 16");
        }
        DISPATCH_FP_DTYPE(output_dtype_code, OutputType, [&] {
            auto launch = [&](auto bias_ptr) {
                using BiasType = std::remove_cv_t<std::remove_pointer_t<decltype(bias_ptr)>>;
                constexpr int warps = 8;
                comfy::int8_gemv2_dequant_warp_kernel<warps, OutputType, BiasType>
                    <<<(num_cols + warps - 1) / warps, warps * 32, 0, stream>>>(
                        static_cast<const int8_t*>(input), static_cast<const int8_t*>(weight),
                        static_cast<const float*>(x_scales), static_cast<const float*>(weight_scales),
                        bias_ptr, static_cast<OutputType*>(output), num_cols, K, weight_scale_size, ring);
            };
            if (has_bias) {
                DISPATCH_FP_DTYPE(bias_dtype_code, BiasType, [&] {
                    launch(static_cast<const BiasType*>(bias));
                });
            } else {
                launch(static_cast<const float*>(nullptr));
            }
        });
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA INT8 two-row GEMV failed: ") + cudaGetErrorString(err));
        }
        return;
    }

    DISPATCH_FP_DTYPE(output_dtype_code, OutputType, [&] {
        auto launch_warp = [&](auto bias_ptr, bool bias_flag) {
            using BiasType = std::remove_cv_t<std::remove_pointer_t<decltype(bias_ptr)>>;
            constexpr int kWarpsPerBlock = 8;
            const unsigned int blocks =
                static_cast<unsigned int>((num_cols + kWarpsPerBlock - 1) / kWarpsPerBlock);
            auto kernel = vec16
                ? comfy::int8_gemv_dequant_warp_kernel<kWarpsPerBlock, OutputType, BiasType, true>
                : comfy::int8_gemv_dequant_warp_kernel<kWarpsPerBlock, OutputType, BiasType, false>;
            kernel<<<blocks, kWarpsPerBlock * comfy::kThreadsPerWarp, 0, stream>>>(
                static_cast<const int8_t*>(input),
                static_cast<const int8_t*>(weight),
                static_cast<const float*>(x_scales),
                static_cast<const float*>(weight_scales),
                bias_ptr,
                static_cast<OutputType*>(output),
                static_cast<int>(num_cols),
                static_cast<int>(K),
                static_cast<int>(weight_scale_size),
                bias_flag,
                static_cast<const OutputType*>(residual),
                static_cast<const OutputType*>(residual_scale),
                ring);
        };
        if (!has_bias) {
            if ((K & 3) == 0) {
                launch_warp(static_cast<const float*>(nullptr), false);
            } else {
                comfy::int8_gemv_dequant_kernel<comfy::kInt8Threads, OutputType, float>
                    <<<static_cast<unsigned int>(num_cols), comfy::kInt8Threads, 0, stream>>>(
                        static_cast<const int8_t*>(input),
                        static_cast<const int8_t*>(weight),
                        static_cast<const float*>(x_scales),
                        static_cast<const float*>(weight_scales),
                        nullptr,
                        static_cast<OutputType*>(output),
                        static_cast<int>(num_cols),
                        static_cast<int>(K),
                        static_cast<int>(weight_scale_size),
                        false);
            }
            return;
        }

        DISPATCH_FP_DTYPE(bias_dtype_code, BiasType, [&] {
            if ((K & 3) == 0) {
                launch_warp(static_cast<const BiasType*>(bias), true);
            } else {
                comfy::int8_gemv_dequant_kernel<comfy::kInt8Threads, OutputType, BiasType>
                    <<<static_cast<unsigned int>(num_cols), comfy::kInt8Threads, 0, stream>>>(
                        static_cast<const int8_t*>(input),
                        static_cast<const int8_t*>(weight),
                        static_cast<const float*>(x_scales),
                        static_cast<const float*>(weight_scales),
                        static_cast<const BiasType*>(bias),
                        static_cast<OutputType*>(output),
                        static_cast<int>(num_cols),
                        static_cast<int>(K),
                        static_cast<int>(weight_scale_size),
                        true);
            }
        });
    });

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA INT8 GEMV dequantization failed: ") + cudaGetErrorString(err));
    }
}

void launch_int8_gemv_convrot_fused_kernel(
    const void* input,
    int input_dtype_code,
    int act_code,
    const void* act_weight,
    float act_eps,
    const void* weight,
    const void* weight_scales,
    const void* bias,
    bool has_bias,
    int bias_dtype_code,
    void* output,
    int output_dtype_code,
    int64_t N,
    int64_t K,
    int64_t weight_scale_size,
    const void* residual,
    const void* residual_scale,
    cudaStream_t stream)
{
    if (N == 0 || K == 0) {
        return;
    }
    if (N > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        K > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("int8 fused convrot GEMV only supports N,K <= INT_MAX");
    }
    if (K % comfy::kConvRotGroup != 0) {
        throw std::runtime_error("int8 fused convrot GEMV requires K divisible by 256");
    }
    if ((reinterpret_cast<uintptr_t>(weight) & 15) != 0) {
        throw std::runtime_error("int8 fused convrot GEMV requires a 16-byte aligned weight");
    }
    if (weight_scale_size != 1 && weight_scale_size != N) {
        throw std::runtime_error("INT8 GEMV weight scale must be scalar or per-output-channel");
    }
    if (act_code != comfy::kActNone && act_code != comfy::kActGeluTanh
        && act_code != comfy::kActSwiGLU && act_code != comfy::kActRmsNorm) {
        throw std::runtime_error("int8 fused convrot GEMV: unsupported input activation code");
    }
    if (act_code == comfy::kActRmsNorm && act_weight == nullptr) {
        throw std::runtime_error("int8 fused convrot GEMV: rms_norm activation requires a weight");
    }
    if (has_bias && bias_dtype_code != output_dtype_code) {
        throw std::runtime_error("int8 fused convrot GEMV expects the bias in the output dtype");
    }

    int device = 0;
    int sm_count = 0;
    if (cudaGetDevice(&device) != cudaSuccess
        || cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device) != cudaSuccess) {
        throw std::runtime_error("int8 fused convrot GEMV: failed to query SM count");
    }
    // Every block repeats the quantizer prologue, so cap the blocks per SM and let
    // each walk more column groups instead (2 measured best on sm_120 for the
    // 2048..12288 x 2048..6144 decode shapes).
    constexpr int blocks_per_sm = 2;
    const int64_t col_groups = (N + comfy::kFusedGemvCols - 1) / comfy::kFusedGemvCols;
    const unsigned int blocks =
        static_cast<unsigned int>(std::min<int64_t>(col_groups, static_cast<int64_t>(blocks_per_sm) * sm_count));
    // fp32 row + FHT scratch + int8 row (int8 offset stays 16-byte aligned since K % 256 == 0)
    const size_t smem_bytes =
        (static_cast<size_t>(K) + (comfy::kFusedGemvThreads / 64) * 2 * comfy::kConvRotGroup) * sizeof(float)
        + static_cast<size_t>(K);
    PrefetchRingState* ring = comfy::int8_prefetch_ring_state();

    DISPATCH_FP_DTYPE(input_dtype_code, InputType, [&] {
        DISPATCH_FP_DTYPE(output_dtype_code, OutputType, [&] {
            // the M=1 binding hands the bias over in the output dtype
            auto launch = [&](auto kernel) {
                cudaError_t attr_err = cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes));
                if (attr_err != cudaSuccess) {
                    throw std::runtime_error(
                        std::string("int8 fused convrot GEMV shared memory request (") +
                        std::to_string(smem_bytes) + " bytes) failed: " + cudaGetErrorString(attr_err));
                }
                kernel<<<blocks, comfy::kFusedGemvThreads, smem_bytes, stream>>>(
                    static_cast<const InputType*>(input),
                    static_cast<const InputType*>(act_weight),
                    act_eps,
                    static_cast<const int8_t*>(weight),
                    static_cast<const float*>(weight_scales),
                    static_cast<const OutputType*>(bias),
                    static_cast<OutputType*>(output),
                    static_cast<int>(N),
                    static_cast<int>(K),
                    static_cast<int>(weight_scale_size),
                    has_bias,
                    static_cast<const OutputType*>(residual),
                    static_cast<const OutputType*>(residual_scale),
                    ring);
            };
            switch (act_code) {
                case comfy::kActGeluTanh:
                    launch(comfy::int8_gemv_convrot_fused_kernel<InputType, OutputType, OutputType, comfy::kActGeluTanh>);
                    break;
                case comfy::kActSwiGLU:
                    launch(comfy::int8_gemv_convrot_fused_kernel<InputType, OutputType, OutputType, comfy::kActSwiGLU>);
                    break;
                case comfy::kActRmsNorm:
                    launch(comfy::int8_gemv_convrot_fused_kernel<InputType, OutputType, OutputType, comfy::kActRmsNorm>);
                    break;
                default:
                    launch(comfy::int8_gemv_convrot_fused_kernel<InputType, OutputType, OutputType, comfy::kActNone>);
                    break;
            }
        });
    });

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA INT8 fused convrot GEMV failed: ") + cudaGetErrorString(err));
    }
}

void launch_dequantize_int8_simple_kernel(
    const void* input,
    const void* scales,
    void* output,
    int64_t total,
    int64_t inner_dim,
    int scale_mode,
    int output_dtype_code,
    cudaStream_t stream)
{
    if (total == 0) {
        return;
    }
    if (inner_dim <= 0 || inner_dim > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("dequantize_int8_simple inner dimension is invalid");
    }
    if (scale_mode < 0 || scale_mode > 2) {
        throw std::runtime_error("dequantize_int8_simple scale mode is invalid");
    }

    if (scale_mode == 2 && (total % 4) == 0 && (inner_dim % 4) == 0) {
        const int64_t total_vec4 = total / 4;
        const int64_t rows = total / inner_dim;
        const int inner_dim_vec4 = static_cast<int>(inner_dim / 4);
        if (inner_dim >= 1024 && rows <= static_cast<int64_t>(std::numeric_limits<int>::max())) {
            const int block_threads = inner_dim >= 4096 ? 512 : comfy::kInt8Threads;
            const int blocks_x = static_cast<int>((inner_dim_vec4 + block_threads - 1) / block_threads);
            dim3 grid(static_cast<unsigned int>(rows), static_cast<unsigned int>(blocks_x));
            DISPATCH_FP_DTYPE(output_dtype_code, OutputType, [&] {
                comfy::dequantize_int8_rowwise_vec4_2d_kernel<OutputType>
                    <<<grid, block_threads, 0, stream>>>(
                        static_cast<const int8_t*>(input),
                        static_cast<const float*>(scales),
                        static_cast<OutputType*>(output),
                        static_cast<int>(rows),
                        inner_dim_vec4);
            });

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                throw std::runtime_error(std::string("CUDA INT8 simple dequantization failed: ") + cudaGetErrorString(err));
            }
            return;
        }

        const int blocks = static_cast<int>((total_vec4 + comfy::kInt8Threads - 1) / comfy::kInt8Threads);
        DISPATCH_FP_DTYPE(output_dtype_code, OutputType, [&] {
            comfy::dequantize_int8_simple_vec4_kernel<OutputType>
                <<<blocks, comfy::kInt8Threads, 0, stream>>>(
                    static_cast<const int8_t*>(input),
                    static_cast<const float*>(scales),
                    static_cast<OutputType*>(output),
                    total_vec4,
                    static_cast<int>(inner_dim / 4),
                    scale_mode);
        });

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA INT8 simple dequantization failed: ") + cudaGetErrorString(err));
        }
        return;
    }

    if (scale_mode == 0 && (total % 4) == 0) {
        const int64_t total_vec4 = total / 4;
        const int block_threads = (total_vec4 >= 8'000'000 && total_vec4 <= 16'000'000)
            ? 512
            : comfy::kInt8Threads;
        const int blocks = static_cast<int>((total_vec4 + block_threads - 1) / block_threads);
        DISPATCH_FP_DTYPE(output_dtype_code, OutputType, [&] {
            comfy::dequantize_int8_simple_vec4_kernel<OutputType>
                <<<blocks, block_threads, 0, stream>>>(
                    static_cast<const int8_t*>(input),
                    static_cast<const float*>(scales),
                    static_cast<OutputType*>(output),
                    total_vec4,
                    1,
                    scale_mode);
        });

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA INT8 simple dequantization failed: ") + cudaGetErrorString(err));
        }
        return;
    }

    const int blocks = static_cast<int>((total + comfy::kInt8Threads - 1) / comfy::kInt8Threads);
    DISPATCH_FP_DTYPE(output_dtype_code, OutputType, [&] {
        comfy::dequantize_int8_simple_kernel<OutputType>
            <<<blocks, comfy::kInt8Threads, 0, stream>>>(
                static_cast<const int8_t*>(input),
                static_cast<const float*>(scales),
                static_cast<OutputType*>(output),
                total,
                static_cast<int>(inner_dim),
                scale_mode);
    });

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA INT8 simple dequantization failed: ") + cudaGetErrorString(err));
    }
}

void launch_dequantize_int8_convrot_kernel(
    const void* input,
    const void* scales,
    void* output,
    int64_t num_rows,
    int64_t num_cols,
    int64_t scale_size,
    int group_size,
    int output_dtype_code,
    cudaStream_t stream)
{
    if (num_rows == 0 || num_cols == 0) {
        return;
    }
    if (group_size != comfy::kConvRotGroup && group_size != 64) {
        throw std::runtime_error("convrot dequant kernel only supports group_size 64 or 256");
    }
    if (num_cols % group_size != 0) {
        throw std::runtime_error("convrot dequant kernel requires K divisible by group_size");
    }
    if (num_cols > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("convrot dequant kernel only supports K <= INT_MAX");
    }
    if (scale_size != 1 && scale_size != num_rows) {
        throw std::runtime_error("convrot dequant scale must be scalar or per-row");
    }

    if (group_size == 64) {
        const int64_t num_groups = num_rows * (num_cols / 64);
        const unsigned int blocks = static_cast<unsigned int>((num_groups + 15) / 16);
        DISPATCH_FP_DTYPE(output_dtype_code, OutputType, [&] {
            comfy::dequantize_int8_convrot_g64_kernel<16, OutputType><<<blocks, 16 * 16, 0, stream>>>(
                static_cast<const int8_t*>(input),
                static_cast<const float*>(scales),
                static_cast<OutputType*>(output),
                num_groups,
                static_cast<int>(num_cols),
                static_cast<int>(scale_size));
        });
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA INT8 convrot dequantization failed: ") + cudaGetErrorString(err));
        }
        return;
    }

    if (num_cols >= comfy::kConvRotGroup) {
        auto launch_groups = [&](auto groups_tag) {
            constexpr int groups_per_block = decltype(groups_tag)::value;
            // const (not constexpr): MSVC 14.42 rejects capturing a constexpr
            // local into the nested dispatch lambda (C3495); it is only a
            // runtime launch dimension here.
            const int block_threads = groups_per_block * 64;
            const int group_blocks =
                static_cast<int>((num_cols / comfy::kConvRotGroup + groups_per_block - 1) / groups_per_block);
            const size_t smem_bytes = groups_per_block * 2 * comfy::kConvRotGroup * sizeof(float);
            const dim3 grid(
                static_cast<unsigned int>(num_rows),
                static_cast<unsigned int>(group_blocks));
            DISPATCH_FP_DTYPE(output_dtype_code, OutputType, [&] {
                // Re-derived from the tag: MSVC 14.42 refuses to capture the
                // enclosing lambda's constexpr local (C3495).
                constexpr int kGroupsPerBlock = decltype(groups_tag)::value;
                comfy::dequantize_int8_convrot_groups64_kernel<kGroupsPerBlock, OutputType>
                    <<<grid, block_threads, smem_bytes, stream>>>(
                        static_cast<const int8_t*>(input),
                        static_cast<const float*>(scales),
                        static_cast<OutputType*>(output),
                        static_cast<int>(num_cols),
                        static_cast<int>(scale_size));
            });
        };

        if (num_cols < 1024) {
            launch_groups(std::integral_constant<int, 1>{});
        } else if (num_cols < 4096) {
            launch_groups(std::integral_constant<int, 2>{});
        } else {
            launch_groups(std::integral_constant<int, 4>{});
        }

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA INT8 convrot dequantization failed: ") + cudaGetErrorString(err));
        }
        return;
    }

    const bool wide = num_cols > 5120;
    const int block_threads = wide ? 1024 : comfy::kInt8Threads;
    const int groups_in_flight = block_threads / comfy::kConvRotGroup;
    const size_t smem_bytes =
        (static_cast<size_t>(num_cols) + groups_in_flight * 2 * comfy::kConvRotGroup) * sizeof(float);

    auto launch = [&](auto kernel, auto* output_typed) {
        cudaError_t attr_err = cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes));
        if (attr_err != cudaSuccess) {
            throw std::runtime_error(
                std::string("convrot dequant kernel shared memory request (") +
                std::to_string(smem_bytes) + " bytes) failed: " +
                cudaGetErrorString(attr_err));
        }
        kernel<<<static_cast<unsigned int>(num_rows), block_threads, smem_bytes, stream>>>(
            static_cast<const int8_t*>(input),
            static_cast<const float*>(scales),
            output_typed,
            static_cast<int>(num_cols),
            static_cast<int>(scale_size));
    };
    DISPATCH_FP_DTYPE(output_dtype_code, OutputType, [&] {
        if (wide) {
            launch(comfy::dequantize_int8_convrot_kernel<1024, OutputType>, static_cast<OutputType*>(output));
        } else {
            launch(comfy::dequantize_int8_convrot_kernel<256, OutputType>, static_cast<OutputType*>(output));
        }
    });

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA INT8 convrot dequantization failed: ") + cudaGetErrorString(err));
    }
}

} // extern "C"
