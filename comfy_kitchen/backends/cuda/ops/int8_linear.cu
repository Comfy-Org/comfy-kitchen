/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils.cuh"
#include "dtype_dispatch.cuh"

#include <cmath>
#include <cfloat>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace comfy {

namespace {

constexpr int kInt8Threads = 256;

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

__device__ __forceinline__ int warp_reduce_sum_i32(int v) {
    for (int offset = kThreadsPerWarp / 2; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
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
    const int row_offset = row * K;

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
        const int idx = row_offset + col;
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

template<int WARPS_PER_BLOCK, typename OutputType, typename BiasType>
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
    bool has_bias)
{
    const int lane = threadIdx.x & (kThreadsPerWarp - 1);
    const int warp = threadIdx.x >> 5;
    const int n = static_cast<int>(blockIdx.x) * WARPS_PER_BLOCK + warp;
    if (n >= N) {
        return;
    }

    const int K4 = K >> 2;
    const int* __restrict__ x4 = reinterpret_cast<const int*>(x);
    const int* __restrict__ w4 = reinterpret_cast<const int*>(weight + static_cast<int64_t>(n) * K);

    int acc = 0;
    for (int k4 = lane; k4 < K4; k4 += kThreadsPerWarp) {
        acc = __dp4a(x4[k4], w4[k4], acc);
    }
    acc = warp_reduce_sum_i32(acc);

    if (lane == 0) {
        const float weight_scale = weight_scales[weight_scale_size == 1 ? 0 : n];
        float value = static_cast<float>(acc) * x_scales[0] * weight_scale;
        if (has_bias) {
            value += to_float(bias[n]);
        }
        output[n] = from_float<OutputType>(value);
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
    const int row = static_cast<int>(blockIdx.y);
    const int col4 = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
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
    const int group = static_cast<int>(blockIdx.x) * GROUPS_PER_BLOCK + sub;
    const int row = static_cast<int>(blockIdx.y);
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
    const int group = static_cast<int>(blockIdx.x) * GROUPS_PER_BLOCK + sub;
    const int row = static_cast<int>(blockIdx.y);
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
    const int group = static_cast<int>(blockIdx.x) * GROUPS_PER_BLOCK + sub;
    const int row = static_cast<int>(blockIdx.y);
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

template<typename InputType, int BLOCK_THREADS, bool STOCHASTIC>
__global__ void quantize_int8_rowwise_convrot64_kernel(
    const InputType* __restrict__ x,
    int8_t* __restrict__ q,
    float* __restrict__ scales,
    int K,
    uint64_t seed)
{
    constexpr int kGroupThreads = 64;
    constexpr int kGroupsInFlight = BLOCK_THREADS / kGroupThreads;
    constexpr int kWarps = BLOCK_THREADS / kThreadsPerWarp;

    extern __shared__ float smem[];
    float* row_buf = smem;
    float* tmp = smem + K;

    __shared__ float warp_smem[kWarps];
    __shared__ float block_smem;

    const int row = static_cast<int>(blockIdx.x);
    const int tid = threadIdx.x;
    const int sub = tid / kGroupThreads;
    const int lane = tid % kGroupThreads;
    const int64_t row_offset = static_cast<int64_t>(row) * K;
    const int n_groups = K / kConvRotGroup;

    float* buf0 = tmp + sub * (2 * kConvRotGroup);
    float* buf1 = buf0 + kConvRotGroup;
    float abs_max = 0.0f;

    const int iters = (n_groups + kGroupsInFlight - 1) / kGroupsInFlight;
    for (int it = 0; it < iters; ++it) {
        const int group = it * kGroupsInFlight + sub;
        const bool active = group < n_groups;
        const int base = lane * 4;
        const int group_col = group * kConvRotGroup;
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
            abs_max = fmaxf(
                abs_max,
                convrot_fht_stage64_store_absmax<64, float>(buf1, row_buf + group_col, lane));
        }
        __syncthreads();
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

} // namespace

} // namespace comfy

extern "C" {

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
    dim3 grid(static_cast<unsigned int>(group_blocks), static_cast<unsigned int>(num_rows));

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
    dim3 rotate_grid(static_cast<unsigned int>(group_blocks), static_cast<unsigned int>(num_rows));

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

void launch_quantize_int8_rowwise_convrot64_kernel(
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
        throw std::runtime_error("convrot64 fused kernel only supports group_size 256");
    }
    if (num_cols % comfy::kConvRotGroup != 0) {
        throw std::runtime_error("convrot64 fused kernel requires K divisible by 256");
    }
    if (num_cols > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("convrot64 fused kernel only supports K <= INT_MAX");
    }

    DISPATCH_FP_DTYPE(input_dtype_code, InputType, [&] {
        constexpr int block_threads_single = 512;
        constexpr int block_threads_multi = 1024;
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
                seed);
        };

        if (num_rows == 1) {
            if (stochastic) {
                launch(comfy::quantize_int8_rowwise_convrot64_kernel<InputType, block_threads_single, true>,
                       block_threads_single);
            } else {
                launch(comfy::quantize_int8_rowwise_convrot64_kernel<InputType, block_threads_single, false>,
                       block_threads_single);
            }
        } else if (num_cols == comfy::kConvRotGroup) {
            constexpr int block_threads_256 = 64;
            if (stochastic) {
                launch(comfy::quantize_int8_rowwise_convrot64_kernel<InputType, block_threads_256, true>,
                       block_threads_256);
            } else {
                launch(comfy::quantize_int8_rowwise_convrot64_kernel<InputType, block_threads_256, false>,
                       block_threads_256);
            }
        } else if (num_cols == 2560) {
            constexpr int block_threads_2560 = 640;
            if (stochastic) {
                launch(comfy::quantize_int8_rowwise_convrot64_kernel<InputType, block_threads_2560, true>,
                       block_threads_2560);
            } else {
                launch(comfy::quantize_int8_rowwise_convrot64_kernel<InputType, block_threads_2560, false>,
                       block_threads_2560);
            }
        } else if (num_cols == 6144) {
            constexpr int block_threads_6144 = 768;
            if (stochastic) {
                launch(comfy::quantize_int8_rowwise_convrot64_kernel<InputType, block_threads_6144, true>,
                       block_threads_6144);
            } else {
                launch(comfy::quantize_int8_rowwise_convrot64_kernel<InputType, block_threads_6144, false>,
                       block_threads_6144);
            }
        } else {
            if (stochastic) {
                launch(comfy::quantize_int8_rowwise_convrot64_kernel<InputType, block_threads_multi, true>,
                       block_threads_multi);
            } else {
                launch(comfy::quantize_int8_rowwise_convrot64_kernel<InputType, block_threads_multi, false>,
                       block_threads_multi);
            }
        }
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
    int64_t num_cols,
    int64_t K,
    int64_t weight_scale_size,
    bool has_bias,
    int output_dtype_code,
    int bias_dtype_code,
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

    DISPATCH_FP_DTYPE(output_dtype_code, OutputType, [&] {
        if (!has_bias) {
            if ((K & 3) == 0) {
                constexpr int kWarpsPerBlock = 8;
                const unsigned int blocks =
                    static_cast<unsigned int>((num_cols + kWarpsPerBlock - 1) / kWarpsPerBlock);
                comfy::int8_gemv_dequant_warp_kernel<kWarpsPerBlock, OutputType, float>
                    <<<blocks, kWarpsPerBlock * comfy::kThreadsPerWarp, 0, stream>>>(
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
                constexpr int kWarpsPerBlock = 8;
                const unsigned int blocks =
                    static_cast<unsigned int>((num_cols + kWarpsPerBlock - 1) / kWarpsPerBlock);
                comfy::int8_gemv_dequant_warp_kernel<kWarpsPerBlock, OutputType, BiasType>
                    <<<blocks, kWarpsPerBlock * comfy::kThreadsPerWarp, 0, stream>>>(
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
            dim3 grid(static_cast<unsigned int>(blocks_x), static_cast<unsigned int>(rows));
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
    if (group_size != comfy::kConvRotGroup) {
        throw std::runtime_error("convrot dequant kernel only supports group_size 256");
    }
    if (num_cols % comfy::kConvRotGroup != 0) {
        throw std::runtime_error("convrot dequant kernel requires K divisible by 256");
    }
    if (num_cols > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("convrot dequant kernel only supports K <= INT_MAX");
    }
    if (scale_size != 1 && scale_size != num_rows) {
        throw std::runtime_error("convrot dequant scale must be scalar or per-row");
    }

    if (num_cols >= comfy::kConvRotGroup) {
        auto launch_groups = [&](auto groups_tag) {
            constexpr int groups_per_block = decltype(groups_tag)::value;
            constexpr int block_threads = groups_per_block * 64;
            const int group_blocks =
                static_cast<int>((num_cols / comfy::kConvRotGroup + groups_per_block - 1) / groups_per_block);
            const size_t smem_bytes = groups_per_block * 2 * comfy::kConvRotGroup * sizeof(float);
            dim3 grid(static_cast<unsigned int>(group_blocks), static_cast<unsigned int>(num_rows));
            DISPATCH_FP_DTYPE(output_dtype_code, OutputType, [&] {
                comfy::dequantize_int8_convrot_groups64_kernel<groups_per_block, OutputType>
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
