/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils.cuh"
#include "dtype_dispatch.cuh"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace comfy {

namespace {

constexpr int kInt8Threads = 256;
constexpr int kInt8Warps = kInt8Threads / kThreadsPerWarp;

template<typename T>
__device__ __forceinline__ float to_float(T val);
template<> __device__ __forceinline__ float to_float<float>(float val) { return val; }
template<> __device__ __forceinline__ float to_float<half>(half val) { return __half2float(val); }
template<> __device__ __forceinline__ float to_float<nv_bfloat16>(nv_bfloat16 val) { return __bfloat162float(val); }

template<typename T>
__device__ __forceinline__ T from_float(float val);
template<> __device__ __forceinline__ float from_float<float>(float val) { return val; }
template<> __device__ __forceinline__ half from_float<half>(float val) { return __float2half_rn(val); }
template<> __device__ __forceinline__ nv_bfloat16 from_float<nv_bfloat16>(float val) { return __float2bfloat16_rn(val); }

__device__ __forceinline__ float warp_reduce_max(float v) {
    for (int offset = kThreadsPerWarp / 2; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
    }
    return v;
}

__device__ __forceinline__ float block_reduce_max(float v, float* warp_smem, float* block_smem) {
    const int lane = threadIdx.x & (kThreadsPerWarp - 1);
    const int wid = threadIdx.x >> 5;

    v = warp_reduce_max(v);
    if (lane == 0) {
        warp_smem[wid] = v;
    }
    __syncthreads();

    float total = 0.0f;
    if (wid == 0) {
        total = lane < kInt8Warps ? warp_smem[lane] : 0.0f;
        total = warp_reduce_max(total);
        if (lane == 0) {
            *block_smem = total;
        }
    }
    __syncthreads();
    return *block_smem;
}

template<typename InputType>
__global__ void quantize_int8_rowwise_kernel(
    const InputType* __restrict__ x,
    int8_t* __restrict__ q,
    float* __restrict__ scales,
    int K)
{
    __shared__ float warp_smem[kInt8Warps];
    __shared__ float block_smem;

    const int row = static_cast<int>(blockIdx.x);
    const int tid = threadIdx.x;
    const int row_offset = row * K;

    float abs_max = 0.0f;
    for (int col = tid; col < K; col += blockDim.x) {
        abs_max = fmaxf(abs_max, fabsf(to_float(x[row_offset + col])));
    }

    abs_max = block_reduce_max(abs_max, warp_smem, &block_smem);
    const float scale = fmaxf(abs_max * (1.0f / 127.0f), 1.0e-30f);
    const float inv_scale = 1.0f / scale;

    if (tid == 0) {
        scales[row] = scale;
    }

    for (int col = tid; col < K; col += blockDim.x) {
        float quantized = nearbyintf(to_float(x[row_offset + col]) * inv_scale);
        quantized = fminf(127.0f, fmaxf(-128.0f, quantized));
        q[row_offset + col] = static_cast<int8_t>(quantized);
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

template<typename InputType, int BLOCK_THREADS>
__global__ void quantize_int8_rowwise_convrot_kernel(
    const InputType* __restrict__ x,
    int8_t* __restrict__ q,
    float* __restrict__ scales,
    int K)
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
    const float scale = fmaxf(abs_max * (1.0f / 127.0f), 1.0e-30f);
    const float inv_scale = 1.0f / scale;
    if (tid == 0) {
        scales[row] = scale;
    }

    for (int col = tid; col < K; col += BLOCK_THREADS) {
        float quantized = nearbyintf(row_buf[col] * inv_scale);
        quantized = fminf(127.0f, fmaxf(-128.0f, quantized));
        q[row_offset + col] = static_cast<int8_t>(quantized);
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
    cudaStream_t stream)
{
    if (num_rows == 0 || num_cols == 0) {
        return;
    }
    if (num_cols > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("quantize_int8_rowwise only supports K <= INT_MAX");
    }

    DISPATCH_FP_DTYPE(input_dtype_code, InputType, [&] {
        comfy::quantize_int8_rowwise_kernel<InputType>
            <<<static_cast<unsigned int>(num_rows), comfy::kInt8Threads, 0, stream>>>(
                static_cast<const InputType*>(input),
                static_cast<int8_t*>(output),
                static_cast<float*>(scales),
                static_cast<int>(num_cols));
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
                static_cast<int>(num_cols));
        };
        if (wide) {
            launch(comfy::quantize_int8_rowwise_convrot_kernel<InputType, 1024>);
        } else {
            launch(comfy::quantize_int8_rowwise_convrot_kernel<InputType, 256>);
        }
    });

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA INT8 rowwise convrot quantization failed: ") + cudaGetErrorString(err));
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

} // extern "C"
