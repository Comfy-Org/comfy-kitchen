// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// INT8 quantization kernels for the WMMA int8 GEMM.
#include <stdexcept>

#include "../hadamard.h"

namespace comfy::hip_backend {

constexpr float kInt8Max = 127.0f;

__forceinline__ __device__ int8_t quant_round(float v, float inv) {
    // rintf is round-half-to-even, matching torch.round.
    int q = static_cast<int>(rintf(v * inv));
    q = q < -128 ? -128 : (q > 127 ? 127 : q);
    return static_cast<int8_t>(q);
}

// One block per row: block-reduce the row absmax, then quantize the row.
__global__ __launch_bounds__(256) void quantize_int8_rowwise_kernel(
    const void* __restrict__ x, int in_code, int8_t* __restrict__ q,
    float* __restrict__ scales, int M, int K) {

    __shared__ float red[256];
    const int row = blockIdx.x;
    const int t = threadIdx.x;

    float lmax = 0.0f;
    for (int j = t; j < K; j += 256)
        lmax = fmaxf(lmax, fabsf(load_in(x, static_cast<int64_t>(row) * K + j, in_code)));

    red[t] = lmax;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (t < s) red[t] = fmaxf(red[t], red[t + s]);
        __syncthreads();
    }

    const float scale = fmaxf(red[0] / kInt8Max, 1e-30f);
    if (t == 0) scales[row] = scale;
    const float inv = 1.0f / scale;

    for (int j = t; j < K; j += 256) {
        const float v = load_in(x, static_cast<int64_t>(row) * K + j, in_code);
        q[static_cast<int64_t>(row) * K + j] = quant_round(v, inv);
    }
}

__global__ void absmax_kernel(const void* __restrict__ x, int in_code,
                              unsigned int* __restrict__ out, int64_t numel) {
    __shared__ float red[256];
    const int t = threadIdx.x;

    float lmax = 0.0f;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + t; i < numel;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x)
        lmax = fmaxf(lmax, fabsf(load_in(x, i, in_code)));

    red[t] = lmax;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (t < s) red[t] = fmaxf(red[t], red[t + s]);
        __syncthreads();
    }
    // Non-negative floats compare identically to their bit patterns as unsigned.
    if (t == 0) atomicMax(out, __float_as_uint(red[0]));
}

__global__ void scale_from_absmax_kernel(const unsigned int* __restrict__ absmax,
                                         float* __restrict__ scale) {
    if (threadIdx.x == 0) *scale = fmaxf(__uint_as_float(*absmax) / kInt8Max, 1e-30f);
}

__global__ void quantize_int8_tensorwise_kernel(
    const void* __restrict__ x, int in_code, int8_t* __restrict__ q,
    const float* __restrict__ scale, int64_t numel) {

    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    q[i] = quant_round(load_in(x, i, in_code), 1.0f / scale[0]);
}

}  // namespace comfy::hip_backend

extern "C" void launch_quantize_int8_rowwise_kernel(
    const void* x, int in_code, void* q, void* scales, int M, int K, hipStream_t stream) {

    using namespace comfy::hip_backend;
    if (M == 0) {
        return;  // a zero-block launch is hipErrorInvalidConfiguration
    }
    quantize_int8_rowwise_kernel<<<M, 256, 0, stream>>>(
        x, in_code, static_cast<int8_t*>(q), static_cast<float*>(scales), M, K);
}

// Fused ConvRot rotation + rowwise int8 quantize.
extern "C" void launch_quantize_int8_convrot_kernel(
    const void* x, int in_code, void* q, void* scales,
    int M, int K, int group_size, hipStream_t stream) {

    using namespace comfy::hip_backend;
    check_convrot_group_size(group_size);
    check_convrot_k(K, group_size);
    if (M == 0) {
        return;  // a zero-block launch is hipErrorInvalidConfiguration
    }
    const size_t shmem = static_cast<size_t>(K) * sizeof(__bf16);
    convrot_quant_kernel<false><<<M, 256, shmem, stream>>>(
        x, in_code, static_cast<int8_t*>(q), static_cast<float*>(scales), M, K, group_size);
}

// scratch is a 4-byte device buffer, pre-zeroed by the caller.
extern "C" void launch_quantize_int8_tensorwise_kernel(
    const void* x, int in_code, void* q, void* scale, void* scratch,
    int64_t numel, hipStream_t stream) {

    using namespace comfy::hip_backend;
    const int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    const int rblocks = blocks < 1024 ? blocks : 1024;

    // An empty input has no absmax and nothing to quantize, but the scale is still
    // an output, so only the zero-block launches are skipped. scratch is zeroed,
    // which the scale kernel clamps.
    if (numel > 0) {
        absmax_kernel<<<rblocks, threads, 0, stream>>>(
            x, in_code, static_cast<unsigned int*>(scratch), numel);
    }
    scale_from_absmax_kernel<<<1, 1, 0, stream>>>(
        static_cast<const unsigned int*>(scratch), static_cast<float*>(scale));
    if (numel > 0) {
        quantize_int8_tensorwise_kernel<<<blocks, threads, 0, stream>>>(
            x, in_code, static_cast<int8_t*>(q), static_cast<const float*>(scale), numel);
    }
}
