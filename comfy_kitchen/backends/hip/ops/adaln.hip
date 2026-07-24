// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Fused AdaLN: layernorm(x, no affine) * (1 + scale) + shift.
//
// One block per row. The row is re-read for the normalization pass rather than
// held in registers, which does not scale to large D.
//
// scale/shift broadcast over inner dims, expressed as a row group: output row r
// reads modulation row r / group (see backends/_modulation.py).
#include <stdexcept>

#include "../hadamard.h"  // load_in / dtype codes

namespace comfy::hip_backend {

constexpr int kAdaLNThreads = 256;

template <typename T>
__forceinline__ __device__ void store_out(T* p, int64_t i, float v);

template <>
__forceinline__ __device__ void store_out<float>(float* p, int64_t i, float v) {
    p[i] = v;
}
template <>
__forceinline__ __device__ void store_out<__half>(__half* p, int64_t i, float v) {
    p[i] = __float2half(v);
}
template <>
__forceinline__ __device__ void store_out<__bf16>(__bf16* p, int64_t i, float v) {
    p[i] = static_cast<__bf16>(v);
}

// scale/shift need not share x's dtype (e.g. fp32 modulation against a bf16
// activation), so each buffer carries its own dtype code.
template <typename T>
__global__ __launch_bounds__(kAdaLNThreads) void adaln_kernel(
    const void* __restrict__ x, const void* __restrict__ scale,
    const void* __restrict__ shift, T* __restrict__ out,
    int D, int scale_group, int shift_group, float eps,
    int code, int scale_code, int shift_code) {

    __shared__ float red_sum[kAdaLNThreads];
    __shared__ float red_sq[kAdaLNThreads];

    const int row = blockIdx.x;
    const int t = threadIdx.x;

    const int64_t xbase = static_cast<int64_t>(row) * D;
    const int64_t sbase = static_cast<int64_t>(row / scale_group) * D;
    const int64_t hbase = static_cast<int64_t>(row / shift_group) * D;

    float sum = 0.0f;
    for (int i = t; i < D; i += kAdaLNThreads) {
        sum += load_in(x, xbase + i, code);
    }

    red_sum[t] = sum;
    __syncthreads();
    for (int s = kAdaLNThreads / 2; s > 0; s >>= 1) {
        if (t < s) red_sum[t] += red_sum[t + s];
        __syncthreads();
    }

    const float inv_d = 1.0f / static_cast<float>(D);
    const float mean = red_sum[0] * inv_d;

    // Sum the squared deviations rather than E[x^2] - mean^2: the latter cancels
    // catastrophically once the row's offset dwarfs its spread.
    float sq = 0.0f;
    for (int i = t; i < D; i += kAdaLNThreads) {
        const float d = load_in(x, xbase + i, code) - mean;
        sq += d * d;
    }

    red_sq[t] = sq;
    __syncthreads();
    for (int s = kAdaLNThreads / 2; s > 0; s >>= 1) {
        if (t < s) red_sq[t] += red_sq[t + s];
        __syncthreads();
    }

    const float var = red_sq[0] * inv_d;
    const float rstd = rsqrtf(var + eps);

    for (int i = t; i < D; i += kAdaLNThreads) {
        const float v = load_in(x, xbase + i, code);
        const float norm = (v - mean) * rstd;
        const float s = load_in(scale, sbase + i, scale_code);
        const float h = load_in(shift, hbase + i, shift_code);
        store_out<T>(out, xbase + i, norm * (1.0f + s) + h);
    }
}

}  // namespace comfy::hip_backend

extern "C" void launch_adaln_kernel(
    const void* x, const void* scale, const void* shift, void* out,
    int N, int D, int scale_group, int shift_group, float eps,
    int code, int scale_code, int shift_code, hipStream_t stream) {

    using namespace comfy::hip_backend;

    if (N == 0 || D == 0) {
        // No rows, or zero-width rows: nothing to normalize, and D == 0 would make
        // the kernel's group divisions divide by zero.
        return;  // a zero-block launch is hipErrorInvalidConfiguration
    }

#define CK_ADALN_LAUNCH(T)                                                                    \
    adaln_kernel<T><<<N, kAdaLNThreads, 0, stream>>>(x, scale, shift, static_cast<T*>(out), D, \
                                                     scale_group, shift_group, eps, code,      \
                                                     scale_code, shift_code)

    if (code == 0) {
        CK_ADALN_LAUNCH(float);
    } else if (code == 1) {
        CK_ADALN_LAUNCH(__half);
    } else if (code == 2) {
        CK_ADALN_LAUNCH(__bf16);
    } else {
        throw std::runtime_error("adaln: unsupported dtype");
    }
#undef CK_ADALN_LAUNCH
}
