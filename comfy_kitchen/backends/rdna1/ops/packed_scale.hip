// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// The packed convs' accumulator scale in two launches: the largest |x| and |weight| by
// atomic max, then the power of two that keeps every packed partial sum in fp16's range.
// The same result took a dozen tensor ops, which cost more than a small conv's GEMM.
#include "../epilogue.h"

namespace comfy::hip_backend {

constexpr int kAbsmaxThreads = 256;

// bits[slot] = max(bits[slot], |t|) over a dense tensor, as raw float bits: for
// non-negative floats the unsigned order of the bits is the order of the values, and a
// NaN's bits sit above infinity's, so it survives the reduction.
__global__ void __launch_bounds__(kAbsmaxThreads)
absmax_bits_kernel(const void* __restrict__ t, int code, int64_t n, unsigned int* __restrict__ bits,
                   int slot) {
    const int64_t threads = static_cast<int64_t>(gridDim.x) * blockDim.x;
    float m = 0.0f;
    bool nan = false;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += threads) {
        const float v = fabsf(load_scalar(t, code, i));
        nan |= v != v;
        m = fmaxf(m, v);
    }
    atomicMax(bits + slot, nan ? 0x7fc00000u : __float_as_uint(m));
}

// out[0] = 2**e with e the smallest integer that bounds a stage's partial sum
// (stage_ratio = products per stage / the fp16 sum limit). With x scaled, e also keeps x's
// largest element under 2**15 and is floored at -64 so an all-zero x divides by a finite
// value; with the weight scaled, e >= 0 leaves a small weight untouched.
__global__ void packed_scale_kernel(const unsigned int* __restrict__ bits, float* __restrict__ out,
                                    float stage_ratio, bool scale_weight) {
    const float x_amax = __uint_as_float(bits[0]);
    const float bound = x_amax * __uint_as_float(bits[1]) * stage_ratio;
    float e = log2f(bound);
    if (scale_weight) {
        e = fmaxf(ceilf(e), 0.0f);
    } else {
        e = fmaxf(ceilf(fmaxf(e, log2f(x_amax) - 15.0f)), -64.0f);
    }
    // fmaxf drops a NaN operand; a NaN in either tensor still has to reach the output
    out[0] = bound != bound ? bound : exp2f(e);
}

}  // namespace comfy::hip_backend

// bits is a zeroed two-element scratch on the same stream; out receives the scale.
extern "C" void launch_packed_acc_scale_kernel(const void* x, int x_code, int64_t x_n,
                                               const void* w, int w_code, int64_t w_n, void* bits,
                                               void* out, float stage_ratio, bool scale_weight,
                                               hipStream_t stream) {
    using namespace comfy::hip_backend;
    auto* b = static_cast<unsigned int*>(bits);
    const auto launch = [&](const void* t, int code, int64_t n, int slot) {
        if (n == 0) return;
        const int64_t blocks = std::min<int64_t>(1024, (n + 16 * kAbsmaxThreads - 1) / (16 * kAbsmaxThreads));
        hipLaunchKernelGGL(absmax_bits_kernel, dim3(static_cast<unsigned>(blocks)), dim3(kAbsmaxThreads),
                           0, stream, t, code, n, b, slot);
    };
    launch(x, x_code, x_n, 0);
    launch(w, w_code, w_n, 1);
    hipLaunchKernelGGL(packed_scale_kernel, dim3(1), dim3(1), 0, stream, b,
                       static_cast<float*>(out), stage_ratio, scale_weight);
}
