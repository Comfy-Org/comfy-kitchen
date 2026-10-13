// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <stdexcept>

#include "../epilogue.h"
#include "../hadamard.h"
#include "../launchers.h"

namespace comfy::hip_backend {
namespace {

// Decode-sized M (<= 8) is bound on streaming the weight, which the GEMM's tiles would
// only read once per row: one wave per output column reads its fp16 weight row once
// for every x row, in fp32 math (a fp16 weight is exact in fp32), so the result is
// that of linear(x, w.float()) without the fp32 copy of the weight. Built wave64
// (CMakeLists.txt), where it measured 1.2-1.5x faster at 4-8 rows.
constexpr int kGemvWaves = 8;

template <typename InT, typename OutT>
__global__ __launch_bounds__(kWarpSize* kGemvWaves) void gemv_f16_kernel(
    const InT* __restrict__ x, const __half* __restrict__ w, const void* __restrict__ bias,
    int bias_code, OutT* __restrict__ out, int M, int N, int K) {
    const int n = blockIdx.x * kGemvWaves + threadIdx.y;
    if (n >= N) return;
    const __half* wrow = w + static_cast<int64_t>(n) * K;

    float sum[kGemvF16MaxRows] = {};
    for (int k = threadIdx.x * 8; k < K; k += kWarpSize * 8) {
        const uint4 wv = *reinterpret_cast<const uint4*>(wrow + k);
        const __half* wh = reinterpret_cast<const __half*>(&wv);
        float wf[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) wf[i] = __half2float(wh[i]);
#pragma unroll
        for (int m = 0; m < kGemvF16MaxRows; ++m) {
            if (m < M) {
                const InT* xr = x + static_cast<int64_t>(m) * K + k;
#pragma unroll
                for (int i = 0; i < 8; ++i) sum[m] = fmaf(load_row_value(xr[i]), wf[i], sum[m]);
            }
        }
    }
#pragma unroll
    for (int m = 0; m < kGemvF16MaxRows; ++m) {
        if (m < M) {
            float v = warp_reduce_sum(sum[m]);
            if (threadIdx.x == 0) {
                if (bias) v += load_scalar(bias, bias_code, n);
                out[static_cast<int64_t>(m) * N + n] = static_cast<OutT>(v);
            }
        }
    }
}

template <typename InT>
void launch_gemv(const void* x, const void* w, const void* bias, int bias_code, void* d,
                 int out_code, int M, int N, int K, hipStream_t stream) {
    const dim3 block(kWarpSize, kGemvWaves);
    const dim3 grid((N + kGemvWaves - 1) / kGemvWaves);
    const auto* xi = static_cast<const InT*>(x);
    const auto* wh = static_cast<const __half*>(w);
    if (out_code == kF32) {
        gemv_f16_kernel<InT, float><<<grid, block, 0, stream>>>(xi, wh, bias, bias_code, static_cast<float*>(d), M, N, K);
    } else if (out_code == kBF16) {
        gemv_f16_kernel<InT, __bf16><<<grid, block, 0, stream>>>(xi, wh, bias, bias_code, static_cast<__bf16*>(d), M, N, K);
    } else if (out_code == kF16) {
        gemv_f16_kernel<InT, __half><<<grid, block, 0, stream>>>(xi, wh, bias, bias_code, static_cast<__half*>(d), M, N, K);
    } else {
        throw std::runtime_error("fp16_packed_linear: unsupported output dtype");
    }
}

}  // namespace
}  // namespace comfy::hip_backend

extern "C" void launch_gemv_f16_kernel(const void* x, int x_code, const void* w, const void* bias,
                                       int bias_code, void* d, int out_code, int M, int N, int K,
                                       hipStream_t stream) {
    using namespace comfy::hip_backend;

    if (x_code == kF32) {
        launch_gemv<float>(x, w, bias, bias_code, d, out_code, M, N, K, stream);
    } else if (x_code == kF16) {
        launch_gemv<__half>(x, w, bias, bias_code, d, out_code, M, N, K, stream);
    } else if (x_code == kBF16) {
        launch_gemv<__bf16>(x, w, bias, bias_code, d, out_code, M, N, K, stream);
    } else {
        throw std::runtime_error("fp16_packed_linear: unsupported input dtype");
    }
}
