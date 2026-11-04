// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <stdexcept>
#include <type_traits>

#include "../epilogue.h"
#include "../hadamard.h"
#include "../launchers.h"

namespace comfy::hip_backend {
namespace {

// Decode-sized M (<= 8) is bound on streaming the weight, which the GEMM's tiles would
// only read once per row: one wave per output column reads its weight row once for
// every x row, in fp32 math (an fp16 or bf16 weight is exact in fp32), so the result is
// that of linear(x, w.float()) without a cast copy of the weight. Built wave64
// (CMakeLists.txt), where it measured 1.2-1.5x faster at 4-8 rows.
constexpr int kGemvWaves = 8;

// Eight weight elements as fp32, read 16 bytes at a time.
template <typename WT>
__forceinline__ __device__ void load_weight8(const WT* p, float* wf) {
    if constexpr (sizeof(WT) == 4) {
        const float4 a = *reinterpret_cast<const float4*>(p);
        const float4 b = *reinterpret_cast<const float4*>(p + 4);
        wf[0] = a.x; wf[1] = a.y; wf[2] = a.z; wf[3] = a.w;
        wf[4] = b.x; wf[5] = b.y; wf[6] = b.z; wf[7] = b.w;
    } else {
        const uint4 wv = *reinterpret_cast<const uint4*>(p);
        const unsigned int words[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const unsigned int word = words[i / 2];
            if constexpr (std::is_same_v<WT, __half>) {
                wf[i] = __half2float(__ushort_as_half(static_cast<unsigned short>(i % 2 ? word >> 16 : word)));
            } else {
                wf[i] = __uint_as_float(i % 2 ? word & 0xffff0000u : word << 16);
            }
        }
    }
}

template <typename InT, typename WT, typename OutT>
__global__ __launch_bounds__(kWarpSize* kGemvWaves) void gemv_f16_kernel(
    const InT* __restrict__ x, const WT* __restrict__ w, const void* __restrict__ bias,
    int bias_code, OutT* __restrict__ out, int M, int N, int K) {
    const int n = blockIdx.x * kGemvWaves + threadIdx.y;
    if (n >= N) return;
    const WT* wrow = w + static_cast<int64_t>(n) * K;

    float sum[kGemvF16MaxRows] = {};
    for (int k = threadIdx.x * 8; k < K; k += kWarpSize * 8) {
        float wf[8];
        load_weight8(wrow + k, wf);
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

template <typename InT, typename WT>
void launch_gemv(const void* x, const void* w, const void* bias, int bias_code, void* d,
                 int out_code, int M, int N, int K, hipStream_t stream) {
    const dim3 block(kWarpSize, kGemvWaves);
    const dim3 grid((N + kGemvWaves - 1) / kGemvWaves);
    const auto* xi = static_cast<const InT*>(x);
    const auto* wt = static_cast<const WT*>(w);
    if (out_code == kF32) {
        gemv_f16_kernel<InT, WT, float><<<grid, block, 0, stream>>>(xi, wt, bias, bias_code, static_cast<float*>(d), M, N, K);
    } else if (out_code == kBF16) {
        gemv_f16_kernel<InT, WT, __bf16><<<grid, block, 0, stream>>>(xi, wt, bias, bias_code, static_cast<__bf16*>(d), M, N, K);
    } else if (out_code == kF16) {
        gemv_f16_kernel<InT, WT, __half><<<grid, block, 0, stream>>>(xi, wt, bias, bias_code, static_cast<__half*>(d), M, N, K);
    } else {
        throw std::runtime_error("fp16_packed_linear: unsupported output dtype");
    }
}

template <typename InT>
void launch_gemv_weight(const void* x, const void* w, int w_code, const void* bias, int bias_code,
                        void* d, int out_code, int M, int N, int K, hipStream_t stream) {
    if (w_code == kF16) {
        launch_gemv<InT, __half>(x, w, bias, bias_code, d, out_code, M, N, K, stream);
    } else if (w_code == kBF16) {
        launch_gemv<InT, __bf16>(x, w, bias, bias_code, d, out_code, M, N, K, stream);
    } else if (w_code == kF32) {
        launch_gemv<InT, float>(x, w, bias, bias_code, d, out_code, M, N, K, stream);
    } else {
        throw std::runtime_error("fp16_packed_linear: unsupported weight dtype");
    }
}

}  // namespace
}  // namespace comfy::hip_backend

extern "C" void launch_gemv_f16_kernel(const void* x, int x_code, const void* w, int w_code,
                                       const void* bias, int bias_code, void* d, int out_code,
                                       int M, int N, int K, hipStream_t stream) {
    using namespace comfy::hip_backend;

    if (x_code == kF32) {
        launch_gemv_weight<float>(x, w, w_code, bias, bias_code, d, out_code, M, N, K, stream);
    } else if (x_code == kF16) {
        launch_gemv_weight<__half>(x, w, w_code, bias, bias_code, d, out_code, M, N, K, stream);
    } else if (x_code == kBF16) {
        launch_gemv_weight<__bf16>(x, w, w_code, bias, bias_code, d, out_code, M, N, K, stream);
    } else {
        throw std::runtime_error("fp16_packed_linear: unsupported input dtype");
    }
}
