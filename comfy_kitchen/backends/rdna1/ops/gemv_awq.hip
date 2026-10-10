// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// AWQ W4A16 GEMV: 4-bit weight, bf16/fp16 activation, per-group asymmetric.
//
//   W[n, k] = (q[n, k] - 8) * wscale[k / G, n] + wzero[k / G, n]
//   y       = x @ W^T + bias
//
// Small-batch shapes are bandwidth bound on streaming the weight and offer no
// operand reuse for a WMMA tile, so one wave reduces one output column.
#include <stdexcept>
#include <string>

#include "../epilogue.h"
#include "../mma.h"

namespace comfy::hip_backend {

constexpr int kAwqWavesPerBlock = 8;

template <typename OutT>
__global__ __launch_bounds__(kWave* kAwqWavesPerBlock) void gemv_awq_kernel(
    const void* __restrict__ x, const int8_t* __restrict__ qweight,
    const void* __restrict__ wscales, const void* __restrict__ wzeros,
    const void* __restrict__ bias, OutT* __restrict__ out,
    int M, int N, int K, int group_size, int x_code, int s_code, int bias_code) {

    const int lane = threadIdx.x;
    const int n = blockIdx.x * kAwqWavesPerBlock + threadIdx.y;
    const int m = blockIdx.y;
    if (n >= N || m >= M) return;

    const int8_t* wrow = qweight + static_cast<int64_t>(n) * (K / 2);

    float sum = 0.0f;
    // 4 packed bytes = 8 weights. group_size is a multiple of 8, so a chunk
    // never straddles a group boundary and scale/zero are looked up per chunk.
    for (int k = lane * 8; k < K; k += kWave * 8) {
        const int g = k / group_size;
        const float s = load_scalar(wscales, s_code, g * N + n);
        const float z = load_scalar(wzeros, s_code, g * N + n);

        const uchar4 packed = *reinterpret_cast<const uchar4*>(wrow + k / 2);
        const uint8_t* pb = reinterpret_cast<const uint8_t*>(&packed);

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int lo = pb[i] & 0x0F;
            const int hi = (pb[i] >> 4) & 0x0F;
            const float x0 = load_scalar(x, x_code, static_cast<int64_t>(m) * K + k + 2 * i);
            const float x1 = load_scalar(x, x_code, static_cast<int64_t>(m) * K + k + 2 * i + 1);
            sum += x0 * ((lo - 8.0f) * s + z);
            sum += x1 * ((hi - 8.0f) * s + z);
        }
    }

    sum = wave_reduce_sum(sum);

    if (lane == 0) {
        if (bias != nullptr) sum += load_scalar(bias, bias_code, n);
        out[static_cast<int64_t>(m) * N + n] = static_cast<OutT>(sum);
    }
}

}  // namespace comfy::hip_backend

extern "C" void launch_gemv_awq_kernel(
    const void* x, const void* qweight, const void* wscales, const void* wzeros,
    const void* bias, void* out, int M, int N, int K, int group_size,
    int x_code, int s_code, int bias_code, int out_code, hipStream_t stream) {

    using namespace comfy::hip_backend;

    if (M == 0 || N == 0) {
        return;  // a zero-dimension grid is hipErrorInvalidConfiguration
    }

    // The kernel consumes 8 weights (4 packed bytes) per step and indexes the
    // scale/zero rows as k / group_size. A group that is not a whole number of
    // those chunks would straddle the boundary and pick up the wrong scale; a K
    // that is not would read past the end of the packed row.
    if (group_size <= 0 || group_size % 8 != 0) {
        throw std::runtime_error("gemv_awq_w4a16: group_size must be a positive multiple of 8, got " +
                                 std::to_string(group_size));
    }
    if (K % 8 != 0) {
        throw std::runtime_error("gemv_awq_w4a16: K must be a multiple of 8, got " +
                                 std::to_string(K));
    }
    if (K % group_size != 0) {
        throw std::runtime_error("gemv_awq_w4a16: K=" + std::to_string(K) +
                                 " must be a multiple of group_size=" + std::to_string(group_size));
    }

    dim3 block(kWave, kAwqWavesPerBlock);
    dim3 grid((N + kAwqWavesPerBlock - 1) / kAwqWavesPerBlock, M);
    const int8_t* q = static_cast<const int8_t*>(qweight);

#define CK_AWQ_LAUNCH(T)                                                                     \
    gemv_awq_kernel<T><<<grid, block, 0, stream>>>(x, q, wscales, wzeros, bias,               \
                                                   static_cast<T*>(out), M, N, K, group_size, \
                                                   x_code, s_code, bias_code)

    if (out_code == kBF16) {
        CK_AWQ_LAUNCH(__bf16);
    } else if (out_code == kF16) {
        CK_AWQ_LAUNCH(__half);
    } else if (out_code == kF32) {
        CK_AWQ_LAUNCH(float);
    } else {
        throw std::runtime_error("gemv_awq_w4a16: unsupported output dtype");
    }
#undef CK_AWQ_LAUNCH
}
