// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// AWQ W4A16 GEMV: 4-bit weight, bf16/fp16 activation, per-group asymmetric.
//
//   W[n, k] = (q[n, k] - 8) * wscale[k / G, n] + wzero[k / G, n]
//   y       = x @ W^T + bias
//
// Small-M shapes use one wave per output column. Larger M tiles reuse each
// dequantized weight group across 16 rows and accumulate with WMMA.
#include <stdexcept>
#include <string>

#include "../epilogue.h"
#include "../mma.h"

namespace comfy::hip_backend {

constexpr int kAwqWavesPerBlock = 8;
constexpr int kAwqMmaMThreshold = 8;
constexpr int kAwqGroupSize = 64;

template <typename Mma, typename OutT>
__global__ __launch_bounds__(128) void gemm_awq_w4a16_kernel(
    const typename Mma::Elem* __restrict__ x, const int8_t* __restrict__ qweight,
    const void* __restrict__ wscales, const void* __restrict__ wzeros,
    const void* __restrict__ bias, OutT* __restrict__ out,
    int M, int N, int K, int s_code, int bias_code) {

    using Elem = typename Mma::Elem;
    constexpr int BM = 16;
    constexpr int BN = 128;
    constexpr int BK = kAwqGroupSize;
    // Pad each LDS row by eight elements (16 bytes).
    constexpr int kStride = BK + 8;
    constexpr int kNTiles = 2;

    __shared__ __align__(16) Elem xs[BM * kStride];
    __shared__ __align__(16) Elem ws[BN * kStride];

    const int tid = threadIdx.x;
    const int lane = tid % kWave;
    const int warp = tid / kWave;
    const int m0 = blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;
    const int row = frag_row(lane);

    typename Mma::Acc acc[kNTiles];
    #pragma unroll
    for (int j = 0; j < kNTiles; ++j) acc[j] = Mma::zero();

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Each thread copies one aligned 16-byte activation chunk. The padded
        // LDS row stride keeps successive WMMA rows on different banks.
        const int xv = tid;
        const int xm = xv / (BK / 8);
        const int xk = (xv % (BK / 8)) * 8;
        uint4* xdst = reinterpret_cast<uint4*>(xs + xm * kStride + xk);
        if (m0 + xm < M) {
            *xdst = *reinterpret_cast<const uint4*>(
                x + static_cast<int64_t>(m0 + xm) * K + k0 + xk);
        } else {
            *xdst = make_uint4(0, 0, 0, 0);
        }

        // One thread owns one output column and dequantizes its 64-value weight
        // group directly into the LDS tile shared by the four compute waves.
        const int n = n0 + tid;
        if (n < N) {
            const int g = k0 / BK;
            const float scale = load_scalar(wscales, s_code, static_cast<int64_t>(g) * N + n);
            const float zero = load_scalar(wzeros, s_code, static_cast<int64_t>(g) * N + n);
            const uint8_t* src = reinterpret_cast<const uint8_t*>(qweight) +
                                 static_cast<int64_t>(n) * (K / 2) + k0 / 2;
            Elem* dst = ws + tid * kStride;
            #pragma unroll
            for (int i = 0; i < BK / 2; ++i) {
                const uint8_t packed = src[i];
                dst[2 * i] =
                    static_cast<Elem>((static_cast<int>(packed & 0x0f) - 8) * scale + zero);
                dst[2 * i + 1] =
                    static_cast<Elem>((static_cast<int>(packed >> 4) - 8) * scale + zero);
            }
        } else {
            Elem* dst = ws + tid * kStride;
            #pragma unroll
            for (int i = 0; i < BK; ++i) dst[i] = static_cast<Elem>(0.0f);
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            const typename Mma::Frag af = load_frag_16bit<Mma>(xs + row * kStride + kk, lane);
            #pragma unroll
            for (int j = 0; j < kNTiles; ++j) {
                const int wn = warp * (kNTiles * 16) + j * 16;
                const typename Mma::Frag bf =
                    load_frag_16bit<Mma>(ws + (wn + row) * kStride + kk, lane);
                acc[j] = Mma::mma(af, bf, acc[j]);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int j = 0; j < kNTiles; ++j) {
        const int col = n0 + warp * (kNTiles * 16) + j * 16 + acc_col(lane);
        if (col >= N) continue;
        const float b = bias != nullptr ? load_scalar(bias, bias_code, col) : 0.0f;
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int m = m0 + acc_row(lane, e);
            if (m < M) {
                out[static_cast<int64_t>(m) * N + col] =
                    static_cast<OutT>(Mma::get(acc[j], e) + b);
            }
        }
    }
}

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
    int x_code, int s_code, int bias_code, int out_code, bool use_wmma,
    hipStream_t stream) {

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

    // The fused path converts the weights to the scale dtype before the MMA,
    // matching the eager and CUDA implementations. Keep mixed-dtype operands
    // on the scalar path, which accumulates their original values in fp32.
    if (use_wmma && M > kAwqMmaMThreshold && group_size == kAwqGroupSize &&
        x_code == s_code && out_code == s_code) {
        dim3 block(128);
        dim3 grid((N + 127) / 128, (M + 15) / 16);
        if (out_code == kBF16) {
            gemm_awq_w4a16_kernel<MmaBf16, __bf16><<<grid, block, 0, stream>>>(
                static_cast<const __bf16*>(x), static_cast<const int8_t*>(qweight), wscales,
                wzeros, bias, static_cast<__bf16*>(out), M, N, K, s_code, bias_code);
            return;
        }
        if (out_code == kF16) {
            gemm_awq_w4a16_kernel<MmaF16, _Float16><<<grid, block, 0, stream>>>(
                static_cast<const _Float16*>(x), static_cast<const int8_t*>(qweight), wscales,
                wzeros, bias, static_cast<_Float16*>(out), M, N, K, s_code, bias_code);
            return;
        }
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
