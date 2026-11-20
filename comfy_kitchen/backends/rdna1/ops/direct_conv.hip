// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Direct conv3d over contiguous NCDHW tensors, for the convs the implicit GEMM
// (conv3d_fp16.hip) serves badly: few input channels, which it zero-pads to 32 per tap, and
// few output channels, where a patch row it gathers feeds only a handful of products. Here
// x is read in place in its own dtype, nothing is padded and the math is fp32 FMAs, so the
// result matches an fp32 conv.
//
// A thread owns 8 consecutive output positions of one output row and 8 output channels.
// For each input element of the patch (channel, tap) it loads the 8 input values under its
// positions, one 16- or 32-byte read at stride 1, and multiplies them into its 64
// accumulators against the 8 weights, which the block stages in LDS 32 patch elements at a
// time. The weight is read in torch's [K, C, T, R, S] order, so no re-layout exists either.
//
// Measured on gfx1010 against MIOpen (time ratio, under 1 is faster), 3x3 convs:
//   3 -> 128 over 512x512    fp32 0.42   fp16 0.41      16 -> 512 over 128x128  fp32 1.5   fp16 0.97
//   128 -> 3 over 512x512    fp32 0.93   fp16 0.17      128 -> 3 over 96x96     fp32 1.55  fp16 0.39
// and against the implicit GEMM 2.7-3.4x faster at 128 -> 3. Past about 64 patch elements
// in fp32 its one MAC per FMA loses to MIOpen's kernels, and past 8 output channels with
// 32 or more input channels the implicit GEMM is ahead.
#include "../epilogue.h"

#include <algorithm>

namespace comfy::hip_backend {

struct DirectConvDims {
    int C, K;        // input and output channels
    int D, H, W;     // input extents
    int Z, P, Q;     // output extents
    int T, R, S;     // filter taps
    int sd, sh, sw;  // stride
    int pd, ph, pw;  // zero padding on each side
    int dd, dh, dw;  // dilation
};

constexpr int kDirectStage = 32;  // patch elements staged per LDS round
constexpr int kDirectTapBits = 10;

__forceinline__ __device__ float direct_load(const float* p) { return *p; }
__forceinline__ __device__ float direct_load(const __half* p) { return __half2float(*p); }
__forceinline__ __device__ float direct_load(const __bf16* p) { return static_cast<float>(*p); }

__forceinline__ __device__ void direct_load8(const float* p, float a[8]) {
    const float4 lo = *reinterpret_cast<const float4*>(p), hi = *reinterpret_cast<const float4*>(p + 4);
    a[0] = lo.x, a[1] = lo.y, a[2] = lo.z, a[3] = lo.w, a[4] = hi.x, a[5] = hi.y, a[6] = hi.z, a[7] = hi.w;
}
template <typename T>
__forceinline__ __device__ void direct_load8(const T* p, float a[8]) {
    T h[8];
    *reinterpret_cast<uint4*>(h) = *reinterpret_cast<const uint4*>(p);
#pragma unroll
    for (int j = 0; j < 8; ++j) a[j] = direct_load(h + j);
}

__forceinline__ __device__ void direct_store8(float* p, const float v[8]) {
    *reinterpret_cast<float4*>(p) = make_float4(v[0], v[1], v[2], v[3]);
    *reinterpret_cast<float4*>(p + 4) = make_float4(v[4], v[5], v[6], v[7]);
}
template <typename T>
__forceinline__ __device__ void direct_store8(T* p, const float v[8]) {
    T h[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) h[j] = static_cast<T>(v[j]);
    *reinterpret_cast<uint4*>(p) = *reinterpret_cast<const uint4*>(h);
}

// 256 threads as groups of 8 positions by CG groups of TN output channels by KS slices of
// the patch, each slice taking every KS-th element and the slices' sums added through LDS.
// groups counts the position groups: ceil(Q / 8) per (n, z, p) output row.
template <typename T, int CG, int TN, int KS>
__global__ __launch_bounds__(256) void direct_conv3d_kernel(
    const T* __restrict__ x, const float* __restrict__ w, const float* __restrict__ bias,
    T* __restrict__ out, int groups, DirectConvDims g) {
    constexpr int PG = 256 / (CG * KS), BN = TN * CG, kStage = kDirectStage * KS;
    __shared__ __align__(16) float Ws[kStage * BN];  // weight of patch element i, channel n at [i * BN + n]
    __shared__ int Ko[kStage];  // x offset of each staged patch element from a position's first tap
    __shared__ int Kt[kStage];  // its tap's input offset on each axis, kDirectTapBits apiece
    __shared__ __align__(16) float Red[KS > 1 ? 256 * 8 : 1];  // the slices' sums of one channel

    const int tid = threadIdx.x, slice = tid % KS, cg = tid / KS % CG;
    const int grp = blockIdx.x * PG + tid / (KS * CG);
    const int nb0 = blockIdx.y * BN, n0 = nb0 + cg * TN;
    const bool valid = grp < groups;
    const int qg = (g.Q + 7) / 8;
    const int row = grp / qg, q0 = (grp - row * qg) * 8;
    const int zp = g.Z * g.P;
    const int nb = row / zp, rem = row - nb * zp;
    const int z = rem / g.P, p = rem - z * g.P;
    const int zi0 = z * g.sd - g.pd, pi0 = p * g.sh - g.ph, qi0 = q0 * g.sw - g.pw;
    const int hw = g.H * g.W, dhw = g.D * hw;
    const int taps = g.T * g.R * g.S, kk = g.C * taps;
    // The first tap's input position, which may lie in the padding: only dereferenced
    // after the bounds checks below.
    const T* base = x + static_cast<int64_t>(nb) * g.C * dhw + static_cast<int64_t>(zi0) * hw +
                    static_cast<int64_t>(pi0) * g.W + qi0;

    float acc[8][TN];
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int n = 0; n < TN; ++n) acc[j][n] = 0.0f;

    for (int k0 = 0; k0 < kk; k0 += kStage) {
        if (k0) __syncthreads();  // every thread is done with the previous stage
#pragma unroll
        for (int i = 0; i < BN * kStage / 256; ++i) {
            const int e = tid + i * 256;
            const int k = k0 + e / BN, n = nb0 + e % BN;
            Ws[e] = (n < g.K && k < kk) ? w[static_cast<int64_t>(n) * kk + k] : 0.0f;
        }
        for (int e = tid; e < kStage && k0 + e < kk; e += 256) {
            const int k = k0 + e;
            const int c = k / taps, tap = k - c * taps;
            const int t = tap / (g.R * g.S), rs = tap - t * (g.R * g.S);
            const int r = rs / g.S, s = rs - r * g.S;
            Ko[e] = c * dhw + t * g.dd * hw + r * g.dh * g.W + s * g.dw;
            Kt[e] = t * g.dd | (r * g.dh) << kDirectTapBits | (s * g.dw) << (2 * kDirectTapBits);
        }
        __syncthreads();

        const int kn = min(kStage, kk - k0);
        for (int i = slice; i < kn; i += KS) {
            const int kt = Kt[i];
            constexpr int kMask = (1 << kDirectTapBits) - 1;
            const int zi = zi0 + (kt & kMask), pi = pi0 + ((kt >> kDirectTapBits) & kMask);
            const int qi = qi0 + (kt >> (2 * kDirectTapBits));
            const bool inside = valid && static_cast<unsigned>(zi) < static_cast<unsigned>(g.D) &&
                                static_cast<unsigned>(pi) < static_cast<unsigned>(g.H);
            const T* src = base + Ko[i];
            float a[8];
            if (inside && g.sw == 1 && qi >= 0 && qi + 8 <= g.W) {
                direct_load8(src, a);
            } else {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int qj = qi + j * g.sw;
                    a[j] = (inside && static_cast<unsigned>(qj) < static_cast<unsigned>(g.W))
                               ? direct_load(src + j * g.sw) : 0.0f;
                }
            }
            float wn[TN];
#pragma unroll
            for (int n = 0; n < TN; n += 8) direct_load8(Ws + i * BN + cg * TN + n, wn + n);
#pragma unroll
            for (int j = 0; j < 8; ++j)
#pragma unroll
                for (int n = 0; n < TN; ++n) acc[j][n] = fmaf(a[j], wn[n], acc[j][n]);
        }
    }

    const int64_t zpq = static_cast<int64_t>(zp) * g.Q;
#pragma unroll
    for (int n = 0; n < TN; ++n) {
        const int ch = n0 + n;
        float v[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) v[j] = acc[j][n];
        if constexpr (KS > 1) {
            direct_store8(Red + tid * 8, v);
            __syncthreads();
            if (slice == 0) {
#pragma unroll
                for (int u = 1; u < KS; ++u) {
                    float o[8];
                    direct_load8(Red + (tid + u) * 8, o);
#pragma unroll
                    for (int j = 0; j < 8; ++j) v[j] += o[j];
                }
            }
            __syncthreads();
        }
        if (!valid || slice != 0 || ch >= g.K) continue;
        const float b = bias ? bias[ch] : 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) v[j] += b;
        T* dst = out + (static_cast<int64_t>(nb) * g.K + ch) * zpq + static_cast<int64_t>(rem) * g.Q + q0;
        if (q0 + 8 <= g.Q) {
            direct_store8(dst, v);
        } else {
#pragma unroll
            for (int j = 0; j < 8; ++j)
                if (q0 + j < g.Q) dst[j] = static_cast<T>(v[j]);
        }
    }
}

template <typename T>
void launch_direct_conv3d(const void* x, const float* w, const float* bias, void* out, int groups,
                          const DirectConvDims& g, hipStream_t stream) {
    const auto launch = [&]<int CG, int TN, int KS>() {
        const int pg = 256 / (CG * KS);
        const dim3 grid((groups + pg - 1) / pg, (g.K + TN * CG - 1) / (TN * CG));
        direct_conv3d_kernel<T, CG, TN, KS><<<grid, 256, 0, stream>>>(
            static_cast<const T*>(x), w, bias, static_cast<T*>(out), groups, g);
    };
    // Measured on gfx1010 at 3 -> 128 channels over 512x512: 4 channel groups 1.0 ms, 8 1.6,
    // 16 3.3 (MIOpen 2.4). A conv of at most 8 output channels has one group to fill, and
    // under 8192 position groups too few threads to hide each one's load latency behind
    // the others: 16 slices then ran 512 -> 8 over 64x64 in 0.36 ms against 3.75 unsliced,
    // and 128 -> 3 over 512x512 in 3.2 against 2.1.
    if (g.K > 8) {
        launch.template operator()<4, 8, 1>();
    } else if (groups < 8192) {
        launch.template operator()<1, 8, 16>();
    } else {
        launch.template operator()<1, 8, 1>();
    }
}

}  // namespace comfy::hip_backend

// x [N, C, D, H, W] and out [N, K, Z, P, Q] contiguous in the same dtype (code), w fp32
// [K, C, T, R, S], bias fp32 [K] or null. Returns false, having launched nothing, for a
// shape past the kernel's 32-bit indexing.
extern "C" bool launch_direct_conv3d_kernel(
    const void* x, const void* w, const void* bias, void* out, int code, int N, int C, int K,
    int D, int H, int W, int Z, int P, int Q, int T, int R, int S, int sd, int sh, int sw, int pd,
    int ph, int pw, int dd, int dh, int dw, hipStream_t stream) {
    using namespace comfy::hip_backend;
    const int64_t groups = static_cast<int64_t>(N) * Z * P * ((Q + 7) / 8);
    if (groups == 0 || K == 0) {
        return true;
    }
    constexpr int kTapMax = (1 << kDirectTapBits) - 1;
    if (groups > INT_MAX || static_cast<int64_t>(C) * D * H * W > INT_MAX ||
        static_cast<int64_t>(C) * T * R * S > INT_MAX || (T - 1) * dd > kTapMax ||
        (R - 1) * dh > kTapMax || (S - 1) * dw > kTapMax) {
        return false;
    }
    const DirectConvDims g{C, K, D, H, W, Z, P, Q, T, R, S, sd, sh, sw, pd, ph, pw, dd, dh, dw};
    const auto* wf = static_cast<const float*>(w);
    const auto* bf = static_cast<const float*>(bias);
    const int n = static_cast<int>(groups);
    if (code == kF32) {
        launch_direct_conv3d<float>(x, wf, bf, out, n, g, stream);
    } else if (code == kF16) {
        launch_direct_conv3d<__half>(x, wf, bf, out, n, g, stream);
    } else {
        launch_direct_conv3d<__bf16>(x, wf, bf, out, n, g, stream);
    }
    return true;
}
