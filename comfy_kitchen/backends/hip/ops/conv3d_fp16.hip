// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// FP16 NDHWC conv3d as an implicit GEMM on the WMMA tile core, with bias and an
// optional residual in the epilogue. Zero padding only: callers pre-pad
// (group_norm_pad3d.hip). Row m of the implicit A operand is the input patch under
// output position m, gathered 8 channels at a time straight from x, so no im2col
// buffer exists; B is the weight in its channels_last_3d (K, T, R, S, C) layout and
// the output is written NZPQK. Accumulation is fp32. x may be an NDHWC-ordered view of a
// larger tensor, so a tiled convolution reads its tile in place.
#include "../epilogue.h"
#include "../gemm_wmma.h"

namespace comfy::hip_backend {

// One 16-byte chunk (8 fp16 channels) of the patch row under output position `row`.
struct ConvPatchSrc {
    const uint8_t* x;
    int zpq, pq, q_out;
    int C;
    int64_t xw, xh, xd, xn;  // element strides
    int rs, s_taps;
    int sd, sh, sw;

    __forceinline__ __device__ uint4 chunk(int row, int kbyte) const {
        const int n = row / zpq;
        int rem = row - n * zpq;
        const int z = rem / pq;
        rem -= z * pq;
        const int p = rem / q_out;
        const int q = rem - p * q_out;

        const int e = kbyte / 2;
        const int tap = e / C;
        const int c0 = e - tap * C;
        const int t = tap / rs;
        const int r = (tap - t * rs) / s_taps;
        const int s = tap - t * rs - r * s_taps;

        const int64_t idx = n * xn + (z * sd + t) * xd + (p * sh + r) * xh + (q * sw + s) * xw + c0;
        return *reinterpret_cast<const uint4*>(x + 2 * idx);
    }
};

}  // namespace comfy::hip_backend

// resid is the full NZPQK output-shaped residual, or null. xs_* are the element strides of
// x; xs_h == 0 means packed NDHWC.
extern "C" bool launch_fp16_conv3d_kernel(
    const void* x, const void* w, const void* bias, const void* resid, void* out, int N, int D,
    int H, int W, int C, int K, int T, int R, int S, int Z, int P, int Q, int sd, int sh, int sw,
    int xs_w, int xs_h, int xs_d, int xs_n, hipStream_t stream) {

    using namespace comfy::hip_backend;

    if (C % 8 != 0 || K % 8 != 0 || sd <= 0 || sh <= 0 || sw <= 0) {
        return false;
    }
    const int64_t m64 = static_cast<int64_t>(N) * Z * P * Q;
    const int64_t depth = static_cast<int64_t>(C) * T * R * S;
    if (m64 == 0 || K == 0) {
        return true;
    }
    if (m64 > 0x7fffffff || 2 * depth > 0x7fffffff) {
        return false;
    }
    const int M = static_cast<int>(m64);

    // CUDA's envelope, in its own tile units: launches too small to fill the device
    // run the framework conv, deep small launches are taken down to 16 of its 64x64
    // tiles, and past a reduction depth of 16384 only those small launches are.
    auto tiles = [&](int64_t tm, int64_t tn) { return ((M + tm - 1) / tm) * ((K + tn - 1) / tn); };
    const bool small_launch = tiles(128, 128) < 128;
    const bool served = depth > 16384 || (small_launch && depth > 8192)
                            ? small_launch && tiles(64, 64) >= 16
                            : !small_launch;
    if (!served) {
        return false;
    }

    const bool packed = xs_h == 0;
    const int64_t xw = packed ? C : xs_w;
    const int64_t xh = packed ? static_cast<int64_t>(W) * C : xs_h;
    const int64_t xd = packed ? static_cast<int64_t>(H) * W * C : xs_d;
    const int64_t xn = packed ? static_cast<int64_t>(D) * H * W * C : xs_n;
    const ConvPatchSrc src{static_cast<const uint8_t*>(x), Z * P * Q, P * Q, Q, C, xw, xh, xd, xn,
                           R * S, S, sd, sh, sw};
    EpiFp16 epi{static_cast<const __half*>(bias), nullptr, static_cast<const __half*>(resid), K};
    launch_gemm_wmma<MmaF16, EpiFp16, __half, ConvPatchSrc>(
        src, static_cast<const uint8_t*>(w), static_cast<__half*>(out), M, K,
        static_cast<int>(2 * depth), K, epi, stream);
    return true;
}
