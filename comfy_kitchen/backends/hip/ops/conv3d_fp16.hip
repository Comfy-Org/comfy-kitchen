// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// FP16 NDHWC conv3d as an implicit GEMM on the WMMA tile core, with bias and an
// optional residual in the epilogue. On gfx90c and gfx1010 it runs on the thread-level
// packed fp16 GEMM instead (gemm_simt.h), which also zero-pads in the gather. Other
// pre-WMMA devices run a windowed MIOpen conv from the Python layer. The WMMA path takes
// no padding: callers pre-pad (group_norm_pad3d.hip). Row m of the implicit A operand is the input patch under
// output position m, gathered 8 channels at a time straight from x, so no im2col
// buffer exists; B is the weight in its channels_last_3d (K, T, R, S, C) layout and
// the output is written NZPQK. Accumulation is fp32. x may be an NDHWC-ordered view of a
// larger tensor, so a tiled convolution reads its tile in place.
#include "../epilogue.h"
#include "../gemm_simt.h"
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

// The same patch rows for the thread-level GEMM, zero-padded. C % 32 == 0, so each
// 64-byte stage of a row lies within one filter tap: the tap is resolved once per stage
// for the whole block, and a row once per thread, leaving a bounds check and an add
// per chunk.
struct ConvPatchSimtSrc {
    const uint8_t* x;
    int zpq, pq, q_out;
    int C;
    int64_t xw, xh, xd, xn;  // element strides
    int rs, s_taps;
    int sd, sh, sw;
    int pd, ph, pw;
    int D, H, W;

    struct Row {
        int64_t base;  // element offset of tap (0, 0, 0), which may lie in the padding
        int z, p, q;   // its input position; z = D past M, so every tap is outside
    };
    struct Stage {
        int64_t offset;  // element offset of the stage's first channel from a row's base
        int t, r, s;
    };

    __forceinline__ __device__ Row row(int grow, bool valid) const {
        const int n = grow / zpq;
        int rem = grow - n * zpq;
        const int z = rem / pq;
        rem -= z * pq;
        const int p = rem / q_out;
        const int q = rem - p * q_out;
        const int zi = valid ? z * sd - pd : D, pi = p * sh - ph, qi = q * sw - pw;
        return {n * xn + zi * xd + pi * xh + qi * xw, zi, pi, qi};
    }

    __forceinline__ __device__ Stage stage(int kbyte0) const {
        const int e = kbyte0 / 2;
        const int tap = e / C;
        const int t = tap / rs;
        const int r = (tap - t * rs) / s_taps;
        const int s = tap - t * rs - r * s_taps;
        return {t * xd + r * xh + s * xw + (e - tap * C), t, r, s};
    }

    __forceinline__ __device__ uint4 chunk(const Row& w, const Stage& st, int offset) const {
        const bool inside = static_cast<unsigned>(w.z + st.t) < static_cast<unsigned>(D) &&
                            static_cast<unsigned>(w.p + st.r) < static_cast<unsigned>(H) &&
                            static_cast<unsigned>(w.q + st.s) < static_cast<unsigned>(W);
        return inside ? *reinterpret_cast<const uint4*>(x + 2 * (w.base + st.offset) + offset)
                      : make_uint4(0, 0, 0, 0);
    }
};

// fp16 output of the packed GEMM: out = acc * acc_scale[0] + bias[col] (+ resid[row, col]).
// acc_scale undoes the power of two the caller scaled the weight by.
struct EpiConvScaled {
    const float* acc_scale;
    const __half* bias;
    const __half* resid;
    int resid_stride;
    float scale;

    __forceinline__ __device__ void init() { scale = acc_scale[0]; }

    __forceinline__ __device__ float operator()(int row, int col, float acc) const {
        float v = acc * scale;
        if (bias) v += __half2float(bias[col]);
        if (resid) v += __half2float(resid[static_cast<int64_t>(row) * resid_stride + col]);
        return v;
    }
};

}  // namespace comfy::hip_backend

// resid is the full NZPQK output-shaped residual, or null. xs_* are the element strides of
// x; xs_h == 0 means packed NDHWC. acc_scale (one float, see EpiConvScaled) selects the
// SIMT path, on the devices that have it, and is required there; pd/ph/pw zero-pad each
// side, on that path only. D, H and W are x's extents before padding.
extern "C" bool launch_fp16_conv3d_kernel(
    const void* x, const void* w, const void* bias, const void* resid, void* out, int N, int D,
    int H, int W, int C, int K, int T, int R, int S, int Z, int P, int Q, int sd, int sh, int sw,
    int xs_w, int xs_h, int xs_d, int xs_n, int pd, int ph, int pw, const void* acc_scale,
    hipStream_t stream) {

    using namespace comfy::hip_backend;

    if (C % 8 != 0 || K % 8 != 0 || sd <= 0 || sh <= 0 || sw <= 0) {
        return false;
    }
    const bool simt = acc_scale != nullptr;
    if (simt ? (C % 32 != 0 || !device_uses_simt_gemm()) : (pd || ph || pw)) {
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

    const bool packed = xs_h == 0;
    const int64_t xw = packed ? C : xs_w;
    const int64_t xh = packed ? static_cast<int64_t>(W) * C : xs_h;
    const int64_t xd = packed ? static_cast<int64_t>(H) * W * C : xs_d;
    const int64_t xn = packed ? static_cast<int64_t>(D) * H * W * C : xs_n;
    if (simt) {
        const ConvPatchSimtSrc src{static_cast<const uint8_t*>(x), Z * P * Q, P * Q, Q, C,
                                   xw, xh, xd, xn, R * S, S, sd, sh, sw, pd, ph, pw, D, H, W};
        const EpiConvScaled epi{static_cast<const float*>(acc_scale),
                                static_cast<const __half*>(bias),
                                static_cast<const __half*>(resid), K, 1.0f};
        launch_gemm_simt<SimtF16Packed, EpiConvScaled, __half, ConvPatchSimtSrc>(
            src, static_cast<const uint8_t*>(w), static_cast<__half*>(out), M, K,
            static_cast<int>(2 * depth), K, epi, stream);
        return true;
    }

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

    const ConvPatchSrc src{static_cast<const uint8_t*>(x), Z * P * Q, P * Q, Q, C, xw, xh, xd, xn,
                           R * S, S, sd, sh, sw};
    EpiFp16 epi{static_cast<const __half*>(bias), nullptr, static_cast<const __half*>(resid), K};
    launch_gemm_wmma<MmaF16, EpiFp16, __half, ConvPatchSrc>(
        src, static_cast<const uint8_t*>(w), static_cast<__half*>(out), M, K,
        static_cast<int>(2 * depth), K, epi, stream);
    return true;
}
