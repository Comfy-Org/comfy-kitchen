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

// The same patch rows for the thread-level GEMM, zero-padded and dilated. C % 32 == 0, so each
// 64-byte stage of a row lies within one filter tap: the tap is resolved once per stage
// for the whole block, and a row once per thread, leaving a bounds check and an add
// per chunk.
//
// Row m's K axis runs over taps (t, r, s) in order with C channels each, the same order
// as the weight's (K, T, R, S, C) layout, so A row m dotted with B row k is output
// channel k at output position m. Padding is never materialized: a tap that lands in the
// padding reads as zeros, which is what a zero-padded input would hold there.
struct ConvPatchSimtSrc {
    const uint8_t* x;
    int zpq, pq, q_out;
    int C;
    int64_t xw, xh, xd, xn;  // element strides
    int rs, s_taps;
    int sd, sh, sw;
    int pd, ph, pw;
    int D, H, W;
    int dd, dh, dw;

    // base may point before x or past its end when the tap lies in the padding; it is
    // only dereferenced by chunk() after the bounds check, so that is never read.
    struct Row {
        int64_t base;  // element offset of tap (0, 0, 0), which may lie in the padding
        int z, p, q;   // its input position; z = D past M, so every tap is outside
    };
    struct Stage {
        int64_t offset;  // element offset of the stage's first channel from a row's base
        int t, r, s;     // the tap's input offset from the row's position, dilation applied
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
        return {t * dd * xd + r * dh * xh + s * dw * xw + (e - tap * C), t * dd, r * dh, s * dw};
    }

    // The casts to unsigned fold "0 <= i && i < extent" into one compare per axis: a
    // negative position (in the leading padding) wraps to a huge value.
    __forceinline__ __device__ uint4 chunk(const Row& w, const Stage& st, int offset) const {
        const bool inside = static_cast<unsigned>(w.z + st.t) < static_cast<unsigned>(D) &&
                            static_cast<unsigned>(w.p + st.r) < static_cast<unsigned>(H) &&
                            static_cast<unsigned>(w.q + st.s) < static_cast<unsigned>(W);
        return inside ? *reinterpret_cast<const uint4*>(x + 2 * (w.base + st.offset) + offset)
                      : make_uint4(0, 0, 0, 0);
    }
};

// Output of the packed GEMM: out = acc * acc_scale[0] + bias[col] (+ resid[row, col]), with
// bias, resid and out all of dtype `code`, and resid in out's layout: NZPQK, or NCZPQ when
// out_s (Z * P * Q) is set, so a channels-first caller needs no transpose of the result.
// acc_scale undoes the power of two the caller scaled one operand by: the weight when x is
// fp16, since the implicit GEMM reads x in place and a per-row scaled copy of x would be the
// im2col-sized buffer this path exists to avoid, and x when the caller converts a wider x
// into an fp16 copy anyway. One scale for the whole conv, from both operands' absmax, gives
// the same fp16 overflow bound as gemm_f16_packed.hip's per-row one.
struct EpiConvScaled {
    const float* acc_scale;
    const void* bias;
    const void* resid;
    int k;
    int out_s;
    int code;
    float scale;

    __forceinline__ __device__ void init() { scale = acc_scale[0]; }

    __forceinline__ __device__ int64_t out_offset(int row, int col) const {
        if (!out_s) return static_cast<int64_t>(row) * k + col;
        const int n = row / out_s;
        return (static_cast<int64_t>(n) * k + col) * out_s + (row - n * out_s);
    }

    __forceinline__ __device__ float operator()(int row, int col, float acc) const {
        float v = acc * scale;
        if (bias) v += load_scalar(bias, code, col);
        if (resid) v += load_scalar(resid, code, out_offset(row, col));
        return v;
    }
};

}  // namespace comfy::hip_backend

// resid is the full NZPQK output-shaped residual, or null. xs_* are the element strides of
// x; xs_h == 0 means packed NDHWC. acc_scale (one float, see EpiConvScaled) selects the
// SIMT path, on the devices that have it, and is required there; pd/ph/pw zero-pad each
// side, dd/dh/dw dilate the filter, out_code picks an fp32, fp16 or bf16 out (bias and
// resid share its dtype), and out_ncdhw lays out and resid as NCZPQ rather than NZPQK, on
// that path only. D, H and W are x's extents before padding.
extern "C" bool launch_fp16_conv3d_kernel(
    const void* x, const void* w, const void* bias, const void* resid, void* out, int N, int D,
    int H, int W, int C, int K, int T, int R, int S, int Z, int P, int Q, int sd, int sh, int sw,
    int xs_w, int xs_h, int xs_d, int xs_n, int pd, int ph, int pw, int dd, int dh, int dw,
    const void* acc_scale, int out_code, bool out_ncdhw, hipStream_t stream) {

    using namespace comfy::hip_backend;

    if (C % 8 != 0 || K % 8 != 0 || sd <= 0 || sh <= 0 || sw <= 0) {
        return false;
    }
    const bool simt = acc_scale != nullptr;
    if (simt ? (C % 32 != 0 || !device_uses_simt_gemm())
             : (pd || ph || pw || dd != 1 || dh != 1 || dw != 1 || out_code != kF16 ||
                out_ncdhw)) {
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
                                   xw, xh, xd, xn, R * S, S, sd, sh, sw, pd, ph, pw, D, H, W,
                                   dd, dh, dw};
        const EpiConvScaled epi{static_cast<const float*>(acc_scale), bias, resid, K,
                                out_ncdhw ? Z * P * Q : 0, out_code, 1.0f};
        const auto* b = static_cast<const uint8_t*>(w);
        const int kbytes = static_cast<int>(2 * depth);
        if (out_code == kF32) {
            launch_gemm_simt<SimtF16Packed, EpiConvScaled, float, ConvPatchSimtSrc>(
                src, b, static_cast<float*>(out), M, K, kbytes, K, epi, stream);
        } else if (out_code == kBF16) {
            launch_gemm_simt<SimtF16Packed, EpiConvScaled, __bf16, ConvPatchSimtSrc>(
                src, b, static_cast<__bf16*>(out), M, K, kbytes, K, epi, stream);
        } else {
            launch_gemm_simt<SimtF16Packed, EpiConvScaled, __half, ConvPatchSimtSrc>(
                src, b, static_cast<__half*>(out), M, K, kbytes, K, epi, stream);
        }
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
