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
//
// On the 6-WGP 780M this is 6x-11x faster than MIOpen's conv3d on the same shapes
// (9.5 ms against 80.6 ms at C=128 256x256, 6.10 TOPS against 0.72), because
// MIOpen has no optimized gfx1103 kernel for the grouped case. The gather is
// instruction-bound: the inner loop issues its WMMA in about one percent of the
// slots, and the staging path's six integer divisions by launch-time constants
// are 43% of that path's instructions (RDNA has no integer divider, so each one
// is a float-reciprocal sequence of about twenty).
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
    // The envelope sends a small launch to the framework's conv, which is the
    // right call where the framework's conv is good. On this part it is not:
    // MIOpen has no optimized gfx1103 kernel for the grouped case and falls
    // back to one that runs at 0.7-1.3 TOPS, while the kernel here runs at
    // 1.1-8 TOPS on the same shapes. The VAE's own convolutions are the small
    // ones this drops -- C=128 or 256 at 64x64 is 32 or 64 tiles of 128x128 --
    // so they ran MIOpen's fallback at four to eleven times the cost.
    //
    // The tile choice is not this launcher's to make: the GEMM's own policy
    // already takes the narrow tile for a launch this small (see the padding
    // waste test in gemm_wmma.h), and the shapes that reach here keep the
    // kernel's numbers.
    if (!served && !comfy_small_igpu()) {
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
    // `kbytes` here is 2 * depth -- a **patch volume** (output positions times the
    // filter extent), not a K -- so the heuristic's K bands, which were calibrated
    // on the plain GEMMs, do not transfer to this caller's units. Declining
    // kstep32 keeps that from silently re-banding a conv by a quantity it was
    // never measured in. See GemmPolicy in gemm_wmma.h.
    //
    // Measured to be a no-op either way, not just precaution: depth is at least
    // Z*P*Q*R*S, so kbytes is far above both band edges for every conv shape in
    // tests/test_fp16_conv3d.py, and the kstep32 arm's `else` (deep K) selects the
    // same 128x128 BKB128 4x4 grid the plain path does. The shapes where a conv
    // *does* diverge (c=k=320, 32^3, 3x3x3: 25.5 ms) do so through the pre-existing
    // `padding_waste` -> skinny test, which is evaluated before any tile branch,
    // so no value of this flag reaches them. Seven conv shapes, interleaved,
    // identical with kstep32 true and false.
    launch_gemm_wmma<MmaF16, EpiFp16, __half, ConvPatchSrc>(
        src, static_cast<const uint8_t*>(w), static_cast<__half*>(out), M, K,
        static_cast<int>(2 * depth), K, epi, stream, GemmPolicy{false, false});
    return true;
}
