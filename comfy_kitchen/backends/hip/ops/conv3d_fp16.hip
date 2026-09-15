// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// FP16 NDHWC conv3d as an implicit GEMM on the WMMA tile core, with bias and an
// optional residual in the epilogue. Zero padding only: callers pre-pad
// (group_norm_pad3d.hip). Row m of the implicit A operand is the input patch under
// output position m, gathered 8 channels at a time straight from x, so no im2col
// buffer exists; B is the weight in its channels_last_3d (K, T, R, S, C) layout and
// the output is written NZPQK. Accumulation is fp32.
#include "../epilogue.h"
#include "../gemm_wmma.h"

namespace comfy::hip_backend {

// One 16-byte chunk (8 fp16 channels) of the patch row under output position `row`.
struct ConvPatchSrc {
    const uint8_t* x;
    int zpq, pq, q_out;
    int D, H, W, C;
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

        const int64_t idx =
            (((static_cast<int64_t>(n) * D + z * sd + t) * H + p * sh + r) * W + q * sw + s) * C +
            c0;
        return *reinterpret_cast<const uint4*>(x + 2 * idx);
    }
};

}  // namespace comfy::hip_backend

// resid is the full NZPQK output-shaped residual, or null.
extern "C" bool launch_fp16_conv3d_kernel(
    const void* x, const void* w, const void* bias, const void* resid, void* out, int N, int D,
    int H, int W, int C, int K, int T, int R, int S, int Z, int P, int Q, int sd, int sh, int sw,
    hipStream_t stream) {

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
    // run the framework conv, and past a reduction depth of 8192 only small launches
    // are taken.
    auto tiles = [&](int64_t tm, int64_t tn) { return ((M + tm - 1) / tm) * ((K + tn - 1) / tn); };
    const bool served = depth > 8192
                            ? tiles(128, 128) < 128 && tiles(64, 64) >= 16
                            : (K >= 256 && tiles(128, 256) >= 128) || tiles(128, 128) >= 128;
    if (!served) {
        return false;
    }

    const ConvPatchSrc src{static_cast<const uint8_t*>(x), Z * P * Q, P * Q, Q, D, H, W, C,
                           R * S, S, sd, sh, sw};
    EpiFp16 epi{static_cast<const __half*>(bias), nullptr, static_cast<const __half*>(resid), K};
    launch_gemm_wmma<MmaF16, EpiFp16, __half, ConvPatchSrc>(
        src, static_cast<const uint8_t*>(w), static_cast<__half*>(out), M, K,
        static_cast<int>(2 * depth), K, epi, stream);
    return true;
}
