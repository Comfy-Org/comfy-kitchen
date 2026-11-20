// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Depthwise conv3d and its transpose (groups == channels, one filter per channel) over
// contiguous NCDHW tensors of any float dtype, accumulated in fp32. A depthwise conv has
// no GEMM in it: each output element reads T * R * S taps of its own channel, so this is a
// plain gather with one thread per output. It covers the anti-aliasing FIR filters of the
// audio vocoders, which MIOpen serves through its generic depthwise and transposed paths.
#include "../epilogue.h"

namespace comfy::hip_backend {

struct DepthwiseDims {
    int C;
    int D, H, W;     // input extents
    int Z, P, Q;     // output extents
    int T, R, S;     // filter taps
    int sd, sh, sw;  // stride
    int pd, ph, pw;  // zero padding on each side
    int dd, dh, dw;  // dilation (1 for the transpose)
};

// One block covers up to 256 output positions of one (n, c, z, p) row: grid.x runs along Q
// and grid.y/z over the rows, so a thread finds its position with 32-bit arithmetic.
struct DepthwisePos {
    int q, p, z, c;
    int64_t x_base, w_base, out_index;
    bool valid;
};

__forceinline__ __device__ DepthwisePos depthwise_pos(const DepthwiseDims& g, int rows) {
    DepthwisePos o;
    o.q = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y + blockIdx.z * gridDim.y;
    o.valid = o.q < g.Q && row < rows;
    o.p = row % g.P;
    const int zc = row / g.P;
    o.z = zc % g.Z;
    const int nc = zc / g.Z;
    o.c = nc % g.C;
    o.x_base = static_cast<int64_t>(nc) * g.D * g.H * g.W;
    o.w_base = static_cast<int64_t>(o.c) * g.T * g.R * g.S;
    o.out_index = static_cast<int64_t>(row) * g.Q + o.q;
    return o;
}

// Forward: out[z, p, q] = sum_t w[t] * x[z * s - pad + t * dil].
__global__ void __launch_bounds__(256)
depthwise_conv3d_kernel(const void* __restrict__ x, int x_code, const void* __restrict__ w,
                        int w_code, const void* __restrict__ bias, int bias_code,
                        void* __restrict__ out, int out_code, int rows, DepthwiseDims g) {
    const DepthwisePos o = depthwise_pos(g, rows);
    if (!o.valid) return;
    const int z0 = o.z * g.sd - g.pd, p0 = o.p * g.sh - g.ph, q0 = o.q * g.sw - g.pw;
    float acc = bias ? load_scalar(bias, bias_code, o.c) : 0.0f;
    for (int t = 0; t < g.T; ++t) {
        const int zi = z0 + t * g.dd;
        if (static_cast<unsigned>(zi) >= static_cast<unsigned>(g.D)) continue;
        for (int r = 0; r < g.R; ++r) {
            const int pi = p0 + r * g.dh;
            if (static_cast<unsigned>(pi) >= static_cast<unsigned>(g.H)) continue;
            const int64_t row = o.x_base + (static_cast<int64_t>(zi) * g.H + pi) * g.W;
            const int64_t wrow = o.w_base + (t * g.R + r) * g.S;
            for (int s = 0; s < g.S; ++s) {
                const int qi = q0 + s * g.dw;
                if (static_cast<unsigned>(qi) < static_cast<unsigned>(g.W)) {
                    acc = fmaf(load_scalar(w, w_code, wrow + s), load_scalar(x, x_code, row + qi), acc);
                }
            }
        }
    }
    store_scalar(out, out_code, o.out_index, acc);
}

// Transpose: out[o] = sum over taps j with (o + pad - j) % s == 0 of w[j] * x[(o + pad - j) / s],
// so each axis steps its taps by the stride from (o + pad) % s.
__global__ void __launch_bounds__(256)
depthwise_conv_transpose3d_kernel(const void* __restrict__ x, int x_code,
                                  const void* __restrict__ w, int w_code,
                                  const void* __restrict__ bias, int bias_code,
                                  void* __restrict__ out, int out_code, int rows,
                                  DepthwiseDims g) {
    const DepthwisePos o = depthwise_pos(g, rows);
    if (!o.valid) return;
    const int zn = o.z + g.pd, pn = o.p + g.ph, qn = o.q + g.pw;
    float acc = bias ? load_scalar(bias, bias_code, o.c) : 0.0f;
    for (int t = zn % g.sd; t < g.T && t <= zn; t += g.sd) {
        const int zi = (zn - t) / g.sd;
        if (zi >= g.D) continue;
        for (int r = pn % g.sh; r < g.R && r <= pn; r += g.sh) {
            const int pi = (pn - r) / g.sh;
            if (pi >= g.H) continue;
            const int64_t row = o.x_base + (static_cast<int64_t>(zi) * g.H + pi) * g.W;
            const int64_t wrow = o.w_base + (t * g.R + r) * g.S;
            // the first tap's input position, stepping down by one per tap
            int s = qn % g.sw;
            for (int qi = (qn - s) / g.sw; s < g.S && qi >= 0; s += g.sw, --qi) {
                if (qi < g.W) {
                    acc = fmaf(load_scalar(w, w_code, wrow + s), load_scalar(x, x_code, row + qi), acc);
                }
            }
        }
    }
    store_scalar(out, out_code, o.out_index, acc);
}

}  // namespace comfy::hip_backend

// x [N, C, D, H, W], w [C, 1, T, R, S], bias [C] or null, out [N, C, Z, P, Q], all contiguous.
extern "C" void launch_depthwise_conv3d_kernel(
    const void* x, int x_code, const void* w, int w_code, const void* bias, int bias_code,
    void* out, int out_code, int N, int C, int D, int H, int W, int Z, int P, int Q, int T, int R,
    int S, int sd, int sh, int sw, int pd, int ph, int pw, int dd, int dh, int dw, bool transpose,
    hipStream_t stream) {
    using namespace comfy::hip_backend;
    const int64_t rows64 = static_cast<int64_t>(N) * C * Z * P;
    if (rows64 == 0 || Q == 0) {
        return;
    }
    const int rows = static_cast<int>(rows64);
    const DepthwiseDims g{C, D, H, W, Z, P, Q, T, R, S, sd, sh, sw, pd, ph, pw, dd, dh, dw};
    const int bx = Q >= 256 ? 256 : Q > 64 ? 128 : Q > 32 ? 64 : 32;
    const int gy = std::min(rows, 32768);
    const dim3 grid((Q + bx - 1) / bx, gy, (rows + gy - 1) / gy);
    const dim3 block(bx);
    if (transpose) {
        hipLaunchKernelGGL(depthwise_conv_transpose3d_kernel, grid, block, 0, stream, x, x_code, w,
                           w_code, bias, bias_code, out, out_code, rows, g);
    } else {
        hipLaunchKernelGGL(depthwise_conv3d_kernel, grid, block, 0, stream, x, x_code, w, w_code,
                           bias, bias_code, out, out_code, rows, g);
    }
}
