// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// RoPE. x is (batch, dim1, dim2, head_dim); freqs_cis is
// (fb, fd1, fd2, head_dim/2, 2, 2), broadcasting over any of its leading dims.
//
// For pair k:
//   out[a] = f[k][0][0] * x[a] + f[k][0][1] * x[b]
//   out[b] = f[k][1][0] * x[a] + f[k][1][1] * x[b]
// with (a, b) = (2k, 2k+1) for the interleaved layout and (k, k + head_dim/2)
// for the split-half layout. The rotation is evaluated in fp32 regardless of the
// storage dtype, matching the eager reference, which upcasts to the freqs dtype.
#include <stdexcept>

#include "../hadamard.h"  // load_in / dtype codes

namespace comfy::hip_backend {

constexpr int kRopeThreads = 256;

template <typename T>
__forceinline__ __device__ void rope_store(T* p, int64_t i, float v);

template <>
__forceinline__ __device__ void rope_store<float>(float* p, int64_t i, float v) {
    p[i] = v;
}
template <>
__forceinline__ __device__ void rope_store<__half>(__half* p, int64_t i, float v) {
    p[i] = __float2half(v);
}
template <>
__forceinline__ __device__ void rope_store<__hip_bfloat16>(__hip_bfloat16* p, int64_t i, float v) {
    p[i] = __float2bfloat16(v);
}

template <typename T>
__global__ void apply_rope_kernel(
    const void* __restrict__ xq, const void* __restrict__ xk,
    const void* __restrict__ freqs, T* __restrict__ xq_out, T* __restrict__ xk_out,
    int64_t batch, int64_t dim1, int64_t dim2, int64_t head_dim,
    int64_t fb, int64_t fd1, int64_t fd2,
    int64_t sx_b, int64_t sx_d1, int64_t sx_d2, int64_t sx_d,
    int64_t sf_b, int64_t sf_d1, int64_t sf_d2, int64_t sf_d, int64_t sf_rot,
    int64_t sf_pair,
    int x_code, int f_code, bool split_half) {

    const int64_t n_pairs = head_dim / 2;
    const int64_t total = batch * dim1 * dim2 * n_pairs;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int64_t k = idx % n_pairs;
    int64_t tmp = idx / n_pairs;
    const int64_t d2 = tmp % dim2;
    tmp /= dim2;
    const int64_t d1 = tmp % dim1;
    const int64_t b = tmp / dim1;

    const int64_t xoff = b * sx_b + d1 * sx_d1 + d2 * sx_d2;
    const int64_t ia = split_half ? k : 2 * k;
    const int64_t ib = split_half ? k + n_pairs : 2 * k + 1;
    const int64_t oa = xoff + ia * sx_d;
    const int64_t ob = xoff + ib * sx_d;

    const int64_t foff = (fb == 1 ? 0 : b) * sf_b + (fd1 == 1 ? 0 : d1) * sf_d1 +
                         (fd2 == 1 ? 0 : d2) * sf_d2 + k * sf_d;

    const float f00 = load_in(freqs, foff, f_code);
    const float f01 = load_in(freqs, foff + sf_pair, f_code);
    const float f10 = load_in(freqs, foff + sf_rot, f_code);
    const float f11 = load_in(freqs, foff + sf_rot + sf_pair, f_code);

    const float qa = load_in(xq, oa, x_code);
    const float qb = load_in(xq, ob, x_code);
    rope_store<T>(xq_out, oa, f00 * qa + f01 * qb);
    rope_store<T>(xq_out, ob, f10 * qa + f11 * qb);

    if (xk != nullptr) {
        const float ka = load_in(xk, oa, x_code);
        const float kb = load_in(xk, ob, x_code);
        rope_store<T>(xk_out, oa, f00 * ka + f01 * kb);
        rope_store<T>(xk_out, ob, f10 * ka + f11 * kb);
    }
}

}  // namespace comfy::hip_backend

extern "C" void launch_apply_rope_kernel(
    const void* xq, const void* xk, const void* freqs, void* xq_out, void* xk_out,
    int64_t batch, int64_t dim1, int64_t dim2, int64_t head_dim,
    int64_t fb, int64_t fd1, int64_t fd2,
    int64_t sx_b, int64_t sx_d1, int64_t sx_d2, int64_t sx_d,
    int64_t sf_b, int64_t sf_d1, int64_t sf_d2, int64_t sf_d, int64_t sf_rot, int64_t sf_pair,
    int x_code, int f_code, bool split_half, hipStream_t stream) {

    using namespace comfy::hip_backend;

    const int64_t total = batch * dim1 * dim2 * (head_dim / 2);
    const int blocks = static_cast<int>((total + kRopeThreads - 1) / kRopeThreads);
    if (blocks == 0) return;

#define CK_ROPE_LAUNCH(T)                                                                  \
    apply_rope_kernel<T><<<blocks, kRopeThreads, 0, stream>>>(                             \
        xq, xk, freqs, static_cast<T*>(xq_out), static_cast<T*>(xk_out), batch, dim1, dim2, \
        head_dim, fb, fd1, fd2, sx_b, sx_d1, sx_d2, sx_d, sf_b, sf_d1, sf_d2, sf_d, sf_rot, \
        sf_pair, x_code, f_code, split_half)

    if (x_code == 0) {
        CK_ROPE_LAUNCH(float);
    } else if (x_code == 1) {
        CK_ROPE_LAUNCH(__half);
    } else if (x_code == 2) {
        CK_ROPE_LAUNCH(__hip_bfloat16);
    } else {
        throw std::runtime_error("apply_rope: unsupported input dtype");
    }
#undef CK_ROPE_LAUNCH
}
