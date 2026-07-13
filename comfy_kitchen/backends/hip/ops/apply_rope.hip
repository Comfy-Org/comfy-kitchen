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
// for the split-half layout. The rotation is evaluated in the freqs dtype,
// rounding after each multiply and add, matching the eager reference, which
// upcasts x to the freqs dtype and evaluates there. Doing the same makes the
// kernel track eager's rounding instead of out-accuracy-ing it in fp32, which
// otherwise drifts past the comparison tolerance for bf16 freqs.
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
__forceinline__ __device__ void rope_store<__bf16>(__bf16* p, int64_t i, float v) {
    p[i] = static_cast<__bf16>(v);
}

// Round a finite fp32 to bf16 (round to nearest, ties to even) and widen back.
// This whole file builds with -ffast-math, under which the native __bf16 type
// carries excess precision: casting fp32 -> __bf16 -> fp32 is folded away and the
// intermediate rounding never happens. The integer path here survives that fold
// because it is bitwise, not floating-point. Inputs are finite (rope data), so
// NaN/Inf handling is not needed.
__forceinline__ __device__ float round_bf16(float x) {
    unsigned int u = __float_as_uint(x);
    u += 0x7fffu + ((u >> 16) & 1u);
    return __uint_as_float(u & 0xffff0000u);
}

// Round a finite fp32 to fp16 and widen back. __float2half_rn is an intrinsic
// conversion, so unlike a __bf16 cast it is not folded under -ffast-math.
__forceinline__ __device__ float round_fp16(float x) {
    return __half2float(__float2half_rn(x));
}

// out = f_a * x_a + f_b * x_b, evaluated in the freqs dtype with the same rounding
// eager applies after upcasting x to the freqs dtype. The two eager layouts round
// differently: split-half (apply_rope_split_half1) forms both products as separate
// tensors, so each is rounded before the add; interleaved (apply_rope1) uses
// addcmul_, which fuses the second product into the add under one rounding. Rounding
// inputs first mirrors eager's cast of x. f_code: 0=fp32, 1=fp16, 2=bf16.
__forceinline__ __device__ float rope_combine(
    float f_a, float x_a, float f_b, float x_b, int f_code, bool split_half) {
    if (f_code == 2) {
        const float pa = round_bf16(round_bf16(f_a) * round_bf16(x_a));
        const float second = round_bf16(f_b) * round_bf16(x_b);
        return round_bf16(pa + (split_half ? round_bf16(second) : second));
    }
    if (f_code == 1) {
        const float pa = round_fp16(round_fp16(f_a) * round_fp16(x_a));
        const float second = round_fp16(f_b) * round_fp16(x_b);
        return round_fp16(pa + (split_half ? round_fp16(second) : second));
    }
    return f_a * x_a + f_b * x_b;
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
    rope_store<T>(xq_out, oa, rope_combine(f00, qa, f01, qb, f_code, split_half));
    rope_store<T>(xq_out, ob, rope_combine(f10, qa, f11, qb, f_code, split_half));

    if (xk != nullptr) {
        const float ka = load_in(xk, oa, x_code);
        const float kb = load_in(xk, ob, x_code);
        rope_store<T>(xk_out, oa, rope_combine(f00, ka, f01, kb, f_code, split_half));
        rope_store<T>(xk_out, ob, rope_combine(f10, ka, f11, kb, f_code, split_half));
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
        CK_ROPE_LAUNCH(__bf16);
    } else {
        throw std::runtime_error("apply_rope: unsupported input dtype");
    }
#undef CK_ROPE_LAUNCH
}
