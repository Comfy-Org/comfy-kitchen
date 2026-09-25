// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Rounding and rotation shared by ops/apply_rope.hip and ops/rms_rope.hip.
//
// The rotation is evaluated in the freqs dtype, rounding after each multiply and
// add, matching the eager reference, which upcasts x to the freqs dtype and
// evaluates there. Doing the same makes the kernels track eager's rounding instead
// of out-accuracy-ing it in fp32, which otherwise drifts past the comparison
// tolerance for bf16 freqs.
#pragma once

#include <hip/hip_runtime.h>

namespace comfy::hip_backend {

// Round fp32 to bf16 (round to nearest, ties to even) and widen back.
// Both callers build with -ffast-math, under which the native __bf16 type carries
// excess precision: casting fp32 -> __bf16 -> fp32 is folded away and the
// intermediate rounding never happens. The integer path here survives that fold
// because it is bitwise, not floating-point.
//
// A NaN survives if its mantissa exceeds 0x8000, which every reachable one does
// (torch quiets a signaling NaN on the way in). Below that it rounds to Inf, and
// -ffinite-math-only folds a guard away, so there is no fixing it here.
__forceinline__ __device__ float round_bf16(float x) {
    unsigned int u = __float_as_uint(x);
    u += 0x7fffu + ((u >> 16) & 1u);
    return __uint_as_float(u & 0xffff0000u);
}

// Round fp32 to fp16 (round to nearest, ties to even) and widen back.
// `__half2float(__float2half_rn(x))` does not survive -ffast-math: clang sees
// fpext(fptrunc(x)), drops the pair, then contracts the multiply it guarded into
// the following add. Inline asm is opaque to that, the way round_bf16's integer
// arithmetic is. These two converts are base VALU on every target built here.
__forceinline__ __device__ float round_fp16(float x) {
    float r;
    asm("v_cvt_f16_f32 %0, %1\n\tv_cvt_f32_f16 %0, %0" : "=v"(r) : "v"(x));
    return r;
}

// Round to the storage type of T without storing. rms_rope needs this for the
// normalized value, which the unfused contract materializes in x's dtype before
// the rotation reads it back.
template <typename T>
__forceinline__ __device__ float round_to(float v);

template <>
__forceinline__ __device__ float round_to<float>(float v) {
    return v;
}
template <>
__forceinline__ __device__ float round_to<__half>(float v) {
    return round_fp16(v);
}
template <>
__forceinline__ __device__ float round_to<__bf16>(float v) {
    return round_bf16(v);
}

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

// out = f_a * x_a + f_b * x_b, evaluated in the freqs dtype with the same rounding
// eager applies after upcasting x to the freqs dtype.
//
// Eager rounds exactly twice: the standalone product (apply_rope1's
// `freqs[..., 0] * x_[..., 0]`, split-half's `t.unflatten * diagonal`) is
// materialized as a tensor, so it is rounded to the freqs dtype; the addcmul_
// that follows then evaluates `p + f_b * x_b` in fp32 opmath and rounds once.
// The second product is never rounded on its own -- and does not need to be,
// since the product of two fp16 (11-bit) or bf16 (8-bit) mantissas is exact in
// fp32. Both eager layouts round identically, so split_half needs no special
// case; rounding the second product here cost an extra rounding and desynced the
// kernel from eager for apply_rope_split_half / rms_rope_split_half.
// f_code: 0=fp32, 1=fp16, 2=bf16.
__forceinline__ __device__ float rope_combine(
    float f_a, float x_a, float f_b, float x_b, int f_code) {
    if (f_code == 2) {
        const float pa = round_bf16(round_bf16(f_a) * round_bf16(x_a));
        return round_bf16(pa + round_bf16(f_b) * round_bf16(x_b));
    }
    if (f_code == 1) {
        const float pa = round_fp16(round_fp16(f_a) * round_fp16(x_a));
        return round_fp16(pa + round_fp16(f_b) * round_fp16(x_b));
    }
    return f_a * x_a + f_b * x_b;
}

// Output b of a rotation pair (out[b] = f_b * x_b + f_a * x_a). Which product is
// rounded before the fused add depends on the layout, because eager rounds the
// term it materializes as a tensor:
//   interleaved (apply_rope1): x_out = f0 * x[a]; addcmul_(f1, x[b])  -> f_a*x_a
//   split-half (apply_rope_split_half1): out = t * diagonal (f_b*x_b for the
//     b output); addcmul_(t[a], f_a)                                  -> f_b*x_b
// Same value either way, one rounding apart, so the two layouts need different
// argument orders to stay bit-identical to eager.
__forceinline__ __device__ float rope_combine_b(
    float f_a, float x_a, float f_b, float x_b, int f_code, bool split_half) {
    return split_half ? rope_combine(f_b, x_b, f_a, x_a, f_code)
                      : rope_combine(f_a, x_a, f_b, x_b, f_code);
}

}  // namespace comfy::hip_backend
