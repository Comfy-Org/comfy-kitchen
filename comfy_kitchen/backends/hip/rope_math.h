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

extern "C" __device__ float __ocml_rsqrt_f32(float);

__forceinline__ __device__ float ieee_div_f32(
    float numerator, float denominator) {
    bool denominator_scale = false;
    bool scale = false;
    const float scaled_denominator = __builtin_amdgcn_div_scalef(
        numerator, denominator, false, &denominator_scale);
    const float scaled_numerator = __builtin_amdgcn_div_scalef(
        numerator, denominator, true, &scale);
    float reciprocal = __builtin_amdgcn_rcpf(scaled_denominator);
    const float reciprocal_error =
        fmaf(-scaled_denominator, reciprocal, 1.0f);
    reciprocal = fmaf(reciprocal_error, reciprocal, reciprocal);
    float quotient = scaled_numerator * reciprocal;
    float remainder =
        fmaf(-scaled_denominator, quotient, scaled_numerator);
    quotient = fmaf(remainder, reciprocal, quotient);
    remainder = fmaf(-scaled_denominator, quotient, scaled_numerator);
    quotient = __builtin_amdgcn_div_fmasf(
        remainder, reciprocal, quotient, scale);
    return __builtin_amdgcn_div_fixupf(
        quotient, denominator, numerator);
}

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

// Round a finite fp32 to fp16 and widen back. __float2half_rn is an intrinsic
// conversion, so unlike a __bf16 cast it is not folded under -ffast-math.
__forceinline__ __device__ float round_fp16(float x) {
    return __half2float(__float2half_rn(x));
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

}  // namespace comfy::hip_backend
