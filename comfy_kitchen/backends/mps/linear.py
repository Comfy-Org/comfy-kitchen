# SPDX-License-Identifier: Apache-2.0
"""Weight-only quantized linear layers for Apple GPUs.

Each call runs four steps, all range-safe by construction:

1. Prologue. Each activation row is divided by an exact power of two so that its
   largest magnitude, after the optional ConvRot rotation, is at most 1. With
   ConvRot the rotation (a 4x4 butterfly per base-4 digit, in fp32) runs in the
   same kernel. The result is written in the compute dtype (fp16, or fp32).
2. Decode. The integer weight is decoded with each output row divided by an
   exact power of two, so every decoded value has magnitude at most 1. The
   power moves into a per-output-channel exponent.
3. The stock GEMM. Every product is at most 1 in magnitude, so a dot product is
   at most K. That is finite in fp16 for K <= 65504; wider layers compute in fp32.
4. Epilogue. One kernel restores the combined exponent with integer exponent
   arithmetic (no 2**128 ever formed), adds bias before the single rounding to
   the output dtype, applies residual/residual_scale, and writes the output.

An unmerged LoRA (lora_down [R, K], lora_up [N, R]) rides along without forming the
[N, K] delta. The down weight goes through the same prologue as an activation (rotated
into the stored weight's basis, rows normalized), and its R rows are appended to the
first decoded weight chunk, so the main GEMM also produces the low-rank activation of
the normalized row in R extra columns: the down projection costs about 1% of the GEMM
instead of a separate skinny GEMM. The epilogue multiplies that activation by the up
weight on the simdgroup matrix units in fp32 and adds it before the rounding, bias and
residual, where a merged weight's GEMM would have. The up weight is carried as mantissa
and per-output exponent and the two terms meet in a common scaled domain before the range
is restored, so the LoRA path is range-safe the same way the base path is.

Apple GPUs flush fp32 subnormals in arithmetic (fp16 subnormals survive), so
exponents are read and applied on raw bits, never by multiplying.
"""
from __future__ import annotations

import os

import torch

from comfy_kitchen.backends._activations import apply_input_act, apply_residual
from comfy_kitchen.tensor.int8_utils import _rotate_activation, _build_hadamard

# Metal helpers shared with w4a8.py. bf16 tensors are read and written as ushort.
METAL_COMMON = r"""
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// fp32 bit pattern of element i; bf16 is the top half of an fp32.
inline uint f32_bits(device const float* p, ulong i) { return as_type<uint>(p[i]); }
inline uint f32_bits(device const half* p, ulong i) { return as_type<uint>(float(p[i])); }
inline uint f32_bits(device const ushort* p, ulong i) { return uint(p[i]) << 16; }

inline float load_f(device const float* p, ulong i) { return p[i]; }
inline float load_f(device const half* p, ulong i) { return float(p[i]); }
inline float load_f(device const ushort* p, ulong i) { return as_type<float>(uint(p[i]) << 16); }

// float -> bf16, round to nearest even; NaN -> 0x7FC0, as torch does.
inline ushort bf16_rne(uint u) {
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) {
        return ushort(0x7FC0);
    }
    return ushort((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

// Store an fp32 bit pattern rounded to the destination type.
inline void store_b(device float* p, ulong i, uint b) { p[i] = as_type<float>(b); }
inline void store_b(device half* p, ulong i, uint b) { p[i] = half(as_type<float>(b)); }
inline void store_b(device ushort* p, ulong i, uint b) { p[i] = bf16_rne(b); }

// v rounded to the destination type and widened back (eager rounds before residual).
inline float round_as(device float*, float v) { return v; }
inline float round_as(device half*, float v) { return float(half(v)); }
inline float round_as(device ushort*, float v) { return as_type<float>(uint(bf16_rne(as_type<uint>(v))) << 16); }

// Four consecutive elements; i must be a multiple of 4 on a 4-element-aligned buffer.
inline uint4 f32_bits4(device const float* p, ulong i) { return as_type<uint4>(*(device const float4*)(p + i)); }
inline uint4 f32_bits4(device const half* p, ulong i) { return as_type<uint4>(float4(*(device const half4*)(p + i))); }
inline uint4 f32_bits4(device const ushort* p, ulong i) { return uint4(*(device const ushort4*)(p + i)) << 16; }
inline void store_b4(device float* p, ulong i, uint4 b) { *(device float4*)(p + i) = as_type<float4>(b); }
inline void store_b4(device half* p, ulong i, uint4 b) { *(device half4*)(p + i) = half4(as_type<float4>(b)); }
inline void store_b4(device ushort* p, ulong i, uint4 b) {
    *(device ushort4*)(p + i) = ushort4(bf16_rne(b.x), bf16_rne(b.y), bf16_rne(b.z), bf16_rne(b.w));
}

// a = |x| bits, finite and nonzero. Returns E with 2^(E-1) <= |x| < 2^E.
inline int ceil_exp(uint a) {
    const uint e = a >> 23;
    return e == 0u ? int(32u - clz(a)) - 149 : int(e) - 126;
}

// Bits of x * 2^s, correctly rounded (to nearest even) when the result is subnormal.
// Integer exponent arithmetic: exact for any s, no intermediate overflow or flush.
inline uint ldexp_bits(uint u, int s) {
    const uint sign = u & 0x80000000u;
    const uint a = u & 0x7FFFFFFFu;
    int e = int(a >> 23);
    if (e >= 1 && e <= 254 && e + s >= 1 && e + s <= 254) {
        return u + uint(s << 23);  // normal in, normal out: only the exponent field moves
    }
    if (a == 0u || a >= 0x7F800000u) {
        return u;
    }
    uint m = a & 0x7FFFFFu;
    if (e == 0) {
        const int sh = int(clz(m)) - 8;
        m = (m << sh) & 0x7FFFFFu;
        e = 1 - sh;
    }
    e += s;
    if (e >= 255) {
        return sign | 0x7F800000u;
    }
    if (e >= 1) {
        return sign | (uint(e) << 23) | m;
    }
    const int sh = 1 - e;
    if (sh > 25) {
        return sign;
    }
    const uint full = m | 0x800000u;
    uint q = full >> sh;
    const uint rem = full & ((1u << sh) - 1u);
    const uint halfway = 1u << (sh - 1);
    if (rem > halfway || (rem == halfway && (q & 1u))) {
        q += 1u;
    }
    return sign | q;
}

// Split x = mant * 2^f with |mant| in [2^-(headroom+1), 2^-headroom). Zero and
// non-finite x come back unchanged with f = 0.
inline float split_pow2(float x, int headroom, thread int& f) {
    const uint u = as_type<uint>(x);
    const uint a = u & 0x7FFFFFFFu;
    if (a == 0u || a >= 0x7F800000u) {
        f = 0;
        return x;
    }
    f = ceil_exp(a) + headroom;
    return as_type<float>(ldexp_bits(u, -f));
}

// Bits of (a * 2^ea + b * 2^eb) * 2^e. Both terms are first brought to the larger one's own
// exponent (so each is below 1 in magnitude) and added there: neither side overflows or
// restores its range before they cancel, and a zero term never pulls the other's exponent.
// The result is rounded once by the add and once more only if it lands out of normal range.
inline uint add_scaled(uint a, int ea, uint b, int eb, int e) {
    const uint aa = a & 0x7FFFFFFFu, ab = b & 0x7FFFFFFFu;
    if (ab == 0u) {
        return ldexp_bits(a, ea + e);
    }
    if (aa == 0u) {
        return ldexp_bits(b, eb + e);
    }
    if (aa >= 0x7F800000u || ab >= 0x7F800000u) {  // inf or NaN: propagate as an add would
        return as_type<uint>(as_type<float>(a) + as_type<float>(b));
    }
    const int top = max(ceil_exp(aa) + ea, ceil_exp(ab) + eb);
    const float t = as_type<float>(ldexp_bits(a, ea - top)) + as_type<float>(ldexp_bits(b, eb - top));
    return ldexp_bits(as_type<uint>(t), top + e);
}
"""

_SHADERS = METAL_COMMON + r"""
// Four consecutive elements, as one vector load when VEC (aligned rows, K % 4 == 0).
template <bool VEC, typename T>
inline uint4 bits4(device const T* p, ulong i) {
    if (VEC) {
        return f32_bits4(p, i);
    }
    return uint4(f32_bits(p, i), f32_bits(p, i + 1), f32_bits(p, i + 2), f32_bits(p, i + 3));
}

// Activation prologue, one threadgroup of 256 threads per row; grid (256, M).
// Pass 1 finds the row's largest magnitude on raw bits (integer order == |float| order, so
// subnormals count) and picks row_exp = E + extra, E the smallest power with |x| < 2^E.
// Zero and non-finite rows get 0, so non-finite values pass through to the output.
// Pass 2 writes x * 2^-row_exp, rotated per group of G = 4^D when D > 0: the normalized
// regular Hadamard (Kronecker power of a symmetric 4x4 block) as one 4-point butterfly per
// base-4 digit, in fp32. The row is walked in chunks of 1024 elements, 4 consecutive per
// thread; the lowest digit's butterfly runs in registers, the rest in threadgroup memory.
// Chunks start at multiples of 1024 and G <= 1024 divides both 1024 and K, so every group
// lies inside one chunk.
template <typename TI, typename TO, bool VEC>
kernel void act_prologue(device TO* out,
                         device int* row_exp,
                         device const TI* x,
                         constant long& K_,
                         constant long& row_stride,
                         constant long& extra,
                         constant long& D,
                         uint2 pos [[thread_position_in_grid]],
                         uint2 lid [[thread_position_in_threadgroup]],
                         uint sid [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint part[8];
    threadgroup float buf[1024];
    const ulong K = ulong(K_);
    const ulong row = pos.y;
    const uint t = lid.x;
    device const TI* xr = x + row * ulong(row_stride);
    device TO* o = out + row * K;

    uint amax = 0u;
    if (VEC) {
        for (ulong k = 4 * t; k < K; k += 1024) {
            const uint4 b = f32_bits4(xr, k) & 0x7FFFFFFFu;
            amax = max(amax, max(max(b.x, b.y), max(b.z, b.w)));
        }
    } else {
        for (ulong k = t; k < K; k += 256) {
            amax = max(amax, f32_bits(xr, k) & 0x7FFFFFFFu);
        }
    }
    amax = simd_max(amax);
    if (lane == 0u) {
        part[sid] = amax;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint a = part[0];
    for (uint i = 1; i < 8u; ++i) {
        a = max(a, part[i]);
    }
    const int e = (a == 0u || a >= 0x7F800000u) ? 0 : ceil_exp(a) + int(extra);
    if (t == 0u) {
        row_exp[row] = e;
    }
    // The butterflies apply the unnormalized +-1 matrix; H_normalized is that over 2^D.
    const int s = -(e + int(D));

    if (D == 0) {
        for (ulong k = 4 * t; k < K; k += 1024) {
            if (VEC) {
                const uint4 b = f32_bits4(xr, k);
                store_b4(o, k, uint4(ldexp_bits(b.x, s), ldexp_bits(b.y, s), ldexp_bits(b.z, s), ldexp_bits(b.w, s)));
            } else {
                for (ulong i = k; i < min(k + 4, K); ++i) {
                    store_b(o, i, ldexp_bits(f32_bits(xr, i), s));
                }
            }
        }
        return;
    }
    for (ulong base = 0; base < K; base += 1024) {
        const ulong k = base + 4 * t;
        const bool live = k < K;  // groups are whole: a thread is live iff its group is
        if (live) {
            const uint4 b = bits4<VEC>(xr, k);
            const float a0 = as_type<float>(ldexp_bits(b.x, s)), a1 = as_type<float>(ldexp_bits(b.y, s));
            const float a2 = as_type<float>(ldexp_bits(b.z, s)), a3 = as_type<float>(ldexp_bits(b.w, s));
            buf[4 * t] = a0 + a1 + a2 - a3;
            buf[4 * t + 1] = a0 + a1 - a2 + a3;
            buf[4 * t + 2] = a0 - a1 + a2 + a3;
            buf[4 * t + 3] = -a0 + a1 + a2 + a3;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Butterflies for digits 1..D-1: thread j of its group (G/4 threads) owns the 4
        // slots of one butterfly, so there is no barrier between its read and write.
        const uint qlog = 2u * uint(D) - 2u;
        const uint j = t & ((1u << qlog) - 1u);
        threadgroup float* g = buf + ((t >> qlog) << (qlog + 2u));
        for (uint d = 1; d < uint(D); ++d) {
            const uint sl = 2u * d;
            const uint stride = 1u << sl;
            const uint bi = ((j >> sl) << (sl + 2u)) | (j & (stride - 1u));
            const float a0 = g[bi], a1 = g[bi + stride], a2 = g[bi + 2u * stride], a3 = g[bi + 3u * stride];
            g[bi] = a0 + a1 + a2 - a3;
            g[bi + stride] = a0 + a1 - a2 + a3;
            g[bi + 2u * stride] = a0 - a1 + a2 + a3;
            g[bi + 3u * stride] = -a0 + a1 + a2 + a3;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (live) {
            store_b4(o, k, uint4(as_type<uint>(buf[4 * t]), as_type<uint>(buf[4 * t + 1]),
                                 as_type<uint>(buf[4 * t + 2]), as_type<uint>(buf[4 * t + 3])));
        }
    }
}

// Rows [row0, row0 + rows) of w * scale, each row divided by 2^ch_exp[n] so |out| < 1.
// out is [rows, K]. Grid (K, rows).
template <typename T>
kernel void i8_decode(device T* out,
                      device int* ch_exp,
                      device const char* w,
                      device const float* scale,
                      constant long& K,
                      constant long& row0,
                      constant long& scale_stride,
                      uint2 pos [[thread_position_in_grid]]) {
    const ulong k = pos.x, r = pos.y, n = ulong(row0) + r;
    int f;
    const float mant = split_pow2(scale[n * ulong(scale_stride)], 7, f);  // |int8| <= 127 < 2^7
    out[r * ulong(K) + k] = T(float(w[n * ulong(K) + k]) * mant);
    if (k == 0) {
        ch_exp[n] = f;
    }
}

// Four columns per thread when K % 4 == 0. Grid (K/4, rows).
template <typename T>
kernel void i8_decode4(device vec<T, 4>* out,
                       device int* ch_exp,
                       device const char4* w,
                       device const float* scale,
                       constant long& K,
                       constant long& row0,
                       constant long& scale_stride,
                       uint2 pos [[thread_position_in_grid]]) {
    const ulong k4 = pos.x, r = pos.y, n = ulong(row0) + r, quads = ulong(K) / 4;
    int f;
    const float mant = split_pow2(scale[n * ulong(scale_stride)], 7, f);
    out[r * quads + k4] = vec<T, 4>(float4(w[n * quads + k4]) * mant);
    if (k4 == 0) {
        ch_exp[n] = f;
    }
}

// Same decode written transposed, out[k, r] with row stride ld >= rows, through a 32x32
// threadgroup tile so both the int8 read and the write are coalesced. Grid (K rounded up to
// 32, rows rounded up to 32).
template <typename T>
kernel void i8_decode_t(device T* out,
                        device int* ch_exp,
                        device const char* w,
                        device const float* scale,
                        constant long& K,
                        constant long& rows,
                        constant long& row0,
                        constant long& scale_stride,
                        constant long& ld,
                        uint2 tg [[threadgroup_position_in_grid]],
                        uint2 lid [[thread_position_in_threadgroup]]) {
    threadgroup float tile[32][33];
    const ulong k = ulong(tg.x) * 32 + lid.x, r = ulong(tg.y) * 32 + lid.y;
    if (k < ulong(K) && r < ulong(rows)) {
        const ulong n = ulong(row0) + r;
        int f;
        const float mant = split_pow2(scale[n * ulong(scale_stride)], 7, f);
        tile[lid.y][lid.x] = float(w[n * ulong(K) + k]) * mant;
        if (k == 0) {
            ch_exp[n] = f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const ulong ko = ulong(tg.x) * 32 + lid.y, ro = ulong(tg.y) * 32 + lid.x;
    if (ko < ulong(K) && ro < ulong(rows)) {
        out[ko * ulong(ld) + ro] = T(tile[lid.x][lid.y]);
    }
}

// Transposed decode, four values per load and per store: a threadgroup of 32x8 threads
// moves a tile of 32 rows x 128 columns through threadgroup memory. Needs K, rows and the
// row stride ld to be multiples of 4. Grid (ceil(K/128) * 32, ceil(rows/32) * 8).
template <typename T>
kernel void i8_decode_t4(device T* out,
                         device int* ch_exp,
                         device const char4* w,
                         device const float* scale,
                         constant long& K,
                         constant long& rows,
                         constant long& row0,
                         constant long& scale_stride,
                         constant long& ld,
                         uint2 tg [[threadgroup_position_in_grid]],
                         uint2 lid [[thread_position_in_threadgroup]]) {
    threadgroup T tile[128][36];  // [k][r], padded
    const ulong quads = ulong(K) / 4;
    const ulong k4 = ulong(tg.x) * 32 + lid.x;
    for (uint i = 0; i < 4u; ++i) {
        const uint rl = lid.y + 8u * i;
        const ulong r = ulong(tg.y) * 32 + rl;
        if (k4 < quads && r < ulong(rows)) {
            const ulong n = ulong(row0) + r;
            int f;
            const float mant = split_pow2(scale[n * ulong(scale_stride)], 7, f);
            const float4 v = float4(w[n * quads + k4]) * mant;
            for (uint j = 0; j < 4u; ++j) {
                tile[lid.x * 4u + j][rl] = T(v[j]);
            }
            if (k4 == 0) {
                ch_exp[n] = f;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint tid = lid.y * 32u + lid.x;
    const uint rq = (tid & 7u) * 4u;
    const ulong r0 = ulong(tg.y) * 32 + rq;
    for (uint p = 0; p < 4u; ++p) {
        const uint kl = (tid >> 3) + 32u * p;
        const ulong k = ulong(tg.x) * 128 + kl;
        if (k < ulong(K) && r0 < ulong(rows)) {
            *(device vec<T, 4>*)(out + k * ulong(ld) + r0) =
                vec<T, 4>(tile[kl][rq], tile[kl][rq + 1], tile[kl][rq + 2], tile[kl][rq + 3]);
        }
    }
}

// out[m, col0 + n] = c[m, n] * 2^(row_exp[m] + ch_exp[col0 + n]) with bias, then
// residual + residual_scale * that, rounded again (eager's order). Bias arrives as fp32
// values already rounded the way eager rounds them. FL is a compile-time bit set:
// 1 = bias, 2 = residual, 4 = round to the output dtype before adding bias (eager's INT8
// order; without it bias is added before the one rounding, as eager's F.linear does),
// 8 = LoRA: the GEMM value is c * 2^(re + ce) with ex = re + ce, the LoRA value
// lora * 2^(re + lue); the two are summed in their common scaled domain and only then
// restored (add_scaled), so a LoRA that cancels the weight gives 0, as a merged weight
// would, not inf - inf. As a runtime argument the flags cost 1.5-2.5x here.
// Fast path: when the two scales are within 2^64 of each other, both terms are brought to
// the larger scale by exact power-of-two multipliers in [2^-64, 1] and summed by one fma.
// It is taken only when both scaled terms stay at or above 2^-102 (biased exponent >= 25):
// then any nonzero sum, even after cancellation, is a multiple of an ulp >= 2^-125 and so
// normal. A zero or tiny term must not set the scale, or the other term or the cancelled
// sum would flush (a LoRA contributing 0 would erase a small base). Everything else takes
// add_scaled, which follows the terms' own exponents.
template <int FL, typename TO>
inline uint epilogue_value(device TO* out, uint cb, int ex, float bv, float r, float rs,
                           float lora = 0.0f, int lue = 0, int re = 0) {
    uint b;
    if (FL & 8) {
        const int ce = ex - re, d = ce - lue, top = max(ce, lue);
        const int ea = int((cb >> 23) & 0xFFu), el = int((as_type<uint>(lora) >> 23) & 0xFFu);
        if (d >= -64 && d <= 64 && ea + ce - top >= 25 && el + lue - top >= 25) {
            const float pa = as_type<float>(uint(127 + ce - top) << 23);
            const float pb = as_type<float>(uint(127 + lue - top) << 23);
            b = ldexp_bits(as_type<uint>(fma(lora, pb, as_type<float>(cb) * pa)), re + top);
        } else {
            b = add_scaled(cb, ce, as_type<uint>(lora), lue, re);
        }
    } else {
        b = ldexp_bits(cb, ex);
    }
    if (FL & 1) {
        if (FL & 4) {
            b = as_type<uint>(round_as(out, as_type<float>(b)));
        }
        if (bv != 0.0f) {  // adding zero would flush a subnormal result
            b = as_type<uint>(as_type<float>(b) + bv);
        }
    }
    if (FL & 2) {
        b = as_type<uint>(r + rs * round_as(out, as_type<float>(b)));
    }
    return b;
}

// One column per thread. Grid (rows, M).
template <int FL, typename TC, typename TO>
kernel void epilogue(device TO* out,
                     device const TC* c,
                     device const int* row_exp,
                     device const int* ch_exp,
                     device const float* bias,
                     device const TO* res,
                     device const TO* rscale,
                     constant long& N,
                     constant long& col0,
                     constant long& bias_stride,
                     constant long& res_stride,
                     constant long& rscale_stride,
                     uint2 pos [[thread_position_in_grid]],
                     uint2 grid [[threads_per_grid]]) {
    const ulong n = pos.x, m = pos.y;
    const ulong col = ulong(col0) + n;
    const float bv = (FL & 1) ? bias[col * ulong(bias_stride)] : 0.0f;
    const float r = (FL & 2) ? load_f(res, m * ulong(res_stride) + col) : 0.0f;
    const float rs = (FL & 2) ? load_f(rscale, col * ulong(rscale_stride)) : 0.0f;
    const uint b = epilogue_value<FL>(out, f32_bits(c, m * ulong(grid.x) + n), row_exp[m] + ch_exp[col], bv, r, rs);
    store_b(out, m * ulong(N) + col, b);
}

// Four columns per thread with vector loads and stores: needs rows, N and col0 to be
// multiples of 4. Grid (rows / 4, M).
template <int FL, typename TC, typename TO>
kernel void epilogue4(device TO* out,
                      device const TC* c,
                      device const int* row_exp,
                      device const int* ch_exp,
                      device const float* bias,
                      device const TO* res,
                      device const TO* rscale,
                      constant long& N,
                      constant long& col0,
                      constant long& bias_stride,
                      constant long& res_stride,
                      constant long& rscale_stride,
                      uint2 pos [[thread_position_in_grid]],
                      uint2 grid [[threads_per_grid]]) {
    const ulong n = ulong(pos.x) * 4, m = pos.y;
    const ulong col = ulong(col0) + n;
    const int re = row_exp[m];
    const uint4 cb = f32_bits4(c, m * ulong(grid.x) * 4 + n);
    uint4 b;
    for (int v = 0; v < 4; ++v) {
        const ulong cv = col + v;
        const float bv = (FL & 1) ? bias[cv * ulong(bias_stride)] : 0.0f;
        const float r = (FL & 2) ? load_f(res, m * ulong(res_stride) + cv) : 0.0f;
        const float rs = (FL & 2) ? load_f(rscale, cv * ulong(rscale_stride)) : 0.0f;
        b[v] = epilogue_value<FL>(out, cb[v], re + ch_exp[cv], bv, r, rs);
    }
    store_b4(out, m * ulong(N) + col, b);
}

// The epilogue with a LoRA up-projection folded in. la holds the low-rank activation of the
// normalized row, la[m * la_stride + la_off + r] = x'[m] . D'[r] (the GEMM's extra columns,
// each at most K), and upT [R, ups] the normalized up weight (|upT| < 1, from lora_up_prep,
// zero-padded so a tile may read 64 columns past col0 + rows), so the LoRA output of row m
// is 2^(row_exp[m] + up_exp[col]) * sum_r la[m, r] * upT[r, col]. The sum is at most R * K,
// so the fp32 accumulation on the simdgroup matrix units cannot overflow; its exponent is
// applied only when it meets the base value (flag 8 of epilogue_value). A threadgroup
// of LT_M / 8 simdgroups owns a tile of LT_M rows x LT_N columns: la is staged through
// threadgroup memory LT_R ranks at a time, each simdgroup accumulates 8 rows x LT_N columns,
// and the tile then goes through the usual per-element epilogue, 4 threads per row. R is a
// multiple of 8. VEC: rows, N and col0 multiples of 4. c is [M, rows] with row stride
// c_stride. Grid (ceil(rows / LT_N) * LT_T, ceil(M / LT_M)), threadgroups of LT_T x 1.
constant constexpr uint LT_M = 32, LT_N = 64, LT_R = 32, LT_T = LT_M * 4;

template <int FL, typename TC, typename TO, bool VEC>
kernel void epilogue_lora(device TO* out,
                          device const TC* c,
                          device const int* row_exp,
                          device const int* ch_exp,
                          device const float* bias,
                          device const TO* res,
                          device const TO* rscale,
                          device const TC* la,
                          device const float* upT,
                          device const int* up_exp,
                          constant long& N,
                          constant long& col0,
                          constant long& bias_stride,
                          constant long& res_stride,
                          constant long& rscale_stride,
                          constant long& M_,
                          constant long& R_,
                          constant long& rows_,
                          constant long& c_stride,
                          constant long& la_stride,
                          constant long& la_off,
                          constant long& ups,
                          uint2 tg [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float la_t[LT_M][LT_R];
    threadgroup float lora_t[LT_M][LT_N];
    const ulong M = ulong(M_), R = ulong(R_), rows = ulong(rows_), cs = ulong(c_stride);
    const ulong m0 = ulong(tg.y) * LT_M, n0 = ulong(tg.x) * LT_N;
    simdgroup_float8x8 acc[LT_N / 8];
    for (uint j = 0; j < LT_N / 8; ++j) {
        acc[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    device const float* ub = upT + ulong(col0) + n0;
    for (ulong r0 = 0; r0 < R; r0 += LT_R) {
        for (uint e = tid; e < LT_M * LT_R; e += LT_T) {
            const uint i = e / LT_R, j = e % LT_R;
            const ulong m = m0 + i, r = r0 + j;
            la_t[i][j] = (m < M && r < R) ? float(la[m * ulong(la_stride) + ulong(la_off) + r]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const ulong kend = min(ulong(LT_R), R - r0);
        for (ulong kk = 0; kk < kend; kk += 8) {
            simdgroup_float8x8 a;
            simdgroup_load(a, &la_t[sg * 8][kk], LT_R);
            for (uint j = 0; j < LT_N / 8; ++j) {
                simdgroup_float8x8 b;  // the 4 simdgroups read the same b; the cache serves it
                simdgroup_load(b, ub + (r0 + kk) * ulong(ups) + j * 8, ulong(ups));
                simdgroup_multiply_accumulate(acc[j], a, b, acc[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint j = 0; j < LT_N / 8; ++j) {
        simdgroup_store(acc[j], &lora_t[sg * 8][j * 8], LT_N);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 4 threads per row; quad q of thread t covers columns 16q + 4(t % 4) .. + 3.
    const uint i = tid / 4u;
    const ulong m = m0 + i;
    if (m >= M) {
        return;
    }
    const int re = row_exp[m];
    for (uint q = 0; q < LT_N / 16u; ++q) {
        const uint jl = q * 16u + (tid % 4u) * 4u;
        const ulong n = n0 + jl;
        if (n >= rows) {
            break;
        }
        const ulong col = ulong(col0) + n;
        if (VEC) {
            const uint4 cb = f32_bits4(c, m * cs + n);
            uint4 b;
            for (uint v = 0; v < 4u; ++v) {
                const float bv = (FL & 1) ? bias[(col + v) * ulong(bias_stride)] : 0.0f;
                const float r = (FL & 2) ? load_f(res, m * ulong(res_stride) + col + v) : 0.0f;
                const float rs = (FL & 2) ? load_f(rscale, (col + v) * ulong(rscale_stride)) : 0.0f;
                b[v] = epilogue_value<FL>(out, cb[v], re + ch_exp[col + v], bv, r, rs, lora_t[i][jl + v],
                                          up_exp[col + v], re);
            }
            store_b4(out, m * ulong(N) + col, b);
        } else {
            for (uint v = 0; v < 4u && n + v < rows; ++v) {
                const ulong cv = col + v;
                const float bv = (FL & 1) ? bias[cv * ulong(bias_stride)] : 0.0f;
                const float r = (FL & 2) ? load_f(res, m * ulong(res_stride) + cv) : 0.0f;
                const float rs = (FL & 2) ? load_f(rscale, cv * ulong(rscale_stride)) : 0.0f;
                store_b(out, m * ulong(N) + cv,
                        epilogue_value<FL>(out, f32_bits(c, m * cs + n + v), re + ch_exp[cv], bv, r, rs,
                                           lora_t[i][jl + v], up_exp[cv], re));
            }
        }
    }
}

// The LoRA up weight as mantissa and exponent, never materialized at full range: with
// scale = sm * 2^se (|sm| in [1/2, 1)), up[n, r] * scale * 2^dexp[r] = upT[r, n] * 2^up_exp[n],
// where up_exp[n] is the smallest exponent with every |upT[r, n]| < 1 for that output row.
// fp32, zero in the padding (R8 rows, ups columns) so the LoRA epilogue reads aligned,
// bounded tiles. One thread per column n; grid (ups).
template <typename TU>
kernel void lora_up_prep(device float* upT,
                         device int* up_exp,
                         device const TU* up,
                         device const int* dexp,
                         constant float& scale,
                         constant long& N,
                         constant long& R,
                         constant long& R8,
                         constant long& ups,
                         uint pos [[thread_position_in_grid]]) {
    const ulong n = pos;
    int se;
    const float sm = split_pow2(scale, 0, se);
    const int none = -(1 << 30);
    int e = none;
    if (n < ulong(N)) {
        for (ulong r = 0; r < ulong(R); ++r) {
            const uint a = f32_bits(up, n * ulong(R) + r) & 0x7FFFFFFFu;
            if (a != 0u && a < 0x7F800000u) {
                e = max(e, ceil_exp(a) + dexp[r]);
            }
        }
        if (e == none) {
            e = 0;  // an all-zero (or non-finite) row: nothing to scale
        }
        up_exp[n] = e + se;
    }
    for (ulong r = 0; r < ulong(R8); ++r) {
        float v = 0.0f;
        if (n < ulong(N) && r < ulong(R)) {
            v = as_type<float>(ldexp_bits(f32_bits(up, n * ulong(R) + r), dexp[r] - e)) * sm;
        }
        upT[r * ulong(ups) + n] = v;
    }
}
#define UPP(TU, NAME)                                                                        \
    template [[host_name("lora_up_prep_" NAME)]] kernel void lora_up_prep<TU>(               \
        device float*, device int*, device const TU*, device const int*, constant float&,    \
        constant long&, constant long&, constant long&, constant long&, uint);
UPP(half, "f16")
UPP(float, "f32")
UPP(ushort, "bf16")

#define ACT1(TI, TO, VEC, NAME)                                                             \
    template [[host_name(NAME)]] kernel void act_prologue<TI, TO, VEC>(                    \
        device TO*, device int*, device const TI*, constant long&, constant long&,         \
        constant long&, constant long&, uint2, uint2, uint, uint);
#define ACT(TI, TO, NAME) ACT1(TI, TO, true, "act_prologue4_" NAME) ACT1(TI, TO, false, "act_prologue_" NAME)
ACT(half, half, "f16_f16")
ACT(float, half, "f32_f16")
ACT(ushort, half, "bf16_f16")
ACT(half, float, "f16_f32")
ACT(float, float, "f32_f32")
ACT(ushort, float, "bf16_f32")

#define DEC(T, NAME)                                                                         \
    template [[host_name("i8_decode_" NAME)]] kernel void i8_decode<T>(                      \
        device T*, device int*, device const char*, device const float*, constant long&,     \
        constant long&, constant long&, uint2);                                              \
    template [[host_name("i8_decode4_" NAME)]] kernel void i8_decode4<T>(                    \
        device vec<T, 4>*, device int*, device const char4*, device const float*,            \
        constant long&, constant long&, constant long&, uint2);                              \
    template [[host_name("i8_decode_t4_" NAME)]] kernel void i8_decode_t4<T>(                \
        device T*, device int*, device const char4*, device const float*, constant long&,    \
        constant long&, constant long&, constant long&, constant long&, uint2, uint2);       \
    template [[host_name("i8_decode_t_" NAME)]] kernel void i8_decode_t<T>(                  \
        device T*, device int*, device const char*, device const float*, constant long&,     \
        constant long&, constant long&, constant long&, constant long&, uint2, uint2);
DEC(half, "f16")
DEC(float, "f32")

#define EPI1(FL, TC, TO, NAME)                                                               \
    template [[host_name("epilogue_" #FL "_" NAME)]] kernel void epilogue<FL, TC, TO>(       \
        device TO*, device const TC*, device const int*, device const int*, device const float*, \
        device const TO*, device const TO*, constant long&, constant long&, constant long&,  \
        constant long&, constant long&, uint2, uint2);                                       \
    template [[host_name("epilogue4_" #FL "_" NAME)]] kernel void epilogue4<FL, TC, TO>(     \
        device TO*, device const TC*, device const int*, device const int*, device const float*, \
        device const TO*, device const TO*, constant long&, constant long&, constant long&,  \
        constant long&, constant long&, uint2, uint2);
#define EPI(TC, TO, NAME) EPI1(0, TC, TO, NAME) EPI1(1, TC, TO, NAME) EPI1(2, TC, TO, NAME) EPI1(3, TC, TO, NAME) \
    EPI1(5, TC, TO, NAME) EPI1(7, TC, TO, NAME)
EPI(half, half, "f16_f16")
EPI(half, float, "f16_f32")
EPI(half, ushort, "f16_bf16")
EPI(float, half, "f32_f16")
EPI(float, float, "f32_f32")
EPI(float, ushort, "f32_bf16")

#define LEPI1(FL, TC, TO, VEC, NAME)                                                           \
    template [[host_name(NAME)]] kernel void epilogue_lora<FL, TC, TO, VEC>(                   \
        device TO*, device const TC*, device const int*, device const int*, device const float*, \
        device const TO*, device const TO*, device const TC*, device const float*,              \
        device const int*, constant long&, constant long&, constant long&, constant long&,       \
        constant long&, constant long&, constant long&, constant long&, constant long&,          \
        constant long&, constant long&, constant long&, uint2, uint, uint);
#define LEPI2(FL, TC, TO, NAME) LEPI1(FL, TC, TO, true, "epilogue_lora4_" #FL "_" NAME) \
    LEPI1(FL, TC, TO, false, "epilogue_lora_" #FL "_" NAME)
#define LEPI(TC, TO, NAME) LEPI2(8, TC, TO, NAME) LEPI2(9, TC, TO, NAME) LEPI2(10, TC, TO, NAME) \
    LEPI2(11, TC, TO, NAME) LEPI2(13, TC, TO, NAME) LEPI2(15, TC, TO, NAME)
LEPI(half, half, "f16_f16")
LEPI(half, float, "f16_f32")
LEPI(half, ushort, "f16_bf16")
LEPI(float, half, "f32_f16")
LEPI(float, float, "f32_f32")
LEPI(float, ushort, "f32_bf16")
"""

_lib = None

# Kernel-name suffix per dtype; bf16 travels as ushort.
DTYPE_SUFFIX = {torch.float16: "f16", torch.float32: "f32", torch.bfloat16: "bf16"}

# Decode at most this many bytes of weight at once. Bounds the transient buffer for very
# wide layers; the GEMM then runs per row-chunk of the weight.
_MAX_DECODE_BYTES = int(os.environ.get("COMFY_KITCHEN_MPS_DECODE_BYTES", 512 * 1024 * 1024))

# |x'| <= 1 and |w'| <= 1 bound every dot product by K, which fp16 holds up to 65504.
_FP16_MAX_K = 65504

# The fused rotation holds 1024 fp32 values per threadgroup.
_MAX_FUSED_GROUP = 1024

# The LoRA epilogue's tile: LT_M rows x LT_N columns per threadgroup of LT_M * 4 (see the shader).
_LORA_TILE_M, _LORA_TILE_N = 32, 64

# MPS matmul against a lazily transposed weight (what F.linear does) falls onto a
# slow path once the input width passes ~8k: 2x slower at K=16384 on an M1 Max.
# Decoding straight into [K, N] layout avoids it.
_TRANSPOSED_DECODE_MIN_K = int(os.environ.get("COMFY_KITCHEN_MPS_TRANSPOSED_MIN_K", 12288))


def _shaders():
    global _lib
    if _lib is None:
        _lib = torch.mps.compile_shader(_SHADERS)
    return _lib


def _compute_dtype(x: torch.Tensor, k: int) -> torch.dtype:
    """fp16 unless the caller works in fp32 or K is too wide for fp16's range.

    bf16 is emulated on M1-M4 GPUs and measurably slower than fp16, so it is never used.
    """
    forced = os.environ.get("COMFY_KITCHEN_MPS_COMPUTE_DTYPE")
    if forced:
        cdtype = getattr(torch, forced)
        if cdtype not in (torch.float16, torch.float32):
            raise ValueError(f"COMFY_KITCHEN_MPS_COMPUTE_DTYPE must be float16 or float32, got {forced}")
    else:
        cdtype = torch.float32 if x.dtype == torch.float32 else torch.float16
    return torch.float32 if k > _FP16_MAX_K else cdtype


def _hadamard_depth(group: int) -> int:
    """D with group == 4**D, validated like _build_hadamard."""
    d = 0
    while 4 ** d < group:
        d += 1
    if group < 4 or 4 ** d != group:
        raise ValueError(f"Regular Hadamard size must be a power of 4, got {group}")
    return d


def prologue(x2: torch.Tensor, group: int | None, cdtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Rows of x2 [M, K] divided by 2^row_exp (and rotated by groups of ``group``) in cdtype.

    Afterwards every value has magnitude at most 1, and x2 = 2^row_exp * x' (@ H^T).
    """
    m, k = x2.shape
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    extra = 0 if group is None else _hadamard_depth(group)
    fused = group is None or group <= _MAX_FUSED_GROUP
    xc = torch.empty((m, k), dtype=cdtype if fused else torch.float32, device=x2.device)
    row_exp = torch.empty(m, dtype=torch.int32, device=x2.device)
    # Vector loads need 4-element-aligned rows; a sliced view can start anywhere.
    vec = k % 4 == 0 and x2.stride(0) % 4 == 0 and x2.storage_offset() % 4 == 0
    kernel = f"act_prologue{4 if vec else ''}_{DTYPE_SUFFIX[x2.dtype]}_{DTYPE_SUFFIX[xc.dtype]}"
    getattr(_shaders(), kernel)(
        xc, row_exp, x2, k, x2.stride(0), extra, extra if fused else 0,
        threads=(256, m), group_size=(256, 1),
    )
    if not fused:  # groups past the fused kernel's buffer: rotate in fp32
        h = _build_hadamard(group, device=x2.device, dtype=torch.float32)
        xc = _rotate_activation(xc, h, group).to(cdtype)
    return xc, row_exp


def _bias_operand(bias, n, dtype, device):
    """Eager's ``bias.to(dtype).reshape(1, -1)``, carried as exact fp32 values: flattened,
    N values or one broadcast."""
    if bias is None:
        return None, 0
    b = bias.to(device=device, dtype=dtype).reshape(-1).float()
    if b.numel() == 1:
        return b, 0
    if b.numel() != n:
        raise ValueError(f"bias with {b.numel()} elements does not broadcast to {n} output features")
    return b.contiguous(), 1


def _residual_operands(residual, residual_scale, out_shape, n, out_dtype, device):
    """(residual [M, N] view, row stride, scale vector, scale stride) when the epilogue can
    apply eager's ``addcmul(residual, out, residual_scale)`` without changing its shape,
    else None (the caller falls back to eager's apply_residual)."""
    if residual is None:
        return None
    if residual_scale is None:
        raise ValueError("residual requires residual_scale")
    rs = residual_scale
    if tuple(residual.shape) != tuple(out_shape) or rs.dim() > len(out_shape):
        return None
    if rs.numel() != 1 and (rs.shape[-1] != n or rs.numel() != n):
        return None
    res2 = residual.to(device=device, dtype=out_dtype).reshape(-1, n)
    if res2.stride(-1) != 1:
        res2 = res2.contiguous()
    rs_v = rs.to(device=device, dtype=out_dtype).reshape(-1).contiguous()
    return res2, res2.stride(0), rs_v, 0 if rs_v.numel() == 1 else 1


def lora_operands(lora_down, lora_up, lora_scale, n, k, device):
    """Validated ``(down [R, K], up [N, R], scale)``, or None when there is no LoRA term.

    Several LoRAs on one layer are one LoRA of the summed rank: concatenate the downs along
    R and the ups along R, each up multiplied by its own scale (and pass ``lora_scale=1``).
    """
    if lora_down is None and lora_up is None:
        return None
    if lora_down is None or lora_up is None:
        raise ValueError("lora_down and lora_up must be given together")
    if lora_down.dim() != 2 or lora_up.dim() != 2:
        raise ValueError(
            f"lora_down must be [R, K] and lora_up [N, R], got {tuple(lora_down.shape)} and {tuple(lora_up.shape)}"
        )
    r = lora_down.shape[0]
    if lora_down.shape[1] != k or tuple(lora_up.shape) != (n, r):
        raise ValueError(
            f"lora_down {tuple(lora_down.shape)} and lora_up {tuple(lora_up.shape)} do not fit "
            f"a layer with N={n}, K={k}: expected [R, {k}] and [{n}, R]"
        )
    scale = float(lora_scale)
    if r == 0 or scale == 0.0:
        return None
    down, up = lora_down.to(device=device), lora_up.to(device=device).contiguous()
    if down.dtype not in DTYPE_SUFFIX:
        down = down.float()
    if up.dtype not in DTYPE_SUFFIX:
        up = up.float()
    return down, up, scale


def _lora_gemm_operands(lora, group, cdtype):
    """(dn [R8, K] in cdtype, upT [R8, N4 + 64] in fp32, up_exp [N] int32), R8 = R rounded up
    to 8 and N4 = N rounded up to 4, padded with zeros (the epilogue's 8x8 matrix tiles and
    aligned rows; a tile may read 64 columns past the end).

    The down weight goes through the activation prologue: rotated into the stored weight's
    basis (D H pairs with the rotated row x' exactly as the stored W H does, H being
    orthogonal) and each row divided by 2^dexp[r] so |D'| <= 1. Appended to the decoded
    weight, it makes the main GEMM also produce la = x' @ D'^T, each value at most K like the
    base product. The up weight with the scale and 2^dexp folded in is never formed: it is
    carried as upT[r, n] * 2^up_exp[n] with |upT| < 1, so the LoRA output of row m is
    2^(row_exp[m] + up_exp[n]) * la[m] @ upT[:, n], its sum at most R * K."""
    down, up, scale = lora
    r, n = down.shape[0], up.shape[0]
    r8 = -(-r // 8) * 8
    dn, dexp = prologue(down, group, cdtype)
    ups = -(-n // 4) * 4 + _LORA_TILE_N
    upt = torch.empty((r8, ups), dtype=torch.float32, device=up.device)
    up_exp = torch.empty(n, dtype=torch.int32, device=up.device)
    getattr(_shaders(), f"lora_up_prep_{DTYPE_SUFFIX[up.dtype]}")(
        upt, up_exp, up, dexp, scale, n, r, r8, ups, threads=(ups,), group_size=(min(ups, 256),),
    )
    if r8 != r:
        dn = torch.cat([dn, dn.new_zeros((r8 - r, dn.shape[1]))])
    return dn, upt, up_exp


def weight_only_linear(x, n, k, decode, *, bias, out_dtype, group, residual=None, residual_scale=None,
                       bias_in_linear=False, lora=None):
    """Shared driver for INT8 and W4A8: prologue, chunked decode + GEMM, fused epilogue.

    Bias follows eager per format. By default (eager INT8 and symmetric W4A8) the result is
    rounded to ``out_dtype`` and bias, cast to ``out_dtype``, is added after. With
    ``bias_in_linear`` (eager's asymmetric W4A8, ``F.linear(x, w, bias)``) bias is cast to
    x's dtype and added before the single rounding.

    ``lora`` (from ``lora_operands``) adds ``scale * (x @ down^T) @ up^T`` where a merged
    weight would: in fp32, before the rounding, bias and residual. ``down`` is in the layer's
    original (un-rotated) basis.

    ``decode(row0, rows, cdtype, transposed, ch_exp)`` returns rows [row0, row0 + rows) of the
    normalized weight, [rows, K] (or [K, rows] when transposed), and fills ch_exp[row0:row0+rows].
    """
    lead = x.shape[:-1]
    out_shape = (*lead, n)
    device = x.device
    bias_v, bias_stride = _bias_operand(bias, n, x.dtype if bias_in_linear else out_dtype, device)
    if x.numel() == 0 or n == 0:  # an empty K makes the LoRA term zero as well
        out = torch.zeros(out_shape, dtype=out_dtype, device=device)
        if bias_v is not None:
            out = out + bias_v.to(out_dtype)
        return apply_residual(out, residual, residual_scale)

    x2 = x.reshape(-1, k)
    m = x2.shape[0]
    res = _residual_operands(residual, residual_scale, out_shape, n, out_dtype, device)
    cdtype = _compute_dtype(x, k)
    xc, row_exp = prologue(x2, group, cdtype)

    lib = _shaders()
    epi_name = f"{DTYPE_SUFFIX[cdtype]}_{DTYPE_SUFFIX[out_dtype]}"
    out = torch.empty((m, n), dtype=out_dtype, device=device)
    ch_exp = torch.empty(n, dtype=torch.int32, device=device)
    flags = (bias_v is not None) | ((res is not None) << 1)
    if bias_v is not None and not bias_in_linear:
        flags |= 4
    if lora is not None:
        dn, upt, up_exp = _lora_gemm_operands(lora, group, cdtype)
        rank = dn.shape[0]
        flags |= 8
    unused = torch.zeros(1, dtype=out_dtype, device=device)  # placeholder for absent operands
    res2, res_stride, rs_v, rs_stride = res if res is not None else (unused, 0, unused, 0)
    no_bias = torch.zeros(1, dtype=torch.float32, device=device)
    transposed = k >= _TRANSPOSED_DECODE_MIN_K
    rows_per_chunk = max(1, _MAX_DECODE_BYTES // (torch.finfo(cdtype).bits // 8 * k))
    if rows_per_chunk > 4:
        rows_per_chunk -= rows_per_chunk % 4  # keeps chunk starts on the vector epilogue's grid
    for row0 in range(0, n, rows_per_chunk):
        rows = min(rows_per_chunk, n - row0)
        if lora is not None and row0 == 0:
            # The first chunk's GEMM also computes the low-rank activation: D' rides along as
            # rank extra weight rows, la lands in c's last rank columns.
            w, view = decode_buffer(rows, k, cdtype, transposed, rank, device)
            decode(row0, rows, cdtype, transposed, ch_exp, view)
            del view  # otherwise this view keeps the first decoded weight buffer alive
            if transposed:
                w[:, rows:] = dn.t()
            else:
                w[rows:] = dn
        else:
            w = decode(row0, rows, cdtype, transposed, ch_exp)
        c = xc @ w if transposed else torch.nn.functional.linear(xc, w)
        v = 4 if rows % 4 == 0 and n % 4 == 0 and row0 % 4 == 0 else 1
        operands = (out, c, row_exp, ch_exp, bias_v if bias_v is not None else no_bias, res2, rs_v)
        if lora is None:
            getattr(lib, f"epilogue{4 if v == 4 else ''}_{flags}_{epi_name}")(
                *operands, n, row0, bias_stride, res_stride, rs_stride,
                threads=(rows // v, m), group_size=(min(rows // v, 256), 1),
            )
        else:
            if row0 == 0:
                la_args = (c, c.stride(0), rows)  # (tensor, row stride, column offset)
            getattr(lib, f"epilogue_lora{4 if v == 4 else ''}_{flags}_{epi_name}")(
                *operands, la_args[0], upt, up_exp, n, row0, bias_stride, res_stride, rs_stride, m, rank, rows,
                c.stride(0), la_args[1], la_args[2], upt.stride(0),
                threads=(-(-rows // _LORA_TILE_N) * _LORA_TILE_M * 4, -(-m // _LORA_TILE_M)),
                group_size=(_LORA_TILE_M * 4, 1),
            )
            if row0 == 0 and rows < n:  # later chunks read a compact copy; clone, since at
                # M = 1 the slice is already contiguous and .contiguous() would keep c alive
                la_args = (c[:, rows:].clone(memory_format=torch.contiguous_format), rank, 0)
        del w, c, operands  # operands also holds c; release it before allocating the next chunk
    out = out.reshape(out_shape)
    if residual is not None and res is None:
        out = apply_residual(out, residual, residual_scale)
    return out


def decode_buffer(rows, k, cdtype, transposed, extra, device):
    """A decode destination with ``extra`` spare weight rows after the decoded ones: [rows +
    extra, K], or [K, rows + extra] when transposed. Returns (buffer, view of the decoded
    rows); the view is what the decode writes, the buffer what the GEMM reads."""
    if transposed:
        buf = torch.empty((k, rows + extra), dtype=cdtype, device=device)
        return buf, buf[:, :rows]
    buf = torch.empty((rows + extra, k), dtype=cdtype, device=device)
    return buf, buf[:rows]


def decode_int8(weight, scale, row0, rows, cdtype, transposed, ch_exp, out=None):
    """Rows [row0, row0 + rows) of the INT8 weight, each divided by 2^ch_exp so |w'| < 1.

    ``out`` (from ``decode_buffer``) receives them in place; otherwise a new tensor is returned.
    """
    k = weight.shape[1]
    lib = _shaders()
    suffix = DTYPE_SUFFIX[cdtype]
    scale_stride = 0 if scale.numel() == 1 else 1
    if out is None:
        shape = (k, rows) if transposed else (rows, k)
        out = torch.empty(shape, dtype=cdtype, device=weight.device)
    ld = out.stride(0)
    if transposed and k % 4 == 0 and rows % 4 == 0 and ld % 4 == 0:
        getattr(lib, f"i8_decode_t4_{suffix}")(
            out, ch_exp, weight, scale, k, rows, row0, scale_stride, ld,
            threads=(-(-k // 128) * 32, -(-rows // 32) * 8), group_size=(32, 8),
        )
    elif transposed:
        getattr(lib, f"i8_decode_t_{suffix}")(
            out, ch_exp, weight, scale, k, rows, row0, scale_stride, ld,
            threads=(-(-k // 32) * 32, -(-rows // 32) * 32), group_size=(32, 32),
        )
    elif k % 4 == 0:
        getattr(lib, f"i8_decode4_{suffix}")(
            out, ch_exp, weight, scale, k, row0, scale_stride,
            threads=(k // 4, rows), group_size=(min(k // 4, 256), 1),
        )
    else:
        getattr(lib, f"i8_decode_{suffix}")(
            out, ch_exp, weight, scale, k, row0, scale_stride,
            threads=(k, rows), group_size=(min(k, 256), 1),
        )
    return out


def int8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    convrot: bool = False,
    convrot_groupsize: int = 256,
    input_act: str | None = None,
    input_act_weight: torch.Tensor | None = None,
    input_act_eps: float = 0.0,
    residual: torch.Tensor | None = None,
    residual_scale: torch.Tensor | None = None,
    lora_down: torch.Tensor | None = None,
    lora_up: torch.Tensor | None = None,
    lora_scale: float = 1.0,
) -> torch.Tensor:
    """INT8-weight linear, computed as fp16 GEMM over on-the-fly decoded weights.

    ``lora_down`` [R, K] and ``lora_up`` [N, R] (the layer's original, un-rotated basis) add
    ``lora_scale * (x @ lora_down^T) @ lora_up^T`` without merging, as if the weight were
    ``W + lora_scale * lora_up @ lora_down``. See ``lora_operands`` for several LoRAs.
    """
    x = apply_input_act(x, input_act, input_act_weight, input_act_eps)
    n, k = weight.shape
    if x.shape[-1] != k:
        raise ValueError(
            f"Input and weight inner dimensions must match, got {x.shape[-1]} and {k}"
        )
    if convrot and k % convrot_groupsize != 0:
        raise ValueError(
            f"ConvRot group size {convrot_groupsize} does not divide input features {k}"
        )
    weight = weight.to(device=x.device).contiguous()
    scale = weight_scale.to(device=x.device, dtype=torch.float32).reshape(-1).contiguous()
    if scale.numel() not in (1, n):
        raise ValueError(
            f"INT8 weight scale must be scalar or per-output-channel, got {tuple(weight_scale.shape)} "
            f"for weight shape {tuple(weight.shape)}"
        )

    lora = lora_operands(lora_down, lora_up, lora_scale, n, k, x.device)

    def decode(row0, rows, cdtype, transposed, ch_exp, out=None):
        return decode_int8(weight, scale, row0, rows, cdtype, transposed, ch_exp, out)

    return weight_only_linear(
        x, n, k, decode, bias=bias, out_dtype=out_dtype,
        group=convrot_groupsize if convrot else None,
        residual=residual, residual_scale=residual_scale, lora=lora,
    )
