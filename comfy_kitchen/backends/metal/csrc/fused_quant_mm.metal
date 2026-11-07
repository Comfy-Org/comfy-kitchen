// SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Fused low-bit GEMMs: dequant in-register during the K loop, with threadgroup
// tiles so a row of A / B is loaded once per tile and reused across the MN tile.

#include <metal_stdlib>
using namespace metal;

// ---- shared helpers ----

inline float e4m3_to_f32(uchar b) {
  const uint sign = (b >> 7) & 1u;
  const uint exp = (b >> 3) & 0xFu;
  const uint mant = b & 0x7u;
  float v;
  if (exp == 0u) {
    v = (mant == 0u) ? 0.0f : ldexp(float(mant) / 8.0f, -6);
  } else if (exp == 0xFu && mant == 0x7u) {
    v = NAN;
  } else {
    v = ldexp(1.0f + float(mant) / 8.0f, int(exp) - 7);
  }
  return sign ? -v : v;
}

constant float E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

inline float e2m1_from_nibble(uint n) { return E2M1_LUT[n & 0xFu]; }

inline int svd_nibble_signed(uint n) {
  int v = int(n & 0xFu);
  return v >= 8 ? v - 16 : v;
}

inline int svd_nibble_unsigned(uint n) { return int(n & 0xFu); }

// Tile: 8x8 MN, K-stage 32 (2 NVFP4 blocks / half SVDQuant group).
constant uint TM = 8u;
constant uint TN = 8u;
constant uint TK = 32u;

// =============================================================================
// NVFP4 fused C = A @ B.T  (tiled)
// =============================================================================

struct Nvfp4MmParams {
  uint M;
  uint N;
  uint K;
  uint a_packed_stride;
  uint b_packed_stride;
  uint a_scale_stride;
  uint b_scale_stride;
  uint out_stride;
  float tensor_scale_a;
  float tensor_scale_b;
  uint hi_first;
};

kernel void scaled_mm_nvfp4_fused(
    device const uchar* a_packed [[buffer(0)]],
    device const uchar* b_packed [[buffer(1)]],
    device const uchar* a_scales [[buffer(2)]],
    device const uchar* b_scales [[buffer(3)]],
    device half* out [[buffer(4)]],
    constant Nvfp4MmParams& p [[buffer(5)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]) {
  threadgroup float As[TM * TK];
  threadgroup float Bs[TN * TK];

  const uint m0 = tgid.y * TM;
  const uint n0 = tgid.x * TN;
  const uint mi = tid.y;
  const uint ni = tid.x;
  const uint m = m0 + mi;
  const uint n = n0 + ni;
  const float tsa = p.tensor_scale_a;
  const float tsb = p.tensor_scale_b;
  float acc = 0.0f;

  const uint tlinear = mi * TN + ni;
  const uint tptg = TM * TN;

  for (uint k0 = 0; k0 < p.K; k0 += TK) {
    // Cooperative dequant of A tile rows into As[mi, k]
    for (uint i = tlinear; i < TM * TK; i += tptg) {
      const uint row = i / TK;
      const uint kk = i % TK;
      const uint mm = m0 + row;
      float v = 0.0f;
      if (mm < p.M && (k0 + kk) < p.K) {
        const uint kb = k0 + kk;
        const uint block = kb / 16u;
        const float sa = tsa * e4m3_to_f32(a_scales[ulong(mm) * p.a_scale_stride + block]);
        const uchar ab = a_packed[ulong(mm) * p.a_packed_stride + kb / 2u];
        const uint nib = ((kb & 1u) == 0u)
                             ? (p.hi_first ? ((ab >> 4) & 0xFu) : (ab & 0xFu))
                             : (p.hi_first ? (ab & 0xFu) : ((ab >> 4) & 0xFu));
        v = e2m1_from_nibble(nib) * sa;
      }
      As[i] = v;
    }
    // Cooperative dequant of B tile rows into Bs[ni, k]
    for (uint i = tlinear; i < TN * TK; i += tptg) {
      const uint row = i / TK;
      const uint kk = i % TK;
      const uint nn = n0 + row;
      float v = 0.0f;
      if (nn < p.N && (k0 + kk) < p.K) {
        const uint kb = k0 + kk;
        const uint block = kb / 16u;
        const float sb = tsb * e4m3_to_f32(b_scales[ulong(nn) * p.b_scale_stride + block]);
        const uchar bb = b_packed[ulong(nn) * p.b_packed_stride + kb / 2u];
        const uint nib = ((kb & 1u) == 0u)
                             ? (p.hi_first ? ((bb >> 4) & 0xFu) : (bb & 0xFu))
                             : (p.hi_first ? (bb & 0xFu) : ((bb >> 4) & 0xFu));
        v = e2m1_from_nibble(nib) * sb;
      }
      Bs[i] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (m < p.M && n < p.N) {
      for (uint k = 0; k < TK; ++k) {
        acc = fma(As[mi * TK + k], Bs[ni * TK + k], acc);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (m < p.M && n < p.N) {
    out[ulong(m) * p.out_stride + n] = half(acc);
  }
}

// =============================================================================
// SVDQuant W4A4 fused (tiled, group size 64)
// =============================================================================

struct SvdMmParams {
  uint M;
  uint N;
  uint K;
  uint groups;
  uint a_packed_stride;
  uint w_packed_stride;
  uint a_scale_stride;
  uint w_scale_stride;
  uint out_stride;
  uint act_unsigned;
};

kernel void scaled_mm_svdquant_fused(
    device const uchar* a_packed [[buffer(0)]],
    device const uchar* w_packed [[buffer(1)]],
    device const half* a_scales [[buffer(2)]],
    device const half* w_scales [[buffer(3)]],
    device half* out [[buffer(4)]],
    constant SvdMmParams& p [[buffer(5)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]) {
  threadgroup float As[TM * TK];
  threadgroup float Ws[TN * TK];

  const uint m0 = tgid.y * TM;
  const uint n0 = tgid.x * TN;
  const uint mi = tid.y;
  const uint ni = tid.x;
  const uint m = m0 + mi;
  const uint n = n0 + ni;
  float acc = 0.0f;
  const uint tlinear = mi * TN + ni;
  const uint tptg = TM * TN;
  const uint group_size = 64u;

  for (uint k0 = 0; k0 < p.K; k0 += TK) {
    for (uint i = tlinear; i < TM * TK; i += tptg) {
      const uint row = i / TK;
      const uint kk = i % TK;
      const uint mm = m0 + row;
      float v = 0.0f;
      if (mm < p.M && (k0 + kk) < p.K) {
        const uint kb = k0 + kk;
        const uint g = kb / group_size;
        const float sa = float(a_scales[g * p.a_scale_stride + mm]);
        const uchar ab = a_packed[ulong(mm) * p.a_packed_stride + kb / 2u];
        const uint nib = (kb & 1u) ? ((ab >> 4) & 0xFu) : (ab & 0xFu);
        const int qi = p.act_unsigned ? svd_nibble_unsigned(nib) : svd_nibble_signed(nib);
        v = float(qi) * sa;
      }
      As[i] = v;
    }
    for (uint i = tlinear; i < TN * TK; i += tptg) {
      const uint row = i / TK;
      const uint kk = i % TK;
      const uint nn = n0 + row;
      float v = 0.0f;
      if (nn < p.N && (k0 + kk) < p.K) {
        const uint kb = k0 + kk;
        const uint g = kb / group_size;
        const float sw = float(w_scales[g * p.w_scale_stride + nn]);
        const uchar wb = w_packed[ulong(nn) * p.w_packed_stride + kb / 2u];
        const uint nib = (kb & 1u) ? ((wb >> 4) & 0xFu) : (wb & 0xFu);
        v = float(svd_nibble_signed(nib)) * sw;
      }
      Ws[i] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (m < p.M && n < p.N) {
      for (uint k = 0; k < TK; ++k) {
        acc = fma(As[mi * TK + k], Ws[ni * TK + k], acc);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (m < p.M && n < p.N) {
    out[ulong(m) * p.out_stride + n] = half(acc);
  }
}

// =============================================================================
// AWQ W4A16 GEMV / small-batch (tiled over N; M usually tiny)
// =============================================================================

struct AwqParams {
  uint M;
  uint N;
  uint K;
  uint group_size;
  uint x_stride;
  uint q_stride;
  uint scale_stride;
  uint out_stride;
  uint has_bias;
};

kernel void gemv_awq_w4a16_fused(
    device const half* x [[buffer(0)]],
    device const uchar* qweight [[buffer(1)]],
    device const half* wscales [[buffer(2)]],
    device const half* wzeros [[buffer(3)]],
    device const half* bias [[buffer(4)]],
    device half* out [[buffer(5)]],
    constant AwqParams& p [[buffer(6)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]) {
  threadgroup float Xs[TM * TK];
  threadgroup float Ws[TN * TK];

  const uint m0 = tgid.y * TM;
  const uint n0 = tgid.x * TN;
  const uint mi = tid.y;
  const uint ni = tid.x;
  const uint m = m0 + mi;
  const uint n = n0 + ni;
  float acc = 0.0f;
  const uint tlinear = mi * TN + ni;
  const uint tptg = TM * TN;

  for (uint k0 = 0; k0 < p.K; k0 += TK) {
    for (uint i = tlinear; i < TM * TK; i += tptg) {
      const uint row = i / TK;
      const uint kk = i % TK;
      const uint mm = m0 + row;
      float v = 0.0f;
      if (mm < p.M && (k0 + kk) < p.K) {
        v = float(x[ulong(mm) * p.x_stride + k0 + kk]);
      }
      Xs[i] = v;
    }
    for (uint i = tlinear; i < TN * TK; i += tptg) {
      const uint row = i / TK;
      const uint kk = i % TK;
      const uint nn = n0 + row;
      float v = 0.0f;
      if (nn < p.N && (k0 + kk) < p.K) {
        const uint kb = k0 + kk;
        const uint g = kb / p.group_size;
        const float scale = float(wscales[g * p.scale_stride + nn]);
        const float zero = float(wzeros[g * p.scale_stride + nn]);
        const uchar byte = qweight[ulong(nn) * p.q_stride + kb / 2u];
        const uint nib = (kb & 1u) ? ((byte >> 4) & 0xFu) : (byte & 0xFu);
        v = (float(nib) - 8.0f) * scale + zero;
      }
      Ws[i] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (m < p.M && n < p.N) {
      for (uint k = 0; k < TK; ++k) {
        acc = fma(Xs[mi * TK + k], Ws[ni * TK + k], acc);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (m < p.M && n < p.N) {
    if (p.has_bias) {
      acc += float(bias[n]);
    }
    out[ulong(m) * p.out_stride + n] = half(acc);
  }
}
