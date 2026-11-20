// SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Fused ConvRot Fast Hadamard Transform + quantization / GEMM for Metal.
// Matches CUDA's regular H4 Kronecker FHT: stages at strides 1,4,16,... with
// 0.5 scale per stage so the overall transform is orthonormal.

#include <metal_stdlib>
using namespace metal;

inline uint pcg_hash_cr(uint x) {
  const uint state = x * 747796405u + 2891336453u;
  const uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

inline float stochastic_rng_cr(ulong idx, ulong seed) {
  const ulong key = idx + seed;
  const uint folded = uint(key) ^ uint(key >> 32);
  return float(pcg_hash_cr(folded) >> 8) * 0x1.0p-24f;
}

inline float h4_row_dot(uint d, float x0, float x1, float x2, float x3) {
  switch (d) {
    case 0u: return x0 + x1 + x2 - x3;
    case 1u: return x0 + x1 - x2 + x3;
    case 2u: return x0 - x1 + x2 + x3;
    default: return -x0 + x1 + x2 + x3;
  }
}

inline uint log4_group(uint group_size) {
  // group_size is 4^n; return n.
  uint n = 0u;
  uint g = group_size;
  while (g > 1u) {
    g >>= 2;
    n += 1u;
  }
  return n;
}

// In-place Fast Hadamard over one group of `group_size` floats in `buf`,
// using `tmp` as ping-pong (also group_size floats). All threads in the
// threadgroup participate; inactive lanes still hit barriers.
inline void fht_group(
    threadgroup float* buf,
    threadgroup float* tmp,
    uint group_size,
    uint tid,
    uint tptg) {
  const uint stages = log4_group(group_size);
  threadgroup float* src = buf;
  threadgroup float* dst = tmp;
  for (uint stage = 0u; stage < stages; ++stage) {
    uint s = 1u;
    for (uint i = 0u; i < stage; ++i) {
      s *= 4u;
    }
    for (uint i = tid; i < group_size; i += tptg) {
      const uint d = (i / s) & 3u;
      const uint base = i - d * s;
      dst[i] = 0.5f * h4_row_dot(d, src[base], src[base + s], src[base + 2u * s], src[base + 3u * s]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float* t = src;
    src = dst;
    dst = t;
  }
  // After an even number of swaps src==buf; after odd, result is in tmp.
  if ((stages & 1u) != 0u) {
    for (uint i = tid; i < group_size; i += tptg) {
      buf[i] = tmp[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

struct ConvRotQuantParams {
  uint rows;
  uint dim;
  uint group_size;
  uint stochastic;
  uint seed_lo;
  uint seed_hi;
  uint x_stride;
  uint q_stride;
};

inline ulong cr_seed(constant ConvRotQuantParams& p) {
  return (ulong(p.seed_hi) << 32) | ulong(p.seed_lo);
}

// One threadgroup per row. Dynamic TG memory layout:
//   [0, dim)                 float row buffer (rotated in place)
//   [dim, dim+group)         float FHT temp
//   [dim+group, dim+group+8) float absmax reduction
template <typename T>
void quantize_int8_convrot_row(
    device const T* x,
    device char* q,
    device float* scale,
    constant ConvRotQuantParams& p,
    uint row,
    uint tid,
    uint tptg,
    threadgroup float* smem) {
  if (row >= p.rows) {
    return;
  }
  threadgroup float* row_buf = smem;
  threadgroup float* tmp = smem + p.dim;
  threadgroup float* red = tmp + p.group_size;
  device const T* xrow = x + ulong(row) * p.x_stride;
  device char* qrow = q + ulong(row) * p.q_stride;

  for (uint i = tid; i < p.dim; i += tptg) {
    row_buf[i] = float(xrow[i]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint n_groups = p.dim / p.group_size;
  for (uint g = 0u; g < n_groups; ++g) {
    fht_group(row_buf + g * p.group_size, tmp, p.group_size, tid, tptg);
  }

  // Round-trip through T so absmax/quant match a separate fp16/bf16 rotate +
  // quantize path (eager / two-kernel Metal).
  for (uint i = tid; i < p.dim; i += tptg) {
    row_buf[i] = float(T(row_buf[i]));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float amax = 0.0f;
  for (uint i = tid; i < p.dim; i += tptg) {
    amax = max(amax, fabs(row_buf[i]));
  }
  amax = simd_max(amax);
  const uint simd_lane = tid % 32u;
  const uint simd_id = tid / 32u;
  if (simd_lane == 0u) {
    red[simd_id] = amax;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float m = 0.0f;
    const uint num_simd = (tptg + 31u) / 32u;
    for (uint i = 0u; i < num_simd; ++i) {
      m = max(m, red[i]);
    }
    float s = max(m / 127.0f, 1e-30f);
    scale[row] = s;
    red[0] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = 1.0f / red[0];
  const bool stochastic = p.stochastic != 0u;
  const ulong seed = cr_seed(p);
  const ulong row_base = ulong(row) * ulong(p.dim);
  for (uint i = tid; i < p.dim; i += tptg) {
    float v = row_buf[i] * inv;
    float qi = stochastic ? floor(v + stochastic_rng_cr(row_base + i, seed)) : round(v);
    qi = clamp(qi, -128.0f, 127.0f);
    qrow[i] = char(int(qi));
  }
}

kernel void quantize_int8_rowwise_convrot_fp16(
    device const half* x [[buffer(0)]],
    device char* q [[buffer(1)]],
    device float* scale [[buffer(2)]],
    constant ConvRotQuantParams& p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    threadgroup float* smem [[threadgroup(0)]]) {
  quantize_int8_convrot_row<half>(x, q, scale, p, row, tid, tptg, smem);
}

kernel void quantize_int8_rowwise_convrot_bf16(
    device const bfloat* x [[buffer(0)]],
    device char* q [[buffer(1)]],
    device float* scale [[buffer(2)]],
    constant ConvRotQuantParams& p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    threadgroup float* smem [[threadgroup(0)]]) {
  quantize_int8_convrot_row<bfloat>(x, q, scale, p, row, tid, tptg, smem);
}

template <typename T>
void quantize_int4_convrot_row(
    device const T* x,
    device char* packed,
    device float* scale,
    constant ConvRotQuantParams& p,
    uint row,
    uint tid,
    uint tptg,
    threadgroup float* smem) {
  if (row >= p.rows) {
    return;
  }
  threadgroup float* row_buf = smem;
  threadgroup float* tmp = smem + p.dim;
  threadgroup float* red = tmp + p.group_size;
  device const T* xrow = x + ulong(row) * p.x_stride;
  device char* prow = packed + ulong(row) * p.q_stride;

  for (uint i = tid; i < p.dim; i += tptg) {
    row_buf[i] = float(xrow[i]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint n_groups = p.dim / p.group_size;
  for (uint g = 0u; g < n_groups; ++g) {
    fht_group(row_buf + g * p.group_size, tmp, p.group_size, tid, tptg);
  }

  for (uint i = tid; i < p.dim; i += tptg) {
    row_buf[i] = float(T(row_buf[i]));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float amax = 0.0f;
  for (uint i = tid; i < p.dim; i += tptg) {
    amax = max(amax, fabs(row_buf[i]));
  }
  amax = simd_max(amax);
  const uint simd_lane = tid % 32u;
  const uint simd_id = tid / 32u;
  if (simd_lane == 0u) {
    red[simd_id] = amax;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float m = 0.0f;
    const uint num_simd = (tptg + 31u) / 32u;
    for (uint i = 0u; i < num_simd; ++i) {
      m = max(m, red[i]);
    }
    float s = max(m / 7.0f, 1e-10f);
    scale[row] = s;
    red[0] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = 1.0f / red[0];
  const bool stochastic = p.stochastic != 0u;
  const ulong seed = cr_seed(p);
  const ulong row_base = ulong(row) * ulong(p.dim);
  const uint pairs = p.dim / 2u;
  for (uint i = tid; i < pairs; i += tptg) {
    const uint d = i * 2u;
    float vlo = row_buf[d] * inv;
    float vhi = row_buf[d + 1u] * inv;
    float qlo = stochastic ? floor(vlo + stochastic_rng_cr(row_base + d, seed)) : round(vlo);
    float qhi = stochastic ? floor(vhi + stochastic_rng_cr(row_base + d + 1u, seed)) : round(vhi);
    int lo = clamp(int(qlo), -7, 7);
    int hi = clamp(int(qhi), -7, 7);
    prow[i] = char((lo & 0xF) | ((hi & 0xF) << 4));
  }
}

kernel void quantize_int4_rowwise_convrot_fp16(
    device const half* x [[buffer(0)]],
    device char* packed [[buffer(1)]],
    device float* scale [[buffer(2)]],
    constant ConvRotQuantParams& p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    threadgroup float* smem [[threadgroup(0)]]) {
  quantize_int4_convrot_row<half>(x, packed, scale, p, row, tid, tptg, smem);
}

kernel void quantize_int4_rowwise_convrot_bf16(
    device const bfloat* x [[buffer(0)]],
    device char* packed [[buffer(1)]],
    device float* scale [[buffer(2)]],
    constant ConvRotQuantParams& p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    threadgroup float* smem [[threadgroup(0)]]) {
  quantize_int4_convrot_row<bfloat>(x, packed, scale, p, row, tid, tptg, smem);
}

// Pure FHT rotate over rows: out = FHT(x) with orthonormal H.
// One threadgroup per (row, group). TG memory is only 2*group_size floats, so
// K can be arbitrarily large (LTX uses K=16384 with 32KB TG caps).
struct RotateParams {
  uint rows;
  uint dim;
  uint group_size;
  uint x_stride;
  uint out_stride;
};

template <typename T>
void rotate_hadamard_group_impl(
    device const T* x,
    device T* out,
    constant RotateParams& p,
    uint group,
    uint row,
    uint tid,
    uint tptg,
    threadgroup float* smem) {
  if (row >= p.rows) {
    return;
  }
  const uint n_groups = p.dim / p.group_size;
  if (group >= n_groups) {
    return;
  }
  threadgroup float* buf = smem;
  threadgroup float* tmp = smem + p.group_size;
  device const T* xrow = x + ulong(row) * p.x_stride + ulong(group) * p.group_size;
  device T* orow = out + ulong(row) * p.out_stride + ulong(group) * p.group_size;
  for (uint i = tid; i < p.group_size; i += tptg) {
    buf[i] = float(xrow[i]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  fht_group(buf, tmp, p.group_size, tid, tptg);
  for (uint i = tid; i < p.group_size; i += tptg) {
    orow[i] = T(buf[i]);
  }
}

kernel void rotate_hadamard_rows_fp16(
    device const half* x [[buffer(0)]],
    device half* out [[buffer(1)]],
    constant RotateParams& p [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    threadgroup float* smem [[threadgroup(0)]]) {
  const uint n_groups = p.dim / p.group_size;
  rotate_hadamard_group_impl<half>(
      x, out, p, tg % n_groups, tg / n_groups, tid, tptg, smem);
}

kernel void rotate_hadamard_rows_bf16(
    device const bfloat* x [[buffer(0)]],
    device bfloat* out [[buffer(1)]],
    constant RotateParams& p [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    threadgroup float* smem [[threadgroup(0)]]) {
  const uint n_groups = p.dim / p.group_size;
  rotate_hadamard_group_impl<bfloat>(
      x, out, p, tg % n_groups, tg / n_groups, tid, tptg, smem);
}

// Fused W4A4 linear: C = (A_int4 * a_scale) @ (W_int4 * w_scale).T
// A and W are signed nibble-packed along K (low nibble = even col).
struct W4A4MmParams {
  uint M;
  uint N;
  uint K; // full feature dim (even)
  uint a_packed_stride;
  uint w_packed_stride;
  uint out_stride;
};

constant uint W4_TM = 8u;
constant uint W4_TN = 8u;
constant uint W4_TK = 64u; // features per stage (32 packed bytes)

inline int w4_nibble_signed(uint n) {
  int v = int(n & 0xFu);
  return v >= 8 ? v - 16 : v;
}

kernel void convrot_w4a4_linear_fused(
    device const char* a_packed [[buffer(0)]],
    device const char* w_packed [[buffer(1)]],
    device const float* a_scales [[buffer(2)]],
    device const float* w_scales [[buffer(3)]],
    device half* out [[buffer(4)]],
    device const half* bias [[buffer(5)]],
    constant W4A4MmParams& p [[buffer(6)]],
    constant uint& has_bias [[buffer(7)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]) {
  threadgroup float As[W4_TM * W4_TK];
  threadgroup float Ws[W4_TN * W4_TK];

  const uint m0 = tgid.y * W4_TM;
  const uint n0 = tgid.x * W4_TN;
  const uint mi = tid.y;
  const uint ni = tid.x;
  const uint m = m0 + mi;
  const uint n = n0 + ni;
  float acc = 0.0f;
  const uint tlinear = mi * W4_TN + ni;
  const uint tptg = W4_TM * W4_TN;
  for (uint k0 = 0u; k0 < p.K; k0 += W4_TK) {
    for (uint i = tlinear; i < W4_TM * W4_TK; i += tptg) {
      const uint row = i / W4_TK;
      const uint kk = i % W4_TK;
      const uint mm = m0 + row;
      float v = 0.0f;
      if (mm < p.M && (k0 + kk) < p.K) {
        const uint kb = k0 + kk;
        const uchar b = uchar(a_packed[ulong(mm) * p.a_packed_stride + kb / 2u]);
        const int nib = ((kb & 1u) == 0u) ? w4_nibble_signed(b) : w4_nibble_signed(b >> 4);
        v = float(nib) * a_scales[mm];
      }
      As[i] = v;
    }
    for (uint i = tlinear; i < W4_TN * W4_TK; i += tptg) {
      const uint row = i / W4_TK;
      const uint kk = i % W4_TK;
      const uint nn = n0 + row;
      float v = 0.0f;
      if (nn < p.N && (k0 + kk) < p.K) {
        const uint kb = k0 + kk;
        const uchar b = uchar(w_packed[ulong(nn) * p.w_packed_stride + kb / 2u]);
        const int nib = ((kb & 1u) == 0u) ? w4_nibble_signed(b) : w4_nibble_signed(b >> 4);
        v = float(nib) * w_scales[nn];
      }
      Ws[i] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (m < p.M && n < p.N) {
      for (uint k = 0u; k < W4_TK; ++k) {
        acc = fma(As[mi * W4_TK + k], Ws[ni * W4_TK + k], acc);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (m < p.M && n < p.N) {
    if (has_bias != 0u) {
      acc += float(bias[n]);
    }
    out[ulong(m) * p.out_stride + n] = half(acc);
  }
}

// Scale epilogue for W8A8: out = (acc_i32 * (x_scale * w_scale)).to(half) + bias
struct Int8EpilogueParams {
  uint M;
  uint N;
  uint acc_stride;
  uint out_stride;
  uint has_bias;
  uint has_residual;
};

kernel void int8_mm_epilogue_fp16(
    device const int* acc [[buffer(0)]],
    device const float* x_scale [[buffer(1)]],
    device const float* w_scale [[buffer(2)]],
    device const half* bias [[buffer(3)]],
    device const half* residual [[buffer(4)]],
    device const half* residual_scale [[buffer(5)]],
    device half* out [[buffer(6)]],
    constant Int8EpilogueParams& p [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint n = gid.x;
  const uint m = gid.y;
  if (m >= p.M || n >= p.N) {
    return;
  }
  float v = float(acc[ulong(m) * p.acc_stride + n]) * x_scale[m] * w_scale[n];
  if (p.has_bias != 0u) {
    v += float(bias[n]);
  }
  // residual + residual_scale * linear_out (matches apply_residual / addcmul)
  if (p.has_residual != 0u) {
    v = float(residual[ulong(m) * p.out_stride + n]) + v * float(residual_scale[n]);
  }
  out[ulong(m) * p.out_stride + n] = half(v);
}

kernel void int8_mm_epilogue_bf16(
    device const int* acc [[buffer(0)]],
    device const float* x_scale [[buffer(1)]],
    device const float* w_scale [[buffer(2)]],
    device const bfloat* bias [[buffer(3)]],
    device const bfloat* residual [[buffer(4)]],
    device const bfloat* residual_scale [[buffer(5)]],
    device bfloat* out [[buffer(6)]],
    constant Int8EpilogueParams& p [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint n = gid.x;
  const uint m = gid.y;
  if (m >= p.M || n >= p.N) {
    return;
  }
  float v = float(acc[ulong(m) * p.acc_stride + n]) * x_scale[m] * w_scale[n];
  if (p.has_bias != 0u) {
    v += float(bias[n]);
  }
  if (p.has_residual != 0u) {
    v = float(residual[ulong(m) * p.out_stride + n]) + v * float(residual_scale[n]);
  }
  out[ulong(m) * p.out_stride + n] = bfloat(v);
}
