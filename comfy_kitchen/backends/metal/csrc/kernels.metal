#include <metal_stdlib>
using namespace metal;

// Apple GPU SIMD width is 32. Prefer simdgroup reductions + vector loads.

struct RmsNormParams {
  uint rows;
  uint dim;
  float eps;
  uint x_stride;
  uint out_stride;
  uint has_weight;
  uint has_residual;
};

struct AdaLNParams {
  uint rows;
  uint dim;
  float eps;
  uint x_stride;
  uint scale_stride; // 0 => broadcast row 0
  uint shift_stride;
  uint out_stride;
  uint use_rms; // 1 = RMSNorm, 0 = LayerNorm
};

struct RopePairParams {
  uint B;
  uint H;
  uint T;
  uint pairs;
  uint x_stride_b;
  uint x_stride_h;
  uint x_stride_t;
  uint f_stride_b;
  uint f_stride_h;  // 0 => broadcast over heads
  uint f_stride_t;
  uint f_stride_pair;
};

struct RopeSplitParams {
  uint B;
  uint H;
  uint T;
  uint half_d;
  uint x_stride_b;
  uint x_stride_h;
  uint x_stride_t;
  uint f_stride_b;
  uint f_stride_h;  // 0 => broadcast over heads
  uint f_stride_t;
  uint f_stride_pair;
};

struct QuantRowParams {
  uint rows;
  uint dim;
  uint x_stride;
  uint q_stride;
  uint stochastic; // 1 => floor(x + U), 0 => round(x)
  uint seed_lo;
  uint seed_hi;
  uint _pad;
};

inline uint pcg_hash(uint x) {
  const uint state = x * 747796405u + 2891336453u;
  const uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

inline float stochastic_rng_value(ulong idx, ulong seed) {
  const ulong key = idx + seed;
  const uint folded = uint(key) ^ uint(key >> 32);
  return float(pcg_hash(folded) >> 8) * 0x1.0p-24f;
}

inline ulong quant_seed(constant QuantRowParams& p) {
  return (ulong(p.seed_hi) << 32) | ulong(p.seed_lo);
}

inline int quantize_int8_value(float scaled, bool stochastic, ulong idx, ulong seed) {
  float q = stochastic ? floor(scaled + stochastic_rng_value(idx, seed)) : round(scaled);
  return clamp(int(q), -128, 127);
}

inline int quantize_int4_value(float scaled, bool stochastic, ulong idx, ulong seed) {
  float q = stochastic ? floor(scaled + stochastic_rng_value(idx, seed)) : round(scaled);
  return clamp(int(q), -7, 7);
}

struct RmsRopeParams {
  uint B;
  uint H;
  uint T;
  uint D;
  float eps;
  uint x_stride_b;
  uint x_stride_h;
  uint x_stride_t;
  uint out_stride_b;
  uint out_stride_h;
  uint out_stride_t;
  uint f_stride_b;
  uint f_stride_h;  // 0 => broadcast over heads
  uint f_stride_t;
  uint f_stride_pair;
  uint rot_dim;     // 0 => full D
  uint split_half;  // 0 interleaved, 1 split-half
  uint bnhd;        // 1 => inv_rms row order is (b,t,h)
};

struct Fp8Params {
  uint numel;
  uint fmt; // 0 = e4m3fn, 1 = e5m2
};

// ---- RMS / residual RMS: simdgroup tree + threadgroup spill ----

template <typename T>
void rms_norm_simd(
    device const T* x,
    device const T* weight,
    device const T* residual,
    device T* out,
    constant RmsNormParams& p,
    uint row,
    uint tid,
    uint tptg,
    threadgroup float* red) {
  if (row >= p.rows) {
    return;
  }
  device const T* xrow = x + ulong(row) * p.x_stride;
  device T* orow = out + ulong(row) * p.out_stride;

  float partial = 0.0f;
  // Each thread owns a contiguous float4 chunk when possible (coalesced + SIMD-friendly).
  const uint vec_count = p.dim / 4u;
  for (uint i = tid; i < vec_count; i += tptg) {
    const uint d = i * 4u;
    float4 v;
    v.x = float(xrow[d + 0u]);
    v.y = float(xrow[d + 1u]);
    v.z = float(xrow[d + 2u]);
    v.w = float(xrow[d + 3u]);
    if (p.has_residual) {
      const ulong base = ulong(row) * p.x_stride;
      v.x += float(residual[base + d + 0u]);
      v.y += float(residual[base + d + 1u]);
      v.z += float(residual[base + d + 2u]);
      v.w += float(residual[base + d + 3u]);
    }
    partial += dot(v, v);
  }
  for (uint s = vec_count * 4u + tid; s < p.dim; s += tptg) {
    float v = float(xrow[s]);
    if (p.has_residual) {
      v += float(residual[ulong(row) * p.x_stride + s]);
    }
    partial += v * v;
  }

  // SIMD-group reduce, then threadgroup reduce across groups.
  partial = simd_sum(partial);
  const uint simd_lane = tid % 32u;
  const uint simd_id = tid / 32u;
  if (simd_lane == 0u) {
    red[simd_id] = partial;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint num_simd = (tptg + 31u) / 32u;
  if (tid == 0u) {
    float total = 0.0f;
    for (uint i = 0; i < num_simd; ++i) {
      total += red[i];
    }
    red[0] = rsqrt(total / float(p.dim) + p.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv_rms = red[0];

  for (uint s = tid; s < p.dim; s += tptg) {
    float v = float(xrow[s]);
    if (p.has_residual) {
      v += float(residual[ulong(row) * p.x_stride + s]);
    }
    v *= inv_rms;
    if (p.has_weight) {
      v *= float(weight[s]);
    }
    orow[s] = T(v);
  }
}

kernel void residual_rms_norm_fp16(
    device const half* x [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device const half* residual [[buffer(2)]],
    device half* out [[buffer(3)]],
    constant RmsNormParams& p [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]]) {
  threadgroup float red[8];
  rms_norm_simd<half>(x, weight, residual, out, p, row, tid, tptg, red);
}

kernel void residual_rms_norm_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device const bfloat* residual [[buffer(2)]],
    device bfloat* out [[buffer(3)]],
    constant RmsNormParams& p [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]]) {
  threadgroup float red[8];
  rms_norm_simd<bfloat>(x, weight, residual, out, p, row, tid, tptg, red);
}

// residual + gate * rms_norm(activation, weight)
template <typename T>
void rms_gated_residual_impl(
    device const T* x,
    device const T* weight,
    device const T* residual,
    device const T* gate,
    device T* out,
    constant RmsNormParams& p,
    uint row,
    uint tid,
    uint tptg,
    threadgroup float* red) {
  if (row >= p.rows) {
    return;
  }
  device const T* xrow = x + ulong(row) * p.x_stride;
  device const T* rrow = residual + ulong(row) * p.x_stride;
  device T* orow = out + ulong(row) * p.out_stride;

  float partial = 0.0f;
  const uint vec_count = p.dim / 4u;
  for (uint i = tid; i < vec_count; i += tptg) {
    const uint d = i * 4u;
    float4 v = float4(
        float(xrow[d]), float(xrow[d + 1u]), float(xrow[d + 2u]), float(xrow[d + 3u]));
    partial += dot(v, v);
  }
  for (uint s = vec_count * 4u + tid; s < p.dim; s += tptg) {
    float v = float(xrow[s]);
    partial += v * v;
  }
  partial = simd_sum(partial);
  const uint simd_lane = tid % 32u;
  const uint simd_id = tid / 32u;
  if (simd_lane == 0u) {
    red[simd_id] = partial;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float total = 0.0f;
    const uint num_simd = (tptg + 31u) / 32u;
    for (uint i = 0; i < num_simd; ++i) {
      total += red[i];
    }
    red[0] = rsqrt(total / float(p.dim) + p.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv_rms = red[0];
  for (uint s = tid; s < p.dim; s += tptg) {
    float n = float(xrow[s]) * inv_rms * float(weight[s]);
    orow[s] = T(float(rrow[s]) + float(gate[s]) * n);
  }
}

kernel void rms_gated_residual_fp16(
    device const half* x [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device const half* residual [[buffer(2)]],
    device const half* gate [[buffer(3)]],
    device half* out [[buffer(4)]],
    constant RmsNormParams& p [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]]) {
  threadgroup float red[8];
  rms_gated_residual_impl<half>(x, weight, residual, gate, out, p, row, tid, tptg, red);
}

kernel void rms_gated_residual_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device const bfloat* residual [[buffer(2)]],
    device const bfloat* gate [[buffer(3)]],
    device bfloat* out [[buffer(4)]],
    constant RmsNormParams& p [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]]) {
  threadgroup float red[8];
  rms_gated_residual_impl<bfloat>(x, weight, residual, gate, out, p, row, tid, tptg, red);
}

// ---- Fused AdaLN / RMS-AdaLN ----

template <typename T>
void adaln_simd(
    device const T* x,
    device const T* scale,
    device const T* shift,
    device T* out,
    constant AdaLNParams& p,
    uint row,
    uint tid,
    uint tptg,
    threadgroup float* red) {
  if (row >= p.rows) {
    return;
  }
  device const T* xrow = x + ulong(row) * p.x_stride;
  device T* orow = out + ulong(row) * p.out_stride;
  const uint scale_row = p.scale_stride == 0u ? 0u : row;
  const uint shift_row = p.shift_stride == 0u ? 0u : row;
  device const T* srow = scale + ulong(scale_row) * (p.scale_stride == 0u ? p.dim : p.scale_stride);
  device const T* trow = shift + ulong(shift_row) * (p.shift_stride == 0u ? p.dim : p.shift_stride);

  const uint vec_count = p.dim / 4u;
  const uint simd_lane = tid % 32u;
  const uint simd_id = tid / 32u;
  const uint num_simd = (tptg + 31u) / 32u;

  if (p.use_rms) {
    float partial = 0.0f;
    for (uint i = tid; i < vec_count; i += tptg) {
      const uint d = i * 4u;
      float4 v = float4(
          float(xrow[d]), float(xrow[d + 1u]), float(xrow[d + 2u]), float(xrow[d + 3u]));
      partial += dot(v, v);
    }
    for (uint s = vec_count * 4u + tid; s < p.dim; s += tptg) {
      float v = float(xrow[s]);
      partial += v * v;
    }
    partial = simd_sum(partial);
    if (simd_lane == 0u) {
      red[simd_id] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
      float total = 0.0f;
      for (uint i = 0; i < num_simd; ++i) {
        total += red[i];
      }
      red[0] = rsqrt(total / float(p.dim) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = red[0];
    for (uint i = tid; i < vec_count; i += tptg) {
      const uint d = i * 4u;
      float4 v = float4(
          float(xrow[d]), float(xrow[d + 1u]), float(xrow[d + 2u]), float(xrow[d + 3u]));
      float4 sc = float4(
          float(srow[d]), float(srow[d + 1u]), float(srow[d + 2u]), float(srow[d + 3u]));
      float4 sh = float4(
          float(trow[d]), float(trow[d + 1u]), float(trow[d + 2u]), float(trow[d + 3u]));
      v = v * inv;
      v = v * (1.0f + sc) + sh;
      orow[d] = T(v.x);
      orow[d + 1u] = T(v.y);
      orow[d + 2u] = T(v.z);
      orow[d + 3u] = T(v.w);
    }
    for (uint s = vec_count * 4u + tid; s < p.dim; s += tptg) {
      float v = float(xrow[s]) * inv;
      v = v * (1.0f + float(srow[s])) + float(trow[s]);
      orow[s] = T(v);
    }
  } else {
    // LayerNorm: mean then var via two passes, simdgroup each.
    float partial = 0.0f;
    for (uint i = tid; i < vec_count; i += tptg) {
      const uint d = i * 4u;
      float4 v = float4(
          float(xrow[d]), float(xrow[d + 1u]), float(xrow[d + 2u]), float(xrow[d + 3u]));
      partial += v.x + v.y + v.z + v.w;
    }
    for (uint s = vec_count * 4u + tid; s < p.dim; s += tptg) {
      partial += float(xrow[s]);
    }
    partial = simd_sum(partial);
    if (simd_lane == 0u) {
      red[simd_id] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
      float total = 0.0f;
      for (uint i = 0; i < num_simd; ++i) {
        total += red[i];
      }
      red[0] = total / float(p.dim);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float mean = red[0];

    float var_partial = 0.0f;
    for (uint i = tid; i < vec_count; i += tptg) {
      const uint d = i * 4u;
      float4 v = float4(
          float(xrow[d]), float(xrow[d + 1u]), float(xrow[d + 2u]), float(xrow[d + 3u]));
      v -= mean;
      var_partial += dot(v, v);
    }
    for (uint s = vec_count * 4u + tid; s < p.dim; s += tptg) {
      float d = float(xrow[s]) - mean;
      var_partial += d * d;
    }
    var_partial = simd_sum(var_partial);
    if (simd_lane == 0u) {
      red[simd_id] = var_partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
      float total = 0.0f;
      for (uint i = 0; i < num_simd; ++i) {
        total += red[i];
      }
      red[0] = rsqrt(total / float(p.dim) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = red[0];
    for (uint i = tid; i < vec_count; i += tptg) {
      const uint d = i * 4u;
      float4 v = float4(
          float(xrow[d]), float(xrow[d + 1u]), float(xrow[d + 2u]), float(xrow[d + 3u]));
      float4 sc = float4(
          float(srow[d]), float(srow[d + 1u]), float(srow[d + 2u]), float(srow[d + 3u]));
      float4 sh = float4(
          float(trow[d]), float(trow[d + 1u]), float(trow[d + 2u]), float(trow[d + 3u]));
      v = (v - mean) * inv;
      v = v * (1.0f + sc) + sh;
      orow[d] = T(v.x);
      orow[d + 1u] = T(v.y);
      orow[d + 2u] = T(v.z);
      orow[d + 3u] = T(v.w);
    }
    for (uint s = vec_count * 4u + tid; s < p.dim; s += tptg) {
      float v = (float(xrow[s]) - mean) * inv;
      v = v * (1.0f + float(srow[s])) + float(trow[s]);
      orow[s] = T(v);
    }
  }
}

kernel void adaln_fp16(
    device const half* x [[buffer(0)]],
    device const half* scale [[buffer(1)]],
    device const half* shift [[buffer(2)]],
    device half* out [[buffer(3)]],
    constant AdaLNParams& p [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]]) {
  threadgroup float red[8];
  adaln_simd<half>(x, scale, shift, out, p, row, tid, tptg, red);
}

kernel void adaln_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* scale [[buffer(1)]],
    device const bfloat* shift [[buffer(2)]],
    device bfloat* out [[buffer(3)]],
    constant AdaLNParams& p [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]]) {
  threadgroup float red[8];
  adaln_simd<bfloat>(x, scale, shift, out, p, row, tid, tptg, red);
}

// ---- RoPE: one thread per complex pair, float2-style math ----

template <typename T>
void apply_rope1_impl(
    device T* x,
    device const float* freqs,
    constant RopePairParams& p,
    uint3 gid) {
  const uint pair = gid.x;
  const uint t = gid.y;
  const uint bh = gid.z;
  if (pair >= p.pairs || t >= p.T) {
    return;
  }
  const uint b = bh / max(p.H, 1u);
  const uint h = bh % max(p.H, 1u);
  if (b >= p.B) {
    return;
  }

  device T* x_row = x + ulong(b) * p.x_stride_b + ulong(h) * p.x_stride_h + ulong(t) * p.x_stride_t;
  const device float* f = freqs + ulong(b) * p.f_stride_b + ulong(h) * p.f_stride_h +
                          ulong(t) * p.f_stride_t + ulong(pair) * p.f_stride_pair;

  // Contiguous (2,2) row-major: c00,c01,c10,c11
  const float2 col0 = float2(f[0], f[2]);
  const float2 col1 = float2(f[1], f[3]);
  const uint d0 = pair * 2u;
  const float2 xv = float2(float(x_row[d0]), float(x_row[d0 + 1]));
  const float2 y = col0 * xv.x + col1 * xv.y;
  x_row[d0] = T(y.x);
  x_row[d0 + 1] = T(y.y);
}

kernel void apply_rope1_inplace_fp16(
    device half* x [[buffer(0)]],
    device const float* freqs [[buffer(1)]],
    constant RopePairParams& p [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]) {
  apply_rope1_impl<half>(x, freqs, p, gid);
}

kernel void apply_rope1_inplace_bf16(
    device bfloat* x [[buffer(0)]],
    device const float* freqs [[buffer(1)]],
    constant RopePairParams& p [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]) {
  apply_rope1_impl<bfloat>(x, freqs, p, gid);
}

template <typename T>
void apply_rope_split_impl(
    device T* x,
    device const float* freqs,
    constant RopeSplitParams& p,
    uint3 gid) {
  const uint i = gid.x;
  const uint t = gid.y;
  const uint bh = gid.z;
  if (i >= p.half_d || t >= p.T) {
    return;
  }
  const uint b = bh / max(p.H, 1u);
  const uint h = bh % max(p.H, 1u);
  if (b >= p.B) {
    return;
  }

  device T* x_row = x + ulong(b) * p.x_stride_b + ulong(h) * p.x_stride_h + ulong(t) * p.x_stride_t;
  const device float* f = freqs + ulong(b) * p.f_stride_b + ulong(h) * p.f_stride_h +
                          ulong(t) * p.f_stride_t + ulong(i) * p.f_stride_pair;
  const float2 col0 = float2(f[0], f[2]);
  const float2 col1 = float2(f[1], f[3]);
  const float2 xv = float2(float(x_row[i]), float(x_row[i + p.half_d]));
  const float2 y = col0 * xv.x + col1 * xv.y;
  x_row[i] = T(y.x);
  x_row[i + p.half_d] = T(y.y);
}

kernel void apply_rope_split_inplace_fp16(
    device half* x [[buffer(0)]],
    device const float* freqs [[buffer(1)]],
    constant RopeSplitParams& p [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]) {
  apply_rope_split_impl<half>(x, freqs, p, gid);
}

kernel void apply_rope_split_inplace_bf16(
    device bfloat* x [[buffer(0)]],
    device const float* freqs [[buffer(1)]],
    constant RopeSplitParams& p [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]) {
  apply_rope_split_impl<bfloat>(x, freqs, p, gid);
}

// ---- INT8 row-wise quantize: simdgroup absmax ----

template <typename T>
void quantize_int8_rowwise_impl(
    device const T* x,
    device char* q,
    device float* scale,
    constant QuantRowParams& p,
    uint row,
    uint tid,
    uint tptg,
    threadgroup float* red) {
  if (row >= p.rows) {
    return;
  }
  device const T* xrow = x + ulong(row) * p.x_stride;
  device char* qrow = q + ulong(row) * p.q_stride;

  float amax = 0.0f;
  const uint vec_count = p.dim / 4u;
  for (uint i = tid; i < vec_count; i += tptg) {
    const uint d = i * 4u;
    float4 v = float4(
        float(xrow[d]), float(xrow[d + 1u]), float(xrow[d + 2u]), float(xrow[d + 3u]));
    v = fabs(v);
    amax = max(amax, max(max(v.x, v.y), max(v.z, v.w)));
  }
  for (uint s = vec_count * 4u + tid; s < p.dim; s += tptg) {
    amax = max(amax, fabs(float(xrow[s])));
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
    for (uint i = 0; i < num_simd; ++i) {
      m = max(m, red[i]);
    }
    float s = max(m / 127.0f, 1e-30f);
    scale[row] = s;
    red[0] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float s = red[0];
  const float inv = 1.0f / s;
  const bool stochastic = p.stochastic != 0u;
  const ulong seed = quant_seed(p);
  const ulong row_base = ulong(row) * ulong(p.dim);
  for (uint i = tid; i < vec_count; i += tptg) {
    const uint d = i * 4u;
    float4 v = float4(
        float(xrow[d]), float(xrow[d + 1u]), float(xrow[d + 2u]), float(xrow[d + 3u]));
    v *= inv;
    qrow[d] = char(quantize_int8_value(v.x, stochastic, row_base + d, seed));
    qrow[d + 1u] = char(quantize_int8_value(v.y, stochastic, row_base + d + 1u, seed));
    qrow[d + 2u] = char(quantize_int8_value(v.z, stochastic, row_base + d + 2u, seed));
    qrow[d + 3u] = char(quantize_int8_value(v.w, stochastic, row_base + d + 3u, seed));
  }
  for (uint d = vec_count * 4u + tid; d < p.dim; d += tptg) {
    float v = float(xrow[d]) * inv;
    qrow[d] = char(quantize_int8_value(v, stochastic, row_base + d, seed));
  }
}

kernel void quantize_int8_rowwise_fp16(
    device const half* x [[buffer(0)]],
    device char* q [[buffer(1)]],
    device float* scale [[buffer(2)]],
    constant QuantRowParams& p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]]) {
  threadgroup float red[8];
  quantize_int8_rowwise_impl<half>(x, q, scale, p, row, tid, tptg, red);
}

kernel void quantize_int8_rowwise_bf16(
    device const bfloat* x [[buffer(0)]],
    device char* q [[buffer(1)]],
    device float* scale [[buffer(2)]],
    constant QuantRowParams& p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]]) {
  threadgroup float red[8];
  quantize_int8_rowwise_impl<bfloat>(x, q, scale, p, row, tid, tptg, red);
}

// Signed INT4 rowwise absmax quant + nibble pack (low = even col). Emission [-7, 7].
template <typename T>
void quantize_int4_rowwise_impl(
    device const T* x,
    device char* packed,
    device float* scale,
    constant QuantRowParams& p,
    uint row,
    uint tid,
    uint tptg,
    threadgroup float* red) {
  if (row >= p.rows) {
    return;
  }
  device const T* xrow = x + ulong(row) * p.x_stride;
  device char* prow = packed + ulong(row) * p.q_stride;
  const uint pairs = p.dim / 2u;

  float amax = 0.0f;
  for (uint i = tid; i < p.dim; i += tptg) {
    amax = max(amax, fabs(float(xrow[i])));
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
    for (uint i = 0; i < num_simd; ++i) {
      m = max(m, red[i]);
    }
    float s = max(m / 7.0f, 1e-10f);
    scale[row] = s;
    red[0] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = 1.0f / red[0];
  const bool stochastic = p.stochastic != 0u;
  const ulong seed = quant_seed(p);
  const ulong row_base = ulong(row) * ulong(p.dim);
  for (uint i = tid; i < pairs; i += tptg) {
    const uint d = i * 2u;
    int lo = quantize_int4_value(float(xrow[d]) * inv, stochastic, row_base + d, seed);
    int hi = quantize_int4_value(float(xrow[d + 1u]) * inv, stochastic, row_base + d + 1u, seed);
    prow[i] = char((lo & 0xF) | ((hi & 0xF) << 4));
  }
}

kernel void quantize_int4_rowwise_fp16(
    device const half* x [[buffer(0)]],
    device char* packed [[buffer(1)]],
    device float* scale [[buffer(2)]],
    constant QuantRowParams& p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]]) {
  threadgroup float red[8];
  quantize_int4_rowwise_impl<half>(x, packed, scale, p, row, tid, tptg, red);
}

kernel void quantize_int4_rowwise_bf16(
    device const bfloat* x [[buffer(0)]],
    device char* packed [[buffer(1)]],
    device float* scale [[buffer(2)]],
    constant QuantRowParams& p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]]) {
  threadgroup float red[8];
  quantize_int4_rowwise_impl<bfloat>(x, packed, scale, p, row, tid, tptg, red);
}

inline int unpack_signed_nibble(uint nibble) {
  int v = int(nibble & 0xFu);
  return v >= 8 ? v - 16 : v;
}

template <typename T>
void dequantize_int4_rowwise_impl(
    device const char* packed,
    device const float* scale,
    device T* out,
    constant QuantRowParams& p,
    uint row,
    uint tid,
    uint tptg) {
  if (row >= p.rows) {
    return;
  }
  device const char* prow = packed + ulong(row) * p.q_stride;
  device T* orow = out + ulong(row) * p.x_stride;
  const float s = scale[row];
  const uint pairs = p.dim / 2u;
  for (uint i = tid; i < pairs; i += tptg) {
    const uchar b = uchar(prow[i]);
    const int lo = unpack_signed_nibble(b);
    const int hi = unpack_signed_nibble(b >> 4);
    const uint d = i * 2u;
    orow[d] = T(float(lo) * s);
    orow[d + 1u] = T(float(hi) * s);
  }
}

kernel void dequantize_int4_rowwise_fp16(
    device const char* packed [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant QuantRowParams& p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]]) {
  dequantize_int4_rowwise_impl<half>(packed, scale, out, p, row, tid, tptg);
}

kernel void dequantize_int4_rowwise_bf16(
    device const char* packed [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant QuantRowParams& p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]]) {
  dequantize_int4_rowwise_impl<bfloat>(packed, scale, out, p, row, tid, tptg);
}

kernel void dequantize_int8_simple_fp32(
    device const char* q [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant QuantRowParams& p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]]) {
  if (row >= p.rows) {
    return;
  }
  device const char* qrow = q + ulong(row) * p.q_stride;
  device float* orow = out + ulong(row) * p.x_stride;
  const float s = scale[min(row, p.rows - 1u)]; // caller packs broadcast scales
  const uint vec_count = p.dim / 4u;
  for (uint i = tid; i < vec_count; i += tptg) {
    const uint d = i * 4u;
    float4 v = float4(
        float(qrow[d]), float(qrow[d + 1u]), float(qrow[d + 2u]), float(qrow[d + 3u]));
    v *= s;
    orow[d] = v.x;
    orow[d + 1u] = v.y;
    orow[d + 2u] = v.z;
    orow[d + 3u] = v.w;
  }
  for (uint d = vec_count * 4u + tid; d < p.dim; d += tptg) {
    orow[d] = float(qrow[d]) * s;
  }
}

// ---- Fused RMSNorm + RoPE (two device passes, one activation write) ----
// Pass A: per-(b,h,t) inv_rms into a tiny float buffer (no D-wide store).
// Pass B: scale * rope in registers → single write of the activation.

struct RmsInvParams {
  uint rows;
  uint dim;
  float eps;
  uint x_stride;
};

template <typename T>
void rms_inv_rows(
    device const T* x,
    device float* inv_rms,
    constant RmsInvParams& p,
    uint row,
    uint tid,
    uint tptg,
    threadgroup float* red) {
  if (row >= p.rows) {
    return;
  }
  device const T* xrow = x + ulong(row) * p.x_stride;
  float partial = 0.0f;
  const uint vec_count = p.dim / 4u;
  for (uint i = tid; i < vec_count; i += tptg) {
    const uint d = i * 4u;
    float4 v = float4(
        float(xrow[d]), float(xrow[d + 1u]), float(xrow[d + 2u]), float(xrow[d + 3u]));
    partial += dot(v, v);
  }
  for (uint s = vec_count * 4u + tid; s < p.dim; s += tptg) {
    float v = float(xrow[s]);
    partial += v * v;
  }
  partial = simd_sum(partial);
  const uint simd_lane = tid % 32u;
  const uint simd_id = tid / 32u;
  if (simd_lane == 0u) {
    red[simd_id] = partial;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float total = 0.0f;
    const uint num_simd = (tptg + 31u) / 32u;
    for (uint i = 0; i < num_simd; ++i) {
      total += red[i];
    }
    inv_rms[row] = rsqrt(total / float(p.dim) + p.eps);
  }
}

kernel void rms_inv_rows_fp16(
    device const half* x [[buffer(0)]],
    device float* inv_rms [[buffer(1)]],
    constant RmsInvParams& p [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]]) {
  threadgroup float red[8];
  rms_inv_rows<half>(x, inv_rms, p, row, tid, tptg, red);
}

kernel void rms_inv_rows_bf16(
    device const bfloat* x [[buffer(0)]],
    device float* inv_rms [[buffer(1)]],
    constant RmsInvParams& p [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]]) {
  threadgroup float red[8];
  rms_inv_rows<bfloat>(x, inv_rms, p, row, tid, tptg, red);
}

template <typename T>
void rms_rope_apply(
    device const T* x,
    device const T* weight,
    device const float* inv_rms,
    device T* out,
    device const float* freqs,
    constant RmsRopeParams& p,
    uint3 gid) {
  const uint pair_or_i = gid.x;
  const uint t = gid.y;
  const uint bh = gid.z;
  const uint b = bh / max(p.H, 1u);
  const uint h = bh % max(p.H, 1u);
  if (b >= p.B || t >= p.T) {
    return;
  }
  const uint row = p.bnhd ? (b * (p.T * p.H) + t * p.H + h) : (b * (p.H * p.T) + h * p.T + t);
  const float inv = inv_rms[row];
  device const T* x_row =
      x + ulong(b) * p.x_stride_b + ulong(h) * p.x_stride_h + ulong(t) * p.x_stride_t;
  device T* o_row =
      out + ulong(b) * p.out_stride_b + ulong(h) * p.out_stride_h + ulong(t) * p.out_stride_t;
  const uint rot = p.rot_dim == 0u ? p.D : p.rot_dim;
  const device float* f = freqs + ulong(b) * p.f_stride_b + ulong(h) * p.f_stride_h +
                          ulong(t) * p.f_stride_t + ulong(pair_or_i) * p.f_stride_pair;

  if (p.split_half == 0u) {
    const uint pairs = rot / 2u;
    if (pair_or_i >= pairs) {
      return;
    }
    const uint d0 = pair_or_i * 2u;
    float2 xv = float2(float(x_row[d0]), float(x_row[d0 + 1u])) * inv;
    xv.x *= float(weight[d0]);
    xv.y *= float(weight[d0 + 1u]);
    const float2 y = float2(f[0], f[2]) * xv.x + float2(f[1], f[3]) * xv.y;
    o_row[d0] = T(y.x);
    o_row[d0 + 1u] = T(y.y);
  } else {
    const uint half_d = rot / 2u;
    if (pair_or_i >= half_d) {
      return;
    }
    float2 xv = float2(float(x_row[pair_or_i]), float(x_row[pair_or_i + half_d])) * inv;
    xv.x *= float(weight[pair_or_i]);
    xv.y *= float(weight[pair_or_i + half_d]);
    const float2 y = float2(f[0], f[2]) * xv.x + float2(f[1], f[3]) * xv.y;
    o_row[pair_or_i] = T(y.x);
    o_row[pair_or_i + half_d] = T(y.y);
  }
}

// Tail pass-through for partial rotary: one thread per tail element.
template <typename T>
void rms_rope_tail(
    device const T* x,
    device const T* weight,
    device const float* inv_rms,
    device T* out,
    constant RmsRopeParams& p,
    uint3 gid) {
  const uint rot = p.rot_dim == 0u ? p.D : p.rot_dim;
  const uint s = rot + gid.x; // absolute dim index in [rot, D)
  const uint t = gid.y;
  const uint bh = gid.z;
  const uint b = bh / max(p.H, 1u);
  const uint h = bh % max(p.H, 1u);
  if (b >= p.B || t >= p.T || s >= p.D) {
    return;
  }
  const uint row = p.bnhd ? (b * (p.T * p.H) + t * p.H + h) : (b * (p.H * p.T) + h * p.T + t);
  const float inv = inv_rms[row];
  device const T* x_row =
      x + ulong(b) * p.x_stride_b + ulong(h) * p.x_stride_h + ulong(t) * p.x_stride_t;
  device T* o_row =
      out + ulong(b) * p.out_stride_b + ulong(h) * p.out_stride_h + ulong(t) * p.out_stride_t;
  o_row[s] = T(float(x_row[s]) * inv * float(weight[s]));
}

kernel void rms_rope_apply_fp16(
    device const half* x [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device const float* inv_rms [[buffer(2)]],
    device half* out [[buffer(3)]],
    device const float* freqs [[buffer(4)]],
    constant RmsRopeParams& p [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]) {
  rms_rope_apply<half>(x, weight, inv_rms, out, freqs, p, gid);
}

kernel void rms_rope_apply_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device const float* inv_rms [[buffer(2)]],
    device bfloat* out [[buffer(3)]],
    device const float* freqs [[buffer(4)]],
    constant RmsRopeParams& p [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]) {
  rms_rope_apply<bfloat>(x, weight, inv_rms, out, freqs, p, gid);
}

kernel void rms_rope_tail_fp16(
    device const half* x [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device const float* inv_rms [[buffer(2)]],
    device half* out [[buffer(3)]],
    constant RmsRopeParams& p [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]) {
  rms_rope_tail<half>(x, weight, inv_rms, out, p, gid);
}

kernel void rms_rope_tail_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device const float* inv_rms [[buffer(2)]],
    device bfloat* out [[buffer(3)]],
    constant RmsRopeParams& p [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]) {
  rms_rope_tail<bfloat>(x, weight, inv_rms, out, p, gid);
}

// ---- FP8 per-tensor quant / dequant (software E4M3FN / E5M2) ----

constant float kF8E4M3Max = 448.0f;
constant float kF8E5M2Max = 57344.0f;

// Round-ties-to-even float->fp8 packing (matches OCP layout used by torch.float8_*).
inline uchar float_to_fp8(float x, uint fmt) {
  const uint ebits = fmt == 0u ? 4u : 5u;
  const uint mbits = fmt == 0u ? 3u : 2u;
  const float fmax = fmt == 0u ? kF8E4M3Max : kF8E5M2Max;
  const int exp_bias = int((1u << (ebits - 1u)) - 1u);
  const uint max_int = (1u << (ebits + mbits)) - 1u;
  const uint sign_mask = 1u << (ebits + mbits);
  const uint magic_adder = (1u << (23u - mbits - 1u)) - 1u;
  const float max_normal =
      exp2(float(int((1u << ebits) - 1u) - exp_bias)) * (float((1u << (mbits + 1u)) - 1u) / float(1u << mbits));
  const float min_normal = exp2(float(1 - exp_bias));
  const int denorm_exp = (127 - exp_bias) + int(23u - mbits) + 1;
  const uint denorm_mask_int = uint(denorm_exp) << 23u;

  uint bits = as_type<uint>(x);
  const uint sign = bits & 0x80000000u;
  bits ^= sign;
  float ax = as_type<float>(bits);

  // Eager clamps to ±fmax before casting; keep the same contract here.
  ax = min(ax, fmax);

  uchar out;
  if (ax != ax) {
    out = uchar(max_int);
  } else if (ax >= max_normal) {
    // Max finite code (e4m3fn avoids 0x7F NaN; e5m2 avoids Inf/NaN).
    out = uchar(fmt == 0u ? 0x7Eu : 0x7Bu);
  } else if (ax < min_normal) {
    float den = ax + as_type<float>(denorm_mask_int);
    uint di = as_type<uint>(den) - denorm_mask_int;
    out = uchar(di);
  } else {
    uint ni = as_type<uint>(ax);
    const uint mant_odd = (ni >> (23u - mbits)) & 1u;
    const int val_to_add = ((exp_bias - 127) << 23) + int(magic_adder);
    ni = uint(int(ni) + val_to_add);
    ni += mant_odd;
    out = uchar(ni >> (23u - mbits));
  }
  const uchar sign_lp = uchar((sign >> (31u - (ebits + mbits))) & sign_mask);
  return uchar(out | sign_lp);
}

inline float fp8_to_float(uchar in, uint fmt) {
  const uint ebits = fmt == 0u ? 4u : 5u;
  const uint mbits = fmt == 0u ? 3u : 2u;
  const int exp_bias = int((1u << (ebits - 1u)) - 1u);
  const uint sign_mask = 1u << (ebits + mbits);
  const uint exp_mask = ((1u << ebits) - 1u) << mbits;
  const uint mant_mask = (1u << mbits) - 1u;

  const uint sign = (uint(in) & sign_mask) ? 0x80000000u : 0u;
  const uint exp = (uint(in) & exp_mask) >> mbits;
  const uint mant = uint(in) & mant_mask;

  if (exp == 0u) {
    if (mant == 0u) {
      return as_type<float>(sign);
    }
    // subnormal
    float m = float(mant) / float(1u << mbits);
    float v = ldexp(m, 1 - exp_bias);
    return sign ? -v : v;
  }
  // e4m3fn: exp==0xF with mant!=0 is NaN; e5m2: exp==0x1F is Inf/NaN
  if (fmt == 0u && exp == 0xFu && mant != 0u) {
    return NAN;
  }
  if (fmt == 1u && exp == 0x1Fu) {
    return mant == 0u ? (sign ? -INFINITY : INFINITY) : NAN;
  }
  float m = 1.0f + float(mant) / float(1u << mbits);
  float v = ldexp(m, int(exp) - exp_bias);
  return sign ? -v : v;
}

template <typename T>
void quantize_fp8_impl(
    device const T* x,
    device const float* scale,
    device uchar* out,
    constant Fp8Params& p,
    uint gid,
    uint nthreads) {
  const float inv = 1.0f / max(*scale, 1e-30f);
  const float fmax = p.fmt == 0u ? kF8E4M3Max : kF8E5M2Max;
  for (uint i = gid; i < p.numel; i += nthreads) {
    float v = float(x[i]) * inv;
    v = clamp(v, -fmax, fmax);
    out[i] = float_to_fp8(v, p.fmt);
  }
}

kernel void quantize_fp8_fp16(
    device const half* x [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device uchar* out [[buffer(2)]],
    constant Fp8Params& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint nthreads [[threads_per_grid]]) {
  quantize_fp8_impl<half>(x, scale, out, p, gid, nthreads);
}

kernel void quantize_fp8_bf16(
    device const bfloat* x [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device uchar* out [[buffer(2)]],
    constant Fp8Params& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint nthreads [[threads_per_grid]]) {
  quantize_fp8_impl<bfloat>(x, scale, out, p, gid, nthreads);
}

kernel void quantize_fp8_fp32(
    device const float* x [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device uchar* out [[buffer(2)]],
    constant Fp8Params& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint nthreads [[threads_per_grid]]) {
  quantize_fp8_impl<float>(x, scale, out, p, gid, nthreads);
}

template <typename T>
void dequantize_fp8_impl(
    device const uchar* x,
    device const float* scale,
    device T* out,
    constant Fp8Params& p,
    uint gid,
    uint nthreads) {
  const float s = *scale;
  for (uint i = gid; i < p.numel; i += nthreads) {
    out[i] = T(fp8_to_float(x[i], p.fmt) * s);
  }
}

kernel void dequantize_fp8_to_fp16(
    device const uchar* x [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant Fp8Params& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint nthreads [[threads_per_grid]]) {
  dequantize_fp8_impl<half>(x, scale, out, p, gid, nthreads);
}

kernel void dequantize_fp8_to_bf16(
    device const uchar* x [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant Fp8Params& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint nthreads [[threads_per_grid]]) {
  dequantize_fp8_impl<bfloat>(x, scale, out, p, gid, nthreads);
}

kernel void dequantize_fp8_to_fp32(
    device const uchar* x [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant Fp8Params& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint nthreads [[threads_per_grid]]) {
  dequantize_fp8_impl<float>(x, scale, out, p, gid, nthreads);
}

// ---- SVDQuant W4A4: group-wise int4 unpack/dequant + activation quant ----

struct SvdDequantParams {
  uint rows;
  uint k;       // full K (even)
  uint groups;  // K / 64
  uint packed_stride;
  uint scale_stride; // row stride of scales in (groups, rows) layout = rows
  uint out_stride;
  uint unsigned_act;
};

struct SvdQuantParams {
  uint rows;
  uint k;
  uint groups;
  uint x_stride;
  uint packed_stride;
  uint scale_stride; // = rows for (groups, rows)
  uint unsigned_act;
  uint qmax; // 7 or 15
};

inline int unpack_nibble_signed(uint nibble) {
  int v = int(nibble & 0xFu);
  return v >= 8 ? v - 16 : v;
}

inline int unpack_nibble_unsigned(uint nibble) {
  return int(nibble & 0xFu);
}

template <typename T>
void svdquant_dequant_row(
    device const char* packed,
    device const T* scales, // (groups, rows) contiguous
    device T* out,
    constant SvdDequantParams& p,
    uint row,
    uint tid,
    uint tptg) {
  if (row >= p.rows) {
    return;
  }
  device const uchar* prow = (device const uchar*)(packed + ulong(row) * p.packed_stride);
  device T* orow = out + ulong(row) * p.out_stride;
  const uint group_size = p.k / p.groups;
  const uint packed_len = p.k / 2u;

  // One thread per packed byte; scale looked up from the owning group.
  for (uint i = tid; i < packed_len; i += tptg) {
    const uint d0 = i * 2u;
    const uint g = d0 / group_size;
    const float s = float(scales[g * p.scale_stride + row]);
    const uchar byte = prow[i];
    int lo = p.unsigned_act ? unpack_nibble_unsigned(byte) : unpack_nibble_signed(byte);
    int hi = p.unsigned_act ? unpack_nibble_unsigned(byte >> 4) : unpack_nibble_signed(byte >> 4);
    orow[d0] = T(float(lo) * s);
    orow[d0 + 1u] = T(float(hi) * s);
  }
}

kernel void svdquant_dequant_fp16(
    device const char* packed [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant SvdDequantParams& p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]]) {
  svdquant_dequant_row<half>(packed, scales, out, p, row, tid, tptg);
}

kernel void svdquant_dequant_bf16(
    device const char* packed [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant SvdDequantParams& p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]]) {
  svdquant_dequant_row<bfloat>(packed, scales, out, p, row, tid, tptg);
}

template <typename T>
void svdquant_quantize_row(
    device const T* x,
    device const T* smooth,
    device char* packed,
    device T* scales_out, // (groups, rows)
    constant SvdQuantParams& p,
    uint row,
    uint tid,
    uint tptg,
    threadgroup float* red) {
  if (row >= p.rows) {
    return;
  }
  device const T* xrow = x + ulong(row) * p.x_stride;
  device uchar* prow = (device uchar*)(packed + ulong(row) * p.packed_stride);
  const uint group_size = p.k / p.groups;
  const float qmax = float(p.qmax);
  const float qmin = p.unsigned_act ? 0.0f : -qmax;

  // One group at a time owned by the whole TG (groups often = K/64 ~ 32-128).
  for (uint g = 0; g < p.groups; ++g) {
    const uint base = g * group_size;
    float amax = 0.0f;
    for (uint i = tid; i < group_size; i += tptg) {
      float v = float(xrow[base + i]) / max(float(smooth[base + i]), 1e-12f);
      amax = max(amax, fabs(v));
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
      for (uint i = 0; i < num_simd; ++i) {
        m = max(m, red[i]);
      }
      m = max(m, 1e-10f);
      float s = m / qmax;
      scales_out[g * p.scale_stride + row] = T(s);
      red[0] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float s = red[0];
    const float inv = 1.0f / s;
    for (uint i = tid; i < group_size / 2u; i += tptg) {
      const uint d0 = base + i * 2u;
      float v0 = float(xrow[d0]) / max(float(smooth[d0]), 1e-12f) * inv;
      float v1 = float(xrow[d0 + 1u]) / max(float(smooth[d0 + 1u]), 1e-12f) * inv;
      int q0 = int(round(clamp(v0, qmin, qmax)));
      int q1 = int(round(clamp(v1, qmin, qmax)));
      prow[(d0 / 2u)] = uchar((q0 & 0xF) | ((q1 & 0xF) << 4));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

kernel void svdquant_quantize_fp16(
    device const half* x [[buffer(0)]],
    device const half* smooth [[buffer(1)]],
    device char* packed [[buffer(2)]],
    device half* scales [[buffer(3)]],
    constant SvdQuantParams& p [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]]) {
  threadgroup float red[8];
  svdquant_quantize_row<half>(x, smooth, packed, scales, p, row, tid, tptg, red);
}

kernel void svdquant_quantize_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* smooth [[buffer(1)]],
    device char* packed [[buffer(2)]],
    device bfloat* scales [[buffer(3)]],
    constant SvdQuantParams& p [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]]) {
  threadgroup float red[8];
  svdquant_quantize_row<bfloat>(x, smooth, packed, scales, p, row, tid, tptg, red);
}

// ---- INT8 tensorwise quantize + typed dequant + stochastic FP8 ----

struct TensorQuantParams {
  uint numel;
  uint stochastic;
  uint seed_lo;
  uint seed_hi;
};

struct DequantFlatParams {
  uint numel;
  uint inner_dim;
  uint scale_mode; // 0 scalar, 1 per-element, 2 per-row
};

template <typename T>
void quantize_int8_tensorwise_impl(
    device const T* x,
    device const float* scale,
    device char* q,
    constant TensorQuantParams& p,
    uint gid,
    uint nthreads) {
  const float inv = 1.0f / max(*scale, 1e-30f);
  const bool stochastic = p.stochastic != 0u;
  const ulong seed = (ulong(p.seed_hi) << 32) | ulong(p.seed_lo);
  for (uint i = gid; i < p.numel; i += nthreads) {
    q[i] = char(quantize_int8_value(float(x[i]) * inv, stochastic, ulong(i), seed));
  }
}

kernel void quantize_int8_tensorwise_fp16(
    device const half* x [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device char* q [[buffer(2)]],
    constant TensorQuantParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint nthreads [[threads_per_grid]]) {
  quantize_int8_tensorwise_impl<half>(x, scale, q, p, gid, nthreads);
}

kernel void quantize_int8_tensorwise_bf16(
    device const bfloat* x [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device char* q [[buffer(2)]],
    constant TensorQuantParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint nthreads [[threads_per_grid]]) {
  quantize_int8_tensorwise_impl<bfloat>(x, scale, q, p, gid, nthreads);
}

inline float dequant_scale_at(device const float* scale, uint i, constant DequantFlatParams& p) {
  if (p.scale_mode == 1u) {
    return scale[i];
  }
  if (p.scale_mode == 2u) {
    return scale[i / max(p.inner_dim, 1u)];
  }
  return scale[0];
}

template <typename T>
void dequantize_int8_flat_impl(
    device const char* q,
    device const float* scale,
    device T* out,
    constant DequantFlatParams& p,
    uint gid,
    uint nthreads) {
  for (uint i = gid; i < p.numel; i += nthreads) {
    out[i] = T(float(q[i]) * dequant_scale_at(scale, i, p));
  }
}

kernel void dequantize_int8_flat_fp32(
    device const char* q [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant DequantFlatParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint nthreads [[threads_per_grid]]) {
  dequantize_int8_flat_impl<float>(q, scale, out, p, gid, nthreads);
}

kernel void dequantize_int8_flat_fp16(
    device const char* q [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant DequantFlatParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint nthreads [[threads_per_grid]]) {
  dequantize_int8_flat_impl<half>(q, scale, out, p, gid, nthreads);
}

kernel void dequantize_int8_flat_bf16(
    device const char* q [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant DequantFlatParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint nthreads [[threads_per_grid]]) {
  dequantize_int8_flat_impl<bfloat>(q, scale, out, p, gid, nthreads);
}

// Stochastic FP8: match eager/HIP (half-reduce → mantissa floor(rng) → pack).
inline uchar stochastic_round_fp8_one(float src, uchar rng_byte, uint fmt) {
  const bool e4m3 = fmt == 0u;
  const float fp8_max = e4m3 ? kF8E4M3Max : kF8E5M2Max;
  const int exponent_bits = e4m3 ? 4 : 5;
  const int mantissa_bits = e4m3 ? 3 : 2;
  const int exponent_bias = e4m3 ? 7 : 15;
  const float mantissa_levels = float(1 << mantissa_bits);
  const float subnormal_mantissa_scale = exp2(float(-exponent_bias + 1 - mantissa_bits));
  const float subnormal_value_scale = exp2(float(-exponent_bias + 1));

  const float value = float(half(src));
  if (value != value) {
    return float_to_fp8(value, fmt);
  }
  const float abs_value = fabs(value);
  const float sign = value < 0.0f ? -1.0f : (value > 0.0f ? 1.0f : 0.0f);
  if (abs_value == 0.0f) {
    return float_to_fp8(0.0f, fmt);
  }

  float exponent = floor(log2(abs_value)) + float(exponent_bias);
  exponent = clamp(exponent, 0.0f, float((1 << exponent_bits) - 1));
  const bool normal = exponent != 0.0f;
  const float exponent_scale = exp2(exponent - float(exponent_bias));
  const float normal_mantissa = (abs_value / exponent_scale - 1.0f) * mantissa_levels;
  const float subnormal_mantissa = abs_value / subnormal_mantissa_scale;
  const float random = float(rng_byte) * (1.0f / 256.0f);
  const float mantissa =
      floor((normal ? normal_mantissa : subnormal_mantissa) + random) / mantissa_levels;
  const float rounded =
      sign * (normal ? exponent_scale * (1.0f + mantissa) : subnormal_value_scale * mantissa);
  return float_to_fp8(clamp(rounded, -fp8_max, fp8_max), fmt);
}

template <typename T>
void stochastic_round_fp8_impl(
    device const T* x,
    device uchar* rng_and_dst,
    constant Fp8Params& p,
    uint gid,
    uint nthreads) {
  for (uint i = gid; i < p.numel; i += nthreads) {
    rng_and_dst[i] = stochastic_round_fp8_one(float(x[i]), rng_and_dst[i], p.fmt);
  }
}

kernel void stochastic_round_fp8_fp16(
    device const half* x [[buffer(0)]],
    device uchar* rng_and_dst [[buffer(1)]],
    constant Fp8Params& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint nthreads [[threads_per_grid]]) {
  stochastic_round_fp8_impl<half>(x, rng_and_dst, p, gid, nthreads);
}

kernel void stochastic_round_fp8_bf16(
    device const bfloat* x [[buffer(0)]],
    device uchar* rng_and_dst [[buffer(1)]],
    constant Fp8Params& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint nthreads [[threads_per_grid]]) {
  stochastic_round_fp8_impl<bfloat>(x, rng_and_dst, p, gid, nthreads);
}

kernel void stochastic_round_fp8_fp32(
    device const float* x [[buffer(0)]],
    device uchar* rng_and_dst [[buffer(1)]],
    constant Fp8Params& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint nthreads [[threads_per_grid]]) {
  stochastic_round_fp8_impl<float>(x, rng_and_dst, p, gid, nthreads);
}
