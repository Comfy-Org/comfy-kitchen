// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
//
// Optimized INT8 per-thread quantization for Q and K (SageAttention).
//
// All tensors assumed contiguous [B, H, L, D] (HND layout).
// Two-kernel launch: Q and K in separate kernels for better I-cache
// utilization, with a fused fallback for non-standard tile configs.
//   Q path: warp-per-group, single-pass, vectorized float2 loads / int32
//   stores. K path: warp-per-group, single-pass, vectorized loads / stores.
//   No __syncthreads anywhere – pure warp-level reductions.
//
// Block / warp tile sizes and alignment are template parameters so the
// compiler can constant-fold address arithmetic (divisions, modulos) and
// eliminate dead scalar-fallback code when C is a multiple of 4.
//
// Smooth-K: when enabled, a custom k_mean_reduce kernel computes per-channel
// means across the sequence dimension using vectorized loads and shared-memory
// reduction, then quant_k_kernel subtracts them inline during quantization.
// Both kernels run back-to-back on the same stream so K data stays warm in L2
// cache between the two reads.

#include "dtype_dispatch.cuh"
#include "float_utils.cuh"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

using comfy::quant_int8;
using comfy::store4_i8;
using comfy::warp_reduce_fmax;

namespace {

template <typename T>
struct VectorLoader4;

template <>
struct VectorLoader4<half> {
  __forceinline__ __device__ static void load(const half* ptr, float* out) {
    float2 raw = __ldg(reinterpret_cast<const float2 *>(ptr));
    const half *vals = reinterpret_cast<const half *>(&raw);
    out[0] = static_cast<float>(vals[0]);
    out[1] = static_cast<float>(vals[1]);
    out[2] = static_cast<float>(vals[2]);
    out[3] = static_cast<float>(vals[3]);
  }
};

template <>
struct VectorLoader4<nv_bfloat16> {
  __forceinline__ __device__ static void load(const nv_bfloat16* ptr, float* out) {
    float2 raw = __ldg(reinterpret_cast<const float2 *>(ptr));
    const nv_bfloat16 *vals = reinterpret_cast<const nv_bfloat16 *>(&raw);
    out[0] = static_cast<float>(vals[0]);
    out[1] = static_cast<float>(vals[1]);
    out[2] = static_cast<float>(vals[2]);
    out[3] = static_cast<float>(vals[3]);
  }
};

template <>
struct VectorLoader4<float> {
  __forceinline__ __device__ static void load(const float* ptr, float* out) {
    float4 raw = __ldg(reinterpret_cast<const float4 *>(ptr));
    out[0] = raw.x;
    out[1] = raw.y;
    out[2] = raw.z;
    out[3] = raw.w;
  }
};

__forceinline__ __device__ void convrot4(float *values) {
  const float x0 = values[0];
  const float x1 = values[1];
  const float x2 = values[2];
  const float x3 = values[3];
  const float a0 = x0 + x1;
  const float a1 = x0 - x1;
  const float a2 = x2 + x3;
  const float a3 = x2 - x3;
  values[0] = (a0 + a2) * 0.5f;
  values[1] = (a1 + a3) * 0.5f;
  values[2] = (a0 - a2) * 0.5f;
  values[3] = (a1 - a3) * 0.5f;
}

// Apply a normalized Walsh-Hadamard H64 to one 64-channel half-warp group.
// H4 covers the four adjacent channels owned by each lane; four shuffle
// butterflies cover the remaining 16-lane dimension. This uses half as many
// shuffles as evaluating H4 x H4 x H4 directly.
__forceinline__ __device__ void convrot64(float *values) {
  convrot4(values);
  const int half_lane = threadIdx.x & 15;
  const unsigned mask = (threadIdx.x & 16) ? 0xffff0000u : 0x0000ffffu;

#pragma unroll
  for (int bit = 1; bit < 16; bit <<= 1) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const float other = __shfl_xor_sync(mask, values[c], bit, 16);
      values[c] =
          (half_lane & bit) ? other - values[c] : values[c] + other;
    }
  }

#pragma unroll
  for (int c = 0; c < 4; ++c)
    values[c] *= 0.25f;
}

// Apply a normalized Walsh-Hadamard H128 across all four-channel warp groups.
__forceinline__ __device__ void convrot128(float *values) {
  convrot4(values);
  const int lane = threadIdx.x & 31;

#pragma unroll
  for (int bit = 1; bit < 32; bit <<= 1) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const float other = __shfl_xor_sync(0xffffffffu, values[c], bit);
      values[c] = (lane & bit) ? other - values[c] : values[c] + other;
    }
  }

#pragma unroll
  for (int c = 0; c < 4; ++c)
    values[c] *= 0.1767766952966369f;
}

// ---------------------------------------------------------------------------
// Q processing device function
// ---------------------------------------------------------------------------
#pragma nv_diag_suppress 1056
template <typename T, int NR, int BLKQ, int WARPQ, int CHANNEL_TILES,
          int ROTATION, bool ALIGNED4>
__forceinline__ __device__ void
process_q(const T *__restrict__ in, int8_t *__restrict__ out,
          float *__restrict__ sc_buf, const int oblk, const int L,
          const int C, const int64_t stride_n) {
  constexpr int NSUB = BLKQ / WARPQ;
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;

#pragma unroll
  for (int g = 0; g < 2; ++g) {
    const int otld = wid * 2 + g;
    const int base = (oblk / NSUB) * BLKQ + (oblk % NSUB) * WARPQ + otld;

    float v[CHANNEL_TILES * NR * 4];
    float mx = 0.f;

#pragma unroll
    for (int i = 0; i < CHANNEL_TILES * NR * 4; ++i)
      v[i] = 0.f;

#pragma unroll
    for (int tile = 0; tile < CHANNEL_TILES; ++tile) {
      const int ch = tile * 128 + (lane << 2);
      const int tile_base = tile * NR * 4;
      if (ALIGNED4 || ch + 3 < C) {
#pragma unroll
        for (int j = 0; j < NR; ++j) {
          const int n = base + j * 8;
          const int vi = tile_base + j * 4;
          if (n < L) {
            VectorLoader4<T>::load(&in[(int64_t)n * stride_n + ch], &v[vi]);
            mx = fmaxf(mx, fmaxf(fmaxf(fabsf(v[vi]), fabsf(v[vi + 1])),
                                 fmaxf(fabsf(v[vi + 2]), fabsf(v[vi + 3]))));
          }
        }
      } else if (ch < C) {
#pragma unroll
        for (int j = 0; j < NR; ++j) {
          const int n = base + j * 8;
          const int vi = tile_base + j * 4;
          if (n < L) {
#pragma unroll
            for (int c = 0; c < 4; ++c) {
              v[vi + c] =
                  (ch + c < C)
                      ? static_cast<float>(
                            __ldg(&in[(int64_t)n * stride_n + ch + c]))
                      : 0.f;
              mx = fmaxf(mx, fabsf(v[vi + c]));
            }
          }
        }
      }
    }

    if constexpr (ROTATION != 0) {
#pragma unroll
      for (int tile = 0; tile < CHANNEL_TILES; ++tile) {
#pragma unroll
        for (int j = 0; j < NR; ++j) {
          if constexpr (ROTATION == 128)
            convrot128(&v[(tile * NR + j) * 4]);
          else if constexpr (ROTATION == 64)
            convrot64(&v[(tile * NR + j) * 4]);
          else
            convrot4(&v[(tile * NR + j) * 4]);
        }
      }
      mx = 0.f;
#pragma unroll
      for (int j = 0; j < CHANNEL_TILES * NR * 4; ++j)
        mx = fmaxf(mx, fabsf(v[j]));
    }

    mx = warp_reduce_fmax(mx);
    const float sc = mx / 127.f + 1e-7f;

    if (lane == 0)
      sc_buf[oblk * 8 + otld] = sc;

#pragma unroll
    for (int tile = 0; tile < CHANNEL_TILES; ++tile) {
      const int ch = tile * 128 + (lane << 2);
      const int tile_base = tile * NR * 4;
      if (ALIGNED4 || ch + 3 < C) {
#pragma unroll
        for (int j = 0; j < NR; ++j) {
          const int n = base + j * 8;
          const int vi = tile_base + j * 4;
          if (n < L) {
            store4_i8(&out[(int64_t)n * C + ch], quant_int8(v[vi], sc),
                      quant_int8(v[vi + 1], sc), quant_int8(v[vi + 2], sc),
                      quant_int8(v[vi + 3], sc));
          }
        }
      } else if (ch < C) {
#pragma unroll
        for (int j = 0; j < NR; ++j) {
          const int n = base + j * 8;
          const int vi = tile_base + j * 4;
          if (n < L) {
#pragma unroll
            for (int c = 0; c < 4; ++c) {
              if (ch + c < C)
                out[(int64_t)n * C + ch + c] = quant_int8(v[vi + c], sc);
            }
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// K processing device function
//
// When km != nullptr (float32), subtracts the per-channel mean from each
// loaded value before abs-max reduction and quantization.  km points to
// D floats for the current (b,h) slice.
// ---------------------------------------------------------------------------
template <typename T, int NL, int WARPK, int CHANNEL_TILES, int ROTATION,
          bool ALIGNED4>
__forceinline__ __device__ void
process_k(const T *__restrict__ in, int8_t *__restrict__ out,
          float *__restrict__ sc_buf, const int oblk, const int L, const int C,
          const float *__restrict__ km, const int64_t stride_n) {
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  const int otld = wid;

  float bias[CHANNEL_TILES * 4];
#pragma unroll
  for (int i = 0; i < CHANNEL_TILES * 4; ++i)
    bias[i] = 0.f;

  if (km) {
#pragma unroll
    for (int tile = 0; tile < CHANNEL_TILES; ++tile) {
      const int ch = tile * 128 + (lane << 2);
      if (ALIGNED4 || ch + 3 < C) {
        float4 b4 = __ldg(reinterpret_cast<const float4 *>(&km[ch]));
        bias[tile * 4] = b4.x;
        bias[tile * 4 + 1] = b4.y;
        bias[tile * 4 + 2] = b4.z;
        bias[tile * 4 + 3] = b4.w;
      } else if (ch < C) {
#pragma unroll
        for (int c = 0; c < 4; ++c)
          bias[tile * 4 + c] =
              (ch + c < C) ? __ldg(&km[ch + c]) : 0.f;
      }
    }
  }

  float v[CHANNEL_TILES * 2 * NL * 4];
  float mx = 0.f;

#pragma unroll
  for (int i = 0; i < CHANNEL_TILES * 2 * NL * 4; ++i)
    v[i] = 0.f;

#pragma unroll
  for (int tile = 0; tile < CHANNEL_TILES; ++tile) {
    const int ch = tile * 128 + (lane << 2);
    const int tile_base = tile * 2 * NL * 4;
    if (ALIGNED4 || ch + 3 < C) {
#pragma unroll
      for (int j = 0; j < NL; ++j) {
#pragma unroll
        for (int p = 0; p < 2; ++p) {
          const int n = oblk * WARPK + j * 8 + otld * 2 + p;
          const int vi = tile_base + (j * 2 + p) * 4;
          if (n < L) {
            VectorLoader4<T>::load(&in[(int64_t)n * stride_n + ch], &v[vi]);
#pragma unroll
            for (int c = 0; c < 4; ++c)
              v[vi + c] -= bias[tile * 4 + c];
            mx = fmaxf(mx, fmaxf(fmaxf(fabsf(v[vi]), fabsf(v[vi + 1])),
                                 fmaxf(fabsf(v[vi + 2]), fabsf(v[vi + 3]))));
          }
        }
      }
    } else if (ch < C) {
#pragma unroll
      for (int j = 0; j < NL; ++j) {
#pragma unroll
        for (int p = 0; p < 2; ++p) {
          const int n = oblk * WARPK + j * 8 + otld * 2 + p;
          const int vi = tile_base + (j * 2 + p) * 4;
          if (n < L) {
#pragma unroll
            for (int c = 0; c < 4; ++c) {
              v[vi + c] =
                  (ch + c < C)
                      ? static_cast<float>(
                            __ldg(&in[(int64_t)n * stride_n + ch + c])) -
                            bias[tile * 4 + c]
                      : 0.f;
              mx = fmaxf(mx, fabsf(v[vi + c]));
            }
          }
        }
      }
    }
  }

  if constexpr (ROTATION != 0) {
#pragma unroll
    for (int tile = 0; tile < CHANNEL_TILES; ++tile) {
#pragma unroll
      for (int j = 0; j < 2 * NL; ++j) {
        if constexpr (ROTATION == 128)
          convrot128(&v[(tile * 2 * NL + j) * 4]);
        else if constexpr (ROTATION == 64)
          convrot64(&v[(tile * 2 * NL + j) * 4]);
        else
          convrot4(&v[(tile * 2 * NL + j) * 4]);
      }
    }
    mx = 0.f;
#pragma unroll
    for (int j = 0; j < CHANNEL_TILES * 2 * NL * 4; ++j)
      mx = fmaxf(mx, fabsf(v[j]));
  }

  mx = warp_reduce_fmax(mx);
  const float sc = mx / 127.f + 1e-7f;

  if (lane == 0)
    sc_buf[oblk * 4 + otld] = sc;

#pragma unroll
  for (int tile = 0; tile < CHANNEL_TILES; ++tile) {
    const int ch = tile * 128 + (lane << 2);
    const int tile_base = tile * 2 * NL * 4;
    if (ALIGNED4 || ch + 3 < C) {
#pragma unroll
      for (int j = 0; j < NL; ++j) {
#pragma unroll
        for (int p = 0; p < 2; ++p) {
          const int n = oblk * WARPK + j * 8 + otld * 2 + p;
          const int vi = tile_base + (j * 2 + p) * 4;
          if (n < L) {
            store4_i8(&out[(int64_t)n * C + ch], quant_int8(v[vi], sc),
                      quant_int8(v[vi + 1], sc), quant_int8(v[vi + 2], sc),
                      quant_int8(v[vi + 3], sc));
          }
        }
      }
    } else if (ch < C) {
#pragma unroll
      for (int j = 0; j < NL; ++j) {
#pragma unroll
        for (int p = 0; p < 2; ++p) {
          const int n = oblk * WARPK + j * 8 + otld * 2 + p;
          const int vi = tile_base + (j * 2 + p) * 4;
          if (n < L) {
#pragma unroll
            for (int c = 0; c < 4; ++c) {
              if (ch + c < C)
                out[(int64_t)n * C + ch + c] = quant_int8(v[vi + c], sc);
            }
          }
        }
      }
    }
  }
}
#pragma nv_diag_default 1056

// ---------------------------------------------------------------------------
// K channel-mean reduction kernel (smooth-k)
//
// Computes km[b,h,d] = (1/Lk) * sum_n k[b,h,n,d]   ∀ (b,h,d).
//
// Grid: (num_tile_blks, H_kv, B)   — multiple blocks per (b,h) head.
// Block: MEAN_BLK_DIM threads.
//
// Each thread covers 4 channels (via float2 vectorized loads of 4 bf16).
// D=128 → 32 "channel groups".  With MEAN_BLK_DIM threads we have
// MEAN_BLK_DIM/32 "row workers" per channel group.  Each row worker
// loops over a subset of N rows (strided), accumulating 4 fp32 partial
// sums.  A shared-memory reduction then sums across all row workers that
// share the same channel group, producing one partial per channel group
// per block.  The partial is atomicAdd'd to the output.  The last block
// for each (b,h) divides by Lk to get the mean.
//
// km_out: [B * H_kv * C]  float32  (pre-zeroed by the caller)
// done:   [B * H_kv]      int32    (pre-zeroed by the caller)
// ---------------------------------------------------------------------------
constexpr int MEAN_BLK_DIM = 256;
constexpr int MEAN_ROWS_PER_BLK = 512;

template <typename T, int CHANNEL_GROUPS>
__global__ __launch_bounds__(MEAN_BLK_DIM) void k_mean_reduce(
    const T *__restrict__ k_in, float *__restrict__ km_out,
    int *__restrict__ done, const int Lk, const int C, const int H_kv,
    const int n_blks, const float inv_Lk, const int64_t stride_b,
    const int64_t stride_h, const int64_t stride_n) {
  const int tile = blockIdx.x;
  const int h = blockIdx.y, b = blockIdx.z;
  const int64_t bh_off = (int64_t)b * stride_b + (int64_t)h * stride_h;
  const int bh_idx = b * H_kv + h;

  constexpr int ROW_WORKERS = MEAN_BLK_DIM / CHANNEL_GROUPS;
  const int cg = threadIdx.x % CHANNEL_GROUPS;
  const int rw = threadIdx.x / CHANNEL_GROUPS;
  const int ch = cg << 2;                        // starting channel

  const int row_base = tile * MEAN_ROWS_PER_BLK;

  float4 acc = {0.f, 0.f, 0.f, 0.f};

  if (ch < C) {
    for (int r = rw; r < MEAN_ROWS_PER_BLK; r += ROW_WORKERS) {
      const int n = row_base + r;
      if (n < Lk) {
        float vals[4];
        VectorLoader4<T>::load(&k_in[bh_off + (int64_t)n * stride_n + ch], vals);
        acc.x += vals[0];
        acc.y += vals[1];
        acc.z += vals[2];
        acc.w += vals[3];
      }
    }
  }

  // Reduce across all 8 row-workers that share the same channel group.
  // Row workers for the same channel group are in different warps, so we
  // use shared memory: each worker writes its float4 partial, then
  // worker 0 sums all 8 partials for its channel group.
  __shared__ float4 smem[ROW_WORKERS][CHANNEL_GROUPS];
  smem[rw][cg] = acc;
  __syncthreads();

  // First row worker (rw==0) sums all 8 partials for its channel group.
  if (rw == 0 && ch < C) {
    float4 s = smem[0][cg];
#pragma unroll
    for (int i = 1; i < ROW_WORKERS; ++i) {
      float4 v = smem[i][cg];
      s.x += v.x;
      s.y += v.y;
      s.z += v.z;
      s.w += v.w;
    }

    atomicAdd(&km_out[bh_idx * C + ch], s.x);
    atomicAdd(&km_out[bh_idx * C + ch + 1], s.y);
    atomicAdd(&km_out[bh_idx * C + ch + 2], s.z);
    atomicAdd(&km_out[bh_idx * C + ch + 3], s.w);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    __threadfence();
    int prev = atomicAdd(&done[bh_idx], 1);
    if (prev == n_blks - 1) {
      for (int c = 0; c < C; ++c)
        km_out[bh_idx * C + c] *= inv_Lk;
    }
  }
}

// ---------------------------------------------------------------------------
// Standalone Q kernel
// ---------------------------------------------------------------------------
template <typename T, int NR, int BLKQ, int WARPQ, int CHANNEL_TILES,
          int ROTATION, bool ALIGNED4>
__global__ __launch_bounds__(128, 4) void quant_q_kernel(
    const T *__restrict__ q_in, int8_t *__restrict__ q_out,
    float *__restrict__ q_sb, const int Lq, const int C, const int H_q,
    const int q_sc_per_h, const int64_t stride_b, const int64_t stride_h,
    const int64_t stride_n) {
  const int oblk = blockIdx.x;
  const int h = blockIdx.y, b = blockIdx.z;
  const int64_t in_bh = (int64_t)b * stride_b + (int64_t)h * stride_h;
  const int64_t out_bh = ((int64_t)b * H_q + h) * Lq * C;
  const int64_t sbh = ((int64_t)b * H_q + h) * q_sc_per_h;
  process_q<T, NR, BLKQ, WARPQ, CHANNEL_TILES, ROTATION, ALIGNED4>(
      q_in + in_bh, q_out + out_bh, q_sb + sbh, oblk, Lq, C, stride_n);
}

// ---------------------------------------------------------------------------
// Standalone K kernel  (km may be nullptr when smooth-k is disabled)
// km is float32 [B, H_kv, C] computed by k_mean_reduce.
// ---------------------------------------------------------------------------
template <typename T, int NL, int WARPK, int CHANNEL_TILES, int ROTATION,
          bool ALIGNED4>
__global__ __launch_bounds__(128, 4) void quant_k_kernel(
    const T *__restrict__ k_in, int8_t *__restrict__ k_out,
    float *__restrict__ k_sb, const float *__restrict__ km, const int Lk,
    const int C, const int H_kv, const int k_sc_per_h, const int64_t stride_b,
    const int64_t stride_h, const int64_t stride_n) {
  const int oblk = blockIdx.x;
  const int h = blockIdx.y, b = blockIdx.z;
  const int64_t in_bh = (int64_t)b * stride_b + (int64_t)h * stride_h;
  const int64_t out_bh = ((int64_t)b * H_kv + h) * Lk * C;
  const int64_t sbh = ((int64_t)b * H_kv + h) * k_sc_per_h;
  const float *km_bh = km ? km + ((int64_t)b * H_kv + h) * C : nullptr;
  process_k<T, NL, WARPK, CHANNEL_TILES, ROTATION, ALIGNED4>(
      k_in + in_bh, k_out + out_bh, k_sb + sbh, oblk, Lk, C, km_bh,
      stride_n);
}

// ---------------------------------------------------------------------------
// Fused Q+K kernel – fallback for non-standard tile configs.
// blockIdx.x < q_oblk_count  →  Q path
// blockIdx.x >= q_oblk_count →  K path
// ---------------------------------------------------------------------------
template <typename T, int NR, int NL, int BLKQ, int WARPQ, int BLKK, int WARPK,
          int CHANNEL_TILES, int ROTATION, bool ALIGNED4>
__global__ __launch_bounds__(128, 3) void quant_qk_fused(
    const T *__restrict__ q_in, int8_t *__restrict__ q_out,
    float *__restrict__ q_sb, const T *__restrict__ k_in,
    int8_t *__restrict__ k_out, float *__restrict__ k_sb,
    const float *__restrict__ km, const int Lq, const int Lk, const int C,
    const int q_oblk_count, const int H_q, const int H_kv, const int q_sc_per_h,
    const int k_sc_per_h, const int64_t q_stride_b, const int64_t q_stride_h,
    const int64_t q_stride_n, const int64_t k_stride_b,
    const int64_t k_stride_h, const int64_t k_stride_n) {
  const int h = blockIdx.y, b = blockIdx.z;

  if (blockIdx.x < (unsigned)q_oblk_count) {
    if (h >= H_q)
      return;
    const int64_t in_bh = (int64_t)b * q_stride_b + (int64_t)h * q_stride_h;
    const int64_t out_bh = ((int64_t)b * H_q + h) * Lq * C;
    const int64_t sbh = ((int64_t)b * H_q + h) * q_sc_per_h;
    process_q<T, NR, BLKQ, WARPQ, CHANNEL_TILES, ROTATION, ALIGNED4>(
        q_in + in_bh, q_out + out_bh, q_sb + sbh, blockIdx.x, Lq, C,
        q_stride_n);
  } else {
    if (h >= H_kv)
      return;
    const int64_t in_bh = (int64_t)b * k_stride_b + (int64_t)h * k_stride_h;
    const int64_t out_bh = ((int64_t)b * H_kv + h) * Lk * C;
    const int64_t sbh = ((int64_t)b * H_kv + h) * k_sc_per_h;
    const float *km_bh = km ? km + ((int64_t)b * H_kv + h) * C : nullptr;
    process_k<T, NL, WARPK, CHANNEL_TILES, ROTATION, ALIGNED4>(
        k_in + in_bh, k_out + out_bh, k_sb + sbh,
        (int)blockIdx.x - q_oblk_count, Lk, C, km_bh, k_stride_n);
  }
}

} // namespace

// smooth_k == 1 → compute km into km_scratch, pass to quant_k_kernel.
// km_scratch: [B * H_kv * C] float32  (will be zeroed internally).
// km_done:    [B * H_kv]     int32    (will be zeroed internally).
extern "C" void launch_quant_qk_per_thread_int8(
    const void *q, void *q_int8, void *q_scale, const void *k, void *k_int8,
    void *k_scale, int smooth_k, void *km_scratch, void *km_done, int B,
    int H_q, int Lq, int H_kv, int Lk, int C, int BLKQ, int WARPQ, int BLKK,
    int WARPK, int64_t q_stride_b, int64_t q_stride_h, int64_t q_stride_n,
    int64_t k_stride_b, int64_t k_stride_h, int64_t k_stride_n,
    int input_dtype_code, int convrot, cudaStream_t stream) {
  const int q_oblk = (Lq + BLKQ - 1) / BLKQ * (BLKQ / WARPQ);
  const int k_oblk = (Lk + BLKK - 1) / BLKK * (BLKK / WARPK);
  const int q_sc_per_h = q_oblk * 8;
  const int k_sc_per_h = k_oblk * 4;
  // ALIGNED4 means every warp lane's 4-channel group is fully in-bounds,
  // so we can skip per-lane ch<C checks and always use vectorized loads.
  // 32 lanes × 4 channels = 128, so C must be ≥128 AND 4-aligned.
  const bool aligned4 = C == 128 || C == 256;

  float *km_ptr = nullptr;
  if (smooth_k && km_scratch && km_done) {
    const int mean_blks = (Lk + MEAN_ROWS_PER_BLK - 1) / MEAN_ROWS_PER_BLK;
    const float inv_Lk = 1.f / static_cast<float>(Lk);

    const size_t km_bytes = (size_t)B * H_kv * C * sizeof(float);
    const size_t done_bytes = (size_t)B * H_kv * sizeof(int);
    cudaMemsetAsync(km_scratch, 0, km_bytes, stream);
    cudaMemsetAsync(km_done, 0, done_bytes, stream);

    dim3 gm(mean_blks, H_kv, B);
    DISPATCH_FP_DTYPE(input_dtype_code, T, [&] {
      if (C > 128) {
        k_mean_reduce<T, 64><<<gm, MEAN_BLK_DIM, 0, stream>>>(
            (const T *)k, (float *)km_scratch, (int *)km_done, Lk, C, H_kv,
            mean_blks, inv_Lk, k_stride_b, k_stride_h, k_stride_n);
      } else {
        k_mean_reduce<T, 32><<<gm, MEAN_BLK_DIM, 0, stream>>>(
            (const T *)k, (float *)km_scratch, (int *)km_done, Lk, C, H_kv,
            mean_blks, inv_Lk, k_stride_b, k_stride_h, k_stride_n);
      }
    });
    km_ptr = (float *)km_scratch;
  }

#define LAUNCH_SPLIT(T, NR, NL, BQ, WQ, BK, WK, CT, ROT, A4)                   \
  do {                                                                         \
    dim3 gq(q_oblk, H_q, B);                                                   \
    quant_q_kernel<T, NR, BQ, WQ, CT, ROT, A4>                                 \
        <<<gq, 128, 0, stream>>>((const T *)q, (int8_t *)q_int8,               \
                                 (float *)q_scale, Lq, C, H_q, q_sc_per_h,     \
                                 q_stride_b, q_stride_h, q_stride_n);           \
    dim3 gk(k_oblk, H_kv, B);                                                  \
    quant_k_kernel<T, NL, WK, CT, ROT, A4><<<gk, 128, 0, stream>>>(            \
        (const T *)k, (int8_t *)k_int8, (float *)k_scale, km_ptr, Lk, C, H_kv, \
        k_sc_per_h, k_stride_b, k_stride_h, k_stride_n);                       \
  } while (0)

#define LAUNCH_FUSED(T, NR, NL, BQ, WQ, BK, WK, CT, ROT, A4)                   \
  do {                                                                         \
    const int H_max = H_q > H_kv ? H_q : H_kv;                                 \
    dim3 g(q_oblk + k_oblk, H_max, B);                                         \
    quant_qk_fused<T, NR, NL, BQ, WQ, BK, WK, CT, ROT, A4>                     \
        <<<g, 128, 0, stream>>>(                                                \
        (const T *)q, (int8_t *)q_int8, (float *)q_scale, (const T *)k,        \
        (int8_t *)k_int8, (float *)k_scale, km_ptr, Lq, Lk, C, q_oblk, H_q,    \
        H_kv, q_sc_per_h, k_sc_per_h, q_stride_b, q_stride_h, q_stride_n,       \
        k_stride_b, k_stride_h, k_stride_n);                                   \
  } while (0)

#define LAUNCH_SELECTED(T, ROT)                                                \
  do {                                                                         \
    if (BLKQ == 128 && WARPQ == 16 && BLKK == 128 && WARPK == 128) {           \
      LAUNCH_SPLIT(T, 2, 16, 128, 16, 128, 128, 2, ROT, true);                 \
    } else if (BLKQ == 128 && WARPQ == 16 && BLKK == 64 && WARPK == 64) {      \
      LAUNCH_SPLIT(T, 2, 8, 128, 16, 64, 64, 2, ROT, true);                    \
    } else if (BLKK == 128 && WARPK == 128 && C == 256) {                      \
      LAUNCH_SPLIT(T, 4, 16, 128, 32, 128, 128, 2, ROT, true);                 \
    } else if (C == 256) {                                                     \
      LAUNCH_SPLIT(T, 4, 8, 128, 32, 64, 64, 2, ROT, true);                    \
    } else if (BLKK == 128 && WARPK == 128 && aligned4) {                      \
      LAUNCH_FUSED(T, 4, 16, 128, 32, 128, 128, 1, ROT, true);                 \
    } else if (BLKK == 128 && WARPK == 128) {                                  \
      LAUNCH_FUSED(T, 4, 16, 128, 32, 128, 128, 1, ROT, false);                \
    } else if (aligned4) {                                                     \
      LAUNCH_FUSED(T, 4, 8, 128, 32, 64, 64, 1, ROT, true);                    \
    } else {                                                                   \
      LAUNCH_FUSED(T, 4, 8, 128, 32, 64, 64, 1, ROT, false);                   \
    }                                                                          \
  } while (0)

#define DO(T)                                                                  \
  if (!convrot) {                                                              \
    LAUNCH_SELECTED(T, 0);                                                     \
  } else if (Lk <= 256) {                                                      \
    LAUNCH_SELECTED(T, 4);                                                     \
  } else if (C >= 128) {                                                       \
    LAUNCH_SELECTED(T, 128);                                                   \
  } else {                                                                     \
    LAUNCH_SELECTED(T, 64);                                                    \
  }

  DISPATCH_FP_DTYPE(input_dtype_code, T, [&] { DO(T); });

#undef LAUNCH_SPLIT
#undef LAUNCH_FUSED
#undef LAUNCH_SELECTED
#undef DO
}
