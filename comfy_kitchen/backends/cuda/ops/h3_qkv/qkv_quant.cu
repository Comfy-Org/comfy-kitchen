// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// H3 INT8 projection with BF16 per-head RMS/RoPE rounding.
#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifdef COMFY_HAVE_CUTLASS
extern "C" int h3_qkv_available() { return 1; }

#include "mma.cuh"

namespace h3_qkv_full {
using h3_qkv::Mma;
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

// A fixed random diagonal makes the following Hadamard a randomized
// orthogonal transform instead of aligning every row to the same structured
// basis. Q and K use the same signs, so their exact dot product is unchanged.
// Flip the IEEE sign bit directly: this is exact and avoids an FP multiply.
__forceinline__ __device__ void apply_convrot_sign128(float *values,
                                                      const int lane) {
  constexpr uint32_t signs_0 = 0x1035997bu;
  constexpr uint32_t signs_1 = 0x8087f5eeu;
  constexpr uint32_t signs_2 = 0xee2e4e1au;
  constexpr uint32_t signs_3 = 0x71132418u;
  const uint32_t signs = lane < 8    ? signs_0
                         : lane < 16 ? signs_1
                         : lane < 24 ? signs_2
                                     : signs_3;
  const int shift = (lane & 7) * 4;
#pragma unroll
  for (int channel = 0; channel < 4; ++channel) {
    const uint32_t flip = ((signs >> (shift + channel)) & 1u) ^ 1u;
    values[channel] =
        __uint_as_float(__float_as_uint(values[channel]) ^ (flip << 31));
  }
}

__forceinline__ __device__ void convrot128_plain(float *values) {
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

__forceinline__ __device__ void convrot128(float *values) {
  apply_convrot_sign128(values, threadIdx.x & 31);
  convrot128_plain(values);
}

template <int Log>
__global__ __launch_bounds__(256) void direct(
    const int8_t *A, const int8_t *B, const float *xs, const float *ws,
    __nv_bfloat16 *D, const __nv_bfloat16 *freqs, const __nv_bfloat16 *qw,
    const __nv_bfloat16 *kw, float eps, int8_t *QI, float *QS, float *VP,
    int8_t *KI, float *KS, const __nv_bfloat16 *SK, const int *anchors, int M,
    int N, int K) {
  int tm = blockIdx.x >> Log,
      tn = (blockIdx.y << Log) + (blockIdx.x & ((1 << Log) - 1));
  if (tm * 128 >= M || tn * 256 >= N)
    return;
  extern __shared__ __align__(16) char storage[];
  auto &smem = *reinterpret_cast<typename Mma::SharedStorage *>(storage);
  typename Mma::IteratorA::Params pa{cutlass::layout::RowMajor(K)};
  typename Mma::IteratorB::Params pb{cutlass::layout::ColumnMajor(K)};
  int tid = threadIdx.x, warp = cutlass::canonical_warp_idx_sync(),
      lane = tid & 31;
  typename Mma::IteratorA ia(pa, const_cast<int8_t *>(A), {M, K}, tid,
                             {tm * 128, 0});
  typename Mma::IteratorB ib(pb, const_cast<int8_t *>(B), {K, N}, tid,
                             {0, tn * 256});
  Mma mma(smem, tid, warp, lane);
  typename Mma::FragmentC acc;
  acc.clear();
  mma((K + 63) / 64, acc, ia, ib, acc);
  __syncthreads();
  auto *tile = reinterpret_cast<__nv_bfloat16 *>(storage);
#pragma unroll
  for (int nn = 0; nn < 8; nn++) {
    int col = tn * 256 + (warp / 2) * 64 + nn * 8 + (lane & 3) * 2;
    float s0 = ws[col], s1 = ws[col + 1];
#pragma unroll
    for (int mm = 0; mm < 4; mm++) {
#pragma unroll
      for (int rr = 0; rr < 2; rr++) {
        int row = tm * 128 + (warp % 2) * 64 + mm * 16 + lane / 4 + rr * 8;
        int ix = (nn * 4 + mm) * 4 + rr * 2;
        if (row < M) {
          float x = xs[row];
          float y0 = __fadd_rn(
              __fmul_rn(__fmul_rn(__int2float_rn(acc[ix]), x), s0), 0.0f);
          float y1 = __fadd_rn(
              __fmul_rn(__fmul_rn(__int2float_rn(acc[ix + 1]), x), s1), 0.0f);
          unsigned u =
              unsigned(__bfloat16_as_ushort(__float2bfloat16_rn(y0))) |
              (unsigned(__bfloat16_as_ushort(__float2bfloat16_rn(y1))) << 16);
          int tr = row - tm * 128, tc = col - tn * 256;
          *reinterpret_cast<unsigned *>(tile + tr * 256 +
                                        (tc ^ ((tr & 7) * 8))) = u;
        }
      }
    }
  }
  __syncthreads();

  // Original per-head RMS reduction order, then two exact BF16 rounding points.
  for (int it = 0; it < 32; it++) {
    int rh = warp + it * 8, tr = rh / 2, hh = rh % 2, row = tm * 128 + tr,
        col = tn * 256 + hh * 128;
    if (row >= M)
      continue;
    int group = col / 7168, head = (col % 7168) / 128;
    __nv_bfloat16 *t = tile + tr * 256;
    __nv_bfloat16 *dest =
        D + int64_t(group * 56 + head) * M * 128 + int64_t(row) * 128;
    if (group == 2) {
      if (lane < 16) {
        uint4 x = *reinterpret_cast<uint4 *>(
            t + ((hh * 128 + lane * 8) ^ ((tr & 7) * 8)));
        *reinterpret_cast<uint4 *>(dest + lane * 8) = x;
      }
      continue;
    }
    float sum = 0.f;
#pragma unroll
    for (int d = lane; d < 128; d += 32) {
      float v = float(t[(hh * 128 + d) ^ ((tr & 7) * 8)]);
      sum = fmaf(v, v, sum);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      sum += __shfl_down_sync(0xffffffffu, sum, off);
    sum = __shfl_sync(0xffffffffu, sum, 0);
    float rrms = rsqrtf(sum / 128.f + eps);
    const __nv_bfloat16 *w = group == 0 ? qw : kw;
    if (lane < 24) {
#pragma unroll
      for (int i = 0; i < 2; i++) {
        int d = lane * 2 + i;
        float x0 = float(t[(hh * 128 + d) ^ ((tr & 7) * 8)]),
              x1 = float(t[(hh * 128 + d + 48) ^ ((tr & 7) * 8)]);
        x0 = float(__float2bfloat16_rn(x0 * rrms * float(w[d])));
        x1 = float(__float2bfloat16_rn(x1 * rrms * float(w[d + 48])));
        auto f = freqs + int64_t(row) * 192 + d * 4;
        float a = float(f[0]), b = float(f[1]), c = float(f[2]),
              e = float(f[3]);
        auto y0 = __float2bfloat16_rn(a * x0 + b * x1),
             y1 = __float2bfloat16_rn(c * x0 + e * x1);
        dest[d] = y0;
        dest[d + 48] = y1;
        if (group == 0 || group == 1) {
          t[(hh * 128 + d) ^ ((tr & 7) * 8)] = y0;
          t[(hh * 128 + d + 48) ^ ((tr & 7) * 8)] = y1;
        }
      }
    }
    int d = 96 + lane;
    float x = float(t[(hh * 128 + d) ^ ((tr & 7) * 8)]);
    auto tail = __float2bfloat16_rn(x * rrms * float(w[d]));
    dest[d] = tail;
    if (group == 0 || group == 1)
      t[(hh * 128 + d) ^ ((tr & 7) * 8)] = tail;
  }

  __syncthreads();
  if (tn * 256 < 7168) {
#pragma unroll
    for (int it = 0; it < 8; it++) {
      int group = warp + it * 8, hh = group / 32, sub = group % 32,
          base = (sub / 8) * 32 + sub % 8;
      float v[16], mx = 0.f;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        int tr = base + j * 8, row = tm * 128 + tr, ch = lane * 4;
#pragma unroll
        for (int c = 0; c < 4; c++)
          v[j * 4 + c] =
              row < M
                  ? float(
                        tile[tr * 256 + ((hh * 128 + ch + c) ^ ((tr & 7) * 8))])
                  : 0.f;
        convrot128(v + j * 4);
      }
#pragma unroll
      for (int i = 0; i < 16; i++)
        mx = fmaxf(mx, fabsf(v[i]));
#pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
      float sc = fmaf(mx, 1.f / 127.f, 1e-7f), inv;
      asm volatile("rcp.approx.ftz.f32 %0,%1;" : "=f"(inv) : "f"(sc));
      int head = tn * 2 + hh;
      if (lane == 0)
        QS[int64_t(head) * ((M + 127) / 128) * 32 + tm * 32 + sub] = sc;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        int row = tm * 128 + base + j * 8;
        if (row < M) {
          unsigned packed = 0;
#pragma unroll
          for (int c = 0; c < 4; c++) {
            float x = v[j * 4 + c] * inv;
            int t;
            asm volatile("cvt.rni.sat.s8.f32 %0,%1;" : "=r"(t) : "f"(x));
            packed |= (unsigned(t) & 255u) << (8 * c);
          }
          *reinterpret_cast<unsigned *>(QI + (int64_t(head) * M + row) * 128 +
                                        lane * 4) = packed;
        }
      }
    }
  }

  // Two warps per original K-scale group. Full BF16 K remains for auditing.
  if (tn * 256 >= 7168 && tn * 256 < 14336) {
    __shared__ float kmaxima[8];
#pragma unroll
    for (int it = 0; it < 4; it++) {
      int group = warp / 4 + it * 2, part = warp % 4, hh = group / 4,
          kg = group % 4, head = tn * 2 - 56 + hh;
      int anchor = anchors[head];
      float bias[4] = {0.f, 0.f, 0.f, 0.f};
      if (anchor >= 0) {
#pragma unroll
        for (int c = 0; c < 4; c++)
          bias[c] =
              float(SK[(int64_t(head) * 9 + anchor) * 128 + lane * 4 + c]);
      }
      float values[32], mx = 0.f;
#pragma unroll
      for (int j = 0; j < 4; j++) {
#pragma unroll
        for (int p = 0; p < 2; p++) {
          int tr = (j + part * 4) * 8 + kg * 2 + p, row = tm * 128 + tr,
              vi = (j * 2 + p) * 4;
#pragma unroll
          for (int c = 0; c < 4; c++)
            values[vi + c] =
                row < M ? float(tile[tr * 256 + ((hh * 128 + lane * 4 + c) ^
                                                 ((tr & 7) * 8))]) -
                              bias[c]
                        : 0.f;
          convrot128(values + vi);
        }
      }
#pragma unroll
      for (int i = 0; i < 32; i++)
        mx = fmaxf(mx, fabsf(values[i]));
#pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
      if (lane == 0)
        kmaxima[warp] = mx;
      __syncthreads();
      mx = fmaxf(
          fmaxf(kmaxima[(warp / 4) * 4], kmaxima[(warp / 4) * 4 + 1]),
          fmaxf(kmaxima[(warp / 4) * 4 + 2], kmaxima[(warp / 4) * 4 + 3]));
      float sc = fmaf(mx, 1.f / 127.f, 1e-7f), inv;
      asm volatile("rcp.approx.ftz.f32 %0,%1;" : "=f"(inv) : "f"(sc));
      if (lane == 0 && part == 0)
        KS[(int64_t(head) * ((M + 127) / 128) + tm) * 4 + kg] = sc;
#pragma unroll
      for (int j = 0; j < 4; j++) {
#pragma unroll
        for (int p = 0; p < 2; p++) {
          int tr = (j + part * 4) * 8 + kg * 2 + p, row = tm * 128 + tr,
              vi = (j * 2 + p) * 4;
          if (row < M) {
            unsigned packed = 0;
#pragma unroll
            for (int c = 0; c < 4; c++) {
              float y = values[vi + c] * inv;
              int z;
              asm volatile("cvt.rni.sat.s8.f32 %0,%1;" : "=r"(z) : "f"(y));
              packed |= (unsigned(z) & 255u) << (8 * c);
            }
            *reinterpret_cast<unsigned *>(KI + (int64_t(head) * M + row) * 128 +
                                          lane * 4) = packed;
          }
        }
      }
      __syncthreads();
    }
  }

  // Exact max of finite BF16 magnitudes; shared tile still has original V.
  // One lane owns one channel and scans the 128 rows used by QuantVPlan.
  if (tn * 256 >= 14336) {
    int ch = warp * 32 + lane, head = tn * 2 - 112 + ch / 128;
    unsigned mx = 0;
#pragma unroll 4
    for (int tr = 0; tr < 128; tr++) {
      if (tm * 128 + tr < M) {
        unsigned v =
            __bfloat16_as_ushort(tile[tr * 256 + (ch ^ ((tr & 7) * 8))]) &
            0x7fffu;
        mx = max(mx, v);
      }
    }
    VP[(int64_t(head) * ((M + 127) / 128) + tm) * 128 + ch % 128] =
        float(__ushort_as_bfloat16((unsigned short)mx));
  }
}
template <int Log>
int launch(const int8_t *A, const int8_t *B, const float *xs, const float *ws,
           void *D, const void *freqs, const void *qw, const void *kw,
           float eps, int8_t *QI, float *QS, float *VP, int8_t *KI, float *KS,
           const __nv_bfloat16 *SK, const int *anchors, int M, int N, int K,
           uintptr_t stream) {
  constexpr int smem = sizeof(typename Mma::SharedStorage) > 65536
                           ? sizeof(typename Mma::SharedStorage)
                           : 65536;
  auto k = direct<Log>;
  if (cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           smem) != cudaSuccess)
    return 0;
  dim3 grid(((M + 127) / 128) * (1 << Log),
            (((N + 255) / 256) + (1 << Log) - 1) >> Log);
  k<<<grid, 256, smem, (cudaStream_t)stream>>>(
      A, B, xs, ws, (__nv_bfloat16 *)D, (const __nv_bfloat16 *)freqs,
      (const __nv_bfloat16 *)qw, (const __nv_bfloat16 *)kw, eps, QI, QS, VP, KI,
      KS, SK, anchors, M, N, K);
  return cudaGetLastError() == cudaSuccess;
}
} // namespace h3_qkv_full

extern "C" int h3_qkv_quant_kv(const int8_t *A, const int8_t *B,
                               const float *xs, const float *ws, void *D,
                               const void *freqs, const void *qw,
                               const void *kw, float eps, int8_t *QI, float *QS,
                               float *VP, int8_t *KI, float *KS,
                               const __nv_bfloat16 *SK, const int *anchors,
                               int M, uintptr_t stream) {
  if (!A || !B || !xs || !ws || !D || !freqs || !qw || !kw || M < 1 ||
      M > 200000)
    return 0;
  return h3_qkv_full::launch<5>(A, B, xs, ws, D, freqs, qw, kw, eps, QI, QS, VP,
                                KI, KS, SK, anchors, M, 21504, 5376, stream);
}

#else
extern "C" int h3_qkv_available() { return 0; }
extern "C" int h3_qkv_quant_kv(const int8_t *, const int8_t *, const float *,
                               const float *, void *, const void *,
                               const void *, const void *, float, int8_t *,
                               float *, float *, int8_t *, float *,
                               const __nv_bfloat16 *, const int *, int,
                               uintptr_t) {
  return 0;
}
#endif
