// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// H3 INT8 projection with BF16 per-head RMS/RoPE rounding.
#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifdef COMFY_HAVE_CUTLASS

#include "mma.cuh"

namespace h3_qkv_sample {
using h3_qkv::Mma;
template <int Log>
__global__ __launch_bounds__(256) void direct(
    const int8_t *A, const int8_t *B, const float *xs, const float *ws,
    __nv_bfloat16 *D, const __nv_bfloat16 *freqs, const __nv_bfloat16 *qw,
    const __nv_bfloat16 *kw, float eps, int8_t *QI, float *QS, float *VP, int M,
    int N, int K) {
  int tm = blockIdx.x >> Log,
      tn = 28 + (blockIdx.y << Log) + (blockIdx.x & ((1 << Log) - 1));
  if (tm * 128 >= M || tn >= 28 + 28)
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
    constexpr int group = 1;
    int head = tn * 2 + hh - 56;
    __nv_bfloat16 *t = tile + tr * 256;
    __nv_bfloat16 *dest =
        D + int64_t(group * 56 + head) * M * 128 + int64_t(row) * 128;
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
      }
    }
    int d = 96 + lane;
    float x = float(t[(hh * 128 + d) ^ ((tr & 7) * 8)]);
    auto tail = __float2bfloat16_rn(x * rrms * float(w[d]));
    dest[d] = tail;
  }
}
template <int Log>
int launch(const int8_t *A, const int8_t *B, const float *xs, const float *ws,
           void *D, const void *freqs, const void *qw, const void *kw,
           float eps, int8_t *QI, float *QS, float *VP, int M, int N, int K,
           uintptr_t stream) {
  constexpr int smem = sizeof(typename Mma::SharedStorage) > 65536
                           ? sizeof(typename Mma::SharedStorage)
                           : 65536;
  auto k = direct<Log>;
  if (cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           smem) != cudaSuccess)
    return 0;
  dim3 grid(((M + 127) / 128) * (1 << Log), (28 + (1 << Log) - 1) >> Log);
  k<<<grid, 256, smem, (cudaStream_t)stream>>>(
      A, B, xs, ws, (__nv_bfloat16 *)D, (const __nv_bfloat16 *)freqs,
      (const __nv_bfloat16 *)qw, (const __nv_bfloat16 *)kw, eps, QI, QS, VP, M,
      N, K);
  return cudaGetLastError() == cudaSuccess;
}
} // namespace h3_qkv_sample

extern "C" int h3_sample_konly(const int8_t *A, const int8_t *B,
                               const float *xs, const float *ws, void *D,
                               const void *freqs, const void *qw,
                               const void *kw, float eps, int8_t *QI, float *QS,
                               float *VP, int M, uintptr_t stream) {
  if (!A || !B || !xs || !ws || !D || !freqs || !qw || !kw || M < 1 ||
      M > 200000)
    return 0;
  if (M != 9)
    return 0;
  return h3_qkv_sample::launch<4>(A, B, xs, ws, D, freqs, qw, kw, eps, QI, QS,
                                  VP, M, 21504, 5376, stream);
}

#else
extern "C" int h3_sample_konly(const int8_t *, const int8_t *, const float *,
                               const float *, void *, const void *,
                               const void *, const void *, float, int8_t *,
                               float *, float *, int, uintptr_t) {
  return 0;
}
#endif

__global__ void gather_sample_rows(const int8_t *Q, const float *XS,
                                   const __nv_bfloat16 *ROPE, int8_t *SQ,
                                   float *SX, __nv_bfloat16 *SR, int M) {
  int i = blockIdx.x * 256 + threadIdx.x;
  if (i < 9 * 5376) {
    int s = i / 5376, c = i % 5376;
    int row = (int64_t(s) * (M - 1)) / 8;
    SQ[i] = Q[int64_t(row) * 5376 + c];
  }
  if (i < 9 * 192) {
    int s = i / 192, c = i % 192;
    int row = (int64_t(s) * (M - 1)) / 8;
    SR[i] = ROPE[int64_t(row) * 192 + c];
  }
  if (i < 9) {
    int row = (int64_t(i) * (M - 1)) / 8;
    SX[i] = XS[row];
  }
}
extern "C" int h3_sample_gather(const void *Q, const void *XS, const void *ROPE,
                                void *SQ, void *SX, void *SR, int M,
                                uintptr_t st) {
  if (M < 1 || M > 200000)
    return 0;
  gather_sample_rows<<<(9 * 5376 + 255) / 256, 256, 0, (cudaStream_t)st>>>(
      (const int8_t *)Q, (const float *)XS, (const __nv_bfloat16 *)ROPE,
      (int8_t *)SQ, (float *)SX, (__nv_bfloat16 *)SR, M);
  return cudaGetLastError() == cudaSuccess;
}
