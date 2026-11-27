// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
// Norm reduction derived from PyTorch commit
// 7661cd9c6b841b62b7f411aa52ec51f05457263b. See
// comfy_kitchen/licenses/PYTORCH_LICENSE.txt. ConvRot layout derived from
// SGLang PR #38040, commit 3299804f81b686d3752287e19d9c9d2987d1afc1, Copyright
// 2026 SGLang Team. ConvRot arithmetic/quantization from Comfy Kitchen; see
// vendor and Apache license. Changes: join the previously qualified native
// norm/mod and register-v2 kernels within one 192-thread CTA. No intermediate
// norm/mod global store in mode 0.
#include "utils.cuh"
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_bf16.h>
#include <stdexcept>

namespace {
using bf16 = nv_bfloat16;
constexpr int K = 5376, Threads = 192, Warps = 6;
constexpr unsigned Mask = 0xffffffffu;

// This TU uses v2's --use_fast_math. The native norm prefix was compiled
// WITHOUT fast math. Explicit PTX excludes .ftz and contraction in every
// native FP32 operation, preserving subnormal values and its original tree.
__device__ __forceinline__ float native_add(float a, float b) {
  float c;
  asm("add.rn.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
  return c;
}
__device__ __forceinline__ float native_mul(float a, float b) {
  float c;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
  return c;
}
__device__ __forceinline__ float native_fma(float a, float b, float c) {
  float d;
  asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
  return d;
}
__device__ __forceinline__ float native_div(float a, float b) {
  float c;
  asm("div.rn.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
  return c;
}
__device__ __forceinline__ float native_rsqrt(float a) {
  // ABI requires eps >= FLT_MIN. The rsqrt input is therefore positive and
  // normal, where native non-fast-math rsqrtf uses this same approximation.
  float b;
  asm("rsqrt.approx.f32 %0, %1;" : "=f"(b) : "f"(a));
  return b;
}
__device__ __forceinline__ float from_bits(unsigned short value) {
  return __uint_as_float(static_cast<unsigned>(value) << 16);
}
__device__ __forceinline__ unsigned short bf_bits(float a) {
  unsigned short b;
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(b) : "f"(a));
  return b;
}
__device__ __forceinline__ float bfround(float a) {
  return from_bits(bf_bits(a));
}
template <class T> __device__ __forceinline__ float load_mod(const T *p) {
  return *p;
}
template <> __device__ __forceinline__ float load_mod<bf16>(const bf16 *p) {
  return from_bits(*reinterpret_cast<const unsigned short *>(p));
}

__device__ __forceinline__ float stats(const bf16 *x, float *buf) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  float sum = 0.0f;
  // Only original logical tid=0..127 contributes; all 192 physical threads
  // execute every CTA barrier. Warp 4/5 are never folded into the native sum.
  if (tid < 128) {
    for (int i = tid; i < K / 4; i += 128) {
      const uint2 raw = reinterpret_cast<const uint2 *>(x)[i];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const unsigned word = j < 2 ? raw.x : raw.y;
        const float value =
            from_bits(static_cast<unsigned short>(word >> ((j & 1) * 16)));
        sum = native_fma(value, value, sum);
      }
    }
    for (int offset = 16; offset > 0; offset >>= 1)
      sum = native_add(sum, __shfl_down_sync(Mask, sum, offset));
  }
  for (int offset = 2; offset > 0; offset /= 2) {
    if (lane == 0 && warp >= offset && warp < 2 * offset)
      buf[warp - offset] = sum;
    __syncthreads();
    if (lane == 0 && warp < offset)
      sum = native_add(sum, buf[warp]);
    __syncthreads();
  }
  if (tid == 0)
    buf[0] = native_div(sum, float(K));
  __syncthreads();
  return buf[0];
}

// Copied from the already selected register/vector v2; arithmetic expressions
// and global Q packing remain unchanged. The source manifest pins its origin.
__device__ __forceinline__ int run_offset(int q, int j) {
  return ((q & 1) << 3) | ((j & 1) << 4) | (((q >> 1) & 1) << 5) |
         ((j >> 1) << 6) | (((q >> 2) & 1) << 7);
}
__device__ __forceinline__ void stage_register(float (&v)[32]) {
#pragma unroll
  for (int r = 0; r < 32; r += 4) {
    const float x0 = v[r], x1 = v[r + 1], x2 = v[r + 2], x3 = v[r + 3];
    v[r] = 0.5f * (x0 + x1 + x2 - x3);
    v[r + 1] = 0.5f * (x0 + x1 - x2 + x3);
    v[r + 2] = 0.5f * (x0 - x1 + x2 + x3);
    v[r + 3] = 0.5f * (-x0 + x1 + x2 + x3);
  }
}
template <int RegisterBit, int LaneXor>
__device__ __forceinline__ void stage_split(float (&v)[32], int lane) {
  const bool high = (lane & LaneXor) != 0;
#pragma unroll
  for (int r = 0; r < 32; ++r)
    if ((r & RegisterBit) == 0) {
      const float own0 = v[r], own1 = v[r | RegisterBit];
      const float peer0 = __shfl_xor_sync(Mask, own0, LaneXor);
      const float peer1 = __shfl_xor_sync(Mask, own1, LaneXor);
      const float x0 = high ? peer0 : own0, x1 = high ? peer1 : own1;
      const float x2 = high ? own0 : peer0, x3 = high ? own1 : peer1;
      if (high) {
        v[r] = 0.5f * (x0 - x1 + x2 + x3);
        v[r | RegisterBit] = 0.5f * (-x0 + x1 + x2 + x3);
      } else {
        v[r] = 0.5f * (x0 + x1 + x2 - x3);
        v[r | RegisterBit] = 0.5f * (x0 + x1 - x2 + x3);
      }
    }
}
__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    value = fmaxf(value, __shfl_xor_sync(Mask, value, offset));
  return value;
}

template <class ModT>
__global__ void __launch_bounds__(Threads)
    norm_quant_kernel(const bf16 *__restrict__ X, const bf16 *__restrict__ W,
                      const ModT *__restrict__ Shift,
                      const ModT *__restrict__ Scale,
                      const int32_t *__restrict__ Rows, int8_t *__restrict__ Q,
                      float *__restrict__ QScale, int64_t XS, int64_t SS,
                      int64_t CS, int64_t table_rows, float eps) {
  __shared__ float shared[Warps];
  const int row = blockIdx.x, tid = threadIdx.x, lane = tid & 31,
            warp = tid >> 5;
  const bf16 *xr = X + int64_t(row) * XS;
  const float variance = stats(xr, shared);
  const float rrms = native_rsqrt(native_add(variance, eps));
  // All threads must finish reading variance before shared is reused for
  // absmax. The later data-dependent path does not synchronize those reads.
  __syncthreads();
  const int mr = Rows[row];
  const bool valid = mr >= 0 && int64_t(mr) < table_rows;
  const ModT *sr = valid ? Shift + int64_t(mr) * SS : nullptr;
  const ModT *cr = valid ? Scale + int64_t(mr) * CS : nullptr;
  const int group = warp * 4 + lane / 8, cluster_lane = lane & 7;
  const bool active = group < K / 256;
  const int64_t out_offset = int64_t(row) * K;
  float v[32];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int col = group * 256 + run_offset(cluster_lane, j);
    // Two 8-byte vectors also support native's XS % 4 == 0 contract. In the
    // usual contiguous case both are adjacent, aligned BF16 vec4 loads.
    uint2 xa = make_uint2(0, 0), xb = make_uint2(0, 0), wa = make_uint2(0, 0),
          wb = make_uint2(0, 0);
    if (active) {
      xa = *reinterpret_cast<const uint2 *>(xr + col);
      xb = *reinterpret_cast<const uint2 *>(xr + col + 4);
      wa = *reinterpret_cast<const uint2 *>(W + col);
      wb = *reinterpret_cast<const uint2 *>(W + col + 4);
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const unsigned xword = e < 2 ? xa.x : e < 4 ? xa.y : e < 6 ? xb.x : xb.y;
      const unsigned wword = e < 2 ? wa.x : e < 4 ? wa.y : e < 6 ? wb.x : wb.y;
      const float xv =
          from_bits(static_cast<unsigned short>(xword >> ((e & 1) * 16)));
      const float wv =
          from_bits(static_cast<unsigned short>(wword >> ((e & 1) * 16)));
      float n = 0.0f;
      if (active) {
        n = bfround(native_mul(wv, native_mul(rrms, xv)));
        const float s = valid ? bfround(load_mod(sr + col + e)) : 0.0f;
        const float c = valid ? bfround(load_mod(cr + col + e)) : 0.0f;
        const float one_c = bfround(native_add(1.0f, c));
        const float p = bfround(native_mul(n, one_c));
        n = bfround(native_add(p, s));
      }
      v[j * 8 + e] = n;
    }
  }
  stage_register(v);
  stage_split<4, 1>(v, lane);
  stage_split<8, 2>(v, lane);
  stage_split<16, 4>(v, lane);
  float local = 0.0f;
#pragma unroll
  for (int r = 0; r < 32; ++r)
    local = fmaxf(local, fabsf(v[r]));
  local = warp_max(local);
  if (lane == 0)
    shared[warp] = local;
  __syncthreads();
  const float abs_max = warp_max(lane < Warps ? shared[lane] : 0.0f);
  const float scale =
      fmaxf(fminf(abs_max, 3.38953139e38f) * (1.0f / 127.0f), 1.0e-30f);
  if (tid == 0) {
    QScale[row] = scale;
  }
  if (active) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int col = group * 256 + run_offset(cluster_lane, j);
      uint32_t low = 0, high = 0;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float value = v[j * 8 + e];

        const float scaled = bfround(bfround(value) / bfround(scale));
        float quantized = nearbyintf(scaled);
        quantized = fminf(127.0f, fmaxf(-128.0f, quantized));
        const uint32_t byte =
            static_cast<unsigned char>(static_cast<int8_t>(quantized));
        if (e < 4)
          low |= byte << (8 * e);
        else
          high |= byte << (8 * (e - 4));
      }
      *reinterpret_cast<uint2 *>(Q + out_offset + col) = make_uint2(low, high);
    }
  }
}

} // namespace

extern "C" bool
launch_indexed_norm_convrot(const void *x, const void *w, const void *shift,
                            const void *scale, const int32_t *rows, int8_t *q,
                            float *qscale, int64_t m, int64_t k, int64_t xs,
                            int64_t ss, int64_t cs, int64_t table_rows,
                            float eps, int table_dtype, cudaStream_t stream) {
  if (k != K || m < 1 || m > INT32_MAX || xs < K || xs % 4 || ss < K ||
      cs < K || table_rows > INT32_MAX || (uintptr_t(x) & 7u) ||
      (uintptr_t(w) & 7u) || (uintptr_t(q) & 7u) || (uintptr_t(qscale) & 3u) ||
      (uintptr_t(rows) & 3u))
    return false;
  int device = 0, major = 0, minor = 0;
  auto error = cudaGetDevice(&device);
  if (error == cudaSuccess)
    error = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                                   device);
  if (error == cudaSuccess)
    error = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                                   device);
  if (error != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(error));
  if (major != 12 || minor != 0)
    return false;
#define LAUNCH(T)                                                              \
  norm_quant_kernel<T><<<m, Threads, 0, stream>>>(                             \
      static_cast<const bf16 *>(x), static_cast<const bf16 *>(w),              \
      static_cast<const T *>(shift), static_cast<const T *>(scale), rows, q,   \
      qscale, xs, ss, cs, table_rows, eps)
  if (table_dtype == 0) {
    LAUNCH(float);
  } else {
    LAUNCH(bf16);
  }
#undef LAUNCH
  error = cudaGetLastError();
  if (error != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(error));
  return true;
}
