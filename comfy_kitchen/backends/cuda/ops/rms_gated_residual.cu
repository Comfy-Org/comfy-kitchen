// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Exact BF16 RMSNorm followed by the separately rounded BF16 gate product and
// residual add. One 128-thread block owns one row.
//
// The statistics reproduce PyTorch's CUDA vectorized RMSNorm
// (aten/src/ATen/native/cuda/layer_norm_kernel.cu, compute_stats<T, true>):
// four warps, each thread accumulating x * x in strided four-element vectors,
// a shuffle-down warp tree, the pairwise inter-warp merge, then an IEEE
// division by N and rsqrt. This file is built with --use_fast_math, so the
// division is spelled __fdiv_rn to keep PyTorch's correctly rounded quotient.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace comfy {
namespace {

constexpr int kRmsGatedThreads = 128;
constexpr int kRmsGatedWarps = kRmsGatedThreads / 32;
constexpr int kRmsGatedMaxWidth = 16384;

struct alignas(8) Bf16x4 {
  __nv_bfloat16 v[4];
};

__global__ __launch_bounds__(kRmsGatedThreads) void rms_gated_residual_bf16_kernel(
    const __nv_bfloat16 *__restrict__ activation,
    const __nv_bfloat16 *__restrict__ norm_weight,
    const __nv_bfloat16 *__restrict__ residual,
    const __nv_bfloat16 *__restrict__ gate,
    __nv_bfloat16 *__restrict__ output, int width, float eps) {
  __shared__ float warp_sums[kRmsGatedWarps];

  const int thread = threadIdx.x;
  const int lane = thread & 31;
  const int warp = thread >> 5;
  const int vectors = width / 4;
  const int64_t row_offset = static_cast<int64_t>(blockIdx.x) * width;

  const auto *input_vectors =
      reinterpret_cast<const Bf16x4 *>(activation + row_offset);

  float sum_sq = 0.0f;
  for (int vector = thread; vector < vectors; vector += kRmsGatedThreads) {
    const Bf16x4 values = input_vectors[vector];
#pragma unroll
    for (int element = 0; element < 4; ++element) {
      const float value = __bfloat162float(values.v[element]);
      sum_sq = __fmaf_rn(value, value, sum_sq);
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum_sq += __shfl_down_sync(0xffffffffu, sum_sq, offset);
  }

  // PyTorch's pairwise compute_stats() merge: upper half writes, lower adds.
#pragma unroll
  for (int offset = kRmsGatedWarps / 2; offset > 0; offset >>= 1) {
    if (lane == 0 && warp >= offset && warp < 2 * offset) {
      warp_sums[warp - offset] = sum_sq;
    }
    __syncthreads();
    if (lane == 0 && warp < offset) {
      sum_sq += warp_sums[warp];
    }
    __syncthreads();
  }
  if (thread == 0) {
    warp_sums[0] =
        rsqrtf(__fdiv_rn(sum_sq, static_cast<float>(width)) + eps);
  }
  __syncthreads();
  const float rstd = warp_sums[0];

  const auto *weight_vectors = reinterpret_cast<const Bf16x4 *>(norm_weight);
  const auto *residual_vectors =
      reinterpret_cast<const Bf16x4 *>(residual + row_offset);
  const auto *gate_vectors = reinterpret_cast<const Bf16x4 *>(gate);
  auto *output_vectors = reinterpret_cast<Bf16x4 *>(output + row_offset);

  for (int vector = thread; vector < vectors; vector += kRmsGatedThreads) {
    const Bf16x4 raw = input_vectors[vector];
    const Bf16x4 weight = weight_vectors[vector];
    const Bf16x4 res = residual_vectors[vector];
    const Bf16x4 g = gate_vectors[vector];
    Bf16x4 result;
#pragma unroll
    for (int element = 0; element < 4; ++element) {
      // rms_norm materializes gamma * (rstd * x) in BF16.
      const float normalized = __bfloat162float(__float2bfloat16_rn(__fmul_rn(
          __bfloat162float(weight.v[element]),
          __fmul_rn(rstd, __bfloat162float(raw.v[element])))));
      // Eager exit is two BF16 kernels, each with its own rounding point.
      const float product = __bfloat162float(__float2bfloat16_rn(
          __fmul_rn(__bfloat162float(g.v[element]), normalized)));
      result.v[element] = __float2bfloat16_rn(
          __fadd_rn(__bfloat162float(res.v[element]), product));
    }
    output_vectors[vector] = result;
  }
}

}  // namespace
}  // namespace comfy

extern "C" void launch_rms_gated_residual_bf16_kernel(
    const void *activation, const void *norm_weight, const void *residual,
    const void *gate, void *output, int rows, int width, float eps,
    cudaStream_t stream) {
  using namespace comfy;
  if (rows < 0 || width <= 0 || width % 4 != 0 ||
      width > kRmsGatedMaxWidth) {
    throw std::runtime_error(
        "rms_gated_residual_bf16: requires non-negative rows and positive "
        "width divisible by 4 and at most 16384");
  }
  if (rows == 0) return;
  rms_gated_residual_bf16_kernel<<<rows, kRmsGatedThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(activation),
      static_cast<const __nv_bfloat16 *>(norm_weight),
      static_cast<const __nv_bfloat16 *>(residual),
      static_cast<const __nv_bfloat16 *>(gate),
      static_cast<__nv_bfloat16 *>(output), width, eps);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("rms_gated_residual_bf16 kernel launch failed: ") +
        cudaGetErrorString(err));
  }
}
