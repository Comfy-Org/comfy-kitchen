// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// K detector from Kitchen/SageAttention; same nine-key criterion as
// sage_sdpa_quantize.
#include "runtime.cuh"
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
namespace {
constexpr int CENTER_DETECT_THREADS = 128;
constexpr int CENTER_SAMPLES = 9;
constexpr int CENTER_MAX_CHANNELS = 256;
template <typename T>
__global__ __launch_bounds__(CENTER_DETECT_THREADS) void detect_k_anchor(
    const T *__restrict__ k_in, int *__restrict__ anchor_indices, const int Lk,
    const int C, const int H_kv, const int64_t stride_b, const int64_t stride_h,
    const int64_t stride_n) {
  const int h = blockIdx.x;
  const int b = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int64_t bh_offset = (int64_t)b * stride_b + (int64_t)h * stride_h;

  __shared__ float samples[CENTER_SAMPLES * CENTER_MAX_CHANNELS];
  __shared__ float warp_original_energy[4];
  __shared__ float warp_original_max[4];
  __shared__ float warp_candidate_distance[CENTER_SAMPLES][4];
  __shared__ float warp_best_energy[4];
  __shared__ float warp_best_max[4];
  __shared__ int selected_candidate;

  for (int index = tid; index < CENTER_SAMPLES * C;
       index += CENTER_DETECT_THREADS) {
    const int sample = index / C;
    const int channel = index - sample * C;
    const int row = sample * (Lk - 1) / (CENTER_SAMPLES - 1);
    samples[index] = static_cast<float>(
        __ldg(&k_in[bh_offset + (int64_t)row * stride_n + channel]));
  }
  __syncthreads();

  float original_energy = 0.f;
  float original_max = 0.f;
  float candidate_distance[CENTER_SAMPLES];
#pragma unroll
  for (int candidate = 0; candidate < CENTER_SAMPLES; ++candidate) {
    candidate_distance[candidate] = 0.f;
  }

  for (int channel = tid; channel < C; channel += CENTER_DETECT_THREADS) {
    float channel_sum = 0.f;
#pragma unroll
    for (int sample = 0; sample < CENTER_SAMPLES; ++sample) {
      const float value = samples[sample * C + channel];
      original_energy = fmaf(value, value, original_energy);
      original_max = fmaxf(original_max, fabsf(value));
      channel_sum += value;
    }
#pragma unroll
    for (int candidate = 0; candidate < CENTER_SAMPLES; ++candidate) {
      const float distance =
          CENTER_SAMPLES * samples[candidate * C + channel] - channel_sum;
      candidate_distance[candidate] =
          fmaf(distance, distance, candidate_distance[candidate]);
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    original_energy += __shfl_down_sync(0xffffffffu, original_energy, offset);
    original_max = fmaxf(original_max,
                         __shfl_down_sync(0xffffffffu, original_max, offset));
#pragma unroll
    for (int candidate = 0; candidate < CENTER_SAMPLES; ++candidate) {
      candidate_distance[candidate] +=
          __shfl_down_sync(0xffffffffu, candidate_distance[candidate], offset);
    }
  }

  if (lane == 0) {
    warp_original_energy[warp] = original_energy;
    warp_original_max[warp] = original_max;
#pragma unroll
    for (int candidate = 0; candidate < CENTER_SAMPLES; ++candidate) {
      warp_candidate_distance[candidate][warp] = candidate_distance[candidate];
    }
  }
  __syncthreads();

  if (tid == 0) {
    int best_candidate = 0;
    float best_distance = 3.402823466e+38F;
#pragma unroll
    for (int candidate = 0; candidate < CENTER_SAMPLES; ++candidate) {
      float distance = 0.f;
#pragma unroll
      for (int w = 0; w < 4; ++w) {
        distance += warp_candidate_distance[candidate][w];
      }
      if (distance < best_distance) {
        best_candidate = candidate;
        best_distance = distance;
      }
    }
    selected_candidate = best_candidate;
  }
  __syncthreads();

  float best_energy = 0.f;
  float best_max = 0.f;
  for (int channel = tid; channel < C; channel += CENTER_DETECT_THREADS) {
    const float anchor = samples[selected_candidate * C + channel];
#pragma unroll
    for (int sample = 0; sample < CENTER_SAMPLES; ++sample) {
      const float residual = samples[sample * C + channel] - anchor;
      best_energy = fmaf(residual, residual, best_energy);
      best_max = fmaxf(best_max, fabsf(residual));
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    best_energy += __shfl_down_sync(0xffffffffu, best_energy, offset);
    best_max = fmaxf(best_max, __shfl_down_sync(0xffffffffu, best_max, offset));
  }
  if (lane == 0) {
    warp_best_energy[warp] = best_energy;
    warp_best_max[warp] = best_max;
  }
  __syncthreads();

  if (tid == 0) {
    float total_original_energy = 0.f;
    float total_original_max = 0.f;
    float total_best_energy = 0.f;
    float total_best_max = 0.f;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      total_original_energy += warp_original_energy[w];
      total_original_max = fmaxf(total_original_max, warp_original_max[w]);
      total_best_energy += warp_best_energy[w];
      total_best_max = fmaxf(total_best_max, warp_best_max[w]);
    }

    const bool improves_range = total_best_energy < total_original_energy &&
                                total_best_max <= total_original_max * 1.125f;
    anchor_indices[b * H_kv + h] =
        improves_range ? selected_candidate * (Lk - 1) / (CENTER_SAMPLES - 1)
                       : -1;
  }
}
} // namespace
extern "C" int h3_sample_anchor(const void *k, void *anchors,
                                uintptr_t stream) {
  detect_k_anchor<__nv_bfloat16><<<dim3(56, 1), 128, 0, (cudaStream_t)stream>>>(
      (const __nv_bfloat16 *)k, (int *)anchors, 9, 128, 56,
      int64_t(56) * 9 * 128, int64_t(9) * 128, 128);
  return cudaGetLastError() == cudaSuccess;
}

__global__ void finish_scales(const float *partial, float *scales,
                              float *inverse, int parts) {
  int ch = blockIdx.x * 256 + threadIdx.x;
  if (ch >= 56 * 128)
    return;
  int head = ch / 128, channel = ch % 128;
  float mx = 0.f;
  for (int p = 0; p < parts; ++p)
    mx = fmaxf(mx, partial[(int64_t(head) * parts + p) * 128 + channel]);
  float product;
  asm("mul.rn.ftz.f32 %0, %1, 0f3C010204;" : "=f"(product) : "f"(mx));
  float scale = fmaxf(product, 1e-12f), inv;
  asm("rcp.approx.ftz.f32 %0,%1;" : "=f"(inv) : "f"(scale));
  scales[ch] = scale;
  inverse[ch] = inv;
}
__global__ void quantize_v(const __nv_bfloat16 *v, int8_t *out,
                           const float *inv, int m, int padded) {
  int64_t index = int64_t(blockIdx.x) * 256 + threadIdx.x;
  if (index >= int64_t(56) * 128 * padded)
    return;
  int dst = index % padded, ch = index / padded;
  int src = (dst & ~15) | (dst & 1) | ((dst & 2) << 2) | ((dst & 4) >> 1) |
            ((dst & 8) >> 1);
  float value =
      src < m ? float(v[(int64_t(ch / 128) * m + src) * 128 + ch % 128]) : 0.f;
  float product;
  int code;
  asm("mul.rn.ftz.f32 %0,%1,%2;" : "=f"(product) : "f"(value), "f"(inv[ch]));
  asm("cvt.rni.sat.s8.f32 %0,%1;" : "=r"(code) : "f"(product));
  out[index] = int8_t(code);
}
extern "C" int h3_finish_v(const void *v, const float *partial, float *scales,
                           float *inv, int8_t *out, int m, uintptr_t st) {
  int parts = (m + 127) / 128, padded = parts * 128;
  finish_scales<<<28, 256, 0, (cudaStream_t)st>>>(partial, scales, inv, parts);
  if (cudaGetLastError() != cudaSuccess)
    return 0;
  quantize_v<<<(int64_t(56) * 128 * padded + 255) / 256, 256, 0,
               (cudaStream_t)st>>>((const __nv_bfloat16 *)v, out, inv, m,
                                   padded);
  return cudaGetLastError() == cudaSuccess;
}

extern "C" int h3_finish_available() {
  return h3_qkv::loadable(detect_k_anchor<__nv_bfloat16>) &&
         h3_qkv::loadable(finish_scales) && h3_qkv::loadable(quantize_v);
}
