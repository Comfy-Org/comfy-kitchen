// SPDX-License-Identifier: Apache-2.0
#include "anemoi_dispatch.h"
#include <cuda_fp16.h>
#include <math_constants.h>

namespace comfy::anemoi {
namespace {
struct Packed {
  const half *data;
  int64_t batch_stride, head_stride, token_stride;
  int heads, blocks, dim, block;
};

__global__ void count_keys(const int *counts, int *physical, int prefix,
                           int prefix_blocks, int video_blocks, int split) {
  int i = int(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < prefix_blocks) {
    physical[i] = min(64, max(0, prefix - i * 64));
  } else if (i < prefix_blocks + video_blocks * split) {
    int v = i - prefix_blocks;
    physical[i] = min(64, max(0, counts[v / split] - (v % split) * 64));
  }
}

template <bool Second>
__global__ void block_statistics(Packed x, const int *counts, half *mean, half *maximum) {
  int row = blockIdx.x;
  int d = threadIdx.x;
  if (d >= x.dim)
    return;
  int r = row % x.blocks;
  int head = (row / x.blocks) % x.heads;
  int batch = row / (x.blocks * x.heads);
  auto input = x.data + batch * x.batch_stride + head * x.head_stride +
               int64_t(r * x.block) * x.token_stride + d;
  // Match ATen's non-fastest-dimension FP32 sum: four independent
  // accumulators per reduction lane, combined left-to-right, then a descending
  // binary tree across reduction lanes. A sequential sum changes Half-rounded
  // Jensen moments near ties, even when multiply/add contraction is disabled.
  const int lane = threadIdx.y;
  const int lanes = blockDim.y;
  float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float high = -CUDART_INF_F;
  for (int t = lane; t < x.block; t += 4 * lanes) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int token = t + j * lanes;
      float value = __half2float(input[int64_t(token) * x.token_stride]);
      sums[j] = __fadd_rn(sums[j], Second ? __fmul_rn(value, value) : value);
      if (token < counts[r])
        high = fmaxf(high, value);
    }
  }
  float sum = __fadd_rn(__fadd_rn(__fadd_rn(sums[0], sums[1]), sums[2]), sums[3]);
  __shared__ float partial[8][128];
  __shared__ float maxima[8][128];
  partial[lane][d] = sum;
  if (maximum) maxima[lane][d] = high;
  for (int offset = lanes / 2; offset > 0; offset /= 2) {
    __syncthreads();
    if (lane < offset) {
      sum = __fadd_rn(sum, partial[lane + offset][d]);
      partial[lane][d] = sum;
      if (maximum) {
        high = fmaxf(high, maxima[lane + offset][d]);
        maxima[lane][d] = high;
      }
    }
  }
  if (lane == 0) {
    mean[int64_t(row) * x.dim + d] = __float2half_rn(__fdiv_rn(sum, float(counts[r])));
    if (maximum)
      maximum[int64_t(row) * x.dim + d] = __float2half_rn(high);
  }
}

__global__ void write_scale(float *q, float *k, float *v, int nq, int nk, int nv,
                            float qs, float ks, float vs) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nq) q[i] = qs;
  if (i < nk) k[i] = ks;
  if (i < nv) v[i] = vs;
}

// ATen vectorizes four contiguous FP32 output channels. Its 512-thread
// budget becomes 128 vector threads; width is capped at a warp and height
// splits the reduction only when each lane retains at least 16 values.
// For normal pools this is four lanes. The single-row D64 case uses one lane
// for Q64 and eight lanes for Q128. No cross-CTA reduction is needed here.
dim3 statistics_threads(const Tensor &t, const Packed &x) {
  int64_t outputs = t.size(0) * x.heads * x.blocks * x.dim;
  int width = 1;
  while (width < 32 && width * 2 <= outputs / 4) width *= 2;
  int height = 128 / width;
  int lanes = x.block >= height * 16 ? height : 1;
  return dim3(x.dim, lanes);
}

Packed packed_view(const Tensor &t, const Tensor &counts, int block) {
  anemoi_native::check(t.dim() == 4 && t.scalar_type() == DType::Half &&
                          (block == 64 || block == 128) &&
                          (t.size(3) == 64 || t.size(3) == 128) &&
                          t.stride(3) == 1 && t.size(2) == counts.numel() * block,
                      "invalid packed tensor for block statistics");
  return {t.data_ptr<half>(), t.stride(0), t.stride(1), t.stride(2),
          int(t.size(1)), int(counts.numel()), int(t.size(3)), block};
}
}

void valid_key_counts(const Config &c, const Tensor &logical, const Tensor &physical,
                      cudaStream_t stream) {
  int prefix_blocks = (c.prefix_tokens + 63) / 64;
  int n = prefix_blocks + c.video_blocks * (c.query_block / 64);
  anemoi_native::check(physical.numel() == n, "physical K64 count capacity mismatch");
  count_keys<<<(n + 255) / 256, 256, 0, stream>>>(
      logical.data_ptr<int>(), physical.data_ptr<int>(), c.prefix_tokens,
      prefix_blocks, c.video_blocks, c.query_block / 64);
  anemoi_native::cuda_check(cudaGetLastError());
}
void moments(const Tensor &packed, const Tensor &counts, int block,
             const Tensor &second, cudaStream_t stream) {
  auto x = packed_view(packed, counts, block);
  block_statistics<true><<<packed.size(0) * x.heads * x.blocks,
                            statistics_threads(packed, x), 0, stream>>>(
      x, counts.data_ptr<int>(), second.data_ptr<half>(), nullptr);
  anemoi_native::cuda_check(cudaGetLastError());
}
void pool(const Tensor &packed, const Tensor &counts, int block,
          const Tensor &mean, const Tensor &maximum, cudaStream_t stream) {
  auto x = packed_view(packed, counts, block);
  block_statistics<false><<<packed.size(0) * x.heads * x.blocks,
                             statistics_threads(packed, x), 0, stream>>>(
      x, counts.data_ptr<int>(), mean.data_ptr<half>(),
      maximum.numel() ? maximum.data_ptr<half>() : nullptr);
  anemoi_native::cuda_check(cudaGetLastError());
}
void fill_scales(const Config &c, const Tensor &q, const Tensor &k,
                 const Tensor &v, cudaStream_t stream) {
  int n = int(std::max({q.numel(), k.numel(), v.numel()}));
  if (n) {
    write_scale<<<(n + 255) / 256, 256, 0, stream>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        int(q.numel()), int(k.numel()), int(v.numel()), c.global_scales[0],
        c.global_scales[1], c.global_scales[2]);
    anemoi_native::cuda_check(cudaGetLastError());
  }
}
} // namespace comfy::anemoi
