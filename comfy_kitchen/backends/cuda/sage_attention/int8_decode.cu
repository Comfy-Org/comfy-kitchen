// SPDX-License-Identifier: Apache-2.0
#include "qk_int_sv_i8_cuda.cuh"
#include <stdexcept>

namespace {
// One CK integer-attention CTA per KV page/head. GQA groups occupy query rows.
__global__ void int8_decode_pages(
    int8_t* q, int8_t* k, int8_t* v, float* qs, float* ks, float* vs,
    const int64_t* lengths, nv_bfloat16* partial, float* lse,
    int heads, int rows, int pages, int page_size) {
  const int b = blockIdx.z, page = blockIdx.x;
  const int live = min(page_size, max(0, static_cast<int>(lengths[0]) - page * page_size));
  if (!live) return;
  const int64_t slab = (static_cast<int64_t>(b) * pages + page) * heads;
  const int capacity = pages * page_size;
  qk_int_sv_i8_attn_body<false, 64, 64, 16, 64, 256, DataType::kInt8,
      QuantGranularity::kPerThread, QuantGranularity::kPerThread, float, false,
      nv_bfloat16, ComputeUnit::kCudaCore, MaskMode::kNone, true, true, false,
      false, true, false, uint32_t, true>(
      q + static_cast<int64_t>(b) * heads * rows * 256,
      k + (static_cast<int64_t>(b) * heads * capacity + page * page_size) * 256,
      v + static_cast<int64_t>(b) * heads * 256 * capacity + page * page_size,
      partial + slab * rows * 256, lse + slab * rows,
      qs + b * heads * 64,
      ks + static_cast<int64_t>(b) * heads * (capacity / 64) * 4 + page * (page_size / 64) * 4,
      vs + slab * 256,
      nullptr, nullptr, 0, 0, 0, 0, 0, rows, live, 1,
      0, 256, rows * 256, 0, 256, capacity * 256,
      0, capacity * 256, capacity, 0, 256, rows * 256, 0.0625f, nullptr);
}

// Merge page-local normalized outputs in their shared (uncentered) score domain.
__global__ void int8_decode_combine(
    const nv_bfloat16* partial, const float* lse, const int64_t* lengths,
    nv_bfloat16* out, float* out_lse, int heads, int rows, int pages, int page_size, int seq) {
  const int row = blockIdx.x % rows, h = blockIdx.x / rows % heads;
  const int b = blockIdx.x / (heads * rows), d = threadIdx.x;
  const int live_pages = (static_cast<int>(lengths[0]) + page_size - 1) / page_size;
  const int64_t base = (static_cast<int64_t>(b) * pages * heads + h) * rows + row;
  float m = -INFINITY;
  for (int p = 0; p < live_pages; ++p)
    m = fmaxf(m, lse[base + static_cast<int64_t>(p) * heads * rows]);
  float sum = 0.0f, val = 0.0f;
  for (int p = 0; p < live_pages; ++p) {
    const int64_t offset = base + static_cast<int64_t>(p) * heads * rows;
    const float w = exp2f(lse[offset] - m);
    sum += w;
    val = fmaf(w, __bfloat162float(partial[offset * 256 + d]), val);
  }
  const int groups = rows / seq, query_head = h * groups + row / seq, token = row % seq;
  out[((static_cast<int64_t>(b) * seq + token) * heads * groups + query_head) * 256 + d] =
      __float2bfloat16_rn(sum > 0 ? val / sum : 0.0f);
  if (d == 0)
    out_lse[(b * heads * groups + query_head) * seq + token] =
        sum > 0 ? (m + log2f(sum)) * 0.6931471805599453f : -INFINITY;
}
}

extern "C" void launch_int8_decode(
    int8_t* q, int8_t* k, int8_t* v, float* qs, float* ks, float* vs, const int64_t* lengths,
    void* partial, float* lse, void* out, float* out_lse,
    int batch, int heads, int rows, int pages, int page_size, int seq, cudaStream_t stream) {
  constexpr int smem = (64 + 64 + 64) * 256;
  int8_decode_pages<<<dim3(pages, heads, batch), dim3(32, 4), smem, stream>>>(
      q, k, v, qs, ks, vs, lengths, static_cast<nv_bfloat16*>(partial), lse, heads, rows, pages, page_size);
  int8_decode_combine<<<batch * heads * rows, 256, 0, stream>>>(
      static_cast<nv_bfloat16*>(partial), lse, lengths, static_cast<nv_bfloat16*>(out), out_lse,
      heads, rows, pages, page_size, seq);
  const auto error = cudaGetLastError();
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

namespace {
template <MaskMode Mask, bool FuseProbabilities = true>
__global__ void int8_prefill(
    int8_t* q, int8_t* k, int8_t* v, float* qs, float* ks, float* vs, nv_bfloat16* out,
    const void* mask, int mask_dtype, int64_t mask_b, int64_t mask_h, int64_t mask_q, int64_t mask_k,
    int heads, int kv_heads, int rows, int length, int capacity, int chunk_size) {
  qk_int_sv_i8_attn_body<false, 128, 64, 16, 64, 256, DataType::kInt8,
      QuantGranularity::kPerThread, QuantGranularity::kPerThread, float, false,
      nv_bfloat16, ComputeUnit::kCudaCore, Mask, false, true, false,
      false, FuseProbabilities, false, uint32_t, false, true>(
      q, k, v, out, nullptr, qs, ks, vs, nullptr, mask, mask_b, mask_h, mask_q, mask_k,
      mask_dtype, rows, length, heads / kv_heads,
      heads * rows * 256, 256, rows * 256, kv_heads * capacity * 256, 256, capacity * 256,
      kv_heads * 256 * capacity, 256 * capacity, capacity,
      rows * heads * 256, heads * 256, 256, 0.0625f, nullptr, capacity / chunk_size, chunk_size);
}
}

extern "C" void launch_int8_prefill(
    int8_t* q, int8_t* k, int8_t* v, float* qs, float* ks, float* vs, void* out,
    const void* mask, int mask_dtype, int64_t mask_b, int64_t mask_h, int64_t mask_q, int64_t mask_k,
    int batch, int heads, int kv_heads, int rows, int length, int capacity, int chunk_size, cudaStream_t stream) {
  constexpr int smem = (128 + 64 + 64) * 256;
  const dim3 grid((rows + 127) / 128, heads, batch), block(32, 8);
  if (mask) {
    cudaFuncSetAttribute(int8_prefill<MaskMode::kCustom>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    int8_prefill<MaskMode::kCustom><<<grid, block, smem, stream>>>(
        q, k, v, qs, ks, vs, static_cast<nv_bfloat16*>(out), mask, mask_dtype, mask_b, mask_h, mask_q, mask_k,
        heads, kv_heads, rows, length, capacity, chunk_size);
  } else {
    auto kernel = length <= 512 ? int8_prefill<MaskMode::kNone, false> : int8_prefill<MaskMode::kNone>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    kernel<<<grid, block, smem, stream>>>(
        q, k, v, qs, ks, vs, static_cast<nv_bfloat16*>(out), nullptr, 0, 0, 0, 0, 0,
        heads, kv_heads, rows, length, capacity, chunk_size);
  }
  const auto error = cudaGetLastError();
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
