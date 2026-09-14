// SPDX-License-Identifier: Apache-2.0
// FP32 DraftMap math for the public mean/Jensen policy. No framework ABI.
#include "anemoi_dispatch.h"
#include "cublas_runtime.h"
#include <math_constants.h>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace comfy::anemoi {
namespace {
using anemoi_native::check;
using anemoi_native::cuda_check;

// Operands are reused for the three Jensen GEMMs. Their consumers and rewrites
// are ordered on the explicit stream; no tensor allocation or copy is needed.
struct DraftScratch {
  size_t bytes = 0, q, k, scores, variance = 0, squared = 0;
  int pools, maps, heads, rows, dim;
  DraftScratch(int batch, int h, int r, int d, bool jensen)
      : rows(r), dim(d) {
    check(batch > 0 && h > 0 && r > 0 && d > 0,
          "DraftMap dimensions must be positive");
    auto product = [](int64_t a, int64_t b) {
      check(a <= INT32_MAX / b, "DraftMap exceeds signed indexing range");
      return int(a * b);
    };
    heads = product(batch,h);
    pools = product(product(heads,r),d);
    maps = product(product(heads,r),r);
    auto reserve = [&](int elements) {
      check(bytes <= SIZE_MAX - 255, "DraftMap scratch alignment overflow");
      const size_t offset = (bytes + 255) & ~size_t(255);
      check(size_t(elements) <= (SIZE_MAX - offset) / sizeof(float),
            "DraftMap scratch size overflow");
      bytes = offset + size_t(elements) * sizeof(float);
      return offset;
    };
    q = reserve(pools); k = reserve(pools); scores = reserve(maps);
    if (jensen) { variance = reserve(maps); squared = reserve(maps); }
  }
};

void blas_check(cublasStatus_t status, const char *operation) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string(operation) + " failed, cuBLAS status " +
                             std::to_string(int(status)));
}

cublasHandle_t draft_handle(int device, cudaStream_t stream) {
  struct HandleCache {
    const AnemoiCublasFunctions *api = nullptr;
    std::vector<std::pair<int,cublasHandle_t>> handles;
    ~HandleCache() {
      if (!api || handles.empty()) return;
      int previous;
      if (cudaGetDevice(&previous) != cudaSuccess) return;
      for (const auto &entry : handles)
        if (cudaSetDevice(entry.first) == cudaSuccess) api->destroy(entry.second);
      cudaSetDevice(previous);
    }
  };
  thread_local HandleCache cache;
  const auto &api = comfy::anemoi_cublas();
  cache.api = &api; // The lazy runtime intentionally outlives thread-local state.
  cublasHandle_t handle = nullptr;
  for (const auto &entry : cache.handles)
    if (entry.first == device) handle = entry.second;
  if (!handle) {
    blas_check(api.create(&handle), "DraftMap cublasCreate");
    try { cache.handles.emplace_back(device,handle); }
    catch (...) { api.destroy(handle); throw; }
  }
  blas_check(api.set_stream(handle,stream), "DraftMap cublasSetStream");
  blas_check(api.set_pointer_mode(handle,CUBLAS_POINTER_MODE_HOST),
             "DraftMap cublasSetPointerMode");
  // Match the original FP32 torch.matmul contract, not TF32/FP16 tensor math.
  blas_check(api.set_math_mode(handle,CUBLAS_DEFAULT_MATH),
             "DraftMap cublasSetMathMode");
  return handle;
}

void product(cublasHandle_t handle, const DraftScratch &s,
             const float *q, const float *k, float *out) {
  const float alpha = 1.0f, beta = 0.0f;
  const long long operand_stride = static_cast<long long>(s.rows) * s.dim;
  const long long output_stride = static_cast<long long>(s.rows) * s.rows;
  // Row-major Q K^T is column-major K Q^T. Scaling must NOT be GEMM alpha:
  // torch rounds the FP32 product before its distinct pointwise multiplication.
  blas_check(comfy::anemoi_cublas().gemm_strided_batched_ex(
      handle,CUBLAS_OP_T,CUBLAS_OP_N,s.rows,s.rows,s.dim,
      &alpha,k,CUDA_R_32F,s.dim,operand_stride,
      q,CUDA_R_32F,s.dim,operand_stride,&beta,
      out,CUDA_R_32F,s.rows,output_stride,s.heads,
      CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT), "DraftMap FP32 GEMM");
}

template<bool Square>
__global__ void convert_pools(const half *q, const half *k,
                              float *qf, float *kf, int count) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  float a = __half2float(q[i]), b = __half2float(k[i]);
  if constexpr (Square) { a = __fmul_rn(a,a); b = __fmul_rn(b,b); }
  qf[i] = a; kf[i] = b;
}

__global__ void scale_scores(float *scores, int count, float scale) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) scores[i] = __fmul_rn(scores[i],scale);
}

__global__ void add_jensen(float *scores, const float *variance,
                           const float *squared, int count, float inv_dim) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  // Separate RN operations reproduce sub_ -> div_ -> clamp_min_ -> add_(alpha).
  // Division by a host scalar in Torch uses multiplication by its reciprocal.
  float correction = __fmul_rn(__fsub_rn(variance[i],squared[i]),inv_dim);
  // Unlike fmaxf, the comparison preserves a NaN as torch.clamp_min does.
  correction = correction < 0.0f ? 0.0f : correction;
  scores[i] = __fmaf_rn(0.5f,correction,scores[i]);
}

// Persistent row reduction: sequential lane accumulation followed by descending
// XOR butterfly, as in ATen's persistent softmax. Keep ordinary expf (compile
// this source without --use_fast_math) and FP32 division before the final cast.
// Ordering reference: PyTorch v2.8.0 aten/src/ATen/native/cuda/PersistentSoftmax.cuh.
template<int Log2>
__global__ void softmax_rows(const float *scores, half *out, int rows, int width) {
  constexpr int extent = 1 << Log2;
  constexpr int lanes = extent < 32 ? extent : 32;
  constexpr int iterations = extent / lanes;
  constexpr int batches = extent <= 128 ? 2 : 1;
  const int64_t first = (int64_t(blockIdx.x) * blockDim.y + threadIdx.y) * batches;
  float values[batches][iterations], maximum[batches], sums[batches];
#pragma unroll
  for (int b = 0; b < batches; ++b) {
#pragma unroll
    for (int j = 0; j < iterations; ++j) {
      const int column = threadIdx.x + j * lanes;
      values[b][j] = first+b < rows && column < width
          ? scores[int64_t(first+b)*width+column] : -CUDART_INF_F;
    }
    maximum[b] = values[b][0];
#pragma unroll
    for (int j = 0; j < iterations; ++j)
      maximum[b] = maximum[b] > values[b][j] ? maximum[b] : values[b][j];
  }
#pragma unroll
  for (int offset = lanes/2; offset > 0; offset /= 2) {
#pragma unroll
    for (int b = 0; b < batches; ++b) {
      const float other = __shfl_xor_sync(0xffffffff,maximum[b],offset,lanes);
      maximum[b] = maximum[b] < other ? other : maximum[b];
    }
  }
#pragma unroll
  for (int b = 0; b < batches; ++b) {
    sums[b] = 0.0f;
#pragma unroll
    for (int j = 0; j < iterations; ++j) {
      values[b][j] = expf(__fsub_rn(values[b][j],maximum[b]));
      sums[b] = __fadd_rn(sums[b],values[b][j]);
    }
  }
#pragma unroll
  for (int offset = lanes/2; offset > 0; offset /= 2) {
#pragma unroll
    for (int b = 0; b < batches; ++b)
      sums[b] = __fadd_rn(sums[b],__shfl_xor_sync(0xffffffff,sums[b],offset,lanes));
  }
#pragma unroll
  for (int b = 0; b < batches; ++b) {
#pragma unroll
    for (int j = 0; j < iterations; ++j) {
      const int column = threadIdx.x + j * lanes;
      if (first+b < rows && column < width)
        out[int64_t(first+b)*width+column] = __float2half_rn(
            sums[b] == 0.0f ? CUDART_NAN_F : __fdiv_rn(values[b][j],sums[b]));
    }
  }
}

// Wide-row fallback remains stable FP32 throughout and never needs another
// workspace. Unlike the small-row kernel its tree need not match ATen bitwise.
__global__ void softmax_wide(const float *scores, half *out, int width) {
  __shared__ float reduction[256];
  const int lane = threadIdx.x;
  const int64_t start = int64_t(blockIdx.x)*width;
  float maximum = -CUDART_INF_F;
  for (int column = lane; column < width; column += 256) {
    const float x = scores[start+column];
    maximum = maximum > x ? maximum : x;
  }
  reduction[lane] = maximum;
  __syncthreads();
  for (int offset = 128; offset; offset /= 2) {
    if (lane < offset) {
      const float other = reduction[lane+offset];
      reduction[lane] = reduction[lane] < other ? other : reduction[lane];
    }
    __syncthreads();
  }
  maximum = reduction[0];
  __syncthreads();
  float sum = 0.0f;
  for (int column = lane; column < width; column += 256)
    sum = __fadd_rn(sum,expf(__fsub_rn(scores[start+column],maximum)));
  reduction[lane] = sum;
  __syncthreads();
  for (int offset = 128; offset; offset /= 2) {
    if (lane < offset) reduction[lane] = __fadd_rn(reduction[lane],reduction[lane+offset]);
    __syncthreads();
  }
  sum = reduction[0];
  for (int column = lane; column < width; column += 256)
    out[start+column] = __float2half_rn(__fdiv_rn(
        expf(__fsub_rn(scores[start+column],maximum)),sum));
}

void softmax(const DraftScratch &s, const float *scores, half *out, cudaStream_t stream) {
  const int rows = s.heads*s.rows;
  int log2 = 0;
  while ((1 << log2) < s.rows && log2 < 11) ++log2;
  if (log2 > 10) { softmax_wide<<<rows,256,0,stream>>>(scores,out,s.rows); }
  else {
    const int lanes = std::min(1 << log2,32);
    const int batches = (1 << log2) <= 128 ? 2 : 1;
    const int rows_per_block = (128/lanes)*batches;
    const int blocks = (rows-1)/rows_per_block+1;
    const dim3 threads(lanes,128/lanes);
#define DRAFT_SOFTMAX_CASE(N) case N: softmax_rows<N><<<blocks,threads,0,stream>>>(scores,out,rows,s.rows); break
    switch (log2) {
      DRAFT_SOFTMAX_CASE(0); DRAFT_SOFTMAX_CASE(1); DRAFT_SOFTMAX_CASE(2);
      DRAFT_SOFTMAX_CASE(3); DRAFT_SOFTMAX_CASE(4); DRAFT_SOFTMAX_CASE(5);
      DRAFT_SOFTMAX_CASE(6); DRAFT_SOFTMAX_CASE(7); DRAFT_SOFTMAX_CASE(8);
      DRAFT_SOFTMAX_CASE(9); DRAFT_SOFTMAX_CASE(10);
    }
#undef DRAFT_SOFTMAX_CASE
  }
  cuda_check(cudaGetLastError());
}

void check_pool(const Tensor &t, const Tensor &q, const char *name) {
  check(t.defined() && t.pointer && t.sizes() == q.sizes() &&
        t.scalar_type() == DType::Half && t.is_contiguous() && t.device() == q.device(),
        name," must be contiguous FP16 matching Q pool on its device");
}
} // namespace

size_t draft_workspace_bytes(int batch, int heads, int blocks, int dim, bool jensen) {
  return DraftScratch(batch,heads,blocks,dim,jensen).bytes;
}

void draft_probability(const Tensor &q_pool, const Tensor &k_pool,
                       const Tensor &q_second, const Tensor &k_second,
                       const Tensor &probability, void *scratch,
                       size_t scratch_bytes, cudaStream_t stream) {
  check(q_pool.dim() == 4, "DraftMap pools must be BHSD tensors");
  check_pool(q_pool,q_pool,"Q pool"); check_pool(k_pool,q_pool,"K pool");
  const bool jensen = q_second.numel() != 0;
  check(jensen == (k_second.numel() != 0), "Jensen needs both second moments");
  if (jensen) {
    check_pool(q_second,q_pool,"Q second moment");
    check_pool(k_second,q_pool,"K second moment");
  }
  for (auto dim : q_pool.sizes()) check(dim > 0 && dim <= INT32_MAX,"invalid DraftMap extent");
  const DraftScratch s(int(q_pool.size(0)),int(q_pool.size(1)),int(q_pool.size(2)),
                       int(q_pool.size(3)),jensen);
  check(probability.defined() && probability.pointer && probability.is_contiguous() &&
        probability.scalar_type() == DType::Half && probability.device() == q_pool.device() &&
        probability.sizes() == anemoi_native::IntArrayRef({q_pool.size(0),q_pool.size(1),s.rows,s.rows}),
        "DraftMap probability must be contiguous FP16 [B,H,R,R]");
  check(scratch && scratch_bytes >= s.bytes && uintptr_t(scratch)%alignof(float) == 0,
        "insufficient or misaligned DraftMap scratch");
  anemoi_native::cuda::CUDAGuard guard(q_pool.device());
  auto f = [&](size_t offset) { return reinterpret_cast<float *>(static_cast<char *>(scratch)+offset); };
  const auto handle = draft_handle(q_pool.device(),stream);
  const int pool_blocks = (s.pools-1)/256+1, map_blocks = (s.maps-1)/256+1;
  convert_pools<false><<<pool_blocks,256,0,stream>>>(
      q_pool.data_ptr<half>(),k_pool.data_ptr<half>(),f(s.q),f(s.k),s.pools);
  cuda_check(cudaGetLastError());
  product(handle,s,f(s.q),f(s.k),f(s.scores));
  scale_scores<<<map_blocks,256,0,stream>>>(f(s.scores),s.maps,
                                          float(std::pow(double(s.dim),-0.5)));
  cuda_check(cudaGetLastError());
  if (jensen) {
    convert_pools<false><<<pool_blocks,256,0,stream>>>(
        q_second.data_ptr<half>(),k_second.data_ptr<half>(),f(s.q),f(s.k),s.pools);
    cuda_check(cudaGetLastError());
    product(handle,s,f(s.q),f(s.k),f(s.variance));
    convert_pools<true><<<pool_blocks,256,0,stream>>>(
        q_pool.data_ptr<half>(),k_pool.data_ptr<half>(),f(s.q),f(s.k),s.pools);
    cuda_check(cudaGetLastError());
    product(handle,s,f(s.q),f(s.k),f(s.squared));
    add_jensen<<<map_blocks,256,0,stream>>>(f(s.scores),f(s.variance),f(s.squared),
                                          s.maps,float(1.0/double(s.dim)));
    cuda_check(cudaGetLastError());
  }
  softmax(s,f(s.scores),probability.data_ptr<half>(),stream);
}
} // namespace comfy::anemoi
