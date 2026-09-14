/*
 * Copyright 2026 Mixed Attention Project Contributors
 * SPDX-License-Identifier: Apache-2.0
 * Ported from mixed_precision_attention e318368c15a2962885df2117c35710e86837d5d1.
 * FP16 pooled operands feed a cuBLAS GEMM with FP32 accumulation. Shared
 * normalization/GEMM math lives in common.cuh; K-tail is SM120-only.
 */
#include "common.cuh"
#include <limits>
#include <utility>
#include "assembly_route_draft_native.cuh"

namespace {
inline int64_t checked_positive_product(
    int64_t lhs, int64_t rhs, const char* description) {
  ANEMOI_CHECK(lhs > 0 && rhs > 0, description, " factors must be positive");
  ANEMOI_CHECK(
      lhs <= std::numeric_limits<int64_t>::max() / rhs,
      description,
      " exceeds int64 range");
  return lhs * rhs;
}
inline void check_cublas(cublasStatus_t status, const char* operation) {
  ANEMOI_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      operation,
      " failed with cuBLAS status ",
      static_cast<int>(status));
}
inline void check_pool_tensor(
    const anemoi_native::Tensor& tensor, const char* name) {
  ANEMOI_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  ANEMOI_CHECK(tensor.is_contiguous(), name, " must be contiguous BHSD");
  ANEMOI_CHECK(
      tensor.scalar_type() == anemoi_native::ScalarType::Half,
      name,
      " must have dtype torch.float16");
  ANEMOI_CHECK(tensor.dim() == 4, name, " must have shape [B,H,R,D]");
  ANEMOI_CHECK(
      tensor.size(0) > 0 && tensor.size(1) > 0 && tensor.size(2) > 0,
      name,
      " batch, head, and row dimensions must be positive");
  ANEMOI_CHECK(
      tensor.size(3) == 64 || tensor.size(3) == 128,
      name,
      " head dimension must be 64 or 128");
}

void launch_draft_gemm(
    const anemoi_native::Tensor& q_pool,
    const anemoi_native::Tensor& k_pool,
    anemoi_native::Tensor& logits,
    cublasHandle_t handle) {
  const int64_t batch_size = q_pool.size(0);
  const int64_t q_heads = q_pool.size(1);
  const int64_t kv_heads = k_pool.size(1);
  const int64_t query_rows = q_pool.size(2);
  const int64_t key_rows = k_pool.size(2);
  const int64_t head_dim = q_pool.size(3);
  const int64_t queries_per_kv = q_heads / kv_heads;
  const int64_t batch_kv_heads = checked_positive_product(
      batch_size, kv_heads, "Draft B*Hkv");
  const int64_t q_operand_stride = checked_positive_product(
      query_rows, head_dim, "Draft Q operand head stride");
  checked_positive_product(key_rows, head_dim, "Draft K operand head stride");
  const int64_t output_stride = checked_positive_product(
      query_rows, key_rows, "Draft logits head stride");
  checked_positive_product(queries_per_kv, q_operand_stride, "Draft grouped Q head stride");
  checked_positive_product(queries_per_kv, output_stride, "Draft grouped logits head stride");
  ANEMOI_CHECK(
      query_rows <= std::numeric_limits<int>::max() &&
          key_rows <= std::numeric_limits<int>::max(),
      "Draft row/column count exceeds cuBLAS int range");
  ANEMOI_CHECK(head_dim <= std::numeric_limits<int>::max(),
               "Draft head dimension exceeds cuBLAS int range");
  ANEMOI_CHECK(queries_per_kv <= std::numeric_limits<int>::max(),
               "Draft GQA group exceeds cuBLAS batch range");
  ANEMOI_CHECK(batch_kv_heads <= std::numeric_limits<int>::max(),
               "Draft B*Hkv exceeds cuBLAS batch range");
  check_cublas(draft_gemm(q_pool.data_ptr(), k_pool.data_ptr(), logits.data_ptr(),
      batch_size, q_heads, kv_heads, query_rows, key_rows, head_dim, handle),
      "Draft cublasGemmStridedBatchedEx");
}

cublasHandle_t checked_current_draft_handle(cudaStream_t stream) {
  // cuBLAS library state is cached per host thread and device. Tensor/output
  // storage remains entirely caller-owned; no framework handle/ABI is used.
  struct HandleCache {
    std::vector<std::pair<int, cublasHandle_t>> handles;
    ~HandleCache() {
      int previous = 0;
      if (cudaGetDevice(&previous) != cudaSuccess) return;
      for (const auto& entry : handles) {
        if (cudaSetDevice(entry.first) == cudaSuccess) comfy::anemoi_cublas().destroy(entry.second);
      }
      cudaSetDevice(previous);
    }
  };
  thread_local HandleCache cache;
  int device = 0;
  ANEMOI_CUDA_CHECK(cudaGetDevice(&device));
  cublasHandle_t handle = nullptr;
  for (const auto& entry : cache.handles) if (entry.first == device) handle = entry.second;
  if (!handle) {
    check_cublas(comfy::anemoi_cublas().create(&handle), "Draft cublasCreate");
    cache.handles.emplace_back(device, handle);
  }
  check_cublas(comfy::anemoi_cublas().set_stream(handle, stream), "Draft cublasSetStream");
  check_cublas(comfy::anemoi_cublas().set_pointer_mode(handle, CUBLAS_POINTER_MODE_HOST), "Draft cublasSetPointerMode");
  return handle;
}

anemoi_native::Tensor h3_draft_probability_impl(
    anemoi_native::Tensor q_pool,
    anemoi_native::Tensor k_pool,
    std::optional<anemoi_native::Tensor> q_max_pool_optional,
    std::optional<anemoi_native::Tensor> k_max_pool_optional,
    double maxpool_weight,
    const char* operation_name) {
  ANEMOI_CHECK(
      std::isfinite(maxpool_weight) && maxpool_weight >= 0.0 &&
          maxpool_weight <= 1.0,
      "maxpool_weight must be finite and in [0, 1]");
  check_pool_tensor(q_pool, "q_pool");
  check_pool_tensor(k_pool, "k_pool");
  ANEMOI_CHECK(
      q_pool.device() == k_pool.device(),
      "q_pool and k_pool must share one CUDA device");
  ANEMOI_CHECK(
      q_pool.size(0) == k_pool.size(0),
      "q_pool and k_pool batch dimensions must match");
  ANEMOI_CHECK(
      q_pool.size(2) == k_pool.size(2),
      "q_pool and k_pool row dimensions must match");
  ANEMOI_CHECK(
      q_pool.size(3) == k_pool.size(3),
      "q_pool and k_pool head dimensions must match");
  ANEMOI_CHECK(
      q_pool.size(1) % k_pool.size(1) == 0,
      "Q pool heads must be divisible by KV pool heads");
  if (maxpool_weight != 0.0) {
    ANEMOI_CHECK(
        q_max_pool_optional.has_value() && k_max_pool_optional.has_value(),
        "nonzero maxpool_weight requires q_max_pool and k_max_pool");
    check_pool_tensor(*q_max_pool_optional, "q_max_pool");
    check_pool_tensor(*k_max_pool_optional, "k_max_pool");
    ANEMOI_CHECK(
        q_max_pool_optional->device() == q_pool.device() &&
            k_max_pool_optional->device() == k_pool.device() &&
            assembly_route_draft_native::shape(*q_max_pool_optional) == assembly_route_draft_native::shape(q_pool) &&
            assembly_route_draft_native::shape(*k_max_pool_optional) == assembly_route_draft_native::shape(k_pool),
        "max pools must match their mean-pool shapes and devices");
  }

  assembly_route_draft_native::DeviceGuard device_guard(q_pool.device());
  const auto device_properties = assembly_route_draft_native::device_properties(q_pool.device());
  const cudaDeviceProp* properties = &device_properties;
  ANEMOI_CHECK(
      anemoi_native::ada_serves_device(properties),
      operation_name,
      " serves SM89 to SM119 (SM120+ uses the Blackwell kernels)");

  const int64_t rows = q_pool.size(2);
  const int64_t output_head_stride = checked_positive_product(
      rows, rows, "Draft probability head stride");
  const int64_t output_heads = checked_positive_product(
      q_pool.size(0), q_pool.size(1), "Draft B*Hq");
  checked_positive_product(
      output_heads, output_head_stride, "Draft probability elements");

  auto logits = anemoi_native::empty(
      {q_pool.size(0), q_pool.size(1), rows, rows}, q_pool.options());
  const cudaStream_t stream =
      anemoi_native::cuda::getCurrentCUDAStream();
  cublasHandle_t handle = checked_current_draft_handle(stream);

  const int64_t softmax_rows = checked_positive_product(
      output_heads, rows, "Draft softmax row count");
  ANEMOI_CHECK(
      softmax_rows <= kMaxGridX, "Draft softmax grid.x exceeds CUDA limit");
  if (maxpool_weight == 0.0) {
    launch_draft_gemm(q_pool, k_pool, logits, handle);
    row_softmax_fp16_kernel<<<
        static_cast<unsigned int>(softmax_rows),
        kSoftmaxThreads,
        0,
        stream>>>(
        reinterpret_cast<half*>(logits.data_ptr<half>()),
        softmax_rows,
        rows);
    ANEMOI_CUDA_KERNEL_LAUNCH_CHECK();
    return logits;
  }

  const anemoi_native::Tensor& q_max_pool = *q_max_pool_optional;
  const anemoi_native::Tensor& k_max_pool = *k_max_pool_optional;
  if (maxpool_weight == 1.0) {
    launch_draft_gemm(q_max_pool, k_max_pool, logits, handle);
    row_softmax_fp16_kernel<<<
        static_cast<unsigned int>(softmax_rows),
        kSoftmaxThreads,
        0,
        stream>>>(
        reinterpret_cast<half*>(logits.data_ptr<half>()),
        softmax_rows,
        rows);
    ANEMOI_CUDA_KERNEL_LAUNCH_CHECK();
    return logits;
  }

  launch_draft_gemm(q_pool, k_pool, logits, handle);
  auto max_logits = anemoi_native::empty_like(logits);
  launch_draft_gemm(q_max_pool, k_max_pool, max_logits, handle);
  row_softmax_fusion_fp16_kernel<<<
      static_cast<unsigned int>(softmax_rows),
      kSoftmaxThreads,
      0,
      stream>>>(
      reinterpret_cast<half*>(logits.data_ptr<half>()),
      reinterpret_cast<const half*>(max_logits.data_ptr<half>()),
      softmax_rows,
      rows,
      static_cast<float>(maxpool_weight));
  ANEMOI_CUDA_KERNEL_LAUNCH_CHECK();
  return logits;
}
}  // namespace

anemoi_native::Tensor sm89_h3_draft_probability(
    anemoi_native::Tensor q_pool,
    anemoi_native::Tensor k_pool,
    std::optional<anemoi_native::Tensor> q_max_pool,
    std::optional<anemoi_native::Tensor> k_max_pool,
    double maxpool_weight) {
  return h3_draft_probability_impl(
      std::move(q_pool), std::move(k_pool), std::move(q_max_pool), std::move(k_max_pool),
      maxpool_weight, "sm89_h3_draft_probability");
}
