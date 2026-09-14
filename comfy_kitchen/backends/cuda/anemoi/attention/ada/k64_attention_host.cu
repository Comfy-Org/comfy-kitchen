// Adapted from Anemoi 4e85afba741bdeaf2d9486cab19cb76d3e7985a4; see LICENSE.
#include "native_tensor.h"
/*
 * Native Q64 x K64 FP16 attention host dispatch for controlled Sol-H3
 * alignment experiments on SM89.
 */


#include <cmath>
#include <cstdint>
#include <tuple>
#include <type_traits>

#include "api.h"
#include "k64_attention_decl.cuh"

namespace {

void check_cuda_contiguous(const anemoi_native::Tensor& tensor, const char* name) {
  ANEMOI_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  ANEMOI_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_same_device(
    const anemoi_native::Tensor& tensor,
    const anemoi_native::Tensor& reference,
    const char* name) {
  ANEMOI_CHECK(
      tensor.device() == reference.device(), name,
      " must be on the same CUDA device as query");
}

}  // namespace

template <uint32_t QueryBlock>
std::tuple<anemoi_native::Tensor, anemoi_native::Tensor> fp16_attention_forward(
    anemoi_native::Tensor query,
    anemoi_native::Tensor key,
    anemoi_native::Tensor value,
    anemoi_native::Tensor block_ids,
    anemoi_native::Tensor block_counts,
    anemoi_native::Tensor valid_k_counts,
    double softmax_scale) {
  static_assert(QueryBlock == 64 || QueryBlock == 128);
  for (const auto& item : {
           std::pair<const anemoi_native::Tensor*, const char*>(&query, "query"),
           std::pair<const anemoi_native::Tensor*, const char*>(&key, "key"),
           std::pair<const anemoi_native::Tensor*, const char*>(&value, "value"),
           std::pair<const anemoi_native::Tensor*, const char*>(&block_ids, "block_ids"),
           std::pair<const anemoi_native::Tensor*, const char*>(&block_counts, "block_counts"),
           std::pair<const anemoi_native::Tensor*, const char*>(
               &valid_k_counts, "valid_k_counts")}) {
    check_cuda_contiguous(*item.first, item.second);
    check_same_device(*item.first, query, item.second);
  }
  ANEMOI_CHECK(query.dim() == 4, "query must have shape [B,Hq,Q,D]");
  ANEMOI_CHECK(key.dim() == 4 && value.dim() == 4,
              "key/value must have shape [B,Hkv,K,D]");
  ANEMOI_CHECK(query.scalar_type() == anemoi_native::ScalarType::Half,
              "query must be FP16");
  ANEMOI_CHECK(key.scalar_type() == anemoi_native::ScalarType::Half,
              "key must be FP16");
  ANEMOI_CHECK(value.scalar_type() == anemoi_native::ScalarType::Half,
              "value must be FP16");
  ANEMOI_CHECK(key.sizes() == value.sizes(), "key/value shapes must match");
  ANEMOI_CHECK(query.size(0) > 0 && query.size(1) > 0 && query.size(2) > 0,
              "query dimensions must be positive");
  ANEMOI_CHECK(key.size(0) == query.size(0), "query/key batch mismatch");
  ANEMOI_CHECK(key.size(2) > 0 && key.size(2) % 64 == 0,
              "K must be a positive multiple of 64 physical slots");
  const bool supports_head_dim =
      query.size(3) == 64 || query.size(3) == 128;
  ANEMOI_CHECK(
      supports_head_dim && key.size(3) == query.size(3),
      "native K64 FP16 requires head_dim=64 or 128");
  ANEMOI_CHECK(key.size(1) > 0 && query.size(1) % key.size(1) == 0,
              "query heads must be divisible by KV heads");
  ANEMOI_CHECK(
      std::isfinite(softmax_scale) && softmax_scale > 0.0,
      "softmax_scale must be finite and positive");

  ANEMOI_CHECK(block_ids.scalar_type() == anemoi_native::ScalarType::Int,
              "block_ids must be int32");
  ANEMOI_CHECK(block_counts.scalar_type() == anemoi_native::ScalarType::Int,
              "block_counts must be int32");
  ANEMOI_CHECK(valid_k_counts.scalar_type() == anemoi_native::ScalarType::Int,
              "valid_k_counts must be int32");
  const int64_t query_blocks =
      (query.size(2) + QueryBlock - 1) / QueryBlock;
  const int64_t key_blocks = key.size(2) / 64;
  ANEMOI_CHECK(
      block_ids.sizes() == anemoi_native::IntArrayRef(
          {query.size(0), query.size(1), query_blocks, key_blocks}),
      "block_ids must have shape [B,Hq,ceil(Q/query_block),K/64]");
  ANEMOI_CHECK(
      block_counts.sizes() ==
          anemoi_native::IntArrayRef({query.size(0), query.size(1), query_blocks}),
      "block_counts must have shape [B,Hq,ceil(Q/query_block)]");
  ANEMOI_CHECK(
      valid_k_counts.sizes() ==
          anemoi_native::IntArrayRef({query.size(0), key_blocks}),
      "valid_k_counts must have shape [B,K/64]");

  auto output = anemoi_native::empty_like(query);
  auto lse = anemoi_native::empty(
      {0}, query.options().dtype(anemoi_native::ScalarType::Float));

  anemoi_native::cuda::CUDAGuard device_guard(query.device());
  cudaStream_t stream = anemoi_native::cuda::getCurrentCUDAStream(query.get_device());
  if constexpr (QueryBlock == 64) {
    if (query.size(3) == 64) {
      launch_mixed_attention_sm89_k64<64, false, true, false>(
          nullptr, nullptr, nullptr,
          reinterpret_cast<half*>(query.data_ptr<anemoi_native::Half>()),
          reinterpret_cast<half*>(key.data_ptr<anemoi_native::Half>()),
          reinterpret_cast<half*>(value.data_ptr<anemoi_native::Half>()), nullptr,
          reinterpret_cast<half*>(output.data_ptr<anemoi_native::Half>()),
          nullptr, nullptr, block_ids.data_ptr<int32_t>(),
          block_counts.data_ptr<int32_t>(), nullptr, nullptr, nullptr,
          valid_k_counts.data_ptr<int32_t>(), nullptr, 0,
          static_cast<uint32_t>(query.size(0)),
          static_cast<uint32_t>(query.size(2)),
          static_cast<uint32_t>(key.size(2)), 0,
          static_cast<uint32_t>(query.size(1)),
          static_cast<uint32_t>(key.size(1)),
          static_cast<float>(softmax_scale), stream);
    } else {
      launch_mixed_attention_sm89_k64<128, false, true, false>(
          nullptr, nullptr, nullptr,
          reinterpret_cast<half*>(query.data_ptr<anemoi_native::Half>()),
          reinterpret_cast<half*>(key.data_ptr<anemoi_native::Half>()),
          reinterpret_cast<half*>(value.data_ptr<anemoi_native::Half>()), nullptr,
          reinterpret_cast<half*>(output.data_ptr<anemoi_native::Half>()),
          nullptr, nullptr, block_ids.data_ptr<int32_t>(),
          block_counts.data_ptr<int32_t>(), nullptr, nullptr, nullptr,
          valid_k_counts.data_ptr<int32_t>(), nullptr, 0,
          static_cast<uint32_t>(query.size(0)),
          static_cast<uint32_t>(query.size(2)),
          static_cast<uint32_t>(key.size(2)), 0,
          static_cast<uint32_t>(query.size(1)),
          static_cast<uint32_t>(key.size(1)),
          static_cast<float>(softmax_scale), stream);
    }
  } else if (query.size(3) == 64) {
    launch_mixed_attention_sm89_q128_k64<64, false, true, false>(
        nullptr, nullptr, nullptr,
        reinterpret_cast<half*>(query.data_ptr<anemoi_native::Half>()),
        reinterpret_cast<half*>(key.data_ptr<anemoi_native::Half>()),
        reinterpret_cast<half*>(value.data_ptr<anemoi_native::Half>()), nullptr,
        reinterpret_cast<half*>(output.data_ptr<anemoi_native::Half>()),
        nullptr, nullptr, block_ids.data_ptr<int32_t>(),
        block_counts.data_ptr<int32_t>(), nullptr, nullptr, nullptr,
        valid_k_counts.data_ptr<int32_t>(), nullptr, 0,
        static_cast<uint32_t>(query.size(0)),
        static_cast<uint32_t>(query.size(2)),
        static_cast<uint32_t>(key.size(2)), 0,
        static_cast<uint32_t>(query.size(1)),
        static_cast<uint32_t>(key.size(1)),
        static_cast<float>(softmax_scale), stream);
  } else {
    launch_mixed_attention_sm89_q128_k64<128, false, true, false>(
        nullptr, nullptr, nullptr,
        reinterpret_cast<half*>(query.data_ptr<anemoi_native::Half>()),
        reinterpret_cast<half*>(key.data_ptr<anemoi_native::Half>()),
        reinterpret_cast<half*>(value.data_ptr<anemoi_native::Half>()), nullptr,
        reinterpret_cast<half*>(output.data_ptr<anemoi_native::Half>()),
        nullptr, nullptr, block_ids.data_ptr<int32_t>(),
        block_counts.data_ptr<int32_t>(), nullptr, nullptr, nullptr,
        valid_k_counts.data_ptr<int32_t>(), nullptr, 0,
        static_cast<uint32_t>(query.size(0)),
        static_cast<uint32_t>(query.size(2)),
        static_cast<uint32_t>(key.size(2)), 0,
        static_cast<uint32_t>(query.size(1)),
        static_cast<uint32_t>(key.size(1)),
        static_cast<float>(softmax_scale), stream);
  }
  ANEMOI_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, lse};
}

std::tuple<anemoi_native::Tensor, anemoi_native::Tensor> k64_fp16_attention_forward(
    anemoi_native::Tensor query, anemoi_native::Tensor key, anemoi_native::Tensor value,
    anemoi_native::Tensor block_ids, anemoi_native::Tensor block_counts,
    anemoi_native::Tensor valid_k_counts, double softmax_scale) {
  return fp16_attention_forward<64>(
      query, key, value, block_ids, block_counts, valid_k_counts,
      softmax_scale);
}

std::tuple<anemoi_native::Tensor, anemoi_native::Tensor> q128_k64_fp16_attention_forward(
    anemoi_native::Tensor query, anemoi_native::Tensor key, anemoi_native::Tensor value,
    anemoi_native::Tensor block_ids, anemoi_native::Tensor block_counts,
    anemoi_native::Tensor valid_k_counts, double softmax_scale) {
  return fp16_attention_forward<128>(
      query, key, value, block_ids, block_counts, valid_k_counts,
      softmax_scale);
}

template <uint32_t QueryBlock, bool SmoothK>
std::tuple<anemoi_native::Tensor, anemoi_native::Tensor> mixed_attention_forward(
    anemoi_native::Tensor q8,
    anemoi_native::Tensor k8,
    anemoi_native::Tensor v8,
    anemoi_native::Tensor q16,
    anemoi_native::Tensor k16,
    anemoi_native::Tensor v16,
    anemoi_native::Tensor key_mean,
    anemoi_native::Tensor fp8_block_ids,
    anemoi_native::Tensor fp8_block_counts,
    anemoi_native::Tensor fp16_block_ids,
    anemoi_native::Tensor fp16_block_counts,
    anemoi_native::Tensor q_scale,
    anemoi_native::Tensor k_scale,
    anemoi_native::Tensor v_scale,
    anemoi_native::Tensor valid_k_counts,
    int64_t fp16_prefix_blocks,
    double softmax_scale) {
  static_assert(QueryBlock == 64 || QueryBlock == 128);
  for (const auto& item : {
           std::pair<const anemoi_native::Tensor*, const char*>(&q8, "q8"),
           std::pair<const anemoi_native::Tensor*, const char*>(&k8, "k8"),
           std::pair<const anemoi_native::Tensor*, const char*>(&v8, "v8"),
           std::pair<const anemoi_native::Tensor*, const char*>(&q16, "q16"),
           std::pair<const anemoi_native::Tensor*, const char*>(&k16, "k16"),
           std::pair<const anemoi_native::Tensor*, const char*>(&v16, "v16"),
           std::pair<const anemoi_native::Tensor*, const char*>(
               &fp8_block_ids, "fp8_block_ids"),
           std::pair<const anemoi_native::Tensor*, const char*>(
               &fp8_block_counts, "fp8_block_counts"),
           std::pair<const anemoi_native::Tensor*, const char*>(
               &fp16_block_ids, "fp16_block_ids"),
           std::pair<const anemoi_native::Tensor*, const char*>(
               &fp16_block_counts, "fp16_block_counts"),
           std::pair<const anemoi_native::Tensor*, const char*>(&q_scale, "q_scale"),
           std::pair<const anemoi_native::Tensor*, const char*>(&k_scale, "k_scale"),
           std::pair<const anemoi_native::Tensor*, const char*>(&v_scale, "v_scale"),
           std::pair<const anemoi_native::Tensor*, const char*>(
               &valid_k_counts, "valid_k_counts")}) {
    check_cuda_contiguous(*item.first, item.second);
    check_same_device(*item.first, q16, item.second);
  }
  ANEMOI_CHECK(q16.dim() == 4, "q16 must have shape [B,Hq,Q,D]");
  ANEMOI_CHECK(k16.dim() == 4 && v16.dim() == 4,
              "k16/v16 must have shape [B,Hkv,K,D]");
  ANEMOI_CHECK(q16.scalar_type() == anemoi_native::ScalarType::Half,
              "q16 must be FP16");
  ANEMOI_CHECK(k16.scalar_type() == anemoi_native::ScalarType::Half,
              "k16 must be FP16");
  ANEMOI_CHECK(v16.scalar_type() == anemoi_native::ScalarType::Half,
              "v16 must be FP16");
  ANEMOI_CHECK(k16.sizes() == v16.sizes(), "k16/v16 shapes must match");
  ANEMOI_CHECK(
      q16.size(0) > 0 && q16.size(1) > 0 && q16.size(2) > 0 &&
          q16.size(2) % QueryBlock == 0,
      "q16 dimensions must be positive and Q must match the query block");
  ANEMOI_CHECK(k16.size(0) == q16.size(0), "query/key batch mismatch");
  ANEMOI_CHECK(k16.size(2) > 0 && k16.size(2) % 64 == 0,
              "K must be a positive multiple of 64 physical slots");
  const bool supports_head_dim =
      q16.size(3) == 64 || q16.size(3) == 128;
  ANEMOI_CHECK(
      supports_head_dim && k16.size(3) == q16.size(3),
      "native mixed K64 requires head_dim=64 or 128");
  ANEMOI_CHECK(k16.size(1) > 0 && q16.size(1) % k16.size(1) == 0,
              "query heads must be divisible by KV heads");
  if constexpr (SmoothK) {
    check_cuda_contiguous(key_mean, "key_mean");
    check_same_device(key_mean, q16, "key_mean");
    ANEMOI_CHECK(
        key_mean.scalar_type() == anemoi_native::ScalarType::Half &&
            key_mean.sizes() == anemoi_native::IntArrayRef(
                {q16.size(0), k16.size(1), q16.size(3)}),
        "key_mean must have FP16 shape [B,Hkv,D] when K-smooth is enabled");
  }
  ANEMOI_CHECK(
      std::isfinite(softmax_scale) && softmax_scale > 0.0,
      "softmax_scale must be finite and positive");

  ANEMOI_CHECK(q8.scalar_type() == anemoi_native::ScalarType::Char,
              "q8 must be int8");
  ANEMOI_CHECK(k8.scalar_type() == anemoi_native::ScalarType::Char,
              "k8 must be int8");
  ANEMOI_CHECK(v8.scalar_type() == anemoi_native::ScalarType::Float8_e4m3fn,
              "v8 must be float8_e4m3fn");
  ANEMOI_CHECK(q8.sizes() == q16.sizes(), "q8 must match q16 shape");
  ANEMOI_CHECK(k8.sizes() == k16.sizes(), "k8 must match k16 shape");
  ANEMOI_CHECK(
      v8.dim() == 4 && v8.size(0) == q16.size(0) &&
          v8.size(1) == k16.size(1) && v8.size(2) == q16.size(3),
      "v8 must have shape [B,Hkv,D,padded_K]");
  const int64_t padded_kv_len = v8.size(3);
  ANEMOI_CHECK(
      padded_kv_len >= ((k16.size(2) + 127) / 128) * 128 &&
          padded_kv_len % 128 == 0,
      "v8 token dimension must cover K and be padded to 128");
  for (const auto& item : {
           std::pair<const anemoi_native::Tensor*, const char*>(&q_scale, "q_scale"),
           std::pair<const anemoi_native::Tensor*, const char*>(&k_scale, "k_scale"),
           std::pair<const anemoi_native::Tensor*, const char*>(&v_scale, "v_scale")}) {
    ANEMOI_CHECK(item.first->scalar_type() == anemoi_native::ScalarType::Float,
                item.second, " must be FP32");
  }

  const int64_t query_blocks = q16.size(2) / QueryBlock;
  const int64_t key_blocks = k16.size(2) / 64;
  ANEMOI_CHECK(
      q_scale.sizes() == anemoi_native::IntArrayRef(
          {q16.size(0), q16.size(1), query_blocks}),
      "q_scale must have shape [B,Hq,Q/query_block]");
  ANEMOI_CHECK(
      k_scale.sizes() == anemoi_native::IntArrayRef(
          {q16.size(0), k16.size(1), key_blocks}),
      "k_scale must have K64 shape [B,Hkv,K/64]");
  ANEMOI_CHECK(
      v_scale.sizes() == anemoi_native::IntArrayRef(
          {q16.size(0), k16.size(1), q16.size(3)}),
      "v_scale must have shape [B,Hkv,D]");
  for (const auto& item : {
           std::pair<const anemoi_native::Tensor*, const char*>(
               &fp8_block_ids, "fp8_block_ids"),
           std::pair<const anemoi_native::Tensor*, const char*>(
               &fp16_block_ids, "fp16_block_ids")}) {
    ANEMOI_CHECK(item.first->scalar_type() == anemoi_native::ScalarType::Int,
                item.second, " must be int32");
    ANEMOI_CHECK(
        item.first->sizes() == anemoi_native::IntArrayRef(
            {q16.size(0), q16.size(1), query_blocks, key_blocks}),
        item.second, " must have shape [B,Hq,Q/query_block,K/64]");
  }
  for (const auto& item : {
           std::pair<const anemoi_native::Tensor*, const char*>(
               &fp8_block_counts, "fp8_block_counts"),
           std::pair<const anemoi_native::Tensor*, const char*>(
               &fp16_block_counts, "fp16_block_counts")}) {
    ANEMOI_CHECK(item.first->scalar_type() == anemoi_native::ScalarType::Int,
                item.second, " must be int32");
    ANEMOI_CHECK(
        item.first->sizes() == anemoi_native::IntArrayRef(
            {q16.size(0), q16.size(1), query_blocks}),
        item.second, " must have shape [B,Hq,Q/query_block]");
  }
  ANEMOI_CHECK(valid_k_counts.scalar_type() == anemoi_native::ScalarType::Int,
              "valid_k_counts must be int32");
  ANEMOI_CHECK(
      valid_k_counts.sizes() ==
          anemoi_native::IntArrayRef({q16.size(0), key_blocks}),
      "valid_k_counts must have shape [B,K/64]");
  ANEMOI_CHECK(
      fp16_prefix_blocks >= 0 && fp16_prefix_blocks <= key_blocks,
      "fp16_prefix_blocks must be in [0,K/64]");
  ANEMOI_CHECK(
      fp8_block_ids.data_ptr<int32_t>() == fp16_block_ids.data_ptr<int32_t>(),
      "native mixed K64 requires one aliased compact FP8/FP16 route list");

  auto output = anemoi_native::empty_like(q16);
  auto lse = anemoi_native::empty(
      {0}, q16.options().dtype(anemoi_native::ScalarType::Float));
  anemoi_native::cuda::CUDAGuard device_guard(q16.device());
  cudaStream_t stream = anemoi_native::cuda::getCurrentCUDAStream(q16.get_device());
  const auto launch = [&](auto head_dim_tag) {
    constexpr uint32_t HeadDim = decltype(head_dim_tag)::value;
    if constexpr (QueryBlock == 64) {
      launch_mixed_attention_sm89_k64<HeadDim, true, true, SmoothK>(
          q8.data_ptr<int8_t>(), k8.data_ptr<int8_t>(),
          reinterpret_cast<__nv_fp8_e4m3*>(v8.data_ptr()),
          reinterpret_cast<half*>(q16.data_ptr<anemoi_native::Half>()),
          reinterpret_cast<half*>(k16.data_ptr<anemoi_native::Half>()),
          reinterpret_cast<half*>(v16.data_ptr<anemoi_native::Half>()),
          SmoothK
              ? reinterpret_cast<half*>(key_mean.data_ptr<anemoi_native::Half>())
              : nullptr,
          reinterpret_cast<half*>(output.data_ptr<anemoi_native::Half>()),
          fp8_block_ids.data_ptr<int32_t>(),
          fp8_block_counts.data_ptr<int32_t>(),
          fp8_block_ids.data_ptr<int32_t>(),
          fp16_block_counts.data_ptr<int32_t>(), q_scale.data_ptr<float>(),
          k_scale.data_ptr<float>(), v_scale.data_ptr<float>(),
          valid_k_counts.data_ptr<int32_t>(), nullptr,
          static_cast<uint32_t>(fp16_prefix_blocks),
          static_cast<uint32_t>(q16.size(0)),
          static_cast<uint32_t>(q16.size(2)),
          static_cast<uint32_t>(k16.size(2)),
          static_cast<uint32_t>(padded_kv_len),
          static_cast<uint32_t>(q16.size(1)),
          static_cast<uint32_t>(k16.size(1)),
          static_cast<float>(softmax_scale), stream);
    } else {
      launch_mixed_attention_sm89_q128_k64<HeadDim, true, true, SmoothK>(
          q8.data_ptr<int8_t>(), k8.data_ptr<int8_t>(),
          reinterpret_cast<__nv_fp8_e4m3*>(v8.data_ptr()),
          reinterpret_cast<half*>(q16.data_ptr<anemoi_native::Half>()),
          reinterpret_cast<half*>(k16.data_ptr<anemoi_native::Half>()),
          reinterpret_cast<half*>(v16.data_ptr<anemoi_native::Half>()),
          SmoothK
              ? reinterpret_cast<half*>(key_mean.data_ptr<anemoi_native::Half>())
              : nullptr,
          reinterpret_cast<half*>(output.data_ptr<anemoi_native::Half>()),
          fp8_block_ids.data_ptr<int32_t>(),
          fp8_block_counts.data_ptr<int32_t>(),
          fp8_block_ids.data_ptr<int32_t>(),
          fp16_block_counts.data_ptr<int32_t>(), q_scale.data_ptr<float>(),
          k_scale.data_ptr<float>(), v_scale.data_ptr<float>(),
          valid_k_counts.data_ptr<int32_t>(), nullptr,
          static_cast<uint32_t>(fp16_prefix_blocks),
          static_cast<uint32_t>(q16.size(0)),
          static_cast<uint32_t>(q16.size(2)),
          static_cast<uint32_t>(k16.size(2)),
          static_cast<uint32_t>(padded_kv_len),
          static_cast<uint32_t>(q16.size(1)),
          static_cast<uint32_t>(k16.size(1)),
          static_cast<float>(softmax_scale), stream);
    }
  };
  if (q16.size(3) == 64) {
    launch(std::integral_constant<uint32_t, 64>{});
  } else {
    launch(std::integral_constant<uint32_t, 128>{});
  }
  ANEMOI_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, lse};
}

std::tuple<anemoi_native::Tensor, anemoi_native::Tensor> k64_mixed_attention_forward(
    anemoi_native::Tensor q8, anemoi_native::Tensor k8, anemoi_native::Tensor v8,
    anemoi_native::Tensor q16, anemoi_native::Tensor k16, anemoi_native::Tensor v16,
    anemoi_native::Tensor fp8_block_ids, anemoi_native::Tensor fp8_block_counts,
    anemoi_native::Tensor fp16_block_ids, anemoi_native::Tensor fp16_block_counts,
    anemoi_native::Tensor q_scale, anemoi_native::Tensor k_scale, anemoi_native::Tensor v_scale,
    anemoi_native::Tensor valid_k_counts, int64_t fp16_prefix_blocks,
    double softmax_scale) {
  return mixed_attention_forward<64, false>(
      q8, k8, v8, q16, k16, v16, anemoi_native::Tensor(),
      fp8_block_ids, fp8_block_counts,
      fp16_block_ids, fp16_block_counts, q_scale, k_scale, v_scale,
      valid_k_counts, fp16_prefix_blocks, softmax_scale);
}

std::tuple<anemoi_native::Tensor, anemoi_native::Tensor> q128_k64_mixed_attention_forward(
    anemoi_native::Tensor q8, anemoi_native::Tensor k8, anemoi_native::Tensor v8,
    anemoi_native::Tensor q16, anemoi_native::Tensor k16, anemoi_native::Tensor v16,
    anemoi_native::Tensor fp8_block_ids, anemoi_native::Tensor fp8_block_counts,
    anemoi_native::Tensor fp16_block_ids, anemoi_native::Tensor fp16_block_counts,
    anemoi_native::Tensor q_scale, anemoi_native::Tensor k_scale, anemoi_native::Tensor v_scale,
    anemoi_native::Tensor valid_k_counts, int64_t fp16_prefix_blocks,
    double softmax_scale) {
  return mixed_attention_forward<128, false>(
      q8, k8, v8, q16, k16, v16, anemoi_native::Tensor(),
      fp8_block_ids, fp8_block_counts,
      fp16_block_ids, fp16_block_counts, q_scale, k_scale, v_scale,
      valid_k_counts, fp16_prefix_blocks, softmax_scale);
}

std::tuple<anemoi_native::Tensor, anemoi_native::Tensor> k64_smooth_mixed_attention_forward(
    anemoi_native::Tensor q8, anemoi_native::Tensor k8, anemoi_native::Tensor v8,
    anemoi_native::Tensor q16, anemoi_native::Tensor k16, anemoi_native::Tensor v16,
    anemoi_native::Tensor key_mean,
    anemoi_native::Tensor fp8_block_ids, anemoi_native::Tensor fp8_block_counts,
    anemoi_native::Tensor fp16_block_ids, anemoi_native::Tensor fp16_block_counts,
    anemoi_native::Tensor q_scale, anemoi_native::Tensor k_scale, anemoi_native::Tensor v_scale,
    anemoi_native::Tensor valid_k_counts, int64_t fp16_prefix_blocks,
    double softmax_scale) {
  return mixed_attention_forward<64, true>(
      q8, k8, v8, q16, k16, v16, key_mean,
      fp8_block_ids, fp8_block_counts, fp16_block_ids, fp16_block_counts,
      q_scale, k_scale, v_scale, valid_k_counts, fp16_prefix_blocks,
      softmax_scale);
}

std::tuple<anemoi_native::Tensor, anemoi_native::Tensor>
q128_k64_smooth_mixed_attention_forward(
    anemoi_native::Tensor q8, anemoi_native::Tensor k8, anemoi_native::Tensor v8,
    anemoi_native::Tensor q16, anemoi_native::Tensor k16, anemoi_native::Tensor v16,
    anemoi_native::Tensor key_mean,
    anemoi_native::Tensor fp8_block_ids, anemoi_native::Tensor fp8_block_counts,
    anemoi_native::Tensor fp16_block_ids, anemoi_native::Tensor fp16_block_counts,
    anemoi_native::Tensor q_scale, anemoi_native::Tensor k_scale, anemoi_native::Tensor v_scale,
    anemoi_native::Tensor valid_k_counts, int64_t fp16_prefix_blocks,
    double softmax_scale) {
  return mixed_attention_forward<128, true>(
      q8, k8, v8, q16, k16, v16, key_mean,
      fp8_block_ids, fp8_block_counts, fp16_block_ids, fp16_block_counts,
      q_scale, k_scale, v_scale, valid_k_counts, fp16_prefix_blocks,
      softmax_scale);
}

template <uint32_t QueryBlock, bool SmoothK>
std::tuple<anemoi_native::Tensor, anemoi_native::Tensor> pure_fp8_attention_forward(
    anemoi_native::Tensor q8,
    anemoi_native::Tensor k8,
    anemoi_native::Tensor v8,
    anemoi_native::Tensor block_ids,
    anemoi_native::Tensor block_counts,
    anemoi_native::Tensor q_scale,
    anemoi_native::Tensor k_scale,
    anemoi_native::Tensor v_scale,
    anemoi_native::Tensor valid_k_counts,
    anemoi_native::Tensor q16,
    anemoi_native::Tensor key_mean,
    double softmax_scale) {
  for (const auto& item : {
           std::pair<const anemoi_native::Tensor*, const char*>(&q8, "q8"),
           std::pair<const anemoi_native::Tensor*, const char*>(&k8, "k8"),
           std::pair<const anemoi_native::Tensor*, const char*>(&v8, "v8"),
           std::pair<const anemoi_native::Tensor*, const char*>(&block_ids, "block_ids"),
           std::pair<const anemoi_native::Tensor*, const char*>(&block_counts, "block_counts"),
           std::pair<const anemoi_native::Tensor*, const char*>(&q_scale, "q_scale"),
           std::pair<const anemoi_native::Tensor*, const char*>(&k_scale, "k_scale"),
           std::pair<const anemoi_native::Tensor*, const char*>(&v_scale, "v_scale"),
           std::pair<const anemoi_native::Tensor*, const char*>(
               &valid_k_counts, "valid_k_counts")}) {
    check_cuda_contiguous(*item.first, item.second);
    check_same_device(*item.first, q8, item.second);
  }
  ANEMOI_CHECK(q8.scalar_type() == anemoi_native::ScalarType::Char &&
                  k8.scalar_type() == anemoi_native::ScalarType::Char &&
                  v8.scalar_type() == anemoi_native::ScalarType::Float8_e4m3fn,
              "pure K64 audit requires INT8 Q/K and E4M3 V");
  ANEMOI_CHECK(q8.dim() == 4 && k8.dim() == 4 && v8.dim() == 4 &&
                  (q8.size(3) == 64 || q8.size(3) == 128) &&
                  k8.size(3) == q8.size(3) &&
                  k8.size(2) > 0 && k8.size(2) % 64 == 0 &&
                  q8.size(0) == k8.size(0) && k8.size(1) > 0 && q8.size(1) % k8.size(1) == 0 &&
                  v8.size(0) == k8.size(0) && v8.size(1) == k8.size(1) &&
                  v8.size(2) == q8.size(3) && v8.size(3) >= ((k8.size(2) + 127) / 128) * 128 && v8.size(3) % 128 == 0 && q8.size(0) > 0 && q8.size(1) > 0 && q8.size(2) > 0,
              "invalid pure K64 audit operand shapes");
  static_assert(QueryBlock == 64 || QueryBlock == 128);
  const int64_t query_blocks =
      (q8.size(2) + QueryBlock - 1) / QueryBlock;
  const int64_t key_blocks = k8.size(2) / 64;
  ANEMOI_CHECK(
      block_ids.scalar_type() == anemoi_native::ScalarType::Int &&
          block_ids.sizes() == anemoi_native::IntArrayRef(
              {q8.size(0), q8.size(1), query_blocks, key_blocks}) &&
          block_counts.scalar_type() == anemoi_native::ScalarType::Int &&
          block_counts.sizes() == anemoi_native::IntArrayRef(
              {q8.size(0), q8.size(1), query_blocks}) &&
          valid_k_counts.scalar_type() == anemoi_native::ScalarType::Int &&
          valid_k_counts.sizes() == anemoi_native::IntArrayRef(
              {q8.size(0), key_blocks}),
      "invalid pure K64 audit route shapes");
  ANEMOI_CHECK(
      q_scale.scalar_type() == anemoi_native::ScalarType::Float &&
          q_scale.sizes() == anemoi_native::IntArrayRef(
              {q8.size(0), q8.size(1), query_blocks}) &&
          k_scale.scalar_type() == anemoi_native::ScalarType::Float &&
          k_scale.sizes() == anemoi_native::IntArrayRef(
              {q8.size(0), k8.size(1), key_blocks}) &&
          v_scale.scalar_type() == anemoi_native::ScalarType::Float &&
          v_scale.sizes() == anemoi_native::IntArrayRef(
              {q8.size(0), k8.size(1), q8.size(3)}),
      "invalid pure K64 audit scale shapes");
  ANEMOI_CHECK(
      std::isfinite(softmax_scale) && softmax_scale > 0.0,
      "softmax_scale must be finite and positive");
  if constexpr (SmoothK) {
    check_cuda_contiguous(q16, "q16");
    check_same_device(q16, q8, "q16");
    check_cuda_contiguous(key_mean, "key_mean");
    check_same_device(key_mean, q8, "key_mean");
    ANEMOI_CHECK(
        q16.scalar_type() == anemoi_native::ScalarType::Half && q16.sizes() == q8.sizes(),
        "q16 must be FP16 and match q8 when K-smooth is enabled");
    ANEMOI_CHECK(
        key_mean.scalar_type() == anemoi_native::ScalarType::Half &&
            key_mean.sizes() == anemoi_native::IntArrayRef(
                {q8.size(0), k8.size(1), q8.size(3)}),
        "key_mean must have FP16 shape [B,Hkv,D] when K-smooth is enabled");
  }

  auto output =
      anemoi_native::empty(q8.sizes(), q8.options().dtype(anemoi_native::ScalarType::Half));
  auto lse = anemoi_native::empty(
      {0}, q8.options().dtype(anemoi_native::ScalarType::Float));
  anemoi_native::cuda::CUDAGuard device_guard(q8.device());
  cudaStream_t stream = anemoi_native::cuda::getCurrentCUDAStream(q8.get_device());
  const auto launch = [&](auto head_dim_tag) {
    constexpr uint32_t HeadDim = decltype(head_dim_tag)::value;
    if constexpr (QueryBlock == 64) {
      launch_mixed_attention_sm89_k64<HeadDim, true, false, false>(
          q8.data_ptr<int8_t>(), k8.data_ptr<int8_t>(),
          reinterpret_cast<__nv_fp8_e4m3*>(v8.data_ptr()),
          nullptr, nullptr, nullptr, nullptr,
          reinterpret_cast<half*>(output.data_ptr<anemoi_native::Half>()),
          block_ids.data_ptr<int32_t>(), block_counts.data_ptr<int32_t>(),
          nullptr, nullptr, q_scale.data_ptr<float>(),
          k_scale.data_ptr<float>(), v_scale.data_ptr<float>(),
          valid_k_counts.data_ptr<int32_t>(), nullptr, 0,
          static_cast<uint32_t>(q8.size(0)),
          static_cast<uint32_t>(q8.size(2)),
          static_cast<uint32_t>(k8.size(2)),
          static_cast<uint32_t>(v8.size(3)),
          static_cast<uint32_t>(q8.size(1)),
          static_cast<uint32_t>(k8.size(1)),
          static_cast<float>(softmax_scale), stream);
    } else {
      launch_mixed_attention_sm89_q128_k64<HeadDim, true, false, false>(
          q8.data_ptr<int8_t>(), k8.data_ptr<int8_t>(),
          reinterpret_cast<__nv_fp8_e4m3*>(v8.data_ptr()),
          nullptr, nullptr, nullptr, nullptr,
          reinterpret_cast<half*>(output.data_ptr<anemoi_native::Half>()),
          block_ids.data_ptr<int32_t>(), block_counts.data_ptr<int32_t>(),
          nullptr, nullptr, q_scale.data_ptr<float>(),
          k_scale.data_ptr<float>(), v_scale.data_ptr<float>(),
          valid_k_counts.data_ptr<int32_t>(), nullptr, 0,
          static_cast<uint32_t>(q8.size(0)),
          static_cast<uint32_t>(q8.size(2)),
          static_cast<uint32_t>(k8.size(2)),
          static_cast<uint32_t>(v8.size(3)),
          static_cast<uint32_t>(q8.size(1)),
          static_cast<uint32_t>(k8.size(1)),
          static_cast<float>(softmax_scale), stream);
    }
  };
  if (q8.size(3) == 64) {
    launch(std::integral_constant<uint32_t, 64>{});
  } else {
    launch(std::integral_constant<uint32_t, 128>{});
  }
  ANEMOI_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, lse};
}

std::tuple<anemoi_native::Tensor, anemoi_native::Tensor> k64_fp8_attention_forward(
    anemoi_native::Tensor q8, anemoi_native::Tensor k8, anemoi_native::Tensor v8,
    anemoi_native::Tensor block_ids, anemoi_native::Tensor block_counts,
    anemoi_native::Tensor q_scale, anemoi_native::Tensor k_scale, anemoi_native::Tensor v_scale,
    anemoi_native::Tensor valid_k_counts, double softmax_scale) {
  return pure_fp8_attention_forward<64, false>(
      q8, k8, v8, block_ids, block_counts, q_scale, k_scale, v_scale,
      valid_k_counts, anemoi_native::Tensor(), anemoi_native::Tensor(), softmax_scale);
}

std::tuple<anemoi_native::Tensor, anemoi_native::Tensor> q128_k64_fp8_attention_forward(
    anemoi_native::Tensor q8, anemoi_native::Tensor k8, anemoi_native::Tensor v8,
    anemoi_native::Tensor block_ids, anemoi_native::Tensor block_counts,
    anemoi_native::Tensor q_scale, anemoi_native::Tensor k_scale, anemoi_native::Tensor v_scale,
    anemoi_native::Tensor valid_k_counts, double softmax_scale) {
  return pure_fp8_attention_forward<128, false>(
      q8, k8, v8, block_ids, block_counts, q_scale, k_scale, v_scale,
      valid_k_counts, anemoi_native::Tensor(), anemoi_native::Tensor(), softmax_scale);
}

std::tuple<anemoi_native::Tensor, anemoi_native::Tensor> k64_smooth_fp8_attention_forward(
    anemoi_native::Tensor q8, anemoi_native::Tensor k8, anemoi_native::Tensor v8,
    anemoi_native::Tensor q16, anemoi_native::Tensor key_mean,
    anemoi_native::Tensor block_ids, anemoi_native::Tensor block_counts,
    anemoi_native::Tensor q_scale, anemoi_native::Tensor k_scale, anemoi_native::Tensor v_scale,
    anemoi_native::Tensor valid_k_counts, double softmax_scale) {
  return pure_fp8_attention_forward<64, true>(
      q8, k8, v8, block_ids, block_counts, q_scale, k_scale, v_scale,
      valid_k_counts, q16, key_mean, softmax_scale);
}

std::tuple<anemoi_native::Tensor, anemoi_native::Tensor>
q128_k64_smooth_fp8_attention_forward(
    anemoi_native::Tensor q8, anemoi_native::Tensor k8, anemoi_native::Tensor v8,
    anemoi_native::Tensor q16, anemoi_native::Tensor key_mean,
    anemoi_native::Tensor block_ids, anemoi_native::Tensor block_counts,
    anemoi_native::Tensor q_scale, anemoi_native::Tensor k_scale, anemoi_native::Tensor v_scale,
    anemoi_native::Tensor valid_k_counts, double softmax_scale) {
  return pure_fp8_attention_forward<128, true>(
      q8, k8, v8, block_ids, block_counts, q_scale, k_scale, v_scale,
      valid_k_counts, q16, key_mean, softmax_scale);
}

anemoi_native::Tensor sm89_q64_prefix_int8_attention_forward(
    anemoi_native::Tensor q8,
    anemoi_native::Tensor k8,
    anemoi_native::Tensor v8,
    anemoi_native::Tensor q_scale,
    anemoi_native::Tensor k_scale,
    anemoi_native::Tensor v_scale,
    anemoi_native::Tensor valid_k_counts,
    int64_t prefix_tokens,
    double softmax_scale) {
  for (const auto& item : {
           std::pair<const anemoi_native::Tensor*, const char*>(&q8, "q8"),
           std::pair<const anemoi_native::Tensor*, const char*>(&k8, "k8"),
           std::pair<const anemoi_native::Tensor*, const char*>(&v8, "v8"),
           std::pair<const anemoi_native::Tensor*, const char*>(&q_scale, "q_scale"),
           std::pair<const anemoi_native::Tensor*, const char*>(&k_scale, "k_scale"),
           std::pair<const anemoi_native::Tensor*, const char*>(&v_scale, "v_scale"),
           std::pair<const anemoi_native::Tensor*, const char*>(
               &valid_k_counts, "valid_k_counts")}) {
    check_cuda_contiguous(*item.first, item.second);
    check_same_device(*item.first, q8, item.second);
  }
  ANEMOI_CHECK(
      q8.scalar_type() == anemoi_native::ScalarType::Char &&
          k8.scalar_type() == anemoi_native::ScalarType::Char,
      "q8/k8 must be int8");
  ANEMOI_CHECK(
      v8.scalar_type() == anemoi_native::ScalarType::Float8_e4m3fn,
      "v8 must be float8_e4m3fn");
  ANEMOI_CHECK(
      q8.dim() == 4 && q8.size(0) > 0 && q8.size(1) > 0 &&
          q8.size(2) > 0 && q8.size(2) % 64 == 0 &&
          (q8.size(3) == 64 || q8.size(3) == 128),
      "q8 must have padded shape [B,Hq,Q64,D] with D=64 or 128");
  ANEMOI_CHECK(
      k8.dim() == 4 && k8.size(0) == q8.size(0) && k8.size(1) > 0 &&
          k8.size(2) > 0 && k8.size(2) % 64 == 0 &&
          k8.size(3) == q8.size(3) &&
          k8.size(1) > 0 && q8.size(1) % k8.size(1) == 0,
      "k8 must have compatible [B,Hkv,K64,D] layout");
  ANEMOI_CHECK(
      prefix_tokens > 0 && prefix_tokens <= q8.size(2) &&
          prefix_tokens > q8.size(2) - 64,
      "prefix_tokens must select the nonempty final-padded Q64 prefix");
  ANEMOI_CHECK(
      std::isfinite(softmax_scale) && softmax_scale > 0.0,
      "softmax_scale must be finite and positive");

  const int64_t query_blocks = q8.size(2) / 64;
  const int64_t key_blocks = k8.size(2) / 64;
  ANEMOI_CHECK(
      v8.dim() == 4 && v8.size(0) == q8.size(0) &&
          v8.size(1) == k8.size(1) && v8.size(2) == q8.size(3) &&
          v8.size(3) >= ((k8.size(2) + 127) / 128) * 128 &&
          v8.size(3) % 128 == 0,
      "v8 must have verified [B,Hkv,D,padded_K] layout");
  ANEMOI_CHECK(
      q_scale.scalar_type() == anemoi_native::ScalarType::Float &&
          q_scale.sizes() == anemoi_native::IntArrayRef(
              {q8.size(0), q8.size(1), query_blocks}),
      "q_scale must have shape [B,Hq,Q/64]");
  ANEMOI_CHECK(
      k_scale.scalar_type() == anemoi_native::ScalarType::Float &&
          k_scale.sizes() == anemoi_native::IntArrayRef(
              {q8.size(0), k8.size(1), key_blocks}),
      "k_scale must have shape [B,Hkv,K/64]");
  ANEMOI_CHECK(
      v_scale.scalar_type() == anemoi_native::ScalarType::Float &&
          v_scale.sizes() == anemoi_native::IntArrayRef(
              {q8.size(0), k8.size(1), q8.size(3)}),
      "v_scale must have shape [B,Hkv,D]");
  ANEMOI_CHECK(
      valid_k_counts.scalar_type() == anemoi_native::ScalarType::Int &&
          valid_k_counts.sizes() ==
              anemoi_native::IntArrayRef({q8.size(0), key_blocks}),
      "valid_k_counts must have shape [B,K/64]");

  auto output = anemoi_native::empty(
      q8.sizes(), q8.options().dtype(anemoi_native::ScalarType::Half));
  anemoi_native::cuda::CUDAGuard device_guard(q8.device());
  cudaStream_t stream = anemoi_native::cuda::getCurrentCUDAStream(q8.get_device());
  const auto launch = [&](auto head_dim_tag) {
    constexpr uint32_t HeadDim = decltype(head_dim_tag)::value;
    launch_mixed_attention_sm89_k64_int8_dense<
        HeadDim, true, false, false>(
        q8.data_ptr<int8_t>(), k8.data_ptr<int8_t>(),
        reinterpret_cast<__nv_fp8_e4m3*>(v8.data_ptr()),
        nullptr, nullptr, nullptr, nullptr,
        reinterpret_cast<half*>(output.data_ptr<anemoi_native::Half>()),
        nullptr, nullptr, nullptr, nullptr,
        q_scale.data_ptr<float>(), k_scale.data_ptr<float>(),
        v_scale.data_ptr<float>(), valid_k_counts.data_ptr<int32_t>(),
        nullptr, 0, static_cast<uint32_t>(q8.size(0)),
        static_cast<uint32_t>(q8.size(2)),
        static_cast<uint32_t>(k8.size(2)),
        static_cast<uint32_t>(v8.size(3)),
        static_cast<uint32_t>(q8.size(1)),
        static_cast<uint32_t>(k8.size(1)),
        static_cast<float>(softmax_scale), stream);
  };
  if (q8.size(3) == 64) {
    launch(std::integral_constant<uint32_t, 64>{});
  } else {
    launch(std::integral_constant<uint32_t, 128>{});
  }
  ANEMOI_CUDA_KERNEL_LAUNCH_CHECK();
  return output.narrow(2, 0, prefix_tokens);
}
