// Adapted from Anemoi 4e85afba741bdeaf2d9486cab19cb76d3e7985a4, updated
// through 1765573bea31159adba32165770d19171d5153bd (native SM120 D64
// support); Apache-2.0.
/*
 * Native Q64 x K64 FP16 attention host dispatch for controlled Sol-H3
 * alignment experiments on SM120.
 */


#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <tuple>
#include <type_traits>
#include <utility>

#include "attention_api.h"
using namespace anemoi_sm120;
#include "q64_attention_decl.cuh"

namespace {

void check_cuda_contiguous(const TensorView& tensor, const char* name) {
  ANEMOI_SM120_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  ANEMOI_SM120_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  for (auto d : tensor.shape) ANEMOI_SM120_CHECK(d >= 0 && d <= UINT32_MAX, name, " dimension exceeds uint32");
  ANEMOI_SM120_CHECK(tensor.numel() <= UINT32_MAX, name, " exceeds donor uint32 indexing capacity");
  if (tensor.dim() == 4) ANEMOI_SM120_CHECK(tensor.size(0) > 0 && tensor.size(1) > 0, name, " batch/heads must be positive");
}

void check_same_device(
    const TensorView& tensor,
    const TensorView& reference,
    const char* name) {
  ANEMOI_SM120_CHECK(
      tensor.device() == reference.device(), name,
      " must be on the same CUDA device as query");
}

template <uint32_t QueryBlock>
void fp16_attention_forward(
    TensorView query,
    TensorView key,
    TensorView value,
    TensorView block_ids,
    TensorView block_counts,
    TensorView valid_k_counts,
    double softmax_scale, TensorView output, cudaStream_t stream) {
  static_assert(QueryBlock == 64 || QueryBlock == 128);
  for (const auto& item : {
           std::pair<const TensorView*, const char*>(&query, "query"),
           std::pair<const TensorView*, const char*>(&key, "key"),
           std::pair<const TensorView*, const char*>(&value, "value"),
           std::pair<const TensorView*, const char*>(&block_ids, "block_ids"),
           std::pair<const TensorView*, const char*>(&block_counts, "block_counts"),
           std::pair<const TensorView*, const char*>(
               &valid_k_counts, "valid_k_counts")}) {
    check_cuda_contiguous(*item.first, item.second);
    check_same_device(*item.first, query, item.second);
  }
  ANEMOI_SM120_CHECK(query.dim() == 4, "query must have shape [B,Hq,Q,D]");
  ANEMOI_SM120_CHECK(key.dim() == 4 && value.dim() == 4,
              "key/value must have shape [B,Hkv,K,D]");
  ANEMOI_SM120_CHECK(query.scalar_type() == ScalarType::Half,
              "query must be FP16");
  ANEMOI_SM120_CHECK(key.scalar_type() == ScalarType::Half,
              "key must be FP16");
  ANEMOI_SM120_CHECK(value.scalar_type() == ScalarType::Half,
              "value must be FP16");
  ANEMOI_SM120_CHECK(key.sizes() == value.sizes(), "key/value shapes must match");
  ANEMOI_SM120_CHECK(query.size(0) > 0 && query.size(1) > 0 && query.size(2) > 0,
              "query dimensions must be positive");
  if constexpr (QueryBlock == 128) {
    ANEMOI_SM120_CHECK(query.size(2) % 128 == 0, "Q must be a multiple of 128");
  }
  ANEMOI_SM120_CHECK(key.size(0) == query.size(0), "query/key batch mismatch");
  ANEMOI_SM120_CHECK(key.size(2) > 0 && key.size(2) % 64 == 0,
              "K must be a positive multiple of 64 physical slots");
  ANEMOI_SM120_CHECK((query.size(3) == 64 || query.size(3) == 128) && key.size(3) == query.size(3),
              "native K64 attention requires head_dim in {64,128}");
  ANEMOI_SM120_CHECK(query.size(1) % key.size(1) == 0,
              "query heads must be divisible by KV heads");
  ANEMOI_SM120_CHECK(
      std::isfinite(softmax_scale) && softmax_scale > 0.0,
      "softmax_scale must be finite and positive");

  ANEMOI_SM120_CHECK(block_ids.scalar_type() == ScalarType::Int,
              "block_ids must be int32");
  ANEMOI_SM120_CHECK(block_counts.scalar_type() == ScalarType::Int,
              "block_counts must be int32");
  ANEMOI_SM120_CHECK(valid_k_counts.scalar_type() == ScalarType::Int,
              "valid_k_counts must be int32");
  const int64_t query_blocks =
      (query.size(2) + QueryBlock - 1) / QueryBlock;
  const int64_t key_blocks = key.size(2) / 64;
  ANEMOI_SM120_CHECK(
      block_ids.sizes() == Shape(
          {query.size(0), query.size(1), query_blocks, key_blocks}),
      "block_ids must have shape [B,Hq,ceil(Q/query_block),K/64]");
  ANEMOI_SM120_CHECK(
      block_counts.sizes() ==
          Shape({query.size(0), query.size(1), query_blocks}),
      "block_counts must have shape [B,Hq,ceil(Q/query_block)]");
  ANEMOI_SM120_CHECK(
      valid_k_counts.sizes() ==
          Shape({query.size(0), key_blocks}),
      "valid_k_counts must have shape [B,K/64]");

  check_output(output, query, stream);

  const auto launch = [&](auto head_dim_tag) {
    constexpr uint32_t HeadDim = decltype(head_dim_tag)::value;
    if constexpr (QueryBlock == 64) {
      launch_mixed_attention_sm120_q64<HeadDim, false, true, false>(
          nullptr, nullptr, nullptr,
          reinterpret_cast<half*>(query.data_ptr<half>()),
          reinterpret_cast<half*>(key.data_ptr<half>()),
          reinterpret_cast<half*>(value.data_ptr<half>()), nullptr,
          reinterpret_cast<half*>(output.data_ptr<half>()),
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
      launch_mixed_attention_sm120_q128_fp16<HeadDim, false, true, false>(
          nullptr, nullptr, nullptr,
          reinterpret_cast<half*>(query.data_ptr<half>()),
          reinterpret_cast<half*>(key.data_ptr<half>()),
          reinterpret_cast<half*>(value.data_ptr<half>()), nullptr,
          reinterpret_cast<half*>(output.data_ptr<half>()),
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
  };
  if (query.size(3) == 64) {
    launch(std::integral_constant<uint32_t, 64>{});
  } else {
    launch(std::integral_constant<uint32_t, 128>{});
  }
  ANEMOI_SM120_CUDA_CHECK(cudaGetLastError());
  return;
}

}  // namespace

void sm120_q64_fp16_attention_forward(
    TensorView query,
    TensorView key,
    TensorView value,
    TensorView block_ids,
    TensorView block_counts,
    TensorView valid_k_counts,
    double softmax_scale, TensorView output, cudaStream_t stream) {
  return fp16_attention_forward<64>(
      query, key, value, block_ids, block_counts, valid_k_counts,
      softmax_scale, output, stream);
}

void sm120_q128_fp16_attention_forward(
    TensorView query,
    TensorView key,
    TensorView value,
    TensorView block_ids,
    TensorView block_counts,
    TensorView valid_k_counts,
    double softmax_scale, TensorView output, cudaStream_t stream) {
  return fp16_attention_forward<128>(
      query, key, value, block_ids, block_counts, valid_k_counts,
      softmax_scale, output, stream);
}

template <uint32_t QueryBlock>
void int8_attention_forward(
    TensorView q8,
    TensorView k8,
    TensorView v8,
    TensorView q16,
    TensorView k16,
    TensorView v16,
    TensorView block_ids,
    TensorView int8_block_counts,
    TensorView fp16_block_counts,
    TensorView q_scale,
    TensorView k_scale,
    TensorView v_scale,
    TensorView valid_k_counts,
    int64_t fp16_prefix_blocks,
    double softmax_scale,
    bool active_fp16, TensorView output, cudaStream_t stream) {
  static_assert(QueryBlock == 64 || QueryBlock == 128);
  for (const auto& item : {
           std::pair<const TensorView*, const char*>(&q8, "q8"),
           std::pair<const TensorView*, const char*>(&k8, "k8"),
           std::pair<const TensorView*, const char*>(&v8, "v8"),
           std::pair<const TensorView*, const char*>(&q16, "q16"),
           std::pair<const TensorView*, const char*>(&k16, "k16"),
           std::pair<const TensorView*, const char*>(&v16, "v16"),
           std::pair<const TensorView*, const char*>(
               &block_ids, "block_ids"),
           std::pair<const TensorView*, const char*>(
               &int8_block_counts, "int8_block_counts"),
           std::pair<const TensorView*, const char*>(
               &fp16_block_counts, "fp16_block_counts"),
           std::pair<const TensorView*, const char*>(&q_scale, "q_scale"),
           std::pair<const TensorView*, const char*>(&k_scale, "k_scale"),
           std::pair<const TensorView*, const char*>(&v_scale, "v_scale"),
           std::pair<const TensorView*, const char*>(
               &valid_k_counts, "valid_k_counts")}) {
    check_cuda_contiguous(*item.first, item.second);
    check_same_device(*item.first, q16, item.second);
  }

  ANEMOI_SM120_CHECK(
      q16.dim() == 4 && k16.dim() == 4 && v16.dim() == 4,
      "q16/k16/v16 must have shape [B,H,Q_or_K,D]");
  ANEMOI_SM120_CHECK(
      q16.scalar_type() == ScalarType::Half &&
          k16.scalar_type() == ScalarType::Half &&
          v16.scalar_type() == ScalarType::Half,
      "q16/k16/v16 must be FP16");
  ANEMOI_SM120_CHECK(k16.sizes() == v16.sizes(), "k16/v16 shapes must match");
  ANEMOI_SM120_CHECK(
      q16.size(0) > 0 && q16.size(1) > 0 && q16.size(2) > 0 &&
          q16.size(2) % QueryBlock == 0,
      "Q must be positive and divisible by the query block");
  ANEMOI_SM120_CHECK(
      k16.size(0) == q16.size(0) && k16.size(2) > 0 &&
          k16.size(2) % 64 == 0,
      "K must share the batch and contain complete physical K64 slots");
  ANEMOI_SM120_CHECK(
      (q16.size(3) == 64 || q16.size(3) == 128) && k16.size(3) == q16.size(3),
      "native INT8 K64 attention requires head_dim in {64,128}");
  ANEMOI_SM120_CHECK(
      q16.size(1) % k16.size(1) == 0,
      "query heads must be divisible by KV heads");
  ANEMOI_SM120_CHECK(
      std::isfinite(softmax_scale) && softmax_scale > 0.0,
      "softmax_scale must be finite and positive");

  ANEMOI_SM120_CHECK(
      q8.scalar_type() == ScalarType::Char &&
          k8.scalar_type() == ScalarType::Char,
      "q8/k8 must be int8");
  ANEMOI_SM120_CHECK(
      v8.scalar_type() == ScalarType::Float8_e4m3fn,
      "v8 must be float8_e4m3fn");
  ANEMOI_SM120_CHECK(q8.sizes() == q16.sizes(), "q8 must match q16 shape");
  ANEMOI_SM120_CHECK(k8.sizes() == k16.sizes(), "k8 must match k16 shape");
  ANEMOI_SM120_CHECK(
      v8.dim() == 4 && v8.size(0) == q16.size(0) &&
          v8.size(1) == k16.size(1) && v8.size(2) == q16.size(3),
      "v8 must have shape [B,Hkv,D,padded_K]");
  const int64_t padded_kv_len = v8.size(3);
  ANEMOI_SM120_CHECK(
      padded_kv_len >= ((k16.size(2) + 127) / 128) * 128 &&
          padded_kv_len % 128 == 0,
      "v8 token dimension must cover K and be padded to 128");

  for (const auto& item : {
           std::pair<const TensorView*, const char*>(&q_scale, "q_scale"),
           std::pair<const TensorView*, const char*>(&k_scale, "k_scale"),
           std::pair<const TensorView*, const char*>(&v_scale, "v_scale")}) {
    ANEMOI_SM120_CHECK(
        item.first->scalar_type() == ScalarType::Float,
        item.second, " must be FP32");
  }
  const int64_t query_blocks = q16.size(2) / QueryBlock;
  const int64_t key_blocks = k16.size(2) / 64;
  ANEMOI_SM120_CHECK(
      q_scale.sizes() == Shape(
          {q16.size(0), q16.size(1), query_blocks}),
      "q_scale must have shape [B,Hq,Q/query_block]");
  ANEMOI_SM120_CHECK(
      k_scale.sizes() == Shape(
          {q16.size(0), k16.size(1), key_blocks}),
      "k_scale must have shape [B,Hkv,K/64]");
  ANEMOI_SM120_CHECK(
      v_scale.sizes() == Shape(
          {q16.size(0), k16.size(1), q16.size(3)}),
      "v_scale must have shape [B,Hkv,D]");

  ANEMOI_SM120_CHECK(
      block_ids.scalar_type() == ScalarType::Int &&
          block_ids.sizes() == Shape(
              {q16.size(0), q16.size(1), query_blocks, key_blocks}),
      "block_ids must have shape [B,Hq,Q/query_block,K/64]");
  for (const auto& item : {
           std::pair<const TensorView*, const char*>(
               &int8_block_counts, "int8_block_counts")}) {
    ANEMOI_SM120_CHECK(
        item.first->scalar_type() == ScalarType::Int &&
            item.first->sizes() == Shape(
                {q16.size(0), q16.size(1), query_blocks}),
        item.second, " must have shape [B,Hq,Q/query_block]");
  }
  ANEMOI_SM120_CHECK(
      fp16_block_counts.scalar_type() == ScalarType::Int &&
          (active_fp16
               ? fp16_block_counts.sizes() == Shape(
                     {q16.size(0), q16.size(1), query_blocks})
               : fp16_block_counts.numel() == 0),
      "active FP16 counts must match route rows; inactive FP16 requires "
      "an empty count tensor");
  ANEMOI_SM120_CHECK(
      valid_k_counts.scalar_type() == ScalarType::Int &&
          valid_k_counts.sizes() == Shape(
              {q16.size(0), key_blocks}),
      "valid_k_counts must have shape [B,K/64]");
  ANEMOI_SM120_CHECK(
      fp16_prefix_blocks >= 0 && fp16_prefix_blocks <= key_blocks,
      "fp16_prefix_blocks must be in [0,K/64]");
  ANEMOI_SM120_CHECK(
      active_fp16 || fp16_prefix_blocks == 0,
      "inactive FP16 requires zero prefix stages");

  check_output(output, q16, stream);

#define MPA_LAUNCH_INT8(launcher) \
  launcher( \
      q8.data_ptr<int8_t>(), k8.data_ptr<int8_t>(), \
      reinterpret_cast<__nv_fp8_e4m3*>(v8.data_ptr()), \
      reinterpret_cast<half*>(q16.data_ptr<half>()), \
      reinterpret_cast<half*>(k16.data_ptr<half>()), \
      reinterpret_cast<half*>(v16.data_ptr<half>()), nullptr, \
      reinterpret_cast<half*>(output.data_ptr<half>()), \
      block_ids.data_ptr<int32_t>(), int8_block_counts.data_ptr<int32_t>(), \
      block_ids.data_ptr<int32_t>(), fp16_block_counts.data_ptr<int32_t>(), \
      q_scale.data_ptr<float>(), k_scale.data_ptr<float>(), \
      v_scale.data_ptr<float>(), valid_k_counts.data_ptr<int32_t>(), \
      nullptr, static_cast<uint32_t>(fp16_prefix_blocks), \
      static_cast<uint32_t>(q16.size(0)), \
      static_cast<uint32_t>(q16.size(2)), \
      static_cast<uint32_t>(k16.size(2)), \
      static_cast<uint32_t>(padded_kv_len), \
      static_cast<uint32_t>(q16.size(1)), \
      static_cast<uint32_t>(k16.size(1)), \
      static_cast<float>(softmax_scale), stream)

  const auto launch = [&](auto head_dim_tag) {
    constexpr uint32_t HeadDim = decltype(head_dim_tag)::value;
    if constexpr (QueryBlock == 64) {
      auto launcher = active_fp16
          ? launch_mixed_attention_sm120_q64_int8_fp16<HeadDim, true, true, false>
          : launch_mixed_attention_sm120_q64_int8<HeadDim, true, false, false>;
      MPA_LAUNCH_INT8(launcher);
    } else {
      auto launcher = active_fp16
          ? launch_mixed_attention_sm120_q128_int8<HeadDim, true, true, false>
          : launch_mixed_attention_sm120_q128_int8<HeadDim, true, false, false>;
      MPA_LAUNCH_INT8(launcher);
    }
  };
  if (q16.size(3) == 64) {
    launch(std::integral_constant<uint32_t, 64>{});
  } else {
    launch(std::integral_constant<uint32_t, 128>{});
  }
#undef MPA_LAUNCH_INT8

  ANEMOI_SM120_CUDA_CHECK(cudaGetLastError());
  return;
}

void sm120_q64_int8_attention_forward(
    TensorView q8, TensorView k8, TensorView v8,
    TensorView q16, TensorView k16, TensorView v16,
    TensorView block_ids, TensorView int8_block_counts,
    TensorView fp16_block_counts, TensorView q_scale,
    TensorView k_scale, TensorView v_scale,
    TensorView valid_k_counts, int64_t fp16_prefix_blocks,
    double softmax_scale, bool active_fp16, TensorView output, cudaStream_t stream) {
  return int8_attention_forward<64>(
      q8, k8, v8, q16, k16, v16, block_ids, int8_block_counts,
      fp16_block_counts, q_scale, k_scale, v_scale, valid_k_counts,
      fp16_prefix_blocks, softmax_scale, active_fp16, output, stream);
}

void sm120_q128_int8_attention_forward(
    TensorView q8, TensorView k8, TensorView v8,
    TensorView q16, TensorView k16, TensorView v16,
    TensorView block_ids, TensorView int8_block_counts,
    TensorView fp16_block_counts, TensorView q_scale,
    TensorView k_scale, TensorView v_scale,
    TensorView valid_k_counts, int64_t fp16_prefix_blocks,
    double softmax_scale, bool active_fp16, TensorView output, cudaStream_t stream) {
  return int8_attention_forward<128>(
      q8, k8, v8, q16, k16, v16, block_ids, int8_block_counts,
      fp16_block_counts, q_scale, k_scale, v_scale, valid_k_counts,
      fp16_prefix_blocks, softmax_scale, active_fp16, output, stream);
}

template <uint32_t QueryBlock>
void prefix_int8_attention_forward(
    TensorView q8,
    TensorView k8,
    TensorView v8,
    TensorView q_scale,
    TensorView k_scale,
    TensorView v_scale,
    TensorView valid_k_counts,
    int64_t prefix_tokens,
    double softmax_scale, TensorView output, cudaStream_t stream) {
  static_assert(QueryBlock == 64 || QueryBlock == 128);
  for (const auto& item : {
           std::pair<const TensorView*, const char*>(&q8, "q8"),
           std::pair<const TensorView*, const char*>(&k8, "k8"),
           std::pair<const TensorView*, const char*>(&v8, "v8"),
           std::pair<const TensorView*, const char*>(&q_scale, "q_scale"),
           std::pair<const TensorView*, const char*>(&k_scale, "k_scale"),
           std::pair<const TensorView*, const char*>(&v_scale, "v_scale"),
           std::pair<const TensorView*, const char*>(
               &valid_k_counts, "valid_k_counts")}) {
    check_cuda_contiguous(*item.first, item.second);
    check_same_device(*item.first, q8, item.second);
  }
  ANEMOI_SM120_CHECK(
      q8.scalar_type() == ScalarType::Char &&
          k8.scalar_type() == ScalarType::Char,
      "q8/k8 must be int8");
  ANEMOI_SM120_CHECK(
      v8.scalar_type() == ScalarType::Float8_e4m3fn,
      "v8 must be float8_e4m3fn");
  ANEMOI_SM120_CHECK(
      q8.dim() == 4 && q8.size(0) > 0 && q8.size(1) > 0 &&
          q8.size(2) > 0 && q8.size(2) % QueryBlock == 0 &&
          (q8.size(3) == 64 || q8.size(3) == 128),
      "q8 must have shape [B,Hq,Q,D] with Q divisible by query block");
  ANEMOI_SM120_CHECK(
      k8.dim() == 4 && k8.size(0) == q8.size(0) && k8.size(1) > 0 &&
          k8.size(2) > 0 && k8.size(2) % 64 == 0 && k8.size(3) == q8.size(3) &&
          q8.size(1) % k8.size(1) == 0,
      "k8 must have compatible [B,Hkv,K,D] K64 layout");
  ANEMOI_SM120_CHECK(
      prefix_tokens > 0 && prefix_tokens <= q8.size(2),
      "prefix_tokens must be in (0,Q]");
  ANEMOI_SM120_CHECK(
      std::isfinite(softmax_scale) && softmax_scale > 0.0,
      "softmax_scale must be finite and positive");

  const int64_t query_blocks = q8.size(2) / QueryBlock;
  const int64_t key_blocks = k8.size(2) / 64;
  ANEMOI_SM120_CHECK(
      v8.dim() == 4 && v8.size(0) == q8.size(0) &&
          v8.size(1) == k8.size(1) && v8.size(2) == q8.size(3) &&
          v8.size(3) >= ((k8.size(2) + 127) / 128) * 128 &&
          v8.size(3) % 128 == 0,
      "v8 must have verified [B,Hkv,D,padded_K] layout");
  for (const auto& item : {
           std::pair<const TensorView*, const char*>(&q_scale, "q_scale"),
           std::pair<const TensorView*, const char*>(&k_scale, "k_scale"),
           std::pair<const TensorView*, const char*>(&v_scale, "v_scale")}) {
    ANEMOI_SM120_CHECK(
        item.first->scalar_type() == ScalarType::Float,
        item.second, " must be FP32");
  }
  ANEMOI_SM120_CHECK(
      q_scale.sizes() == Shape(
          {q8.size(0), q8.size(1), query_blocks}),
      "q_scale must have shape [B,Hq,Q/query_block]");
  ANEMOI_SM120_CHECK(
      k_scale.sizes() == Shape(
          {q8.size(0), k8.size(1), key_blocks}),
      "k_scale must have shape [B,Hkv,K/64]");
  ANEMOI_SM120_CHECK(
      v_scale.sizes() == Shape(
          {q8.size(0), k8.size(1), q8.size(3)}),
      "v_scale must have shape [B,Hkv,D]");
  ANEMOI_SM120_CHECK(
      valid_k_counts.scalar_type() == ScalarType::Int &&
          valid_k_counts.sizes() ==
              Shape({q8.size(0), key_blocks}),
      "valid_k_counts must have shape [B,K/64]");

  check_output(output, q8, stream);
  const auto launch = [&](auto head_dim_tag) {
    constexpr uint32_t HeadDim = decltype(head_dim_tag)::value;
    auto launcher = QueryBlock == 64
        ? launch_mixed_attention_sm120_q64_int8_dense<HeadDim, true, false, false>
        : launch_mixed_attention_sm120_q128_int8_dense<HeadDim, true, false, false>;
    launcher(
        q8.data_ptr<int8_t>(), k8.data_ptr<int8_t>(),
        reinterpret_cast<__nv_fp8_e4m3*>(v8.data_ptr()),
        nullptr, nullptr, nullptr, nullptr,
        reinterpret_cast<half*>(output.data_ptr<half>()),
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
  ANEMOI_SM120_CUDA_CHECK(cudaGetLastError());
  return; // Python slices the padded output to prefix_tokens.
}

void sm120_q64_prefix_int8_attention_forward(
    TensorView q8, TensorView k8, TensorView v8,
    TensorView q_scale, TensorView k_scale, TensorView v_scale,
    TensorView valid_k_counts, int64_t prefix_tokens,
    double softmax_scale, TensorView output, cudaStream_t stream) {
  return prefix_int8_attention_forward<64>(
      q8, k8, v8, q_scale, k_scale, v_scale, valid_k_counts,
      prefix_tokens, softmax_scale, output, stream);
}

void sm120_q128_prefix_int8_attention_forward(
    TensorView q8, TensorView k8, TensorView v8,
    TensorView q_scale, TensorView k_scale, TensorView v_scale,
    TensorView valid_k_counts, int64_t prefix_tokens,
    double softmax_scale, TensorView output, cudaStream_t stream) {
  return prefix_int8_attention_forward<128>(
      q8, k8, v8, q_scale, k_scale, v_scale, valid_k_counts,
      prefix_tokens, softmax_scale, output, stream);
}

template <uint32_t QueryBlock, bool CompactSequential = false>
void mxfp8_attention_forward(
    TensorView q_mxfp8,
    TensorView q_mxfp8_scale,
    TensorView k_mxfp8,
    TensorView k_mxfp8_scale,
    TensorView v_mxfp8,
    TensorView v_mxfp8_scale,
    TensorView q_fp16,
    TensorView k_fp16,
    TensorView v_fp16,
    TensorView block_ids,
    TensorView mxfp8_block_counts,
    TensorView fp16_block_counts,
    TensorView valid_k_counts,
    int64_t fp16_prefix_blocks,
    double softmax_scale,
    bool active_fp16, TensorView output, cudaStream_t stream) {
  static_assert(QueryBlock == 64 || QueryBlock == 128);
  for (const auto& item : {
           std::pair<const TensorView*, const char*>(&q_mxfp8, "q_mxfp8"),
           std::pair<const TensorView*, const char*>(
               &q_mxfp8_scale, "q_mxfp8_scale"),
           std::pair<const TensorView*, const char*>(&k_mxfp8, "k_mxfp8"),
           std::pair<const TensorView*, const char*>(
               &k_mxfp8_scale, "k_mxfp8_scale"),
           std::pair<const TensorView*, const char*>(&v_mxfp8, "v_mxfp8"),
           std::pair<const TensorView*, const char*>(
               &v_mxfp8_scale, "v_mxfp8_scale"),
           std::pair<const TensorView*, const char*>(&q_fp16, "q_fp16"),
           std::pair<const TensorView*, const char*>(&k_fp16, "k_fp16"),
           std::pair<const TensorView*, const char*>(&v_fp16, "v_fp16"),
           std::pair<const TensorView*, const char*>(&block_ids, "block_ids"),
           std::pair<const TensorView*, const char*>(
               &mxfp8_block_counts, "mxfp8_block_counts"),
           std::pair<const TensorView*, const char*>(
               &fp16_block_counts, "fp16_block_counts"),
           std::pair<const TensorView*, const char*>(
               &valid_k_counts, "valid_k_counts")}) {
    check_cuda_contiguous(*item.first, item.second);
    check_same_device(*item.first, q_fp16, item.second);
  }
  ANEMOI_SM120_CHECK(
      q_fp16.dim() == 4 && k_fp16.dim() == 4 && v_fp16.dim() == 4,
      "FP16 operands must have shape [B,H,S,D]");
  ANEMOI_SM120_CHECK(
      q_fp16.scalar_type() == ScalarType::Half &&
          k_fp16.scalar_type() == ScalarType::Half &&
          v_fp16.scalar_type() == ScalarType::Half,
      "FP16 operands must be FP16");
  ANEMOI_SM120_CHECK(k_fp16.sizes() == v_fp16.sizes(), "FP16 K/V shapes must match");
  ANEMOI_SM120_CHECK(
      q_fp16.size(0) > 0 && q_fp16.size(1) > 0 &&
          q_fp16.size(2) > 0 && q_fp16.size(2) % QueryBlock == 0,
      "MXFP8 Q length must be a positive multiple of the query block");
  ANEMOI_SM120_CHECK(
      k_fp16.size(0) == q_fp16.size(0) && k_fp16.size(2) > 0 &&
          k_fp16.size(2) % 64 == 0,
      "MXFP8 K length must be a positive multiple of 64 with matching batch");
  ANEMOI_SM120_CHECK(
      (q_fp16.size(3) == 64 || q_fp16.size(3) == 128) && k_fp16.size(3) == q_fp16.size(3) &&
          q_fp16.size(1) % k_fp16.size(1) == 0,
      "SM120 MXFP8 requires D64/D128 and divisible Q/KV heads");
  ANEMOI_SM120_CHECK(
      std::isfinite(softmax_scale) && softmax_scale > 0.0,
      "softmax_scale must be finite and positive");

  for (const auto& item : {
           std::pair<const TensorView*, const char*>(&q_mxfp8, "q_mxfp8"),
           std::pair<const TensorView*, const char*>(
               &q_mxfp8_scale, "q_mxfp8_scale"),
           std::pair<const TensorView*, const char*>(&k_mxfp8, "k_mxfp8"),
           std::pair<const TensorView*, const char*>(
               &k_mxfp8_scale, "k_mxfp8_scale"),
           std::pair<const TensorView*, const char*>(&v_mxfp8, "v_mxfp8"),
           std::pair<const TensorView*, const char*>(
               &v_mxfp8_scale, "v_mxfp8_scale")}) {
    ANEMOI_SM120_CHECK(
        item.first->scalar_type() == ScalarType::Byte,
        item.second, " must be uint8");
  }
  ANEMOI_SM120_CHECK(q_mxfp8.sizes() == q_fp16.sizes(), "q_mxfp8 must match q_fp16");
  ANEMOI_SM120_CHECK(k_mxfp8.sizes() == k_fp16.sizes(), "k_mxfp8 must match k_fp16");
  ANEMOI_SM120_CHECK(
      q_mxfp8_scale.sizes() == Shape(
          {q_fp16.size(0), q_fp16.size(1), q_fp16.size(2), q_fp16.size(3) / 32}),
      "q_mxfp8_scale must have shape [B,Hq,Q,D/32]");
  ANEMOI_SM120_CHECK(
      k_mxfp8_scale.sizes() == Shape(
          {k_fp16.size(0), k_fp16.size(1), k_fp16.size(2), q_fp16.size(3) / 32}),
      "k_mxfp8_scale must have shape [B,Hkv,K,D/32]");
  ANEMOI_SM120_CHECK(
      v_mxfp8.dim() == 4 && v_mxfp8.size(0) == k_fp16.size(0) &&
          v_mxfp8.size(1) == k_fp16.size(1) &&
          v_mxfp8.size(2) == q_fp16.size(3) &&
          v_mxfp8.size(3) >= k_fp16.size(2) &&
          v_mxfp8.size(3) % 64 == 0,
      "v_mxfp8 must have shape [B,Hkv,D,padded_K]");
  const int64_t query_blocks = q_fp16.size(2) / QueryBlock;
  const int64_t key_blocks = k_fp16.size(2) / 64;
  ANEMOI_SM120_CHECK(
      v_mxfp8_scale.sizes() == Shape(
          {k_fp16.size(0), k_fp16.size(1), key_blocks, q_fp16.size(3) * 2}),
      "v_mxfp8_scale must have K64 consumer shape [B,Hkv,K/64,2*D]");

  ANEMOI_SM120_CHECK(block_ids.scalar_type() == ScalarType::Int,
              "block_ids must be int32");
  ANEMOI_SM120_CHECK(mxfp8_block_counts.scalar_type() == ScalarType::Int,
              "mxfp8_block_counts must be int32");
  ANEMOI_SM120_CHECK(fp16_block_counts.scalar_type() == ScalarType::Int,
              "fp16_block_counts must be int32");
  ANEMOI_SM120_CHECK(valid_k_counts.scalar_type() == ScalarType::Int,
              "valid_k_counts must be int32");
  ANEMOI_SM120_CHECK(
      block_ids.sizes() == Shape(
          {q_fp16.size(0), q_fp16.size(1), query_blocks, key_blocks}),
      "block_ids must have shape [B,Hq,Q/query_block,K/64]");
  ANEMOI_SM120_CHECK(
      mxfp8_block_counts.sizes() == Shape(
          {q_fp16.size(0), q_fp16.size(1), query_blocks}),
              "mxfp8_block_counts must have shape [B,Hq,Q/query_block]");
  ANEMOI_SM120_CHECK(
      active_fp16
          ? fp16_block_counts.sizes() == Shape(
                {q_fp16.size(0), q_fp16.size(1), query_blocks})
          : fp16_block_counts.numel() == 0,
      "active FP16 counts must match route rows; inactive FP16 requires "
      "an empty count tensor");
  ANEMOI_SM120_CHECK(
      valid_k_counts.sizes() == Shape(
          {q_fp16.size(0), key_blocks}),
      "valid_k_counts must have shape [B,K/64]");
  ANEMOI_SM120_CHECK(
      fp16_prefix_blocks >= 0 && fp16_prefix_blocks <= key_blocks,
      "fp16_prefix_blocks must be in [0,K/64]");
  ANEMOI_SM120_CHECK(
      active_fp16 || fp16_prefix_blocks == 0,
      "inactive FP16 requires zero prefix stages");
  check_output(output, q_fp16, stream);

  const auto launch = [&](auto head_dim_tag) {
    constexpr uint32_t HeadDim = decltype(head_dim_tag)::value;
    if constexpr (CompactSequential) {
      static_assert(QueryBlock == 128);
      ANEMOI_SM120_CHECK(!active_fp16, "compact MXFP8 is a pure compute ceiling");
      launch_mixed_attention_sm120_q128_mxfp8_compact<
          HeadDim, true, false, false>(
          reinterpret_cast<int8_t*>(q_mxfp8.data_ptr<uint8_t>()),
          reinterpret_cast<int8_t*>(k_mxfp8.data_ptr<uint8_t>()),
          reinterpret_cast<__nv_fp8_e4m3*>(v_mxfp8.data_ptr<uint8_t>()),
          reinterpret_cast<half*>(q_fp16.data_ptr<half>()),
          reinterpret_cast<half*>(k_fp16.data_ptr<half>()),
          reinterpret_cast<half*>(v_fp16.data_ptr<half>()), nullptr,
          reinterpret_cast<half*>(output.data_ptr<half>()),
          block_ids.data_ptr<int32_t>(),
          mxfp8_block_counts.data_ptr<int32_t>(),
          block_ids.data_ptr<int32_t>(), fp16_block_counts.data_ptr<int32_t>(),
          q_mxfp8_scale.data_ptr<uint8_t>(),
          k_mxfp8_scale.data_ptr<uint8_t>(),
          v_mxfp8_scale.data_ptr<uint8_t>(),
          valid_k_counts.data_ptr<int32_t>(), nullptr, 0,
          static_cast<uint32_t>(q_fp16.size(0)),
          static_cast<uint32_t>(q_fp16.size(2)),
          static_cast<uint32_t>(k_fp16.size(2)),
          static_cast<uint32_t>(v_mxfp8.size(3)),
          static_cast<uint32_t>(q_fp16.size(1)),
          static_cast<uint32_t>(k_fp16.size(1)),
          static_cast<float>(softmax_scale), stream);
    } else if constexpr (QueryBlock == 64) {
      auto launcher = active_fp16
          ? launch_mixed_attention_sm120_q64<HeadDim, true, true, false>
          : launch_mixed_attention_sm120_q64<HeadDim, true, false, false>;
      launcher(
          reinterpret_cast<int8_t*>(q_mxfp8.data_ptr<uint8_t>()),
          reinterpret_cast<int8_t*>(k_mxfp8.data_ptr<uint8_t>()),
          reinterpret_cast<__nv_fp8_e4m3*>(v_mxfp8.data_ptr<uint8_t>()),
          reinterpret_cast<half*>(q_fp16.data_ptr<half>()),
          reinterpret_cast<half*>(k_fp16.data_ptr<half>()),
          reinterpret_cast<half*>(v_fp16.data_ptr<half>()), nullptr,
          reinterpret_cast<half*>(output.data_ptr<half>()),
          block_ids.data_ptr<int32_t>(),
          mxfp8_block_counts.data_ptr<int32_t>(),
          block_ids.data_ptr<int32_t>(),
          fp16_block_counts.data_ptr<int32_t>(),
          q_mxfp8_scale.data_ptr<uint8_t>(),
          k_mxfp8_scale.data_ptr<uint8_t>(),
          v_mxfp8_scale.data_ptr<uint8_t>(),
          valid_k_counts.data_ptr<int32_t>(), nullptr,
          static_cast<uint32_t>(fp16_prefix_blocks),
          static_cast<uint32_t>(q_fp16.size(0)),
          static_cast<uint32_t>(q_fp16.size(2)),
          static_cast<uint32_t>(k_fp16.size(2)),
          static_cast<uint32_t>(v_mxfp8.size(3)),
          static_cast<uint32_t>(q_fp16.size(1)),
          static_cast<uint32_t>(k_fp16.size(1)),
          static_cast<float>(softmax_scale), stream);
    } else {
      auto launcher = active_fp16
          ? launch_mixed_attention_sm120_q128_mxfp8<HeadDim, true, true, false>
          : launch_mixed_attention_sm120_q128_mxfp8<HeadDim, true, false, false>;
      launcher(
        reinterpret_cast<int8_t*>(q_mxfp8.data_ptr<uint8_t>()),
        reinterpret_cast<int8_t*>(k_mxfp8.data_ptr<uint8_t>()),
        reinterpret_cast<__nv_fp8_e4m3*>(v_mxfp8.data_ptr<uint8_t>()),
        reinterpret_cast<half*>(q_fp16.data_ptr<half>()),
        reinterpret_cast<half*>(k_fp16.data_ptr<half>()),
        reinterpret_cast<half*>(v_fp16.data_ptr<half>()), nullptr,
        reinterpret_cast<half*>(output.data_ptr<half>()),
        block_ids.data_ptr<int32_t>(),
        mxfp8_block_counts.data_ptr<int32_t>(),
        block_ids.data_ptr<int32_t>(),
        fp16_block_counts.data_ptr<int32_t>(),
        q_mxfp8_scale.data_ptr<uint8_t>(),
        k_mxfp8_scale.data_ptr<uint8_t>(),
        v_mxfp8_scale.data_ptr<uint8_t>(),
        valid_k_counts.data_ptr<int32_t>(), nullptr,
        static_cast<uint32_t>(fp16_prefix_blocks),
        static_cast<uint32_t>(q_fp16.size(0)),
        static_cast<uint32_t>(q_fp16.size(2)),
        static_cast<uint32_t>(k_fp16.size(2)),
        static_cast<uint32_t>(v_mxfp8.size(3)),
        static_cast<uint32_t>(q_fp16.size(1)),
        static_cast<uint32_t>(k_fp16.size(1)),
        static_cast<float>(softmax_scale), stream);
    }
  };
  if (q_fp16.size(3) == 64) {
    launch(std::integral_constant<uint32_t, 64>{});
  } else {
    launch(std::integral_constant<uint32_t, 128>{});
  }
  ANEMOI_SM120_CUDA_CHECK(cudaGetLastError());
  return;
}

void sm120_q64_mxfp8_attention_forward(
    TensorView q_mxfp8, TensorView q_mxfp8_scale,
    TensorView k_mxfp8, TensorView k_mxfp8_scale,
    TensorView v_mxfp8, TensorView v_mxfp8_scale,
    TensorView q_fp16, TensorView k_fp16, TensorView v_fp16,
    TensorView block_ids, TensorView mxfp8_block_counts,
    TensorView fp16_block_counts, TensorView valid_k_counts,
    int64_t fp16_prefix_blocks, double softmax_scale, bool active_fp16, TensorView output, cudaStream_t stream) {
  return mxfp8_attention_forward<64>(
      q_mxfp8, q_mxfp8_scale, k_mxfp8, k_mxfp8_scale, v_mxfp8,
      v_mxfp8_scale, q_fp16, k_fp16, v_fp16, block_ids,
      mxfp8_block_counts, fp16_block_counts, valid_k_counts,
      fp16_prefix_blocks, softmax_scale, active_fp16, output, stream);
}

void sm120_q128_mxfp8_attention_forward(
    TensorView q_mxfp8, TensorView q_mxfp8_scale,
    TensorView k_mxfp8, TensorView k_mxfp8_scale,
    TensorView v_mxfp8, TensorView v_mxfp8_scale,
    TensorView q_fp16, TensorView k_fp16, TensorView v_fp16,
    TensorView block_ids, TensorView mxfp8_block_counts,
    TensorView fp16_block_counts, TensorView valid_k_counts,
    int64_t fp16_prefix_blocks, double softmax_scale, bool active_fp16, TensorView output, cudaStream_t stream) {
  return mxfp8_attention_forward<128>(
      q_mxfp8, q_mxfp8_scale, k_mxfp8, k_mxfp8_scale, v_mxfp8,
      v_mxfp8_scale, q_fp16, k_fp16, v_fp16, block_ids,
      mxfp8_block_counts, fp16_block_counts, valid_k_counts,
      fp16_prefix_blocks, softmax_scale, active_fp16, output, stream);
}

void
sm120_q128_mxfp8_compact_attention_forward(
    TensorView q_mxfp8, TensorView q_mxfp8_scale,
    TensorView k_mxfp8, TensorView k_mxfp8_scale,
    TensorView v_mxfp8, TensorView v_mxfp8_scale,
    TensorView q_fp16, TensorView k_fp16, TensorView v_fp16,
    TensorView block_ids, TensorView mxfp8_block_counts,
    TensorView fp16_block_counts, TensorView valid_k_counts,
    int64_t fp16_prefix_blocks, double softmax_scale, bool active_fp16, TensorView output, cudaStream_t stream) {
  return mxfp8_attention_forward<128, true>(
      q_mxfp8, q_mxfp8_scale, k_mxfp8, k_mxfp8_scale, v_mxfp8,
      v_mxfp8_scale, q_fp16, k_fp16, v_fp16, block_ids,
      mxfp8_block_counts, fp16_block_counts, valid_k_counts,
      fp16_prefix_blocks, softmax_scale, active_fp16, output, stream);
}

template <uint32_t QueryBlock>
void nvfp4_attention_forward(
    TensorView q_nvfp4, TensorView q_nvfp4_scale,
    TensorView k_nvfp4, TensorView k_nvfp4_scale,
    TensorView v_nvfp4, TensorView v_nvfp4_scale,
    TensorView q_fp16, TensorView k_fp16, TensorView v_fp16,
    TensorView block_ids, TensorView nvfp4_block_counts,
    TensorView fp16_block_counts, TensorView valid_k_counts,
    TensorView q_global_scale, TensorView k_global_scale,
    TensorView v_global_scale, int64_t fp16_prefix_blocks,
    double softmax_scale, bool active_fp16, TensorView output, cudaStream_t stream) {
  static_assert(QueryBlock == 64 || QueryBlock == 128);
  for (const auto& item : {
           std::pair<const TensorView*, const char*>(&q_nvfp4, "q_nvfp4"),
           std::pair<const TensorView*, const char*>(&q_nvfp4_scale, "q_nvfp4_scale"),
           std::pair<const TensorView*, const char*>(&k_nvfp4, "k_nvfp4"),
           std::pair<const TensorView*, const char*>(&k_nvfp4_scale, "k_nvfp4_scale"),
           std::pair<const TensorView*, const char*>(&v_nvfp4, "v_nvfp4"),
           std::pair<const TensorView*, const char*>(&v_nvfp4_scale, "v_nvfp4_scale"),
           std::pair<const TensorView*, const char*>(&q_fp16, "q_fp16"),
           std::pair<const TensorView*, const char*>(&k_fp16, "k_fp16"),
           std::pair<const TensorView*, const char*>(&v_fp16, "v_fp16"),
           std::pair<const TensorView*, const char*>(&block_ids, "block_ids"),
           std::pair<const TensorView*, const char*>(&nvfp4_block_counts, "nvfp4_block_counts"),
           std::pair<const TensorView*, const char*>(&fp16_block_counts, "fp16_block_counts"),
           std::pair<const TensorView*, const char*>(&valid_k_counts, "valid_k_counts"),
           std::pair<const TensorView*, const char*>(&q_global_scale, "q_global_scale"),
           std::pair<const TensorView*, const char*>(&k_global_scale, "k_global_scale"),
           std::pair<const TensorView*, const char*>(&v_global_scale, "v_global_scale")}) {
    check_cuda_contiguous(*item.first, item.second);
    check_same_device(*item.first, q_fp16, item.second);
  }
  ANEMOI_SM120_CHECK(
      q_fp16.scalar_type() == ScalarType::Half &&
          k_fp16.scalar_type() == ScalarType::Half &&
          v_fp16.scalar_type() == ScalarType::Half,
      "FP16 operands must be FP16");
  ANEMOI_SM120_CHECK(
      q_fp16.dim() == 4 && (q_fp16.size(3) == 64 || q_fp16.size(3) == 128) &&
          q_fp16.size(2) > 0 && q_fp16.size(2) % QueryBlock == 0,
      "Q must have shape [B,Hq,Q,D] with Q divisible by query block");
  ANEMOI_SM120_CHECK(
      k_fp16.dim() == 4 && k_fp16.sizes() == v_fp16.sizes() &&
          k_fp16.size(0) == q_fp16.size(0) &&
          k_fp16.size(2) > 0 && k_fp16.size(2) % 64 == 0 &&
          k_fp16.size(3) == q_fp16.size(3) && q_fp16.size(1) % k_fp16.size(1) == 0,
      "K/V must have matching [B,Hkv,K,D] shapes with K divisible by 64");
  for (const auto& item : {
           std::pair<const TensorView*, const char*>(&q_nvfp4, "q_nvfp4"),
           std::pair<const TensorView*, const char*>(&q_nvfp4_scale, "q_nvfp4_scale"),
           std::pair<const TensorView*, const char*>(&k_nvfp4, "k_nvfp4"),
           std::pair<const TensorView*, const char*>(&k_nvfp4_scale, "k_nvfp4_scale"),
           std::pair<const TensorView*, const char*>(&v_nvfp4, "v_nvfp4"),
           std::pair<const TensorView*, const char*>(&v_nvfp4_scale, "v_nvfp4_scale")}) {
    ANEMOI_SM120_CHECK(item.first->scalar_type() == ScalarType::Byte,
                item.second, " must be uint8");
  }
  ANEMOI_SM120_CHECK(
      q_nvfp4.sizes() == Shape(
          {q_fp16.size(0), q_fp16.size(1), q_fp16.size(2), q_fp16.size(3) / 2}) &&
          q_nvfp4_scale.sizes() == Shape(
              {q_fp16.size(0), q_fp16.size(1), q_fp16.size(2), q_fp16.size(3) / 16}),
      "invalid NVFP4 Q shapes");
  ANEMOI_SM120_CHECK(
      k_nvfp4.sizes() == Shape(
          {k_fp16.size(0), k_fp16.size(1), k_fp16.size(2), q_fp16.size(3) / 2}) &&
          k_nvfp4_scale.sizes() == Shape(
              {k_fp16.size(0), k_fp16.size(1), k_fp16.size(2), q_fp16.size(3) / 16}),
      "invalid NVFP4 K shapes");
  const int64_t query_blocks = q_fp16.size(2) / QueryBlock;
  const int64_t key_blocks = k_fp16.size(2) / 64;
  ANEMOI_SM120_CHECK(
      v_nvfp4.sizes() == Shape(
          {v_fp16.size(0), v_fp16.size(1), q_fp16.size(3), v_fp16.size(2) / 2}) &&
          v_nvfp4_scale.sizes() == Shape(
              {v_fp16.size(0), v_fp16.size(1), key_blocks, q_fp16.size(3) * 4}),
      "invalid NVFP4 V consumer shapes");
  ANEMOI_SM120_CHECK(
      block_ids.scalar_type() == ScalarType::Int &&
          nvfp4_block_counts.scalar_type() == ScalarType::Int &&
          fp16_block_counts.scalar_type() == ScalarType::Int &&
          valid_k_counts.scalar_type() == ScalarType::Int,
      "route metadata must be int32");
  ANEMOI_SM120_CHECK(
      block_ids.sizes() == Shape(
          {q_fp16.size(0), q_fp16.size(1), query_blocks, key_blocks}) &&
          nvfp4_block_counts.sizes() == Shape(
              {q_fp16.size(0), q_fp16.size(1), query_blocks}) &&
          (active_fp16
               ? fp16_block_counts.sizes() == Shape(
                     {q_fp16.size(0), q_fp16.size(1), query_blocks})
               : fp16_block_counts.numel() == 0) &&
          valid_k_counts.sizes() == Shape(
              {q_fp16.size(0), key_blocks}),
      "invalid route metadata shapes");
  for (const auto& item : {
           std::pair<const TensorView*, const char*>(&q_global_scale, "q_global_scale"),
           std::pair<const TensorView*, const char*>(&k_global_scale, "k_global_scale"),
           std::pair<const TensorView*, const char*>(&v_global_scale, "v_global_scale")}) {
    ANEMOI_SM120_CHECK(
        item.first->scalar_type() == ScalarType::Float &&
            item.first->numel() == 1,
        item.second, " must be one FP32 scalar");
  }
  ANEMOI_SM120_CHECK(
      fp16_prefix_blocks >= 0 && fp16_prefix_blocks <= key_blocks,
      "fp16_prefix_blocks must be in [0,K/64]");
  ANEMOI_SM120_CHECK(
      active_fp16 || fp16_prefix_blocks == 0,
      "inactive FP16 requires zero prefix stages");
  ANEMOI_SM120_CHECK(std::isfinite(softmax_scale) && softmax_scale > 0.0,
              "softmax_scale must be finite and positive");

  check_output(output, q_fp16, stream);
  auto launch_nvfp4 = [&](auto launcher) {
    launcher(
      reinterpret_cast<int8_t*>(q_nvfp4.data_ptr<uint8_t>()),
      reinterpret_cast<int8_t*>(k_nvfp4.data_ptr<uint8_t>()),
      reinterpret_cast<__nv_fp8_e4m3*>(v_nvfp4.data_ptr<uint8_t>()),
      reinterpret_cast<half*>(q_fp16.data_ptr<half>()),
      reinterpret_cast<half*>(k_fp16.data_ptr<half>()),
      reinterpret_cast<half*>(v_fp16.data_ptr<half>()), nullptr,
      reinterpret_cast<half*>(output.data_ptr<half>()),
      block_ids.data_ptr<int32_t>(), nvfp4_block_counts.data_ptr<int32_t>(),
      block_ids.data_ptr<int32_t>(), fp16_block_counts.data_ptr<int32_t>(),
      q_nvfp4_scale.data_ptr<uint8_t>(),
      k_nvfp4_scale.data_ptr<uint8_t>(),
      v_nvfp4_scale.data_ptr<uint8_t>(),
      q_global_scale.data_ptr<float>(), k_global_scale.data_ptr<float>(),
      v_global_scale.data_ptr<float>(), valid_k_counts.data_ptr<int32_t>(),
      nullptr, static_cast<uint32_t>(fp16_prefix_blocks),
      static_cast<uint32_t>(q_fp16.size(0)),
      static_cast<uint32_t>(q_fp16.size(2)),
      static_cast<uint32_t>(k_fp16.size(2)),
      static_cast<uint32_t>(k_fp16.size(2)),
      static_cast<uint32_t>(q_fp16.size(1)),
      static_cast<uint32_t>(k_fp16.size(1)),
      static_cast<float>(softmax_scale), stream);
  };
  const auto launch = [&](auto head_dim_tag) {
    constexpr uint32_t HeadDim = decltype(head_dim_tag)::value;
    if constexpr (QueryBlock == 64) {
      auto launcher = active_fp16
          ? launch_mixed_attention_sm120_q64_nvfp4<HeadDim, true, true, false>
          : launch_mixed_attention_sm120_q64_nvfp4<HeadDim, true, false, false>;
      launch_nvfp4(launcher);
    } else {
      auto launcher = active_fp16
          ? launch_mixed_attention_sm120_q128_nvfp4<HeadDim, true, true, false>
          : launch_mixed_attention_sm120_q128_nvfp4<HeadDim, true, false, false>;
      launch_nvfp4(launcher);
    }
  };
  if (q_fp16.size(3) == 64) {
    launch(std::integral_constant<uint32_t, 64>{});
  } else {
    launch(std::integral_constant<uint32_t, 128>{});
  }
  ANEMOI_SM120_CUDA_CHECK(cudaGetLastError());
  return;
}


void sm120_q64_nvfp4_attention_forward(
    TensorView q_nvfp4, TensorView q_nvfp4_scale,
    TensorView k_nvfp4, TensorView k_nvfp4_scale,
    TensorView v_nvfp4, TensorView v_nvfp4_scale,
    TensorView q_fp16, TensorView k_fp16, TensorView v_fp16,
    TensorView block_ids, TensorView nvfp4_block_counts,
    TensorView fp16_block_counts, TensorView valid_k_counts,
    TensorView q_global_scale, TensorView k_global_scale,
    TensorView v_global_scale, int64_t fp16_prefix_blocks,
    double softmax_scale, bool active_fp16, TensorView output, cudaStream_t stream) {
  return nvfp4_attention_forward<64>(
      q_nvfp4, q_nvfp4_scale, k_nvfp4, k_nvfp4_scale, v_nvfp4,
      v_nvfp4_scale, q_fp16, k_fp16, v_fp16, block_ids,
      nvfp4_block_counts, fp16_block_counts, valid_k_counts,
      q_global_scale, k_global_scale, v_global_scale, fp16_prefix_blocks,
      softmax_scale, active_fp16, output, stream); \
}
void sm120_q128_nvfp4_attention_forward(
    TensorView q_nvfp4, TensorView q_nvfp4_scale,
    TensorView k_nvfp4, TensorView k_nvfp4_scale,
    TensorView v_nvfp4, TensorView v_nvfp4_scale,
    TensorView q_fp16, TensorView k_fp16, TensorView v_fp16,
    TensorView block_ids, TensorView nvfp4_block_counts,
    TensorView fp16_block_counts, TensorView valid_k_counts,
    TensorView q_global_scale, TensorView k_global_scale,
    TensorView v_global_scale, int64_t fp16_prefix_blocks,
    double softmax_scale, bool active_fp16, TensorView output, cudaStream_t stream) {
  return nvfp4_attention_forward<128>(
      q_nvfp4, q_nvfp4_scale, k_nvfp4, k_nvfp4_scale, v_nvfp4,
      v_nvfp4_scale, q_fp16, k_fp16, v_fp16, block_ids,
      nvfp4_block_counts, fp16_block_counts, valid_k_counts,
      q_global_scale, k_global_scale, v_global_scale, fp16_prefix_blocks,
      softmax_scale, active_fp16, output, stream); \
}


template <uint32_t QueryBlock, bool MiddleInt8>
void three_phase_forward(
    TensorView q4, TensorView q4_scale,
    TensorView k4, TensorView k4_scale,
    TensorView v4, TensorView v4_scale,
    TensorView q8, TensorView q8_scale,
    TensorView k8, TensorView k8_scale,
    TensorView v8, TensorView v8_scale,
    TensorView q16, TensorView k16, TensorView v16,
    TensorView block_ids, TensorView nvfp4_counts,
    TensorView middle_counts, TensorView fp16_counts,
    TensorView valid_k_counts, TensorView q_global_scale,
    TensorView k_global_scale, TensorView v_global_scale,
    int64_t fp16_prefix_blocks, double softmax_scale, bool active_fp16, TensorView output, cudaStream_t stream) {
  static_assert(QueryBlock == 64 || QueryBlock == 128);
  for (const auto& item : {
           std::pair<const TensorView*, const char*>(&q4, "q4"),
           std::pair<const TensorView*, const char*>(&q4_scale, "q4_scale"),
           std::pair<const TensorView*, const char*>(&k4, "k4"),
           std::pair<const TensorView*, const char*>(&k4_scale, "k4_scale"),
           std::pair<const TensorView*, const char*>(&v4, "v4"),
           std::pair<const TensorView*, const char*>(&v4_scale, "v4_scale"),
           std::pair<const TensorView*, const char*>(&q8, "q8"),
           std::pair<const TensorView*, const char*>(&q8_scale, "q8_scale"),
           std::pair<const TensorView*, const char*>(&k8, "k8"),
           std::pair<const TensorView*, const char*>(&k8_scale, "k8_scale"),
           std::pair<const TensorView*, const char*>(&v8, "v8"),
           std::pair<const TensorView*, const char*>(&v8_scale, "v8_scale"),
           std::pair<const TensorView*, const char*>(&q16, "q16"),
           std::pair<const TensorView*, const char*>(&k16, "k16"),
           std::pair<const TensorView*, const char*>(&v16, "v16"),
           std::pair<const TensorView*, const char*>(&block_ids, "block_ids"),
           std::pair<const TensorView*, const char*>(&nvfp4_counts, "nvfp4_counts"),
           std::pair<const TensorView*, const char*>(&middle_counts, "middle_counts"),
           std::pair<const TensorView*, const char*>(&fp16_counts, "fp16_counts"),
           std::pair<const TensorView*, const char*>(&valid_k_counts, "valid_k_counts"),
           std::pair<const TensorView*, const char*>(&q_global_scale, "q_global_scale"),
           std::pair<const TensorView*, const char*>(&k_global_scale, "k_global_scale"),
           std::pair<const TensorView*, const char*>(&v_global_scale, "v_global_scale")}) {
    check_cuda_contiguous(*item.first, item.second);
    check_same_device(*item.first, q16, item.second);
  }
  ANEMOI_SM120_CHECK(
      q16.scalar_type() == ScalarType::Half &&
          k16.scalar_type() == ScalarType::Half &&
          v16.scalar_type() == ScalarType::Half &&
          q16.dim() == 4 && k16.dim() == 4 && k16.sizes() == v16.sizes() &&
          q16.size(0) == k16.size(0) && q16.size(2) > 0 &&
          q16.size(2) % QueryBlock == 0 && k16.size(2) > 0 &&
          k16.size(2) % 64 == 0 && (q16.size(3) == 64 || q16.size(3) == 128) &&
          k16.size(3) == q16.size(3) && q16.size(1) % k16.size(1) == 0,
      "FP16 operands must be compatible [B,H,Q/K,D] tensors");
  const int64_t query_blocks = q16.size(2) / QueryBlock;
  const int64_t key_blocks = k16.size(2) / 64;
  const std::array<int64_t, 3> count_dims = {
      q16.size(0), q16.size(1), query_blocks};
  const Shape count_shape(count_dims);
  ANEMOI_SM120_CHECK(
      block_ids.scalar_type() == ScalarType::Int &&
          block_ids.sizes() == Shape(
              {q16.size(0), q16.size(1), query_blocks, key_blocks}) &&
          nvfp4_counts.scalar_type() == ScalarType::Int &&
          nvfp4_counts.sizes() == count_shape &&
          middle_counts.scalar_type() == ScalarType::Int &&
          middle_counts.sizes() == count_shape &&
          fp16_counts.scalar_type() == ScalarType::Int &&
          (active_fp16
               ? fp16_counts.sizes() == count_shape
               : fp16_counts.numel() == 0) &&
          valid_k_counts.scalar_type() == ScalarType::Int &&
          valid_k_counts.sizes() == Shape(
              {q16.size(0), key_blocks}),
      "invalid compact three-phase route metadata");
  ANEMOI_SM120_CHECK(
      q4.scalar_type() == ScalarType::Byte &&
          k4.scalar_type() == ScalarType::Byte &&
          v4.scalar_type() == ScalarType::Byte &&
          q4_scale.scalar_type() == ScalarType::Byte &&
          k4_scale.scalar_type() == ScalarType::Byte &&
          v4_scale.scalar_type() == ScalarType::Byte &&
          q4.sizes() == Shape(
              {q16.size(0), q16.size(1), q16.size(2), q16.size(3) / 2}) &&
          q4_scale.sizes() == Shape(
              {q16.size(0), q16.size(1), q16.size(2), q16.size(3) / 16}) &&
          k4.sizes() == Shape(
              {k16.size(0), k16.size(1), k16.size(2), q16.size(3) / 2}) &&
          k4_scale.sizes() == Shape(
              {k16.size(0), k16.size(1), k16.size(2), q16.size(3) / 16}) &&
          v4.sizes() == Shape(
              {v16.size(0), v16.size(1), q16.size(3), v16.size(2) / 2}) &&
          v4_scale.sizes() == Shape(
              {v16.size(0), v16.size(1), key_blocks, q16.size(3) * 4}),
      "invalid NVFP4 operand shapes or dtypes");
  for (const auto* scale : {&q_global_scale, &k_global_scale, &v_global_scale}) {
    ANEMOI_SM120_CHECK(
        scale->scalar_type() == ScalarType::Float && scale->numel() == 1,
        "NVFP4 tensor-global scales must be FP32 scalars");
  }
  if constexpr (MiddleInt8) {
    ANEMOI_SM120_CHECK(
        q8.scalar_type() == ScalarType::Char &&
            k8.scalar_type() == ScalarType::Char &&
            v8.scalar_type() == ScalarType::Float8_e4m3fn &&
            q8.sizes() == q16.sizes() && k8.sizes() == k16.sizes() &&
            q8_scale.scalar_type() == ScalarType::Float &&
            q8_scale.sizes() == count_shape &&
            k8_scale.scalar_type() == ScalarType::Float &&
            k8_scale.sizes() == Shape(
                {q16.size(0), k16.size(1), key_blocks}) &&
            v8_scale.scalar_type() == ScalarType::Float &&
            v8_scale.sizes() == Shape(
                {q16.size(0), k16.size(1), q16.size(3)}),
        "invalid INT8/E4M3 middle-phase operands");
  } else {
    ANEMOI_SM120_CHECK(
        q8.scalar_type() == ScalarType::Byte &&
            k8.scalar_type() == ScalarType::Byte &&
            v8.scalar_type() == ScalarType::Byte &&
            q8.sizes() == q16.sizes() && k8.sizes() == k16.sizes() &&
            q8_scale.scalar_type() == ScalarType::Byte &&
            q8_scale.sizes() == Shape(
                {q16.size(0), q16.size(1), q16.size(2), q16.size(3) / 32}) &&
            k8_scale.scalar_type() == ScalarType::Byte &&
            k8_scale.sizes() == Shape(
                {k16.size(0), k16.size(1), k16.size(2), q16.size(3) / 32}) &&
            v8_scale.scalar_type() == ScalarType::Byte &&
            v8_scale.sizes() == Shape(
                {k16.size(0), k16.size(1), key_blocks, q16.size(3) * 2}),
        "invalid MXFP8 middle-phase operands");
  }
  ANEMOI_SM120_CHECK(
      v8.dim() == 4 && v8.size(0) == q16.size(0) &&
          v8.size(1) == k16.size(1) && v8.size(2) == q16.size(3) &&
          v8.size(3) >= k16.size(2) && v8.size(3) % 64 == 0,
      "middle V must have shape [B,Hkv,D,padded_K]");
  ANEMOI_SM120_CHECK(
      fp16_prefix_blocks >= 0 && fp16_prefix_blocks <= key_blocks &&
          std::isfinite(softmax_scale) && softmax_scale > 0.0,
      "invalid prefix count or softmax scale");
  ANEMOI_SM120_CHECK(
      active_fp16 || fp16_prefix_blocks == 0,
      "inactive FP16 requires zero prefix stages");
  check_output(output, q16, stream);
#define MPA_STACK_COMMON_ARGS \
      reinterpret_cast<half*>(q16.data_ptr<half>()), \
      reinterpret_cast<half*>(k16.data_ptr<half>()), \
      reinterpret_cast<half*>(v16.data_ptr<half>()), nullptr, \
      reinterpret_cast<half*>(output.data_ptr<half>()), \
      block_ids.data_ptr<int32_t>(), nvfp4_counts.data_ptr<int32_t>(), \
      block_ids.data_ptr<int32_t>(), fp16_counts.data_ptr<int32_t>()
#define MPA_STACK_NV_ARGS \
      reinterpret_cast<int8_t*>(q4.data_ptr<uint8_t>()), \
      reinterpret_cast<int8_t*>(k4.data_ptr<uint8_t>()), \
      reinterpret_cast<__nv_fp8_e4m3*>(v4.data_ptr<uint8_t>()), \
      q4_scale.data_ptr<uint8_t>(), k4_scale.data_ptr<uint8_t>(), \
      v4_scale.data_ptr<uint8_t>(), middle_counts.data_ptr<int32_t>(), \
      q_global_scale.data_ptr<float>(), k_global_scale.data_ptr<float>(), \
      v_global_scale.data_ptr<float>(), valid_k_counts.data_ptr<int32_t>(), \
      nullptr, static_cast<uint32_t>(fp16_prefix_blocks), \
      static_cast<uint32_t>(q16.size(0)), static_cast<uint32_t>(q16.size(2)), \
      static_cast<uint32_t>(k16.size(2)), static_cast<uint32_t>(v8.size(3)), \
      static_cast<uint32_t>(q16.size(1)), static_cast<uint32_t>(k16.size(1)), \
      static_cast<float>(softmax_scale), stream
  const auto launch = [&](auto head_dim_tag) {
    constexpr uint32_t HeadDim = decltype(head_dim_tag)::value;
    if constexpr (MiddleInt8) {
      auto launch_int8 = [&](auto launcher) {
        launcher(
            q8.data_ptr<int8_t>(), k8.data_ptr<int8_t>(),
            reinterpret_cast<__nv_fp8_e4m3*>(v8.data_ptr()),
            MPA_STACK_COMMON_ARGS,
            q8_scale.data_ptr<float>(), k8_scale.data_ptr<float>(),
            v8_scale.data_ptr<float>(), MPA_STACK_NV_ARGS);
      };
      if constexpr (QueryBlock == 64) {
        auto launcher = active_fp16
            ? launch_mixed_attention_sm120_q64_nv_int8_fp16<HeadDim, true, true, false>
            : launch_mixed_attention_sm120_q64_nv_int8_fp16<HeadDim, true, false, false>;
        launch_int8(launcher);
      } else {
        auto launcher = active_fp16
            ? launch_mixed_attention_sm120_q128_nv_int8_fp16<HeadDim, true, true, false>
            : launch_mixed_attention_sm120_q128_nv_int8_fp16<HeadDim, true, false, false>;
        launch_int8(launcher);
      }
    } else {
      auto launch_mx = [&](auto launcher) {
        launcher(
            reinterpret_cast<int8_t*>(q8.data_ptr<uint8_t>()),
            reinterpret_cast<int8_t*>(k8.data_ptr<uint8_t>()),
            reinterpret_cast<__nv_fp8_e4m3*>(v8.data_ptr<uint8_t>()),
            MPA_STACK_COMMON_ARGS,
            q8_scale.data_ptr<uint8_t>(), k8_scale.data_ptr<uint8_t>(),
            v8_scale.data_ptr<uint8_t>(), MPA_STACK_NV_ARGS);
      };
      if constexpr (QueryBlock == 64) {
        auto launcher = active_fp16
            ? launch_mixed_attention_sm120_q64_nv_mx_fp16<HeadDim, true, true, false>
            : launch_mixed_attention_sm120_q64_nv_mx_fp16<HeadDim, true, false, false>;
        launch_mx(launcher);
      } else {
        auto launcher = active_fp16
            ? launch_mixed_attention_sm120_q128_nv_mx_fp16<HeadDim, true, true, false>
            : launch_mixed_attention_sm120_q128_nv_mx_fp16<HeadDim, true, false, false>;
        launch_mx(launcher);
      }
    }
  };
  if (q16.size(3) == 64) {
    launch(std::integral_constant<uint32_t, 64>{});
  } else {
    launch(std::integral_constant<uint32_t, 128>{});
  }
#undef MPA_STACK_NV_ARGS
#undef MPA_STACK_COMMON_ARGS
  ANEMOI_SM120_CUDA_CHECK(cudaGetLastError());
  return;
}


void sm120_q64_nv_int8_fp16_attention_forward(
    TensorView q4, TensorView q4_scale,
    TensorView k4, TensorView k4_scale,
    TensorView v4, TensorView v4_scale,
    TensorView q8, TensorView q8_scale,
    TensorView k8, TensorView k8_scale,
    TensorView v8, TensorView v8_scale,
    TensorView q16, TensorView k16, TensorView v16,
    TensorView block_ids, TensorView nvfp4_counts,
    TensorView middle_counts, TensorView fp16_counts,
    TensorView valid_k_counts, TensorView q_global_scale,
    TensorView k_global_scale, TensorView v_global_scale,
    int64_t fp16_prefix_blocks, double softmax_scale, bool active_fp16, TensorView output, cudaStream_t stream) {
  return three_phase_forward<64, true>(
      q4, q4_scale, k4, k4_scale, v4, v4_scale, q8, q8_scale, k8,
      k8_scale, v8, v8_scale, q16, k16, v16, block_ids, nvfp4_counts,
      middle_counts, fp16_counts, valid_k_counts, q_global_scale,
      k_global_scale, v_global_scale, fp16_prefix_blocks, softmax_scale,
      active_fp16, output, stream); \
}
void sm120_q128_nv_int8_fp16_attention_forward(
    TensorView q4, TensorView q4_scale,
    TensorView k4, TensorView k4_scale,
    TensorView v4, TensorView v4_scale,
    TensorView q8, TensorView q8_scale,
    TensorView k8, TensorView k8_scale,
    TensorView v8, TensorView v8_scale,
    TensorView q16, TensorView k16, TensorView v16,
    TensorView block_ids, TensorView nvfp4_counts,
    TensorView middle_counts, TensorView fp16_counts,
    TensorView valid_k_counts, TensorView q_global_scale,
    TensorView k_global_scale, TensorView v_global_scale,
    int64_t fp16_prefix_blocks, double softmax_scale, bool active_fp16, TensorView output, cudaStream_t stream) {
  return three_phase_forward<128, true>(
      q4, q4_scale, k4, k4_scale, v4, v4_scale, q8, q8_scale, k8,
      k8_scale, v8, v8_scale, q16, k16, v16, block_ids, nvfp4_counts,
      middle_counts, fp16_counts, valid_k_counts, q_global_scale,
      k_global_scale, v_global_scale, fp16_prefix_blocks, softmax_scale,
      active_fp16, output, stream); \
}



void sm120_q64_nv_mx_fp16_attention_forward(
    TensorView q4, TensorView q4_scale,
    TensorView k4, TensorView k4_scale,
    TensorView v4, TensorView v4_scale,
    TensorView q8, TensorView q8_scale,
    TensorView k8, TensorView k8_scale,
    TensorView v8, TensorView v8_scale,
    TensorView q16, TensorView k16, TensorView v16,
    TensorView block_ids, TensorView nvfp4_counts,
    TensorView middle_counts, TensorView fp16_counts,
    TensorView valid_k_counts, TensorView q_global_scale,
    TensorView k_global_scale, TensorView v_global_scale,
    int64_t fp16_prefix_blocks, double softmax_scale, bool active_fp16, TensorView output, cudaStream_t stream) {
  return three_phase_forward<64, false>(
      q4, q4_scale, k4, k4_scale, v4, v4_scale, q8, q8_scale, k8,
      k8_scale, v8, v8_scale, q16, k16, v16, block_ids, nvfp4_counts,
      middle_counts, fp16_counts, valid_k_counts, q_global_scale,
      k_global_scale, v_global_scale, fp16_prefix_blocks, softmax_scale,
      active_fp16, output, stream); \
}
void sm120_q128_nv_mx_fp16_attention_forward(
    TensorView q4, TensorView q4_scale,
    TensorView k4, TensorView k4_scale,
    TensorView v4, TensorView v4_scale,
    TensorView q8, TensorView q8_scale,
    TensorView k8, TensorView k8_scale,
    TensorView v8, TensorView v8_scale,
    TensorView q16, TensorView k16, TensorView v16,
    TensorView block_ids, TensorView nvfp4_counts,
    TensorView middle_counts, TensorView fp16_counts,
    TensorView valid_k_counts, TensorView q_global_scale,
    TensorView k_global_scale, TensorView v_global_scale,
    int64_t fp16_prefix_blocks, double softmax_scale, bool active_fp16, TensorView output, cudaStream_t stream) {
  return three_phase_forward<128, false>(
      q4, q4_scale, k4, k4_scale, v4, v4_scale, q8, q8_scale, k8,
      k8_scale, v8, v8_scale, q16, k16, v16, block_ids, nvfp4_counts,
      middle_counts, fp16_counts, valid_k_counts, q_global_scale,
      k_global_scale, v_global_scale, fp16_prefix_blocks, softmax_scale,
      active_fp16, output, stream); \
}
