// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. Derived from SageAttention
// (https://github.com/thu-ml/SageAttention) commit
// d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5. DLPack-compatible launcher for the
// Pure integer attention kernel: signed INT8 Q/K/V, unsigned INT8 softmax
// probabilities, INT32 tensor-core P*V accumulation, and FP32 online-softmax
// state. V scaling is fused and LSE is not returned.

#include "qk_int_sv_i8_cuda.cuh"
#include <algorithm>
#include <stdexcept>
#include <string>

namespace {

template <int HEAD_DIM, int CTA_K, MaskMode mask_mode, typename DTypeOut,
          bool fuse_fp32_probabilities = true>
void launch_impl(int8_t *q, int8_t *k, int8_t *v, DTypeOut *o, float *q_scale,
                 float *k_scale, float *v_scale, const void *mask,
                 int64_t mask_stride_b, int64_t mask_stride_h,
                 int64_t mask_stride_q, int64_t mask_stride_k,
                 int mask_dtype_code, int qo_len, int kv_len,
                 int num_qo_heads, int num_kv_groups, int stride_bz_q,
                 int stride_seq_q, int stride_h_q, int stride_bz_k,
                 int stride_seq_k, int stride_h_k, int stride_bz_v,
                 int stride_h_v, int stride_d_v, int stride_bz_o,
                 int stride_seq_o, int stride_h_o, float sm_scale,
                 int batch_size, cudaStream_t stream) {
  // Tiling constants — must match sage_attention.py and dlpack_bindings.cpp.
  constexpr int CTA_Q = 128;
  // D>=128 otherwise needs too many live FP32 output accumulators per thread.
  // A 16-row warp tile halves that accumulator set.
  constexpr int WARP_Q = HEAD_DIM >= 128 ? 16 : 32;
  constexpr int WARP_K = CTA_K;

  size_t smem_max =
      std::max(static_cast<size_t>(CTA_Q * HEAD_DIM * sizeof(int8_t) +
                                   CTA_K * HEAD_DIM * sizeof(int8_t) +
                                   CTA_K * HEAD_DIM * sizeof(int8_t)),
               static_cast<size_t>(CTA_Q * HEAD_DIM * sizeof(half)));

  auto kernel = qk_int_sv_i8_attn_kernel<
      CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM, DataType::kInt8,
      QuantGranularity::kPerThread, QuantGranularity::kPerThread, float, false,
      DTypeOut, ComputeUnit::kCudaCore, mask_mode, false, true, false, false,
      fuse_fp32_probabilities>;

  cudaError_t error = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_max));
  if (error != cudaSuccess) {
    throw std::runtime_error(
        "sage_attn failed to request " + std::to_string(smem_max) +
        " bytes of dynamic shared memory: " + cudaGetErrorString(error));
  }

  dim3 grid(div_ceil(qo_len, CTA_Q), num_qo_heads, batch_size);
  dim3 block(32, (CTA_Q / WARP_Q) * (CTA_K / WARP_K));

  kernel<<<grid, block, smem_max, stream>>>(
      q, k, v, o, nullptr, q_scale, k_scale, v_scale, nullptr, mask,
      mask_stride_b, mask_stride_h, mask_stride_q, mask_stride_k,
      mask_dtype_code, qo_len, kv_len, num_kv_groups, stride_bz_q,
      stride_seq_q, stride_h_q, stride_bz_k, stride_seq_k, stride_h_k,
      stride_bz_v, stride_h_v, stride_d_v, stride_bz_o, stride_seq_o,
      stride_h_o, sm_scale, 0, 0, nullptr, nullptr, 0);

  error = cudaGetLastError();
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string("sage_attn kernel launch failed: ") +
                             cudaGetErrorString(error));
  }
}

} // anonymous namespace

extern "C" void launch_sage_attn_kernel(
    const void *q, const void *k, const void *v, void *o, const void *q_scale,
    const void *k_scale, const void *v_scale, const void *mask,
    int64_t mask_stride_b, int64_t mask_stride_h, int64_t mask_stride_q,
    int64_t mask_stride_k, int mask_dtype_code, int cta_k, int batch_size,
    int qo_len,
    int kv_len, int num_qo_heads, int num_kv_heads, int head_dim,
    int stride_bz_q, int stride_seq_q, int stride_h_q, int stride_bz_k,
    int stride_seq_k, int stride_h_k, int stride_bz_v, int stride_h_v,
    int stride_d_v, int stride_bz_o, int stride_seq_o, int stride_h_o,
    float sm_scale, int output_dtype_code, cudaStream_t stream) {
  if (cta_k != 64 && cta_k != 128) {
    throw std::runtime_error("sage_attn: cta_k must be 64 or 128");
  }
  if (cta_k == 128 && (head_dim == 64 || mask != nullptr)) {
    throw std::runtime_error(
        "sage_attn: cta_k 128 requires unmasked head_dim 128 or 256");
  }
  int num_kv_groups = num_qo_heads / num_kv_heads;

  // Upstream kernel uses non-const pointers; cast away const from the
  // extern "C" boundary (kernel does not modify inputs).
  auto q_ = const_cast<int8_t *>(static_cast<const int8_t *>(q));
  auto k_ = const_cast<int8_t *>(static_cast<const int8_t *>(k));
  auto v_ = const_cast<int8_t *>(static_cast<const int8_t *>(v));
  auto qs_ = const_cast<float *>(static_cast<const float *>(q_scale));
  auto ks_ = const_cast<float *>(static_cast<const float *>(k_scale));
  auto vs_ = const_cast<float *>(static_cast<const float *>(v_scale));

#define LAUNCH(HD, CK, MM, DT, FUSE_FP32)                                     \
  launch_impl<HD, CK, MM, DT, FUSE_FP32>(                                    \
                          q_, k_, v_, static_cast<DT *>(o), qs_, ks_, vs_,     \
                          mask, mask_stride_b, mask_stride_h, mask_stride_q,   \
                          mask_stride_k, mask_dtype_code, qo_len, kv_len,      \
                          num_qo_heads, num_kv_groups,                         \
                          stride_bz_q, stride_seq_q, stride_h_q, stride_bz_k,  \
                          stride_seq_k, stride_h_k, stride_bz_v, stride_h_v,   \
                          stride_d_v, stride_bz_o, stride_seq_o, stride_h_o,   \
                          sm_scale, batch_size, stream)

#define LAUNCH_CTA(HD, MM, DT, FUSE_FP32)                                     \
  if constexpr (HD == 64) {                                                   \
    LAUNCH(HD, 64, MM, DT, FUSE_FP32);                                        \
  } else if (cta_k == 128) {                                                   \
    LAUNCH(HD, 128, MM, DT, FUSE_FP32);                                       \
  } else {                                                                     \
    LAUNCH(HD, 64, MM, DT, FUSE_FP32);                                        \
  }

#define DISPATCH_DTYPE(HD, MM)                                                 \
  if (output_dtype_code == 1) {                                                \
    if constexpr (MM == MaskMode::kNone) {                                    \
      if (kv_len <= 512 && cta_k == 64) {                                      \
        LAUNCH(HD, 64, MM, half, false);                                       \
      } else {                                                                 \
        LAUNCH_CTA(HD, MM, half, true);                                        \
      }                                                                        \
    } else {                                                                   \
      LAUNCH(HD, 64, MM, half, true);                                          \
    }                                                                          \
  } else {                                                                     \
    if constexpr (MM == MaskMode::kNone) {                                    \
      if (kv_len <= 512 && cta_k == 64) {                                      \
        LAUNCH(HD, 64, MM, nv_bfloat16, false);                                \
      } else {                                                                 \
        LAUNCH_CTA(HD, MM, nv_bfloat16, true);                                 \
      }                                                                        \
    } else {                                                                   \
      LAUNCH(HD, 64, MM, nv_bfloat16, true);                                   \
    }                                                                          \
  }

#define DISPATCH_MASK(HD)                                                      \
  if (mask != nullptr) {                                                       \
    if (mask_stride_q == 0) {                                                  \
      DISPATCH_DTYPE(HD, MaskMode::kCustomKey);                                \
    } else {                                                                   \
      DISPATCH_DTYPE(HD, MaskMode::kCustom);                                   \
    }                                                                          \
  } else {                                                                     \
    DISPATCH_DTYPE(HD, MaskMode::kNone);                                       \
  }

  if (head_dim == 64) {
    DISPATCH_MASK(64);
  } else if (head_dim == 128) {
    DISPATCH_MASK(128);
  } else if (head_dim == 256) {
    DISPATCH_MASK(256);
  } else {
    throw std::runtime_error("sage_attn: unsupported head_dim " +
                             std::to_string(head_dim));
  }

#undef LAUNCH
#undef LAUNCH_CTA
#undef DISPATCH_DTYPE
#undef DISPATCH_MASK
}
