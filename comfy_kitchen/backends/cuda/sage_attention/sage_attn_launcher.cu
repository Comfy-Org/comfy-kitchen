// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. Derived from SageAttention
// (https://github.com/thu-ml/SageAttention) commit
// d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5. DLPack-compatible launcher for the
// Pure integer attention kernel: signed INT8 Q/K/V, unsigned INT8 softmax
// probabilities, INT32 tensor-core P*V accumulation, and FP32 online-softmax
// state. V scaling is fused and LSE is not returned.

#include "qk_int_sv_i8_cuda.cuh"
#include <math_constants.h>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace {

template <int HEAD_DIM, int CTA_K, MaskMode mask_mode, typename DTypeOut,
          bool fuse_fp32_probabilities = true, int CTA_Q = 128>
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
                 int batch_size, cudaStream_t stream, const float *mask_tile_bias) {
  // Tiling constants — must match sage_attention.py and dlpack_bindings.cpp.
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
      stride_h_o, sm_scale, mask_tile_bias);

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
    float sm_scale, int output_dtype_code, cudaStream_t stream,
    const float *mask_tile_bias) {
  if (cta_k != 64 && cta_k != 128) {
    throw std::runtime_error("sage_attn: cta_k must be 64 or 128");
  }
  if (cta_k == 128 && (head_dim == 64 || (mask != nullptr && mask_tile_bias == nullptr))) {
    throw std::runtime_error(
        "sage_attn: cta_k 128 requires unmasked head_dim 128/256 or a prepared key mask");
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

#define LAUNCH_Q(HD, CK, MM, DT, FUSE_FP32, CQ)                                     \
  launch_impl<HD, CK, MM, DT, FUSE_FP32, CQ>(                                    \
                          q_, k_, v_, static_cast<DT *>(o), qs_, ks_, vs_,     \
                          mask, mask_stride_b, mask_stride_h, mask_stride_q,   \
                          mask_stride_k, mask_dtype_code, qo_len, kv_len,      \
                          num_qo_heads, num_kv_groups,                         \
                          stride_bz_q, stride_seq_q, stride_h_q, stride_bz_k,  \
                          stride_seq_k, stride_h_k, stride_bz_v, stride_h_v,   \
                          stride_d_v, stride_bz_o, stride_seq_o, stride_h_o,   \
                          sm_scale, batch_size, stream, mask_tile_bias)

#define LAUNCH(HD, CK, MM, DT, FUSE_FP32) \
  LAUNCH_Q(HD, CK, MM, DT, FUSE_FP32, 128)

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

#define DISPATCH_PREPARED(HD, CK, CQ)                                          \
  if (output_dtype_code == 1) {                                                \
    LAUNCH_Q(HD, CK, MaskMode::kPreparedKey, half, true, CQ);                    \
  } else {                                                                     \
    LAUNCH_Q(HD, CK, MaskMode::kPreparedKey, nv_bfloat16, true, CQ);             \
  }

  if (mask_tile_bias != nullptr) {
    if ((head_dim != 64 && head_dim != 128 && head_dim != 256) ||
        cta_k != (head_dim == 64 ? 64 : 128)) {
      throw std::runtime_error("sage_attn: incompatible head dimension or tile size for prepared mask");
    }
    if (head_dim == 64) {
      DISPATCH_PREPARED(64, 64, 128);
      return;
    }
    if (head_dim == 256) {
      DISPATCH_PREPARED(256, 128, 128);
      return;
    }
    // For D=128, Blackwell benefits from two 64-query CTAs per SM; Ada
    // performs better with one 128-query CTA. Q scales retain 128-row groups.
    int device, major;
    cudaError_t error = cudaGetDevice(&device);
    if (error == cudaSuccess) {
      error = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    }
    if (error != cudaSuccess) {
      throw std::runtime_error(std::string("sage_attn device query failed: ") +
                               cudaGetErrorString(error));
    }
    if (major >= 10) {
      DISPATCH_PREPARED(128, 128, 64);
    } else {
      DISPATCH_PREPARED(128, 128, 128);
    }
    return;
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
#undef LAUNCH_Q
#undef LAUNCH_CTA
#undef DISPATCH_DTYPE
#undef DISPATCH_MASK
#undef DISPATCH_PREPARED
}

namespace {

// Normalize and classify each key-mask tile once, instead of repeating dtype
// conversion and validity checks for every query and query head. The packed
// row contains padded FP32 biases followed by one constant-bias value per tile;
// NaN marks a varying tile. All non-finite input values retain the existing
// masking semantics and become -inf.
template <typename T>
__global__ void prepare_key_mask_kernel(
    const T *mask, float *packed, int length, int tiles,
    int64_t stride_b, int64_t stride_h, int64_t stride_k, int heads) {
  constexpr int tile_size = 128;
  const int key = blockIdx.x * tile_size + threadIdx.x;
  const int width = ((tiles * (tile_size + 1) + 3) / 4) * 4;
  const int64_t output_row = (static_cast<int64_t>(blockIdx.z) * heads + blockIdx.y) * width;
  float bias = -CUDART_INF_F;
  if (key < length) {
    const int64_t offset = blockIdx.z * stride_b + blockIdx.y * stride_h + key * stride_k;
    if constexpr (std::is_same_v<T, uint8_t>) {
      bias = mask[offset] ? 0.0f : -CUDART_INF_F;
    } else {
      bias = static_cast<float>(mask[offset]);
      if (!isfinite(bias)) bias = -CUDART_INF_F;
    }
  }
  packed[output_row + key] = bias;
  float low = bias;
  float high = bias;
#pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1) {
    low = fminf(low, __shfl_down_sync(0xffffffff, low, delta));
    high = fmaxf(high, __shfl_down_sync(0xffffffff, high, delta));
  }
  __shared__ float lows[4], highs[4];
  if (threadIdx.x % 32 == 0) {
    lows[threadIdx.x / 32] = low;
    highs[threadIdx.x / 32] = high;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int warp = 1; warp < 4; ++warp) {
      low = fminf(low, lows[warp]);
      high = fmaxf(high, highs[warp]);
    }
    packed[output_row + tiles * tile_size + blockIdx.x] =
        low == high ? high : CUDART_NAN_F;
  }
}

} // namespace

extern "C" void launch_sage_prepare_key_mask(
    const void *mask, float *packed, int batch, int heads, int length,
    int64_t stride_b, int64_t stride_h, int64_t stride_k,
    int dtype_code, cudaStream_t stream) {
  const int tiles = div_ceil(length, 128);
  const dim3 grid(tiles, heads, batch);
#define PREPARE_MASK(T) \
  prepare_key_mask_kernel<T><<<grid, 128, 0, stream>>>( \
      static_cast<const T *>(mask), packed, length, tiles, \
      stride_b, stride_h, stride_k, heads)
  switch (dtype_code) {
  case 0: PREPARE_MASK(float); break;
  case 1: PREPARE_MASK(half); break;
  case 2: PREPARE_MASK(nv_bfloat16); break;
  case 3: PREPARE_MASK(uint8_t); break;
  default: throw std::runtime_error("sage_prepare_key_mask: unsupported dtype");
  }
#undef PREPARE_MASK
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string("sage_prepare_key_mask launch failed: ") +
                             cudaGetErrorString(error));
  }
}
