// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// This private ABI is only used after the ordinary attention validation.
struct DenseTmaArgs {
  void *q, *k, *v, *o, *q_scale, *k_scale, *v_scale;
  uint32_t batch, qo_len, kv_len, qo_heads, kv_heads;
  uint32_t q_stride_b, q_stride_h, q_stride_s;
  uint32_t k_stride_b, k_stride_h, k_stride_s;
  uint32_t v_stride_b, v_stride_h, v_stride_d;
  uint32_t o_stride_b, o_stride_h, o_stride_s;
  float sm_scale;
};
bool launch_dense_tma_sm120(const DenseTmaArgs &args, cudaStream_t stream);
