// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 SageAttention team; 2025 NVIDIA CORPORATION & AFFILIATES.
// SM120 D128 unmasked specialization. Arithmetic helpers are shared with the
// ordinary kernel. Positive-scale arithmetic is retained; nonpositive scales
// apply scaling before the maximum and the padding mask.
#pragma once
#include "qk_int_sv_i8_cuda.cuh"
#include "tma_pipeline.cuh"
#include <cassert>
#include <cuda.h>

namespace comfy_sm120 {
template <bool positive_scale>
__global__ __launch_bounds__(384, 1) void dense_tma(
    int8_t *__restrict__ Q, int8_t *__restrict__ K, int8_t *__restrict__ V, nv_bfloat16 *__restrict__ O, float *__restrict__ Q_scale,
    float *__restrict__ K_scale, float *__restrict__ V_scale, uint32_t qo_len, uint32_t kv_len,
    uint32_t num_kv_groups, uint32_t stride_bz_q, uint32_t stride_seq_q,
    uint32_t stride_h_q, uint32_t stride_bz_k, uint32_t stride_seq_k,
    uint32_t stride_h_k, uint32_t stride_bz_v, uint32_t stride_h_v,
    uint32_t stride_d_v, uint32_t stride_bz_o, uint32_t stride_seq_o,
    uint32_t stride_h_o, float sm_scale,
    const __grid_constant__ CUtensorMap mapK,
    const __grid_constant__ CUtensorMap mapV) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200
  constexpr uint32_t CTA_Q = 128, CTA_K = 128, WARP_Q = 16, WARP_K = 128,
                     head_dim = 128;
  constexpr uint32_t num_warps_q = 8, num_warps_k = 1, num_warps = 8;
  constexpr uint32_t num_tiles_q = 1, num_tiles_k = 8, num_tiles_v = 8,
                     num_tiles_qk_inner = 4;
  constexpr uint32_t QK_SMEM_STRIDE = 128, O_SMEM_STRIDE = 128,
                     V_SMEM_STRIDE = 128;
  constexpr auto DTypeQK = DataType::kInt8;
  using DTypeSVAccum = float;
  using DTypeOut = nv_bfloat16;
  constexpr auto Q_GRAN = QuantGranularity::kPerThread,
                 K_GRAN = QuantGranularity::kPerThread;
  extern __shared__ __align__(1024) int8_t smem[];
  uint64_t *ready = (uint64_t *)(smem + 65536), *empty = ready + 3;
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    for (int i = 0; i < 3; i++) {
      tma_bar_init(ready + i);
      tma_bar_init_count(empty + i, 256);
    }
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  if (threadIdx.y >= 8) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 24;" ::: "memory");
    if (threadIdx.y == 8 && threadIdx.x == 0) {
      unsigned h =
          blockIdx.z * (gridDim.y / num_kv_groups) + blockIdx.y / num_kv_groups;
      for (unsigned op = 0; op < 2 * ((kv_len + 127) / 128); op++) {
        unsigned slot = op % 3, epoch = op / 3;
        if (op >= 3)
          tma_bar_wait(empty + slot, (epoch - 1) & 1);
        tma_issue((op & 1) ? &mapV : &mapK, smem + 16384 + slot * 16384,
                  ready + slot, (op & 1) ? int((op / 2) * 128) : 0,
                  (op & 1) ? 0 : int((op / 2) * 128), h);
      }
    }
    return;
  }
  asm volatile("setmaxnreg.inc.sync.aligned.u32 240;" ::: "memory");
  const uint32_t lane_id = get_lane_id();
  const uint32_t warp_id = get_warp_id();

  // maximize L2 hit rate
  const uint32_t batch_id = blockIdx.z;
  const uint32_t bx = blockIdx.x;
  const uint32_t num_qo_heads = gridDim.y;
  const uint32_t head_id = blockIdx.y;

  // transfer to base 2 instead of base e with better numerical efficiency
  sm_scale *= math::log2e;

  // RS holds the fragment of S
  int32_t RS[num_tiles_q][num_tiles_k][8];
  DTypeSVAccum RO[num_tiles_q][num_tiles_v][8];
  float m[num_tiles_q][2]; // max
  float d[num_tiles_q][2]; // denominator

  // Same per-thread quantization ABI as prequantize_int8_attention.
  const uint32_t q_blocks=div_ceil(qo_len,128);
  const uint32_t q_scale_idx=(batch_id*num_qo_heads+head_id)*q_blocks*32
      + bx*32 + (warp_id/2)*8 + lane_id/4;
  const uint32_t k_blocks=div_ceil(kv_len,128);
  const uint32_t k_scale_idx=(batch_id*(num_qo_heads/num_kv_groups)
      + head_id/num_kv_groups)*k_blocks*4 + lane_id%4;
  constexpr uint32_t k_scale_advance_offset=4;

  // initialize o, m, d
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
      
#pragma unroll
        for (uint32_t k = 0; k < 8; k++) {
          RO[fq][fv][k] = 0.0f;
        }
      
    }
  }
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
    for (uint32_t k = 0; k < 2; k++) {
      m[fq][k] = -50000.0f;
      d[fq][k] = 1.0f;
    }
  }

  constexpr uint32_t K_smem_idx_offset = CTA_Q;
  constexpr uint32_t V_smem_idx_offset = CTA_Q + CTA_K;

  constexpr SwizzleMode swizzle_mode_QK =
      (QK_SMEM_STRIDE == 32)   ? SwizzleMode::k32B
      : (QK_SMEM_STRIDE == 64) ? SwizzleMode::k64B
                               : SwizzleMode::k128B;
  smem_t<swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK> smem_Q(smem);
  smem_t<swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK> smem_K(
      smem + K_smem_idx_offset * QK_SMEM_STRIDE);
  constexpr SwizzleMode swizzle_mode_V =
      (V_SMEM_STRIDE == 64) ? SwizzleMode::k64B : SwizzleMode::k128B;
  smem_t<swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V> smem_V(
      smem + V_smem_idx_offset * QK_SMEM_STRIDE);
  constexpr SwizzleMode swizzle_mode_O =
      (O_SMEM_STRIDE == 32) ? SwizzleMode::k64B : SwizzleMode::k128B;
  smem_t<swizzle_mode_O, O_SMEM_STRIDE / PACK_SIZE_O> smem_O(smem);

  constexpr uint32_t global_to_shared_line_lanes_QK = (QK_SMEM_STRIDE == 32) ? 2
                                                      : (QK_SMEM_STRIDE == 64)
                                                          ? 4
                                                          : 8;
  constexpr uint32_t global_to_shared_copy_lines_per_warp_QK =
      (QK_SMEM_STRIDE == 32)   ? 16
      : (QK_SMEM_STRIDE == 64) ? 8
                               : 4;
  constexpr uint32_t global_to_shared_line_lanes_V =
      (V_SMEM_STRIDE == 64) ? 4 : 8;
  constexpr uint32_t global_to_shared_copy_lines_per_warp_V =
      (V_SMEM_STRIDE == 64) ? 8 : 4;
  constexpr uint32_t global_to_shared_line_lanes_O =
      (O_SMEM_STRIDE == 32) ? 4 : 8;
  constexpr uint32_t global_to_shared_copy_lines_per_warp_O =
      (O_SMEM_STRIDE == 32) ? 8 : 4;

  constexpr uint32_t QK_smem_iters_row =
      QK_SMEM_STRIDE / (global_to_shared_line_lanes_QK * PACK_SIZE_QK);
  constexpr uint32_t Q_smem_iters_col =
      CTA_Q / (num_warps * global_to_shared_copy_lines_per_warp_QK);
  constexpr uint32_t K_smem_iters_col =
      CTA_K / (num_warps * global_to_shared_copy_lines_per_warp_QK);
  constexpr uint32_t V_smem_iters_row =
      V_SMEM_STRIDE / (global_to_shared_line_lanes_V * PACK_SIZE_V);
  constexpr uint32_t V_smem_iters_col =
      head_dim / (num_warps * global_to_shared_copy_lines_per_warp_V);
  constexpr uint32_t O_smem_iters_row =
      O_SMEM_STRIDE / (global_to_shared_line_lanes_O * PACK_SIZE_O);
  constexpr uint32_t O_smem_iters_col =
      CTA_Q / (num_warps * global_to_shared_copy_lines_per_warp_O);

  int8_t *Q_lane_base_ptr =
      Q + batch_id * stride_bz_q + head_id * stride_h_q +
      (bx * CTA_Q + CTA_Q / num_warps * warp_id +
       lane_id / global_to_shared_line_lanes_QK) *
          stride_seq_q +
      (lane_id % global_to_shared_line_lanes_QK) * PACK_SIZE_QK;
  uint32_t Q_smem_offset_load = smem_Q.get_permuted_offset(
      warp_id * global_to_shared_copy_lines_per_warp_QK * Q_smem_iters_col +
          lane_id / global_to_shared_line_lanes_QK,
      lane_id % global_to_shared_line_lanes_QK);

  uint32_t Q_smem_offset_mma = smem_Q.get_permuted_offset(
      get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q + lane_id % 16,
      lane_id / 16);
  uint32_t K_smem_offset_mma = smem_K.get_permuted_offset(
      get_warp_idx_k<num_warps_q, num_warps_k>() * WARP_K + lane_id % 8 +
          (lane_id / 16) * 8,
      (lane_id / 8) % 2);
  // for causal masking

  // for loading
  uint32_t Q_load_idx_lane_base = bx * CTA_Q + CTA_Q / num_warps * warp_id +
                                  lane_id / global_to_shared_line_lanes_QK;

  const uint32_t num_iterations = div_ceil(kv_len, CTA_K);

  // load Q with predicate
  load_global_to_share<global_to_shared_line_lanes_QK,
                       global_to_shared_copy_lines_per_warp_QK,
                       QK_smem_iters_row, Q_smem_iters_col, swizzle_mode_QK,
                       QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_Q>(
      &Q_lane_base_ptr, Q_smem_offset_load, stride_seq_q, smem_Q,
      Q_load_idx_lane_base, qo_len);
  cp_async::commit_group();
  cp_async::wait_group<0>();
  asm volatile("bar.sync 1, 256;" ::: "memory");

  uint32_t RQ[num_tiles_q][num_tiles_qk_inner][4];
  uint32_t q_offset = Q_smem_offset_mma;
#pragma unroll
  for (uint32_t inner = 0; inner < 4; ++inner) {
    smem_Q.ldmatrix_m8n8x4(q_offset, RQ[0][inner]);
    q_offset = smem_Q.advance_offset_by_column<2>(q_offset, inner);
  }
  const float original_sm_scale = sm_scale;
  const float q_scale = Q_scale[q_scale_idx];
  // A single producer rotates K,V through three 16 KiB slots. Every compute
  // thread releases a slot only after its last ldmatrix load has completed.
#pragma unroll
  for (uint32_t iter = 0; iter < num_iterations; ++iter) {
    const uint32_t kop = 2 * iter, vop = kop + 1;
    tma_bar_wait(ready + kop % 3, (kop / 3) & 1);
    smem_K.set_base(smem + 16384 + (kop % 3) * 16384);
    compute_int_qk_cached<8, 1, 1, 8, 4, SwizzleMode::k128B, 8,
                          DataType::kInt8>(smem_K, RS, RQ, K_smem_offset_mma);
    tma_bar_arrive(empty + kop % 3);
    const float dequant_scale =
        q_scale * K_scale[k_scale_idx + iter * k_scale_advance_offset];
    const float tile_scale = original_sm_scale * dequant_scale;
    uint32_t RS_u8[1][4][4];
    if constexpr (!positive_scale) {
      // Max-before-scale is valid only for a positive scale. Scale first for
      // zero/negative values, then mask in the scaled domain so padding cannot
      // turn into positive logits (negative scale) or valid logits (zero).
      auto &scores = reinterpret_cast<float (&)[1][8][8]>(RS);
#pragma unroll
      for (int fk = 0; fk < 8; ++fk) {
#pragma unroll
        for (int i = 0; i < 8; ++i)
          scores[0][fk][i] = __int2float_rz(RS[0][fk][i]) * tile_scale;
      }
      if (iter + 1 == num_iterations)
        apply_out_of_bound_mask<1, 8>(iter * 128 + 2 * (lane_id % 4), scores,
                                      kv_len);
      update_mdo_f32_u8<1, 8, 8>(scores, RO, m, d, S_U8_OFFSET, RS_u8);
    } else if (iter + 1 < num_iterations) {
      update_mdo_i32_u8<1, 8, 8>(RS, RO, m, d, tile_scale, S_U8_OFFSET, RS_u8);
    } else {
      // Retain the ordinary kernel's last-tile conversion and OOB masking.
      float scores[1][8][8], pv_scale[1][2];
#pragma unroll
      for (int fk = 0; fk < 8; ++fk) {
#pragma unroll
        for (int i = 0; i < 8; ++i)
          scores[0][fk][i] = __int2float_rz(RS[0][fk][i]);
      }
      apply_out_of_bound_mask<1, 8>(iter * 128 + 2 * (lane_id % 4), scores,
                                    kv_len, -1.0e30f);
      update_mdo<1, 8, 8, true, false>(scores, RO, m, d, pv_scale, tile_scale,
                                       S_U8_OFFSET);
      RS_to_u8<1, 8>(scores, RS_u8);
      accumulate_d<1, 8>(scores, d, pv_scale);
      RS[0][0][0] = __float_as_int(pv_scale[0][0]);
      RS[0][0][1] = __float_as_int(pv_scale[0][1]);
    }
    tma_bar_wait(ready + vop % 3, (vop / 3) & 1);
    smem_V.set_base(smem + 16384 + (vop % 3) * 16384);
    compute_int8_sv<8, 1, 1, 8, 8, SwizzleMode::k128B, 8>(smem_V, RS, RS_u8,
                                                          RO);
    tma_bar_arrive(empty + vop % 3);
  }
  // All asynchronous writes have completed before the output reuses shared
  // memory occupied by Q and the first ring slot.
  asm volatile("bar.sync 1, 256;" ::: "memory");
  normalize_d<1, 8, ComputeUnit::kCudaCore>(RO, m, d);
  {
    float v_scale[4];
    float *V_scale_base_ptr =
        V_scale + batch_id * (num_qo_heads / num_kv_groups) * head_dim +
        (head_id / num_kv_groups) * head_dim + (lane_id % 4) * 2;
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
      ((float2 *)v_scale)[0] = *((float2 *)(V_scale_base_ptr + fv * 16));
      ((float2 *)v_scale)[1] = *((float2 *)(V_scale_base_ptr + fv * 16 + 8));
#pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
        for (uint32_t k = 0; k < 8; k++) {
          const float scale_value = v_scale[(k / 4) * 2 + (k % 2)];
          RO[fq][fv][k] *= scale_value;
        }
      }
    }
  }
  // save the result to shared memory
  uint32_t smem_O_row_base =
      get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q + lane_id / 4;
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
      uint32_t offset_O = smem_O.get_permuted_offset(
          smem_O_row_base + fq * MMA_QK_M, fv * (MMA_SV_N / PACK_SIZE_O));

      
        // convert RO to half
        uint32_t RO_f16[4];
#pragma unroll
        for (uint32_t k = 0; k < 4; k++) {
          
            ((nv_bfloat162 *)RO_f16)[k] =
                __float22bfloat162_rn(((float2 *)RO[fq][fv])[k]);
          
        }

        ((uint32_t *)(smem_O.base + offset_O))[lane_id % 4] = RO_f16[0];
        ((uint32_t *)(smem_O.base + offset_O +
                      8 * (O_SMEM_STRIDE / PACK_SIZE_O)))[lane_id % 4] =
            RO_f16[1];

        offset_O = smem_O.get_permuted_offset(
            smem_O_row_base + fq * MMA_QK_M, fv * (MMA_SV_N / PACK_SIZE_O) + 1);
        ((uint32_t *)(smem_O.base + offset_O))[lane_id % 4] = RO_f16[2];
        ((uint32_t *)(smem_O.base + offset_O +
                      8 * (O_SMEM_STRIDE / PACK_SIZE_O)))[lane_id % 4] =
            RO_f16[3];
      
    }
  }

  // ! do we need to sync here?
  __syncwarp();

  // shared memory to global memory
  DTypeOut *O_lane_ptr =
      O + batch_id * stride_bz_o + head_id * stride_h_o +
      (bx * CTA_Q + WARP_Q * get_warp_idx_q<num_warps_q, num_warps_k>() +
       lane_id / global_to_shared_line_lanes_O) *
          stride_seq_o +
      lane_id % global_to_shared_line_lanes_O * PACK_SIZE_O;
  uint32_t offset_O = smem_O.get_permuted_offset(
      get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q +
          lane_id / global_to_shared_line_lanes_O,
      lane_id % global_to_shared_line_lanes_O);
  uint32_t O_load_idx_lane_base = bx * CTA_Q + CTA_Q / num_warps * warp_id +
                                  lane_id / global_to_shared_line_lanes_O;

#pragma unroll
  for (uint32_t i = 0; i < O_smem_iters_col; i++) {
#pragma unroll
    for (uint32_t j = 0; j < O_smem_iters_row; j++) {
      if (O_load_idx_lane_base < qo_len) {
        smem_O.store_128b(offset_O, O_lane_ptr);
      }
      O_lane_ptr += (global_to_shared_line_lanes_O * PACK_SIZE_O);
      offset_O = smem_O.advance_offset_by_column<global_to_shared_line_lanes_O>(
          offset_O);
    }

    offset_O =
        smem_O.advance_offset_by_row<global_to_shared_copy_lines_per_warp_O>(
            offset_O - (O_smem_iters_row * global_to_shared_line_lanes_O));
    O_lane_ptr +=
        ((global_to_shared_copy_lines_per_warp_O * stride_seq_o) -
         (O_smem_iters_row * global_to_shared_line_lanes_O * PACK_SIZE_O));
    O_load_idx_lane_base += global_to_shared_copy_lines_per_warp_O;
  }

#endif
}
} // namespace comfy_sm120
