// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org. All rights reserved.
// BF16 SOL attention: fused summary preprocessing, block routing, approximate
// centroid attention, and exact attention for locally important blocks.

#include "attention/sol_common.cuh"

#include <float.h>
#include <stdint.h>
#include <stdexcept>
#include <string>

#include <cuda_bf16.h>

using namespace comfy::sol_attention;

namespace {

constexpr int BLOCK_Q = 64;
constexpr int BLOCK_KV = 64;
constexpr int GROUP = 32;
constexpr int DIM = 128;
constexpr int NUM_WARPS = 4;
constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;
constexpr int WARP_Q = BLOCK_Q / NUM_WARPS;
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;
constexpr float NEG_INF = -1.0e20f;

__device__ inline void load_Q_rmem(
    uint32_t Q_rmem[WARP_Q / MMA_M][DIM / MMA_K][4], uint32_t Q_smem_thread) {
  for (int mq = 0; mq < WARP_Q / MMA_M; mq++)
    for (int md = 0; md < DIM / MMA_K; md++) {
      uint32_t addr = Q_smem_thread;
      addr += mq * MMA_M * DIM * sizeof(nv_bfloat16);
      addr ^= md * MMA_K * sizeof(nv_bfloat16);
      ldmatrix_x4(Q_rmem[mq][md], addr);
    }
}

__device__ inline void load_K_rmem_full(
    uint32_t K_rmem[BLOCK_KV / MMA_N][DIM / MMA_K][2], uint32_t K_smem_thread) {
  for (int mk = 0; mk < BLOCK_KV / MMA_N; mk++)
    for (int md = 0; md < DIM / MMA_K; md += 2) {
      uint32_t addr = K_smem_thread;
      addr += mk * MMA_N * DIM * sizeof(nv_bfloat16);
      addr ^= md * MMA_K * sizeof(nv_bfloat16);
      ldmatrix_x4(K_rmem[mk][md], addr);
    }
}

__device__ inline void load_V_rmem_full(
    uint32_t V_rmem[BLOCK_KV / MMA_K][DIM / MMA_N][2], uint32_t V_smem_thread) {
  for (int mk = 0; mk < BLOCK_KV / MMA_K; mk++)
    for (int md = 0; md < DIM / MMA_N; md += 2) {
      uint32_t addr = V_smem_thread;
      addr += mk * MMA_K * DIM * sizeof(nv_bfloat16);
      addr ^= md * MMA_N * sizeof(nv_bfloat16);
      ldmatrix_x4_trans(V_rmem[mk][md], addr);
    }
}

__device__ inline void gemm_qk(
    uint32_t Q_rmem[WARP_Q / MMA_M][DIM / MMA_K][4],
    uint32_t K_rmem[BLOCK_KV / MMA_N][DIM / MMA_K][2],
    float S[WARP_Q / MMA_M][BLOCK_KV / MMA_N][4], float scale, int n_kv_tiles) {
  for (int mq = 0; mq < WARP_Q / MMA_M; mq++)
    for (int mk = 0; mk < n_kv_tiles; mk++) {
#pragma unroll
      for (int r = 0; r < 4; r++)
        S[mq][mk][r] = 0.f;
      for (int md = 0; md < DIM / MMA_K; md++)
        mma_bf16(Q_rmem[mq][md], K_rmem[mk][md], S[mq][mk]);
#pragma unroll
      for (int r = 0; r < 4; r++)
        S[mq][mk][r] *= scale;
    }
}

__device__ inline void gemm_pv(
    uint32_t P_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_K][4],
    uint32_t V_rmem[BLOCK_KV / MMA_K][DIM / MMA_N][2],
    float O_rmem[WARP_Q / MMA_M][DIM / MMA_N][4], int n_k_tiles) {
  for (int mq = 0; mq < WARP_Q / MMA_M; mq++)
    for (int md = 0; md < DIM / MMA_N; md++)
      for (int mk = 0; mk < n_k_tiles; mk++)
        mma_bf16(P_rmem[mq][mk], V_rmem[mk][md], O_rmem[mq][md]);
}

// Inline copy of attention_v5's online softmax for one KV tile.
__device__ inline void softmax_exact_tile(
    float S_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_N][4],
    float rowmax[WARP_Q / MMA_M][2], float rowsumexp[WARP_Q / MMA_M][2],
    float O_rmem[WARP_Q / MMA_M][DIM / MMA_N][4],
    uint32_t P_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_K][4]) {
  for (int mq = 0; mq < WARP_Q / MMA_M; mq++) {
    float this_rowmax[2];
    for (int mk = 0; mk < BLOCK_KV / MMA_N; mk++) {
      float *regs = S_rmem[mq][mk];
      if (mk == 0) {
        this_rowmax[0] = max(regs[0], regs[1]);
        this_rowmax[1] = max(regs[2], regs[3]);
      } else {
        this_rowmax[0] = max(this_rowmax[0], max(regs[0], regs[1]));
        this_rowmax[1] = max(this_rowmax[1], max(regs[2], regs[3]));
      }
    }
    this_rowmax[0] =
        max(this_rowmax[0], __shfl_xor_sync(0xffffffffu, this_rowmax[0], 1));
    this_rowmax[0] =
        max(this_rowmax[0], __shfl_xor_sync(0xffffffffu, this_rowmax[0], 2));
    this_rowmax[1] =
        max(this_rowmax[1], __shfl_xor_sync(0xffffffffu, this_rowmax[1], 1));
    this_rowmax[1] =
        max(this_rowmax[1], __shfl_xor_sync(0xffffffffu, this_rowmax[1], 2));
    this_rowmax[0] = max(this_rowmax[0], rowmax[mq][0]);
    this_rowmax[1] = max(this_rowmax[1], rowmax[mq][1]);

    float rescale[2] = {__expf(rowmax[mq][0] - this_rowmax[0]),
                        __expf(rowmax[mq][1] - this_rowmax[1])};
    for (int md = 0; md < DIM / MMA_N; md++) {
      O_rmem[mq][md][0] *= rescale[0];
      O_rmem[mq][md][1] *= rescale[0];
      O_rmem[mq][md][2] *= rescale[1];
      O_rmem[mq][md][3] *= rescale[1];
    }
    rowmax[mq][0] = this_rowmax[0];
    rowmax[mq][1] = this_rowmax[1];

    float this_rowsumexp[2];
    for (int mk = 0; mk < BLOCK_KV / MMA_N; mk++) {
      float *regs = S_rmem[mq][mk];
      regs[0] = __expf(regs[0] - rowmax[mq][0]);
      regs[1] = __expf(regs[1] - rowmax[mq][0]);
      regs[2] = __expf(regs[2] - rowmax[mq][1]);
      regs[3] = __expf(regs[3] - rowmax[mq][1]);
      if (mk == 0) {
        this_rowsumexp[0] = regs[0] + regs[1];
        this_rowsumexp[1] = regs[2] + regs[3];
      } else {
        this_rowsumexp[0] += regs[0] + regs[1];
        this_rowsumexp[1] += regs[2] + regs[3];
      }
      auto *P = reinterpret_cast<nv_bfloat162 *>(P_rmem[mq][mk / 2]);
      P[(mk % 2) * 2] = __float22bfloat162_rn({regs[0], regs[1]});
      P[(mk % 2) * 2 + 1] = __float22bfloat162_rn({regs[2], regs[3]});
    }
    this_rowsumexp[0] += __shfl_xor_sync(0xffffffffu, this_rowsumexp[0], 1);
    this_rowsumexp[0] += __shfl_xor_sync(0xffffffffu, this_rowsumexp[0], 2);
    this_rowsumexp[1] += __shfl_xor_sync(0xffffffffu, this_rowsumexp[1], 1);
    this_rowsumexp[1] += __shfl_xor_sync(0xffffffffu, this_rowsumexp[1], 2);
    rowsumexp[mq][0] = rowsumexp[mq][0] * rescale[0] + this_rowsumexp[0];
    rowsumexp[mq][1] = rowsumexp[mq][1] * rescale[1] + this_rowsumexp[1];
  }
}

__device__ inline void setup_smem_threads(
    uint32_t Q_smem, uint32_t K_smem, uint32_t V_smem, int warp_id, int lane_id,
    uint32_t &Q_smem_thread, uint32_t &K_smem_thread,
    uint32_t &V_smem_thread) {
  {
    const int row_off = warp_id * WARP_Q + (lane_id % 16);
    const int col_off = lane_id / 16 * 8;
    Q_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)>(
        Q_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
  }
  {
    const int row_off = lane_id % 8;
    const int col_off = lane_id / 8 * 8;
    K_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)>(
        K_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
  }
  {
    const int row_off = lane_id % 16;
    const int col_off = lane_id / 16 * 8;
    V_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)>(
        V_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
  }
}

__device__ inline void write_O(
    nv_bfloat16 *O_ptr, float O_rmem[WARP_Q / MMA_M][DIM / MMA_N][4],
    float rowsumexp[WARP_Q / MMA_M][2], int warp_id, int lane_id) {
  for (int mq = 0; mq < WARP_Q / MMA_M; mq++)
    for (int md = 0; md < DIM / MMA_N; md++) {
      const int row = warp_id * WARP_Q + mq * MMA_M + (lane_id / 4);
      const int col = md * MMA_N + (lane_id % 4) * 2;
      float *regs = O_rmem[mq][md];
      regs[0] /= rowsumexp[mq][0];
      regs[1] /= rowsumexp[mq][0];
      regs[2] /= rowsumexp[mq][1];
      regs[3] /= rowsumexp[mq][1];
      reinterpret_cast<nv_bfloat162 *>(O_ptr + (row + 0) * DIM + col)[0] =
          __float22bfloat162_rn({regs[0], regs[1]});
      reinterpret_cast<nv_bfloat162 *>(O_ptr + (row + 8) * DIM + col)[0] =
          __float22bfloat162_rn({regs[2], regs[3]});
    }
}

} // namespace
__device__ inline void softmax_approx(
    float S[WARP_Q / MMA_M][BLOCK_KV / MMA_N][4],
    float rowmax[WARP_Q / MMA_M][2], float rowsumexp[WARP_Q / MMA_M][2],
    float O_rmem[WARP_Q / MMA_M][DIM / MMA_N][4],
    uint32_t P_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_K][4], int n_kv_tiles) {
  for (int mq = 0; mq < WARP_Q / MMA_M; mq++) {
    float tmax[2] = {NEG_INF, NEG_INF};
    for (int mk = 0; mk < n_kv_tiles; mk++) {
      float *r = S[mq][mk];
      tmax[0] = fmaxf(tmax[0], fmaxf(r[0], r[1]));
      tmax[1] = fmaxf(tmax[1], fmaxf(r[2], r[3]));
    }
    tmax[0] = fmaxf(tmax[0], __shfl_xor_sync(0xffffffffu, tmax[0], 1));
    tmax[0] = fmaxf(tmax[0], __shfl_xor_sync(0xffffffffu, tmax[0], 2));
    tmax[1] = fmaxf(tmax[1], __shfl_xor_sync(0xffffffffu, tmax[1], 1));
    tmax[1] = fmaxf(tmax[1], __shfl_xor_sync(0xffffffffu, tmax[1], 2));
    if (!(tmax[0] > NEG_INF / 2.f || tmax[1] > NEG_INF / 2.f)) {
      for (int t = 0; t < (n_kv_tiles + 1) / 2; t++)
#pragma unroll
        for (int u = 0; u < 4; u++)
          P_rmem[mq][t][u] = 0;
      continue;
    }
    tmax[0] = fmaxf(tmax[0], rowmax[mq][0]);
    tmax[1] = fmaxf(tmax[1], rowmax[mq][1]);
    float rescale[2];
    rescale[0] =
        (rowmax[mq][0] > NEG_INF / 2.f) ? __expf(rowmax[mq][0] - tmax[0]) : 0.f;
    rescale[1] =
        (rowmax[mq][1] > NEG_INF / 2.f) ? __expf(rowmax[mq][1] - tmax[1]) : 0.f;
    for (int md = 0; md < DIM / MMA_N; md++) {
      O_rmem[mq][md][0] *= rescale[0];
      O_rmem[mq][md][1] *= rescale[0];
      O_rmem[mq][md][2] *= rescale[1];
      O_rmem[mq][md][3] *= rescale[1];
    }
    rowmax[mq][0] = tmax[0];
    rowmax[mq][1] = tmax[1];
    float tsum[2] = {0.f, 0.f};
    for (int mk = 0; mk < n_kv_tiles; mk++) {
      float *r = S[mq][mk];
      r[0] = __expf(r[0] - rowmax[mq][0]);
      r[1] = __expf(r[1] - rowmax[mq][0]);
      r[2] = __expf(r[2] - rowmax[mq][1]);
      r[3] = __expf(r[3] - rowmax[mq][1]);
      constexpr float block_length = static_cast<float>(BLOCK_KV);
      tsum[0] += (r[0] + r[1]) * block_length;
      tsum[1] += (r[2] + r[3]) * block_length;
      auto *P = reinterpret_cast<nv_bfloat162 *>(P_rmem[mq][mk / 2]);
      P[(mk % 2) * 2] = __float22bfloat162_rn({r[0], r[1]});
      P[(mk % 2) * 2 + 1] = __float22bfloat162_rn({r[2], r[3]});
    }
    tsum[0] += __shfl_xor_sync(0xffffffffu, tsum[0], 1);
    tsum[0] += __shfl_xor_sync(0xffffffffu, tsum[0], 2);
    tsum[1] += __shfl_xor_sync(0xffffffffu, tsum[1], 1);
    tsum[1] += __shfl_xor_sync(0xffffffffu, tsum[1], 2);
    rowsumexp[mq][0] = rowsumexp[mq][0] * rescale[0] + tsum[0];
    rowsumexp[mq][1] = rowsumexp[mq][1] * rescale[1] + tsum[1];
  }
}
// Kernel C: SM89/SM120 route-first + pipelined exact tiles.
//
// Routing reduces proxy columns directly from MMA fragments (no 64x32 score
// dump), accumulates a compact uint16 exact-block list, and finishes all
// approximate contributions before entering a v5-style K/V pipeline.
// ---------------------------------------------------------------------------
template <bool PIPELINED>
__launch_bounds__(TB_SIZE, PIPELINED ? 2 : 3)
__global__ void sol_attn_optimized_kernel(
    const nv_bfloat16 *__restrict__ Q, const nv_bfloat16 *__restrict__ K,
    const nv_bfloat16 *__restrict__ V, nv_bfloat16 *__restrict__ O,
    const nv_bfloat16 *__restrict__ kc, const nv_bfloat16 *__restrict__ vc,
    const float *__restrict__ threshold, int bs, int seq_len, int num_blocks,
    float scale) {

  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;
  const int bid = blockIdx.x;
  const int bs_id = bid / num_blocks;
  const int q_block_id = bid % num_blocks;
  if (bs_id >= bs)
    return;

  const int q_row0 = q_block_id * BLOCK_Q;
  const float thr = threshold[bs_id * num_blocks + q_block_id];
  const nv_bfloat16 *Q_ptr =
      Q + (static_cast<int64_t>(bs_id) * seq_len + q_row0) * DIM;
  nv_bfloat16 *O_ptr =
      O + (static_cast<int64_t>(bs_id) * seq_len + q_row0) * DIM;
  const nv_bfloat16 *K_base =
      K + static_cast<int64_t>(bs_id) * seq_len * DIM;
  const nv_bfloat16 *V_base =
      V + static_cast<int64_t>(bs_id) * seq_len * DIM;
  const nv_bfloat16 *kc_base =
      kc + static_cast<int64_t>(bs_id) * num_blocks * DIM;
  const nv_bfloat16 *vc_base =
      vc + static_cast<int64_t>(bs_id) * num_blocks * DIM;

  constexpr int BF16_TILE_BYTES =
      BLOCK_KV * DIM * static_cast<int>(sizeof(nv_bfloat16));
  extern __shared__ char smem_raw[];
  const uint32_t Q_smem = __cvta_generic_to_shared(smem_raw);
  const uint32_t K_smem = Q_smem;            // two exact-K buffers
  const uint32_t V_smem =
      K_smem + (PIPELINED ? 2 * BF16_TILE_BYTES : BF16_TILE_BYTES);

  // During routing only the first 32 rows of K_smem are occupied by kc.
  // Put routing reductions in the unused half of that tile.
  float *route_partial =
      reinterpret_cast<float *>(smem_raw + GROUP * DIM * sizeof(nv_bfloat16));
  uint32_t *exact_mask_smem =
      reinterpret_cast<uint32_t *>(route_partial + NUM_WARPS * GROUP);

  // O is private to this query CTA and is not committed until the epilogue.
  // Use its first num_blocks bf16 slots as a compact uint16 routing scratch.
  // Keeping the list out of smem holds the kernel at exactly 48 KiB, allowing
  // two resident CTAs on SM120 at long sequence lengths.
  uint16_t *exact_list = reinterpret_cast<uint16_t *>(O_ptr);
  int *exact_count_smem =
      reinterpret_cast<int *>(exact_mask_smem + 1);

  uint32_t Q_rmem[WARP_Q / MMA_M][DIM / MMA_K][4];
  uint32_t K_rmem[BLOCK_KV / MMA_N][DIM / MMA_K][2];
  uint32_t P_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_K][4] = {};
  uint32_t V_rmem[BLOCK_KV / MMA_K][DIM / MMA_N][2];
  float O_rmem[WARP_Q / MMA_M][DIM / MMA_N][4] = {};
  float rowmax[WARP_Q / MMA_M][2];
  float rowsumexp[WARP_Q / MMA_M][2] = {};
  for (int mq = 0; mq < WARP_Q / MMA_M; mq++) {
    rowmax[mq][0] = -FLT_MAX;
    rowmax[mq][1] = -FLT_MAX;
  }

  uint32_t Q_smem_thread, K_smem_thread, V_smem_thread;
  setup_smem_threads(Q_smem, K_smem, V_smem, warp_id, lane_id, Q_smem_thread,
                     K_smem_thread, V_smem_thread);

  copy_bf16_tile<BLOCK_Q, DIM, TB_SIZE>(Q_smem, Q_ptr, DIM, tid);
  cp_async::commit_group();
  cp_async::wait_group<0>();
  __syncthreads();
  load_Q_rmem(Q_rmem, Q_smem_thread);
  __syncthreads();
  if (tid == 0)
    exact_count_smem[0] = 0;
  __syncthreads();

  constexpr int PROXY_KV_TILES = GROUP / MMA_N;
  constexpr int PROXY_K_TILES = GROUP / MMA_K;

  // Route all groups and accumulate all approximate contributions first.
  for (int g0 = 0; g0 < num_blocks; g0 += GROUP) {
    const int g = min(GROUP, num_blocks - g0);
    copy_bf16_tile_masked<GROUP, DIM, TB_SIZE>(
        K_smem, kc_base + g0 * DIM, DIM, g, tid);
    copy_bf16_tile_masked<GROUP, DIM, TB_SIZE>(
        V_smem, vc_base + g0 * DIM, DIM, g, tid);
    cp_async::commit_group();
    cp_async::wait_group<0>();
    __syncthreads();

    float S[WARP_Q / MMA_M][BLOCK_KV / MMA_N][4];
    for (int mk = 0; mk < PROXY_KV_TILES; mk++)
      for (int md = 0; md < DIM / MMA_K; md += 2) {
        uint32_t addr = K_smem_thread;
        addr += mk * MMA_N * DIM * sizeof(nv_bfloat16);
        addr ^= md * MMA_K * sizeof(nv_bfloat16);
        ldmatrix_x4(K_rmem[mk][md], addr);
      }
    gemm_qk(Q_rmem, K_rmem, S, scale, PROXY_KV_TILES);

    // Each warp owns 16 Q rows. Sum its two 8-row MMA halves, then reduce
    // lanes {0,4,...,28} for each output column.
    for (int mk = 0; mk < PROXY_KV_TILES; mk++) {
      float *r = S[0][mk];
      float sum0 = r[0] + r[2];
      float sum1 = r[1] + r[3];
      sum0 += __shfl_xor_sync(0xffffffffu, sum0, 4);
      sum1 += __shfl_xor_sync(0xffffffffu, sum1, 4);
      sum0 += __shfl_xor_sync(0xffffffffu, sum0, 8);
      sum1 += __shfl_xor_sync(0xffffffffu, sum1, 8);
      sum0 += __shfl_xor_sync(0xffffffffu, sum0, 16);
      sum1 += __shfl_xor_sync(0xffffffffu, sum1, 16);
      if (lane_id < 4) {
        const int col0 = mk * MMA_N + lane_id * 2;
        route_partial[warp_id * GROUP + col0] = sum0;
        route_partial[warp_id * GROUP + col0 + 1] = sum1;
      }
    }
    __syncthreads();

    // Warp 0 computes one route predicate per lane and compacts exact IDs
    // with a ballot/prefix rank.
    if (warp_id == 0) {
      bool is_exact = false;
      if (lane_id < g) {
        float colsum = 0.f;
#pragma unroll
        for (int w = 0; w < NUM_WARPS; ++w)
          colsum += route_partial[w * GROUP + lane_id];
        const int kv_block = g0 + lane_id;
        is_exact = (colsum * (1.f / 64.f) > thr) ||
                   (abs(q_block_id - kv_block) <= 1);
      }
      const uint32_t mask = __ballot_sync(0xffffffffu, is_exact);
      if (lane_id == 0)
        exact_mask_smem[0] = mask;
      const int count = __popc(mask);
      int base = 0;
      if (lane_id == 0) {
        base = exact_count_smem[0];
        exact_count_smem[0] = base + count;
      }
      base = __shfl_sync(0xffffffffu, base, 0);
      if (is_exact) {
        const uint32_t lower =
            lane_id == 0 ? 0u : (mask & ((1u << lane_id) - 1u));
        exact_list[base + __popc(lower)] =
            static_cast<uint16_t>(g0 + lane_id);
      }
    }
    __syncthreads();

    const uint32_t exact_mask = exact_mask_smem[0];
    const int approx_count = g - __popc(exact_mask);
    if (approx_count > 0) {
      for (int mk = 0; mk < PROXY_KV_TILES; mk++) {
        const int col0 = mk * MMA_N + (lane_id % 4) * 2;
        float *r = S[0][mk];
        if (col0 >= g || ((exact_mask >> col0) & 1u)) {
          r[0] = NEG_INF;
          r[2] = NEG_INF;
        }
        if (col0 + 1 >= g || ((exact_mask >> (col0 + 1)) & 1u)) {
          r[1] = NEG_INF;
          r[3] = NEG_INF;
        }
      }
      softmax_approx(S, rowmax, rowsumexp, O_rmem, P_rmem, PROXY_KV_TILES);
      for (int mk = 0; mk < PROXY_K_TILES; mk++)
        for (int md = 0; md < DIM / MMA_N; md += 2) {
          uint32_t addr = V_smem_thread;
          addr += mk * MMA_K * DIM * sizeof(nv_bfloat16);
          addr ^= md * MMA_N * sizeof(nv_bfloat16);
          ldmatrix_x4_trans(V_rmem[mk][md], addr);
        }
      gemm_pv(P_rmem, V_rmem, O_rmem, PROXY_K_TILES);
    }
    __syncthreads();
  }

  const int exact_count = exact_count_smem[0];
  if constexpr (PIPELINED) {
    auto load_exact_K = [&](int e) {
      if (e < exact_count) {
        const int kv_block = static_cast<int>(exact_list[e]);
        const uint32_t destination = K_smem + (e % 2) * BF16_TILE_BYTES;
        copy_bf16_tile<BLOCK_KV, DIM, TB_SIZE,
                       cp_async::PrefetchMode::kPrefetch>(
            destination, K_base + kv_block * BLOCK_KV * DIM, DIM, tid);
      }
      cp_async::commit_group();
    };
    auto load_exact_V = [&](int e) {
      const int kv_block = static_cast<int>(exact_list[e]);
      copy_bf16_tile<BLOCK_KV, DIM, TB_SIZE,
                     cp_async::PrefetchMode::kPrefetch>(
          V_smem, V_base + kv_block * BLOCK_KV * DIM, DIM, tid);
      cp_async::commit_group();
    };

    // attention_v5 pipeline over the compact, potentially non-contiguous list.
    load_exact_K(0);
    for (int e = 0; e < exact_count; ++e) {
      float S_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_N][4];
      __syncthreads();
      load_exact_V(e);
      cp_async::wait_group<1>();
      __syncthreads();

      for (int mk = 0; mk < BLOCK_KV / MMA_N; mk++)
        for (int md = 0; md < DIM / MMA_K; md += 2) {
          uint32_t addr = K_smem_thread + (e % 2) * BF16_TILE_BYTES;
          addr += mk * MMA_N * DIM * sizeof(nv_bfloat16);
          addr ^= md * MMA_K * sizeof(nv_bfloat16);
          ldmatrix_x4(K_rmem[mk][md], addr);
        }
      gemm_qk(Q_rmem, K_rmem, S_rmem, scale, BLOCK_KV / MMA_N);
      load_exact_K(e + 1);
      softmax_exact_tile(S_rmem, rowmax, rowsumexp, O_rmem, P_rmem);

      cp_async::wait_group<1>();
      __syncthreads();
      load_V_rmem_full(V_rmem, V_smem_thread);
      gemm_pv(P_rmem, V_rmem, O_rmem, BLOCK_KV / MMA_K);
    }
  } else {
    // Occupancy-first SM89 path: 32 KiB smem permits three resident CTAs.
    for (int e = 0; e < exact_count; ++e) {
      const int kv_block = static_cast<int>(exact_list[e]);
      copy_bf16_tile<BLOCK_KV, DIM, TB_SIZE,
                       cp_async::PrefetchMode::kPrefetch>(
          K_smem, K_base + kv_block * BLOCK_KV * DIM, DIM, tid);
      copy_bf16_tile<BLOCK_KV, DIM, TB_SIZE,
                       cp_async::PrefetchMode::kPrefetch>(
          V_smem, V_base + kv_block * BLOCK_KV * DIM, DIM, tid);
      cp_async::commit_group();
      cp_async::wait_group<0>();
      __syncthreads();

      float S_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_N][4];
      load_K_rmem_full(K_rmem, K_smem_thread);
      gemm_qk(Q_rmem, K_rmem, S_rmem, scale, BLOCK_KV / MMA_N);
      softmax_exact_tile(S_rmem, rowmax, rowsumexp, O_rmem, P_rmem);
      load_V_rmem_full(V_rmem, V_smem_thread);
      gemm_pv(P_rmem, V_rmem, O_rmem, BLOCK_KV / MMA_K);
      __syncthreads();
    }
  }

  write_O(O_ptr, O_rmem, rowsumexp, warp_id, lane_id);
}

namespace {

__global__ void sol_block_summaries_kernel(
    const nv_bfloat16 *__restrict__ k, const nv_bfloat16 *__restrict__ v,
    nv_bfloat16 *__restrict__ kc, nv_bfloat16 *__restrict__ vc,
    int sequence_length, int num_blocks) {
  const int batch_head = blockIdx.x / num_blocks;
  const int block = blockIdx.x % num_blocks;
  const int channel = threadIdx.x;
  if (channel >= DIM)
    return;

  const int64_t input_base =
      (static_cast<int64_t>(batch_head) * sequence_length +
       block * BLOCK_KV) *
      DIM;
  float key_sum = 0.0f;
  float value_sum = 0.0f;
#pragma unroll
  for (int row = 0; row < BLOCK_KV; ++row) {
    key_sum += __bfloat162float(k[input_base + row * DIM + channel]);
    value_sum += __bfloat162float(v[input_base + row * DIM + channel]);
  }

  const int64_t output =
      (static_cast<int64_t>(batch_head) * num_blocks + block) * DIM +
      channel;
  kc[output] = __float2bfloat16_rn(key_sum * (1.0f / BLOCK_KV));
  vc[output] = __float2bfloat16_rn(value_sum);
}

__global__ void sol_centroid_stats_kernel(
    const nv_bfloat16 *__restrict__ kc, float *__restrict__ mean,
    float *__restrict__ variance, int num_blocks) {
  const int batch_head = blockIdx.x;
  const int channel = threadIdx.x;
  if (channel >= DIM)
    return;

  const int64_t base = static_cast<int64_t>(batch_head) * num_blocks * DIM;
  float sum = 0.0f;
  float square_sum = 0.0f;
  for (int block = 0; block < num_blocks; ++block) {
    const float value =
        __bfloat162float(kc[base + static_cast<int64_t>(block) * DIM +
                           channel]);
    sum += value;
    square_sum += value * value;
  }
  const float average = sum / num_blocks;
  const int64_t output = static_cast<int64_t>(batch_head) * DIM + channel;
  mean[output] = average;
  variance[output] =
      fmaxf(square_sum / num_blocks - average * average, 0.0f);
}

__device__ __forceinline__ float block_sum(float value, float *shared) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_down_sync(0xffffffffu, value, offset);
  if (lane == 0)
    shared[warp] = value;
  __syncthreads();

  value = threadIdx.x < 4 ? shared[lane] : 0.0f;
  if (warp == 0) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__global__ void sol_threshold_kernel(
    const nv_bfloat16 *__restrict__ q, const float *__restrict__ key_mean,
    const float *__restrict__ key_variance, float *__restrict__ threshold,
    int sequence_length, int num_blocks, float tau, float scale) {
  __shared__ float reduction[8];
  const int batch_head = blockIdx.x / num_blocks;
  const int block = blockIdx.x % num_blocks;
  const int channel = threadIdx.x;

  const int64_t q_base =
      (static_cast<int64_t>(batch_head) * sequence_length +
       block * BLOCK_Q) *
      DIM;
  float query_sum = 0.0f;
#pragma unroll
  for (int row = 0; row < BLOCK_Q; ++row)
    query_sum +=
        __bfloat162float(q[q_base + static_cast<int64_t>(row) * DIM +
                          channel]);
  const float query_mean = query_sum * (1.0f / BLOCK_Q);
  const int64_t stat = static_cast<int64_t>(batch_head) * DIM + channel;
  const float mean_part = scale * query_mean * key_mean[stat];
  const float variance_part =
      scale * scale * query_mean * query_mean * key_variance[stat];

  const float reduced_mean = block_sum(mean_part, reduction);
  const float reduced_variance = block_sum(variance_part, reduction + 4);
  if (threadIdx.x == 0) {
    threshold[static_cast<int64_t>(batch_head) * num_blocks + block] =
        reduced_mean + tau * sqrtf(reduced_variance + 1.0e-6f);
  }
}

} // namespace

extern "C" void launch_sol_attention_bf16(
    const void *q, const void *k, const void *v, void *output, void *kc,
    void *vc, void *key_mean, void *key_variance, void *threshold, int batch,
    int heads, int sequence_length, float tau, float scale,
    cudaStream_t stream) {
  const int batch_heads = batch * heads;
  const int num_blocks = sequence_length / BLOCK_Q;
  const auto *q_bf16 = static_cast<const nv_bfloat16 *>(q);
  const auto *k_bf16 = static_cast<const nv_bfloat16 *>(k);
  const auto *v_bf16 = static_cast<const nv_bfloat16 *>(v);

  sol_block_summaries_kernel<<<batch_heads * num_blocks, DIM, 0, stream>>>(
      k_bf16, v_bf16, static_cast<nv_bfloat16 *>(kc),
      static_cast<nv_bfloat16 *>(vc), sequence_length, num_blocks);
  sol_centroid_stats_kernel<<<batch_heads, DIM, 0, stream>>>(
      static_cast<const nv_bfloat16 *>(kc), static_cast<float *>(key_mean),
      static_cast<float *>(key_variance), num_blocks);
  sol_threshold_kernel<<<batch_heads * num_blocks, DIM, 0, stream>>>(
      q_bf16, static_cast<const float *>(key_mean),
      static_cast<const float *>(key_variance),
      static_cast<float *>(threshold), sequence_length, num_blocks, tau,
      scale);

  constexpr int shared_memory =
      3 * BLOCK_KV * DIM * static_cast<int>(sizeof(nv_bfloat16));
  auto kernel = sol_attn_optimized_kernel<true>;
  const cudaError_t attribute_error = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory);
  if (attribute_error != cudaSuccess) {
    throw std::runtime_error(
        std::string("SOL attention shared-memory configuration failed: ") +
        cudaGetErrorString(attribute_error));
  }

  kernel<<<batch_heads * num_blocks, TB_SIZE, shared_memory, stream>>>(
      q_bf16, k_bf16, v_bf16, static_cast<nv_bfloat16 *>(output),
      static_cast<const nv_bfloat16 *>(kc),
      static_cast<const nv_bfloat16 *>(vc), static_cast<const float *>(threshold),
      batch_heads, sequence_length,
      num_blocks, scale);

  const cudaError_t launch_error = cudaGetLastError();
  if (launch_error != cudaSuccess) {
    throw std::runtime_error(
        std::string("SOL attention kernel launch failed: ") +
        cudaGetErrorString(launch_error));
  }
}
