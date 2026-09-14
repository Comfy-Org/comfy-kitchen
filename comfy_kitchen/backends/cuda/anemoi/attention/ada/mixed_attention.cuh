/*
 * Copyright (c) 2025 by SpargeAttn team.
 * Copyright (c) 2026 mixed-attention project contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * This project-owned SM89 derivative starts from the locked SpargeAttention
 * qk_int_sv_f8_block_sparse_attn_kernel (commit
 * ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a).  It retains Sparge/Sage's
 * shared-memory wrappers, INT8 QK path, online softmax, E4M3 P/V path and
 * stage-local FP16 PV accumulator.  The project adds homogeneous precision
 * lists, the raw-FP16 rescue phase, and the in-register precision boundary.
 */

#pragma once

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "native_tensor.h"

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "primitives/cp_async.cuh"
#include "primitives/math.cuh"
#include "primitives/mma.cuh"
#include "primitives/permuted_smem.cuh"
#include "primitives/qattn/attn_utils.cuh"
#include "attention_decl.cuh"

#ifndef MPA_DENSE_SEQUENTIAL
#define MPA_DENSE_SEQUENTIAL 0
#endif
#ifndef MPA_STORE_LSE
#define MPA_STORE_LSE 1
#endif

#if !defined(MPA_ATTENTION_KERNEL_ENTRY) || !defined(MPA_ATTENTION_LAUNCH_ENTRY)
#if defined(MPA_PACKED_RASTER_MODE)
#define MPA_ATTENTION_KERNEL_ENTRY mixed_attention_sm89_packed_raster_kernel
#define MPA_ATTENTION_LAUNCH_ENTRY launch_mixed_attention_sm89_packed_raster
#else
#define MPA_ATTENTION_KERNEL_ENTRY mixed_attention_sm89_kernel
#define MPA_ATTENTION_LAUNCH_ENTRY launch_mixed_attention_sm89
#endif
#endif

namespace mpa::attention {

#ifndef MPA_CTA_Q
#define MPA_CTA_Q 128
#endif
constexpr uint32_t kCtaQ = MPA_CTA_Q;
static_assert(kCtaQ == 64 || kCtaQ == 128, "Q CTA must be 64 or 128");
constexpr uint32_t kCtaK = 64;
#ifndef MPA_FP16_CTA_K
#define MPA_FP16_CTA_K 64
#endif
constexpr uint32_t kFp16CtaK = MPA_FP16_CTA_K;
static_assert(
    kFp16CtaK == 32 || kFp16CtaK == 64,
    "FP16 stage K must be 32 or 64");
constexpr uint32_t kFp16StagesPerRasterPatch = kCtaQ / kFp16CtaK;
#ifndef MPA_FP16_PIPELINE
#define MPA_FP16_PIPELINE 0
#endif
constexpr bool kFp16Pipeline = MPA_FP16_PIPELINE != 0;
static_assert(!kFp16Pipeline || kFp16CtaK == 32);
#ifndef MPA_WARP_Q
#define MPA_WARP_Q 32
#endif
constexpr uint32_t kWarpQ = MPA_WARP_Q;
static_assert(
    kWarpQ == 16 || kWarpQ == 32,
    "each physical warp must own 16 or 32 query rows");
constexpr uint32_t kWarpK = 64;
constexpr uint32_t kPackInt8 = 16;
constexpr uint32_t kPackFp8 = 16;
constexpr uint32_t kPackHalf = 8;
constexpr uint32_t kMmaQkM = 16;
constexpr uint32_t kMmaQkN = 16;
constexpr uint32_t kMmaInt8K = 32;
constexpr uint32_t kMmaHalfK = 16;
constexpr uint32_t kMmaPvN = 16;

template <uint32_t HeadDim>
using LowQKSmem = smem_t<
    (HeadDim == 64 ? SwizzleMode::k64B : SwizzleMode::k128B),
    HeadDim / kPackInt8>;

using LowVSmem = smem_t<SwizzleMode::k64B, kCtaK / kPackFp8>;

template <uint32_t HeadDim>
using HalfQKVSmem = smem_t<SwizzleMode::k128B, HeadDim / kPackHalf>;

template <uint32_t HeadDim>
using OutputSmem = smem_t<SwizzleMode::k128B, HeadDim / kPackHalf>;

template <uint32_t NumTilesQ, uint32_t NumTilesV>
__device__ __forceinline__ void scale_ro_by_v_scale(
    float (&ro)[NumTilesQ][NumTilesV][8],
    const float* v_scale_base,
    float common_scale) {
  const uint32_t lane_id = get_lane_id();
  float scales[4];
#pragma unroll
  for (uint32_t fv = 0; fv < NumTilesV; ++fv) {
    reinterpret_cast<float2*>(scales)[0] =
        *reinterpret_cast<const float2*>(
            v_scale_base + fv * 16 + (lane_id % 4) * 2);
    reinterpret_cast<float2*>(scales)[1] =
        *reinterpret_cast<const float2*>(
            v_scale_base + fv * 16 + 8 + (lane_id % 4) * 2);
#pragma unroll
    for (uint32_t index = 0; index < 4; ++index) {
      scales[index] *= common_scale;
    }
#pragma unroll
    for (uint32_t fq = 0; fq < NumTilesQ; ++fq) {
      ro[fq][fv][0] *= scales[0];
      ro[fq][fv][1] *= scales[1];
      ro[fq][fv][2] *= scales[0];
      ro[fq][fv][3] *= scales[1];
      ro[fq][fv][4] *= scales[2];
      ro[fq][fv][5] *= scales[3];
      ro[fq][fv][6] *= scales[2];
      ro[fq][fv][7] *= scales[3];
    }
  }
}

template <uint32_t NumTilesQ, uint32_t NumTilesV>
__device__ __forceinline__ void leave_fp8_probability_and_v_domains(
    float (&ro)[NumTilesQ][NumTilesV][8],
    float (&d)[NumTilesQ][2],
    const float* v_scale_base) {
  scale_ro_by_v_scale(ro, v_scale_base, S_FP8_OFFSET_EXP_INV);
#pragma unroll
  for (uint32_t fq = 0; fq < NumTilesQ; ++fq) {
#pragma unroll
    for (uint32_t row = 0; row < 2; ++row) {
      d[fq][row] *= S_FP8_OFFSET_EXP_INV;
    }
  }
}

// Give ptxas an explicit register-to-register ownership boundary between the
// ordered INT8 and FP16 phases.  The self moves preserve the online-softmax
// state exactly while creating fresh definitions after every INT8-local value
// has died.  Unlike a device call or a memory-backed handoff, this keeps the
// only cross-phase payload (ro/m/d) in registers.
template <uint32_t NumTilesQ, uint32_t NumTilesV>
__device__ __forceinline__ void handoff_online_softmax_registers(
    float (&ro)[NumTilesQ][NumTilesV][8],
    float (&m)[NumTilesQ][2],
    float (&d)[NumTilesQ][2]) {
#pragma unroll
  for (uint32_t fq = 0; fq < NumTilesQ; ++fq) {
#pragma unroll
    for (uint32_t fv = 0; fv < NumTilesV; ++fv) {
#pragma unroll
      for (uint32_t element = 0; element < 8; ++element) {
        asm volatile("mov.f32 %0, %0;" : "+f"(ro[fq][fv][element]));
      }
    }
#pragma unroll
    for (uint32_t row = 0; row < 2; ++row) {
      asm volatile("mov.f32 %0, %0;" : "+f"(m[fq][row]));
      asm volatile("mov.f32 %0, %0;" : "+f"(d[fq][row]));
    }
  }
}

template <uint32_t NumTilesQ, uint32_t NumTilesV>
__device__ __forceinline__ void normalize_d_inplace(
    float ro[][NumTilesV][8],
    float d[][2]) {
#pragma unroll
  for (uint32_t fq = 0; fq < NumTilesQ; ++fq) {
#pragma unroll
    for (uint32_t row = 0; row < 2; ++row) {
      d[fq][row] += __shfl_xor_sync(0xffffffff, d[fq][row], 1);
      d[fq][row] += __shfl_xor_sync(0xffffffff, d[fq][row], 2);
      d[fq][row] = math::ptx_rcp(d[fq][row]);
    }
  }
#pragma unroll
  for (uint32_t fq = 0; fq < NumTilesQ; ++fq) {
#pragma unroll
    for (uint32_t fv = 0; fv < NumTilesV; ++fv) {
#pragma unroll
      for (uint32_t element = 0; element < 8; ++element) {
        ro[fq][fv][element] *= d[fq][(element % 4) / 2];
      }
    }
  }
}

template <uint32_t HeadDim, uint32_t NumTilesQ>
__device__ __forceinline__ void restore_smoothed_low_rowmax(
    const HalfQKVSmem<HeadDim>& smem_q,
    const half* k_mean,
    float (&m)[NumTilesQ][2],
    uint32_t query_block,
    uint32_t qo_len,
    float base2_softmax_scale) {
  const uint32_t lane_id = get_lane_id();
  const uint32_t warp_q = get_warp_idx_q<kCtaQ / kWarpQ, kCtaK / kWarpK>();
  const uint32_t lane_in_row = lane_id % 4;

#pragma unroll
  for (uint32_t fq = 0; fq < NumTilesQ; ++fq) {
#pragma unroll
    for (uint32_t row_half = 0; row_half < 2; ++row_half) {
      const uint32_t local_row =
          warp_q * kWarpQ + fq * kMmaQkM + lane_id / 4 + row_half * 8;
      const uint32_t global_row = query_block * kCtaQ + local_row;
      float dot = 0.0f;
      const bool valid_row = global_row < qo_len;
      if (valid_row) {
#pragma unroll
        for (uint32_t pack = lane_in_row; pack < HeadDim / kPackHalf; pack += 4) {
          const b128_t q_pack =
              smem_q.base[smem_q.get_permuted_offset(local_row, pack)];
          const b128_t mean_pack = reinterpret_cast<const b128_t*>(k_mean)[pack];
          const half* q_values = reinterpret_cast<const half*>(&q_pack);
          const half* mean_values = reinterpret_cast<const half*>(&mean_pack);
#pragma unroll
          for (uint32_t element = 0; element < kPackHalf; ++element) {
            dot = fmaf(
                __half2float(q_values[element]),
                __half2float(mean_values[element]),
                dot);
          }
        }
      }
      // Every lane in the warp must execute both shuffles.  Query-tail
      // predicates vary between four-lane row groups, so placing these under
      // valid_row while using a full mask would be undefined.
      dot += __shfl_xor_sync(0xffffffff, dot, 1);
      dot += __shfl_xor_sync(0xffffffff, dot, 2);
      if (valid_row) {
        m[fq][row_half] = fmaf(dot, base2_softmax_scale, m[fq][row_half]);
      }
    }
  }
}

union Half2Bits {
  half2 value;
  uint32_t bits;
};

template <uint32_t NumTilesQ, uint32_t NumTilesK>
__device__ __forceinline__ void pack_rs_fp16_inplace(
    float rs[][NumTilesK][8]) {
#pragma unroll
  for (uint32_t fq = 0; fq < NumTilesQ; ++fq) {
#pragma unroll
    for (uint32_t fk = 0; fk < NumTilesK; ++fk) {
      // Materialize all four words before overwriting the first half of the
      // FP32 fragment.  The second half then dies and the first half becomes
      // the packed FP16 operand without a second 32-register D128 array.
      Half2Bits packed[4];
#pragma unroll
      for (uint32_t pair = 0; pair < 4; ++pair) {
        packed[pair].value = __float22half2_rn(
            reinterpret_cast<float2*>(rs[fq][fk])[pair]);
      }
#pragma unroll
      for (uint32_t pair = 0; pair < 4; ++pair) {
        rs[fq][fk][pair] = __uint_as_float(packed[pair].bits);
      }
    }
  }
}

// The donor helper keeps every V-tile FP16 stage accumulator live at once.
// That is ideal for instruction scheduling in a homogeneous kernel, but the
// D128 mixed kernel already carries 128 FP32 output values per lane and would
// spill.  Process one compile-time V tile at a time instead.  Each tile still
// accumulates all K=64 in FP16 and is promoted exactly once at the physical
// stage boundary, preserving the mixed-precision accumulator contract.
template <
    uint32_t Fv, uint32_t NumTilesQ, uint32_t NumTilesK,
    uint32_t NumTilesV, SwizzleMode Swizzle, uint32_t Stride,
    bool SplitN8 = false>
__device__ __forceinline__ void compute_fp16_sv_stage_tilewise(
    const smem_t<Swizzle, Stride>& smem_v,
    float rs_packed[][NumTilesK][8],
    float ro[][NumTilesV][8]) {
  if constexpr (Fv < NumTilesV) {
    {
      const uint32_t lane_id = get_lane_id();
      if constexpr (SplitN8) {
        static_assert(NumTilesQ == 1, "m16n8 split requires one Q tile");
        {
          uint32_t ro_half[2];

#pragma unroll
          for (uint32_t fk = 0; fk < NumTilesK; ++fk) {
            uint32_t rv[2];
            const uint32_t v_offset = smem_v.get_permuted_offset(
                (lane_id & 15U) + fk * kMmaHalfK, Fv * 2);
            b128_t* smem_ptr = smem_v.base + v_offset;
            const uint32_t smem_int_ptr =
                static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
                "{%0, %1}, [%2];\n"
                : "=r"(rv[0]), "=r"(rv[1])
                : "r"(smem_int_ptr));
            uint32_t rs_fragment[4];
#pragma unroll
            for (uint32_t word = 0; word < 4; ++word) {
              rs_fragment[word] = __float_as_uint(rs_packed[0][fk][word]);
            }
            if (fk == 0) {
              mma::mma_sync_m16n8k16_row_col_f16f16f16<
                  mma::MMAMode::kInit>(ro_half, rs_fragment, rv);
            } else {
              mma::mma_sync_m16n8k16_row_col_f16f16f16<
                  mma::MMAMode::kInplaceUpdate>(
                  ro_half, rs_fragment, rv);
            }
          }

#pragma unroll
          for (uint32_t pair = 0; pair < 2; ++pair) {
            const half2 value =
                reinterpret_cast<const half2*>(ro_half)[pair];
            ro[0][Fv][pair * 2] += __half2float(value.x);
            ro[0][Fv][pair * 2 + 1] += __half2float(value.y);
          }
        }

        {
          uint32_t ro_half[2];

#pragma unroll
          for (uint32_t fk = 0; fk < NumTilesK; ++fk) {
            uint32_t rv[2];
            const uint32_t v_offset = smem_v.get_permuted_offset(
                (lane_id & 15U) + fk * kMmaHalfK, Fv * 2 + 1);
            b128_t* smem_ptr = smem_v.base + v_offset;
            const uint32_t smem_int_ptr =
                static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
                "{%0, %1}, [%2];\n"
                : "=r"(rv[0]), "=r"(rv[1])
                : "r"(smem_int_ptr));
            uint32_t rs_fragment[4];
#pragma unroll
            for (uint32_t word = 0; word < 4; ++word) {
              rs_fragment[word] = __float_as_uint(rs_packed[0][fk][word]);
            }
            if (fk == 0) {
              mma::mma_sync_m16n8k16_row_col_f16f16f16<
                  mma::MMAMode::kInit>(ro_half, rs_fragment, rv);
            } else {
              mma::mma_sync_m16n8k16_row_col_f16f16f16<
                  mma::MMAMode::kInplaceUpdate>(
                  ro_half, rs_fragment, rv);
            }
          }

#pragma unroll
          for (uint32_t pair = 0; pair < 2; ++pair) {
            const half2 value =
                reinterpret_cast<const half2*>(ro_half)[pair];
            ro[0][Fv][4 + pair * 2] += __half2float(value.x);
            ro[0][Fv][5 + pair * 2] += __half2float(value.y);
          }
        }
      } else {
        uint32_t ro_stage[NumTilesQ][4];

#pragma unroll
        for (uint32_t fk = 0; fk < NumTilesK; ++fk) {
          uint32_t rv[4];
          const uint32_t v_offset = smem_v.get_permuted_offset(
              lane_id % 16 + fk * kMmaHalfK,
              lane_id / 16 + Fv * (kMmaPvN / kPackHalf));
          smem_v.ldmatrix_m8n8x4_trans(v_offset, rv);
#pragma unroll
          for (uint32_t fq = 0; fq < NumTilesQ; ++fq) {
            uint32_t rs_fragment[4];
#pragma unroll
            for (uint32_t word = 0; word < 4; ++word) {
              rs_fragment[word] = __float_as_uint(rs_packed[fq][fk][word]);
            }
            if (fk == 0) {
              mma::mma_sync_m16n16k16_row_col_f16f16f16<
                  mma::MMAMode::kInit>(ro_stage[fq], rs_fragment, rv);
            } else {
              mma::mma_sync_m16n16k16_row_col_f16f16f16<
                  mma::MMAMode::kInplaceUpdate>(
                  ro_stage[fq], rs_fragment, rv);
            }
          }
        }

#pragma unroll
        for (uint32_t fq = 0; fq < NumTilesQ; ++fq) {
#pragma unroll
          for (uint32_t pair = 0; pair < 4; ++pair) {
            const half2 value =
                reinterpret_cast<const half2*>(ro_stage[fq])[pair];
            ro[fq][Fv][pair * 2] += __half2float(value.x);
            ro[fq][Fv][pair * 2 + 1] += __half2float(value.y);
          }
        }
      }
    }
    compute_fp16_sv_stage_tilewise<
        Fv + 1, NumTilesQ, NumTilesK, NumTilesV, Swizzle, Stride,
        SplitN8>(
        smem_v, rs_packed, ro);
  }
}

// D128 mixed kernels also need to lower the donor FP8 PV peak.  The all-FP8
// family keeps the donor helper unchanged; only the compile-time mixed family
// changes loop order so one V tile's stage accumulator is live at once. CUDA
// 12.8 added FP8-input/FP16-accumulator MMA; CUDA 12.4-12.7 use the equivalent
// FP32-accumulator instruction instead of compiling the unsupported path to a
// device breakpoint.
template <
    uint32_t Fv, uint32_t NumTilesQ, uint32_t NumTilesK,
    uint32_t NumTilesV, SwizzleMode Swizzle, uint32_t Stride>
__device__ __forceinline__ void compute_fp8_sv_stage_tilewise(
    const smem_t<Swizzle, Stride>& smem_v,
    uint32_t rs_fp8[][NumTilesK / 2][4],
    float ro[][NumTilesV][8]) {
  if constexpr (Fv < NumTilesV) {
    {
#ifdef MMA_F8F8F16_M16N8K16_ENABLED
      uint32_t ro_stage[NumTilesQ][4];
#else
      float ro_stage[NumTilesQ][8];
#endif
      const uint32_t lane_id = get_lane_id();
      const uint32_t row_base = lane_id % 8 + (lane_id / 16) * 8;
      const uint32_t col_base = (lane_id / 8) % 2;

#pragma unroll
      for (uint32_t fk = 0; fk < NumTilesK / 2; ++fk) {
        uint32_t rv[4];
        const uint32_t v_offset = smem_v.get_permuted_offset(
            row_base + Fv * kMmaPvN, col_base + fk * 2);
        smem_v.ldmatrix_m8n8x4(v_offset, rv);
#pragma unroll
        for (uint32_t fq = 0; fq < NumTilesQ; ++fq) {
#ifdef MMA_F8F8F16_M16N8K16_ENABLED
          if (fk == 0) {
            mma::mma_sync_m16n16k32_row_col_f8f8f16<
                mma::MMAMode::kInit>(ro_stage[fq], rs_fp8[fq][fk], rv);
          } else {
            mma::mma_sync_m16n16k32_row_col_f8f8f16<
                mma::MMAMode::kInplaceUpdate>(
                ro_stage[fq], rs_fp8[fq][fk], rv);
          }
#else
          if (fk == 0) {
            mma::mma_sync_m16n16k32_row_col_f8f8f32<
                mma::MMAMode::kInit>(
                ro_stage[fq], rs_fp8[fq][fk], rv);
          } else {
            mma::mma_sync_m16n16k32_row_col_f8f8f32<
                mma::MMAMode::kInplaceUpdate>(
                ro_stage[fq], rs_fp8[fq][fk], rv);
          }
#endif
        }
      }

#pragma unroll
      for (uint32_t fq = 0; fq < NumTilesQ; ++fq) {
#ifdef MMA_F8F8F16_M16N8K16_ENABLED
#pragma unroll
        for (uint32_t pair = 0; pair < 4; ++pair) {
          float unpacked[2];
          unpack_half2_from_uint32_to_float(unpacked, ro_stage[fq][pair]);
          ro[fq][Fv][pair * 2] += unpacked[0];
          ro[fq][Fv][pair * 2 + 1] += unpacked[1];
        }
#else
#pragma unroll
        for (uint32_t element = 0; element < 8; ++element) {
          ro[fq][Fv][element] += ro_stage[fq][element];
        }
#endif
      }
    }
    compute_fp8_sv_stage_tilewise<
        Fv + 1, NumTilesQ, NumTilesK, NumTilesV, Swizzle, Stride>(
        smem_v, rs_fp8, ro);
  }
}

// Keep every Q64 precision family on the same three-CTA scheduling target.
// Pure INT8 then remains spill-free while its standalone resource envelope
// matches the INT8 phase inside the ordered INT8 -> FP16 specialization.
template <uint32_t HeadDim, bool HasFp8, bool HasFp16, bool SmoothK>
__global__ __launch_bounds__(
    32 * (kCtaQ / kWarpQ) * (kCtaK / kWarpK),
    kCtaQ == 64 ? 3 : 1)
void MPA_ATTENTION_KERNEL_ENTRY(
    int8_t* __restrict__ q8,
    int8_t* __restrict__ k8,
    __nv_fp8_e4m3* __restrict__ v8,
    half* __restrict__ q16,
    half* __restrict__ k16,
    half* __restrict__ v16,
    half* __restrict__ k_mean,
    half* __restrict__ output,
    int32_t* __restrict__ fp8_lut,
    int32_t* __restrict__ fp8_count,
    int32_t* __restrict__ fp16_lut,
    int32_t* __restrict__ fp16_count,
    float* __restrict__ q_scale,
    float* __restrict__ k_scale,
    float* __restrict__ v_scale,
#if defined(MPA_K64_BLOCK_MODE)
    const int32_t* __restrict__ valid_k_counts,
    float* __restrict__ lse,
    uint32_t fp16_prefix_stages,
#endif
    uint32_t qo_len,
    uint32_t kv_len,
    uint32_t padded_kv_len,
    uint32_t num_kv_groups,
    float softmax_scale) {
  static_assert(HeadDim == 64 || HeadDim == 128);
  static_assert(HasFp8 || HasFp16);
  static_assert(!SmoothK || (HasFp8 && HasFp16));

  constexpr uint32_t num_warps_q = kCtaQ / kWarpQ;
  constexpr uint32_t num_warps_k = kCtaK / kWarpK;
  constexpr uint32_t num_warps = num_warps_q * num_warps_k;
  constexpr uint32_t num_tiles_q = kWarpQ / kMmaQkM;
  constexpr uint32_t num_tiles_k = kWarpK / kMmaQkN;
  constexpr uint32_t num_tiles_v = HeadDim / kMmaPvN;
  constexpr uint32_t low_qk_inner = HeadDim / kMmaInt8K;
  constexpr uint32_t half_qk_inner = HeadDim / kMmaHalfK;

  extern __shared__ int8_t smem[];
  uint32_t lane_id = get_lane_id();
  const uint32_t warp_id = get_warp_id();
  const uint32_t batch_id = blockIdx.z;
  const uint32_t query_block = blockIdx.x;
  const uint32_t head_id = blockIdx.y;
  const uint32_t num_qo_heads = gridDim.y;
  const uint32_t kv_head = head_id / num_kv_groups;
  const uint32_t num_kv_heads = num_qo_heads / num_kv_groups;
  const uint32_t num_query_blocks = gridDim.x;
  const uint32_t num_physical_stages = div_ceil(kv_len, kCtaK);
  const uint32_t metadata_row =
      (batch_id * num_qo_heads + head_id) * num_query_blocks + query_block;

#if MPA_DENSE_SEQUENTIAL
  const uint32_t low_iterations = HasFp8 ? num_physical_stages : 0;
  const uint32_t initial_high_iterations = 0;
#else
  const uint32_t low_iterations = HasFp8 ? fp8_count[metadata_row] : 0;
  const uint32_t initial_high_iterations = HasFp16
      ? (HasFp8 && kCtaQ == 128 && low_iterations != 0
             ? 0
             : fp16_count[metadata_row])
      : 0;
#endif
  if (__builtin_expect(
          low_iterations == 0 && initial_high_iterations == 0, false)) {
#if defined(MPA_K64_BLOCK_MODE)
    // Final production output assembly owns empty-row zeroing. Preserve the
    // legacy store shape behind a host-rejected prefix sentinel because ptxas
    // uses this cold block while coloring the register-heavy nonempty path.
    const uint32_t query_begin = query_block * kCtaQ;
    const uint32_t query_rows =
        query_begin < qo_len ? min(kCtaQ, qo_len - query_begin) : 0;
    const uint32_t linear_thread = warp_id * 32 + lane_id;
    constexpr uint32_t threads = num_warps * 32;
    if (__builtin_expect(fp16_prefix_stages == 0xffffffffu, false)) {
      constexpr uint32_t output_words_per_row =
          HeadDim * sizeof(half) / sizeof(uint32_t);
      static_assert(
          HeadDim * sizeof(half) % sizeof(uint32_t) == 0,
          "empty-route output must be vector aligned");
      uint32_t* output_words = reinterpret_cast<uint32_t*>(output);
      const uint32_t output_row =
          (batch_id * num_qo_heads + head_id) * qo_len + query_begin;
      const uint32_t output_word = output_row * output_words_per_row;
      const uint32_t word_count = query_rows * output_words_per_row;
      for (uint32_t word = linear_thread; word < word_count; word += threads) {
        output_words[output_word + word] = 0;
      }
    }
#if MPA_STORE_LSE
    // Preserve dormant LSE code shape for ptxas coloring; production sparse
    // launches pass nullptr and execute no global LSE store.
    if (lse != nullptr) {
      for (uint32_t row = linear_thread; row < query_rows; row += threads) {
        lse[(batch_id * num_qo_heads + head_id) * qo_len + query_begin + row] =
            -__int_as_float(0x7f800000);
      }
    }
#endif
#endif
    return;
  }

  const float base2_softmax_scale = softmax_scale * math::log2e;

  float ro[num_tiles_q][num_tiles_v][8];
  float m[num_tiles_q][2];
  float d[num_tiles_q][2];
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; ++fq) {
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; ++fv) {
#pragma unroll
      for (uint32_t element = 0; element < 8; ++element) {
        ro[fq][fv][element] = 0.0f;
      }
    }
#pragma unroll
    for (uint32_t row = 0; row < 2; ++row) {
      m[fq][row] = -5000000.0f;
      d[fq][row] = 1.0f;
    }
  }

#include "mixed_attention_phase_composer.inl"

#if defined(MPA_K64_BLOCK_MODE)
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; ++fq) {
#pragma unroll
    for (uint32_t row = 0; row < 2; ++row) {
      float denominator = d[fq][row];
      denominator += __shfl_xor_sync(0xffffffff, denominator, 1);
      denominator += __shfl_xor_sync(0xffffffff, denominator, 2);
      m[fq][row] =
          (m[fq][row] + math::ptx_log2(denominator)) * math::log2e_recp;
      d[fq][row] = math::ptx_rcp(denominator);
    }
  }
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; ++fq) {
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; ++fv) {
#pragma unroll
      for (uint32_t element = 0; element < 8; ++element) {
        ro[fq][fv][element] *= d[fq][(element % 4) / 2];
      }
    }
  }
#if MPA_STORE_LSE
  if (lse != nullptr && lane_id % 4 == 0) {
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; ++fq) {
#pragma unroll
      for (uint32_t row_half = 0; row_half < 2; ++row_half) {
        const uint32_t local_row =
            get_warp_idx_q<num_warps_q, num_warps_k>() * kWarpQ +
            fq * kMmaQkM + lane_id / 4 + row_half * 8;
        const uint32_t global_row = query_block * kCtaQ + local_row;
        if (global_row < qo_len) {
          lse[(batch_id * num_qo_heads + head_id) * qo_len + global_row] =
              m[fq][row_half];
        }
      }
    }
  }
#endif
#else
  normalize_d_inplace<num_tiles_q, num_tiles_v>(ro, d);
#endif

  uint32_t output_warp_id = warp_id;
  if constexpr (HeadDim == 128 && kCtaQ == 64 && HasFp8 && !HasFp16) {
    asm volatile("mov.u32 %0, %%tid.y;" : "=r"(output_warp_id));
  }

  OutputSmem<HeadDim> smem_output(smem);
  const uint32_t output_row_base =
      output_warp_id * kWarpQ + lane_id / 4;
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; ++fq) {
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; ++fv) {
      uint32_t output_offset = smem_output.get_permuted_offset(
          output_row_base + fq * kMmaQkM,
          fv * (kMmaPvN / kPackHalf));
      Half2Bits packed_half;
      packed_half.value =
          __float22half2_rn(reinterpret_cast<float2*>(ro[fq][fv])[0]);
      reinterpret_cast<uint32_t*>(smem_output.base + output_offset)
          [lane_id % 4] =
          packed_half.bits;
      packed_half.value =
          __float22half2_rn(reinterpret_cast<float2*>(ro[fq][fv])[1]);
      reinterpret_cast<uint32_t*>(
          smem_output.base + output_offset + 8 * (HeadDim / kPackHalf))
          [lane_id % 4] =
          packed_half.bits;
      output_offset = smem_output.get_permuted_offset(
          output_row_base + fq * kMmaQkM,
          fv * (kMmaPvN / kPackHalf) + 1);
      packed_half.value =
          __float22half2_rn(reinterpret_cast<float2*>(ro[fq][fv])[2]);
      reinterpret_cast<uint32_t*>(smem_output.base + output_offset)
          [lane_id % 4] =
          packed_half.bits;
      packed_half.value =
          __float22half2_rn(reinterpret_cast<float2*>(ro[fq][fv])[3]);
      reinterpret_cast<uint32_t*>(
          smem_output.base + output_offset + 8 * (HeadDim / kPackHalf))
          [lane_id % 4] =
          packed_half.bits;
    }
  }
  __syncwarp();
  if constexpr (HeadDim == 128 && HasFp16 && !HasFp8) {
    asm volatile("mov.u32 %0, %%tid.x;" : "=r"(lane_id));
  }

  constexpr uint32_t output_line_lanes = 8;
  constexpr uint32_t output_lines_per_warp = 4;
  constexpr uint32_t output_smem_iters_row =
      HeadDim / (output_line_lanes * kPackHalf);
  constexpr uint32_t output_smem_iters_col =
      kCtaQ / (num_warps * output_lines_per_warp);
  half* output_lane =
      output + ((batch_id * num_qo_heads + head_id) * qo_len +
                query_block * kCtaQ +
                kWarpQ * output_warp_id +
                lane_id / output_line_lanes) *
                   HeadDim +
      (lane_id % output_line_lanes) * kPackHalf;
  uint32_t output_offset = smem_output.get_permuted_offset(
      output_warp_id * kWarpQ + lane_id / output_line_lanes,
      lane_id % output_line_lanes);
  uint32_t output_global_row =
      query_block * kCtaQ +
      kCtaQ / num_warps * output_warp_id + lane_id / output_line_lanes;
#pragma unroll
  for (uint32_t row_iter = 0; row_iter < output_smem_iters_col; ++row_iter) {
#pragma unroll
    for (uint32_t col_iter = 0; col_iter < output_smem_iters_row; ++col_iter) {
      if (output_global_row < qo_len) {
        smem_output.store_128b(output_offset, output_lane);
      }
      output_lane += output_line_lanes * kPackHalf;
      output_offset =
          smem_output.advance_offset_by_column<output_line_lanes>(output_offset);
    }
    output_offset = smem_output.advance_offset_by_row<output_lines_per_warp>(
        output_offset - output_smem_iters_row * output_line_lanes);
    output_lane += output_lines_per_warp * HeadDim -
                   output_smem_iters_row * output_line_lanes * kPackHalf;
    output_global_row += output_lines_per_warp;
  }
}

}  // namespace mpa::attention

template <uint32_t HeadDim, bool HasFp8, bool HasFp16, bool SmoothK>
void MPA_ATTENTION_LAUNCH_ENTRY(
    int8_t* q8,
    int8_t* k8,
    __nv_fp8_e4m3* v8,
    half* q16,
    half* k16,
    half* v16,
    half* k_mean,
    half* output,
    int32_t* fp8_lut,
    int32_t* fp8_count,
    int32_t* fp16_lut,
    int32_t* fp16_count,
    float* q_scale,
    float* k_scale,
    float* v_scale,
#if defined(MPA_K64_BLOCK_MODE)
    const int32_t* valid_k_counts,
    float* lse,
    uint32_t fp16_prefix_stages,
#endif
    uint32_t batch_size,
    uint32_t qo_len,
    uint32_t kv_len,
    uint32_t padded_kv_len,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    float softmax_scale,
    cudaStream_t stream) {
  anemoi_native::require_sm89();
  constexpr uint32_t low_smem_bytes =
      HasFp8 ? (mpa::attention::kCtaQ + 2 * mpa::attention::kCtaK) *
                   HeadDim
             : 0;
  constexpr uint32_t high_smem_bytes =
      HasFp16 ? (mpa::attention::kCtaQ + mpa::attention::kCtaK) *
                    HeadDim * sizeof(half)
              : 0;
  constexpr uint32_t output_smem_bytes =
      mpa::attention::kCtaQ * HeadDim * sizeof(half);
  constexpr uint32_t smem_bytes =
      std::max(std::max(low_smem_bytes, high_smem_bytes), output_smem_bytes);

  auto kernel =
      mpa::attention::MPA_ATTENTION_KERNEL_ENTRY<
          HeadDim, HasFp8, HasFp16, SmoothK>;
  ANEMOI_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
  const dim3 grid(
      div_ceil(qo_len, mpa::attention::kCtaQ), num_qo_heads, batch_size);
  const dim3 block(
      32,
      (mpa::attention::kCtaQ / mpa::attention::kWarpQ) *
          (mpa::attention::kCtaK / mpa::attention::kWarpK));
  kernel<<<grid, block, smem_bytes, stream>>>(
      q8, k8, v8, q16, k16, v16, k_mean, output, fp8_lut, fp8_count,
      fp16_lut, fp16_count, q_scale, k_scale, v_scale,
#if defined(MPA_K64_BLOCK_MODE)
      valid_k_counts, lse, fp16_prefix_stages,
#endif
      qo_len, kv_len,
      padded_kv_len, num_qo_heads / num_kv_heads, softmax_scale);
}

#undef MPA_ATTENTION_KERNEL_ENTRY
#undef MPA_ATTENTION_LAUNCH_ENTRY
#undef MPA_DENSE_SEQUENTIAL
#undef MPA_STORE_LSE
