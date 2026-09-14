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
 * This project-owned SM120 Q64 derivative starts from Anemoi SM89@7ebeb084 and the locked SpargeAttention
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

// Kitchen adaptation: framework-independent CUDA error propagation.
#include "attention_view.h"

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "primitives/cp_async.cuh"
#include "primitives/math.cuh"
#include "primitives/mma.cuh"
#include "primitives/permuted_smem.cuh"
#include "primitives/qattn/attn_utils.cuh"
#include "q64_attention_decl.cuh"

#ifndef MPA_LOW4_NVFP4
#define MPA_LOW4_NVFP4 0
#endif
#ifndef MPA_MIDDLE_MXFP8
#define MPA_MIDDLE_MXFP8 0
#endif
#ifndef MPA_MIDDLE_INT8
#define MPA_MIDDLE_INT8 0
#endif
#ifndef MPA_DENSE_SEQUENTIAL
#define MPA_DENSE_SEQUENTIAL 0
#endif
#ifndef MPA_STORE_LSE
#define MPA_STORE_LSE 1
#endif
#define MPA_THREE_PHASE \
  (MPA_LOW4_NVFP4 && (MPA_MIDDLE_MXFP8 || MPA_MIDDLE_INT8))

static_assert(!(MPA_MIDDLE_MXFP8 && MPA_MIDDLE_INT8));
static_assert(
    !MPA_DENSE_SEQUENTIAL ||
    ((MPA_MIDDLE_MXFP8 || MPA_MIDDLE_INT8) && !MPA_LOW4_NVFP4));

#if !defined(MPA_ATTENTION_KERNEL_ENTRY) || !defined(MPA_ATTENTION_LAUNCH_ENTRY)
#if defined(MPA_PACKED_RASTER_MODE)
#define MPA_ATTENTION_KERNEL_ENTRY mixed_attention_sm120_q64_packed_raster_kernel
#define MPA_ATTENTION_LAUNCH_ENTRY launch_mixed_attention_sm120_q64_packed_raster
#else
#define MPA_ATTENTION_KERNEL_ENTRY mixed_attention_sm120_q64_kernel
#define MPA_ATTENTION_LAUNCH_ENTRY launch_mixed_attention_sm120_q64
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

constexpr uint8_t kMxProbabilityScaleBits = 119U;  // E8M0 2^-8.
constexpr float kMxProbabilityInverseScale = 256.0f;
constexpr uint8_t kNvProbabilityScaleBits = 126U;  // E4M3 448.
constexpr float kNvProbabilityInverseScale = 6.0f;
constexpr float kNvProbabilityGlobalScale = 1.0f / (6.0f * 448.0f);

template <uint32_t NumTilesQ, uint32_t NumTilesV>
__device__ __forceinline__ void enter_fp8_probability_and_v_domains(
    float (&ro)[NumTilesQ][NumTilesV][8],
    float (&d)[NumTilesQ][2],
    const float* v_scale) {
  const uint32_t lane = get_lane_id();
  float scales[4];
#pragma unroll
  for (uint32_t fv = 0; fv < NumTilesV; ++fv) {
    reinterpret_cast<float2*>(scales)[0] =
        *reinterpret_cast<const float2*>(
            v_scale + fv * 16 + (lane % 4) * 2);
    reinterpret_cast<float2*>(scales)[1] =
        *reinterpret_cast<const float2*>(
            v_scale + fv * 16 + 8 + (lane % 4) * 2);
#pragma unroll
    for (uint32_t index = 0; index < 4; ++index) {
      scales[index] = (1.0f / S_FP8_OFFSET_EXP_INV) / scales[index];
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
#pragma unroll
  for (uint32_t fq = 0; fq < NumTilesQ; ++fq) {
#pragma unroll
    for (uint32_t row = 0; row < 2; ++row) {
      d[fq][row] *= 1.0f / S_FP8_OFFSET_EXP_INV;
    }
  }
}

template <uint32_t NumTilesQ, uint32_t NumTilesV>
__device__ __forceinline__ void leave_fp8_probability_and_v_domains(
    float (&ro)[NumTilesQ][NumTilesV][8],
    float (&d)[NumTilesQ][2],
    const float* v_scale) {
  const uint32_t lane = get_lane_id();
  float scales[4];
#pragma unroll
  for (uint32_t fv = 0; fv < NumTilesV; ++fv) {
    reinterpret_cast<float2*>(scales)[0] =
        *reinterpret_cast<const float2*>(
            v_scale + fv * 16 + (lane % 4) * 2);
    reinterpret_cast<float2*>(scales)[1] =
        *reinterpret_cast<const float2*>(
            v_scale + fv * 16 + 8 + (lane % 4) * 2);
#pragma unroll
    for (uint32_t index = 0; index < 4; ++index) {
      scales[index] *= S_FP8_OFFSET_EXP_INV;
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
#pragma unroll
  for (uint32_t fq = 0; fq < NumTilesQ; ++fq) {
#pragma unroll
    for (uint32_t row = 0; row < 2; ++row) {
      d[fq][row] *= S_FP8_OFFSET_EXP_INV;
    }
  }
}

__device__ __forceinline__ void mma_m16n8k32_mxfp8(
    const uint32_t (&a)[4],
    const uint32_t (&b)[2],
    uint32_t scale_a,
    uint16_t byte_id_a,
    uint32_t scale_b,
    uint16_t byte_id_b,
    float (&acc)[4]) {
  const uint16_t thread_id = 0;
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4"
      ".block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};"
      : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
        "r"(b[0]), "r"(b[1]),
        "f"(acc[0]), "f"(acc[1]), "f"(acc[2]), "f"(acc[3]),
        "r"(scale_a), "h"(byte_id_a), "h"(thread_id),
        "r"(scale_b), "h"(byte_id_b), "h"(thread_id));
}

// D64 owns two E8M0 bytes per row; zero-extend that halfword instead of
// reading the next row. MMA selects byte 0/1 for the two K32 groups.
template <uint32_t HeadDim>
__device__ __forceinline__ uint32_t load_mxfp8_scale_word(
    const uint8_t* scales, uint32_t row) {
  static_assert(HeadDim == 64 || HeadDim == 128);
  if constexpr (HeadDim == 64) {
    return *reinterpret_cast<const uint16_t*>(scales + row * 2);
  } else {
    return *reinterpret_cast<const uint32_t*>(scales + row * 4);
  }
}

template <uint32_t HeadDim, uint32_t NumTilesQ, uint32_t NumTilesK>
__device__ __forceinline__ void compute_mxfp8_qk(
    const LowQKSmem<HeadDim>& smem_q,
    const LowQKSmem<HeadDim>& smem_k,
    const uint8_t* q_scale,
    const uint8_t* k_scale,
    float (&scores)[NumTilesQ][NumTilesK][8]) {
  static_assert(HeadDim == 64 || HeadDim == 128);
  static_assert((NumTilesQ == 1 || NumTilesQ == 2) && NumTilesK == 4);
  constexpr uint32_t NumWarpsQ = kCtaQ / kWarpQ;
  constexpr uint32_t NumWarpsK = kCtaK / kWarpK;
  const uint32_t lane = get_lane_id();
  const uint32_t group = lane >> 2;
  const uint32_t thread_in_quad = lane & 3;
  const uint32_t warp_q = get_warp_idx_q<NumWarpsQ, NumWarpsK>();

#pragma unroll
  for (uint32_t fq = 0; fq < NumTilesQ; ++fq) {
#pragma unroll
    for (uint32_t fk = 0; fk < NumTilesK; ++fk) {
#pragma unroll
      for (uint32_t element = 0; element < 8; ++element) {
        scores[fq][fk][element] = 0.0f;
      }
    }
  }

  uint32_t sf_q[NumTilesQ] = {};
#pragma unroll
  for (uint32_t fq = 0; fq < NumTilesQ; ++fq) {
    if (thread_in_quad <= 1) {
      const uint32_t row =
          warp_q * kWarpQ + fq * 16 + group + thread_in_quad * 8;
      sf_q[fq] = load_mxfp8_scale_word<HeadDim>(q_scale, row);
    }
  }

#pragma unroll
  for (uint32_t d32 = 0; d32 < HeadDim / 32; ++d32) {
    uint32_t q_data[NumTilesQ][4];
#pragma unroll
    for (uint32_t fq = 0; fq < NumTilesQ; ++fq) {
      const uint32_t q_offset = smem_q.get_permuted_offset(
          warp_q * kWarpQ + fq * 16 + lane % 16,
          lane / 16 + d32 * 2);
      smem_q.ldmatrix_m8n8x4(q_offset, q_data[fq]);
    }

#pragma unroll
    for (uint32_t fk = 0; fk < NumTilesK; ++fk) {
      const uint32_t key_row0 = fk * 16 + group;
      uint32_t sf_k0 = 0U;
      uint32_t sf_k1 = 0U;
      if (thread_in_quad == 0) {
        sf_k0 = load_mxfp8_scale_word<HeadDim>(k_scale, key_row0);
        sf_k1 = load_mxfp8_scale_word<HeadDim>(k_scale, key_row0 + 8);
      }
      const uint32_t k_offset = smem_k.get_permuted_offset(
          fk * 16 + lane % 8 + (lane / 16) * 8,
          (lane / 8) % 2 + d32 * 2);
      uint32_t k_data[4];
      smem_k.ldmatrix_m8n8x4(k_offset, k_data);
      const uint32_t k_data0[2] = {k_data[0], k_data[1]};
      const uint32_t k_data1[2] = {k_data[2], k_data[3]};
#pragma unroll
      for (uint32_t fq = 0; fq < NumTilesQ; ++fq) {
        float (&score0)[4] =
            *reinterpret_cast<float (*)[4]>(&scores[fq][fk][0]);
        float (&score1)[4] =
            *reinterpret_cast<float (*)[4]>(&scores[fq][fk][4]);
        mma_m16n8k32_mxfp8(
            q_data[fq], k_data0, sf_q[fq], static_cast<uint16_t>(d32),
            sf_k0, static_cast<uint16_t>(d32), score0);
        mma_m16n8k32_mxfp8(
            q_data[fq], k_data1, sf_q[fq], static_cast<uint16_t>(d32),
            sf_k1, static_cast<uint16_t>(d32), score1);
      }
    }
  }
}

__device__ __forceinline__ uint32_t pack_e4m3x4(
    float f0, float f1, float f2, float f3) {
  uint32_t packed;
  asm volatile(
      "{\n\t"
      ".reg .b16 lo, hi;\n\t"
      "cvt.rn.satfinite.e4m3x2.f32 lo, %1, %2;\n\t"
      "cvt.rn.satfinite.e4m3x2.f32 hi, %3, %4;\n\t"
      "mov.b32 %0, {lo, hi};\n\t"
      "}"
      : "=r"(packed)
      : "f"(f0), "f"(f1), "f"(f2), "f"(f3));
  return packed;
}

__device__ __forceinline__ uint32_t pack_sage_mxfp8_probability(
    const float* first, const float* second) {
  return pack_e4m3x4(
      first[1] * kMxProbabilityInverseScale,
      first[0] * kMxProbabilityInverseScale,
      second[1] * kMxProbabilityInverseScale,
      second[0] * kMxProbabilityInverseScale);
}

struct MxProbabilityFragment {
  uint32_t data[4];
  uint32_t scale;
};

template <uint32_t TokenChunk, uint32_t NumTilesK>
__device__ __forceinline__ MxProbabilityFragment
prepare_mxfp8_probability(float (&scores)[NumTilesK][8]) {
  static_assert(NumTilesK == 4);
  static_assert(TokenChunk < 2);
  constexpr uint32_t TileBase = TokenChunk * 2;
  MxProbabilityFragment probability;
  probability.data[0] = pack_sage_mxfp8_probability(
      scores[TileBase], scores[TileBase] + 4);
  probability.data[1] = pack_sage_mxfp8_probability(
      scores[TileBase] + 2, scores[TileBase] + 6);
  probability.data[2] = pack_sage_mxfp8_probability(
      scores[TileBase + 1], scores[TileBase + 1] + 4);
  probability.data[3] = pack_sage_mxfp8_probability(
      scores[TileBase + 1] + 2, scores[TileBase + 1] + 6);
  const uint32_t thread_in_quad = get_lane_id() & 3;
  probability.scale = thread_in_quad <= 1 ? kMxProbabilityScaleBits : 0U;
  return probability;
}

template <uint32_t TokenChunk, uint32_t HeadDim, uint32_t NumTilesV>
__device__ __forceinline__ void accumulate_prepared_mxfp8_pv(
    float (&ro)[NumTilesV][8],
    const MxProbabilityFragment& probability,
    const LowVSmem& smem_v,
    const uint8_t* v_scale) {
  static_assert((HeadDim == 64 || HeadDim == 128) && NumTilesV == HeadDim / 16);
  const uint32_t lane = get_lane_id();
  const uint32_t group = lane >> 2;
  const uint32_t thread_in_quad = lane & 3;
  const uint32_t row_base = lane % 8 + (lane / 16) * 8;
  const uint32_t col_base = (lane / 8) % 2 + TokenChunk * 2;

#pragma unroll
  for (uint32_t fv = 0; fv < NumTilesV; ++fv) {
    uint32_t v_data[4];
    const uint32_t offset = smem_v.get_permuted_offset(
        row_base + fv * 16, col_base);
    smem_v.ldmatrix_m8n8x4(offset, v_data);
    uint32_t sf_v = 0U;
    if (thread_in_quad == 0) {
      sf_v = *reinterpret_cast<const uint32_t*>(
          v_scale + (fv * 8 + group) * 4);
    }
#pragma unroll
    for (uint32_t pair = 0; pair < 2; ++pair) {
      const uint32_t operand[2] = {
          v_data[pair * 2], v_data[pair * 2 + 1]};
      float (&destination)[4] =
          *reinterpret_cast<float (*)[4]>(&ro[fv][pair * 4]);
      mma_m16n8k32_mxfp8(
          probability.data, operand, probability.scale, 0,
          sf_v, static_cast<uint16_t>(TokenChunk + pair * 2),
          destination);
    }
  }
}

template <uint32_t TokenChunk, uint32_t HeadDim, uint32_t NumTilesV>
__device__ __forceinline__ void accumulate_mxfp8_pv(
    float (&ro)[NumTilesV][8],
    float (&scores)[4][8],
    const LowVSmem& smem_v,
    const uint8_t* v_scale) {
  const MxProbabilityFragment probability =
      prepare_mxfp8_probability<TokenChunk>(scores);
  accumulate_prepared_mxfp8_pv<TokenChunk, HeadDim>(
      ro, probability, smem_v, v_scale);
}

// Localized from Anemoi 8551120418d1af149bbb3ce0bfb33b23ce7cec8d.  The
// byte-address swizzle is the exact layout paired with the native NVFP4
// ldmatrix fragments; it changes only physical operand placement.
template <uint32_t StrideBytes>
__device__ __forceinline__ uint32_t nvfp4_swizzle(uint32_t address) {
  if constexpr (StrideBytes == 16) {
    return address;
  }
  const uint32_t row = (address / StrideBytes) % 8;
  const uint32_t xor_bits = row / max(64U / StrideBytes, 1U);
  return address ^ (xor_bits << 4);
}

template <uint32_t Rows, uint32_t Width, uint32_t Threads>
__device__ __forceinline__ void copy_nvfp4_async_swizzled(
    uint8_t* dst,
    const uint8_t* src,
    uint32_t tid,
    uint32_t src_stride) {
  static_assert((Rows * Width) % 16 == 0);
  constexpr uint32_t vectors = Rows * Width / 16;
  const uint32_t shared = __cvta_generic_to_shared(dst);
  for (uint32_t vector = tid; vector < vectors; vector += Threads) {
    const uint32_t index = vector * 16;
    const uint32_t row = index / Width;
    const uint32_t col = index - row * Width;
    const uint32_t address = nvfp4_swizzle<Width>(shared + index);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                 :: "r"(address), "l"(src + row * src_stride + col));
  }
}

template <uint32_t Bytes, uint32_t Threads>
__device__ __forceinline__ void copy_nvfp4_async_contiguous(
    uint8_t* dst,
    const uint8_t* src,
    uint32_t tid) {
  static_assert(Bytes % 16 == 0);
  for (uint32_t offset = tid * 16; offset < Bytes;
       offset += Threads * 16) {
    cp_async::load_128b<cp_async::PrefetchMode::kNoPrefetch>(
        dst + offset, src + offset);
  }
}

template <uint32_t HeadDim, uint32_t NumTilesQ>
__device__ __forceinline__ void load_nvfp4_query(
    const uint8_t* q,
    const uint8_t* q_scale,
    uint32_t (&q_data)[NumTilesQ][HeadDim / 64][4],
    uint32_t (&scale_data)[NumTilesQ][HeadDim / 64]) {
  static_assert((HeadDim == 64 || HeadDim == 128) && (NumTilesQ == 1 || NumTilesQ == 2));
  constexpr uint32_t data_stride = HeadDim / 2;
  constexpr uint32_t scale_stride = HeadDim / 16;
  constexpr uint32_t num_warps_q = kCtaQ / kWarpQ;
  constexpr uint32_t num_warps_k = kCtaK / kWarpK;
  const uint32_t lane = get_lane_id();
  const uint32_t group = lane >> 2;
  const uint32_t thread_in_quad = lane & 3;
  const uint32_t warp_q = get_warp_idx_q<num_warps_q, num_warps_k>();
#pragma unroll
  for (uint32_t fq = 0; fq < NumTilesQ; ++fq) {
    const uint32_t upper = warp_q * kWarpQ + fq * 16 + group;
    const uint32_t lower = upper + 8;
#pragma unroll
    for (uint32_t d64 = 0; d64 < HeadDim / 64; ++d64) {
      const uint32_t byte_col = d64 * 32 + thread_in_quad * 4;
      q_data[fq][d64][0] = *reinterpret_cast<const uint32_t*>(
          q + upper * data_stride + byte_col);
      q_data[fq][d64][1] = *reinterpret_cast<const uint32_t*>(
          q + lower * data_stride + byte_col);
      q_data[fq][d64][2] = *reinterpret_cast<const uint32_t*>(
          q + upper * data_stride + byte_col + 16);
      q_data[fq][d64][3] = *reinterpret_cast<const uint32_t*>(
          q + lower * data_stride + byte_col + 16);
      if (thread_in_quad == 0) {
        scale_data[fq][d64] = *reinterpret_cast<const uint32_t*>(
            q_scale + upper * scale_stride + d64 * 4);
      } else if (thread_in_quad == 1) {
        scale_data[fq][d64] = *reinterpret_cast<const uint32_t*>(
            q_scale + lower * scale_stride + d64 * 4);
      } else {
        scale_data[fq][d64] = 0U;
      }
    }
  }
}

template <uint32_t HeadDim, uint32_t NumTilesQ, uint32_t NumTilesK>
__device__ __forceinline__ void compute_nvfp4_qk(
    const uint32_t (&q_data)[NumTilesQ][HeadDim / 64][4],
    const uint32_t (&q_scale)[NumTilesQ][HeadDim / 64],
    const uint8_t* k_smem,
    const uint8_t* k_scale,
    float (&scores)[NumTilesQ][NumTilesK][8]) {
  static_assert(
      (HeadDim == 64 || HeadDim == 128) && (NumTilesQ == 1 || NumTilesQ == 2) &&
      NumTilesK == 4);
  constexpr uint32_t data_stride = HeadDim / 2;
  constexpr uint32_t scale_stride = HeadDim / 16;
  const uint32_t lane = get_lane_id();
  const uint32_t group = lane >> 2;
  const uint32_t k_base = nvfp4_swizzle<data_stride>(
      __cvta_generic_to_shared(k_smem) +
      (lane % 8) * data_stride + ((lane / 8) % (HeadDim / 32)) * 16);
#pragma unroll
  for (uint32_t fq = 0; fq < NumTilesQ; ++fq) {
#pragma unroll
    for (uint32_t fk = 0; fk < NumTilesK; ++fk) {
#pragma unroll
      for (uint32_t element = 0; element < 8; ++element) {
        scores[fq][fk][element] = 0.0f;
      }
    }
  }
#pragma unroll
  for (uint32_t tile = 0; tile < 8; ++tile) {
    const uint32_t key_row = tile * 8 + group;
    uint32_t k_fragment[HeadDim / 32];
    const uint32_t address = k_base + tile * 8 * data_stride;
    if constexpr (HeadDim == 64) {
      // Two matrices contain exactly one packed K64 group per key row.
      asm volatile(
          "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
          : "=r"(k_fragment[0]), "=r"(k_fragment[1])
          : "r"(address));
    } else {
      asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(k_fragment[0]), "=r"(k_fragment[1]),
          "=r"(k_fragment[2]), "=r"(k_fragment[3])
        : "r"(address));
    }
#pragma unroll
    for (uint32_t d64 = 0; d64 < HeadDim / 64; ++d64) {
      const uint32_t k_data[2] = {
          k_fragment[d64 * 2], k_fragment[d64 * 2 + 1]};
      const uint32_t scale_k = *reinterpret_cast<const uint32_t*>(
          k_scale + key_row * scale_stride + d64 * 4);
#pragma unroll
      for (uint32_t fq = 0; fq < NumTilesQ; ++fq) {
        float (&destination)[4] = *reinterpret_cast<float (*)[4]>(
            &scores[fq][tile / 2][(tile & 1) * 4]);
        mma::mma_m16n8k64_nvfp4(
            q_data[fq][d64], k_data,
            q_scale[fq][d64], scale_k, destination);
      }
    }
  }
}

__device__ __forceinline__ uint32_t pack_nvfp4_e2m1x8(
    float f0, float f1, float f2, float f3,
    float f4, float f5, float f6, float f7) {
  uint32_t packed;
  asm volatile(
      "{\n\t"
      ".reg .b8 a0, a1, a2, a3;\n\t"
      ".reg .b16 lo, hi;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 a0, %1, %2;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 a1, %3, %4;\n\t"
      "mov.b16 lo, {a0, a1};\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 a2, %5, %6;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 a3, %7, %8;\n\t"
      "mov.b16 hi, {a2, a3};\n\t"
      "mov.b32 %0, {lo, hi};\n\t"
      "}"
      : "=r"(packed)
      : "f"(f0), "f"(f1), "f"(f2), "f"(f3),
        "f"(f4), "f"(f5), "f"(f6), "f"(f7));
  return packed;
}

template <uint32_t TileBase, uint32_t Component>
__device__ __forceinline__ uint32_t pack_nvfp4_probability_fragment(
    float (&scores)[8][4]) {
  static_assert((TileBase == 0 || TileBase == 4) &&
                (Component == 0 || Component == 2));
  return pack_nvfp4_e2m1x8(
      scores[TileBase + 0][Component + 1] * kNvProbabilityInverseScale,
      scores[TileBase + 0][Component + 0] * kNvProbabilityInverseScale,
      scores[TileBase + 1][Component + 1] * kNvProbabilityInverseScale,
      scores[TileBase + 1][Component + 0] * kNvProbabilityInverseScale,
      scores[TileBase + 2][Component + 1] * kNvProbabilityInverseScale,
      scores[TileBase + 2][Component + 0] * kNvProbabilityInverseScale,
      scores[TileBase + 3][Component + 1] * kNvProbabilityInverseScale,
      scores[TileBase + 3][Component + 0] * kNvProbabilityInverseScale);
}

struct NvProbabilityFragment {
  uint32_t data[4];
  uint32_t scale;
};

__device__ __forceinline__ NvProbabilityFragment
prepare_nvfp4_probability(float (&current_scores)[4][8]) {
  // K is pre-permuted within K32, so four local N8 score fragments already
  // form the native PV A register.  No cross-lane data transpose is needed.
  float (&scores)[8][4] = *reinterpret_cast<float (*)[8][4]>(current_scores);
  NvProbabilityFragment result{
      {pack_nvfp4_probability_fragment<0, 0>(scores),
       pack_nvfp4_probability_fragment<0, 2>(scores),
       pack_nvfp4_probability_fragment<4, 0>(scores),
       pack_nvfp4_probability_fragment<4, 2>(scores)},
      0x7e7e7e7eU};
  return result;
}

template <uint32_t HeadDim>
__device__ __forceinline__ void accumulate_nvfp4_pv(
    float (&current_ro)[HeadDim / 16][8],
    const NvProbabilityFragment& probability,
    const uint8_t* v_smem,
    const uint8_t* v_scale) {
  static_assert(HeadDim == 64 || HeadDim == 128);
  float (&ro)[HeadDim / 8][4] =
      *reinterpret_cast<float (*)[HeadDim / 8][4]>(current_ro);
  constexpr uint32_t v_stride = 64 / 2;
  const uint32_t lane = get_lane_id();
  const uint32_t group = lane >> 2;
#pragma unroll
  for (uint32_t d = 0; d < HeadDim / 8; d += 2) {
    const uint32_t n_index =
        d * 8 + (lane / 16) * 8 + (lane % 8);
    const uint32_t token_byte = ((lane % 16) / 8) * 16;
    const uint32_t address = nvfp4_swizzle<v_stride>(
        __cvta_generic_to_shared(v_smem) +
        n_index * v_stride + token_byte);
    uint32_t v_fragment[4];
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(v_fragment[0]), "=r"(v_fragment[1]),
          "=r"(v_fragment[2]), "=r"(v_fragment[3])
        : "r"(address));
#pragma unroll
    for (uint32_t pair = 0; pair < 2; ++pair) {
      const uint32_t output_tile = d + pair;
      const uint32_t v_col = output_tile * 8 + group;
      const uint32_t v_data[2] = {
          v_fragment[pair * 2], v_fragment[pair * 2 + 1]};
      const uint32_t scale_v = *reinterpret_cast<const uint32_t*>(
          v_scale + v_col * 4);
      mma::mma_m16n8k64_nvfp4(
          probability.data, v_data, probability.scale,
          scale_v, ro[output_tile]);
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

// D128 uses the already-consumed fq=0 half of each warp's Q shared-memory
// slice as a transient 16-KiB score stash.  Only those 64 Q rows need to be
// restored before a later FP16 physical stage; fq=1 remains resident.
template <bool Drain = true, bool WarpLocal = false>
__device__ __forceinline__ void reload_d128_fq0_q(
    const HalfQKVSmem<128>& smem_q,
    const half* q16,
#if MPA_MIDDLE_INT8 && !MPA_LOW4_NVFP4
    uint32_t q_head_row,
#else
    uint32_t batch_id,
    uint32_t head_id,
    uint32_t num_qo_heads,
#endif
    uint32_t query_block,
    uint32_t qo_len) {
  constexpr uint32_t packs_per_row = 128 / kPackHalf;
  constexpr uint32_t rows_per_warp = kWarpQ / 2;
  constexpr uint32_t compact_rows =
      (kCtaQ / kWarpQ) * rows_per_warp;
  constexpr uint32_t reload_rows = WarpLocal ? rows_per_warp : compact_rows;
  constexpr uint32_t total_packs = reload_rows * packs_per_row;
  constexpr uint32_t cta_threads =
      32 * (kCtaQ / kWarpQ) * (kCtaK / kWarpK);
  constexpr uint32_t reload_threads = WarpLocal ? 32 : cta_threads;
  const uint32_t linear_thread =
      WarpLocal ? get_lane_id() : get_warp_id() * 32 + get_lane_id();

#pragma unroll 1
  for (uint32_t copy = 0; copy < total_packs / reload_threads; ++copy) {
    const uint32_t compact_index = linear_thread + copy * reload_threads;
    const uint32_t compact_row = compact_index / packs_per_row;
    const uint32_t pack = compact_index % packs_per_row;
    const uint32_t local_row = WarpLocal
        ? get_warp_id() * kWarpQ + compact_row
        : (compact_row / rows_per_warp) * kWarpQ +
              compact_row % rows_per_warp;
    const uint32_t global_row = query_block * kCtaQ + local_row;
    const half* source =
#if MPA_MIDDLE_INT8 && !MPA_LOW4_NVFP4
        q16 + (q_head_row * qo_len + global_row) * 128 +
#else
        q16 + ((batch_id * num_qo_heads + head_id) * qo_len + global_row) *
                  128 +
#endif
        pack * kPackHalf;
    smem_q.load_128b_async<cp_async::SharedMemFillMode::kFillZero>(
        smem_q.get_permuted_offset(local_row, pack),
        source,
        global_row < qo_len);
  }
  cp_async::commit_group();
  if constexpr (Drain) {
    cp_async::wait_group<0>();
    __syncthreads();
  }
}

template <uint32_t NumTilesK>
__device__ __forceinline__ void stash_d128_fq0_scores(
    int8_t* smem,
    float scores[][NumTilesK][8]) {
  constexpr uint32_t score_values = NumTilesK * 8;
  constexpr uint32_t score_packs = score_values / 4;
  constexpr uint32_t warp_lanes = 32;
  constexpr uint32_t q_slice_values =
      kWarpQ * 128 * sizeof(half) / sizeof(float);
  constexpr uint32_t q_slice_packs = q_slice_values / 4;
  static_assert(score_packs * warp_lanes <= q_slice_packs);
  float4* warp_slots = reinterpret_cast<float4*>(smem) +
      get_warp_id() * q_slice_packs;
  const float4* score_frag = reinterpret_cast<float4*>(scores[0][0]);
#pragma unroll
  for (uint32_t pack = 0; pack < score_packs; ++pack) {
    // Keep the 16-byte vector access, but make lanes adjacent within each
    // pack.  A lane-major score array strides every lane by 128 bytes and
    // maps the entire warp to the same four shared-memory banks.
    warp_slots[pack * warp_lanes + get_lane_id()] = score_frag[pack];
  }
}

template <uint32_t NumTilesK>
__device__ __forceinline__ void load_d128_fq0_scores(
    const int8_t* smem,
    float scores[][NumTilesK][8]) {
  constexpr uint32_t score_values = NumTilesK * 8;
  constexpr uint32_t score_packs = score_values / 4;
  constexpr uint32_t warp_lanes = 32;
  constexpr uint32_t q_slice_values =
      kWarpQ * 128 * sizeof(half) / sizeof(float);
  constexpr uint32_t q_slice_packs = q_slice_values / 4;
  static_assert(score_packs * warp_lanes <= q_slice_packs);
  const float4* warp_slots = reinterpret_cast<const float4*>(smem) +
      get_warp_id() * q_slice_packs;
  float4* score_frag = reinterpret_cast<float4*>(scores[0][0]);
#pragma unroll
  for (uint32_t pack = 0; pack < score_packs; ++pack) {
    score_frag[pack] = warp_slots[pack * warp_lanes + get_lane_id()];
  }
}

template <uint32_t HeadDim, bool HasFp8, bool HasFp16, bool SmoothK>
#if defined(MPA_MIN_BLOCKS_PER_SM)
__global__ __launch_bounds__(
    32 * (kCtaQ / kWarpQ) * (kCtaK / kWarpK), MPA_MIN_BLOCKS_PER_SM)
#else
__global__ __launch_bounds__(
    32 * (kCtaQ / kWarpQ) * (kCtaK / kWarpK))
#endif
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
#if MPA_MIDDLE_INT8
    float* __restrict__ q_scale,
    float* __restrict__ k_scale,
    float* __restrict__ v_scale,
#else
    uint8_t* __restrict__ q_scale,
    uint8_t* __restrict__ k_scale,
    uint8_t* __restrict__ v_scale,
#endif
#if MPA_THREE_PHASE
    int8_t* __restrict__ q4,
    int8_t* __restrict__ k4,
    __nv_fp8_e4m3* __restrict__ v4,
    uint8_t* __restrict__ q4_scale,
    uint8_t* __restrict__ k4_scale,
    uint8_t* __restrict__ v4_scale,
    int32_t* __restrict__ middle_count,
#endif
#if MPA_LOW4_NVFP4
    const float* __restrict__ q_global_scale,
    const float* __restrict__ k_global_scale,
    const float* __restrict__ v_global_scale,
#endif
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
#if !MPA_MIDDLE_INT8 && !MPA_LOW4_NVFP4 && !MPA_MIDDLE_MXFP8
  static_assert(!HasFp8 || HasFp16,
                "MXFP8 endpoints share the unified MXFP8+FP16 binary");
#endif
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
#if MPA_MIDDLE_INT8 && !MPA_LOW4_NVFP4
  uint32_t batch_id;
  uint32_t head_id;
  if constexpr (HasFp16) {
    // The scratch is consumed before any phase owns shared memory.  The
    // volatile round trip prevents ptxas from extending entry CTA IDs across
    // INT8 and substituting them for the FP16 boundary reads.
    volatile uint32_t* int8_ids =
        reinterpret_cast<volatile uint32_t*>(smem) +
        2 * (warp_id * 32 + lane_id);
    int8_ids[0] = blockIdx.z;
    int8_ids[1] = blockIdx.y;
    batch_id = int8_ids[0];
    head_id = int8_ids[1];
  } else {
    batch_id = blockIdx.z;
    head_id = blockIdx.y;
  }
  const uint32_t query_block = blockIdx.x;
#else
  const uint32_t batch_id = blockIdx.z;
  const uint32_t query_block = blockIdx.x;
  const uint32_t head_id = blockIdx.y;
#endif
  const uint32_t num_qo_heads = gridDim.y;
  const uint32_t kv_head = head_id / num_kv_groups;
  const uint32_t num_kv_heads = num_qo_heads / num_kv_groups;
  const uint32_t num_query_blocks = gridDim.x;
  const uint32_t num_physical_stages = div_ceil(kv_len, kCtaK);
  const uint32_t metadata_row =
      (batch_id * num_qo_heads + head_id) * num_query_blocks + query_block;

#if MPA_DENSE_SEQUENTIAL
  const uint32_t nv_iterations = 0;
  const uint32_t low_iterations = HasFp8 ? num_physical_stages : 0;
#elif MPA_THREE_PHASE
  const uint32_t nv_iterations = HasFp8 ? fp8_count[metadata_row] : 0;
  const uint32_t low_iterations = HasFp8 ? middle_count[metadata_row] : 0;
#elif MPA_LOW4_NVFP4
  const uint32_t nv_iterations = HasFp8 ? fp8_count[metadata_row] : 0;
  const uint32_t low_iterations = 0;
#else
  const uint32_t nv_iterations = 0;
  const uint32_t low_iterations = HasFp8 ? fp8_count[metadata_row] : 0;
#endif
  const uint32_t route_low_iterations = nv_iterations + low_iterations;
#if MPA_LOW4_NVFP4 || MPA_MIDDLE_INT8 || MPA_MIDDLE_MXFP8
  const uint32_t initial_high_iterations =
      HasFp16 && route_low_iterations == 0 ? fp16_count[metadata_row] : 0;
#else
  const uint32_t initial_high_iterations =
      HasFp16 ? fp16_count[metadata_row] : 0;
#endif
  if (route_low_iterations == 0 && initial_high_iterations == 0) {
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

#include "q64_attention_phase_composer.inl"
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
  constexpr bool kInnerLseNullGuard =
      kCtaQ == 128 && MPA_LOW4_NVFP4 && MPA_MIDDLE_MXFP8 && !HasFp16;
  if ((kInnerLseNullGuard || lse != nullptr) && lane_id % 4 == 0) {
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; ++fq) {
#pragma unroll
      for (uint32_t row_half = 0; row_half < 2; ++row_half) {
        const uint32_t local_row =
            get_warp_idx_q<num_warps_q, num_warps_k>() * kWarpQ +
            fq * kMmaQkM + lane_id / 4 + row_half * 8;
        const uint32_t global_row = query_block * kCtaQ + local_row;
        if (global_row < qo_len &&
            (!kInnerLseNullGuard || lse != nullptr)) {
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

  OutputSmem<HeadDim> smem_output(smem);
  const uint32_t output_row_base =
      get_warp_idx_q<num_warps_q, num_warps_k>() * kWarpQ + lane_id / 4;
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
                kWarpQ * get_warp_idx_q<num_warps_q, num_warps_k>() +
                lane_id / output_line_lanes) *
                   HeadDim +
      (lane_id % output_line_lanes) * kPackHalf;
  uint32_t output_offset = smem_output.get_permuted_offset(
      get_warp_idx_q<num_warps_q, num_warps_k>() * kWarpQ +
          lane_id / output_line_lanes,
      lane_id % output_line_lanes);
  uint32_t output_global_row =
      query_block * kCtaQ +
      kCtaQ / num_warps * warp_id + lane_id / output_line_lanes;
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
#if MPA_MIDDLE_INT8
    float* q_scale,
    float* k_scale,
    float* v_scale,
#else
    uint8_t* q_scale,
    uint8_t* k_scale,
    uint8_t* v_scale,
#endif
#if MPA_THREE_PHASE
    int8_t* q4,
    int8_t* k4,
    __nv_fp8_e4m3* v4,
    uint8_t* q4_scale,
    uint8_t* k4_scale,
    uint8_t* v4_scale,
    int32_t* middle_count,
#endif
#if MPA_LOW4_NVFP4
    const float* q_global_scale,
    const float* k_global_scale,
    const float* v_global_scale,
#endif
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
  constexpr uint32_t low_smem_bytes =
      HasFp8 ? (mpa::attention::kCtaQ + 2 * mpa::attention::kCtaK) *
                       HeadDim +
                   mpa::attention::kCtaK * (HeadDim / 32) + HeadDim * 2
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
  ANEMOI_SM120_CUDA_CHECK(cudaFuncSetAttribute(
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
#if MPA_THREE_PHASE
      q4, k4, v4, q4_scale, k4_scale, v4_scale, middle_count,
#endif
#if MPA_LOW4_NVFP4
      q_global_scale, k_global_scale, v_global_scale,
#endif
#if defined(MPA_K64_BLOCK_MODE)
      valid_k_counts, lse, fp16_prefix_stages,
#endif
      qo_len, kv_len,
      padded_kv_len, num_qo_heads / num_kv_heads, softmax_scale);
}

#undef MPA_ATTENTION_KERNEL_ENTRY
#undef MPA_ATTENTION_LAUNCH_ENTRY
#undef MPA_THREE_PHASE
#undef MPA_DENSE_SEQUENTIAL
#undef MPA_STORE_LSE
#undef MPA_MIN_BLOCKS_PER_SM
