// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
#pragma once
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
namespace h3_qkv {
using namespace cute;
template <typename ThreadMap, bool Scalar> struct WeightScaleBroadcast;

template <typename ThreadMap> struct WeightScaleBroadcast<ThreadMap, false> {
  using Type = cutlass::epilogue::threadblock::VisitorRowBroadcast<
      ThreadMap, float, cute::Stride<_0, _1, int32_t>>;

  static typename Type::Arguments arguments(const float *scale, int n) {
    return {scale, 0.f, {_0{}, _1{}, n}};
  }
};

template <typename ThreadMap> struct WeightScaleBroadcast<ThreadMap, true> {
  using Type = cutlass::epilogue::threadblock::VisitorScalarBroadcast<float>;

  static typename Type::Arguments arguments(const float *scale, int) {
    typename Type::Arguments result{};
    result.scalar_ptrs[0] = scale;
    return result;
  }
};

// One fused int8 GEMM, parameterized on output type AND tile/warp/stage config.
// bias is read in ElementOutput (nullptr broadcasts 0).
template <typename ElementOutput, int TBM, int TBN, int TBK, int WM, int WN,
          int WK, int NumStages, typename ArchTag = cutlass::arch::Sm80,
          bool ScalarWeightScale = false, int AlignmentAB = 16,
          typename ThreadblockSwizzle =
              cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>>
struct FusedInt8Gemm {
  using ElementA = int8_t;
  using ElementB = int8_t;
  using ElementC = ElementOutput;
  using ElementAcc = int32_t;
  using ElementCompute = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor; // B[N,K] row == [K,N] col
  using LayoutC = cutlass::layout::RowMajor;
  static constexpr int AlignA = AlignmentAB, AlignB = AlignmentAB;
  static constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementC>::value;
  using TB = cutlass::gemm::GemmShape<TBM, TBN, TBK>;
  using Warp = cutlass::gemm::GemmShape<WM, WN, WK>;
  using Inst = cutlass::gemm::GemmShape<16, 8, 32>;
  static constexpr int EVTStages = 2;

  using ThreadMap =
      cutlass::epilogue::threadblock::OutputTileThreadLayout<TB, Warp, ElementC,
                                                             AlignC, EVTStages>;
  using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;
  using XScale = cutlass::epilogue::threadblock::VisitorColBroadcast<
      ThreadMap, ElementCompute, cute::Stride<_1, _0, int32_t>>;
  using WScale =
      typename WeightScaleBroadcast<ThreadMap, ScalarWeightScale>::Type;
  using Bias =
      cutlass::epilogue::threadblock::VisitorScalarBroadcast<ElementOutput>;
  using Mul0 = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, ElementCompute, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using EVT0 = cutlass::epilogue::threadblock::Sm80EVT<Mul0, Accum, XScale>;
  using Mul1 = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, ElementCompute, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using EVT1 = cutlass::epilogue::threadblock::Sm80EVT<Mul1, EVT0, WScale>;
  using Add2 = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::plus, ElementOutput, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using EVT2 = cutlass::epilogue::threadblock::Sm80EVT<Add2, EVT1, Bias>;
  using StoreD = cutlass::epilogue::threadblock::VisitorAuxStore<
      ThreadMap, ElementOutput, cutlass::FloatRoundStyle::round_to_nearest,
      cute::Stride<int64_t, _1, int64_t>>;
  using EVTD = cutlass::epilogue::threadblock::Sm80EVT<StoreD, EVT2>;

  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
      ElementA, LayoutA, cutlass::ComplexTransform::kNone, AlignA, ElementB,
      LayoutB, cutlass::ComplexTransform::kNone, AlignB, ElementC, LayoutC,
      AlignC, ElementAcc, ElementCompute, cutlass::arch::OpClassTensorOp,
      ArchTag, TB, Warp, Inst, EVTD, ThreadblockSwizzle, NumStages,
      cutlass::arch::OpMultiplyAddSaturate, EVTStages>::GemmKernel;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};
using G = FusedInt8Gemm<cutlass::bfloat16_t, 128, 256, 64, 64, 64, 64, 3>;
using Mma = typename G::GemmKernel::Mma;
static_assert(Mma::FragmentC::kElements == 128);
static_assert(Mma::Base::WarpCount::kM == 2 && Mma::Base::WarpCount::kN == 4 &&
              Mma::Base::WarpCount::kK == 1);
} // namespace h3_qkv
