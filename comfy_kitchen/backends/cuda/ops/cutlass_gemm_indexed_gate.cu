/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Prequantized INT8 GEMM with an indexed BF16 gate + residual epilogue.
 * The GEMM result is rounded to BF16 before FP32 fma and final BF16 rounding.
 * Invalid gate row indices load zero, without host synchronization.
 */
#include <climits>
#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifdef COMFY_HAVE_CUTLASS

// clang-format off: CUTLASS visitor headers require GEMM declarations first.
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"

#include "cutlass_gemm_common.cuh"
#include "indexed_gate.cuh"
// clang-format on

namespace {
using namespace cute;
template <class T> struct IndexedGateFMA;
template <int N> struct IndexedGateFMA<cutlass::Array<float, N>> {
  CUTLASS_DEVICE cutlass::Array<float, N>
  operator()(cutlass::Array<float, N> const &a,
             cutlass::Array<float, N> const &b,
             cutlass::Array<float, N> const &c) const {
    cutlass::Array<float, N> out;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; i++)
      out[i] = __fmaf_rn(a[i], b[i], c[i]);
    return out;
  }
};
using comfy_cutlass::ThreadblockSwizzleLeanStreamK;

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
  static constexpr int EVTStages = 1;

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
  using Gate = cutlass::epilogue::threadblock::VisitorIndexedGateLoad<
      ThreadMap, ElementOutput, cute::Stride<int64_t, _1, int64_t>>;
  using Residual = cutlass::epilogue::threadblock::VisitorAuxLoad<
      ThreadMap, ElementOutput, cute::Stride<int64_t, _1, int64_t>>;
  using FMA = cutlass::epilogue::threadblock::VisitorCompute<
      IndexedGateFMA, ElementOutput, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using GateResidual =
      cutlass::epilogue::threadblock::Sm80EVT<FMA, EVT2, Gate, Residual>;
  using EVTD = cutlass::epilogue::threadblock::Sm80EVT<StoreD, GateResidual>;

  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
      ElementA, LayoutA, cutlass::ComplexTransform::kNone, AlignA, ElementB,
      LayoutB, cutlass::ComplexTransform::kNone, AlignB, ElementC, LayoutC,
      AlignC, ElementAcc, ElementCompute, cutlass::arch::OpClassTensorOp,
      ArchTag, TB, Warp, Inst, EVTD, ThreadblockSwizzle, NumStages,
      cutlass::arch::OpMultiplyAddSaturate, EVTStages>::GemmKernel;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  static bool run_strided(const int8_t *A, const int8_t *B, const float *xs,
                          const float *ws, const ElementOutput *bias,
                          ElementOutput *D, int M, int N, int K,
                          int output_stride, cudaStream_t stream,
                          const ElementOutput *gate, const int32_t *rows,
                          const ElementOutput *residual, int gate_rows) {
    const auto weight_scale_args =
        WeightScaleBroadcast<ThreadMap, ScalarWeightScale>::arguments(ws, N);
    typename EVT2::Arguments dq{
        {{{}, {const_cast<float *>(xs), 0.f, {_1{}, _0{}, M}}, {}},
         weight_scale_args,
         {}},
        {ElementOutput(0)},
        {}};
    typename EVTD::Arguments cb{
        {dq,
         {const_cast<ElementOutput *>(gate),
          ElementOutput(0),
          {int64_t(N), _1{}, int64_t(M) * N},
          rows,
          N,
          gate_rows},
         {const_cast<ElementOutput *>(residual),
          ElementOutput(0),
          {int64_t(N), _1{}, int64_t(M) * N}},
         {}},
        {D, {int64_t(output_stride), _1{}, int64_t(M) * output_stride}}};

    return comfy_cutlass::launch_universal<Gemm>(A, B, cb, M, N, K, stream);
  }
};

} // namespace
// Same tile and Stream-K implementation as the ordinary fused GEMM. The
// specialization changes the epilogue only and is opt-in through a new API.
extern "C" bool launch_cutlass_int8_indexed_gate(
    const int8_t *A, const int8_t *B, const float *xs, const float *ws, void *D,
    const void *gate, const int32_t *rows, const void *residual, int64_t M,
    int64_t N, int64_t K, int64_t gate_rows, cudaStream_t stream) {
  if (M == 0 || N == 0)
    return true;
  if (K == 0 || M > INT_MAX || N > INT_MAX || K > 131071 ||
      gate_rows > INT_MAX || K % 16 || N % 8 || !A || !B || !xs || !ws || !D ||
      !gate || !rows || !residual)
    return false;
  if ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B) |
       reinterpret_cast<uintptr_t>(D) | reinterpret_cast<uintptr_t>(gate) |
       reinterpret_cast<uintptr_t>(residual)) &
      15)
    return false;
  int device = 0, major = 0, minor = 0;
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                             device) != cudaSuccess ||
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                             device) != cudaSuccess)
    return false;
  // Broaden only with measurements; the first deployment target is SM120.
  if (major != 12 || minor != 0)
    return false;
  using O = cutlass::bfloat16_t;
  using G =
      FusedInt8Gemm<O, 128, 256, 64, 64, 64, 64, 3, cutlass::arch::Sm80, false,
                    16, comfy_cutlass::ThreadblockSwizzleLeanStreamK>;
  // Stream-K may grow a private workspace on a new stream. CUDA Graph
  // capture must never attempt cudaMalloc/cudaFree; use the workspace-free
  // identity schedule during capture, retaining the identical epilogue.
  cudaStreamCaptureStatus capture;
  if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess)
    return false;
  if (capture != cudaStreamCaptureStatusNone) {
    using CaptureGemm = FusedInt8Gemm<O, 128, 256, 64, 64, 64, 64, 3>;
    return CaptureGemm::run_strided(
        A, B, xs, ws, nullptr, static_cast<O *>(D), M, N, K, N, stream,
        static_cast<const O *>(gate), rows, static_cast<const O *>(residual),
        gate_rows);
  }
  return G::run_strided(A, B, xs, ws, nullptr, static_cast<O *>(D), M, N, K, N,
                        stream, static_cast<const O *>(gate), rows,
                        static_cast<const O *>(residual), gate_rows);
}
#else
extern "C" bool launch_cutlass_int8_indexed_gate(const int8_t *, const int8_t *,
                                                 const float *, const float *,
                                                 void *, const void *,
                                                 const int32_t *, const void *,
                                                 int64_t, int64_t, int64_t,
                                                 int64_t, cudaStream_t) {
  return false;
}
#endif
