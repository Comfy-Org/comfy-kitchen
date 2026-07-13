// SPDX-License-Identifier: Apache-2.0
// Native SM75 packed signed-INT4 GEMM. Python owns all architecture dispatch.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstdint>

#ifdef COMFY_HAVE_CUTLASS
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"

namespace {

template <typename ElementOutput, int TBM, int TBN, int WM, int WN>
struct TuringInt4Gemm {
    using ElementA = cutlass::int4b_t;
    using ElementB = cutlass::int4b_t;
    using ElementC = ElementOutput;
    using ElementAcc = int32_t;
    using ElementCompute = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    static constexpr int AlignA = 32;
    static constexpr int AlignB = 32;
    static constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementC>::value;
    using TB = cutlass::gemm::GemmShape<TBM, TBN, 128>;
    using Warp = cutlass::gemm::GemmShape<WM, WN, 128>;
    using Inst = cutlass::gemm::GemmShape<8, 8, 32>;
    static constexpr int EVTStages = 1;

    using ThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
        TB, Warp, ElementC, AlignC, EVTStages>;
    using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;
    using XScale = cutlass::epilogue::threadblock::VisitorColBroadcast<
        ThreadMap, ElementCompute, cute::Stride<cute::_1, cute::_0, int32_t>>;
    using WScale = cutlass::epilogue::threadblock::VisitorRowBroadcast<
        ThreadMap, ElementCompute, cute::Stride<cute::_0, cute::_1, int32_t>>;
    using Bias = cutlass::epilogue::threadblock::VisitorRowBroadcast<
        ThreadMap, ElementCompute, cute::Stride<cute::_0, cute::_1, int32_t>>;
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
        cute::Stride<int64_t, cute::_1, int64_t>>;
    using EVTD = cutlass::epilogue::threadblock::Sm80EVT<StoreD, EVT2>;

    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
        ElementA, LayoutA, cutlass::ComplexTransform::kNone, AlignA,
        ElementB, LayoutB, cutlass::ComplexTransform::kNone, AlignB,
        ElementC, LayoutC, AlignC, ElementAcc, ElementCompute,
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
        TB, Warp, Inst, EVTD,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2, cutlass::arch::OpMultiplyAddSaturate, EVTStages>::GemmKernel;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    static bool run(
        const int8_t* a,
        const int8_t* b,
        const float* activation_scale,
        const float* weight_scale,
        const float* bias,
        ElementOutput* output,
        int m,
        int n,
        int k,
        cudaStream_t stream) {
        cutlass::gemm::GemmCoord problem(m, n, k);
        typename EVTD::Arguments callbacks{
            {{{{}, {const_cast<float*>(activation_scale), 0.f,
                    {cute::_1{}, cute::_0{}, m}}, {}},
              {const_cast<float*>(weight_scale), 0.f,
               {cute::_0{}, cute::_1{}, n}}, {}},
             {const_cast<float*>(bias), 0.f,
              {cute::_0{}, cute::_1{}, n}}, {}},
            {output, {n, cute::_1{}, m * n}}};
        typename Gemm::Arguments arguments(
            cutlass::gemm::GemmUniversalMode::kGemm,
            problem,
            1,
            callbacks,
            reinterpret_cast<cutlass::int4b_t*>(const_cast<int8_t*>(a)),
            reinterpret_cast<cutlass::int4b_t*>(const_cast<int8_t*>(b)),
            nullptr,
            nullptr,
            static_cast<int64_t>(m) * k,
            static_cast<int64_t>(n) * k,
            0,
            0,
            k,
            k,
            0,
            0);
        Gemm gemm;
        if (gemm.can_implement(arguments) != cutlass::Status::kSuccess) return false;
        if (Gemm::get_workspace_size(arguments) != 0) return false;
        if (gemm.initialize(arguments, nullptr, stream) != cutlass::Status::kSuccess) return false;
        return gemm(stream) == cutlass::Status::kSuccess;
    }
};

template <typename ElementOutput, int TBM, int TBN, int WM, int WN>
struct TuringInt4GemmNoBias {
    using ElementA = cutlass::int4b_t;
    using ElementB = cutlass::int4b_t;
    using ElementC = ElementOutput;
    using ElementAcc = int32_t;
    using ElementCompute = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    static constexpr int AlignA = 32;
    static constexpr int AlignB = 32;
    static constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementC>::value;
    using TB = cutlass::gemm::GemmShape<TBM, TBN, 128>;
    using Warp = cutlass::gemm::GemmShape<WM, WN, 128>;
    using Inst = cutlass::gemm::GemmShape<8, 8, 32>;
    static constexpr int EVTStages = 1;

    using ThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
        TB, Warp, ElementC, AlignC, EVTStages>;
    using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;
    using XScale = cutlass::epilogue::threadblock::VisitorColBroadcast<
        ThreadMap, ElementCompute, cute::Stride<cute::_1, cute::_0, int32_t>>;
    using WScale = cutlass::epilogue::threadblock::VisitorRowBroadcast<
        ThreadMap, ElementCompute, cute::Stride<cute::_0, cute::_1, int32_t>>;
    using Mul0 = cutlass::epilogue::threadblock::VisitorCompute<
        cutlass::multiplies, ElementCompute, ElementCompute,
        cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT0 = cutlass::epilogue::threadblock::Sm80EVT<Mul0, Accum, XScale>;
    using Mul1 = cutlass::epilogue::threadblock::VisitorCompute<
        cutlass::multiplies, ElementOutput, ElementCompute,
        cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT1 = cutlass::epilogue::threadblock::Sm80EVT<Mul1, EVT0, WScale>;
    using StoreD = cutlass::epilogue::threadblock::VisitorAuxStore<
        ThreadMap, ElementOutput, cutlass::FloatRoundStyle::round_to_nearest,
        cute::Stride<int64_t, cute::_1, int64_t>>;
    using EVTD = cutlass::epilogue::threadblock::Sm80EVT<StoreD, EVT1>;

    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
        ElementA, LayoutA, cutlass::ComplexTransform::kNone, AlignA,
        ElementB, LayoutB, cutlass::ComplexTransform::kNone, AlignB,
        ElementC, LayoutC, AlignC, ElementAcc, ElementCompute,
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
        TB, Warp, Inst, EVTD,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2, cutlass::arch::OpMultiplyAddSaturate, EVTStages>::GemmKernel;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    static bool run(
        const int8_t* a,
        const int8_t* b,
        const float* activation_scale,
        const float* weight_scale,
        ElementOutput* output,
        int m,
        int n,
        int k,
        cudaStream_t stream) {
        cutlass::gemm::GemmCoord problem(m, n, k);
        typename EVTD::Arguments callbacks{
            {{{}, {const_cast<float*>(activation_scale), 0.f,
                   {cute::_1{}, cute::_0{}, m}}, {}},
             {const_cast<float*>(weight_scale), 0.f,
              {cute::_0{}, cute::_1{}, n}}, {}},
            {output, {n, cute::_1{}, m * n}}};
        typename Gemm::Arguments arguments(
            cutlass::gemm::GemmUniversalMode::kGemm,
            problem,
            1,
            callbacks,
            reinterpret_cast<cutlass::int4b_t*>(const_cast<int8_t*>(a)),
            reinterpret_cast<cutlass::int4b_t*>(const_cast<int8_t*>(b)),
            nullptr,
            nullptr,
            static_cast<int64_t>(m) * k,
            static_cast<int64_t>(n) * k,
            0,
            0,
            k,
            k,
            0,
            0);
        Gemm gemm;
        if (gemm.can_implement(arguments) != cutlass::Status::kSuccess) return false;
        if (Gemm::get_workspace_size(arguments) != 0) return false;
        if (gemm.initialize(arguments, nullptr, stream) != cutlass::Status::kSuccess) return false;
        return gemm(stream) == cutlass::Status::kSuccess;
    }
};

template <typename Output>
bool dispatch_turing_int4(
    const int8_t* a,
    const int8_t* b,
    const float* activation_scale,
    const float* weight_scale,
    const float* bias,
    Output* output,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
    if (m <= 64) {
        if (bias != nullptr) {
            return TuringInt4Gemm<Output, 16, 64, 16, 32>::run(
                a, b, activation_scale, weight_scale, bias, output, m, n, k, stream);
        }
        return TuringInt4GemmNoBias<Output, 16, 64, 16, 32>::run(
            a, b, activation_scale, weight_scale, output, m, n, k, stream);
    }
    if (m <= 128) {
        if (bias != nullptr) {
            return TuringInt4Gemm<Output, 32, 64, 32, 32>::run(
                a, b, activation_scale, weight_scale, bias, output, m, n, k, stream);
        }
        return TuringInt4GemmNoBias<Output, 32, 64, 32, 32>::run(
            a, b, activation_scale, weight_scale, output, m, n, k, stream);
    }
    if (m <= 256) {
        if (bias != nullptr) {
            return TuringInt4Gemm<Output, 64, 64, 32, 32>::run(
                a, b, activation_scale, weight_scale, bias, output, m, n, k, stream);
        }
        return TuringInt4GemmNoBias<Output, 64, 64, 32, 32>::run(
            a, b, activation_scale, weight_scale, output, m, n, k, stream);
    }
    if (bias != nullptr) {
        return TuringInt4Gemm<Output, 128, 256, 64, 64>::run(
            a, b, activation_scale, weight_scale, bias, output, m, n, k, stream);
    }
    return TuringInt4GemmNoBias<Output, 128, 256, 64, 64>::run(
        a, b, activation_scale, weight_scale, output, m, n, k, stream);
}

}  // namespace
#endif

extern "C" bool launch_cutlass_turing_int4_dequant(
    const void* a,
    const void* b,
    const void* activation_scale,
    const void* weight_scale,
    const void* bias,
    void* output,
    int64_t m,
    int64_t n,
    int64_t k,
    int output_dtype_code,
    cudaStream_t stream) {
#ifdef COMFY_HAVE_CUTLASS
    if (m == 0 || n == 0 || k == 0) return true;
    if (k % 64 != 0 || m > INT_MAX || n > INT_MAX || k > INT_MAX) return false;
    const auto* activation = static_cast<const int8_t*>(a);
    const auto* weights = static_cast<const int8_t*>(b);
    const auto* x_scale = static_cast<const float*>(activation_scale);
    const auto* w_scale = static_cast<const float*>(weight_scale);
    const auto* bias_values = static_cast<const float*>(bias);
    switch (output_dtype_code) {
        case 0:
            return dispatch_turing_int4(
                activation, weights, x_scale, w_scale, bias_values,
                static_cast<float*>(output), m, n, k, stream);
        case 1:
            return dispatch_turing_int4(
                activation, weights, x_scale, w_scale, bias_values,
                static_cast<cutlass::half_t*>(output), m, n, k, stream);
        case 2:
            return dispatch_turing_int4(
                activation, weights, x_scale, w_scale, bias_values,
                static_cast<cutlass::bfloat16_t*>(output), m, n, k, stream);
        default:
            return false;
    }
#else
    (void)a;
    (void)b;
    (void)activation_scale;
    (void)weight_scale;
    (void)bias;
    (void)output;
    (void)m;
    (void)n;
    (void)k;
    (void)output_dtype_code;
    (void)stream;
    return false;
#endif
}
