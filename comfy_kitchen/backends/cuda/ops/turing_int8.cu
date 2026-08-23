// SPDX-License-Identifier: Apache-2.0
// Native SM75 signed-INT8 GEMM with fused dequantization.
// Python owns all architecture dispatch.

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
using namespace cute;

template <typename ThreadMap, bool Scalar>
struct TuringWeightScaleBroadcast;

template <typename ThreadMap>
struct TuringWeightScaleBroadcast<ThreadMap, false> {
    using Type = cutlass::epilogue::threadblock::VisitorRowBroadcast<
        ThreadMap, float, cute::Stride<_0, _1, int32_t>>;

    static typename Type::Arguments arguments(const float* scale, int n) {
        return {scale, 0.f, {_0{}, _1{}, n}};
    }
};

template <typename ThreadMap>
struct TuringWeightScaleBroadcast<ThreadMap, true> {
    using Type = cutlass::epilogue::threadblock::VisitorScalarBroadcast<float>;

    static typename Type::Arguments arguments(const float* scale, int) {
        typename Type::Arguments result{};
        result.scalar_ptrs[0] = scale;
        return result;
    }
};

template <typename Output, bool ScalarWeightScale, int TBM, int TBN, int WM, int WN>
struct TuringInt8Gemm {
    using ElementA = int8_t;
    using ElementB = int8_t;
    using ElementC = Output;
    using ElementAcc = int32_t;
    using ElementCompute = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    static constexpr int AlignA = 16;
    static constexpr int AlignB = 16;
    static constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementC>::value;
    using TB = cutlass::gemm::GemmShape<TBM, TBN, 64>;
    using Warp = cutlass::gemm::GemmShape<WM, WN, 64>;
    using Inst = cutlass::gemm::GemmShape<8, 8, 16>;
    static constexpr int EVTStages = 1;

    using ThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
        TB, Warp, ElementC, AlignC, EVTStages>;
    using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;
    using XScale = cutlass::epilogue::threadblock::VisitorColBroadcast<
        ThreadMap, ElementCompute, cute::Stride<_1, _0, int32_t>>;
    using WScale = typename TuringWeightScaleBroadcast<
        ThreadMap, ScalarWeightScale>::Type;
    using Bias = cutlass::epilogue::threadblock::VisitorRowBroadcast<
        ThreadMap, Output, cute::Stride<_0, _1, int32_t>>;
    using Mul0 = cutlass::epilogue::threadblock::VisitorCompute<
        cutlass::multiplies, ElementCompute, ElementCompute,
        cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT0 = cutlass::epilogue::threadblock::Sm80EVT<Mul0, Accum, XScale>;
    using Mul1 = cutlass::epilogue::threadblock::VisitorCompute<
        cutlass::multiplies, ElementCompute, ElementCompute,
        cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT1 = cutlass::epilogue::threadblock::Sm80EVT<Mul1, EVT0, WScale>;
    using Add2 = cutlass::epilogue::threadblock::VisitorCompute<
        cutlass::plus, Output, ElementCompute,
        cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT2 = cutlass::epilogue::threadblock::Sm80EVT<Add2, EVT1, Bias>;
    using StoreD = cutlass::epilogue::threadblock::VisitorAuxStore<
        ThreadMap, Output, cutlass::FloatRoundStyle::round_to_nearest,
        cute::Stride<int64_t, _1, int64_t>>;
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
        const int8_t* activation,
        const int8_t* weight,
        const float* activation_scale,
        const float* weight_scale,
        const Output* bias,
        Output* output,
        int m,
        int n,
        int k,
        cudaStream_t stream) {
        cutlass::gemm::GemmCoord problem(m, n, k);
        const auto weight_scale_args = TuringWeightScaleBroadcast<
            ThreadMap, ScalarWeightScale>::arguments(weight_scale, n);
        typename EVTD::Arguments callbacks{
            {{{{}, {const_cast<float*>(activation_scale), 0.f,
                    {_1{}, _0{}, m}}, {}},
              weight_scale_args, {}},
             {const_cast<Output*>(bias), Output(0), {_0{}, _1{}, n}}, {}},
            {output, {n, _1{}, m * n}}};
        typename Gemm::Arguments arguments(
            cutlass::gemm::GemmUniversalMode::kGemm,
            problem,
            1,
            callbacks,
            const_cast<int8_t*>(activation),
            const_cast<int8_t*>(weight),
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

template <typename Output, bool ScalarWeightScale, int TBM, int TBN, int WM, int WN>
struct TuringInt8GemmNoBias {
    using ElementA = int8_t;
    using ElementB = int8_t;
    using ElementC = Output;
    using ElementAcc = int32_t;
    using ElementCompute = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    static constexpr int AlignA = 16;
    static constexpr int AlignB = 16;
    static constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementC>::value;
    using TB = cutlass::gemm::GemmShape<TBM, TBN, 64>;
    using Warp = cutlass::gemm::GemmShape<WM, WN, 64>;
    using Inst = cutlass::gemm::GemmShape<8, 8, 16>;
    static constexpr int EVTStages = 1;

    using ThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
        TB, Warp, ElementC, AlignC, EVTStages>;
    using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;
    using XScale = cutlass::epilogue::threadblock::VisitorColBroadcast<
        ThreadMap, ElementCompute, cute::Stride<_1, _0, int32_t>>;
    using WScale = typename TuringWeightScaleBroadcast<
        ThreadMap, ScalarWeightScale>::Type;
    using Mul0 = cutlass::epilogue::threadblock::VisitorCompute<
        cutlass::multiplies, ElementCompute, ElementCompute,
        cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT0 = cutlass::epilogue::threadblock::Sm80EVT<Mul0, Accum, XScale>;
    using Mul1 = cutlass::epilogue::threadblock::VisitorCompute<
        cutlass::multiplies, Output, ElementCompute,
        cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT1 = cutlass::epilogue::threadblock::Sm80EVT<Mul1, EVT0, WScale>;
    using StoreD = cutlass::epilogue::threadblock::VisitorAuxStore<
        ThreadMap, Output, cutlass::FloatRoundStyle::round_to_nearest,
        cute::Stride<int64_t, _1, int64_t>>;
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
        const int8_t* activation,
        const int8_t* weight,
        const float* activation_scale,
        const float* weight_scale,
        Output* output,
        int m,
        int n,
        int k,
        cudaStream_t stream) {
        cutlass::gemm::GemmCoord problem(m, n, k);
        const auto weight_scale_args = TuringWeightScaleBroadcast<
            ThreadMap, ScalarWeightScale>::arguments(weight_scale, n);
        typename EVTD::Arguments callbacks{
            {{{}, {const_cast<float*>(activation_scale), 0.f,
                   {_1{}, _0{}, m}}, {}},
             weight_scale_args, {}},
            {output, {n, _1{}, m * n}}};
        typename Gemm::Arguments arguments(
            cutlass::gemm::GemmUniversalMode::kGemm,
            problem,
            1,
            callbacks,
            const_cast<int8_t*>(activation),
            const_cast<int8_t*>(weight),
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

template <typename Output, bool ScalarWeightScale, int TBM, int TBN, int WM, int WN>
bool run_turing_int8_tile(
    const int8_t* activation,
    const int8_t* weight,
    const float* activation_scale,
    const float* weight_scale,
    const Output* bias,
    Output* output,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
    if (bias != nullptr) {
        return TuringInt8Gemm<Output, ScalarWeightScale, TBM, TBN, WM, WN>::run(
            activation, weight, activation_scale, weight_scale, bias,
            output, m, n, k, stream);
    }
    return TuringInt8GemmNoBias<Output, ScalarWeightScale, TBM, TBN, WM, WN>::run(
        activation, weight, activation_scale, weight_scale,
        output, m, n, k, stream);
}

template <typename Output, bool ScalarWeightScale>
bool dispatch_turing_int8(
    const int8_t* activation,
    const int8_t* weight,
    const float* activation_scale,
    const float* weight_scale,
    const Output* bias,
    Output* output,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
    if (m <= 32) {
        return run_turing_int8_tile<Output, ScalarWeightScale, 16, 64, 16, 32>(
            activation, weight, activation_scale, weight_scale, bias,
            output, m, n, k, stream);
    }
    if (m <= 128 && n < 8192) {
        return run_turing_int8_tile<Output, ScalarWeightScale, 32, 64, 32, 32>(
            activation, weight, activation_scale, weight_scale, bias,
            output, m, n, k, stream);
    }
    if (m <= 512) {
        return run_turing_int8_tile<Output, ScalarWeightScale, 64, 128, 32, 64>(
            activation, weight, activation_scale, weight_scale, bias,
            output, m, n, k, stream);
    }
    return run_turing_int8_tile<Output, ScalarWeightScale, 128, 256, 64, 64>(
        activation, weight, activation_scale, weight_scale, bias,
        output, m, n, k, stream);
}

template <typename Output>
bool dispatch_scale_mode(
    const int8_t* activation,
    const int8_t* weight,
    const float* activation_scale,
    const float* weight_scale,
    const Output* bias,
    Output* output,
    int m,
    int n,
    int k,
    bool scalar_weight_scale,
    cudaStream_t stream) {
    if (scalar_weight_scale) {
        return dispatch_turing_int8<Output, true>(
            activation, weight, activation_scale, weight_scale, bias,
            output, m, n, k, stream);
    }
    return dispatch_turing_int8<Output, false>(
        activation, weight, activation_scale, weight_scale, bias,
        output, m, n, k, stream);
}

}  // namespace
#endif

extern "C" bool launch_cutlass_turing_int8_dequant(
    const void* activation,
    const void* weight,
    const void* activation_scale,
    const void* weight_scale,
    const void* bias,
    void* output,
    int64_t m,
    int64_t n,
    int64_t k,
    int output_dtype_code,
    bool scalar_weight_scale,
    cudaStream_t stream) {
#ifdef COMFY_HAVE_CUTLASS
    if (m == 0 || n == 0 || k == 0) return true;
    if (k % 16 != 0 || m > INT_MAX || n > INT_MAX || k > INT_MAX) return false;
    const auto* a = static_cast<const int8_t*>(activation);
    const auto* b = static_cast<const int8_t*>(weight);
    const auto* x_scale = static_cast<const float*>(activation_scale);
    const auto* w_scale = static_cast<const float*>(weight_scale);
    switch (output_dtype_code) {
        case 0:
            return dispatch_scale_mode(
                a, b, x_scale, w_scale, static_cast<const float*>(bias),
                static_cast<float*>(output), m, n, k, scalar_weight_scale, stream);
        case 1:
            return dispatch_scale_mode(
                a, b, x_scale, w_scale, static_cast<const cutlass::half_t*>(bias),
                static_cast<cutlass::half_t*>(output), m, n, k,
                scalar_weight_scale, stream);
        case 2:
            return dispatch_scale_mode(
                a, b, x_scale, w_scale, static_cast<const cutlass::bfloat16_t*>(bias),
                static_cast<cutlass::bfloat16_t*>(output), m, n, k,
                scalar_weight_scale, stream);
        default:
            return false;
    }
#else
    (void)activation;
    (void)weight;
    (void)activation_scale;
    (void)weight_scale;
    (void)bias;
    (void)output;
    (void)m;
    (void)n;
    (void)k;
    (void)output_dtype_code;
    (void)scalar_weight_scale;
    (void)stream;
    return false;
#endif
}
