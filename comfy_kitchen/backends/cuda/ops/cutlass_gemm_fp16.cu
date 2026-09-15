/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
  * FP16 GEMM with fp16 accumulators via CUTLASS (EVT epilogue):
  *   D = A @ B^T (+ bias), or resid + rscale * (acc + bias) fused.
  * On sm_120 fp16-accumulate HMMA issues at twice the rate of fp32-accumulate;
  * only route here when the user opted into fp16 accumulation (torch's
  * allow_fp16_accumulation numerics). Returns false -> caller uses cuBLAS.
 */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#ifdef COMFY_HAVE_CUTLASS

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"

#include "cutlass_gemm_common.cuh"

namespace {
using namespace cute;
using comfy_cutlass::ThreadblockSwizzleLeanStreamK;

using half_t = cutlass::half_t;

// Shared kernel plumbing for the fp16-accum GEMM variants: row-major A [M,K],
// B given as [N,K] row-major read column-major, row-major D [M,N].
template <int TBM, int TBN, int TBK, int WM, int WN, int WK, int NumStages,
          typename EVTD, typename ThreadblockSwizzle>
struct Fp16GemmBase {
    using TB   = cutlass::gemm::GemmShape<TBM, TBN, TBK>;
    using Warp = cutlass::gemm::GemmShape<WM, WN, WK>;
    using Inst = cutlass::gemm::GemmShape<16, 8, 16>;
    static constexpr int EVTStages = 1;

    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
        half_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,
        half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,
        half_t, cutlass::layout::RowMajor, 8,
        half_t, float,
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        TB, Warp, Inst, EVTD,
        ThreadblockSwizzle,
        NumStages, cutlass::arch::OpMultiplyAdd, EVTStages>::GemmKernel;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    template <typename Callback>
    static bool launch(const half_t* A, const half_t* B, const Callback& cb,
                       int M, int N, int K, cudaStream_t stream) {
        return comfy_cutlass::launch_universal<Gemm>(
            A, B, cb, M, N, K, stream);
    }
};

template <int TBM, int TBN, int TBK, int WM, int WN, int WK, int NumStages,
          typename ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>>
struct Fp16Gemm {
    using TB   = cutlass::gemm::GemmShape<TBM, TBN, TBK>;
    using Warp = cutlass::gemm::GemmShape<WM, WN, WK>;
    using ThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<TB, Warp, half_t, 8, 1>;
    using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;
    // bias == nullptr broadcasts null_default(0) and is an identity add.
    using Bias  = cutlass::epilogue::threadblock::VisitorRowBroadcast<ThreadMap, half_t, cute::Stride<_0, _1, int32_t>>;
    using Add0 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::plus, half_t, float, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT0 = cutlass::epilogue::threadblock::Sm80EVT<Add0, Accum, Bias>;
    using StoreD = cutlass::epilogue::threadblock::VisitorAuxStore<ThreadMap, half_t, cutlass::FloatRoundStyle::round_to_nearest, cute::Stride<int64_t, _1, int64_t>>;
    using EVTD = cutlass::epilogue::threadblock::Sm80EVT<StoreD, EVT0>;
    using Base = Fp16GemmBase<TBM, TBN, TBK, WM, WN, WK, NumStages, EVTD, ThreadblockSwizzle>;

    static bool run(const half_t* A, const half_t* B, const half_t* bias,
                    half_t* D, int M, int N, int K, cudaStream_t stream) {
        typename EVTD::Arguments cb{
            { {}, {const_cast<half_t*>(bias), half_t(0), {_0{}, _1{}, N}}, {} },
            {D, {int64_t(N), _1{}, int64_t(M) * N}} };
        return Base::launch(A, B, cb, M, N, K, stream);
    }
};

// D = resid + rscale[n] * (acc + bias[n]) -- a pre-norm block's addcmul.
// The multiply and adds run in float; only loads/stores are half.
template <int TBM, int TBN, int TBK, int WM, int WN, int WK, int NumStages,
          typename ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>>
struct Fp16GemmResidual {
    using TB   = cutlass::gemm::GemmShape<TBM, TBN, TBK>;
    using Warp = cutlass::gemm::GemmShape<WM, WN, WK>;
    using ThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<TB, Warp, half_t, 8, 1>;
    using Accum  = cutlass::epilogue::threadblock::VisitorAccFetch;
    using Bias   = cutlass::epilogue::threadblock::VisitorRowBroadcast<ThreadMap, half_t, cute::Stride<_0, _1, int32_t>>;
    using RScale = cutlass::epilogue::threadblock::VisitorRowBroadcast<ThreadMap, half_t, cute::Stride<_0, _1, int32_t>>;
    using Resid  = cutlass::epilogue::threadblock::VisitorAuxLoad<ThreadMap, half_t, cute::Stride<int64_t, _1, int64_t>>;
    using Add0 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::plus, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT0 = cutlass::epilogue::threadblock::Sm80EVT<Add0, Accum, Bias>;
    using Mul1 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT1 = cutlass::epilogue::threadblock::Sm80EVT<Mul1, EVT0, RScale>;
    using Add2 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::plus, half_t, float, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT2 = cutlass::epilogue::threadblock::Sm80EVT<Add2, EVT1, Resid>;
    using StoreD = cutlass::epilogue::threadblock::VisitorAuxStore<ThreadMap, half_t, cutlass::FloatRoundStyle::round_to_nearest, cute::Stride<int64_t, _1, int64_t>>;
    using EVTD = cutlass::epilogue::threadblock::Sm80EVT<StoreD, EVT2>;
    using Base = Fp16GemmBase<TBM, TBN, TBK, WM, WN, WK, NumStages, EVTD, ThreadblockSwizzle>;

    static bool run(const half_t* A, const half_t* B, const half_t* bias,
                    const half_t* rscale, const half_t* resid,
                    half_t* D, int M, int N, int K, cudaStream_t stream) {
        typename EVTD::Arguments cb{
            { { { {}, {const_cast<half_t*>(bias), half_t(0), {_0{}, _1{}, N}}, {} },
                {const_cast<half_t*>(rscale), half_t(0), {_0{}, _1{}, N}}, {} },
              {const_cast<half_t*>(resid), half_t(0), {int64_t(N), _1{}, int64_t(M) * N}}, {} },
            {D, {int64_t(N), _1{}, int64_t(M) * N}} };
        return Base::launch(A, B, cb, M, N, K, stream);
    }
};

// Benchmarked on sm_120 at the H3 decoder shapes; 2 is stream-K for deep K.
// select_fp16_config indexes this list; both epilogue variants instantiate from it.
using comfy_cutlass::TileConfig;
using comfy_cutlass::ConfigList;
using Fp16Configs = ConfigList<
    TileConfig<128, 256, 32, 64, 64, 32, 3>,                                     // 0
    TileConfig<128, 128, 32, 64, 64, 32, 4>,                                     // 1
    TileConfig<128, 128, 32, 64, 64, 32, 4, 16, ThreadblockSwizzleLeanStreamK>>; // 2 (stream-K)

// Launches too small to fill the GPU lose to cuBLAS's split-K (up to 10x on
// sm_120): plain tiles need ~96 threadblocks, stream-K ~32. -1 -> cuBLAS.
constexpr int64_t kMinTilesPlain = 96;
constexpr int64_t kMinTilesStreamK = 32;

int select_fp16_config(int m, int n, int k) {
    const int config = k > 4096 ? 2 : (n <= 3072 ? 0 : (n <= 8192 ? 1 : 0));
    const int tile_n = config == 0 ? 256 : 128;
    const int64_t tiles = static_cast<int64_t>((m + 127) / 128) * ((n + tile_n - 1) / tile_n);
    if (tiles < (config == 2 ? kMinTilesStreamK : kMinTilesPlain)) return -1;
    return config;
}

template <typename Launch>
bool launch_fp16_heuristic(int m, int n, int k, Launch launch) {
    const int selected = select_fp16_config(m, n, k);
    if (selected < 0) return false;
    if (launch(selected)) return true;
    for (int config = 0; config < Fp16Configs::size; ++config) {
        if (config != selected && launch(config)) return true;
    }
    return false;
}

template <typename C>
using PlainGemm = Fp16Gemm<C::TBM, C::TBN, C::TBK, C::WM, C::WN, C::WK, C::NumStages,
                           typename C::ThreadblockSwizzle>;
template <typename C>
using ResidualGemm = Fp16GemmResidual<C::TBM, C::TBN, C::TBK, C::WM, C::WN, C::WK, C::NumStages,
                                      typename C::ThreadblockSwizzle>;

using Fp16Fn = bool (*)(const half_t*, const half_t*, const half_t*, half_t*, int, int, int, cudaStream_t);
using Fp16ResidualFn = bool (*)(const half_t*, const half_t*, const half_t*, const half_t*,
                                const half_t*, half_t*, int, int, int, cudaStream_t);

template <typename... Cs>
const Fp16Fn* fp16_runners(ConfigList<Cs...>) {
    static const Fp16Fn runners[sizeof...(Cs)] = {&PlainGemm<Cs>::run...};
    return runners;
}

template <typename... Cs>
const Fp16ResidualFn* fp16_residual_runners(ConfigList<Cs...>) {
    static const Fp16ResidualFn runners[sizeof...(Cs)] = {&ResidualGemm<Cs>::run...};
    return runners;
}

bool dispatch_fp16(const half_t* A, const half_t* B, const half_t* bias,
                   half_t* D, int M, int N, int K, cudaStream_t stream) {
    // bias == nullptr is fine: VisitorRowBroadcast's EnableNullptr fallback
    // broadcasts null_default (0), verified bitwise-identical to a plain store.
    const Fp16Fn* runners = fp16_runners(Fp16Configs{});
    return launch_fp16_heuristic(M, N, K, [&](int config) {
        return runners[config](A, B, bias, D, M, N, K, stream);
    });
}

bool dispatch_fp16_residual(const half_t* A, const half_t* B, const half_t* bias,
                            const half_t* rscale, const half_t* resid,
                            half_t* D, int M, int N, int K, cudaStream_t stream) {
    const Fp16ResidualFn* runners = fp16_residual_runners(Fp16Configs{});
    return launch_fp16_heuristic(M, N, K, [&](int config) {
        return runners[config](A, B, bias, rscale, resid, D, M, N, K, stream);
    });
}

// Shape gate for both entry points: a zero-K linear is the bias broadcast, and
// past a few batched decoder tiles (M > 8192) cuBLAS wins. Both decline.
bool fp16_gemm_shape_ok(int64_t M, int64_t N, int64_t K) {
    return K != 0 && K % 8 == 0 && N % 8 == 0 && M <= 8192;
}

}  // namespace

extern "C" {

bool launch_cutlass_fp16_linear(
    const void* A, const void* B, const void* bias, void* D,
    int64_t M, int64_t N, int64_t K, cudaStream_t stream)
{
    if (M == 0 || N == 0) return true;
    if (!fp16_gemm_shape_ok(M, N, K)) return false;
    return dispatch_fp16(
        static_cast<const half_t*>(A), static_cast<const half_t*>(B),
        static_cast<const half_t*>(bias), static_cast<half_t*>(D),
        M, N, K, stream);
}

bool launch_cutlass_fp16_linear_residual(
    const void* A, const void* B, const void* bias, const void* rscale,
    const void* resid, void* D, int64_t M, int64_t N, int64_t K,
    cudaStream_t stream)
{
    if (M == 0 || N == 0) return true;
    if (!fp16_gemm_shape_ok(M, N, K)) return false;
    // bias may be nullptr: the RowBroadcast visitor broadcasts null_default(0).
    if (rscale == nullptr || resid == nullptr) return false;
    return dispatch_fp16_residual(
        static_cast<const half_t*>(A), static_cast<const half_t*>(B),
        static_cast<const half_t*>(bias), static_cast<const half_t*>(rscale),
        static_cast<const half_t*>(resid), static_cast<half_t*>(D),
        M, N, K, stream);
}

}  // extern "C"

#else  // !COMFY_HAVE_CUTLASS -- stub; caller falls back to cuBLAS.

extern "C" bool launch_cutlass_fp16_linear(
    const void*, const void*, const void*, void*,
    int64_t, int64_t, int64_t, cudaStream_t) {
    return false;
}

extern "C" bool launch_cutlass_fp16_linear_residual(
    const void*, const void*, const void*, const void*, const void*,
    void*, int64_t, int64_t, int64_t, cudaStream_t) {
    return false;
}

#endif
