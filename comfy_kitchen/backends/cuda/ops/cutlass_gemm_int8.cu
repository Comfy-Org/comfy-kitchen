/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * INT8 GEMM with a FUSED dequant epilogue via CUTLASS (EVT):
 *   D[m,n] = (sum_k A[m,k]*B[n,k]) * x_scale[m] * w_scale[n] + bias[n]   -> out dtype
 *
 * Replaces cuBLAS-GEMM(int32) + separate dequant with one near-peak kernel.
 * Multiple tile configs are instantiated and the fastest for each (M,N,K) is
 * picked at runtime and cached (like Triton's autotuner / cuBLAS's heuristic),
 * so it adapts to the GPU instead of relying on one hand-tuned tile.
 * Falls back to cuBLAS when CUTLASS is unavailable or no config can run.
 */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

#ifdef COMFY_HAVE_CUTLASS

#include <map>
#include <tuple>
#include <mutex>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"

namespace {
using namespace cute;

// One fused int8 GEMM, parameterized on output type AND tile/warp/stage config.
template <typename ElementOutput, int TBM, int TBN, int TBK, int WM, int WN, int WK, int NumStages>
struct FusedInt8Gemm {
    using ElementA = int8_t; using ElementB = int8_t;
    using ElementC = ElementOutput;
    using ElementAcc = int32_t; using ElementCompute = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;   // B[N,K] row == [K,N] col
    using LayoutC = cutlass::layout::RowMajor;
    static constexpr int AlignA = 16, AlignB = 16;
    static constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementC>::value;
    using TB   = cutlass::gemm::GemmShape<TBM, TBN, TBK>;
    using Warp = cutlass::gemm::GemmShape<WM, WN, WK>;
    using Inst = cutlass::gemm::GemmShape<16, 8, 32>;
    static constexpr int EVTStages = 1;

    using ThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<TB, Warp, ElementC, AlignC, EVTStages>;
    using Accum  = cutlass::epilogue::threadblock::VisitorAccFetch;
    using XScale = cutlass::epilogue::threadblock::VisitorColBroadcast<ThreadMap, ElementCompute, cute::Stride<_1, _0, int32_t>>;
    using WScale = cutlass::epilogue::threadblock::VisitorRowBroadcast<ThreadMap, ElementCompute, cute::Stride<_0, _1, int32_t>>;
    using Bias   = cutlass::epilogue::threadblock::VisitorRowBroadcast<ThreadMap, ElementCompute, cute::Stride<_0, _1, int32_t>>;
    using Mul0 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, ElementCompute, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT0 = cutlass::epilogue::threadblock::Sm80EVT<Mul0, Accum, XScale>;
    using Mul1 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, ElementCompute, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT1 = cutlass::epilogue::threadblock::Sm80EVT<Mul1, EVT0, WScale>;
    using Add2 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::plus, ElementOutput, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT2 = cutlass::epilogue::threadblock::Sm80EVT<Add2, EVT1, Bias>;
    using StoreD = cutlass::epilogue::threadblock::VisitorAuxStore<ThreadMap, ElementOutput, cutlass::FloatRoundStyle::round_to_nearest, cute::Stride<int64_t, _1, int64_t>>;
    using EVTD = cutlass::epilogue::threadblock::Sm80EVT<StoreD, EVT2>;

    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
        ElementA, LayoutA, cutlass::ComplexTransform::kNone, AlignA,
        ElementB, LayoutB, cutlass::ComplexTransform::kNone, AlignB,
        ElementC, LayoutC, AlignC,
        ElementAcc, ElementCompute,
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        TB, Warp, Inst, EVTD,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        NumStages, cutlass::arch::OpMultiplyAddSaturate, EVTStages>::GemmKernel;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    static bool run(const int8_t* A, const int8_t* B, const float* xs, const float* ws,
                    const float* bias, ElementOutput* D, int M, int N, int K, cudaStream_t stream) {
        cutlass::gemm::GemmCoord problem(M, N, K);
        typename EVTD::Arguments cb{
            { {  { {}, {const_cast<float*>(xs), 0.f, {_1{}, _0{}, M}}, {} },
                 {const_cast<float*>(ws), 0.f, {_0{}, _1{}, N}}, {} },
              {const_cast<float*>(bias), 0.f, {_0{}, _1{}, N}}, {} },
            {D, {N, _1{}, M * N}} };
        typename Gemm::Arguments args(
            cutlass::gemm::GemmUniversalMode::kGemm, problem, 1, cb,
            const_cast<int8_t*>(A), const_cast<int8_t*>(B), nullptr, nullptr,
            (int64_t)M * K, (int64_t)N * K, 0, 0, K, K, 0, 0);

        Gemm gemm;
        if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;
        if (Gemm::get_workspace_size(args) != 0) return false;  // kGemm mode -> 0; bail if not
        if (gemm.initialize(args, nullptr, stream) != cutlass::Status::kSuccess) return false;
        return gemm(stream) == cutlass::Status::kSuccess;
    }
};

template <typename ElementOutput, int TBM, int TBN, int TBK, int WM, int WN, int WK, int NumStages>
struct FusedInt8GemmNoBias {
    using ElementA = int8_t; using ElementB = int8_t;
    using ElementC = ElementOutput;
    using ElementAcc = int32_t; using ElementCompute = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    static constexpr int AlignA = 16, AlignB = 16;
    static constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementC>::value;
    using TB   = cutlass::gemm::GemmShape<TBM, TBN, TBK>;
    using Warp = cutlass::gemm::GemmShape<WM, WN, WK>;
    using Inst = cutlass::gemm::GemmShape<16, 8, 32>;
    static constexpr int EVTStages = 1;

    using ThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<TB, Warp, ElementC, AlignC, EVTStages>;
    using Accum  = cutlass::epilogue::threadblock::VisitorAccFetch;
    using XScale = cutlass::epilogue::threadblock::VisitorColBroadcast<ThreadMap, ElementCompute, cute::Stride<_1, _0, int32_t>>;
    using WScale = cutlass::epilogue::threadblock::VisitorRowBroadcast<ThreadMap, ElementCompute, cute::Stride<_0, _1, int32_t>>;
    using Mul0 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, ElementCompute, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT0 = cutlass::epilogue::threadblock::Sm80EVT<Mul0, Accum, XScale>;
    using Mul1 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, ElementOutput, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT1 = cutlass::epilogue::threadblock::Sm80EVT<Mul1, EVT0, WScale>;
    using StoreD = cutlass::epilogue::threadblock::VisitorAuxStore<ThreadMap, ElementOutput, cutlass::FloatRoundStyle::round_to_nearest, cute::Stride<int64_t, _1, int64_t>>;
    using EVTD = cutlass::epilogue::threadblock::Sm80EVT<StoreD, EVT1>;

    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
        ElementA, LayoutA, cutlass::ComplexTransform::kNone, AlignA,
        ElementB, LayoutB, cutlass::ComplexTransform::kNone, AlignB,
        ElementC, LayoutC, AlignC,
        ElementAcc, ElementCompute,
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        TB, Warp, Inst, EVTD,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        NumStages, cutlass::arch::OpMultiplyAddSaturate, EVTStages>::GemmKernel;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    static bool run(const int8_t* A, const int8_t* B, const float* xs, const float* ws,
                    ElementOutput* D, int M, int N, int K, cudaStream_t stream) {
        cutlass::gemm::GemmCoord problem(M, N, K);
        typename EVTD::Arguments cb{
            { { {}, {const_cast<float*>(xs), 0.f, {_1{}, _0{}, M}}, {} },
              {const_cast<float*>(ws), 0.f, {_0{}, _1{}, N}}, {} },
            {D, {N, _1{}, M * N}} };
        typename Gemm::Arguments args(
            cutlass::gemm::GemmUniversalMode::kGemm, problem, 1, cb,
            const_cast<int8_t*>(A), const_cast<int8_t*>(B), nullptr, nullptr,
            (int64_t)M * K, (int64_t)N * K, 0, 0, K, K, 0, 0);

        Gemm gemm;
        if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;
        if (Gemm::get_workspace_size(args) != 0) return false;
        if (gemm.initialize(args, nullptr, stream) != cutlass::Status::kSuccess) return false;
        return gemm(stream) == cutlass::Status::kSuccess;
    }
};

// Autotuning dispatcher: try each tile config, time it, cache the fastest per
// (M,N,K). First call for a shape pays the tuning cost; the rest hit the cache.
template <typename OutT>
bool dispatch_fused(const int8_t* A, const int8_t* B, const float* xs, const float* ws,
                    const float* bias, OutT* D, int M, int N, int K, cudaStream_t stream) {
    using Fn = bool (*)(const int8_t*, const int8_t*, const float*, const float*, const float*, OutT*, int, int, int, cudaStream_t);
    // Tile configs spanning big-GPU/large-M (wide) to small-GPU/small-M (more CTAs).
    static const Fn runners[] = {
        &FusedInt8Gemm<OutT, 128, 256, 64, 64, 64, 64, 3>::run,
        &FusedInt8Gemm<OutT, 128, 128, 64, 64, 64, 64, 4>::run,
        &FusedInt8Gemm<OutT,  64, 128, 64, 32, 64, 64, 4>::run,
    };
    constexpr int NC = sizeof(runners) / sizeof(runners[0]);

    static std::mutex mtx;
    static std::map<std::tuple<int, int, int>, int> cache;   // (M,N,K) -> best config (or -1 = none)
    const std::tuple<int, int, int> key{M, N, K};

    static thread_local int last_m = -1;
    static thread_local int last_n = -1;
    static thread_local int last_k = -1;
    static thread_local int last_best = -2;
    if (M == last_m && N == last_n && K == last_k) {
        if (last_best < 0) return false;
        return runners[last_best](A, B, xs, ws, bias, D, M, N, K, stream);
    }

    int best;
    {
        std::lock_guard<std::mutex> lk(mtx);
        auto it = cache.find(key);
        best = (it != cache.end()) ? it->second : -2;
    }
    if (best == -2) {  // not tuned yet
        best = -1;
        float best_ms = 1e30f;
        cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
        for (int i = 0; i < NC; ++i) {
            if (!runners[i](A, B, xs, ws, bias, D, M, N, K, stream)) continue;  // can't run / failed
            cudaStreamSynchronize(stream);
            cudaEventRecord(s, stream);
            for (int r = 0; r < 3; ++r) runners[i](A, B, xs, ws, bias, D, M, N, K, stream);
            cudaEventRecord(e, stream); cudaEventSynchronize(e);
            float ms = 0.f; cudaEventElapsedTime(&ms, s, e);
            if (ms < best_ms) { best_ms = ms; best = i; }
        }
        cudaEventDestroy(s); cudaEventDestroy(e);
        std::lock_guard<std::mutex> lk(mtx);
        cache[key] = best;
    }
    last_m = M;
    last_n = N;
    last_k = K;
    last_best = best;
    if (best < 0) return false;                              // fall back to cuBLAS
    return runners[best](A, B, xs, ws, bias, D, M, N, K, stream);  // final, correct write with best config
}

template <typename OutT>
bool dispatch_fused_no_bias(const int8_t* A, const int8_t* B, const float* xs, const float* ws,
                            OutT* D, int M, int N, int K, cudaStream_t stream) {
    using Fn = bool (*)(const int8_t*, const int8_t*, const float*, const float*, OutT*, int, int, int, cudaStream_t);
    static const Fn runners[] = {
        &FusedInt8GemmNoBias<OutT, 128, 256, 64, 64, 64, 64, 3>::run,
        &FusedInt8GemmNoBias<OutT, 128, 128, 64, 64, 64, 64, 4>::run,
        &FusedInt8GemmNoBias<OutT,  64, 128, 64, 32, 64, 64, 4>::run,
    };
    constexpr int NC = sizeof(runners) / sizeof(runners[0]);

    static std::mutex mtx;
    static std::map<std::tuple<int, int, int>, int> cache;
    const std::tuple<int, int, int> key{M, N, K};

    static thread_local int last_m = -1;
    static thread_local int last_n = -1;
    static thread_local int last_k = -1;
    static thread_local int last_best = -2;
    if (M == last_m && N == last_n && K == last_k) {
        if (last_best < 0) return false;
        return runners[last_best](A, B, xs, ws, D, M, N, K, stream);
    }

    int best;
    {
        std::lock_guard<std::mutex> lk(mtx);
        auto it = cache.find(key);
        best = (it != cache.end()) ? it->second : -2;
    }
    if (best == -2) {
        best = -1;
        float best_ms = 1e30f;
        cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
        for (int i = 0; i < NC; ++i) {
            if (!runners[i](A, B, xs, ws, D, M, N, K, stream)) continue;
            cudaStreamSynchronize(stream);
            cudaEventRecord(s, stream);
            for (int r = 0; r < 3; ++r) runners[i](A, B, xs, ws, D, M, N, K, stream);
            cudaEventRecord(e, stream); cudaEventSynchronize(e);
            float ms = 0.f; cudaEventElapsedTime(&ms, s, e);
            if (ms < best_ms) { best_ms = ms; best = i; }
        }
        cudaEventDestroy(s); cudaEventDestroy(e);
        std::lock_guard<std::mutex> lk(mtx);
        cache[key] = best;
    }
    last_m = M;
    last_n = N;
    last_k = K;
    last_best = best;
    if (best < 0) return false;
    return runners[best](A, B, xs, ws, D, M, N, K, stream);
}
}  // namespace

extern "C" {
// out_dtype_code: 0=float32, 1=float16, 2=bfloat16 (DTYPE_TO_CODE).
bool launch_cutlass_int8_dequant(
    const void* A, const void* B, const void* xs, const void* ws, const void* bias,
    void* D, int64_t M, int64_t N, int64_t K, int out_dtype_code, cudaStream_t stream)
{
    if (M == 0 || N == 0 || K == 0) return true;
    const int8_t* a = static_cast<const int8_t*>(A);
    const int8_t* b = static_cast<const int8_t*>(B);
    const float* x = static_cast<const float*>(xs);
    const float* w = static_cast<const float*>(ws);
    const float* bs = static_cast<const float*>(bias);
    if (bs == nullptr && M <= 1024) {
        switch (out_dtype_code) {
            case 0: return dispatch_fused_no_bias<float>(a, b, x, w, static_cast<float*>(D), M, N, K, stream);
            case 1: return dispatch_fused_no_bias<cutlass::half_t>(a, b, x, w, static_cast<cutlass::half_t*>(D), M, N, K, stream);
            case 2: return dispatch_fused_no_bias<cutlass::bfloat16_t>(a, b, x, w, static_cast<cutlass::bfloat16_t*>(D), M, N, K, stream);
            default: return false;
        }
    }
    switch (out_dtype_code) {
        case 0: return dispatch_fused<float>(a, b, x, w, bs, static_cast<float*>(D), M, N, K, stream);
        case 1: return dispatch_fused<cutlass::half_t>(a, b, x, w, bs, static_cast<cutlass::half_t*>(D), M, N, K, stream);
        case 2: return dispatch_fused<cutlass::bfloat16_t>(a, b, x, w, bs, static_cast<cutlass::bfloat16_t*>(D), M, N, K, stream);
        default: return false;
    }
}
}  // extern "C"

#else  // !COMFY_HAVE_CUTLASS -- stub; caller falls back to cuBLAS + separate dequant.

extern "C" bool launch_cutlass_int8_dequant(
    const void*, const void*, const void*, const void*, const void*,
    void*, int64_t, int64_t, int64_t, int, cudaStream_t) {
    return false;
}

#endif
