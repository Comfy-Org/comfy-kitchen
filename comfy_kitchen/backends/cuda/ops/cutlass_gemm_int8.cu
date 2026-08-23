/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * INT8 GEMM with a FUSED dequant epilogue via CUTLASS (EVT):
 *   D[m,n] = (sum_k A[m,k]*B[n,k]) * x_scale[m] * w_scale[n] + bias[n]   -> out dtype
 *
 * Replaces cuBLAS-GEMM(int32) + separate dequant with one near-peak kernel.
 * Multiple tile configs are instantiated and selected with a shape heuristic
 * fitted from sustained Ada and Blackwell benchmarks.
 * Falls back to cuBLAS when CUTLASS is unavailable or no config can run.
 */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <climits>
#include <cstdint>
#include <type_traits>

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

struct StreamWorkspace {
    void* data = nullptr;
    size_t size = 0;
};

void* get_stream_workspace(size_t size, cudaStream_t stream) {
    if (size == 0) return nullptr;

    int device;
    if (cudaGetDevice(&device) != cudaSuccess) return nullptr;

    static std::mutex mutex;
    static std::map<std::tuple<int, uintptr_t>, StreamWorkspace> workspaces;
    std::lock_guard<std::mutex> lock(mutex);
    auto& workspace = workspaces[{device, reinterpret_cast<uintptr_t>(stream)}];
    if (workspace.size >= size) return workspace.data;

    if (workspace.data != nullptr && cudaFree(workspace.data) != cudaSuccess) return nullptr;
    workspace = {};
    if (cudaMalloc(&workspace.data, size) != cudaSuccess) return nullptr;
    workspace.size = size;
    return workspace.data;
}

struct ThreadblockSwizzleLeanStreamK {
    using StreamkFeature = void;

    template <typename GemmKernel>
    struct KernelTraits {};

    enum ReductionStrategy {
        kNone,
        kAtomic,
        kMixed,
    };

    static constexpr ReductionStrategy kReductionStrategy = kAtomic;

    cutlass::gemm::GemmCoord problem_size;
    cutlass::gemm::GemmCoord tiled_shape_;
    int iterations_per_tile;
    int sk_blocks;
    int sk_iterations;
    int avail_sms;
    int dp_blocks;
    int dp_first_wave_tiles = 1;
    int reduction_blocks = 0;
    int sk_tiles;
    int sk_waves;
    bool cohort_raster = false;

    ThreadblockSwizzleLeanStreamK() = default;

    ThreadblockSwizzleLeanStreamK(
        cutlass::gemm::GemmUniversalMode,
        cutlass::gemm::GemmCoord problem_size_arg,
        cutlass::gemm::GemmCoord tile_size,
        int,
        int,
        int device_sms,
        int available_sms,
        size_t,
        size_t,
        size_t,
        int)
        : problem_size(problem_size_arg),
          tiled_shape_(
              (problem_size.m() + tile_size.m() - 1) / tile_size.m(),
              (problem_size.n() + tile_size.n() - 1) / tile_size.n(),
              1),
          iterations_per_tile(
              (problem_size.k() + tile_size.k() - 1) / tile_size.k()),
          avail_sms(available_sms < 0 || available_sms > device_sms
                        ? device_sms
                        : available_sms) {
        const int output_tiles = tiled_shape_.m() * tiled_shape_.n();
        const int partial_wave_tiles = output_tiles % avail_sms;
        if (partial_wave_tiles == 0) {
            sk_tiles = 0;
            sk_blocks = 0;
            sk_waves = 0;
            dp_blocks = output_tiles;
        } else {
            sk_tiles = output_tiles < avail_sms
                ? output_tiles
                : avail_sms + partial_wave_tiles;
            sk_iterations = sk_tiles * iterations_per_tile;
            sk_blocks = sk_iterations / 2 < avail_sms
                ? sk_iterations / 2
                : avail_sms;
            if (sk_blocks < 1) sk_blocks = 1;
            sk_waves = (sk_blocks + avail_sms - 1) / avail_sms;
            dp_blocks = output_tiles - sk_tiles;
        }
        sk_iterations = sk_tiles * iterations_per_tile;
    }

    CUTLASS_HOST_DEVICE
    cutlass::gemm::GemmCoord tiled_shape() const {
        return tiled_shape_;
    }

    CUTLASS_HOST_DEVICE
    int iters_per_tile() const {
        return iterations_per_tile;
    }

    CUTLASS_HOST_DEVICE
    int sk_regions() const {
        return sk_blocks == 0 ? 0 : 1;
    }

    CUTLASS_HOST_DEVICE
    int sk_blocks_per_region() const {
        return sk_blocks;
    }

    dim3 get_grid_dims() const {
        return dim3(sk_waves * avail_sms + dp_blocks, 1, 1);
    }

    CUTLASS_DEVICE
    int get_sk_tile_idx(int iteration) const {
        return iteration / iterations_per_tile;
    }

    CUTLASS_DEVICE
    cutlass::gemm::GemmCoord get_tile_offset(int tile_index) const {
        int tile_m;
        int tile_n;
        if (tiled_shape_.m() < tiled_shape_.n()) {
            tile_n = tile_index / tiled_shape_.m();
            tile_m = tile_index - tile_n * tiled_shape_.m();
        } else {
            tile_m = tile_index / tiled_shape_.n();
            tile_n = tile_index - tile_m * tiled_shape_.n();
        }
        return {tile_m, tile_n, 0};
    }

    CUTLASS_DEVICE
    int get_block_idx() const {
        return blockIdx.x;
    }

    CUTLASS_DEVICE
    int get_sk_block_idx(int iteration) const {
        const int small_iterations = sk_iterations / sk_blocks;
        const int big_blocks = sk_iterations - small_iterations * sk_blocks;
        const int big_iterations = small_iterations + 1;
        if (iteration < big_blocks * big_iterations) {
            return iteration / big_iterations;
        }
        return big_blocks +
            (iteration - big_blocks * big_iterations) / small_iterations;
    }

    CUTLASS_DEVICE
    void get_iter_extents(
        int block_index,
        int& iteration_begin,
        int& iteration_end) const {
        const int small_iterations = sk_iterations / sk_blocks;
        const int big_blocks = sk_iterations - small_iterations * sk_blocks;
        iteration_begin = block_index * small_iterations +
            (block_index < big_blocks ? block_index : big_blocks);
        iteration_end = iteration_begin + small_iterations +
            (block_index < big_blocks ? 1 : 0);
    }

    CUTLASS_DEVICE
    int get_first_block_idx(int tile_index, int block_index) const {
        return tile_index < sk_tiles
            ? get_sk_block_idx(tile_index * iterations_per_tile)
            : block_index;
    }
};

template <typename T, typename = void>
struct IsStreamKSwizzle : std::false_type {};

template <typename T>
struct IsStreamKSwizzle<T, std::void_t<typename T::StreamkFeature>> : std::true_type {};

template <typename ThreadMap, bool Scalar>
struct WeightScaleBroadcast;

template <typename ThreadMap>
struct WeightScaleBroadcast<ThreadMap, false> {
    using Type = cutlass::epilogue::threadblock::VisitorRowBroadcast<
        ThreadMap, float, cute::Stride<_0, _1, int32_t>>;

    static typename Type::Arguments arguments(const float* scale, int n) {
        return {scale, 0.f, {_0{}, _1{}, n}};
    }
};

template <typename ThreadMap>
struct WeightScaleBroadcast<ThreadMap, true> {
    using Type = cutlass::epilogue::threadblock::VisitorScalarBroadcast<float>;

    static typename Type::Arguments arguments(const float* scale, int) {
        typename Type::Arguments result{};
        result.scalar_ptrs[0] = scale;
        return result;
    }
};

// One fused int8 GEMM, parameterized on output type AND tile/warp/stage config.
template <typename ElementOutput, int TBM, int TBN, int TBK, int WM, int WN, int WK, int NumStages,
          typename ArchTag = cutlass::arch::Sm80, typename ElementBias = float,
          bool ScalarWeightScale = false, int AlignmentAB = 16,
          typename ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>>
struct FusedInt8Gemm {
    using ElementA = int8_t; using ElementB = int8_t;
    using ElementC = ElementOutput;
    using ElementAcc = int32_t; using ElementCompute = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;   // B[N,K] row == [K,N] col
    using LayoutC = cutlass::layout::RowMajor;
    static constexpr int AlignA = AlignmentAB, AlignB = AlignmentAB;
    static constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementC>::value;
    using TB   = cutlass::gemm::GemmShape<TBM, TBN, TBK>;
    using Warp = cutlass::gemm::GemmShape<WM, WN, WK>;
    using Inst = cutlass::gemm::GemmShape<16, 8, 32>;
    static constexpr int EVTStages = 1;

    using ThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<TB, Warp, ElementC, AlignC, EVTStages>;
    using Accum  = cutlass::epilogue::threadblock::VisitorAccFetch;
    using XScale = cutlass::epilogue::threadblock::VisitorColBroadcast<ThreadMap, ElementCompute, cute::Stride<_1, _0, int32_t>>;
    using WScale = typename WeightScaleBroadcast<ThreadMap, ScalarWeightScale>::Type;
    using Bias   = cutlass::epilogue::threadblock::VisitorRowBroadcast<ThreadMap, ElementBias, cute::Stride<_0, _1, int32_t>>;
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
        cutlass::arch::OpClassTensorOp, ArchTag,
        TB, Warp, Inst, EVTD,
        ThreadblockSwizzle,
        NumStages, cutlass::arch::OpMultiplyAddSaturate, EVTStages>::GemmKernel;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    static bool run(const int8_t* A, const int8_t* B, const float* xs, const float* ws,
                    const ElementBias* bias, ElementOutput* D, int M, int N, int K, cudaStream_t stream) {
        return run_strided(A, B, xs, ws, bias, D, M, N, K, N, stream);
    }

    static bool run_strided(const int8_t* A, const int8_t* B, const float* xs, const float* ws,
                            const ElementBias* bias, ElementOutput* D, int M, int N, int K,
                            int output_stride, cudaStream_t stream) {
        cutlass::gemm::GemmCoord problem(M, N, K);
        const auto weight_scale_args = WeightScaleBroadcast<ThreadMap, ScalarWeightScale>::arguments(ws, N);
        typename EVTD::Arguments cb{
            { {  { {}, {const_cast<float*>(xs), 0.f, {_1{}, _0{}, M}}, {} },
                 weight_scale_args, {} },
              {const_cast<ElementBias*>(bias), ElementBias(0), {_0{}, _1{}, N}}, {} },
            {D, {output_stride, _1{}, M * output_stride}} };
        const auto make_args = [&]() {
            if constexpr (IsStreamKSwizzle<ThreadblockSwizzle>::value) {
                return typename Gemm::Arguments(
                    cutlass::gemm::GemmUniversalMode::kGemm, problem, 1, cb,
                    const_cast<int8_t*>(A), const_cast<int8_t*>(B), nullptr, nullptr,
                    (int64_t)M * K, (int64_t)N * K, 0, 0, K, K, 0, 0, -1);
            } else {
                return typename Gemm::Arguments(
                    cutlass::gemm::GemmUniversalMode::kGemm, problem, 1, cb,
                    const_cast<int8_t*>(A), const_cast<int8_t*>(B), nullptr, nullptr,
                    (int64_t)M * K, (int64_t)N * K, 0, 0, K, K, 0, 0);
            }
        };
        typename Gemm::Arguments args = make_args();

        Gemm gemm;
        if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;
        const size_t workspace_size = Gemm::get_workspace_size(args);
        void* workspace = get_stream_workspace(workspace_size, stream);
        if (workspace_size != 0 && workspace == nullptr) return false;
        if (gemm.initialize(args, workspace, stream) != cutlass::Status::kSuccess) return false;
        return gemm(stream) == cutlass::Status::kSuccess;
    }
};

template <typename ElementOutput, int TBM, int TBN, int TBK, int WM, int WN, int WK, int NumStages,
          typename ArchTag = cutlass::arch::Sm80, bool ScalarWeightScale = false,
          int AlignmentAB = 16,
          typename ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>>
struct FusedInt8GemmNoBias {
    using ElementA = int8_t; using ElementB = int8_t;
    using ElementC = ElementOutput;
    using ElementAcc = int32_t; using ElementCompute = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    static constexpr int AlignA = AlignmentAB, AlignB = AlignmentAB;
    static constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementC>::value;
    using TB   = cutlass::gemm::GemmShape<TBM, TBN, TBK>;
    using Warp = cutlass::gemm::GemmShape<WM, WN, WK>;
    using Inst = cutlass::gemm::GemmShape<16, 8, 32>;
    static constexpr int EVTStages = 1;

    using ThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<TB, Warp, ElementC, AlignC, EVTStages>;
    using Accum  = cutlass::epilogue::threadblock::VisitorAccFetch;
    using XScale = cutlass::epilogue::threadblock::VisitorColBroadcast<ThreadMap, ElementCompute, cute::Stride<_1, _0, int32_t>>;
    using WScale = typename WeightScaleBroadcast<ThreadMap, ScalarWeightScale>::Type;
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
        cutlass::arch::OpClassTensorOp, ArchTag,
        TB, Warp, Inst, EVTD,
        ThreadblockSwizzle,
        NumStages, cutlass::arch::OpMultiplyAddSaturate, EVTStages>::GemmKernel;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    static bool run(const int8_t* A, const int8_t* B, const float* xs, const float* ws,
                    ElementOutput* D, int M, int N, int K, cudaStream_t stream) {
        return run_strided(A, B, xs, ws, D, M, N, K, N, stream);
    }

    static bool run_strided(const int8_t* A, const int8_t* B, const float* xs, const float* ws,
                            ElementOutput* D, int M, int N, int K, int output_stride,
                            cudaStream_t stream) {
        cutlass::gemm::GemmCoord problem(M, N, K);
        const auto weight_scale_args = WeightScaleBroadcast<ThreadMap, ScalarWeightScale>::arguments(ws, N);
        typename EVTD::Arguments cb{
            { { {}, {const_cast<float*>(xs), 0.f, {_1{}, _0{}, M}}, {} },
              weight_scale_args, {} },
            {D, {output_stride, _1{}, M * output_stride}} };
        const auto make_args = [&]() {
            if constexpr (IsStreamKSwizzle<ThreadblockSwizzle>::value) {
                return typename Gemm::Arguments(
                    cutlass::gemm::GemmUniversalMode::kGemm, problem, 1, cb,
                    const_cast<int8_t*>(A), const_cast<int8_t*>(B), nullptr, nullptr,
                    (int64_t)M * K, (int64_t)N * K, 0, 0, K, K, 0, 0, -1);
            } else {
                return typename Gemm::Arguments(
                    cutlass::gemm::GemmUniversalMode::kGemm, problem, 1, cb,
                    const_cast<int8_t*>(A), const_cast<int8_t*>(B), nullptr, nullptr,
                    (int64_t)M * K, (int64_t)N * K, 0, 0, K, K, 0, 0);
            }
        };
        typename Gemm::Arguments args = make_args();

        Gemm gemm;
        if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;
        const size_t workspace_size = Gemm::get_workspace_size(args);
        void* workspace = get_stream_workspace(workspace_size, stream);
        if (workspace_size != 0 && workspace == nullptr) return false;
        if (gemm.initialize(args, workspace, stream) != cutlass::Status::kSuccess) return false;
        return gemm(stream) == cutlass::Status::kSuccess;
    }
};

int select_fused_int8_config(int m, int n, int k) {
    if (k % 16 != 0) return 9;

    const int64_t mn = int64_t(m) * n;
    if (n <= 24832) {
        if (mn <= 1477632) {
            if (mn <= 259072) return k <= 7296 ? 6 : 12;
            return int64_t(n) * k <= 16252928 ? 2 : 12;
        }
        if (mn <= 4193792) {
            return int64_t(n) * k <= 5275648 ? 1 : 12;
        }
        return int64_t(m) * k <= int64_t(n) * 5675 ? 0 : 13;
    }
    if (int64_t(n) * k <= int64_t(m) * 11096) return 0;
    return mn <= 108003328 ? 0 : 13;
}

template <typename Launch>
bool launch_fused_int8_heuristic(int m, int n, int k, Launch launch) {
    const int selected = select_fused_int8_config(m, n, k);
    if (launch(selected)) return true;

    static constexpr int aligned_fallbacks[] = {2, 12, 0, 13, 1, 6, 8, 7, 3, 4, 5};
    static constexpr int low_alignment_fallbacks[] = {9, 10, 11};
    if (k % 16 == 0) {
        for (int config : aligned_fallbacks) {
            if (config != selected && launch(config)) return true;
        }
    } else {
        for (int config : low_alignment_fallbacks) {
            if (config != selected && launch(config)) return true;
        }
    }
    return false;
}

template <typename OutT>
bool dispatch_fused(const int8_t* A, const int8_t* B, const float* xs, const float* ws,
                    const float* bias, OutT* D, int M, int N, int K, cudaStream_t stream) {
    using Fn = bool (*)(const int8_t*, const int8_t*, const float*, const float*, const float*, OutT*, int, int, int, cudaStream_t);
    // Tile configs spanning big-GPU/large-M (wide) to small-GPU/small-M (more CTAs).
    static const Fn runners[] = {
        &FusedInt8Gemm<OutT, 128, 256, 64, 64, 64, 64, 3>::run,
        &FusedInt8Gemm<OutT, 128, 128, 64, 64, 64, 64, 4>::run,
        &FusedInt8Gemm<OutT,  64, 128, 64, 32, 64, 64, 4>::run,
        &FusedInt8Gemm<OutT,  64, 256, 64, 32, 64, 64, 3>::run,
        &FusedInt8Gemm<OutT,  32, 256, 64, 32, 64, 64, 4>::run,
        &FusedInt8Gemm<OutT,  32, 128, 64, 32, 64, 64, 4>::run,
        &FusedInt8Gemm<OutT,  16, 128, 64, 16, 64, 64, 4>::run,
        &FusedInt8Gemm<OutT,  64, 128, 128, 32, 64, 128, 3>::run,
        &FusedInt8Gemm<OutT, 128,  64, 128, 64, 32, 128, 3>::run,
        &FusedInt8Gemm<OutT,  64, 128, 64, 32, 64, 64, 4,
                       cutlass::arch::Sm80, float, false, 8>::run,
        &FusedInt8Gemm<OutT,  32, 128, 64, 32, 64, 64, 4,
                       cutlass::arch::Sm80, float, false, 8>::run,
        &FusedInt8Gemm<OutT,  16, 128, 64, 16, 64, 64, 4,
                       cutlass::arch::Sm80, float, false, 8>::run,
        &FusedInt8Gemm<OutT, 128, 128, 64, 64, 64, 64, 4,
                       cutlass::arch::Sm80, float, false, 16,
                       ThreadblockSwizzleLeanStreamK>::run,
        &FusedInt8Gemm<OutT, 128, 256, 64, 64, 64, 64, 3,
                       cutlass::arch::Sm80, float, false, 16,
                       ThreadblockSwizzleLeanStreamK>::run,
    };
    return launch_fused_int8_heuristic(M, N, K, [&](int config) {
        return runners[config](A, B, xs, ws, bias, D, M, N, K, stream);
    });
}

template <typename OutT>
bool dispatch_fused_no_bias(const int8_t* A, const int8_t* B, const float* xs, const float* ws,
                            OutT* D, int M, int N, int K, cudaStream_t stream) {
    using Fn = bool (*)(const int8_t*, const int8_t*, const float*, const float*,
                        OutT*, int, int, int, cudaStream_t);
    static const Fn runners[] = {
        &FusedInt8GemmNoBias<OutT, 128, 256, 64, 64, 64, 64, 3>::run,
        &FusedInt8GemmNoBias<OutT, 128, 128, 64, 64, 64, 64, 4>::run,
        &FusedInt8GemmNoBias<OutT,  64, 128, 64, 32, 64, 64, 4>::run,
        &FusedInt8GemmNoBias<OutT,  64, 256, 64, 32, 64, 64, 3>::run,
        &FusedInt8GemmNoBias<OutT,  32, 256, 64, 32, 64, 64, 4>::run,
        &FusedInt8GemmNoBias<OutT,  32, 128, 64, 32, 64, 64, 4>::run,
        &FusedInt8GemmNoBias<OutT,  16, 128, 64, 16, 64, 64, 4>::run,
        &FusedInt8GemmNoBias<OutT,  64, 128, 128, 32, 64, 128, 3>::run,
        &FusedInt8GemmNoBias<OutT, 128,  64, 128, 64, 32, 128, 3>::run,
        &FusedInt8GemmNoBias<OutT,  64, 128, 64, 32, 64, 64, 4,
                             cutlass::arch::Sm80, false, 8>::run,
        &FusedInt8GemmNoBias<OutT,  32, 128, 64, 32, 64, 64, 4,
                             cutlass::arch::Sm80, false, 8>::run,
        &FusedInt8GemmNoBias<OutT,  16, 128, 64, 16, 64, 64, 4,
                             cutlass::arch::Sm80, false, 8>::run,
        &FusedInt8GemmNoBias<OutT, 128, 128, 64, 64, 64, 64, 4,
                             cutlass::arch::Sm80, false, 16,
                             ThreadblockSwizzleLeanStreamK>::run,
        &FusedInt8GemmNoBias<OutT, 128, 256, 64, 64, 64, 64, 3,
                             cutlass::arch::Sm80, false, 16,
                             ThreadblockSwizzleLeanStreamK>::run,
    };
    return launch_fused_int8_heuristic(M, N, K, [&](int config) {
        return runners[config](A, B, xs, ws, D, M, N, K, stream);
    });
}

template <typename OutT>
bool dispatch_fused_no_bias_config(
    const int8_t* A, const int8_t* B, const float* xs, const float* ws,
    OutT* D, int M, int N, int K, int config, cudaStream_t stream) {
    using Fn = bool (*)(const int8_t*, const int8_t*, const float*, const float*,
                        OutT*, int, int, int, cudaStream_t);
    static const Fn runners[] = {
        &FusedInt8GemmNoBias<OutT, 128, 256, 64, 64, 64, 64, 3>::run,
        &FusedInt8GemmNoBias<OutT, 128, 128, 64, 64, 64, 64, 4>::run,
        &FusedInt8GemmNoBias<OutT,  64, 128, 64, 32, 64, 64, 4>::run,
        &FusedInt8GemmNoBias<OutT,  64, 256, 64, 32, 64, 64, 3>::run,
        &FusedInt8GemmNoBias<OutT,  32, 256, 64, 32, 64, 64, 4>::run,
        &FusedInt8GemmNoBias<OutT,  32, 128, 64, 32, 64, 64, 4>::run,
        &FusedInt8GemmNoBias<OutT,  16, 128, 64, 16, 64, 64, 4>::run,
        &FusedInt8GemmNoBias<OutT,  64, 128, 128, 32, 64, 128, 3>::run,
        &FusedInt8GemmNoBias<OutT, 128,  64, 128, 64, 32, 128, 3>::run,
        &FusedInt8GemmNoBias<OutT,  64, 128, 64, 32, 64, 64, 4,
                             cutlass::arch::Sm80, false, 8>::run,
        &FusedInt8GemmNoBias<OutT,  32, 128, 64, 32, 64, 64, 4,
                             cutlass::arch::Sm80, false, 8>::run,
        &FusedInt8GemmNoBias<OutT,  16, 128, 64, 16, 64, 64, 4,
                             cutlass::arch::Sm80, false, 8>::run,
        &FusedInt8GemmNoBias<OutT, 128, 128, 64, 64, 64, 64, 4,
                             cutlass::arch::Sm80, false, 16,
                             ThreadblockSwizzleLeanStreamK>::run,
        &FusedInt8GemmNoBias<OutT, 128, 256, 64, 64, 64, 64, 3,
                             cutlass::arch::Sm80, false, 16,
                             ThreadblockSwizzleLeanStreamK>::run,
    };
    constexpr int config_count = sizeof(runners) / sizeof(runners[0]);
    if (config < 0 || config >= config_count) return false;
    return runners[config](A, B, xs, ws, D, M, N, K, stream);
}

template <typename OutT>
bool dispatch_fused_strided(const int8_t* A, const int8_t* B, const float* xs, const float* ws,
                            const float* bias, OutT* D, int M, int N, int K, int output_stride,
                            cudaStream_t stream) {
    using Fn = bool (*)(const int8_t*, const int8_t*, const float*, const float*, const float*, OutT*, int, int, int, int, cudaStream_t);
    static const Fn runners[] = {
        &FusedInt8Gemm<OutT, 128, 256, 64, 64, 64, 64, 3>::run_strided,
        &FusedInt8Gemm<OutT, 128, 128, 64, 64, 64, 64, 4>::run_strided,
        &FusedInt8Gemm<OutT,  64, 128, 64, 32, 64, 64, 4>::run_strided,
        &FusedInt8Gemm<OutT,  64, 256, 64, 32, 64, 64, 3>::run_strided,
        &FusedInt8Gemm<OutT,  32, 256, 64, 32, 64, 64, 4>::run_strided,
        &FusedInt8Gemm<OutT,  32, 128, 64, 32, 64, 64, 4>::run_strided,
        &FusedInt8Gemm<OutT,  16, 128, 64, 16, 64, 64, 4>::run_strided,
        &FusedInt8Gemm<OutT,  64, 128, 128, 32, 64, 128, 3>::run_strided,
        &FusedInt8Gemm<OutT, 128,  64, 128, 64, 32, 128, 3>::run_strided,
        &FusedInt8Gemm<OutT,  64, 128, 64, 32, 64, 64, 4,
                       cutlass::arch::Sm80, float, false, 8>::run_strided,
        &FusedInt8Gemm<OutT,  32, 128, 64, 32, 64, 64, 4,
                       cutlass::arch::Sm80, float, false, 8>::run_strided,
        &FusedInt8Gemm<OutT,  16, 128, 64, 16, 64, 64, 4,
                       cutlass::arch::Sm80, float, false, 8>::run_strided,
        &FusedInt8Gemm<OutT, 128, 128, 64, 64, 64, 64, 4,
                       cutlass::arch::Sm80, float, false, 16,
                       ThreadblockSwizzleLeanStreamK>::run_strided,
        &FusedInt8Gemm<OutT, 128, 256, 64, 64, 64, 64, 3,
                       cutlass::arch::Sm80, float, false, 16,
                       ThreadblockSwizzleLeanStreamK>::run_strided,
    };
    return launch_fused_int8_heuristic(M, N, K, [&](int config) {
        return runners[config](
            A, B, xs, ws, bias, D, M, N, K, output_stride, stream);
    });
}

template <typename OutT>
bool dispatch_fused_no_bias_strided(
    const int8_t* A, const int8_t* B, const float* xs, const float* ws,
    OutT* D, int M, int N, int K, int output_stride, cudaStream_t stream) {
    using Fn = bool (*)(const int8_t*, const int8_t*, const float*, const float*,
                        OutT*, int, int, int, int, cudaStream_t);
    static const Fn runners[] = {
        &FusedInt8GemmNoBias<OutT, 128, 256, 64, 64, 64, 64, 3>::run_strided,
        &FusedInt8GemmNoBias<OutT, 128, 128, 64, 64, 64, 64, 4>::run_strided,
        &FusedInt8GemmNoBias<OutT,  64, 128, 64, 32, 64, 64, 4>::run_strided,
        &FusedInt8GemmNoBias<OutT,  64, 256, 64, 32, 64, 64, 3>::run_strided,
        &FusedInt8GemmNoBias<OutT,  32, 256, 64, 32, 64, 64, 4>::run_strided,
        &FusedInt8GemmNoBias<OutT,  32, 128, 64, 32, 64, 64, 4>::run_strided,
        &FusedInt8GemmNoBias<OutT,  16, 128, 64, 16, 64, 64, 4>::run_strided,
        &FusedInt8GemmNoBias<OutT,  64, 128, 128, 32, 64, 128, 3>::run_strided,
        &FusedInt8GemmNoBias<OutT, 128,  64, 128, 64, 32, 128, 3>::run_strided,
        &FusedInt8GemmNoBias<OutT,  64, 128, 64, 32, 64, 64, 4,
                             cutlass::arch::Sm80, false, 8>::run_strided,
        &FusedInt8GemmNoBias<OutT,  32, 128, 64, 32, 64, 64, 4,
                             cutlass::arch::Sm80, false, 8>::run_strided,
        &FusedInt8GemmNoBias<OutT,  16, 128, 64, 16, 64, 64, 4,
                             cutlass::arch::Sm80, false, 8>::run_strided,
        &FusedInt8GemmNoBias<OutT, 128, 128, 64, 64, 64, 64, 4,
                             cutlass::arch::Sm80, false, 16,
                             ThreadblockSwizzleLeanStreamK>::run_strided,
        &FusedInt8GemmNoBias<OutT, 128, 256, 64, 64, 64, 64, 3,
                             cutlass::arch::Sm80, false, 16,
                             ThreadblockSwizzleLeanStreamK>::run_strided,
    };
    return launch_fused_int8_heuristic(M, N, K, [&](int config) {
        return runners[config](A, B, xs, ws, D, M, N, K, output_stride, stream);
    });
}

}  // namespace

extern "C" {
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
    if (bs == nullptr) {
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

bool launch_cutlass_int8_dequant_strided(
    const void* A, const void* B, const void* xs, const void* ws, const void* bias,
    void* D, int64_t M, int64_t N, int64_t K, int64_t output_stride, int out_dtype_code,
    cudaStream_t stream)
{
    if (M == 0 || N == 0 || K == 0) return true;
    if (output_stride < N) return false;
    const int8_t* a = static_cast<const int8_t*>(A);
    const int8_t* b = static_cast<const int8_t*>(B);
    const float* x = static_cast<const float*>(xs);
    const float* w = static_cast<const float*>(ws);
    const float* bs = static_cast<const float*>(bias);
    if (bs == nullptr) {
        switch (out_dtype_code) {
            case 0: return dispatch_fused_no_bias_strided<float>(a, b, x, w, static_cast<float*>(D), M, N, K, output_stride, stream);
            case 1: return dispatch_fused_no_bias_strided<cutlass::half_t>(a, b, x, w, static_cast<cutlass::half_t*>(D), M, N, K, output_stride, stream);
            case 2: return dispatch_fused_no_bias_strided<cutlass::bfloat16_t>(a, b, x, w, static_cast<cutlass::bfloat16_t*>(D), M, N, K, output_stride, stream);
            default: return false;
        }
    }
    switch (out_dtype_code) {
        case 0: return dispatch_fused_strided<float>(a, b, x, w, bs, static_cast<float*>(D), M, N, K, output_stride, stream);
        case 1: return dispatch_fused_strided<cutlass::half_t>(a, b, x, w, bs, static_cast<cutlass::half_t*>(D), M, N, K, output_stride, stream);
        case 2: return dispatch_fused_strided<cutlass::bfloat16_t>(a, b, x, w, bs, static_cast<cutlass::bfloat16_t*>(D), M, N, K, output_stride, stream);
        default: return false;
    }
}

bool launch_cutlass_int8_dequant_config(
    const void* A, const void* B, const void* xs, const void* ws, void* D,
    int64_t M, int64_t N, int64_t K, int out_dtype_code, int config,
    cudaStream_t stream)
{
    if (M == 0 || N == 0 || K == 0) return true;
    const int8_t* a = static_cast<const int8_t*>(A);
    const int8_t* b = static_cast<const int8_t*>(B);
    const float* x = static_cast<const float*>(xs);
    const float* w = static_cast<const float*>(ws);
    switch (out_dtype_code) {
        case 2: return dispatch_fused_no_bias_config<cutlass::bfloat16_t>(
            a, b, x, w, static_cast<cutlass::bfloat16_t*>(D), M, N, K, config, stream);
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

extern "C" bool launch_cutlass_int8_dequant_strided(
    const void*, const void*, const void*, const void*, const void*,
    void*, int64_t, int64_t, int64_t, int64_t, int, cudaStream_t) {
    return false;
}

extern "C" bool launch_cutlass_int8_dequant_config(
    const void*, const void*, const void*, const void*, void*,
    int64_t, int64_t, int64_t, int, int, cudaStream_t) {
    return false;
}

#endif
