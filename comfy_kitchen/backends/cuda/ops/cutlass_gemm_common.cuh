/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
  * Shared by the CUTLASS GEMM units (int8, fp16): per-stream workspace cache,
  * lean stream-K swizzle, GemmUniversal launch tail.
 */
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <map>
#include <mutex>
#include <tuple>
#include <type_traits>

#include "cutlass/cutlass.h"
#include "cutlass/gemm_coord.h"
#include "cutlass/gemm/gemm_enumerated_types.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

namespace comfy_cutlass {

struct StreamWorkspace {
    void* data = nullptr;
    size_t size = 0;
};

// One buffer per (device, stream), grown on demand and kept for the process
// lifetime; assumes a stream is driven by one host thread at a time.
inline void* get_stream_workspace(size_t size, cudaStream_t stream) {
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

// A tile configuration; each kernel file instantiates every epilogue variant
// from one list of these, so heuristic indices mean the same tile in every table.
template <int TBM_, int TBN_, int TBK_, int WM_, int WN_, int WK_, int NumStages_,
          int AlignmentAB_ = 16,
          typename ThreadblockSwizzle_ = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>>
struct TileConfig {
    static constexpr int TBM = TBM_, TBN = TBN_, TBK = TBK_;
    static constexpr int WM = WM_, WN = WN_, WK = WK_;
    static constexpr int NumStages = NumStages_;
    static constexpr int AlignmentAB = AlignmentAB_;
    using ThreadblockSwizzle = ThreadblockSwizzle_;
};

template <typename... Configs>
struct ConfigList {
    static constexpr int size = sizeof...(Configs);
};

// GemmUniversal launch tail: A [M,K] row-major x B [N,K] (column-major view),
// can_implement / workspace / initialize / run. Stream-K takes one extra argument.
template <typename Gemm, typename Callback, typename ElementA, typename ElementB>
bool launch_universal(const ElementA* A, const ElementB* B, const Callback& cb,
                      int M, int N, int K, cudaStream_t stream) {
    using Swizzle = typename Gemm::ThreadblockSwizzle;
    cutlass::gemm::GemmCoord problem(M, N, K);
    const auto make_args = [&]() {
        if constexpr (IsStreamKSwizzle<Swizzle>::value) {
            return typename Gemm::Arguments(
                cutlass::gemm::GemmUniversalMode::kGemm, problem, 1, cb,
                const_cast<ElementA*>(A), const_cast<ElementB*>(B), nullptr, nullptr,
                (int64_t)M * K, (int64_t)N * K, 0, 0, K, K, 0, 0, -1);
        } else {
            return typename Gemm::Arguments(
                cutlass::gemm::GemmUniversalMode::kGemm, problem, 1, cb,
                const_cast<ElementA*>(A), const_cast<ElementB*>(B), nullptr, nullptr,
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

}  // namespace comfy_cutlass
