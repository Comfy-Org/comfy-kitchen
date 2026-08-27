// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// INT8 GEMM on WMMA (v_wmma_i32_16x16x16_iu8), gfx11 and gfx12. See mma.h.
// C[M, N] = (A[M, K] @ B[N, K]^T) * scale_a[row] * scale_b[col] + bias
#include <atomic>
#include <cstring>
#include <stdexcept>
#include <string>

#include "../epilogue.h"
#include "../gemm_wmma.h"
#include "../launchers.h"

namespace comfy::hip_backend {

inline bool use_nonduplicated_int8_schedule() {
    // The launch contract stays dimensional. Only the hardware choice is
    // architecture-specific, and it remains private to this translation unit.
    constexpr int kMaxDevices = 16;
    static std::atomic<int> cache[kMaxDevices] = {};
    int device = 0;
    if (hipGetDevice(&device) != hipSuccess) return false;

    auto select = [device] {
        hipDeviceProp_t properties{};
        return hipGetDeviceProperties(&properties, device) == hipSuccess &&
            std::strncmp(properties.gcnArchName, "gfx12", 5) == 0;
    };
    if (device < 0 || device >= kMaxDevices) return select();

    int selected = cache[device].load(std::memory_order_relaxed);
    if (selected == 0) {
        selected = select() ? 2 : 1;
        cache[device].store(selected, std::memory_order_relaxed);
    }
    return selected == 2;
}

// Small-M path: reduce K across the wave, one wave per output column.
template <typename OutT>
__global__ __launch_bounds__(256) void gemv_int8_kernel(
    const int8_t* __restrict__ A, const int8_t* __restrict__ B, OutT* __restrict__ C,
    int M, int N, int K, int ldc, EpiRowwise epi) {

    const int col = blockIdx.x * blockDim.y + threadIdx.y;
    const int row = blockIdx.y;
    if (col >= N || row >= M) return;

    const int8_t* a = A + static_cast<int64_t>(row) * K;
    const int8_t* b = B + static_cast<int64_t>(col) * K;

    int sum = 0;
    for (int k = threadIdx.x * 16; k < K; k += kWave * 16) {
        const uint4 av = *reinterpret_cast<const uint4*>(a + k);
        const uint4 bv = *reinterpret_cast<const uint4*>(b + k);
        const int8_t* ab = reinterpret_cast<const int8_t*>(&av);
        const int8_t* bb = reinterpret_cast<const int8_t*>(&bv);
        #pragma unroll
        for (int i = 0; i < 16; ++i) sum += ab[i] * bb[i];
    }
    sum = wave_reduce_sum(sum);

    if (threadIdx.x == 0) {
        epi.init();
        C[static_cast<int64_t>(row) * ldc + col] =
            static_cast<OutT>(epi(row, col, static_cast<float>(sum)));
    }
}

template <typename OutT>
static void launch_int8(const int8_t* A, const int8_t* B, OutT* C, int M, int N, int K, int ldc,
                        EpiRowwise epi, hipStream_t stream) {
    const uint8_t* Au = reinterpret_cast<const uint8_t*>(A);
    const uint8_t* Bu = reinterpret_cast<const uint8_t*>(B);

    if (M <= 8) {
        dim3 block(kWave, 8);
        dim3 grid((N + 7) / 8, M);
        gemv_int8_kernel<OutT><<<grid, block, 0, stream>>>(A, B, C, M, N, K, ldc, epi);
        return;
    }
    const bool tuned = use_nonduplicated_int8_schedule();
    if (M >= 96 && N >= 96) {
        if (tuned && M >= 512 && K == 3840 &&
            (N == 3840 || N == 10240 || N == 11520)) {
            // Square, packed-QKV, and MLP projections all reuse one staged B
            // tile across two adjacent M tiles.
            constexpr int BM = 128, BN = 128, BKB = 64;
            dim3 grid((N + BN - 1) / BN, (M + 2 * BM - 1) / (2 * BM));
            if (epi.bias == nullptr && epi.scale_b_stride == 1) {
                EpiRowwiseNoBias no_bias{epi.scale_a, epi.scale_b};
                gemm_wmma_dual_m_kernel<MmaInt8, EpiRowwiseNoBias, OutT,
                                        BM, BN, BKB, 4, 2, 2, 4>
                    <<<grid, 512, 0, stream>>>(
                        Au, Bu, C, M, N, K, ldc, no_bias);
            } else {
                gemm_wmma_dual_m_kernel<MmaInt8, EpiRowwise, OutT,
                                        BM, BN, BKB, 4, 2, 2, 4>
                    <<<grid, 512, 0, stream>>>(Au, Bu, C, M, N, K, ldc, epi);
            }
        } else if (tuned && M >= 512 && N >= 128 && K > N) {
            constexpr int BM = 128, BN = 128, BKB = 64;
            dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
            gemm_wmma_kernel<MmaInt8, EpiRowwise, OutT, BM, BN, BKB,
                             4, 2, 2, 4, true>
                <<<grid, 256, 0, stream>>>(Au, Bu, C, M, N, K, ldc, epi);
        } else if (tuned) {
            // Non-duplicated WMMA operands favor the lower-LDS stage.
            constexpr int BM = 128, BN = 128, BKB = 64;
            dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
            gemm_wmma_kernel<MmaInt8, EpiRowwise, OutT, BM, BN, BKB,
                             4, 2, 2, 4>
                <<<grid, 256, 0, stream>>>(Au, Bu, C, M, N, K, ldc, epi);
        } else {
            // Portable starting geometry for other WMMA targets.
            constexpr int BM = 128, BN = 128, BKB = 128;
            dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
            gemm_wmma_kernel<MmaInt8, EpiRowwise, OutT, BM, BN, BKB, 4, 2, 2, 4>
                <<<grid, 256, 0, stream>>>(Au, Bu, C, M, N, K, ldc, epi);
        }
    } else {
        constexpr int BM = 64, BN = 64, BKB = 64;
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        gemm_wmma_kernel<MmaInt8, EpiRowwise, OutT, BM, BN, BKB, 2, 2, 2, 2>
            <<<grid, 128, 0, stream>>>(Au, Bu, C, M, N, K, ldc, epi);
    }
}

}  // namespace comfy::hip_backend

// scale_b_stride is 1 for a per-output-channel weight scale, 0 for a scalar one.
extern "C" void launch_int8_gemm_kernel(
    const void* a, const void* b, void* c,
    const void* scale_a, const void* scale_b, int scale_b_stride,
    const void* bias, int bias_code,
    int M, int N, int K, int ldc, int out_code, hipStream_t stream) {

    using namespace comfy::hip_backend;

    if (M == 0 || N == 0) {
        return;  // a zero-dimension grid is hipErrorInvalidConfiguration
    }
    if (ldc < N) {
        throw std::runtime_error("int8_gemm: ldc=" + std::to_string(ldc) +
                                 " is narrower than N=" + std::to_string(N));
    }
    // Both the tile loader and the small-M GEMV read a row 16 bytes at a time; an
    // unaligned K would take the last chunk past the end of the row. An empty GEMM
    // does no reads, so this only applies once there is real work above.
    if (K % 16 != 0) {
        throw std::runtime_error("int8_gemm: K=" + std::to_string(K) +
                                 " must be a multiple of 16");
    }

    EpiRowwise epi{static_cast<const float*>(scale_a), static_cast<const float*>(scale_b),
                   scale_b_stride, bias, bias_code};

    const int8_t* A = static_cast<const int8_t*>(a);
    const int8_t* B = static_cast<const int8_t*>(b);

    if (out_code == kBF16) {
        launch_int8<__bf16>(A, B, static_cast<__bf16*>(c), M, N, K, ldc, epi, stream);
    } else if (out_code == kF16) {
        launch_int8<__half>(A, B, static_cast<__half*>(c), M, N, K, ldc, epi, stream);
    } else if (out_code == kF32) {
        launch_int8<float>(A, B, static_cast<float*>(c), M, N, K, ldc, epi, stream);
    } else {
        throw std::runtime_error("int8_gemm: unsupported output dtype");
    }
}

template <typename OutT, int BKB, bool DUAL_M, typename Epi>
static void launch_int8_b_tiled(
    const uint8_t* A, const uint8_t* B, OutT* C,
    int M, int N, int K, int ldc, Epi epi,
    hipStream_t stream) {
    using namespace comfy::hip_backend;
    constexpr int BM = 128, BN = 128;
    if constexpr (DUAL_M) {
        dim3 grid((N + BN - 1) / BN, (M + 2 * BM - 1) / (2 * BM));
        gemm_wmma_dual_m_kernel<MmaInt8, Epi, OutT,
                                BM, BN, BKB, 4, 2, 2, 4,
                                false, true>
            <<<grid, 512, 0, stream>>>(A, B, C, M, N, K, ldc, epi);
    } else {
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        gemm_wmma_kernel<MmaInt8, Epi, OutT,
                         BM, BN, BKB, 4, 2, 2, 4,
                         true, false, true, true>
            <<<grid, 256, 0, stream>>>(A, B, C, M, N, K, ldc, epi);
    }
}

extern "C" void launch_int8_gemm_b_tiled_kernel(
    const void* a, const void* b, void* c,
    const void* scale_a, const void* scale_b, int scale_b_stride,
    const void* bias, int bias_code, int M, int N, int K, int ldc,
    int out_code, int tile_k, bool dual_m, hipStream_t stream) {
    using namespace comfy::hip_backend;
    if (M == 0 || N == 0) return;
    if (M < 96 || N % 128 != 0 ||
        (tile_k != 64 && tile_k != 128) || K % tile_k != 0 || ldc < N) {
        throw std::runtime_error(
            "int8_gemm_b_tiled: requires M>=96, N divisible by 128, "
            "K divisible by tile_k, tile_k in {64,128}, and ldc>=N");
    }
    EpiRowwise epi{
        static_cast<const float*>(scale_a), static_cast<const float*>(scale_b),
        scale_b_stride, bias, bias_code};
    const uint8_t* A = static_cast<const uint8_t*>(a);
    const uint8_t* B = static_cast<const uint8_t*>(b);

#define LAUNCH_TILED(OUT_T, EPI_ARG)                                            \
    do {                                                                        \
        if (tile_k == 64) {                                                     \
            if (dual_m) launch_int8_b_tiled<OUT_T, 64, true>(                   \
                A, B, static_cast<OUT_T*>(c), M, N, K, ldc, EPI_ARG, stream);   \
            else launch_int8_b_tiled<OUT_T, 64, false>(                         \
                A, B, static_cast<OUT_T*>(c), M, N, K, ldc, EPI_ARG, stream);   \
        } else {                                                                \
            if (dual_m) launch_int8_b_tiled<OUT_T, 128, true>(                  \
                A, B, static_cast<OUT_T*>(c), M, N, K, ldc, EPI_ARG, stream);   \
            else launch_int8_b_tiled<OUT_T, 128, false>(                        \
                A, B, static_cast<OUT_T*>(c), M, N, K, ldc, EPI_ARG, stream);   \
        }                                                                       \
    } while (false)

    if (epi.bias == nullptr && epi.scale_b_stride == 1) {
        EpiRowwiseNoBias no_bias{epi.scale_a, epi.scale_b};
        if (out_code == kBF16) {
            LAUNCH_TILED(__bf16, no_bias);
        } else if (out_code == kF16) {
            LAUNCH_TILED(__half, no_bias);
        } else if (out_code == kF32) {
            LAUNCH_TILED(float, no_bias);
        } else {
            throw std::runtime_error("int8_gemm_b_tiled: unsupported output dtype");
        }
    } else if (out_code == kBF16) {
        LAUNCH_TILED(__bf16, epi);
    } else if (out_code == kF16) {
        LAUNCH_TILED(__half, epi);
    } else if (out_code == kF32) {
        LAUNCH_TILED(float, epi);
    } else {
        throw std::runtime_error("int8_gemm_b_tiled: unsupported output dtype");
    }
#undef LAUNCH_TILED
}

extern "C" void launch_int8_gemm_b_tiled_gated_residual_kernel(
    const void* a, const void* b, void* c,
    const void* scale_a, const void* scale_b, int scale_b_stride,
    const void* bias, int bias_code, const void* residual, const void* gate,
    int M, int N, int K, int ldc, int tile_k, bool dual_m,
    hipStream_t stream) {
    using namespace comfy::hip_backend;
    if (M == 0 || N == 0) return;
    if (M < 96 || N % 128 != 0 ||
        (tile_k != 64 && tile_k != 128) || K % tile_k != 0 || ldc < N) {
        throw std::runtime_error(
            "int8_gemm_b_tiled_gated_residual: requires M>=96, N divisible "
            "by 128, K divisible by tile_k, tile_k in {64,128}, and ldc>=N");
    }
    if (residual == nullptr || gate == nullptr) {
        throw std::runtime_error(
            "int8_gemm_b_tiled_gated_residual: residual and gate are required");
    }
    EpiRowwiseGatedResidual epi{
        static_cast<const float*>(scale_a), static_cast<const float*>(scale_b),
        scale_b_stride, bias, bias_code,
        static_cast<const __bf16*>(residual), static_cast<const __bf16*>(gate),
        ldc};
    const uint8_t* A = static_cast<const uint8_t*>(a);
    const uint8_t* B = static_cast<const uint8_t*>(b);

#define LAUNCH_TILED_GATED(TILE_K, DUAL_M)                                     \
    launch_int8_b_tiled<__bf16, TILE_K, DUAL_M>(                              \
        A, B, static_cast<__bf16*>(c), M, N, K, ldc, epi, stream)

    if (tile_k == 64) {
        if (dual_m) LAUNCH_TILED_GATED(64, true);
        else LAUNCH_TILED_GATED(64, false);
    } else {
        if (dual_m) LAUNCH_TILED_GATED(128, true);
        else LAUNCH_TILED_GATED(128, false);
    }
#undef LAUNCH_TILED_GATED
}

template <typename OutT>
static void launch_int8_pair(
    const uint8_t* A, const uint8_t* B0, const uint8_t* B1,
    OutT* C0, OutT* C1, int M, int N, int K, int ldc,
    comfy::hip_backend::EpiRowwise epi0,
    comfy::hip_backend::EpiRowwise epi1, hipStream_t stream) {
    using namespace comfy::hip_backend;
    constexpr int BM = 128, BN = 128;
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    if (use_nonduplicated_int8_schedule() &&
        epi0.bias == nullptr && epi1.bias == nullptr &&
        epi0.scale_b_stride == 1 && epi1.scale_b_stride == 1) {
        constexpr int BKB = 128;
        EpiRowwiseNoBias no_bias0{epi0.scale_a, epi0.scale_b};
        EpiRowwiseNoBias no_bias1{epi1.scale_a, epi1.scale_b};
        gemm_wmma_pair_kernel<MmaInt8, EpiRowwiseNoBias, OutT,
                              BM, BN, BKB, 4, 2, 2, 4>
            <<<grid, 512, 0, stream>>>(
                A, B0, B1, C0, C1, M, N, K, ldc, no_bias0, no_bias1);
    } else {
        constexpr int BKB = 128;
        gemm_wmma_pair_kernel<MmaInt8, EpiRowwise, OutT,
                              BM, BN, BKB, 4, 2, 2, 4>
            <<<grid, 512, 0, stream>>>(
                A, B0, B1, C0, C1, M, N, K, ldc, epi0, epi1);
    }
}

extern "C" void launch_int8_gemm_pair_kernel(
    const void* a, const void* b0, const void* b1, void* c0, void* c1,
    const void* scale_a, const void* scale_b0, int scale_b_stride0,
    const void* scale_b1, int scale_b_stride1, const void* bias0,
    int bias_code0, const void* bias1, int bias_code1, int M, int N, int K,
    int ldc, int out_code, hipStream_t stream) {

    using namespace comfy::hip_backend;
    if (M == 0) return;
    if (M < 96 || N < 96 || K <= 0 || K % 64 != 0 || ldc < N) {
        throw std::runtime_error(
            "int8_gemm_pair: requires M,N>=96, positive K divisible by 64, and ldc>=N");
    }

    EpiRowwise epi0{
        static_cast<const float*>(scale_a),
        static_cast<const float*>(scale_b0), scale_b_stride0,
        bias0, bias_code0};
    EpiRowwise epi1{
        static_cast<const float*>(scale_a),
        static_cast<const float*>(scale_b1), scale_b_stride1,
        bias1, bias_code1};
    const uint8_t* A = static_cast<const uint8_t*>(a);
    const uint8_t* B0 = static_cast<const uint8_t*>(b0);
    const uint8_t* B1 = static_cast<const uint8_t*>(b1);

    if (out_code == kBF16) {
        launch_int8_pair(
            A, B0, B1, static_cast<__bf16*>(c0), static_cast<__bf16*>(c1),
            M, N, K, ldc, epi0, epi1, stream);
    } else if (out_code == kF16) {
        launch_int8_pair(
            A, B0, B1, static_cast<__half*>(c0), static_cast<__half*>(c1),
            M, N, K, ldc, epi0, epi1, stream);
    } else if (out_code == kF32) {
        launch_int8_pair(
            A, B0, B1, static_cast<float*>(c0), static_cast<float*>(c1),
            M, N, K, ldc, epi0, epi1, stream);
    } else {
        throw std::runtime_error("int8_gemm_pair: unsupported output dtype");
    }
}

template <typename OutT, typename Epi>
static void launch_int8_tiled_a_down(
    const uint8_t* A, const uint8_t* B, OutT* C,
    int M, int N, int K, int ldc, Epi epi,
    bool tiled_b, hipStream_t stream) {
    using namespace comfy::hip_backend;
    constexpr int BM = 128, BN = 128, BKB = 128;
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    if (tiled_b) {
        gemm_wmma_kernel<MmaInt8, Epi, OutT, BM, BN, BKB,
                         4, 2, 2, 4, false, true, false, true>
            <<<grid, 256, 0, stream>>>(A, B, C, M, N, K, ldc, epi);
    } else {
        gemm_wmma_kernel<MmaInt8, Epi, OutT, BM, BN, BKB,
                         4, 2, 2, 4, true, true>
            <<<grid, 256, 0, stream>>>(A, B, C, M, N, K, ldc, epi);
    }
}

extern "C" void launch_int8_gemm_a_tiled128_down_kernel(
    const void* a, const void* b, void* c,
    const void* scale_a, const void* scale_b, int scale_b_stride,
    const void* bias, int bias_code,
    int M, int N, int K, int ldc, int out_code, bool tiled_b,
    hipStream_t stream) {

    using namespace comfy::hip_backend;
    if (M == 0) return;
    if (N <= 0 || K <= 0 || N % 128 != 0 || K % 128 != 0 || ldc < N) {
        throw std::runtime_error(
            "tiled-A int8 GEMM: requires positive N/K divisible by "
            "128 and ldc>=N");
    }
    EpiRowwise epi{
        static_cast<const float*>(scale_a), static_cast<const float*>(scale_b),
        scale_b_stride, bias, bias_code};
    const uint8_t* A = static_cast<const uint8_t*>(a);
    const uint8_t* B = static_cast<const uint8_t*>(b);
    if (out_code == kBF16) {
        if (epi.bias == nullptr && epi.scale_b_stride == 1) {
            EpiRowwiseNoBias no_bias{epi.scale_a, epi.scale_b};
            launch_int8_tiled_a_down(
                A, B, static_cast<__bf16*>(c), M, N, K, ldc, no_bias, tiled_b, stream);
        } else {
            launch_int8_tiled_a_down(
                A, B, static_cast<__bf16*>(c), M, N, K, ldc, epi, tiled_b, stream);
        }
    } else if (out_code == kF16) {
        launch_int8_tiled_a_down(
            A, B, static_cast<__half*>(c), M, N, K, ldc, epi, tiled_b, stream);
    } else if (out_code == kF32) {
        launch_int8_tiled_a_down(
            A, B, static_cast<float*>(c), M, N, K, ldc, epi, tiled_b, stream);
    } else {
        throw std::runtime_error("tiled-A int8 GEMM: unsupported output dtype");
    }
}
