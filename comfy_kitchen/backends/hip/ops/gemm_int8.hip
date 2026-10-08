// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// INT8 GEMM on WMMA (v_wmma_i32_16x16x16_iu8), gfx11 and gfx12. See mma.h.
// C[M, N] = (A[M, K] @ B[N, K]^T) * scale_a[row] * scale_b[col] + bias
#include <stdexcept>
#include <string>

#include "../device_arch.h"
#include "../epilogue.h"
#include "../gemm_wmma.h"
#include "../launchers.h"

namespace comfy::hip_backend {

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

template <typename OutT, int BKB>
static void launch_dual_m_tiles(const uint8_t* A, const uint8_t* B, OutT* C, int M, int N,
                                int K, int ldc, EpiRowwise epi, hipStream_t stream) {
    constexpr int BM = 128, BN = 128;
    dim3 grid((N + BN - 1) / BN, (M + 2 * BM - 1) / (2 * BM));
    if (epi.bias == nullptr && epi.scale_b_stride == 1) {
        EpiRowwiseNoBias no_bias{epi.scale_a, epi.scale_b};
        gemm_wmma_dual_m_kernel<MmaInt8, EpiRowwiseNoBias, OutT, BM, BN, BKB, 4, 2, 2, 4>
            <<<grid, 512, 0, stream>>>(A, B, C, M, N, K, ldc, no_bias);
    } else {
        gemm_wmma_dual_m_kernel<MmaInt8, EpiRowwise, OutT, BM, BN, BKB, 4, 2, 2, 4>
            <<<grid, 512, 0, stream>>>(A, B, C, M, N, K, ldc, epi);
    }
}

// Two adjacent M tiles share one staged B tile, halving B traffic per MMA. This
// wins once the grid covers the device and two blocks fit per WGP, so one
// block's loads overlap the other's MMAs; use the deepest K stage that fits
// twice. An odd tile count idles half of the last pair, so require an even
// count or at least seven tiles. Returns false to fall back to launch_gemm_wmma.
template <typename OutT>
static bool launch_int8_dual_m(const uint8_t* A, const uint8_t* B, OutT* C, int M, int N, int K,
                               int ldc, EpiRowwise epi, hipStream_t stream) {
    const int m_tiles = (M + 127) / 128;
    const int n_tiles = (N + 127) / 128;
    if (N <= 64 || m_tiles < 2 || (m_tiles % 2 != 0 && m_tiles < 7) ||
        m_tiles * n_tiles < device_wgp_count()) {
        return false;
    }
    constexpr auto kDualK128 =
        gemm_wmma_dual_m_kernel<MmaInt8, EpiRowwise, OutT, 128, 128, 128, 4, 2, 2, 4>;
    constexpr auto kDualK64 =
        gemm_wmma_dual_m_kernel<MmaInt8, EpiRowwise, OutT, 128, 128, 64, 4, 2, 2, 4>;
    if (resident_blocks<kDualK128, 512>() >= 2) {
        launch_dual_m_tiles<OutT, 128>(A, B, C, M, N, K, ldc, epi, stream);
    } else if (resident_blocks<kDualK64, 512>() >= 2) {
        launch_dual_m_tiles<OutT, 64>(A, B, C, M, N, K, ldc, epi, stream);
    } else {
        return false;
    }
    return true;
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
    if (!use_nonduplicated_wmma_schedule() ||
        !launch_int8_dual_m(Au, Bu, C, M, N, K, ldc, epi, stream)) {
        launch_gemm_wmma<MmaInt8, EpiRowwise, OutT>(Au, Bu, C, M, N, K, ldc, epi, stream);
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

template <typename OutT>
static void launch_int8_pair(
    const uint8_t* A, const uint8_t* B0, const uint8_t* B1,
    OutT* C0, OutT* C1, int M, int N, int K, int ldc,
    comfy::hip_backend::EpiRowwise epi0,
    comfy::hip_backend::EpiRowwise epi1, hipStream_t stream) {
    using namespace comfy::hip_backend;
    constexpr int BM = 128, BN = 128;
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    if (use_nonduplicated_wmma_schedule() &&
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
    int M, int N, int K, int ldc, Epi epi, hipStream_t stream) {
    using namespace comfy::hip_backend;
    constexpr int BM = 128, BN = 128, BKB = 128;
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_wmma_kernel<MmaInt8, Epi, OutT, BM, BN, BKB,
                     4, 2, 2, 4, const uint8_t*, true, true>
        <<<grid, 256, 0, stream>>>(A, B, C, M, N, K, ldc, epi);
}

extern "C" void launch_int8_gemm_a_tiled128_down_kernel(
    const void* a, const void* b, void* c,
    const void* scale_a, const void* scale_b, int scale_b_stride,
    const void* bias, int bias_code,
    int M, int N, int K, int ldc, int out_code, hipStream_t stream) {

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
                A, B, static_cast<__bf16*>(c), M, N, K, ldc, no_bias, stream);
        } else {
            launch_int8_tiled_a_down(
                A, B, static_cast<__bf16*>(c), M, N, K, ldc, epi, stream);
        }
    } else if (out_code == kF16) {
        launch_int8_tiled_a_down(
            A, B, static_cast<__half*>(c), M, N, K, ldc, epi, stream);
    } else if (out_code == kF32) {
        launch_int8_tiled_a_down(
            A, B, static_cast<float*>(c), M, N, K, ldc, epi, stream);
    } else {
        throw std::runtime_error("tiled-A int8 GEMM: unsupported output dtype");
    }
}
