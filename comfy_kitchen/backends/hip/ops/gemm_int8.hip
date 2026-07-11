// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// INT8 GEMM on gfx12 WMMA (v_wmma_i32_16x16x16_iu8).
// C[M, N] = (A[M, K] @ B[N, K]^T) * scale_a[row] * scale_b[col] + bias
#include <stdexcept>
#include <string>

#include "../epilogue.h"
#include "../gemm_wmma.h"

namespace comfy::hip_backend {

// Small-M path: reduce K across the wave, one wave per output column.
template <typename OutT>
__global__ __launch_bounds__(256) void gemv_int8_kernel(
    const int8_t* __restrict__ A, const int8_t* __restrict__ B, OutT* __restrict__ C,
    int M, int N, int K, EpiRowwise epi) {

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
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor(sum, off, kWave);

    if (threadIdx.x == 0) {
        epi.init();
        C[static_cast<int64_t>(row) * N + col] =
            static_cast<OutT>(epi(row, col, static_cast<float>(sum)));
    }
}

template <typename OutT>
static void launch_int8(const int8_t* A, const int8_t* B, OutT* C, int M, int N, int K,
                        EpiRowwise epi, hipStream_t stream) {
    const uint8_t* Au = reinterpret_cast<const uint8_t*>(A);
    const uint8_t* Bu = reinterpret_cast<const uint8_t*>(B);

    if (M <= 8) {
        dim3 block(kWave, 8);
        dim3 grid((N + 7) / 8, M);
        gemv_int8_kernel<OutT><<<grid, block, 0, stream>>>(A, B, C, M, N, K, epi);
        return;
    }
    if (M >= 512 && N >= 128 && K > N) {
        constexpr int BM = 256, BN = 128, BKB = 64;
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        gemm_wmma_kernel<MmaInt8, EpiRowwise, OutT, BM, BN, BKB, 4, 2, 4, 4>
            <<<grid, 256, 0, stream>>>(Au, Bu, C, M, N, K, epi);
    } else if (M >= 96 && N >= 96) {
        constexpr int BM = 128, BN = 128, BKB = 64;
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        gemm_wmma_kernel<MmaInt8, EpiRowwise, OutT, BM, BN, BKB, 4, 2, 2, 4>
            <<<grid, 256, 0, stream>>>(Au, Bu, C, M, N, K, epi);
    } else {
        constexpr int BM = 64, BN = 64, BKB = 64;
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        gemm_wmma_kernel<MmaInt8, EpiRowwise, OutT, BM, BN, BKB, 2, 2, 2, 2>
            <<<grid, 128, 0, stream>>>(Au, Bu, C, M, N, K, epi);
    }
}

}  // namespace comfy::hip_backend

// scale_b_stride is 1 for a per-output-channel weight scale, 0 for a scalar one.
extern "C" void launch_int8_gemm_kernel(
    const void* a, const void* b, void* c,
    const void* scale_a, const void* scale_b, int scale_b_stride,
    const void* bias, int bias_code,
    int M, int N, int K, int out_code, hipStream_t stream) {

    using namespace comfy::hip_backend;

    // Both the tile loader and the small-M GEMV read a row 16 bytes at a time; an
    // unaligned K would take the last chunk past the end of the row.
    if (K % 16 != 0) {
        throw std::runtime_error("int8_gemm: K=" + std::to_string(K) +
                                 " must be a multiple of 16");
    }
    if (M == 0 || N == 0) {
        return;  // a zero-dimension grid is hipErrorInvalidConfiguration
    }

    EpiRowwise epi{static_cast<const float*>(scale_a), static_cast<const float*>(scale_b),
                   scale_b_stride, bias, bias_code};

    const int8_t* A = static_cast<const int8_t*>(a);
    const int8_t* B = static_cast<const int8_t*>(b);

    if (out_code == kBF16) {
        launch_int8<__bf16>(A, B, static_cast<__bf16*>(c), M, N, K, epi, stream);
    } else if (out_code == kF16) {
        launch_int8<__half>(A, B, static_cast<__half*>(c), M, N, K, epi, stream);
    } else if (out_code == kF32) {
        launch_int8<float>(A, B, static_cast<float*>(c), M, N, K, epi, stream);
    } else {
        throw std::runtime_error("int8_gemm: unsupported output dtype");
    }
}
