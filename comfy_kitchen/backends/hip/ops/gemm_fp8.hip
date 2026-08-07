// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// FP8 scaled GEMM on WMMA (v_wmma_f32_16x16x16_fp8_fp8 on gfx12; gfx11 has no
// fp8 WMMA and widens the operands to bf16). See mma.h.
// C[M, N] = (A[M, K] @ B[N, K]^T) * scale_a * scale_b + bias
#include <stdexcept>
#include <string>

#include "../epilogue.h"
#include "../fp8_utils.h"
#include "../gemm_wmma.h"

namespace comfy::hip_backend {

// Small-M path. A 16-row WMMA tile is mostly idle at small M, so reduce K
// across the wave instead: one wave per output column.
template <typename OutT>
__global__ __launch_bounds__(256) void gemv_fp8_kernel(
    const uint8_t* __restrict__ A, const uint8_t* __restrict__ B, OutT* __restrict__ C,
    int M, int N, int K, EpiTensorwise epi) {

    const int col = blockIdx.x * blockDim.y + threadIdx.y;
    const int row = blockIdx.y;
    if (col >= N || row >= M) return;

    const uint8_t* a = A + static_cast<int64_t>(row) * K;
    const uint8_t* b = B + static_cast<int64_t>(col) * K;

    // Bandwidth-bound on streaming B: 16 bytes per lane is a 512-byte wave
    // transaction.
    float sum = 0.0f;
    for (int k = threadIdx.x * 16; k < K; k += kWave * 16) {
        // K is a multiple of 16, so a 16-byte chunk never straddles the end.
        const uint4 av = *reinterpret_cast<const uint4*>(a + k);
        const uint4 bv = *reinterpret_cast<const uint4*>(b + k);
        const uint8_t* ab = reinterpret_cast<const uint8_t*>(&av);
        const uint8_t* bb = reinterpret_cast<const uint8_t*>(&bv);
        #pragma unroll
        for (int i = 0; i < 16; ++i) sum += fp8_to_float(ab[i]) * fp8_to_float(bb[i]);
    }
    sum = wave_reduce_sum(sum);

    if (threadIdx.x == 0) {
        epi.init();
        C[static_cast<int64_t>(row) * N + col] = static_cast<OutT>(epi(row, col, sum));
    }
}

template <typename OutT>
static void launch_fp8(const uint8_t* A, const uint8_t* B, OutT* C, int M, int N, int K,
                       EpiTensorwise epi, hipStream_t stream) {
    if (M <= 8) {
        dim3 block(kWave, 8);
        dim3 grid((N + 7) / 8, M);
        gemv_fp8_kernel<OutT><<<grid, block, 0, stream>>>(A, B, C, M, N, K, epi);
        return;
    }
    launch_gemm_wmma<MmaFp8, EpiTensorwise, OutT>(A, B, C, M, N, K, N, epi, stream);
}

}  // namespace comfy::hip_backend

extern "C" void launch_scaled_mm_fp8_kernel(
    const void* a, const void* b, void* c,
    const void* scale_a, const void* scale_b, const void* bias, int bias_code,
    int M, int N, int K, int out_code, hipStream_t stream) {

    using namespace comfy::hip_backend;

    if (M == 0 || N == 0) {
        return;  // a zero-dimension grid is hipErrorInvalidConfiguration
    }
    // Both the tile loader and the small-M GEMV read a row 16 bytes at a time; an
    // unaligned K would take the last chunk past the end of the row. An empty GEMM
    // does no reads, so this only applies once there is real work above.
    if (K % 16 != 0) {
        throw std::runtime_error("scaled_mm_fp8: K=" + std::to_string(K) +
                                 " must be a multiple of 16");
    }

    EpiTensorwise epi{static_cast<const float*>(scale_a), static_cast<const float*>(scale_b),
                      bias, bias_code, 0.0f};

    const uint8_t* A = static_cast<const uint8_t*>(a);
    const uint8_t* B = static_cast<const uint8_t*>(b);

    if (out_code == kBF16) {
        launch_fp8<__bf16>(A, B, static_cast<__bf16*>(c), M, N, K, epi, stream);
    } else if (out_code == kF16) {
        launch_fp8<__half>(A, B, static_cast<__half*>(c), M, N, K, epi, stream);
    } else if (out_code == kF32) {
        launch_fp8<float>(A, B, static_cast<float*>(c), M, N, K, epi, stream);
    } else {
        throw std::runtime_error("scaled_mm_fp8: unsupported output dtype");
    }
}
