// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// FP8 scaled GEMV for M <= 8, built wave64 (CMakeLists.txt), where it measured 6%
// faster. See ops/gemm_fp8.hip for the GEMM.
#include <stdexcept>

#include "../epilogue.h"
#include "../fp8_utils.h"
#include "../launchers.h"
#include "../mma.h"

namespace comfy::hip_backend {
namespace {

// Reduce K across 32 lanes, one output column per 32.
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
void launch_gemv(const void* a, const void* b, void* c, int M, int N, int K, EpiTensorwise epi,
                 hipStream_t stream) {
    dim3 block(kWave, 8);
    dim3 grid((N + 7) / 8, M);
    gemv_fp8_kernel<OutT><<<grid, block, 0, stream>>>(
        static_cast<const uint8_t*>(a), static_cast<const uint8_t*>(b), static_cast<OutT*>(c), M,
        N, K, epi);
}

}  // namespace
}  // namespace comfy::hip_backend

extern "C" void launch_gemv_fp8_kernel(const void* a, const void* b, void* c, const void* scale_a,
                                       const void* scale_b, const void* bias, int bias_code,
                                       int M, int N, int K, int out_code, hipStream_t stream) {
    using namespace comfy::hip_backend;

    EpiTensorwise epi{static_cast<const float*>(scale_a), static_cast<const float*>(scale_b),
                      bias, bias_code, 0.0f};
    if (out_code == kBF16) {
        launch_gemv<__bf16>(a, b, c, M, N, K, epi, stream);
    } else if (out_code == kF16) {
        launch_gemv<__half>(a, b, c, M, N, K, epi, stream);
    } else if (out_code == kF32) {
        launch_gemv<float>(a, b, c, M, N, K, epi, stream);
    } else {
        throw std::runtime_error("scaled_mm_fp8: unsupported output dtype");
    }
}
