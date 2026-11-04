// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// FP8 scaled GEMM, as the thread-level GEMM in gemm_simt.h.
// C[M, N] = (A[M, K] @ B[N, K]^T) * scale_a * scale_b + bias
#include <stdexcept>
#include <string>

#include "../epilogue.h"
#include "../fp8_utils.h"
#include "../gemm_simt.h"
#include "../launchers.h"

namespace comfy::hip_backend {

template <typename OutT>
static void launch_fp8(const uint8_t* A, const uint8_t* B, OutT* C, int M, int N, int K,
                       EpiTensorwise epi, hipStream_t stream) {
    launch_gemm_simt<SimtFp8, EpiTensorwise, OutT>(A, B, C, M, N, K, N, epi, stream);
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

    if (M <= 8) {
        launch_gemv_fp8_kernel(a, b, c, scale_a, scale_b, bias, bias_code, M, N, K, out_code,
                               stream);
        return;
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
