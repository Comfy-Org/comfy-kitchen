// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// FP16 GEMM on WMMA (v_wmma_f32_16x16x16_f16), gfx11 and gfx12. See mma.h.
//   D[M, N] = A[M, K] @ B[N, K]^T + bias, or resid + rscale * (that) fused.
// The accumulators are fp32, so this is at least as accurate as the fp16-accumulate
// contract fp16_linear promises. Returns false for shapes the caller must serve.
#include "../epilogue.h"
#include "../gemm_wmma.h"

extern "C" bool launch_fp16_gemm_kernel(
    const void* a, const void* b, void* d, const void* bias, const void* rscale, const void* resid,
    int M, int N, int K, hipStream_t stream) {

    using namespace comfy::hip_backend;

    if (M == 0 || N == 0) {
        return true;  // a zero-dimension grid is hipErrorInvalidConfiguration
    }
    // CUDA's envelope: a zero-K linear is the bias broadcast, and past M=8192 the
    // vendor GEMM is preferred. Shallower than the deep-K boundary CUDA moves to
    // stream-K at, an fp16 tile holds half the K-steps of an 8-bit one and the vendor
    // GEMM is the faster path, so those shapes are declined too.
    if (K <= 4096 || K % 8 != 0 || N % 8 != 0 || M > 8192) {
        return false;
    }
    EpiFp16 epi{static_cast<const __half*>(bias), static_cast<const __half*>(rscale),
                static_cast<const __half*>(resid), N};
    launch_gemm_wmma<MmaF16, EpiFp16, __half>(
        static_cast<const uint8_t*>(a), static_cast<const uint8_t*>(b), static_cast<__half*>(d),
        M, N, 2 * K, N, epi, stream);
    return true;
}
