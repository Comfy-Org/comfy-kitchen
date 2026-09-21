// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// FP16/BF16 GEMM on WMMA (v_wmma_f32_16x16x16_f16 / _bf16), gfx11 and gfx12. See mma.h.
//   D[M, N] = A[M, K] @ B[N, K]^T + bias, or resid + rscale * (that) fused.
// The accumulators are fp32, so this is at least as accurate as the fp16-accumulate
// contract fp16_linear promises. Returns false for shapes the caller must serve.
#include "../epilogue.h"
#include "../gemm_wmma.h"

namespace comfy::hip_backend {

// Both launchers share the shape envelope. On high-WGP (dGPU) devices the vendor
// GEMM is preferred for shallow K and M > 8192 (CUDA's envelope; below the
// deep-K boundary where CUDA moves to stream-K, and past M=8192 the vendor GEMM
// wins). On low-WGP integrated GPUs the vendor fp16 GEMM is far slower than WMMA
// (measured 4.5-6 TF vs 7-10 TF on a 6-WGP 780M), so those shapes are served by
// WMMA instead.
inline bool fp16_shape_served(int M, int N, int K) {
    if (K % 8 != 0 || N % 8 != 0) return false;
    if (device_wgp_count() > 8 && (K <= 4096 || M > 8192)) return false;
    return true;
}

}  // namespace comfy::hip_backend

extern "C" bool launch_fp16_gemm_kernel(
    const void* a, const void* b, void* d, const void* bias, const void* rscale, const void* resid,
    int M, int N, int K, hipStream_t stream) {

    using namespace comfy::hip_backend;

    if (M == 0 || N == 0) {
        return true;  // a zero-dimension grid is hipErrorInvalidConfiguration
    }
    if (!fp16_shape_served(M, N, K)) {
        return false;
    }
    EpiFp16 epi{static_cast<const __half*>(bias), static_cast<const __half*>(rscale),
                static_cast<const __half*>(resid), N};
    launch_gemm_wmma<MmaF16, EpiFp16, __half>(
        static_cast<const uint8_t*>(a), static_cast<const uint8_t*>(b), static_cast<__half*>(d),
        M, N, 2 * K, N, epi, stream);
    return true;
}

extern "C" bool launch_bf16_gemm_kernel(
    const void* a, const void* b, void* d, const void* bias, const void* rscale, const void* resid,
    int M, int N, int K, hipStream_t stream) {

    using namespace comfy::hip_backend;

    if (M == 0 || N == 0) {
        return true;  // a zero-dimension grid is hipErrorInvalidConfiguration
    }
    if (!fp16_shape_served(M, N, K)) {
        return false;
    }
    EpiFp16 epi{static_cast<const __half*>(bias), static_cast<const __half*>(rscale),
                static_cast<const __half*>(resid), N};
    launch_gemm_wmma<MmaBf16G, EpiFp16, __bf16>(
        static_cast<const uint8_t*>(a), static_cast<const uint8_t*>(b), static_cast<__bf16*>(d),
        M, N, 2 * K, N, epi, stream);
    return true;
}