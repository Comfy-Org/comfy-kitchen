// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// FP16/BF16 GEMM on WMMA (v_wmma_f32_16x16x16_f16 / _bf16), gfx11 and gfx12. See mma.h.
//   D[M, N] = A[M, K] @ B[N, K]^T + bias, or resid + rscale * (that) fused.
// The accumulators are fp32, so this is at least as accurate as the fp16-accumulate
// contract fp16_linear promises. Returns false for shapes the caller must serve.
#include "../arch_compat.h"

#include "../epilogue.h"
#include "../gemm_gfx10.h"
#include "../gemm_wmma.h"
#include "../launchers.h"

namespace comfy::hip_backend {

// Both launchers share the shape envelope. Every device but the gfx1103 iGPU
// keeps upstream's: the vendor GEMM is preferred for shallow K and M > 8192
// (CUDA's envelope; below the deep-K boundary where CUDA moves to stream-K, and
// past M=8192 the vendor GEMM wins). On the gfx1103 iGPU the vendor fp16 GEMM is
// far slower than WMMA (measured 4.5-6 TF vs 7-10 TF on its 6 WGPs), so those
// shapes are served by WMMA instead.
inline bool fp16_shape_served(int M, int N, int K) {
    if (K % 8 != 0 || N % 8 != 0) return false;
    if (comfy_is_gfx10()) {
        // RDNA2: VALU path serves all shapes the WMMA path would
        return true;
    }
    if (comfy_small_igpu()) return true;
    if (K <= 4096 || M > 8192) return false;
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

    if (comfy_is_gfx10()) {
        // RDNA2: VALU v_dot2_f32_f16 path
        EpiFp16 epi{static_cast<const __half*>(bias), static_cast<const __half*>(rscale),
                    static_cast<const __half*>(resid), N};
        constexpr int BM = 64;
        constexpr int BN = 64;
        const int num_tiles = ((N + BN - 1) / BN) * ((M + BM - 1) / BM);
        if (num_tiles > 24) {
            dim3 block(256);
            dim3 grid(6);
            gemm_fp16_persistent_kernel<__half><<<grid, block, 0, stream>>>(
                static_cast<const __half*>(a), static_cast<const __half*>(b),
                epi, static_cast<__half*>(d), M, N, K, N);
        } else {
            dim3 block(256);
            dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
            gemm_fp16_valu_kernel<__half><<<grid, block, 0, stream>>>(
                static_cast<const __half*>(a), static_cast<const __half*>(b),
                epi, static_cast<__half*>(d), M, N, K, N);
        }
        return true;
    } else {
        EpiFp16 epi{static_cast<const __half*>(bias), static_cast<const __half*>(rscale),
                    static_cast<const __half*>(resid), N};
        launch_gemm_wmma<MmaF16, EpiFp16, __half>(
            static_cast<const uint8_t*>(a), static_cast<const uint8_t*>(b), static_cast<__half*>(d),
            M, N, 2 * K, N, epi, stream);
        return true;
    }
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

    if (comfy_is_gfx10()) {
        // RDNA2: VALU v_dot2_f32_f16 path for BF16 (uses f16 dot product on bf16 data)
        EpiFp16 epi{static_cast<const __half*>(bias), static_cast<const __half*>(rscale),
                    static_cast<const __half*>(resid), N};
        constexpr int BM = 64;
        constexpr int BN = 64;
        const int num_tiles = ((N + BN - 1) / BN) * ((M + BM - 1) / BM);
        if (num_tiles > 24) {
            dim3 block(256);
            dim3 grid(6);
            gemm_fp16_persistent_kernel<__bf16><<<grid, block, 0, stream>>>(
                static_cast<const __half*>(a), static_cast<const __half*>(b),
                epi, static_cast<__bf16*>(d), M, N, K, N);
        } else {
            dim3 block(256);
            dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
            gemm_fp16_valu_kernel<__bf16><<<grid, block, 0, stream>>>(
                static_cast<const __half*>(a), static_cast<const __half*>(b),
                epi, static_cast<__bf16*>(d), M, N, K, N);
        }
        return true;
    } else {
        EpiFp16 epi{static_cast<const __half*>(bias), static_cast<const __half*>(rscale),
                    static_cast<const __half*>(resid), N};
        launch_gemm_wmma<MmaBf16G, EpiFp16, __bf16>(
            static_cast<const uint8_t*>(a), static_cast<const uint8_t*>(b), static_cast<__bf16*>(d),
            M, N, 2 * K, N, epi, stream);
        return true;
    }
}