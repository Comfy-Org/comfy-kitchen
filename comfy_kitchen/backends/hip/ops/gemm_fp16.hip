// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// FP16/BF16 GEMM on WMMA (v_wmma_f32_16x16x16_f16 / _bf16), gfx11 and gfx12. See mma.h.
//   D[M, N] = A[M, K] @ B[N, K]^T + bias, or resid + rscale * (that) fused.
// The accumulators are fp32, so this is at least as accurate as the fp16-accumulate
// contract fp16_linear promises. Returns false for shapes the caller must serve.
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
        // RDNA2 has no matrix cores, so this arch reaches the VALU
        // v_dot2_f32_f16 kernel instead of WMMA. That kernel is gated off.
        //
        // Why: it is correct and slower than torch. Measured on the 6-CU gfx1035,
        // min-of-12 in one process, calling _C.fp16_gemm -- i.e.
        // launch_fp16_gemm_kernel -- with this gate temporarily lifted:
        //
        //   shape                blocks   valu ms  torch ms  torch/valu   rel err
        //   128x128 K128               4     0.127     0.188        1.48   2.07e-04
        //   512x512 K512              64     0.385     0.315        0.82   2.07e-04
        //   1024x1024 K1024          256     1.901     0.767        0.40   2.08e-04
        //   1536x1536 K1536          576     6.072     2.075        0.34   2.08e-04
        //   1024x1024 K8192          256    14.490     6.267        0.43   2.07e-04
        //   1728x1728 K512           729     2.486     0.844        0.34   2.07e-04
        //   2048x2048 K2048         1024    13.224     4.616        0.35   2.07e-04
        //   4096x4096 K4096         4096   110.406    34.059        0.31   2.08e-04
        //   2048x11008 K4096        5504   149.018    53.945        0.36   2.08e-04
        //
        // Every rel err is fp32-accumulate noise for an fp16 product (~2e-04),
        // and the correlation against the true product is 1.0000 throughout.
        // Both epilogues (bias, and resid + rscale * (acc + bias)) agree, ragged
        // M/N that are not multiples of the 64x64 tile are right, and K up to
        // 32768 is right. So torch wins on speed alone: 2.4-3.2x from 1024^3 up,
        // and 1.2x even at 512^2. The one shape this kernel wins is 128x128, too
        // small to be worth an arch carve-out on its own.
        //
        // This replaces an earlier table here that read 1.2e+00 with correlation
        // 0.247 on the five largest shapes and reported a "correct through 676
        // blocks, wrong from 729" boundary. That was measured through
        // fp16_linear, which this gate makes the fallback rather than this
        // kernel, and it does not reproduce: the corrected kernel is right at
        // 676, 784, 841, 729 and 5504 blocks alike. The 8x8/128x128/BK32 retune
        // note in gemm_gfx10.h rests on the same discarded measurement and is
        // unverified for the same reason.
        //
        // gfx11/gfx12 have WMMA and never arrive here, so this gate is inert
        // off RDNA2 and cannot affect a dGPU.
        return false;
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
        // RDNA2: VALU v_dot2_f32_f16 path. Unreachable while
        // fp16_shape_served declines gfx10; kept so the kernel stays compiled and
        // the fix has somewhere to land. Do not re-enable without re-measuring
        // the large-grid boundary described in fp16_shape_served.
        EpiFp16 epi{static_cast<const __half*>(bias), static_cast<const __half*>(rscale),
                    static_cast<const __half*>(resid), N};
        // These must match BM/BN/THREADS in gemm_fp16_valu_kernel. They are
        // spelled out here rather than shared because the kernel's constants are
        // local to its definition. If the tile is ever retuned, change both and
        // re-verify through fp16_linear at 1792^2 and 2048^2, not just in a
        // standalone harness: an 8x8/128x128/BK32 tile measured 19% faster and
        // validated standalone, but was wrong in the library from 196 blocks up.
        constexpr int BM = 64;
        constexpr int BN = 64;
        dim3 block(256);
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        gemm_fp16_valu_kernel<__half><<<grid, block, 0, stream>>>(
            static_cast<const __half*>(a), static_cast<const __half*>(b),
            epi, static_cast<__half*>(d), M, N, K, N);
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

    // RDNA2 has no bf16 dot instruction (v_dot2_f32_bf16 is gfx11+), so there is
    // no VALU path here. This arm used to instantiate the fp16 VALU kernel on
    // bf16 data -- "uses f16 dot product on bf16 data", as the comment said --
    // which feeds bf16 bit patterns to v_dot2_f32_f16 and returns silently wrong
    // numbers: measured on the 6-CU gfx1035, _C.bf16_gemm returned served=true
    // with max_abs_err 169.5 on data whose absmax was 69.5. Decline instead, so
    // the caller serves the shape from torch. Nothing reaches this through the
    // public API today (hip/__init__.py only asks for a bf16 GEMM on the gfx1103
    // iGPU, which has MmaBf16G below), but _C.bf16_gemm is a public binding.
    if (comfy_is_gfx10()) {
        return false;
    }

    EpiFp16 epi{static_cast<const __half*>(bias), static_cast<const __half*>(rscale),
                static_cast<const __half*>(resid), N};
    launch_gemm_wmma<MmaBf16G, EpiFp16, __bf16>(
        static_cast<const uint8_t*>(a), static_cast<const uint8_t*>(b), static_cast<__bf16*>(d),
        M, N, 2 * K, N, epi, stream);
    return true;
}
