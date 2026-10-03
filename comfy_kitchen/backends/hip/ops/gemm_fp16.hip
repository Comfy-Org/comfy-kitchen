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
        // It is wrong above a size threshold whose mechanism is unknown, and it
        // is not faster than torch anywhere it is correct. Measured on the 6-CU
        // gfx1035, min-of-12 interleaved, through fp16_linear:
        //
        //   shape                hip ms   eager ms   eager/hip   rel err
        //   128x128 K128           0.155     0.154       0.99      3.2e-04 ok
        //   512x512 K512           0.305     0.319       1.04      2.6e-04 ok
        //   1024x1024 K1024        0.717     0.702       0.98      3.9e-04 ok
        //   1536x1536 K1536        2.476     2.080       0.84      3.1e-04 ok
        //   1024x1024 K8192       10.242     6.659       0.65      1.2e+00 WRONG
        //   1728x1728 K512         2.272     0.825       0.36      1.2e+00 WRONG
        //   2048x2048 K2048        9.235     4.849       0.53      1.2e+00 WRONG
        //   4096x4096 K4096       73.372    34.450       0.47      1.2e+00 WRONG
        //   2048x11008 K4096      98.946    52.261       0.53      1.2e+00 WRONG
        //
        // Correct shapes tie eager; wrong shapes are 1.5-2.8x slower *and*
        // wrong, so serving them from torch is strictly better on both axes.
        //
        // The failure: output fully written, value distribution preserved to
        // within 1% (std ratio 0.989) but correlation against the true product
        // only 0.247, bitwise repeatable, and independent of K. The boundary is
        // per-axis and moves as the other axis grows. Restoring upstream's
        // persistent kernel and its num_tiles > 24 gate did not move it, so this
        // predates the restructuring in this tree. Not yet root-caused; see
        // benchmark_attn/待验证.md B7.
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
