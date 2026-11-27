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

template <typename T>
__forceinline__ __device__ float gemv_to_float(T v) {
    return static_cast<float>(v);
}
template <>
__forceinline__ __device__ float gemv_to_float<__half>(__half v) {
    return __half2float(v);
}

// Small-M path: one wave per output column, the K reduction inside the wave.
//
// The WMMA path stages a 128-row tile for a single row. The weight traffic is
// the same -- one pass over the weight either way -- but the stager's loop
// runs over 128 rows to drop 127 of them, and the tile's shape is a poor fit
// for one row. Measured on the 6-WGP 780M with K=4096, against the int8
// GEMV's rate on the same bytes:
//
//     N=1024    17.3 GB/s    the WMMA path
//     N=4096    26.7 GB/s
//     N=12288   45.6 GB/s
//     N=151936  70.0 GB/s    (the weight is big enough that the stager's waste
//                             is a small part of it)
//
// The GEMV reaches the int8 GEMV's 57-80 GB/s at the small and middle sizes.
//
// Eight elements per lane is four 16-byte loads' worth of reduction; the
// conversions and the multiplies are 24 instructions per 16 bytes, which at
// 50 GB/s of weight is about half the memory time, so the kernel stays
// bandwidth-bound.
template <typename T, typename OutT>
__global__ __launch_bounds__(256) void gemv_fp16_kernel(
    const T* __restrict__ A, const T* __restrict__ B, OutT* __restrict__ C,
    int M, int N, int K, int ldc, EpiFp16 epi) {

    const int col = blockIdx.x * blockDim.y + threadIdx.y;
    const int row = blockIdx.y;
    if (col >= N || row >= M) return;

    const T* a = A + static_cast<int64_t>(row) * K;
    const T* b = B + static_cast<int64_t>(col) * K;

    float sum = 0.0f;
    for (int k = threadIdx.x * 8; k < K; k += kWave * 8) {
        const uint4 av = *reinterpret_cast<const uint4*>(a + k);
        const uint4 bv = *reinterpret_cast<const uint4*>(b + k);
        const T* ah = reinterpret_cast<const T*>(&av);
        const T* bh = reinterpret_cast<const T*>(&bv);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            sum += gemv_to_float<T>(ah[i]) * gemv_to_float<T>(bh[i]);
        }
    }
    sum = wave_reduce_sum(sum);

    if (threadIdx.x == 0) {
        epi.init();
        C[static_cast<int64_t>(row) * ldc + col] = static_cast<OutT>(
            epi(row, col, sum, epi.row_scale(row), epi.col_scale(col), epi.col_bias(col)));
    }
}

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

// Whether the one-row GEMV beats the WMMA path for an output of N columns.
//
// The GEMV reads the weight once, like the WMMA path, but it does the products
// in the vector unit rather than the matrix pipe, and it has no 128-row tile to
// stage. Which of those matters depends on how much weight there is to read.
//
// Measured on the 6-WGP 780M at K=4096, the two paths interleaved in one
// process (M=9 leaves the GEMV, whose cutoff is M <= 8, and takes the WMMA
// path, so the comparison is at the same shape):
//
//     N=1024    16.7 GB/s   against 17.2    0.97x
//     N=4096    27.1        against 35.1    0.77x
//     N=12288   45.2        against 51.7    0.87x
//     N=24576   78.3        against 66.8    1.17x
//     N=32768   80.2        against 68.0    1.18x
//     N=65536   80.0        against 66.7    1.20x
//     N=151936  80.5        against 67.4    1.19x
//
// The crossover is a step, not a slope: below it the WMMA path's matrix pipe
// is worth more than the GEMV's simpler staging, above it the weight traffic
// is the whole cost and the GEMV reaches the roofline where the WMMA path
// does not.
//
// The lm_head of a ConvRot text encoder is the shape that wants this: M=1,
// K=4096, N=151936, which is 1.24 GB of weight per token. The projections
// below it (N=12288 and under) are better on the WMMA path.
//
// iGPU only: the crossover is a property of this part's vector and matrix
// pipes, and the dGPU path is not measured here.
inline bool fp16_gemv_preferred(int N) {
    return comfy_small_igpu() && N >= 24576;
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
        if (M <= 8 && fp16_gemv_preferred(N)) {
            // One wave per output column. The int8 launcher has taken this path
            // for M <= 8 since it was written; see gemv_fp16_kernel for what the
            // WMMA path costs at one row.
            dim3 block(kWave, 8);
            dim3 grid((N + 7) / 8, M);
            gemv_fp16_kernel<__half, __half><<<grid, block, 0, stream>>>(
                static_cast<const __half*>(a), static_cast<const __half*>(b),
                static_cast<__half*>(d), M, N, K, N, epi);
            return true;
        }
        launch_gemm_wmma<MmaF16, EpiFp16, __half>(
            static_cast<const uint8_t*>(a), static_cast<const uint8_t*>(b), static_cast<__half*>(d),
            M, N, 2 * K, N, epi, stream, GemmPolicy{true, false});
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
    if (M <= 8 && fp16_gemv_preferred(N)) {
        // One wave per output column; see gemv_fp16_kernel.
        dim3 block(kWave, 8);
        dim3 grid((N + 7) / 8, M);
        gemv_fp16_kernel<__bf16, __bf16><<<grid, block, 0, stream>>>(
            static_cast<const __bf16*>(a), static_cast<const __bf16*>(b),
            static_cast<__bf16*>(d), M, N, K, N, epi);
        return true;
    }
    launch_gemm_wmma<MmaBf16G, EpiFp16, __bf16>(
        static_cast<const uint8_t*>(a), static_cast<const uint8_t*>(b), static_cast<__bf16*>(d),
        M, N, 2 * K, N, epi, stream, GemmPolicy{true, false});
    return true;
}
