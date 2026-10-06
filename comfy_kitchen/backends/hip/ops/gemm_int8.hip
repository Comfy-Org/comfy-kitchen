// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// INT8 GEMM on WMMA (v_wmma_i32_16x16x16_iu8), gfx11 and gfx12. See mma.h.
// C[M, N] = (A[M, K] @ B[N, K]^T) * scale_a[row] * scale_b[col] + bias
#include <stdexcept>
#include <string>

#include "../epilogue.h"
#include "../gemm_gfx10.h"
#include "../gemm_wmma.h"
#include "../launchers.h"

namespace comfy::hip_backend {

// Small-M path: reduce K across the wave, one wave per output column.
template <typename OutT>
__global__ __launch_bounds__(256) void gemv_int8_kernel(
    const int8_t* __restrict__ A, const int8_t* __restrict__ B, OutT* __restrict__ C,
    int M, int N, int K, int ldc, EpiRowwise epi) {

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
    sum = wave_reduce_sum(sum);

    if (threadIdx.x == 0) {
        epi.init();
        C[static_cast<int64_t>(row) * ldc + col] =
            static_cast<OutT>(epi(row, col, static_cast<float>(sum)));
    }
}

template <typename OutT>
static void launch_int8(const int8_t* A, const int8_t* B, OutT* C, int M, int N, int K, int ldc,
                        EpiRowwise epi, hipStream_t stream) {
    const uint8_t* Au = reinterpret_cast<const uint8_t*>(A);
    const uint8_t* Bu = reinterpret_cast<const uint8_t*>(B);

    if (M <= 8) {
        dim3 block(kWave, 8);
        dim3 grid((N + 7) / 8, M);
        gemv_int8_kernel<OutT><<<grid, block, 0, stream>>>(A, B, C, M, N, K, ldc, epi);
        return;
    }

    if (comfy_is_gfx10()) {
        // RDNA2: no matrix cores, use VALU v_dot4_i32_i8.
        //
        // One block per tile, never a persistent walk. The persistent variant
        // this replaces was 1.14x-1.80x slower on every shape measured on the
        // 6-CU gfx1035 (256^3 .. 4096x4096x8192), because a fixed grid of
        // min(tiles, wgps * blocks_per_cu) gives the hardware's own block
        // scheduler no tiles to overlap: consecutive tiles run back to back on
        // one CU with a barrier between them. Letting the scheduler stream the
        // grid is both simpler and faster. See the standalone benchmark notes B5.
        // Shared with gemm_int8_valu_kernel so the grid cannot drift from the
        // tile the kernel assumes. See kInt8BM in gemm_gfx10.h.
        constexpr int BM = kInt8BM;
        constexpr int BN = kInt8BN;
        dim3 block(kInt8Threads);
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        gemm_int8_valu_kernel<OutT><<<grid, block, 0, stream>>>(
            A, B, epi, C, M, N, K, ldc);
    } else {
        launch_gemm_wmma<MmaInt8, EpiRowwise, OutT>(Au, Bu, C, M, N, K, ldc, epi, stream);
    }
}

}  // namespace comfy::hip_backend

// scale_b_stride is 1 for a per-output-channel weight scale, 0 for a scalar one.
extern "C" void launch_int8_gemm_kernel(
    const void* a, const void* b, void* c,
    const void* scale_a, const void* scale_b, int scale_b_stride,
    const void* bias, int bias_code,
    int M, int N, int K, int ldc, int out_code, hipStream_t stream) {

    using namespace comfy::hip_backend;

    if (M == 0 || N == 0) {
        return;  // a zero-dimension grid is hipErrorInvalidConfiguration
    }
    if (ldc < N) {
        throw std::runtime_error("int8_gemm: ldc=" + std::to_string(ldc) +
                                 " is narrower than N=" + std::to_string(N));
    }
    // Both the tile loader and the small-M GEMV read a row 16 bytes at a time; an
    // unaligned K would take the last chunk past the end of the row. An empty GEMM
    // does no reads, so this only applies once there is real work above.
    if (K % 16 != 0) {
        throw std::runtime_error("int8_gemm: K=" + std::to_string(K) +
                                 " must be a multiple of 16");
    }

    EpiRowwise epi{static_cast<const float*>(scale_a), static_cast<const float*>(scale_b),
                   scale_b_stride, bias, bias_code};

    const int8_t* A = static_cast<const int8_t*>(a);
    const int8_t* B = static_cast<const int8_t*>(b);

    if (out_code == kBF16) {
        launch_int8<__bf16>(A, B, static_cast<__bf16*>(c), M, N, K, ldc, epi, stream);
    } else if (out_code == kF16) {
        launch_int8<__half>(A, B, static_cast<__half*>(c), M, N, K, ldc, epi, stream);
    } else if (out_code == kF32) {
        launch_int8<float>(A, B, static_cast<float*>(c), M, N, K, ldc, epi, stream);
    } else {
        throw std::runtime_error("int8_gemm: unsupported output dtype");
    }
}

// INT4 on VALU, the int4 counterpart of the above. a and b are signed int4
// packed two per byte (low nibble = even k), so a row is K / 2 bytes wide and a
// 16-byte LDS read covers 32 k-values -- hence K must be a multiple of 32.
//
// The launcher lives here rather than in gemm_gfx10.h because that header is
// included by ops/gemm_fp16.hip as well, and a plain extern "C" definition would
// be emitted into both objects and collide at link time.
extern "C" void launch_int4_gemm_kernel(
    const void* a, const void* b, void* c,
    const void* scale_a, const void* scale_b, int scale_b_stride,
    const void* bias, int bias_code,
    int M, int N, int K, int ldc, int out_code, hipStream_t stream) {

    using namespace comfy::hip_backend;

    if (M == 0 || N == 0) {
        return;
    }
    // v_dot8_i32_i4 exists on gfx10 and nowhere else: the stub in gemm_gfx10.h's
    // !__GFX10__ arm is a trap. This binding is exposed on every target, so a
    // caller on an RDNA3/4 part would get that trap rather than an answer --
    // refuse here instead. convrot_w4a4.hip checks comfy_is_gfx10() before
    // calling, so the only way to land here is a direct call on the wrong device.
    if (!comfy_is_gfx10()) {
        throw std::runtime_error("int4_gemm: v_dot8_i32_i4 is gfx10 only");
    }
    if (ldc < N) {
        throw std::runtime_error("int4_gemm: ldc=" + std::to_string(ldc) +
                                 " is narrower than N=" + std::to_string(N));
    }
    // The tile reads a row 16 bytes at a time and each read is 32 k-values, so
    // the row has to be a whole number of reads. An empty GEMM does no reads, so
    // this only applies once there is real work above.
    if (K % 32 != 0) {
        throw std::runtime_error("int4_gemm: K=" + std::to_string(K) +
                                 " must be a multiple of 32");
    }

    EpiRowwise epi{static_cast<const float*>(scale_a), static_cast<const float*>(scale_b),
                   scale_b_stride, bias, bias_code};

    const int8_t* A = static_cast<const int8_t*>(a);
    const int8_t* B = static_cast<const int8_t*>(b);

    // One block per tile, the same grid launch_int8 settled on.
    dim3 block(kInt8Threads);
    dim3 grid((N + kInt8BN - 1) / kInt8BN, (M + kInt8BM - 1) / kInt8BM);

    if (out_code == kBF16) {
        gemm_int4_valu_kernel<__bf16><<<grid, block, 0, stream>>>(A, B, epi,
                                                                  static_cast<__bf16*>(c), M, N, K, ldc);
    } else if (out_code == kF16) {
        gemm_int4_valu_kernel<__half><<<grid, block, 0, stream>>>(A, B, epi,
                                                                 static_cast<__half*>(c), M, N, K, ldc);
    } else if (out_code == kF32) {
        gemm_int4_valu_kernel<float><<<grid, block, 0, stream>>>(A, B, epi,
                                                                 static_cast<float*>(c), M, N, K, ldc);
    } else {
        throw std::runtime_error("int4_gemm: unsupported output dtype");
    }
}
