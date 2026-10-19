// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// INT8 GEMM on WMMA (v_wmma_i32_16x16x16_iu8), gfx11 and gfx12. See mma.h.
// C[M, N] = (A[M, K] @ B[N, K]^T) * scale_a[row] * scale_b[col] + bias
#include <stdexcept>
#include <string>

#include "../device_arch.h"
#include "../epilogue.h"
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
    launch_gemm_wmma<MmaInt8, EpiRowwise, OutT>(Au, Bu, C, M, N, K, ldc, epi, stream);
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

namespace comfy::hip_backend {

// Two 128-row activation tiles share one weight tile. Each half of the block
// retains the single-M accumulator shape and K order; only B staging is shared.
// The producer still writes 128x128 tiles, including a padded final M tile.
template <typename Epi>
__global__ __launch_bounds__(512) void gemm_int8_dual_m_tiled_kernel(
    const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
    __bf16* __restrict__ C, int M, int N, int K, int ldc, Epi epi) {
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    constexpr int BM = 128, BN = 128, BK = 128, stride = BK + kLdsPad;
    constexpr int TM = 2, TN = 4, steps = BK / MmaInt8::kStepBytes;
    __shared__ __align__(16) uint8_t As[2][BM * stride];
    __shared__ __align__(16) uint8_t Bs[BN * stride];

    const int lane = threadIdx.x % kWave;
    const int warp = threadIdx.x / kWave;
    const int half = warp / 8;
    const int wm = (warp % 8) / 2;
    const int wn = warp % 2;

    // Two block-rows retain the four-M-tile B locality of the single-M kernel.
    const int bid = blockIdx.y * gridDim.x + blockIdx.x;
    const int per_group = 2 * gridDim.x;
    const int group = bid / per_group;
    const int within = bid % per_group;
    const int group_rows = min(2, static_cast<int>(gridDim.y) - group * 2);
    const int pair = group * 2 + within % group_rows;
    const int n0 = (within / group_rows) * BN;
    const int m_pair = pair * 2 * BM;
    const int m0 = m_pair + half * BM;
    const bool second_tile = m_pair + BM < M;

    MmaInt8::Acc acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = MmaInt8::zero();

    TileStager<BM, BK, 512, true> sa0, sa1;
    TileStager<BN, BK, 512> sb;
    const int64_t tile0 = static_cast<int64_t>(pair * 2) * (K / BK);
    const int64_t tile1 = tile0 + K / BK;
    sa0.load_contiguous(A + tile0 * BM * BK);
    if (second_tile) sa1.load_contiguous(A + tile1 * BM * BK);
    sb.load(B, n0, N, 0, K);
    sa0.store(As[0]);
    if (second_tile) sa1.store(As[1]);
    sb.store(Bs);
    __syncthreads();

    const int row = frag_row(lane);
    for (int kb = 0; kb < K; kb += BK) {
        const int next = kb + BK;
        const bool has_next = next < K;
        if (has_next) {
            sa0.load_contiguous(A + (tile0 + next / BK) * BM * BK);
            if (second_tile)
                sa1.load_contiguous(A + (tile1 + next / BK) * BM * BK);
            sb.load(B, n0, N, next, K);
        }
        // An absent second tile must participate in barriers, but not read LDS.
        if (m0 < M) {
            MmaInt8::Frag af[2][TM], bf[2][TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                af[0][i] = MmaInt8::load(As[half], wm * TM * 16 + i * 16 + row,
                                        0, stride, lane);
            #pragma unroll
            for (int j = 0; j < TN; ++j)
                bf[0][j] = MmaInt8::load(Bs, wn * TN * 16 + j * 16 + row,
                                        0, stride, lane);
            #pragma unroll
            for (int kk = 0; kk < steps; ++kk) {
                const int cur = kk & 1, nxt = cur ^ 1;
                if (kk + 1 < steps) {
                    const int kbyte = (kk + 1) * MmaInt8::kStepBytes;
                    #pragma unroll
                    for (int i = 0; i < TM; ++i)
                        af[nxt][i] = MmaInt8::load(
                            As[half], wm * TM * 16 + i * 16 + row, kbyte, stride, lane);
                    #pragma unroll
                    for (int j = 0; j < TN; ++j)
                        bf[nxt][j] = MmaInt8::load(
                            Bs, wn * TN * 16 + j * 16 + row, kbyte, stride, lane);
                }
                #pragma unroll
                for (int i = 0; i < TM; ++i)
                    #pragma unroll
                    for (int j = 0; j < TN; ++j)
                        acc[i][j] = MmaInt8::mma(af[cur][i], bf[cur][j], acc[i][j]);
            }
        }
        if (has_next) {
            __syncthreads();
            sa0.store(As[0]);
            if (second_tile) sa1.store(As[1]);
            sb.store(Bs);
            __syncthreads();
        }
    }

    // Keep channel operands in registers across output rows. Preserve the
    // producer's (acc * row_scale) * channel_scale, followed by a separate add.
    float col_scale[TN], col_bias[TN];
    const int col_lane = acc_col(lane);
    #pragma unroll
    for (int j = 0; j < TN; ++j) {
        const int col = n0 + wn * TN * 16 + j * 16 + col_lane;
        if constexpr (std::is_same_v<Epi, EpiRowwiseNoBias>) {
            col_scale[j] = col < N ? epi.scale_b[col] : 0.0f;
        } else {
            col_scale[j] = col < N ? epi.scale_b[col * epi.scale_b_stride] : 0.0f;
            col_bias[j] = col < N && epi.bias
                ? load_scalar(epi.bias, epi.bias_code, col) : 0.0f;
        }
    }
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int r = m0 + wm * TM * 16 + i * 16 + acc_row(lane, e);
            if (r >= M) continue;
            const float row_scale = epi.scale_a[r];
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int col = n0 + wn * TN * 16 + j * 16 + col_lane;
                if (col >= N) continue;
                float value = MmaInt8::get(acc[i][j], e) * row_scale;
                value *= col_scale[j];
                if constexpr (!std::is_same_v<Epi, EpiRowwiseNoBias>)
                    value += col_bias[j];
                C[static_cast<int64_t>(r) * ldc + col] = static_cast<__bf16>(value);
            }
        }
    }
}

}  // namespace comfy::hip_backend

template <typename OutT, typename Epi>
static void launch_int8_tiled_a(
    const uint8_t* A, const uint8_t* B, OutT* C,
    int M, int N, int K, int ldc, Epi epi, hipStream_t stream) {
    using namespace comfy::hip_backend;
    constexpr int BM = 128, BN = 128, BKB = 128;
    if constexpr (std::is_same_v<OutT, __bf16>) {
        // ROCm 10.1 gfx12 sweep: sharing B wins the model-sized rows; retain
        // single-M below this one cut to avoid underfilled dual-M blocks.
        if (use_nonduplicated_wmma_schedule() && M >= 1024) {
            dim3 grid((N + BN - 1) / BN, (M + 2 * BM - 1) / (2 * BM));
            gemm_int8_dual_m_tiled_kernel<<<grid, 512, 0, stream>>>(
                A, B, C, M, N, K, ldc, epi);
            return;
        }
    }
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_wmma_kernel<MmaInt8, Epi, OutT, BM, BN, BKB,
                     4, 2, 2, 4, const uint8_t*, true, true>
        <<<grid, 256, 0, stream>>>(A, B, C, M, N, K, ldc, epi);
}

extern "C" void launch_int8_gemm_a_tiled128_kernel(
    const void* a, const void* b, void* c,
    const void* scale_a, const void* scale_b, int scale_b_stride,
    const void* bias, int bias_code,
    int M, int N, int K, int ldc, int out_code, hipStream_t stream) {

    using namespace comfy::hip_backend;
    if (M == 0) return;
    if (N <= 0 || K <= 0 || K % 128 != 0 || ldc < N) {
        throw std::runtime_error(
            "tiled-A int8 GEMM: requires positive N/K, K divisible by 128 and ldc>=N");
    }
    EpiRowwise epi{
        static_cast<const float*>(scale_a), static_cast<const float*>(scale_b),
        scale_b_stride, bias, bias_code};
    const uint8_t* A = static_cast<const uint8_t*>(a);
    const uint8_t* B = static_cast<const uint8_t*>(b);
    if (out_code == kBF16) {
        if (epi.bias == nullptr && epi.scale_b_stride == 1) {
            EpiRowwiseNoBias no_bias{epi.scale_a, epi.scale_b};
            launch_int8_tiled_a(
                A, B, static_cast<__bf16*>(c), M, N, K, ldc, no_bias, stream);
        } else {
            launch_int8_tiled_a(
                A, B, static_cast<__bf16*>(c), M, N, K, ldc, epi, stream);
        }
    } else if (out_code == kF16) {
        launch_int8_tiled_a(
            A, B, static_cast<__half*>(c), M, N, K, ldc, epi, stream);
    } else if (out_code == kF32) {
        launch_int8_tiled_a(
            A, B, static_cast<float*>(c), M, N, K, ldc, epi, stream);
    } else {
        throw std::runtime_error("tiled-A int8 GEMM: unsupported output dtype");
    }
}
