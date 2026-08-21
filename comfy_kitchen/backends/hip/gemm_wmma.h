// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Tiled WMMA GEMM core, shared by the fp8, int8 and int4 paths on both gfx11 and
// gfx12.
//
// Computes C[M, N] = epilogue(A[M, K] @ B[N, K]^T). The B operand is the weight
// in its natural (N, K) row-major form, matching torch linear. C is written with
// row stride ldc, so a caller splitting N into column chunks can point at a slice
// of a wider output; ldc == N for a whole GEMM.
//
// The tile loop is byte-addressed: LDS holds raw rows, and how many bytes of a
// row one MMA consumes (Mma::kStepBytes) and how a lane reads its fragment out of
// them (Mma::load) belong to the policy, which is where the two architectures
// differ. See mma.h.
//
// The K loop is software-pipelined: the next tile's global loads are issued into
// registers before the current tile's math.
#pragma once

#include <atomic>
#include <type_traits>

#include "mma.h"

namespace comfy::hip_backend {

// LDS row padding, in bytes. 8 spreads the 16 lanes of a fragment read across
// distinct banks. 16 would preserve 128-bit LDS access but aliases rows
// 0/4/8/... onto the same bank.
constexpr int kLdsPad = 8;

// Staging for one ROWS x BKB byte tile. load() issues only the global reads, so
// the caller can place the tile's math between load() and store().
//
// kbytes is the source row length in bytes (K for 8-bit types, K/2 for int4) and
// is a multiple of 16, so a 16-byte chunk starting inside a row also ends inside
// it. Out-of-range rows and the K tail are zero-filled.
template <int ROWS, int BKB, int THREADS, bool COALESCED_STAGING = false>
struct TileStager {
    static constexpr int kChunksPerRow = BKB / 16;
    static constexpr int kChunks = ROWS * kChunksPerRow;
    static constexpr int kPerThread = kChunks / THREADS;
    static constexpr int kStride = BKB + kLdsPad;
    // Both divisions truncate, and either remainder would leave part of the tile
    // unwritten in LDS for the math to then read as stale.
    static_assert(BKB % 16 == 0, "BKB must be a whole number of 16-byte chunks");
    static_assert(kChunks % THREADS == 0, "THREADS must divide the tile's 16-byte chunks");

    uint4 regs[kPerThread];

    template <bool ASSUME_ROWS_FULL = false, bool ASSUME_K_FULL = false>
    __forceinline__ __device__ void load(const uint8_t* __restrict__ src, int row0, int rows_total,
                                         int kbyte0, int kbytes) {
        const int tid = threadIdx.x;
        #pragma unroll
        for (int i = 0; i < kPerThread; ++i) {
            // A thread-strided assignment makes every load instruction cover
            // consecutive 16-byte chunks across a wave. Keep the original
            // thread-major order as the control until exact-shape profiling
            // promotes this layout.
            const int c = COALESCED_STAGING ? tid + i * THREADS
                                             : tid * kPerThread + i;
            const int grow = row0 + c / kChunksPerRow;
            const int gk = kbyte0 + (c % kChunksPerRow) * 16;

            if constexpr (ASSUME_ROWS_FULL && ASSUME_K_FULL) {
                const uint8_t* const p =
                    src + static_cast<int64_t>(grow) * kbytes + gk;
                regs[i] = *reinterpret_cast<const uint4*>(p);
            } else {
                const bool valid_row = ASSUME_ROWS_FULL || grow < rows_total;
                const bool valid_k = ASSUME_K_FULL || gk < kbytes;
                if (valid_row && valid_k) {
                    const uint8_t* const p =
                        src + static_cast<int64_t>(grow) * kbytes + gk;
                    regs[i] = *reinterpret_cast<const uint4*>(p);
                } else {
                    regs[i] = make_uint4(0, 0, 0, 0);
                }
            }
        }
    }

    // Load one physically contiguous [ROWS, BKB] activation tile. The caller
    // supplies A as [M_tile, K_tile, ROWS, BKB], so the wave-coalesced chunk
    // assignment becomes a dense global-memory span instead of touching four
    // K-strided rows per wave instruction.
    __forceinline__ __device__ void load_contiguous(
        const uint8_t* __restrict__ tile) {
        const int tid = threadIdx.x;
        #pragma unroll
        for (int i = 0; i < kPerThread; ++i) {
            const int c = COALESCED_STAGING ? tid + i * THREADS
                                             : tid * kPerThread + i;
            const uint8_t* const p = tile + static_cast<int64_t>(c) * 16;
            regs[i] = *reinterpret_cast<const uint4*>(p);
        }
    }

    __forceinline__ __device__ void store(uint8_t* __restrict__ lds) const {
        const int tid = threadIdx.x;
        #pragma unroll
        for (int i = 0; i < kPerThread; ++i) {
            const int c = COALESCED_STAGING ? tid + i * THREADS
                                             : tid * kPerThread + i;
            uint8_t* dst = lds + (c / kChunksPerRow) * kStride + (c % kChunksPerRow) * 16;
            // 8-byte stores: kLdsPad breaks 16-byte LDS alignment.
            *reinterpret_cast<uint2*>(dst) = make_uint2(regs[i].x, regs[i].y);
            *reinterpret_cast<uint2*>(dst + 8) = make_uint2(regs[i].z, regs[i].w);
        }
    }
};

// Epi is a functor: float operator()(int row, int col, float acc) const.
template <typename Mma, typename Epi, typename OutT,
          int BM, int BN, int BKB, int WARPS_M, int WARPS_N, int TM, int TN,
          bool COALESCED_STAGING = false, bool TILED_A = false,
          bool VOPD_CROSS_E = false, bool ASSUME_NK_FULL = false,
          bool TILED_B = false>
__global__ __launch_bounds__(WARPS_M* WARPS_N* kWave) void gemm_wmma_kernel(
    const uint8_t* __restrict__ A, const uint8_t* __restrict__ B, OutT* __restrict__ C,
    int M, int N, int kbytes, int ldc, Epi epi) {

    constexpr int kThreads = WARPS_M * WARPS_N * kWave;
    constexpr int kStride = BKB + kLdsPad;
    constexpr int kStepBytes = Mma::kStepBytes;
    constexpr int kSteps = BKB / kStepBytes;

    // The fragment reads below index As by wm * (TM * 16) + i * 16 + row, which
    // reaches WARPS_M * TM * 16 - 1, and Bs likewise. The warp grid has to tile the
    // block exactly: a smaller product leaves part of the tile unread, a larger one
    // walks off the end of the LDS array.
    static_assert(BM == WARPS_M * TM * 16, "the M warp grid must tile BM exactly");
    static_assert(BN == WARPS_N * TN * 16, "the N warp grid must tile BN exactly");
    // A partial K-step would read past the tile's bytes in LDS.
    static_assert(BKB % kStepBytes == 0, "BKB must be a whole number of MMA K-steps");

    // Byte arrays, but every access is a uint2/v2i/v4i reinterpret at a multiple
    // of 8 from the base, so the base itself has to be at least 8-byte aligned.
    // uint8_t alone only promises 1.
    __shared__ __align__(16) uint8_t As[BM * kStride];
    __shared__ __align__(16) uint8_t Bs[BN * kStride];

    const int tid = threadIdx.x;
    const int lane = tid % kWave;
    const int warp = tid / kWave;
    const int wm = warp / WARPS_N;
    const int wn = warp % WARPS_N;

    // Grouped block ordering for L2 locality: consecutive blocks advance along M
    // within a group of kGroupM block-rows, so concurrently resident blocks share
    // the same B columns.
    constexpr int kGroupM = 4;
    const int blocks_n = gridDim.x;
    const int blocks_m = gridDim.y;
    const int bid = blockIdx.y * blocks_n + blockIdx.x;
    const int per_group = kGroupM * blocks_n;
    const int group = bid / per_group;
    const int idx_in_group = bid - group * per_group;
    const int group_rows = min(kGroupM, blocks_m - group * kGroupM);
    const int bm = group * kGroupM + idx_in_group % group_rows;
    const int bn = idx_in_group / group_rows;

    const int m0 = bm * BM;
    const int n0 = bn * BN;

    typename Mma::Acc acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = Mma::zero();

    const int row = frag_row(lane);

    TileStager<BM, BKB, kThreads, COALESCED_STAGING> sa;
    TileStager<BN, BKB, kThreads, COALESCED_STAGING> sb;

    if constexpr (TILED_A) {
        const int64_t tile = static_cast<int64_t>(bm) * (kbytes / BKB);
        sa.load_contiguous(A + tile * BM * BKB);
    } else {
        sa.template load<false, ASSUME_NK_FULL>(A, m0, M, 0, kbytes);
    }
    if constexpr (TILED_B) {
        const int64_t tile = static_cast<int64_t>(bn) * (kbytes / BKB);
        sb.load_contiguous(B + tile * BN * BKB);
    } else {
        sb.template load<ASSUME_NK_FULL, ASSUME_NK_FULL>(
            B, n0, N, 0, kbytes);
    }
    sa.store(As);
    sb.store(Bs);
    __syncthreads();

    for (int kb0 = 0; kb0 < kbytes; kb0 += BKB) {
        const int knext = kb0 + BKB;
        const bool has_next = knext < kbytes;

        // Prefetch the next tile's global reads ahead of the current tile's math.
        if (has_next) {
            if constexpr (TILED_A) {
                const int64_t tile =
                    static_cast<int64_t>(bm) * (kbytes / BKB) + knext / BKB;
                sa.load_contiguous(A + tile * BM * BKB);
            } else {
                sa.template load<false, ASSUME_NK_FULL>(
                    A, m0, M, knext, kbytes);
            }
            if constexpr (TILED_B) {
                const int64_t tile =
                    static_cast<int64_t>(bn) * (kbytes / BKB)
                    + knext / BKB;
                sb.load_contiguous(B + tile * BN * BKB);
            } else {
                sb.template load<ASSUME_NK_FULL, ASSUME_NK_FULL>(
                    B, n0, N, knext, kbytes);
            }
        }

        // Register-level pipeline over K-steps: the LDS reads for step kk+1 are
        // issued before the MMAs of step kk.
        typename Mma::Frag af[2][TM];
        typename Mma::Frag bf[2][TN];

        #pragma unroll
        for (int i = 0; i < TM; ++i)
            af[0][i] = Mma::load(As, wm * (TM * 16) + i * 16 + row, 0, kStride, lane);
        #pragma unroll
        for (int j = 0; j < TN; ++j)
            bf[0][j] = Mma::load(Bs, wn * (TN * 16) + j * 16 + row, 0, kStride, lane);

        #pragma unroll
        for (int kk = 0; kk < kSteps; ++kk) {
            const int cur = kk & 1;
            const int nxt = cur ^ 1;

            if (kk + 1 < kSteps) {
                const int kbyte = (kk + 1) * kStepBytes;
                #pragma unroll
                for (int i = 0; i < TM; ++i)
                    af[nxt][i] =
                        Mma::load(As, wm * (TM * 16) + i * 16 + row, kbyte, kStride, lane);
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    bf[nxt][j] =
                        Mma::load(Bs, wn * (TN * 16) + j * 16 + row, kbyte, kStride, lane);
            }

            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] = Mma::mma(af[cur][i], bf[cur][j], acc[i][j]);
        }

        if (has_next) {
            __syncthreads();  // all warps have finished reading the current tile
            sa.store(As);
            sb.store(Bs);
            __syncthreads();
        }
    }

    epi.init();

    // Row-major writeback: the TN column tiles of one accumulator row cover
    // TN*16 consecutive columns, keeping the stores of an iteration contiguous.
    const int col_lane = acc_col(lane);
    if constexpr (VOPD_CROSS_E) {
        // Adjacent accumulator elements occupy different gfx12 VGPR banks.
        // Keep two independent row epilogues adjacent in the instruction graph
        // so the backend can form legal VOPD mul/FMA pairs. Each expression and
        // output location retain the control path's exact order.
        #pragma clang fp reassociate(off)
        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            #pragma unroll
            for (int e = 0; e < 8; e += 2) {
                const int r0 =
                    m0 + wm * (TM * 16) + i * 16 + acc_row(lane, e);
                const int r1 =
                    m0 + wm * (TM * 16) + i * 16 + acc_row(lane, e + 1);
                const bool valid0 = r0 < M;
                const bool valid1 = r1 < M;
                if (!valid0 && !valid1) continue;
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    const int col =
                        n0 + wn * (TN * 16) + j * 16 + col_lane;
                    if constexpr (!ASSUME_NK_FULL) {
                        if (col >= N) continue;
                    }
                    if (valid0) {
                        OutT* const crow0 =
                            C + static_cast<int64_t>(r0) * ldc;
                        crow0[col] = static_cast<OutT>(
                            epi(r0, col, Mma::get(acc[i][j], e)));
                    }
                    if (valid1) {
                        OutT* const crow1 =
                            C + static_cast<int64_t>(r1) * ldc;
                        crow1[col] = static_cast<OutT>(
                            epi(r1, col, Mma::get(acc[i][j], e + 1)));
                    }
                }
            }
        }
    } else {
        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int r =
                    m0 + wm * (TM * 16) + i * 16 + acc_row(lane, e);
                if (r >= M) continue;
                OutT* crow = C + static_cast<int64_t>(r) * ldc;
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    const int col =
                        n0 + wn * (TN * 16) + j * 16 + col_lane;
                    if constexpr (!ASSUME_NK_FULL) {
                        if (col >= N) continue;
                    }
                    crow[col] = static_cast<OutT>(
                        epi(r, col, Mma::get(acc[i][j], e)));
                }
            }
        }
    }
}

// Two equal-width projections over the same activation tile. Half of the
// workgroup computes each projection, so each thread retains the same single
// accumulator set as gemm_wmma_kernel while both halves share every global/LDS
// load of A. The two outputs remain independent and keep the single-GEMM MMA and
// epilogue order exactly.
template <typename Mma, typename Epi, typename OutT,
          int BM, int BN, int BKB, int WARPS_M, int WARPS_N, int TM, int TN,
          bool TILED_B = false>
__global__ __launch_bounds__(2 * WARPS_M * WARPS_N * kWave)
void gemm_wmma_pair_kernel(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ B0,
    const uint8_t* __restrict__ B1,
    OutT* __restrict__ C0,
    OutT* __restrict__ C1,
    int M, int N, int kbytes, int ldc, Epi epi0, Epi epi1) {

    constexpr int kProjectionWarps = WARPS_M * WARPS_N;
    constexpr int kThreads = 2 * kProjectionWarps * kWave;
    constexpr int kStride = BKB + kLdsPad;
    constexpr int kStepBytes = Mma::kStepBytes;
    constexpr int kSteps = BKB / kStepBytes;

    static_assert(BM == WARPS_M * TM * 16,
                  "the M warp grid must tile BM exactly");
    static_assert(BN == WARPS_N * TN * 16,
                  "the N warp grid must tile BN exactly");
    static_assert(BKB % kStepBytes == 0,
                  "BKB must be a whole number of MMA K-steps");

    __shared__ __align__(16) uint8_t As[BM * kStride];
    __shared__ __align__(16) uint8_t Bs[2][BN * kStride];

    const int tid = threadIdx.x;
    const int lane = tid % kWave;
    const int warp = tid / kWave;
    const int projection = warp / kProjectionWarps;
    const int projection_warp = warp - projection * kProjectionWarps;
    const int wm = projection_warp / WARPS_N;
    const int wn = projection_warp % WARPS_N;

    constexpr int kGroupM = 4;
    const int blocks_n = gridDim.x;
    const int blocks_m = gridDim.y;
    const int bid = blockIdx.y * blocks_n + blockIdx.x;
    const int per_group = kGroupM * blocks_n;
    const int group = bid / per_group;
    const int idx_in_group = bid - group * per_group;
    const int group_rows = min(kGroupM, blocks_m - group * kGroupM);
    const int bm = group * kGroupM + idx_in_group % group_rows;
    const int bn = idx_in_group / group_rows;

    const int m0 = bm * BM;
    const int n0 = bn * BN;

    typename Mma::Acc acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = Mma::zero();

    const int row = frag_row(lane);
    TileStager<BM, BKB, kThreads> sa;
    TileStager<BN, BKB, kThreads> sb0;
    TileStager<BN, BKB, kThreads> sb1;

    sa.load(A, m0, M, 0, kbytes);
    if constexpr (TILED_B) {
        const int64_t tile = static_cast<int64_t>(bn) * (kbytes / BKB);
        sb0.load_contiguous(B0 + tile * BN * BKB);
        sb1.load_contiguous(B1 + tile * BN * BKB);
    } else {
        sb0.load(B0, n0, N, 0, kbytes);
        sb1.load(B1, n0, N, 0, kbytes);
    }
    sa.store(As);
    sb0.store(Bs[0]);
    sb1.store(Bs[1]);
    __syncthreads();

    for (int kb0 = 0; kb0 < kbytes; kb0 += BKB) {
        const int knext = kb0 + BKB;
        const bool has_next = knext < kbytes;
        if (has_next) {
            sa.load(A, m0, M, knext, kbytes);
            if constexpr (TILED_B) {
                const int64_t tile =
                    static_cast<int64_t>(bn) * (kbytes / BKB)
                    + knext / BKB;
                sb0.load_contiguous(B0 + tile * BN * BKB);
                sb1.load_contiguous(B1 + tile * BN * BKB);
            } else {
                sb0.load(B0, n0, N, knext, kbytes);
                sb1.load(B1, n0, N, knext, kbytes);
            }
        }

        typename Mma::Frag af[2][TM];
        typename Mma::Frag bf[2][TN];
        #pragma unroll
        for (int i = 0; i < TM; ++i)
            af[0][i] = Mma::load(
                As, wm * (TM * 16) + i * 16 + row, 0, kStride, lane);
        #pragma unroll
        for (int j = 0; j < TN; ++j)
            bf[0][j] = Mma::load(
                Bs[projection], wn * (TN * 16) + j * 16 + row,
                0, kStride, lane);

        #pragma unroll
        for (int kk = 0; kk < kSteps; ++kk) {
            const int cur = kk & 1;
            const int nxt = cur ^ 1;
            if (kk + 1 < kSteps) {
                const int kbyte = (kk + 1) * kStepBytes;
                #pragma unroll
                for (int i = 0; i < TM; ++i)
                    af[nxt][i] = Mma::load(
                        As, wm * (TM * 16) + i * 16 + row,
                        kbyte, kStride, lane);
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    bf[nxt][j] = Mma::load(
                        Bs[projection], wn * (TN * 16) + j * 16 + row,
                        kbyte, kStride, lane);
            }

            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] = Mma::mma(
                        af[cur][i], bf[cur][j], acc[i][j]);
        }

        if (has_next) {
            __syncthreads();
            sa.store(As);
            sb0.store(Bs[0]);
            sb1.store(Bs[1]);
            __syncthreads();
        }
    }

    constexpr bool kCacheRowwiseEpilogue =
        std::is_same_v<Epi, EpiRowwiseNoBias> ||
        std::is_same_v<Epi, EpiRowwise>;
    float* scale_a_lds = nullptr;
    float* scale_b_lds = nullptr;
    float* bias_lds = nullptr;
    if constexpr (kCacheRowwiseEpilogue) {
        // The matrix loop is finished, so its A/B tiles can hold the much
        // smaller epilogue operands. Synchronize before overwriting them, then
        // load each unique row/channel scale and optional bias once per
        // workgroup instead of once per unrolled accumulator element.
        __syncthreads();
        scale_a_lds = reinterpret_cast<float*>(As);
        scale_b_lds = reinterpret_cast<float*>(&Bs[0][0]);
        bias_lds = scale_b_lds + 2 * BN;
        if (tid < BM) {
            const int r = m0 + tid;
            scale_a_lds[tid] = r < M ? epi0.scale_a[r] : 0.0f;
        }
        if (tid < BN) {
            const int col = n0 + tid;
            if (col < N) {
                if constexpr (std::is_same_v<Epi, EpiRowwiseNoBias>) {
                    scale_b_lds[tid] = epi0.scale_b[col];
                } else {
                    scale_b_lds[tid] =
                        epi0.scale_b[col * epi0.scale_b_stride];
                    bias_lds[tid] = epi0.bias
                        ? load_scalar(epi0.bias, epi0.bias_code, col)
                        : 0.0f;
                }
            } else {
                scale_b_lds[tid] = 0.0f;
                if constexpr (std::is_same_v<Epi, EpiRowwise>) {
                    bias_lds[tid] = 0.0f;
                }
            }
        } else if (tid < 2 * BN) {
            const int local_col = tid - BN;
            const int col = n0 + local_col;
            if (col < N) {
                if constexpr (std::is_same_v<Epi, EpiRowwiseNoBias>) {
                    scale_b_lds[BN + local_col] = epi1.scale_b[col];
                } else {
                    scale_b_lds[BN + local_col] =
                        epi1.scale_b[col * epi1.scale_b_stride];
                    bias_lds[BN + local_col] = epi1.bias
                        ? load_scalar(epi1.bias, epi1.bias_code, col)
                        : 0.0f;
                }
            } else {
                scale_b_lds[BN + local_col] = 0.0f;
                if constexpr (std::is_same_v<Epi, EpiRowwise>) {
                    bias_lds[BN + local_col] = 0.0f;
                }
            }
        }
        __syncthreads();
    }

    Epi epi = projection == 0 ? epi0 : epi1;
    OutT* C = projection == 0 ? C0 : C1;
    epi.init();
    const int col_lane = acc_col(lane);
    if constexpr (kCacheRowwiseEpilogue) {
        // The accepted pair is row-major at writeback. Load its activation
        // scale once for the four adjacent column fragments instead of letting
        // the generic functor reload it for every fully unrolled output.
        #pragma clang fp reassociate(off)
        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int r =
                    m0 + wm * (TM * 16) + i * 16 + acc_row(lane, e);
                if (r >= M) continue;
                OutT* crow = C + static_cast<int64_t>(r) * ldc;
                const int local_row =
                    wm * (TM * 16) + i * 16 + acc_row(lane, e);
                const float row_scale = scale_a_lds[local_row];
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    const int col =
                        n0 + wn * (TN * 16) + j * 16 + col_lane;
                    if (col >= N) continue;
                    const int local_col =
                        wn * (TN * 16) + j * 16 + col_lane;
                    float value = Mma::get(acc[i][j], e) * row_scale;
                    value *= scale_b_lds[projection * BN + local_col];
                    if constexpr (std::is_same_v<Epi, EpiRowwise>) {
                        if (epi.bias) {
                            value +=
                                bias_lds[projection * BN + local_col];
                        }
                    }
                    crow[col] = static_cast<OutT>(value);
                }
            }
        }
    } else {
        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int r =
                    m0 + wm * (TM * 16) + i * 16 + acc_row(lane, e);
                if (r >= M) continue;
                OutT* crow = C + static_cast<int64_t>(r) * ldc;
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    const int col =
                        n0 + wn * (TN * 16) + j * 16 + col_lane;
                    if (col >= N) continue;
                    crow[col] = static_cast<OutT>(
                        epi(r, col, Mma::get(acc[i][j], e)));
                }
            }
        }
    }
}

// Two adjacent M tiles over one weight tile. Half of the workgroup computes
// each M tile, so every thread keeps the same accumulator shape and arithmetic
// order as gemm_wmma_kernel while both halves share each global/LDS B stage.
// gemm_wmma_pair_kernel shares A across two projections; this kernel instead
// shares B across two row tiles.
template <typename Mma, typename Epi, typename OutT,
          int BM, int BN, int BKB, int WARPS_M, int WARPS_N, int TM, int TN,
          bool VOPD_CROSS_E = false, bool TILED_B = false>
__global__ __launch_bounds__(2 * WARPS_M * WARPS_N * kWave)
void gemm_wmma_dual_m_kernel(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ B,
    OutT* __restrict__ C,
    int M, int N, int kbytes, int ldc, Epi epi) {

    constexpr int kTileWarps = WARPS_M * WARPS_N;
    constexpr int kThreads = 2 * kTileWarps * kWave;
    constexpr int kStride = BKB + kLdsPad;
    constexpr int kStepBytes = Mma::kStepBytes;
    constexpr int kSteps = BKB / kStepBytes;

    static_assert(BM == WARPS_M * TM * 16,
                  "the M warp grid must tile BM exactly");
    static_assert(BN == WARPS_N * TN * 16,
                  "the N warp grid must tile BN exactly");
    static_assert(BKB % kStepBytes == 0,
                  "BKB must be a whole number of MMA K-steps");

    __shared__ __align__(16) uint8_t As[2][BM * kStride];
    __shared__ __align__(16) uint8_t Bs[BN * kStride];

    const int tid = threadIdx.x;
    const int lane = tid % kWave;
    const int warp = tid / kWave;
    const int mtile = warp / kTileWarps;
    const int tile_warp = warp - mtile * kTileWarps;
    const int wm = tile_warp / WARPS_N;
    const int wn = tile_warp % WARPS_N;

    // Each block covers two adjacent M tiles. Group two such blocks so the
    // traversal retains the production kernel's four-M-tile B locality.
    constexpr int kGroupM = 2;
    const int blocks_n = gridDim.x;
    const int blocks_m = gridDim.y;
    const int bid = blockIdx.y * blocks_n + blockIdx.x;
    const int per_group = kGroupM * blocks_n;
    const int group = bid / per_group;
    const int idx_in_group = bid - group * per_group;
    const int group_rows = min(kGroupM, blocks_m - group * kGroupM);
    const int bm_pair = group * kGroupM + idx_in_group % group_rows;
    const int bn = idx_in_group / group_rows;

    const int m0_pair = bm_pair * (2 * BM);
    const int m0 = m0_pair + mtile * BM;
    const int n0 = bn * BN;

    typename Mma::Acc acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = Mma::zero();

    const int row = frag_row(lane);
    TileStager<BM, BKB, kThreads> sa0;
    TileStager<BM, BKB, kThreads> sa1;
    TileStager<BN, BKB, kThreads> sb;

    sa0.load(A, m0_pair, M, 0, kbytes);
    sa1.load(A, m0_pair + BM, M, 0, kbytes);
    if constexpr (TILED_B) {
        const int64_t tile = static_cast<int64_t>(bn) * (kbytes / BKB);
        sb.load_contiguous(B + tile * BN * BKB);
    } else {
        sb.load(B, n0, N, 0, kbytes);
    }
    sa0.store(As[0]);
    sa1.store(As[1]);
    sb.store(Bs);
    __syncthreads();

    for (int kb0 = 0; kb0 < kbytes; kb0 += BKB) {
        const int knext = kb0 + BKB;
        const bool has_next = knext < kbytes;
        if (has_next) {
            sa0.load(A, m0_pair, M, knext, kbytes);
            sa1.load(A, m0_pair + BM, M, knext, kbytes);
            if constexpr (TILED_B) {
                const int64_t tile =
                    static_cast<int64_t>(bn) * (kbytes / BKB)
                    + knext / BKB;
                sb.load_contiguous(B + tile * BN * BKB);
            } else {
                sb.load(B, n0, N, knext, kbytes);
            }
        }

        typename Mma::Frag af[2][TM];
        typename Mma::Frag bf[2][TN];
        #pragma unroll
        for (int i = 0; i < TM; ++i)
            af[0][i] = Mma::load(
                As[mtile], wm * (TM * 16) + i * 16 + row,
                0, kStride, lane);
        #pragma unroll
        for (int j = 0; j < TN; ++j)
            bf[0][j] = Mma::load(
                Bs, wn * (TN * 16) + j * 16 + row,
                0, kStride, lane);

        #pragma unroll
        for (int kk = 0; kk < kSteps; ++kk) {
            const int cur = kk & 1;
            const int nxt = cur ^ 1;
            if (kk + 1 < kSteps) {
                const int kbyte = (kk + 1) * kStepBytes;
                #pragma unroll
                for (int i = 0; i < TM; ++i)
                    af[nxt][i] = Mma::load(
                        As[mtile], wm * (TM * 16) + i * 16 + row,
                        kbyte, kStride, lane);
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    bf[nxt][j] = Mma::load(
                        Bs, wn * (TN * 16) + j * 16 + row,
                        kbyte, kStride, lane);
            }

            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] = Mma::mma(
                        af[cur][i], bf[cur][j], acc[i][j]);
        }

        if (has_next) {
            __syncthreads();
            sa0.store(As[0]);
            sa1.store(As[1]);
            sb.store(Bs);
            __syncthreads();
        }
    }

    epi.init();
    const int col_lane = acc_col(lane);
    if constexpr (std::is_same_v<Epi, EpiRowwiseNoBias>) {
        #pragma clang fp reassociate(off)
        // Cache the four channel scales owned by this lane directly in VGPRs.
        // This avoids both the generic functor's fully-unrolled reloads and an
        // LDS round trip/barrier after the matrix loop.
        float col_scale[TN];
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int col = n0 + wn * (TN * 16) + j * 16 + col_lane;
            col_scale[j] = col < N ? epi.scale_b[col] : 0.0f;
        }
        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int r =
                    m0 + wm * (TM * 16) + i * 16 + acc_row(lane, e);
                if (r >= M) continue;
                OutT* const crow = C + static_cast<int64_t>(r) * ldc;
                const float row_scale = epi.scale_a[r];
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    const int col =
                        n0 + wn * (TN * 16) + j * 16 + col_lane;
                    if (col >= N) continue;
                    float value = Mma::get(acc[i][j], e) * row_scale;
                    value *= col_scale[j];
                    crow[col] = static_cast<OutT>(value);
                }
            }
        }
    } else if constexpr (std::is_same_v<Epi, EpiRowwise>) {
        #pragma clang fp contract(off)
        #pragma clang fp reassociate(off)
        // Bias-bearing checkpoints otherwise reload the same channel scale and
        // bias for every accumulator element. Each lane owns TN columns, so
        // retain those operands across all of its output rows just as the
        // bias-free specialization does.
        float col_scale[TN];
        float col_bias[TN];
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int col = n0 + wn * (TN * 16) + j * 16 + col_lane;
            if (col < N) {
                col_scale[j] = epi.scale_b[col * epi.scale_b_stride];
                col_bias[j] = epi.bias
                    ? load_scalar(epi.bias, epi.bias_code, col) : 0.0f;
            } else {
                col_scale[j] = 0.0f;
                col_bias[j] = 0.0f;
            }
        }
        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int r =
                    m0 + wm * (TM * 16) + i * 16 + acc_row(lane, e);
                if (r >= M) continue;
                OutT* const crow = C + static_cast<int64_t>(r) * ldc;
                const float row_scale = epi.scale_a[r];
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    const int col =
                        n0 + wn * (TN * 16) + j * 16 + col_lane;
                    if (col >= N) continue;
                    float value = Mma::get(acc[i][j], e) * row_scale;
                    value *= col_scale[j];
                    value += col_bias[j];
                    crow[col] = static_cast<OutT>(value);
                }
            }
        }
    } else if constexpr (std::is_same_v<Epi, EpiRowwiseGatedResidual>) {
        #pragma clang fp contract(off)
        #pragma clang fp reassociate(off)
        // This is the same cached rowwise linear epilogue above, followed by
        // the visible BF16 materialization and addcmul arithmetic.  Gate is a
        // row-broadcast vector, so each lane retains its TN values while the
        // residual is read once at the final output location.
        float col_scale[TN];
        float col_bias[TN];
        float col_gate[TN];
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int col = n0 + wn * (TN * 16) + j * 16 + col_lane;
            if (col < N) {
                col_scale[j] = epi.scale_b[col * epi.scale_b_stride];
                col_bias[j] = epi.bias
                    ? load_scalar(epi.bias, epi.bias_code, col) : 0.0f;
                col_gate[j] = static_cast<float>(epi.gate[col]);
            } else {
                col_scale[j] = 0.0f;
                col_bias[j] = 0.0f;
                col_gate[j] = 0.0f;
            }
        }
        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int r =
                    m0 + wm * (TM * 16) + i * 16 + acc_row(lane, e);
                if (r >= M) continue;
                OutT* const crow = C + static_cast<int64_t>(r) * ldc;
                const __bf16* const residual_row =
                    epi.residual + static_cast<int64_t>(r) * epi.residual_stride;
                const float row_scale = epi.scale_a[r];
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    const int col =
                        n0 + wn * (TN * 16) + j * 16 + col_lane;
                    if (col >= N) continue;
                    float linear = Mma::get(acc[i][j], e) * row_scale;
                    linear *= col_scale[j];
                    linear += col_bias[j];
                    const __bf16 rounded = static_cast<__bf16>(linear);
                    const float value = fmaf(
                        col_gate[j], static_cast<float>(rounded),
                        static_cast<float>(residual_row[col]));
                    crow[col] = static_cast<OutT>(value);
                }
            }
        }
    } else if constexpr (VOPD_CROSS_E) {
        #pragma clang fp reassociate(off)
        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            #pragma unroll
            for (int e = 0; e < 8; e += 2) {
                const int r0 =
                    m0 + wm * (TM * 16) + i * 16 + acc_row(lane, e);
                const int r1 =
                    m0 + wm * (TM * 16) + i * 16 + acc_row(lane, e + 1);
                const bool valid0 = r0 < M;
                const bool valid1 = r1 < M;
                if (!valid0 && !valid1) continue;
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    const int col =
                        n0 + wn * (TN * 16) + j * 16 + col_lane;
                    if (col >= N) continue;
                    if (valid0) {
                        OutT* const crow0 =
                            C + static_cast<int64_t>(r0) * ldc;
                        crow0[col] = static_cast<OutT>(
                            epi(r0, col, Mma::get(acc[i][j], e)));
                    }
                    if (valid1) {
                        OutT* const crow1 =
                            C + static_cast<int64_t>(r1) * ldc;
                        crow1[col] = static_cast<OutT>(
                            epi(r1, col, Mma::get(acc[i][j], e + 1)));
                    }
                }
            }
        }
    } else {
        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int r =
                    m0 + wm * (TM * 16) + i * 16 + acc_row(lane, e);
                if (r >= M) continue;
                OutT* crow = C + static_cast<int64_t>(r) * ldc;
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    const int col =
                        n0 + wn * (TN * 16) + j * 16 + col_lane;
                    if (col >= N) continue;
                    crow[col] = static_cast<OutT>(
                        epi(r, col, Mma::get(acc[i][j], e)));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tile selection, shared by the fp8 and int8 launchers.
// ---------------------------------------------------------------------------

// hipDeviceAttributeMultiprocessorCount reports WGPs on RDNA, not CUs (32 on a
// 64-CU gfx1201), and a workgroup schedules onto a WGP, so WGPs are the unit the
// grid-coverage test needs. Cached per ordinal to keep the query off the launch
// path; the fallback only mis-sizes that test, never a result.
inline int device_wgp_count() {
    constexpr int kMaxDevices = 16;
    // A GEMM can be launched from several host threads at once. Racing threads
    // write the same value and nothing is published through the cache, so relaxed.
    static std::atomic<int> cache[kMaxDevices] = {};
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= kMaxDevices) return 16;
    int n = cache[dev].load(std::memory_order_relaxed);
    if (n == 0) {
        if (hipDeviceGetAttribute(&n, hipDeviceAttributeMultiprocessorCount, dev) != hipSuccess ||
            n <= 0) {
            n = 16;
        }
        cache[dev].store(n, std::memory_order_relaxed);
    }
    return n;
}

// Pick and launch a tile for C[M, N] = A[M, K] @ B[N, K]^T, selecting on grid
// coverage, K depth and warp grid. 128x128 has the best arithmetic intensity but
// wastes the device when it yields fewer blocks than there are WGPs; BKB=128
// halves the LDS round trips per K element once K amortizes the coarser tail.
// The thresholds are tuned on RDNA4 and govern tile choice only, never
// correctness. kbytes is bytes of K, equal to K only for the 8-bit policies, so
// an int4 caller passing K/2 fires at twice the K these read as.
template <typename Mma, typename Epi, typename OutT>
void launch_gemm_wmma(const uint8_t* A, const uint8_t* B, OutT* C, int M, int N, int kbytes,
                      int ldc, Epi epi, hipStream_t stream) {
    const int blocks_128 = ((M + 127) / 128) * ((N + 127) / 128);

    const int wgps = device_wgp_count();

    // Zero padding the block count cannot see: at M <= 64 the 128-row tile is at
    // least half empty, and the finer 64x64 grid recovers the wasted MMAs.
    const bool skinny = (M <= 64 || N <= 64);

    if (!skinny && blocks_128 >= wgps) {
        if (kbytes >= 4096) {
            constexpr int BM = 128, BN = 128, BKB = 128;
            dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
            // With few blocks per WGP there is nothing to interleave across, so
            // the 16-wave grid hides latency within a block instead.
            if (blocks_128 <= 4 * wgps) {
                gemm_wmma_kernel<Mma, Epi, OutT, BM, BN, BKB, 4, 4, 2, 2>
                    <<<grid, 512, 0, stream>>>(A, B, C, M, N, kbytes, ldc, epi);
            } else {
                gemm_wmma_kernel<Mma, Epi, OutT, BM, BN, BKB, 4, 2, 2, 4>
                    <<<grid, 256, 0, stream>>>(A, B, C, M, N, kbytes, ldc, epi);
            }
        } else {
            constexpr int BM = 128, BN = 128, BKB = 64;
            dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
            gemm_wmma_kernel<Mma, Epi, OutT, BM, BN, BKB, 4, 2, 2, 4>
                <<<grid, 256, 0, stream>>>(A, B, C, M, N, kbytes, ldc, epi);
        }
    } else if (kbytes >= 2048) {
        constexpr int BM = 64, BN = 64, BKB = 128;
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        gemm_wmma_kernel<Mma, Epi, OutT, BM, BN, BKB, 2, 2, 2, 2>
            <<<grid, 128, 0, stream>>>(A, B, C, M, N, kbytes, ldc, epi);
    } else {
        constexpr int BM = 64, BN = 64, BKB = 64;
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        gemm_wmma_kernel<Mma, Epi, OutT, BM, BN, BKB, 2, 2, 2, 2>
            <<<grid, 128, 0, stream>>>(A, B, C, M, N, kbytes, ldc, epi);
    }
}

}  // namespace comfy::hip_backend
