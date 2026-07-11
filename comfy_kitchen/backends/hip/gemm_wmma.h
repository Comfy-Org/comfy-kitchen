// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Tiled WMMA GEMM core for gfx12, shared by the fp8, int8 and int4 paths.
//
// Computes C[M, N] = epilogue(A[M, K] @ B[N, K]^T). The B operand is the weight
// in its natural (N, K) row-major form, matching torch linear.
//
// All supported element types hold 8 bytes per lane per WMMA, so one K-step is
// 16 bytes of a row whether those bytes hold 16 fp8/int8 values or 32 int4
// nibbles. The tile loop is byte-addressed and the element type appears only in
// the Mma policy.
//
// The K loop is software-pipelined: the next tile's global loads are issued into
// registers before the current tile's math.
#pragma once

#include "wmma_gfx12.h"

namespace comfy::hip_backend {

// LDS row padding, in bytes. 8 spreads the 16 lanes of a fragment read across
// distinct banks. 16 would preserve 128-bit LDS access but aliases rows
// 0/4/8/... onto the same bank.
constexpr int kLdsPad = 8;

struct MmaFp8 {
    using Acc = v8f;
    static __forceinline__ __device__ Acc zero() { return Acc{0, 0, 0, 0, 0, 0, 0, 0}; }
    static __forceinline__ __device__ Acc mma(v2i a, v2i b, Acc c) { return wmma_fp8(a, b, c); }
    static __forceinline__ __device__ float get(Acc c, int e) { return c[e]; }
};

struct MmaBf8 {
    using Acc = v8f;
    static __forceinline__ __device__ Acc zero() { return Acc{0, 0, 0, 0, 0, 0, 0, 0}; }
    static __forceinline__ __device__ Acc mma(v2i a, v2i b, Acc c) { return wmma_bf8(a, b, c); }
    static __forceinline__ __device__ float get(Acc c, int e) { return c[e]; }
};

struct MmaInt8 {
    using Acc = v8i;
    static __forceinline__ __device__ Acc zero() { return Acc{0, 0, 0, 0, 0, 0, 0, 0}; }
    static __forceinline__ __device__ Acc mma(v2i a, v2i b, Acc c) { return wmma_iu8(a, b, c); }
    static __forceinline__ __device__ float get(Acc c, int e) { return static_cast<float>(c[e]); }
};

struct MmaInt4 {
    using Acc = v8i;
    static __forceinline__ __device__ Acc zero() { return Acc{0, 0, 0, 0, 0, 0, 0, 0}; }
    static __forceinline__ __device__ Acc mma(v2i a, v2i b, Acc c) { return wmma_iu4(a, b, c); }
    static __forceinline__ __device__ float get(Acc c, int e) { return static_cast<float>(c[e]); }
};

// Staging for one ROWS x BKB byte tile. load() issues only the global reads, so
// the caller can place the tile's math between load() and store().
//
// kbytes is the source row length in bytes (K for 8-bit types, K/2 for int4) and
// is a multiple of 16, so a 16-byte chunk starting inside a row also ends inside
// it. Out-of-range rows and the K tail are zero-filled.
template <int ROWS, int BKB, int THREADS>
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

    __forceinline__ __device__ void load(const uint8_t* __restrict__ src, int row0, int rows_total,
                                         int kbyte0, int kbytes) {
        const int tid = threadIdx.x;
        #pragma unroll
        for (int i = 0; i < kPerThread; ++i) {
            const int c = tid * kPerThread + i;
            const int grow = row0 + c / kChunksPerRow;
            const int gk = kbyte0 + (c % kChunksPerRow) * 16;

            regs[i] = (grow < rows_total && gk < kbytes)
                          ? *reinterpret_cast<const uint4*>(
                                src + static_cast<int64_t>(grow) * kbytes + gk)
                          : make_uint4(0, 0, 0, 0);
        }
    }

    __forceinline__ __device__ void store(uint8_t* __restrict__ lds) const {
        const int tid = threadIdx.x;
        #pragma unroll
        for (int i = 0; i < kPerThread; ++i) {
            const int c = tid * kPerThread + i;
            uint8_t* dst = lds + (c / kChunksPerRow) * kStride + (c % kChunksPerRow) * 16;
            // 8-byte stores: kLdsPad breaks 16-byte LDS alignment.
            *reinterpret_cast<uint2*>(dst) = make_uint2(regs[i].x, regs[i].y);
            *reinterpret_cast<uint2*>(dst + 8) = make_uint2(regs[i].z, regs[i].w);
        }
    }
};

// Epi is a functor: float operator()(int row, int col, float acc) const.
template <typename Mma, typename Epi, typename OutT,
          int BM, int BN, int BKB, int WARPS_M, int WARPS_N, int TM, int TN>
__global__ __launch_bounds__(WARPS_M* WARPS_N* kWave) void gemm_wmma_kernel(
    const uint8_t* __restrict__ A, const uint8_t* __restrict__ B, OutT* __restrict__ C,
    int M, int N, int kbytes, Epi epi) {

    constexpr int kThreads = WARPS_M * WARPS_N * kWave;
    constexpr int kStride = BKB + kLdsPad;

    // The fragment reads below index As by wm * (TM * 16) + i * 16 + row, which
    // reaches WARPS_M * TM * 16 - 1, and Bs likewise. The warp grid has to tile the
    // block exactly: a smaller product leaves part of the tile unread, a larger one
    // walks off the end of the LDS array.
    static_assert(BM == WARPS_M * TM * 16, "the M warp grid must tile BM exactly");
    static_assert(BN == WARPS_N * TN * 16, "the N warp grid must tile BN exactly");

    __shared__ uint8_t As[BM * kStride];
    __shared__ uint8_t Bs[BN * kStride];

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
    const int khalf = frag_khalf(lane);

    TileStager<BM, BKB, kThreads> sa;
    TileStager<BN, BKB, kThreads> sb;

    sa.load(A, m0, M, 0, kbytes);
    sb.load(B, n0, N, 0, kbytes);
    sa.store(As);
    sb.store(Bs);
    __syncthreads();

    for (int kb0 = 0; kb0 < kbytes; kb0 += BKB) {
        const int knext = kb0 + BKB;
        const bool has_next = knext < kbytes;

        // Prefetch the next tile's global reads ahead of the current tile's math.
        if (has_next) {
            sa.load(A, m0, M, knext, kbytes);
            sb.load(B, n0, N, knext, kbytes);
        }

        // Register-level pipeline over K-steps: the LDS reads for step kk+1 are
        // issued before the MMAs of step kk.
        v2i af[2][TM];
        v2i bf[2][TN];

        #pragma unroll
        for (int i = 0; i < TM; ++i)
            af[0][i] = load_frag_b64(As, wm * (TM * 16) + i * 16 + row, 8 * khalf, kStride);
        #pragma unroll
        for (int j = 0; j < TN; ++j)
            bf[0][j] = load_frag_b64(Bs, wn * (TN * 16) + j * 16 + row, 8 * khalf, kStride);

        #pragma unroll
        for (int kk = 0; kk < BKB / 16; ++kk) {
            const int cur = kk & 1;
            const int nxt = cur ^ 1;

            if (kk + 1 < BKB / 16) {
                const int kbyte = (kk + 1) * 16 + 8 * khalf;
                #pragma unroll
                for (int i = 0; i < TM; ++i)
                    af[nxt][i] = load_frag_b64(As, wm * (TM * 16) + i * 16 + row, kbyte, kStride);
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    bf[nxt][j] = load_frag_b64(Bs, wn * (TN * 16) + j * 16 + row, kbyte, kStride);
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
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int r = m0 + wm * (TM * 16) + i * 16 + acc_row(lane, e);
            if (r >= M) continue;
            OutT* crow = C + static_cast<int64_t>(r) * N;
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int col = n0 + wn * (TN * 16) + j * 16 + col_lane;
                if (col >= N) continue;
                crow[col] = static_cast<OutT>(epi(r, col, Mma::get(acc[i][j], e)));
            }
        }
    }
}

}  // namespace comfy::hip_backend
