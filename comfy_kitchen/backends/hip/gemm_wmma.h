// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Tiled WMMA GEMM core, shared by the fp8, int8, int4 and fp16 paths and the fp16
// conv3d on both gfx11 and gfx12.
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
#include <cstdlib>  // strtol, for the experiment knob below

#include "launchers.h"  // comfy_small_igpu
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

    // Same, for an operand whose rows are gathered rather than stored contiguously:
    // Src::chunk(row, kbyte) returns those 16 bytes, and neither is out of range.
    template <typename Src>
    __forceinline__ __device__ void load(const Src& src, int row0, int rows_total, int kbyte0,
                                         int kbytes) {
        const int tid = threadIdx.x;
        #pragma unroll
        for (int i = 0; i < kPerThread; ++i) {
            const int c = tid * kPerThread + i;
            const int grow = row0 + c / kChunksPerRow;
            const int gk = kbyte0 + (c % kChunksPerRow) * 16;

            regs[i] = (grow < rows_total && gk < kbytes) ? src.chunk(grow, gk)
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

// A is the raw operand rows, or a TileStager source that gathers them. Only the
// pointer form carries __restrict__.
template <typename T>
struct GemmOperandA {
    using type = T;
};
template <>
struct GemmOperandA<const uint8_t*> {
    using type = const uint8_t* __restrict__;
};

// Epi is a functor: float operator()(int row, int col, float acc) const.
template <typename Mma, typename Epi, typename OutT,
          int BM, int BN, int BKB, int WARPS_M, int WARPS_N, int TM, int TN,
          typename ASrc = const uint8_t*>
__global__ __launch_bounds__(WARPS_M* WARPS_N* kWave) void gemm_wmma_kernel(
    typename GemmOperandA<ASrc>::type A, const uint8_t* __restrict__ B, OutT* __restrict__ C,
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
    //
    // The epilogue's per-row and per-column operands are hoisted into registers
    // first. The stores below go through OutT*, so the compiler has to assume they
    // may alias the scale / bias / residual pointers and re-issues every load in
    // the loop -- and this loop runs TM*8*TN times per lane. The values handed to
    // operator() are exactly the ones it used to load, so the arithmetic is
    // unchanged; see epilogue.h.
    const int col_lane = acc_col(lane);
    float col_scale[TN], col_bias_v[TN];
    #pragma unroll
    for (int j = 0; j < TN; ++j) {
        const int col = n0 + wn * (TN * 16) + j * 16 + col_lane;
        col_scale[j] = (col < N) ? epi.col_scale(col) : 0.0f;
        col_bias_v[j] = (col < N) ? epi.col_bias(col) : 0.0f;
    }
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int r = m0 + wm * (TM * 16) + i * 16 + acc_row(lane, e);
            if (r >= M) continue;
            const float rs = epi.row_scale(r);
            OutT* crow = C + static_cast<int64_t>(r) * ldc;
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int col = n0 + wn * (TN * 16) + j * 16 + col_lane;
                if (col >= N) continue;
                crow[col] = static_cast<OutT>(epi(r, col, Mma::get(acc[i][j], e), rs, col_scale[j],
                                                 col_bias_v[j]));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Experiment knob: force a tile, bypassing the heuristic below.
//
// COMFY_GEMM_WMMA_TILE=BM,BN,BKB,WARPS_M,WARPS_N,TM,TN
//
// This exists only so one process can interleave every candidate against the
// heuristic under the same clocks. A rebuild-per-candidate sweep cannot be
// trusted on this device: the first GEMM after a rebuild runs on cold clocks
// (measured 1.9x), which is larger than the effects being chased. Read on every
// launch, like the ConvRot block overrides. Unset means the tuned heuristic.
struct GemmTileOverride {
    int bm, bn, bkb, wm, wn, tm, tn;
};

inline GemmTileOverride comfy_gemm_wmma_tile_override() {
    const char* s = getenv("COMFY_GEMM_WMMA_TILE");
    if (!s || !*s) return GemmTileOverride{0, 0, 0, 0, 0, 0, 0};
    int v[7] = {0, 0, 0, 0, 0, 0, 0};
    const char* p = s;
    for (int i = 0; i < 7; ++i) {
        char* end = nullptr;
        v[i] = static_cast<int>(strtol(p, &end, 10));
        if (end == p || v[i] <= 0) return GemmTileOverride{0, 0, 0, 0, 0, 0, 0};
        p = (*end == ',') ? end + 1 : end;
        if (*end != ',') {
            if (i != 6) return GemmTileOverride{0, 0, 0, 0, 0, 0, 0};
        }
    }
    return GemmTileOverride{v[0], v[1], v[2], v[3], v[4], v[5], v[6]};
}

// Launch a forced tile. Only the combinations the sweep needs are instantiated;
// **an unlisted tile returns false and the caller silently runs the heuristic**, so a
// sweep that asks for something not instantiated measures the heuristic and reads
// exactly like "that tile does not matter". The instantiations below are therefore
// the list a sweep may use.
//
// The outcome on this part (gfx1103, one process, every candidate interleaved with
// the heuristic, 11 production shapes): the heuristic is 0.99x-1.04x of the best
// forced tile on every one of them, so the three failures to beat it above still
// stand. The one real exception was 128x128x128 on 256 threads (WARPS_M=4,
// WARPS_N=2), which measured 0.77x at K=4096 and 0.94x at K=4096/N=8192 and
// 1.02x-1.04x on every other K -- and **no Anima or SDXL shape has K=4096**, so
// there is nothing here to take. Two further candidates (128x256x128, and the
// 256x128 / 256x256 tiles) showed 0.94x on one shape each and did not reproduce
// across runs: exactly the trap the interleaved A/B exists to catch.
template <typename Mma, typename Epi, typename OutT, typename ASrc = const uint8_t*>
inline bool launch_gemm_wmma_forced(const ASrc& A, const uint8_t* B, OutT* C, int M, int N,
                                    int kbytes, int ldc, Epi epi, hipStream_t stream) {
    const GemmTileOverride o = comfy_gemm_wmma_tile_override();
    if (o.bm == 0) return false;
#define CK_LAUNCH_WMMA_TILE(BM, BN, BKB, WM, WN, TM, TN)                                    \
    if (o.bm == BM && o.bn == BN && o.bkb == BKB && o.wm == WM && o.wn == WN &&            \
        o.tm == TM && o.tn == TN) {                                                         \
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);                                   \
        gemm_wmma_kernel<Mma, Epi, OutT, BM, BN, BKB, WM, WN, TM, TN, ASrc>                 \
            <<<grid, WM * WN * kWave, 0, stream>>>(A, B, C, M, N, kbytes, ldc, epi);       \
        return true;                                                                       \
    }
    // The heuristic's own tiles, so a forced run reproduces it exactly.
    CK_LAUNCH_WMMA_TILE(128, 128, 128, 4, 4, 2, 2)
    CK_LAUNCH_WMMA_TILE(128, 128, 64, 4, 4, 2, 2)
    CK_LAUNCH_WMMA_TILE(64, 64, 128, 2, 2, 2, 2)
    CK_LAUNCH_WMMA_TILE(64, 64, 64, 2, 2, 2, 2)
    CK_LAUNCH_WMMA_TILE(128, 128, 128, 4, 2, 2, 4)
    CK_LAUNCH_WMMA_TILE(128, 128, 64, 4, 2, 2, 4)
    // Wider / narrower tiles and shallow K-steps, the shapes the sweep asks about.
    CK_LAUNCH_WMMA_TILE(256, 128, 128, 4, 4, 4, 2)
    CK_LAUNCH_WMMA_TILE(256, 128, 64, 4, 4, 4, 2)
    CK_LAUNCH_WMMA_TILE(128, 256, 128, 4, 4, 2, 4)
    CK_LAUNCH_WMMA_TILE(128, 256, 64, 4, 4, 2, 4)
    CK_LAUNCH_WMMA_TILE(256, 256, 64, 4, 4, 4, 4)
    CK_LAUNCH_WMMA_TILE(64, 128, 64, 2, 4, 2, 2)
    CK_LAUNCH_WMMA_TILE(128, 64, 64, 4, 2, 2, 2)
    // BKB=32 needs ROWS*2 chunks to cover THREADS, so only the 128-thread tiles fit.
    CK_LAUNCH_WMMA_TILE(64, 64, 32, 2, 2, 2, 2)
    CK_LAUNCH_WMMA_TILE(64, 128, 32, 2, 2, 2, 4)
#undef CK_LAUNCH_WMMA_TILE
    return false;
}

// ---------------------------------------------------------------------------
// Tile selection, shared by the fp8 and int8 launchers.
// ---------------------------------------------------------------------------
//
// Do not retune these numbers without re-running the sweeps in ck_tools/; three
// separate attempts to do better all measured worse on the 6-WGP 780M, and the
// kernel is at the machine's limit rather than short of one. All three used
// ck_tools/ck_tile_ab.py, which A/Bs every candidate against this heuristic in a
// single process with forward *and* reverse rotation (the first candidate in a
// rotation is otherwise systematically slow and invents ~1.4x on shapes that
// change nothing).
//
//   1. Wider tiles. 128x128 has 2*BM*BN/(BM+BN) = 128 ops per byte of global
//      traffic, and the kernel moves that at ~100 GB/s, which accounts for the
//      ~12 TOPS it measures -- so widening the tile looked like the obvious fix.
//      256x128 (170 ops/byte) and 256x256 (256 ops/byte) measured 0.49x-0.92x.
//      TM/TN doubles with the tile, and the accumulators alone reach 64-128 VGPRs,
//      which spills. Arithmetic intensity is bought with the register file here.
//   2. More independent WMMA chains. ck_tools/wmma_peak.hip issues MMAs from a
//      fixed accumulator set with no memory traffic and finds 1/2/4 accumulators
//      all plateau at 7.9 TOPS while 8 reach 31 and 16 reach 62, so TM*TN = 4
//      looked like a stall. Reshaping the warp grid to 8 chains at the same tile
//      (TM=4,TN=2 or TM=2,TN=4) measured 0.88x-0.94x. The microkernel stalls
//      because nothing else is in flight; this kernel has LDS loads and the K
//      loop around each MMA, so there is already enough to cover the latency, and
//      the extra registers only cost occupancy.
//   3. 64x64 and 64x128 tiles are within 1% on every workload shape. The two
//      that beat this heuristic do so by 1.03x-1.09x on shapes worth 0.1-0.25% of
//      a step, and both lose on other shapes in the same family.

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
template <typename Mma, typename Epi, typename OutT, typename ASrc = const uint8_t*>
void launch_gemm_wmma(ASrc A, const uint8_t* B, OutT* C, int M, int N, int kbytes,
                      int ldc, Epi epi, hipStream_t stream) {
    if (launch_gemm_wmma_forced<Mma, Epi, OutT, ASrc>(A, B, C, M, N, kbytes, ldc, epi, stream)) {
        return;
    }
    const int wgps = device_wgp_count();
    const int blocks_128 = ((M + 127) / 128) * ((N + 127) / 128);

    // Zero padding the block count cannot see. At M <= 64 the 128-row tile is at
    // least half empty, and the finer 64x64 grid recovers the wasted MMAs.
    //
    // The padded-area test below is **gfx1103-only**, for the same reason the tile
    // branch underneath it is: 64x64 halves the arithmetic intensity, so whether the
    // padding it saves outweighs that depends on how many workgroups there are to
    // interleave -- and that was measured on 6, not on the 8-96 WGP parts. Keeping it
    // gated means every other architecture sees upstream's `M <= 64 || N <= 64`
    // unchanged, which is the only claim backed by a measurement on all of them.
    //
    // What it catches that `M <= 64` does not: a row that rounds *up* to a second
    // 128-row tile. At M=154 (SDXL's cross-attention context, 2*77 tokens) the 128
    // grid launches two tiles and runs 256 rows' worth of MMAs for 154 rows (1.66x
    // the work), where 64x64 launches three and runs 192 (1.25x). Interleaved A/B of
    // 64x64x64 against the 128x128x64 that was being picked, 20 samples per arm,
    // non-overlapping spreads: 0.806x at M=154/N=2048/K=1280, 0.813x at N=640,
    // 0.760x at K=2560, 0.835x at N=1280.
    //
    // Compared as padded area rather than tested as M <= 64 because 64x64 also has
    // half the arithmetic intensity (64 vs 128 ops/byte), so the narrower grid has to
    // save more than that before it wins -- hence the 115/100. A row just under a tile
    // boundary (M=100) pads to the same area either way and correctly keeps 128x128.
    const int padded64 = ((M + 63) / 64) * 64 * ((N + 63) / 64) * 64;
    const int padded128 = ((M + 127) / 128) * 128 * ((N + 127) / 128) * 128;
    const bool padding_waste =
        static_cast<int64_t>(padded64) * 115 < static_cast<int64_t>(padded128) * 100;
    const bool skinny = (M <= 64 || N <= 64) || (comfy_small_igpu() && padding_waste);


    // The gfx1103-only branch, keyed on the architecture rather than a WGP
    // threshold: low-WGP devices (6 WGPs on a 780M, 8 on a 680M) have too few
    // workgroup processors to interleave many small blocks, so a 16-wave
    // 512-thread block that hides WMMA latency within the block wins. Deeper K
    // amortizes the BKB=128 tile's LDS round trips; shallower K runs faster with
    // BKB=64 on the 512-thread grid (measured on the 6-WGP 780M: the
    // Anima/SDXL K<=2048..2880 shapes prefer 128x128 BKB64 16w, while K=8192
    // (kbytes 16384) wants 128x128 BKB128 16w -- which is the split both arms
    // below make).
    //
    // Every other architecture falls through to the upstream heuristic below:
    // the branch above was tuned against 6 WGPs and is a regression elsewhere.
    //
    // Do not retune the 8-bit arm (the `kbytes >= 4096` chain below) on K alone.
    // An attempt to route int8 K in [2048, 5120] to the 64x64 BKB128 tile looked
    // like a 1.07-1.57x win when measured at M=8192/12288, and then *regressed*
    // 9 of the 16 shapes the benchmark actually runs (301.6 -> 306.4 ms summed).
    // Re-measuring with K and N fixed and M swept (4096..32768) showed the small
    // tile at 0.97-1.00x of the 128x128 for K=2048 at every M, so the original
    // sweep's signal was an artifact, not a K effect. Two traps, both worth
    // avoiding: the first GPU work in a process runs on cold clocks (it inflated
    // that sweep's apparent effect by 14-24%, above its own noise floor), and
    // shapes must include the M the workload really uses -- ComfyUI CFG at B=2
    // reaches M=2*sq=18432, well past where the effect was assumed to hold.
    if (!skinny && comfy_small_igpu()) {
        if (Mma::kStepBytes >= 32) {
            // 32-byte K-steps (fp16/bf16) halve the K-steps per tile versus 8-bit
            // operands. Re-measured on the 6-WGP 780M across ten Anima/SDXL/conv
            // shapes (ck_tools/sweep_gemm_fp.py, same-process interleaved): for a
            // moderate K the 64x64 BKB128 4-wave tile beats 128x128 by 3-27% -- it
            // yields 4x the blocks to interleave and its smaller LDS footprint
            // keeps more of them resident -- while deep K (mlp2, K=8192) and very
            // shallow K (adaln2, K=256) both want 128x128 back. kbytes is 2*K here,
            // so the 1024..4096 window is K in [512, 2048]; every measured shape
            // inside it agreed, and conv320 (kbytes 5760) plus adaln2 (512)
            // regress outside it.
            if (kbytes >= 1024 && kbytes <= 4096) {
                constexpr int BM = 64, BN = 64, BKB = 128;
                dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
                gemm_wmma_kernel<Mma, Epi, OutT, BM, BN, BKB, 2, 2, 2, 2, ASrc>
                    <<<grid, 128, 0, stream>>>(A, B, C, M, N, kbytes, ldc, epi);
            } else {
                constexpr int BM = 128, BN = 128, BKB = 128;
                dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
                gemm_wmma_kernel<Mma, Epi, OutT, BM, BN, BKB, 4, 4, 2, 2, ASrc>
                    <<<grid, 512, 0, stream>>>(A, B, C, M, N, kbytes, ldc, epi);
            }
        } else if (kbytes >= 4096) {
            constexpr int BM = 128, BN = 128, BKB = 128;
            dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
            gemm_wmma_kernel<Mma, Epi, OutT, BM, BN, BKB, 4, 4, 2, 2, ASrc>
                <<<grid, 512, 0, stream>>>(A, B, C, M, N, kbytes, ldc, epi);
        } else if (blocks_128 >= wgps) {
            constexpr int BM = 128, BN = 128, BKB = 64;
            dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
            gemm_wmma_kernel<Mma, Epi, OutT, BM, BN, BKB, 4, 4, 2, 2, ASrc>
                <<<grid, 512, 0, stream>>>(A, B, C, M, N, kbytes, ldc, epi);
        } else {
            constexpr int BM = 64, BN = 64, BKB = 64;
            dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
            gemm_wmma_kernel<Mma, Epi, OutT, BM, BN, BKB, 2, 2, 2, 2, ASrc>
                <<<grid, 128, 0, stream>>>(A, B, C, M, N, kbytes, ldc, epi);
        }
    } else if (!skinny && blocks_128 >= wgps) {
        if (kbytes >= 4096) {
            constexpr int BM = 128, BN = 128, BKB = 128;
            dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
            // With few blocks per WGP there is nothing to interleave across, so
            // the 16-wave grid hides latency within a block instead.
            if (blocks_128 <= 4 * wgps) {
                gemm_wmma_kernel<Mma, Epi, OutT, BM, BN, BKB, 4, 4, 2, 2, ASrc>
                    <<<grid, 512, 0, stream>>>(A, B, C, M, N, kbytes, ldc, epi);
            } else {
                gemm_wmma_kernel<Mma, Epi, OutT, BM, BN, BKB, 4, 2, 2, 4, ASrc>
                    <<<grid, 256, 0, stream>>>(A, B, C, M, N, kbytes, ldc, epi);
            }
        } else {
            constexpr int BM = 128, BN = 128, BKB = 64;
            dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
            gemm_wmma_kernel<Mma, Epi, OutT, BM, BN, BKB, 4, 2, 2, 4, ASrc>
                <<<grid, 256, 0, stream>>>(A, B, C, M, N, kbytes, ldc, epi);
        }
    } else if (kbytes >= 2048) {
        constexpr int BM = 64, BN = 64, BKB = 128;
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        gemm_wmma_kernel<Mma, Epi, OutT, BM, BN, BKB, 2, 2, 2, 2, ASrc>
            <<<grid, 128, 0, stream>>>(A, B, C, M, N, kbytes, ldc, epi);
    } else {
        constexpr int BM = 64, BN = 64, BKB = 64;
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        gemm_wmma_kernel<Mma, Epi, OutT, BM, BN, BKB, 2, 2, 2, 2, ASrc>
            <<<grid, 128, 0, stream>>>(A, B, C, M, N, kbytes, ldc, epi);
    }
}

}  // namespace comfy::hip_backend
