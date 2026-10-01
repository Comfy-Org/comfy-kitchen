// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Tiled VALU GEMM for RDNA2 (gfx1035), which has no matrix cores.
// Uses v_dot4_i32_i8 for INT8 and v_dot2_f32_f16 for FP16/BF16.
//
// Computes C[M, N] = epilogue(A[M, K] @ B[N, K]^T). The B operand is the weight
// in its natural (N, K) row-major form, matching torch linear. C is written with
// row stride ldc, so a caller splitting N into column chunks can point at a slice
// of a wider output; ldc == N for a whole GEMM.
//
// Algorithm: register-tiled GEMM with 4x4 accumulators per thread.
// - BM=64, BN=64, BK=64 (INT8) / BK=32 (FP16)
// - 256 threads, each computes 4x4=16 outputs
// - Each k-step loads 4 A values + 4 B values, reuses for 16 sdot4/fdot2
// - Double buffering: prefetch next tile while computing current
// - STRIDE = BK + 4 to avoid bank conflicts
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "epilogue.h"

namespace comfy::hip_backend {

// Every kernel in this file needs a gfx10-only dot instruction --
// v_dot4_i32_i8 for INT8 (__builtin_amdgcn_sdot4) and v_dot2_f32_f16 for FP16
// (__builtin_amdgcn_fdot2) -- and clang rejects the first with "needs target
// feature dot1-insts" on every other target, while gfx11/gfx12 spell dot4 as
// __builtin_amdgcn_sudot4. They are therefore compiled in the gfx103x device
// pass only (__GFX10__, predefined by clang for gfx1030-1036 and for nothing
// else), the one pass that runs them -- ops/gemm_int8.hip and
// ops/gemm_fp16.hip launch them behind comfy_is_gfx10().
//
// The other passes still need the symbols, because the launcher is host code
// and a fat binary is linked per architecture: a kernel defined in no pass
// leaves the host reference unresolved instead of declining to run. So each
// pass gets a trapping definition, as mma.h does for the WMMA policies.

// Row padding for the LDS staging buffers. The inner loops read 16 bytes at a
// time, so every row's first byte has to be 16-byte aligned. The original
// padding of 4 gave 68 bytes for the int8 tile (17 dwords) and 136 for the fp16
// one (34 dwords), so every odd row landed on a 4-byte boundary -- and clang
// still emitted ds_read_b128, which the ISA leaves undefined for a misaligned
// address.
//
// 16 is what ships for the int8 tile: on the 6-CU gfx1035 it was worth 2.6x on
// its own (0.87 -> 2.29 TOPS at 4096^3) once the persistent-grid bug below was
// also fixed, because halving the bank conflicts on the operand reads outweighs
// the 18% larger LDS footprint. 20 dwords still leaves a 2-way conflict on the
// 8-lane phase, so an XOR row swizzle is the next step if this kernel is tuned
// further. The fp16 tile uses a different padding for a different reason; see
// kFp16Pad below.
constexpr int kPad = 16;

// INT8 VALU block tile. Defined here rather than inside the kernel so the
// launcher in ops/gemm_int8.hip derives its grid from the same values. They were
// independent copies of 128/64; when the kernel was retuned to 256/128 the
// launcher's copy would have kept launching 128x64 grids over a 256x128 kernel,
// computing a quarter of the output per block and leaving three quarters of it
// unwritten. That is a wrong-answer bug rather than a slowdown, so the tile now
// has exactly one definition.
constexpr int kInt8BM = 256;
constexpr int kInt8BN = 128;
constexpr int kInt8Threads = 256;

// Row padding for the fp16 tile. The inner loop reads 16 bytes at a time and a
// row's byte offset is 2 * (row * STRIDE + k), so STRIDE must be a multiple of 8
// halves for every row to land on a 16-byte boundary. PAD = 4 (the ported value,
// STRIDE = 68 halves) is 4 mod 8 and therefore misaligned, which is what the
// int8 tile had too; 8 fixes it for 8% more LDS. With the launch grid now one
// block per tile, LDS occupancy no longer caps the grid, so the smaller of the
// two aligned options wins on footprint alone.
constexpr int kFp16Pad = 8;

#if defined(__GFX10__)

// ===========================================================================
// INT8 GEMM (triple-buffered): C[M, N] = A[M, K] @ B[N, K]^T
// Triple buffering hides DRAM load latency by overlapping prefetch with compute.
// With double buffering, the __syncthreads between prefetch and compute serialized
// them, leaving DRAM latency unhidden. Triple buffering issues the load for tile k+1
// BEFORE computing tile k, so the load overlaps with the compute.
//
// Thread tile is 16x8 over a 256x128 block, still 256 threads (16 rows x 32 cols
// of threads), double buffered.
//
// This replaces 8x4/128x64 triple-buffered, which the earlier sweep had chosen on
// cubic shapes. The real workloads are not cubic: the dispatch trace of one step
// at 1024x1536 with B=2 issues M=12288 or M=3072 with K/N of 1280, 2048, 640,
// 8192, and 74% of the int8 GEMM time sits in three M=12288 shapes. Re-measured
// there, 16x8/256x128 is 1.30-1.37x faster than 8x4/128x64 on every one of them,
// output verified bit-exact against a host reference in double:
//
//   M     N     K    8x4/128x64   16x8/256x128   speedup   triton
//   12288 2048 2048   2.93 TOPS     4.02 TOPS      1.37x    4.89
//   12288 2048 8192   2.94          4.04           1.37x    4.93
//   12288 8192 2048   2.97          4.00           1.35x    5.22
//   3072  10240 1280  2.90          4.12           1.42x    5.33
//   4096  4096 4096   2.94          3.93           1.34x    4.90
//
// The mechanism is MACs per LDS read: for a TM x TN tile on v_dot4c_i32_i8 that
// is 16*TM*TN/(TM+TN), so 8x4 gives 42.7 and 16x8 gives 53.3. Arithmetic issue
// rate is not the limit (dot4 and fp32 FMA time the same) and DRAM is not either
// (35-43 GB/s against a measured 89.7 GB/s ceiling), so the LDS read rate is
// what the tile has to amortise over.
//
// Two rejected alternatives, both measured, both worth keeping out of the tree:
//   * 16x8/128x128 keeps the 2.90 TOPS class (3.64-3.85) -- it is the block tile
//     width, not the thread tile, that carries the rest of the gain.
//   * 16x16 needs BM*BN coverable by 256 threads, which forces either a 128x256
//     block (LDS 60 KiB, fits) at 128 threads or a 256x256 block, and the latter
//     only fits at BK=32. Both were measured and both are far worse: 0.52 TOPS
//     at 128 threads, 0.72 at 256 threads with BK=32, 1.16 with BK=16 and four
//     buffers. The cause is register spilling, not LDS, and the compiler reports
//     it directly under -Rpass-analysis=kernel-resource-usage:
//
//       8x4  /128x64  32 accumulators   VGPR 228   scratch 0    spill   0   2.95 TOPS
//       16x8 /256x128 128 accumulators  VGPR 232   scratch 0    spill   0   4.01 TOPS
//       16x16/256x256 256 accumulators  VGPR 256   scratch 712  spill 332   0.72 TOPS
//
//     A 16x16 int32 accumulator tile needs 256 VGPRs and the architecture grants
//     128 per thread, so 332 spill to scratch and the inner loop becomes bound by
//     local-memory traffic -- 712 bytes per lane per k-tile. 16x8 sits exactly at
//     the 128 budget with zero spill, which is why it is the optimum rather than
//     a compromise. 32x8, 16x32 and 8x32 were measured for the same reason and
//     are worse still (0.24-0.70 TOPS).
//
//     So triton's 10.04 dot4-per-LDS-read is not a 16x16 thread tile. Its ratio
//     is what a 16x8 tile reaches with a wider k-unroll, and the ceiling here is
//     the register file, not the LDS budget.
//   * widening the k-unroll was tried too, on the theory that more dot4 per LDS
//     read is free once the accumulators fit. It is not: 16x8/128x128 at BK=96
//     drops to 2.41 TOPS from 3.73 at BK=64, because the LDS row grows with BK
//     and the wider row costs more than the extra unroll buys. BK=128 does not fit
//     at two buffers anywhere (2*(BM+BN)*144 exceeds 64 KiB for both 256x64 and
//     128x128) and a single buffer cannot overlap a load with compute. So BK=64
//     stands, and 16x8/256x128/K64 is the measured optimum, not a stopping point.
//
// The grid is one block per tile, NOT a persistent walk -- see the note on why
// the persistent variant was deleted.
// ===========================================================================

template <typename OutT>
__global__ __launch_bounds__(256) void gemm_int8_valu_kernel(
    const int8_t* __restrict__ A, const int8_t* __restrict__ B,
    EpiRowwise epi,
    OutT* __restrict__ C,
    int M, int N, int K, int ldc) {

    constexpr int TM = 16;     // rows per thread
    constexpr int TN = 8;      // cols per thread
    constexpr int BM = kInt8BM;
    constexpr int BN = kInt8BN;
    constexpr int BK = 64;
    constexpr int THREADS = kInt8Threads;
    constexpr int STRIDE = BK + kPad;
    constexpr int A_SIZE = BM * STRIDE;
    constexpr int B_SIZE = BN * STRIDE;
    // Double buffered: 2 * (256 + 128) * 80 = 61440 B, just inside the 64 KiB
    // per-block limit. A third buffer would be 92160 and would be demoted to local
    // memory by the compiler without any error being reported.
    constexpr int NBUFFERS = 2;
    static_assert((BM / TM) * (BN / TN) == THREADS,
                  "thread tile and block tile must cover the block exactly");
    // Over 64 KiB per block the compiler demotes the shared arrays to local
    // memory and reports no error: the kernel still compiles, still runs, and is
    // just slow (measured 0.5 TOPS for a 16x16 variant that would have needed
    // 90 KiB). Assert the footprint rather than trusting it.
    static_assert(NBUFFERS * (A_SIZE + B_SIZE) <= 64 * 1024,
                  "int8 LDS footprint exceeds 64 KiB and will spill to local");

    __shared__ int8_t A_buf[NBUFFERS][A_SIZE];
    __shared__ int8_t B_buf[NBUFFERS][B_SIZE];

    const int tid = threadIdx.x;
    const int trow = tid / (BN / TN);
    const int tcol = tid % (BN / TN);
    const int orow = trow * TM;
    const int ocol = tcol * TN;

    const int m0 = blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;

    int acc[TM][TN] = {};
    const int chunks = BK / 16;

    // Helper: load a tile from global to shared memory buffer
    auto load_tile = [&](int buf_idx, int k0) {
        for (int i = tid; i < BM * chunks; i += THREADS) {
            const int r = i / chunks;
            const int c = i % chunks;
            const int gk = k0 + c * 16;
            int4 val = make_int4(0, 0, 0, 0);
            if (m0 + r < M && gk + 15 < K)
                val = *reinterpret_cast<const int4*>(A + (m0 + r) * K + gk);
            *reinterpret_cast<int4*>(&A_buf[buf_idx][r * STRIDE + c * 16]) = val;
        }
        for (int i = tid; i < BN * chunks; i += THREADS) {
            const int r = i / chunks;
            const int c = i % chunks;
            const int gk = k0 + c * 16;
            int4 val = make_int4(0, 0, 0, 0);
            if (n0 + r < N && gk + 15 < K)
                val = *reinterpret_cast<const int4*>(B + (n0 + r) * K + gk);
            *reinterpret_cast<int4*>(&B_buf[buf_idx][r * STRIDE + c * 16]) = val;
        }
    };

    // Helper: compute one k-tile from shared memory.
    //
    // For each output (row, col), every field of the a-vector pairs with the
    // matching field of the b-vector: the field selects the K-range, not the
    // output column. Written as a nest over the compile-time TM/TN/4 rather
    // than spelled out per (row, col, field) -- the compiler unrolls all three
    // loops, so this is the same v_dot4_i32_i8 sequence the flat version emitted.
    auto compute_tile = [&](int buf_idx) {
        const int8_t* Ac = &A_buf[buf_idx][0];
        const int8_t* Bc = &B_buf[buf_idx][0];
        #pragma unroll
        for (int k = 0; k < BK; k += 16) {
            int4 av[TM], bv[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                av[i] = *reinterpret_cast<const int4*>(&Ac[(orow + i) * STRIDE + k]);
            #pragma unroll
            for (int j = 0; j < TN; ++j)
                bv[j] = *reinterpret_cast<const int4*>(&Bc[(ocol + j) * STRIDE + k]);
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] = __builtin_amdgcn_sdot4(av[i].x, bv[j].x, acc[i][j], true);
                    acc[i][j] = __builtin_amdgcn_sdot4(av[i].y, bv[j].y, acc[i][j], true);
                    acc[i][j] = __builtin_amdgcn_sdot4(av[i].z, bv[j].z, acc[i][j], true);
                    acc[i][j] = __builtin_amdgcn_sdot4(av[i].w, bv[j].w, acc[i][j], true);
                }
            }
        }
    };

    // Prologue: fill NBUFFERS-1 slices ahead, so slice k0 is always resident when
    // the loop reaches it. The prefetch distance is NBUFFERS-1, never 2: with
    // NBUFFERS == 2 a distance of 2 resolves to (buf + 2) % 2 == buf, i.e. the
    // prefetch writes the slice the compute is reading and every element comes
    // out wrong. That is not a slowdown, it is a silent wrong answer, and it is
    // the reason the distance is written as NBUFFERS-1 rather than as a literal.
    #pragma unroll 1
    for (int p = 0; p < NBUFFERS - 1; ++p)
        if (p * BK < K) load_tile(p, p * BK);
    __syncthreads();

    constexpr int PREFETCH = NBUFFERS - 1;
    int buf = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
        const int next_buf = (buf + 1) % NBUFFERS;
        const int prefetch_buf = (buf + PREFETCH) % NBUFFERS;
        const int kprefetch = k0 + PREFETCH * BK;

        // Issue the load for a slice PREFETCH ahead, so it overlaps the compute
        // below. The barrier between them is what makes that safe, and the
        // barrier after the compute is what makes the *next* iteration's
        // prefetch safe against this iteration's reads.
        if (kprefetch < K) {
            load_tile(prefetch_buf, kprefetch);
        }

        compute_tile(buf);

        __syncthreads();
        buf = next_buf;
    }

    // Epilogue: write the TM x TN outputs with scaling and bias
    #pragma unroll
    for (int ri = 0; ri < TM; ri++) {
        const int row = m0 + orow + ri;
        if (row >= M) continue;
        #pragma unroll
        for (int ci = 0; ci < TN; ci++) {
            const int col = n0 + ocol + ci;
            if (col >= N) continue;
            C[row * ldc + col] = static_cast<OutT>(epi(row, col, static_cast<float>(acc[ri][ci])));
        }
    }
}

// ===========================================================================
// FP16 GEMM: C[M, N] = A[M, K] @ B[N, K]^T

//
// Output is fp16 only. RDNA2 has no v_dot2_f32_bf16, so a bf16 output would
// have to arrive by reinterpreting bf16 bit patterns as fp16 -- which is
// silently wrong, and did ship: see the note on launch_bf16_gemm_kernel in
// ops/gemm_fp16.hip. C stays concrete so that mistake cannot be made here.
// ===========================================================================

// Templated only to get inline linkage: this header is included by both
// ops/gemm_fp16.hip and ops/gemm_int8.hip, so a plain non-template definition
// would be emitted into two objects and collide at link time. OutT is fixed to
// __half and asserted, so the template cannot be used to reintroduce a bf16
// output.
template <typename OutT>
__global__ __launch_bounds__(256) void gemm_fp16_valu_kernel(
    const __half* __restrict__ A, const __half* __restrict__ B,
    EpiFp16 epi,
    OutT* __restrict__ C,
    int M, int N, int K, int ldc) {

    static_assert(std::is_same_v<OutT, __half>,
                  "RDNA2 has no bf16 dot product; see the note above");

    // Tile: 4x4 per thread over a 64x64 block, BK 64, double buffered.
    //
    // An 8x8 / 128x128 / BK32 tile was measured 19% faster standalone (1.96 vs
    // 1.65 TOPS at 4096^3, benchmark_attn/待验证.md B6) and validated at that
    // size in the harness, but it is wrong in the library from 196 blocks
    // upward: rel err 2.8e-04 at 169 blocks, 1.2e+00 at 196, in every element,
    // repeatably. Not yet diagnosed. Do not retune this tile on harness numbers
    // alone -- re-measure through fp16_linear at 1792^2 and 2048^2 at minimum.
    constexpr int TM = 4;      // rows per thread
    constexpr int TN = 4;      // cols per thread
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 64;
    constexpr int THREADS = 256;
    constexpr int STRIDE = BK + kFp16Pad;
    constexpr int A_SIZE = BM * STRIDE;
    constexpr int B_SIZE = BN * STRIDE;
    static_assert((BM / TM) * (BN / TN) == THREADS,
                  "thread tile and block tile must cover the block exactly");
    // Over 64 KiB the compiler demotes the shared arrays to local memory without
    // reporting an error, so the kernel still compiles and still runs, just
    // slowly. Assert the footprint rather than trusting it.
    static_assert(2 * (A_SIZE + B_SIZE) * sizeof(__half) <= 64 * 1024,
                  "fp16 LDS footprint exceeds 64 KiB and will spill to local");

    typedef _Float16 v2h __attribute__((ext_vector_type(2)));
    typedef _Float16 v8h __attribute__((ext_vector_type(8)));

    __shared__ __half A_buf[2][A_SIZE];
    __shared__ __half B_buf[2][B_SIZE];

    const int tid = threadIdx.x;

    // 16x16 threads, each owning a 4x4 block of the 64x64 tile.
    const int trow = tid / (BN / TN);
    const int tcol = tid % (BN / TN);
    const int orow = trow * TM;
    const int ocol = tcol * TN;

    const int m0 = blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;

    float acc[TM][TN] = {};

    // Stage one BK-deep k slice of A and B into LDS buffer `buf`, reading global
    // row-major and writing with 16-byte stores. STRIDE is a multiple of 8
    // halves (kFp16Pad = 8) so every row starts 16-byte aligned.
    auto load_tile = [&](int buf, int k0) {
        const int chunks = BK / 8;  // 8 halves per int4
        for (int i = tid; i < BM * chunks; i += THREADS) {
            const int r = i / chunks;
            const int c = i % chunks;
            const int gk = k0 + c * 8;
            int4 val = make_int4(0, 0, 0, 0);
            if (m0 + r < M && gk + 7 < K)
                val = *reinterpret_cast<const int4*>(A + (m0 + r) * K + gk);
            *reinterpret_cast<int4*>(&A_buf[buf][r * STRIDE + c * 8]) = val;
        }
        for (int i = tid; i < BN * chunks; i += THREADS) {
            const int r = i / chunks;
            const int c = i % chunks;
            const int gk = k0 + c * 8;
            int4 val = make_int4(0, 0, 0, 0);
            if (n0 + r < N && gk + 7 < K)
                val = *reinterpret_cast<const int4*>(B + (n0 + r) * K + gk);
            *reinterpret_cast<int4*>(&B_buf[buf][r * STRIDE + c * 8]) = val;
        }
    };

    // Both buffers are filled before the single barrier, so iteration 0 reads a
    // complete buffer[0] and iteration 1 a complete buffer[1], with one barrier
    // per k-tile rather than two.
    load_tile(0, 0);
    if (BK < K) load_tile(1, BK);
    __syncthreads();

    int buf = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
        const int nb = 1 - buf;
        const int kn = k0 + BK;

        // Prefetch the slice after the one now in `buf`. The barrier below orders
        // these stores against every thread's compute of `buf`, and the barrier
        // at the end of the body orders them against the next iteration's compute
        // of `nb`, so the two barriers per iteration are what make the buffer
        // hand-off safe. With a single barrier per iteration one of the two
        // orderings is lost and a block can compute a slice another thread is
        // still writing.
        if (kn < K) load_tile(nb, kn);

        __syncthreads();

        // Compute: TM x TN outputs. One 16-byte LDS read per row per 8 k
        // (8 halves), then four fdot2 per (row, col) pair. Written as loops
        // over the compile-time TM/TN rather than as 64 literal fdot2 calls,
        // so a tile change cannot leave the arithmetic computing a different
        // width than the geometry claims.
        const __half* Ac = &A_buf[buf][0];
        const __half* Bc = &B_buf[buf][0];
#pragma unroll
        for (int k = 0; k < BK; k += 8) {
            v8h av[TM], bv[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i)
                av[i] = *reinterpret_cast<const v8h*>(&Ac[(orow + i) * STRIDE + k]);
#pragma unroll
            for (int j = 0; j < TN; ++j)
                bv[j] = *reinterpret_cast<const v8h*>(&Bc[(ocol + j) * STRIDE + k]);
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
#pragma unroll
                    for (int q = 0; q < 4; ++q)
                        acc[i][j] = __builtin_amdgcn_fdot2(
                            reinterpret_cast<const v2h*>(&av[i])[q],
                            reinterpret_cast<const v2h*>(&bv[j])[0], acc[i][j], true);
        }

        __syncthreads();
        buf = nb;
    }

    // Epilogue: write the TM x TN block with bias/residual
    for (int ri = 0; ri < TM; ri++) {
        const int row = m0 + orow + ri;
        if (row >= M) continue;
        for (int ci = 0; ci < TN; ci++) {
            const int col = n0 + ocol + ci;
            if (col >= N) continue;
            C[row * ldc + col] = static_cast<__half>(epi(row, col, acc[ri][ci]));
        }
    }
}

// ===========================================================================
#else  // !__GFX10__

// No v_dot4_i32_i8 / v_dot2_f32_f16 on this target. comfy_is_gfx10() keeps the
// launches off these devices; trap rather than return a plausible wrong answer
// if one ever runs. See the note at the top of this file for why the stubs have
// to exist at all.
template <typename OutT>
__global__ __launch_bounds__(256) void gemm_int8_valu_kernel(
    const int8_t*, const int8_t*, EpiRowwise, OutT*, int, int, int, int) {
    __builtin_trap();
}

template <typename OutT>
__global__ __launch_bounds__(256) void gemm_fp16_valu_kernel(
    const __half*, const __half*, EpiFp16, OutT*, int, int, int, int) {
    __builtin_trap();
}

#endif  // __GFX10__

}  // namespace comfy::hip_backend
