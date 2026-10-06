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
#include <stdexcept>
#include <string>
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

// INT8 VALU block tile. Defined here rather than inside the kernel so the
// launcher in ops/gemm_int8.hip derives its grid from the same values. They were
// independent copies of 128/64; when the kernel was retuned to 256/128 the
// launcher's copy would have kept launching 128x64 grids over a 256x128 kernel,
// computing a quarter of the output per block and leaving three quarters of it
// unwritten. That is a wrong-answer bug rather than a slowdown, so the tile now
// has exactly one definition.
constexpr int kInt8BM = 128;
constexpr int kInt8BN = 128;
constexpr int kInt8Threads = 256;

// Row padding for the int8 A staging buffer, in bytes. Zero, and that is a
// measured result rather than an oversight. There used to be a shared kPad = 16
// here that both GEMMs used, because the inner loops read 16 bytes at a time and
// a row-major LDS row has to start 16-byte aligned for the int4 staging store to
// be legal -- clang emits ds_read_b128 regardless, which the ISA leaves undefined
// for a misaligned address, so the alignment was load-bearing. The int8 tile no
// longer needs it: B is staged chunk-major (see gemm_int8_valu_kernel), which
// drops the constraint on B entirely, and A is better off unpadded because the
// staging writes become one contiguous 512-byte run per warp instead of eight
// 64-byte runs. 5.63 -> 5.97 TOPS at 12288x2048x2048 from this constant alone.
constexpr int kInt8Pad = 0;

// Row padding for the int4 A staging buffer, in bytes. A k-tile of 64 k-values
// is 32 bytes per row, so a row still starts 16-byte aligned at pad 0 and the
// ds_read_b128 in the inner loop is legal. Zero here is not measured the way
// kInt8Pad's zero was: the int4 tile is new, and this constant exists so that a
// sweep does not have to edit the tile's body.
constexpr int kInt4Pad = 0;

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
// Thread tile 16x4 over a 128x128 block, 256 threads, BK 64, double buffered.
//
// This replaces 16x8/256x128, which the previous sweep had chosen. It is 1.45x
// faster on the real dispatch mix and it overtakes triton, which 16x8 did not.
//
// What changed the conclusion is the register file, not the arithmetic. The
// device code for 16x8 was disassembled out of the .pyd's .hip_fat section (the
// section is named .hip_fat, not .hip_fatbin, and it holds 90 embedded AMDGPU
// ELF objects -- llvm-objdump reads one of those fine, it just cannot read them
// out of the fat binary itself). Its inner loop is 2048 v_dot4_i32_i8 against 96
// ds_read_b128, i.e. 21.33 dot4 per LDS read, which is 4*TM*TN/(TM+TN) for a
// 16x8 tile exactly. Triton's is 10.04. So triton was doing *half* our MAC per LDS
// read and still winning by 1.5x, which rules out both the LDS read rate and
// arithmetic intensity as the binding constraint.
//
// -Rpass-analysis=kernel-resource-usage is what settled it. gfx10 has a 512-entry
// register file per SIMD and allocates it in whole waves, so waves/SIMD is
// floor(512 / VGPRs) with no partial credit, and the value is piecewise constant:
//
//   16x8 /256x128  128 accumulators   VGPR 214   2 waves/SIMD   3.97 TOPS
//   16x4 /128x128   64 accumulators   VGPR 154   3 waves/SIMD   5.97 TOPS
//   8x8  /256x128   64 accumulators   VGPR 125   4 waves/SIMD   2.75 TOPS
//   8x4  /128x64    32 accumulators   VGPR  97   5 waves/SIMD   3.48 TOPS
//   16x16/256x256  256 accumulators   VGPR 256   1 wave/SIMD    0.72 TOPS
//
// triton's own autotune configs land at n_regs = 126..128, i.e. 4 waves/SIMD, for
// the ones that win. It reaches 4 waves with 128x128 block tiles and 4 or 8 warps.
// 128 int32 accumulators cannot fit under 128 VGPRs at all, so no amount of
// scheduling work makes 16x8 reach 4 waves; the accumulator count is what has to
// come down. The last row is why: past 128 accumulators it spills to scratch and
// collapses.
//
// Occupancy is not the whole story either, because the two 64-accumulator tiles
// differ by 2.2x despite both reaching at least 3 waves. Two things separate
// 16x4 from 8x8:
//
//   * Wave count within a block. 512 threads is 16 waves on 4 SIMDs; at 4
//     waves/SIMD that is exactly one resident block and 3 of every 4 waves never
//     run. The 4-wave configs at 512 threads all measured badly (8x8 at 512 is
//     2.19-2.75 TOPS, worse than any 256-thread entry). Occupancy is worth having
//     only if the block is small enough that several fit.
//   * Bank conflicts, which is what the B layout below is for.
//
// The B layout, and why it is a layout change rather than a padding change.
// On gfx10 a 16-byte lane read is serviced in phases of 8 lanes, since 8 x 16 B =
// 128 B is the whole 32-bank array, and a phase is conflict-free only when those
// 8 addresses are 16 B apart -- TN * STRIDE must be 16 (mod 128). Row-major
// cannot satisfy that at any pitch that also keeps the int4 staging store
// 16-byte aligned, so padding only trades 8-way for 4-way: measured 5.63 TOPS at
// pad 16, 4.50 at pad 0, 3.81 at pad 32, 4.84 at pad 48, 3.79 at pad 64. Storing
// chunk c of row r at (c * BN + r) * 16 puts consecutive lanes on consecutive
// addresses with no constraint on the pitch at all: 5.63 -> 5.97. The A side needs
// no change; with TX = 32 the whole warp shares trow and therefore shares the A
// address, and an LDS broadcast is free.
//
// Measured on the real shapes, 16x8/256x128 -> 16x4/128x128, output verified
// bit-exact against a host reference in double on every entry:
//
//   M       N      K     16x8/256x128   16x4/128x128   ratio   triton
//   12288  2048   2048     3.97 TOPS       5.97 TOPS    0.67x    4.89
//   12288  2048   8192     4.06            5.91         0.69x    4.93
//   12288  8192   2048     3.97            5.99         0.66x    5.22
//   3072   10240  1280     4.23            6.09         0.69x    5.33
//   3072   1280   1280     4.28            6.26         0.68x      -
//   4096   4096   4096     4.05            5.79         0.70x    4.90
//
// Weighted by call count from the dispatch trace that is 0.65x.
//
// One caveat about that table, because it was got wrong once. An earlier run of
// the same harness put 12288x8192x2048 at 4.60 TOPS (0.86x) rather than 5.99
// (0.66x). It is not reproducible: the same binary in a second harness, and with
// the other nine configurations run first to move the clock, both give 68.2-68.4
// ms, and the shipped 16x8 baseline reproduces to within 0.2% in every one of
// those runs. A single 30% outlier in a table is not a property of the kernel.
//
// Two things that look like they should help and do not:
//
//   * Capping the k-chunk unroll does nothing for 16x8: at KU=1/2/4 the compiler
//     reports 234/213/214 VGPRs, all 2 waves. For 64 accumulators it is worth a
//     lot (8x8 goes 178 -> 138), so the hoisting is real but 128 accumulators
//     dominate it.
//   * 16x16 is still out. 256 accumulators needs 256 VGPRs, the architecture
//     grants 128 per thread, 332 spill to scratch, and the inner loop becomes
//     bound by 712 bytes per lane per k-tile of local-memory traffic.
//
// The staging's 16-byte loads, which look like the last obvious thing and are not.
// The disassembly puts 1109 instructions of 64-bit address arithmetic in this
// kernel, 21.5% of it, and shows the int4 staging read split into 4-byte and
// 2-byte pieces:
//
//   global_load*  324 ins  1088 B/lane   dwordx4:4x16B  dword:192x4B  ushort:128x2B
//
// which is 4.8x the load-pipeline requests for the same bytes, because the row
// address A + (m0 + r) * K has a runtime stride K that the compiler cannot prove
// 16-byte aligned. Three spellings of the promise were measured and none pays:
//
//   __builtin_assume((K & 15) == 0) and the same on both pointers -- leaves the
//       emitted loads byte-identical to the plain cast, 2287 instructions and
//       4 dwordx4 / 128 dword either way. Documented not to reach codegen.
//   __builtin_assume_aligned on each loaded address -- also byte-identical.
//   Rewriting the row offset structurally as (m0 + r) * (K >> 4) * 16, so the
//       multiply by the literal 16 makes the alignment evident in the IR with no
//       assumption involved -- 2285 instructions, still 4 dwordx4 / 128 dword,
//       and a tie in wall clock on every real shape.
//
// Unrolling the staging loop does produce dwordx4 (8 of them) and costs 3-9%,
// which settles it: more 16-byte requests do not make this kernel faster, so the
// staging is limited by bytes moved rather than by requests. None of the 1109
// address instructions are inside the dot4 span either -- 0% -- so the compute loop
// is already clean and there is nothing to hoist out of it.
//
// And one that is not a tile question at all, recorded because it looked like the
// obvious next thing:
//
//   * The block order. triton groups its program ids into GROUP_M-tall bands
//     (group_size_m=8) before mapping them to tiles, and this launcher uses the
//     plain blockIdx mapping. Measured GROUP_M = 1/2/4/8/16 on every real shape:
//     all within 3% of each other, with the winner different on each shape. The
//     reason is occupancy: only 6 blocks are resident at a time on 6 CUs, and 6
//     consecutive blockIdx.x already share one A tile, so there is nothing for a
//     band to improve. Same for 512-thread geometries (BN or BM 256), which need
//     2 waves/SIMD's worth of block to be resident and are 1.3-1.7x slower.
//
// No geometry dispatch is justified inside this kernel. A scan over N (640 to
// 10240), K (640 to 4096) and M (128 to 12288) found no boundary on the first two
// and one on the third: at M = 154, where BM = 128 leaves the last block row 20%
// occupied and BM = 64 would leave it 41%, the 64-row block wins by 1.15-1.30x.
// That shape is 0.4% of a step's int8 GEMM time, so the dispatch would be worth
// 0.08%. Wider blocks (BN or BM = 256) need 512 threads and lose 1.3-1.7x; three
// and four buffers and BK = 32 lose 1.04-1.26x.
//
// The grid is one block per tile, NOT a persistent walk -- see the note on why
// the persistent variant was deleted.
// ===========================================================================

template <typename OutT>
__global__ __launch_bounds__(kInt8Threads) void gemm_int8_valu_kernel(
    const int8_t* __restrict__ A, const int8_t* __restrict__ B,
    EpiRowwise epi,
    OutT* __restrict__ C,
    int M, int N, int K, int ldc) {

    constexpr int TM = 16;     // rows per thread
    constexpr int TN = 4;      // cols per thread
    constexpr int BM = kInt8BM;
    constexpr int BN = kInt8BN;
    constexpr int BK = 64;
    constexpr int THREADS = kInt8Threads;
    constexpr int STRIDE = BK + kInt8Pad;
    constexpr int VPR = BK / 16;          // 16-byte chunks per row per k-tile
    constexpr int CHUNKS = BK / 16;
    constexpr int A_SIZE = BM * STRIDE;
    // B is staged chunk-major: B_buf[buf][chunk * BN + row][0..15]. See the note
    // on the layout below for why that is worth 6% and why padding cannot achieve
    // the same thing. Its footprint has no padding term at all.
    constexpr int B_SIZE = BN * BK;
    // Double buffered: 2 * (128*64 + 128*64) = 32768 B, half the 64 KiB per-block
    // limit. A third buffer would be 49152 and does fit, but measured 5.06 TOPS
    // against 5.97 for two at 12288x2048x2048: the third slice is 16 KiB of LDS
    // per block resident for prefetch distance the hardware does not repay.
    constexpr int NBUFFERS = 2;
    static_assert((BM / TM) * (BN / TN) == THREADS,
                  "thread tile and block tile must cover the block exactly");
    // Over 64 KiB per block the compiler demotes the shared arrays to local
    // memory and reports no error: the kernel still compiles, still runs, and is
    // just slow (measured 0.5 TOPS for a 16x16 variant that would have needed
    // 90 KiB). Assert the footprint rather than trusting it.
    static_assert(NBUFFERS * (A_SIZE + B_SIZE) <= 64 * 1024,
                  "int8 LDS footprint exceeds 64 KiB and will spill to local");
    // Chunk-major B is addressed as (chunk * BN + row) * 16, so it needs BN and
    // 16-byte granularity, nothing else. Row-major would need STRIDE % 16 == 0 for
    // the int4 staging store, which is what kInt8Pad used to have to guarantee.
    static_assert((BN * 16) % 16 == 0 && VPR == CHUNKS, "chunk-major B layout");

    __shared__ int8_t A_buf[NBUFFERS][A_SIZE];
    __shared__ int8_t B_buf[NBUFFERS][B_SIZE];

    const int tid = threadIdx.x;
    // TX = 32 puts one warp entirely inside one row-group, which matters below.
    const int trow = tid / (BN / TN);
    const int tcol = tid % (BN / TN);
    const int orow = trow * TM;
    const int ocol = tcol * TN;

    const int m0 = blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;

    int acc[TM][TN] = {};

    // Helper: load a tile from global to shared memory buffer.
    //
    // Out-of-range elements are written as zero rather than skipped, so the compute
    // loop needs no mask: a K that is not a multiple of BK or 16 still produces the
    // right answer, which matters because these weights come from a checkpoint
    // whose K is whatever the layer happened to have.
    auto load_tile = [&](int buf_idx, int k0) {
        for (int i = tid; i < BM * VPR; i += THREADS) {
            const int r = i / VPR;
            const int c = i % VPR;
            const int gk = k0 + c * 16;
            int4 val = make_int4(0, 0, 0, 0);
            if (m0 + r < M && gk + 15 < K)
                val = *reinterpret_cast<const int4*>(A + (m0 + r) * K + gk);
            *reinterpret_cast<int4*>(&A_buf[buf_idx][r * STRIDE + c * 16]) = val;
        }
        for (int i = tid; i < BN * VPR; i += THREADS) {
            const int r = i / VPR;
            const int c = i % VPR;
            const int gk = k0 + c * 16;
            int4 val = make_int4(0, 0, 0, 0);
            if (n0 + r < N && gk + 15 < K)
                val = *reinterpret_cast<const int4*>(B + (n0 + r) * K + gk);
            *reinterpret_cast<int4*>(&B_buf[buf_idx][(c * BN + r) * 16]) = val;
        }
    };

    // Helper: compute one k-tile from shared memory.
    //
    // For each output (row, col), every field of the a-vector pairs with the
    // matching field of the b-vector: the field selects the K-range, not the
    // output column. Written as a nest over the compile-time TM/TN/4 rather
    // than spelled out per (row, col, field) -- the compiler unrolls all three
    // loops, so this is the same v_dot4_i32_i8 sequence the flat version emitted.
    //
    // The chunk loop is "#pragma unroll 1": one 16-byte k-chunk's worth of
    // av/bv is live at a time instead of all four. That is what keeps the
    // register file from deciding the occupancy -- see the note on the tile.
    auto compute_tile = [&](int buf_idx) {
        const int8_t* Ac = &A_buf[buf_idx][0];
        const int8_t* Bc = &B_buf[buf_idx][0];
        #pragma unroll 1
        for (int c = 0; c < CHUNKS; ++c) {
            const int k = c * 16;
            int4 av[TM], bv[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                av[i] = *reinterpret_cast<const int4*>(&Ac[(orow + i) * STRIDE + k]);
            #pragma unroll
            for (int j = 0; j < TN; ++j)
                bv[j] = *reinterpret_cast<const int4*>(&Bc[(c * BN + ocol + j) * 16]);
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
// INT4 GEMM on VALU: C[M, N] = (A[M, K] @ B[N, K]^T) * scale_a[row] * scale_b[col]
//
// A and B are signed int4 packed two per byte, low nibble = even k, which is the
// layout ops/convrot_w4a4.hip's quantized weights already use. The tile is the
// int8 tile above with one change: v_dot8_i32_i4 does eight 4x4-bit products per
// instruction against v_dot4_i32_i8's four, and retires at the same rate on this
// part (measured 1.62 vs 1.58 Tera-instr/s), so a k-tile needs half the dot
// instructions and half the ds_read_b128 for the same arithmetic. LDS therefore
// holds half the bytes per row and the chunk count per k-tile halves with it.
//
// Not the same entry point as convrot_w4a4_linear: that one is WMMA-only (MmaInt4
// traps without matrix cores) and this one needs none.
// ===========================================================================
__device__ __forceinline__ int dot8_i4(int a, int b, int c) {
    int r;
    asm("v_dot8_i32_i4 %0, %1, %2, %3" : "=v"(r) : "v"(a), "v"(b), "v"(c));
    return r;
}

template <typename OutT>
__global__ __launch_bounds__(kInt8Threads) void gemm_int4_valu_kernel(
    const int8_t* __restrict__ A, const int8_t* __restrict__ B,
    EpiRowwise epi,
    OutT* __restrict__ C,
    int M, int N, int K, int ldc) {

    constexpr int TM = 16;
    constexpr int TN = 4;
    constexpr int BM = kInt8BM;
    constexpr int BN = kInt8BN;
    constexpr int BK = 64;                 // k-values per tile, i.e. 32 bytes
    constexpr int THREADS = kInt8Threads;
    // A 16-byte LDS read is 32 k-values, so a BK=64 tile is CPR=2 chunks.
    constexpr int CPR = BK / 32;
    constexpr int ROWB = BK / 2;           // bytes per row per k-tile
    constexpr int STRIDE = ROWB + kInt4Pad;
    constexpr int VPR = CPR;               // 16-byte chunks per row per k-tile
    constexpr int B_SIZE = CPR * BN * 16;  // chunk-major, no padding term
    constexpr int A_SIZE = BM * STRIDE;
    constexpr int NBUFFERS = 2;
    static_assert((BM / TM) * (BN / TN) == THREADS, "thread tile and block tile must agree");
    static_assert(NBUFFERS * (A_SIZE + B_SIZE) <= 64 * 1024, "int4 LDS footprint would spill");
    static_assert(BK % 32 == 0, "a chunk must be a whole number of 16-byte reads");

    __shared__ int8_t A_buf[NBUFFERS][A_SIZE];
    __shared__ int8_t B_buf[NBUFFERS][B_SIZE];

    const int tid = threadIdx.x;
    const int trow = tid / (BN / TN);
    const int tcol = tid % (BN / TN);
    const int orow = trow * TM;
    const int ocol = tcol * TN;

    const int m0 = blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;
    const int K2 = K / 2;                  // bytes per operand row

    int acc[TM][TN] = {};

    auto load_tile = [&](int buf_idx, int k0) {
        for (int i = tid; i < BM * VPR; i += THREADS) {
            const int r = i / VPR;
            const int c = i % VPR;
            const int gk = k0 / 2 + c * 16;
            int4 val = make_int4(0, 0, 0, 0);
            if (m0 + r < M && gk + 15 < K2)
                val = *reinterpret_cast<const int4*>(A + (m0 + r) * K2 + gk);
            *reinterpret_cast<int4*>(&A_buf[buf_idx][r * STRIDE + c * 16]) = val;
        }
        for (int i = tid; i < BN * VPR; i += THREADS) {
            const int r = i / VPR;
            const int c = i % VPR;
            const int gk = k0 / 2 + c * 16;
            int4 val = make_int4(0, 0, 0, 0);
            if (n0 + r < N && gk + 15 < K2)
                val = *reinterpret_cast<const int4*>(B + (n0 + r) * K2 + gk);
            *reinterpret_cast<int4*>(&B_buf[buf_idx][(c * BN + r) * 16]) = val;
        }
    };

    auto compute_tile = [&](int buf_idx) {
        #pragma unroll 1
        for (int c = 0; c < CPR; ++c) {
            int4 av[TM], bv[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                av[i] = *reinterpret_cast<const int4*>(&A_buf[buf_idx][(orow + i) * STRIDE + c * 16]);
            #pragma unroll
            for (int j = 0; j < TN; ++j)
                bv[j] = *reinterpret_cast<const int4*>(&B_buf[buf_idx][(c * BN + ocol + j) * 16]);
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] = dot8_i4(av[i].x, bv[j].x, acc[i][j]);
                    acc[i][j] = dot8_i4(av[i].y, bv[j].y, acc[i][j]);
                    acc[i][j] = dot8_i4(av[i].z, bv[j].z, acc[i][j]);
                    acc[i][j] = dot8_i4(av[i].w, bv[j].w, acc[i][j]);
                }
            }
        }
    };

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
        if (kprefetch < K) {
            load_tile(prefetch_buf, kprefetch);
        }
        compute_tile(buf);
        __syncthreads();
        buf = next_buf;
    }

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
    // repeatably. Do not retune this tile on harness numbers alone -- re-measure
    // through fp16_linear at 1792^2 and 2048^2 at minimum.
    //
    // That failure is not yet diagnosed, and this fix does not diagnose it. What it
    // does remove is a second, different fault that this same commit introduced: the
    // q loop below was pairing A pair q with B pair 0, so only one k-pair in four was
    // multiplied correctly. That fault is size-independent -- every shape would read
    // 1.2e+00 with correlation 0.247 -- and the table above has shapes reading
    // 3e-04, so it cannot have been measured on a build carrying it. The table
    // therefore describes the pre-loop kernel, and its "correct through 676 blocks,
    // wrong from 729" boundary is a still-open question that this one-character fix
    // neither explains nor settles.
    //
    // Both of those are now settled, by measuring the corrected kernel directly on
    // the 6-CU gfx1035 with the fp16_shape_served gate temporarily lifted. It is
    // correct on every shape tried -- rel err ~2.07e-04 (fp32-accumulate noise for
    // an fp16 product) and correlation 1.0000, including all five the old table
    // called WRONG, and including 676 / 784 / 841 / 729 / 5504 blocks, so the
    // boundary above does not exist. Both epilogues and ragged M/N agree. So the
    // q-loop fix is what removed the reported failure, and nothing above this line
    // describes the kernel as it stands.
    //
    // The gate stays shut for speed, not correctness: 2.4-3.2x slower than torch
    // from 1024^3 up, and 1.2x slower even at 512^2. See fp16_shape_served.
    //
    // The 8x8/128x128/BK32 retune above is still unverified. It was rejected on a
    // measurement taken through fp16_linear, which measures the fallback rather
    // than this kernel, so "wrong from 196 blocks" is not established either way.
    // Retuning means re-measuring that tile through _C.fp16_gemm with the gate
    // lifted, not through fp16_linear.
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
            // Both operands are indexed by q in the q loop below. fdot2 pairs
            // element 2q with element 2q+1 of each operand, so A pair q has to meet
            // B pair q. Holding B at pair 0 computes sum_q a[2q..2q+1] * b[0..1]:
            // the right terms in the wrong grouping, one k-pair in four correct and
            // the rest re-reading K/8, K/4 and 3K/8 -- a product with the right
            // distribution, correlation ~1/4 against the true one, and no error.
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
#pragma unroll
                    for (int q = 0; q < 4; ++q)
                        acc[i][j] = __builtin_amdgcn_fdot2(
                            reinterpret_cast<const v2h*>(&av[i])[q],
                            reinterpret_cast<const v2h*>(&bv[j])[q], acc[i][j], true);
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

// v_dot8_i32_i4 is gfx10-only for the same reason: the stub below is what a
// build for another target gets, so the binding has to refuse to launch here
// rather than let this run.
template <typename OutT>
__global__ __launch_bounds__(256) void gemm_int4_valu_kernel(
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

// launch_int4_gemm_kernel is in ops/gemm_int8.hip next to
// launch_int8_gemm_kernel, not here: this header is included by two translation
// units and a plain extern "C" definition would be emitted into both and collide
// at link time. See the note above gemm_fp16_valu_kernel for the same reason the
// kernels here are templates.
