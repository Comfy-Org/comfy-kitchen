// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// The NVFP4 scaled GEMM on RDNA2 (gfx103x), which has no matrix cores.
//
// This is the gfx103x arm of ops/nvfp4.hip's launch_scaled_mm_nvfp4_kernel, and
// the launcher reaches it through the same `comfy_is_gfx10()` test the other
// gfx103x VALU paths use (launch_int4_gemm_kernel, launch_convrot_w4a4_gemm_kernel,
// gemm_gfx10.h). Everything here therefore sits BEHIND that architecture gate: the
// translation unit is compiled for every target this extension builds, the body
// below is compiled only for the gfx103x device pass (__GFX10__), the launcher
// consults the gate per call, and the registry entry that advertises the op is
// itself gated on the RDNA2 half of the architecture manifest (see hip/__init__.py's
// _VALU_GEMM_ARCHS). There is no path by which this file can be entered on a part
// that is not gfx103x, and the !__GFX10__ arm is a trap rather than a plausible
// wrong answer, for the same reason mma.h and gemm_gfx10.h carry traps.
//
// WHY INTEGER AND NOT fp16
// ------------------------
// ops/nvfp4.hip stages the e2m1 nibbles as fp16 with each block scale folded into
// the staged value and runs them through gfx11's fp16 WMMA. None of the three
// ingredients survives onto RDNA2: there is no matrix core, and there is no
// v_dot2_f32_f16 fast enough to stand in for one. On this part gfx10 retires
// v_dot2_f32_f16 at a quarter of v_dot4_i32_i8's rate and moves two MACs per
// instruction against four, so a MAC through the fp16 form costs 8x the dot
// instructions of the same MAC through the integer form. The end-to-end ratio the
// two VALU tiles already measure on this 6-CU part is 5.97 against 1.96 TOPS
// (gemm_gfx10.h), which is the same story told in the only units that matter.
//
// The integer form gives up the fold-into-the-operand trick, and with it the
// single tensor-wise epilogue scale, because an e4m3 scale has no integer form. So
// the scale is applied to the ACCUMULATOR, once per 16-element block -- which is
// possible here and is not under WMMA, and the difference is the whole design.
// Under WMMA a lane's accumulator element spans eight rows (gfx11 acc_row(lane, e)
// = 2e + lane/16) with eight different row scales, so one output element cannot be
// rescaled; ops/nvfp4.hip's note on the integer WMMA path is that experiment, and
// it is why that path measured 2.28x slower rather than merely slower. In a
// register-tiled VALU kernel every accumulator belongs to exactly one (row, col)
// pair, and v_dot4_i32_i8's four products never straddle a block, so a block's
// int32 sum is a whole dot product and can be scaled on its own. The cost is three
// instructions per accumulator per block against four dot instructions, which is
// 1.75x the instruction count of a plain int8 GEMM and nothing at all compared to
// what the fp16 form costs.
//
// The decode is exact end to end. E2M1's magnitudes DOUBLE into {0, 1, 2, 3, 4, 6,
// 8, 12}, every one an exact int8 (nvfp4_utils.h), so the products are integers, a
// 16-element block sums to at most 16 * 144 = 2304, and the float accumulator sees
// exactly one multiply by one e4m3 per block. The 1/4 that both operands' doubling
// contributes rides on the A-side scale, which is a multiply already being done.
//
// TWO KERNELS, AND WHY THE SPLIT IS AT SMALL M
// ---------------------------------------------
// Text-encoder decode is M = 1, and at M = 1 this is a bandwidth problem: the weight
// bytes are read once per token and nothing else about the step is worth measuring.
// A register tile cannot serve that. A BM=64 tile at M = 1 does 64x the multiply-
// accumulate work it needs and -- worse -- re-reads the A side 64 times over, which
// at M = 1 is the entire activation. So the GEMV below is not an optimization of the
// tiled path, it is the only shape that can reach the read bandwidth, and it is what
// carries this workload.
//
// Where the split falls is measured, not argued, and the MMAX ladder is in
// ops/nvfp4.hip's CK_NVFP4_DISPATCH_GFX10: M <= 4 to the GEMV, above that the tile.
// The GEMV stays ahead to M=4 because hoisting the activation window out of the
// column loop (see the note there) makes the per-column cost one weight load, one
// decode, one scale load and MMAX * (4 dot4 + 6 float) -- cheap enough that the
// weight bytes still dominate the time at M=4. It stops there because acc is
// kColsPerWarp x MMAX floats and registers, not bandwidth, are the wall.
//
// The budget says how much arithmetic there is to hide behind it. Decode reads
// 13.7 GB per token. The measured read ceiling on this part is 90 GB/s, not the 78 a
// generic stream reports, so the floor for a decode step is ~150 ms and the whole
// design below is about how close to it the seven GEMMs get. At the time of writing
// the best of them reach 60 GB/s and the worst 11, and the gap is concurrency, not
// arithmetic: the inner loop retires about 0.31 instructions per weight byte and 24
// SIMDs at ~2.4 GHz issue 57.6e9 warp-instructions per second, which at 60 GB/s is
// about one instruction per byte, so the loop spends a third of the issue budget and
// the dots are half a percent of it. Nothing below is shaped by the dot count; it is
// shaped by how many blocks the machine has queued and by getting the weight and
// block-scale bytes to come in as sectors.
//
// The GEMV's shape is the one the int8 GEMV already uses (ops/gemm_int8.hip): a
// wave per output column, lanes striding K. That is what makes the weight reads
// coalesced -- 32 lanes x 16 bytes is 512 contiguous bytes per instruction, sixteen
// whole sectors -- and a warp then walks a run of columns. That 512 is a hard ceiling
// and not a starting point: see kGemvLaneBlocksWide.
//
// COLUMNS PER BLOCK: 32
// ----------------------
// See the note on kGemvCols, which has the measurement. The short version is that this
// file used to size a block at 128 columns so that it would own a whole 512-byte
// block-scale window, and that is wrong on a six-CU part: the grid is worth more than
// the scale locality, four times over, at every N in a decode step.
//
// The tiled kernel above the threshold is gemm_int4_valu_kernel's shape with the
// staging replaced by the decode, and with the scale applied to the accumulator as
// described above. It does not get this treatment: at M above the GEMV's reach the
// scale bytes are 1/(8*BM) of the weight bytes and no longer worth a window's worth
// of thought.
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cstdint>

#include "epilogue.h"
#include "fp8_utils.h"
#include "mma.h"
#include "nvfp4_utils.h"

namespace comfy::hip_backend {

// The GEMV's launch geometry, outside the __GFX10__ arm on purpose: the launcher in
// ops/nvfp4.hip is HOST code, and a host pass never sees __GFX10__, so a constant
// defined inside the device arm is an undeclared identifier at every launch site --
// the same trap CK_NVFP4_DISPATCH_OUT documents for COMFY_MMA_GFX11.
//
// The register budget is the performance argument for this kernel, and it took an
// ISA dump to see it. Left alone the compiler spends 129 VGPRs here -- and gfx10
// allocates VGPRs in granules of 8 out of 512 per SIMD, so 129 rounds up to 136 and
// buys THREE waves a SIMD where 128 would have bought FOUR. One register over a
// line.
//
// The 129 is the compiler prefetching: the inner loop runs s_waitcnt lgkmcnt(14),
// so fourteen 16-byte weight loads are in flight per lane, and that is 56 of the
// registers. It is latency hiding bought with occupancy -- and occupancy is the
// better buy here, because the kernel does not need the registers for anything.
//
// Nothing in this kernel is compute bound. The inner loop issues 161 instructions
// per 512 bytes of weight a wave consumes, which against a measured ceiling of
// 90 GB/s on 24 SIMDs at ~2.4 GHz is a third of the issue budget, and the dots
// themselves are half a percent of it. The weight bytes reach 64% of the ceiling on
// the shapes with enough blocks to fill the machine, and the access pattern is not
// the problem either: measured, a strided read of the same buffer in this exact
// blocked order runs at 90.4 GB/s against 88.1 for one contiguous stream. What is
// left is concurrency.
//
// kGemvThreads is therefore 512, not 256. Doubling the block halves kColsPerWarp,
// which halves the accumulators (kColsPerWarp x MMAX floats) and lets the compiler
// halve its prefetch depth. Measured both ways with everything else fixed: 512 threads
// is ahead on every shape (o_proj -15%, down_proj -9%, gate_proj -3%) and 259 ms a
// token against 241 for 256. It was tried a second time after the register budget
// above had already bought the extra wave, in case the earlier result had been an
// occupancy effect rather than a register effect. It was not.
constexpr int kGemvThreads = 512;

// COLUMNS PER BLOCK: 32, AND WHY THE OBVIOUS 128 IS WRONG HERE
// -----------------------------------------------------------
// nvfp4_scale_offset packs a (row, block) pair into cuBLAS's blocked layout, in which
// one 128-row group's four consecutive block-columns occupy 512 CONTIGUOUS bytes
// indexed (d3, d4, d5) = (row%32, (row/32)%4, block%4). A block owning 128 adjacent
// columns owns every byte of those 512 -- all four d4 groups -- so its scale reads are
// exactly 1 byte per 8 weight bytes, the format's own cost, with no amplification. A
// block owning 32 reads one d4 group: 128 of those 512 bytes, spread across all sixteen
// sectors of the window, so it pulls the whole 512 and uses a quarter of it.
//
// That argument says 128 is better and it is WRONG on this part. Measured, 32 beats
// 128 at every N in the decode step, and the reason is that the grid is what this
// kernel is short of. A COLS=128 block is 128 blocks at N=1024 -- eight blocks on six
// CUs -- and the achieved bandwidth tracks the block count almost perfectly:
//
//     blocks   24    40    64   128   200
//     GB/s    25.2  33.2  44.0  57.9  66.6
//
// a saturation curve that has still not reached its asymptote at N=25600, and the only
// thing that moves it is how many blocks the machine has queued. Trading 1 byte of
// scale per 8 weight bytes for 1 per 16 -- 12.5% more total traffic -- buys four times
// the blocks, and on a part this small that is worth far more than 12.5%.
//
// What the amplification actually costs, measured: nothing visible. Every shape at
// COLS=32 is within noise of COLS=64 and ahead of COLS=128:
//
//     shape        COLS=128   COLS=64   COLS=32
//     q_proj         0.493     0.471     0.463
//     o_proj         0.587     0.504     0.485
//     gate_proj      1.120     1.113     1.094
//     down_proj      1.213     1.112     1.108
//
// So there is one arm and it is the narrow one. COLS=64 was measured too and buys
// nothing over 32, which is the expected shape: past about 128 blocks the curve has
// flattened and more scale locality has nothing left to pay for itself with.
//
// The file used to carry a threshold that picked between 128 and 32 by grid size, on
// the theory that 32 was worth its 12.5% only where 128 could not fill the machine.
// The table above is the refutation: 32 is ahead of 128 where 128 also fills the
// machine. The threshold is gone rather than retuned, because a knob that selects
// between two settings when one of them always loses is a knob that misleads.
constexpr int kGemvCols = 32;

// NVFP4 blocks one lane owns per column per iteration, and therefore how much of a
// column a wave reads per iteration.
//
// 512 CONTINUOUS BYTES A WAVE A STEP IS THE CEILING, AND IT IS A HARD ONE. Tried 4
// blocks a lane, which would give 1024 bytes a step and halve the number of DRAM
// streams a block has open: measured, gate_proj went from 1.105 ms to 1.909 ms, a
// 73% regression, with no register spill to explain it. The reason is that the
// coalescer works per INSTRUCTION. The widest load gfx10 has is 16 bytes a lane, so
// a warp can move 512 contiguous bytes in one instruction and no more; at 4 blocks a
// lane the four 8-byte loads sit at a 32-byte stride, every one of them touches 32
// distinct sectors and uses 8 bytes of each, and the over-fetch is 4x. Widening the
// lane's run does not widen the step, it only spreads it out. So this is 2, and the
// argument above it is kept because it is the reason not to try again.
//
// What the experiment did leave behind is the scale load, which was two
// global_load_ubyte in the ISA where the source said one 16-bit load: the compiler
// cannot prove sb_off is even, so it split it. At 2 blocks a lane the two scale bytes
// are the top or bottom half of one four-block group, so reading the group's dword and
// shifting is one instruction and half the sector lookups, and the evenness comes out
// of the address arithmetic instead of having to be assumed.
constexpr int kGemvLaneBlocksWide = 2;
constexpr int kGemvLaneBlocksNarrow = 2;

#if defined(__GFX10__)

constexpr int kGemvWarps = kGemvThreads / kWave;

// Bytes one wave reads per GEMV iteration, and therefore the elements and NVFP4
// blocks it covers.
template <int LANEB>
constexpr int gemv_iter_bytes() {
    return kWave * LANEB * 8;
}
template <int LANEB>
constexpr int gemv_iter_elems() {
    return gemv_iter_bytes<LANEB>() * 2;
}
template <int LANEB>
constexpr int gemv_iter_blocks() {
    return gemv_iter_elems<LANEB>() / kNvfp4Block;
}
// An iteration must advance a whole number of four-block scale groups, or the group's
// contribution to the scale offset below would not be a whole kScaleGroupStride step.
static_assert(gemv_iter_blocks<kGemvLaneBlocksWide>() % 4 == 0 &&
                  gemv_iter_blocks<kGemvLaneBlocksNarrow>() % 4 == 0,
              "an iteration must advance a whole number of four-block scale groups");
// A lane's blocks must fit inside ONE four-block scale group, so that they are one
// dword read rather than LANEB byte reads, and so that (kb_local & 3) is constant
// across the lane's run. 4 is the largest that fits.
static_assert(kGemvLaneBlocksWide <= 4 && kGemvLaneBlocksNarrow <= 4 &&
                  (4 % kGemvLaneBlocksWide) == 0 && (4 % kGemvLaneBlocksNarrow) == 0,
              "a lane's blocks must fit inside one four-block scale group");
// And a lane's run must be exactly one 16-byte load, which is the widest gfx10 has and
// the only width at which the warp's accesses coalesce. See the note above.
static_assert(kGemvLaneBlocksWide * 8 <= 16 && kGemvLaneBlocksNarrow * 8 <= 16,
              "a lane's run must fit in one 16-byte load to coalesce");

// K elements of the activation staged in LDS per GEMV chunk, as a whole number of
// iterations' worth at this lane count. The staged bytes are MMAX * KC, held well
// inside the 64 KB a WGP has. Staging the activation DECODED is what pays for itself:
// every warp reads the same staged bytes for every one of the block's COLS columns,
// so the nibble decode is paid once per column-block rather than once per column, and
// so is the A-side block-scale read.
//
// MMAX=1 gets 8192 rather than 4096 because it is the only one that can afford it and
// it is the only one that matters: decode is M = 1 and K = 5120, which is 320 NVFP4
// blocks, so a 4096-element chunk is TWO chunks -- two staging passes and two barriers
// for a weight buffer of 2.6 MB. Every one of those is paid per block, and on the
// narrow shapes there are only 32 blocks, so the machine spends most of the launch
// draining one block's barrier into the next block's prologue. 8192 makes K=5120 a
// single chunk. At MMAX=2 the staged activation doubles to 16 KB and the second chunk
// comes back, which is the right trade: M=2 is 6% of a step and M=1 is 94%.
template <int MMAX, int LANEB>
constexpr int gemv_kc() {
    constexpr int one = gemv_iter_elems<LANEB>();
    constexpr int base = (MMAX == 1) ? 8192 : ((MMAX <= 2) ? 4096 : 2048);
    return base < one ? one : ((base / one) * one);
}

// Four signed int8 products accumulated into one int32. gfx10 spells this
// v_dot4c_i32_i8 and it is the reason the whole kernel is behind __GFX10__:
// clang rejects it with "needs target feature dot1-insts" on every other target,
// while gfx11/gfx12 have v_dot4_i32_iu8 with the opposite signedness spelling.
__forceinline__ __device__ int nvfp4_dot4(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, true);
}

// The LANEB*8 bytes at `p` covering NVALS whole NVFP4 blocks, zero-filled past them.
// Only the last iteration of the last chunk ever calls it with NVALS < LANEB, and the
// zeros are what let the rest of the loop stay unconditional: the staged activation is
// zero past the same point, so the products are zero and the block's scale can be
// anything finite.
// The LANEB 8-byte NVFP4 blocks at `p`, of which only NVALS are real, the rest
// zero-filled. Only the last iteration of the last chunk ever has NVALS < LANEB, and
// the zeros are what let the rest of the loop stay unconditional: the staged
// activation is zero past the same point, so the products are zero and the block's
// scale can be anything finite.
//
// There is no half-block case and there must not be one. A block is 8 packed bytes
// whole, nvals counts whole blocks, so a block is either there or it is not -- reading
// a short piece of one would be reading past the end of the weight buffer on the last
// block of the last column.
template <int LANEB>
__forceinline__ __device__ void nvfp4_load_blocks(const uint8_t* p, int nvals, uint2* out) {
    #pragma unroll
    for (int q = 0; q < LANEB; ++q) {
        out[q] = (q < nvals) ? *reinterpret_cast<const uint2*>(p + q * 8)
                             : make_uint2(0u, 0u);
    }
}

// ---------------------------------------------------------------------------
// GEMV: C[M, N] = (a @ b.T) * (tsa * tsb) + bias[n], for M <= MMAX
// ---------------------------------------------------------------------------
//
// Layout and the math are the module's: a is (M, K/2) and b is (N, K/2) packed
// nibbles, high nibble = even column, and sa/sb are the e4m3 block scales in the
// blocked layout nvfp4_scale_offset indexes.
// The second __launch_bounds__ argument is here because the register budget above
// is, and it is the form LLVM actually honours on gfx10: the first argument alone
// lets the compiler pick, and it picks 129. (An explicit min-blocks-per-SIMD of 2
// is silently ignored by the AMDGPU backend here -- it left vgpr_count at 129 --
// so the budget has to be bought in the source rather than requested.)
template <typename OutT, int MMAX, int COLS, int LANEB>
__global__ __launch_bounds__(kGemvThreads) void nvfp4_gemv_valu_kernel(
    const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
    const uint8_t* __restrict__ sa, const uint8_t* __restrict__ sb,
    OutT* __restrict__ C, int M, int N, int K, int a_blocks, int b_blocks, int ldc,
    EpiTensorwise epi) {

    constexpr int KC = gemv_kc<MMAX, LANEB>();
    constexpr int kChunks = KC / kNvfp4Block;  // NVFP4 blocks staged per chunk
    constexpr int kColsPerWarp = COLS / kGemvWarps;
    static_assert(COLS % kGemvWarps == 0, "the warps must divide the columns");

    __shared__ __align__(16) uint16_t lut[kE2m1I8ByteLutSize];
    // The activation, decoded: one int8 per element, in the SAME pair-swapped order
    // the weight decode produces (both go through e2m1x2_to_i8x4, which is what
    // makes v_dot4 pair matching k values -- see nvfp4_utils.h).
    __shared__ __align__(16) int8_t Ads[MMAX * KC];
    // Its block scales, already widened to float with the 1/4 that undoes the
    // doubling folded in, so the inner loop's first multiply is the only work the
    // A-side scale costs. Indexed [m][block within the chunk].
    __shared__ __align__(16) float Asc[MMAX * kChunks];

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & (kWave - 1);
    const int warp = tid / kWave;

    e2m1_init_i8_byte_lut(lut, tid, kGemvThreads);
    // The barrier the table needs and does not get for free, and it is needed in BOTH
    // kernels rather than in one of them. Both DECODE during staging -- that is the
    // whole reason their staged operands are usable as int8 at all -- so both read
    // lut[] before any barrier, and a thread can index an entry another thread has not
    // written yet and silently decode zeros. nvfp4_gemm_kernel does not need it
    // because its stager defers the decode to store().
    //
    // The two failures look nothing alike, which is what makes this worth writing down:
    // in the GEMV the staging covers the whole row one block at a time, so most of
    // what a thread wants was written by a DIFFERENT thread and the output was zero
    // almost everywhere, which reads as an indexing bug rather than a race. In the
    // tile it was 1 launch in 30 and depended on which warps were resident, which
    // reads as "it passes" -- the worst shape a race can have. Missing barrier in the
    // tile: __syncthreads() right here.
    __syncthreads();

    const int kbytes = K / 2;
    const int col_base = static_cast<int>(blockIdx.x) * COLS;
    const int cols = min(COLS, N - col_base);

    float acc[kColsPerWarp][MMAX] = {};

    for (int kb_base = 0; kb_base < a_blocks; kb_base += kChunks) {
        const int nblk = min(kChunks, a_blocks - kb_base);

        // Stage this chunk of the activation and its scales. Rows at or past M stage
        // as zero rather than not at all: the compute loop below runs a compile-time
        // MMAX rows so that acc stays in registers, which means it also reads these
        // buffers for a row that is not there, and an unwritten float could be a NaN
        // and 0 * NaN is NaN.
        for (int m = 0; m < MMAX; ++m) {
            const bool live = m < M;
            int8_t* adst = Ads + m * KC;
            float* asdst = Asc + m * kChunks;
            for (int j = tid; j < kChunks; j += kGemvThreads) {
                const int kb = kb_base + j;
                uint2 raw = make_uint2(0u, 0u);
                if (live && kb < a_blocks) {
                    raw = *reinterpret_cast<const uint2*>(
                        A + static_cast<int64_t>(m) * kbytes + kb * 8);
                }
                // One NVFP4 block is 8 packed bytes = 4 dwords of int8, and dword t of
                // a block holds elements [4t, 4t+4), which is the order the GEMV's
                // activation window is staged in.
                uint32_t d[4];
                d[0] = e2m1x2_to_i8x4(raw.x & 0xFFFFu, lut);
                d[1] = e2m1x2_to_i8x4(raw.x >> 16, lut);
                d[2] = e2m1x2_to_i8x4(raw.y & 0xFFFFu, lut);
                d[3] = e2m1x2_to_i8x4(raw.y >> 16, lut);
                *reinterpret_cast<uint4*>(adst + j * kNvfp4Block) =
                    make_uint4(d[0], d[1], d[2], d[3]);
                asdst[j] = (live && kb < a_blocks)
                               ? kNvfp4Double * fp8_to_float(
                                     sa[nvfp4_scale_offset(m, kb, a_blocks)])
                               : 0.0f;
            }
        }
        // One barrier covers the table, the activation and its scales: everything
        // staged above is read only below it.
        __syncthreads();

        // `it` outside and the column loop inside, so the COLS/kGemvWarps weight loads a
        // warp issues at one iteration are independent of each other and of the ones
        // after them. That is the memory-level parallelism this kernel has, and it is
        // why the loop is written this way round.
        //
        // Ceiling, not exact division. An iteration always covers gemv_iter_blocks
        // blocks because its 32 lanes hold LANEB each, and the lanes past the end of a
        // short chunk must still run, masked, or they would contribute nothing and the
        // chunk's real lanes would have nothing to reduce with. So the trip count is
        // ceil(nblk / iter_blocks) and at least 1: `it * iter_blocks < nblk` is zero
        // trips for any nblk under one iteration, which for the K below one chunk is
        // the whole chunk and silently produced an all-zero output.
        const int iters =
            (nblk + gemv_iter_blocks<LANEB>() - 1) / gemv_iter_blocks<LANEB>();
        #pragma unroll 1
        for (int it = 0; it < iters; ++it) {
            // Warp-uniform: false only on the last iteration of the last chunk, and
            // only when K is not a whole number of iterations.
            const bool full = (it + 1) * gemv_iter_blocks<LANEB>() <= nblk;
            const int kb_local = it * gemv_iter_blocks<LANEB>() + lane * LANEB;
            const int elem_off = it * gemv_iter_elems<LANEB>() + lane * LANEB * 16;

            // THE ACTIVATION WINDOW, HOISTED OUT OF THE COLUMN LOOP.
            //
            // The window is LANEB*16 int8 at this lane's slice of K -- it does NOT
            // depend on the column, and the loop below runs kColsPerWarp columns over
            // it. Reading it inside that loop, as the obvious spelling does, issues
            // the same two ds_read_b128 and the same scale reads once per column: at
            // MMAX=1, kColsPerWarp=2 that is 6 LDS reads per 16 weight bytes where 2
            // will do, and LDS -- not the global loads -- becomes the thing standing
            // between this kernel and the read bandwidth.
            //
            // The cost is MMAX * LANEB * 4 dwords of register (8 at MMAX=1, 32 at
            // MMAX=4), held across the column loop. That is what buys the GEMV its
            // reach above M=1: the per-column cost is one weight load, one decode, one
            // scale load and MMAX * (4 dot4 + 6 float) per NVFP4 block, which the
            // weight bytes still dominate through MMAX=4. The issue budget is not what
            // stops the ladder -- registers are. acc is kColsPerWarp x MMAX floats on
            // top of this window. See the crossover table in ops/nvfp4.hip's
            // CK_NVFP4_DISPATCH_GFX10.
            uint32_t av[MMAX][LANEB * 4];
            float asc[MMAX][LANEB];
            #pragma unroll
            for (int m = 0; m < MMAX; ++m) {
                const int8_t* arow = Ads + m * KC + elem_off;
                #pragma unroll
                for (int q = 0; q < LANEB; ++q) {
                    const uint4 t = *reinterpret_cast<const uint4*>(arow + q * 16);
                    av[m][q * 4 + 0] = t.x;
                    av[m][q * 4 + 1] = t.y;
                    av[m][q * 4 + 2] = t.z;
                    av[m][q * 4 + 3] = t.w;
                    asc[m][q] = Asc[m * kChunks + kb_local + q];
                }
            }

            #pragma unroll
            for (int ci = 0; ci < kColsPerWarp; ++ci) {
                const int c = warp + ci * kGemvWarps;
                if (c >= cols) continue;
                const int col = col_base + c;
                const uint8_t* __restrict__ b = B + static_cast<int64_t>(col) * kbytes;
                // Block scales for this column, for the lane's LANEB blocks.
                //
                // Inside a four-block group consecutive blocks are one byte apart and
                // groups are kScaleGroupStride apart. The lane's blocks are kb_local
                // and kb_local+1, and kb_local is even, so (kb_local & 3) is 0 or 2 --
                // either way the pair is the bottom or the top half of ONE group. So
                // the whole group is one dword read and the half is a shift, which is
                // also what stops the compiler splitting a 16-bit source load into two
                // global_load_ubyte: it cannot prove sb_off is even, and the group
                // dword is even by construction rather than by assumption.
                //
                // All THREE terms are needed and dropping any of them fails SILENTLY, which
                // is what makes this worth writing out. nvfp4_scale_offset(row, kb)
                // decomposes kb into a group index kb >> 2 and a byte-within-group
                // kb & 3, so the offset for the lane's blocks is
                //
                //   base(col, kb_base)                 the column and the chunk base
                // + (kb_local >> 2) * kScaleGroupStride   which GROUP inside the chunk
                // + (kb_local & 3)                        which byte inside the group
                //
                // and the middle term is the one that is easy to leave out, because
                // iter_blocks / 4 alone accounts for the *iteration's* groups and
                // looks like it accounts for the lane's. It does not: lane L's blocks
                // sit in group L*LANEB/4 of the iteration, so the warp's 32 lanes read
                // 32 different windows 512 bytes apart. Without the term the whole warp
                // reads a quarter of ONE window, i.e. one quarter of the scale bytes it
                // needs from the wrong place -- which scales most blocks by a
                // neighbouring e4m3 and is a ~3e-1 relative error rather than an
                // obvious one. That was the whole of the error here.
                const int sb_off = nvfp4_scale_offset(col, kb_base, b_blocks) +
                                   (kb_local >> 2) * kScaleGroupStride + (kb_local & 3);
                const int kbyte = (kb_base + kb_local) * 8;
                const int nvals = min(LANEB, nblk - kb_local);

                uint2 raw[LANEB];
                uint32_t sbraw = 0;
                if (full) {
                    #pragma unroll
                    for (int q = 0; q < LANEB; ++q) {
                        raw[q] = *reinterpret_cast<const uint2*>(b + kbyte + q * 8);
                    }
                    // The lane's LANEB scale bytes live in ONE four-block group: every
                    // term of sb_off above is a multiple of 4 except (kb_local & 3), and
                    // kb_local is even, so (kb_local & 3) is 0 or 2 and the pair is
                    // either half of a group. Reading the group's dword and shifting is
                    // one instruction and one sector lookup instead of the two
                    // global_load_ubyte the compiler produced from a 16-bit source load
                    // it could not prove aligned.
                    //
                    // It also reads two bytes this lane does not need, and that is only
                    // safe because of an invariant in the binding rather than in this
                    // file: nvfp4_scale_elems rounds BOTH extents up (rows to 128,
                    // blocks to 4), so a group whose last two blocks are past K still
                    // has its four bytes allocated. Change that rounding and this
                    // becomes a two-byte overread on the last row of the last column,
                    // for every K that is 32 (mod 64) -- which is not a shape anyone
                    // would test.
                    const int sb_grp = sb_off & ~3;
                    const int sb_idx = sb_off & 3;
                    sbraw = *reinterpret_cast<const uint32_t*>(sb + sb_grp);
                    sbraw >>= 8 * sb_idx;
                } else {
                    nvfp4_load_blocks<LANEB>(b + kbyte, nvals, raw);
                    #pragma unroll
                    for (int j = 0; j < LANEB; ++j) {
                        if (j < nvals) {
                            sbraw |= static_cast<uint32_t>(sb[sb_off + j]) << (8 * j);
                        }
                    }
                }

                // One NVFP4 block is 8 packed bytes = 4 dwords of int8, and dword t of
                // a block holds elements [4t, 4t+4) on BOTH sides -- which is the
                // order the activation window above is staged in.
                uint32_t d[LANEB][4];
                #pragma unroll
                for (int q = 0; q < LANEB; ++q) {
                    d[q][0] = e2m1x2_to_i8x4(raw[q].x & 0xFFFFu, lut);
                    d[q][1] = e2m1x2_to_i8x4(raw[q].x >> 16, lut);
                    d[q][2] = e2m1x2_to_i8x4(raw[q].y & 0xFFFFu, lut);
                    d[q][3] = e2m1x2_to_i8x4(raw[q].y >> 16, lut);
                }

                #pragma unroll
                for (int m = 0; m < MMAX; ++m) {
                    #pragma unroll
                    for (int q = 0; q < LANEB; ++q) {
                        int p = 0;
                        #pragma unroll
                        for (int t = 0; t < 4; ++t) {
                            p = nvfp4_dot4(av[m][q * 4 + t], d[q][t], p);
                        }
                        const float sq = fp8_to_float(
                            static_cast<uint8_t>((sbraw >> (8 * q)) & 0xFFu));
                        acc[ci][m] = nvfp4_block_acc(acc[ci][m], p, asc[m][q], sq);
                    }
                }
            }
        }
        __syncthreads();
    }

    epi.init();
    #pragma unroll
    for (int ci = 0; ci < kColsPerWarp; ++ci) {
        const int c = warp + ci * kGemvWarps;
        if (c >= cols) continue;
        const int col = col_base + c;
        const float bias = epi.col_bias(col);
        // epi.row_scale(col) is the tensor-wise alpha (EpiTensorwise::row_scale
        // ignores the row and returns init()'s scale_a[0]*scale_b[0]) and
        // epi.col_scale is 1.0f for the same policy -- both are hoisted out of the
        // loop above because the store goes through OutT*, which the compiler must
        // assume aliases the scale pointer, so every load in the loop would be
        // re-issued per output element. Passing 1.0f here instead is a silent
        // ~1e5x under-report of the result, which is what the first run of
        // nvfp4_gfx10_check.py caught.
        const float rs = epi.row_scale(0);
        const float cs = epi.col_scale(0);
        // THE CROSS-LANE REDUCTION
        // -----------------------
        // The compute loop gave every lane of the wave a different slice of K -- lane L
        // holds blocks L*2 and L*2+1 of the iteration -- so all 32 lanes hold a PARTIAL
        // sum for the SAME output element. Without a reduction they all write to the same
        // C[m * ldc + col], the store is a plain store, and the answer is whichever lane
        // retires last; a lane holding only zeros makes the common case exactly zero.
        // That is a data race whose symptom is "sometimes right, usually zero", which is
        // why it is worth stating rather than leaving as an apparent bug in the store.
        //
        // __shfl_xor with every offset folds all 32 lanes' values into all 32, so the
        // result is broadcast and any lane may store it. Only ONE lane needs to, and
        // doing that from lane 0 is what the int8 GEMV does (gemm_int8.hip:38): on this
        // part a 128-column block would otherwise issue 32 stores to each of 128
        // columns where 128 will do.
        //
        // The shuffle must be OUTSIDE the lane guard. __shfl_xor reads the partner lane's
        // register, so a "lane 0 only" reduction would fold in whatever the inactive
        // lanes' registers happen to hold rather than the wave's total. All 32 lanes run
        // it, all 32 come out with the same sum, and only the store is guarded.
        //
        // The reduction is over kWave rather than a literal, because this part is built
        // -mno-wavefrontize64 and kWave is 32 here but is not 32 everywhere mma.h is
        // included -- the same reason wave_reduce_sum takes it from there.
        float sum[MMAX];
        #pragma unroll
        for (int m = 0; m < MMAX; ++m) {
            if (m >= M) continue;
            sum[m] = wave_reduce_sum(acc[ci][m]);
        }
        if (lane == 0) {
            #pragma unroll
            for (int m = 0; m < MMAX; ++m) {
                if (m >= M) continue;
                C[static_cast<int64_t>(m) * ldc + col] =
                    static_cast<OutT>(epi(m, col, sum[m], rs, cs, bias));
            }
        }
    }
}

// ===========================================================================
// Tiled VALU GEMM, for M above the GEMV's reach
// ===========================================================================
//
// BM x BN x BK, TM x TN per thread, THREADS threads, double buffered. The tile
// geometry and the staging shapes are gemm_int8_valu_kernel's, which is the one
// measured on this part (5.97 TOPS at 16x4/128x128, and the tile choice there is
// read off the register file rather than swept). What changes is the staging --
// packed nibbles in, decoded int8 out -- and the compute loop, which applies each
// block's scales to its accumulator instead of folding them into the operand.
//
// BK is in ELEMENTS and must be a whole number of 16-element blocks, so BK/16 is
// the number of chunks a staged row is cut into; the WMMA path's BKB is bytes and
// its kSteps come from the MMA policy, which is exactly the difference between the
// two.
template <typename OutT, int BM, int BN, int BK, int TM, int TN, int THREADS>
__global__ __launch_bounds__(THREADS) void nvfp4_gemm_valu_kernel(
    const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
    const uint8_t* __restrict__ sa, const uint8_t* __restrict__ sb,
    OutT* __restrict__ C, int M, int N, int K, int a_blocks, int b_blocks, int ldc,
    EpiTensorwise epi) {

    constexpr int kChunks = BK / kNvfp4Block;  // NVFP4 blocks per staged row
    constexpr int kBuffers = 2;
    constexpr int kThreads = THREADS;
    static_assert(BK % kNvfp4Block == 0, "BK must be a whole number of NVFP4 blocks");
    static_assert((BM / TM) * (BN / TN) == kThreads,
                  "the thread tile must cover the block exactly");

    __shared__ __align__(16) uint16_t lut[kE2m1I8ByteLutSize];
    __shared__ __align__(16) int8_t A_buf[kBuffers][BM * BK];
    // Chunk-major, exactly as gemm_int8_valu_kernel's B: it is a layout change
    // rather than a padding change that avoids the bank conflict (padding only
    // trades 8-way for 4-way there), and it costs nothing in pitch.
    __shared__ __align__(16) int8_t B_buf[kBuffers][kChunks * BN * kNvfp4Block];
    // Block scales for the staged tile, widened once during staging with the A-side
    // carrying the 1/4 that undoes the doubling. As floats rather than bytes because
    // the compute loop reads TM + TN of them per chunk: converting there instead
    // would be a conversion per read rather than per staged element, and this kernel
    // is close enough to the issue limit that the trade is not free.
    __shared__ __align__(16) float Asc[kBuffers][BM * kChunks];
    __shared__ __align__(16) float Bsc[kBuffers][BN * kChunks];

    static_assert(kBuffers * (BM * BK + kChunks * BN * kNvfp4Block) +
                          kBuffers * (BM + BN) * kChunks * (int)sizeof(float) <=
                      64 * 1024,
                  "LDS footprint exceeds 64 KiB and will spill to local");

    const int tid = static_cast<int>(threadIdx.x);
    // TX puts a whole warp inside one row group of the thread tile, which is what
    // makes the A-side LDS reads broadcasts.
    constexpr int TX = BN / TN;
    const int trow = tid / TX;
    const int tcol = tid % TX;
    const int orow = trow * TM;
    const int ocol = tcol * TN;

    const int m0 = static_cast<int>(blockIdx.y) * BM;
    const int n0 = static_cast<int>(blockIdx.x) * BN;
    const int kbytes = K / 2;

    e2m1_init_i8_byte_lut(lut, tid, kThreads);
    // The barrier the table needs and does not get for free. See the note on the same
    // barrier in nvfp4_gemv_valu_kernel: both kernels decode during staging, both read
    // lut[] before any barrier, and here it showed up as 1 launch in 30 at M=40/N=8192
    // rather than as a consistent wrong answer.
    __syncthreads();

    float acc[TM][TN] = {};

    // Load one BK-deep slice of both operands, decoded, plus their block scales.
    // Out-of-range rows and the K tail stage as zero, which is what the compute loop
    // needs: a zeroed row contributes nothing. The scales are zeroed with them rather
    // than skipped, because the compute loop reads TM + TN of them per chunk
    // unconditionally and an unwritten float could be a NaN, and 0 * NaN is NaN.
    auto load_tile = [&](int buf, int k0) {
        const int kb0 = k0 / kNvfp4Block;
        for (int i = tid; i < BM * kChunks; i += kThreads) {
            const int r = i / kChunks;
            const int c = i - r * kChunks;
            const int kb = kb0 + c;
            const bool live = (m0 + r < M) && (kb < a_blocks);
            uint2 raw = make_uint2(0u, 0u);
            if (live) {
                raw = *reinterpret_cast<const uint2*>(
                    A + static_cast<int64_t>(m0 + r) * kbytes + kb * 8);
            }
            uint32_t d[4];
            d[0] = e2m1x2_to_i8x4(raw.x & 0xFFFFu, lut);
            d[1] = e2m1x2_to_i8x4(raw.x >> 16, lut);
            d[2] = e2m1x2_to_i8x4(raw.y & 0xFFFFu, lut);
            d[3] = e2m1x2_to_i8x4(raw.y >> 16, lut);
            int8_t* dst = A_buf[buf] + r * BK + c * kNvfp4Block;
            *reinterpret_cast<uint4*>(dst) = make_uint4(d[0], d[1], d[2], d[3]);
            Asc[buf][r * kChunks + c] =
                live ? kNvfp4Double *
                            fp8_to_float(sa[nvfp4_scale_offset(m0 + r, kb, a_blocks)])
                      : 0.0f;
        }
        for (int i = tid; i < BN * kChunks; i += kThreads) {
            const int r = i / kChunks;
            const int c = i - r * kChunks;
            const int kb = kb0 + c;
            const bool live = (n0 + r < N) && (kb < b_blocks);
            uint2 raw = make_uint2(0u, 0u);
            if (live) {
                raw = *reinterpret_cast<const uint2*>(
                    B + static_cast<int64_t>(n0 + r) * kbytes + kb * 8);
            }
            uint32_t d[4];
            d[0] = e2m1x2_to_i8x4(raw.x & 0xFFFFu, lut);
            d[1] = e2m1x2_to_i8x4(raw.x >> 16, lut);
            d[2] = e2m1x2_to_i8x4(raw.y & 0xFFFFu, lut);
            d[3] = e2m1x2_to_i8x4(raw.y >> 16, lut);
            // Chunk-major: chunk c of row r at (c * BN + r), which is the layout
            // that takes the B reads conflict-free without a padding term.
            int8_t* dst = B_buf[buf] + (c * BN + r) * kNvfp4Block;
            *reinterpret_cast<uint4*>(dst) = make_uint4(d[0], d[1], d[2], d[3]);
            Bsc[buf][r * kChunks + c] =
                live ? fp8_to_float(sb[nvfp4_scale_offset(n0 + r, kb, b_blocks)]) : 0.0f;
        }
    };

    #pragma unroll 1
    for (int p = 0; p < kBuffers - 1; ++p) {
        if (p * BK < K) load_tile(p, p * BK);
    }
    __syncthreads();

    constexpr int kPrefetch = kBuffers - 1;
    int buf = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
        const int nb = 1 - buf;
        const int kp = k0 + kPrefetch * BK;
        // Issue the next slice's global reads before the current slice's math; the
        // barrier between them is what makes that safe.
        if (kp < K) load_tile(nb, kp);
        {
            // One 16-element block per step. Four v_dot4 consume it, and the block
            // scale is applied to the accumulator once, here.
            #pragma unroll 1
            for (int c = 0; c < kChunks; ++c) {
                uint32_t av[TM][4];
                uint32_t bv[TN][4];
                #pragma unroll
                for (int i = 0; i < TM; ++i) {
                    const uint4 v = *reinterpret_cast<const uint4*>(
                        A_buf[buf] + (orow + i) * BK + c * kNvfp4Block);
                    av[i][0] = v.x; av[i][1] = v.y; av[i][2] = v.z; av[i][3] = v.w;
                }
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    const uint4 v = *reinterpret_cast<const uint4*>(
                        B_buf[buf] + (c * BN + ocol + j) * kNvfp4Block);
                    bv[j][0] = v.x; bv[j][1] = v.y; bv[j][2] = v.z; bv[j][3] = v.w;
                }
                #pragma unroll
                for (int i = 0; i < TM; ++i) {
                    const float sa_sc = Asc[buf][(orow + i) * kChunks + c];
                    #pragma unroll
                    for (int j = 0; j < TN; ++j) {
                        int p = 0;
                        #pragma unroll
                        for (int t = 0; t < 4; ++t) p = nvfp4_dot4(av[i][t], bv[j][t], p);
                        acc[i][j] = nvfp4_block_acc(
                            acc[i][j], p, sa_sc, Bsc[buf][(ocol + j) * kChunks + c]);
                    }
                }
            }
        }
        __syncthreads();
        buf = nb;
    }

    epi.init();
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int row = m0 + orow + i;
        if (row >= M) continue;
        const float rs = epi.row_scale(row);
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int col = n0 + ocol + j;
            if (col >= N) continue;
            C[static_cast<int64_t>(row) * ldc + col] = static_cast<OutT>(
                epi(row, col, acc[i][j], rs, epi.col_scale(col), epi.col_bias(col)));
        }
    }
}

#else  // !__GFX10__

// v_dot4_i32_i8 is gfx1030-1036 only and clang rejects it everywhere else, so the
// kernels above have no body on any other target. The launcher reaches them behind
// comfy_is_gfx10(), which is the same arrangement gemm_gfx10.h and mma.h use, and
// these traps are what make a direct call on the wrong device a trap rather than a
// plausible wrong answer. Templated for the same reason: a plain definition here
// would be emitted into several objects and collide at link time.
// The trap stub takes kGemvThreads rather than a literal, because the launcher
// passes exactly that and a launch_bounds below it is a launch failure rather than
// the trap -- the one thing these stubs are here to be.
template <typename OutT, int MMAX, int COLS, int LANEB>
__global__ __launch_bounds__(kGemvThreads) void nvfp4_gemv_valu_kernel(
    const uint8_t*, const uint8_t*, const uint8_t*, const uint8_t*, OutT*, int, int, int,
    int, int, int, EpiTensorwise) {
    __builtin_trap();
}

template <typename OutT, int BM, int BN, int BK, int TM, int TN, int THREADS>
__global__ __launch_bounds__(THREADS) void nvfp4_gemm_valu_kernel(
    const uint8_t*, const uint8_t*, const uint8_t*, const uint8_t*, OutT*, int, int, int,
    int, int, int, EpiTensorwise) {
    __builtin_trap();
}

#endif  // __GFX10__

}  // namespace comfy::hip_backend
