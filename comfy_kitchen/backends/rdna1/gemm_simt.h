// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Thread-level GEMM for gfx1010 (RDNA1), which has neither matrix cores nor
// dot instructions.
//
// The software MMA policy in mma.h emulates a WMMA fragment contract: every MMA
// broadcasts the A row eight times across the wave, so most of its time goes to
// cross-lane shuffles rather than arithmetic. This kernel is the classic
// register-blocked form instead. Each thread owns a TM x TN block of C, LDS holds
// K-major 32-bit words so a thread reads its rows and columns with aligned 16-byte
// loads, and each word it reads is unpacked once and then used TN (or TM) times.
//
// The word-level dot is plain arithmetic:
//   int8  sign-extended bytes into v_mad_i32_i24.
//   int4  sign-extended nibbles, half a word at a time, into 24-bit multiplies.
//   fp8   e4m3 is re-encoded as fp16 (exact, see SimtFp8) once, as the tile is
//         stored to LDS, then v_fma_mix_f32.
//
// Measured against the software policy at M=4096, N=K=3072, the int8 / fp8 / int4
// GEMMs went from 0.69 / 0.22 / 0.63 to 4.7 / 5.3 / 4.8 TOPS.
//
// Short M (M <= 48) packs its rows into 16- or 32-row tiles, which spend the
// threads on columns rather than on rows past M, and splits K across blocks when
// that leaves too few of them (see launch_gemm_simt). Against the 64-row tiles
// this measured 1.4-3x faster for M <= 32 at N >= 3072 or K >= 3072, and about
// even on the smallest shapes (N=1024 with K=768).
//
// Computes C[M, N] = epilogue(A[M, K] @ B[N, K]^T) with the operands in their
// natural row-major form and C written with row stride ldc.
// kbytes, the row length in bytes, must be a multiple of 16.
//
// Fetching the operands, not arithmetic, bounds the packed GEMM on gfx1010: a probe of
// its 8x8 block ran 10.6 TFLOPS with no global loads and 5.6-7 with row-major ones. Two
// options cut that cost:
//   TILED  A is stored a 128-row tile's stage at a time (see simt_tiled_offset), so the
//          128 rows' worth of K a block stages is one contiguous run of global memory
//          rather than 128 pieces a row length apart.
//   BI8    B is int8, row-major with kbytes / 2 bytes a row, and is widened to the Op's
//          fp16 pairs as it is stored to LDS (Op::expand_i8): half the bytes fetched, and
//          an INT8 weight is read where it lies with no fp16 copy.
//
// A may instead be gathered (ASrc, an implicit-GEMM convolution's patch rows): the
// source resolves each of a thread's rows once, ASrc::row(grow, valid), and each
// stage of 64 K bytes once, ASrc::stage(kbyte0), and ASrc::chunk(row, stage, offset)
// then returns the 16 bytes at that offset into the stage. kbytes must then be a
// multiple of 64, so no stage runs past the end of a row.
#pragma once

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "gemm_common.h"

namespace comfy::hip_backend {

// Device pass only; the host pass compiles the kernels' bodies out.
#if defined(__gfx1010__)
#define COMFY_SIMT_GEMM 1
#endif
// K words unrolled per LDS stage. gfx1010 spills the 8x8 tile's registers from 2 up.
#define COMFY_SIMT_UNROLL 1

// ---------------------------------------------------------------------------
// Operand policies. A policy says how a 32-bit word of K from global memory (4
// int8 or fp8 elements, 8 int4) is kept in LDS and multiplied:
//   expand   runs once per word per block, at the LDS store, and writes kExpand
//            LDS words. Work done here is shared by the 16 threads that read the
//            word, so conversions belong here rather than in unpack.
//   unpack   turns one LDS word into a Frag, in kSubs pieces, per thread per use.
//   dot      accumulates one Frag pair into Acc.
//   finish   turns the accumulator into the epilogue's float.
// ---------------------------------------------------------------------------

// c += a * b for operands known to fit 24 bits, as one v_mad_i32_i24. Left to
// itself the compiler folds the byte extraction into v_mul_i32_i24_sdwa and sums
// with v_add3_u32, about 1.75 instructions a product against this 1 plus the
// extraction, which unpack's callers share across TN (or TM) uses: measured 17%
// faster for int8.
__forceinline__ __device__ int mad_i24(int a, int b, int c) {
    asm("v_mad_i32_i24 %0, %1, %2, %0" : "+v"(c) : "v"(a), "v"(b));
    return c;
}

struct SimtInt8 {
    using Acc = int;
    static constexpr int kExpand = 1;
    static constexpr int kSubs = 1;
    static constexpr bool kPack16 = false;  // see launch_gemm_simt
    static __forceinline__ __device__ void expand(uint32_t w, uint32_t out[kExpand]) { out[0] = w; }
    struct Frag {
        int v[4];
    };
    static __forceinline__ __device__ Frag unpack(uint32_t w, int) {
        Frag f;
#pragma unroll
        for (int i = 0; i < 4; ++i) f.v[i] = static_cast<int>(w << (24 - 8 * i)) >> 24;
        return f;
    }
    static __forceinline__ __device__ int dot(const Frag& a, const Frag& b, int c) {
#pragma unroll
        for (int i = 0; i < 4; ++i) c = mad_i24(a.v[i], b.v[i], c);
        return c;
    }
    static __forceinline__ __device__ float finish(int c) { return static_cast<float>(c); }
};

// Signed int4, two per byte with the low nibble first, so element e of a word sits
// at bits 4e..4e+3 of both operands. Eight nibbles in two halves of four keep the
// unpacked registers per word at int8's count.
struct SimtInt4 {
    using Acc = int;
    static constexpr int kExpand = 1;
    static constexpr int kSubs = 2;
    static constexpr bool kPack16 = true;  // see launch_gemm_simt
    static __forceinline__ __device__ void expand(uint32_t w, uint32_t out[kExpand]) { out[0] = w; }
    struct Frag {
        int v[4];
    };
    static __forceinline__ __device__ Frag unpack(uint32_t w, int sub) {
        Frag f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            f.v[i] = static_cast<int>(w << (28 - 4 * (4 * sub + i))) >> 28;
        }
        return f;
    }
    static __forceinline__ __device__ int dot(const Frag& a, const Frag& b, int c) {
#pragma unroll
        // Unlike int8, the compiler's own form measured faster here than mad_i24
        // (nibbles do not fold into an operand select, so it extracts them anyway).
        for (int i = 0; i < 4; ++i) c += a.v[i] * b.v[i];
        return c;
    }
    static __forceinline__ __device__ float finish(int c) { return static_cast<float>(c); }
};

// e4m3fn, kept in LDS as fp16 pairs. Moving an e4m3 byte's low seven bits up by
// seven places lands its exponent and mantissa in fp16's fields, and since the two
// formats differ only in bias (7 against 15) and exponent width, the fp16 value is
// exactly x * 2^-8 for every finite e4m3, subnormals included. Products therefore
// come out 2^-16 low and finish() scales them back, exactly, being a power of two.
// The e4m3 NaN (S.1111.111) would otherwise land on a finite fp16, so it is mapped
// to one.
struct SimtFp8 {
    using Acc = float;
    static constexpr int kExpand = 2;
    static constexpr int kSubs = 1;
    static constexpr bool kPack16 = false;  // see launch_gemm_simt
    typedef _Float16 half2_t __attribute__((ext_vector_type(2)));

    // bytes 0,1 (sel 0) or 2,3 (sel 1) of w as two fp16 bit patterns
    static __forceinline__ __device__ uint32_t to_half2_bits(uint32_t w, int sel) {
        const uint32_t t = __byte_perm(w, 0u, sel ? 0x4342u : 0x4140u);  // [b, 0, b', 0]
        const uint32_t h = ((t & 0x007F007Fu) << 7) | ((t & 0x00800080u) << 8);
        // (t & 0x7F) + 1 reaches 0x80 only for the NaN pattern
        const uint32_t nan = (((t & 0x007F007Fu) + 0x00010001u) & 0x00800080u) >> 7;
        return h | nan * 0x7E00u;
    }
    static __forceinline__ __device__ void expand(uint32_t w, uint32_t out[kExpand]) {
        out[0] = to_half2_bits(w, 0);
        out[1] = to_half2_bits(w, 1);
    }

    struct Frag {
        half2_t h;
    };
    static __forceinline__ __device__ Frag unpack(uint32_t w, int) {
        return {__builtin_bit_cast(half2_t, w)};
    }
    // fp16 operands straight into an fp32 FMA: v_fma_mix_f32, no conversions
    static __forceinline__ __device__ float dot(Frag a, Frag b, float c) {
        c = __builtin_fmaf(static_cast<float>(a.h.x), static_cast<float>(b.h.x), c);
        return __builtin_fmaf(static_cast<float>(a.h.y), static_cast<float>(b.h.y), c);
    }
    static __forceinline__ __device__ float finish(float c) { return c * 65536.0f; }
};

// fp16 pairs, kept in LDS as is, into v_fma_mix_f32.
// At M=4096 and MiniMax-H3's linear shapes this measured 4.2-4.9 TFLOPS,
// against 1.6-2.3 for rocBLAS fp16, while accumulating in fp32.
struct SimtF16 {
    using Acc = float;
    static constexpr int kExpand = 1;
    static constexpr int kSubs = 1;
    static constexpr bool kPack16 = false;  // see launch_gemm_simt
    using Frag = SimtFp8::Frag;
    static __forceinline__ __device__ void expand(uint32_t w, uint32_t out[kExpand]) { out[0] = w; }
    static __forceinline__ __device__ Frag unpack(uint32_t w, int s) { return SimtFp8::unpack(w, s); }
    static __forceinline__ __device__ float dot(const Frag& a, const Frag& b, float c) {
        return SimtFp8::dot(a, b, c);
    }
    static __forceinline__ __device__ float finish(float c) { return c; }
};

// fp16 pairs multiplied two at a time by v_pk_fma_f16, which both targets have: twice
// SimtF16's MAC rate. Each half of a pair accumulates kFoldStages stages' products in
// fp16 (one per K word) before the kernel folds the pair into the fp32 accumulator, so
// the caller only has to keep that many products' partial sums inside fp16's range
// (see ops/gemm_f16_packed.hip).
//
// A K word holds elements (2i, 2i + 1) of both operands, so the low half of the packed
// accumulator sums the even-K products and the high half the odd-K ones: two independent
// fp16 chains of kSimtWords * kFoldStages = 32 products each. Only those 32-product
// partial sums are ever rounded to fp16 (11-bit mantissa); everything across folds adds
// in fp32, so the rounding error does not grow with K the way a pure fp16 GEMM's does.
struct SimtF16Packed {
    using Acc = float;
    using Pack = SimtFp8::half2_t;
    // Folding after every stage cost 5-7% at H3's shapes on gfx1010 against every
    // second one; fp32 error vs the fp64 GEMM went from 6.7e-4 to 8.9e-4.
    static constexpr int kFoldStages = 2;
    static constexpr int kExpand = 1;
    static constexpr int kSubs = 1;
    static constexpr bool kPack16 = false;  // see launch_gemm_simt
    struct Frag {
        Pack h;
    };
    static __forceinline__ __device__ void expand(uint32_t w, uint32_t out[kExpand]) { out[0] = w; }
    // Four int8 as two fp16 pairs (exact). SDWA sign-extends a byte of the source and
    // converts it into one half of the destination, so each int8 is one instruction:
    // measured 1% faster in the GEMM than the byte-permute form (two v_perm_b32 and two
    // packed subtracts a word).
    static __forceinline__ __device__ void expand_i8(uint32_t w, uint32_t& lo, uint32_t& hi) {
        asm("v_cvt_f16_i16_sdwa %0, sext(%1) dst_sel:WORD_0 dst_unused:UNUSED_PAD src0_sel:BYTE_0\n\t"
            "v_cvt_f16_i16_sdwa %0, sext(%1) dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:BYTE_1"
            : "=&v"(lo) : "v"(w));
        asm("v_cvt_f16_i16_sdwa %0, sext(%1) dst_sel:WORD_0 dst_unused:UNUSED_PAD src0_sel:BYTE_2\n\t"
            "v_cvt_f16_i16_sdwa %0, sext(%1) dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:BYTE_3"
            : "=&v"(hi) : "v"(w));
    }
    static __forceinline__ __device__ Frag unpack(uint32_t w, int) { return {__builtin_bit_cast(Pack, w)}; }
    static __forceinline__ __device__ Pack dot(Frag a, Frag b, Pack c) {
        return __builtin_elementwise_fma(a.h, b.h, c);
    }
    // The two chains are widened separately: adding them in fp16 first would round
    // once more and could overflow where each chain alone does not.
    static __forceinline__ __device__ float fold(Pack p, float c) {
        return c + static_cast<float>(p.x) + static_cast<float>(p.y);
    }
    static __forceinline__ __device__ float finish(float c) { return c; }
};

// A policy with a Pack type accumulates into it, folded into Acc every Op::kFoldStages
// stages.
template <typename Op, typename = void>
struct SimtPack {
    using type = char;
    static constexpr bool kPacked = false;
};
template <typename Op>
struct SimtPack<Op, std::void_t<typename Op::Pack>> {
    using type = typename Op::Pack;
    static constexpr bool kPacked = true;
};

// A gathered A operand's per-row state; nothing for dense rows.
template <typename ASrc>
struct SimtARow {
    using type = typename ASrc::Row;
};
template <>
struct SimtARow<const uint8_t*> {
    using type = char;
};

// ---------------------------------------------------------------------------
// The kernel. 256 threads as a TY x TX grid (TX = 256 / TY); thread (tx, ty)
// owns rows ty*4 + {0..3} of each TY*4-row slice of the block tile and likewise
// columns, so BM = TY * TM and BN = TX * TN, and its reads of a K word are one
// 16-byte LDS load per four rows or columns. TY = 16 is the square grid; a
// smaller TY packs a short M into a squat tile, spending the threads on columns
// rather than on rows past M.
// ---------------------------------------------------------------------------

constexpr int kSimtThreads = 256;
constexpr int kSimtWords = 16;  // K words (64 bytes) staged per iteration
// Split-K slices take turns by grains of this many stages (256 bytes). Contiguous
// slices of power-of-two size alias onto the same memory channels (see the kernel);
// a one-stage grain wastes half of each 128-byte line a block fetches; 4 and 8 both
// avoided both on gfx1010, and 32 (2 KiB) brought the aliasing back.
constexpr int kSimtGrainStages = 4;

// Tiled layout of A, for the 128-row tile: rows are grouped 128 at a time, and within a
// group each stage (64 bytes of K) of all 128 rows is contiguous, in the order the
// kernel's threads load it: 16-byte chunk q of row r at [(q * 128 + r) * 16]. An A of R
// rows therefore occupies ceil(R / 128) * 128 rows' worth of bytes, and kbytes must be
// a multiple of 64.
constexpr int kSimtTileRows = 128;
__forceinline__ __host__ __device__ int64_t simt_tiled_offset(int row, int kbyte, int kbytes) {
    constexpr int kStage = kSimtWords * 4;
    const int r = row % kSimtTileRows;
    return static_cast<int64_t>(row - r) * kbytes +
           static_cast<int64_t>(kbyte / kStage) * (kSimtTileRows * kStage) +
           ((kbyte % kStage) / 16 * kSimtTileRows + r) * 16 + kbyte % 16;
}

// Element offset of C[r, col]: row-major with row stride ldc, unless the epilogue places
// its output itself through out_offset(r, col) (the packed conv writing NCDHW).
template <typename Epi>
__forceinline__ __device__ int64_t simt_out_offset(const Epi& epi, int r, int col, int ldc) {
    if constexpr (requires { epi.out_offset(r, col); }) {
        return epi.out_offset(r, col);
    } else {
        return static_cast<int64_t>(r) * ldc + col;
    }
}

template <typename Op, typename Epi, typename OutT, int TY, int TM, int TN, bool SPLIT,
          typename ASrc = const uint8_t*, bool TILED = false, bool BI8 = false>
__global__ __launch_bounds__(kSimtThreads, 2) void gemm_simt_kernel(
    typename GemmOperandA<ASrc>::type A, const uint8_t* __restrict__ B, OutT* __restrict__ C,
    int M, int N, int kbytes, int ldc, Epi epi, typename Op::Acc* __restrict__ partial) {
#if defined(COMFY_SIMT_GEMM)
    constexpr int TX = kSimtThreads / TY;
    constexpr int BM = TY * TM, BN = TX * TN;
    constexpr int kBytes = kSimtWords * 4;
    constexpr int kChunks = kBytes / 16;  // 16-byte global chunks per row per stage
    // A squat tile has fewer A chunks than threads, so the per-thread counts round
    // up and the loads below guard c against the chunk count.
    constexpr int kAChunks = BM * kChunks, kBChunks = BN * kChunks / (BI8 ? 2 : 1);
    constexpr int kAPer = (kAChunks + kSimtThreads - 1) / kSimtThreads;
    constexpr int kBPer = (kBChunks + kSimtThreads - 1) / kSimtThreads;
    static_assert(TX * TY == kSimtThreads, "TY must divide the block's threads");
    static_assert(!TILED || BM == kSimtTileRows, "a tiled A is laid out for the 128-row tile");
    static_assert(!BI8 || Op::kExpand == 1, "an int8 B is widened by Op::expand_i8 alone");
    static_assert(TM % 4 == 0 && TN % 4 == 0, "a thread reads its rows four at a time");
    static_assert(kAChunks % kSimtThreads == 0 || kAChunks < kSimtThreads,
                  "a partial last round of chunk loads is only handled for a tile under one round");
    static_assert(kBChunks % kSimtThreads == 0 || kBChunks < kSimtThreads,
                  "a partial last round of chunk loads is only handled for a tile under one round");
    constexpr int kE = Op::kExpand;
    constexpr int kLdsWords = kSimtWords * kE;  // LDS words per row per stage

    // K-major: LDS word lw of row r at [lw * BM + r], where global word kw expands
    // to lw = kw * kE .. kw * kE + kE - 1. Every read is a 16-byte load at a
    // multiple of 16 bytes, which also keeps clear of gfx1010's misaligned-LDS
    // hazard (LLVM's lds-misaligned-bug).
    __shared__ __align__(16) uint32_t As[kLdsWords * BM];
    __shared__ __align__(16) uint32_t Bs[kLdsWords * BN];

    const int tid = threadIdx.x;
    const int tx = tid % TX, ty = tid / TX;

    // Grouped block order: consecutive blocks walk M within a
    // group of rows so resident blocks share B columns in L2.
    constexpr int kGroupM = 4;
    const int blocks_n = gridDim.x, blocks_m = gridDim.y;
    const int bid = blockIdx.y * blocks_n + blockIdx.x;
    const int per_group = kGroupM * blocks_n;
    const int group = bid / per_group;
    const int idx_in_group = bid - group * per_group;
    const int group_rows = min(kGroupM, blocks_m - group * kGroupM);
    const int m0 = (group * kGroupM + idx_in_group % group_rows) * BM;
    const int n0 = (idx_in_group / group_rows) * BN;

    // K slice of this block: all of K, or with split-K (gridDim.z > 1) every
    // gridDim.z-th grain of kgrain bytes from the z-th. Interleaved rather than
    // contiguous, so the slices of a tile read neighbouring grains of the same rows
    // at any moment. Contiguous slices read them K / slices bytes apart instead, and
    // whenever that is a power of two of 2 KiB or more those reads alias onto the
    // same memory channels: measured on gfx1010, contiguous 4 KiB slices ran at
    // under half the speed of interleaved ones.
    // Without SPLIT, all of K in order, and none of this or the partial writeback
    // below is compiled in: carried unused, it cost the int4 8x8 tile 5% on gfx1010.
    constexpr int kGrain = kSimtGrainStages * kBytes;
    const int kb_begin = SPLIT ? blockIdx.z * kGrain : 0;
    const int kskip = SPLIT ? (gridDim.z - 1) * kGrain : 0;  // other slices' grains
    const auto next_k = [&](int kb) {
        kb += kBytes;
        return (SPLIT && kb % kGrain == 0) ? kb + kskip : kb;
    };

    // Chunk c covers row c % ROWS and K bytes (c / ROWS) * 16, so consecutive
    // threads store consecutive rows of one LDS word: distinct banks.
    constexpr bool kDenseA = std::is_same_v<ASrc, const uint8_t*>;
    uint4 ra[kAPer], rb[kBPer];
    [[maybe_unused]] typename SimtARow<ASrc>::type arow[kAPer];
    if constexpr (!kDenseA) {
#pragma unroll
        for (int i = 0; i < kAPer; ++i) {
            const int c = tid + i * kSimtThreads;
            arow[i] = A.row(m0 + c % BM,
                            (kAChunks % kSimtThreads == 0 || c < kAChunks) && m0 + c % BM < M);
        }
    }
    auto load = [&](int kb0) {
        if constexpr (kDenseA) {
#pragma unroll
            for (int i = 0; i < kAPer; ++i) {
                const int c = tid + i * kSimtThreads;
                const int grow = m0 + c % BM, gk = kb0 + (c / BM) * 16;
                const int64_t off = TILED ? static_cast<int64_t>(m0) * kbytes +
                                                static_cast<int64_t>(kb0) * BM + c * 16
                                          : static_cast<int64_t>(grow) * kbytes + gk;
                ra[i] = ((kAChunks % kSimtThreads == 0 || c < kAChunks) && grow < M && gk < kbytes)
                            ? *reinterpret_cast<const uint4*>(A + off)
                            : make_uint4(0, 0, 0, 0);
            }
        } else {
            const auto stage = A.stage(kb0);
#pragma unroll
            for (int i = 0; i < kAPer; ++i) {
                const int c = tid + i * kSimtThreads;
                ra[i] = A.chunk(arow[i], stage, (c / BM) * 16);
            }
        }
#pragma unroll
        for (int i = 0; i < kBPer; ++i) {
            const int c = tid + i * kSimtThreads;
            // B's bytes of K per row and per stage: half of A's when it is int8
            constexpr int kDiv = BI8 ? 2 : 1;
            const int grow = n0 + c % BN, gk = kb0 / kDiv + (c / BN) * 16;
            rb[i] = ((kBChunks % kSimtThreads == 0 || c < kBChunks) && grow < N && gk < kbytes / kDiv)
                        ? *reinterpret_cast<const uint4*>(B + static_cast<int64_t>(grow) * (kbytes / kDiv) + gk)
                        : make_uint4(0, 0, 0, 0);
        }
    };
    // Word q of a chunk is global word (c / ROWS) * 4 + q of the row.
    auto store_chunk = [&](uint32_t* lds, int rows, int c, const uint4& v) {
        const uint32_t w[4] = {v.x, v.y, v.z, v.w};
        uint32_t* dst = lds + (c / rows) * 4 * kE * rows + c % rows;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            uint32_t e[kE];
            Op::expand(w[q], e);
#pragma unroll
            for (int x = 0; x < kE; ++x) dst[(q * kE + x) * rows] = e[x];
        }
    };
    auto store = [&]() {
#pragma unroll
        for (int i = 0; i < kAPer; ++i) {
            const int c = tid + i * kSimtThreads;
            if (kAChunks % kSimtThreads == 0 || c < kAChunks) store_chunk(As, BM, c, ra[i]);
        }
#pragma unroll
        for (int i = 0; i < kBPer; ++i) {
            const int c = tid + i * kSimtThreads;
            if (!(kBChunks % kSimtThreads == 0 || c < kBChunks)) continue;
            if constexpr (BI8) {
                // 16 int8 of row c % BN: K words (c / BN) * 8 .. + 7 of the stage
                const uint32_t w[4] = {rb[i].x, rb[i].y, rb[i].z, rb[i].w};
                uint32_t* dst = Bs + (c / BN) * 8 * BN + c % BN;
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    uint32_t lo, hi;
                    Op::expand_i8(w[q], lo, hi);
                    dst[2 * q * BN] = lo;
                    dst[(2 * q + 1) * BN] = hi;
                }
            } else {
                store_chunk(Bs, BN, c, rb[i]);
            }
        }
    };

    constexpr bool kPacked = SimtPack<Op>::kPacked;
    typename Op::Acc acc[TM][TN];
    [[maybe_unused]] typename SimtPack<Op>::type pack[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0;
            if constexpr (kPacked) pack[i][j] = {};
        }

    load(kb_begin);
    store();
    __syncthreads();
    [[maybe_unused]] int stage = 0;

    for (int kb0 = kb_begin; kb0 < kbytes; kb0 = next_k(kb0)) {
        const bool has_next = next_k(kb0) < kbytes;
        if (has_next) load(next_k(kb0));  // in flight across this tile's math

#pragma unroll COMFY_SIMT_UNROLL
        for (int lw = 0; lw < kLdsWords; ++lw) {
            uint32_t aw[TM], bw[TN];
#pragma unroll
            for (int g = 0; g < TM / 4; ++g) {
                const uint4 v = *reinterpret_cast<const uint4*>(As + lw * BM + g * (TY * 4) + ty * 4);
                aw[4 * g] = v.x, aw[4 * g + 1] = v.y, aw[4 * g + 2] = v.z, aw[4 * g + 3] = v.w;
            }
#pragma unroll
            for (int g = 0; g < TN / 4; ++g) {
                const uint4 v = *reinterpret_cast<const uint4*>(Bs + lw * BN + g * (TX * 4) + tx * 4);
                bw[4 * g] = v.x, bw[4 * g + 1] = v.y, bw[4 * g + 2] = v.z, bw[4 * g + 3] = v.w;
            }
#pragma unroll
            for (int s = 0; s < Op::kSubs; ++s) {
                // B is unpacked whole and A one row at a time, so only one A Frag
                // need be live across its TN dots.
                typename Op::Frag bf[TN];
#pragma unroll
                for (int j = 0; j < TN; ++j) bf[j] = Op::unpack(bw[j], s);
#pragma unroll
                for (int i = 0; i < TM; ++i) {
                    const typename Op::Frag af = Op::unpack(aw[i], s);
#pragma unroll
                    for (int j = 0; j < TN; ++j) {
                        if constexpr (kPacked) {
                            pack[i][j] = Op::dot(af, bf[j], pack[i][j]);
                        } else {
                            acc[i][j] = Op::dot(af, bf[j], acc[i][j]);
                        }
                    }
                }
            }
        }
        // Fold the fp16 chains into fp32 before they hold more products than the caller's
        // operand scaling bounds, and after this slice's last stage so no partial is lost
        // (a slice's stage count need not be a multiple of kFoldStages).
        if constexpr (kPacked) {
            if (++stage % Op::kFoldStages == 0 || !has_next) {
#pragma unroll
                for (int i = 0; i < TM; ++i)
#pragma unroll
                    for (int j = 0; j < TN; ++j) {
                        acc[i][j] = Op::fold(pack[i][j], acc[i][j]);
                        pack[i][j] = {};
                    }
            }
        }

        if (has_next) {
            __syncthreads();  // every thread is done reading this tile
            store();
            __syncthreads();
        }
    }

    if constexpr (SPLIT) {
        // split-K: this slice's raw sums, for gemm_simt_reduce_kernel to combine
        typename Op::Acc* slice = partial + static_cast<int64_t>(blockIdx.z) * M * N;
#pragma unroll
        for (int i = 0; i < TM; ++i) {
            const int r = m0 + (i / 4) * (TY * 4) + ty * 4 + i % 4;
            if (r >= M) continue;
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int col = n0 + (j / 4) * (TX * 4) + tx * 4 + j % 4;
                if (col < N) slice[static_cast<int64_t>(r) * N + col] = acc[i][j];
            }
        }
        return;
    }

    epi.init();
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int r = m0 + (i / 4) * (TY * 4) + ty * 4 + i % 4;
        if (r >= M) continue;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int col = n0 + (j / 4) * (TX * 4) + tx * 4 + j % 4;
            if (col >= N) continue;
            C[simt_out_offset(epi, r, col, ldc)] =
                static_cast<OutT>(epi(r, col, Op::finish(acc[i][j])));
        }
    }
#endif // COMFY_SIMT_GEMM
}

// Split-K combine: C[r, c] = epilogue(sum over the slices of partial[z, r, c]),
// summed in slice order so the result is the same every run.
template <typename Op, typename Epi, typename OutT>
__global__ __launch_bounds__(256) void gemm_simt_reduce_kernel(
    const typename Op::Acc* __restrict__ partial, OutT* __restrict__ C, int M, int N, int ldc,
    int slices, Epi epi) {
#if defined(COMFY_SIMT_GEMM)
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t mn = static_cast<int64_t>(M) * N;
    if (i >= mn) return;
    typename Op::Acc sum = partial[i];
    for (int z = 1; z < slices; ++z) sum += partial[z * mn + i];
    const int r = static_cast<int>(i / N), col = static_cast<int>(i % N);
    epi.init();
    C[simt_out_offset(epi, r, col, ldc)] = static_cast<OutT>(epi(r, col, Op::finish(sum)));
#endif
}

// ---------------------------------------------------------------------------
// Host side.
// ---------------------------------------------------------------------------

// One tile shape, over `slices` K slices. With more than one, each slice writes
// its raw sums to a stream-ordered scratch buffer and a second kernel adds them
// in order and applies the epilogue. A scratch allocation that fails falls back
// to a single slice rather than failing the GEMM.
template <typename Op, typename Epi, typename OutT, int TY, int TM, int TN,
          typename ASrc = const uint8_t*, bool TILED = false, bool BI8 = false>
void launch_simt_tile(ASrc A, const uint8_t* B, OutT* C, int M, int N, int kbytes,
                      int ldc, Epi epi, int slices, hipStream_t stream) {
    constexpr int bm = TY * TM, bn = (kSimtThreads / TY) * TN;
    constexpr int kStage = kSimtWords * 4;
    dim3 grid((N + bn - 1) / bn, (M + bm - 1) / bm, 1);
    using Acc = typename Op::Acc;
    Acc* partial = nullptr;
    const int kgrain = kSimtGrainStages * kStage;
    slices = std::min(slices, (kbytes + kgrain - 1) / kgrain);  // no slice without a grain
    if (slices > 1) {
        const size_t bytes = static_cast<size_t>(slices) * M * N * sizeof(Acc);
        if (hipMallocAsync(reinterpret_cast<void**>(&partial), bytes, stream) != hipSuccess) {
            (void)hipGetLastError();
            partial = nullptr;
            slices = 1;
        }
    }
    grid.z = slices;
    if (!partial) {
        gemm_simt_kernel<Op, Epi, OutT, TY, TM, TN, false, ASrc, TILED, BI8>
            <<<grid, kSimtThreads, 0, stream>>>(A, B, C, M, N, kbytes, ldc, epi, nullptr);
        return;
    }
    gemm_simt_kernel<Op, Epi, OutT, TY, TM, TN, true, ASrc, TILED, BI8>
        <<<grid, kSimtThreads, 0, stream>>>(A, B, C, M, N, kbytes, ldc, epi, partial);
    const int64_t mn = static_cast<int64_t>(M) * N;
    gemm_simt_reduce_kernel<Op, Epi, OutT>
        <<<static_cast<unsigned>((mn + 255) / 256), 256, 0, stream>>>(partial, C, M, N, ldc,
                                                                     slices, epi);
    (void)hipFreeAsync(partial, stream);
}

// Tile choice. 128x128 has the most reuse per LDS word; the smaller tiles trade
// it for enough blocks to fill the device when M or N is small. device_wgp_count()
// is the WGP count, the unit a block lands on.
//
// Short M (M <= 48) packs the rows into a 16- or 32-row tile instead of
// a 64-row one that is mostly empty, and splits K until there are 48 blocks. Every
// choice below was the fastest, or within a few percent of it, of the measured
// candidates (tile shape x 1..8 slices, for int8, fp8 and int4, M 9..48, N 1024 to
// 12288, K 768 to 12288). The exceptions to "smallest tile that covers M" are
// measured, not derived:
//   - int8 and fp8 lose to the 32-row tile at M <= 16 once the grid has 24 blocks
//     of 16x256, or K is past 4 KiB.
//   - int4 keeps the 16-row tile up to M = 48 (three of them beat one 64-row tile)
//     while its rows are at most 2 KiB, and loses past that.
// Whether launch_gemm_simt gives this shape the square 128-row tile, the one a tiled A
// is laid out for.
inline bool simt_square_tile(int M, int N) {
    return M > 64 && static_cast<int64_t>((M + 127) / 128) * ((N + 127) / 128) >= 2 * device_wgp_count();
}

// launch_gemm_simt for a simt_square_tile shape with A in the tiled layout.
template <typename Op, typename Epi, typename OutT, bool BI8 = false>
void launch_gemm_simt_tiled(const uint8_t* A, const uint8_t* B, OutT* C, int M, int N, int kbytes,
                            int ldc, Epi epi, hipStream_t stream) {
    launch_simt_tile<Op, Epi, OutT, 16, 8, 8, const uint8_t*, true, BI8>(A, B, C, M, N, kbytes,
                                                                        ldc, epi, 1, stream);
}

template <typename Op, typename Epi, typename OutT, typename ASrc = const uint8_t*, bool BI8 = false>
void launch_gemm_simt(ASrc A, const uint8_t* B, OutT* C, int M, int N, int kbytes,
                      int ldc, Epi epi, hipStream_t stream) {
    const int units = device_wgp_count();
    const auto blocks = [&](int bm, int bn) {
        return static_cast<int64_t>((M + bm - 1) / bm) * ((N + bn - 1) / bn);
    };
    // Past about 48 blocks, more slices only add partial traffic: measured, a
    // 96-block grid split 4 ways ran 20% slower than unsplit.
    const auto slices = [&](int bm, int bn) {
        const int64_t b = blocks(bm, bn);
        return static_cast<int>(std::min<int64_t>(8, (48 + b - 1) / b));
    };
    // The 64-row tiles at M 33..48 also want 8 stages (512 bytes of K) per slice:
    // below that the second kernel and the scratch allocation cost more than the
    // split saves (K=768 ran up to 12% slower split). The packed tiles still gain
    // there, having far fewer blocks to start from.
    const auto slices64 = [&](int bm, int bn) {
        return std::min(slices(bm, bn), std::max(1, kbytes / (8 * kSimtWords * 4)));
    };
    const bool packed = M <= 48;
    if (packed && M <= 16 &&
        (Op::kPack16 || (blocks(16, 256) < 24 && kbytes <= 4096))) {
        launch_simt_tile<Op, Epi, OutT, 4, 4, 4, ASrc, false, BI8>(A, B, C, M, N, kbytes, ldc, epi,
                                                 slices(16, 256), stream);
    } else if (packed && M <= 32) {
        launch_simt_tile<Op, Epi, OutT, 8, 4, 4, ASrc, false, BI8>(A, B, C, M, N, kbytes, ldc, epi,
                                                 slices(32, 128), stream);
    } else if (packed && Op::kPack16 && kbytes <= 2048) {
        launch_simt_tile<Op, Epi, OutT, 4, 4, 4, ASrc, false, BI8>(A, B, C, M, N, kbytes, ldc, epi,
                                                 slices(16, 256), stream);
    } else if (simt_square_tile(M, N)) {
        launch_simt_tile<Op, Epi, OutT, 16, 8, 8, ASrc, false, BI8>(A, B, C, M, N, kbytes, ldc, epi, 1, stream);
    } else if (blocks(64, 128) >= 2 * units) {
        launch_simt_tile<Op, Epi, OutT, 16, 4, 8, ASrc, false, BI8>(A, B, C, M, N, kbytes, ldc, epi,
                                                  packed ? slices64(64, 128) : 1, stream);
    } else {
        launch_simt_tile<Op, Epi, OutT, 16, 4, 4, ASrc, false, BI8>(A, B, C, M, N, kbytes, ldc, epi,
                                                  packed ? slices64(64, 64) : 1, stream);
    }
}

}  // namespace comfy::hip_backend
