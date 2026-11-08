// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// NVFP4 (E2M1) codec shared by the quantizer, the dequantizer and both scaled
// GEMM paths. RDNA has no fp4 matrix instruction (see mma.h: gfx11 widens fp8
// operands to bf16 for the same reason), so every path here decodes a nibble to
// a float and lets the caller scale it.
//
// Storage, matching comfy_kitchen.backends.eager.quantization:
//   data:  (rows, K/2) uint8, two nibbles per byte, HIGH nibble = EVEN column
//          (hi_first=True). K is a multiple of 32, so a row is a multiple of 16
//          bytes and a 16-byte load never straddles its end.
//   block scale: one e4m3 per 16 columns of a row, in the cuBLAS blocked layout
//          over (RoundUp(rows,128), RoundUp(K/16,4)). The flat offset is
//          nvfp4_scale_offset below.
//
// Decoding a nibble is a table read rather than arithmetic, and that is a
// performance decision, not a stylistic one. Measured on the 6-WGP gfx1103 this
// tree targets: ~21 VALU instructions per byte of weight are available at the
// measured 72 GB/s streaming rate (24 SIMDs at ~2.0 GHz), and a weight byte is
// two elements. Assembling the float from the code's bits -- sign shift, exponent
// shift-or, a select for the subnormal case -- is ~11 instructions per element,
// which exceeds the whole budget before any multiply happens. A 16-entry table in
// LDS is one load per element and leaves the arithmetic for the dot product.
//
// The conversion itself is bit assembly, because e2m1 is a float format: 1-2-1,
// bias 1, so exponent field 0 is subnormal (m * 2^-1) and fields 1..3 are 2^0..2^2
// times (1 + m/2). Both cases are the same fp32 bit pattern, so one shift/or chain
// covers all 16 codes. It fills the table rather than replacing it, so the table
// cannot disagree with the format definition.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace comfy::hip_backend {

// One scale per 16 elements. Always 16 for NVFP4.
constexpr int kNvfp4Block = 16;

// The dtype code for e4m3, matching eager's DTYPE_TO_CODE and fp8_utils.h.
constexpr int kFp8E4M3CodeLocal = 5;

// Entries in the decode table. One per E2M1 code, including the sign codes 8..15,
// so the table lookup carries the sign and no separate negation is needed.
constexpr int kE2m1LutSize = 16;

// ---------------------------------------------------------------------------
// Nibble decode
// ---------------------------------------------------------------------------

// One E2M1 code -> its fp32 bit pattern.
//
// Verified against comfy_kitchen.float_utils._floatx_unpacked_to_f32 and
// E2M1_LUT: code 0 -> +0.0, 7 -> 6.0, 8 -> -0.0 (distinct bits from 0x0),
// 0xF -> -6.0.
__forceinline__ __device__ uint32_t e2m1_code_to_f32_bits(uint32_t code) {
    const uint32_t sign = (code & 0x8u) << 28;  // bit 3 -> fp32 bit 31
    const uint32_t m = code & 1u;
    const uint32_t e = (code >> 1) & 3u;
    // e == 0 is subnormal: 0.0 or 0.5, whose fp32 exponent field is 126.
    // e >= 1: unbiased exponent e - 1, so the field is 126 + e.
    const uint32_t bits = (e == 0u) ? (m * 0x3F000000u) : (((126u + e) << 23) | (m << 22));
    return bits | sign;
}

__forceinline__ __device__ float e2m1_code_to_f32(uint32_t code) {
    return __uint_as_float(e2m1_code_to_f32_bits(code));
}

// Fill a 16-entry table. Called once per block; 16 threads write, then the caller
// barriers. Cheap enough that it never shows up, and it keeps the table derived
// from the format instead of spelled out a second time.
__forceinline__ __device__ void e2m1_init_lut(float* lut, int tid) {
    if (tid < kE2m1LutSize) lut[tid] = e2m1_code_to_f32(tid);
}

// ---------------------------------------------------------------------------
// Blocked scale layout
// ---------------------------------------------------------------------------

// Bytes one 128-row group's four-block column group occupies, and so the stride
// between two ADJACENT such groups in the buffer. Spelled out because it is the one
// number a kernel walking the scales needs and it is not legible inside the offset
// expression below: it is what makes a lane's two consecutive block scales one byte
// apart, and what lets a kernel advance along K by a whole number of four-block
// groups with a multiply. Also the reason a 128-row group's scales are one run of
// 512 CONTIGUOUS bytes with nothing missing -- which is what nvfp4_gfx10.h sizes its
// column block from.
constexpr int kScaleRowsPerGroup = 32;                      // d3, rows inside a group
constexpr int kScaleGroupStride = kScaleRowsPerGroup * 16;  // 512 bytes

// Flat offset of the e4m3 scale for logical (row, col) -- col counts 16-element
// blocks along K -- inside a buffer stored as (RoundUp(rows,128),
// RoundUp(K/16,4)) and viewed row-major.
//
// This is cuBLAS's "D block scaling factors layout", mirrored here from
// cuda/float_utils.cuh:scale_factor_swizzled_offset and identical to
// comfy_kitchen.float_utils.to_blocked, which it inverts: verified as an exact
// inverse of to_blocked over the whole grid for (rows, K) in
// {128, 1024, 25600, 5120} x {4096, 5120, 8192, 25600}, and against real layers
// of qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors, where decoding through this
// offset reproduces ck.dequantize_nvfp4 exactly (relative error 0.000000, against
// 0.28-0.54 for a row-major read).
__forceinline__ __device__ int nvfp4_scale_offset(int row, int col, int col_length) {
    const int rb = row >> 7;    // row / 128
    const int rem = row & 127; // row % 128
    const int d4 = rem >> 5;    // rem / 32
    const int d3 = rem & 31;    // rem % 32
    const int cbg = col >> 2;   // col / 4
    const int d5 = col & 3;     // col % 4
    const int cbg_cnt = (col_length + 3) >> 2;
    return ((rb * cbg_cnt + cbg) * kScaleRowsPerGroup + d3) * 16 + d4 * 4 + d5;
}

// ---------------------------------------------------------------------------
// Vector decode through the table
// ---------------------------------------------------------------------------

// 8 packed bytes -> 16 floats of E2M1 values, HIGH nibble of byte i at even
// element 2i. This is the unit of the block-quantized layout: one 16-element block
// is exactly 8 bytes, so this is what the dequantizer and any one-block-per-lane
// path wants. e2m1x16_to_f32_lut below is the 32-element form, for a lane that
// spans two blocks.
template <bool HI_FIRST>
__forceinline__ __device__ void e2m1x8_to_f32_lut(const uint2 packed, const float* lut,
                                                  float* out) {
    const uint32_t words[2] = {packed.x, packed.y};
    #pragma unroll
    for (int w = 0; w < 2; ++w) {
        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            const uint32_t byte = (words[w] >> (8 * b)) & 0xFFu;
            const uint32_t hi = byte >> 4;
            const uint32_t lo = byte & 0xFu;
            out[8 * w + 2 * b + (HI_FIRST ? 0 : 1)] = lut[hi];
            out[8 * w + 2 * b + (HI_FIRST ? 1 : 0)] = lut[lo];
        }
    }
}

// 16 packed bytes -> 32 floats of E2M1 values, HIGH nibble of byte i at even
// element 2i. HI_FIRST=false swaps the two, which is the only difference between
// the packing orders (ck.swap_nibbles converts between them without requantizing).
template <bool HI_FIRST>
__forceinline__ __device__ void e2m1x16_to_f32_lut(const uint4 packed, const float* lut,
                                                   float* out) {
    const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
    #pragma unroll
    for (int w = 0; w < 4; ++w) {
        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            const uint32_t byte = (words[w] >> (8 * b)) & 0xFFu;
            const uint32_t hi = byte >> 4;
            const uint32_t lo = byte & 0xFu;
            out[8 * w + 2 * b + (HI_FIRST ? 0 : 1)] = lut[hi];
            out[8 * w + 2 * b + (HI_FIRST ? 1 : 0)] = lut[lo];
        }
    }
}

// As above but writing half, scaled by s0 for the first 16 elements and s1 for
// the second 16, as 4 uint4 of 8 halves each. The two scales are the block scales
// of the two NVFP4 blocks a 32-element chunk spans, and folding them in here is
// what lets the WMMA accumulate straight into an fp32 total with no per-block
// rescale: the staged value is (e2m1 x e4m3), which needs at most 4 mantissa bits
// and so is exact in fp16, and the products need at most 8 and are exact in the
// fp32 accumulator.
//
// Every E2M1 magnitude (0, 0.5, 1, 1.5, 2, 3, 4, 6) is exactly representable in
// fp16 and in bf16 -- both carry more mantissa bits than e2m1's one and a wider
// exponent range -- so the widening is exact and adds no error the scaled GEMM
// would not otherwise have. Same argument mma.h makes for widening fp8 to bf16 on
// gfx11.
//
// One byte out of a dword, as a single v_bfe. The shift amount and width are
// immediates at every call site (the loops that use this are unrolled over the four
// bytes of a word), which is what lets the extract fold to one instruction where
// shift-then-mask costs two. Byte 0 still lowers to a plain and, which is the same
// instruction either way, so this is a win on three of every four bytes and never
// a loss.
// The table entry for byte v is (half(e2m1(v >> 4)) << 16) | half(e2m1(v & 0xF)), so
// its high half is the byte's high nibble and its low half the low one. Which of the
// two ends up at which half-index after the stores is not claimed here; see the note
// above e2m1x16_to_half_scaled for why that is deliberate.
//
// **The table is indexed by the BYTE, not the nibble, and holds the two halves
// already packed.** That is the whole optimization. The narrow version -- extract
// the byte, split it into two nibbles, do two table reads, convert and pack --
// costs about 12 instructions per byte, and the ISA put it at 192 of the 578 in
// the tile kernel's inner loop. Here the byte IS the index, so there is nothing to
// split, one read returns both values, and the scale is a single packed multiply:
//
//   byte extraction        2   v_lshrrev + v_and (the index cannot be the whole
//                              shifted word -- it would run off the table)
//   table read             1   ds_read_b32, giving both halves packed
//   scale                  1   v_mul_f16, one instruction for both halves
//
// which is 4 per byte, or 2 per element, against about 6.
//
// The multiply is exact and stays exact. e2m1 needs 1 mantissa bit and e4m3 needs
// 3, so the product needs at most 4 and fp16 carries 10; the previous form
// multiplied in fp32 and rounded to fp16, which also landed on an exactly
// representable value, so both spellings produce the same bits. Range: the largest
// product is 6 * 448 = 2688 against fp16's 65504, and the smallest nonzero is
// 0.5 * 2^-9, above fp16's smallest normal 2^-14, so no overflow and no subnormal
// result -- which is what would make v_mul_f16's denormal handling a question
// worth asking, and it does not arise.
//
// The table is 1 KB, filled from the format the same way the narrow one was, so it
// cannot disagree with the definition. 256 entries is 2 per thread at 128 threads
// and 1 at 256; a 64 KB LDS per WGP absorbs it without moving a block.
constexpr int kE2m1ByteLutSize = 256;

// Fill the byte-indexed table. THREADS threads cooperate; the caller barriers.
__forceinline__ __device__ void e2m1_init_byte_lut(uint32_t* lut, int tid, int threads) {
    for (int i = tid; i < kE2m1ByteLutSize; i += threads) {
        // HI_FIRST: the byte's high nibble goes in the high half of the entry.
        const uint32_t hi = __half_as_ushort(__float2half(e2m1_code_to_f32(i >> 4)));
        const uint32_t lo = __half_as_ushort(__float2half(e2m1_code_to_f32(i & 0xFu)));
        lut[i] = (hi << 16) | lo;
    }
}

// Which nibble lands in which half of a staged dword is deliberately NOT recorded
// here. It was, and it was wrong: the claim was that the dword at index b of word w
// holds elements 8w+2b and 8w+2b+1 "high half first", and in fact it holds them
// pair-swapped against logical K order. Three facts combine, and the one that is
// easy to miss is that a uint32's LOW 16 bits are what land at the LOWER half-index
// on a little-endian target:
//
//   lut[i]        = (half(hi_nibble) << 16) | half(lo_nibble)   high half = high nibble
//   p             = *(const __half2*)(lut + byte)              p.x is bits 0..15 = LOW nibble
//   d[w][b]       = *(uint32_t*)&(__hmul2(p, sp))              bits 0..15 = the LOW nibble
//   wide[w]       = make_uint4(d[w][0..3])                     components at bytes 0,4,8,12
//   store         = *(uint4*)(dst + 8*w) = wide[w]             so component j covers
//                                                                  half-indices 8w+2j, 8w+2j+1
//                                                                  with the LOW 16 bits first
//   hi_first pack = (c0 << 4) | (c1 & 0xF)                    c0 is the even column
//
// hence dst[8w+2b] ends up holding the ODD element 8w+2b+1. Simulating the writes end
// to end puts a wrong element in all 32 half-indices of a chunk.
//
// **Nothing depends on that, and that is the only reason it is tolerable.** Both
// operands are staged by this function, so any within-block permutation sigma applied
// here is applied to a and b alike, and sum_k a[sigma(k)] * b[sigma(k)] equals
// sum_k a[k] * b[k] exactly -- a swap of the two halves changes no output bit. So
// there is nothing to correct; a corrected claim would just be one more assertion
// that cannot be checked from the GEMM, and this comment already got it backwards
// once by being written from the table rather than from the stores.
//
// If either operand is ever staged by something else -- a second stager, an unpacked
// path, a fused prologue -- pin the order down then, make it explicit, and fix both
// sides. A split-K that reassembled partial sums in a different order would be the
// other way to break it.
//
// This used to be a HI_FIRST template parameter with a swap for the lo-first case.
// The GEMM has no such parameter -- the NVFP4 format is hi-first and this is the
// only caller -- so the lo-first instantiation was dead, and dead in the wrong
// direction: it inverted an order that was already as arbitrary as this one.
// e2m1x8_to_f32_lut still takes HI_FIRST, because the dequantizer is reached through
// dequantize_nvfp4's hi_first argument and genuinely has both orders to serve.
__forceinline__ __device__ void e2m1x16_to_half_scaled(const uint4 packed,
                                                       const uint32_t* lut, __half s0,
                                                       __half s1, uint4* out) {
    const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
    // The scale, packed the same way the table entry is, once per block rather
    // than once per byte. Words 0-1 are elements 0-15 (block 0) and words 2-3 are
    // elements 16-31 (block 1).
    const uint32_t h0 = __half_as_ushort(s0);
    const uint32_t h1 = __half_as_ushort(s1);
    const uint32_t spair[2] = {(h0 << 16) | h0, (h1 << 16) | h1};
    uint32_t d[4][4];
    #pragma unroll
    for (int w = 0; w < 4; ++w) {
        const __half2 sp = *reinterpret_cast<const __half2*>(&spair[w >> 1]);
        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            // Shift and mask, not v_bfe. The inline-asm spelling of a bit-field
            // extract does emit the v_bfe, but the offset constraint does not
            // reach the assembler as an immediate: LLVM materializes the shift
            // amounts in scalar registers instead, and the inner loop measured 532
            // instructions against 432 for this. Byte 0 already folds to a single
            // and, so the pair is one instruction for one of every four bytes and
            // two for the rest, and that is the cheaper spelling.
            const uint32_t byte = (words[w] >> (8 * b)) & 0xFFu;
            const __half2 p = *reinterpret_cast<const __half2*>(lut + byte);
            const __half2 r = __hmul2(p, sp);
            d[w][b] = *reinterpret_cast<const uint32_t*>(&r);
        }
        out[w] = make_uint4(d[w][0], d[w][1], d[w][2], d[w][3]);
    }
}

// ---------------------------------------------------------------------------
// Integer decode: E2M1 -> doubled E2M1 in int8
// ---------------------------------------------------------------------------
//
// This is the form the gfx103x GEMM consumes, and it exists because E2M1's
// magnitude grid DOUBLES into {0, 1, 2, 3, 4, 6, 8, 12} -- every one of them an
// exact int8, the largest a quarter of the format's range. So the decode is a
// widening by multiplication with 2, and what follows is integer arithmetic end to
// end: v_dot4_i32_i8 accumulates the products exactly, and a 16-element block sums
// to at most 16 * 144 = 2304, nowhere near the int32 the accumulator holds.
//
// The alternative on this part is widening to fp16 and using v_dot2_f32_f16, which
// folds the e4m3 block scale into the staged operand instead of applying it to the
// accumulator (see e2m1x16_to_half_scaled above). That is the better *shape* -- no
// per-block rescale -- and it is what gfx11 does, because WMMA has no integer
// counterpart to trade against. It loses here on the instruction count, and that is
// not close: gfx10 retires v_dot2_f32_f16 at a quarter of v_dot4_i32_i8's rate and
// does two MACs per instruction against four, so one MAC through the fp16 form
// costs 8x the dot instructions of one through the integer form. That is the same
// ratio the measured int8 and fp16 VALU tiles show end to end (5.97 against 1.96
// TOPS on this 6-CU part, gemm_gfx10.h).
//
// The scale is what the integer form gives up, and it gives it up exactly once per
// 16-element block: the 4-element products v_dot4_i32_i8 forms never straddle a
// block, so a whole block's int32 sum can be scaled and added to a float
// accumulator. That is impossible under WMMA (where one lane's accumulator element
// spans eight rows with eight different scales, ops/nvfp4.hip's note on the integer
// WMMA path) and it is the entire reason the integer form is usable at all.
//
// The table is indexed by the packed BYTE, like the fp16 one above and for the same
// reason: the byte IS the index, so one read returns both of its values and there is
// nothing to split. Entry layout is (doubled value of the high nibble) in the high
// byte, (doubled value of the low nibble) in the low byte -- the same high-nibble
// convention the format uses for the elements themselves.
//
// Which nibble lands in which byte of a staged dword is likewise not recorded, for
// the same reason as above and the same argument: both operands go through this, so
// the pair swap is applied to a and b alike and sum_k a[sigma(k)] * b[sigma(k)] is
// sum_k a[k] * b[k] exactly. See the long note before e2m1x16_to_half_scaled.
constexpr int kE2m1I8ByteLutSize = 256;

// Fill the byte-indexed integer table. THREADS threads cooperate; caller barriers.
__forceinline__ __device__ void e2m1_init_i8_byte_lut(uint16_t* lut, int tid, int threads) {
    for (int i = tid; i < kE2m1I8ByteLutSize; i += threads) {
        // 2 * value, exact for every E2M1 code (see the note above). The cast is
        // through float because e2m1_code_to_f32 is the definition and the
        // multiplication is by a power of two, so neither step rounds.
        const int32_t hi = static_cast<int32_t>(2.0f * e2m1_code_to_f32(i >> 4));
        const int32_t lo = static_cast<int32_t>(2.0f * e2m1_code_to_f32(i & 0xFu));
        lut[i] = static_cast<uint16_t>(((hi & 0xFF) << 8) | (lo & 0xFF));
    }
}

// Two packed bytes -> the four doubled-E2M1 int8 of one v_dot4 group, as the dword
// that instruction consumes.
__forceinline__ __device__ uint32_t e2m1x2_to_i8x4(uint16_t packed, const uint16_t* lut) {
    const uint32_t b0 = packed & 0xFFu;
    const uint32_t b1 = packed >> 8;
    return static_cast<uint32_t>(lut[b0]) | (static_cast<uint32_t>(lut[b1]) << 16);
}

// 16 packed bytes -> 8 such dwords, 32 elements.
__forceinline__ __device__ void e2m1x16_to_i8x8(const uint4 packed, const uint16_t* lut,
                                                uint32_t* out) {
    const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
    #pragma unroll
    for (int w = 0; w < 4; ++w) {
        // Shift and mask rather than v_bfe, for the reason spelled out in
        // e2m1x16_to_half_scaled: the inline-asm bit-field extract does not get its
        // offset to the assembler as an immediate, and LLVM materializes the shift
        // amounts in scalar registers instead. Byte 0 folds to a single and.
        out[2 * w] = e2m1x2_to_i8x4(static_cast<uint16_t>(words[w] & 0xFFFFu), lut);
        out[2 * w + 1] = e2m1x2_to_i8x4(static_cast<uint16_t>(words[w] >> 16), lut);
    }
}

// One 16-element block's contribution to one output element: the integer sum of the
// products, scaled by the two e4m3 block scales and accumulated.
//
// The two operands are both doubled, so this sum is FOUR times the value the format
// defines. kNvfp4Double is the 1/4 that undoes it, and the caller folds it into the
// A-side scale, because a power of two rides into a multiply that is being done
// anyway -- putting it anywhere else would cost an instruction per output element
// for nothing.
//
// Three instructions per accumulator per block -- cvt, mul, fma -- against four
// v_dot4 for the same 16 MACs, which is where the 1.75x instruction count over a
// plain int8 GEMM comes from. There is no cheaper spelling: the sum has to leave
// int32 before either scale can touch it, and there are two different scales.
constexpr float kNvfp4Double = 0.25f;

__forceinline__ __device__ float nvfp4_block_acc(float acc, int dot4_sum, float sa, float sb) {
    return fmaf(static_cast<float>(dot4_sum) * sa, sb, acc);
}

}  // namespace comfy::hip_backend
