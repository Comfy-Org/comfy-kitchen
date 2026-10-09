// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// NVFP4 (E2M1 + per-16 e4m3 block scales): quantize, dequantize and scaled GEMM.
//
// RDNA has no fp4 matrix instruction -- gfx11 lacks even the fp8 WMMA gfx12 has
// and widens fp8 to bf16 for the same reason (see mma.h) -- and there is no
// micro-scaling hardware either. So the arithmetic is: decode the nibble, widen
// to fp16 (exact for every e2m1 magnitude), and use gfx11's fp16 WMMA.
//
//   C[M, N] = alpha * sum_kb (sa[m, kb] * sb[n, kb]
//                             * dot(a[m, kb*16 .. +16], b[n, kb*16 .. +16]))
//             + bias[n]
//
// a, b are packed two nibbles per byte (high nibble = even column); sa, sb are
// the e4m3 block scales in cuBLAS blocked layout; alpha = tensor_scale_a *
// tensor_scale_b.
//
// Folding each block scale into its own operand is what reduces the epilogue to a
// single tensor-wise scale, and it is exact:
//   - (e2m1 x e4m3) needs at most 4 mantissa bits, which fp16 holds exactly, and
//     the pair's product needs at most 8, which the fp32 MMA accumulator holds
//     exactly over any K this sees;
//   - a WMMA K-step is 16 elements, i.e. exactly one NVFP4 block, so no block's
//     scale ever spans two MMAs and there are no partial sums to reassemble.
//
// The decode is the budget, and nvfp4_utils.h explains why it is a table read
// rather than bit assembly: at the measured 72 GB/s this device affords ~21 VALU
// instructions per weight byte and there are two elements per byte, while an
// arithmetic decode costs ~11 per element.
//
// What is architecture-dependent here, and where it is gated:
//
//   The codec, the quantizer, the dequantizer and the stager are VALU and LDS
//   work. They compile and run anywhere, and they are what makes the packed
//   representation usable when there is no fused GEMM.
//
//   The GEMM needs WMMA. mma.h supplies it per architecture, including a
//   __builtin_trap() stub where there is no matrix core, so on RDNA2 the kernel
//   is compiled -- the translation unit must build in every device pass -- but
//   must never be launched, and __init__.py's _WMMA_ONLY_OPS is what keeps
//   scaled_mm_nvfp4 from being advertised there. quantize_nvfp4 and
//   dequantize_nvfp4 stay advertised on purpose.
//
//   The tile family and the WGP-count arm of the dispatch are gated to gfx11.
//   Both are read off the machine's shape -- 4 SIMDs a WGP, 512 VGPRs a SIMD,
//   64 KB of LDS a WGP, and 6 WGPs on the part it was swept on -- so they are
//   measurements of one architecture, and another one is better served by a
//   conservative tile than by a schedule that was never run there. See
//   CK_NVFP4_DISPATCH_OUT.
//
// See nvfp4_utils.h for the codec and the blocked scale layout.
#include <stdexcept>
#include <string>

#include "../epilogue.h"
#include "../fp8_utils.h"   // fp8_to_float, for the e4m3 block scales
#include "../gemm_wmma.h"   // frag_row / acc_row come from mma.h via here
#include "../mma.h"
#include "../nvfp4_utils.h"

namespace comfy::hip_backend {

// ---------------------------------------------------------------------------
// Quantize
// ---------------------------------------------------------------------------

// Entries in the decode table lives in nvfp4_utils.h; this is the block width the
// quantizer and dequantizer share, chosen so one block's 32-element staging chunk
// is exactly one 16-byte load per lane and a wave covers 128 blocks.
constexpr int kThreadsQuant = 256;

// The quotient has to be the IEEE one, and this file is compiled with
// -ffast-math. Measured on gfx1103 at these flags, `a / b`, __fdiv_rn and
// __ddiv_rn all return one ULP BELOW the correctly-rounded result -- -ffast-math's
// -freciprocal-math rewrites them through a single fp32 reciprocal, and the
// intrinsics are lowered through the same path.
//
// That is not cosmetic here. The quantizer is built so a block's own maximum maps
// to exactly 6.0, so |x| * 6 / absmax is a dyadic rational that lands EXACTLY on a
// rounding midpoint (1.75, 3.5, ...) a fair fraction of the time -- 1698 of
// 1310720 nibbles on a (512, 5120) tensor, every one of them a value the fast form
// read as one ULP below the midpoint and rounded the wrong way.
//
// This is the refined-reciprocal sequence the correctly-rounded division is built
// from: one correctly-rounded reciprocal, then one Newton correction, with the
// residuals kept in fma so nothing is lost between the steps. It is 4 fp32
// instructions and it reproduces torch's own fp32 division bit for bit, which is
// the arithmetic the reference expressions are evaluated with.
__forceinline__ __device__ float nvfp4_quant_div(float v, float total) {
    const float r = __frcp_rn(total);
    const float q0 = v * r;
    const float rem = __fmaf_rn(-q0, total, v);
    return __fmaf_rn(rem, r, q0);
}

// One E2M1 code from a scaled magnitude. This is the arithmetic spelling of the
// round-to-nearest-even saturating conversion that
// float_utils._f32_to_floatx_unpacked and triton/quantization.py:55-71 both
// implement; keeping it in one place means the quantizer cannot drift from the
// reference it is checked against.
//
// The sign comes from the input's bits rather than from a float comparison:
// -ffast-math implies -ffinite-math-only, so `x < 0` folds away for a NaN, while
// eager takes code bit 3 straight off the sign (float_utils.py:137). The bits
// agree for every finite input.
__forceinline__ __device__ uint32_t e2m1_encode(float x, uint32_t x_bits) {
    const float a = fabsf(x);
    // One comparison per midpoint; the sum of the seven bools is the magnitude
    // code. The comparison direction is what puts each tie on the even code:
    // 0.25 -> 0, 0.75 -> 1.0, 1.25 -> 1.0, 1.75 -> 2.0, 3.5 -> 4.0. Above 5.0 the
    // code saturates to 7 (6.0), and exactly 5.0 falls to 6 (4.0).
    const uint32_t mag = static_cast<uint32_t>((a > 0.25f) + (a >= 0.75f) + (a > 1.25f) +
                                               (a >= 1.75f) + (a > 2.5f) + (a >= 3.5f) +
                                               (a > 5.0f));
    return mag | ((x_bits >> 28) & 0x8u);
}

// 16 values per block, 4 lanes per block, 4 values per lane: four lanes then
// cover one block exactly, so the block's absmax is a two-step shuffle reduction
// and no value crosses a lane to be reduced. A lane reads its 8 input bytes as
// one uint2 and writes its 8 output bytes as one uint16_t.
template <typename IType, bool HI_FIRST>
__global__ __launch_bounds__(kThreadsQuant) void quantize_nvfp4_kernel(
    const IType* __restrict__ x, const float* __restrict__ global_scale,
    uint8_t* __restrict__ out, uint8_t* __restrict__ block_scales, int rows, int cols,
    int orig_rows, int orig_cols) {

    // One group is 4 threads, so the block covers threads/4 groups and the group
    // index has to come from the flat thread id -- taking it from the lane (tid %
    // wavefront) alone would give the same 8 groups to all 8 waves of the block.
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & (kWave - 1);
    const int in_block = tid & 3;       // lane % 4
    const int group = blockIdx.x * (kThreadsQuant / 4) + (tid >> 2);

    const int blocks_per_row = cols / kNvfp4Block;
    const int row = group / blocks_per_row;
    const int col_block = group - row * blocks_per_row;
    if (row >= rows) return;

    const int col0 = col_block * kNvfp4Block + in_block * 4;

    // pad_16x pads the OUTPUT into a 16x16 grid, so the input is addressed at its own
    // extents and a read past them becomes the zero a padded block encodes as.
    //
    // Two things have to hold for the vectorized read. The four elements this lane
    // owns must all be in range -- which is NOT the same as the lane's first one
    // being in range, and a padded width can leave a block straddling the end (a
    // (33, 65) tensor pads to (48, 80), where column 64 exists and 67 does not).
    // And the row start must be 4-element aligned for the 8- or 16-byte load, which
    // orig_cols being a multiple of 4 guarantees and a padded width need not be.
    // When either fails, read element by element and zero the tail.
    float v[4];
    const bool vectorizable = (col0 + 3 < orig_cols) && (row < orig_rows) &&
                              (orig_cols & 3) == 0;
    if (vectorizable) {
        const int64_t base = static_cast<int64_t>(row) * orig_cols + col0;
        if (sizeof(IType) == 2) {
            const uint2 raw = *reinterpret_cast<const uint2*>(x + base);
            const IType* p = reinterpret_cast<const IType*>(&raw);
            #pragma unroll
            for (int i = 0; i < 4; ++i) v[i] = static_cast<float>(p[i]);
        } else if (sizeof(IType) == 4) {
            const uint4 raw = *reinterpret_cast<const uint4*>(x + base);
            const IType* p = reinterpret_cast<const IType*>(&raw);
            #pragma unroll
            for (int i = 0; i < 4; ++i) v[i] = static_cast<float>(p[i]);
        } else {
            #pragma unroll
            for (int i = 0; i < 4; ++i) v[i] = static_cast<float>(x[base + i]);
        }
    } else {
        const bool live_row = row < orig_rows;
        const int64_t base = static_cast<int64_t>(row) * orig_cols + col0;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            v[i] = (live_row && (col0 + i) < orig_cols)
                       ? static_cast<float>(x[base + i])
                       : 0.0f;
        }
    }

    float absmax = fabsf(v[0]);
    #pragma unroll
    for (int i = 1; i < 4; ++i) absmax = fmaxf(absmax, fabsf(v[i]));

    // 4 lanes -> 1 block. The mask is 64-bit because HIP's warp intrinsics require
    // it, and every lane of the wave is active here: `group` is shared by the four
    // lanes of a group, so the early return above is wave-uniform.
    constexpr uint64_t kFullMask = 0xffffffffffffffffull;
    #pragma unroll
    for (int off = 2; off >= 1; off >>= 1) {
        absmax = fmaxf(absmax, __shfl_down_sync(kFullMask, absmax, off, kWave));
    }
    // Back over the group, so all four lanes encode against the same scale. The
    // source has to be the group's FIRST lane: __shfl_sync's srcLane names one lane
    // of the whole wave, so passing `in_block` broadcasts four unrelated lanes'
    // maxima to the entire wave and hands all eight of its groups a scale taken
    // from the first four lanes. The wave is 32 lanes and a group is 4, so tid & ~3
    // and lane & ~3 name the same lane.
    absmax = __shfl_sync(kFullMask, absmax, tid & ~3, kWave);

    // Reference (eager/quantization.py:143-147): decode_scale = absmax / 6 /
    // global_scale, clamped at 448, rounded to e4m3 -- and the ROUNDED value is
    // what the data is then encoded against. Encoding against the unrounded scale
    // instead would move every element by up to one e4m3 step (~6%).
    //
    // Both divisions go through nvfp4_quant_div for the reason given there: -ffast-math
    // lands them one ULP low, and e4m3's 3-bit mantissa turns a 1-ULP-low pre-rounding
    // value into a WHOLE-step-low scale. Of an 8-bit e4m3 exponent range there are
    // only 1024 representable magnitudes, so 19 of the 2048 midpoints in the encoding
    // are exactly representable while 2029 are not; a midpoint that is representable
    // gets a 1-ULP error for free, and here that is the difference between a scale of
    // 224 and 208 for an exact tie.
    const float gscale = global_scale[0];
    const float decode_scale =
        fminf(nvfp4_quant_div(nvfp4_quant_div(absmax, 6.0f), gscale), 448.0f);
    const uint8_t scale_fp8 = pack_fp8(decode_scale, kFp8E4M3CodeLocal);
    const float decode_scale_fp8 = fp8_to_float(scale_fp8);

    if (in_block == 0) {
        block_scales[nvfp4_scale_offset(row, col_block, blocks_per_row)] = scale_fp8;
    }

    // total_scale is what the data is divided by, built the way the reference builds
    // it (eager/quantization.py:147 and triton/quantization.py:331): per_tensor *
    // the ROUNDED block scale. A scale this small cannot represent a value the
    // e2m1 grid can hold, and the reference zeroes the block rather than dividing
    // by ~0.
    const float total_scale = gscale * decode_scale_fp8;
    const bool zero_scale = total_scale < 1e-10f;
    const float safe_scale = zero_scale ? 1.0f : total_scale;

    // Element 2i is the high nibble when hi_first. This lane owns consecutive
    // elements, so (0,1) share a byte and (2,3) the next, and the two go out as one
    // 16-bit store. b0 is the LOWER address, and x86/AMD are little-endian, so it
    // is the low half of the stored value -- getting this backwards exchanges the
    // two bytes of every pair.
    const uint32_t c0 = e2m1_encode(zero_scale ? 0.0f : nvfp4_quant_div(v[0], safe_scale),
                                    __float_as_uint(v[0]));
    const uint32_t c1 = e2m1_encode(zero_scale ? 0.0f : nvfp4_quant_div(v[1], safe_scale),
                                    __float_as_uint(v[1]));
    const uint32_t c2 = e2m1_encode(zero_scale ? 0.0f : nvfp4_quant_div(v[2], safe_scale),
                                    __float_as_uint(v[2]));
    const uint32_t c3 = e2m1_encode(zero_scale ? 0.0f : nvfp4_quant_div(v[3], safe_scale),
                                    __float_as_uint(v[3]));
    const uint32_t b0 = HI_FIRST ? ((c0 << 4) | (c1 & 0xFu)) : ((c1 << 4) | (c0 & 0xFu));
    const uint32_t b1 = HI_FIRST ? ((c2 << 4) | (c3 & 0xFu)) : ((c3 << 4) | (c2 & 0xFu));

    *reinterpret_cast<uint16_t*>(out + static_cast<int64_t>(row) * (cols / 2) + col0 / 2) =
        static_cast<uint16_t>(b0 | (b1 << 8));
}

// ---------------------------------------------------------------------------
// Dequantize
// ---------------------------------------------------------------------------

// One 16-element block per lane: a lane covers one packed uint2 (8 bytes), so a
// wave covers 32 whole blocks. Not a uint4 -- 16 bytes is two blocks, and the
// block's single scale would then be applied to both. And one thread per block,
// not four as in the quantizer: there the block's absmax spans four lanes, here
// there is nothing to reduce.
template <typename OType, bool HI_FIRST>
__global__ __launch_bounds__(kThreadsQuant) void dequantize_nvfp4_kernel(
    const uint8_t* __restrict__ in, const float* __restrict__ global_scale,
    const uint8_t* __restrict__ block_scales, OType* __restrict__ out, int rows,
    int cols) {

    __shared__ float lut[kE2m1LutSize];
    e2m1_init_lut(lut, threadIdx.x);
    __syncthreads();

    const int group = blockIdx.x * kThreadsQuant + static_cast<int>(threadIdx.x);

    const int blocks_per_row = cols / kNvfp4Block;
    const int row = group / blocks_per_row;
    const int col_block = group - row * blocks_per_row;
    if (row >= rows) return;

    const uint8_t scale_fp8 =
        block_scales[nvfp4_scale_offset(row, col_block, blocks_per_row)];
    const float gs = global_scale[0];
    const float bs = fp8_to_float(scale_fp8);

    const int col0 = col_block * kNvfp4Block;
    const uint2 packed = *reinterpret_cast<const uint2*>(
        in + static_cast<int64_t>(row) * (cols / 2) + col0 / 2);

    float v[16];
    e2m1x8_to_f32_lut<HI_FIRST>(packed, lut, v);

    // Two multiplies, value by block scale then by global, in that order: that is
    // the expression triton/quantization.py:597 evaluates, and it is the one whose
    // rounding this backend is replacing. Folding them into one product first --
    // v * (bs * gs) -- moves ~1e-4 of bf16 elements onto the other side of their
    // rounding boundary. eager differs again (it casts both scales to the output
    // dtype and multiplies there), so the three implementations are not bit-equal
    // to each other; matching triton is what leaves this op's numbers on this
    // platform unchanged.
    OType* dst = out + static_cast<int64_t>(row) * cols + col0;
    #pragma unroll
    for (int i = 0; i < 16; ++i) dst[i] = static_cast<OType>(v[i] * bs * gs);
}

// ---------------------------------------------------------------------------
// Scaled GEMM: fp16 WMMA
// ---------------------------------------------------------------------------

// LDS row stride, in halves. The +8 is 16 bytes and keeps every row 16-byte
// aligned, which is what lets a fragment's 32-byte K-step load as two
// ds_read_b128 rather than sixteen scalar reads. (gemm_wmma.h's kLdsPad is 8
// BYTES, which breaks that alignment; it can afford to, because its operands are
// already 16-bit in LDS from a byte-addressed stager.)
constexpr int kNvfp4LdsPadHalves = 8;

// Staging for one ROWS x BKB tile, where BKB counts ELEMENTS along K.
//
// A staging chunk is 32 elements -- 16 packed bytes in, 4 uint4 of fp16 out --
// which is exactly two NVFP4 blocks, so a chunk's two e4m3 scales are adjacent in
// the blocked layout (BKB is a multiple of 32, so a tile always starts on a
// 32-element boundary and a chunk's pair never straddles a 4-block column group)
// and one 16-bit load covers both. That is what makes the block scales
// foldable into the staged operand; see the note at the top of this file.
//
// Structure and naming follow gemm_wmma.h's TileStager, which stages the same
// tiles from byte-addressed operands: load() issues only the global reads so the
// caller can place the tile's math between load() and store().
template <int ROWS, int BKB, int THREADS>
struct Nvfp4Stager {
    static constexpr int kChunksPerRow = BKB / 32;
    static constexpr int kChunks = ROWS * kChunksPerRow;
    // Ceiling, not exact division: a tile can have fewer 32-element chunks than the
    // block has threads, and the surplus threads stage nothing. Rounding up and
    // skipping past kChunks below is what lets a tile be resized without also
    // having to resize its warp grid.
    static constexpr int kPerThread = (kChunks + THREADS - 1) / THREADS;
    static constexpr int kStrideHalves = BKB + kNvfp4LdsPadHalves;
    static_assert(BKB % 32 == 0, "BKB must be a whole number of 32-element chunks");

    // The chunk is staged WIDENED: load() reads and decodes, store() writes. Deferring
    // the decode to store() instead -- so the global read has a full MMA phase
    // before its first use -- was measured and is a loss, 600 against 552 ms on
    // the decode weighted total. It does shrink the staging set to one uint4 plus a
    // dword of scale, and the kernel does come out at 121 VGPRs instead of 143, but
    // that buys nothing: LDS already caps the block count at three per WGP, so
    // registers had no slack to trade and the only effect was to move the decode
    // behind the barrier. Keep the decode where the load puts it.
    uint4 regs[kPerThread][4];

    // row0/rows_total bound the tile against the real extents and kbyte0/kbytes
    // against K. Out-of-range rows and the K tail stage as zero, which the MMA
    // then contributes nothing from. A zeroed chunk also needs no scale read --
    // reading one there would index past the scale buffer for a padded row.
    //
    // The two out-of-range cases are handled alike, by zeroing, and both are worth
    // it against the alternatives:
    //
    //   M/N out of range: clamp the row to the last real one and read it anyway,
    //     letting the `r >= M` guard at writeback drop the accumulator. It is
    //     correct -- no lane of one row feeds another, so the value cannot reach the
    //     output -- and it does cut about seventeen instructions a chunk. But at M=16
    //     with BM=64 it makes three of every four A chunks issue a read they used to
    //     skip, fetching addresses already in cache, and that costs more than the
    //     branch: 583 against 570 ms on the decode weighted total.
    //
    //   The e4m3 scale is likewise not tabled. 256 entries of the fp16 already
    //     doubled, replacing two fp8_to_float conversions per chunk with two LDS
    //     reads, is 570 against 552 ms. It trades about 57 ALU and scalar issue
    //     slots for four LDS reads and comes out slower, which puts a number on what
    //     an LDS instruction costs here: roughly fourteen ALU slots. That is why the
    //     byte-indexed data table is already at its floor -- one read per byte, the
    //     minimum a 4-bit-indexed table admits -- and why trimming ALU further is
    //     not where the remaining time is.
    __forceinline__ __device__ void load(const uint8_t* __restrict__ src,
                                         const uint8_t* __restrict__ scales, int row0,
                                         int rows_total, int kbyte0, int kbytes,
                                         int blocks_per_row, const uint32_t* lut) {
        const int tid = threadIdx.x;
        #pragma unroll
        for (int i = 0; i < kPerThread; ++i) {
            const int c = tid + i * THREADS;
            if (c >= kChunks) break;
            const int grow = row0 + c / kChunksPerRow;
            const int gk = kbyte0 + (c % kChunksPerRow) * 16;
            const int kb = gk >> 3;  // 16 packed bytes = 2 blocks
            if (grow < rows_total && (gk + 16) <= kbytes) {
                const uint4 raw = *reinterpret_cast<const uint4*>(
                    src + static_cast<int64_t>(grow) * kbytes + gk);
                const uint16_t sp = *reinterpret_cast<const uint16_t*>(
                    scales + nvfp4_scale_offset(grow, kb, blocks_per_row));
                e2m1x16_to_half_scaled(
                    raw, lut, __float2half(fp8_to_float(sp & 0xFFu)),
                    __float2half(fp8_to_float(sp >> 8)), regs[i]);
            } else {
                #pragma unroll
                for (int w = 0; w < 4; ++w) regs[i][w] = make_uint4(0, 0, 0, 0);
            }
        }
    }

    __forceinline__ __device__ void store(__half* lds) const {
        const int tid = threadIdx.x;
        #pragma unroll
        for (int i = 0; i < kPerThread; ++i) {
            const int c = tid + i * THREADS;
            if (c >= kChunks) break;
            __half* dst =
                lds + (c / kChunksPerRow) * kStrideHalves + (c % kChunksPerRow) * 32;
            // kStrideHalves is even, so every chunk base is 32-byte aligned and
            // these are four 16-byte stores. The order the two halves of each dword
            // end up in is not claimed here, and deliberately so: nvfp4_utils.h above
            // e2m1x16_to_half_scaled says why, and the short version is that both
            // operands come through here, so any within-block permutation is shared
            // and no output bit moves.
            *reinterpret_cast<uint4*>(dst) = regs[i][0];
            *reinterpret_cast<uint4*>(dst + 8) = regs[i][1];
            *reinterpret_cast<uint4*>(dst + 16) = regs[i][2];
            *reinterpret_cast<uint4*>(dst + 24) = regs[i][3];
        }
    }
};

// Block-rows grouped so that concurrently resident blocks share the same B
// columns, which is what keeps the streamed operand in L2. Shared by both tile
// kernels: the grouping is about the grid, not about the MMA.
constexpr int kNvfp4GroupM = 4;

// C[M, N] with A[M, K] and B[N, K] NVFP4, block scales folded into the staged fp16
// operands. Mirrors gemm_wmma_kernel's structure -- grouped block ordering for L2
// reuse, a register-level pipeline over K-steps inside each tile -- with the
// staging replaced by the decode.
template <typename OutT, int BM, int BN, int BKB, int WARPS_M, int WARPS_N, int TM, int TN,
          int PIPE = 2>
// PIPE is the K-step register pipeline depth; see the comment where the fragments
// are declared for why it is a parameter and not a constant.
// VGPRs per SIMD, so 4 waves costs 128 VGPRs a lane and 5 costs 102. The natural
// allocation for these tiles is 142 -- which is 3 waves, because 512/142 rounds
// down -- so the third wave is the last one the shape gets for free and the
// fourth has to be asked for.
//
// __launch_bounds__'s second argument is meant to be that lever (min blocks per
// SM: 4 for a 128-thread tile, 2 for a 256-thread one), but it is ignored on this
// target: with it the allocation stayed at 142. So PIPE is the lever instead, and
// whether it pays is measured rather than argued -- hipisa.ps1 then isa_analyze.py
// report the VGPR count and any spill, and the sweep below reports the time.
__global__ __launch_bounds__(WARPS_M* WARPS_N* kWave) void nvfp4_gemm_kernel(
    const uint8_t* __restrict__ A, const uint8_t* __restrict__ B, OutT* __restrict__ C,
    const uint8_t* __restrict__ sa, const uint8_t* __restrict__ sb, int M, int N, int K,
    int a_blocks, int b_blocks, int ldc, EpiTensorwise epi) {

    constexpr int kThreads = WARPS_M * WARPS_N * kWave;
    constexpr int kStride = (BKB + kNvfp4LdsPadHalves) * 2;  // MMA policies address in bytes
    // MmaF16::kStepBytes is in bytes and the tile is in elements, so a K-step is
    // half of it. On gfx11 that is 16 elements -- which is exactly one NVFP4 block,
    // the property the whole scale-folding argument rests on.
    constexpr int kStepElems = MmaF16::kStepBytes / 2;
    constexpr int kSteps = BKB / kStepElems;
    static_assert(BM == WARPS_M * TM * 16, "the M warp grid must tile BM exactly");
    static_assert(BN == WARPS_N * TN * 16, "the N warp grid must tile BN exactly");
    static_assert(BKB % kStepElems == 0, "BKB must hold whole MMA K-steps");

    __shared__ __align__(16) __half As[BM * (BKB + kNvfp4LdsPadHalves)];
    __shared__ __align__(16) __half Bs[BN * (BKB + kNvfp4LdsPadHalves)];
    // Byte-indexed decode table: 256 packed half-pairs, 1 KB. See
    // e2m1x16_to_half_scaled for why the byte is the index rather than the nibble.
    __shared__ __align__(16) uint32_t lut[kE2m1ByteLutSize];

    e2m1_init_byte_lut(lut, threadIdx.x, kThreads);
    __syncthreads();

    const int tid = threadIdx.x;
    const int lane = tid % kWave;
    const int warp = tid / kWave;
    const int wm = warp / WARPS_N;
    const int wn = warp % WARPS_N;

    // Grouped block ordering: consecutive blocks advance along M within a group of
    // kNvfp4GroupM block-rows, so concurrently resident blocks share the same B
    // columns and the streamed operand stays in L2. The constant is file-scope
    // because both tile kernels index their blocks the same way.
    const int blocks_n = gridDim.x;
    const int blocks_m = gridDim.y;
    const int bid = blockIdx.y * blocks_n + blockIdx.x;
    const int per_group = kNvfp4GroupM * blocks_n;
    const int group = bid / per_group;
    const int idx_in_group = bid - group * per_group;
    const int group_rows = min(kNvfp4GroupM, blocks_m - group * kNvfp4GroupM);
    const int bm = group * kNvfp4GroupM + idx_in_group % group_rows;
    const int bn = idx_in_group / group_rows;

    const int m0 = bm * BM;
    const int n0 = bn * BN;
    const int kbytes = K / 2;

    typename MmaF16::Acc acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = MmaF16::zero();

    const int row = frag_row(lane);

    Nvfp4Stager<BM, BKB, kThreads> stage_a;
    Nvfp4Stager<BN, BKB, kThreads> stage_b;

    stage_a.load(A, sa, m0, M, 0, kbytes, a_blocks, lut);
    stage_b.load(B, sb, n0, N, 0, kbytes, b_blocks, lut);
    stage_a.store(As);
    stage_b.store(Bs);
    __syncthreads();

    for (int kb0 = 0; kb0 < K; kb0 += BKB) {
        const int knext = kb0 + BKB;
        const bool has_next = knext < K;

        // Issue the next tile's global reads before the current tile's math.
        if (has_next) {
            stage_a.load(A, sa, m0, M, knext / 2, kbytes, a_blocks, lut);
            stage_b.load(B, sb, n0, N, knext / 2, kbytes, b_blocks, lut);
        }

        // Register-level pipeline over K-steps: step kk+1's LDS reads are issued
        // before step kk's MMAs. PIPE is the depth, and it is a template parameter
        // because it trades LDS latency hiding against the register count, and the
        // register count is what caps the waves: 512 VGPRs per SIMD means 142 costs
        // 3 waves and 128 costs 4. At depth 2 the fragments alone are
        // 2*(TM+TN) fragments -- 32 VGPRs at TM=TN=2 -- so dropping to depth 1 is
        // what a tile would try to reach the fourth wave. Which way it lands is
        // measured, not argued: hipisa.ps1 + isa_analyze.py give the VGPR count and
        // any spill, and the sweep gives the time.
        typename MmaF16::Frag af[PIPE][TM];
typename MmaF16::Frag bf[PIPE][TN];

// Step 0's fragments, so the depth-2 loop has something to issue the step-1
        // loads against. Depth 1 loads step kk inside the loop and needs no
        // priming -- leaving this unconditional would be dead code the compiler
        // happens to remove, which is one more thing to have to check.
        if (PIPE == 2) {
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                af[0][i] = MmaF16::load(As, wm * (TM * 16) + i * 16 + row, 0, kStride, lane);
            #pragma unroll
            for (int j = 0; j < TN; ++j)
                bf[0][j] = MmaF16::load(Bs, wn * (TN * 16) + j * 16 + row, 0, kStride, lane);
        }

        #pragma unroll
        for (int kk = 0; kk < kSteps; ++kk) {
        // af[cur] holds step kk's fragments. At depth 2 the next step's loads go
        // into the other buffer so they overlap step kk's MMAs; at depth 1 there is
        // nowhere to put them, so each step loads its own immediately before its
        // MMAs and gives up the overlap.
        //
        // `cur` is indexed by step for BOTH depths. Pinning it to 0 -- which is
        // what `PIPE == 2 ? (kk & 1) : 0` did -- is only correct if every step
        // reloads, and at depth 1 the reload was gated behind `PIPE == 2`, so the
        // buffer kept the kbyte=0 fragment the pre-loop load had put there and all
        // kSteps MMAs ran the same 16-element dot product. Measured at
        // M=8/N=32/K=128: relative error 2.28, against 0.0025 for depth 2.
        //
        // This matters beyond the knob itself: the tile table in this file reports
        // 64x64x64 PIPE=1 as a 1.02x loser, and that number was timing the wrong
        // answer. The PIPE=1 rows were re-measured after the fix.
        const int cur = kk & (PIPE - 1);
        if (PIPE == 2) {
            if (kk + 1 < kSteps) {
                const int kbyte = (kk + 1) * MmaF16::kStepBytes;
                #pragma unroll
                for (int i = 0; i < TM; ++i)
                    af[cur ^ 1][i] =
                        MmaF16::load(As, wm * (TM * 16) + i * 16 + row, kbyte, kStride, lane);
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    bf[cur ^ 1][j] =
                        MmaF16::load(Bs, wn * (TN * 16) + j * 16 + row, kbyte, kStride, lane);
            }
        } else {
            const int kbyte = kk * MmaF16::kStepBytes;
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                af[0][i] = MmaF16::load(As, wm * (TM * 16) + i * 16 + row, kbyte, kStride, lane);
            #pragma unroll
            for (int j = 0; j < TN; ++j)
                bf[0][j] = MmaF16::load(Bs, wn * (TN * 16) + j * 16 + row, kbyte, kStride, lane);
        }
        #pragma unroll
        for (int i = 0; i < TM; ++i)
            #pragma unroll
            for (int j = 0; j < TN; ++j)
                acc[i][j] = MmaF16::mma(af[cur][i], bf[cur][j], acc[i][j]);
        }

        if (has_next) {
            __syncthreads();  // every warp has finished reading the current tile
            stage_a.store(As);
            stage_b.store(Bs);
            __syncthreads();
        }
    }

    // Identical to gemm_wmma_kernel's writeback, including the hand-hoisting: the
    // stores go through OutT*, which the compiler must assume may alias the scale
    // and bias pointers, so without this every load is re-issued per output.
    epi.init();

    const int col_lane = acc_col(lane);
    float col_bias_v[TN];
    #pragma unroll
    for (int j = 0; j < TN; ++j) {
        const int col = n0 + wn * (TN * 16) + j * 16 + col_lane;
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
                crow[col] = static_cast<OutT>(
                    epi(r, col, MmaF16::get(acc[i][j], e), rs, 1.0f, col_bias_v[j]));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Integer WMMA: measured, and it does not work for NVFP4
// ---------------------------------------------------------------------------

// There is no int8 kernel here, and that is a result rather than an omission. It
// was built, made exact, and measured; the numbers and the argument are kept
// because the argument is the useful part.
//
// **The MMA, the decode, the fragment layout and the K loop were all exact.** With
// the block scales held constant and the operands random, the result matched a
// Python reference to 0 across K = 64..512 and M/N up to 512x1024. Two real bugs
// came out of getting there, both worth knowing if this is ever revisited: the
// nibble decode indexed its output with stride 4 where two packed bytes need 8,
// writing 16 values into 11 slots; and the staging register was declared 64 bits
// for a 128-bit chunk, so the upper half of every chunk's LDS slot was never
// written and the MMA read it uninitialized.
//
// **What cannot work is the scale, and it is structural.** e2m1's magnitude grid
// doubles into {0, 1, 2, 3, 4, 6, 8, 12}, all exact in int8, so an int8 path
// would be lossless rather than a precision trade -- and half a bit the speed of
// an fp16 MMA here. But an e4m3 scale has no integer form, so it cannot be folded
// into an int8 operand the way e2m1x16_to_half_scaled folds it for fp16 -- and
// that fold is not an optimization, it is the only reason a per-16 block scale is
// compatible with a 16x16 MMA at all. Left out of the operand, the scale has to be
// applied to the MMA's output, where on gfx11 acc_row(lane, e) = 2e + lane/16 puts
// 8 different rows with 8 different scales in one lane's accumulator while the
// lane holds exactly one A fragment. No single scale is right for all eight.
// Per element would mean one MMA per element: 8x the MMAs, 8x the fragment loads,
// and nothing of the int8 advantage left.
//
// gemm_wmma.h already says this in its type system: its int8 MMA runs with
// EpiRowwise, per-row scales, and there is no block-scaled int8 epilogue anywhere,
// while the fp16 and fp8 paths have one. NVFP4's granularity is per 16 elements;
// int8 WMMA's is per row. int4 is worse -- the doubled grid's top value 12 does not
// fit a 4-bit integer, so int4 would drop the doubling and start rounding the
// 0.5, 1.5 and 3 entries.
//
// The timing, in case the shape of the loss is useful: 1378 ms against the fp16
// tile's 605 on the decode loop's weighted total, 2.28x slower. The ISA explains
// it -- 128 v_cvt_f32_i32 and 94 v_fmac_f32 of rescale inside a 535-instruction
// inner loop, plus 211 VGPRs where the fp16 tile needs 143, which is 2 waves per
// SIMD against 3. Even with a correct scale it could not have won.

// K-step register pipeline depth, 1 or 2. Read once per process, so it costs
// nothing after the first call. 2 hides the LDS latency behind the MMAs; 1 halves
// the fragment registers, which is the only route these tiles have to a fourth wave
// per SIMD (see the kernel's template comment). Which one wins is a measurement,
// and it is the one knob here whose answer is not obvious from the ISA: the
// register count says 1 should buy the fourth wave, the instruction count says 2
// should hide more latency. On this part 2 wins, so that is the default.
inline int comfy_nvfp4_pipe() {
    static const int v = [] {
        const char* s = getenv("COMFY_NVFP4_PIPE");
        const int p = (s && *s) ? atoi(s) : 2;
        return (p == 1) ? 1 : 2;
    }();
    return v;
}

}  // namespace comfy::hip_backend

extern "C" void launch_quantize_nvfp4_kernel(
    const void* x, const void* global_scale, void* out, void* block_scales, int rows,
    int cols, int orig_rows, int orig_cols, int input_dtype_code, bool hi_first,
    hipStream_t stream) {

    using namespace comfy::hip_backend;

    if (rows == 0 || cols == 0) {
        return;  // a zero-dimension grid is hipErrorInvalidConfiguration
    }
    // One scale per 16 elements and one packed byte per 2, so the PADDED column
    // count must be a whole number of blocks or a block would straddle the row's
    // end. orig_rows/orig_cols are unconstrained -- a padded call is exactly the
    // case where they are not multiples of 16, which is why they are separate.
    if (cols % kNvfp4Block != 0) {
        throw std::runtime_error("quantize_nvfp4: the padded column count " +
                                 std::to_string(cols) + " must be a multiple of 16");
    }
    if (rows < orig_rows || cols < orig_cols) {
        throw std::runtime_error("quantize_nvfp4: the padded extents must not be smaller "
                                 "than the original ones");
    }

    dim3 block(kThreadsQuant);
    // One block covers kThreadsQuant/4 blocks; one wave covers kWave/4.
    dim3 grid((rows * (cols / kNvfp4Block) + kThreadsQuant / 4 - 1) / (kThreadsQuant / 4));

#define CK_NVFP4_QUANT_LAUNCH(T, HF)                                                          \
    quantize_nvfp4_kernel<T, HF><<<grid, block, 0, stream>>>(                                 \
        static_cast<const T*>(x), static_cast<const float*>(global_scale),                   \
        static_cast<uint8_t*>(out), static_cast<uint8_t*>(block_scales), rows, cols, orig_rows, \
        orig_cols)

    if (input_dtype_code == kF32) {
        if (hi_first) { CK_NVFP4_QUANT_LAUNCH(float, true); }
        else { CK_NVFP4_QUANT_LAUNCH(float, false); }
    } else if (input_dtype_code == kF16) {
        if (hi_first) { CK_NVFP4_QUANT_LAUNCH(__half, true); }
        else { CK_NVFP4_QUANT_LAUNCH(__half, false); }
    } else if (input_dtype_code == kBF16) {
        if (hi_first) { CK_NVFP4_QUANT_LAUNCH(__bf16, true); }
        else { CK_NVFP4_QUANT_LAUNCH(__bf16, false); }
    } else {
        throw std::runtime_error("quantize_nvfp4: unsupported input dtype");
    }
#undef CK_NVFP4_QUANT_LAUNCH
}

extern "C" void launch_dequantize_nvfp4_kernel(
    const void* in, const void* global_scale, const void* block_scales, void* out, int rows,
    int cols, int output_dtype_code, bool hi_first, hipStream_t stream) {

    using namespace comfy::hip_backend;

    if (rows == 0 || cols == 0) {
        return;
    }
    if (cols % kNvfp4Block != 0) {
        throw std::runtime_error("dequantize_nvfp4: cols=" + std::to_string(cols) +
                                 " must be a multiple of 16");
    }

    dim3 block(kThreadsQuant);
    // One thread per 16-element block, so a block of threads covers kThreadsQuant
    // of them. (The quantizer above is the opposite: four threads per block, to
    // reduce the absmax, so it covers kThreadsQuant/4.)
    dim3 grid((rows * (cols / kNvfp4Block) + kThreadsQuant - 1) / kThreadsQuant);

#define CK_NVFP4_DEQUANT_LAUNCH(T, HF)                                                         \
    dequantize_nvfp4_kernel<T, HF><<<grid, block, 0, stream>>>(                               \
        static_cast<const uint8_t*>(in), static_cast<const float*>(global_scale),              \
        static_cast<const uint8_t*>(block_scales), static_cast<T*>(out), rows, cols)

    if (output_dtype_code == kF32) {
        if (hi_first) { CK_NVFP4_DEQUANT_LAUNCH(float, true); }
        else { CK_NVFP4_DEQUANT_LAUNCH(float, false); }
    } else if (output_dtype_code == kF16) {
        if (hi_first) { CK_NVFP4_DEQUANT_LAUNCH(__half, true); }
        else { CK_NVFP4_DEQUANT_LAUNCH(__half, false); }
    } else if (output_dtype_code == kBF16) {
        if (hi_first) { CK_NVFP4_DEQUANT_LAUNCH(__bf16, true); }
        else { CK_NVFP4_DEQUANT_LAUNCH(__bf16, false); }
    } else {
        throw std::runtime_error("dequantize_nvfp4: unsupported output dtype");
    }
#undef CK_NVFP4_DEQUANT_LAUNCH
}

extern "C" void launch_scaled_mm_nvfp4_kernel(
    const void* a, const void* b, void* c, const void* sa, const void* sb,
    const void* tsa, const void* tsb, const void* bias, int bias_code, int M, int N, int K,
    int ldc, int out_code, hipStream_t stream) {

    using namespace comfy::hip_backend;

    if (M == 0 || N == 0) {
        return;
    }
    // Every path reads a packed row a 16-byte chunk at a time, and a block scale
    // covers 16 elements, so K must be a whole number of blocks and the packed row
    // a whole number of chunks.
    if (K % 32 != 0) {
        throw std::runtime_error("scaled_mm_nvfp4: K=" + std::to_string(K) +
                                 " must be a multiple of 32");
    }

    const int a_blocks = K / kNvfp4Block;
    const int b_blocks = a_blocks;
    const uint8_t* A = static_cast<const uint8_t*>(a);
    const uint8_t* B = static_cast<const uint8_t*>(b);
    const uint8_t* SA = static_cast<const uint8_t*>(sa);
    const uint8_t* SB = static_cast<const uint8_t*>(sb);
    const float* TSA = static_cast<const float*>(tsa);
    const float* TSB = static_cast<const float*>(tsb);

#define CK_NVFP4_GEMM_LAUNCH_P(T, BM, BN, BKB, WM, WN, TM, TN, PIPE)                         \
    do {                                                                                      \
        dim3 grid((N + (BN)-1) / (BN), (M + (BM)-1) / (BM));                                   \
        nvfp4_gemm_kernel<T, BM, BN, BKB, WM, WN, TM, TN, PIPE>                              \
            <<<grid, (WM) * (WN) * kWave, 0, stream>>>(                                       \
                A, B, static_cast<T*>(c), SA, SB, M, N, K, a_blocks, b_blocks, ldc, epi);    \
        return;                                                                               \
    } while (0)
// PIPE is a template argument, so COMFY_NVFP4_PIPE has to be a runtime branch
// between two instantiations rather than a value passed in.
#define CK_NVFP4_GEMM_LAUNCH(T, BM, BN, BKB, WM, WN, TM, TN)                                  \
    do {                                                                                      \
        if (comfy_nvfp4_pipe() == 1) {                                                        \
            CK_NVFP4_GEMM_LAUNCH_P(T, BM, BN, BKB, WM, WN, TM, TN, 1);                       \
        } else {                                                                              \
            CK_NVFP4_GEMM_LAUNCH_P(T, BM, BN, BKB, WM, WN, TM, TN, 2);                       \
        }                                                                                     \
    } while (0)

    // The small-M path, and the tiled path's own tile choice, are the two knobs

// Tile choice on the WMMA path.
//
// Which tile wins is decided by occupancy, and occupancy is a property the
// assembly reports without any hardware support: dump the ISA and the tiles
// differ by their VGPR count and their LDS footprint, which on gfx1103 (512 VGPRs
// and 64 KB of LDS per WGP, 4 SIMDs) is what sets the waves per SIMD:
//
//   128x128x64  253 VGPR  37888 B LDS  -> 2 waves/SIMD, 1 block/WGP
//    64x64x64   143 VGPR  19456 B LDS  -> 3 waves/SIMD, 3 blocks/WGP
//    64x64x64   127 VGPR  19456 B LDS  -> PIPE=1: still 3, LDS-capped
//    64x64x32   118 VGPR  11264 B LDS  -> 5 blocks/WGP either depth
//   128x64x32   118 VGPR  16384 B LDS  -> PIPE=1: 4 waves/SIMD
//
// Every one of those VGPR and LDS counts is read out of the assembly by
// hipisa.ps1 + isa_analyze.py rather than computed, because the byte-indexed decode
// table is 1 KB of the LDS figure and hand-arithmetic is how that gets forgotten:
// 64x64x64 reads 19456 B where it used to read 18496 B, and 128x64x32 lands on
// exactly 16384 B -- which is the four-block line, reached but not passed, so the
// table has cost that tile its fourth wave.
//
// Occupancy is the lever because the kernel is waiting on memory, not on
// dispatch: its 611 instructions per BK step predict 1.36 ms for N=25600/K=5120
// at 128x128 and 1.29 ms at 64x64, against 4.56 ms and 3.27 ms measured.
//
// A tile stages (BM+BN) rows of BKB elements, so halving BKB is what buys the LDS
// back; BKB=32 is legal because the stager's unit is a 32-element chunk and the
// WMMA K-step is 16.
//
// Measured on gfx1103, one process per candidate with 300 warmup calls first and
// the choice made per launch through COMFY_GEMM_WMMA_TILE and COMFY_NVFP4_PIPE (a
// rebuild per candidate is not usable here: the first GEMM after a rebuild runs on
// cold clocks, measured 1.9x). Decode is the weighted per-token total over the
// five shapes the decode loop issues, 350 GEMMs to a token:
//
//   tile            decode ms   vs 64x64x64
//   128x128x64 p2       840.5        1.50
//    64x64x64  p2       562.2        1.00   <- the default
//    64x64x64  p1       559.6        1.00
//    64x64x32  p2       775.5        1.39
//    64x64x32  p1       808.0        1.44
//   128x64x32  p2      1097.5        1.96
//   128x64x32  p1      1082.4        1.93
//
// **The PIPE=1 rows in this table were re-measured, and the old ones were wrong.**
// The earlier table reported 64x64x64 PIPE=1 as a 1.02x loser and 64x64x32 PIPE=1 as
// a 1.45x loser; those kernels ran the same 16-element dot product kSteps times (see
// the K-step loop) and were not producing the numbers at all. Re-measured, the two
// depths are a TIE at 64x64x64: interleaved in one process over 7 rounds, median
// 482.1 against 481.3 ms, ratio 1.002, per-round spread 9.6% and 10.8%. That spread
// is why a single-shot sequential sweep cannot separate them -- two full runs of
// the table above put the gap at 5.3% and then at 0.5%.
//
// So depth 1 stays off by default, on the tie and not on a measured loss: it buys
// no wave, because the LDS footprint already capped the block count at 3 per WGP
// (19456 B, against a 16384 B budget for four), so what it saves is the second
// fragment buffer -- 16 VGPRs, 143 down to 127 -- and what it gives up is the LDS
// overlap. Those cancel here. On a part with more LDS headroom, or a tile whose
// fragments were the bottleneck, the trade would not cancel.
//
// Note that BKB=32 reaches 5 blocks per WGP at 11264 B but not 4 waves per SIMD
// (512/118 is 4.3, and 5 blocks of 4 warps over 4 SIMDs is 5 waves' worth of
// threads -- the register count is no longer the binding constraint there, and
// BKB=32 still loses by 1.4x). And 128x64x32 sits at exactly 16384 B, the
// four-block budget, which the decode table's 1 KB would have to share.
//
// The two stable results in the table are the ones worth keeping: a 128-row tile
// costs 1.4-2.0x, and halving BKB costs 1.4x. Halving BKB does relieve the LDS --
// 11264 B is 5 blocks per WGP -- but it starves the stager, because at BKB=32 a
// 64-row tile has 64 chunks for 128 threads and half of them stage nothing, and
// every saving comes back as loop trips and barriers. Occupancy is the ceiling on
// this kernel and neither route up to it is free, so what is left is 64x64x64 at
// 3 waves.
//
// The sweep driver now checks each candidate's answer before timing it and reports
// a wrong one as a wrong one, rather than as a row in this table. That is the only
// reason the two PIPE=1 rows above mean anything.
//
// COMFY_GEMM_WMMA_TILE forces a tile and COMFY_NVFP4_PIPE the K-step pipeline
// depth, on the same reasoning as gemm_wmma.h.
//
// The tile family below, and the WGP-count arm of it, are measured on gfx1103 and
// mean nothing on another architecture. gfx11 has 4 SIMDs per WGP, 512 VGPRs per
// SIMD and 64 KB of LDS per WGP, and the 780M has 6 of them; gfx12 has 8 SIMDs per
// WGP and a different LDS budget, and the dGPU parts have 20-96 WGPs rather than 6.
// Every number in the table above is a property of that machine, so the family is
// gated to gfx11 and the other architectures get one conservative tile instead of
// inheriting a schedule whose measurements do not apply to them.
//
// COMFY_MMA_GFX11 rather than COMFY_MMA_GFX1103: the tile table was swept on the
// 780M, but what it is reading off the architecture is the 4-SIMD WGP and the
// 512-VGPR/64 KB-LDS budget, which is the whole gfx11 family and not one SKU of it.
// This is the same arch gate mma.h's hw_exp2 uses, resolved at compile time because
// the choice sits in a launcher.
#if defined(COMFY_MMA_GFX11)
#define CK_NVFP4_DISPATCH_OUT(T)                                                              \
    do {                                                                                      \
        const EpiTensorwise epi{TSA, TSB, bias, bias_code, 0.0f};                              \
        const GemmTileOverride ovr = comfy_gemm_wmma_tile_override();                         \
        if (ovr.bm == 64 && ovr.bkb == 32) {                                                   \
            CK_NVFP4_GEMM_LAUNCH(T, 64, 64, 32, 2, 2, 2, 2);                                 \
        } else if (ovr.bm == 64) {                                                            \
            CK_NVFP4_GEMM_LAUNCH(T, 64, 64, 64, 2, 2, 2, 2);                                 \
        } else if (ovr.bm == 128 && ovr.bkb == 32) {                                          \
            CK_NVFP4_GEMM_LAUNCH(T, 128, 64, 32, 4, 2, 2, 2);                                \
        } else if (ovr.bm == 128) {                                                           \
            CK_NVFP4_GEMM_LAUNCH_P(T, 128, 128, 64, 4, 2, 2, 4, 2);                           \
        } else if (((M + 63) / 64) * ((N + 63) / 64) >= device_wgp_count()) {                 \
            CK_NVFP4_GEMM_LAUNCH(T, 64, 64, 64, 2, 2, 2, 2);                                 \
        } else {                                                                              \
            CK_NVFP4_GEMM_LAUNCH_P(T, 128, 128, 64, 4, 2, 2, 4, 2);                           \
        }                                                                                     \
    } while (0)
#else
// Not gfx11. The override still reaches the WHOLE family, because that is the
// point of it: a sweep that silently substitutes a tile for the one it asked for
// reports "that tile does not matter" for a tile it never ran, which is the exact
// failure gemm_wmma.h warns about for an unlisted tile. So every override arm is
// spelled out here as it is above. What is gfx1103-only and therefore NOT here is
// the `device_wgp_count()` arm of the default: it asks whether the grid is big
// enough to keep the machine busy, and the answer that was right for 6 WGPs is not
// the answer for 20-96. The default is the single tile the decode loop spends its
// time in, chosen from the shape rather than from a measurement on this part.
//
// The reasoning behind that default: M is 16-40 against an N in the thousands, so
// the grid is one row of M-tiles and the choice comes down to how little padding a
// narrow M tile wastes -- 64 rows against a 16-row M is 4x the MMAs, and 128 would
// be 8x. 64x64x64 with 2x2 warps is also the tile the gfx1103 sweep found best on
// that shape, which is an argument from the shape and one measurement on one part,
// not a measurement on this architecture.
#define CK_NVFP4_DISPATCH_OUT(T)                                                              \
    do {                                                                                      \
        const EpiTensorwise epi{TSA, TSB, bias, bias_code, 0.0f};                              \
        const GemmTileOverride ovr = comfy_gemm_wmma_tile_override();                         \
        if (ovr.bm == 64 && ovr.bkb == 32) {                                                   \
            CK_NVFP4_GEMM_LAUNCH(T, 64, 64, 32, 2, 2, 2, 2);                                 \
        } else if (ovr.bm == 64) {                                                            \
            CK_NVFP4_GEMM_LAUNCH(T, 64, 64, 64, 2, 2, 2, 2);                                 \
        } else if (ovr.bm == 128 && ovr.bkb == 32) {                                          \
            CK_NVFP4_GEMM_LAUNCH(T, 128, 64, 32, 4, 2, 2, 2);                                \
        } else if (ovr.bm == 128) {                                                           \
            CK_NVFP4_GEMM_LAUNCH_P(T, 128, 128, 64, 4, 2, 2, 4, 2);                           \
        } else {                                                                              \
            CK_NVFP4_GEMM_LAUNCH(T, 64, 64, 64, 2, 2, 2, 2);                                 \
        }                                                                                     \
    } while (0)
#endif

    if (out_code == kBF16) {
        CK_NVFP4_DISPATCH_OUT(__bf16);
    } else if (out_code == kF16) {
        CK_NVFP4_DISPATCH_OUT(__half);
    } else if (out_code == kF32) {
        CK_NVFP4_DISPATCH_OUT(float);
    } else {
        throw std::runtime_error("scaled_mm_nvfp4: unsupported output dtype");
    }
#undef CK_NVFP4_DISPATCH_OUT
#undef CK_NVFP4_GEMM_LAUNCH
}
