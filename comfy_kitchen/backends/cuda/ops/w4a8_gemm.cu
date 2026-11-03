// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// W4A8 / W6A8 weight dequant: grouped int4 or int6 -> int8 for the tuned int8-GEMM path.
//
// AsymW4A8Int8Layout dequantizes packed weights to "grouped int8" (per-group scale
// folded in, per-channel scale left for the int8 GEMM epilogue), then runs comfy's
// tuned int8 CUTLASS GEMM. So this file is just the memory-bound dequant kernel
// (fp32/fp8-e4m3 group scales, optional 4-bit codebook); the matmul is cutlass_gemm_int8.
//
// Storage contract (bits = 3, 4, 5 or 6, implied by the packed row width K*bits/8):
//   4/5/6 bits: bytes [0, K/2) nibble plane, even col = low nibble
//   6 bits: then [K/2, 3K/4) each code's top 2 bits: col c -> byte c/4, bit 2*(c%4)
//   5 bits: then [K/2, 5K/8) each code's top bit: col c -> byte c/8, bit c%8
//   3 bits: [0, K/4) each code's low 2 bits (byte c/4, bit 2*(c%4)), then [K/4, 3K/8) the top bit
// 5-bit levels are uniform (c - 16), no codebook; 3-bit codes index an 8-entry codebook.
// 6-bit levels are uniform (c - 32), no codebook; K % 32 == 0 keeps rows 8-byte aligned and
// G is a multiple of 16 so a decode vector never spans groups.

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <type_traits>
#include <limits>
#include <stdexcept>
#include <string>

#include "dtype_dispatch.cuh"
#include "float_utils.cuh"

// Grouped int4 -> int8 dequant for the int8-GEMM W4A8 path: out[n,k] =
// round((q_u[n,k]-8) * s_rel[n, k/G]), q_u packed uint4 (even col=low nibble).
// s_rel = per-group scale / per-channel scale (so the int8 range is used). The
// per-channel scale is applied later in the int8 GEMM epilogue. Memory-bound.
namespace {

// Runs f(std::integral_constant<int, bits>) for the code width (3, 5, 6; anything else is 4).
template <class F>
void dispatch_bits(int bits, F&& f) {
    switch (bits) {
        case 3: f(std::integral_constant<int, 3>{}); break;
        case 5: f(std::integral_constant<int, 5>{}); break;
        case 6: f(std::integral_constant<int, 6>{}); break;
        default: f(std::integral_constant<int, 4>{}); break;
    }
}
// Per-group scale is fp32 or fp8 (e4m3). fp8 halves the scale metadata at a tiny
// quality cost. uint8_t storage == e4m3 raw bits.
template <typename ScaleT> __device__ __forceinline__ float load_scale(ScaleT v);
template <> __device__ __forceinline__ float load_scale<float>(float v) { return v; }
template <> __device__ __forceinline__ float load_scale<uint8_t>(uint8_t v) {
    return __half2float(__nv_cvt_fp8_to_halfraw(v, __NV_E4M3));
}

template <typename T> __device__ __forceinline__ T store_output(float value);
template <> __device__ __forceinline__ float store_output<float>(float value) { return value; }
template <> __device__ __forceinline__ __half store_output<__half>(float value) { return __float2half(value); }
template <> __device__ __forceinline__ __nv_bfloat16 store_output<__nv_bfloat16>(float value) { return __float2bfloat16(value); }

// The int8 grid shared by every W4A8/W6A8 path (and the eager/Triton decoders):
// round-to-nearest-even of the fp32 product, clamped to +-127. Done without cvt
// instructions (16/clk/SM, the decode bottleneck): adding 1.5 * 2^23 puts the fp32
// rounding on the unit grid, so the sum is 0x4B400000 + q bit for bit (|v * s| < 2^22,
// true for every stored scale), the clamp stays in fp32, and the low byte of that
// pattern is q's two's complement.
constexpr float kRoundMagic = 12582912.0f;
__device__ __forceinline__ float grid_clamp(float biased) {
    return fminf(fmaxf(biased, kRoundMagic - 127.0f), kRoundMagic + 127.0f);
}
__device__ __forceinline__ int8_t level_to_int8(float v, float s) {
    // product rounded first, as the eager path (v * s need not be exact)
    return static_cast<int8_t>(__float_as_int(grid_clamp(__fadd_rn(__fmul_rn(v, s), kRoundMagic))));
}
// Uniform level (c - zero) of a code c < 2^23 as fp32, exactly, without an I2F.
__device__ __forceinline__ float code_level(unsigned c, float zero) {
    return __int_as_float(0x4B000000u | c) - (8388608.0f + zero);
}

// Decode 16 columns to 16 int8. BITS==4/5/6: pk is the uint2 of 8 nibble-plane bytes (low
// nibble = even col). BITS==4: cb is the 16-entry level table or nullptr for the uniform (q-8)
// levels, and sc0..sc3 are the up-to-4 distinct group scales the 16 cols can span (equal for
// G>=16). BITS==6: hi is the uint32 of the 2-bit plane (col i's top 2 bits at bit 2*i), levels
// uniform (c-32). BITS==5: hi is the uint16 of the 1-bit plane (col i's top bit at bit i),
// levels uniform (c-16). BITS==3: pk.x is the uint32 of the 2-bit plane (col i's low 2 bits at
// bit 2*i), hi the uint16 of the 1-bit plane, and cb the 8-entry table. Multi-plane widths
// have one scale per vector (their groups are multiples of 16).
template <int BITS>
__device__ __forceinline__ void dequant16_to_int8(
    uint2 pk, unsigned hi, const float* __restrict__ cb,
    float sc0, float sc1, float sc2, float sc3, int G, char4 out4[4])
{
    const unsigned words[2] = {pk.x, pk.y};
    int8_t* o = reinterpret_cast<int8_t*>(out4);
    if constexpr (BITS == 3) {
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const unsigned c = ((pk.x >> (2 * i)) & 3u) | (((hi >> i) & 1u) << 2);
            o[i] = level_to_int8(cb[c], sc0);
        }
        return;
    } else {
    #pragma unroll
    for (int w = 0; w < 2; ++w) {
        #pragma unroll
        for (int bi = 0; bi < 4; ++bi) {
            const int oo = w * 4 + bi;             // 0..7 -> cols oo*2, oo*2+1
            const unsigned byte = (words[w] >> (bi * 8)) & 0xFF;
            float v0, v1, s;
            if constexpr (BITS == 6) {
                const unsigned c0 = (byte & 0xF) | (((hi >> (4 * oo)) & 3u) << 4);
                const unsigned c1 = ((byte >> 4) & 0xF) | (((hi >> (4 * oo + 2)) & 3u) << 4);
                v0 = code_level(c0, 32.0f);
                v1 = code_level(c1, 32.0f);
                s = sc0;
            } else if constexpr (BITS == 5) {
                const unsigned c0 = (byte & 0xF) | (((hi >> (2 * oo)) & 1u) << 4);
                const unsigned c1 = ((byte >> 4) & 0xF) | (((hi >> (2 * oo + 1)) & 1u) << 4);
                v0 = code_level(c0, 16.0f);
                v1 = code_level(c1, 16.0f);
                s = sc0;
            } else {
                const int lg = (G >= 16) ? 0 : ((oo * 2) / G);  // local group in the vec
                s = (lg == 0) ? sc0 : (lg == 1 ? sc1 : (lg == 2 ? sc2 : sc3));
                const unsigned c0 = byte & 0xF, c1 = (byte >> 4) & 0xF;
                v0 = cb ? cb[c0] : code_level(c0, 8.0f);
                v1 = cb ? cb[c1] : code_level(c1, 8.0f);
            }
            o[2 * oo]     = level_to_int8(v0, s);
            o[2 * oo + 1] = level_to_int8(v1, s);
        }
    }
    }
}

// Each thread: 8 packed bytes (uint2) -> 16 int8 (uint4 store). The 16 output
// cols may span multiple groups when G<16 (finer groups = better int4 quality),
// so the scale is (re)loaded per output pair from its own group. Only 4 group
// scales (sc0..sc3) are loaded, so a 16-col vec may span at most 4 groups: G must
// be in {4, 8, 16} or a multiple of 16 (G<4 would span >4 groups and mis-scale).
// If codebook != nullptr, the 4-bit code indexes a shared 16-entry non-uniform
// codebook (Lloyd-Max on the rotated-Gaussian weight) instead of the uniform
// level (q-8); same storage/speed, ~14% lower weight error at coarse groups.
// The 16-column vector v of a packed row: the uint2 of its 8 nibble-plane bytes (4/5/6-bit) or
// the 2-bit plane word (3-bit), and in hi the high plane's bits for those columns.
template <int BITS>
__device__ __forceinline__ uint2 load_vec_planes(const int8_t* __restrict__ wrow, int v, int K, unsigned& hi) {
    hi = 0u;
    if constexpr (BITS == 6) hi = *reinterpret_cast<const unsigned*>(wrow + K / 2 + v * 4);
    else if constexpr (BITS == 5) hi = *reinterpret_cast<const unsigned short*>(wrow + K / 2 + v * 2);
    else if constexpr (BITS == 3) hi = *reinterpret_cast<const unsigned short*>(wrow + K / 4 + v * 2);
    if constexpr (BITS == 3) return make_uint2(*reinterpret_cast<const unsigned*>(wrow + v * 4), 0u);
    else return *reinterpret_cast<const uint2*>(wrow + v * 8);
}

template <typename ScaleT, int BITS>
__global__ void dequant_int4_grouped_to_int8_kernel(
    const int8_t* __restrict__ qw,   // (N, K*BITS/8) packed codes
    const ScaleT* __restrict__ s_rel,// (N, K/G) fp32 or e4m3 raw
    const float*  __restrict__ codebook, // 16 floats (4-bit, or nullptr), 8 floats (3-bit)
    int8_t*       __restrict__ out,  // (N, K)
    long n_vec, int K, int G)
{
    __shared__ float cb[16];
    if constexpr (BITS == 4 || BITS == 3) {
        if (codebook && threadIdx.x < (BITS == 4 ? 16 : 8)) cb[threadIdx.x] = codebook[threadIdx.x];
        if (codebook) __syncthreads();
    }
    long v = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n_vec) return;                       // n_vec = N*K/16: one 16-col vector per thread
    const int vec_per_row = K / 16;
    const int n = v / vec_per_row;
    const int hv = v % vec_per_row;               // which 16-col vector in the row
    const int k0 = hv * 16;                       // output col base (16 wide)
    const int nG = K / G;
    const long srow = (long)n * nG;
    const long row_bytes = (long)K * BITS / 8;
    const int8_t* __restrict__ wrow = qw + (long)n * row_bytes;
    unsigned hi;
    const uint2 pk = load_vec_planes<BITS>(wrow, hv, K, hi);
    // The 16-col vec spans 1 group (G>=16, the common case), 2 (G=8), or 4 (G=4).
    // Load+decode each distinct group scale ONCE instead of per output pair.
    const int base_g = k0 / G;
    float sc0 = load_scale<ScaleT>(s_rel[srow + base_g]);
    float sc1 = sc0, sc2 = sc0, sc3 = sc0;
    if constexpr (BITS == 4) {
        if (G < 16) {
            sc1 = load_scale<ScaleT>(s_rel[srow + base_g + 1]);
            if (G < 8) {  // G == 4
                sc2 = load_scale<ScaleT>(s_rel[srow + base_g + 2]);
                sc3 = load_scale<ScaleT>(s_rel[srow + base_g + 3]);
            }
        }
    }
    char4 o4[4];
    dequant16_to_int8<BITS>(pk, hi, codebook ? cb : nullptr, sc0, sc1, sc2, sc3, G, o4);
    *reinterpret_cast<uint4*>(&out[(long)n * K + k0]) = *reinterpret_cast<uint4*>(o4);
}

// Decode one 16-code chunk (one scale byte) with the LUT row for that scale held in
// shared memory: two byte permutes pick codes 0-7 / 8-15, code bit 3 selects between
// them.
__device__ __forceinline__ unsigned decode_w4a8_chunk(
    unsigned codes, unsigned scale_bits, const uint4* __restrict__ lut)
{
    const uint4 table = lut[scale_bits];
    // selector nibbles use bits 0-2 only; bit 3 of each code then picks lo/hi per byte
    const unsigned lo = __byte_perm(table.x, table.y, codes);
    const unsigned hi = __byte_perm(table.z, table.w, codes);
    return __byte_perm(lo, hi, ((codes >> 1) & 0x4444u) | 0x3210u);
}

__device__ __forceinline__ void decode_w4a8_record_smem(
    const uint2& packed, unsigned scales,
    const uint4* __restrict__ lut, unsigned (&decoded)[4])
{
    decoded[0] = decode_w4a8_chunk(packed.x, scales & 0xffu, lut);
    decoded[1] = decode_w4a8_chunk(packed.x >> 16, (scales >> 8) & 0xffu, lut);
    decoded[2] = decode_w4a8_chunk(packed.y, (scales >> 16) & 0xffu, lut);
    decoded[3] = decode_w4a8_chunk(packed.y >> 16, scales >> 24, lut);
}

// A lane's 16 codes of a 3/5/6-bit record: one little-endian string of 16*BITS bits, code q at
// bit q*BITS, stored as plane A (32 lanes x 4 or 8 bytes) then plane B (32 lanes x 2 or 4),
// which is how pack_w4a8_slabs_kernel writes them. Chunk j (codes 4j..4j+3) starts at bit
// 4*BITS*j; chunk() leaves whatever follows in the high bits, the decoders mask per code.
template <int BITS> struct LaneCodes {
    static_assert(BITS == 3 || BITS == 5 || BITS == 6, "4-bit records are plain nibble pairs");
    static constexpr int kABytes = BITS == 3 ? 4 : 8, kBBytes = BITS == 6 ? 4 : 2;
    static constexpr int kBOffset = 32 * kABytes;
    unsigned w[3];
    __device__ __forceinline__ void load(const uint8_t* __restrict__ record, int lane) {
        if constexpr (BITS == 3) {
            w[0] = *reinterpret_cast<const unsigned*>(record + lane * 4);
            w[1] = *reinterpret_cast<const unsigned short*>(record + kBOffset + lane * 2);
            w[2] = 0u;
        } else {
            const uint2 a = *reinterpret_cast<const uint2*>(record + lane * 8);
            w[0] = a.x;
            w[1] = a.y;
            if constexpr (BITS == 6) w[2] = *reinterpret_cast<const unsigned*>(record + kBOffset + lane * 4);
            else w[2] = *reinterpret_cast<const unsigned short*>(record + kBOffset + lane * 2);
        }
    }
    template <int J> __device__ __forceinline__ unsigned chunk() const {
        constexpr int bit = 4 * BITS * J, i = bit / 32, sh = bit % 32;
        if constexpr (sh == 0) return w[i];
        else if constexpr (sh + 4 * BITS <= 32) return w[i] >> sh;
        else return __funnelshift_r(w[i], w[i + 1], sh);
    }
    __device__ __forceinline__ unsigned chunk(int j) const {
        switch (j) { case 0: return chunk<0>(); case 1: return chunk<1>(); case 2: return chunk<2>(); default: return chunk<3>(); }
    }
};

// 5/6-bit: four codes per 4*BITS-bit chunk, uniform levels (c - 2^(BITS-1)) on the int8 grid
// of level_to_int8. s is e4m3 (256-entry fp8 -> fp32 table in shared memory), so the
// product has <= 10 significant bits and the single-rounding fma onto the magic grid
// equals the rounded product of level_to_int8. The clamped patterns are packed straight
// from their low bytes.
template <int BITS>
__device__ __forceinline__ unsigned decode_uniform_chunk(unsigned codes, float s)
{
    unsigned v[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float level = code_level((codes >> (BITS * i)) & ((1u << BITS) - 1), static_cast<float>(1 << (BITS - 1)));
        v[i] = __float_as_uint(grid_clamp(fmaf(level, s, kRoundMagic)));
    }
    return __byte_perm(__byte_perm(v[0], v[1], 0x0040u), __byte_perm(v[2], v[3], 0x0040u), 0x5410u);
}

// 3-bit: four codes per 12-bit chunk; the LUT row for the scale holds the 8 int8 levels,
// so one byte permute with the codes widened to selector nibbles decodes the chunk.
__device__ __forceinline__ unsigned decode_w3a8_chunk(
    unsigned codes, unsigned scale_bits, const uint2* __restrict__ lut)
{
    const uint2 table = lut[scale_bits];
    const unsigned sel = (codes & 7u) | ((codes >> 3) & 7u) << 4 | ((codes >> 6) & 7u) << 8 | ((codes >> 9) & 7u) << 12;
    return __byte_perm(table.x, table.y, sel);
}

// One chunk through the stream kernel's shared-memory table: the [256][8] int8 LUT at 3 bits,
// the 256 fp32 scales at 5/6.
template <int BITS>
__device__ __forceinline__ unsigned decode_chunk(unsigned codes, unsigned scale_bits, const void* __restrict__ table)
{
    if constexpr (BITS == 3) return decode_w3a8_chunk(codes, scale_bits, static_cast<const uint2*>(table));
    else return decode_uniform_chunk<BITS>(codes, static_cast<const float*>(table)[scale_bits]);
}

template <int BITS>
__device__ __forceinline__ void decode_record_smem(
    const LaneCodes<BITS>& codes, unsigned scales, const void* __restrict__ table, unsigned (&decoded)[4])
{
    decoded[0] = decode_chunk<BITS>(codes.template chunk<0>(), scales & 0xffu, table);
    decoded[1] = decode_chunk<BITS>(codes.template chunk<1>(), (scales >> 8) & 0xffu, table);
    decoded[2] = decode_chunk<BITS>(codes.template chunk<2>(), (scales >> 16) & 0xffu, table);
    decoded[3] = decode_chunk<BITS>(codes.template chunk<3>(), scales >> 24, table);
}

__device__ __forceinline__ void mma_m16n8k32_s8(
    int (&acc)[4], const unsigned (&a)[4], const unsigned (&b)[2])
{
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
#endif
}

// cp.async is sm_80+, like the mma above; the host launcher rejects older devices
// so the pre-sm_80 passes only need to compile.
__device__ __forceinline__ void cp_async16(uint32_t smem, const void* gptr) {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem), "l"(gptr));
#endif
}
__device__ __forceinline__ void cp_async_commit() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n");
#endif
}
template <int N> __device__ __forceinline__ void cp_async_wait() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}

// Streaming decode GEMM (M <= 8). A warp owns one 16-output tile over
// `rows` consecutive K records and streams them through a Stages-deep per-warp
// cp.async ring in shared memory: one record is decoded + MMA'd while Stages-1 are
// in flight, for the warp's whole life. Register rings can't do this: ptxas gives
// every load in the loop one scoreboard, so waiting on one record waits on all.
//
// Weight layout: kRecordBytes records (512 codes at BITS
// bits in mma m16n8k32 fragment order + 32 scale bytes: 224 B at 3 bits, 288 at 4,
// 352 at 5, 416 at 6) for a 16-output x 32-K tile, stored [tile][k_row][record]. Each warp walks
// contiguous records within its tile and K split; no global record permutation.
//
// The decode table (4/3-bit: the [256 scales][16 or 8 codes] int8 LUT; 6/5-bit: the 256
// fp8 scales as fp32) and the block's x slice are staged in shared memory after the
// first loads are issued. The split-K reduction finishes in-kernel: each warp adds its tile into the
// int32 workspace, bumps the tile's counter, and the warp that arrives last reads
// the tile back from L2, applies the scales/bias, writes the output, and returns
// the workspace tile and counter to zero. So the workspace and counters are zero
// on entry and on exit, and the caller keeps both across launches.
// residual (nullable): out = residual + residual_scale * out, with the linear rounded
// to OutputT first, as the unfused addcmul sees it (int8_linear.cu's contract).
constexpr int kStreamStages = 4;
constexpr int kMmaWarps = 8;

template <int BITS> struct StreamRecord {
    static constexpr int kBytes = 512 * BITS / 8 + 32;
    static constexpr int kTableBytes = BITS == 4 ? 4096 : BITS == 3 ? 2048 : 256 * static_cast<int>(sizeof(float));
};

template <int BITS>
__host__ __device__ constexpr int w4a8_stream_smem_bytes(int rows) {
    return StreamRecord<BITS>::kTableBytes + 8 * (rows * 32 + 16)
        + kMmaWarps * kStreamStages * StreamRecord<BITS>::kBytes;
}


template <typename OutputT, int BITS>
__global__ __launch_bounds__(kMmaWarps * 32)
void w4a8_codebook_mma_stream_kernel(
    const int8_t* __restrict__ x,
    const int8_t* __restrict__ weight,
    const int8_t* __restrict__ decode_lut,
    const float* __restrict__ s_channel,
    const float* __restrict__ x_scales,
    const float* __restrict__ bias,
    const OutputT* __restrict__ residual,
    const OutputT* __restrict__ residual_scale,
    int* __restrict__ workspace,
    int* __restrict__ counters,
    OutputT* __restrict__ output,
    int M, int N, int K, int rows)
{
    constexpr int S = kStreamStages;
    constexpr int R = StreamRecord<BITS>::kBytes;
    extern __shared__ __align__(16) uint8_t stream_smem[];
    uint4* lut = reinterpret_cast<uint4*>(stream_smem);
    float* scale_table = reinterpret_cast<float*>(stream_smem);
    uint8_t* xs_s = stream_smem + StreamRecord<BITS>::kTableBytes;
    const int x_stride = rows * 32 + 16;   // +16 keeps the per-token rows off one bank pattern
    uint8_t* stages = xs_s + 8 * x_stride;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int tiles = N / 16;
    const int output_tile = static_cast<int>(blockIdx.x) * kMmaWarps + warp;
    const bool active = output_tile < tiles;
    const int split = static_cast<int>(blockIdx.y);
    const int splits = static_cast<int>(gridDim.y);
    const int k_begin = split * rows * 32;

    // lanes 0..R/16-1 each move 16 B of the record
    const int8_t* tile_base = weight
        + (static_cast<int64_t>(output_tile) * (K / 32) + split * rows) * R + lane * 16;
    auto record = [&](int i) -> const int8_t* {
        return tile_base + i * R;
    };
    const bool issuer = active && lane < R / 16;
    const uint8_t* my_stages = stages + warp * S * R;
    const uint32_t st_s = static_cast<uint32_t>(__cvta_generic_to_shared(my_stages)) + lane * 16;

    #pragma unroll
    for (int s = 0; s < S; ++s) {
        if (issuer) cp_async16(st_s + s * R, record(s));
        cp_async_commit();
    }
    for (int i = threadIdx.x; i < 256; i += kMmaWarps * 32) {
        if constexpr (BITS == 4) {
            lut[i] = reinterpret_cast<const uint4*>(decode_lut)[i];
        } else if constexpr (BITS == 3) {
            if (i < 128) lut[i] = reinterpret_cast<const uint4*>(decode_lut)[i];
        } else {
            scale_table[i] = load_scale<uint8_t>(static_cast<uint8_t>(i));
        }
    }
    for (int i = threadIdx.x; i < 8 * rows * 2; i += kMmaWarps * 32) {
        const int token = i / (rows * 2), v = i % (rows * 2);
        uint4 value = make_uint4(0u, 0u, 0u, 0u);
        if (token < M) {
            value = reinterpret_cast<const uint4*>(x + static_cast<int64_t>(token) * K + k_begin)[v];
        }
        *reinterpret_cast<uint4*>(xs_s + token * x_stride + v * 16) = value;
    }
    __syncthreads();
    if (!active) return;

    const int token = lane >> 2;
    const int thread_in_group = lane & 3;
    const uint8_t* xrow = xs_s + token * x_stride + thread_in_group * 4;
    const uint8_t* rec_codes = my_stages + lane * 8;   // 4-bit: the lane's 8 nibble bytes
    const uint8_t* rec_scales = my_stages + (R - 32) + token * 4;
    int acc[4] = {};

    auto consume = [&](int stage, int i) {
        const unsigned scales = *reinterpret_cast<const unsigned*>(rec_scales + stage * R);
        unsigned decoded[4];
        if constexpr (BITS == 4) {
            decode_w4a8_record_smem(*reinterpret_cast<const uint2*>(rec_codes + stage * R), scales, lut, decoded);
        } else {
            LaneCodes<BITS> codes;
            codes.load(my_stages + stage * R, lane);
            decode_record_smem<BITS>(codes, scales, stream_smem, decoded);
        }
        const unsigned input[2] = {
            *reinterpret_cast<const unsigned*>(xrow + i * 32),
            *reinterpret_cast<const unsigned*>(xrow + i * 32 + 16),
        };
        mma_m16n8k32_s8(acc, decoded, input);
    };

    // Record i is consumed once at most S-1 newer groups are pending, then its stage
    // is refilled with record i+S. Unrolled by S so stage indices are compile-time.
    int i = 0;
    for (; i + S <= rows - S; i += S) {
        #pragma unroll
        for (int j = 0; j < S; ++j) {
            cp_async_wait<S - 1>();
            __syncwarp();
            consume(j, i + j);
            __syncwarp();
            if (issuer) cp_async16(st_s + j * R, record(i + S + j));
            cp_async_commit();
        }
    }
    for (; i < rows; ++i) {
        cp_async_wait<S - 1>();
        __syncwarp();
        consume(i % S, i);
        __syncwarp();
        if (issuer && i + S < rows) cp_async16(st_s + (i % S) * R, record(i + S));
        cp_async_commit();
    }

    const int token0 = thread_in_group * 2;
    const int token1 = token0 + 1;
    const int n_top = output_tile * 16 + token;
    const int n_bottom = n_top + 8;
    int* ws00 = &workspace[static_cast<int64_t>(token0) * N + n_top];
    int* ws01 = &workspace[static_cast<int64_t>(token0) * N + n_bottom];
    int* ws10 = &workspace[static_cast<int64_t>(token1) * N + n_top];
    int* ws11 = &workspace[static_cast<int64_t>(token1) * N + n_bottom];
    // Residual operands are issued here so they are in flight behind the atomics and
    // fences below instead of on the finishing warp's dependent tail.
    float r00 = 0.f, r01 = 0.f, r10 = 0.f, r11 = 0.f, rs_top = 0.f, rs_bottom = 0.f;
    if (residual) {
        rs_top = static_cast<float>(residual_scale[n_top]);
        rs_bottom = static_cast<float>(residual_scale[n_bottom]);
        if (token0 < M) {
            r00 = static_cast<float>(residual[static_cast<int64_t>(token0) * N + n_top]);
            r01 = static_cast<float>(residual[static_cast<int64_t>(token0) * N + n_bottom]);
        }
        if (token1 < M) {
            r10 = static_cast<float>(residual[static_cast<int64_t>(token1) * N + n_top]);
            r11 = static_cast<float>(residual[static_cast<int64_t>(token1) * N + n_bottom]);
        }
    }
    if (token0 < M) {
        atomicAdd(ws00, acc[0]);
        atomicAdd(ws01, acc[2]);
    }
    if (token1 < M) {
        atomicAdd(ws10, acc[1]);
        atomicAdd(ws11, acc[3]);
    }

    // Publish this warp's adds, then arrive on the tile counter. The lane that sees
    // splits-1 prior arrivals knows every split's adds are visible at L2.
    __threadfence();
    int prior = 0;
    if (lane == 0) prior = atomicAdd(&counters[output_tile], 1);
    prior = __shfl_sync(0xffffffffu, prior, 0);
    if (prior != splits - 1) return;
    __threadfence();

    // Same lane->element mapping as the adds above, so the warp covers the whole
    // [M, 16] tile exactly once: read (L2), scale, store, and re-zero.
    auto finish = [&](int* ws, int tok, int n, float res, float res_scale) {
        float value = static_cast<float>(__ldcg(ws)) * x_scales[tok] * s_channel[n];
        if (bias) value += bias[n];
        if (residual) {
            const float linear = static_cast<float>(store_output<OutputT>(value));
            value = res + res_scale * linear;
        }
        output[static_cast<int64_t>(tok) * N + n] = store_output<OutputT>(value);
        *ws = 0;
    };
    if (token0 < M) {
        finish(ws00, token0, n_top, r00, rs_top);
        finish(ws01, token0, n_bottom, r01, rs_bottom);
    }
    if (token1 < M) {
        finish(ws10, token1, n_top, r10, rs_top);
        finish(ws11, token1, n_bottom, r11, rs_bottom);
    }
    if (lane == 0) counters[output_tile] = 0;
}


template <typename ScaleT>
void launch_dequant_grouped_to_int8(
    const void* qw, const void* s_rel, const void* codebook, void* out,
    int64_t N, int64_t K, int64_t G, int64_t bits, cudaStream_t stream)
{
    const long n_vec = (long)N * K / 16;
    const int block = 256;
    const long grid = (n_vec + block - 1) / block;
    dispatch_bits(bits, [&](auto tag) {
        constexpr int B = decltype(tag)::value;
        dequant_int4_grouped_to_int8_kernel<ScaleT, B><<<grid, block, 0, stream>>>(
            static_cast<const int8_t*>(qw), static_cast<const ScaleT*>(s_rel),
            (B == 4 || B == 3) ? static_cast<const float*>(codebook) : nullptr,
            static_cast<int8_t*>(out), n_vec, static_cast<int>(K), static_cast<int>(G));
    });
}
// Non-overlapping backwards batches pack directly from source to destination.
// Only the bottom slabs overlap: one CTA saves each of those before expanding it.
template <int BITS, bool OVERLAP>
__global__ void pack_w4a8_slabs_kernel(uint8_t* data, const uint8_t* scales,
                                     uint8_t* scratch, int first, int last, int K) {
    constexpr int code_bytes = 512 * BITS / 8, record_bytes = code_bytes + 32;
    constexpr int parts = OVERLAP ? 1 : 8;
    const int row_bytes = K * BITS / 8, slab_bytes = 16 * row_bytes, k_rows = K / 32;
    for (int tile = last - 1 - blockIdx.x / parts; tile >= first; tile -= gridDim.x / parts) {
        const uint8_t* slab = data + (int64_t)tile * slab_bytes;
        if constexpr (OVERLAP) {
            for (int i = threadIdx.x; i < slab_bytes / 4; i += blockDim.x)
                reinterpret_cast<unsigned*>(scratch)[i] = reinterpret_cast<const unsigned*>(slab)[i];
            __syncthreads();
            slab = scratch;
        }
        for (int i = threadIdx.x + (blockIdx.x % parts) * blockDim.x; i < k_rows * record_bytes; i += blockDim.x * parts) {
            const int kr = i / record_bytes, b = i % record_bytes;
            uint8_t value;
            if (b >= code_bytes) {
                const int s = b - code_bytes, row = (s / 4) + 8 * (s & 1), half = (s % 4) / 2;
                value = scales[((int64_t)tile * 16 + row) * (K / 16) + kr * 2 + half];
            } else if constexpr (BITS == 4) {
                const int lane = b / 8, j = (b % 8) / 2;
                const int row = lane / 4 + 8 * (j & 1);
                const int col = kr * 32 + (lane % 4) * 4 + (j / 2) * 16 + (b & 1) * 2;
                value = slab[row * row_bytes + col / 2];
            } else {
                // byte lb of the lane's LaneCodes<BITS> string: every code q whose bits q*BITS..
                // overlap it, read from the canonical planes
                using Lane = LaneCodes<BITS>;
                constexpr int a_bytes = Lane::kABytes, b_bytes = Lane::kBBytes, a_total = Lane::kBOffset;
                const int lane = b < a_total ? b / a_bytes : (b - a_total) / b_bytes;
                const int lb = b < a_total ? b % a_bytes : a_bytes + (b - a_total) % b_bytes;
                unsigned v = 0;
                #pragma unroll
                for (int q = 0; q < 16; ++q) {
                    const int shift = q * BITS - 8 * lb;
                    if (shift <= -BITS || shift >= 8) continue;
                    const int j = q >> 2, c = q & 3;
                    const int row = lane / 4 + 8 * (j & 1);
                    const int k = kr * 32 + (lane % 4) * 4 + (j / 2) * 16 + c;
                    const uint8_t* srow = slab + row * row_bytes;
                    unsigned code;
                    if constexpr (BITS == 3)
                        code = ((srow[k / 4] >> (2 * (k & 3))) & 3u) | (((srow[K / 4 + k / 8] >> (k & 7)) & 1u) << 2);
                    else if constexpr (BITS == 5)
                        code = ((srow[k / 2] >> (4 * (k & 1))) & 15u) | (((srow[K / 2 + k / 8] >> (k & 7)) & 1u) << 4);
                    else
                        code = ((srow[k / 2] >> (4 * (k & 1))) & 15u) | (((srow[K / 2 + k / 4] >> (2 * (k & 3))) & 3u) << 4);
                    v |= shift >= 0 ? (code << shift) : (code >> -shift);
                }
                value = static_cast<uint8_t>(v);
            }
            data[(int64_t)tile * k_rows * record_bytes + i] = value;
        }
        if constexpr (OVERLAP) __syncthreads();
    }
}

// Read record-stream weights directly into one canonical INT8 GEMM chunk.
// One thread writes a 16-code group; n0/cols may cut across a 16-output record.
template <int BITS>
__global__ void dequant_w4a8_mma_chunk_kernel(
    const uint8_t* __restrict__ packed, const float* __restrict__ codebook,
    int8_t* __restrict__ out, long n_groups, int n0, int K)
{
    const long v = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n_groups) return;
    constexpr int kRecordBytes = 512 * BITS / 8 + 32;
    const int n = n0 + v / (K / 16);
    const int kg = v % (K / 16), kr = kg / 2;
    const int h = (n & 15) >> 3, r = n & 7, kh = kg & 1;
    const long rec = (long)(n / 16) * (K / 32) + kr;
    const uint8_t* __restrict__ record = packed + rec * kRecordBytes;
    const float scale = load_scale<uint8_t>(record[512 * BITS / 8 + r * 4 + h + 2 * kh]);
    uint4 decoded;
    int8_t* values = reinterpret_cast<int8_t*>(&decoded);
    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        const int j = kh * 2 + h;
        const int lane = r * 4 + g;
        unsigned codes;
        if constexpr (BITS == 4) {
            codes = *reinterpret_cast<const unsigned short*>(record + lane * 8 + j * 2);
        } else {
            LaneCodes<BITS> lc;
            lc.load(record, lane);
            codes = lc.chunk(j);
        }
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            const unsigned code = (codes >> (BITS * c)) & ((1u << BITS) - 1);
            const float level = (BITS == 3 || (BITS == 4 && codebook)) ? codebook[code]
                                                                      : code_level(code, static_cast<float>(1 << (BITS - 1)));
            values[g * 4 + c] = level_to_int8(level, scale);
        }
    }
    reinterpret_cast<uint4*>(out)[v] = decoded;
}
}  // namespace

extern "C" void launch_pack_w4a8_mma_inplace(
    void* data, const void* scales, void* scratch,
    int64_t N, int64_t K, int bits, cudaStream_t stream) {
    auto* bytes = static_cast<uint8_t*>(data);
    auto* temp = static_cast<uint8_t*>(scratch);
    int last = N / 16;
    while (last > 2 * bits) {
        // first * expanded_slab_bytes >= last * canonical_slab_bytes:
        // the entire destination batch is above its entire source batch.
        const int first = (last * (2 * bits) + 2 * bits) / (2 * bits + 1);
        dispatch_bits(bits, [&](auto tag) {
            pack_w4a8_slabs_kernel<decltype(tag)::value, false><<<(last - first) * 8, 256, 0, stream>>>(
                bytes, static_cast<const uint8_t*>(scales), temp, first, last, K);
        });
        last = first;
    }
    dispatch_bits(bits, [&](auto tag) {
        pack_w4a8_slabs_kernel<decltype(tag)::value, true><<<1, 1024, 0, stream>>>(
            bytes, static_cast<const uint8_t*>(scales), temp, 0, last, K);
    });
}

static void launch_dequant_w4a8_mma_chunk(
    const void* packed, const void* codebook, void* out,
    int64_t n0, int64_t cols, int64_t K, int64_t bits, cudaStream_t stream)
{
    const long n_groups = (long)cols * K / 16;
    const int block = 256;
    const long grid = (n_groups + block - 1) / block;
    dispatch_bits(bits, [&](auto tag) {
        constexpr int B = decltype(tag)::value;
        dequant_w4a8_mma_chunk_kernel<B><<<grid, block, 0, stream>>>(
            static_cast<const uint8_t*>(packed), (B == 4 || B == 3) ? static_cast<const float*>(codebook) : nullptr,
            static_cast<int8_t*>(out), n_groups, n0, K);
    });
}

// codebook: 16 floats (non-uniform levels) or nullptr for uniform (q-8); ignored at 6 bits.
extern "C" void launch_dequant_int4_grouped_to_int8(
    const void* qw, const void* s_rel, const void* codebook, void* out,
    int64_t N, int64_t K, int64_t G, int64_t bits, cudaStream_t stream)
{
    launch_dequant_grouped_to_int8<float>(qw, s_rel, codebook, out, N, K, G, bits, stream);
}

// fp8 (e4m3) per-group scale variant; s_rel passed as raw uint8 bits.
extern "C" void launch_dequant_int4_grouped_to_int8_e4m3(
    const void* qw, const void* s_rel, const void* codebook, void* out,
    int64_t N, int64_t K, int64_t G, int64_t bits, cudaStream_t stream)
{
    launch_dequant_grouped_to_int8<uint8_t>(qw, s_rel, codebook, out, N, K, G, bits, stream);
}

// Fused-quality W4A8: dequant int4 -> int8 in column chunks (codebook + per-group
// s_rel) feeding the tuned STRIDED int8 GEMM, so each int8 weight chunk stays
// L2-resident instead of the full [N,K] round-tripping global (the convrot_w4a4
// chunking trick, run at our group-16 codebook quality). Returns false if the
// strided GEMM rejects a chunk config -> caller falls back to the 2-pass path.
// bias is read in the OUTPUT dtype (cutlass_gemm_int8.cu); never pass fp32 here.
extern "C" bool launch_cutlass_int8_dequant_strided(
    const void* A, const void* B, const void* xs, const void* ws, const void* bias,
    void* D, int64_t M, int64_t N, int64_t K, int64_t output_stride, int out_dtype_code,
    cudaStream_t stream);

extern "C" bool launch_w4a8_codebook_gemm_chunked(
    const void* xq,        // [M, K] int8 activation
    const void* weight,    // [N, K*bits/8] packed codes
    const void* s_rel,     // [N, K/G] fp8 (e4m3) per-group scale
    const void* codebook,  // [16] fp32 or nullptr
    const void* s_channel, // [N] fp32 per-channel scale
    const void* xs,        // [M] fp32 per-row activation scale
    const void* bias,      // [N] in out_dtype, or nullptr
    void* workspace,       // [chunk_cols, K] int8 scratch (preallocated, reused)
    void* out,             // [M, N] output (out_dtype)
    int64_t M, int64_t N, int64_t K, int64_t G, int64_t chunk_cols, int64_t bits,
    bool mma_packed, int out_dtype_code, cudaStream_t stream)
{
    // A non-positive chunk stride never advances n0 -> would loop forever; a non-positive
    // K/G would divide by zero below. Bail so the caller uses the 2-pass path.
    if (chunk_cols <= 0 || K <= 0 || G <= 0) return false;
    const int64_t row_bytes = K * bits / 8, KG = K / G, osz = (out_dtype_code == 0) ? 4 : 2;
    for (int64_t n0 = 0; n0 < N; n0 += chunk_cols) {
        const int64_t cols = (chunk_cols < N - n0) ? chunk_cols : (N - n0);
        if (mma_packed) {
            launch_dequant_w4a8_mma_chunk(weight, codebook, workspace, n0, cols, K, bits, stream);
        } else {
            launch_dequant_int4_grouped_to_int8_e4m3(
                static_cast<const int8_t*>(weight) + n0 * row_bytes,
                static_cast<const uint8_t*>(s_rel) + n0 * KG,
                codebook, workspace, cols, K, G, bits, stream);
        }
        // bias is in the output dtype (the strided GEMM's contract), so it
        // advances by the same element size as the output.
        const void* bias_chunk = bias ? static_cast<const char*>(bias) + n0 * osz : nullptr;
        void* out_chunk = static_cast<char*>(out) + n0 * osz;
        if (!launch_cutlass_int8_dequant_strided(
                xq, workspace, xs, static_cast<const float*>(s_channel) + n0, bias_chunk,
                out_chunk, M, cols, K, N /*output_stride*/, out_dtype_code, stream))
            return false;
    }
    return true;
}

// bits (3..6) is the code width. The workspace [M, N] int32 and counters [N/16] int32 must
// be zero on entry and are zero again on exit, so the caller keeps both across
// launches. residual [M, N] and residual_scale [N] are in the output dtype (both or
// neither).
extern "C" bool launch_w4a8_codebook_mma(
    const void* xq, const void* weight, const void* decode_lut,
    const void* s_channel, const void* xs, const void* bias, const void* residual,
    const void* residual_scale, void* workspace, void* counters, void* out,
    int64_t M, int64_t N, int64_t K, int64_t G, int64_t bits,
    int out_dtype_code, cudaStream_t stream)
{
    if (M == 0 || N == 0 || K == 0) return true;
    if (M > 8 || ((bits == 4 || bits == 3) && decode_lut == nullptr) || N > std::numeric_limits<int>::max()
            || (bits != 3 && bits != 4 && bits != 5 && bits != 6)
            || K > std::numeric_limits<int>::max() || N % 16 != 0
            || G != 16 || K % 256 != 0) {
        return false;
    }
    int device = 0;
    int compute_capability_major = 0;
    if (cudaGetDevice(&device) != cudaSuccess
            || cudaDeviceGetAttribute(
                &compute_capability_major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess
            || compute_capability_major < 8) {
        return false;
    }
    const int rows = K % 512 == 0 ? 16 : 8;
    const int splits = static_cast<int>(K) / 32 / rows;
    auto launch = [&]<int BITS>() {
        constexpr int OutputsPerBlock = kMmaWarps * 16;
        const dim3 grid(
            static_cast<unsigned int>((N + OutputsPerBlock - 1) / OutputsPerBlock),
            static_cast<unsigned int>(splits));
        DISPATCH_FP_DTYPE(out_dtype_code, OutputT, [&] {
            w4a8_codebook_mma_stream_kernel<OutputT, BITS>
                <<<grid, kMmaWarps * 32, w4a8_stream_smem_bytes<BITS>(rows), stream>>>(
                static_cast<const int8_t*>(xq),
                static_cast<const int8_t*>(weight),
                static_cast<const int8_t*>(decode_lut),
                static_cast<const float*>(s_channel),
                static_cast<const float*>(xs),
                static_cast<const float*>(bias),
                static_cast<const OutputT*>(residual),
                static_cast<const OutputT*>(residual_scale),
                static_cast<int*>(workspace),
                static_cast<int*>(counters),
                static_cast<OutputT*>(out),
                static_cast<int>(M), static_cast<int>(N), static_cast<int>(K), rows);
        });
    };
    dispatch_bits(bits, [&](auto tag) { launch.template operator()<decltype(tag)::value>(); });
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("W4A8 packed MMA failed: ") + cudaGetErrorString(error));
    }
    return true;
}


// W4A8 / W6A8 requantize in one launch: raw weight -> ConvRot -> packed codes + fp8 s_rel +
// f32 s_channel. Same math as the eager quantizer, up to fp32 rounding order.
namespace {

__device__ __forceinline__ float rq_to_float(float v) { return v; }
__device__ __forceinline__ float rq_to_float(__half v) { return __half2float(v); }
__device__ __forceinline__ float rq_to_float(__nv_bfloat16 v) { return __bfloat162float(v); }
template <typename T> __device__ __forceinline__ T rq_from_float(float v);
template <> __device__ __forceinline__ __half rq_from_float<__half>(float v) { return __float2half(v); }
template <> __device__ __forceinline__ __nv_bfloat16 rq_from_float<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

__device__ __forceinline__ uint32_t rq_pcg(uint32_t x) {
    x = x * 747796405u + 2891336453u;
    uint32_t w = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    return (w >> 22u) ^ w;
}
// uniform in [0,1) keyed by a global element index + seed
__device__ __forceinline__ float rq_uniform(int64_t idx, uint64_t seed) {
    uint32_t h = rq_pcg(static_cast<uint32_t>(idx) ^ static_cast<uint32_t>(seed)
                        ^ static_cast<uint32_t>(seed >> 32));
    return static_cast<float>(h >> 8) * (1.0f / 16777216.0f);
}
__device__ __forceinline__ float rq_warp_max(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, o));
    return v;
}
// Block-wide max, returned on every thread.
__device__ __forceinline__ float rq_block_max(float v, float* warp_max) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = rq_warp_max(v);
    if (lane == 0) warp_max[wid] = v;
    __syncthreads();
    if (wid == 0) {
        const int nwarps = (blockDim.x + 31) >> 5;
        float t = (lane < nwarps) ? warp_max[lane] : 0.0f;
        t = rq_warp_max(t);
        if (lane == 0) warp_max[0] = t;
    }
    __syncthreads();
    return warp_max[0];
}

// Entries of a sorted L-entry (16 or 8) table below x (<= x with LE); branch-free bisection.
template <bool LE, int L = 16, typename Table>
__device__ __forceinline__ int rq_count16(Table tab, float x) {
    auto lt = [&](int j) { return LE ? (tab(j) <= x) : (tab(j) < x); };
    int p = 0;
    #pragma unroll
    for (int step = L / 2; step >= 1; step >>= 1) p += lt(p + step - 1) ? step : 0;
    return p;
}
// Nearest entry, lowest index on ties; DISTINCT skips the duplicate-value search.
template <bool DISTINCT, int L = 16, typename Table>
__device__ __forceinline__ int rq_nearest16(Table tab, float x) {
    const int pos = rq_count16<false, L>(tab, x);
    const int h = min(pos, L - 1), l = max(pos - 1, 0);
    const float th = tab(h), tl = tab(l);
    if (fabsf(x - th) < fabsf(x - tl)) return h;
    if (DISTINCT) return l;
    return rq_count16<false, L>(tab, tl);   // first index holding the value tl
}

// Rotated row in smem, in the weight dtype, 2 elements of padding per group (bank-conflict free).
template <int LOG2G>
__device__ __forceinline__ int rq_padded_index(int i) { return i + 2 * (i >> LOG2G); }
template <int G> struct RqLog2 { static constexpr int value = G == 16 ? 4 : G == 32 ? 5 : 6; };

// One W4A8 (L=16) or W3A8 (L=8) row (group 16, codebook, fp8 s_rel) from the padded smem row.
template <typename InputType, int L = 16>
__device__ __forceinline__ void w4_quantize_row(
    const InputType* row_buf, int K, int64_t elem_base, const float* cb,   // cb: smem [L], sorted
    int8_t* __restrict__ packed_row,   // [K/2] nibbles (L=16) or [3K/8] 2-bit plane + 1-bit plane (L=8)
    uint8_t* __restrict__ s_rel_row,   // [K/16] e4m3 bits
    float* __restrict__ s_channel_out, // [1]
    float* gscale,                     // smem [K/16]
    float* warp_max,                   // smem [32]
    bool stochastic, uint64_t seed)
{
    constexpr int G = 16;
    auto src = [&](int i) { return rq_to_float(row_buf[rq_padded_index<4>(i)]); };
    auto cbt = [&](int j) { return cb[j]; };
    const int groups = K / G;

    // --- Phase 1: per-group ALS group scale; accumulate row shifted-amax ---
    float thr_amax = 0.0f;
    for (int g = threadIdx.x; g < groups; g += blockDim.x) {
        const int base = g * G;
        float w[G];
        float amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < G; ++i) { w[i] = src(base + i); amax = fmaxf(amax, fabsf(w[i])); }
        float gs = fmaxf(amax, 1e-8f);
        float inv = 1.0f / gs;
        int idx[G];
        #pragma unroll
        for (int i = 0; i < G; ++i) idx[i] = rq_nearest16<true, L>(cbt, w[i] * inv);
        #pragma unroll
        for (int it = 0; it < 2; ++it) {       // matches eager _ALS_ITERS
            float num = 0.0f, den = 0.0f;
            #pragma unroll
            for (int i = 0; i < G; ++i) { const float c = cb[idx[i]]; num += w[i] * c; den += c * c; }
            gs = fmaxf(num / fmaxf(den, 1e-8f), 1e-8f);
            inv = 1.0f / gs;
            #pragma unroll
            for (int i = 0; i < G; ++i) idx[i] = rq_nearest16<true, L>(cbt, w[i] * inv);
        }
        gscale[g] = gs;
        #pragma unroll
        for (int i = 0; i < G; ++i) thr_amax = fmaxf(thr_amax, fabsf(cb[idx[i]] * gs));
    }
    const float sc = fmaxf(rq_block_max(thr_amax, warp_max) / 127.0f, 1e-8f);
    const float inv_sc = 1.0f / sc;
    if (threadIdx.x == 0) *s_channel_out = sc;

    // --- Phase 2: s_rel (fp8) -> int8 levels -> assign (+SR) -> pack ---
    auto phase2 = [&](auto sr_tag) {
        constexpr bool SR = decltype(sr_tag)::value;
        for (int g = threadIdx.x; g < groups; g += blockDim.x) {
            const uint8_t srel_bits = __nv_cvt_float_to_fp8(gscale[g] / sc, __NV_SATFINITE, __NV_E4M3);
            s_rel_row[g] = srel_bits;
            const float srel_r = __half2float(__nv_cvt_fp8_to_halfraw(srel_bits, __NV_E4M3));
            // int8 level j, computed on demand (a register table would spill to local memory)
            auto lv = [&](int j) { return fminf(127.0f, fmaxf(-127.0f, nearbyintf(cb[j] * srel_r))); };
            const int base = g * G;
            uint32_t words[2] = {0u, 0u};
            #pragma unroll
            for (int i = 0; i < G; ++i) {
                const float t = src(base + i) * inv_sc;
                int a;
                if constexpr (SR) {
                    const int lo = min(max(rq_count16<true, L>(lv, t) - 1, 0), L - 2);   // levels <= t, minus one
                    const float l0 = lv(lo);
                    const float thr = l0 + rq_uniform(elem_base + base + i, seed) * (lv(lo + 1) - l0);
                    a = min(lo + (t > thr ? 1 : 0), L - 1);
                } else {
                    a = rq_nearest16<false, L>(lv, t);
                }
                if constexpr (L == 16) {
                    words[i >> 3] |= static_cast<uint32_t>(a & 0xF) << (4 * (i & 7));
                } else {
                    words[0] |= static_cast<uint32_t>(a & 3) << (2 * i);        // 2-bit plane word
                    words[1] |= static_cast<uint32_t>((a >> 2) & 1) << i;       // 1-bit plane half-word
                }
            }
            if constexpr (L == 16) {
                uint32_t* out = reinterpret_cast<uint32_t*>(packed_row + g * (G / 2));
                out[0] = words[0]; out[1] = words[1];
            } else {
                *reinterpret_cast<uint32_t*>(packed_row + g * 4) = words[0];
                *reinterpret_cast<uint16_t*>(packed_row + K / 4 + g * 2) = static_cast<uint16_t>(words[1]);
            }
        }
    };
    if (stochastic) phase2(std::true_type{}); else phase2(std::false_type{});
}

// int8 value of uniform level index i (0..2*LMAX) at group step s: eager's _grid_levels, on
// the fly. LMAX is 31 (6-bit) or 15 (5-bit).
template <int LMAX>
__device__ __forceinline__ float w6_level(int i, float s) {
    return fminf(127.0f, fmaxf(-127.0f, nearbyintf(static_cast<float>(i - LMAX) * s)));
}
// First level index >= t (searchsorted): closed-form estimate, fixup loops make it exact.
template <int LMAX>
__device__ __forceinline__ int w6_lower_bound(float t, float s, float inv_s) {
    constexpr int top = 2 * LMAX + 1;
    int i = __float2int_rn((ceilf(t) - 0.5f) * inv_s) + LMAX;
    i = min(max(i, 0), top);
    while (i > 0 && w6_level<LMAX>(i - 1, s) >= t) --i;
    while (i < top && w6_level<LMAX>(i, s) < t) ++i;
    return i;
}
// Eager _assign_grid: nearer bracket around t (lower on ties), or an SR draw in the bracket.
template <bool SR, int LMAX>
__device__ __forceinline__ int w6_assign(float t, float s, float inv_s, int64_t idx, uint64_t seed) {
    constexpr int last = 2 * LMAX;
    const int pos = w6_lower_bound<LMAX>(t, s, inv_s);
    if constexpr (SR) {
        const int l = min(max(pos - 1, 0), last - 1);
        const float level_lo = w6_level<LMAX>(l, s);
        const float thr = level_lo + rq_uniform(idx, seed) * (w6_level<LMAX>(l + 1, s) - level_lo);
        return min(l + (t > thr ? 1 : 0), last);
    }
    const int l = min(max(pos - 1, 0), last), h = min(pos, last);
    return (fabsf(t - w6_level<LMAX>(h, s)) < fabsf(t - w6_level<LMAX>(l, s))) ? h : l;
}

// One W6A8 (LMAX=31) or W5A8 (LMAX=15) row (uniform levels, fp8 s_rel, no scale search)
// from the padded smem row.
template <typename InputType, int G, int LMAX = 31>
__device__ __forceinline__ void w6_quantize_row(
    const InputType* row_buf, int K, int64_t elem_base,
    int8_t* __restrict__ packed_row,   // [3K/4]: nibble plane then 2-bit plane; [5K/8]: nibble plane then 1-bit plane
    uint8_t* __restrict__ s_rel_row,   // [K/G] e4m3 bits
    float* __restrict__ s_channel_out, // [1]
    float* gscale,                     // smem [K/G]
    float* warp_max,                   // smem [32]
    bool stochastic, uint64_t seed)
{
    auto src = [&](int i) { return rq_to_float(row_buf[rq_padded_index<RqLog2<G>::value>(i)]); };
    const int groups = K / G;

    // --- Phase 1: per-group step (2 ALS rounds); row amax of the decoded weight ---
    float thr_amax = 0.0f;
    for (int g = threadIdx.x; g < groups; g += blockDim.x) {
        const int base = g * G;
        float w[G];
        float amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < G; ++i) { w[i] = src(base + i); amax = fmaxf(amax, fabsf(w[i])); }
        constexpr float lmax = static_cast<float>(LMAX);
        float gs, inv = 0.0f;
        if constexpr (LMAX == 31) {
            gs = fmaxf(amax / lmax, 1e-8f);   // the shipped 6-bit form: reciprocal multiply
            inv = 1.0f / gs;
        } else {
            // Eager's amax / lmax divides by a Python scalar, which torch turns into a product
            // with the fp32-rounded reciprocal; 1/15 is 5e-8 off (1/31 is not) and decides
            // which way exact k + 0.5 quotients (w = amax / 2 is common with bf16 weights)
            // round. The element quotients are tensor/tensor there, so IEEE division here
            // (the build is fast-math) keeps those ties resolved the same way; reciprocal
            // multiply measured 3.5% channel-scale misses at 5 bits (+20% kernel time is
            // the price, 0.23 vs 0.19 ms on 10240x5120).
            constexpr float inv_lmax = 1.0f / lmax;
            gs = fmaxf(amax * inv_lmax, 1e-8f);
        }
        auto quo = [&](float v) -> float {
            if constexpr (LMAX == 31) return v * inv;
            else return __fdiv_rn(v, gs);
        };
        #pragma unroll
        for (int it = 0; it < 2; ++it) {       // eager _ALS_ITERS
            float num = 0.0f, den = 0.0f;
            #pragma unroll
            for (int i = 0; i < G; ++i) {
                const float q = fminf(lmax, fmaxf(-lmax, nearbyintf(quo(w[i]))));
                num += w[i] * q; den += q * q;
            }
            if constexpr (LMAX == 31) {
                gs = fmaxf(num / fmaxf(den, 1e-8f), 1e-8f);
                inv = 1.0f / gs;
            } else {
                gs = fmaxf(__fdiv_rn(num, fmaxf(den, 1e-8f)), 1e-8f);
            }
        }
        gscale[g] = gs;
        #pragma unroll
        for (int i = 0; i < G; ++i)
            thr_amax = fmaxf(thr_amax, fabsf(fminf(lmax, fmaxf(-lmax, nearbyintf(quo(w[i])))) * gs));
    }
    const float sc = fmaxf(rq_block_max(thr_amax, warp_max) / 127.0f, 1e-8f);
    const float inv_sc = 1.0f / sc;
    if (threadIdx.x == 0) *s_channel_out = sc;

    // --- Phase 2: fp8 s_rel -> grid assignment -> code q+32 -> two-plane pack ---
    int8_t* hi_plane = packed_row + K / 2;
    auto phase2 = [&](auto sr_tag) {
        constexpr bool SR = decltype(sr_tag)::value;
        for (int g = threadIdx.x; g < groups; g += blockDim.x) {
            const uint8_t srel_bits = __nv_cvt_float_to_fp8(gscale[g] / sc, __NV_SATFINITE, __NV_E4M3);
            s_rel_row[g] = srel_bits;
            const float srel_r = __half2float(__nv_cvt_fp8_to_halfraw(srel_bits, __NV_E4M3));
            const float inv_srel = 1.0f / fmaxf(srel_r, 1e-30f);
            const int base = g * G;
            int8_t* lo = packed_row + g * (G / 2);
            int8_t* hi = hi_plane + g * (LMAX == 31 ? G / 4 : G / 8);
            [[maybe_unused]] uint8_t hib[G / 8] = {};   // 5-bit: one top bit per code, 8 codes per byte
            #pragma unroll
            for (int p = 0; p < G / 4; ++p) {
                const int c = 4 * p;
                int u[4];
                #pragma unroll
                for (int k = 0; k < 4; ++k)
                    u[k] = w6_assign<SR, LMAX>(src(base + c + k) * inv_sc, srel_r, inv_srel, elem_base + base + c + k, seed) + 1;
                lo[2 * p] = static_cast<int8_t>((u[0] & 0xF) | ((u[1] & 0xF) << 4));
                lo[2 * p + 1] = static_cast<int8_t>((u[2] & 0xF) | ((u[3] & 0xF) << 4));
                if constexpr (LMAX == 31)
                    hi[p] = static_cast<int8_t>(((u[0] >> 4) & 3) | (((u[1] >> 4) & 3) << 2) | (((u[2] >> 4) & 3) << 4) | (((u[3] >> 4) & 3) << 6));
                else
                    hib[p >> 1] |= static_cast<uint8_t>((((u[0] >> 4) & 1) | (((u[1] >> 4) & 1) << 1) | (((u[2] >> 4) & 1) << 2) | (((u[3] >> 4) & 1) << 3)) << (4 * (p & 1)));
            }
            if constexpr (LMAX != 31) {
                #pragma unroll
                for (int p = 0; p < G / 8; ++p) hi[p] = static_cast<int8_t>(hib[p]);
            }
        }
    };
    if (stochastic) phase2(std::true_type{}); else phase2(std::false_type{});
}

// Row k of H4, same association as int8_linear.cu's h4_row_dot (bit-identical).
__device__ __forceinline__ float rq_h4_row(int k, float x0, float x1, float x2, float x3) {
    return 0.5f * ((((k == 3 ? -x0 : x0) + (k == 2 ? -x1 : x1)) + (k == 1 ? -x2 : x2)) + (k == 0 ? -x3 : x3));
}

// Radix-4 stage pairing register bit RB with the lane bit selected by LX.
template <int RB, int LX>
__device__ __forceinline__ void rq_fht_mixed(float (&v)[32], int lane_bit) {
    #pragma unroll
    for (int r = 0; r < 32; ++r) {
        if (r & (1 << RB)) continue;
        const int r1 = r | (1 << RB);
        const float a0 = v[r], a1 = v[r1];
        const float p0 = __shfl_xor_sync(0xffffffffu, a0, LX);
        const float p1 = __shfl_xor_sync(0xffffffffu, a1, LX);
        const float x0 = lane_bit ? p0 : a0, x1 = lane_bit ? p1 : a1;
        const float x2 = lane_bit ? a0 : p0, x3 = lane_bit ? a1 : p1;
        v[r] = rq_h4_row(2 * lane_bit, x0, x1, x2, x3);
        v[r1] = rq_h4_row(2 * lane_bit + 1, x0, x1, x2, x3);
    }
}

// ConvRot256 of one group in registers: 8 lanes x 32 values. Index bits 0,1,2,4,6 are register
// bits, 3,5,7 lane bits. Same stage order and arithmetic as rotate_int8_convrot_weight; the
// rounded result goes to the padded smem row.
template <typename InputType, int LOG2G>
__device__ __forceinline__ void rq_rotate_group_regs(const InputType* __restrict__ xg, InputType* row_buf,
                                                     int e_base, int lane8, bool store)
{
    const int b3 = lane8 & 1, b5 = (lane8 >> 1) & 1, b7 = lane8 >> 2;
    float v[32];   // r = b0 | b1<<1 | b2<<2 | b4<<3 | b6<<4
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int off = 8 * b3 + 16 * (q & 1) + 32 * b5 + 64 * (q >> 1) + 128 * b7;
        const uint4 raw = *reinterpret_cast<const uint4*>(xg + off);   // 8 x 16-bit
        const InputType* e = reinterpret_cast<const InputType*>(&raw);
        #pragma unroll
        for (int i = 0; i < 8; ++i) v[q * 8 + i] = rq_to_float(e[i]);
    }
    #pragma unroll
    for (int base = 0; base < 32; base += 4) {       // stage 0: bits (0,1), register-only
        const float x0 = v[base], x1 = v[base + 1], x2 = v[base + 2], x3 = v[base + 3];
        v[base] = 0.5f * (x0 + x1 + x2 - x3);
        v[base + 1] = 0.5f * (x0 + x1 - x2 + x3);
        v[base + 2] = 0.5f * (x0 - x1 + x2 + x3);
        v[base + 3] = 0.5f * (-x0 + x1 + x2 + x3);
    }
    rq_fht_mixed<2, 1>(v, b3);                        // bits (2,3)
    rq_fht_mixed<3, 2>(v, b5);                        // bits (4,5)
    rq_fht_mixed<4, 4>(v, b7);                        // bits (6,7)
    if (!store) return;
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int off = 8 * b3 + 16 * (q & 1) + 32 * b5 + 64 * (q >> 1) + 128 * b7;
        InputType* dst = row_buf + rq_padded_index<LOG2G>(e_base + off);   // even index: 4-byte pairs
        #pragma unroll
        for (int i = 0; i < 8; i += 2) {
            InputType pair[2] = {rq_from_float<InputType>(v[q * 8 + i]), rq_from_float<InputType>(v[q * 8 + i + 1])};
            *reinterpret_cast<uint32_t*>(dst + i) = *reinterpret_cast<const uint32_t*>(pair);
        }
    }
}

// Fused requant: rotate the row in registers, park it in smem, quantize (4/3-bit codebook g16 or 6/5-bit uniform).
template <typename InputType, int BITS, int G>
__global__ void __launch_bounds__(512)
quantize_wxa8_convrot_fused_kernel(
    const InputType* __restrict__ weight,   // [N, K]
    const float* __restrict__ codebook,     // [16] (4-bit), [8] (3-bit) or nullptr
    int8_t* __restrict__ packed,            // [N, K*BITS/8]
    uint8_t* __restrict__ s_rel,            // [N, K/G] e4m3 bits
    float* __restrict__ s_channel,          // [N]
    int K, bool stochastic, uint64_t seed)
{
    constexpr int LOG2G = RqLog2<G>::value;
    __shared__ float warp_max[32];
    __shared__ float cb[16];
    extern __shared__ float smem[];
    float* gscale = smem;                                                 // [K/G]
    InputType* row_buf = reinterpret_cast<InputType*>(gscale + (K >> LOG2G));   // padded rotated row
    if ((BITS == 4 || BITS == 3) && threadIdx.x < (BITS == 4 ? 16 : 8)) cb[threadIdx.x] = codebook[threadIdx.x];

    const int64_t row = blockIdx.x;
    const InputType* wrow = weight + row * K;
    const int clusters = blockDim.x / 8, cluster = threadIdx.x / 8, lane8 = threadIdx.x % 8;
    const int n_groups = K / 256;
    for (int it = 0; it < (n_groups + clusters - 1) / clusters; ++it) {   // uniform trip count (shuffles)
        const int g = it * clusters + cluster;
        const bool active = g < n_groups;
        rq_rotate_group_regs<InputType, LOG2G>(wrow + (active ? g : 0) * 256, row_buf, g * 256, lane8, active);
    }
    __syncthreads();
    if constexpr (BITS == 4 || BITS == 3) {
        w4_quantize_row<InputType, (BITS == 4 ? 16 : 8)>(
            row_buf, K, row * K, cb,
            packed + row * (K * BITS / 8), s_rel + row * (K / G), s_channel + row, gscale, warp_max, stochastic, seed);
    } else {
        w6_quantize_row<InputType, G, (BITS == 6 ? 31 : 15)>(
            row_buf, K, row * K,
            packed + row * (K * BITS / 8), s_rel + row * (K / G), s_channel + row, gscale, warp_max, stochastic, seed);
    }
}

template <int BITS, int G>
bool launch_wxa8_fused(const void* weight, const float* codebook, void* packed, void* s_rel, void* s_channel,
                       int64_t N, int64_t K, int in_dtype_code, bool stochastic, uint64_t seed, cudaStream_t stream)
{
    const bool wide = K > (BITS <= 4 ? 16384 : 8192);   // long rows: one block per SM anyway, use more warps
    // Short W4/W3 rows have at most 128 quantization groups. Avoid reserving
    // eight warps per row when most of them cannot contribute to quantization.
    const int threads = (BITS <= 4 && K <= 2048) ? ((static_cast<int>(K / G) + 31) / 32) * 32
                                               : (wide ? 512 : 256);
    const size_t shmem = (K / G) * sizeof(float) + (static_cast<size_t>(K) + 2 * (K / G)) * 2;   // group steps + 16-bit padded row
    int dev = 0, max_shmem = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    if (cudaDeviceGetAttribute(&max_shmem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess)
        return false;
    if (shmem + 256 > static_cast<size_t>(max_shmem)) return false;
    dim3 grid(static_cast<unsigned>(N));
    auto run = [&](auto* tag) -> bool {
        using IT = std::remove_pointer_t<decltype(tag)>;
        auto kern = quantize_wxa8_convrot_fused_kernel<IT, BITS, G>;
        if (cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shmem)) != cudaSuccess) {
            (void)cudaGetLastError();
            return false;
        }
        kern<<<grid, threads, shmem, stream>>>(static_cast<const IT*>(weight), codebook, static_cast<int8_t*>(packed),
                                               static_cast<uint8_t*>(s_rel), static_cast<float*>(s_channel),
                                               static_cast<int>(K), stochastic, stochastic ? seed : 0);
        return true;
    };
    bool ok;
    if (in_dtype_code == 0) return false;   // fp32 weights use staged requantization
    if (in_dtype_code == 1) ok = run(static_cast<__half*>(nullptr));
    else ok = run(static_cast<__nv_bfloat16*>(nullptr));
    if (!ok) return false;
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace

// weight [N,K] raw, ConvRot group 256, fp16/bf16 (fp32 declines); bits 4/3 (codebook, G=16) or
// 6/5 (G in {16,32,64}). Returns false when the row does not fit shared memory: caller runs eager.
extern "C" bool launch_quantize_wxa8_convrot_fused(
    const void* weight, const void* codebook, void* packed, void* s_rel, void* s_channel,
    int64_t N, int64_t K, int bits, int G, int in_dtype_code, bool stochastic, uint64_t seed, cudaStream_t stream)
{
    if (K % 256 != 0 || K % G != 0) return false;
    const float* cb = static_cast<const float*>(codebook);
    if (bits == 4 && G == 16) return launch_wxa8_fused<4, 16>(weight, cb, packed, s_rel, s_channel, N, K, in_dtype_code, stochastic, seed, stream);
    if (bits == 6 && G == 16) return launch_wxa8_fused<6, 16>(weight, nullptr, packed, s_rel, s_channel, N, K, in_dtype_code, stochastic, seed, stream);
    if (bits == 6 && G == 32) return launch_wxa8_fused<6, 32>(weight, nullptr, packed, s_rel, s_channel, N, K, in_dtype_code, stochastic, seed, stream);
    if (bits == 6 && G == 64) return launch_wxa8_fused<6, 64>(weight, nullptr, packed, s_rel, s_channel, N, K, in_dtype_code, stochastic, seed, stream);
    if (bits == 3 && G == 16) return launch_wxa8_fused<3, 16>(weight, cb, packed, s_rel, s_channel, N, K, in_dtype_code, stochastic, seed, stream);
    if (bits == 5 && G == 16) return launch_wxa8_fused<5, 16>(weight, nullptr, packed, s_rel, s_channel, N, K, in_dtype_code, stochastic, seed, stream);
    if (bits == 5 && G == 32) return launch_wxa8_fused<5, 32>(weight, nullptr, packed, s_rel, s_channel, N, K, in_dtype_code, stochastic, seed, stream);
    if (bits == 5 && G == 64) return launch_wxa8_fused<5, 64>(weight, nullptr, packed, s_rel, s_channel, N, K, in_dtype_code, stochastic, seed, stream);
    return false;
}

// Staged W4A8 requant for FP32 and rows too wide for the fused rotated-row buffer.
// Only group scales occupy shared memory; ConvRot runs separately in bounded row chunks.
namespace {

// nearest index in cb[16] (lowest index on tie); cb sorted ascending
__device__ __forceinline__ int rq_nearest(float x, const float* cb) {
    int best = 0; float bd = fabsf(x - cb[0]);
    #pragma unroll
    for (int j = 1; j < 16; ++j) { float d = fabsf(x - cb[j]); if (d < bd) { bd = d; best = j; } }
    return best;
}

template <typename InputType, bool STOCHASTIC>
__global__ void quantize_w4a8_convrot_kernel(
    const InputType* __restrict__ rotated,  // [N, K]
    const float* __restrict__ codebook,     // [16]
    int8_t* __restrict__ packed,            // [N, K/2]
    uint8_t* __restrict__ s_rel,            // [N, K/16] e4m3 bits
    float* __restrict__ s_channel,          // [N]
    int K, uint64_t seed)
{
    constexpr int G = 16;
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int groups = K / G;
    const int64_t row_off = static_cast<int64_t>(row) * K;

    __shared__ float cb[16];
    __shared__ float warp_max[32];
    extern __shared__ float gscale[];  // [groups]
    if (tid < 16) cb[tid] = codebook[tid];
    __syncthreads();

    // --- Phase 1: per-group ALS group scale; accumulate row shifted-amax ---
    float thr_amax = 0.0f;
    for (int g = tid; g < groups; g += nthreads) {
        const int64_t base = row_off + static_cast<int64_t>(g) * G;
        float w[G];
        float amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < G; ++i) { w[i] = rq_to_float(rotated[base + i]); amax = fmaxf(amax, fabsf(w[i])); }
        float gs = fmaxf(amax, 1e-8f);
        int idx[G];
        #pragma unroll
        for (int i = 0; i < G; ++i) idx[i] = rq_nearest(w[i] / gs, cb);
        #pragma unroll
        for (int it = 0; it < 2; ++it) {       // matches eager _ALS_ITERS
            float num = 0.0f, den = 0.0f;
            #pragma unroll
            for (int i = 0; i < G; ++i) { float c = cb[idx[i]]; num += w[i] * c; den += c * c; }
            gs = fmaxf(num / fmaxf(den, 1e-8f), 1e-8f);
            #pragma unroll
            for (int i = 0; i < G; ++i) idx[i] = rq_nearest(w[i] / gs, cb);
        }
        gscale[g] = gs;
        #pragma unroll
        for (int i = 0; i < G; ++i) thr_amax = fmaxf(thr_amax, fabsf(cb[idx[i]] * gs));
    }

    // --- block reduce thr_amax -> s_channel = row_amax / 127 ---
    float wm = rq_warp_max(thr_amax);
    const int lane = tid & 31, wid = tid >> 5;
    if (lane == 0) warp_max[wid] = wm;
    __syncthreads();
    if (wid == 0) {
        const int nwarps = (nthreads + 31) >> 5;
        float t = (lane < nwarps) ? warp_max[lane] : 0.0f;
        t = rq_warp_max(t);
        if (lane == 0) warp_max[0] = t;
    }
    __syncthreads();
    const float sc = fmaxf(warp_max[0] / 127.0f, 1e-8f);
    if (tid == 0) s_channel[row] = sc;

    // --- Phase 2: s_rel (fp8) -> int8 levels -> assign (+SR) -> pack ---
    const int64_t prow_off = static_cast<int64_t>(row) * (K / 2);
    const int64_t srow_off = static_cast<int64_t>(row) * groups;
    for (int g = tid; g < groups; g += nthreads) {
        const float gs = gscale[g];
        const float srel_f = gs / sc;
        const uint8_t srel_bits = __nv_cvt_float_to_fp8(srel_f, __NV_SATFINITE, __NV_E4M3);
        s_rel[srow_off + g] = srel_bits;
        const float srel_r = __half2float(__nv_cvt_fp8_to_halfraw(srel_bits, __NV_E4M3));
        float lv[16];
        #pragma unroll
        for (int j = 0; j < 16; ++j) lv[j] = fminf(127.0f, fmaxf(-127.0f, nearbyintf(cb[j] * srel_r)));
        const int64_t base = row_off + static_cast<int64_t>(g) * G;
        int u[G];
        #pragma unroll
        for (int i = 0; i < G; ++i) {
            const float t = rq_to_float(rotated[base + i]) / sc;
            int a;
            if constexpr (STOCHASTIC) {
                int lo = 0;
                #pragma unroll
                for (int j = 0; j < 16; ++j) lo += (lv[j] <= t);
                lo = min(max(lo - 1, 0), 14);
                const float thr = lv[lo] + rq_uniform(base + i, seed) * (lv[lo + 1] - lv[lo]);
                a = min(lo + (t > thr ? 1 : 0), 15);
            } else {
                a = rq_nearest(t, lv);
            }
            u[i] = a;
        }
        const int64_t base_p = prow_off + static_cast<int64_t>(g) * (G / 2);
        #pragma unroll
        for (int p = 0; p < G / 2; ++p)
            packed[base_p + p] = static_cast<int8_t>((u[2 * p] & 0xF) | ((u[2 * p + 1] & 0xF) << 4));
    }
}

}  // namespace

// rotated: [N,K] in in_dtype (0=fp32,1=fp16,2=bf16); s_rel: [N,K/16] e4m3 bits (uint8).
// Returns false (caller must fall back / raise) if the group-scale shared memory won't fit
// or the launch is rejected, so uninitialized outputs are never mistaken for a result.
extern "C" bool launch_quantize_w4a8_convrot(
    const void* rotated, const void* codebook, void* packed, void* s_rel, void* s_channel,
    int64_t N, int64_t K, int in_dtype_code, bool stochastic, uint64_t seed, cudaStream_t stream)
{
    const int threads = 256;
    const size_t shmem = static_cast<size_t>(K / 16) * sizeof(float);
    // Static shared is cb[16] + warp_max[32] = 192 bytes; bail if static+dynamic won't fit.
    int dev = 0, max_shmem = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    if (cudaDeviceGetAttribute(&max_shmem, cudaDevAttrMaxSharedMemoryPerBlock, dev) != cudaSuccess)
        return false;
    if (shmem + 192 > static_cast<size_t>(max_shmem)) return false;
    dim3 grid(static_cast<unsigned>(N));
#define RQ_LAUNCH(IT)                                                                              \
    do {                                                                                           \
        if (stochastic)                                                                            \
            quantize_w4a8_convrot_kernel<IT, true><<<grid, threads, shmem, stream>>>(              \
                static_cast<const IT*>(rotated), static_cast<const float*>(codebook),              \
                static_cast<int8_t*>(packed), static_cast<uint8_t*>(s_rel),                        \
                static_cast<float*>(s_channel), static_cast<int>(K), seed);                        \
        else                                                                                       \
            quantize_w4a8_convrot_kernel<IT, false><<<grid, threads, shmem, stream>>>(             \
                static_cast<const IT*>(rotated), static_cast<const float*>(codebook),              \
                static_cast<int8_t*>(packed), static_cast<uint8_t*>(s_rel),                        \
                static_cast<float*>(s_channel), static_cast<int>(K), 0);                           \
    } while (0)
    if (in_dtype_code == 0) RQ_LAUNCH(float);
    else if (in_dtype_code == 1) RQ_LAUNCH(__half);
    else RQ_LAUNCH(__nv_bfloat16);
#undef RQ_LAUNCH
    return cudaGetLastError() == cudaSuccess;
}

// Fused W4A8/W6A8 (and 3/5-bit) GEMV for decode (M<=8): dequantize the code planes in registers and
// dp4a against the int8 activation in one pass — no int8 workspace round-trip.
// Bit-exact with the chunked path: same __float2int_rn(cb[c]*s_rel) int8 grid,
// same acc*xs*s_channel(+bias) epilogue. Requires G>=16 and G%16==0 so one
// 16-col vec never spans two groups.
namespace {

constexpr int kGemvMaxM = 8;  // sizes acc[] below and gates the launcher

template <int WARPS_PER_BLOCK, typename OutT, int BITS>
__global__ void w4a8_codebook_gemv_kernel(
    const int8_t* __restrict__ xq,        // (M, K) int8 rotated+quantized activation
    const int8_t* __restrict__ qw,        // (N, K*BITS/8) packed codes
    const uint8_t* __restrict__ s_rel,    // (N, K/G) e4m3 raw
    const float* __restrict__ codebook,   // 16 floats (4-bit, or nullptr), 8 floats (3-bit), else unused
    const float* __restrict__ s_channel,  // (N)
    const float* __restrict__ xs,         // (M)
    const OutT* __restrict__ bias,        // (N) in the output dtype, or nullptr
    OutT* __restrict__ out,               // (M, N)
    int M, int N, int K, int G)
{
    __shared__ float cb[16];
    if constexpr (BITS == 4) {
        if (threadIdx.x < 16)
            cb[threadIdx.x] = codebook ? codebook[threadIdx.x] : (static_cast<float>(threadIdx.x) - 8.0f);
        __syncthreads();
    } else if constexpr (BITS == 3) {
        if (threadIdx.x < 8)
            cb[threadIdx.x] = codebook[threadIdx.x];
        __syncthreads();
    }

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int n = static_cast<int>(blockIdx.x) * WARPS_PER_BLOCK + warp;
    if (n >= N)
        return;

    const int nvec = K >> 4;                      // 16-col vecs per row
    const int nG = K / G;
    const int8_t* __restrict__ wrow = qw + static_cast<int64_t>(n) * (static_cast<int64_t>(K) * BITS / 8);
    const uint8_t* __restrict__ srow = s_rel + static_cast<int64_t>(n) * nG;

    // one weight pass shared across all M rows (M <= kGemvMaxM)
    int acc[kGemvMaxM] = {};
    for (int v = lane; v < nvec; v += 32) {
        unsigned hi;
        const uint2 pk = load_vec_planes<BITS>(wrow, v, K, hi);
        const int k0 = v * 16;
        const float s = load_scale<uint8_t>(srow[k0 / G]);
        char4 w4[4];
        dequant16_to_int8<BITS>(pk, hi, cb, s, s, s, s, G, w4);
        const int kw = k0 >> 2;
        for (int m = 0; m < M; ++m) {
            const int* __restrict__ x4 = reinterpret_cast<const int*>(xq + static_cast<int64_t>(m) * K);
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                acc[m] = __dp4a(x4[kw + j], *reinterpret_cast<const int*>(&w4[j]), acc[m]);
        }
    }

    for (int m = 0; m < M; ++m) {
        int a = acc[m];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            a += __shfl_down_sync(0xffffffffu, a, offset);
        if (lane == 0) {
            float value = static_cast<float>(a) * xs[m] * s_channel[n];
            if (bias)
                value += static_cast<float>(bias[n]);
            out[static_cast<int64_t>(m) * N + n] = comfy::from_float<OutT>(value);
        }
    }
}

}  // namespace

extern "C" bool launch_w4a8_codebook_gemv(
    const void* xq, const void* weight, const void* s_rel, const void* codebook,
    const void* s_channel, const void* xs, const void* bias, void* out,
    int64_t M, int64_t N, int64_t K, int64_t G, int64_t bits,
    int out_dtype_code, cudaStream_t stream)
{
    if (M < 1 || M > kGemvMaxM)
        return false;
    constexpr int kWarps = 4;
    dim3 block(kWarps * 32);
    dim3 grid((N + kWarps - 1) / kWarps);
#define GEMV_LAUNCH_B(OT, B)                                                                       \
    w4a8_codebook_gemv_kernel<kWarps, OT, B><<<grid, block, 0, stream>>>(                          \
        static_cast<const int8_t*>(xq), static_cast<const int8_t*>(weight),                        \
        static_cast<const uint8_t*>(s_rel), static_cast<const float*>(codebook),                   \
        static_cast<const float*>(s_channel), static_cast<const float*>(xs),                       \
        static_cast<const OT*>(bias), static_cast<OT*>(out),                                       \
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K), static_cast<int>(G))
#define GEMV_LAUNCH(OT) dispatch_bits(bits, [&](auto tag) { GEMV_LAUNCH_B(OT, decltype(tag)::value); })
    if (out_dtype_code == 0) GEMV_LAUNCH(float);
    else if (out_dtype_code == 1) GEMV_LAUNCH(__half);
    else GEMV_LAUNCH(__nv_bfloat16);
#undef GEMV_LAUNCH
#undef GEMV_LAUNCH_B
    return cudaGetLastError() == cudaSuccess;
}
