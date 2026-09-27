// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Thread-level GEMM for gfx90c (Vega APU) and gfx1010 (RDNA1), the two software
// tile targets it has been measured and validated on. Every other target keeps
// gemm_wmma.h (the native WMMA kernels, or the software policy in mma.h on the
// other pre-WMMA parts), and this kernel's body compiles to nothing for them.
//
// The software policy emulates the WMMA fragment contract: every MMA broadcasts
// the A row eight times across the wave, so most of its time goes to cross-lane
// shuffles rather than arithmetic. This kernel is the classic register-blocked
// form instead. Each thread owns a TM x TN block of C, LDS holds K-major 32-bit
// words so a thread reads its rows and columns with aligned 16-byte loads, and
// each word it reads is unpacked once and then used TN (or TM) times. No
// cross-lane operation remains, so the same code serves gfx90c's wave64 and
// gfx1010's wave32.
//
// Neither target has the dot-product instructions, so the word-level dot is
// plain arithmetic, shaped for what each has:
//   int8  sign-extended bytes into v_mad_i32_i24 (both).
//   int4  sign-extended nibbles, half a word at a time, into 24-bit multiplies.
//   fp8   e4m3 is re-encoded as fp16 (exact, see SimtFp8) once, as the tile is
//         stored to LDS, then v_fma_mix_f32 on gfx1010 and fp32 FMAs on gfx90c,
//         which has no mixed FMA.
//
// Measured against the software policy at M=4096, N=K=3072, the int8 / fp8 / int4
// GEMMs went from 0.69 / 0.22 / 0.63 to 4.7 / 5.3 / 4.8 TOPS on gfx1010 and from
// 0.10 / 0.03 / 0.15 to 0.79 / 0.76 / 0.85 on gfx90c.
//
// Computes C[M, N] = epilogue(A[M, K] @ B[N, K]^T) with the operands in their
// natural row-major form and C written with row stride ldc, like gemm_wmma.h.
// kbytes, the row length in bytes, must be a multiple of 16.
#pragma once

#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstring>

#include "gemm_wmma.h"

namespace comfy::hip_backend {

// Device pass: the targets this kernel is built for. Keep in step with
// kSimtArchNames below.
#if defined(__gfx90c__) || defined(__gfx1010__)
#define COMFY_SIMT_GEMM 1
#endif
#if defined(__gfx1010__)
#define COMFY_SIMT_FMA_MIX 1  // v_fma_mix_f32: fp16 operands, fp32 accumulate
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
// faster for int8 on both gfx1010 and gfx90c.
__forceinline__ __device__ int mad_i24(int a, int b, int c) {
    asm("v_mad_i32_i24 %0, %1, %2, %0" : "+v"(c) : "v"(a), "v"(b));
    return c;
}

struct SimtInt8 {
    using Acc = int;
    static constexpr int kExpand = 1;
    static constexpr int kSubs = 1;
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

#if defined(COMFY_SIMT_FMA_MIX)
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
#else
    // No mixed FMA: convert at unpack, where the cost is shared by TN (or TM) uses.
    struct Frag {
        float v[2];
    };
    static __forceinline__ __device__ Frag unpack(uint32_t w, int) {
        const half2_t h = __builtin_bit_cast(half2_t, w);
        return {{static_cast<float>(h.x), static_cast<float>(h.y)}};
    }
    static __forceinline__ __device__ float dot(const Frag& a, const Frag& b, float c) {
        c = __builtin_fmaf(a.v[0], b.v[0], c);
        return __builtin_fmaf(a.v[1], b.v[1], c);
    }
#endif
    static __forceinline__ __device__ float finish(float c) { return c * 65536.0f; }
};

// ---------------------------------------------------------------------------
// The kernel. 256 threads as a 16 x 16 grid; thread (tx, ty) owns rows
// ty*4 + {0..3} of each 64-row slice of the block tile and likewise columns, so
// BM = 16 * TM and BN = 16 * TN, and its reads of a K word are one 16-byte LDS
// load per four rows or columns.
// ---------------------------------------------------------------------------

constexpr int kSimtThreads = 256;
constexpr int kSimtWords = 16;  // K words (64 bytes) staged per iteration

// Two blocks per CU caps gfx90c at the 128 VGPRs that keep two wave64s per SIMD;
// left uncapped its fp8 and int4 tiles took ~200 and ran 15-30% slower. gfx1010's
// wave32 budget is wider and it measured the same either way.
template <typename Op, typename Epi, typename OutT, int TM, int TN>
__global__ __launch_bounds__(kSimtThreads, 2) void gemm_simt_kernel(
    const uint8_t* __restrict__ A, const uint8_t* __restrict__ B, OutT* __restrict__ C, int M,
    int N, int kbytes, int ldc, Epi epi) {
#if defined(COMFY_SIMT_GEMM)
    constexpr int BM = 16 * TM, BN = 16 * TN;
    constexpr int kBytes = kSimtWords * 4;
    constexpr int kChunks = kBytes / 16;  // 16-byte global chunks per row per stage
    constexpr int kAPer = BM * kChunks / kSimtThreads;
    constexpr int kBPer = BN * kChunks / kSimtThreads;
    static_assert(TM % 4 == 0 && TN % 4 == 0, "a thread reads its rows four at a time");
    static_assert(kAPer * kSimtThreads == BM * kChunks && kBPer * kSimtThreads == BN * kChunks,
                  "the block's threads must split the tile's chunks evenly");
    constexpr int kE = Op::kExpand;
    constexpr int kLdsWords = kSimtWords * kE;  // LDS words per row per stage

    // K-major: LDS word lw of row r at [lw * BM + r], where global word kw expands
    // to lw = kw * kE .. kw * kE + kE - 1. Every read is a 16-byte load at a
    // multiple of 16 bytes, which also keeps clear of gfx1010's misaligned-LDS
    // hazard (LLVM's lds-misaligned-bug).
    __shared__ __align__(16) uint32_t As[kLdsWords * BM];
    __shared__ __align__(16) uint32_t Bs[kLdsWords * BN];

    const int tid = threadIdx.x;
    const int tx = tid % 16, ty = tid / 16;

    // Grouped block order, as in gemm_wmma.h: consecutive blocks walk M within a
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

    // Chunk c covers row c % ROWS and K bytes (c / ROWS) * 16, so consecutive
    // threads store consecutive rows of one LDS word: distinct banks.
    uint4 ra[kAPer], rb[kBPer];
    auto load = [&](int kb0) {
#pragma unroll
        for (int i = 0; i < kAPer; ++i) {
            const int c = tid + i * kSimtThreads;
            const int grow = m0 + c % BM, gk = kb0 + (c / BM) * 16;
            ra[i] = (grow < M && gk < kbytes)
                        ? *reinterpret_cast<const uint4*>(A + static_cast<int64_t>(grow) * kbytes + gk)
                        : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int i = 0; i < kBPer; ++i) {
            const int c = tid + i * kSimtThreads;
            const int grow = n0 + c % BN, gk = kb0 + (c / BN) * 16;
            rb[i] = (grow < N && gk < kbytes)
                        ? *reinterpret_cast<const uint4*>(B + static_cast<int64_t>(grow) * kbytes + gk)
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
        for (int i = 0; i < kAPer; ++i) store_chunk(As, BM, tid + i * kSimtThreads, ra[i]);
#pragma unroll
        for (int i = 0; i < kBPer; ++i) store_chunk(Bs, BN, tid + i * kSimtThreads, rb[i]);
    };

    typename Op::Acc acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0;

    load(0);
    store();
    __syncthreads();

    for (int kb0 = 0; kb0 < kbytes; kb0 += kBytes) {
        const bool has_next = kb0 + kBytes < kbytes;
        if (has_next) load(kb0 + kBytes);  // in flight across this tile's math

#pragma unroll COMFY_SIMT_UNROLL
        for (int lw = 0; lw < kLdsWords; ++lw) {
            uint32_t aw[TM], bw[TN];
#pragma unroll
            for (int g = 0; g < TM / 4; ++g) {
                const uint4 v = *reinterpret_cast<const uint4*>(As + lw * BM + g * 64 + ty * 4);
                aw[4 * g] = v.x, aw[4 * g + 1] = v.y, aw[4 * g + 2] = v.z, aw[4 * g + 3] = v.w;
            }
#pragma unroll
            for (int g = 0; g < TN / 4; ++g) {
                const uint4 v = *reinterpret_cast<const uint4*>(Bs + lw * BN + g * 64 + tx * 4);
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
                    for (int j = 0; j < TN; ++j) acc[i][j] = Op::dot(af, bf[j], acc[i][j]);
                }
            }
        }

        if (has_next) {
            __syncthreads();  // every thread is done reading this tile
            store();
            __syncthreads();
        }
    }

    epi.init();
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int r = m0 + (i / 4) * 64 + ty * 4 + i % 4;
        if (r >= M) continue;
        OutT* crow = C + static_cast<int64_t>(r) * ldc;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int col = n0 + (j / 4) * 64 + tx * 4 + j % 4;
            if (col >= N) continue;
            crow[col] = static_cast<OutT>(epi(r, col, Op::finish(acc[i][j])));
        }
    }
#endif  // COMFY_SIMT_GEMM
}

// ---------------------------------------------------------------------------
// Host side.
// ---------------------------------------------------------------------------

// Whether the current device is one this kernel is built for, by architecture
// name. Cached per ordinal; an unreadable name answers false, which keeps the
// caller on its existing path. Keep in step with COMFY_SIMT_GEMM above.
inline bool device_uses_simt_gemm() {
    static constexpr const char* kSimtArchNames[] = {"gfx90c", "gfx1010"};
    constexpr int kMaxDevices = 16;
    static std::atomic<int> cache[kMaxDevices] = {};  // 0 unknown, 1 no, 2 yes
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= kMaxDevices) return false;
    int v = cache[dev].load(std::memory_order_relaxed);
    if (v == 0) {
        v = 1;
        hipDeviceProp_t props{};
        if (hipGetDeviceProperties(&props, dev) == hipSuccess) {
            const char* name = props.gcnArchName;
            const size_t len = strcspn(name, ":");
            for (const char* arch : kSimtArchNames) {
                if (strlen(arch) == len && strncmp(name, arch, len) == 0) v = 2;
            }
        }
        cache[dev].store(v, std::memory_order_relaxed);
    }
    return v == 2;
}

// Tile choice. 128x128 has the most reuse per LDS word; the smaller tiles trade
// it for enough blocks to fill the device when M or N is small.
// device_wgp_count() is CUs on gfx90c and WGPs on gfx1010, the unit a block lands on.
template <typename Op, typename Epi, typename OutT>
void launch_gemm_simt(const uint8_t* A, const uint8_t* B, OutT* C, int M, int N, int kbytes,
                      int ldc, Epi epi, hipStream_t stream) {
    const int units = device_wgp_count();
    const auto blocks = [&](int bm, int bn) {
        return static_cast<int64_t>((M + bm - 1) / bm) * ((N + bn - 1) / bn);
    };
    if (M > 64 && blocks(128, 128) >= 2 * units) {
        dim3 grid((N + 127) / 128, (M + 127) / 128);
        gemm_simt_kernel<Op, Epi, OutT, 8, 8>
            <<<grid, kSimtThreads, 0, stream>>>(A, B, C, M, N, kbytes, ldc, epi);
    } else if (blocks(64, 128) >= 2 * units) {
        dim3 grid((N + 127) / 128, (M + 63) / 64);
        gemm_simt_kernel<Op, Epi, OutT, 4, 8>
            <<<grid, kSimtThreads, 0, stream>>>(A, B, C, M, N, kbytes, ldc, epi);
    } else {
        dim3 grid((N + 63) / 64, (M + 63) / 64);
        gemm_simt_kernel<Op, Epi, OutT, 4, 4>
            <<<grid, kSimtThreads, 0, stream>>>(A, B, C, M, N, kbytes, ldc, epi);
    }
}

}  // namespace comfy::hip_backend
