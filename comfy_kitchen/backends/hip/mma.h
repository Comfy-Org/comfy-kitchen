// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// WMMA policies for RDNA3 (gfx11) and RDNA4 (gfx12), wave32.
//
// A policy holds the operand fragment type, the bytes of a K-row one MMA consumes
// (kStepBytes), the LDS read, and the MMA. The tile kernels stay byte-addressed;
// only the policy changes per architecture.
//
// Operand layout differs:
//   gfx12: K is split across the half-waves. A lane holds 8 bytes at byte offset
//          8 * (lane / 16) of its row. Every type is a v2i, which is what lets one
//          kernel serve fp8, iu8 and iu4.
//   gfx11: no K split. A lane holds the whole K-step of its row and both
//          half-waves hold the same data (rocWMMA calls it input duplication),
//          which reading at row = lane % 16 with no half-wave offset gives for
//          free. Fragment width is the K-step: 16B iu8, 8B iu4, 32B bf16 for fp8.
//
// Accumulators are v8f/v8i on both, column lane % 16, but the row differs:
// gfx12 gives each half-wave a contiguous block, D[e + 8 * (lane / 16)]; gfx11
// interleaves the halves, D[2 * e + (lane / 16)]. See acc_row. rocWMMA's "padded
// acc" gfx11 quirk applies to the 16-bit accumulators, not these.
//
// gfx11 has no fp8 WMMA, so MmaFp8/MmaBf8 widen to bf16 there. The widening is
// exact (e4m3 and e5m2 mantissas fit bf16's 7) and both paths accumulate in f32,
// so gfx11 fp8 matches gfx12 numerically, just slower.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

#include "architecture_config.h"
#include "fp8_utils.h"

namespace comfy::hip_backend {

typedef int int32_t_v2 __attribute__((ext_vector_type(2)));
typedef int int32_t_v4 __attribute__((ext_vector_type(4)));

typedef int32_t_v2 v2i;
typedef int32_t_v4 v4i;
typedef int v8i __attribute__((ext_vector_type(8)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef __bf16 v8bf __attribute__((ext_vector_type(8)));
typedef __bf16 v16bf __attribute__((ext_vector_type(16)));
typedef _Float16 v8h __attribute__((ext_vector_type(8)));
typedef _Float16 v16h __attribute__((ext_vector_type(16)));

constexpr int kWave = 32;

// LDS row padding the tiled GEMM core must use for a given policy's fragment
// reads, expressed as two base classes so every policy inherits one and only the
// ones with a reason to differ say so.
//
// 8 spreads 16 rows across 16 distinct banks (stride 34 words: lane r lands on
// bank 2r), and is what a 32-bit or 64-bit fragment read wants. 16 keeps every
// row 16-byte aligned at the cost of a two-way bank conflict (stride 36 words:
// lanes 0-7 of a 128-bit phase still cover all 32 banks), which a policy whose
// fragment is a whole 128-bit K-step trades for four times the issue slots per
// read. The int8 attention kernels reached that trade on gfx11 long ago
// (sage_attention/int8_attn.hip's kLdsPad); the GEMM core did not, and its
// fragment reads were the four-scalar form of that.
struct LdsPad8 {
    static constexpr int kLdsPad = 8;
};
struct LdsPad16 {
    static constexpr int kLdsPad = 16;
};

// 2^x for the softmax paths. Under -ffast-math clang lowers exp2f to a
// range-reduction sequence (compare, two selects, add, v_exp_f32, ldexp): about
// five VALU instructions per call, and every attention kernel here calls it once
// per score element. On gfx11 the hardware instruction computes 2^x directly and
// is bit-identical to exp2f over [-150, 10], which covers every argument these
// kernels produce (a score minus a running max, so <= 0, plus the small
// probability offset). Attention is issue-bound with WMMA and VALU sharing one
// port, so the substitution is a direct cut of the non-WMMA VALU work: the D64
// int8 key-tile loop drops 1759 -> 1474 instructions.
//
// Restricted to the gfx1103 device pass (COMFY_MMA_GFX1103 is generated from the
// architecture manifest and defined only when the compiler targets gfx1103): the
// equivalence was verified on gfx1103, which is where it was measured, so every
// other target -- gfx1100/1102 dGPUs, gfx115x, gfx12 -- keeps upstream's exp2f.
// This is the same arch gate the host-side comfy_small_igpu() applies, just
// resolved at compile time because the choice sits in device code.
__device__ __forceinline__ float hw_exp2(float x) {
#if defined(COMFY_MMA_GFX1103)
    return __builtin_amdgcn_exp2f(x);
#else
    return exp2f(x);
#endif
}

// architecture_config.h is generated from architectures.json. __gfx*__ is
// defined only in device passes; the host pass falls through to the stubs below.

// ---------------------------------------------------------------------------
// Fragment addressing, shared by both architectures.
// ---------------------------------------------------------------------------

// Sum a value across the wave. The first offset comes from kWave rather than a
// literal 16: at wave64 a hardcoded 16 would fold only half the lanes and say
// nothing about it.
template <typename T>
__forceinline__ __device__ T wave_reduce_sum(T v) {
    #pragma unroll
    for (int off = kWave / 2; off > 0; off >>= 1) v += __shfl_xor(v, off, kWave);
    return v;
}

// Four int8 products accumulated into a 32-bit sum, the operands packed one per
// byte. RDNA3/4 spell it v_dot4_i32_iu8 with both operands marked signed; RDNA2
// has the older v_dot4c_i32_i8 under a different builtin, and the host pass has
// neither, so both fall back to the arithmetic the instruction performs.
__forceinline__ __device__ int dot4_i8(int a, int b, int c) {
#if defined(COMFY_HAS_WMMA)
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#else
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        c += static_cast<int>(static_cast<int8_t>((a >> (i * 8)) & 0xFF)) *
             static_cast<int>(static_cast<int8_t>((b >> (i * 8)) & 0xFF));
    }
    return c;
#endif
}

__forceinline__ __device__ int frag_row(int lane) { return lane % 16; }

// Row of accumulator element `e` for this lane; the column is lane % 16. The two
// architectures lay the 16 result rows across the wave differently. gfx12 gives
// each half-wave a contiguous block of 8 rows. gfx11 interleaves them: element e
// is row 2e for the low half-wave and 2e + 1 for the high one, so consecutive
// elements are two rows apart. Both are 32-bit accumulators (v8f/v8i), so neither
// has the padded-acc read the 16-bit gfx11 accumulators need.
#if defined(COMFY_MMA_GFX11)
__forceinline__ __device__ int acc_row(int lane, int e) { return 2 * e + (lane / 16); }
#else
__forceinline__ __device__ int acc_row(int lane, int e) { return e + 8 * (lane / 16); }
#endif
__forceinline__ __device__ int acc_col(int lane) { return lane % 16; }

// Bytes of a row that live `stride` apart in LDS. `row` is the lane's row (A) or
// column (B); `kbyte` is the byte offset within that row.
__forceinline__ __device__ v2i load_frag_b64(const void* lds, int row, int kbyte, int stride) {
    const char* p = static_cast<const char*>(lds) + row * stride + kbyte;
    return *reinterpret_cast<const v2i*>(p);
}

// Same, for a row the caller guarantees is 16-byte aligned: one LDS instruction
// instead of four. gfx11 hands every lane the whole 16-byte K-step, so this is
// the difference between one ds_read_b128 and four ds_read_b32 on every single
// fragment read of an inner loop -- in attention and in the tiled GEMM core
// alike. The caller guarantees it by padding the LDS row stride to a multiple of
// 16 (LdsPad16 below); at stride +8 a row is only 8-byte aligned, and four
// separate ds_read_b32 is the only legal spelling, which is what this tree
// shipped until the GEMM core adopted the padded stride.
__forceinline__ __device__ v4i load_frag_b128_aligned(const void* lds, int row, int kbyte,
                                                      int stride) {
    const char* p = static_cast<const char*>(lds) + row * stride + kbyte;
    return *reinterpret_cast<const v4i*>(p);
}

// 16 fp8 bytes of a row -> the 16 bf16 lanes of one gfx11 K-step.
template <bool E5M2>
__forceinline__ __device__ v16bf load_frag_fp8_as_bf16(const void* lds, int row, int kbyte,
                                                       int stride) {
    const uint8_t* p = static_cast<const uint8_t*>(lds) + row * stride + kbyte;
    v16bf r;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        r[i] = static_cast<__bf16>(E5M2 ? bf8_to_float(p[i]) : fp8_to_float(p[i]));
    }
    return r;
}

// A K-step of a 16-bit row, straight out of whatever address space `row` points
// at. The half-wave slice is the policy's, since gfx11 hands every lane the whole
// K-step and gfx12 splits it. The elements are contiguous, so this folds into
// vector loads.
template <typename Mma>
__forceinline__ __device__ typename Mma::Frag load_frag_16bit(const typename Mma::Elem* row,
                                                              int lane) {
    typename Mma::Frag f;
    const typename Mma::Elem* p = row + Mma::frag_base(lane);
    #pragma unroll
    for (int i = 0; i < Mma::kFragElems; ++i) {
        f[i] = p[i];
    }
    return f;
}

// ---------------------------------------------------------------------------
// Policies
// ---------------------------------------------------------------------------

#if defined(COMFY_MMA_GFX12)

struct MmaFp8 : LdsPad8 {
    using Acc = v8f;
    using Frag = v2i;
    static constexpr int kStepBytes = 16;
    static __forceinline__ __device__ Frag load(const void* lds, int row, int kbyte, int stride,
                                                int lane) {
        return load_frag_b64(lds, row, kbyte + 8 * (lane / 16), stride);
    }
    static __forceinline__ __device__ Acc zero() { return Acc{0, 0, 0, 0, 0, 0, 0, 0}; }
    static __forceinline__ __device__ Acc mma(Frag a, Frag b, Acc c) {
        return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);
    }
    static __forceinline__ __device__ float get(Acc c, int e) { return c[e]; }
};

struct MmaBf8 : LdsPad8 {
    using Acc = v8f;
    using Frag = v2i;
    static constexpr int kStepBytes = 16;
    static __forceinline__ __device__ Frag load(const void* lds, int row, int kbyte, int stride,
                                                int lane) {
        return load_frag_b64(lds, row, kbyte + 8 * (lane / 16), stride);
    }
    static __forceinline__ __device__ Acc zero() { return Acc{0, 0, 0, 0, 0, 0, 0, 0}; }
    static __forceinline__ __device__ Acc mma(Frag a, Frag b, Acc c) {
        return __builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12(a, b, c);
    }
    static __forceinline__ __device__ float get(Acc c, int e) { return c[e]; }
};

struct MmaInt8 : LdsPad8 {
    using Acc = v8i;
    using Frag = v2i;
    static constexpr int kStepBytes = 16;
    static __forceinline__ __device__ Frag load(const void* lds, int row, int kbyte, int stride,
                                                int lane) {
        return load_frag_b64(lds, row, kbyte + 8 * (lane / 16), stride);
    }
    static __forceinline__ __device__ Acc zero() { return Acc{0, 0, 0, 0, 0, 0, 0, 0}; }
    static __forceinline__ __device__ Frag load_aligned(const void* lds, int row, int kbyte,
                                                        int stride, int lane) {
        return load_frag_b64(lds, row, kbyte + 8 * (lane / 16), stride);
    }
    static __forceinline__ __device__ Acc mma(Frag a, Frag b, Acc c) {
        return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c, false);
    }
    // Signedness is an instruction immediate, so an unsigned A needs its own entry.
    static __forceinline__ __device__ Acc mma_ua(Frag a, Frag b, Acc c) {
        return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a, true, b, c, false);
    }
    // ... and so does an unsigned B, which is the P operand of int8 attention.
    static __forceinline__ __device__ Acc mma_ub(Frag a, Frag b, Acc c) {
        return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, false, b, c, false);
    }
    static __forceinline__ __device__ float get(Acc c, int e) { return static_cast<float>(c[e]); }
};

// The one type whose K-step differs: gfx12 does K=32 per MMA, gfx11 does K=16.
struct MmaInt4 : LdsPad8 {
    using Acc = v8i;
    using Frag = v2i;
    static constexpr int kStepBytes = 16;
    static __forceinline__ __device__ Frag load(const void* lds, int row, int kbyte, int stride,
                                                int lane) {
        return load_frag_b64(lds, row, kbyte + 8 * (lane / 16), stride);
    }
    static __forceinline__ __device__ Acc zero() { return Acc{0, 0, 0, 0, 0, 0, 0, 0}; }
    static __forceinline__ __device__ Acc mma(Frag a, Frag b, Acc c) {
        return __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true, a, true, b, c, false);
    }
    static __forceinline__ __device__ Acc mma_ua(Frag a, Frag b, Acc c) {
        return __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(false, a, true, b, c, false);
    }
    static __forceinline__ __device__ float get(Acc c, int e) { return static_cast<float>(c[e]); }
};

// 16-bit float MMA, used by the attention kernels rather than the GEMM core: they
// build fragments from global memory and from LDS scratch, so the policy exposes
// the element type and the half-wave's K slice instead of a byte-addressed load.
struct MmaBf16 : LdsPad8 {
    using Acc = v8f;
    using Frag = v8bf;
    using Elem = __bf16;
    static constexpr int kFragElems = 8;
    static __forceinline__ __device__ int frag_base(int lane) { return 8 * (lane / 16); }
    static __forceinline__ __device__ Acc zero() { return Acc{0, 0, 0, 0, 0, 0, 0, 0}; }
    static __forceinline__ __device__ Acc mma(Frag a, Frag b, Acc c) {
        return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
    }
    static __forceinline__ __device__ float get(Acc c, int e) { return c[e]; }
};

struct MmaF16 : LdsPad8 {
    using Acc = v8f;
    using Frag = v8h;
    using Elem = _Float16;
    static constexpr int kFragElems = 8;
    // The GEMM core's byte-addressed view: 16 fp16 elements are 32 bytes.
    static constexpr int kStepBytes = 32;
    static __forceinline__ __device__ int frag_base(int lane) { return 8 * (lane / 16); }
    static __forceinline__ __device__ Frag load(const void* lds, int row, int kbyte, int stride,
                                                int lane) {
        return load_frag_16bit<MmaF16>(
            reinterpret_cast<const _Float16*>(static_cast<const char*>(lds) + row * stride + kbyte),
            lane);
    }
    static __forceinline__ __device__ Acc zero() { return Acc{0, 0, 0, 0, 0, 0, 0, 0}; }
    static __forceinline__ __device__ Acc mma(Frag a, Frag b, Acc c) {
        return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
    }
    static __forceinline__ __device__ float get(Acc c, int e) { return c[e]; }
};

// BF16 GEMM policy for gfx12, mirroring MmaF16's half-wave K split.
struct MmaBf16G : LdsPad8 {
    using Acc = v8f;
    using Frag = v8bf;
    using Elem = __bf16;
    static constexpr int kFragElems = 8;
    static constexpr int kStepBytes = 32;
    static __forceinline__ __device__ int frag_base(int lane) { return 8 * (lane / 16); }
    static __forceinline__ __device__ Frag load(const void* lds, int row, int kbyte, int stride,
                                                int lane) {
        return load_frag_16bit<MmaBf16G>(
            reinterpret_cast<const __bf16*>(static_cast<const char*>(lds) + row * stride + kbyte),
            lane);
    }
    static __forceinline__ __device__ Acc zero() { return Acc{0, 0, 0, 0, 0, 0, 0, 0}; }
    static __forceinline__ __device__ Acc mma(Frag a, Frag b, Acc c) {
        return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
    }
    static __forceinline__ __device__ float get(Acc c, int e) { return c[e]; }
};

#elif defined(COMFY_MMA_GFX11)

struct MmaFp8 : LdsPad8 {
    using Acc = v8f;
    using Frag = v16bf;
    static constexpr int kStepBytes = 16;  // 16 fp8 bytes == K-step of 16
    static __forceinline__ __device__ Frag load(const void* lds, int row, int kbyte, int stride,
                                                int lane) {
        return load_frag_fp8_as_bf16<false>(lds, row, kbyte, stride);
    }
    static __forceinline__ __device__ Acc zero() { return Acc{0, 0, 0, 0, 0, 0, 0, 0}; }
    static __forceinline__ __device__ Acc mma(Frag a, Frag b, Acc c) {
        return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
    }
    static __forceinline__ __device__ float get(Acc c, int e) { return c[e]; }
};

struct MmaBf8 : LdsPad8 {
    using Acc = v8f;
    using Frag = v16bf;
    static constexpr int kStepBytes = 16;
    static __forceinline__ __device__ Frag load(const void* lds, int row, int kbyte, int stride,
                                                int lane) {
        return load_frag_fp8_as_bf16<true>(lds, row, kbyte, stride);
    }
    static __forceinline__ __device__ Acc zero() { return Acc{0, 0, 0, 0, 0, 0, 0, 0}; }
    static __forceinline__ __device__ Acc mma(Frag a, Frag b, Acc c) {
        return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
    }
    static __forceinline__ __device__ float get(Acc c, int e) { return c[e]; }
};

struct MmaInt8 : LdsPad16 {
    using Acc = v8i;
    using Frag = v4i;  // whole K-step of 16 int8, duplicated across half-waves
    static constexpr int kStepBytes = 16;
    static __forceinline__ __device__ Frag load(const void* lds, int row, int kbyte, int stride,
                                                int lane) {
        // LdsPad16 makes every LDS row 16-byte aligned (BKB is already a multiple
        // of 16, kbyte is a whole K-step), so this is one ds_read_b128. At the
        // 8-byte pad the same address is only dword-aligned and the read has to
        // split into four ds_read_b32 -- four times the issue slots, once per
        // fragment, per K-step, on the innermost loop of the dominant GEMM.
        return load_frag_b128_aligned(lds, row, kbyte, stride);
    }
    static __forceinline__ __device__ Acc zero() { return Acc{0, 0, 0, 0, 0, 0, 0, 0}; }
    static __forceinline__ __device__ Frag load_aligned(const void* lds, int row, int kbyte,
                                                        int stride, int) {
        return load_frag_b128_aligned(lds, row, kbyte, stride);
    }
    static __forceinline__ __device__ Acc mma(Frag a, Frag b, Acc c) {
        return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, false);
    }
    static __forceinline__ __device__ Acc mma_ua(Frag a, Frag b, Acc c) {
        return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, a, true, b, c, false);
    }
    static __forceinline__ __device__ Acc mma_ub(Frag a, Frag b, Acc c) {
        return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, false, b, c, false);
    }
    static __forceinline__ __device__ float get(Acc c, int e) { return static_cast<float>(c[e]); }
};

struct MmaInt4 : LdsPad8 {
    using Acc = v8i;
    using Frag = v2i;  // K=16 nibbles == 8 bytes, whole step, duplicated
    static constexpr int kStepBytes = 8;
    static __forceinline__ __device__ Frag load(const void* lds, int row, int kbyte, int stride,
                                                int lane) {
        return load_frag_b64(lds, row, kbyte, stride);
    }
    static __forceinline__ __device__ Acc zero() { return Acc{0, 0, 0, 0, 0, 0, 0, 0}; }
    static __forceinline__ __device__ Acc mma(Frag a, Frag b, Acc c) {
        return __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, c, false);
    }
    static __forceinline__ __device__ Acc mma_ua(Frag a, Frag b, Acc c) {
        return __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, a, true, b, c, false);
    }
    static __forceinline__ __device__ float get(Acc c, int e) { return static_cast<float>(c[e]); }
};


// gfx11 hands every lane the whole 16-wide K-step, duplicated across half-waves.
struct MmaBf16 : LdsPad8 {
    using Acc = v8f;
    using Frag = v16bf;
    using Elem = __bf16;
    static constexpr int kFragElems = 16;
    static __forceinline__ __device__ int frag_base(int) { return 0; }
    static __forceinline__ __device__ Acc zero() { return Acc{0, 0, 0, 0, 0, 0, 0, 0}; }
    static __forceinline__ __device__ Acc mma(Frag a, Frag b, Acc c) {
        return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
    }
    static __forceinline__ __device__ float get(Acc c, int e) { return c[e]; }
};

// BF16 policy for the tiled GEMM core: unlike MmaBf16 (the fp8-widening path),
// a bf16 GEMM consumes 32 bytes of a row per 16-element K-step, same as MmaF16.
struct MmaBf16G : LdsPad8 {
    using Acc = v8f;
    using Frag = v16bf;
    using Elem = __bf16;
    static constexpr int kFragElems = 16;
    static constexpr int kStepBytes = 32;
    static __forceinline__ __device__ int frag_base(int) { return 0; }
    static __forceinline__ __device__ Frag load(const void* lds, int row, int kbyte, int stride,
                                                int lane) {
        return load_frag_16bit<MmaBf16G>(
            reinterpret_cast<const __bf16*>(static_cast<const char*>(lds) + row * stride + kbyte),
            lane);
    }
    static __forceinline__ __device__ Acc zero() { return Acc{0, 0, 0, 0, 0, 0, 0, 0}; }
    static __forceinline__ __device__ Acc mma(Frag a, Frag b, Acc c) {
        return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
    }
    static __forceinline__ __device__ float get(Acc c, int e) { return c[e]; }
};

struct MmaF16 : LdsPad8 {
    using Acc = v8f;
    using Frag = v16h;
    using Elem = _Float16;
    static constexpr int kFragElems = 16;
    static constexpr int kStepBytes = 32;
    static __forceinline__ __device__ int frag_base(int) { return 0; }
    static __forceinline__ __device__ Frag load(const void* lds, int row, int kbyte, int stride,
                                                int lane) {
        return load_frag_16bit<MmaF16>(
            reinterpret_cast<const _Float16*>(static_cast<const char*>(lds) + row * stride + kbyte),
            lane);
    }
    static __forceinline__ __device__ Acc zero() { return Acc{0, 0, 0, 0, 0, 0, 0, 0}; }
    static __forceinline__ __device__ Acc mma(Frag a, Frag b, Acc c) {
        return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
    }
    static __forceinline__ __device__ float get(Acc c, int e) { return c[e]; }
};

#else

// No matrix cores (RDNA2 and older) and the host pass. The GEMM kernels are
// instantiated in every device pass so they must still compile, but the backend
// reports no WMMA capability there and never launches them. Trap rather than
// return a plausible wrong answer.
#define COMFY_MMA_STUB_BODY \
    __builtin_trap();       \
    return c;

struct MmaFp8 : LdsPad8 {
    using Acc = v8f;
    using Frag = v2i;
    static constexpr int kStepBytes = 16;
    static __forceinline__ __device__ Frag load(const void*, int, int, int, int) { return Frag{}; }
    static __forceinline__ __device__ Acc zero() { return Acc{0, 0, 0, 0, 0, 0, 0, 0}; }
    static __forceinline__ __device__ Acc mma(Frag, Frag, Acc c) { COMFY_MMA_STUB_BODY }
    static __forceinline__ __device__ float get(Acc c, int e) { return c[e]; }
};

struct MmaBf8 : MmaFp8 {};

struct MmaInt8 : LdsPad8 {
    using Acc = v8i;
    using Frag = v2i;
    static constexpr int kStepBytes = 16;
    static __forceinline__ __device__ Frag load(const void*, int, int, int, int) { return Frag{}; }
    static __forceinline__ __device__ Acc zero() { return Acc{0, 0, 0, 0, 0, 0, 0, 0}; }
    static __forceinline__ __device__ Frag load_aligned(const void*, int, int, int, int) {
        return Frag{};
    }
    static __forceinline__ __device__ Acc mma(Frag, Frag, Acc c) { COMFY_MMA_STUB_BODY }
    static __forceinline__ __device__ Acc mma_ua(Frag, Frag, Acc c) { COMFY_MMA_STUB_BODY }
    static __forceinline__ __device__ Acc mma_ub(Frag, Frag, Acc c) { COMFY_MMA_STUB_BODY }
    static __forceinline__ __device__ float get(Acc c, int e) { return static_cast<float>(c[e]); }
};

struct MmaInt4 : MmaInt8 {};

struct MmaBf16 : LdsPad8 {
    using Acc = v8f;
    using Frag = v8bf;
    using Elem = __bf16;
    static constexpr int kFragElems = 8;
    static __forceinline__ __device__ int frag_base(int) { return 0; }
    static __forceinline__ __device__ Acc zero() { return Acc{0, 0, 0, 0, 0, 0, 0, 0}; }
    static __forceinline__ __device__ Acc mma(Frag, Frag, Acc c) { COMFY_MMA_STUB_BODY }
    static __forceinline__ __device__ float get(Acc c, int e) { return c[e]; }
};

struct MmaBf16G : MmaBf16 {
    static constexpr int kStepBytes = 32;
    static __forceinline__ __device__ Frag load(const void*, int, int, int, int) { return Frag{}; }
};

struct MmaF16 : LdsPad8 {
    using Acc = v8f;
    using Frag = v8h;
    using Elem = _Float16;
    static constexpr int kFragElems = 8;
    static constexpr int kStepBytes = 32;
    static __forceinline__ __device__ int frag_base(int) { return 0; }
    static __forceinline__ __device__ Frag load(const void*, int, int, int, int) { return Frag{}; }
    static __forceinline__ __device__ Acc zero() { return Acc{0, 0, 0, 0, 0, 0, 0, 0}; }
    static __forceinline__ __device__ Acc mma(Frag, Frag, Acc c) { COMFY_MMA_STUB_BODY }
    static __forceinline__ __device__ float get(Acc c, int e) { return c[e]; }
};

#undef COMFY_MMA_STUB_BODY

#endif

}  // namespace comfy::hip_backend
