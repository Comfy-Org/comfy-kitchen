// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Tile MMA policies for RDNA1 (gfx1010), wave32. gfx1010 has neither matrix cores
// nor dot instructions, so every policy computes the 16x16 tile in vector
// arithmetic.
//
// A policy holds the operand fragment type, the bytes of a K-row one MMA
// consumes (kStepBytes), the LDS read, and the MMA. The tile kernels stay
// byte-addressed; only the policy changes per operand type.
//
// A lane holds the whole K-step of its row (16B fp8/iu8, 8B iu4, 32B f16/bf16).
// Accumulators are v8f/v8i, column lane % 16, and each half-wave owns a
// contiguous block of 8 rows: D[e + 8 * (lane / 16)]. See acc_row.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

#include "fp8_utils.h"

namespace comfy::hip_backend {

typedef int int32_t_v2 __attribute__((ext_vector_type(2)));
typedef int int32_t_v4 __attribute__((ext_vector_type(4)));

typedef int32_t_v2 v2i;
typedef int32_t_v4 v4i;
typedef int v8i __attribute__((ext_vector_type(8)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef __bf16 v16bf __attribute__((ext_vector_type(16)));
typedef _Float16 v16h __attribute__((ext_vector_type(16)));

constexpr int kWave = 32;

// ---------------------------------------------------------------------------
// Fragment addressing
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
// byte. gfx1010 has no dot4 instruction, so this is the arithmetic one performs.
__forceinline__ __device__ int dot4_i8(int a, int b, int c) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        c += static_cast<int>(static_cast<int8_t>((a >> (i * 8)) & 0xFF)) *
             static_cast<int>(static_cast<int8_t>((b >> (i * 8)) & 0xFF));
    }
    return c;
}

__forceinline__ __device__ int frag_row(int lane) { return lane % 16; }

// Row of accumulator element `e` for this lane; the column is lane % 16.
__forceinline__ __device__ int acc_row(int lane, int e) { return e + 8 * (lane / 16); }
__forceinline__ __device__ int acc_col(int lane) { return lane % 16; }

// Bytes of a row that live `stride` apart in LDS. `row` is the lane's row (A)
// or column (B); `kbyte` is the byte offset within that row.
__forceinline__ __device__ v2i load_frag_b64(const void* lds, int row, int kbyte, int stride) {
    const char* p = static_cast<const char*>(lds) + row * stride + kbyte;
    return *reinterpret_cast<const v2i*>(p);
}

__forceinline__ __device__ v4i load_frag_b128(const void* lds, int row, int kbyte, int stride) {
    const char* p = static_cast<const char*>(lds) + row * stride + kbyte;
    // kLdsPad breaks 16-byte alignment, so this must not become a single b128
    // load.
    v4i r;
    const int* q = reinterpret_cast<const int*>(p);
    r[0] = q[0];
    r[1] = q[1];
    r[2] = q[2];
    r[3] = q[3];
    return r;
}

// A K-step of a 16-bit row, straight out of whatever address space `row` points
// at. The elements are contiguous, so this folds into vector loads.
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

// A lane owns one B row/output column and broadcasts the A row needed by each of
// its eight accumulator elements.
template <typename Frag>
__forceinline__ __device__ Frag broadcast_frag(Frag value, int source_lane) {
    Frag result;
#pragma unroll
    for (int i = 0; i < sizeof(Frag) / sizeof(int); ++i) {
        reinterpret_cast<int*>(&result)[i] =
            __shfl(reinterpret_cast<const int*>(&value)[i], source_lane, kWave);
    }
    return result;
}

template <typename Acc, typename Frag, typename Dot>
__forceinline__ __device__ Acc software_mma(Frag a, Frag b, Acc c, Dot dot) {
    const int lane = threadIdx.x % kWave;
    const int half = lane / 16;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int source_lane = acc_row(lane, e) + 16 * half;
        c[e] = dot(broadcast_frag(a, source_lane), b, c[e]);
    }
    return c;
}

struct MmaFp8 {
    using Acc = v8f;
    using Frag = v4i;
    static constexpr int kStepBytes = 16;
    static __forceinline__ __device__ Frag load(const void* lds, int row, int kbyte, int stride,
                                                int) {
        return load_frag_b128(lds, row, kbyte, stride);
    }
    static __forceinline__ __device__ Acc zero() { return Acc{0, 0, 0, 0, 0, 0, 0, 0}; }
    static __forceinline__ __device__ Acc mma(Frag a, Frag b, Acc c) {
        return software_mma(a, b, c, [] __device__(Frag av, Frag bv, float sum) {
            const uint8_t* ap = reinterpret_cast<const uint8_t*>(&av);
            const uint8_t* bp = reinterpret_cast<const uint8_t*>(&bv);
#pragma unroll
            for (int i = 0; i < 16; ++i) sum += fp8_to_float(ap[i]) * fp8_to_float(bp[i]);
            return sum;
        });
    }
    static __forceinline__ __device__ float get(Acc c, int e) { return c[e]; }
};

struct MmaBf8 : MmaFp8 {
    static __forceinline__ __device__ Acc mma(Frag a, Frag b, Acc c) {
        return software_mma(a, b, c, [] __device__(Frag av, Frag bv, float sum) {
            const uint8_t* ap = reinterpret_cast<const uint8_t*>(&av);
            const uint8_t* bp = reinterpret_cast<const uint8_t*>(&bv);
#pragma unroll
            for (int i = 0; i < 16; ++i) sum += bf8_to_float(ap[i]) * bf8_to_float(bp[i]);
            return sum;
        });
    }
};

struct MmaInt8 {
    using Acc = v8i;
    using Frag = v4i;
    static constexpr int kStepBytes = 16;
    static __forceinline__ __device__ Frag load(const void* lds, int row, int kbyte, int stride,
                                                int) {
        return load_frag_b128(lds, row, kbyte, stride);
    }
    static __forceinline__ __device__ Acc zero() { return Acc{0, 0, 0, 0, 0, 0, 0, 0}; }
    // Callers pad their LDS rows by 8 bytes, so the row is only 8-byte aligned here. One b128 read of it is
    // misaligned, which gfx101x's LDS answers wrongly, and only now and then, in WGP
    // mode (LLVM's lds-misaligned-bug); two b64 reads are always naturally aligned.
    static __forceinline__ __device__ Frag load_aligned(const void* lds, int row, int kbyte,
                                                        int stride, int) {
        const v2i lo = load_frag_b64(lds, row, kbyte, stride);
        const v2i hi = load_frag_b64(lds, row, kbyte + 8, stride);
        return Frag{lo[0], lo[1], hi[0], hi[1]};
    }
    static __forceinline__ __device__ Acc mma(Frag a, Frag b, Acc c) {
        return software_mma(a, b, c, [] __device__(Frag av, Frag bv, int sum) {
#pragma unroll
            for (int i = 0; i < 4; ++i) sum = dot4_i8(av[i], bv[i], sum);
            return sum;
        });
    }
    static __forceinline__ __device__ Acc mma_ua(Frag a, Frag b, Acc c) {
        return software_mma(a, b, c, [] __device__(Frag av, Frag bv, int sum) {
            const uint8_t* ap = reinterpret_cast<const uint8_t*>(&av);
            const int8_t* bp = reinterpret_cast<const int8_t*>(&bv);
#pragma unroll
            for (int i = 0; i < 16; ++i) sum += static_cast<int>(ap[i]) * bp[i];
            return sum;
        });
    }
    static __forceinline__ __device__ Acc mma_ub(Frag a, Frag b, Acc c) {
        return software_mma(a, b, c, [] __device__(Frag av, Frag bv, int sum) {
            const int8_t* ap = reinterpret_cast<const int8_t*>(&av);
            const uint8_t* bp = reinterpret_cast<const uint8_t*>(&bv);
#pragma unroll
            for (int i = 0; i < 16; ++i) sum += static_cast<int>(ap[i]) * bp[i];
            return sum;
        });
    }
    static __forceinline__ __device__ float get(Acc c, int e) { return static_cast<float>(c[e]); }
};

struct MmaInt4 {
    using Acc = v8i;
    using Frag = v2i;
    static constexpr int kStepBytes = 8;
    static __forceinline__ __device__ Frag load(const void* lds, int row, int kbyte, int stride,
                                                int) {
        return load_frag_b64(lds, row, kbyte, stride);
    }
    static __forceinline__ __device__ Acc zero() { return Acc{0, 0, 0, 0, 0, 0, 0, 0}; }
    static __forceinline__ __device__ Acc mma(Frag a, Frag b, Acc c) {
        return software_mma(a, b, c, [] __device__(Frag av, Frag bv, int sum) {
            const uint8_t* ap = reinterpret_cast<const uint8_t*>(&av);
            const uint8_t* bp = reinterpret_cast<const uint8_t*>(&bv);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int alo = static_cast<int8_t>(ap[i] << 4) >> 4;
                const int ahi = static_cast<int8_t>(ap[i]) >> 4;
                const int blo = static_cast<int8_t>(bp[i] << 4) >> 4;
                const int bhi = static_cast<int8_t>(bp[i]) >> 4;
                sum += alo * blo + ahi * bhi;
            }
            return sum;
        });
    }
    static __forceinline__ __device__ Acc mma_ua(Frag a, Frag b, Acc c) {
        return software_mma(a, b, c, [] __device__(Frag av, Frag bv, int sum) {
            const uint8_t* ap = reinterpret_cast<const uint8_t*>(&av);
            const uint8_t* bp = reinterpret_cast<const uint8_t*>(&bv);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int blo = static_cast<int8_t>(bp[i] << 4) >> 4;
                const int bhi = static_cast<int8_t>(bp[i]) >> 4;
                sum += (ap[i] & 15) * blo + (ap[i] >> 4) * bhi;
            }
            return sum;
        });
    }
    static __forceinline__ __device__ float get(Acc c, int e) { return static_cast<float>(c[e]); }
};

struct MmaBf16 {
    using Acc = v8f;
    using Frag = v16bf;
    using Elem = __bf16;
    static constexpr int kFragElems = 16;
    static __forceinline__ __device__ int frag_base(int) { return 0; }
    static __forceinline__ __device__ Acc zero() { return Acc{0, 0, 0, 0, 0, 0, 0, 0}; }
    static __forceinline__ __device__ Acc mma(Frag a, Frag b, Acc c) {
        return software_mma(a, b, c, [] __device__(Frag av, Frag bv, float sum) {
#pragma unroll
            for (int i = 0; i < 16; ++i)
                sum += static_cast<float>(av[i]) * static_cast<float>(bv[i]);
            return sum;
        });
    }
    static __forceinline__ __device__ float get(Acc c, int e) { return c[e]; }
};

struct MmaF16 {
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
        return software_mma(a, b, c, [] __device__(Frag av, Frag bv, float sum) {
#pragma unroll
            for (int i = 0; i < 16; ++i)
                sum += static_cast<float>(av[i]) * static_cast<float>(bv[i]);
            return sum;
        });
    }
    static __forceinline__ __device__ float get(Acc c, int e) { return c[e]; }
};

}  // namespace comfy::hip_backend
