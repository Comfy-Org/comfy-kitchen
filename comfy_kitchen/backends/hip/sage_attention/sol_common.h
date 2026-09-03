// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Layout contract shared by every RDNA Sol-Attn kernel: tile staging, the INT8
// carriers, the block reductions and the fused norm+rope the chunked producer
// applies. The CUDA backend's sage_attention/sol_layout.cuh is the reference for
// what these compute.
//
// One difference runs through the whole port. The CUDA carriers are stored under
// two permutations, sol::perm_d on the contraction axis and sol::perm_key on the
// key axis, whose only job is to make an m16n8k32 lane's two operand words one
// 8-byte load. RDNA's 16x16x16 WMMA hands a lane a contiguous K slice of its own
// row, so natural order is already the fragment order and both permutations are
// dropped: the carriers here are plain [.., D] and [.., key] arrays. Anything
// that reads them from Python (the top-k threshold) must not un-permute.
//
// The scores are evaluated transposed, as in int8_attn.hip: S^T = K @ Q^T puts one
// query on a lane and eight keys in its accumulator, so the online softmax is
// scalar per lane and P^T is already the PV operand.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "../mma.h"
#include "../rope_math.h"
#include "sage_common.h"

namespace comfy::hip_backend::sol {

using namespace comfy::hip_backend::sage;

constexpr int kHeadDim = 128;  // the only head_dim these kernels handle
constexpr int kBlock = 64;     // Sol-Attn's routing granularity, in tokens

// Finite, so kNeg - kNeg == 0 (unlike -inf) and every masked score survives the
// fma and exp2 it passes through before anything tests it. Same value as the CUDA
// sol::NEG, so both backends drop the same blocks.
constexpr float kNeg = -3.0e38f;

// 16-bit row stride of a staged tile; keeps the 16-byte stores aligned.
constexpr int kLdTile = kHeadDim + 8;

// The stage launchers are host code and throw, so a failed async copy is reported
// rather than silently leaving the workspace half-initialised.
inline void sol_check(hipError_t err, const char* what) {
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("sol_attn: ") + what + ": " + hipGetErrorString(err));
    }
}

__forceinline__ __device__ int imin(int a, int b) { return a < b ? a : b; }
__forceinline__ __device__ int imax(int a, int b) { return a > b ? a : b; }

// Valid tokens at the FRONT of 64-block n: the caller's per-block table
// (zero-padded tiles, clamped to [1, rows that exist]) or the plain ragged tail.
// Rows past it are dead.
__forceinline__ __device__ int block_len_of(const int32_t* blen, int n, int T) {
    const int rem = imin(kBlock, T - n * kBlock);
    return blen ? imin(rem, imax(1, blen[n])) : rem;
}

__forceinline__ __device__ int8_t q8(float x, float inv) {
    const int r = static_cast<int>(rintf(x * inv));
    return static_cast<int8_t>(imax(-127, imin(127, r)));
}

// 128-thread reductions, one thread per channel; every thread gets the result and
// `s` is free on return.
__forceinline__ __device__ float block_max128(float x, float* s) {
    const int d = threadIdx.x;
    s[d] = x;
    __syncthreads();
    for (int w = 64; w; w >>= 1) {
        if (d < w) s[d] = fmaxf(s[d], s[d + w]);
        __syncthreads();
    }
    const float r = s[0];
    __syncthreads();
    return r;
}

__forceinline__ __device__ float block_sum128(float x, float* s) {
    const int d = threadIdx.x;
    s[d] = x;
    __syncthreads();
    for (int w = 64; w; w >>= 1) {
        if (d < w) s[d] += s[d + w];
        __syncthreads();
    }
    const float r = s[0];
    __syncthreads();
    return r;
}

// q/k/v/out element type: bf16 or fp16 (_Float16). Everything after the load
// (tiles, scales, int8 carriers) is the same; only the loads and the final store
// convert.
enum SolElem : int { kSolBf16 = 0, kSolFp16 = 1 };

// Stage rows 0..len-1 (row t at src + t * stride, 16-byte loads) into the tile,
// zero past len.
template <typename T>
__forceinline__ __device__ void stage_tile64(T* tile, const T* __restrict__ src,
                                             int64_t stride, int len) {
    for (int idx = threadIdx.x; idx < kBlock * (kHeadDim / 8); idx += kHeadDim) {
        const int t = idx / (kHeadDim / 8), c8 = (idx % (kHeadDim / 8)) * 8;
        uint4 val = make_uint4(0u, 0u, 0u, 0u);
        if (t < len) {
            val = *reinterpret_cast<const uint4*>(src + static_cast<int64_t>(t) * stride + c8);
        }
        *reinterpret_cast<uint4*>(tile + t * kLdTile + c8) = val;
    }
}

// Per-token absmax scale and INT8 row, one thread per token. qi / qs point at
// (this tile's first token, this head). Rows [len, nrows) exist in memory but are
// dead (zero-padded tiles) and get zeros; rows past nrows do not exist.
template <typename T>
__forceinline__ __device__ void quant_q_rows(const T* tile, int len, int nrows,
                                             int8_t* __restrict__ qi, float* __restrict__ qs,
                                             int H) {
    for (int t = threadIdx.x; t < nrows; t += kHeadDim) {
        const T* row = tile + t * kLdTile;  // zero-staged past len
        const bool live = t < len;
        float a = 0.f;
#pragma unroll 8
        for (int d = 0; d < kHeadDim; ++d) a = fmaxf(a, fabsf(static_cast<float>(row[d])));
        const float sc = live ? fmaxf(a / 127.0f, 1e-8f) : 0.f;  // dead rows: deterministic zeros
        qs[static_cast<size_t>(t) * H] = sc;
        const float inv = live ? 1.f / sc : 0.f;
        int8_t* dst = qi + static_cast<size_t>(t) * H * kHeadDim;
        // 16 bytes at a time: a whole 128-byte row in registers spills to scratch.
#pragma unroll
        for (int c = 0; c < kHeadDim; c += 16) {
            __attribute__((aligned(16))) int8_t out[16];
#pragma unroll
            for (int j = 0; j < 16; ++j) out[j] = q8(static_cast<float>(row[c + j]), inv);
            *reinterpret_cast<uint4*>(dst + c) = *reinterpret_cast<const uint4*>(out);
        }
    }
}

// Query-block centroid, quantized like a pseudo-row. One thread per channel;
// returns this thread's channel mean. `sred` holds bytes on return -- sync before
// reusing it.
template <typename T>
__forceinline__ __device__ float centroid_quant(const T* tile, int len, float* sred,
                                                int8_t* __restrict__ cen8,
                                                float* __restrict__ cens) {
    const int d = threadIdx.x;
    float c = 0.f;
    for (int t = 0; t < len; ++t) c += static_cast<float>(tile[t * kLdTile + d]);
    c /= static_cast<float>(len);
    const float csc = fmaxf(block_max128(fabsf(c), sred) / 127.0f, 1e-8f);
    int8_t* s8 = reinterpret_cast<int8_t*>(sred);
    s8[d] = q8(c, 1.f / csc);
    __syncthreads();
    if (d < kHeadDim / 16) {
        reinterpret_cast<uint4*>(cen8)[d] = reinterpret_cast<const uint4*>(s8)[d];
    }
    if (d == 0) *cens = csc;
    return c;
}

// Centred per-key scale and INT8 row. kbias (log2 units, or null) only the exact
// branch reads, so biased blocks must be sink-routed. Dead rows get a zero scale,
// a kNeg bias and zero bytes.
template <typename T>
__forceinline__ __device__ void quant_k_rows(const T* tile, int len,
                                             const float* __restrict__ kmean,
                                             const float* __restrict__ kbias,
                                             int8_t* __restrict__ ki, float2* __restrict__ ksb) {
    for (int p = threadIdx.x; p < kBlock; p += kHeadDim) {
        const bool live = p < len;
        const T* row = tile + p * kLdTile;
        float a = 0.f;
#pragma unroll 8
        for (int d = 0; d < kHeadDim; ++d) {
            a = fmaxf(a, fabsf(static_cast<float>(row[d]) - kmean[d]));
        }
        const float sc = fmaxf(a / 127.0f, 1e-8f);
        const float bias = (kbias && live) ? kbias[p] : 0.f;
        ksb[p] = make_float2(live ? sc : 0.f, live ? bias : kNeg);
        const float inv = 1.f / sc;
#pragma unroll
        for (int c = 0; c < kHeadDim; c += 16) {
            __attribute__((aligned(16))) int8_t out[16];
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                out[j] = live ? q8(static_cast<float>(row[c + j]) - kmean[c + j], inv)
                              : static_cast<int8_t>(0);
            }
            *reinterpret_cast<uint4*>(ki + static_cast<size_t>(p) * kHeadDim + c) =
                *reinterpret_cast<const uint4*>(out);
        }
    }
}

// Fused per-head RMSNorm + split-half RoPE in place, matching ops/rms_rope.hip
// including its bf16 rounding between norm and rotation. One wave per token, lane
// owns channels 4 * lane .. +3.
__forceinline__ __device__ void norm_rope_rows(__bf16* tile, int ld, int len,
                                               const float* __restrict__ fab_t0,
                                               const __bf16* __restrict__ w, float eps, int rot) {
    // fab [T, rot, 2] is per-channel: out[c] = f.x * n[c] + f.y * n[partner(c)]
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    const int nw = static_cast<int>(blockDim.x >> 5), c0 = lane * 4;
    float wreg[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) wreg[i] = static_cast<float>(w[c0 + i]);
    for (int t = wp; t < len; t += nw) {
        __bf16* row = tile + t * ld;
        float x[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) x[i] = static_cast<float>(row[c0 + i]);
        float ss = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3];
#pragma unroll
        for (int off = 16; off; off >>= 1) ss += __shfl_xor(ss, off, 32);
        const float rrms = rsqrtf(ss / static_cast<float>(kHeadDim) + eps);
        float n[4];
        // round_bf16, not a __bf16 round trip: -ffast-math folds the cast away.
#pragma unroll
        for (int i = 0; i < 4; ++i) n[i] = round_bf16(x[i] * rrms * wreg[i]);
        // Explicit source lane: an xor shuffle is only correct for power-of-two
        // offsets and rot/8 need not be one (H3 rot=96 -> 12).
        const int poff = rot >> 3;
        const int src = (c0 < (rot >> 1)) ? lane + poff : (c0 < rot ? lane - poff : lane);
        float p[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) p[i] = __shfl(n[i], src, 32);
        float out[4];
        if (c0 < rot) {
            const float* fr = fab_t0 + static_cast<int64_t>(t) * (rot * 2) + c0 * 2;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 f = *reinterpret_cast<const float2*>(fr + i * 2);
                out[i] = f.x * n[i] + f.y * p[i];
            }
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i) out[i] = n[i];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) row[c0 + i] = static_cast<__bf16>(out[i]);
    }
}

}  // namespace comfy::hip_backend::sol
