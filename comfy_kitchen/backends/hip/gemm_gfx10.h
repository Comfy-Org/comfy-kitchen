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
#include <cstdint>

#include "arch_compat.h"  // __GFX10__
#include "epilogue.h"

namespace comfy::hip_backend {

// The int8 kernels in this file need the gfx10 VALU dot instruction
// v_dot4_i32_i8, which clang exposes as __builtin_amdgcn_sdot4 and rejects
// with "needs target feature dot1-insts" on every other target: gfx11/gfx12
// spell that instruction __builtin_amdgcn_sudot4 instead. They are therefore
// compiled in the gfx103x device pass only (__GFX10__), the one pass that
// runs them -- ops/gemm_int8.hip launches them behind comfy_is_gfx10().
//
// The other passes still need the symbols, because the launcher is host code
// and a fat binary is linked per architecture: a kernel defined in no pass
// leaves the host reference unresolved instead of declining to run. So each
// pass gets a trapping definition, as mma.h does for the WMMA policies.
#if defined(__GFX10__)

// ===========================================================================
// INT8 GEMM (triple-buffered): C[M, N] = A[M, K] @ B[N, K]^T
// Triple buffering hides DRAM load latency by overlapping prefetch with compute.
// With double buffering, the __syncthreads between prefetch and compute serialized
// them, leaving DRAM latency unhidden. Triple buffering issues the load for tile k+1
// BEFORE computing tile k, so the load overlaps with the compute.
// ===========================================================================

template <typename OutT>
__global__ __launch_bounds__(256) void gemm_int8_valu_kernel(
    const int8_t* __restrict__ A, const int8_t* __restrict__ B,
    EpiRowwise epi,
    OutT* __restrict__ C,
    int M, int N, int K, int ldc) {

    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 64;
    constexpr int THREADS = 256;
    constexpr int PAD = 4;
    constexpr int STRIDE = BK + PAD;
    constexpr int A_SIZE = BM * STRIDE;
    constexpr int B_SIZE = BN * STRIDE;
    constexpr int NBUFFERS = 3;

    __shared__ int8_t A_buf[NBUFFERS][A_SIZE];
    __shared__ int8_t B_buf[NBUFFERS][B_SIZE];

    const int tid = threadIdx.x;
    const int trow = tid / 16;
    const int tcol = tid % 16;
    const int orow = trow * 4;
    const int ocol = tcol * 4;

    const int m0 = blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;

    int acc[4][4] = {};
    const int chunks = BK / 16;

    // Helper: load a tile from global to shared memory buffer
    auto load_tile = [&](int buf_idx, int k0) {
        for (int i = tid; i < BM * chunks; i += THREADS) {
            const int r = i / chunks;
            const int c = i % chunks;
            const int gk = k0 + c * 16;
            int4 val = make_int4(0, 0, 0, 0);
            if (m0 + r < M && gk + 15 < K)
                val = *reinterpret_cast<const int4*>(A + (m0 + r) * K + gk);
            *reinterpret_cast<int4*>(&A_buf[buf_idx][r * STRIDE + c * 16]) = val;
        }
        for (int i = tid; i < BN * chunks; i += THREADS) {
            const int r = i / chunks;
            const int c = i % chunks;
            const int gk = k0 + c * 16;
            int4 val = make_int4(0, 0, 0, 0);
            if (n0 + r < N && gk + 15 < K)
                val = *reinterpret_cast<const int4*>(B + (n0 + r) * K + gk);
            *reinterpret_cast<int4*>(&B_buf[buf_idx][r * STRIDE + c * 16]) = val;
        }
    };

    // Helper: compute one k-tile from shared memory
    // Correct pattern: for each b-vector (col), all a-vector fields accumulate
    // into the same output column. The field (x/y/z/w) selects the K-range,
    // not the output column.
    auto compute_tile = [&](int buf_idx) {
        const int8_t* Ac = &A_buf[buf_idx][0];
        const int8_t* Bc = &B_buf[buf_idx][0];
        #pragma unroll
        for (int k = 0; k < BK; k += 16) {
            int4 a0 = *reinterpret_cast<const int4*>(&Ac[(orow + 0) * STRIDE + k]);
            int4 a1 = *reinterpret_cast<const int4*>(&Ac[(orow + 1) * STRIDE + k]);
            int4 a2 = *reinterpret_cast<const int4*>(&Ac[(orow + 2) * STRIDE + k]);
            int4 a3 = *reinterpret_cast<const int4*>(&Ac[(orow + 3) * STRIDE + k]);
            int4 b0 = *reinterpret_cast<const int4*>(&Bc[(ocol + 0) * STRIDE + k]);
            int4 b1 = *reinterpret_cast<const int4*>(&Bc[(ocol + 1) * STRIDE + k]);
            int4 b2 = *reinterpret_cast<const int4*>(&Bc[(ocol + 2) * STRIDE + k]);
            int4 b3 = *reinterpret_cast<const int4*>(&Bc[(ocol + 3) * STRIDE + k]);
            // Col 0: all 4 fields of each a-vector accumulate into col 0
            acc[0][0] = __builtin_amdgcn_sdot4(a0.x, b0.x, acc[0][0], true);
            acc[0][0] = __builtin_amdgcn_sdot4(a0.y, b0.y, acc[0][0], true);
            acc[0][0] = __builtin_amdgcn_sdot4(a0.z, b0.z, acc[0][0], true);
            acc[0][0] = __builtin_amdgcn_sdot4(a0.w, b0.w, acc[0][0], true);
            acc[1][0] = __builtin_amdgcn_sdot4(a1.x, b0.x, acc[1][0], true);
            acc[1][0] = __builtin_amdgcn_sdot4(a1.y, b0.y, acc[1][0], true);
            acc[1][0] = __builtin_amdgcn_sdot4(a1.z, b0.z, acc[1][0], true);
            acc[1][0] = __builtin_amdgcn_sdot4(a1.w, b0.w, acc[1][0], true);
            acc[2][0] = __builtin_amdgcn_sdot4(a2.x, b0.x, acc[2][0], true);
            acc[2][0] = __builtin_amdgcn_sdot4(a2.y, b0.y, acc[2][0], true);
            acc[2][0] = __builtin_amdgcn_sdot4(a2.z, b0.z, acc[2][0], true);
            acc[2][0] = __builtin_amdgcn_sdot4(a2.w, b0.w, acc[2][0], true);
            acc[3][0] = __builtin_amdgcn_sdot4(a3.x, b0.x, acc[3][0], true);
            acc[3][0] = __builtin_amdgcn_sdot4(a3.y, b0.y, acc[3][0], true);
            acc[3][0] = __builtin_amdgcn_sdot4(a3.z, b0.z, acc[3][0], true);
            acc[3][0] = __builtin_amdgcn_sdot4(a3.w, b0.w, acc[3][0], true);
            // Col 1
            acc[0][1] = __builtin_amdgcn_sdot4(a0.x, b1.x, acc[0][1], true);
            acc[0][1] = __builtin_amdgcn_sdot4(a0.y, b1.y, acc[0][1], true);
            acc[0][1] = __builtin_amdgcn_sdot4(a0.z, b1.z, acc[0][1], true);
            acc[0][1] = __builtin_amdgcn_sdot4(a0.w, b1.w, acc[0][1], true);
            acc[1][1] = __builtin_amdgcn_sdot4(a1.x, b1.x, acc[1][1], true);
            acc[1][1] = __builtin_amdgcn_sdot4(a1.y, b1.y, acc[1][1], true);
            acc[1][1] = __builtin_amdgcn_sdot4(a1.z, b1.z, acc[1][1], true);
            acc[1][1] = __builtin_amdgcn_sdot4(a1.w, b1.w, acc[1][1], true);
            acc[2][1] = __builtin_amdgcn_sdot4(a2.x, b1.x, acc[2][1], true);
            acc[2][1] = __builtin_amdgcn_sdot4(a2.y, b1.y, acc[2][1], true);
            acc[2][1] = __builtin_amdgcn_sdot4(a2.z, b1.z, acc[2][1], true);
            acc[2][1] = __builtin_amdgcn_sdot4(a2.w, b1.w, acc[2][1], true);
            acc[3][1] = __builtin_amdgcn_sdot4(a3.x, b1.x, acc[3][1], true);
            acc[3][1] = __builtin_amdgcn_sdot4(a3.y, b1.y, acc[3][1], true);
            acc[3][1] = __builtin_amdgcn_sdot4(a3.z, b1.z, acc[3][1], true);
            acc[3][1] = __builtin_amdgcn_sdot4(a3.w, b1.w, acc[3][1], true);
            // Col 2
            acc[0][2] = __builtin_amdgcn_sdot4(a0.x, b2.x, acc[0][2], true);
            acc[0][2] = __builtin_amdgcn_sdot4(a0.y, b2.y, acc[0][2], true);
            acc[0][2] = __builtin_amdgcn_sdot4(a0.z, b2.z, acc[0][2], true);
            acc[0][2] = __builtin_amdgcn_sdot4(a0.w, b2.w, acc[0][2], true);
            acc[1][2] = __builtin_amdgcn_sdot4(a1.x, b2.x, acc[1][2], true);
            acc[1][2] = __builtin_amdgcn_sdot4(a1.y, b2.y, acc[1][2], true);
            acc[1][2] = __builtin_amdgcn_sdot4(a1.z, b2.z, acc[1][2], true);
            acc[1][2] = __builtin_amdgcn_sdot4(a1.w, b2.w, acc[1][2], true);
            acc[2][2] = __builtin_amdgcn_sdot4(a2.x, b2.x, acc[2][2], true);
            acc[2][2] = __builtin_amdgcn_sdot4(a2.y, b2.y, acc[2][2], true);
            acc[2][2] = __builtin_amdgcn_sdot4(a2.z, b2.z, acc[2][2], true);
            acc[2][2] = __builtin_amdgcn_sdot4(a2.w, b2.w, acc[2][2], true);
            acc[3][2] = __builtin_amdgcn_sdot4(a3.x, b2.x, acc[3][2], true);
            acc[3][2] = __builtin_amdgcn_sdot4(a3.y, b2.y, acc[3][2], true);
            acc[3][2] = __builtin_amdgcn_sdot4(a3.z, b2.z, acc[3][2], true);
            acc[3][2] = __builtin_amdgcn_sdot4(a3.w, b2.w, acc[3][2], true);
            // Col 3
            acc[0][3] = __builtin_amdgcn_sdot4(a0.x, b3.x, acc[0][3], true);
            acc[0][3] = __builtin_amdgcn_sdot4(a0.y, b3.y, acc[0][3], true);
            acc[0][3] = __builtin_amdgcn_sdot4(a0.z, b3.z, acc[0][3], true);
            acc[0][3] = __builtin_amdgcn_sdot4(a0.w, b3.w, acc[0][3], true);
            acc[1][3] = __builtin_amdgcn_sdot4(a1.x, b3.x, acc[1][3], true);
            acc[1][3] = __builtin_amdgcn_sdot4(a1.y, b3.y, acc[1][3], true);
            acc[1][3] = __builtin_amdgcn_sdot4(a1.z, b3.z, acc[1][3], true);
            acc[1][3] = __builtin_amdgcn_sdot4(a1.w, b3.w, acc[1][3], true);
            acc[2][3] = __builtin_amdgcn_sdot4(a2.x, b3.x, acc[2][3], true);
            acc[2][3] = __builtin_amdgcn_sdot4(a2.y, b3.y, acc[2][3], true);
            acc[2][3] = __builtin_amdgcn_sdot4(a2.z, b3.z, acc[2][3], true);
            acc[2][3] = __builtin_amdgcn_sdot4(a2.w, b3.w, acc[2][3], true);
            acc[3][3] = __builtin_amdgcn_sdot4(a3.x, b3.x, acc[3][3], true);
            acc[3][3] = __builtin_amdgcn_sdot4(a3.y, b3.y, acc[3][3], true);
            acc[3][3] = __builtin_amdgcn_sdot4(a3.z, b3.z, acc[3][3], true);
            acc[3][3] = __builtin_amdgcn_sdot4(a3.w, b3.w, acc[3][3], true);
        }
    };

    // Prologue: load first 2 tiles
    load_tile(0, 0);
    if (BK < K) load_tile(1, BK);
    __syncthreads();

    int buf = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
        const int next_buf = (buf + 1) % NBUFFERS;
        const int prefetch_buf = (buf + 2) % NBUFFERS;
        const int kn = k0 + BK;
        const int kprefetch = k0 + 2 * BK;

        // Issue prefetch for tile k+2 (overlaps with compute of tile k)
        if (kprefetch < K) {
            load_tile(prefetch_buf, kprefetch);
        }

        // Compute current tile (overlaps with prefetch of tile k+2)
        compute_tile(buf);

        __syncthreads();
        buf = next_buf;
    }

    // Epilogue: write 4x4 outputs with scaling and bias
    for (int ri = 0; ri < 4; ri++) {
        const int row = m0 + orow + ri;
        if (row >= M) continue;
        for (int ci = 0; ci < 4; ci++) {
            const int col = n0 + ocol + ci;
            if (col >= N) continue;
            C[row * ldc + col] = static_cast<OutT>(epi(row, col, static_cast<float>(acc[ri][ci])));
        }
    }
}

#else  // !__GFX10__

// No v_dot4_i32_i8 on this target. comfy_is_gfx10() keeps the launch off these
// devices; trap rather than return a plausible wrong answer if it ever runs.
template <typename OutT>
__global__ __launch_bounds__(256) void gemm_int8_valu_kernel(
    const int8_t*, const int8_t*, EpiRowwise, OutT*, int, int, int, int) {
    __builtin_trap();
}

#endif  // __GFX10__

// ===========================================================================
// FP16 GEMM: C[M, N] = A[M, K] @ B[N, K]^T
// A is fp16, B is fp16, C is fp16/bf16
// ===========================================================================

template <typename OutT>
__global__ __launch_bounds__(256) void gemm_fp16_valu_kernel(
    const __half* __restrict__ A, const __half* __restrict__ B,
    EpiFp16 epi,
    OutT* __restrict__ C,
    int M, int N, int K, int ldc) {

    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 64;
    constexpr int THREADS = 256;
    // PAD=4: stride=136 bytes=34 words. gcd(34,32)=2 → 4-way conflicts,
    // but 50% aligned (PAD=2 gave 2-way conflicts but 75% misaligned;
    // misalignment penalty on RDNA2 ds_read_b128 dominates).
    constexpr int PAD = 4;
    constexpr int STRIDE = BK + PAD;
    constexpr int A_SIZE = BM * STRIDE;
    constexpr int B_SIZE = BN * STRIDE;

    typedef _Float16 v2h __attribute__((ext_vector_type(2)));
    typedef _Float16 v8h __attribute__((ext_vector_type(8)));

    __shared__ __half A_buf[2][A_SIZE];
    __shared__ __half B_buf[2][B_SIZE];

    const int tid = threadIdx.x;

    // Each thread computes a 4x4 output block
    const int trow = tid / 16;  // 0..15
    const int tcol = tid % 16;  // 0..15
    const int orow = trow * 4;  // 0,4,8,...,60
    const int ocol = tcol * 4;  // 0,4,8,...,60

    const int m0 = blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;

    float acc[4][4] = {};

    // Load first tile into buffer[0]
    {
        const int chunks = BK / 8;  // 8 halves per int4
        for (int i = tid; i < BM * chunks; i += THREADS) {
            const int r = i / chunks;
            const int c = i % chunks;
            const int gk = c * 8;
            int4 val = make_int4(0, 0, 0, 0);
            if (m0 + r < M && gk + 7 < K)
                val = *reinterpret_cast<const int4*>(A + (m0 + r) * K + gk);
            *reinterpret_cast<int4*>(&A_buf[0][r * STRIDE + c * 8]) = val;
        }
        for (int i = tid; i < BN * chunks; i += THREADS) {
            const int r = i / chunks;
            const int c = i % chunks;
            const int gk = c * 8;
            int4 val = make_int4(0, 0, 0, 0);
            if (n0 + r < N && gk + 7 < K)
                val = *reinterpret_cast<const int4*>(B + (n0 + r) * K + gk);
            *reinterpret_cast<int4*>(&B_buf[0][r * STRIDE + c * 8]) = val;
        }
    }

    int buf = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
        const int nb = 1 - buf;
        const int kn = k0 + BK;

        // Prefetch next tile
        if (kn < K) {
            const int chunks = BK / 8;
            for (int i = tid; i < BM * chunks; i += THREADS) {
                const int r = i / chunks;
                const int c = i % chunks;
                const int gk = kn + c * 8;
                int4 val = make_int4(0, 0, 0, 0);
                if (m0 + r < M && gk + 7 < K)
                    val = *reinterpret_cast<const int4*>(A + (m0 + r) * K + gk);
                *reinterpret_cast<int4*>(&A_buf[nb][r * STRIDE + c * 8]) = val;
            }
            for (int i = tid; i < BN * chunks; i += THREADS) {
                const int r = i / chunks;
                const int c = i % chunks;
                const int gk = kn + c * 8;
                int4 val = make_int4(0, 0, 0, 0);
                if (n0 + r < N && gk + 7 < K)
                    val = *reinterpret_cast<const int4*>(B + (n0 + r) * K + gk);
                *reinterpret_cast<int4*>(&B_buf[nb][r * STRIDE + c * 8]) = val;
            }
        }

        __syncthreads();

        // Compute: 4x4 outputs, 16-byte loads (8 halves), 4 fdot2 per pair
        const __half* Ac = &A_buf[buf][0];
        const __half* Bc = &B_buf[buf][0];
        #pragma unroll
        for (int k = 0; k < BK; k += 8) {
            int4 a0 = *reinterpret_cast<const int4*>(&Ac[(orow + 0) * STRIDE + k]);
            int4 a1 = *reinterpret_cast<const int4*>(&Ac[(orow + 1) * STRIDE + k]);
            int4 a2 = *reinterpret_cast<const int4*>(&Ac[(orow + 2) * STRIDE + k]);
            int4 a3 = *reinterpret_cast<const int4*>(&Ac[(orow + 3) * STRIDE + k]);
            int4 b0 = *reinterpret_cast<const int4*>(&Bc[(ocol + 0) * STRIDE + k]);
            int4 b1 = *reinterpret_cast<const int4*>(&Bc[(ocol + 1) * STRIDE + k]);
            int4 b2 = *reinterpret_cast<const int4*>(&Bc[(ocol + 2) * STRIDE + k]);
            int4 b3 = *reinterpret_cast<const int4*>(&Bc[(ocol + 3) * STRIDE + k]);
            // 4 fdot2 per a-b pair (2 halves each)
            acc[0][0] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a0.x), *reinterpret_cast<const v2h*>(&b0.x), acc[0][0], true);
            acc[0][0] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a0.y), *reinterpret_cast<const v2h*>(&b0.y), acc[0][0], true);
            acc[0][0] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a0.z), *reinterpret_cast<const v2h*>(&b0.z), acc[0][0], true);
            acc[0][0] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a0.w), *reinterpret_cast<const v2h*>(&b0.w), acc[0][0], true);
            acc[0][1] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a0.x), *reinterpret_cast<const v2h*>(&b1.x), acc[0][1], true);
            acc[0][1] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a0.y), *reinterpret_cast<const v2h*>(&b1.y), acc[0][1], true);
            acc[0][1] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a0.z), *reinterpret_cast<const v2h*>(&b1.z), acc[0][1], true);
            acc[0][1] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a0.w), *reinterpret_cast<const v2h*>(&b1.w), acc[0][1], true);
            acc[0][2] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a0.x), *reinterpret_cast<const v2h*>(&b2.x), acc[0][2], true);
            acc[0][2] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a0.y), *reinterpret_cast<const v2h*>(&b2.y), acc[0][2], true);
            acc[0][2] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a0.z), *reinterpret_cast<const v2h*>(&b2.z), acc[0][2], true);
            acc[0][2] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a0.w), *reinterpret_cast<const v2h*>(&b2.w), acc[0][2], true);
            acc[0][3] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a0.x), *reinterpret_cast<const v2h*>(&b3.x), acc[0][3], true);
            acc[0][3] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a0.y), *reinterpret_cast<const v2h*>(&b3.y), acc[0][3], true);
            acc[0][3] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a0.z), *reinterpret_cast<const v2h*>(&b3.z), acc[0][3], true);
            acc[0][3] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a0.w), *reinterpret_cast<const v2h*>(&b3.w), acc[0][3], true);
            acc[1][0] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a1.x), *reinterpret_cast<const v2h*>(&b0.x), acc[1][0], true);
            acc[1][0] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a1.y), *reinterpret_cast<const v2h*>(&b0.y), acc[1][0], true);
            acc[1][0] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a1.z), *reinterpret_cast<const v2h*>(&b0.z), acc[1][0], true);
            acc[1][0] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a1.w), *reinterpret_cast<const v2h*>(&b0.w), acc[1][0], true);
            acc[1][1] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a1.x), *reinterpret_cast<const v2h*>(&b1.x), acc[1][1], true);
            acc[1][1] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a1.y), *reinterpret_cast<const v2h*>(&b1.y), acc[1][1], true);
            acc[1][1] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a1.z), *reinterpret_cast<const v2h*>(&b1.z), acc[1][1], true);
            acc[1][1] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a1.w), *reinterpret_cast<const v2h*>(&b1.w), acc[1][1], true);
            acc[1][2] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a1.x), *reinterpret_cast<const v2h*>(&b2.x), acc[1][2], true);
            acc[1][2] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a1.y), *reinterpret_cast<const v2h*>(&b2.y), acc[1][2], true);
            acc[1][2] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a1.z), *reinterpret_cast<const v2h*>(&b2.z), acc[1][2], true);
            acc[1][2] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a1.w), *reinterpret_cast<const v2h*>(&b2.w), acc[1][2], true);
            acc[1][3] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a1.x), *reinterpret_cast<const v2h*>(&b3.x), acc[1][3], true);
            acc[1][3] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a1.y), *reinterpret_cast<const v2h*>(&b3.y), acc[1][3], true);
            acc[1][3] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a1.z), *reinterpret_cast<const v2h*>(&b3.z), acc[1][3], true);
            acc[1][3] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a1.w), *reinterpret_cast<const v2h*>(&b3.w), acc[1][3], true);
            acc[2][0] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a2.x), *reinterpret_cast<const v2h*>(&b0.x), acc[2][0], true);
            acc[2][0] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a2.y), *reinterpret_cast<const v2h*>(&b0.y), acc[2][0], true);
            acc[2][0] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a2.z), *reinterpret_cast<const v2h*>(&b0.z), acc[2][0], true);
            acc[2][0] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a2.w), *reinterpret_cast<const v2h*>(&b0.w), acc[2][0], true);
            acc[2][1] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a2.x), *reinterpret_cast<const v2h*>(&b1.x), acc[2][1], true);
            acc[2][1] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a2.y), *reinterpret_cast<const v2h*>(&b1.y), acc[2][1], true);
            acc[2][1] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a2.z), *reinterpret_cast<const v2h*>(&b1.z), acc[2][1], true);
            acc[2][1] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a2.w), *reinterpret_cast<const v2h*>(&b1.w), acc[2][1], true);
            acc[2][2] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a2.x), *reinterpret_cast<const v2h*>(&b2.x), acc[2][2], true);
            acc[2][2] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a2.y), *reinterpret_cast<const v2h*>(&b2.y), acc[2][2], true);
            acc[2][2] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a2.z), *reinterpret_cast<const v2h*>(&b2.z), acc[2][2], true);
            acc[2][2] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a2.w), *reinterpret_cast<const v2h*>(&b2.w), acc[2][2], true);
            acc[2][3] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a2.x), *reinterpret_cast<const v2h*>(&b3.x), acc[2][3], true);
            acc[2][3] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a2.y), *reinterpret_cast<const v2h*>(&b3.y), acc[2][3], true);
            acc[2][3] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a2.z), *reinterpret_cast<const v2h*>(&b3.z), acc[2][3], true);
            acc[2][3] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a2.w), *reinterpret_cast<const v2h*>(&b3.w), acc[2][3], true);
            acc[3][0] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a3.x), *reinterpret_cast<const v2h*>(&b0.x), acc[3][0], true);
            acc[3][0] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a3.y), *reinterpret_cast<const v2h*>(&b0.y), acc[3][0], true);
            acc[3][0] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a3.z), *reinterpret_cast<const v2h*>(&b0.z), acc[3][0], true);
            acc[3][0] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a3.w), *reinterpret_cast<const v2h*>(&b0.w), acc[3][0], true);
            acc[3][1] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a3.x), *reinterpret_cast<const v2h*>(&b1.x), acc[3][1], true);
            acc[3][1] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a3.y), *reinterpret_cast<const v2h*>(&b1.y), acc[3][1], true);
            acc[3][1] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a3.z), *reinterpret_cast<const v2h*>(&b1.z), acc[3][1], true);
            acc[3][1] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a3.w), *reinterpret_cast<const v2h*>(&b1.w), acc[3][1], true);
            acc[3][2] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a3.x), *reinterpret_cast<const v2h*>(&b2.x), acc[3][2], true);
            acc[3][2] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a3.y), *reinterpret_cast<const v2h*>(&b2.y), acc[3][2], true);
            acc[3][2] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a3.z), *reinterpret_cast<const v2h*>(&b2.z), acc[3][2], true);
            acc[3][2] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a3.w), *reinterpret_cast<const v2h*>(&b2.w), acc[3][2], true);
            acc[3][3] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a3.x), *reinterpret_cast<const v2h*>(&b3.x), acc[3][3], true);
            acc[3][3] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a3.y), *reinterpret_cast<const v2h*>(&b3.y), acc[3][3], true);
            acc[3][3] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a3.z), *reinterpret_cast<const v2h*>(&b3.z), acc[3][3], true);
            acc[3][3] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a3.w), *reinterpret_cast<const v2h*>(&b3.w), acc[3][3], true);
        }

        __syncthreads();
        buf = nb;
    }

    // Epilogue: write 4x4 outputs with bias/residual
    for (int ri = 0; ri < 4; ri++) {
        const int row = m0 + orow + ri;
        if (row >= M) continue;
        for (int ci = 0; ci < 4; ci++) {
            const int col = n0 + ocol + ci;
            if (col >= N) continue;
            C[row * ldc + col] = static_cast<OutT>(epi(row, col, acc[ri][ci]));
        }
    }
}

// ===========================================================================
// Persistent INT8 GEMM (triple-buffered): 6 fixed blocks process tiles in a loop
// Triple buffering hides DRAM latency by overlapping prefetch with compute.
// ===========================================================================

#if defined(__GFX10__)

template <typename OutT>
__global__ __launch_bounds__(256) void gemm_int8_persistent_kernel(
    const int8_t* __restrict__ A, const int8_t* __restrict__ B,
    EpiRowwise epi,
    OutT* __restrict__ C,
    int M, int N, int K, int ldc) {

    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 64;
    constexpr int THREADS = 256;
    constexpr int PAD = 4;
    constexpr int STRIDE = BK + PAD;
    constexpr int A_SIZE = BM * STRIDE;
    constexpr int B_SIZE = BN * STRIDE;
    constexpr int NBUFFERS = 3;

    __shared__ int8_t A_buf[NBUFFERS][A_SIZE];
    __shared__ int8_t B_buf[NBUFFERS][B_SIZE];

    const int tid = threadIdx.x;
    const int trow = tid / 16;
    const int tcol = tid % 16;
    const int orow = trow * 4;
    const int ocol = tcol * 4;

    const int num_n_tiles = (N + BN - 1) / BN;
    const int num_m_tiles = (M + BM - 1) / BM;
    const int total_tiles = num_m_tiles * num_n_tiles;
    const int chunks = BK / 16;

    for (int tile = blockIdx.x; tile < total_tiles; tile += gridDim.x) {
        const int n0 = (tile % num_n_tiles) * BN;
        const int m0 = (tile / num_n_tiles) * BM;

        int acc[4][4] = {};

        // Helper: load a tile
        auto load_tile = [&](int buf_idx, int k0) {
            for (int i = tid; i < BM * chunks; i += THREADS) {
                const int r = i / chunks;
                const int c = i % chunks;
                const int gk = k0 + c * 16;
                int4 val = make_int4(0, 0, 0, 0);
                if (m0 + r < M && gk + 15 < K)
                    val = *reinterpret_cast<const int4*>(A + (m0 + r) * K + gk);
                *reinterpret_cast<int4*>(&A_buf[buf_idx][r * STRIDE + c * 16]) = val;
            }
            for (int i = tid; i < BN * chunks; i += THREADS) {
                const int r = i / chunks;
                const int c = i % chunks;
                const int gk = k0 + c * 16;
                int4 val = make_int4(0, 0, 0, 0);
                if (n0 + r < N && gk + 15 < K)
                    val = *reinterpret_cast<const int4*>(B + (n0 + r) * K + gk);
                *reinterpret_cast<int4*>(&B_buf[buf_idx][r * STRIDE + c * 16]) = val;
            }
        };

        // Helper: compute one k-tile
        auto compute_tile = [&](int buf_idx) {
            const int8_t* Ac = &A_buf[buf_idx][0];
            const int8_t* Bc = &B_buf[buf_idx][0];
            #pragma unroll
            for (int k = 0; k < BK; k += 16) {
                int4 a0 = *reinterpret_cast<const int4*>(&Ac[(orow + 0) * STRIDE + k]);
                int4 a1 = *reinterpret_cast<const int4*>(&Ac[(orow + 1) * STRIDE + k]);
                int4 a2 = *reinterpret_cast<const int4*>(&Ac[(orow + 2) * STRIDE + k]);
                int4 a3 = *reinterpret_cast<const int4*>(&Ac[(orow + 3) * STRIDE + k]);
                int4 b0 = *reinterpret_cast<const int4*>(&Bc[(ocol + 0) * STRIDE + k]);
                int4 b1 = *reinterpret_cast<const int4*>(&Bc[(ocol + 1) * STRIDE + k]);
                int4 b2 = *reinterpret_cast<const int4*>(&Bc[(ocol + 2) * STRIDE + k]);
                int4 b3 = *reinterpret_cast<const int4*>(&Bc[(ocol + 3) * STRIDE + k]);
                // Col 0
                acc[0][0] = __builtin_amdgcn_sdot4(a0.x, b0.x, acc[0][0], true);
                acc[0][0] = __builtin_amdgcn_sdot4(a0.y, b0.y, acc[0][0], true);
                acc[0][0] = __builtin_amdgcn_sdot4(a0.z, b0.z, acc[0][0], true);
                acc[0][0] = __builtin_amdgcn_sdot4(a0.w, b0.w, acc[0][0], true);
                acc[1][0] = __builtin_amdgcn_sdot4(a1.x, b0.x, acc[1][0], true);
                acc[1][0] = __builtin_amdgcn_sdot4(a1.y, b0.y, acc[1][0], true);
                acc[1][0] = __builtin_amdgcn_sdot4(a1.z, b0.z, acc[1][0], true);
                acc[1][0] = __builtin_amdgcn_sdot4(a1.w, b0.w, acc[1][0], true);
                acc[2][0] = __builtin_amdgcn_sdot4(a2.x, b0.x, acc[2][0], true);
                acc[2][0] = __builtin_amdgcn_sdot4(a2.y, b0.y, acc[2][0], true);
                acc[2][0] = __builtin_amdgcn_sdot4(a2.z, b0.z, acc[2][0], true);
                acc[2][0] = __builtin_amdgcn_sdot4(a2.w, b0.w, acc[2][0], true);
                acc[3][0] = __builtin_amdgcn_sdot4(a3.x, b0.x, acc[3][0], true);
                acc[3][0] = __builtin_amdgcn_sdot4(a3.y, b0.y, acc[3][0], true);
                acc[3][0] = __builtin_amdgcn_sdot4(a3.z, b0.z, acc[3][0], true);
                acc[3][0] = __builtin_amdgcn_sdot4(a3.w, b0.w, acc[3][0], true);
                // Col 1
                acc[0][1] = __builtin_amdgcn_sdot4(a0.x, b1.x, acc[0][1], true);
                acc[0][1] = __builtin_amdgcn_sdot4(a0.y, b1.y, acc[0][1], true);
                acc[0][1] = __builtin_amdgcn_sdot4(a0.z, b1.z, acc[0][1], true);
                acc[0][1] = __builtin_amdgcn_sdot4(a0.w, b1.w, acc[0][1], true);
                acc[1][1] = __builtin_amdgcn_sdot4(a1.x, b1.x, acc[1][1], true);
                acc[1][1] = __builtin_amdgcn_sdot4(a1.y, b1.y, acc[1][1], true);
                acc[1][1] = __builtin_amdgcn_sdot4(a1.z, b1.z, acc[1][1], true);
                acc[1][1] = __builtin_amdgcn_sdot4(a1.w, b1.w, acc[1][1], true);
                acc[2][1] = __builtin_amdgcn_sdot4(a2.x, b1.x, acc[2][1], true);
                acc[2][1] = __builtin_amdgcn_sdot4(a2.y, b1.y, acc[2][1], true);
                acc[2][1] = __builtin_amdgcn_sdot4(a2.z, b1.z, acc[2][1], true);
                acc[2][1] = __builtin_amdgcn_sdot4(a2.w, b1.w, acc[2][1], true);
                acc[3][1] = __builtin_amdgcn_sdot4(a3.x, b1.x, acc[3][1], true);
                acc[3][1] = __builtin_amdgcn_sdot4(a3.y, b1.y, acc[3][1], true);
                acc[3][1] = __builtin_amdgcn_sdot4(a3.z, b1.z, acc[3][1], true);
                acc[3][1] = __builtin_amdgcn_sdot4(a3.w, b1.w, acc[3][1], true);
                // Col 2
                acc[0][2] = __builtin_amdgcn_sdot4(a0.x, b2.x, acc[0][2], true);
                acc[0][2] = __builtin_amdgcn_sdot4(a0.y, b2.y, acc[0][2], true);
                acc[0][2] = __builtin_amdgcn_sdot4(a0.z, b2.z, acc[0][2], true);
                acc[0][2] = __builtin_amdgcn_sdot4(a0.w, b2.w, acc[0][2], true);
                acc[1][2] = __builtin_amdgcn_sdot4(a1.x, b2.x, acc[1][2], true);
                acc[1][2] = __builtin_amdgcn_sdot4(a1.y, b2.y, acc[1][2], true);
                acc[1][2] = __builtin_amdgcn_sdot4(a1.z, b2.z, acc[1][2], true);
                acc[1][2] = __builtin_amdgcn_sdot4(a1.w, b2.w, acc[1][2], true);
                acc[2][2] = __builtin_amdgcn_sdot4(a2.x, b2.x, acc[2][2], true);
                acc[2][2] = __builtin_amdgcn_sdot4(a2.y, b2.y, acc[2][2], true);
                acc[2][2] = __builtin_amdgcn_sdot4(a2.z, b2.z, acc[2][2], true);
                acc[2][2] = __builtin_amdgcn_sdot4(a2.w, b2.w, acc[2][2], true);
                acc[3][2] = __builtin_amdgcn_sdot4(a3.x, b2.x, acc[3][2], true);
                acc[3][2] = __builtin_amdgcn_sdot4(a3.y, b2.y, acc[3][2], true);
                acc[3][2] = __builtin_amdgcn_sdot4(a3.z, b2.z, acc[3][2], true);
                acc[3][2] = __builtin_amdgcn_sdot4(a3.w, b2.w, acc[3][2], true);
                // Col 3
                acc[0][3] = __builtin_amdgcn_sdot4(a0.x, b3.x, acc[0][3], true);
                acc[0][3] = __builtin_amdgcn_sdot4(a0.y, b3.y, acc[0][3], true);
                acc[0][3] = __builtin_amdgcn_sdot4(a0.z, b3.z, acc[0][3], true);
                acc[0][3] = __builtin_amdgcn_sdot4(a0.w, b3.w, acc[0][3], true);
                acc[1][3] = __builtin_amdgcn_sdot4(a1.x, b3.x, acc[1][3], true);
                acc[1][3] = __builtin_amdgcn_sdot4(a1.y, b3.y, acc[1][3], true);
                acc[1][3] = __builtin_amdgcn_sdot4(a1.z, b3.z, acc[1][3], true);
                acc[1][3] = __builtin_amdgcn_sdot4(a1.w, b3.w, acc[1][3], true);
                acc[2][3] = __builtin_amdgcn_sdot4(a2.x, b3.x, acc[2][3], true);
                acc[2][3] = __builtin_amdgcn_sdot4(a2.y, b3.y, acc[2][3], true);
                acc[2][3] = __builtin_amdgcn_sdot4(a2.z, b3.z, acc[2][3], true);
                acc[2][3] = __builtin_amdgcn_sdot4(a2.w, b3.w, acc[2][3], true);
                acc[3][3] = __builtin_amdgcn_sdot4(a3.x, b3.x, acc[3][3], true);
                acc[3][3] = __builtin_amdgcn_sdot4(a3.y, b3.y, acc[3][3], true);
                acc[3][3] = __builtin_amdgcn_sdot4(a3.z, b3.z, acc[3][3], true);
                acc[3][3] = __builtin_amdgcn_sdot4(a3.w, b3.w, acc[3][3], true);
            }
        };

        // Prologue: load first 2 tiles
        load_tile(0, 0);
        if (BK < K) load_tile(1, BK);
        __syncthreads();

        int buf = 0;
        for (int k0 = 0; k0 < K; k0 += BK) {
            const int next_buf = (buf + 1) % NBUFFERS;
            const int prefetch_buf = (buf + 2) % NBUFFERS;
            const int kprefetch = k0 + 2 * BK;

            // Issue prefetch for tile k+2 (overlaps with compute of tile k)
            if (kprefetch < K) {
                load_tile(prefetch_buf, kprefetch);
            }

            // Compute current tile
            compute_tile(buf);

            __syncthreads();
            buf = next_buf;
        }

        // Epilogue: write 4x4 outputs with scaling and bias
        for (int ri = 0; ri < 4; ri++) {
            const int row = m0 + orow + ri;
            if (row >= M) continue;
            for (int ci = 0; ci < 4; ci++) {
                const int col = n0 + ocol + ci;
                if (col >= N) continue;
                C[row * ldc + col] = static_cast<OutT>(epi(row, col, static_cast<float>(acc[ri][ci])));
            }
        }
    }
}

#else  // !__GFX10__

// See the note at the top of this file: the launcher is host code, so every
// device pass has to define the symbol even where the kernel cannot run.
template <typename OutT>
__global__ __launch_bounds__(256) void gemm_int8_persistent_kernel(
    const int8_t*, const int8_t*, EpiRowwise, OutT*, int, int, int, int) {
    __builtin_trap();
}

#endif  // __GFX10__

// ===========================================================================
// Persistent FP16 GEMM: 6 fixed blocks process tiles in a loop
// Eliminates per-tile block scheduling and shared memory init overhead.
// ===========================================================================

template <typename OutT>
__global__ __launch_bounds__(256) void gemm_fp16_persistent_kernel(
    const __half* __restrict__ A, const __half* __restrict__ B,
    EpiFp16 epi,
    OutT* __restrict__ C,
    int M, int N, int K, int ldc) {

    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 64;
    constexpr int THREADS = 256;
    constexpr int PAD = 4;
    constexpr int STRIDE = BK + PAD;
    constexpr int A_SIZE = BM * STRIDE;
    constexpr int B_SIZE = BN * STRIDE;

    typedef _Float16 v2h __attribute__((ext_vector_type(2)));

    __shared__ __half A_buf[2][A_SIZE];
    __shared__ __half B_buf[2][B_SIZE];

    const int tid = threadIdx.x;
    const int trow = tid / 16;
    const int tcol = tid % 16;
    const int orow = trow * 4;
    const int ocol = tcol * 4;

    const int num_n_tiles = (N + BN - 1) / BN;
    const int num_m_tiles = (M + BM - 1) / BM;
    const int total_tiles = num_m_tiles * num_n_tiles;

    for (int tile = blockIdx.x; tile < total_tiles; tile += gridDim.x) {
        const int n0 = (tile % num_n_tiles) * BN;
        const int m0 = (tile / num_n_tiles) * BM;

        float acc[4][4] = {};
        int buf = 0;

        // Load first k-tile into buf[0]
        {
            const int chunks = BK / 8;
            for (int i = tid; i < BM * chunks; i += THREADS) {
                const int r = i / chunks;
                const int c = i % chunks;
                const int gk = c * 8;
                int4 val = make_int4(0, 0, 0, 0);
                if (m0 + r < M && gk + 7 < K)
                    val = *reinterpret_cast<const int4*>(A + (m0 + r) * K + gk);
                *reinterpret_cast<int4*>(&A_buf[0][r * STRIDE + c * 8]) = val;
            }
            for (int i = tid; i < BN * chunks; i += THREADS) {
                const int r = i / chunks;
                const int c = i % chunks;
                const int gk = c * 8;
                int4 val = make_int4(0, 0, 0, 0);
                if (n0 + r < N && gk + 7 < K)
                    val = *reinterpret_cast<const int4*>(B + (n0 + r) * K + gk);
                *reinterpret_cast<int4*>(&B_buf[0][r * STRIDE + c * 8]) = val;
            }
        }

        for (int k0 = 0; k0 < K; k0 += BK) {
            const int nb = 1 - buf;
            const int kn = k0 + BK;

            if (kn < K) {
                const int chunks = BK / 8;
                for (int i = tid; i < BM * chunks; i += THREADS) {
                    const int r = i / chunks;
                    const int c = i % chunks;
                    const int gk = kn + c * 8;
                    int4 val = make_int4(0, 0, 0, 0);
                    if (m0 + r < M && gk + 7 < K)
                        val = *reinterpret_cast<const int4*>(A + (m0 + r) * K + gk);
                    *reinterpret_cast<int4*>(&A_buf[nb][r * STRIDE + c * 8]) = val;
                }
                for (int i = tid; i < BN * chunks; i += THREADS) {
                    const int r = i / chunks;
                    const int c = i % chunks;
                    const int gk = kn + c * 8;
                    int4 val = make_int4(0, 0, 0, 0);
                    if (n0 + r < N && gk + 7 < K)
                        val = *reinterpret_cast<const int4*>(B + (n0 + r) * K + gk);
                    *reinterpret_cast<int4*>(&B_buf[nb][r * STRIDE + c * 8]) = val;
                }
            }

            __syncthreads();

            const __half* Ac = &A_buf[buf][0];
            const __half* Bc = &B_buf[buf][0];
            #pragma unroll
            for (int k = 0; k < BK; k += 8) {
                int4 a0 = *reinterpret_cast<const int4*>(&Ac[(orow + 0) * STRIDE + k]);
                int4 a1 = *reinterpret_cast<const int4*>(&Ac[(orow + 1) * STRIDE + k]);
                int4 a2 = *reinterpret_cast<const int4*>(&Ac[(orow + 2) * STRIDE + k]);
                int4 a3 = *reinterpret_cast<const int4*>(&Ac[(orow + 3) * STRIDE + k]);
                int4 b0 = *reinterpret_cast<const int4*>(&Bc[(ocol + 0) * STRIDE + k]);
                int4 b1 = *reinterpret_cast<const int4*>(&Bc[(ocol + 1) * STRIDE + k]);
                int4 b2 = *reinterpret_cast<const int4*>(&Bc[(ocol + 2) * STRIDE + k]);
                int4 b3 = *reinterpret_cast<const int4*>(&Bc[(ocol + 3) * STRIDE + k]);
                acc[0][0] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a0.x), *reinterpret_cast<const v2h*>(&b0.x), acc[0][0], true);
                acc[0][0] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a0.y), *reinterpret_cast<const v2h*>(&b0.y), acc[0][0], true);
                acc[0][0] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a0.z), *reinterpret_cast<const v2h*>(&b0.z), acc[0][0], true);
                acc[0][0] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a0.w), *reinterpret_cast<const v2h*>(&b0.w), acc[0][0], true);
                acc[0][1] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a0.x), *reinterpret_cast<const v2h*>(&b1.x), acc[0][1], true);
                acc[0][1] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a0.y), *reinterpret_cast<const v2h*>(&b1.y), acc[0][1], true);
                acc[0][1] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a0.z), *reinterpret_cast<const v2h*>(&b1.z), acc[0][1], true);
                acc[0][1] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a0.w), *reinterpret_cast<const v2h*>(&b1.w), acc[0][1], true);
                acc[0][2] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a0.x), *reinterpret_cast<const v2h*>(&b2.x), acc[0][2], true);
                acc[0][2] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a0.y), *reinterpret_cast<const v2h*>(&b2.y), acc[0][2], true);
                acc[0][2] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a0.z), *reinterpret_cast<const v2h*>(&b2.z), acc[0][2], true);
                acc[0][2] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a0.w), *reinterpret_cast<const v2h*>(&b2.w), acc[0][2], true);
                acc[0][3] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a0.x), *reinterpret_cast<const v2h*>(&b3.x), acc[0][3], true);
                acc[0][3] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a0.y), *reinterpret_cast<const v2h*>(&b3.y), acc[0][3], true);
                acc[0][3] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a0.z), *reinterpret_cast<const v2h*>(&b3.z), acc[0][3], true);
                acc[0][3] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a0.w), *reinterpret_cast<const v2h*>(&b3.w), acc[0][3], true);
                acc[1][0] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a1.x), *reinterpret_cast<const v2h*>(&b0.x), acc[1][0], true);
                acc[1][0] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a1.y), *reinterpret_cast<const v2h*>(&b0.y), acc[1][0], true);
                acc[1][0] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a1.z), *reinterpret_cast<const v2h*>(&b0.z), acc[1][0], true);
                acc[1][0] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a1.w), *reinterpret_cast<const v2h*>(&b0.w), acc[1][0], true);
                acc[1][1] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a1.x), *reinterpret_cast<const v2h*>(&b1.x), acc[1][1], true);
                acc[1][1] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a1.y), *reinterpret_cast<const v2h*>(&b1.y), acc[1][1], true);
                acc[1][1] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a1.z), *reinterpret_cast<const v2h*>(&b1.z), acc[1][1], true);
                acc[1][1] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a1.w), *reinterpret_cast<const v2h*>(&b1.w), acc[1][1], true);
                acc[1][2] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a1.x), *reinterpret_cast<const v2h*>(&b2.x), acc[1][2], true);
                acc[1][2] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a1.y), *reinterpret_cast<const v2h*>(&b2.y), acc[1][2], true);
                acc[1][2] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a1.z), *reinterpret_cast<const v2h*>(&b2.z), acc[1][2], true);
                acc[1][2] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a1.w), *reinterpret_cast<const v2h*>(&b2.w), acc[1][2], true);
                acc[1][3] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a1.x), *reinterpret_cast<const v2h*>(&b3.x), acc[1][3], true);
                acc[1][3] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a1.y), *reinterpret_cast<const v2h*>(&b3.y), acc[1][3], true);
                acc[1][3] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a1.z), *reinterpret_cast<const v2h*>(&b3.z), acc[1][3], true);
                acc[1][3] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a1.w), *reinterpret_cast<const v2h*>(&b3.w), acc[1][3], true);
                acc[2][0] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a2.x), *reinterpret_cast<const v2h*>(&b0.x), acc[2][0], true);
                acc[2][0] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a2.y), *reinterpret_cast<const v2h*>(&b0.y), acc[2][0], true);
                acc[2][0] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a2.z), *reinterpret_cast<const v2h*>(&b0.z), acc[2][0], true);
                acc[2][0] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a2.w), *reinterpret_cast<const v2h*>(&b0.w), acc[2][0], true);
                acc[2][1] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a2.x), *reinterpret_cast<const v2h*>(&b1.x), acc[2][1], true);
                acc[2][1] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a2.y), *reinterpret_cast<const v2h*>(&b1.y), acc[2][1], true);
                acc[2][1] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a2.z), *reinterpret_cast<const v2h*>(&b1.z), acc[2][1], true);
                acc[2][1] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a2.w), *reinterpret_cast<const v2h*>(&b1.w), acc[2][1], true);
                acc[2][2] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a2.x), *reinterpret_cast<const v2h*>(&b2.x), acc[2][2], true);
                acc[2][2] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a2.y), *reinterpret_cast<const v2h*>(&b2.y), acc[2][2], true);
                acc[2][2] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a2.z), *reinterpret_cast<const v2h*>(&b2.z), acc[2][2], true);
                acc[2][2] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a2.w), *reinterpret_cast<const v2h*>(&b2.w), acc[2][2], true);
                acc[2][3] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a2.x), *reinterpret_cast<const v2h*>(&b3.x), acc[2][3], true);
                acc[2][3] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a2.y), *reinterpret_cast<const v2h*>(&b3.y), acc[2][3], true);
                acc[2][3] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a2.z), *reinterpret_cast<const v2h*>(&b3.z), acc[2][3], true);
                acc[2][3] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a2.w), *reinterpret_cast<const v2h*>(&b3.w), acc[2][3], true);
                acc[3][0] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a3.x), *reinterpret_cast<const v2h*>(&b0.x), acc[3][0], true);
                acc[3][0] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a3.y), *reinterpret_cast<const v2h*>(&b0.y), acc[3][0], true);
                acc[3][0] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a3.z), *reinterpret_cast<const v2h*>(&b0.z), acc[3][0], true);
                acc[3][0] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a3.w), *reinterpret_cast<const v2h*>(&b0.w), acc[3][0], true);
                acc[3][1] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a3.x), *reinterpret_cast<const v2h*>(&b1.x), acc[3][1], true);
                acc[3][1] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a3.y), *reinterpret_cast<const v2h*>(&b1.y), acc[3][1], true);
                acc[3][1] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a3.z), *reinterpret_cast<const v2h*>(&b1.z), acc[3][1], true);
                acc[3][1] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a3.w), *reinterpret_cast<const v2h*>(&b1.w), acc[3][1], true);
                acc[3][2] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a3.x), *reinterpret_cast<const v2h*>(&b2.x), acc[3][2], true);
                acc[3][2] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a3.y), *reinterpret_cast<const v2h*>(&b2.y), acc[3][2], true);
                acc[3][2] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a3.z), *reinterpret_cast<const v2h*>(&b2.z), acc[3][2], true);
                acc[3][2] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a3.w), *reinterpret_cast<const v2h*>(&b2.w), acc[3][2], true);
                acc[3][3] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a3.x), *reinterpret_cast<const v2h*>(&b3.x), acc[3][3], true);
                acc[3][3] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a3.y), *reinterpret_cast<const v2h*>(&b3.y), acc[3][3], true);
                acc[3][3] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a3.z), *reinterpret_cast<const v2h*>(&b3.z), acc[3][3], true);
                acc[3][3] = __builtin_amdgcn_fdot2(*reinterpret_cast<const v2h*>(&a3.w), *reinterpret_cast<const v2h*>(&b3.w), acc[3][3], true);
            }

            __syncthreads();
            buf = nb;
        }

        // Epilogue: write 4x4 outputs with bias/residual
        for (int ri = 0; ri < 4; ri++) {
            const int row = m0 + orow + ri;
            if (row >= M) continue;
            for (int ci = 0; ci < 4; ci++) {
                const int col = n0 + ocol + ci;
                if (col >= N) continue;
                C[row * ldc + col] = static_cast<OutT>(epi(row, col, acc[ri][ci]));
            }
        }
    }
}

}  // namespace comfy::hip_backend
