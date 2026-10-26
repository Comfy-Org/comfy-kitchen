// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// SVDQuant W4A4 on WMMA (iu4: 16x16x32 on gfx12, 16x16x16 on gfx11). See mma.h.
//
//   out = (x/smooth quantized to int4) @ (W_int4 * wscales_group)^T
//       + (x @ proj_down) @ proj_up^T
//       + bias
//
// Both operands carry one scale per group of 64 elements along K, so the int32
// accumulator is flushed and rescaled once per group rather than once at the end
// of the K loop.
//
// The LoRA-up correction is folded into the writeback: rank R is small, so an
// R-length dot per output is cheap relative to the K loop and avoids a second
// pass over the output.
#include <stdexcept>
#include <string>

#include "../epilogue.h"
#include "../gemm_wmma.h"
#include "../hadamard.h"  // load_in

namespace comfy::hip_backend {

constexpr int kSvdGroup = 64;       // quantization group, in elements
constexpr int kSvdGroupBytes = 32;  // ... and in packed int4 bytes

// x @ proj_down -> (M, R), fp32, with proj_down passed transposed as (R, K). A
// 16-bit x runs on the WMMA GEMM core; this kernel is the fp32 path. R is narrow,
// so a block covers a 32-row by 32-rank output tile and walks K through LDS, each
// thread holding a 2x2 tile.
constexpr int kLoraRows = 32;
constexpr int kLoraRanks = 32;
constexpr int kLoraK = 64;

__global__ __launch_bounds__(256) void lora_down_kernel(
    const float* __restrict__ x, const float* __restrict__ down_t, float* __restrict__ lora_act,
    int M, int K, int R) {

    // Each thread stages 8 consecutive K elements of one x row and of one down_t row.
    static_assert(kLoraRows * kLoraK == 256 * 8 && kLoraK * kLoraRanks == 256 * 8,
                  "one 8-element run per thread per operand");
    // +1 keeps the half-wave's two x rows on distinct banks.
    __shared__ float xs[kLoraRows][kLoraK + 1];
    __shared__ float ds[kLoraK][kLoraRanks];

    const int tid = threadIdx.x;
    const int m0 = blockIdx.x * kLoraRows;
    const int r0 = blockIdx.y * kLoraRanks;
    const int ty = tid / 16;
    const int tx = tid % 16;

    const int xr = tid / (kLoraK / 8);
    const int xk = tid % (kLoraK / 8) * 8;
    const int dt = tid / (kLoraK / 8);
    const int dk = tid % (kLoraK / 8) * 8;
    const bool x_ok = m0 + xr < M;
    const bool d_ok = r0 + dt < R;
    const float* xrow = x + static_cast<int64_t>(x_ok ? m0 + xr : 0) * K + xk;
    const float* drow = down_t + static_cast<int64_t>(d_ok ? r0 + dt : 0) * K + dk;

    float xv[8], dv[8];
    auto load = [&](int k0) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            xv[i] = x_ok ? xrow[k0 + i] : 0.0f;
            dv[i] = d_ok ? drow[k0 + i] : 0.0f;
        }
    };

    float acc[2][2] = {};
    load(0);
    for (int k0 = 0; k0 < K; k0 += kLoraK) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            xs[xr][xk + i] = xv[i];
            ds[dk + i][dt] = dv[i];
        }
        __syncthreads();
        if (k0 + kLoraK < K) load(k0 + kLoraK);

        #pragma unroll 8
        for (int k = 0; k < kLoraK; ++k) {
            const float a0 = xs[ty * 2][k];
            const float a1 = xs[ty * 2 + 1][k];
            const float b0 = ds[k][tx * 2];
            const float b1 = ds[k][tx * 2 + 1];
            acc[0][0] = fmaf(a0, b0, acc[0][0]);
            acc[0][1] = fmaf(a0, b1, acc[0][1]);
            acc[1][0] = fmaf(a1, b0, acc[1][0]);
            acc[1][1] = fmaf(a1, b1, acc[1][1]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int m = m0 + ty * 2 + i;
        if (m >= M) continue;
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int t = r0 + tx * 2 + j;
            if (t < R) lora_act[static_cast<int64_t>(m) * R + t] = acc[i][j];
        }
    }
}

// Smooth, then per-row per-group int4 quantize + pack. ascales is stored
// transposed as (K/64, M_pad). One wave per row: a group is 64 elements, two per
// lane, so the wave reads each group as one contiguous span and writes it as one
// packed byte per lane.
constexpr int kSvdQuantRowsPerBlock = 8;

template <bool UNSIGNED>
__global__ __launch_bounds__(kSvdQuantRowsPerBlock* kWave) void svdquant_quant_kernel(
    const void* __restrict__ x, const void* __restrict__ smooth,
    int8_t* __restrict__ q, void* __restrict__ ascales,
    int M, int M_pad, int K, int x_code, int s_code) {

    static_assert(kSvdGroup == 2 * kWave, "a group must be two elements per lane");
    const int lane = threadIdx.x % kWave;
    const int m = blockIdx.x * kSvdQuantRowsPerBlock + threadIdx.x / kWave;
    if (m >= M) return;
    const int ngroups = K / kSvdGroup;

    constexpr float kQMax = UNSIGNED ? 15.0f : 7.0f;
    constexpr float kQMin = UNSIGNED ? 0.0f : -7.0f;

    for (int g = 0; g < ngroups; ++g) {
        const int k = g * kSvdGroup + 2 * lane;
        const int64_t idx = static_cast<int64_t>(m) * K + k;
        const float v0 = load_in(x, idx, x_code) / load_in(smooth, k, s_code);
        const float v1 = load_in(x, idx + 1, x_code) / load_in(smooth, k + 1, s_code);

        float absmax = fmaxf(fabsf(v0), fabsf(v1));
        #pragma unroll
        for (int off = kWave / 2; off > 0; off >>= 1)
            absmax = fmaxf(absmax, __shfl_xor(absmax, off, kWave));
        absmax = fmaxf(absmax, 1e-10f);
        const float scale = absmax / kQMax;
        const float inv = kQMax / absmax;

        // (K/64, M_pad), so the group stride is M_pad.
        if (lane == 0) store_scalar(ascales, s_code, static_cast<int64_t>(g) * M_pad + m, scale);

        int q0 = static_cast<int>(rintf(v0 * inv));
        int q1 = static_cast<int>(rintf(v1 * inv));
        q0 = q0 < static_cast<int>(kQMin) ? static_cast<int>(kQMin)
                                          : (q0 > static_cast<int>(kQMax) ? static_cast<int>(kQMax) : q0);
        q1 = q1 < static_cast<int>(kQMin) ? static_cast<int>(kQMin)
                                          : (q1 > static_cast<int>(kQMax) ? static_cast<int>(kQMax) : q1);
        q[static_cast<int64_t>(m) * (K / 2) + g * kSvdGroupBytes + lane] =
            static_cast<int8_t>(((q0 & 0xF) | ((q1 & 0xF) << 4)) & 0xFF);
    }
}

// Group-scaled int4 GEMM with the LoRA-up correction folded into the writeback.
//
// The flush after every group is VALU work the int4 MMAs cannot hide, so it reads
// both scales from LDS as fp32: each one is decoded once per block, where reading
// it from global memory per accumulator element costs more than the MMAs.
template <typename OutT, bool UNSIGNED>
__global__ __launch_bounds__(256) void svdquant_gemm_kernel(
    const uint8_t* __restrict__ A, const uint8_t* __restrict__ B, OutT* __restrict__ C,
    const void* __restrict__ ascales, const void* __restrict__ wscales,
    const void* __restrict__ lora_act, const void* __restrict__ lora_up,
    const void* __restrict__ bias,
    int M, int N, int K, int R,
    int as_code, int ws_code, int la_code, int lu_code, int bias_code) {

    constexpr int BM = 128, BN = 128, BKB = 64, TM = 2, TN = 4;
    constexpr int kStride = BKB + kLdsPad;
    constexpr int kGroups = BKB / kSvdGroupBytes;
    // LoRA ranks staged per pass. The +1 pad keeps the 16 columns a half-wave
    // reads from lora_up on distinct banks.
    constexpr int kRankChunk = 16;
    constexpr int kRankStride = kRankChunk + 1;
    constexpr int kTileBytes = (BM + BN) * kStride;
    constexpr int kLoraBytes = (BM + BN) * kRankStride * static_cast<int>(sizeof(float));
    static_assert(kGroups * BM == 256 && kGroups * BN == 256, "one scale per thread per tile");

    // The LoRA operands reuse the tile's LDS once the K loop is done.
    __shared__ __align__(16) uint8_t smem[kTileBytes > kLoraBytes ? kTileBytes : kLoraBytes];
    __shared__ float as_s[kGroups][BM];
    __shared__ float ws_s[kGroups][BN];
    uint8_t* As = smem;
    uint8_t* Bs = smem + BM * kStride;

    const int tid = threadIdx.x;
    const int lane = tid % kWave;
    const int warp = tid / kWave;
    const int wm = warp / 2;
    const int wn = warp % 2;

    // Grouped block ordering for L2 reuse of B, as in gemm_wmma_kernel.
    constexpr int kGroupM = 4;
    const int blocks_n = gridDim.x;
    const int blocks_m = gridDim.y;
    const int bid = blockIdx.y * blocks_n + blockIdx.x;
    const int per_group = kGroupM * blocks_n;
    const int group = bid / per_group;
    const int idx_in_group = bid - group * per_group;
    const int group_rows = min(kGroupM, blocks_m - group * kGroupM);
    const int m0 = (group * kGroupM + idx_in_group % group_rows) * BM;
    const int n0 = (idx_in_group / group_rows) * BN;

    const int kbytes = K / 2;
    const int ngroups = K / kSvdGroup;

    // Thread tid stages scale tid % 128 of group tid / 128 for both operands. Rows
    // and groups out of range read as 0, which zeroes their contribution.
    const int s_grp = tid / BM;
    const int s_idx = tid % BM;
    auto load_scales = [&](int kb, float& a, float& w) {
        const int g = kb / kSvdGroupBytes + s_grp;
        const int r = m0 + s_idx;
        const int c = n0 + s_idx;
        a = (g < ngroups && r < M) ? load_in(ascales, static_cast<int64_t>(g) * M + r, as_code)
                                   : 0.0f;
        w = (g < ngroups && c < N) ? load_in(wscales, static_cast<int64_t>(g) * N + c, ws_code)
                                   : 0.0f;
    };

    float fout[TM][TN][8] = {};

    const int row = frag_row(lane);
    const int col_lane = acc_col(lane);

    TileStager<BM, BKB, 256> sa;
    TileStager<BN, BKB, 256> sb;
    float a_next, w_next;

    sa.load(A, m0, M, 0, kbytes);
    sb.load(B, n0, N, 0, kbytes);
    load_scales(0, a_next, w_next);
    sa.store(As);
    sb.store(Bs);
    as_s[s_grp][s_idx] = a_next;
    ws_s[s_grp][s_idx] = w_next;
    __syncthreads();

    for (int kb0 = 0; kb0 < kbytes; kb0 += BKB) {
        const int knext = kb0 + BKB;
        const bool has_next = knext < kbytes;
        if (has_next) {
            sa.load(A, m0, M, knext, kbytes);
            sb.load(B, n0, N, knext, kbytes);
            load_scales(knext, a_next, w_next);
        }

        // BKB bytes = 128 elements = 2 groups of 64. A group is kSvdGroupBytes of
        // packed int4, which is kSvdSteps MMAs: 2 on gfx12 (K=32 each), 4 on gfx11
        // (K=16 each). The accumulator is flushed per group, so the scales apply to
        // exactly the 64 elements they were computed for either way.
        constexpr int kSvdSteps = kSvdGroupBytes / MmaInt4::kStepBytes;
        static_assert(kSvdGroupBytes % MmaInt4::kStepBytes == 0,
                      "an int4 K-step must divide the quantization group");

        #pragma unroll
        for (int grp = 0; grp < kGroups; ++grp) {
            // A K that is not a multiple of the tile leaves the trailing group
            // entirely in the zero padding; its scales do not exist.
            if (kb0 + grp * kSvdGroupBytes >= kbytes) continue;

            typename MmaInt4::Acc acc[TM][TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = MmaInt4::zero();

            #pragma unroll
            for (int s = 0; s < kSvdSteps; ++s) {
                const int kbyte = grp * kSvdGroupBytes + s * MmaInt4::kStepBytes;

                typename MmaInt4::Frag af[TM];
                #pragma unroll
                for (int i = 0; i < TM; ++i)
                    af[i] = MmaInt4::load(As, wm * (TM * 16) + i * 16 + row, kbyte, kStride, lane);

                typename MmaInt4::Frag bf[TN];
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    bf[j] = MmaInt4::load(Bs, wn * (TN * 16) + j * 16 + row, kbyte, kStride, lane);

                #pragma unroll
                for (int i = 0; i < TM; ++i)
                    #pragma unroll
                    for (int j = 0; j < TN; ++j) {
                        acc[i][j] = UNSIGNED ? MmaInt4::mma_ua(af[i], bf[j], acc[i][j])
                                             : MmaInt4::mma(af[i], bf[j], acc[i][j]);
                    }
            }

            // Flush the group: rescale the int32 accumulator into fp32.
            float as[TM][8];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int e = 0; e < 8; ++e)
                    as[i][e] = as_s[grp][wm * (TM * 16) + i * 16 + acc_row(lane, e)];
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                const float ws = ws_s[grp][wn * (TN * 16) + j * 16 + col_lane];
                #pragma unroll
                for (int i = 0; i < TM; ++i)
                    #pragma unroll
                    for (int e = 0; e < 8; ++e)
                        fout[i][j][e] =
                            fmaf(MmaInt4::get(acc[i][j], e), as[i][e] * ws, fout[i][j][e]);
            }
        }

        if (has_next) {
            __syncthreads();
            sa.store(As);
            sb.store(Bs);
            as_s[s_grp][s_idx] = a_next;
            ws_s[s_grp][s_idx] = w_next;
            __syncthreads();
        }
    }

    float* la_s = reinterpret_cast<float*>(smem);
    float* lu_s = la_s + BM * kRankStride;
    for (int t0 = 0; t0 < R; t0 += kRankChunk) {
        __syncthreads();  // the K loop or the previous chunk is done with smem
        for (int idx = tid; idx < BM * kRankChunk; idx += 256) {
            const int r = idx / kRankChunk;
            const int t = idx % kRankChunk;
            const bool ok = m0 + r < M && t0 + t < R;
            la_s[r * kRankStride + t] =
                ok ? load_in(lora_act, static_cast<int64_t>(m0 + r) * R + t0 + t, la_code) : 0.0f;
        }
        for (int idx = tid; idx < BN * kRankChunk; idx += 256) {
            const int c = idx / kRankChunk;
            const int t = idx % kRankChunk;
            const bool ok = n0 + c < N && t0 + t < R;
            lu_s[c * kRankStride + t] =
                ok ? load_in(lora_up, static_cast<int64_t>(n0 + c) * R + t0 + t, lu_code) : 0.0f;
        }
        __syncthreads();

        #pragma unroll 4
        for (int t = 0; t < kRankChunk; ++t) {
            float la[TM][8];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int e = 0; e < 8; ++e)
                    la[i][e] = la_s[(wm * (TM * 16) + i * 16 + acc_row(lane, e)) * kRankStride + t];
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                const float lu = lu_s[(wn * (TN * 16) + j * 16 + col_lane) * kRankStride + t];
                #pragma unroll
                for (int i = 0; i < TM; ++i)
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) fout[i][j][e] = fmaf(la[i][e], lu, fout[i][j][e]);
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int r = m0 + wm * (TM * 16) + i * 16 + acc_row(lane, e);
            if (r >= M) continue;
            OutT* crow = C + static_cast<int64_t>(r) * N;
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int col = n0 + wn * (TN * 16) + j * 16 + col_lane;
                if (col >= N) continue;
                float v = fout[i][j][e];
                if (bias != nullptr) v += load_scalar(bias, bias_code, col);
                crow[col] = static_cast<OutT>(v);
            }
        }
    }
}

}  // namespace comfy::hip_backend

// The output tile of the WMMA path: lora_act is written as accumulated.
struct EpiLoraDown {
    __forceinline__ __device__ void init() {}
    __forceinline__ __device__ float operator()(int, int, float acc) const { return acc; }
};

extern "C" void launch_svdquant_lora_down_kernel(
    const void* x, const void* down_t, void* lora_act, int M, int K, int R, int x_code,
    int d_code, hipStream_t stream) {

    using namespace comfy::hip_backend;
    if (M == 0 || R == 0) {
        return;  // a zero-block launch is hipErrorInvalidConfiguration
    }
    // Callers pass the quantizer's K, a whole number of groups, so the K tiles are full.
    if (K % kLoraK != 0) {
        throw std::runtime_error("svdquant_lora_down: K=" + std::to_string(K) +
                                 " must be a multiple of 64");
    }
    if (x_code != d_code) {
        throw std::runtime_error("svdquant_lora_down: x and lora_down must share a dtype");
    }
    float* out = static_cast<float*>(lora_act);
    const uint8_t* a = static_cast<const uint8_t*>(x);
    const uint8_t* b = static_cast<const uint8_t*>(down_t);
    if (x_code == 1) {
        launch_gemm_wmma<MmaF16, EpiLoraDown, float>(a, b, out, M, R, K * 2, R, EpiLoraDown{},
                                                     stream);
    } else if (x_code == 2) {
        launch_gemm_wmma<MmaBf16, EpiLoraDown, float>(a, b, out, M, R, K * 2, R, EpiLoraDown{},
                                                      stream);
    } else {
        dim3 grid((M + kLoraRows - 1) / kLoraRows, (R + kLoraRanks - 1) / kLoraRanks);
        lora_down_kernel<<<grid, 256, 0, stream>>>(
            static_cast<const float*>(x), static_cast<const float*>(down_t), out, M, K, R);
    }
}

extern "C" void launch_svdquant_quant_kernel(
    const void* x, const void* smooth, void* q, void* ascales, int M, int M_pad, int K,
    int x_code, int s_code, bool act_unsigned, hipStream_t stream) {

    using namespace comfy::hip_backend;
    if (M == 0) {
        return;  // a zero-block launch is hipErrorInvalidConfiguration
    }
    // The kernel writes K/64 complete groups; a partial group would leave the
    // packed tail uninitialized. The binding checks this too, but mirror the GEMM
    // launcher's invariant here.
    if (K % kSvdGroup != 0) {
        throw std::runtime_error("svdquant_quantize: K=" + std::to_string(K) +
                                 " must be a multiple of 64");
    }
    const int blocks = (M + kSvdQuantRowsPerBlock - 1) / kSvdQuantRowsPerBlock;
    constexpr int threads = kSvdQuantRowsPerBlock * kWave;
    if (act_unsigned) {
        svdquant_quant_kernel<true><<<blocks, threads, 0, stream>>>(
            x, smooth, static_cast<int8_t*>(q), ascales, M, M_pad, K, x_code, s_code);
    } else {
        svdquant_quant_kernel<false><<<blocks, threads, 0, stream>>>(
            x, smooth, static_cast<int8_t*>(q), ascales, M, M_pad, K, x_code, s_code);
    }
}

extern "C" void launch_svdquant_gemm_kernel(
    const void* a, const void* b, void* c, const void* ascales, const void* wscales,
    const void* lora_act, const void* lora_up, const void* bias,
    int M, int N, int K, int R,
    int as_code, int ws_code, int la_code, int lu_code, int bias_code, int out_code,
    bool act_unsigned, hipStream_t stream) {

    using namespace comfy::hip_backend;

    if (M == 0 || N == 0) {
        return;  // a zero-dimension grid is hipErrorInvalidConfiguration
    }
    // One scale per 64-element group, and the tile loader reads the packed row
    // (K/2 bytes) in 16-byte chunks: K must be a whole number of groups. An empty
    // GEMM reads no groups, so this only applies once there is real work above.
    if (K % kSvdGroup != 0) {
        throw std::runtime_error("scaled_mm_svdquant_w4a4: K=" + std::to_string(K) +
                                 " must be a multiple of 64");
    }

    dim3 grid((N + 127) / 128, (M + 127) / 128);
    const uint8_t* A = static_cast<const uint8_t*>(a);
    const uint8_t* B = static_cast<const uint8_t*>(b);

#define CK_SVD_LAUNCH(T, U)                                                                    \
    svdquant_gemm_kernel<T, U><<<grid, 256, 0, stream>>>(A, B, static_cast<T*>(c), ascales,     \
                                                         wscales, lora_act, lora_up, bias, M, N, \
                                                         K, R, as_code, ws_code, la_code,        \
                                                         lu_code, bias_code)

    if (out_code == kBF16) {
        if (act_unsigned) { CK_SVD_LAUNCH(__bf16, true); } else { CK_SVD_LAUNCH(__bf16, false); }
    } else if (out_code == kF16) {
        if (act_unsigned) { CK_SVD_LAUNCH(__half, true); } else { CK_SVD_LAUNCH(__half, false); }
    } else if (out_code == kF32) {
        if (act_unsigned) { CK_SVD_LAUNCH(float, true); } else { CK_SVD_LAUNCH(float, false); }
    } else {
        throw std::runtime_error("scaled_mm_svdquant_w4a4: unsupported output dtype");
    }
#undef CK_SVD_LAUNCH
}
