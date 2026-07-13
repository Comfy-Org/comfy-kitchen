// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// SVDQuant W4A4 on gfx12 WMMA (v_wmma_i32_16x16x32_iu4).
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

// x @ proj_down -> (M, R), fp32. One thread per output column: proj_down is
// (K, R), so consecutive threads read consecutive addresses.
__global__ __launch_bounds__(256) void lora_down_kernel(
    const void* __restrict__ x, const void* __restrict__ lora_down,
    float* __restrict__ lora_act, int M, int K, int R, int x_code, int d_code) {

    const int m = blockIdx.x;
    const int r = threadIdx.x;
    if (r >= R) return;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += load_in(x, static_cast<int64_t>(m) * K + k, x_code) *
               load_in(lora_down, static_cast<int64_t>(k) * R + r, d_code);
    }
    lora_act[static_cast<int64_t>(m) * R + r] = sum;
}

// Smooth, then per-row per-group int4 quantize + pack. ascales is stored
// transposed as (K/64, M_pad).
template <bool UNSIGNED>
__global__ __launch_bounds__(256) void svdquant_quant_kernel(
    const void* __restrict__ x, const void* __restrict__ smooth,
    int8_t* __restrict__ q, void* __restrict__ ascales,
    int M, int M_pad, int K, int x_code, int s_code) {

    const int m = blockIdx.x;
    const int ngroups = K / kSvdGroup;

    constexpr float kQMax = UNSIGNED ? 15.0f : 7.0f;
    constexpr float kQMin = UNSIGNED ? 0.0f : -7.0f;

    for (int g = threadIdx.x; g < ngroups; g += 256) {
        const int64_t base = static_cast<int64_t>(m) * K + static_cast<int64_t>(g) * kSvdGroup;

        float absmax = 0.0f;
        for (int i = 0; i < kSvdGroup; ++i) {
            const int k = g * kSvdGroup + i;
            const float v = load_in(x, base + i, x_code) / load_in(smooth, k, s_code);
            absmax = fmaxf(absmax, fabsf(v));
        }
        absmax = fmaxf(absmax, 1e-10f);
        const float scale = absmax / kQMax;
        const float inv = kQMax / absmax;

        // (K/64, M_pad), so the group stride is M_pad.
        store_scalar(ascales, s_code, static_cast<int64_t>(g) * M_pad + m, scale);

        int8_t* qrow = q + static_cast<int64_t>(m) * (K / 2) + g * kSvdGroupBytes;
        for (int i = 0; i < kSvdGroup; i += 2) {
            const int k = g * kSvdGroup + i;
            const float v0 = load_in(x, base + i, x_code) / load_in(smooth, k, s_code);
            const float v1 = load_in(x, base + i + 1, x_code) / load_in(smooth, k + 1, s_code);

            int q0 = static_cast<int>(rintf(v0 * inv));
            int q1 = static_cast<int>(rintf(v1 * inv));
            q0 = q0 < static_cast<int>(kQMin) ? static_cast<int>(kQMin)
                                              : (q0 > static_cast<int>(kQMax) ? static_cast<int>(kQMax) : q0);
            q1 = q1 < static_cast<int>(kQMin) ? static_cast<int>(kQMin)
                                              : (q1 > static_cast<int>(kQMax) ? static_cast<int>(kQMax) : q1);
            qrow[i / 2] = static_cast<int8_t>(((q0 & 0xF) | ((q1 & 0xF) << 4)) & 0xFF);
        }
    }
}

// Group-scaled int4 GEMM with the LoRA-up correction folded into the writeback.
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

    __shared__ uint8_t As[BM * kStride];
    __shared__ uint8_t Bs[BN * kStride];

    const int tid = threadIdx.x;
    const int lane = tid % kWave;
    const int warp = tid / kWave;
    const int wm = warp / 2;
    const int wn = warp % 2;

    const int m0 = blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;
    const int kbytes = K / 2;

    float fout[TM][TN][8] = {};

    const int row = frag_row(lane);

    TileStager<BM, BKB, 256> sa;
    TileStager<BN, BKB, 256> sb;

    sa.load(A, m0, M, 0, kbytes);
    sb.load(B, n0, N, 0, kbytes);
    sa.store(As);
    sb.store(Bs);
    __syncthreads();

    for (int kb0 = 0; kb0 < kbytes; kb0 += BKB) {
        const int knext = kb0 + BKB;
        const bool has_next = knext < kbytes;
        if (has_next) {
            sa.load(A, m0, M, knext, kbytes);
            sb.load(B, n0, N, knext, kbytes);
        }

        // BKB bytes = 128 elements = 2 groups of 64. A group is kSvdGroupBytes of
        // packed int4, which is kSvdSteps MMAs: 2 on gfx12 (K=32 each), 4 on gfx11
        // (K=16 each). The accumulator is flushed per group, so the scales apply to
        // exactly the 64 elements they were computed for either way.
        constexpr int kSvdSteps = kSvdGroupBytes / MmaInt4::kStepBytes;
        static_assert(kSvdGroupBytes % MmaInt4::kStepBytes == 0,
                      "an int4 K-step must divide the quantization group");

        #pragma unroll
        for (int grp = 0; grp < BKB / kSvdGroupBytes; ++grp) {
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
            const int g = kb0 / kSvdGroupBytes + grp;
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int col = n0 + wn * (TN * 16) + j * 16 + acc_col(lane);
                if (col >= N) continue;
                const float ws = load_in(wscales, static_cast<int64_t>(g) * N + col, ws_code);
                #pragma unroll
                for (int i = 0; i < TM; ++i) {
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        const int r = m0 + wm * (TM * 16) + i * 16 + acc_row(lane, e);
                        if (r >= M) continue;
                        const float as =
                            load_in(ascales, static_cast<int64_t>(g) * M + r, as_code);
                        fout[i][j][e] += static_cast<float>(acc[i][j][e]) * as * ws;
                    }
                }
            }
        }

        if (has_next) {
            __syncthreads();
            sa.store(As);
            sb.store(Bs);
            __syncthreads();
        }
    }

    #pragma unroll
    for (int j = 0; j < TN; ++j) {
        const int col = n0 + wn * (TN * 16) + j * 16 + acc_col(lane);
        if (col >= N) continue;
        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int r = m0 + wm * (TM * 16) + i * 16 + acc_row(lane, e);
                if (r >= M) continue;

                float v = fout[i][j][e];
                for (int t = 0; t < R; ++t) {
                    v += load_in(lora_act, static_cast<int64_t>(r) * R + t, la_code) *
                         load_in(lora_up, static_cast<int64_t>(col) * R + t, lu_code);
                }
                if (bias != nullptr) v += load_scalar(bias, bias_code, col);
                C[static_cast<int64_t>(r) * N + col] = static_cast<OutT>(v);
            }
        }
    }
}

}  // namespace comfy::hip_backend

extern "C" void launch_svdquant_lora_down_kernel(
    const void* x, const void* lora_down, void* lora_act, int M, int K, int R, int x_code,
    int d_code, hipStream_t stream) {

    using namespace comfy::hip_backend;
    if (R > 256) throw std::runtime_error("svdquant: LoRA rank above 256 is not supported");
    if (M == 0) {
        return;  // a zero-block launch is hipErrorInvalidConfiguration
    }
    lora_down_kernel<<<M, 256, 0, stream>>>(x, lora_down, static_cast<float*>(lora_act), M, K, R,
                                            x_code, d_code);
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
    if (act_unsigned) {
        svdquant_quant_kernel<true><<<M, 256, 0, stream>>>(
            x, smooth, static_cast<int8_t*>(q), ascales, M, M_pad, K, x_code, s_code);
    } else {
        svdquant_quant_kernel<false><<<M, 256, 0, stream>>>(
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
