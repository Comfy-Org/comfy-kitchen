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

// Scalar fallback for FP32 and mixed input dtypes.
__global__ __launch_bounds__(256) void lora_down_scalar_kernel(
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

// x[M, K] @ lora_down[K, R] -> lora_act[M, R], with fp32 accumulation.
//
// One wave computes a 16-row by (TN * 16)-column output tile. The A fragment is
// contiguous in x. lora_down is row-major KxR rather than the transposed RxK
// layout consumed by the WMMA instruction, so each lane gathers one output
// column across the K step. For a fixed K element the 16 fragment rows are
// adjacent in memory, keeping those gathers coalesced without an LDS transpose.
template <typename Mma, int TN>
__global__ __launch_bounds__(kWave) void lora_down_wmma_kernel(
    const typename Mma::Elem* __restrict__ x,
    const typename Mma::Elem* __restrict__ lora_down,
    float* __restrict__ lora_act, int M, int K, int R) {

    const int lane = threadIdx.x;
    const int frag_r = frag_row(lane);
    const int m0 = blockIdx.y * 16;
    const int r0 = blockIdx.x * (TN * 16);

    typename Mma::Acc acc[TN];
#pragma unroll
    for (int n = 0; n < TN; ++n) {
        acc[n] = Mma::zero();
    }

    for (int k0 = 0; k0 < K; k0 += 16) {
        typename Mma::Frag a{};
        const int row = m0 + frag_r;
        if (row < M) {
            a = load_frag_16bit<Mma>(
                x + static_cast<int64_t>(row) * K + k0, lane);
        }

#pragma unroll
        for (int n = 0; n < TN; ++n) {
            typename Mma::Frag b{};
            const int col = r0 + n * 16 + frag_r;
            if (col < R) {
                const int kbase = k0 + Mma::frag_base(lane);
#pragma unroll
                for (int i = 0; i < Mma::kFragElems; ++i) {
                    b[i] = lora_down[static_cast<int64_t>(kbase + i) * R + col];
                }
            }
            acc[n] = Mma::mma(a, b, acc[n]);
        }
    }

    const int col_lane = acc_col(lane);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int row = m0 + acc_row(lane, e);
        if (row >= M) continue;
#pragma unroll
        for (int n = 0; n < TN; ++n) {
            const int col = r0 + n * 16 + col_lane;
            if (col < R) {
                lora_act[static_cast<int64_t>(row) * R + col] = Mma::get(acc[n], e);
            }
        }
    }
}

template <typename Mma>
void launch_lora_down_wmma(
    const void* x, const void* lora_down, float* lora_act,
    int M, int K, int R, hipStream_t stream) {

    using Elem = typename Mma::Elem;
    if (R <= 16) {
        constexpr int TN = 1;
        const dim3 grid((R + TN * 16 - 1) / (TN * 16), (M + 15) / 16);
        lora_down_wmma_kernel<Mma, TN><<<grid, kWave, 0, stream>>>(
            static_cast<const Elem*>(x), static_cast<const Elem*>(lora_down),
            lora_act, M, K, R);
    } else if (R <= 32) {
        constexpr int TN = 2;
        const dim3 grid((R + TN * 16 - 1) / (TN * 16), (M + 15) / 16);
        lora_down_wmma_kernel<Mma, TN><<<grid, kWave, 0, stream>>>(
            static_cast<const Elem*>(x), static_cast<const Elem*>(lora_down),
            lora_act, M, K, R);
    } else {
        constexpr int TN = 4;
        const dim3 grid((R + TN * 16 - 1) / (TN * 16), (M + 15) / 16);
        lora_down_wmma_kernel<Mma, TN><<<grid, kWave, 0, stream>>>(
            static_cast<const Elem*>(x), static_cast<const Elem*>(lora_down),
            lora_act, M, K, R);
    }
}

// Scalar fallback for operands with different dtypes.
template <bool UNSIGNED>
__global__ __launch_bounds__(256) void svdquant_quant_scalar_kernel(
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

// Smooth, then per-row per-group int4 quantize + pack. Sixteen threads process
// one 64-element group, keeping four smoothed values per thread so x and smooth
// are read once. ascales is stored transposed as (K/64, M_pad).
template <typename Elem, bool UNSIGNED>
__global__ __launch_bounds__(256) void svdquant_quant_kernel(
    const Elem* __restrict__ x, const Elem* __restrict__ smooth,
    int8_t* __restrict__ q, Elem* __restrict__ ascales,
    int M, int M_pad, int K) {

    constexpr int kThreadsPerGroup = 16;
    constexpr int kElemsPerThread = kSvdGroup / kThreadsPerGroup;
    constexpr int kGroupsPerBlock = 256 / kThreadsPerGroup;
    constexpr float kQMax = UNSIGNED ? 15.0f : 7.0f;
    constexpr int kQMin = UNSIGNED ? 0 : -7;
    constexpr int kQMaxInt = UNSIGNED ? 15 : 7;

    const int m = blockIdx.x;
    const int lane = threadIdx.x % kThreadsPerGroup;
    const int group_slot = threadIdx.x / kThreadsPerGroup;
    const int ngroups = K / kSvdGroup;

    for (int g = group_slot; g < ngroups; g += kGroupsPerBlock) {
        const int elem = lane * kElemsPerThread;
        const int kbase = g * kSvdGroup + elem;
        const int64_t xbase = static_cast<int64_t>(m) * K + kbase;

        float values[kElemsPerThread];
        float absmax = 0.0f;
#pragma unroll
        for (int i = 0; i < kElemsPerThread; ++i) {
            values[i] = static_cast<float>(x[xbase + i]) /
                        static_cast<float>(smooth[kbase + i]);
            absmax = fmaxf(absmax, fabsf(values[i]));
        }

#pragma unroll
        for (int offset = kThreadsPerGroup / 2; offset > 0; offset >>= 1) {
            absmax = fmaxf(absmax, __shfl_xor(absmax, offset, kThreadsPerGroup));
        }

        absmax = fmaxf(absmax, 1e-10f);
        const float scale = absmax / kQMax;
        const float inv = kQMax / absmax;
        if (lane == 0) {
            ascales[static_cast<int64_t>(g) * M_pad + m] = static_cast<Elem>(scale);
        }

        int quantized[kElemsPerThread];
#pragma unroll
        for (int i = 0; i < kElemsPerThread; ++i) {
            int v = static_cast<int>(rintf(values[i] * inv));
            quantized[i] = v < kQMin ? kQMin : (v > kQMaxInt ? kQMaxInt : v);
        }

        const uint16_t packed =
            static_cast<uint16_t>((quantized[0] & 0xF) |
                                  ((quantized[1] & 0xF) << 4) |
                                  ((quantized[2] & 0xF) << 8) |
                                  ((quantized[3] & 0xF) << 12));
        int8_t* qrow = q + static_cast<int64_t>(m) * (K / 2) + g * kSvdGroupBytes;
        reinterpret_cast<uint16_t*>(qrow)[lane] = packed;
    }
}

template <bool UNSIGNED>
void launch_svdquant_quant(
    const void* x, const void* smooth, int8_t* q, void* ascales,
    int M, int M_pad, int K, int x_code, int s_code, hipStream_t stream) {

    if (x_code == s_code) {
        if (x_code == 0) {
            svdquant_quant_kernel<float, UNSIGNED><<<M, 256, 0, stream>>>(
                static_cast<const float*>(x), static_cast<const float*>(smooth), q,
                static_cast<float*>(ascales), M, M_pad, K);
            return;
        }
        if (x_code == 1) {
            svdquant_quant_kernel<__half, UNSIGNED><<<M, 256, 0, stream>>>(
                static_cast<const __half*>(x), static_cast<const __half*>(smooth), q,
                static_cast<__half*>(ascales), M, M_pad, K);
            return;
        }
        if (x_code == 2) {
            svdquant_quant_kernel<__bf16, UNSIGNED><<<M, 256, 0, stream>>>(
                static_cast<const __bf16*>(x), static_cast<const __bf16*>(smooth), q,
                static_cast<__bf16*>(ascales), M, M_pad, K);
            return;
        }
    }

    svdquant_quant_scalar_kernel<UNSIGNED><<<M, 256, 0, stream>>>(
        x, smooth, q, ascales, M, M_pad, K, x_code, s_code);
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

    // Byte arrays that TileStager::store writes through uint2, as in gemm_wmma.h.
    __shared__ __align__(16) uint8_t As[BM * kStride];
    __shared__ __align__(16) uint8_t Bs[BN * kStride];

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
                        fout[i][j][e] += MmaInt4::get(acc[i][j], e) * as * ws;
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
    if (M == 0 || R == 0) {
        return;  // a zero-block launch is hipErrorInvalidConfiguration
    }
    lora_down_scalar_kernel<<<M, 256, 0, stream>>>(
        x, lora_down, static_cast<float*>(lora_act), M, K, R, x_code, d_code);
}

extern "C" void launch_svdquant_lora_down_wmma_kernel(
    const void* x, const void* lora_down, void* lora_act, int M, int K, int R, int x_code,
    int d_code, hipStream_t stream) {

    using namespace comfy::hip_backend;
    if (R > 256) throw std::runtime_error("svdquant: LoRA rank above 256 is not supported");
    if (M == 0 || R == 0) {
        return;  // a zero-block launch is hipErrorInvalidConfiguration
    }
    if ((K & 15) == 0 && x_code == 1 && d_code == 1) {
        launch_lora_down_wmma<MmaF16>(
            x, lora_down, static_cast<float*>(lora_act), M, K, R, stream);
    } else if ((K & 15) == 0 && x_code == 2 && d_code == 2) {
        launch_lora_down_wmma<MmaBf16>(
            x, lora_down, static_cast<float*>(lora_act), M, K, R, stream);
    } else {
        launch_svdquant_lora_down_kernel(
            x, lora_down, lora_act, M, K, R, x_code, d_code, stream);
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
    if (act_unsigned) {
        launch_svdquant_quant<true>(
            x, smooth, static_cast<int8_t*>(q), ascales,
            M, M_pad, K, x_code, s_code, stream);
    } else {
        launch_svdquant_quant<false>(
            x, smooth, static_cast<int8_t*>(q), ascales,
            M, M_pad, K, x_code, s_code, stream);
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
