// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Linear on packed fp16 math (v_pk_fma_f16) for gfx1010, which has neither
// matrix cores nor dot instructions: twice the MAC rate of its fp32 or mixed FMAs.
//   D[M, N] = (A[M, K] / s[row]) @ B[N, K]^T * s[row] + bias
// SimtF16Packed accumulates kStageProducts products per fp16 half before folding into
// fp32, so each row of A is first scaled by a power of two s (exact) that keeps that
// many products of its largest element with the weight's largest below
// kStageSumLimit. The rows are written to fp16 scratch, which also takes fp32 or bf16
// activations of any range.
//
// The same GEMM serves ConvRot INT8 weights (|w| <= 127, exact in fp16) with fp16
// activations: the rotation kernel writes the scaled rows directly and the int8
// path's rowwise epilogue applies both scales. Measured on gfx1010 at the MiniMax-H3
// VAE decoder's shapes (M=5800): 5.3-5.8 TFLOPS against 4.1-4.5 for the INT8 GEMM,
// and no activation rounding to int8 (error 8.3e-3 vs 1.2e-2, from the weights).
#include <stdexcept>

#include "../epilogue.h"
#include "../gemm_simt.h"
#include "../hadamard.h"

namespace comfy::hip_backend {
namespace {

// Half of fp16's range, the margin for the fp16 rounding of the partial sums.
//
// Why the bound holds: a fp16 chain sums kStageProducts products a * w, so with
// |a| <= amax_row / s and |w| <= w_amax its exact value is at most
// kStageProducts * amax_row * w_amax / s. Choosing s as below makes that <= 32768. Each
// fp16 FMA rounds its running sum by at most half an ulp (2^-11 relative), so the rounded
// chain stays well under fp16's largest finite value, 65504, and never becomes inf.
constexpr float kStageSumLimit = 32768.0f;
// Products per fp16 half between folds, see SimtF16Packed.
constexpr int kStageProducts = kSimtWords * SimtF16Packed::kFoldStages;
constexpr int kScaleThreads = 256;

__forceinline__ __device__ float load_f(const float* p, int64_t i) { return p[i]; }
__forceinline__ __device__ float load_f(const __half* p, int64_t i) { return __half2float(p[i]); }
__forceinline__ __device__ float load_f(const __bf16* p, int64_t i) { return static_cast<float>(p[i]); }

template <typename InT>
__global__ __launch_bounds__(kScaleThreads) void scale_rows_to_f16_kernel(
    const InT* __restrict__ x, __half* __restrict__ out, float* __restrict__ row_scale,
    const float* __restrict__ w_amax, int K) {
    __shared__ float wave_max[kScaleThreads / 32];
    __shared__ float row_inv;
    const int64_t base = static_cast<int64_t>(blockIdx.x) * K;
    const int tid = threadIdx.x;

    float amax = 0.0f;
    for (int i = tid; i < K; i += kScaleThreads) amax = fmaxf(amax, fabsf(load_f(x, base + i)));
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) amax = fmaxf(amax, __shfl_xor(amax, offset, 32));
    if ((tid & 31) == 0) wave_max[tid >> 5] = amax;
    __syncthreads();
    if (tid == 0) {
#pragma unroll
        for (int w = 1; w < kScaleThreads / 32; ++w) amax = fmaxf(amax, wave_max[w]);
        // The smallest power of two that bounds the stage sums, which also scales a
        // small row up into fp16's precision. A power of two only changes the exponent,
        // so dividing by it here and multiplying it back in EpiRowScale is exact: the only
        // rounding this pass adds is fp32 or bf16 -> fp16, whose 11-bit mantissa keeps
        // more of x than bf16's 8. e < 0 (a row of small values) scales the row up, which
        // keeps it out of fp16's subnormal range. An all-zero row (ratio 0) keeps e = 0.
        // Weights small enough that the stage sums allow a scaled row above fp16's range
        // (w_amax * kStageProducts < 1) bound the row itself by kStageSumLimit instead.
        const float ratio = amax * fmaxf(w_amax[0] * kStageProducts, 1.0f) / kStageSumLimit;
        const int e = ratio > 0.0f ? static_cast<int>(ceilf(log2f(ratio))) : 0;
        row_scale[blockIdx.x] = ldexpf(1.0f, e);
        row_inv = ldexpf(1.0f, -e);
    }
    __syncthreads();
    const float inv = row_inv;
    for (int i = tid; i < K; i += kScaleThreads) out[base + i] = __float2half(load_f(x, base + i) * inv);
}

// Bound on a row's largest element against weights of largest magnitude w_amax, the
// same bound as above solved for amax_row. The ConvRot path passes 127, the largest
// |w| of an INT8 weight, since its weight needs no reduction to find it.
constexpr float row_bound(float w_amax) { return kStageSumLimit / (w_amax * kStageProducts); }

template <typename InT>
void launch_scale_rows(const void* x, __half* x16, float* row_scale, const float* w_amax, int M,
                       int K, hipStream_t stream) {
    scale_rows_to_f16_kernel<InT><<<M, kScaleThreads, 0, stream>>>(
        static_cast<const InT*>(x), x16, row_scale, w_amax, K);
}

// fp16 rows x [R, K] into the GEMM's tiled layout (simt_tiled_offset), where out holds
// ceil(R / 128) * 128 rows. One thread per 16-byte chunk, so K is a multiple of 8.
__global__ __launch_bounds__(256) void tile_rows_f16_kernel(
    const __half* __restrict__ x, __half* __restrict__ out, int R, int K) {
    const int64_t chunk = static_cast<int64_t>(blockIdx.x) * 256 + threadIdx.x;
    const int row = static_cast<int>(chunk / (K / 8)), k = static_cast<int>(chunk % (K / 8)) * 8;
    if (row >= R) return;
    *reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(out) + simt_tiled_offset(row, 2 * k, 2 * K)) =
        *reinterpret_cast<const uint4*>(x + static_cast<int64_t>(row) * K + k);
}

// Decode-sized M (<= 8) is bound on streaming the weight, which the GEMM's tiles would
// only read once per row: one wave per output column reads its fp16 weight row once
// for every x row, in fp32 math (a fp16 weight is exact in fp32), so the result is
// that of linear(x, w.float()) without the fp32 copy of the weight.
constexpr int kGemvWaves = 8;
constexpr int kGemvMaxRows = 8;

template <typename InT, typename OutT>
__global__ __launch_bounds__(kWave* kGemvWaves) void gemv_f16_kernel(
    const InT* __restrict__ x, const __half* __restrict__ w, const void* __restrict__ bias,
    int bias_code, OutT* __restrict__ out, int M, int N, int K) {
    const int n = blockIdx.x * kGemvWaves + threadIdx.y;
    if (n >= N) return;
    const __half* wrow = w + static_cast<int64_t>(n) * K;

    float sum[kGemvMaxRows] = {};
    for (int k = threadIdx.x * 8; k < K; k += kWave * 8) {
        const uint4 wv = *reinterpret_cast<const uint4*>(wrow + k);
        const __half* wh = reinterpret_cast<const __half*>(&wv);
        float wf[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) wf[i] = __half2float(wh[i]);
#pragma unroll
        for (int m = 0; m < kGemvMaxRows; ++m) {
            if (m < M) {
                const InT* xr = x + static_cast<int64_t>(m) * K + k;
#pragma unroll
                for (int i = 0; i < 8; ++i) sum[m] = fmaf(load_f(xr, i), wf[i], sum[m]);
            }
        }
    }
#pragma unroll
    for (int m = 0; m < kGemvMaxRows; ++m) {
        if (m < M) {
            float v = wave_reduce_sum(sum[m]);
            if (threadIdx.x == 0) {
                if (bias) v += load_scalar(bias, bias_code, n);
                out[static_cast<int64_t>(m) * N + n] = static_cast<OutT>(v);
            }
        }
    }
}

template <typename InT>
void launch_gemv_f16(const void* x, const void* w, const void* bias, int bias_code, void* d,
                     int out_code, int M, int N, int K, hipStream_t stream) {
    const dim3 block(kWave, kGemvWaves);
    const dim3 grid((N + kGemvWaves - 1) / kGemvWaves);
    const auto* xi = static_cast<const InT*>(x);
    const auto* wh = static_cast<const __half*>(w);
    if (out_code == kF32) {
        gemv_f16_kernel<InT, float><<<grid, block, 0, stream>>>(xi, wh, bias, bias_code, static_cast<float*>(d), M, N, K);
    } else if (out_code == kBF16) {
        gemv_f16_kernel<InT, __bf16><<<grid, block, 0, stream>>>(xi, wh, bias, bias_code, static_cast<__bf16*>(d), M, N, K);
    } else if (out_code == kF16) {
        gemv_f16_kernel<InT, __half><<<grid, block, 0, stream>>>(xi, wh, bias, bias_code, static_cast<__half*>(d), M, N, K);
    } else {
        throw std::runtime_error("fp16_packed_linear: unsupported output dtype");
    }
}

}  // namespace
}  // namespace comfy::hip_backend

extern "C" bool launch_fp16_packed_linear_kernel(
    const void* x, int x_code, const void* w, const void* w_amax, void* x16, void* row_scale,
    const void* bias, int bias_code, void* d, int out_code, int M, int N, int K,
    hipStream_t stream) {

    using namespace comfy::hip_backend;

    if (M == 0 || N == 0) {
        return true;  // a zero-dimension grid is hipErrorInvalidConfiguration
    }
    // Rows are staged 16 bytes (8 halves) at a time.
    if (K % 8 != 0) {
        return false;
    }
    if (M <= kGemvMaxRows) {
        if (x_code == kF32) {
            launch_gemv_f16<float>(x, w, bias, bias_code, d, out_code, M, N, K, stream);
        } else if (x_code == kF16) {
            launch_gemv_f16<__half>(x, w, bias, bias_code, d, out_code, M, N, K, stream);
        } else if (x_code == kBF16) {
            launch_gemv_f16<__bf16>(x, w, bias, bias_code, d, out_code, M, N, K, stream);
        } else {
            throw std::runtime_error("fp16_packed_linear: unsupported input dtype");
        }
        return true;
    }
    auto* xh = static_cast<__half*>(x16);
    auto* scale = static_cast<float*>(row_scale);
    const auto* amax = static_cast<const float*>(w_amax);
    if (x_code == kF32) {
        launch_scale_rows<float>(x, xh, scale, amax, M, K, stream);
    } else if (x_code == kF16) {
        launch_scale_rows<__half>(x, xh, scale, amax, M, K, stream);
    } else if (x_code == kBF16) {
        launch_scale_rows<__bf16>(x, xh, scale, amax, M, K, stream);
    } else {
        throw std::runtime_error("fp16_packed_linear: unsupported input dtype");
    }

    // The GEMM sums (x[row] / s[row]) . w; EpiRowScale multiplies s[row] back in fp32.
    // The GEMM's K is in bytes, two per fp16 element.
    const EpiRowScale epi{scale, bias, bias_code};
    const auto* A = reinterpret_cast<const uint8_t*>(xh);
    const auto* B = static_cast<const uint8_t*>(w);
    if (out_code == kF32) {
        launch_gemm_simt<SimtF16Packed, EpiRowScale, float>(A, B, static_cast<float*>(d), M, N,
                                                            2 * K, N, epi, stream);
    } else if (out_code == kBF16) {
        launch_gemm_simt<SimtF16Packed, EpiRowScale, __bf16>(A, B, static_cast<__bf16*>(d), M, N,
                                                             2 * K, N, epi, stream);
    } else if (out_code == kF16) {
        launch_gemm_simt<SimtF16Packed, EpiRowScale, __half>(A, B, static_cast<__half*>(d), M, N,
                                                             2 * K, N, epi, stream);
    } else {
        throw std::runtime_error("fp16_packed_linear: unsupported output dtype");
    }
    return true;
}

// ConvRot INT8 linear on the packed GEMM: x (in_code, with input activation act_code,
// as launch_quantize_int8_convrot_kernel) is rotated into fp16 scratch x16 with row
// scales row_scale, then D = (x16 @ w^T) * row_scale[row] * w_scale[col] + bias, the
// GEMM reading the INT8 weight w where it lies and widening it to fp16 itself. Where the
// GEMM takes its square tile, x16 is re-staged into the tiled layout in the scratch x16t,
// which holds M rounded up to a multiple of 128 rows. Returns false, having launched
// nothing, where this device or shape has no such route; the caller keeps the INT8 GEMM.
extern "C" bool launch_int8_convrot_packed_linear_kernel(
    const void* x, int in_code, int act_code, const void* act_weight, float act_eps,
    const void* w, const void* w_scale, int w_scale_stride, const void* bias, int bias_code,
    void* x16, void* x16t, void* row_scale, void* d, int out_code, int M, int N, int K,
    hipStream_t stream) {

    using namespace comfy::hip_backend;

    if (M <= 8 || N == 0 || K % kConvRotGroup256 != 0) {
        return false;
    }
    check_convrot_act(act_code);
    auto* xh = static_cast<__half*>(x16);
    auto* scale = static_cast<float*>(row_scale);
    if (!launch_convrot_rotate_f16(x, in_code, xh, scale, M, K, act_code, act_weight, act_eps,
                                   row_bound(127.0f), stream)) {
        return false;
    }

    // INT8 weights times their per-channel scale are the dequantized weight, so the
    // INT8 codes widened to fp16 (exact, |w| <= 127) take the place of the fp16 weight
    // and EpiRowwise applies both the row's power of two and w_scale.
    const EpiRowwise epi{scale, static_cast<const float*>(w_scale), w_scale_stride, bias,
                         bias_code};
    const bool tiled = simt_square_tile(M, N);
    if (tiled) {
        const int64_t chunks = static_cast<int64_t>(M) * (K / 8);
        tile_rows_f16_kernel<<<static_cast<unsigned>((chunks + 255) / 256), 256, 0, stream>>>(
            xh, static_cast<__half*>(x16t), M, K);
    }
    const auto* A = reinterpret_cast<const uint8_t*>(tiled ? x16t : x16);
    const auto* B = static_cast<const uint8_t*>(w);
    const auto gemm = [&]<typename OutT>(OutT* out) {
        if (tiled) {
            launch_gemm_simt_tiled<SimtF16Packed, EpiRowwise, OutT, true>(A, B, out, M, N, 2 * K, N,
                                                                         epi, stream);
        } else {
            launch_gemm_simt<SimtF16Packed, EpiRowwise, OutT, const uint8_t*, true>(
                A, B, out, M, N, 2 * K, N, epi, stream);
        }
    };
    if (out_code == kF32) {
        gemm(static_cast<float*>(d));
    } else if (out_code == kBF16) {
        gemm(static_cast<__bf16*>(d));
    } else if (out_code == kF16) {
        gemm(static_cast<__half*>(d));
    } else {
        throw std::runtime_error("int8_convrot_packed_linear: unsupported output dtype");
    }
    return true;
}
