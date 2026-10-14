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
#include "../launchers.h"

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

// fp16 rows x [R, K] into the GEMM's tiled layout (simt_tiled_offset) of their row-paired
// form (SimtF16Pairs), where out holds ceil(R / 128) * 128 rows. One thread per 16-byte
// chunk of the output, so K is a multiple of 8.
__global__ __launch_bounds__(256) void tile_rows_f16_kernel(
    const __half* __restrict__ x, __half* __restrict__ out, int R, int K) {
    const int64_t chunk = static_cast<int64_t>(blockIdx.x) * 256 + threadIdx.x;
    const int row = static_cast<int>(chunk / (K / 8)), k = static_cast<int>(chunk % (K / 8)) * 8;
    if (row >= ((R + 1) & ~1)) return;
    // Row 2p of the output takes the even K of rows 2p and 2p + 1, row 2p + 1 the odd K.
    const int r0 = row & ~1;
    const uint4 a = *reinterpret_cast<const uint4*>(x + static_cast<int64_t>(r0) * K + k);
    const uint4 b = r0 + 1 < R ? *reinterpret_cast<const uint4*>(x + static_cast<int64_t>(r0 + 1) * K + k)
                               : make_uint4(0, 0, 0, 0);
    const auto pair = [&](uint32_t wa, uint32_t wb) {
        return (row & 1) ? (wa >> 16) | (wb & 0xFFFF0000u) : (wa & 0xFFFFu) | (wb << 16);
    };
    *reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(out) + simt_tiled_offset(row, 2 * k, 2 * K)) =
        make_uint4(pair(a.x, b.x), pair(a.y, b.y), pair(a.z, b.z), pair(a.w, b.w));
}

// GGUF Q8_0 blocks (an fp16 scale then 32 int8 codes, 34 bytes, kQ8Block elements of a
// row) into the BI8 GEMM's operands: the codes as a row-major int8 [N, K] and the scales
// as [K / 32, N], so one stage's scales for a tile's columns are contiguous. One thread
// per block; a block is only 2-byte aligned, so it is read as 17 halfwords.
constexpr int kQ8Block = 32;
__global__ __launch_bounds__(256) void repack_q8_0_kernel(
    const uint16_t* __restrict__ raw, uint4* __restrict__ codes, __half* __restrict__ scales,
    int N, int blocks_per_row) {
    const int64_t b = static_cast<int64_t>(blockIdx.x) * 256 + threadIdx.x;
    if (b >= static_cast<int64_t>(N) * blocks_per_row) return;
    const int n = static_cast<int>(b / blocks_per_row), kb = static_cast<int>(b % blocks_per_row);
    const uint16_t* src = raw + b * 17;
    uint32_t w[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) w[i] = src[1 + 2 * i] | static_cast<uint32_t>(src[2 + 2 * i]) << 16;
    codes[b * 2] = make_uint4(w[0], w[1], w[2], w[3]);
    codes[b * 2 + 1] = make_uint4(w[4], w[5], w[6], w[7]);
    scales[static_cast<int64_t>(kb) * N + n] = __ushort_as_half(src[0]);
}

}  // namespace
}  // namespace comfy::hip_backend

// Linear against a GGUF Q8_0 weight (raw blocks [N, K / 32, 34 bytes]) on the tiled
// SimtF16Pairs GEMM, reading the int8 codes in place (BI8) and applying each block's scale
// as the kernel folds the stage it covers (BLOCK SCALES in gemm_simt.h): half the weight
// bytes of a dequantized fp16 weight, and no dequantization pass. x is scaled per row
// against |code| <= 127 as the ConvRot path does, into x16, then re-staged tiled into x16t
// (M rounded up to a multiple of 128 rows). codes ([N, K] int8) and scales ([K / 32, N]
// fp16) are scratch for the repacked weight. Returns false, having launched nothing,
// where the GEMM would not take its square tile; the caller dequantizes instead.
extern "C" bool launch_q8_0_packed_linear_kernel(
    const void* x, int x_code, const void* raw, const void* w_amax, void* x16, void* x16t,
    void* row_scale, void* codes, void* scales, const void* bias, int bias_code, void* d,
    int out_code, int M, int N, int K, hipStream_t stream) {

    using namespace comfy::hip_backend;

    if (M == 0 || N == 0) {
        return true;
    }
    if (K % kQ8Block != 0 || N % 8 != 0 || !simt_square_tile(M, N)) {
        return false;
    }
    const int blocks = K / kQ8Block;
    const int64_t total = static_cast<int64_t>(N) * blocks;
    repack_q8_0_kernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0, stream>>>(
        static_cast<const uint16_t*>(raw), static_cast<uint4*>(codes), static_cast<__half*>(scales),
        N, blocks);
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
        throw std::runtime_error("q8_0_packed_linear: unsupported input dtype");
    }
    const int64_t chunks = static_cast<int64_t>((M + 1) & ~1) * (K / 8);
    tile_rows_f16_kernel<<<static_cast<unsigned>((chunks + 255) / 256), 256, 0, stream>>>(
        xh, static_cast<__half*>(x16t), M, K);

    const EpiRowScaleBlocks epi{scale, bias, bias_code, static_cast<const __half*>(scales)};
    const auto* A = static_cast<const uint8_t*>(x16t);
    const auto* B = static_cast<const uint8_t*>(codes);
    const auto gemm = [&]<typename OutT>(OutT* out) {
        launch_gemm_simt_tiled<SimtF16Pairs, EpiRowScaleBlocks, OutT, true>(A, B, out, M, N, 2 * K, N,
                                                                            epi, stream);
    };
    if (out_code == kF32) {
        gemm(static_cast<float*>(d));
    } else if (out_code == kBF16) {
        gemm(static_cast<__bf16*>(d));
    } else if (out_code == kF16) {
        gemm(static_cast<__half*>(d));
    } else {
        throw std::runtime_error("q8_0_packed_linear: unsupported output dtype");
    }
    return true;
}

// x16t, scratch for M rounded up to a multiple of 128 fp16 rows or null, takes the scaled
// rows in the GEMM's tiled, row-paired layout where the GEMM takes its square tile, as the
// ConvRot path does. Measured on gfx1010 against the row-major SimtF16Packed GEMM (time
// ratio): 0.83-0.94 from K = 640 (Z-Image's FFN linear, M=4224 K=3840 N=10240: 49.0 ms
// against 55.5), 1.03-1.06 at K = 320-512, where the re-staging pass is a larger share of a
// short GEMM, so those stay row-major. SimtF16Pairs folds its fp16 chains after the same
// 32 products, so the row scaling's bound is unchanged.
extern "C" bool launch_fp16_packed_linear_kernel(
    const void* x, int x_code, const void* w, const void* w_amax, void* x16, void* x16t,
    void* row_scale, const void* bias, int bias_code, void* d, int out_code, int M, int N,
    int K, hipStream_t stream) {

    using namespace comfy::hip_backend;

    if (M == 0 || N == 0) {
        return true;  // a zero-dimension grid is hipErrorInvalidConfiguration
    }
    // Rows are staged 16 bytes (8 halves) at a time.
    if (K % 8 != 0) {
        return false;
    }
    if (M <= kGemvF16MaxRows) {
        launch_gemv_f16_kernel(x, x_code, w, bias, bias_code, d, out_code, M, N, K, stream);
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
    // the tiled layout stages 64 bytes of K at a time
    const bool tiled = x16t != nullptr && K >= 640 && K % 32 == 0 && simt_square_tile(M, N);
    if (tiled) {
        const int64_t chunks = static_cast<int64_t>((M + 1) & ~1) * (K / 8);
        tile_rows_f16_kernel<<<static_cast<unsigned>((chunks + 255) / 256), 256, 0, stream>>>(
            xh, static_cast<__half*>(x16t), M, K);
    }
    const auto* A = reinterpret_cast<const uint8_t*>(tiled ? x16t : x16);
    const auto* B = static_cast<const uint8_t*>(w);
    const auto gemm = [&]<typename OutT>(OutT* out) {
        if (tiled) {
            launch_gemm_simt_tiled<SimtF16Pairs, EpiRowScale, OutT>(A, B, out, M, N, 2 * K, N, epi,
                                                                    stream);
        } else {
            launch_gemm_simt<SimtF16Packed, EpiRowScale, OutT>(A, B, out, M, N, 2 * K, N, epi, stream);
        }
    };
    if (out_code == kF32) {
        gemm(static_cast<float*>(d));
    } else if (out_code == kBF16) {
        gemm(static_cast<__bf16*>(d));
    } else if (out_code == kF16) {
        gemm(static_cast<__half*>(d));
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
        const int64_t chunks = static_cast<int64_t>((M + 1) & ~1) * (K / 8);
        tile_rows_f16_kernel<<<static_cast<unsigned>((chunks + 255) / 256), 256, 0, stream>>>(
            xh, static_cast<__half*>(x16t), M, K);
    }
    const auto* A = reinterpret_cast<const uint8_t*>(tiled ? x16t : x16);
    const auto* B = static_cast<const uint8_t*>(w);
    const auto gemm = [&]<typename OutT>(OutT* out) {
        if (tiled) {
            launch_gemm_simt_tiled<SimtF16Pairs, EpiRowwise, OutT, true>(A, B, out, M, N, 2 * K, N,
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
