// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// INT8 linear for wave64 CDNA (gfx942) on MFMA (v_mfma_i32_32x32x16_i8). Reproduces the
// eager int8_linear on bf16 bit for bit: the row-wise quantizer rounds as torch does,
// and the GEMM epilogue is eager's bf16((float)acc * (x_scale * w_scale)) + bias.
#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

// v_mfma_i32_32x32x16_i8 is gfx942 (CDNA3) and later; a new cdna target needs its own schedule.
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx942__)
#error "gemm_int8_mfma.hip is only built for gfx942"
#endif

namespace comfy::hip_backend {
namespace {

typedef int v16i __attribute__((ext_vector_type(16)));

// 256 threads as 2x2 waves; each wave owns IT x JT 32x32 MFMA tiles.
constexpr int IT = 2, JT = 2;
constexpr int BM = 64 * IT, BN = 64 * JT, BK = 64;
// Row pitch in LDS; the 8-byte pad spreads a wave's 8-byte reads over the banks.
constexpr int PITCH = BK + 8;
constexpr int CHUNKS_A = BM * BK / 16 / 256, CHUNKS_B = BN * BK / 16 / 256;
constexpr int kMaxGridY = 65535;

__device__ __forceinline__ float bf16_round(float v) { return __bfloat162float(__float2bfloat16(v)); }

__device__ __forceinline__ void store_16b(int8_t* dst, int4 v) {
    // LDS rows are 8-byte aligned (PITCH), so the 16 loaded bytes go in as two longs.
    long* p = reinterpret_cast<long*>(dst);
    p[0] = (static_cast<long>(static_cast<unsigned>(v.y)) << 32) | static_cast<unsigned>(v.x);
    p[1] = (static_cast<long>(static_cast<unsigned>(v.w)) << 32) | static_cast<unsigned>(v.z);
}

// C[M, N] = bf16(bf16(A[M, K] @ B[N, K]^T * (xs[row] * ws[col])) + bias[col]).
// Needs K % 16 == 0 and 16-byte aligned A and B.
template <bool PER_CHANNEL, bool HAS_BIAS>
__global__ __launch_bounds__(256) void int8_gemm_mfma_kernel(
    const int8_t* __restrict__ A, const int8_t* __restrict__ B, __hip_bfloat16* __restrict__ C,
    const float* __restrict__ xs, const float* __restrict__ ws, const __hip_bfloat16* __restrict__ bias,
    int M, int N, int K) {
    __shared__ __attribute__((aligned(16))) int8_t sA[BM * PITCH];
    __shared__ __attribute__((aligned(16))) int8_t sB[BN * PITCH];

    const int tid = threadIdx.x, lane = tid & 63, wave = tid >> 6;
    const int wm = wave >> 1, wn = wave & 1;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;

    v16i acc[IT][JT];
#pragma unroll
    for (int i = 0; i < IT; ++i)
#pragma unroll
        for (int j = 0; j < JT; ++j)
#pragma unroll
            for (int r = 0; r < 16; ++r) acc[i][j][r] = 0;

    int4 ra[CHUNKS_A], rb[CHUNKS_B];
    auto gload = [&](int k0) {
#pragma unroll
        for (int c = 0; c < CHUNKS_A; ++c) {
            const int idx = tid + c * 256, row = idx / (BK / 16), col = (idx % (BK / 16)) * 16;
            const int gk = k0 + col, gm = m0 + row;
            ra[c] = (gm < M && gk < K) ? *reinterpret_cast<const int4*>(A + static_cast<int64_t>(gm) * K + gk)
                                       : make_int4(0, 0, 0, 0);
        }
#pragma unroll
        for (int c = 0; c < CHUNKS_B; ++c) {
            const int idx = tid + c * 256, row = idx / (BK / 16), col = (idx % (BK / 16)) * 16;
            const int gk = k0 + col, gn = n0 + row;
            rb[c] = (gn < N && gk < K) ? *reinterpret_cast<const int4*>(B + static_cast<int64_t>(gn) * K + gk)
                                       : make_int4(0, 0, 0, 0);
        }
    };
    auto sstore = [&]() {
#pragma unroll
        for (int c = 0; c < CHUNKS_A; ++c) {
            const int idx = tid + c * 256;
            store_16b(sA + (idx / (BK / 16)) * PITCH + (idx % (BK / 16)) * 16, ra[c]);
        }
#pragma unroll
        for (int c = 0; c < CHUNKS_B; ++c) {
            const int idx = tid + c * 256;
            store_16b(sB + (idx / (BK / 16)) * PITCH + (idx % (BK / 16)) * 16, rb[c]);
        }
    };

    gload(0);
    for (int k0 = 0; k0 < K; k0 += BK) {
        sstore();
        __syncthreads();
        if (k0 + BK < K) gload(k0 + BK);  // the next tile's global loads overlap the MFMAs
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            // Operand layout: lane l holds row l % 32, k bytes (l / 32) * 8 .. + 8.
            const int koff = kk + (lane >> 5) * 8;
            long a[IT], b[JT];
#pragma unroll
            for (int i = 0; i < IT; ++i)
                a[i] = *reinterpret_cast<const long*>(sA + (wm * 32 * IT + i * 32 + (lane & 31)) * PITCH + koff);
#pragma unroll
            for (int j = 0; j < JT; ++j)
                b[j] = *reinterpret_cast<const long*>(sB + (wn * 32 * JT + j * 32 + (lane & 31)) * PITCH + koff);
#pragma unroll
            for (int i = 0; i < IT; ++i)
#pragma unroll
                for (int j = 0; j < JT; ++j)
                    acc[i][j] = __builtin_amdgcn_mfma_i32_32x32x16_i8(a[i], b[j], acc[i][j], 0, 0, 0);
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < IT; ++i)
#pragma unroll
        for (int j = 0; j < JT; ++j) {
            const int col = n0 + wn * 32 * JT + j * 32 + (lane & 31);
            if (col >= N) continue;
            const float wsc = PER_CHANNEL ? ws[col] : ws[0];
            const float bv = HAS_BIAS ? __bfloat162float(bias[col]) : 0.0f;
#pragma unroll
            for (int r = 0; r < 16; ++r) {
                // Accumulator register r of lane l is row 8 * (r / 4) + 4 * (l / 32) + r % 4, column l % 32.
                const int row = m0 + wm * 32 * IT + i * 32 + 8 * (r >> 2) + 4 * (lane >> 5) + (r & 3);
                if (row >= M) continue;
                float v = bf16_round(static_cast<float>(acc[i][j][r]) * (xs[row] * wsc));
                if (HAS_BIAS) v = v + bv;
                C[static_cast<int64_t>(row) * N + col] = __float2bfloat16(v);
            }
        }
}

// One block per row. torch computes amax / 127 as amax * float(1 / 127), divides x by the
// scale rounded to bf16 and rounds the quotient to bf16 before torch.round.
__global__ __launch_bounds__(256) void quantize_int8_rowwise_bf16_exact_kernel(
    const __hip_bfloat16* __restrict__ x, int8_t* __restrict__ q, float* __restrict__ scales, int K) {
    __shared__ float red[256];
    const int64_t row = blockIdx.x;
    const int t = threadIdx.x;
    const __hip_bfloat16* xr = x + row * K;

    float m = 0.0f;
    for (int j = t; j < K; j += 256) m = fmaxf(m, fabsf(__bfloat162float(xr[j])));
    red[t] = m;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (t < s) red[t] = fmaxf(red[t], red[t + s]);
        __syncthreads();
    }

    const float scale = fmaxf(red[0] * (1.0f / 127.0f), 1e-30f);
    if (t == 0) scales[row] = scale;
    const float sb = bf16_round(scale);
    int8_t* qr = q + row * K;
    for (int j = t; j < K; j += 256) {
        // rintf is round-half-to-even, matching torch.round.
        const float v = rintf(bf16_round(__fdiv_rn(__bfloat162float(xr[j]), sb)));
        qr[j] = static_cast<int8_t>(fminf(fmaxf(v, -128.0f), 127.0f));
    }
}

}  // namespace
}  // namespace comfy::hip_backend

extern "C" void launch_quantize_int8_rowwise_bf16_exact_kernel(
    const void* x, void* q, void* scales, int M, int K, hipStream_t stream) {
    using namespace comfy::hip_backend;
    if (M == 0) return;  // a zero-block launch is hipErrorInvalidConfiguration
    quantize_int8_rowwise_bf16_exact_kernel<<<M, 256, 0, stream>>>(
        static_cast<const __hip_bfloat16*>(x), static_cast<int8_t*>(q), static_cast<float*>(scales), K);
}

extern "C" void launch_int8_gemm_mfma_kernel(
    const void* a, const void* b, void* c, const void* scale_a, const void* scale_b, int scale_b_stride,
    const void* bias, int M, int N, int K, hipStream_t stream) {
    using namespace comfy::hip_backend;
    if (K % 16 != 0) throw std::runtime_error("int8_gemm_mfma: K must be divisible by 16");
    if (M == 0 || N == 0) return;
    const bool per_channel = scale_b_stride != 0, has_bias = bias != nullptr;
    const int n_tiles = (N + BN - 1) / BN;
    // gridDim.y is capped, so very tall A is launched in row bands.
    for (int64_t m_base = 0; m_base < M; m_base += static_cast<int64_t>(kMaxGridY) * BM) {
        const int rows = static_cast<int>(std::min<int64_t>(M - m_base, static_cast<int64_t>(kMaxGridY) * BM));
        const dim3 grid(n_tiles, (rows + BM - 1) / BM);
        const auto* ap = static_cast<const int8_t*>(a) + m_base * K;
        auto* cp = static_cast<__hip_bfloat16*>(c) + m_base * N;
        const auto* xs = static_cast<const float*>(scale_a) + m_base;
        const auto* ws = static_cast<const float*>(scale_b);
        const auto* bp = static_cast<const __hip_bfloat16*>(bias);
        const auto* bq = static_cast<const int8_t*>(b);
        if (per_channel) {
            if (has_bias) int8_gemm_mfma_kernel<true, true><<<grid, 256, 0, stream>>>(ap, bq, cp, xs, ws, bp, rows, N, K);
            else int8_gemm_mfma_kernel<true, false><<<grid, 256, 0, stream>>>(ap, bq, cp, xs, ws, bp, rows, N, K);
        } else {
            if (has_bias) int8_gemm_mfma_kernel<false, true><<<grid, 256, 0, stream>>>(ap, bq, cp, xs, ws, bp, rows, N, K);
            else int8_gemm_mfma_kernel<false, false><<<grid, 256, 0, stream>>>(ap, bq, cp, xs, ws, bp, rows, N, K);
        }
    }
}
