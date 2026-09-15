// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// group_norm_silu_pad3d: per-frame GroupNorm + SiLU + causal conv padding (reflect
// in space, zero frames in front) in one pass, NDHWC in and out.
//   stats_partial:  per (frame, 1024-row chunk, channel) sum and sum of squares
//   stats_finalize: reduce those to a per-group mean and rstd
//   apply:          normalize, silu, store into the padded output (reflected rows)
// C % 8 == 0, so a register holds 8 elements of a row. Statistics are reduced in
// fp32, so they agree with eager GroupNorm to fp32 rounding rather than bitwise.
#include <stdexcept>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "../rope_math.h"

namespace comfy::hip_backend {

constexpr int kGnThreads = 256;
constexpr int kGnVec = 8;
constexpr int kGnChunkRows = 1024;

template <typename T>
__forceinline__ __device__ float gn_load(T v);
template <>
__forceinline__ __device__ float gn_load<__half>(__half v) {
    return __half2float(v);
}
template <>
__forceinline__ __device__ float gn_load<__bf16>(__bf16 v) {
    return static_cast<float>(v);
}

template <typename T>
__forceinline__ __device__ T gn_store(float v);
template <>
__forceinline__ __device__ __half gn_store<__half>(float v) {
    return __float2half(v);
}
template <>
__forceinline__ __device__ __bf16 gn_store<__bf16>(float v) {
    return static_cast<__bf16>(v);
}

__forceinline__ __device__ int gn_reflect(int i, int n) {
    if (i < 0) return -i;
    if (i >= n) return 2 * n - 2 - i;
    return i;
}

// One block per (chunk, frame); one thread per (row stride, channel vector).
template <typename T>
__global__ __launch_bounds__(kGnThreads) void gn_stats_partial_kernel(
    const T* __restrict__ x, float2* __restrict__ partial, int64_t rows, int C, int chunks) {
    const int cv = C / kGnVec;
    const int rows_per_iter = kGnThreads / cv;
    const int frame = blockIdx.y;
    const int chunk = blockIdx.x;
    const int cvec = threadIdx.x % cv;
    const int r0 = threadIdx.x / cv;
    const int64_t row_begin = static_cast<int64_t>(chunk) * kGnChunkRows;
    const int64_t row_end = row_begin + kGnChunkRows < rows ? row_begin + kGnChunkRows : rows;
    const T* base = x + static_cast<int64_t>(frame) * rows * C + cvec * kGnVec;

    float s[kGnVec];
    float ss[kGnVec];
#pragma unroll
    for (int i = 0; i < kGnVec; ++i) {
        s[i] = 0.0f;
        ss[i] = 0.0f;
    }
    for (int64_t r = row_begin + r0; r < row_end; r += rows_per_iter) {
        uint4 raw = *reinterpret_cast<const uint4*>(base + r * C);
        const T* v = reinterpret_cast<const T*>(&raw);
#pragma unroll
        for (int i = 0; i < kGnVec; ++i) {
            const float f = gn_load<T>(v[i]);
            s[i] += f;
            ss[i] += f * f;
        }
    }

    __shared__ float sh_s[kGnThreads][kGnVec];
    __shared__ float sh_ss[kGnThreads][kGnVec];
#pragma unroll
    for (int i = 0; i < kGnVec; ++i) {
        sh_s[threadIdx.x][i] = s[i];
        sh_ss[threadIdx.x][i] = ss[i];
    }
    __syncthreads();
    if (r0 == 0) {
        float2* dst = partial + (static_cast<int64_t>(frame) * chunks + chunk) * C + cvec * kGnVec;
#pragma unroll
        for (int i = 0; i < kGnVec; ++i) {
            float ts = 0.0f;
            float tss = 0.0f;
            for (int j = 0; j < rows_per_iter; ++j) {
                ts += sh_s[j * cv + cvec][i];
                tss += sh_ss[j * cv + cvec][i];
            }
            dst[i] = make_float2(ts, tss);
        }
    }
}

// One block per frame, one thread per group.
__global__ void gn_stats_finalize_kernel(const float2* __restrict__ partial,
                                         float2* __restrict__ stats, int C, int G, int chunks,
                                         float n_per_group, float eps) {
    const int frame = blockIdx.x;
    const int g = threadIdx.x;
    const int cpg = C / G;
    float s = 0.0f;
    float ss = 0.0f;
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const float2* p = partial + (static_cast<int64_t>(frame) * chunks + chunk) * C + g * cpg;
        for (int c = 0; c < cpg; ++c) {
            s += p[c].x;
            ss += p[c].y;
        }
    }
    const float mean = s / n_per_group;
    const float var = fmaxf(ss / n_per_group - mean * mean, 0.0f);
    stats[static_cast<int64_t>(frame) * G + g] = make_float2(mean, rsqrtf(var + eps));
}

// One block row per output frame, one thread per output channel vector.
template <typename T, bool kNorm, bool kSilu>
__global__ __launch_bounds__(kGnThreads) void gn_apply_kernel(
    const T* __restrict__ x, const float2* __restrict__ stats, const T* __restrict__ gamma,
    const T* __restrict__ beta, T* __restrict__ out, int T_in, int H, int W, int C, int G,
    int T_out, int H_out, int W_out, int left, int top, int front) {
    extern __shared__ float sh[];
    float* a = sh;
    float* b = sh + C;

    const int cv = C / kGnVec;
    const int oframe = blockIdx.y;
    const int bidx = oframe / T_out;
    const int t = oframe % T_out - front;
    const int64_t out_rows = static_cast<int64_t>(H_out) * W_out;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * kGnThreads + threadIdx.x;
    const bool in_range = idx < out_rows * cv;
    T* dst = in_range ? out + (oframe * out_rows + idx / cv) * C + (idx % cv) * kGnVec : nullptr;

    // Every thread of a block shares the frame, so all of them leave here together.
    if (t < 0) {
        if (in_range) *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
        return;
    }
    const int iframe = bidx * T_in + t;
    if (kNorm) {
        const int cpg = C / G;
        for (int c = threadIdx.x; c < C; c += kGnThreads) {
            const float2 st = stats[static_cast<int64_t>(iframe) * G + c / cpg];
            a[c] = st.y * gn_load<T>(gamma[c]);
            b[c] = gn_load<T>(beta[c]) - st.x * a[c];
        }
        __syncthreads();
    }
    if (!in_range) return;

    const int orow = static_cast<int>(idx / cv);
    const int c0 = static_cast<int>(idx % cv) * kGnVec;
    const int y = gn_reflect(orow / W_out - top, H);
    const int xx = gn_reflect(orow % W_out - left, W);
    const T* src = x + ((static_cast<int64_t>(iframe) * H + y) * W + xx) * C + c0;

    uint4 raw = *reinterpret_cast<const uint4*>(src);
    T* v = reinterpret_cast<T*>(&raw);
#pragma unroll
    for (int i = 0; i < kGnVec; ++i) {
        float f = gn_load<T>(v[i]);
        // round_to, not a store/load pair: -ffast-math folds that round trip, and
        // eager GroupNorm materializes its output in x's dtype before the SiLU.
        if (kNorm) f = round_to<T>(a[c0 + i] * f + b[c0 + i]);
        if (kSilu) f = f / (1.0f + expf(-f));
        v[i] = gn_store<T>(f);
    }
    *reinterpret_cast<uint4*>(dst) = raw;
}

template <typename T>
void launch_gn_typed(const T* x, const T* gamma, const T* beta, T* out, float2* workspace, int B,
                     int C, int T_in, int H, int W, int G, float eps, int left, int right,
                     int top, int bottom, int front, bool silu, hipStream_t stream) {
    const bool norm = gamma != nullptr;
    const int frames = B * T_in;
    const int64_t rows = static_cast<int64_t>(H) * W;
    const int chunks = static_cast<int>((rows + kGnChunkRows - 1) / kGnChunkRows);
    float2* partial = workspace;
    float2* stats = workspace + static_cast<int64_t>(frames) * chunks * C;

    if (norm && frames > 0) {
        gn_stats_partial_kernel<T><<<dim3(chunks, frames), kGnThreads, 0, stream>>>(
            x, partial, rows, C, chunks);
        gn_stats_finalize_kernel<<<frames, G, 0, stream>>>(
            partial, stats, C, G, chunks, static_cast<float>(rows * (C / G)), eps);
    }

    const int T_out = T_in + front;
    const int H_out = H + top + bottom;
    const int W_out = W + left + right;
    const int64_t out_vecs = static_cast<int64_t>(H_out) * W_out * (C / kGnVec);
    const dim3 grid(static_cast<unsigned>((out_vecs + kGnThreads - 1) / kGnThreads), B * T_out);
    const size_t smem = norm ? 2 * static_cast<size_t>(C) * sizeof(float) : 0;
#define CK_GN_APPLY(N, S)                                                                      \
    gn_apply_kernel<T, N, S><<<grid, kGnThreads, smem, stream>>>(                              \
        x, stats, gamma, beta, out, T_in, H, W, C, G, T_out, H_out, W_out, left, top, front)
    if (norm && silu) {
        CK_GN_APPLY(true, true);
    } else if (norm) {
        CK_GN_APPLY(true, false);
    } else if (silu) {
        CK_GN_APPLY(false, true);
    } else {
        CK_GN_APPLY(false, false);
    }
#undef CK_GN_APPLY
}

}  // namespace comfy::hip_backend

// x/out NDHWC; gamma/beta [C] in x's dtype, or both null for pad-only; workspace
// holds B*T*(ceil(H*W/1024)*C + G) float2 and is unused without gamma.
extern "C" void launch_group_norm_silu_pad3d(
    const void* x, const void* gamma, const void* beta, void* out, void* workspace, int B, int C,
    int T, int H, int W, int G, float eps, int left, int right, int top, int bottom, int front,
    bool silu, int dtype_code, hipStream_t stream) {

    using namespace comfy::hip_backend;

    if (C <= 0 || C % kGnVec != 0 || kGnThreads % (C / kGnVec) != 0 ||
        (gamma != nullptr && (G <= 0 || C % G != 0 || G > 1024))) {
        throw std::runtime_error("group_norm_silu_pad3d: unsupported channel/group count");
    }
    if (left < 0 || right < 0 || top < 0 || bottom < 0 || front < 0 || left >= W || right >= W ||
        top >= H || bottom >= H) {
        throw std::runtime_error(
            "group_norm_silu_pad3d: padding must be non-negative and smaller than the input");
    }
    // Frames ride on grid.y; an oversize launch fails and leaves `out` uninitialized.
    if (static_cast<int64_t>(B) * (T + front) > 65535) {
        throw std::runtime_error("group_norm_silu_pad3d: batch * frames must be <= 65535");
    }
    if (B == 0 || T + front == 0 || H == 0 || W == 0) {
        return;  // a zero-block launch is hipErrorInvalidConfiguration
    }
    if (dtype_code == 1) {
        launch_gn_typed<__half>(static_cast<const __half*>(x), static_cast<const __half*>(gamma),
                                static_cast<const __half*>(beta), static_cast<__half*>(out),
                                static_cast<float2*>(workspace), B, C, T, H, W, G, eps, left,
                                right, top, bottom, front, silu, stream);
    } else if (dtype_code == 2) {
        launch_gn_typed<__bf16>(static_cast<const __bf16*>(x), static_cast<const __bf16*>(gamma),
                                static_cast<const __bf16*>(beta), static_cast<__bf16*>(out),
                                static_cast<float2*>(workspace), B, C, T, H, W, G, eps, left,
                                right, top, bottom, front, silu, stream);
    } else {
        throw std::runtime_error("group_norm_silu_pad3d: only float16 and bfloat16 are supported");
    }
}
