/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// group_norm_silu_pad3d: per-frame GroupNorm + SiLU + causal-conv padding (reflect
// in space, zero frames in front) in one pass, NDHWC in and out.
//   stats_partial:  per (frame, 1024-row chunk, channel tile) sum / sum of squares
//   stats_finalize: reduce to per-group mean, rstd
//   apply:          normalize, silu, store into the padded output (reflected indices)
// C % 8 == 0 (16-byte vectors), C <= 2048. fp32 partial sums reduced in double, so results
// agree with torch's Welford to fp32 rounding, not bitwise.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstdint>
#include <algorithm>

#include "dtype_dispatch.cuh"

namespace {

constexpr int kThreads = 256;
constexpr int kVec = 8;            // 16-byte vectors of half / bf16
constexpr int kChunkRows = 1024;   // rows per statistics block

template <typename T> __device__ __forceinline__ float to_f(T v);
template <> __device__ __forceinline__ float to_f<half>(half v) { return __half2float(v); }
template <> __device__ __forceinline__ float to_f<nv_bfloat16>(nv_bfloat16 v) { return __bfloat162float(v); }
template <typename T> __device__ __forceinline__ T from_f(float v);
template <> __device__ __forceinline__ half from_f<half>(float v) { return __float2half_rn(v); }
template <> __device__ __forceinline__ nv_bfloat16 from_f<nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }

__device__ __forceinline__ int reflect_index(int i, int n) {
    if (i < 0) return -i;
    if (i >= n) return 2 * n - 2 - i;
    return i;
}

template <typename T>
__global__ void __launch_bounds__(kThreads)
stats_partial_kernel(const T* __restrict__ x, float2* __restrict__ partial,
                     int64_t rows, int C, int chunks, int W, int64_t row_stride) {
    // rows counts pixels; an image row of W pixels is row_stride elements apart (W * C when packed)
    // Tile channels so wide frames expose more blocks without increasing workspace.
    const int cv = min(C / kVec, 16);
    const int rows_per_iter = blockDim.x / cv;  // launched with the largest multiple of cv <= kThreads
    const int frame = blockIdx.y;
    const int chunk = blockIdx.x;
    const int cvec = blockIdx.z * cv + threadIdx.x % cv;
    const int r0 = threadIdx.x / cv;
    const int64_t row_begin = static_cast<int64_t>(chunk) * kChunkRows;
    const int64_t row_end = min(row_begin + kChunkRows, rows);
    const T* base = x + frame * (rows / W) * row_stride + cvec * kVec;

    float s[kVec], ss[kVec];
#pragma unroll
    for (int i = 0; i < kVec; ++i) { s[i] = 0.f; ss[i] = 0.f; }
    for (int64_t r = row_begin + r0; r < row_end && cvec < C / kVec; r += rows_per_iter) {
        uint4 raw = *reinterpret_cast<const uint4*>(base +
            (row_stride == static_cast<int64_t>(W) * C ? r * C : (r / W) * row_stride + (r % W) * C));
        const T* v = reinterpret_cast<const T*>(&raw);
#pragma unroll
        for (int i = 0; i < kVec; ++i) {
            float f = to_f(v[i]);
            s[i] += f;
            ss[i] += f * f;
        }
    }

    // Padding spreads neighboring channels over distinct banks during the reduction.
    __shared__ float sh_s[kVec][kThreads + 4];
    __shared__ float sh_ss[kVec][kThreads + 4];
#pragma unroll
    for (int i = 0; i < kVec; ++i) { sh_s[i][threadIdx.x] = s[i]; sh_ss[i][threadIdx.x] = ss[i]; }
    __syncthreads();
    const int c = blockIdx.z * cv * kVec + threadIdx.x;
    if (threadIdx.x < cv * kVec && c < C) {
        const int v = threadIdx.x / kVec;
        const int i = threadIdx.x % kVec;
        float ts = 0.f, tss = 0.f;
        for (int j = 0; j < rows_per_iter; ++j) {
            ts += sh_s[i][j * cv + v];
            tss += sh_ss[i][j * cv + v];
        }
        partial[(static_cast<int64_t>(frame) * chunks + chunk) * C + c] = make_float2(ts, tss);
    }
}

// One block per group; keep FP64 accumulation while reducing in parallel.
__global__ void stats_finalize_kernel(const float2* __restrict__ partial, float2* __restrict__ stats,
                                      int C, int G, int chunks, double n_per_group, float eps) {
    const int frame = blockIdx.x;
    const int g = blockIdx.y;
    const int cpg = C / G;
    double s = 0.0, ss = 0.0;
    for (int i = threadIdx.x; i < chunks * cpg; i += blockDim.x) {
        const float2 p = partial[(static_cast<int64_t>(frame) * chunks + i / cpg) * C + g * cpg + i % cpg];
        s += p.x;
        ss += p.y;
    }
    for (int offset = 16; offset; offset /= 2) {
        s += __shfl_down_sync(0xffffffff, s, offset);
        ss += __shfl_down_sync(0xffffffff, ss, offset);
    }
    __shared__ double2 warp_sums[4];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    if (lane == 0) warp_sums[warp] = make_double2(s, ss);
    __syncthreads();
    if (warp == 0) {
        s = lane < 4 ? warp_sums[lane].x : 0.0;
        ss = lane < 4 ? warp_sums[lane].y : 0.0;
        for (int offset = 16; offset; offset /= 2) {
            s += __shfl_down_sync(0xffffffff, s, offset);
            ss += __shfl_down_sync(0xffffffff, ss, offset);
        }
        if (lane == 0) {
            const double mean = s / n_per_group;
            const double var = fmax(ss / n_per_group - mean * mean, 0.0);
            stats[frame * G + g] = make_float2(static_cast<float>(mean), rsqrtf(static_cast<float>(var) + eps));
        }
    }
}

template <typename T, bool kNorm, bool kSilu, bool kZeroPad>
__global__ void __launch_bounds__(kThreads)
apply_kernel(const T* __restrict__ x, const float2* __restrict__ stats,
             const T* __restrict__ gamma, const T* __restrict__ beta, T* __restrict__ out,
             int T_in, int H, int W, int C, int G,
             int T_out, int H_out, int W_out, int left, int top, int front, int64_t row_stride) {
    extern __shared__ float sh[];
    float* a = sh;
    float* b = sh + C;

    const int cv = C / kVec;
    const int oframe = blockIdx.y;
    const int bidx = oframe / T_out;
    const int t = oframe % T_out - front;
    const int64_t out_rows = static_cast<int64_t>(H_out) * W_out;
    const int64_t orow = static_cast<int64_t>(blockIdx.x) * (blockDim.x / cv) + threadIdx.x / cv;
    const int c0 = (threadIdx.x % cv) * kVec;
    const bool in_range = orow < out_rows;
    T* dst = in_range ? out + (oframe * out_rows + orow) * C + c0 : nullptr;

    if (t < 0) {
        if (in_range) *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
        return;
    }
    const int iframe = bidx * T_in + t;
    if (kNorm) {
        const int cpg = C / G;
        for (int c = threadIdx.x; c < C; c += blockDim.x) {
            const float2 st = stats[iframe * G + c / cpg];
            a[c] = st.y * to_f(gamma[c]);
            b[c] = to_f(beta[c]) - st.x * a[c];
        }
        __syncthreads();
    }
    if (!in_range) return;

    const int y0 = static_cast<int>(orow) / W_out - top;
    const int x0 = static_cast<int>(orow) % W_out - left;
    if (kZeroPad && (y0 < 0 || y0 >= H || x0 < 0 || x0 >= W)) {
        // zero border, for models whose convolutions pad with zeros rather than reflecting
        *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
        return;
    }
    const int y = kZeroPad ? y0 : reflect_index(y0, H);
    const int xx = kZeroPad ? x0 : reflect_index(x0, W);
    const T* src = x + (static_cast<int64_t>(iframe) * H + y) * row_stride + static_cast<int64_t>(xx) * C + c0;

    uint4 raw = *reinterpret_cast<const uint4*>(src);
    T* v = reinterpret_cast<T*>(&raw);
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
        float f = to_f(v[i]);
        if (kNorm) f = to_f(from_f<T>(a[c0 + i] * f + b[c0 + i]));  // round like the eager GroupNorm output
        if (kSilu) f = f / (1.f + expf(-f));
        v[i] = from_f<T>(f);
    }
    *reinterpret_cast<uint4*>(dst) = raw;
}

template <typename T>
void stats_typed(const T* x, float2* partial, float2* stats, int frames, int C, int H, int W, int G, float eps,
                 int64_t row_stride, cudaStream_t stream) {
    const int64_t rows = static_cast<int64_t>(H) * W;
    const int chunks = static_cast<int>((rows + kChunkRows - 1) / kChunkRows);
    const int cv = std::min(C / kVec, 16);
    stats_partial_kernel<T><<<dim3(chunks, frames, (C / kVec + cv - 1) / cv), (kThreads / cv) * cv, 0, stream>>>(
        x, partial, rows, C, chunks, W, row_stride);
    stats_finalize_kernel<<<dim3(frames, G), 128, 0, stream>>>(
        partial, stats, C, G, chunks, static_cast<double>(rows) * (C / G), eps);
}

template <typename T>
void launch_typed(const T* x, const T* gamma, const T* beta, T* out, float2* workspace, const float2* stats_in,
                  int B, int C, int T_in, int H, int W, int G, float eps,
                  int left, int right, int top, int bottom, int front, bool silu, bool zero_pad,
                  int64_t row_stride, cudaStream_t stream) {
    const bool norm = gamma != nullptr;
    const int frames = B * T_in;
    const int64_t rows = static_cast<int64_t>(H) * W;
    const int chunks = static_cast<int>((rows + kChunkRows - 1) / kChunkRows);
    const float2* stats = stats_in;

    if (norm && stats_in == nullptr) {
        // whole-frame statistics; a caller normalizing strips of a frame passes its own
        float2* own = workspace + static_cast<int64_t>(frames) * chunks * C;
        stats_typed<T>(x, workspace, own, frames, C, H, W, G, eps, row_stride, stream);
        stats = own;
    }

    const int T_out = T_in + front;
    const int H_out = H + top + bottom;
    const int W_out = W + left + right;
    const int cv = C / kVec;
    const int threads = (kThreads / cv) * cv;
    const int rows_per_block = threads / cv;
    const int64_t out_rows = static_cast<int64_t>(H_out) * W_out;
    const dim3 grid(static_cast<unsigned>((out_rows + rows_per_block - 1) / rows_per_block), B * T_out);
    const size_t smem = norm ? 2 * C * sizeof(float) : 0;
#define LAUNCH_APPLY(N, S, Z)                                                         \
    apply_kernel<T, N, S, Z><<<grid, threads, smem, stream>>>(                       \
        x, stats, gamma, beta, out, T_in, H, W, C, G, T_out, H_out, W_out, left, top, front, row_stride)
    if (zero_pad) {
        if (norm && silu) LAUNCH_APPLY(true, true, true);
        else if (norm) LAUNCH_APPLY(true, false, true);
        else if (silu) LAUNCH_APPLY(false, true, true);
        else LAUNCH_APPLY(false, false, true);
    } else {
        if (norm && silu) LAUNCH_APPLY(true, true, false);
        else if (norm) LAUNCH_APPLY(true, false, false);
        else if (silu) LAUNCH_APPLY(false, true, false);
        else LAUNCH_APPLY(false, false, false);
    }
#undef LAUNCH_APPLY
}

}  // namespace

// Per-frame GroupNorm statistics only: stats [B*T*G] float2 (mean, rstd), workspace
// B*T*ceil(H*W/1024)*C float2. Feeds launch_group_norm_silu_pad3d's stats_in for strips.
// row_stride: elements between image rows (W * C when packed; larger for a row-interleaved view).
extern "C" void launch_group_norm_stats3d(
    const void* x, void* workspace, void* stats,
    int B, int C, int T, int H, int W, int G, float eps, int64_t row_stride,
    int dtype_code, cudaStream_t stream) {
    if (C % kVec != 0 || C / kVec > kThreads || C % G != 0 || G > 1024) {
        throw std::runtime_error("group_norm_stats3d: unsupported channel/group count");
    }
    if (static_cast<int64_t>(B) * T > 65535) {
        throw std::runtime_error("group_norm_stats3d: batch * frames must be <= 65535");
    }
    DISPATCH_HALF_DTYPE(dtype_code, DT, [&] {
        stats_typed<DT>(static_cast<const DT*>(x), static_cast<float2*>(workspace), static_cast<float2*>(stats),
                        B * T, C, H, W, G, eps, row_stride, stream);
    });
}

// x/out NDHWC; gamma/beta [C] or both nullptr for pad-only;
// workspace (B*T*ceil(H*W/1024)*C + B*T*G) float2, unused without gamma or with stats_in;
// stats_in [B*T*G] float2 from launch_group_norm_stats3d, or nullptr to compute them here.
extern "C" void launch_group_norm_silu_pad3d(
    const void* x, const void* gamma, const void* beta, void* out, void* workspace, const void* stats_in,
    int B, int C, int T, int H, int W, int G, float eps,
    int left, int right, int top, int bottom, int front, bool silu, bool zero_pad, int64_t row_stride,
    int dtype_code, cudaStream_t stream) {
    if (C % kVec != 0 || C / kVec > kThreads || (gamma != nullptr && (C % G != 0 || G > 1024))) {
        throw std::runtime_error("group_norm_silu_pad3d: unsupported channel/group count");
    }
    if (left < 0 || right < 0 || top < 0 || bottom < 0 || front < 0) {
        throw std::runtime_error("group_norm_silu_pad3d: padding must be non-negative");
    }
    // reflection needs a row to mirror; a zero border reads nothing
    if (!zero_pad && (left >= W || right >= W || top >= H || bottom >= H)) {
        throw std::runtime_error("group_norm_silu_pad3d: reflect padding must be smaller than the input");
    }
    // frames ride on grid.y; an oversize launch would fail silently and return `out` uninitialized
    if (static_cast<int64_t>(B) * (T + front) > 65535) {
        throw std::runtime_error("group_norm_silu_pad3d: batch * frames must be <= 65535");
    }
    switch (dtype_code) {
        case comfy::DTYPE_CODE_FLOAT16:
            launch_typed<half>(static_cast<const half*>(x), static_cast<const half*>(gamma),
                               static_cast<const half*>(beta), static_cast<half*>(out),
                               static_cast<float2*>(workspace), static_cast<const float2*>(stats_in),
                               B, C, T, H, W, G, eps,
                               left, right, top, bottom, front, silu, zero_pad, row_stride, stream);
            break;
        case comfy::DTYPE_CODE_BFLOAT16:
            launch_typed<nv_bfloat16>(static_cast<const nv_bfloat16*>(x), static_cast<const nv_bfloat16*>(gamma),
                                      static_cast<const nv_bfloat16*>(beta), static_cast<nv_bfloat16*>(out),
                                      static_cast<float2*>(workspace), static_cast<const float2*>(stats_in),
                                      B, C, T, H, W, G, eps,
                                      left, right, top, bottom, front, silu, zero_pad, row_stride, stream);
            break;
        default:
            throw std::runtime_error("group_norm_silu_pad3d: only float16 and bfloat16 are supported");
    }
}
