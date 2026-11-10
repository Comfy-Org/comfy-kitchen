/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// group_norm_silu_pad3d: per-frame GroupNorm + SiLU + causal-conv padding (reflect
// in space, zero frames in front) in one pass, NDHWC in and out.
//   stats_partial:  per (frame, pixel chunk, channel tile) sum / sum of squares
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
inline int stats_chunk_rows(int frames, int C, int64_t rows) {
    // Keep enough blocks for small frames without excessive partials on large ones.
    int chunk_rows = 64;
    while (chunk_rows < 1024 && rows * frames * ((C + 63) / 64) > 512 * chunk_rows) chunk_rows *= 2;
    return chunk_rows;
}

template <typename T> __device__ __forceinline__ float to_f(T v);
template <> __device__ __forceinline__ float to_f<half>(half v) { return __half2float(v); }
template <> __device__ __forceinline__ float to_f<nv_bfloat16>(nv_bfloat16 v) { return __bfloat162float(v); }
template <typename T> __device__ __forceinline__ float2 to_pair(uint32_t raw);
template <> __device__ __forceinline__ float2 to_pair<half>(uint32_t raw) {
    return __half22float2(*reinterpret_cast<const half2*>(&raw));
}
template <> __device__ __forceinline__ float2 to_pair<nv_bfloat16>(uint32_t raw) {
    return __bfloat1622float2(*reinterpret_cast<const nv_bfloat162*>(&raw));
}
template <typename T> __device__ __forceinline__ uint32_t from_pair(float2 f);
template <> __device__ __forceinline__ uint32_t from_pair<half>(float2 f) {
    const half2 pair = __float22half2_rn(f);
    return *reinterpret_cast<const uint32_t*>(&pair);
}
template <> __device__ __forceinline__ uint32_t from_pair<nv_bfloat16>(float2 f) {
    const nv_bfloat162 pair = __float22bfloat162_rn(f);
    return *reinterpret_cast<const uint32_t*>(&pair);
}

__device__ __forceinline__ int reflect_index(int i, int n) {
    if (i < 0) return -i;
    if (i >= n) return n - 2 - (i - n);
    return i;
}

template <typename T>
__global__ void __launch_bounds__(kThreads)
stats_partial_kernel(const T* __restrict__ x, float2* __restrict__ partial,
                     int64_t rows, int C, int chunks, int W, int64_t row_stride, int chunk_rows) {
    // rows counts pixels; an image row of W pixels is row_stride elements apart (W * C when packed)
    // Split channels to expose independent blocks while retaining vectorized input loads.
    const int cv = min(C / kVec, 8);
    const int rows_per_iter = blockDim.x / cv;  // launched with the largest multiple of cv <= kThreads
    const int frame = blockIdx.y;
    const int chunk = blockIdx.x;
    const int cvec = blockIdx.z * cv + threadIdx.x % cv;
    const int r0 = threadIdx.x / cv;
    const int64_t row_begin = static_cast<int64_t>(chunk) * chunk_rows;
    const int64_t row_end = min(row_begin + chunk_rows, rows);
    const T* base = x + frame * (rows / W) * row_stride + cvec * kVec;

    float s[kVec], ss[kVec];
#pragma unroll
    for (int i = 0; i < kVec; ++i) { s[i] = 0.f; ss[i] = 0.f; }
    for (int64_t r = row_begin + r0; r < row_end && cvec < C / kVec; r += rows_per_iter) {
        uint4 raw = *reinterpret_cast<const uint4*>(base + (row_stride == static_cast<int64_t>(W) * C ? r * C : (r / W) * row_stride + (r % W) * C));
        const T* v = reinterpret_cast<const T*>(&raw);
#pragma unroll
        for (int i = 0; i < kVec; ++i) {
            float f = to_f(v[i]);
            s[i] += f;
            ss[i] += f * f;
        }
    }

    // Padding avoids bank conflicts when threads reduce individual channels.
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
             int T_out, int H_out, int W_out, int left, int top, int front, int64_t row_stride, int pixels_per_thread, uint32_t cpg_reciprocal) {
    extern __shared__ float sh[];
    float* a = sh;
    float* b = sh + C + C / kVec;
    const int cv = C / kVec;
    const int oframe = blockIdx.y;
    const int t = oframe % T_out - front;
    const int iframe = oframe / T_out * T_in + t;
    // Row grids avoid width division; linear grids preserve large RTX throughput.
    const int c0 = (threadIdx.x % cv) * kVec;
    const int step = blockDim.x / cv;
    const int64_t pixel0 = static_cast<int64_t>(blockIdx.x) * step * pixels_per_thread + threadIdx.x / cv;
    if (kNorm && t >= 0) {
        const int cpg = C / G;
        for (int c = threadIdx.x; c < C; c += blockDim.x) {
            uint32_t group = cpg == 1 ? c : __umulhi(static_cast<uint32_t>(c), cpg_reciprocal);
            group += static_cast<uint32_t>(c) - group * cpg >= static_cast<uint32_t>(cpg);
            const float2 st = stats[iframe * G + group];
            const int ci = c + c / kVec;
            a[ci] = st.y * to_f(gamma[c]);
            b[ci] = to_f(beta[c]) - st.x * a[ci];
        }
        __syncthreads();
    }
    const bool linear = gridDim.z == 1;
    const int64_t plane_pixels = linear ? static_cast<int64_t>(H_out) * W_out : W_out;
    for (uint32_t row = blockIdx.z; row < (linear ? 1u : static_cast<uint32_t>(H_out)); row += gridDim.z) {
#pragma unroll 1
        for (int j = 0; j < pixels_per_thread; ++j) {
            const int64_t pixel = pixel0 + j * step;
            if (pixel >= plane_pixels) break;
            const int y0 = static_cast<int>(row) + (linear ? static_cast<int>(pixel / W_out) : 0) - top;
            const int x0 = static_cast<int>(linear ? pixel % W_out : pixel) - left;
            const int y = kZeroPad ? y0 : reflect_index(y0, H);
            T* dst = out + ((static_cast<int64_t>(oframe) * H_out + row) * W_out + pixel) * C + c0;
            if (t < 0 || (kZeroPad && (y0 < 0 || y0 >= H || x0 < 0 || x0 >= W))) {
                *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
                continue;
            }
            const int xx = kZeroPad ? x0 : reflect_index(x0, W);
            const T* src = x + (static_cast<int64_t>(iframe) * H + y) * row_stride + static_cast<int64_t>(xx) * C + c0;
            uint4 raw = *reinterpret_cast<const uint4*>(src);
            uint32_t* v = reinterpret_cast<uint32_t*>(&raw);
#pragma unroll
            for (int i = 0; i < kVec / 2; ++i) {
                float2 f = to_pair<T>(v[i]);
                if (kNorm) {
                    const int ci = c0 + c0 / kVec + i * 2;
                    f.x = a[ci] * f.x + b[ci];
                    f.y = a[ci + 1] * f.y + b[ci + 1];
                    f = to_pair<T>(from_pair<T>(f));
                }
                if (kSilu) {
                    f.x = f.x / (1.f + expf(-f.x));
                    f.y = f.y / (1.f + expf(-f.y));
                }
                v[i] = from_pair<T>(f);
            }
            *reinterpret_cast<uint4*>(dst) = raw;
        }
    }
}

template <typename T>
void stats_typed(const T* x, float2* partial, float2* stats, int frames, int C, int H, int W, int G, float eps,
                 int64_t row_stride, cudaStream_t stream) {
    const int64_t rows = static_cast<int64_t>(H) * W;
    const int chunk_rows = stats_chunk_rows(frames, C, rows);
    const int chunks = static_cast<int>((rows + chunk_rows - 1) / chunk_rows);
    const int cv = std::min(C / kVec, 8);
    stats_partial_kernel<T><<<dim3(chunks, frames, (C / kVec + cv - 1) / cv), (kThreads / cv) * cv, 0, stream>>>(
        x, partial, rows, C, chunks, W, row_stride, chunk_rows);
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
    const int chunk_rows = stats_chunk_rows(frames, C, rows);
    const int chunks = static_cast<int>((rows + chunk_rows - 1) / chunk_rows);
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
    int pixels_per_thread = 1;
    bool row_grid = true;
    if (static_cast<int64_t>(H_out) * W_out * cv >= 131072) {
        int device = 0, major = 0;
        if (cudaGetDevice(&device) != cudaSuccess ||
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess) {
            throw std::runtime_error("group_norm_silu_pad3d: device capability query failed");
        }
        if (major >= 9 && major < 12) pixels_per_thread = major >= 10 && C >= 512 ? 8 : 4;
        else row_grid = static_cast<int64_t>(H_out) * W_out * cv < 1048576;
    }
    const int pixels_per_block = threads / cv * pixels_per_thread;
    const int64_t plane_pixels = row_grid ? W_out : static_cast<int64_t>(H_out) * W_out;
    const dim3 grid(static_cast<unsigned>((plane_pixels + pixels_per_block - 1) / pixels_per_block),
                    B * T_out, row_grid ? std::min(H_out, 65535) : 1);
    const size_t smem = norm ? 2 * (C + C / kVec) * sizeof(float) : 0;
    const uint32_t cpg_reciprocal = norm ? static_cast<uint32_t>((uint64_t{1} << 32) / (C / G)) : 0;
#define LAUNCH_APPLY(N, S, Z)                                                         \
    apply_kernel<T, N, S, Z><<<grid, threads, smem, stream>>>(                       \
        x, stats, gamma, beta, out, T_in, H, W, C, G, T_out, H_out, W_out, left, top, front, row_stride, pixels_per_thread, cpg_reciprocal)
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
// B*T*ceil(H*W/chunk_rows)*C float2. Feeds launch_group_norm_silu_pad3d's stats_in for strips.
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
// workspace (B*T*ceil(H*W/chunk_rows)*C + B*T*G) float2, unused without gamma or with stats_in;
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
