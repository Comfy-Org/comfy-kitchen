/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Fused AdaLN kernels:
//   adaln     — out = layernorm(x, eps=eps) * (1 + scale) + shift
//   rms_adaln — out = rmsnorm(x, eps=eps)   * (1 + scale) + shift
// Both operations share kernels templated on kSubtractMean with float32
// statistics. The grouped-row path uses fewer warps per row to improve
// throughput at moderate widths; the full-block streaming path is unchanged.
//
// scale/shift are broadcast: in AdaLN they are produced per-sample with shape
// (..., 1, D) and shared across all tokens of a sample. Rather than materialize
// a full (N, D) copy of each (a separate expand+copy kernel plus N*D of extra
// read traffic), the caller passes them in their distinct-row form together
// with `scale_group` / `shift_group` = the number of consecutive output rows
// that share one vector. Row r reads vector (r / group). group == 1 means a
// distinct vector per row (no broadcast).
//
// All global accesses are 128-bit vectorized when the row is 16-byte aligned
// (D % VEC == 0); the scalar tail handles the remainder and any unaligned D.
// Mean and variance are computed in a single pass (accumulating sum and
// sum-of-squares together) so only one block-wide reduction / barrier is needed
// rather than two. The normalize pass re-reads x; per-row reuse can be
// served by L1/L2 instead of fetching the input from DRAM again.

#include "utils.cuh"
#include "dtype_dispatch.cuh"

#include <limits>
#include <stdexcept>

namespace comfy {

namespace {

constexpr int kAdaLNThreads = 256;
constexpr int kAdaLNWarps = kAdaLNThreads / kThreadsPerWarp;  // 8

template<typename T>
__device__ __forceinline__ float to_float(T val);
template<> __device__ __forceinline__ float to_float<float>(float val) { return val; }
template<> __device__ __forceinline__ float to_float<half>(half val) { return __half2float(val); }
template<> __device__ __forceinline__ float to_float<nv_bfloat16>(nv_bfloat16 val) { return __bfloat162float(val); }

template<typename T>
__device__ __forceinline__ T from_float(float val);
template<> __device__ __forceinline__ float from_float<float>(float val) { return val; }
template<> __device__ __forceinline__ half from_float<half>(float val) { return __float2half_rn(val); }
template<> __device__ __forceinline__ nv_bfloat16 from_float<nv_bfloat16>(float val) { return __float2bfloat16_rn(val); }

// Number of T elements packed into a 128-bit vector access.
template<typename T> struct VecWidth { static constexpr int value = 16 / sizeof(T); };

// 16-byte aligned bundle of VEC elements. An aligned load/store of this type
// compiles to a single 128-bit (LD.128 / ST.128) memory transaction.
template<typename T>
struct alignas(16) Vec {
    static constexpr int W = VecWidth<T>::value;
    T elts[W];
};

// Reduce two independent sums at once (here: sum(x) and sum(x*x)), so mean and
// variance need only a single block-wide reduction / barrier instead of two.
__device__ __forceinline__ float2 warp_reduce_sum2(float2 v) {
    for (int offset = kThreadsPerWarp / 2; offset > 0; offset >>= 1) {
        v.x += __shfl_down_sync(0xffffffff, v.x, offset);
        v.y += __shfl_down_sync(0xffffffff, v.y, offset);
    }
    return v;
}

// Block-wide reduction of a (sum, sum-of-squares) pair. Returns the total to
// every thread. `warp_smem` holds one float2 per warp.
__device__ __forceinline__ float2 block_reduce_sum2(float2 v, float2* warp_smem) {
    const int lane = threadIdx.x & (kThreadsPerWarp - 1);
    const int wid  = threadIdx.x >> 5;

    v = warp_reduce_sum2(v);
    if (lane == 0) warp_smem[wid] = v;
    __syncthreads();

    float2 total = make_float2(0.0f, 0.0f);
    for (int w = 0; w < kAdaLNWarps; ++w) { total.x += warp_smem[w].x; total.y += warp_smem[w].y; }
    return total;
}

// Reduce within each row's power-of-two group of warps.
template<int kRowWarps>
__device__ __forceinline__ float2 row_reduce_sum2(float2 v, float2* warp_smem) {
    const int lane = threadIdx.x & (kThreadsPerWarp - 1);
    const int wid = threadIdx.x >> 5;
    v = warp_reduce_sum2(v);
    if constexpr (kRowWarps == 1) {
        return make_float2(__shfl_sync(0xffffffff, v.x, 0),
                          __shfl_sync(0xffffffff, v.y, 0));
    }
    if (lane == 0) warp_smem[wid] = v;
    __syncthreads();
    float2 total = make_float2(0.0f, 0.0f);
    float2* row_smem = warp_smem + (wid / kRowWarps) * kRowWarps;
    for (int w = 0; w < kRowWarps; ++w) {
        total.x += row_smem[w].x;
        total.y += row_smem[w].y;
    }
    return total;
}

// Derive (mean, rstd) from the (sum, sum-of-squares) pair. For RMSNorm the mean
// is pinned to 0 so the shared epilogue below — (x - mean) * rstd — covers both.
template<bool kSubtractMean>
__device__ __forceinline__ void norm_stats(float2 acc, float inv_d, float eps,
                                           float& mean, float& rstd) {
    if constexpr (kSubtractMean) {
        mean = acc.x * inv_d;
        // var = E[x^2] - mean^2; clamp to guard against tiny negative rounding
        // for (near-)constant rows before the rsqrt.
        rstd = rsqrtf(fmaxf(acc.y * inv_d - mean * mean, 0.0f) + eps);
    } else {
        mean = 0.0f;
        rstd = rsqrtf(acc.y * inv_d + eps);
    }
}

template<typename T, bool kSubtractMean>
__device__ __forceinline__ void adaln_row(
    const T* __restrict__ x,
    const T* __restrict__ scale,
    const T* __restrict__ shift,
    T* __restrict__ out,
    int row, int n_rows, int D,
    int scale_group,
    int shift_group,
    float eps)
{
    __shared__ float2 warp_smem[kAdaLNWarps];
    constexpr int VEC = VecWidth<T>::value;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const int scale_row = modulation_row(row, scale_group, n_rows);
    const int shift_row = (shift_group == scale_group)
        ? scale_row
        : modulation_row(row, shift_group, n_rows);

    const T* x_row  = x     + row * D;
    const T* s_row  = scale + scale_row * D;
    const T* sh_row = shift + shift_row * D;
    T*       o_row  = out   + row * D;

    // Vectorize only when every row starts on a 16-byte boundary, i.e.
    // D * sizeof(T) is a multiple of 16. Otherwise n_vec == 0 and the whole
    // row is handled by the scalar tail loops below.
    const int n_vec   = (D % VEC == 0) ? (D / VEC) : 0;
    const int vec_end = n_vec * VEC;

    const Vec<T>* x_vec = reinterpret_cast<const Vec<T>*>(x_row);

    // ---- Pass 1: mean + variance in one shot (accumulate sum and sum-of-squares) ----
    float2 acc = make_float2(0.0f, 0.0f);  // (sum, sum of squares)
    for (int v = tid; v < n_vec; v += nthreads) {
        Vec<T> xv = x_vec[v];
        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            float f = to_float(xv.elts[j]);
            if constexpr (kSubtractMean) acc.x += f;
            acc.y += f * f;
        }
    }
    for (int i = vec_end + tid; i < D; i += nthreads) {
        float f = to_float(x_row[i]);
        if constexpr (kSubtractMean) acc.x += f;
        acc.y += f * f;
    }

    acc = block_reduce_sum2(acc, warp_smem);
    const float inv_d = 1.0f / static_cast<float>(D);
    float mean, rstd;
    norm_stats<kSubtractMean>(acc, inv_d, eps, mean, rstd);

    // ---- Pass 3: normalize + modulate ----
    const Vec<T>* s_vec  = reinterpret_cast<const Vec<T>*>(s_row);
    const Vec<T>* sh_vec = reinterpret_cast<const Vec<T>*>(sh_row);
    Vec<T>*       o_vec  = reinterpret_cast<Vec<T>*>(o_row);
    for (int v = tid; v < n_vec; v += nthreads) {
        Vec<T> xv  = x_vec[v];
        Vec<T> sv  = s_vec[v];
        Vec<T> shv = sh_vec[v];
        Vec<T> ov;
        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            float result = (to_float(xv.elts[j]) - mean) * rstd
                           * (1.0f + to_float(sv.elts[j])) + to_float(shv.elts[j]);
            ov.elts[j] = from_float<T>(result);
        }
        o_vec[v] = ov;
    }
    for (int i = vec_end + tid; i < D; i += nthreads) {
        float result = (to_float(x_row[i]) - mean) * rstd
                       * (1.0f + to_float(s_row[i])) + to_float(sh_row[i]);
        o_row[i] = from_float<T>(result);
    }
}

template<typename T, bool kSubtractMean>
__global__ void adaln_kernel(
    const T* __restrict__ x,
    const T* __restrict__ scale,
    const T* __restrict__ shift,
    T* __restrict__ out,
    int D,
    int scale_group,
    int shift_group,
    float eps)
{
    adaln_row<T, kSubtractMean>(x, scale, shift, out, static_cast<int>(blockIdx.x),
        static_cast<int>(gridDim.x), D, scale_group, shift_group, eps);
}

// Reuse a block for four BF16 rows on full-size Hopper GPUs. The other
// launch paths retain their original kernel and register allocation.
template<bool kSubtractMean>
__global__ void adaln_persistent_kernel(
    const nv_bfloat16* __restrict__ x,
    const nv_bfloat16* __restrict__ scale,
    const nv_bfloat16* __restrict__ shift,
    nv_bfloat16* __restrict__ out,
    int N, int D, int scale_group, int shift_group, float eps)
{
    const int end = min(N, static_cast<int>(blockIdx.x) * 4 + 4);
    #pragma unroll 1
    for (int row = static_cast<int>(blockIdx.x) * 4; row < end; ++row) {
        adaln_row<nv_bfloat16, kSubtractMean>(x, scale, shift, out, row, N,
            D, scale_group, shift_group, eps);
        if (row + 1 < end) __syncthreads();
    }
}

template<typename T, bool kSubtractMean, int kRowWarps>
__global__ void adaln_multirow_kernel(
    const T* __restrict__ x,
    const T* __restrict__ scale,
    const T* __restrict__ shift,
    T* __restrict__ out,
    int N,
    int D,
    bool aligned,
    int scale_group,
    int shift_group,
    float eps)
{
    constexpr int VEC = VecWidth<T>::value;
    __shared__ float2 warp_smem[kAdaLNWarps];
    constexpr int nthreads = kThreadsPerWarp * kRowWarps;
    const int row = blockIdx.x * (kAdaLNWarps / kRowWarps) + threadIdx.x / nthreads;
    const int tid = threadIdx.x % nthreads;
    const int n_rows = N;

    const int scale_row = modulation_row(row, scale_group, n_rows);
    const int shift_row = (shift_group == scale_group)
        ? scale_row
        : modulation_row(row, shift_group, n_rows);

    const T* x_row  = x     + int64_t(row) * D;
    const T* s_row  = scale + int64_t(scale_row) * D;
    const T* sh_row = shift + int64_t(shift_row) * D;
    T*       o_row  = out   + int64_t(row) * D;

    const int n_vec = D % VEC == 0 && aligned ? D / VEC : 0;
    const int vec_end = n_vec * VEC;

    const Vec<T>* x_vec = reinterpret_cast<const Vec<T>*>(x_row);

    // ---- Pass 1: mean + variance in one shot (accumulate sum and sum-of-squares) ----
    float2 acc = make_float2(0.0f, 0.0f);  // (sum, sum of squares)
    for (int v = tid; row < N && v < n_vec; v += nthreads) {
        Vec<T> xv = x_vec[v];
        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            float f = to_float(xv.elts[j]);
            if constexpr (kSubtractMean) acc.x += f;
            acc.y += f * f;
        }
    }
    for (int i = vec_end + tid; row < N && i < D; i += nthreads) {
        float f = to_float(x_row[i]);
        if constexpr (kSubtractMean) acc.x += f;
        acc.y += f * f;
    }

    acc = row_reduce_sum2<kRowWarps>(acc, warp_smem);
    if (row >= N) return;
    const float inv_d = 1.0f / static_cast<float>(D);
    float mean, rstd;
    norm_stats<kSubtractMean>(acc, inv_d, eps, mean, rstd);

    // ---- Pass 3: normalize + modulate ----
    const Vec<T>* s_vec  = reinterpret_cast<const Vec<T>*>(s_row);
    const Vec<T>* sh_vec = reinterpret_cast<const Vec<T>*>(sh_row);
    Vec<T>*       o_vec  = reinterpret_cast<Vec<T>*>(o_row);
    for (int v = tid; v < n_vec; v += nthreads) {
        Vec<T> xv  = x_vec[v];
        Vec<T> sv  = s_vec[v];
        Vec<T> shv = sh_vec[v];
        Vec<T> ov;
        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            float result = (to_float(xv.elts[j]) - mean) * rstd
                           * (1.0f + to_float(sv.elts[j])) + to_float(shv.elts[j]);
            ov.elts[j] = from_float<T>(result);
        }
        o_vec[v] = ov;
    }
    for (int i = vec_end + tid; i < D; i += nthreads) {
        float result = (to_float(x_row[i]) - mean) * rstd
                       * (1.0f + to_float(s_row[i])) + to_float(sh_row[i]);
        o_row[i] = from_float<T>(result);
    }
}

}  // namespace

}  // namespace comfy


extern "C" {

void launch_adaln_kernel(
    const void* x,
    const void* scale,
    const void* shift,
    void*       out,
    int64_t     N,
    int64_t     D,
    int64_t     scale_group,
    int64_t     shift_group,
    float       eps,
    int         dtype_code,
    bool        subtract_mean,
    cudaStream_t stream)
{
    if (N > std::numeric_limits<int>::max() ||
        D > std::numeric_limits<int>::max() ||
        scale_group > std::numeric_limits<int>::max() ||
        shift_group > std::numeric_limits<int>::max()) {
        throw std::runtime_error("adaln dimensions exceed CUDA kernel int32 indexing limits");
    }

    // Use fewer warps per row on DC GPUs where small rows underfill bandwidth.
    // RTX keeps a full block per streaming row. Small inputs need enough blocks.
    bool persistent = false;
    int row_warps = D < 512 && N <= 1024 ? 1 : comfy::kAdaLNWarps;
    if (N >= 8192 && D <= 6144) {
        int device = 0, major = 0;
        if (cudaGetDevice(&device) != cudaSuccess ||
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess) {
            throw std::runtime_error("adaln: device capability query failed");
        }
        if (major >= 10 && major < 12) row_warps = D <= 3072 ? 1 : 2;
        else if (major == 9) {
            if (D <= 1024) row_warps = 1;
            else if (dtype_code == 2 && D >= 2048 && D <= 4096 && D % 8 == 0) {
                int sm_count = 0;
                if (cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device) != cudaSuccess)
                    throw std::runtime_error("adaln: device SM count query failed");
                persistent = sm_count >= 120;
            }
        }
    }
    const bool aligned = ((reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(scale) |
                           reinterpret_cast<uintptr_t>(shift) | reinterpret_cast<uintptr_t>(out)) & 15) == 0;
    if (!aligned) row_warps = 1;
    // The original full-block path uses int32 row offsets; grouped rows use int64.
    if (N * D > std::numeric_limits<int>::max() && row_warps == comfy::kAdaLNWarps) row_warps = 2;
    dim3 block(comfy::kAdaLNThreads);
#define LAUNCH_ADALN(W) \
            comfy::adaln_multirow_kernel<T, kSubtractMean, W><<<static_cast<unsigned int>((N + 8 / W - 1) / (8 / W)), block, 0, stream>>>( \
                static_cast<const T*>(x), static_cast<const T*>(scale), \
                static_cast<const T*>(shift), static_cast<T*>(out), \
                static_cast<int>(N), static_cast<int>(D), aligned, \
                static_cast<int>(scale_group), static_cast<int>(shift_group), eps)
    if (persistent && aligned && row_warps == comfy::kAdaLNWarps) {
        DISPATCH_BOOL(subtract_mean, kSubtractMean, [&]() {
            comfy::adaln_persistent_kernel<kSubtractMean><<<static_cast<unsigned int>((N + 3) / 4), block, 0, stream>>>(
                static_cast<const nv_bfloat16*>(x), static_cast<const nv_bfloat16*>(scale),
                static_cast<const nv_bfloat16*>(shift), static_cast<nv_bfloat16*>(out),
                static_cast<int>(N), static_cast<int>(D), static_cast<int>(scale_group),
                static_cast<int>(shift_group), eps);
        });
    } else {
        DISPATCH_FP_DTYPE(dtype_code, T, [&]() {
            DISPATCH_BOOL(subtract_mean, kSubtractMean, [&]() {
                if (row_warps == 1) LAUNCH_ADALN(1);
                else if (row_warps == 2) LAUNCH_ADALN(2);
                else comfy::adaln_kernel<T, kSubtractMean><<<static_cast<unsigned int>(N), block, 0, stream>>>(
                    static_cast<const T*>(x), static_cast<const T*>(scale), static_cast<const T*>(shift),
                    static_cast<T*>(out), static_cast<int>(D), static_cast<int>(scale_group),
                    static_cast<int>(shift_group), eps);
            });
        });
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("adaln kernel launch failed: ") + cudaGetErrorString(err));
#undef LAUNCH_ADALN
}

}  // extern "C"
