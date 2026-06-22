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

// Fused AdaLN kernel: out = layernorm(x, eps=eps) * (1 + scale) + shift
// Each block handles one row; 256 threads per block; float32 accumulation.
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
// rather than two. The normalize pass re-reads x, but a row is only a few KiB
// and stays resident in L1/L2, so it is not re-fetched from DRAM.

#include "utils.cuh"
#include "dtype_dispatch.cuh"

#include <limits>
#include <stdexcept>

namespace comfy {

namespace {

constexpr int kAdaLNThreads = 256;
constexpr int kAdaLNWarps = kAdaLNThreads / kThreadsPerWarp;  // 8
constexpr int kAdaLNRowsPerWarpBlock = kAdaLNWarps;

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

__device__ __forceinline__ int modulation_row(int row, int group, int n_rows) {
    if (group == 1) return row;
    if (group == n_rows) return 0;
    if ((group & (group - 1)) == 0) return row >> (__ffs(group) - 1);
    return row / group;
}

template<typename T>
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
    constexpr int VEC = VecWidth<T>::value;
    const int row = static_cast<int>(blockIdx.x);
    const int n_rows = static_cast<int>(gridDim.x);
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    __shared__ float2 warp_smem[kAdaLNWarps];

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
            acc.x += f;
            acc.y += f * f;
        }
    }
    for (int i = vec_end + tid; i < D; i += nthreads) {
        float f = to_float(x_row[i]);
        acc.x += f;
        acc.y += f * f;
    }

    acc = block_reduce_sum2(acc, warp_smem);
    const float inv_d = 1.0f / static_cast<float>(D);
    const float mean  = acc.x * inv_d;
    // var = E[x^2] - mean^2; clamp to guard against tiny negative rounding for
    // (near-)constant rows before the rsqrt.
    const float var   = fmaxf(acc.y * inv_d - mean * mean, 0.0f);
    const float rstd  = rsqrtf(var + eps);

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

template<typename T>
__global__ void adaln_warp_kernel(
    const T* __restrict__ x,
    const T* __restrict__ scale,
    const T* __restrict__ shift,
    T* __restrict__ out,
    int N,
    int D,
    int scale_group,
    int shift_group,
    float eps)
{
    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & (kThreadsPerWarp - 1);
    const int row = blockIdx.x * kAdaLNRowsPerWarpBlock + warp_in_block;
    if (row >= N) return;

    const int scale_row = modulation_row(row, scale_group, N);
    const int shift_row = (shift_group == scale_group)
        ? scale_row
        : modulation_row(row, shift_group, N);

    const T* x_row  = x     + row * D;
    const T* s_row  = scale + scale_row * D;
    const T* sh_row = shift + shift_row * D;
    T*       o_row  = out   + row * D;

    float2 acc = make_float2(0.0f, 0.0f);
    for (int i = lane; i < D; i += kThreadsPerWarp) {
        float f = to_float(x_row[i]);
        acc.x += f;
        acc.y += f * f;
    }

    acc = warp_reduce_sum2(acc);
    const float sum = __shfl_sync(0xffffffff, acc.x, 0);
    const float sumsq = __shfl_sync(0xffffffff, acc.y, 0);
    const float inv_d = 1.0f / static_cast<float>(D);
    const float mean = sum * inv_d;
    const float var = fmaxf(sumsq * inv_d - mean * mean, 0.0f);
    const float rstd = rsqrtf(var + eps);

    for (int i = lane; i < D; i += kThreadsPerWarp) {
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
    cudaStream_t stream)
{
    if (N > std::numeric_limits<int>::max() ||
        D > std::numeric_limits<int>::max() ||
        scale_group > std::numeric_limits<int>::max() ||
        shift_group > std::numeric_limits<int>::max()) {
        throw std::runtime_error("adaln dimensions exceed CUDA kernel int32 indexing limits");
    }

    dim3 grid(static_cast<unsigned int>(N));
    dim3 block(comfy::kAdaLNThreads);

    DISPATCH_FP_DTYPE(dtype_code, T, [&]() {
        if (N <= 1024) {
            dim3 warp_grid(
                static_cast<unsigned int>((N + comfy::kAdaLNRowsPerWarpBlock - 1) /
                                          comfy::kAdaLNRowsPerWarpBlock));
            comfy::adaln_warp_kernel<T><<<warp_grid, block, 0, stream>>>(
                static_cast<const T*>(x),
                static_cast<const T*>(scale),
                static_cast<const T*>(shift),
                static_cast<T*>(out),
                static_cast<int>(N),
                static_cast<int>(D),
                static_cast<int>(scale_group),
                static_cast<int>(shift_group),
                eps);
        } else {
            comfy::adaln_kernel<T><<<grid, block, 0, stream>>>(
                static_cast<const T*>(x),
                static_cast<const T*>(scale),
                static_cast<const T*>(shift),
                static_cast<T*>(out),
                static_cast<int>(D),
                static_cast<int>(scale_group),
                static_cast<int>(shift_group),
                eps);
        }
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error(std::string("adaln kernel launch failed: ") + cudaGetErrorString(err));
    });
}

}  // extern "C"
