// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// group_norm_silu_nchw: GroupNorm over a contiguous NCHW tensor and SiLU in
// one pass.
//
// The diffusion model's resnet blocks run `norm` then `silu` on a contiguous
// NCHW activation, which torch does in four streaming passes (group_norm reads
// x and writes y, silu reads y and writes z). The group statistics need a full
// read before anything can be written, so a fused kernel cannot do fewer than
// three -- but it never materializes the intermediate either.
//
// One block per (sample, group). In NCHW a group is the contiguous run of C/G
// channel planes, so the block walks it as one linear range; each thread holds
// eight consecutive pixels of one channel, which keeps every load and store a
// 16-byte access. Statistics accumulate in fp32.
#include <stdexcept>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "../rope_math.h"

namespace comfy::hip_backend {

constexpr int kNchwThreads = 256;
constexpr int kNchwVec = 8;
constexpr int kNchwWarps = kNchwThreads / 32;

template <typename T>
__forceinline__ __device__ float gn2_load(T v);
template <>
__forceinline__ __device__ float gn2_load<__half>(__half v) {
    return __half2float(v);
}
template <>
__forceinline__ __device__ float gn2_load<__bf16>(__bf16 v) {
    return static_cast<float>(v);
}
template <typename T>
__forceinline__ __device__ T gn2_store(float v);
template <>
__forceinline__ __device__ __half gn2_store<__half>(float v) {
    return __float2half(v);
}
template <>
__forceinline__ __device__ __bf16 gn2_store<__bf16>(float v) {
    return static_cast<__bf16>(v);
}

// Sum and sum of squares over the block. Both partials ride one shuffle chain
// of float2, so the two reductions cost one set of syncthreads.
__device__ __forceinline__ float2 gn2_block_sum(float s, float ss, float2* sh) {
    for (int off = 16; off > 0; off >>= 1) {
        s += __shfl_down(s, off);
        ss += __shfl_down(ss, off);
    }
    const int lane = threadIdx.x & 31;
    if (lane == 0) {
        sh[threadIdx.x >> 5] = make_float2(s, ss);
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        float2 v = (threadIdx.x < kNchwWarps) ? sh[threadIdx.x] : make_float2(0.0f, 0.0f);
        for (int off = 16; off > 0; off >>= 1) {
            v.x += __shfl_down(v.x, off);
            v.y += __shfl_down(v.y, off);
        }
        if (threadIdx.x == 0) {
            sh[0] = v;
        }
    }
    __syncthreads();
    return sh[0];
}

// x, gamma, beta and out are NCHW-contiguous; a group is the run of cpg planes
// at (b, g). planes % 8 != 0 takes the scalar path.
template <typename T, bool kVec>
__global__ __launch_bounds__(kNchwThreads) void gn_silu_nchw_kernel(
    const T* __restrict__ x, const T* __restrict__ gamma, const T* __restrict__ beta,
    T* __restrict__ out, int64_t planes, int G, int cpg, float eps) {
    __shared__ float2 sh[kNchwWarps];
    __shared__ float2 sh_stats;

    const int g = blockIdx.x % G;
    const int64_t b = blockIdx.x / G;
    const int64_t base = (b * G + g) * cpg * planes;
    const T* src = x + base;
    T* dst = out + base;
    const int64_t n = static_cast<int64_t>(cpg) * planes;

    float s = 0.0f;
    float ss = 0.0f;
    if (kVec) {
        for (int64_t i = threadIdx.x * kNchwVec; i < n; i += kNchwThreads * kNchwVec) {
            const uint4 raw = *reinterpret_cast<const uint4*>(src + i);
            const T* v = reinterpret_cast<const T*>(&raw);
#pragma unroll
            for (int k = 0; k < kNchwVec; ++k) {
                const float f = gn2_load<T>(v[k]);
                s += f;
                ss += f * f;
            }
        }
    } else {
        for (int64_t i = threadIdx.x; i < n; i += kNchwThreads) {
            const float f = gn2_load<T>(src[i]);
            s += f;
            ss += f * f;
        }
    }

    const float2 total = gn2_block_sum(s, ss, sh);
    if (threadIdx.x == 0) {
        const float inv = 1.0f / static_cast<float>(n);
        const float mean = total.x * inv;
        const float var = fmaxf(total.y * inv - mean * mean, 0.0f);
        sh_stats = make_float2(mean, rsqrtf(var + eps));
    }
    __syncthreads();
    const float mean = sh_stats.x;
    const float rstd = sh_stats.y;

    // The second pass folds the statistics into the per-channel scale and
    // shift, so the affine weight is read once per eight pixels.
    if (kVec) {
        for (int64_t i = threadIdx.x * kNchwVec; i < n; i += kNchwThreads * kNchwVec) {
            const int c = g * cpg + static_cast<int>(i / planes);
            const float a = rstd * gn2_load<T>(gamma[c]);
            const float shift = gn2_load<T>(beta[c]) - mean * a;
            uint4 raw = *reinterpret_cast<const uint4*>(src + i);
            T* v = reinterpret_cast<T*>(&raw);
#pragma unroll
            for (int k = 0; k < kNchwVec; ++k) {
                // round_to, not a store/load pair: eager GroupNorm materializes
                // its output in x's dtype before the SiLU sees it.
                const float f = round_to<T>(a * gn2_load<T>(v[k]) + shift);
                v[k] = gn2_store<T>(f / (1.0f + expf(-f)));
            }
            *reinterpret_cast<uint4*>(dst + i) = raw;
        }
    } else {
        for (int64_t i = threadIdx.x; i < n; i += kNchwThreads) {
            const int c = g * cpg + static_cast<int>(i / planes);
            const float a = rstd * gn2_load<T>(gamma[c]);
            const float shift = gn2_load<T>(beta[c]) - mean * a;
            const float f = round_to<T>(a * gn2_load<T>(src[i]) + shift);
            dst[i] = gn2_store<T>(f / (1.0f + expf(-f)));
        }
    }
}

}  // namespace comfy::hip_backend

extern "C" void launch_group_norm_silu_nchw(const void* x, const void* gamma, const void* beta,
                                            void* out, int64_t B, int64_t C, int64_t H, int64_t W,
                                            int G, float eps, int dtype_code, hipStream_t stream) {

    using namespace comfy::hip_backend;

    if (B <= 0 || C <= 0 || H <= 0 || W <= 0) {
        return;
    }
    if (G <= 0 || C % G != 0) {
        throw std::runtime_error("group_norm_silu_nchw: num_groups must divide the channels");
    }
    // Groups ride on grid.x; a batch * group count past 65535 is a caller bug
    // worth failing loudly on rather than launching.
    if (B * G > 65535) {
        throw std::runtime_error("group_norm_silu_nchw: batch * groups must be <= 65535");
    }
    const int64_t planes = H * W;
    const int cpg = static_cast<int>(C / G);
    const unsigned blocks = static_cast<unsigned>(B * G);
    if (dtype_code == 1) {
        if (planes % kNchwVec == 0) {
            gn_silu_nchw_kernel<__half, true><<<blocks, kNchwThreads, 0, stream>>>(
                static_cast<const __half*>(x), static_cast<const __half*>(gamma),
                static_cast<const __half*>(beta), static_cast<__half*>(out), planes, G, cpg, eps);
        } else {
            gn_silu_nchw_kernel<__half, false><<<blocks, kNchwThreads, 0, stream>>>(
                static_cast<const __half*>(x), static_cast<const __half*>(gamma),
                static_cast<const __half*>(beta), static_cast<__half*>(out), planes, G, cpg, eps);
        }
    } else if (dtype_code == 2) {
        if (planes % kNchwVec == 0) {
            gn_silu_nchw_kernel<__bf16, true><<<blocks, kNchwThreads, 0, stream>>>(
                static_cast<const __bf16*>(x), static_cast<const __bf16*>(gamma),
                static_cast<const __bf16*>(beta), static_cast<__bf16*>(out), planes, G, cpg, eps);
        } else {
            gn_silu_nchw_kernel<__bf16, false><<<blocks, kNchwThreads, 0, stream>>>(
                static_cast<const __bf16*>(x), static_cast<const __bf16*>(gamma),
                static_cast<const __bf16*>(beta), static_cast<__bf16*>(out), planes, G, cpg, eps);
        }
    } else {
        throw std::runtime_error("group_norm_silu_nchw: only float16 and bfloat16 are supported");
    }
}
