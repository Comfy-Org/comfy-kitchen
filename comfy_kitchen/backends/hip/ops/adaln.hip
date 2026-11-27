// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Fused AdaLN:
//   adaln     - layernorm(x, no affine) * (1 + scale) + shift
//   rms_adaln - rmsnorm(x, no affine)   * (1 + scale) + shift
// RMSNorm is LayerNorm with the mean pinned to zero, so one kernel templated on
// SubtractMean covers both.
//
// One block per row. The row is re-read for the normalization pass rather than
// held in registers, which does not scale to large D.
//
// scale/shift broadcast over inner dims, expressed as a row group: output row r
// reads modulation row r / group (see backends/_modulation.py).
#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include "../hadamard.h"  // load_in / dtype codes

namespace comfy::hip_backend {

constexpr int kAdaLNThreads = 256;

template <typename T>
__forceinline__ __device__ void store_out(T* p, int64_t i, float v);

template <>
__forceinline__ __device__ void store_out<float>(float* p, int64_t i, float v) {
    p[i] = v;
}
template <>
__forceinline__ __device__ void store_out<__half>(__half* p, int64_t i, float v) {
    p[i] = __float2half(v);
}
template <>
__forceinline__ __device__ void store_out<__bf16>(__bf16* p, int64_t i, float v) {
    p[i] = static_cast<__bf16>(v);
}

// scale/shift need not share x's dtype (e.g. fp32 modulation against a bf16
// activation), so each buffer carries its own dtype code.
template <typename T, bool SubtractMean>
__global__ __launch_bounds__(kAdaLNThreads) void adaln_kernel(
    const void* __restrict__ x, const void* __restrict__ scale,
    const void* __restrict__ shift, T* __restrict__ out,
    int D, int scale_group, int shift_group, float eps,
    int code, int scale_code, int shift_code) {

    __shared__ float red_sq[kAdaLNThreads];

    const int row = blockIdx.x;
    const int t = threadIdx.x;

    const int64_t xbase = static_cast<int64_t>(row) * D;
    const int64_t sbase = static_cast<int64_t>(row / scale_group) * D;
    const int64_t hbase = static_cast<int64_t>(row / shift_group) * D;

    const float inv_d = 1.0f / static_cast<float>(D);

    float mean = 0.0f;
    if constexpr (SubtractMean) {
        __shared__ float red_sum[kAdaLNThreads];

        float sum = 0.0f;
        for (int i = t; i < D; i += kAdaLNThreads) {
            sum += load_in(x, xbase + i, code);
        }

        red_sum[t] = sum;
        __syncthreads();
        for (int s = kAdaLNThreads / 2; s > 0; s >>= 1) {
            if (t < s) red_sum[t] += red_sum[t + s];
            __syncthreads();
        }
        mean = red_sum[0] * inv_d;
    }

    // Sum the squared deviations rather than E[x^2] - mean^2: the latter cancels
    // catastrophically once the row's offset dwarfs its spread. mean == 0 makes
    // this E[x^2], which is what RMSNorm wants.
    float sq = 0.0f;
    for (int i = t; i < D; i += kAdaLNThreads) {
        const float d = load_in(x, xbase + i, code) - mean;
        sq += d * d;
    }

    red_sq[t] = sq;
    __syncthreads();
    for (int s = kAdaLNThreads / 2; s > 0; s >>= 1) {
        if (t < s) red_sq[t] += red_sq[t + s];
        __syncthreads();
    }

    const float rstd = rsqrtf(red_sq[0] * inv_d + eps);

    for (int i = t; i < D; i += kAdaLNThreads) {
        const float v = load_in(x, xbase + i, code);
        const float norm = (v - mean) * rstd;
        const float s = load_in(scale, sbase + i, scale_code);
        const float h = load_in(shift, hbase + i, shift_code);
        store_out<T>(out, xbase + i, norm * (1.0f + s) + h);
    }
}

// Vectorized half/bf16 path for D % 8 == 0 and D <= 8 * kAdaLNVecThreads * VPT: each
// thread holds VPT 16-byte chunks of the row in registers, so x is read once, and the
// reductions run over the wave with shuffles. Same math and rounding as adaln_kernel;
// only the order of the fp32 sums differs.
constexpr int kAdaLNVecThreads = 128;

template <typename T>
__forceinline__ __device__ float to_float(T v);
template <>
__forceinline__ __device__ float to_float<__half>(__half v) {
    return __half2float(v);
}
template <>
__forceinline__ __device__ float to_float<__bf16>(__bf16 v) {
    return static_cast<float>(v);
}

template <typename T>
__forceinline__ __device__ void load8(const void* p, int64_t i, float (&o)[8]) {
    const uint4 pack = *reinterpret_cast<const uint4*>(static_cast<const T*>(p) + i);
    T e[8];
    __builtin_memcpy(e, &pack, sizeof(pack));  // no T/uint4 type punning
#pragma unroll
    for (int j = 0; j < 8; ++j) o[j] = to_float<T>(e[j]);
}

__forceinline__ __device__ void load8_code(const void* p, int64_t i, int code, float (&o)[8]) {
    if (code == 0) {
        const float4* f = reinterpret_cast<const float4*>(static_cast<const float*>(p) + i);
        const float4 a = f[0], b = f[1];
        o[0] = a.x; o[1] = a.y; o[2] = a.z; o[3] = a.w;
        o[4] = b.x; o[5] = b.y; o[6] = b.z; o[7] = b.w;
    } else if (code == 1) {
        load8<__half>(p, i, o);
    } else {
        load8<__bf16>(p, i, o);
    }
}

__forceinline__ __device__ float block_sum_vec(float v, float* red) {
    for (int off = warpSize / 2; off > 0; off >>= 1) v += __shfl_xor(v, off);
    const int wave = threadIdx.x / warpSize;
    if (threadIdx.x % warpSize == 0) red[wave] = v;
    __syncthreads();
    float total = 0.0f;
    for (int w = 0; w < kAdaLNVecThreads / warpSize; ++w) total += red[w];
    __syncthreads();  // red is reused by the next reduction
    return total;
}

template <typename T, bool SubtractMean, int VPT>
__global__ __launch_bounds__(kAdaLNVecThreads) void adaln_vec_kernel(
    const T* x, const void* __restrict__ scale, const void* __restrict__ shift, T* out,
    int D, int scale_group, int shift_group, float eps, int scale_code, int shift_code) {

    __shared__ float red[kAdaLNVecThreads / 32];

    const int row = blockIdx.x;
    const int t = threadIdx.x;
    const int nvec = D / 8;
    const int64_t xbase = static_cast<int64_t>(row) * D;
    const int64_t sbase = static_cast<int64_t>(row / scale_group) * D;
    const int64_t hbase = static_cast<int64_t>(row / shift_group) * D;
    const float inv_d = 1.0f / static_cast<float>(D);

    float v[VPT][8];
#pragma unroll
    for (int j = 0; j < VPT; ++j) {
        const int c = t + j * kAdaLNVecThreads;
        if (c < nvec) {
            load8<T>(x, xbase + c * 8, v[j]);
        } else {
#pragma unroll
            for (int e = 0; e < 8; ++e) v[j][e] = 0.0f;
        }
    }

    float mean = 0.0f;
    if constexpr (SubtractMean) {
        float sum = 0.0f;
#pragma unroll
        for (int j = 0; j < VPT; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) sum += v[j][e];
        mean = block_sum_vec(sum, red) * inv_d;
    }

    float sq = 0.0f;
#pragma unroll
    for (int j = 0; j < VPT; ++j) {
        if (t + j * kAdaLNVecThreads < nvec) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const float d = v[j][e] - mean;
                sq += d * d;
            }
        }
    }
    const float rstd = rsqrtf(block_sum_vec(sq, red) * inv_d + eps);

#pragma unroll
    for (int j = 0; j < VPT; ++j) {
        const int c = t + j * kAdaLNVecThreads;
        if (c >= nvec) continue;
        float s[8], h[8];
        load8_code(scale, sbase + c * 8, scale_code, s);
        load8_code(shift, hbase + c * 8, shift_code, h);
        T o[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const float r = (v[j][e] - mean) * rstd * (1.0f + s[e]) + h[e];
            if constexpr (std::is_same_v<T, __half>) {
                o[e] = __float2half(r);
            } else {
                o[e] = static_cast<__bf16>(r);
            }
        }
        uint4 pack;
        __builtin_memcpy(&pack, o, sizeof(pack));
        *reinterpret_cast<uint4*>(out + xbase + c * 8) = pack;
    }
}

}  // namespace comfy::hip_backend

// subtract_mean selects LayerNorm (true) or RMSNorm (false) statistics.
extern "C" void launch_adaln_kernel(
    const void* x, const void* scale, const void* shift, void* out,
    int N, int D, int scale_group, int shift_group, float eps,
    int code, int scale_code, int shift_code, bool subtract_mean, hipStream_t stream) {

    using namespace comfy::hip_backend;

    if (N == 0 || D == 0) {
        // No rows, or zero-width rows: nothing to normalize, and D == 0 would make
        // the kernel's group divisions divide by zero.
        return;  // a zero-block launch is hipErrorInvalidConfiguration
    }

    // 16-byte rows: D % 8 == 0 keeps every row, scale and shift chunk aligned once the
    // base pointers are.
    const auto aligned16 = [](const void* p) { return reinterpret_cast<uintptr_t>(p) % 16 == 0; };
    const int vpt = (D / 8 + kAdaLNVecThreads - 1) / kAdaLNVecThreads;
    if ((code == 1 || code == 2) && D % 8 == 0 && vpt <= 4 && aligned16(x) && aligned16(out) &&
        aligned16(scale) && aligned16(shift)) {
#define CK_ADALN_VEC(T, MEAN, VPT)                                                           \
    adaln_vec_kernel<T, MEAN, VPT><<<N, kAdaLNVecThreads, 0, stream>>>(                      \
        static_cast<const T*>(x), scale, shift, static_cast<T*>(out), D, scale_group,        \
        shift_group, eps, scale_code, shift_code)
#define CK_ADALN_VEC_VPT(T, MEAN)                \
    switch (vpt) {                               \
        case 1: CK_ADALN_VEC(T, MEAN, 1); break; \
        case 2: CK_ADALN_VEC(T, MEAN, 2); break; \
        case 3: CK_ADALN_VEC(T, MEAN, 3); break; \
        default: CK_ADALN_VEC(T, MEAN, 4); break; \
    }
#define CK_ADALN_VEC_T(T)                   \
    if (subtract_mean) {                    \
        CK_ADALN_VEC_VPT(T, true);          \
    } else {                                \
        CK_ADALN_VEC_VPT(T, false);         \
    }
        if (code == 1) {
            CK_ADALN_VEC_T(__half);
        } else {
            CK_ADALN_VEC_T(__bf16);
        }
#undef CK_ADALN_VEC_T
#undef CK_ADALN_VEC_VPT
#undef CK_ADALN_VEC
        return;
    }

#define CK_ADALN_LAUNCH(T, MEAN)                                                             \
    adaln_kernel<T, MEAN><<<N, kAdaLNThreads, 0, stream>>>(                                  \
        x, scale, shift, static_cast<T*>(out), D, scale_group, shift_group, eps, code,       \
        scale_code, shift_code)
#define CK_ADALN_DISPATCH(T)               \
    if (subtract_mean) {                   \
        CK_ADALN_LAUNCH(T, true);          \
    } else {                               \
        CK_ADALN_LAUNCH(T, false);         \
    }

    if (code == 0) {
        CK_ADALN_DISPATCH(float);
    } else if (code == 1) {
        CK_ADALN_DISPATCH(__half);
    } else if (code == 2) {
        CK_ADALN_DISPATCH(__bf16);
    } else {
        throw std::runtime_error("adaln: unsupported dtype");
    }
#undef CK_ADALN_DISPATCH
#undef CK_ADALN_LAUNCH
}
