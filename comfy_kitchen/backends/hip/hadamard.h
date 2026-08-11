// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Fused ConvRot rotation + rowwise quantization.
//
// The regular Hadamard-G (G in {16, 64, 256}) is kron(H4, ...) / sqrt(G) and so
// factors into log4(G) radix-4 butterfly stages over the base-4 digits of the
// index, matching the eager _rotate_activation reference.
//
// One block per row: transform each G-group in LDS, track the row absmax, then
// quantize. INT4 output packs two nibbles per byte (low nibble = even index),
// which is the layout the iu4 A-fragment consumes directly.
#pragma once

#include <stdexcept>
#include <string>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace comfy::hip_backend {

// convrot_quant_kernel handles 256/G groups per pass and rotates in log4(G)
// stages, so a G outside this set either divides to a zero-width pass or is not
// a power of four. The dispatch wrappers fall back to eager before reaching here.
inline void check_convrot_group_size(int group_size) {
    if (group_size != 16 && group_size != 64 && group_size != 256) {
        throw std::runtime_error("convrot: group_size must be 16, 64 or 256");
    }
}

// The kernel's static LDS: two 256-element butterfly buffers below.
constexpr size_t kConvrotStaticLds = 2 * 256 * sizeof(float);

// convrot_quant_kernel stages the whole rotated row in dynamic LDS, preserving
// the input dtype's rounding contract. K is therefore bounded by both the
// workgroup budget and the input element size. Past it the wrappers fall back
// to eager instead. 0 means unknown or an invalid dtype code.
inline size_t convrot_row_element_size(int in_dtype) {
    if (in_dtype == 0) return sizeof(float);
    if (in_dtype == 1) return sizeof(__half);
    if (in_dtype == 2) return sizeof(__bf16);
    return 0;
}

inline int convrot_max_k(int in_dtype) {
    const size_t element_size = convrot_row_element_size(in_dtype);
    if (element_size == 0) {
        return 0;
    }
    int device = 0;
    if (hipGetDevice(&device) != hipSuccess) {
        return 0;
    }
    int lds = 0;
    if (hipDeviceGetAttribute(&lds, hipDeviceAttributeMaxSharedMemoryPerBlock, device) !=
            hipSuccess ||
        lds <= static_cast<int>(kConvrotStaticLds)) {
        return 0;
    }
    return static_cast<int>((static_cast<size_t>(lds) - kConvrotStaticLds) /
                            element_size);
}

inline void check_convrot_k(int k, int group_size, int in_dtype) {
    // The kernel rotates K/G whole groups but reads back all K entries of the row
    // buffer, so a partial trailing group would quantize uninitialized LDS.
    if (group_size <= 0 || k % group_size != 0) {
        throw std::runtime_error("convrot: K=" + std::to_string(k) +
                                 " is not divisible by group_size=" + std::to_string(group_size));
    }
    const int max_k = convrot_max_k(in_dtype);
    if (max_k <= 0 || k > max_k) {
        throw std::runtime_error("convrot: K=" + std::to_string(k) +
                                 " does not fit in LDS (max " + std::to_string(max_k) + ")");
    }
}

// dtype codes match comfy_kitchen.backends.eager.quantization.DTYPE_TO_CODE
__forceinline__ __device__ float load_in(const void* x, int64_t idx, int code) {
    if (code == 0) return static_cast<const float*>(x)[idx];
    if (code == 1) return __half2float(static_cast<const __half*>(x)[idx]);
    return static_cast<float>(static_cast<const __bf16*>(x)[idx]);
}

// Codes match comfy_kitchen.backends._activations.INPUT_ACT_TO_CODE. SwiGLU is
// the gated pair: the raw row is [gate | up] (2*K wide) and the activated row
// silu(gate) * up is K wide; the others are elementwise.
enum : int { kActNone = 0, kActGeluTanh = 1, kActSwiGLU = 2 };

inline void check_convrot_act(int act) {
    if (act != kActNone && act != kActGeluTanh && act != kActSwiGLU) {
        throw std::runtime_error("convrot: unsupported input activation code");
    }
}

template <int ACT>
__forceinline__ __device__ float apply_input_act(float v) {
    if constexpr (ACT == kActGeluTanh) {
        // Matches torch.nn.functional.gelu(x, approximate="tanh").
        constexpr float kBeta = 0.7978845608028654f;  // sqrt(2/pi)
        constexpr float kKappa = 0.044715f;
        return 0.5f * v * (1.0f + tanhf(kBeta * (v + kKappa * v * v * v)));
    }
    return v;
}

// One activated value: column `col` of the K-wide activated row starting at
// `in_row`. SwiGLU reads the gate at col and the up at K + col; every other
// activation reads the same K-wide row it writes.
template <int ACT>
__forceinline__ __device__ float load_input_act(
    const void* x, int64_t in_row, int col, int K, int code) {
    if constexpr (ACT == kActSwiGLU) {
        // Matches torch silu(gate) * up.
        const float gate = load_in(x, in_row + col, code);
        const float up = load_in(x, in_row + K + col, code);
        return (gate / (1.0f + expf(-gate))) * up;
    } else {
        return apply_input_act<ACT>(load_in(x, in_row + col, code));
    }
}

template <typename RowT>
__forceinline__ __device__ RowT store_row_value(float v) {
    return static_cast<RowT>(v);
}

template <>
__forceinline__ __device__ __half store_row_value<__half>(float v) {
    return __float2half(v);
}

template <typename RowT>
__forceinline__ __device__ float load_row_value(RowT v) {
    return static_cast<float>(v);
}

template <>
__forceinline__ __device__ float load_row_value<__half>(__half v) {
    return __half2float(v);
}

template <typename RowT, bool PACK_INT4, int ACT, int G>
__global__ __launch_bounds__(256) void convrot_quant_kernel(
    const void* __restrict__ x, int in_dtype,
    int8_t* __restrict__ qout, float* __restrict__ scaleout,
    int M, int K) {

    const float h4[4][4] = {{1, 1, 1, -1}, {1, 1, -1, 1}, {1, -1, 1, 1}, {-1, 1, 1, 1}};
    __shared__ float g[2][256];
    extern __shared__ unsigned char rowbuf_raw[];
    RowT* rowbuf = reinterpret_cast<RowT*>(rowbuf_raw);  // K entries: the rotated row

    const int row = blockIdx.x;
    const int t = threadIdx.x;

    constexpr int nstages = G == 16 ? 2 : (G == 64 ? 3 : 4);

    const int gpw = 256 / G;      // groups handled per pass
    const int glocal = t / G;     // this thread's group within the pass
    const int e = t % G;          // element within the group
    const int gbase_idx = glocal * G;
    const float norm = rsqrtf(static_cast<float>(G));
    const int ngrp = K / G;
    // SwiGLU reads a [gate | up] raw row twice as wide as the K it writes.
    constexpr int kInWidth = (ACT == kActSwiGLU) ? 2 : 1;
    const int64_t in_row = static_cast<int64_t>(row) * K * kInWidth;

    float lmax = 0.0f;
    for (int gbase = 0; gbase < ngrp; gbase += gpw) {
        const int grp = gbase + glocal;
        const bool active = grp < ngrp;
        g[0][t] = active ? load_input_act<ACT>(x, in_row, grp * G + e, K, in_dtype) : 0.0f;
        __syncthreads();

        int src = 0;
#pragma unroll
        for (int stage = 0; stage < nstages; ++stage) {
            const int stride = 1 << (2 * stage);
            const int ds = (e / stride) & 3;
            const int b = gbase_idx + (e - ds * stride);
            const float v0 = g[src][b], v1 = g[src][b + stride];
            const float v2 = g[src][b + 2 * stride], v3 = g[src][b + 3 * stride];
            const float v = h4[ds][0] * v0 + h4[ds][1] * v1 + h4[ds][2] * v2 + h4[ds][3] * v3;
            const int dst = src ^ 1;
            g[dst][t] = v;
            __syncthreads();
            src = dst;
        }

        if (active) {
            const float tv = g[src][t] * norm;
            const RowT stored = store_row_value<RowT>(tv);
            rowbuf[static_cast<int64_t>(grp) * G + e] = stored;
            lmax = fmaxf(lmax, fabsf(load_row_value(stored)));
        }
    }

    g[0][t] = lmax;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (t < s) g[0][t] = fmaxf(g[0][t], g[0][t + s]);
        __syncthreads();
    }

    constexpr float kQMax = PACK_INT4 ? 7.0f : 127.0f;
    const float rowmax = fmaxf(g[0][0], 1e-10f);
    const float scale = rowmax / kQMax;
    const float inv = kQMax / rowmax;
    if (t == 0) scaleout[row] = scale;

    if constexpr (PACK_INT4) {
        const int Kp = K / 2;
        for (int jb = t; jb < Kp; jb += 256) {
            const float a = load_row_value(rowbuf[2 * jb]);
            const float b = load_row_value(rowbuf[2 * jb + 1]);
            int qa = static_cast<int>(rintf(a * inv));
            int qb = static_cast<int>(rintf(b * inv));
            qa = qa < -7 ? -7 : (qa > 7 ? 7 : qa);
            qb = qb < -7 ? -7 : (qb > 7 ? 7 : qb);
            qout[static_cast<int64_t>(row) * Kp + jb] =
                static_cast<int8_t>(((qa & 0xF) | ((qb & 0xF) << 4)) & 0xFF);
        }
    } else {
        for (int j = t; j < K; j += 256) {
            const float v = load_row_value(rowbuf[j]);
            int q = static_cast<int>(rintf(v * inv));
            q = q < -127 ? -127 : (q > 127 ? 127 : q);
            qout[static_cast<int64_t>(row) * K + j] = static_cast<int8_t>(q);
        }
    }
}

template <bool PACK_INT4, int ACT, int G>
inline void launch_convrot_quant_group(
    const void* x, int in_dtype, int8_t* qout, float* scaleout,
    int M, int K, hipStream_t stream) {

    const size_t shmem = static_cast<size_t>(K) * convrot_row_element_size(in_dtype);
    if (in_dtype == 0) {
        convrot_quant_kernel<float, PACK_INT4, ACT, G><<<M, 256, shmem, stream>>>(
            x, in_dtype, qout, scaleout, M, K);
    } else if (in_dtype == 1) {
        convrot_quant_kernel<__half, PACK_INT4, ACT, G><<<M, 256, shmem, stream>>>(
            x, in_dtype, qout, scaleout, M, K);
    } else if (in_dtype == 2) {
        convrot_quant_kernel<__bf16, PACK_INT4, ACT, G><<<M, 256, shmem, stream>>>(
            x, in_dtype, qout, scaleout, M, K);
    } else {
        throw std::runtime_error("convrot: unsupported input dtype code");
    }
}

template <bool PACK_INT4, int ACT = kActNone>
inline void launch_convrot_quant(
    const void* x, int in_dtype, int8_t* qout, float* scaleout,
    int M, int K, int group_size, hipStream_t stream) {

    check_convrot_group_size(group_size);
    if (group_size == 16) {
        launch_convrot_quant_group<PACK_INT4, ACT, 16>(
            x, in_dtype, qout, scaleout, M, K, stream);
    } else if (group_size == 64) {
        launch_convrot_quant_group<PACK_INT4, ACT, 64>(
            x, in_dtype, qout, scaleout, M, K, stream);
    } else {
        launch_convrot_quant_group<PACK_INT4, ACT, 256>(
            x, in_dtype, qout, scaleout, M, K, stream);
    }
}

}  // namespace comfy::hip_backend
