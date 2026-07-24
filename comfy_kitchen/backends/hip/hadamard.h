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

// The kernel's static LDS: the g[256] and red[256] reductions below.
constexpr size_t kConvrotStaticLds = 2 * 256 * sizeof(float);

// convrot_quant_kernel stages the whole rotated row in dynamic LDS, so K is
// bounded by what is left of the workgroup budget. Past it the launch does not
// fail cleanly, so the wrappers fall back to eager instead. 0 means unknown.
inline int convrot_max_k() {
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
                            sizeof(__bf16));
}

inline void check_convrot_k(int k, int group_size) {
    // The kernel rotates K/G whole groups but reads back all K entries of the row
    // buffer, so a partial trailing group would quantize uninitialized LDS.
    if (group_size <= 0 || k % group_size != 0) {
        throw std::runtime_error("convrot: K=" + std::to_string(k) +
                                 " is not divisible by group_size=" + std::to_string(group_size));
    }
    const int max_k = convrot_max_k();
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

template <bool PACK_INT4>
__global__ __launch_bounds__(256) void convrot_quant_kernel(
    const void* __restrict__ x, int in_dtype,
    int8_t* __restrict__ qout, float* __restrict__ scaleout,
    int M, int K, int G) {

    const float h4[4][4] = {{1, 1, 1, -1}, {1, 1, -1, 1}, {1, -1, 1, 1}, {-1, 1, 1, 1}};
    __shared__ float g[256];
    __shared__ float red[256];
    extern __shared__ __bf16 rowbuf[];  // K entries: the rotated row

    const int row = blockIdx.x;
    const int t = threadIdx.x;

    int nstages = 0;
    while ((1 << (2 * nstages)) < G) nstages++;  // log4(G)

    const int gpw = 256 / G;      // groups handled per pass
    const int glocal = t / G;     // this thread's group within the pass
    const int e = t % G;          // element within the group
    const int gbase_idx = glocal * G;
    const float norm = rsqrtf(static_cast<float>(G));
    const int ngrp = K / G;

    float lmax = 0.0f;
    for (int gbase = 0; gbase < ngrp; gbase += gpw) {
        const int grp = gbase + glocal;
        const bool active = grp < ngrp;
        g[t] = active ? load_in(x, static_cast<int64_t>(row) * K + static_cast<int64_t>(grp) * G + e,
                                in_dtype)
                      : 0.0f;
        __syncthreads();

        for (int stage = 0; stage < nstages; ++stage) {
            const int stride = 1 << (2 * stage);
            const int ds = (e / stride) & 3;
            const int b = gbase_idx + (e - ds * stride);
            const float v0 = g[b], v1 = g[b + stride], v2 = g[b + 2 * stride], v3 = g[b + 3 * stride];
            const float nv = h4[ds][0] * v0 + h4[ds][1] * v1 + h4[ds][2] * v2 + h4[ds][3] * v3;
            __syncthreads();
            g[t] = nv;
            __syncthreads();
        }

        if (active) {
            const float tv = g[t] * norm;
            const __bf16 bv = static_cast<__bf16>(tv);
            rowbuf[static_cast<int64_t>(grp) * G + e] = bv;
            lmax = fmaxf(lmax, fabsf(static_cast<float>(bv)));
        }
        __syncthreads();
    }

    red[t] = lmax;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (t < s) red[t] = fmaxf(red[t], red[t + s]);
        __syncthreads();
    }

    constexpr float kQMax = PACK_INT4 ? 7.0f : 127.0f;
    const float rowmax = fmaxf(red[0], 1e-10f);
    const float scale = rowmax / kQMax;
    const float inv = kQMax / rowmax;
    if (t == 0) scaleout[row] = scale;

    if constexpr (PACK_INT4) {
        const int Kp = K / 2;
        for (int jb = t; jb < Kp; jb += 256) {
            const float a = static_cast<float>(rowbuf[2 * jb]);
            const float b = static_cast<float>(rowbuf[2 * jb + 1]);
            int qa = static_cast<int>(rintf(a * inv));
            int qb = static_cast<int>(rintf(b * inv));
            qa = qa < -7 ? -7 : (qa > 7 ? 7 : qa);
            qb = qb < -7 ? -7 : (qb > 7 ? 7 : qb);
            qout[static_cast<int64_t>(row) * Kp + jb] =
                static_cast<int8_t>(((qa & 0xF) | ((qb & 0xF) << 4)) & 0xFF);
        }
    } else {
        for (int j = t; j < K; j += 256) {
            const float v = static_cast<float>(rowbuf[j]);
            int q = static_cast<int>(rintf(v * inv));
            q = q < -127 ? -127 : (q > 127 ? 127 : q);
            qout[static_cast<int64_t>(row) * K + j] = static_cast<int8_t>(q);
        }
    }
}

}  // namespace comfy::hip_backend
