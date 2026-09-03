// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// V quantize and transpose: the exact kernel's PV A operand needs consecutive KEYS
// per channel, so V is stored [B*H, D, Tp] int8 through shared memory. The CUDA
// source permutes the key axis per 64-block for its MMA layout; RDNA's WMMA reads
// a contiguous K slice, so the key axis stays in natural order here.
#include <hip/hip_runtime.h>

#include <cstdint>

#include "sol_common.h"

namespace comfy::hip_backend::sol {
namespace {

constexpr int HD = kHeadDim;
constexpr int kThreads = 256;
constexpr int kLdsPad = HD + 16;  // must be a multiple of 16: phase 1 writes 16 bytes

template <int TT, typename E>
__global__ __launch_bounds__(kThreads) void vquant_transpose(const E* __restrict__ v,
                                                             const float* __restrict__ vsc,
                                                             int8_t* __restrict__ vT, int T,
                                                             int Tp, int H, int64_t sb, int64_t st,
                                                             int64_t sh) {
    __shared__ __attribute__((aligned(16))) int8_t sV[TT * kLdsPad];
    const int bh = blockIdx.y, head = bh % H, batch = bh / H;
    const int t0 = blockIdx.x * TT;

    // phase 1: coalesced read (D contiguous) -> smem rows
    for (int idx = threadIdx.x; idx < TT * (HD / 16); idx += kThreads) {
        const int t = idx / (HD / 16), c16 = (idx % (HD / 16)) * 16;
        __attribute__((aligned(16))) int8_t out[16];
        if (t0 + t < T) {
            const E* src =
                v + batch * sb + static_cast<int64_t>(t0 + t) * st + head * sh + c16;
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                out[j] = q8(static_cast<float>(src[j]), 1.f / vsc[bh * HD + c16 + j]);
            }
        } else {
#pragma unroll
            for (int j = 0; j < 16; ++j) out[j] = 0;
        }
        *reinterpret_cast<uint4*>(sV + t * kLdsPad + c16) = *reinterpret_cast<uint4*>(out);
    }
    __syncthreads();

    // phase 2: 4x4 byte register transposes -> coalesced write (T contiguous)
    int8_t* dst0 = vT + static_cast<int64_t>(bh) * HD * Tp + t0;
    for (int idx = threadIdx.x; idx < (HD / 4) * (TT / 4); idx += kThreads) {
        const int cg = idx / (TT / 4), tg = (idx % (TT / 4)) * 4;
        if (t0 + tg >= Tp) continue;
        uint32_t r[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            r[j] = *reinterpret_cast<const uint32_t*>(sV + (tg + j) * kLdsPad + cg * 4);
        }
        // r[j] = channels 4cg..+3 of token tg+j  ->  w[i] = channel 4cg+i over tokens tg..+3
        uint32_t w[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            w[i] = static_cast<uint32_t>((r[0] >> (i * 8)) & 0xffu) |
                   (static_cast<uint32_t>((r[1] >> (i * 8)) & 0xffu) << 8) |
                   (static_cast<uint32_t>((r[2] >> (i * 8)) & 0xffu) << 16) |
                   (static_cast<uint32_t>((r[3] >> (i * 8)) & 0xffu) << 24);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            *reinterpret_cast<uint32_t*>(dst0 + static_cast<int64_t>(cg * 4 + i) * Tp + tg) = w[i];
        }
    }
}

}  // namespace
}  // namespace comfy::hip_backend::sol

using namespace comfy::hip_backend::sol;

void launch_sol_vtranspose(const void* v, const void* vsc, void* vT, int B, int T, int Tp, int H,
                           int64_t sb, int64_t st, int64_t sh, int elem, hipStream_t stream) {
    dim3 grid((T + 255) / 256, B * H);
    if (elem == kSolFp16) {
        vquant_transpose<256, _Float16><<<grid, kThreads, 0, stream>>>(
            static_cast<const _Float16*>(v), static_cast<const float*>(vsc),
            static_cast<int8_t*>(vT), T, Tp, H, sb, st, sh);
    } else {
        vquant_transpose<256, __bf16><<<grid, kThreads, 0, stream>>>(
            static_cast<const __bf16*>(v), static_cast<const float*>(vsc),
            static_cast<int8_t*>(vT), T, Tp, H, sb, st, sh);
    }
}
