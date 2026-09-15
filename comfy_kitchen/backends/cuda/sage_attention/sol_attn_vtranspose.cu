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
// V quantize + transpose: the exact kernel's PV B operand needs consecutive
// KEYS per channel, so V is stored [B*H, D, Tp] int8 via shared memory, with
// sol::perm_d applied to the key axis per 64-block (on the phase-1 smem row
// index, which keeps phase 2 a contiguous copy).

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "sol_layout.cuh"

namespace {
using namespace sol;

constexpr int HD = HEAD_DIM;
constexpr int NTHREADS = 256;
constexpr int LDS_PAD = HD + 16;   // must be a multiple of 16: phase 1 writes uint4

template <int TT, typename E>
__global__ void vquant_transpose(const E* __restrict__ v,
                                 const float* __restrict__ vsc,
                                 int8_t* __restrict__ vT, int8_t* __restrict__ vRow,
                                 int T, int Tp, int H,
                                 int64_t sb, int64_t st, int64_t sh) {
    __shared__ __align__(16) int8_t sV[TT * LDS_PAD];
    const int bh = blockIdx.y, head = bh % H, batch = bh / H;
    const int t0 = blockIdx.x * TT;

    // phase 1: coalesced read (D contiguous) -> smem rows
    for (int idx = threadIdx.x; idx < TT * (HD / 16); idx += NTHREADS) {
        const int t = idx / (HD / 16), c16 = (idx % (HD / 16)) * 16;
        __align__(16) int8_t out[16];
        if (t0 + t < T) {
            const E* src =
                v + batch * sb + (int64_t)(t0 + t) * st + head * sh + c16;
            #pragma unroll
            for (int j = 0; j < 16; ++j)
                out[j] = q8(to_f32(src[j]), 1.f / vsc[bh * HD + c16 + j]);
        } else {
            #pragma unroll
            for (int j = 0; j < 16; ++j) out[j] = 0;
        }
        if (vRow && t0 + t < Tp)   // row-major copy for the gathered token tiles (token routing)
            *reinterpret_cast<uint4*>(vRow + ((int64_t)bh * Tp + t0 + t) * HD + c16) = *reinterpret_cast<uint4*>(out);
        // perm_d on the key axis, per 64-block
        const int tp = (t / 64) * 64 + perm_d(t % 64);
        *reinterpret_cast<uint4*>(sV + tp * LDS_PAD + c16) = *reinterpret_cast<uint4*>(out);
    }
    __syncthreads();

    // phase 2: 4x4 byte register transposes -> coalesced write (T contiguous)
    int8_t* dst0 = vT + (int64_t)bh * HD * Tp + t0;
    for (int idx = threadIdx.x; idx < (HD / 4) * (TT / 4); idx += NTHREADS) {
        const int cg = idx / (TT / 4), tg = (idx % (TT / 4)) * 4;
        if (t0 + tg >= Tp) continue;
        uint32_t r[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            r[j] = *reinterpret_cast<const uint32_t*>(sV + (tg + j) * LDS_PAD + cg * 4);
        // r[j] = channels 4cg..+3 of token tg+j  ->  w[i] = channel 4cg+i over tokens tg..+3
        uint32_t w[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            w[i] = (uint32_t)((r[0] >> (i * 8)) & 0xffu)
                 | (uint32_t)(((r[1] >> (i * 8)) & 0xffu) << 8)
                 | (uint32_t)(((r[2] >> (i * 8)) & 0xffu) << 16)
                 | (uint32_t)(((r[3] >> (i * 8)) & 0xffu) << 24);
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            *reinterpret_cast<uint32_t*>(dst0 + (int64_t)(cg * 4 + i) * Tp + tg) = w[i];
    }
}

}  // namespace

void launch_sol_vtranspose(const void* v, const void* vsc, void* vT, void* vRow,
                           int B, int T, int Tp, int H,
                           int64_t sb, int64_t st, int64_t sh,
                           int elem, cudaStream_t stream) {
    dim3 grid((T + 255) / 256, B * H);
    if (elem == sol::SOL_FP16)
        vquant_transpose<256, __half><<<grid, NTHREADS, 0, stream>>>(
            (const __half*)v, (const float*)vsc, (int8_t*)vT, (int8_t*)vRow, T, Tp, H, sb, st, sh);
    else
        vquant_transpose<256, __nv_bfloat16><<<grid, NTHREADS, 0, stream>>>(
            (const __nv_bfloat16*)v, (const float*)vsc, (int8_t*)vT, (int8_t*)vRow, T, Tp, H, sb, st, sh);
}

