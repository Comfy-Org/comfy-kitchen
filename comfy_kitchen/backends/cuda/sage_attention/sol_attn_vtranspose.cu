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
// V quantize + transpose for the CUDA Sol-Attn path.
//
// The exact kernel's PV B operand needs 4 consecutive *keys* for a fixed
// channel, so V is stored transposed: [B*H, D, Tp] int8 -- the one layout the
// preprocess cannot produce in its own pass. Reads want D contiguous, writes
// want T contiguous, so it goes through shared memory.
//
// The key axis also takes sol::perm_d (per 64-block), which the exact kernel's
// 64-bit fragment loads assume. Applying it to the phase-1 smem row index keeps
// phase 2 a plain contiguous copy -- on the write side it would split each
// 16-byte store into four 4-byte stores at stride 8.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "sol_layout.cuh"

namespace {
using namespace sol;

constexpr int HD = HEAD_DIM;
constexpr int NTHREADS = 256;
constexpr int LDS_PAD = HD + 16;   // must be a multiple of 16: phase 1 writes uint4

__device__ __forceinline__ int8_t q8(float x, float inv) {
    const int v = __float2int_rn(x * inv);
    return (int8_t)max(-127, min(127, v));
}

// Transposing quantizer; phase 2 is a 4x4 byte register transpose.
template <int TT>
__global__ void vquant_transpose(const __nv_bfloat16* __restrict__ v,
                                 const float* __restrict__ vsc,
                                 int8_t* __restrict__ vT,
                                 int T, int Tp, int H,
                                 int64_t sb, int64_t st, int64_t sh) {
    __shared__ int8_t sV[TT * LDS_PAD];
    const int bh = blockIdx.y, head = bh % H, batch = bh / H;
    const int t0 = blockIdx.x * TT;

    // Phase 1: coalesced global read (D contiguous), contiguous smem write.
    for (int idx = threadIdx.x; idx < TT * (HD / 16); idx += NTHREADS) {
        const int t = idx / (HD / 16), c16 = (idx % (HD / 16)) * 16;
        int8_t out[16];
        if (t0 + t < T) {
            // The batch term is load-bearing: without it every batch reads
            // batch 0's V, and no B=1 test can tell the difference.
            const __nv_bfloat16* src =
                v + batch * sb + (int64_t)(t0 + t) * st + head * sh + c16;
            #pragma unroll
            for (int j = 0; j < 16; ++j)
                out[j] = q8(__bfloat162float(src[j]), 1.f / vsc[bh * HD + c16 + j]);
        } else {
            #pragma unroll
            for (int j = 0; j < 16; ++j) out[j] = 0;
        }
        // perm_d on the key axis, per 64-block; a tile spans TT/64 blocks.
        const int tp = (t / 64) * 64 + perm_d(t % 64);
        *reinterpret_cast<uint4*>(sV + tp * LDS_PAD + c16) = *reinterpret_cast<uint4*>(out);
    }
    __syncthreads();

    // Phase 2: coalesced global write (T contiguous), strided smem read. Each
    // thread owns a 4x4 byte block: read 4 uint32, transpose in registers,
    // write 4 uint32 -- 4x fewer smem instructions than byte-at-a-time.
    int8_t* dst0 = vT + (int64_t)bh * HD * Tp + t0;
    for (int idx = threadIdx.x; idx < (HD / 4) * (TT / 4); idx += NTHREADS) {
        const int cg = idx / (TT / 4), tg = (idx % (TT / 4)) * 4;
        if (t0 + tg >= Tp) continue;       // last tile may overrun the row
        uint32_t r[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            r[j] = *reinterpret_cast<const uint32_t*>(sV + (tg + j) * LDS_PAD + cg * 4);
        // r[j] holds channels 4cg..4cg+3 of token tg+j; produce w[i] = channel
        // 4cg+i across tokens tg..tg+3.
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

extern "C" void launch_sol_vtranspose(const void* v, const void* vsc, void* vT,
                                      int B, int T, int Tp, int H,
                                      int64_t sb, int64_t st, int64_t sh,
                                      cudaStream_t stream) {
    dim3 grid((T + 255) / 256, B * H);
    vquant_transpose<256><<<grid, NTHREADS, 0, stream>>>(
        (const __nv_bfloat16*)v, (const float*)vsc, (int8_t*)vT, T, Tp, H, sb, st, sh);
}

