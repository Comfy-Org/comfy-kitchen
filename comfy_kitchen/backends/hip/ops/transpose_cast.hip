// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Batched [N, A, B] -> [N, B, A] transpose with a dtype conversion and a divisor, staged
// through a 32x33 LDS tile so both the read and the write are coalesced. It takes an NCDHW x
// (A = C, B = DHW) to the packed conv's scaled NDHWC fp16 copy, where torch's strided copy
// between those layouts runs at 3-36 GB/s on gfx1010, against ~240 GB/s for a plain copy.
#include "../epilogue.h"

namespace comfy::hip_backend {

constexpr int kTransposeTile = 32;
constexpr int kTransposeRows = 8;

// One block per 32x32 tile, flattened into grid.x (A * B can pass grid.y's 65535 tiles);
// grid.y is the batch. out = in / divisor[0].
__global__ void __launch_bounds__(kTransposeTile * kTransposeRows)
transpose_cast_kernel(const void* __restrict__ in, int in_code, void* __restrict__ out,
                      int out_code, int A, int B, int tiles_b,
                      const float* __restrict__ divisor) {
    __shared__ float tile[kTransposeTile][kTransposeTile + 1];
    const int a0 = (blockIdx.x / tiles_b) * kTransposeTile;
    const int b0 = (blockIdx.x % tiles_b) * kTransposeTile;
    const int64_t base = static_cast<int64_t>(blockIdx.y) * A * B;
    const float mul = 1.0f / divisor[0];
    for (int i = threadIdx.y; i < kTransposeTile; i += kTransposeRows) {
        const int a = a0 + i, b = b0 + threadIdx.x;
        if (a < A && b < B) {
            tile[i][threadIdx.x] = load_scalar(in, in_code, base + static_cast<int64_t>(a) * B + b);
        }
    }
    __syncthreads();
    for (int i = threadIdx.y; i < kTransposeTile; i += kTransposeRows) {
        const int a = a0 + threadIdx.x, b = b0 + i;
        if (a < A && b < B) {
            store_scalar(out, out_code, base + static_cast<int64_t>(b) * A + a,
                         tile[threadIdx.x][i] * mul);
        }
    }
}

}  // namespace comfy::hip_backend

extern "C" void launch_transpose_cast_kernel(const void* in, int in_code, void* out, int out_code,
                                             int N, int A, int B, const void* divisor,
                                             hipStream_t stream) {
    using namespace comfy::hip_backend;
    if (N == 0 || A == 0 || B == 0) {
        return;
    }
    const int tiles_a = (A + kTransposeTile - 1) / kTransposeTile;
    const int tiles_b = (B + kTransposeTile - 1) / kTransposeTile;
    const dim3 grid(tiles_a * tiles_b, N);
    const dim3 block(kTransposeTile, kTransposeRows);
    hipLaunchKernelGGL(transpose_cast_kernel, grid, block, 0, stream, in, in_code, out, out_code,
                       A, B, tiles_b, static_cast<const float*>(divisor));
}
