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

// Conjugate gradients for A x = b with a symmetric positive definite CSR matrix and k <= kMaxCols right-hand sides
// (columns of row-major b and x), in one cooperative kernel. Dot products are summed in a fixed order, so results
// do not depend on thread timing. Consecutive sums use alternate halves of the partials, since a block may start the
// next sum while others still read the last one. All columns iterate until every residual is within tol.

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <algorithm>

#include "utils.cuh"

namespace cg = cooperative_groups;

namespace comfy {
namespace conjugate_gradient {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kMaxCols = 2;

struct Solve {
    int n, k, nnz;
    const int* indptr;
    const int* cols;
    const double* vals;
    const double* b;
    double* x;
    double* r;
    double* p;
    double* ap;
    double* partial;  // (2, blocks, 2 * kMaxCols)
    int max_iters;
    double tol2;
};

// sums m values per block into partial[blockIdx.x], then over all blocks into total
__device__ void grid_sum(const cg::grid_group& grid, double* partial, double* acc, int m, double* total) {
    __shared__ double warp_sums[kWarps][2 * kMaxCols];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (int c = 0; c < m; ++c) {
        double v = acc[c];
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0) warp_sums[warp][c] = v;
    }
    __syncthreads();
    if (threadIdx.x < m) {
        double v = 0.0;
        for (int w = 0; w < kWarps; ++w) v += warp_sums[w][threadIdx.x];
        partial[blockIdx.x * 2 * kMaxCols + threadIdx.x] = v;
    }
    grid.sync();
    __shared__ double sums[2 * kMaxCols];
    if (warp == 0) {
        for (int c = 0; c < m; ++c) {
            double v = 0.0;
            for (int blk = lane; blk < static_cast<int>(gridDim.x); blk += 32)
                v += __ldcg(partial + blk * 2 * kMaxCols + c);
            for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
            if (lane == 0) sums[c] = v;
        }
    }
    __syncthreads();
    for (int c = 0; c < m; ++c) total[c] = sums[c];
    __syncthreads();  // sums is reused by the next call
}

__device__ __forceinline__ double row_dot(const Solve& s, int i, const double* v, int c) {
    double out = 0.0;
    for (int a = s.indptr[i]; a < s.indptr[i + 1]; ++a) out += s.vals[a] * __ldcg(v + static_cast<long long>(s.cols[a]) * s.k + c);
    return out;
}

__global__ void cg_kernel(Solve s) {
    const cg::grid_group grid = cg::this_grid();
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int k = s.k;
    double acc[2 * kMaxCols], rs[kMaxCols], stop[kMaxCols], total[2 * kMaxCols];
    double* const half[2] = {s.partial, s.partial + gridDim.x * 2 * kMaxCols};
    int flip = 0;
    for (int c = 0; c < 2 * k; ++c) acc[c] = 0.0;
    for (int i = tid; i < s.n; i += stride)
        for (int c = 0; c < k; ++c) {
            const long long at = static_cast<long long>(i) * k + c;
            const double ri = s.b[at] - row_dot(s, i, s.x, c);
            s.r[at] = ri;
            s.p[at] = ri;
            acc[c] += ri * ri;
            acc[k + c] += s.b[at] * s.b[at];
        }
    grid_sum(grid, half[flip++ % 2], acc, 2 * k, total);
    for (int c = 0; c < k; ++c) {
        rs[c] = total[c];
        stop[c] = s.tol2 * fmax(total[k + c], 1e-300);
    }
    for (int it = 0; it < s.max_iters; ++it) {
        bool go = false;
        for (int c = 0; c < k; ++c) go |= rs[c] > stop[c];
        if (!go) break;
        for (int c = 0; c < k; ++c) acc[c] = 0.0;
        for (int i = tid; i < s.n; i += stride)
            for (int c = 0; c < k; ++c) {
                const long long at = static_cast<long long>(i) * k + c;
                const double v = row_dot(s, i, s.p, c);
                s.ap[at] = v;
                acc[c] += __ldcg(s.p + at) * v;
            }
        grid_sum(grid, half[flip++ % 2], acc, k, total);
        double alpha[kMaxCols];
        for (int c = 0; c < k; ++c) alpha[c] = rs[c] / fmax(total[c], 1e-300);
        for (int c = 0; c < k; ++c) acc[c] = 0.0;
        for (int i = tid; i < s.n; i += stride)
            for (int c = 0; c < k; ++c) {
                const long long at = static_cast<long long>(i) * k + c;
                s.x[at] += alpha[c] * __ldcg(s.p + at);
                const double ri = __ldcg(s.r + at) - alpha[c] * __ldcg(s.ap + at);
                s.r[at] = ri;
                acc[c] += ri * ri;
            }
        grid_sum(grid, half[flip++ % 2], acc, k, total);
        for (long long at = tid; at < static_cast<long long>(s.n) * k; at += stride)
            s.p[at] = __ldcg(s.r + at) + total[at % k] / fmax(rs[at % k], 1e-300) * __ldcg(s.p + at);
        for (int c = 0; c < k; ++c) rs[c] = total[c];
        grid.sync();
    }
}

}  // namespace conjugate_gradient
}  // namespace comfy

using namespace comfy::conjugate_gradient;

extern "C" {

// x: start in, solution out. work: (3 * n * k + 4 * kMaxCols * SMs) scratch.
void launch_conjugate_gradient(int n, int k, int nnz, const void* indptr, const void* cols, const void* vals,
                               const void* b, void* x, void* work, int max_iters, double tol, cudaStream_t stream) {
    if (n <= 0) return;
    const long long nk = static_cast<long long>(n) * k;
    double* r = static_cast<double*>(work);
    Solve s{n, k, nnz, static_cast<const int*>(indptr), static_cast<const int*>(cols), static_cast<const double*>(vals),
            static_cast<const double*>(b), static_cast<double*>(x), r, r + nk, r + 2 * nk, r + 3 * nk, max_iters,
            tol * tol};
    comfy::launch_cooperative(cg_kernel, std::min(comfy::sm_count(), (n + kThreads - 1) / kThreads), kThreads, s, stream);
}

}  // extern "C"
