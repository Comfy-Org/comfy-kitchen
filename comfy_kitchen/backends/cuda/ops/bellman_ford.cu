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

// Jacobi Bellman-Ford over in-arc CSR lists in one cooperative kernel, kLanes lanes per node. Each sweep lowers every
// node to its best in-arc (tail distance + weight); a node that drops by more than kTol takes that arc as parent.
// Every kCheck sweeps it stops if nothing dropped, or if the parent graph has a cycle: pointer jumping moves every
// node onto the cycle its parent walk ends in, and those nodes are marked in on_cycle.

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <algorithm>

#include "utils.cuh"

namespace cg = cooperative_groups;

namespace comfy {
namespace bellman_ford {

constexpr int kThreads = 256;
constexpr int kLanes = 8;  // lanes per node
constexpr int kCheck = 16;
constexpr double kTol = 1e-12;

struct Sweep {
    int n, E;
    const int* indptr;
    const int* tail;
    const double* w;
    double* dist;  // (2, n) Jacobi buffers
    int* parent;
    int* jump;     // (2, n) pointer-jumping buffers
    int* flags;    // (6,) 3 dropped flags, cycle found, bad index, buffer holding the result
    uint8_t* on_cycle;
    int max_iters, steps;
};

__global__ void sweep_kernel(Sweep s) {
    const cg::grid_group grid = cg::this_grid();
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const cg::thread_block_tile<kLanes> tile = cg::tiled_partition<kLanes>(cg::this_thread_block());
    const int lane = tile.thread_rank();
    for (int h = tid; h <= s.n; h += stride)
        if (s.indptr[h] < 0 || s.indptr[h] > s.E) s.flags[4] = 1;
    for (int a = tid; a < s.E; a += stride)
        if (s.tail[a] < 0 || s.tail[a] >= s.n) s.flags[4] = 1;
    grid.sync();
    if (__ldcg(s.flags + 4)) return;
    int it = 0;
    for (; it < s.max_iters; ++it) {
        const double* cur = s.dist + (it & 1) * s.n;
        double* next = s.dist + ((it + 1) & 1) * s.n;
        if (tid == 0) s.flags[(it + 1) % 3] = 0;
        for (int h = tid / kLanes; h < s.n; h += stride / kLanes) {
            const double d = __ldcg(cur + h);
            const int a0 = s.indptr[h], a1 = s.indptr[h + 1];
            double best = d;
            for (int a = a0 + lane; a < a1; a += kLanes) best = fmin(best, __ldcg(cur + s.tail[a]) + s.w[a]);
            best = cg::reduce(tile, best, cg::less<double>());
            if (best < d - kTol) {
                int arc = -1;
                for (int a = a0 + lane; a < a1; a += kLanes)
                    if (__ldcg(cur + s.tail[a]) + s.w[a] == best) arc = a;
                arc = cg::reduce(tile, arc, cg::greater<int>());
                if (lane == 0) {
                    s.parent[h] = arc;
                    s.flags[it % 3] = 1;
                }
            }
            if (lane == 0) next[h] = best;
        }
        grid.sync();
        if ((it + 1) % kCheck && it + 1 < s.max_iters) continue;
        if (!__ldcg(s.flags + it % 3)) break;
        for (int v = tid; v < s.n; v += stride) {
            const int p = __ldcg(s.parent + v);
            s.jump[v] = p >= 0 ? s.tail[p] : v;
        }
        grid.sync();
        for (int k = 0; k < s.steps; ++k) {
            const int* a = s.jump + (k & 1) * s.n;
            int* b = s.jump + ((k + 1) & 1) * s.n;
            for (int v = tid; v < s.n; v += stride) b[v] = __ldcg(a + __ldcg(a + v));
            grid.sync();
        }
        const int* root = s.jump + (s.steps & 1) * s.n;
        for (int v = tid; v < s.n; v += stride) {
            const int r = __ldcg(root + v);
            if (__ldcg(s.parent + r) >= 0) {
                s.flags[3] = 1;
                s.on_cycle[r] = 1;
            }
        }
        grid.sync();
        if (__ldcg(s.flags + 3)) break;
    }
    if (tid == 0) s.flags[5] = it < s.max_iters ? (it + 1) & 1 : it & 1;    // the buffer the last sweep wrote
}

}  // namespace bellman_ford
}  // namespace comfy

using namespace comfy::bellman_ford;

extern "C" {

// dist (2, n): start distances in the first row; the result is in row flags[5]. parent -1, flags and on_cycle 0 on entry.
// Sets flags[4] and does nothing on an out-of-range index.
void launch_bellman_ford(int n, int E, const void* indptr, const void* tail, const void* w, void* dist, void* parent,
                         void* jump, void* flags, void* on_cycle, int max_iters, cudaStream_t stream) {
    if (n <= 0 || max_iters <= 0) return;
    int steps = 1;
    while ((1LL << steps) < n) ++steps;
    Sweep s{n, E, static_cast<const int*>(indptr), static_cast<const int*>(tail), static_cast<const double*>(w),
            static_cast<double*>(dist), static_cast<int*>(parent), static_cast<int*>(jump), static_cast<int*>(flags),
            static_cast<uint8_t*>(on_cycle), max_iters, steps};
    const int blocks = std::min(2 * comfy::sm_count(), static_cast<int>((static_cast<long long>(n) * kLanes + kThreads - 1) / kThreads));
    comfy::launch_cooperative(sweep_kernel, blocks, kThreads, s, stream);
}

}  // extern "C"
