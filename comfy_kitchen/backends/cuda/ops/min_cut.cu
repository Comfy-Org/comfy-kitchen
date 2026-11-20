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

// Min s-t cut of a bounded-degree graph by worklist push-relabel. Node i's K slots hold its
// neighbours (-1 = none) with symmetric residual capacities r[i, k] (the reverse edge sits at
// r[j, rev[i, k]]); s_cap arrives as initial excess, t_cap is the residual to the sink. One
// iteration pushes from the active list (push_kernel: to the sink at height 1, then along
// admissible slots; an edge is admissible in one direction at most, so the pusher owns both
// residuals of the edge and the receiver's incoming slot for the iteration, and no float
// atomics are needed), then the queued receivers and stuck pushers gather their incoming
// amounts in slot order and relabel to min(height of a residual neighbour) + 1 against the
// old heights (relabel_kernel, committed by commit_kernel), which keeps the result independent
// of thread order. Heights are reset now and then by a frontier BFS from the sink over reverse
// residual edges (global_relabel); nodes it cannot reach stay at `big`, and those are the
// source side when no active node is left.

#include <cuda_runtime.h>
#include <cstdint>

#include "utils.cuh"

namespace comfy {
namespace min_cut {

constexpr int kThreads = 128;

inline unsigned int blocks_for(int n) { return static_cast<unsigned int>((n + kThreads - 1) / kThreads); }

__global__ void init_kernel(int n, int K, const int* __restrict__ nbr, const float* __restrict__ cap, float* r,
                            int8_t* rev) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    for (int k = 0; k < K; ++k) {
        const int j = nbr[i * K + k];
        int m = 0;
        if (j >= 0)
            for (int s = 0; s < K; ++s)
                if (nbr[j * K + s] == i) { m = s; break; }
        rev[i * K + k] = static_cast<int8_t>(m);
        r[i * K + k] = j >= 0 ? cap[i * K + k] : 0.0f;
    }
}

__global__ void bfs_init_kernel(int n, const float* __restrict__ rt, float tol, int big, int* h, int* frontier,
                                int* count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const bool to_sink = rt[i] > tol;
    h[i] = to_sink ? 1 : big;
    if (to_sink) frontier[atomicAdd(count, 1)] = i;
}

// node i joins level d when it has residual towards a node j of level d - 1: that is i's slot rev[j, m]
__global__ void bfs_level_kernel(const int* __restrict__ frontier, int count, int d, int K, const int* __restrict__ nbr,
                                 const int8_t* __restrict__ rev, const float* __restrict__ r, float tol, int big, int* h,
                                 int* next, int* next_count) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count) return;
    const int j = frontier[t];
    for (int m = 0; m < K; ++m) {
        const int i = nbr[j * K + m];
        if (i < 0 || h[i] != big) continue;
        if (r[i * K + rev[j * K + m]] > tol && atomicCAS(h + i, big, d) == big) next[atomicAdd(next_count, 1)] = i;
    }
}

__global__ void rebuild_active_kernel(int n, const float* __restrict__ e, const int* __restrict__ h, float tol, int big,
                                      int* list, int* count, int* queued) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    queued[i] = 0;
    if (e[i] > tol && h[i] < big) list[atomicAdd(count, 1)] = i;
}

__global__ void push_kernel(const int* __restrict__ list, const int* __restrict__ count, int K,
                            const int* __restrict__ nbr, const int8_t* __restrict__ rev, float* r, float* e, float* rt,
                            const int* __restrict__ h, float* inc, float tol, int big, int* list_out, int* count_out,
                            int* queued) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= *count) return;
    const int i = list[t];
    const int hi = h[i];
    if (hi >= big) return;
    float ex = e[i];  // ours alone during the pushes: arrivals land in inc
    if (ex <= tol) return;
    if (hi == 1 && rt[i] > tol) {
        const float amt = fminf(ex, rt[i]);
        rt[i] -= amt;
        ex -= amt;
    }
    for (int k = 0; k < K && ex > tol; ++k) {
        const int j = nbr[i * K + k];
        if (j < 0) continue;
        const float rk = r[i * K + k];
        if (rk <= tol || hi != h[j] + 1) continue;
        const float amt = fminf(ex, rk);
        const int back = j * K + rev[i * K + k];  // j's slot towards us; nobody else touches it this iteration
        r[i * K + k] = rk - amt;
        r[back] += amt;
        inc[back] = amt;
        ex -= amt;
        if (atomicExch(queued + j, 1) == 0) list_out[atomicAdd(count_out, 1)] = j;
    }
    e[i] = ex;
    if (ex > tol && atomicExch(queued + i, 1) == 0) list_out[atomicAdd(count_out, 1)] = i;
}

__global__ void relabel_kernel(const int* __restrict__ list, const int* __restrict__ count, int K,
                               const int* __restrict__ nbr, const float* __restrict__ r, const float* __restrict__ rt,
                               float* e, float* inc, const int* __restrict__ h, int* hn, float tol, int big, int* queued,
                               int* active) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= *count) return;
    const int i = list[t];
    queued[i] = 0;
    float ei = e[i];
    for (int s = 0; s < K; ++s) {  // slot order, so the sum does not depend on who pushed first
        ei += inc[i * K + s];
        inc[i * K + s] = 0.0f;
    }
    e[i] = ei;
    int nh = h[i];
    if (ei > tol && nh < big) {
        int hmin = rt[i] > tol ? 0 : big;
        for (int k = 0; k < K; ++k) {
            const int j = nbr[i * K + k];
            if (j >= 0 && r[i * K + k] > tol) hmin = min(hmin, h[j]);
        }
        nh = max(nh, min(hmin + 1, big));
        if (nh < big) atomicAdd(active, 1);
    }
    hn[i] = nh;
}

__global__ void commit_kernel(const int* __restrict__ list, const int* __restrict__ count, const int* __restrict__ hn,
                              int* h) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= *count) return;
    h[list[t]] = hn[list[t]];
}

}  // namespace min_cut
}  // namespace comfy

using namespace comfy::min_cut;

extern "C" {

void launch_min_cut_init(int n, int K, const void* nbr, const void* cap, void* r, void* rev, cudaStream_t stream) {
    if (n <= 0) return;
    init_kernel<<<blocks_for(n), kThreads, 0, stream>>>(n, K, static_cast<const int*>(nbr),
                                                        static_cast<const float*>(cap), static_cast<float*>(r),
                                                        static_cast<int8_t*>(rev));
    CUDA_CHECK(cudaGetLastError());
}

// Heights from a BFS out of the sink; `lists` holds two frontier buffers of n ints, `counts` two ints.
void launch_min_cut_global_relabel(int n, int K, const void* nbr, const void* rev, const void* r, const void* rt,
                                   float tol, int big, void* h, void* lists, void* counts, cudaStream_t stream) {
    if (n <= 0) return;
    int* frontier[2] = {static_cast<int*>(lists), static_cast<int*>(lists) + n};
    int* count[2] = {static_cast<int*>(counts), static_cast<int*>(counts) + 1};
    CUDA_CHECK(cudaMemsetAsync(count[0], 0, sizeof(int), stream));
    bfs_init_kernel<<<blocks_for(n), kThreads, 0, stream>>>(n, static_cast<const float*>(rt), tol, big,
                                                            static_cast<int*>(h), frontier[0], count[0]);
    int cur = 0, size = 0;
    CUDA_CHECK(cudaMemcpyAsync(&size, count[0], sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (int d = 2; size > 0; ++d) {
        CUDA_CHECK(cudaMemsetAsync(count[1 - cur], 0, sizeof(int), stream));
        bfs_level_kernel<<<blocks_for(size), kThreads, 0, stream>>>(
            frontier[cur], size, d, K, static_cast<const int*>(nbr), static_cast<const int8_t*>(rev),
            static_cast<const float*>(r), tol, big, static_cast<int*>(h), frontier[1 - cur], count[1 - cur]);
        CUDA_CHECK(cudaMemcpyAsync(&size, count[1 - cur], sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        cur = 1 - cur;
    }
    CUDA_CHECK(cudaGetLastError());
}

void launch_min_cut_rebuild_active(int n, const void* e, const void* h, float tol, int big, void* list, void* count,
                                   void* queued, cudaStream_t stream) {
    if (n <= 0) return;
    CUDA_CHECK(cudaMemsetAsync(count, 0, sizeof(int), stream));
    rebuild_active_kernel<<<blocks_for(n), kThreads, 0, stream>>>(n, static_cast<const float*>(e),
                                                                  static_cast<const int*>(h), tol, big,
                                                                  static_cast<int*>(list), static_cast<int*>(count),
                                                                  static_cast<int*>(queued));
    CUDA_CHECK(cudaGetLastError());
}

// One iteration: push from `list` (its length read from `count` on the device) into `list_out`, gather and relabel
// `list_out` against the old heights, commit the new ones; `active` receives how many can still push.
void launch_min_cut_iterate(int n, int K, const void* nbr, const void* rev, void* r, void* e, void* rt, void* h, void* inc,
                            void* hn, float tol, int big, const void* list, const void* count, void* list_out,
                            void* count_out, void* queued, void* active, cudaStream_t stream) {
    if (n <= 0) return;
    CUDA_CHECK(cudaMemsetAsync(count_out, 0, sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(active, 0, sizeof(int), stream));
    push_kernel<<<blocks_for(n), kThreads, 0, stream>>>(
        static_cast<const int*>(list), static_cast<const int*>(count), K, static_cast<const int*>(nbr),
        static_cast<const int8_t*>(rev), static_cast<float*>(r), static_cast<float*>(e), static_cast<float*>(rt),
        static_cast<const int*>(h), static_cast<float*>(inc), tol, big, static_cast<int*>(list_out),
        static_cast<int*>(count_out), static_cast<int*>(queued));
    relabel_kernel<<<blocks_for(n), kThreads, 0, stream>>>(
        static_cast<const int*>(list_out), static_cast<const int*>(count_out), K, static_cast<const int*>(nbr),
        static_cast<const float*>(r), static_cast<const float*>(rt), static_cast<float*>(e), static_cast<float*>(inc),
        static_cast<const int*>(h), static_cast<int*>(hn), tol, big, static_cast<int*>(queued),
        static_cast<int*>(active));
    commit_kernel<<<blocks_for(n), kThreads, 0, stream>>>(static_cast<const int*>(list_out),
                                                           static_cast<const int*>(count_out),
                                                           static_cast<const int*>(hn), static_cast<int*>(h));
    CUDA_CHECK(cudaGetLastError());
}

}  // extern "C"
