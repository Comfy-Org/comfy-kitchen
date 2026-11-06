// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Connected components of an edge list by lock-free union-find (ECL-CC, Jaiganesh &
// Burtscher 2018); a port of backends/cuda/ops/connected_components.cu. Each edge hooks
// the larger root under the smaller with atomicCAS, so parents always have smaller
// indices and each root is its component's smallest node. Since pointers only
// decrease, racy path-halving writes still point at an ancestor and need no atomics.
// No thread waits on another, so the kernel needs no forward-progress guarantee.
#include <hip/hip_runtime.h>
#include <cstdint>

namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ int find_root(volatile int* parent, int x) {
    int p = parent[x];
    while (p != x) {
        const int gp = parent[p];
        parent[x] = gp;  // halving
        x = p;
        p = gp;
    }
    return x;
}

template <typename Index>
__global__ void union_kernel(const Index* __restrict__ edges, int64_t n_edges, int* parent) {
    const int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= n_edges) return;
    int a = find_root(parent, static_cast<int>(edges[2 * e]));
    int b = find_root(parent, static_cast<int>(edges[2 * e + 1]));
    while (a != b) {
        if (a < b) { const int t = a; a = b; b = t; }
        const int old = atomicCAS(parent + a, a, b);  // hook root a under b
        if (old == a) break;
        a = find_root(parent, old);  // a was hooked elsewhere meanwhile
        b = find_root(parent, b);
    }
}

__global__ void label_kernel(int n_nodes, int* parent, int64_t* labels) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nodes) return;
    labels[i] = find_root(parent, i);
}

}  // namespace

// parent: int32 (n_nodes,) holding 0..n_nodes-1; labels: int64 (n_nodes,).
extern "C" void launch_connected_components(const void* edges, int64_t n_edges, bool edges_int64, int n_nodes,
                                            void* parent, void* labels, hipStream_t stream) {
    if (n_nodes <= 0) return;
    if (n_edges > 0) {
        const unsigned int blocks = static_cast<unsigned int>((n_edges + kThreads - 1) / kThreads);
        if (edges_int64)
            union_kernel<int64_t><<<blocks, kThreads, 0, stream>>>(static_cast<const int64_t*>(edges), n_edges,
                                                                  static_cast<int*>(parent));
        else
            union_kernel<int32_t><<<blocks, kThreads, 0, stream>>>(static_cast<const int32_t*>(edges), n_edges,
                                                                  static_cast<int*>(parent));
    }
    label_kernel<<<static_cast<unsigned int>((n_nodes + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        n_nodes, static_cast<int*>(parent), static_cast<int64_t*>(labels));
}
