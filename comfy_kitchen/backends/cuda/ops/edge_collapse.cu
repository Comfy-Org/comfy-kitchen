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

// Edge-collapse checks for decimation. One thread per candidate edge (a, b) walks the
// vertex-to-face fans of a and b and reports, over the faces that keep one endpoint, how
// many flip their normal and how skinny they get, plus the link condition.
//
// The fans are filled here with atomic cursors, so their order varies between runs; every
// reduction over a fan is order-independent (counts, set tests, a fixed-point sum).

#include <cuda_runtime.h>

#include "utils.cuh"

namespace comfy {
namespace edge_collapse {

constexpr int kThreads = 128;
constexpr float kThin = 1e-6f;  // |normal| / sum of squared edge lengths below this: no usable normal
constexpr float kFixed = 16777216.0f;  // skinny terms in [0, 1] are summed in 24-bit fixed point

__global__ void fill_fan_kernel(const int* __restrict__ corners, const int* __restrict__ offsets, int* __restrict__ cursor,
                                int* __restrict__ fan, int n_corners) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_corners) return;
    const int v = corners[i];
    fan[offsets[v] + atomicAdd(cursor + v, 1)] = i / 3;
}

__device__ __forceinline__ float3 load3(const float* p, int i) {
    return make_float3(p[3 * i], p[3 * i + 1], p[3 * i + 2]);
}

__device__ __forceinline__ float3 sub(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }

__device__ __forceinline__ float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

__device__ __forceinline__ float3 cross(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__device__ __forceinline__ bool fan_has(const int* faces, const int* fan, int begin, int end, int x) {
    for (int k = begin; k < end; k++) {
        const int* t = faces + 3 * fan[k];
        if (t[0] == x || t[1] == x || t[2] == x) return true;
    }
    return false;
}

// a and b share at most n_on_edge distinct neighbours. Neighbours are taken once each from the
// smaller fan and each adjacency test walks the smaller of the two fans, so high-valence hubs stay cheap.
__device__ bool link_condition(const int* faces, const int* offsets, const int* fan, int a, int b, int n_on_edge) {
    const bool a_small = offsets[a + 1] - offsets[a] <= offsets[b + 1] - offsets[b];
    const int s = a_small ? a : b, h = a_small ? b : a;
    const int s0 = offsets[s], s1 = offsets[s + 1], h0 = offsets[h], h1 = offsets[h + 1];
    int n_common = 0;
    for (int k = s0; k < s1; k++) {
        const int* t = faces + 3 * fan[k];
        for (int j = 0; j < 3; j++) {
            const int x = t[j];
            if (x == a || x == b || fan_has(faces, fan, s0, k, x)) continue;
            const int x0 = offsets[x], x1 = offsets[x + 1];
            const bool adjacent = x1 - x0 < h1 - h0 ? fan_has(faces, fan, x0, x1, h) : fan_has(faces, fan, h0, h1, x);
            if (adjacent && ++n_common > n_on_edge) return false;
        }
    }
    return true;
}

__global__ void checks_kernel(const float* __restrict__ verts, const int* __restrict__ faces,
                              const int* __restrict__ offsets, const int* __restrict__ fan,
                              const int* __restrict__ edges, const float* __restrict__ positions, int n_edges,
                              float cos_threshold, int* __restrict__ flips, float* __restrict__ skinny,
                              bool* __restrict__ link_ok) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_edges) return;
    const int a = edges[2 * e], b = edges[2 * e + 1];
    const float3 p = load3(positions, e);
    const float shape_scale = 2.0f * sqrtf(3.0f);  // 4√3 · area / Σ len² is 1 for an equilateral triangle

    int n_flip = 0, n_moved = 0, n_on_edge = 0;
    long long skinny_sum = 0;
    for (int side = 0; side < 2; side++) {
        const int v = side ? b : a, other = side ? a : b;
        for (int k = offsets[v]; k < offsets[v + 1]; k++) {
            const int* t = faces + 3 * fan[k];
            if (t[0] == other || t[1] == other || t[2] == other) {
                n_on_edge += side == 0;  // removed by the collapse
                continue;
            }
            const float3 p0 = load3(verts, t[0]), p1 = load3(verts, t[1]), p2 = load3(verts, t[2]);
            const float3 q0 = t[0] == v ? p : p0, q1 = t[1] == v ? p : p1, q2 = t[2] == v ? p : p2;
            const float3 d01 = sub(p1, p0), d02 = sub(p2, p0), d12 = sub(p2, p1);
            const float3 n_old = cross(d01, d02);
            const float len_old = sqrtf(dot(n_old, n_old));
            const float3 e01 = sub(q1, q0), e02 = sub(q2, q0), e12 = sub(q2, q1);
            const float3 n_new = cross(e01, e02);
            const float len_new = sqrtf(dot(n_new, n_new));
            const float sq_old = dot(d01, d01) + dot(d02, d02) + dot(d12, d12);
            const float sq_new = dot(e01, e01) + dot(e02, e02) + dot(e12, e12);
            // faces too thin for a normal (shape below ~3e-6 at any scale) never count as flipped
            if (len_old > kThin * sq_old && len_new > kThin * sq_new &&
                dot(n_old, n_new) < cos_threshold * len_old * len_new)
                n_flip++;
            const float shape = shape_scale * len_new / fmaxf(sq_new, 1e-20f);
            skinny_sum += static_cast<long long>((1.0f - fminf(fmaxf(shape, 0.0f), 1.0f)) * kFixed);
            n_moved++;
        }
    }

    flips[e] = n_flip;
    skinny[e] = n_moved > 0 ? static_cast<float>(skinny_sum) / (kFixed * n_moved) : 0.0f;
    link_ok[e] = link_condition(faces, offsets, fan, a, b, n_on_edge);
}

}  // namespace edge_collapse
}  // namespace comfy

using namespace comfy::edge_collapse;

extern "C" {

void launch_edge_collapse_checks(const void* verts, const void* faces, const void* offsets, void* cursor, void* fan,
                                 const void* edges, const void* positions, int n_corners, int n_edges,
                                 float cos_threshold, void* flips, void* skinny, void* link_ok, cudaStream_t stream) {
    if (n_edges <= 0) return;
    if (n_corners > 0) {
        fill_fan_kernel<<<static_cast<unsigned int>((n_corners + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
            static_cast<const int*>(faces), static_cast<const int*>(offsets), static_cast<int*>(cursor),
            static_cast<int*>(fan), n_corners);
        CUDA_CHECK(cudaGetLastError());
    }
    checks_kernel<<<static_cast<unsigned int>((n_edges + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        static_cast<const float*>(verts), static_cast<const int*>(faces), static_cast<const int*>(offsets),
        static_cast<const int*>(fan), static_cast<const int*>(edges), static_cast<const float*>(positions), n_edges,
        cos_threshold, static_cast<int*>(flips), static_cast<float*>(skinny), static_cast<bool*>(link_ok));
    CUDA_CHECK(cudaGetLastError());
}

}  // extern "C"
