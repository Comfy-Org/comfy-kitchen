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

// Triangle BVH, closest-point-on-mesh and closest-hit ray queries.
//
// Build: Karras 2012 linear BVH over triangles sorted by the 63-bit Morton code of their
// box centres (index breaks ties), with boxes fitted bottom up by the second child to reach
// each node. Nodes 0..F-2 are internal (0 is the root), F-1..2F-2 are leaves (leaf i is
// sorted triangle i); with F == 1 the root is the single leaf.
//
// Queries: one thread per point or ray, stack traversal nearer child first, pruning boxes no
// closer than the best hit. Closest point is float32 except for thin triangles; rays use the
// watertight float32 test of Woop et al. 2013.

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

#include "utils.cuh"

namespace comfy {
namespace mesh_bvh {

constexpr int kThreads = 256;
constexpr int kMaxStack = 96;  // tree height is bounded by 63 code bits + 32 index bits

__device__ __forceinline__ int common_prefix(const long long* codes, int n_leaves, int i, int j) {
    if (j < 0 || j >= n_leaves) return -1;
    const unsigned long long a = static_cast<unsigned long long>(codes[i]);
    const unsigned long long b = static_cast<unsigned long long>(codes[j]);
    if (a == b) return 64 + __clz(static_cast<unsigned int>(i ^ j));
    return __clzll(a ^ b);
}

__global__ void bvh_hierarchy_kernel(const long long* __restrict__ codes, int n_leaves, int* __restrict__ child,
                                     int* __restrict__ parent) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_leaves - 1) return;
    const int d = common_prefix(codes, n_leaves, i, i + 1) > common_prefix(codes, n_leaves, i, i - 1) ? 1 : -1;
    const int dmin = common_prefix(codes, n_leaves, i, i - d);
    int lmax = 2;
    while (common_prefix(codes, n_leaves, i, i + lmax * d) > dmin) lmax *= 2;
    int l = 0;
    for (int t = lmax / 2; t >= 1; t /= 2)
        if (common_prefix(codes, n_leaves, i, i + (l + t) * d) > dmin) l += t;
    const int j = i + l * d;
    const int dnode = common_prefix(codes, n_leaves, i, j);
    int s = 0;
    for (int div = 2;; div *= 2) {
        const int t = (l + div - 1) / div;
        if (common_prefix(codes, n_leaves, i, i + (s + t) * d) > dnode) s += t;
        if (t <= 1) break;
    }
    const int gamma = i + s * d + min(d, 0);
    const int leaf0 = n_leaves - 1;
    const int left = min(i, j) == gamma ? leaf0 + gamma : gamma;
    const int right = max(i, j) == gamma + 1 ? leaf0 + gamma + 1 : gamma + 1;
    child[2 * i] = left;
    child[2 * i + 1] = right;
    parent[left] = i;
    parent[right] = i;
}

__global__ void bvh_boxes_kernel(const float* __restrict__ tris, int n_leaves, const int* __restrict__ child,
                                 const int* __restrict__ parent, int* visits, float* box) {
    const int leaf = blockIdx.x * blockDim.x + threadIdx.x;
    if (leaf >= n_leaves) return;
    int node = n_leaves - 1 + leaf;
    const float* t = tris + 9 * static_cast<int64_t>(leaf);
    float* nb = box + 6 * static_cast<int64_t>(node);
    for (int k = 0; k < 3; ++k) {
        nb[k] = fminf(fminf(t[k], t[3 + k]), t[6 + k]);
        nb[3 + k] = fmaxf(fmaxf(t[k], t[3 + k]), t[6 + k]);
    }
    while (node != 0) {
        __threadfence();
        node = parent[node];
        if (atomicAdd(&visits[node], 1) == 0) return;  // first to arrive: the sibling's thread continues
        // the sibling's box was written by another thread: bypass L1
        const float* a = box + 6 * static_cast<int64_t>(child[2 * node]);
        const float* b = box + 6 * static_cast<int64_t>(child[2 * node + 1]);
        nb = box + 6 * static_cast<int64_t>(node);
        for (int k = 0; k < 3; ++k) {
            nb[k] = fminf(__ldcg(a + k), __ldcg(b + k));
            nb[3 + k] = fmaxf(__ldcg(a + 3 + k), __ldcg(b + 3 + k));
        }
    }
}

__device__ __forceinline__ float box_dist2(const float* b, float px, float py, float pz) {
    const float dx = fmaxf(fmaxf(b[0] - px, px - b[3]), 0.0f);
    const float dy = fmaxf(fmaxf(b[1] - py, py - b[4]), 0.0f);
    const float dz = fmaxf(fmaxf(b[2] - pz, pz - b[5]), 0.0f);
    return dx * dx + dy * dy + dz * dz;
}

__device__ __forceinline__ double3 sub3(double3 a, double3 b) { return make_double3(a.x - b.x, a.y - b.y, a.z - b.z); }

__device__ __forceinline__ double dot3(double3 a, double3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

__device__ __forceinline__ double3 cross3(double3 a, double3 b) {
    return make_double3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

// Closest point on a thin or zero-area triangle: nearest edge point, or the plane projection
// when it falls inside. Double, since a thin triangle's plane is badly conditioned in float32.
__device__ void closest_on_thin_triangle(const float* t, float px, float py, float pz, float& cx, float& cy,
                                         float& cz) {
    const double3 v[3] = {make_double3(t[0], t[1], t[2]), make_double3(t[3], t[4], t[5]),
                          make_double3(t[6], t[7], t[8])};
    const double3 p = make_double3(px, py, pz);
    double3 out = v[0];
    double best = INFINITY;
    for (int k = 0; k < 3; ++k) {
        const double3 a = v[k], ab = sub3(v[(k + 1) % 3], a);
        const double len2 = dot3(ab, ab);
        const double s = len2 > 0.0 ? fmin(fmax(dot3(sub3(p, a), ab) / len2, 0.0), 1.0) : 0.0;
        const double3 c = make_double3(a.x + s * ab.x, a.y + s * ab.y, a.z + s * ab.z);
        const double3 d = sub3(p, c);
        if (dot3(d, d) < best) {
            best = dot3(d, d);
            out = c;
        }
    }
    const double3 n = cross3(sub3(v[1], v[0]), sub3(v[2], v[0]));
    const double n2 = dot3(n, n);
    bool inside = n2 > 0.0;
    const double h = inside ? dot3(sub3(p, v[0]), n) / n2 : 0.0;
    const double3 q = make_double3(p.x - h * n.x, p.y - h * n.y, p.z - h * n.z);
    for (int k = 0; k < 3 && inside; ++k)
        inside = dot3(cross3(sub3(v[(k + 1) % 3], v[k]), sub3(q, v[k])), n) >= 0.0;
    if (inside && h * h * n2 < best) out = q;
    cx = static_cast<float>(out.x);
    cy = static_cast<float>(out.y);
    cz = static_cast<float>(out.z);
}

// Closest point on triangle (a, b, c) to p (Ericson, Real-Time Collision Detection 5.1.5).
__device__ __forceinline__ void closest_on_triangle(const float* t, float px, float py, float pz, float& cx, float& cy,
                                                    float& cz) {
    const float ax = t[0], ay = t[1], az = t[2];
    const float abx = t[3] - ax, aby = t[4] - ay, abz = t[5] - az;
    const float acx = t[6] - ax, acy = t[7] - ay, acz = t[8] - az;
    // the region tests cancel in float32 once a triangle is ~100x longer than wide
    const float nx = aby * acz - abz * acy, ny = abz * acx - abx * acz, nz = abx * acy - aby * acx;
    const float bcx = t[6] - t[3], bcy = t[7] - t[4], bcz = t[8] - t[5];
    const float e2 = fmaxf(fmaxf(abx * abx + aby * aby + abz * abz, acx * acx + acy * acy + acz * acz),
                           bcx * bcx + bcy * bcy + bcz * bcz);
    if (nx * nx + ny * ny + nz * nz <= 1e-4f * e2 * e2) {
        closest_on_thin_triangle(t, px, py, pz, cx, cy, cz);
        return;
    }
    const float apx = px - ax, apy = py - ay, apz = pz - az;
    const float d1 = abx * apx + aby * apy + abz * apz;
    const float d2 = acx * apx + acy * apy + acz * apz;
    if (d1 <= 0.0f && d2 <= 0.0f) { cx = ax; cy = ay; cz = az; return; }
    const float bpx = px - t[3], bpy = py - t[4], bpz = pz - t[5];
    const float d3 = abx * bpx + aby * bpy + abz * bpz;
    const float d4 = acx * bpx + acy * bpy + acz * bpz;
    if (d3 >= 0.0f && d4 <= d3) { cx = t[3]; cy = t[4]; cz = t[5]; return; }
    const float cpx = px - t[6], cpy = py - t[7], cpz = pz - t[8];
    const float d5 = abx * cpx + aby * cpy + abz * cpz;
    const float d6 = acx * cpx + acy * cpy + acz * cpz;
    if (d6 >= 0.0f && d5 <= d6) { cx = t[6]; cy = t[7]; cz = t[8]; return; }
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        const float v = d1 / (d1 - d3);
        cx = ax + v * abx; cy = ay + v * aby; cz = az + v * abz;
        return;
    }
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        const float w = d2 / (d2 - d6);
        cx = ax + w * acx; cy = ay + w * acy; cz = az + w * acz;
        return;
    }
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && d4 - d3 >= 0.0f && d5 - d6 >= 0.0f) {
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        cx = t[3] + w * (t[6] - t[3]); cy = t[4] + w * (t[7] - t[4]); cz = t[5] + w * (t[8] - t[5]);
        return;
    }
    const float denom = 1.0f / (va + vb + vc);
    const float v = vb * denom, w = vc * denom;
    cx = ax + abx * v + acx * w; cy = ay + aby * v + acy * w; cz = az + abz * v + acz * w;
}

__global__ void closest_point_kernel(const float* __restrict__ points, int n_points, const float* __restrict__ tris,
                                     const int* __restrict__ tri_index, int n_leaves, const int* __restrict__ child,
                                     const float* __restrict__ box, float max_dist, float* dist, float* closest,
                                     long long* face) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= n_points) return;
    const int64_t q3 = 3 * static_cast<int64_t>(q);
    const float px = points[q3], py = points[q3 + 1], pz = points[q3 + 2];
    const int leaf0 = n_leaves - 1;
    float best = max_dist * max_dist, bx = px, by = py, bz = pz;
    int best_leaf = -1;
    int stack[kMaxStack];
    int sp = 0;
    stack[sp++] = 0;
    while (sp > 0) {
        const int node = stack[--sp];
        if (box_dist2(box + 6 * static_cast<int64_t>(node), px, py, pz) >= best) continue;
        if (node >= leaf0) {
            float cx, cy, cz;
            closest_on_triangle(tris + 9 * static_cast<int64_t>(node - leaf0), px, py, pz, cx, cy, cz);
            const float d2 = (px - cx) * (px - cx) + (py - cy) * (py - cy) + (pz - cz) * (pz - cz);
            if (d2 < best) {
                best = d2;
                bx = cx; by = cy; bz = cz;
                best_leaf = node - leaf0;
            }
            continue;
        }
        const int c0 = child[2 * node], c1 = child[2 * node + 1];
        const float d0 = box_dist2(box + 6 * static_cast<int64_t>(c0), px, py, pz);
        const float d1 = box_dist2(box + 6 * static_cast<int64_t>(c1), px, py, pz);
        const int near = d0 <= d1 ? c0 : c1, far = d0 <= d1 ? c1 : c0;
        if (fmaxf(d0, d1) < best) stack[sp++] = far;  // pushed first so near pops first
        if (fminf(d0, d1) < best) stack[sp++] = near;
    }
    dist[q] = best_leaf < 0 ? max_dist : sqrtf(best);
    closest[q3] = bx;
    closest[q3 + 1] = by;
    closest[q3 + 2] = bz;
    face[q] = best_leaf < 0 ? -1 : tri_index[best_leaf];
}

// Narrow [t_in, t_out] to one axis's slab [lo, hi], given the reciprocal direction. NaN means a
// zero direction with the origin on a face plane: the ray then lies in the closed slab.
__device__ __forceinline__ void clip_slab(float lo, float hi, float o, float inv, float& t_in, float& t_out) {
    const float a = (lo - o) * inv, b = (hi - o) * inv;
    if (a == a && b == b) {
        t_in = fmaxf(t_in, fminf(a, b));
        t_out = fminf(t_out, fmaxf(a, b));
    }
}

// Where the ray enters box b within (tmin, tmax), else infinity. t_out gets 8e-7 relative slack
// (over Ize 2013's 2 gamma(3)) so rounding never drops a hit on a box face.
__device__ __forceinline__ float ray_box_enter(const float* b, float ox, float oy, float oz, float ix, float iy, float iz,
                                               float tmin, float tmax) {
    float t_in = tmin, t_out = tmax;
    clip_slab(b[0], b[3], ox, ix, t_in, t_out);
    clip_slab(b[1], b[4], oy, iy, t_in, t_out);
    clip_slab(b[2], b[5], oz, iz, t_in, t_out);
    return t_in <= t_out + 8e-7f * fabsf(t_out) ? t_in : INFINITY;
}

// The ray in the frame of the watertight test (Woop, Benthin, Wald 2013): axes permuted so the
// direction's largest component is z, keeping handedness, then sheared onto +z.
struct ShearedRay {
    int kx, ky, kz;
    float ox, oy, oz, sx, sy, sz;
};

__device__ __forceinline__ float axis(float x, float y, float z, int k) { return k == 0 ? x : (k == 1 ? y : z); }

__device__ __forceinline__ ShearedRay shear_ray(float ox, float oy, float oz, float dx, float dy, float dz) {
    const float ax = fabsf(dx), ay = fabsf(dy), az = fabsf(dz);
    const int kz = ax > ay ? (ax > az ? 0 : 2) : (ay > az ? 1 : 2);
    const float dk = axis(dx, dy, dz, kz);
    const int kx = (kz + (dk < 0.0f ? 2 : 1)) % 3, ky = (kz + (dk < 0.0f ? 1 : 2)) % 3;
    return {kx, ky, kz, axis(ox, oy, oz, kx), axis(ox, oy, oz, ky), axis(ox, oy, oz, kz),
            axis(dx, dy, dz, kx) / dk, axis(dx, dy, dz, ky) / dk, 1.0f / dk};
}

// Ray parameter of the hit on triangle t in (tmin, tmax), else infinity. Neighbours compute a
// shared edge's function from the same products, negated (no FMA; exact zeros redone in double),
// so no ray slips between them. Edges and corners count as inside; det > 0 when the winding
// normal faces the ray.
__device__ __forceinline__ float ray_triangle(const float* t, const ShearedRay& r, float tmin, float tmax,
                                              bool cull_backface) {
    const float az = t[r.kz] - r.oz, bz = t[3 + r.kz] - r.oz, cz = t[6 + r.kz] - r.oz;
    const float ax = __fmaf_rn(-r.sx, az, t[r.kx] - r.ox), ay = __fmaf_rn(-r.sy, az, t[r.ky] - r.oy);
    const float bx = __fmaf_rn(-r.sx, bz, t[3 + r.kx] - r.ox), by = __fmaf_rn(-r.sy, bz, t[3 + r.ky] - r.oy);
    const float cx = __fmaf_rn(-r.sx, cz, t[6 + r.kx] - r.ox), cy = __fmaf_rn(-r.sy, cz, t[6 + r.ky] - r.oy);
    float u = __fsub_rn(__fmul_rn(cx, by), __fmul_rn(cy, bx));
    float v = __fsub_rn(__fmul_rn(ax, cy), __fmul_rn(ay, cx));
    float w = __fsub_rn(__fmul_rn(bx, ay), __fmul_rn(by, ax));
    if (u == 0.0f || v == 0.0f || w == 0.0f) {
        u = static_cast<float>(static_cast<double>(cx) * by - static_cast<double>(cy) * bx);
        v = static_cast<float>(static_cast<double>(ax) * cy - static_cast<double>(ay) * cx);
        w = static_cast<float>(static_cast<double>(bx) * ay - static_cast<double>(by) * ax);
    }
    if ((u < 0.0f || v < 0.0f || w < 0.0f) && (u > 0.0f || v > 0.0f || w > 0.0f)) return INFINITY;
    const float det = u + v + w;
    if (det == 0.0f || (cull_backface && det < 0.0f)) return INFINITY;
    const float hit = (u * az + v * bz + w * cz) * r.sz / det;
    return hit > tmin && hit < tmax ? hit : INFINITY;
}

__global__ void closest_hit_kernel(const float* __restrict__ origins, const float* __restrict__ directions, int n_rays,
                                   const float* __restrict__ tris, const int* __restrict__ tri_index, int n_leaves,
                                   const int* __restrict__ child, const float* __restrict__ box, float tmin, float tmax,
                                   bool cull_backface, float* t_out, long long* face) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rays) return;
    const int64_t r3 = 3 * static_cast<int64_t>(r);
    const float ox = origins[r3], oy = origins[r3 + 1], oz = origins[r3 + 2];
    const float dx = directions[r3], dy = directions[r3 + 1], dz = directions[r3 + 2];
    const float ix = 1.0f / dx, iy = 1.0f / dy, iz = 1.0f / dz;
    const ShearedRay sheared = shear_ray(ox, oy, oz, dx, dy, dz);
    const int leaf0 = n_leaves - 1;
    float best = tmax;
    int best_leaf = -1;
    int stack[kMaxStack];
    int sp = 0;
    if (ray_box_enter(box, ox, oy, oz, ix, iy, iz, tmin, best) < INFINITY) stack[sp++] = 0;  // misses leave after one box
    while (sp > 0) {
        const int node = stack[--sp];
        if (node >= leaf0) {
            const float hit = ray_triangle(tris + 9 * static_cast<int64_t>(node - leaf0), sheared, tmin, best,
                                           cull_backface);
            if (hit < best) {
                best = hit;
                best_leaf = node - leaf0;
            }
            continue;
        }
        const int c0 = child[2 * node], c1 = child[2 * node + 1];
        const float t0 = ray_box_enter(box + 6 * static_cast<int64_t>(c0), ox, oy, oz, ix, iy, iz, tmin, best);
        const float t1 = ray_box_enter(box + 6 * static_cast<int64_t>(c1), ox, oy, oz, ix, iy, iz, tmin, best);
        const int near = t0 <= t1 ? c0 : c1, far = t0 <= t1 ? c1 : c0;
        if (fmaxf(t0, t1) < best) stack[sp++] = far;  // pushed first so near pops first
        if (fminf(t0, t1) < best) stack[sp++] = near;
    }
    t_out[r] = best;
    face[r] = best_leaf < 0 ? -1 : tri_index[best_leaf];
}

inline unsigned int blocks_for(int n) { return static_cast<unsigned int>((n + kThreads - 1) / kThreads); }


}  // namespace mesh_bvh
}  // namespace comfy

using namespace comfy::mesh_bvh;

extern "C" {

void launch_mesh_bvh_build(const void* codes, const void* tris, int n_leaves, void* child, void* parent, void* visits,
                           void* box, cudaStream_t stream) {
    if (n_leaves <= 0) return;
    if (n_leaves > 1) {
        bvh_hierarchy_kernel<<<blocks_for(n_leaves - 1), kThreads, 0, stream>>>(
            static_cast<const long long*>(codes), n_leaves, static_cast<int*>(child), static_cast<int*>(parent));
        CUDA_CHECK(cudaGetLastError());
    }
    bvh_boxes_kernel<<<blocks_for(n_leaves), kThreads, 0, stream>>>(
        static_cast<const float*>(tris), n_leaves, static_cast<const int*>(child), static_cast<const int*>(parent),
        static_cast<int*>(visits), static_cast<float*>(box));
    CUDA_CHECK(cudaGetLastError());
}

void launch_closest_point_on_mesh(const void* points, int n_points, const void* tris, const void* tri_index,
                                  int n_leaves, const void* child, const void* box, float max_dist, void* dist,
                                  void* closest, void* face, cudaStream_t stream) {
    if (n_points <= 0) return;
    closest_point_kernel<<<blocks_for(n_points), kThreads, 0, stream>>>(
        static_cast<const float*>(points), n_points, static_cast<const float*>(tris),
        static_cast<const int*>(tri_index), n_leaves, static_cast<const int*>(child), static_cast<const float*>(box),
        max_dist, static_cast<float*>(dist), static_cast<float*>(closest), static_cast<long long*>(face));
    CUDA_CHECK(cudaGetLastError());
}

void launch_closest_hit_on_mesh(const void* origins, const void* directions, int n_rays, const void* tris,
                                const void* tri_index, int n_leaves, const void* child, const void* box, float tmin,
                                float tmax, bool cull_backface, void* t, void* face, cudaStream_t stream) {
    if (n_rays <= 0) return;
    closest_hit_kernel<<<blocks_for(n_rays), kThreads, 0, stream>>>(
        static_cast<const float*>(origins), static_cast<const float*>(directions), n_rays,
        static_cast<const float*>(tris), static_cast<const int*>(tri_index), n_leaves, static_cast<const int*>(child),
        static_cast<const float*>(box), tmin, tmax, cull_backface, static_cast<float*>(t), static_cast<long long*>(face));
    CUDA_CHECK(cudaGetLastError());
}

}  // extern "C"
