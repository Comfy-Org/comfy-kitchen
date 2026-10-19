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

// Batched delta-stepping shortest paths on a CSR graph with non-negative double weights, in one cooperative kernel.
// Each row is its own search over slots holding keys r * N + v: dense rows, or an open-addressing hash table.
// Nodes below their row's bucket end go to the near queue, the rest to the far pile; once no row has near work,
// each bucket end moves to the row's smallest far distance + delta. Distances only decrease (atomicMin on the bits
// of non-negative doubles), so every node within the final bound gets its exact distance in any thread order.
//
// Per row: bound (nodes farther are not entered), reach (not expanded), end and stop nodes (entered, not expanded,
// lower the bound), closer (arcs into it lower the bound instead), blocked nodes (entered only as an end or stop).
// -1 disables end / closer. Indices are checked as they are read.

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

#include "utils.cuh"

namespace cg = cooperative_groups;

namespace comfy {
namespace shortest_paths {

constexpr int kThreads = 256;
constexpr int kTile = 16;
constexpr int kProbes = 64;  // a key not placed within this many probes marks the table full

__device__ __forceinline__ double atomic_min_nonneg(double* addr, double v) {
    const unsigned long long old = atomicMin(reinterpret_cast<unsigned long long*>(addr),
                                             static_cast<unsigned long long>(__double_as_longlong(v)));
    return __longlong_as_double(static_cast<long long>(old));
}

// reads first: most values are not a new minimum, and atomics on one address serialize
__device__ __forceinline__ void lower(double* addr, double v) {
    if (v < __ldcg(addr)) atomic_min_nonneg(addr, v);
}

// one atomic per group of divergent threads
__device__ __forceinline__ void append(int* list, int* count, int item) {
    const cg::coalesced_group g = cg::coalesced_threads();
    int at = 0;
    if (g.thread_rank() == 0) at = atomicAdd(count, static_cast<int>(g.size()));
    list[g.shfl(at, 0) + g.thread_rank()] = item;
}

// true when the bit was clear
__device__ __forceinline__ bool claim(unsigned int* bits, int item) {
    const unsigned int bit = 1u << (item & 31);
    return !(atomicOr(bits + (item >> 5), bit) & bit);
}

__device__ __forceinline__ void release(unsigned int* bits, int item) {
    atomicAnd(bits + (item >> 5), ~(1u << (item & 31)));
}

struct Search {
    int R, N, E;
    const int* indptr;
    const int* dst;
    const double* w;
    const uint8_t* active;    // active .. reach may be null
    const uint8_t* blocked;
    const uint8_t* stop;
    const long long* end;
    const long long* closer;
    const double* reach;
    double delta;
    double* bound;
    const long long* front;   // start keys, at their stored distance (dense) or 0 (table)
    int n_front;
    double* dist;             // (cap) distance per slot
    long long* keys;          // (cap) table keys, -1 when free; null for dense rows
    long long cap;            // slots, a power of two for a table
    unsigned int* near_bits;  // (2, words) near queue membership
    unsigned int* far_bits;   // (words) far pile membership
    int* items;               // (2, cap) near queues, alternating per step
    int* far_items;           // (2, cap) far piles, alternating per far phase
    int* counts;              // (7,) 3 near sizes, 2 far sizes, index error, table full
    double* thr;              // (2, R) current and previous bucket end
    double* low;              // (2, R) smallest far distance per row, per far pile

    __device__ int find(long long key) const {
        if (!keys) return static_cast<int>(key);
        const int mask = static_cast<int>(cap - 1);
        int at = static_cast<int>((static_cast<unsigned long long>(key) * 0x9E3779B97F4A7C15ull) >> 32) & mask;
        for (int probe = 0; probe < kProbes; ++probe, at = (at + 1) & mask) {
            const long long k = __ldcg(keys + at);
            if (k == key) return at;
            if (k != -1) continue;
            const long long old = static_cast<long long>(atomicCAS(reinterpret_cast<unsigned long long*>(keys + at),
                                                                   ~0ull, static_cast<unsigned long long>(key)));
            if (old == -1 || old == key) return at;
        }
        counts[6] = 1;
        return -1;
    }
    __device__ __forceinline__ long long key(int slot) const { return keys ? __ldcg(keys + slot) : slot; }
};

struct Step {
    int* items;               // the next near queue
    int* count;
    unsigned int* bits;
    int* far_items;           // the current far pile
    int* far_count;
    double* low;
};

__device__ void visit(const Search& s, int item, int lane, const Step& out) {
    const long long key = s.key(item);
    const int r = static_cast<int>(key / s.N);
    const int v = static_cast<int>(key - static_cast<long long>(r) * s.N);
    const double d = __ldcg(s.dist + item);
    if (d > __ldcg(s.bound + r) || (s.reach && d > s.reach[r])) return;
    const long long e = s.end ? s.end[r] : -1;
    const long long cl = s.closer ? s.closer[r] : -1;
    const int a0 = s.indptr[v], a1 = s.indptr[v + 1];
    if (e < -1 || e >= s.N || cl < -1 || cl >= s.N || a0 < 0 || a1 < a0 || a1 > s.E) {
        s.counts[5] = 1;
        return;
    }
    if (v == e) return;
    const long long base = key - v;
    const double end_of_bucket = __ldcg(s.thr + r);
    for (int a = a0 + lane; a < a1; a += kTile) {
        const int to = s.dst[a];
        if (to < 0 || to >= s.N) {
            s.counts[5] = 1;
            continue;
        }
        const double c = d + s.w[a];
        if (!(c < INFINITY) || (s.active && !s.active[to])) continue;
        if (to == cl) {
            lower(s.bound + r, d);
            continue;
        }
        const bool target = to == e || (s.stop && s.stop[to]);
        if (s.blocked && s.blocked[to] && !target) continue;
        if (c > __ldcg(s.bound + r)) continue;
        const int slot = s.find(base + to);
        if (slot < 0 || !(c < atomic_min_nonneg(s.dist + slot, c))) continue;
        if (target)
            lower(s.bound + r, c);
        else if (s.reach && c > s.reach[r])
            continue;                                         // entered, never expanded
        else if (c < end_of_bucket) {
            if (claim(out.bits, slot)) append(out.items, out.count, slot);
        } else {
            lower(out.low + r, c);
            if (claim(s.far_bits, slot)) append(out.far_items, out.far_count, slot);
        }
    }
}

__global__ void search_kernel(Search s) {
    const cg::grid_group grid = cg::this_grid();
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const long long words = (s.cap + 31) / 32;
    double *thr = s.thr, *prev = s.thr + s.R;
    for (int t = tid; t < s.n_front; t += stride) {          // everything starts in the far pile
        const long long key = s.front[t];
        if (key < 0 || key >= static_cast<long long>(s.R) * s.N) {
            s.counts[5] = 1;
            continue;
        }
        const int slot = s.find(key);
        if (slot < 0) continue;
        if (s.keys) s.dist[slot] = 0.0;
        const double d = __ldcg(s.dist + slot);
        if (!(d < INFINITY)) continue;                        // +inf reaches nothing
        lower(s.low + key / s.N, d);
        if (claim(s.far_bits, slot)) append(s.far_items, s.counts + 3, slot);
    }
    grid.sync();
    for (int it = 0, pile = 0;; ++it) {                       // pile: which far pile is current
        const int count = __ldcg(s.counts + it % 3), far_count = __ldcg(s.counts + 3 + pile);
        if ((count == 0 && far_count == 0) || __ldcg(s.counts + 6)) break;
        if (tid == 0) s.counts[(it + 2) % 3] = 0;            // step it - 1's count is reused for step it + 1
        const Step out{s.items + ((it + 1) & 1) * s.cap, s.counts + (it + 1) % 3, s.near_bits + ((it + 1) & 1) * words,
                       s.far_items + pile * s.cap, s.counts + 3 + pile, s.low + pile * s.R};
        if (count) {
            const int* cur = s.items + (it & 1) * s.cap;
            unsigned int* cur_bits = s.near_bits + (it & 1) * words;
            for (int t = tid / kTile; t < count; t += stride / kTile) {
                const int item = __ldcg(cur + t);
                if (tid % kTile == 0) release(cur_bits, item);
                visit(s, item, tid % kTile, out);
            }
            grid.sync();
            continue;
        }
        // far phase: move the bucket ends, then split the pile into the near queue and the other pile
        double* next_low = s.low + (pile ^ 1) * s.R;
        for (int r = tid; r < s.R; r += stride) {
            prev[r] = thr[r];
            thr[r] = __ldcg(out.low + r) + s.delta;
            next_low[r] = INFINITY;
        }
        if (tid == 0) s.counts[3 + (pile ^ 1)] = 0;
        grid.sync();
        int* kept = s.far_items + (pile ^ 1) * s.cap;
        for (int t = tid; t < far_count; t += stride) {
            const int item = __ldcg(out.far_items + t);
            const int r = static_cast<int>(s.key(item) / s.N);
            const double d = __ldcg(s.dist + item);
            if (d < __ldcg(prev + r) || d > __ldcg(s.bound + r) || (s.reach && d > s.reach[r])) {
                release(s.far_bits, item);
            } else if (d < __ldcg(thr + r)) {
                release(s.far_bits, item);
                if (claim(out.bits, item)) append(out.items, out.count, item);
            } else {
                lower(next_low + r, d);
                append(kept, s.counts + 3 + (pile ^ 1), item);
            }
        }
        grid.sync();
        pile ^= 1;
    }
}

}  // namespace shortest_paths
}  // namespace comfy

using namespace comfy::shortest_paths;

extern "C" {

// dist: R dense rows of N (keys null) or a table of cap slots, a power of two (keys -1, dist +inf on entry).
// bits (3, cap / 32) and counts (7,) zeroed, levels (4, R) = -inf, -inf, +inf, +inf, queues (4, cap) scratch.
// Sets counts[5] on an out-of-range index (that item is skipped), counts[6] when the table is full (search stops).
void launch_shortest_paths(int R, int N, int E, int n_front, long long cap, double delta, const void* indptr,
                           const void* dst, const void* w, const void* active, const void* blocked, const void* stop,
                           const void* end, const void* closer, const void* reach, void* bound, void* dist, void* keys,
                           const void* front, void* bits, void* queues, void* counts, void* levels, cudaStream_t stream) {
    if (R <= 0 || N <= 0 || n_front <= 0) return;
    unsigned int* b = static_cast<unsigned int*>(bits);
    int* q = static_cast<int*>(queues);
    double* l = static_cast<double*>(levels);
    Search s{R, N, E, static_cast<const int*>(indptr), static_cast<const int*>(dst), static_cast<const double*>(w),
             static_cast<const uint8_t*>(active), static_cast<const uint8_t*>(blocked), static_cast<const uint8_t*>(stop),
             static_cast<const long long*>(end), static_cast<const long long*>(closer), static_cast<const double*>(reach),
             delta, static_cast<double*>(bound), static_cast<const long long*>(front), n_front, static_cast<double*>(dist),
             static_cast<long long*>(keys), cap, b, b + 2 * ((cap + 31) / 32), q, q + 2 * cap, static_cast<int*>(counts),
             l, l + 2 * R};
    // one block per SM: rounds are latency bound and grid syncs slow down with more blocks
    comfy::launch_cooperative(search_kernel, comfy::sm_count(), kThreads, s, stream);
}

}  // extern "C"
