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

// Batched shortest paths by delta-stepping with a near queue and a far pile on a CSR graph with non-negative double
// arc weights, the whole search in one cooperative kernel. Row r is one search; the rows share the queues of slots
// holding flat keys r * N + v, either dense (slot = key) or in an open-addressing table. A near round expands the near
// queue (kTile lanes per item, one arc each): a lowered node below its row's bucket end joins the next near queue, any
// other the far pile. When no row has near work left, a far phase moves every row's bucket end to its smallest far
// distance + delta and splits the pile: entries below the old bucket end (already expanded), beyond the bound or
// beyond reach are dropped, those below the new end go near, the rest stay. A slot joins each queue and the pile once
// (a bit per slot, cleared when it leaves). Distances only decrease, through atomicMin on their bit patterns
// (non-negative doubles order like unsigned integers), so every node within the final bound ends at its exact
// shortest distance whatever the thread order.
//
// Per row: nodes farther than bound[r] are not entered and those farther than reach[r] are not expanded; the row's
// end[r] and stop nodes (shared by all rows) are entered but not expanded (stop nodes the search starts from are),
// and reaching one lowers bound[r]; an arc into closer[r] is not relaxed and lowers bound[r] to the distance it
// leaves from; blocked nodes are entered only as an end or stop node. -1 disables end / closer. Indices are checked
// where they are used, so repeated searches on one graph need no separate validation pass.

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
constexpr int kProbes = 64;  // a key that finds no slot this close counts the table as full

__device__ __forceinline__ double atomic_min_nonneg(double* addr, double v) {
    const unsigned long long old = atomicMin(reinterpret_cast<unsigned long long*>(addr),
                                             static_cast<unsigned long long>(__double_as_longlong(v)));
    return __longlong_as_double(static_cast<long long>(old));
}

// lowers *addr to v, reading first: most values are not a new minimum, and atomics on one address serialize
__device__ __forceinline__ void lower(double* addr, double v) {
    if (v < __ldcg(addr)) atomic_min_nonneg(addr, v);
}

// appends item to a list from divergent threads with one atomic per group
__device__ __forceinline__ void append(int* list, int* count, int item) {
    const cg::coalesced_group g = cg::coalesced_threads();
    int at = 0;
    if (g.thread_rank() == 0) at = atomicAdd(count, static_cast<int>(g.size()));
    list[g.shfl(at, 0) + g.thread_rank()] = item;
}

// sets item's bit, true when it was clear
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
    const uint8_t* active;    // the options from here to reach may be null
    const uint8_t* blocked;
    const uint8_t* stop;
    const long long* end;
    const long long* closer;
    const double* reach;
    double delta;
    double* bound;
    const long long* front;   // flat keys to start from, at their stored distance (dense) or at 0 (table)
    int n_front;
    double* dist;             // (cap) distance per slot
    long long* keys;          // (cap) the table's keys, -1 for a free slot; null for dense rows
    long long cap;            // slots, a power of two for a table
    unsigned int* near_bits;  // (2, words) membership of the two near queues
    unsigned int* far_bits;   // (words) membership of the far pile
    int* items;               // (2, cap) near queues, step it takes the queue it & 1
    int* far_items;           // (2, cap) far piles
    int* counts;              // (7,) near sizes (step it takes counts[it % 3]), far sizes, index error, table full
    double* thr;              // (2, R) bucket end and the one before it
    double* low;              // (2, R) smallest far distance per row, the far phase f reads low[f & 1]

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
    int* far_items;           // the far pile and the row minima it feeds
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
    for (int it = 0, pile = 0;; ++it) {                       // pile: the far phases done & 1
        const int count = __ldcg(s.counts + it % 3), far_count = __ldcg(s.counts + 3 + pile);
        if ((count == 0 && far_count == 0) || __ldcg(s.counts + 6)) break;
        if (tid == 0) s.counts[(it + 2) % 3] = 0;            // round it - 1's near count becomes round it + 1's
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
        // far phase: new bucket ends, then the pile split into the next near queue and the other pile
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

// dist holds R dense rows of N (keys null) or the values of a table of cap keys (a power of two), keys -1 and dist
// +inf on entry. bits (3, ceil(cap / 32)) and counts (7,) must hold 0, levels (4, R) -inf in its first two rows and
// +inf in the others; queues is (4, cap) scratch. counts[5] is set when an index (front, end, closer, indptr or dst)
// was out of range, the offending items left out, counts[6] when the table ran full, which stops the search.
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
    int device = 0, sms = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    // one block per SM: rounds are latency bound and the grid barrier gets slower with more blocks
    void* args[] = {&s};
    CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(search_kernel), sms, kThreads, args, 0, stream));
    CUDA_CHECK(cudaGetLastError());
}

}  // extern "C"
