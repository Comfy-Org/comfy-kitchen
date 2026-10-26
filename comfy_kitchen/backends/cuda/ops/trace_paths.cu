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

// Reads shortest paths back from distance fields, one warp per row: from ends[r], step to the in-arc tail whose
// distance + weight matches, preferring the smallest distance and then the largest id (Dijkstra's predecessor).
// Tails beyond reach[r] and stop nodes other than starts[r] are skipped; a walk with no predecessor fails.
// Distances are dense rows (R, N) or sorted keys r * N + v with values (+inf when absent). A first pass counts
// path lengths, a second writes the nodes. Out-of-range indices set *error.

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

#include "utils.cuh"

namespace comfy {
namespace trace_paths {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;

struct Fields {
    int R, N, E;
    long long n_keys;
    const int* indptr;
    const int* tail;
    const double* w;
    const long long* keys;  // sorted, null for dense rows
    const double* values;
    const long long* starts;
    const long long* ends;
    const double* reach;    // may be null
    const uint8_t* stop;    // may be null
    int* error;
};

__device__ __forceinline__ double dist_at(const Fields& f, int r, int v) {
    const long long key = static_cast<long long>(r) * f.N + v;
    if (!f.keys) return f.values[key];
    long long lo = 0, hi = f.n_keys;
    while (lo < hi) {
        const long long mid = (lo + hi) >> 1;
        if (f.keys[mid] < key)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo < f.n_keys && f.keys[lo] == key ? f.values[lo] : INFINITY;
}

// counts the path nodes into lengths, or with out, writes them to out[offsets[r]:]
__global__ void walk_kernel(Fields f, int* lengths, const long long* offsets, long long* out) {
    const int r = blockIdx.x * kWarps + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (r >= f.R || (out && lengths[r] == 0)) return;
    const long long start = f.starts[r], end = f.ends[r];
    if (start < 0 || start >= f.N || end < -1 || end >= f.N) {
        if (lane == 0) *f.error = 1;
        return;
    }
    const long long last = out ? offsets[r] + lengths[r] - 1 : 0;
    int cur = static_cast<int>(end), steps = 0;               // -1 once the walk fails
    while (cur >= 0 && cur != start) {
        if (out && lane == 0) out[last - steps] = cur;
        const double dv = dist_at(f, r, cur);
        if (++steps >= f.N || !(dv < INFINITY)) {
            cur = -1;
            break;
        }
        const int a0 = f.indptr[cur], a1 = f.indptr[cur + 1];
        if (a0 < 0 || a1 < a0 || a1 > f.E) {
            if (lane == 0) *f.error = 1;
            cur = -1;
            break;
        }
        double best = INFINITY;
        int pick = -1;
        for (int a = a0 + lane; a < a1; a += 32) {
            const int u = f.tail[a];
            if (u < 0 || u >= f.N) {
                *f.error = 1;
                continue;
            }
            const double du = dist_at(f, r, u);
            if (!(du + f.w[a] == dv) || (f.reach && du > f.reach[r]) || (f.stop && f.stop[u] && u != start)) continue;
            if (du < best || (du == best && u > pick)) {
                best = du;
                pick = u;
            }
        }
        for (int s = 16; s > 0; s >>= 1) {
            const double ob = __shfl_xor_sync(0xffffffffu, best, s);
            const int op = __shfl_xor_sync(0xffffffffu, pick, s);
            if (ob < best || (ob == best && op > pick)) {
                best = ob;
                pick = op;
            }
        }
        cur = pick;
    }
    if (lane != 0) return;
    if (out)
        out[offsets[r]] = start;
    else
        lengths[r] = cur >= 0 ? steps + 1 : 0;
}

}  // namespace trace_paths
}  // namespace comfy

using namespace comfy::trace_paths;

extern "C" {

// out null: fills lengths; otherwise writes the paths into out at offsets (R,).
void launch_trace_paths(int R, int N, int E, long long n_keys, const void* indptr, const void* tail, const void* w,
                        const void* keys, const void* values, const void* starts, const void* ends, const void* reach,
                        const void* stop, void* error, void* lengths, const void* offsets, void* out,
                        cudaStream_t stream) {
    if (R <= 0) return;
    const Fields f{R, N, E, n_keys, static_cast<const int*>(indptr), static_cast<const int*>(tail),
                   static_cast<const double*>(w), static_cast<const long long*>(keys), static_cast<const double*>(values),
                   static_cast<const long long*>(starts), static_cast<const long long*>(ends),
                   static_cast<const double*>(reach), static_cast<const uint8_t*>(stop), static_cast<int*>(error)};
    walk_kernel<<<(R + kWarps - 1) / kWarps, kThreads, 0, stream>>>(f, static_cast<int*>(lengths),
                                                                    static_cast<const long long*>(offsets),
                                                                    static_cast<long long*>(out));
    CUDA_CHECK(cudaGetLastError());
}

}  // extern "C"
