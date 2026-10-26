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

// Follows succ from each start until a stop node, one thread per walk. A walk fails at a node with no successor,
// an inactive node, or one already visited (visiting a node also visits its active twin). A first pass counts path
// lengths using a visited bitmap per row, a second replays the successful walks to write their nodes.

#include <cuda_runtime.h>
#include <cstdint>

#include "utils.cuh"

namespace comfy {
namespace walk_successors {

constexpr int kThreads = 128;

__device__ __forceinline__ void visit(unsigned int* row, int v) { row[v >> 5] |= 1u << (v & 31); }

// out null: fills lengths (0 for a failed walk) using seen (R, N / 32), zeroed; otherwise writes the walks to out (R, L)
__global__ void walk_kernel(int R, int N, const int* succ, const uint8_t* stop, const uint8_t* active, const int* twin,
                            const long long* starts, unsigned int* seen, int* lengths, long long* out, int L) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= R) return;
    int cur = static_cast<int>(starts[r]);
    if (out) {
        for (int i = 0; i < lengths[r]; ++i, cur = succ[cur]) out[static_cast<long long>(r) * L + i] = cur;
        return;
    }
    unsigned int* row = seen + r * static_cast<long long>((N + 31) / 32);
    int len = 1;
    while (true) {
        visit(row, cur);
        const int t = twin[cur];
        if (t >= 0 && active[t]) visit(row, t);
        cur = succ[cur];
        if (cur < 0 || !active[cur] || (row[cur >> 5] >> (cur & 31) & 1u)) {
            len = 0;
            break;
        }
        ++len;
        if (stop[cur]) break;
    }
    lengths[r] = len;
}

}  // namespace walk_successors
}  // namespace comfy

using namespace comfy::walk_successors;

extern "C" {

void launch_walk_successors(int R, int N, const void* succ, const void* stop, const void* active, const void* twin,
                            const void* starts, void* seen, void* lengths, void* out, int L, cudaStream_t stream) {
    if (R <= 0) return;
    walk_kernel<<<(R + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        R, N, static_cast<const int*>(succ), static_cast<const uint8_t*>(stop), static_cast<const uint8_t*>(active),
        static_cast<const int*>(twin), static_cast<const long long*>(starts), static_cast<unsigned int*>(seen),
        static_cast<int*>(lengths), static_cast<long long*>(out), L);
    CUDA_CHECK(cudaGetLastError());
}

}  // extern "C"
