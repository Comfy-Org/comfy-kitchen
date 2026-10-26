// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// memmove for device memory: the source and destination ranges may overlap.
//
// Blocks take tickets in the order they start and walk 64 KiB chunks towards
// the overlap: from the low end when moving down, from the high end when
// moving up. A block reads its chunk into registers, flags the chunk as read,
// then waits for the flags of the source chunks its write lands on. Those
// chunks hold earlier tickets, so their blocks are already running and the
// wait cannot deadlock, whatever the grid size.
#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int kThreads = 256;
constexpr int kVecs = 16;
constexpr int64_t kChunk = int64_t(kThreads) * kVecs * 16;

// flags[0:nchunks] are the per-chunk read flags, flags[nchunks] the ticket counter.
__global__ void memmove_kernel(
    uint8_t* dst, const uint8_t* src, int64_t nbytes, int64_t nchunks,
    int window, bool backward, int* flags)
{
    __shared__ int ticket;
    if (threadIdx.x == 0)
        ticket = atomicAdd(flags + nchunks, 1);
    __syncthreads();
    const int64_t chunk = backward ? nchunks - 1 - ticket : ticket;
    const int64_t base = chunk * kChunk;

    uint4 v[kVecs];
#pragma unroll
    for (int i = 0; i < kVecs; ++i) {
        const int64_t off = base + (int64_t(i) * kThreads + threadIdx.x) * 16;
        if (off < nbytes)
            v[i] = *reinterpret_cast<const uint4*>(src + off);
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0)
        atomicExch(flags + chunk, 1);

    for (int i = threadIdx.x; i < window; i += kThreads) {
        const int64_t other = backward ? chunk + 1 + i : chunk - 1 - i;
        if (other < 0 || other >= nchunks)
            break;
        while (*reinterpret_cast<volatile int*>(flags + other) == 0) {}
    }
    __syncthreads();
    __threadfence();

#pragma unroll
    for (int i = 0; i < kVecs; ++i) {
        const int64_t off = base + (int64_t(i) * kThreads + threadIdx.x) * 16;
        if (off < nbytes)
            *reinterpret_cast<uint4*>(dst + off) = v[i];
    }
}

}  // namespace

// flags: zeroed int32 scratch of memmove_flag_count(nbytes) entries.
extern "C" int64_t memmove_flag_count(int64_t nbytes) {
    return (nbytes + kChunk - 1) / kChunk + 1;
}

extern "C" void launch_memmove(void* dst, const void* src, int64_t nbytes, void* flags, cudaStream_t stream) {
    auto* d = static_cast<uint8_t*>(dst);
    const auto* s = static_cast<const uint8_t*>(src);
    const bool backward = d > s;
    const int64_t shift = backward ? d - s : s - d;
    if (nbytes == 0 || shift == 0)
        return;
    const int64_t nchunks = (nbytes + kChunk - 1) / kChunk;
    const int window = shift < nbytes ? static_cast<int>((shift + kChunk - 1) / kChunk) : 0;
    memmove_kernel<<<nchunks, kThreads, 0, stream>>>(
        d, s, nbytes, nchunks, window, backward, static_cast<int*>(flags));
}
