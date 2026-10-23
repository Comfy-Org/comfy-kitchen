// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Pieces the tiled GEMM kernels share: the byte-addressed LDS tile stager, the A
// operand's pointer type, and the device's workgroup-processor count.
#pragma once

#include <atomic>

#include "mma.h"

namespace comfy::hip_backend {

// LDS row padding, in bytes. 8 spreads the 16 lanes of a fragment read across
// distinct banks. 16 would preserve 128-bit LDS access but aliases rows
// 0/4/8/... onto the same bank.
constexpr int kLdsPad = 8;

// Staging for one ROWS x BKB byte tile. load() issues only the global reads, so
// the caller can place the tile's math between load() and store().
//
// kbytes is the source row length in bytes (K for 8-bit types, K/2 for int4) and
// is a multiple of 16, so a 16-byte chunk starting inside a row also ends inside
// it. Out-of-range rows and the K tail are zero-filled.
template <int ROWS, int BKB, int THREADS>
struct TileStager {
    static constexpr int kChunksPerRow = BKB / 16;
    static constexpr int kChunks = ROWS * kChunksPerRow;
    static constexpr int kPerThread = kChunks / THREADS;
    static constexpr int kStride = BKB + kLdsPad;
    // Both divisions truncate, and either remainder would leave part of the tile
    // unwritten in LDS for the math to then read as stale.
    static_assert(BKB % 16 == 0, "BKB must be a whole number of 16-byte chunks");
    static_assert(kChunks % THREADS == 0, "THREADS must divide the tile's 16-byte chunks");

    uint4 regs[kPerThread];

    __forceinline__ __device__ void load(const uint8_t* __restrict__ src, int row0, int rows_total,
                                         int kbyte0, int kbytes) {
        const int tid = threadIdx.x;
        #pragma unroll
        for (int i = 0; i < kPerThread; ++i) {
            const int c = tid * kPerThread + i;
            const int grow = row0 + c / kChunksPerRow;
            const int gk = kbyte0 + (c % kChunksPerRow) * 16;

            regs[i] = (grow < rows_total && gk < kbytes)
                          ? *reinterpret_cast<const uint4*>(
                                src + static_cast<int64_t>(grow) * kbytes + gk)
                          : make_uint4(0, 0, 0, 0);
        }
    }

    __forceinline__ __device__ void store(uint8_t* __restrict__ lds) const {
        const int tid = threadIdx.x;
        #pragma unroll
        for (int i = 0; i < kPerThread; ++i) {
            const int c = tid * kPerThread + i;
            uint8_t* dst = lds + (c / kChunksPerRow) * kStride + (c % kChunksPerRow) * 16;
            // 8-byte stores: kLdsPad breaks 16-byte LDS alignment.
            *reinterpret_cast<uint2*>(dst) = make_uint2(regs[i].x, regs[i].y);
            *reinterpret_cast<uint2*>(dst + 8) = make_uint2(regs[i].z, regs[i].w);
        }
    }
};

// A is the raw operand rows, or a source that gathers them (see gemm_simt.h). Only the
// pointer form carries __restrict__.
template <typename T>
struct GemmOperandA {
    using type = T;
};
template <>
struct GemmOperandA<const uint8_t*> {
    using type = const uint8_t* __restrict__;
};

// hipDeviceAttributeMultiprocessorCount reports WGPs on RDNA, not CUs, and a workgroup
// schedules onto a WGP, so WGPs are the unit a grid-coverage test needs. Cached per
// ordinal to keep the query off the launch path; the fallback only mis-sizes that test,
// never a result.
inline int device_wgp_count() {
    constexpr int kMaxDevices = 16;
    // A GEMM can be launched from several host threads at once. Racing threads
    // write the same value and nothing is published through the cache, so relaxed.
    static std::atomic<int> cache[kMaxDevices] = {};
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= kMaxDevices) return 16;
    int n = cache[dev].load(std::memory_order_relaxed);
    if (n == 0) {
        if (hipDeviceGetAttribute(&n, hipDeviceAttributeMultiprocessorCount, dev) != hipSuccess ||
            n <= 0) {
            n = 16;
        }
        cache[dev].store(n, std::memory_order_relaxed);
    }
    return n;
}

}  // namespace comfy::hip_backend
