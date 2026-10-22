// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// NDHWC -> NDCWH transpose: [B, T, P, C] (C innermost) into [B, C, T, P] (P
// innermost).
//
// The destination's inner dimension is the source's outer one, so a plain copy
// writes one element at a time into a 32-byte sector it cannot share with a
// neighbour. Measured on the 6-WGP 780M, a 268 MB copy in this direction runs
// at 2.3 GB/s against 58.7 GB/s the other way round: a scattered write cannot
// be combined, a scattered read can still fill the sector it pulled.
//
// The tile below stages the block's values in LDS and writes each channel's run
// of pixels at once, so both sides of the copy are coalesced. The read is one
// 16-byte chunk per lane, the write is two bytes per lane, and the tile's
// padding puts the eight channels of a pixel on eight banks.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace comfy::hip_backend {
namespace {

constexpr int kTilePix = 64;
constexpr int kTileChan = 32;
constexpr int kThreads = 256;
// 17 words per channel row: the eight channels of a pixel then land on eight
// different banks, and the row base stays 8-byte aligned.
constexpr int kTileStride = 68;

template <typename T>
__global__ __launch_bounds__(kThreads) void ndhwc_to_nchw_kernel(
    const T* __restrict__ src, T* __restrict__ dst, int64_t pixels, int C, int64_t t_stride) {
    __shared__ T tile[kTileChan][kTileStride];

    const int64_t p0 = static_cast<int64_t>(blockIdx.x) * kTilePix;
    const int64_t c0 = static_cast<int64_t>(blockIdx.y) * kTileChan;
    // The source's frames are (batch, time) pairs; the destination puts the
    // channel outside the time axis, so its channel planes are T * P long and
    // the frames of a batch are P apart.
    const int64_t f = blockIdx.z;
    const int64_t b = f / t_stride;
    const int64_t t = f - b * t_stride;
    const T* s = src + f * pixels * C + c0;
    T* d = dst + (b * C * t_stride + t) * pixels + c0 * t_stride * pixels;

    // The tile's source values are one contiguous run: kTilePix consecutive
    // pixels, kTileChan channels each. Eight channels are one 16-byte read.
    constexpr int kVec = 8;
    constexpr int kGroups = kTileChan / kVec;
    for (int i = threadIdx.x; i < kTilePix * kGroups; i += kThreads) {
        const int p = i / kGroups;
        const int g = i % kGroups;
        const int64_t c = c0 + g * kVec;
        if (c >= C || p0 + p >= pixels) {
            continue;
        }
        const T* q = s + (p0 + p) * C + g * kVec;
        union {
            uint4 v;
            T e[kVec];
        } raw;
        raw.v = *reinterpret_cast<const uint4*>(q);
        for (int k = 0; k < kVec; ++k) {
            // A channel tail shorter than the vector is zeroed, and the write
            // phase drops it again.
            tile[g * kVec + k][p] = (c + k < C) ? raw.e[k] : static_cast<T>(0);
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < kTileChan * kTilePix; i += kThreads) {
        const int c = i / kTilePix;
        const int p = i % kTilePix;
        if (c0 + c >= C || p0 + p >= pixels) {
            continue;
        }
        d[c * t_stride * pixels + p0 + p] = tile[c][p];
    }
}

}  // namespace
}  // namespace comfy::hip_backend

// Transpose [B, T, P, C] with C innermost into [B, C, T, P] with P innermost.
// Both halves are 16-bit types.
extern "C" void launch_ndhwc_to_nchw_transpose(
    const void* src, void* dst, int64_t frames, int64_t pixels, int C, int64_t t_stride,
    int dtype_code, hipStream_t stream) {

    using namespace comfy::hip_backend;
    if (frames <= 0 || pixels <= 0 || C <= 0 || C % 8 != 0 || t_stride <= 0) {
        return;
    }
    const dim3 grid(static_cast<unsigned>((pixels + kTilePix - 1) / kTilePix),
                     static_cast<unsigned>((C + kTileChan - 1) / kTileChan),
                     static_cast<unsigned>(frames));
    if (dtype_code == 1) {
        ndhwc_to_nchw_kernel<__half><<<grid, kThreads, 0, stream>>>(
            static_cast<const __half*>(src), static_cast<__half*>(dst), pixels, C, t_stride);
    } else {
        ndhwc_to_nchw_kernel<__bf16><<<grid, kThreads, 0, stream>>>(
            static_cast<const __bf16*>(src), static_cast<__bf16*>(dst), pixels, C, t_stride);
    }
}
