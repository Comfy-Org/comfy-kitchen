// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Shared INT4 / INT6 -> INT8 weight decode for the W4A8 and W6A8 paths. The
// chunked decode writes the result to a workspace the WMMA GEMM reads back; the
// decode GEMV keeps it in registers. Both must land on the same int8 grid, so
// the rounding and the level table live here rather than in either kernel.
//
// Storage, bits = 4 or 6, implied by the packed row width K * bits / 8:
//   bytes [0, K/2):     nibble plane, even column in the low nibble (both widths)
//   bytes [K/2, 3K/4):  6-bit only, the top two bits: column c -> byte c/4, bit 2*(c%4)
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

#include "fp8_utils.h"

namespace comfy::hip_backend {

// The group scale is stored as fp32 or as raw e4m3 bytes.
template <typename ScaleT>
__forceinline__ __device__ float load_group_scale(ScaleT v);

template <>
__forceinline__ __device__ float load_group_scale<float>(float v) {
    return v;
}

template <>
__forceinline__ __device__ float load_group_scale<uint8_t>(uint8_t v) {
    return fp8_to_float(v);
}

__forceinline__ __device__ int8_t dequant_round(float v) {
    // rintf is round-half-to-even, matching torch.round.
    int q = static_cast<int>(rintf(v));
    q = q < -127 ? -127 : (q > 127 ? 127 : q);
    return static_cast<int8_t>(q);
}

// 8 packed bytes of the nibble plane (low nibble = even column) -> 16 int8 packed
// as four dwords, column 0 in the low byte of x.
// BITS 4: cb is the 16-entry level table or null for the uniform (code - 8)
// levels. The 16 columns span one group when group_size >= 16, two at 8 and four
// at 4, so the caller hands in up to four scales; all four are equal in the
// common case.
// BITS 6: hi is the dword of the high plane for these 16 columns, column i's top
// two bits at bit 2*i. Levels are uniform (code - 32) with no codebook, and 6-bit
// groups are multiples of 16, so sc0 covers the whole vector.
template <int BITS>
__forceinline__ __device__ uint4 dequant16_to_int8(
    uint2 packed, unsigned hi, const float* __restrict__ cb, float sc0, float sc1, float sc2,
    float sc3, int group_size) {

    static_assert(BITS == 4 || BITS == 6, "W4A8 storage is 4 or 6 bits");
    const unsigned words[2] = {packed.x, packed.y};
    unsigned d[4] = {0u, 0u, 0u, 0u};
#pragma unroll
    for (int w = 0; w < 2; ++w) {
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const int pair = w * 4 + b;
            const unsigned byte = (words[w] >> (b * 8)) & 0xFFu;
            float v0, v1, s;
            if constexpr (BITS == 6) {
                const unsigned c0 = (byte & 0xFu) | (((hi >> (4 * pair)) & 3u) << 4);
                const unsigned c1 = ((byte >> 4) & 0xFu) | (((hi >> (4 * pair + 2)) & 3u) << 4);
                v0 = static_cast<float>(c0) - 32.0f;
                v1 = static_cast<float>(c1) - 32.0f;
                s = sc0;
            } else {
                const int local = (group_size >= 16) ? 0 : ((pair * 2) / group_size);
                s = (local == 0) ? sc0 : (local == 1 ? sc1 : (local == 2 ? sc2 : sc3));
                const unsigned lo = byte & 0xFu;
                const unsigned hn = (byte >> 4) & 0xFu;
                v0 = cb ? cb[lo] : (static_cast<float>(lo) - 8.0f);
                v1 = cb ? cb[hn] : (static_cast<float>(hn) - 8.0f);
            }
            const unsigned q0 = static_cast<unsigned>(dequant_round(v0 * s)) & 0xFFu;
            const unsigned q1 = static_cast<unsigned>(dequant_round(v1 * s)) & 0xFFu;
            d[pair >> 1] |= (q0 << ((pair & 1) * 16)) | (q1 << ((pair & 1) * 16 + 8));
        }
    }
    return make_uint4(d[0], d[1], d[2], d[3]);
}

}  // namespace comfy::hip_backend
