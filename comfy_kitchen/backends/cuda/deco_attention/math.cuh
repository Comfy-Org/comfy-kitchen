// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "layout.h"
#include <cute/algorithm/tuple_algorithms.hpp>
#include <cute/container/array_aligned.hpp>
#include <cutlass/arch/memory.h>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/mma_sm80.hpp>

namespace comfy_deco::detail {

using Fragment = cute::array_aligned<uint32_t, 4>;

namespace packed_fp4 {

__device__ __forceinline__ uint32_t global_u32(uintptr_t address, bool valid) {
    uint32_t out;
    asm("{ .reg .pred p; setp.ne.s32 p,%2,0; mov.b32 %0,0; @p ld.global.nc.b32 %0,[%1]; }"
        : "=r"(out)
        : "l"(address), "r"(int(valid)));
    return out;
}

template <bool Transposed = false>
__device__ __forceinline__ Fragment load_matrix(unsigned address) {
    auto const& src = *reinterpret_cast<cute::uint128_t const*>(__cvta_shared_to_generic(address));
    Fragment r;
    if constexpr (Transposed)
        cute::SM75_U16x8_LDSM_T::copy(src, r[0], r[1], r[2], r[3]);
    else
        cute::SM75_U32x4_LDSM_N::copy(src, r[0], r[1], r[2], r[3]);
    return r;
}

// The packed scale selectors are (0,0) for A and (0,ThreadIdB) for B.
// CuTe fixes the B thread selector to zero. Here all four lanes supply scales.
template <int ThreadIdB>
__device__ __forceinline__ void mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1, uint32_t sfa,
                                    uint32_t sfb) {
    asm("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X"
        ".f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},%10,{0,0},%11,{0,%12};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa), "r"(sfb),
          "n"(ThreadIdB));
}

} // namespace packed_fp4

template <class State>
__device__ __forceinline__ void load_query(State& s, const void* q, const void* qsf, int S, int H,
                                           int qt, int bh) {
    const int warp = int(threadIdx.x) >> 5;
    const int lane = int(threadIdx.x) & 31;
    const int g = lane >> 2, t = lane & 3;
    const int m4 = lane >> 3, r8 = lane & 7;
    s.t = t;
    s.ldsm_off = (m4 & 1) * 2048 + ((m4 >> 1) * 8 + r8) * 16;
    s.ksf_lane = layout::key_scale_offset + 16 * (8 * (g >> 1) + 2 * t + (g & 1));
    s.vsf_lane = layout::value_scale_offset + lane * 16;
    s.vm_pb = 2 * (r8 & 3) + (r8 >> 2);
    s.vm_mh = m4 >> 1;
    s.vm_lane = ((m4 & 1) * 8 + s.vm_pb) * 256;

    const int row = qt * 128 + warp * 16 + g;
    const int qr = warp * 16 + g + 8 * (t & 1);
    const int tiles = (S + 127) >> 7;
    const uintptr_t qbase = uintptr_t(q) + (uintptr_t(row) * H + bh) * layout::query_row_bytes + 4 * t;
#pragma unroll
    for (int a = 0; a < 3; ++a) {
        const uintptr_t addr = qbase + 32 * a;
        const uintptr_t row8 = addr + uintptr_t(8 * H) * layout::query_row_bytes;
        s.qa[a][0] = packed_fp4::global_u32(addr, row < S);
        s.qa[a][1] = packed_fp4::global_u32(row8, row + 8 < S);
        s.qa[a][2] = packed_fp4::global_u32(addr + 16, row < S);
        s.qa[a][3] = packed_fp4::global_u32(row8 + 16, row + 8 < S);
        const uintptr_t scale = uintptr_t(qsf) +
                                (uintptr_t(bh * tiles + qt) * 3 + a) * layout::scale_fragment_bytes +
                                4 * (4 * (qr & 31) + (qr >> 5));
        s.qsf[a] = packed_fp4::global_u32(scale, true);
    }
}

template <class State>
__device__ __forceinline__ void preload(State const& s, unsigned tile, Fragment (&early)[4]) {
#pragma unroll
    for (int jp = 0; jp < 4; ++jp)
        early[jp] = packed_fp4::load_matrix(tile + s.ldsm_off + 256 * jp);
}

template <bool Masked, class State>
__device__ __forceinline__ void qk(State const& s, unsigned tile, float (&score)[16][4], int valid_keys,
                                   const Fragment (&early)[4]) {
    using namespace packed_fp4;
    Fragment kw[3];
#pragma unroll
    for (int a = 0; a < 3; ++a)
        cutlass::arch::shared_load<16>(&kw[a], tile + s.ksf_lane + layout::scale_fragment_bytes * a);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
#pragma unroll
        for (int e = 0; e < 4; ++e)
            score[j][e] = 0.f;
    }

    cute::for_each(cute::make_seq<3>{}, [&](auto k) {
        cute::for_each(cute::make_seq<8>{}, [&](auto i) {
            Fragment b;
            if constexpr (k == 0 && i < 4)
                b = early[i];
            else
                b = load_matrix(tile + k * layout::fragment_bytes + i * 256 + s.ldsm_off);
            constexpr int j = 2 * i;
            constexpr int scale = 2 * (j >> 3) + ((j >> 2) & 1);
            mma<j % 4>(score[j], s.qa[k], b[0], b[1], s.qsf[k], kw[k][scale]);
            mma<(j + 1) % 4>(score[j + 1], s.qa[k], b[2], b[3], s.qsf[k], kw[k][scale]);
        });
    });

    if constexpr (Masked) {
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int pos = 64 * (j >> 3) + 32 * ((j >> 2) & 1) + 2 * (j & 3) + 8 * s.t;
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                if (pos + e >= valid_keys) score[j][e] = score[j][2 + e] = -__int_as_float(0x7f800000);
            }
        }
    }
}

template <class State>
__device__ __forceinline__ void pv(State& s, unsigned tile, const Fragment (&early)[4]) {
    using namespace packed_fp4;
    Fragment vs[2];
#pragma unroll
    for (int k = 0; k < 2; ++k)
        cutlass::arch::shared_load<16>(&vs[k], tile + s.vsf_lane + k * layout::scale_fragment_bytes);
    cute::for_each(cute::make_seq<2>{}, [&](auto k) {
        cute::for_each(cute::make_seq<8>{}, [&](auto i) {
            Fragment b;
            if constexpr (k == 0 && i < 4)
                b = early[i];
            else
                b = load_matrix(tile + layout::value_offset + k * layout::fragment_bytes + i * 256 +
                                s.ldsm_off);
            constexpr int j = 2 * i;
            constexpr int group = j >> 2;
            const uint32_t scale = vs[group >> 1][2 * (group & 1) + k];
            mma<j % 4>(s.O + 4 * j, s.Pst + 4 * k, b[0], b[1], s.Pst[8 + k], scale);
            mma<(j + 1) % 4>(s.O + 4 * (j + 1), s.Pst + 4 * k, b[2], b[3], s.Pst[8 + k], scale);
        });
    });
}

template <class State>
__device__ __forceinline__ void means(State& s, unsigned mean_tile, uint32_t hi0, uint32_t hi1) {
    using namespace packed_fp4;
    cute::for_each(cute::make_seq<8>{}, [&](auto i) {
        const Fragment b = load_matrix<true>(mean_tile + s.vm_lane + (((2 * i + s.vm_mh) ^ s.vm_pb) << 4));
        cute::for_each(cute::make_seq<2>{}, [&](auto e) {
            float* c = s.O + 4 * (2 * i + e);
            cute::SM80_16x8x16_F32BF16BF16F32_TN::fma(c[0], c[1], c[2], c[3], s.Mlo[0], s.Mlo[1], hi0,
                                                      hi1, b[2 * e], b[2 * e + 1], c[0],
                                                      c[1], c[2], c[3]);
        });
    });
}

} // namespace comfy_deco::detail
