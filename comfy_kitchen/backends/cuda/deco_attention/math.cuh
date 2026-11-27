// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "layout.h"
#include <type_traits>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/mma_sm80.hpp>

namespace comfy_deco::detail {
namespace packed_fp4 {

template <int I, int N, typename F>
__device__ __forceinline__ void unroll(F const& fn) {
    if constexpr (I < N) {
        fn(std::integral_constant<int, I>{});
        unroll<I + 1, N>(fn);
    }
}

template <int I>
__device__ __forceinline__ uint32_t word(uint4 v) {
    static_assert(I >= 0 && I < 4);
    if constexpr (I == 0) return v.x;
    if constexpr (I == 1) return v.y;
    if constexpr (I == 2) return v.z;
    return v.w;
}

__device__ __forceinline__ uint32_t global_u32(uintptr_t address, bool valid) {
    uint32_t out;
    asm("{ .reg .pred p; setp.ne.s32 p,%2,0; mov.b32 %0,0; @p ld.global.nc.b32 %0,[%1]; }"
        : "=r"(out)
        : "l"(address), "r"(int(valid)));
    return out;
}

__device__ __forceinline__ uint4 shared_u128(unsigned address) {
    uint4 r;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3},[%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
                 : "r"(address));
    return r;
}

template <bool Transposed = false>
__device__ __forceinline__ uint4 load_matrix(unsigned address) {
    auto const& src = *reinterpret_cast<cute::uint128_t const*>(__cvta_shared_to_generic(address));
    uint4 r;
    if constexpr (Transposed)
        cute::SM75_U16x8_LDSM_T::copy(src, r.x, r.y, r.z, r.w);
    else
        cute::SM75_U32x4_LDSM_N::copy(src, r.x, r.y, r.z, r.w);
    return r;
}

// The packed scale selectors are (0,0) for A and (0,ThreadIdB) for B.
// The four B selectors consume a single four-lane-distributed block scale.
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
__device__ __forceinline__ void preload_q(State const& s, unsigned tile, uint4 (&early)[4]) {
#pragma unroll
    for (int jp = 0; jp < 4; ++jp)
        early[jp] = packed_fp4::load_matrix(tile + s.ldsm_off + 256 * jp);
}

template <class State>
__device__ __forceinline__ void preload_v(State const& s, unsigned tile, uint4 (&early)[4]) {
#pragma unroll
    for (int jp = 0; jp < 4; ++jp)
        early[jp] = packed_fp4::load_matrix(tile + layout::value_offset + s.ldsm_off + 256 * jp);
}

template <bool Masked, class State>
__device__ __forceinline__ void qk(State const& s, unsigned tile, float (&score)[16][4], int valid_keys,
                                   const uint4 (&early)[4]) {
    using namespace packed_fp4;
    uint4 kw[3];
#pragma unroll
    for (int a = 0; a < 3; ++a)
        kw[a] = shared_u128(tile + s.ksf_lane + layout::scale_fragment_bytes * a);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
#pragma unroll
        for (int e = 0; e < 4; ++e)
            score[j][e] = 0.f;
    }

    unroll<0, 3>([&](auto kk) {
        constexpr int k = decltype(kk)::value;
        unroll<0, 8>([&](auto ii) {
            constexpr int i = decltype(ii)::value;
            uint4 b;
            if constexpr (k == 0 && i < 4)
                b = early[i];
            else
                b = load_matrix(tile + k * layout::fragment_bytes + i * 256 + s.ldsm_off);
            constexpr int j = 2 * i;
            constexpr int scale = 2 * (j >> 3) + ((j >> 2) & 1);
            mma<j % 4>(score[j], s.qa[k], b.x, b.y, s.qsf[k], word<scale>(kw[k]));
            mma<(j + 1) % 4>(score[j + 1], s.qa[k], b.z, b.w, s.qsf[k], word<scale>(kw[k]));
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
__device__ __forceinline__ void pv(State& s, unsigned tile, const uint4 (&early)[4]) {
    using namespace packed_fp4;
    uint4 vs[2] = {shared_u128(tile + s.vsf_lane),
                   shared_u128(tile + s.vsf_lane + layout::scale_fragment_bytes)};
    unroll<0, 2>([&](auto kk) {
        constexpr int k = decltype(kk)::value;
        unroll<0, 8>([&](auto ii) {
            constexpr int i = decltype(ii)::value;
            uint4 b;
            if constexpr (k == 0 && i < 4)
                b = early[i];
            else
                b = load_matrix(tile + layout::value_offset + k * layout::fragment_bytes + i * 256 +
                                s.ldsm_off);
            constexpr int j = 2 * i;
            constexpr int group = j >> 2;
            const uint32_t scale = word<2 * (group & 1) + k>(vs[group >> 1]);
            mma<j % 4>(s.O + 4 * j, s.Pst + 4 * k, b.x, b.y, s.Pst[8 + k], scale);
            mma<(j + 1) % 4>(s.O + 4 * (j + 1), s.Pst + 4 * k, b.z, b.w, s.Pst[8 + k], scale);
        });
    });
}

template <class State>
__device__ __forceinline__ void means(State& s, unsigned mean_tile, uint32_t hi0, uint32_t hi1) {
    using namespace packed_fp4;
    unroll<0, 8>([&](auto ii) {
        constexpr int i = decltype(ii)::value;
        const uint4 b = load_matrix<true>(mean_tile + s.vm_lane + (((2 * i + s.vm_mh) ^ s.vm_pb) << 4));
        unroll<0, 2>([&](auto ee) {
            constexpr int e = decltype(ee)::value;
            float* c = s.O + 4 * (2 * i + e);
            cute::SM80_16x8x16_F32BF16BF16F32_TN::fma(c[0], c[1], c[2], c[3], s.Mlo[0], s.Mlo[1], hi0,
                                                      hi1, word<2 * e>(b), word<2 * e + 1>(b), c[0],
                                                      c[1], c[2], c[3]);
        });
    });
}

} // namespace comfy_deco::detail
