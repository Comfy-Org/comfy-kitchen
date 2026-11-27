// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>
#include "../float_utils.cuh"
#include "../sage_attention/math.cuh"
#include "layout.h"

namespace comfy_deco::detail {

// The landing FMA deliberately generates FP32 subnormal bit patterns which, once
// saturated as integer halfwords, encode the quantizer's piecewise-linear exponential.
__device__ __forceinline__ float make_subnormal_scale(float c) {
    float m;
    asm("mul.rn.f32 %0,%1,0f00000400;" : "=f"(m) : "f"(c));
    return m;
}

__device__ __forceinline__ uint32_t rescale_mean(uint32_t v, float f) {
    return comfy::pack_bf2(__uint_as_float(v << 16) * f, __uint_as_float(v & 0xffff0000u) * f);
}

__device__ __forceinline__ float soft_rcp(float x) {
    float y;
    asm("rcp.approx.f32 %0,%1;" : "=f"(y) : "f"(x));
    return y;
}

struct SoftScale {
    uint32_t decoded, encoded;
    float delta[2];
};

__device__ __forceinline__ SoftScale soft_scale(float b0, float b1, float c, float a0, float a1) {
    SoftScale v;
    asm("{\n"
        ".reg .b64 b,cc,a,z; .reg .f32 t0,t1,u0,u1; .reg .b16 e,d0,d1; .reg .b32 ah,dh,fl;\n"
        "mov.b64 b,{%4,%5}; mov.b64 cc,{%6,%6}; neg.f32 t0,%7; neg.f32 t1,%8;\n"
        "mov.b64 a,{t0,t1}; fma.rn.f32x2 z,b,cc,a;\n"
        "mov.b64 a,{0f4099d5da,0f4099d5da}; add.rn.f32x2 z,z,a; mov.b64 {t0,t1},z;\n"
        "ex2.approx.ftz.f32 u0,t0; ex2.approx.ftz.f32 u1,t1;\n"
        "cvt.rn.f16x2.f32 ah,u1,u0; cvt.rn.satfinite.e4m3x2.f16x2 e,ah;\n"
        "cvt.u32.u16 %1,e; cvt.rn.f16x2.e4m3x2 dh,e; mov.b32 fl,0x14001400; max.f16x2 dh,dh,fl;\n"
        "mov.b32 %0,dh; mov.b32 {d0,d1},dh; cvt.f32.f16 u0,d0; cvt.f32.f16 u1,d1;\n"
        "lg2.approx.ftz.f32 u0,u0; lg2.approx.ftz.f32 u1,u1; sub.rn.f32 %2,t0,u0; sub.rn.f32 "
        "%3,t1,u1;\n}"
        : "=r"(v.decoded), "=r"(v.encoded), "=f"(v.delta[0]), "=f"(v.delta[1])
        : "f"(b0), "f"(b1), "f"(c), "f"(a0), "f"(a1));
    return v;
}

__device__ __forceinline__ uint32_t land_pair(float x, float y, float m, float offset) {
    uint32_t packed;
    asm("{ .reg .b64 xy,mm,off; .reg .b32 a,b;\n"
        "mov.b64 xy,{%1,%2}; mov.b64 mm,{%3,%3}; mov.b64 off,{%4,%4};\n"
        "fma.rn.f32x2 xy,xy,mm,off; mov.b64 {a,b},xy; cvt.pack.sat.u16.s32 %0,b,a; }"
        : "=r"(packed)
        : "f"(x), "f"(y), "f"(m), "f"(offset));
    return packed;
}

// Eight landed half elements for a lane and one 16-key group. Produce the eight
// E2M1 codes, and independently retain the pre-quantization sum for the means MMA.
__device__ __forceinline__ void land_group(const uint32_t (&z)[4], uint32_t& codes, uint32_t& sum) {
    // Buffer outputs locally until every input operand has been consumed.
    // PTX 9.1 added direct FP16x2 -> FP4x2. Earlier toolkits can
    // widen each half exactly before applying the same E2M1 rounding.
#if __CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)
#define DECO_PACK_HALF2(dst, src) "cvt.rn.satfinite.e2m1x2.f16x2 " dst "," src ";\n"
#else
#define DECO_PACK_HALF2(dst, src)                                                                       \
    "{ .reg .f16 x,y; .reg .f32 fx,fy; mov.b32 {x,y}," src "; "                                         \
    "cvt.f32.f16 fx,x; cvt.f32.f16 fy,y; cvt.rn.satfinite.e2m1x2.f32 " dst ",fy,fx; }\n"
#endif
    asm("{ .reg .b8 a,b,c,d; .reg .b32 s0,s1,s2;\n"
        DECO_PACK_HALF2("a", "%2") DECO_PACK_HALF2("b", "%3")
        DECO_PACK_HALF2("c", "%4") DECO_PACK_HALF2("d", "%5")
        "add.rn.f16x2 s0,%2,%3; add.rn.f16x2 s1,%4,%5; add.rn.f16x2 s2,s0,s1;\n"
        "mov.b32 %0,{a,b,c,d}; mov.b32 %1,s2; }"
        : "=r"(codes), "=r"(sum)
        : "r"(z[0]), "r"(z[1]), "r"(z[2]), "r"(z[3]));
#undef DECO_PACK_HALF2
}

template <int Row>
__device__ __forceinline__ float weighted_row_sum(uint32_t own, uint32_t peer, uint32_t scale) {
    float value;
    uint16_t half = static_cast<uint16_t>(scale >> (16 * Row));
    asm("{ .reg .b16 a,b; .reg .b32 h; .reg .f32 p;\n"
        "add.rn.f16x2 h,%1,%2; mov.b32 {a,b},h;\n"
        "fma.rn.f32.f16 p,b,%3,0f00000000; fma.rn.f32.f16 %0,a,%3,p; }"
        : "=f"(value)
        : "r"(own), "r"(peer), "h"(half));
    return value;
}

__device__ __forceinline__ float soft_fma(float a, float b, float c) {
    float r;
    asm("fma.rn.f32 %0,%1,%2,%3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
    return r;
}

template <bool Masked, bool Odd, class State>
__device__ __forceinline__ void softmax(State& st, float (&scores)[16][4], int valid_keys) {
    float owned[2][2];
    bool need = false;
#pragma unroll
    for (int h = 0; h < 2; h++) {
#pragma unroll
        for (int r = 0; r < 2; r++) {
            float local[2];
#pragma unroll
            for (int s = 0; s < 2; s++) {
                const int j = 8 * s + 4 * h;
                float m0 = fmaxf(fmaxf(scores[j][2 * r], scores[j][2 * r + 1]),
                                 fmaxf(scores[j + 1][2 * r], scores[j + 1][2 * r + 1]));
                float m1 = fmaxf(fmaxf(scores[j + 2][2 * r], scores[j + 2][2 * r + 1]),
                                 fmaxf(scores[j + 3][2 * r], scores[j + 3][2 * r + 1]));
                local[s] = fmaxf(m0, m1);
            }
            // One lane owns s=0 and the adjacent lane owns s=1. Exchange
            // just the two complementary contributions to those maxima.
            float peer = __shfl_xor_sync(0xffffffffu, st.par ? local[0] : local[1], 1);
            float m = fmaxf(st.par ? local[1] : local[0], peer);
            if constexpr (Masked) {
                if (valid_keys != 128) m = fmaxf(m, -1e30f);
            }
            owned[h][r] = m;
            need |= m > st.thr[r];
        }
    }
    if (__any_sync(0xffffffffu, need)) {
        float factor[2];
#pragma unroll
        for (int r = 0; r < 2; r++) {
            float m = fmaxf(owned[0][r], owned[1][r]);
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
            const float top = __fmul_rn(m, st.c);
            const float old = st.anchor[r];
            const float next = (top > __fadd_rn(old, 4.f)) ? top : old;
            factor[r] = ::math::ptx_exp2(__fsub_rn(old, next));
            st.anchor[r] = next;
            st.lsum[r] = __fmul_rn(st.lsum[r], factor[r]);
            const uint32_t ci = __float_as_uint(st.c);
            const float shifted = __fadd_rn(next, 4.f);
            if ((ci & 0x7fffff) == 0 && ci >= 0x00800000 && ci <= 0x7e800000)
                st.thr[r] = __fmul_rd(shifted, __uint_as_float(0x7f000000 - ci));
            else
                st.thr[r] = __fdiv_rd(shifted, st.c);
        }
        st.safe = __all_sync(0xffffffffu, fmaxf(fabsf(st.anchor[0]), fabsf(st.anchor[1])) <= 7900.f &&
                                              st.c <= 0.0625f);
#pragma unroll
        for (int i = 0; i < 64; ++i)
            st.O[i] = __fmul_rn(st.O[i], factor[(i >> 1) & 1]);
        if constexpr (Odd) {
            st.Mlo[0] = rescale_mean(st.Mlo[0], factor[0]);
            st.Mlo[1] = rescale_mean(st.Mlo[1], factor[1]);
        }
    }
    float xs[2][2];
    uint32_t e4[2];
    SoftScale scales[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const float b0 = owned[h][0], b1 = owned[h][1];
        scales[h] = soft_scale(b0, b1, st.c, st.anchor[0], st.anchor[1]);
    }
    if (!st.safe) {
#pragma unroll
        for (int s = 0; s < 2; ++s) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    const float other = __shfl_xor_sync(0xffffffffu, owned[h][r], 1);
                    const float mx = st.par == s ? owned[h][r] : other;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        int j = 8 * s + 4 * h + i;
                        auto& a = scores[j][2 * r];
                        auto& b = scores[j][2 * r + 1];
                        asm("{ .reg .b64 v,m; mov.b64 v,{%0,%1}; mov.b64 m,{%2,%2}; sub.rn.f32x2 v,v,m; "
                            "mov.b64 {%0,%1},v; }"
                            : "+f"(a), "+f"(b)
                            : "f"(mx));
                    }
                }
            }
        }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            owned[h][0] = owned[h][1] = 0.f;
        }
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const auto& sc = scales[h];
        e4[h] = sc.encoded;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            // Adjacent lanes have the same maximum for a given group; each
            // computes one complete rounding offset and sends it to its peer.
            const float own_max = owned[h][r];
            const float origin = soft_fma(-st.M, own_max, __uint_as_float(0x4600u));
            const float own_offset = soft_fma(sc.delta[r], __uint_as_float(0x400u), origin);
            const float peer_offset = __shfl_xor_sync(0xffffffffu, own_offset, 1);
            uint32_t sums[2];
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                const float offset = (st.par == s) ? own_offset : peer_offset;
                uint32_t landed[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int j = 8 * s + 4 * h + i;
                    float a = scores[j][2 * r], b = scores[j][2 * r + 1];
                    landed[i] = land_pair(a, b, st.M, offset);
                }
                land_group(landed, st.Pst[4 * s + 2 * h + r], sums[s]);
            }
            const uint32_t keep = st.par ? sums[1] : sums[0];
            const uint32_t send = st.par ? sums[0] : sums[1];
            const uint32_t peer = __shfl_xor_sync(0xffffffffu, send, 1);
            xs[h][r] = r == 0 ? weighted_row_sum<0>(keep, peer, sc.decoded)
                              : weighted_row_sum<1>(keep, peer, sc.decoded);
        }
    }
#pragma unroll
    for (int r = 0; r < 2; r++) {
        st.lsum[r] = __fadd_rn(st.lsum[r], __fadd_rn(xs[0][r], xs[1][r]));
        const uint32_t ma = comfy::pack_bf2(xs[0][r], xs[1][r]);
        if constexpr (Odd)
            st.Pst[10 + r] = ma;
        else
            st.Mlo[r] = ma;
    }
    uint32_t wa = e4[0] | (e4[1] << 16);
    uint32_t wb = __shfl_xor_sync(0xffffffffu, wa, 1);
    uint32_t dw = __byte_perm(wa, wb, st.par ? 0x3175 : 0x6420);
    uint32_t rw = __shfl_xor_sync(0xffffffffu, dw, 2);
    st.Pst[8] = __byte_perm(dw, rw, 0x5140);
    st.Pst[9] = __byte_perm(dw, rw, 0x7362);
}

template <class State>
__device__ __forceinline__ float soft_total(const State& st, int rr) {
    float l = st.lsum[rr];
    l = __fadd_rn(l, __shfl_xor_sync(0xffffffffu, l, 1));
    return __fadd_rn(l, __shfl_xor_sync(0xffffffffu, l, 2));
}

template <bool SingleWriter, class State>
__device__ __forceinline__ void store_final(State& st, uint16_t* out, int row_g, int head) {
#pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
        const int row = row_g + 8 * rr;
        const float inv = st.vscale * soft_rcp(soft_total(st, rr));
        if constexpr (SingleWriter) {
            // Exchange neighboring words so each warp store fills entire sectors.
            const unsigned valid = __ballot_sync(0xffffffffu, row < st.S);
            if (row >= st.S) continue;
            const int source = st.t & ~1;
            const int parity = st.t & 1;
            const int64_t row_word = (int64_t(row) * st.H + head) * 64;
            uint32_t* dst = reinterpret_cast<uint32_t*>(out) + row_word + source + 4 * parity;
#pragma unroll
            for (int j = 0; j < 16; j += 2) {
                const int e = 4 * j + 2 * rr;
                const uint32_t a = comfy::pack_bf2(st.O[e] * inv, st.O[e + 1] * inv);
                const uint32_t b = comfy::pack_bf2(st.O[e + 4] * inv, st.O[e + 5] * inv);
                // Even lanes store both words from fragment j; odd lanes
                // store both from j+1. Neighbors exchange exactly one word.
                const uint32_t peer = __shfl_xor_sync(valid, parity ? a : b, 1);
                *reinterpret_cast<uint2*>(dst + 4 * j) =
                    parity ? make_uint2(peer, b) : make_uint2(a, peer);
            }
        } else if (row < st.S) {
            uint32_t* dst = reinterpret_cast<uint32_t*>(out) + (int64_t(row) * st.H + head) * 64 + st.t;
#pragma unroll
            for (int jc = 0; jc < 16; jc++)
                dst[4 * jc] = comfy::pack_bf2(st.O[4 * jc + 2 * rr] * inv, st.O[4 * jc + 2 * rr + 1] * inv);
        }
    }
}

// Reserve two float-accumulator records per split tile for either policy.
template <bool SingleWriter, class State>
__device__ __forceinline__ void store_partial(State& st, uint16_t* out, int row_g, int head,
                                              void* scratch, int split_count, int slot, int part,
                                              int* shared_arrived) {
    constexpr int threads = layout::threads, vectors = layout::partial_vectors;
    constexpr int stride = SingleWriter ? threads : 1;
    auto* records = static_cast<uint4*>(scratch);
    auto* flag = reinterpret_cast<unsigned int*>(static_cast<unsigned char*>(scratch) +
                                                 split_count * layout::split_pair_bytes) +
                 slot;
    const float sums[2] = {soft_total(st, 0), soft_total(st, 1)};
    auto store_record = [&](uint4* dst) {
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            dst[j * stride] = make_uint4(__float_as_uint(st.O[4 * j]), __float_as_uint(st.O[4 * j + 1]),
                                         __float_as_uint(st.O[4 * j + 2]), __float_as_uint(st.O[4 * j + 3]));
        }
        dst[16 * stride] = make_uint4(__float_as_uint(sums[0]), __float_as_uint(sums[1]),
                                      __float_as_uint(st.anchor[0]), __float_as_uint(st.anchor[1]));
    };
    const uint4* peer;
    if constexpr (SingleWriter) {
        auto* mine = records + int64_t(slot) * threads * vectors + threadIdx.x;
        // Election takes place after this CTA has started. A waiting consumer
        // thus depends exclusively on a publisher already resident on the GPU.
        if (threadIdx.x == 0) *shared_arrived = atomicCAS(flag, 0u, 1u) == 0u;
        __syncthreads();
        if (*shared_arrived) {
            store_record(mine);
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0) {
                asm volatile("st.release.gpu.global.u32 [%0],%1;" ::"l"(flag), "r"(2u) : "memory");
            }
            return;
        }
        if (threadIdx.x == 0) {
            unsigned ready;
            do {
                asm volatile("ld.acquire.gpu.global.u32 %0,[%1];" : "=r"(ready) : "l"(flag) : "memory");
            } while (ready != 2u);
        }
        __syncthreads();
        peer = mine;
    } else {
        uint4* mine = records + ((int64_t(slot) * 2 + part - 1) * threads + threadIdx.x) * vectors;
        store_record(mine);
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned before;
            asm volatile("atom.acq_rel.gpu.global.add.u32 %0,[%1],1;"
                         : "=r"(before)
                         : "l"(flag)
                         : "memory");
            *shared_arrived = before;
        }
        __syncthreads();
        if (*shared_arrived != 1) return;
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
        peer = records + ((int64_t(slot) * 2 + 2 - part) * threads + threadIdx.x) * vectors;
    }
    const uint4 ps = __ldcg(peer + 16 * stride);
    const float peer_sum[2] = {__uint_as_float(ps.x), __uint_as_float(ps.y)};
    const float peer_anchor[2] = {__uint_as_float(ps.z), __uint_as_float(ps.w)};
    float f1[2], f2[2], inv[2];
#pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
        const float a1 = part == 1 ? st.anchor[rr] : peer_anchor[rr];
        const float a2 = part == 1 ? peer_anchor[rr] : st.anchor[rr];
        const float high = fmaxf(a1, a2);
        f1[rr] = ::math::ptx_exp2(a1 - high);
        f2[rr] = ::math::ptx_exp2(a2 - high);
        const float l1 = part == 1 ? sums[rr] : peer_sum[rr];
        const float l2 = part == 1 ? peer_sum[rr] : sums[rr];
        inv[rr] = st.vscale * soft_rcp(l1 * f1[rr] + l2 * f2[rr]);
    }
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const uint4 v = __ldcg(peer + j * stride);
        const float other[4] = {__uint_as_float(v.x), __uint_as_float(v.y), __uint_as_float(v.z),
                                __uint_as_float(v.w)};
#pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
            float x[2];
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                int i = 2 * rr + e;
                const float o1 = part == 1 ? st.O[4 * j + i] : other[i];
                const float o2 = part == 1 ? other[i] : st.O[4 * j + i];
                x[e] = (o1 * f1[rr] + o2 * f2[rr]) * inv[rr];
            }
            int row = row_g + 8 * rr;
            if (row < st.S) {
                uint32_t* dst =
                    reinterpret_cast<uint32_t*>(out) + (int64_t(row) * st.H + head) * 64 + st.t;
                dst[4 * j] = comfy::pack_bf2(x[0], x[1]);
            }
        }
    }
    if constexpr (SingleWriter) {
        __syncthreads();
        if (threadIdx.x == 0)
            asm volatile("st.release.gpu.global.u32 [%0],%1;" ::"l"(flag), "r"(0u) : "memory");
    } else if (threadIdx.x == 0) {
        *flag = 0u;
    }
}

} // namespace comfy_deco::detail
