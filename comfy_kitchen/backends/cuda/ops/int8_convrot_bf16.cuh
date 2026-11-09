/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Exact ordered FP32 ConvRot256 / CK BF16 quantization, K=4096.
// CUDA C++ port of the qualified CuTe DSL fixed-point specialization.
#pragma once
#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace comfy::convrot_bf16 {
__device__ __forceinline__ float2 h4(float2 a, float2 b, float2 c, float2 d) {
    float2 r;
    asm("{ .reg .b64 a,b,c,d,h; mov.b64 a,{%2,%3}; mov.b64 b,{%4,%5};"
        "mov.b64 c,{%6,%7}; mov.b64 d,{%8,%9}; mov.b64 h,0x3f0000003f000000;"
        "add.rn.ftz.f32x2 a,a,b; add.rn.ftz.f32x2 a,a,c; add.rn.ftz.f32x2 a,a,d;"
        "mul.rn.ftz.f32x2 a,a,h; mov.b64 {%0,%1},a; }"
        : "=f"(r.x), "=f"(r.y)
        : "f"(a.x), "f"(a.y), "f"(b.x), "f"(b.y), "f"(c.x), "f"(c.y), "f"(d.x), "f"(d.y));
    return r;
}
// NVCC's fast-math unary minus emits neg.ftz.f32, which becomes an extra
// scalar FADD per lane. A non-flushing negate preserves the sign bit and lets
// ptxas fold it into the packed arithmetic's source modifiers. The H4 adds
// still explicitly flush to zero, exactly as in the original recipe.
__device__ __forceinline__ float sign_flip(float a) {
    float result;
    asm("neg.f32 %0,%1;" : "=f"(result) : "f"(a));
    return result;
}
__device__ __forceinline__ float2 neg(float2 a) { return {sign_flip(a.x), sign_flip(a.y)}; }
__device__ __forceinline__ float2 decode(unsigned bits) {
    float2 r;
    asm("prmt.b32 %0,%2,0,0x1044; prmt.b32 %1,%2,0,0x3244;" : "=f"(r.x), "=f"(r.y) : "r"(bits));
    return r;
}
__device__ __forceinline__ unsigned numerator(float2 a) {
    unsigned r;
    asm("cvt.rn.bf16x2.f32 %0,%2,%1;" : "=r"(r) : "f"(a.x), "f"(a.y));
    return r;
}
// RNE BF16 numerator, BF16-rounded scale, approximate FP32 reciprocal and
// BF16-rounded quotient match CK's fast-math quantizer. S2F6 at scale 64 has
// the signed INT8 byte encoding. Keep the lower clamp for CK's NaN behavior;
// the converter supplies rounding and upper saturation without scalar CVTs.
__device__ __forceinline__ unsigned encode(unsigned n0, unsigned n1, float inv) {
    unsigned r;
    asm("{ .reg .b32 q01,q23,lo; .reg .b16 out01,out23,sf; .reg .f32 a0,a1,a2,a3,c0,c1,c2,c3; .reg "
        ".b64 a,b;"
        "prmt.b32 a0,%1,0,0x1044; prmt.b32 a1,%1,0,0x3244; prmt.b32 a2,%2,0,0x1044; prmt.b32 "
        "a3,%2,0,0x3244;"
        "mov.b64 b,{%3,%3}; mov.b64 a,{a0,a1}; mul.rn.ftz.f32x2 a,a,b; mov.b64 {c0,c1},a;"
        "mov.b64 a,{a2,a3}; mul.rn.ftz.f32x2 a,a,b; mov.b64 {c2,c3},a;"
        "cvt.rn.bf16x2.f32 q01,c1,c0; cvt.rn.bf16x2.f32 q23,c3,c2; mov.b32 lo,0xc300c300;"
        "max.bf16x2 q01,q01,lo; max.bf16x2 q23,q23,lo; mov.b16 sf,0x8585;"
        "cvt.rn.satfinite.scaled::n2::ue8m0.s2f6x2.bf16x2 out01,q01,sf;"
        "cvt.rn.satfinite.scaled::n2::ue8m0.s2f6x2.bf16x2 out23,q23,sf; mov.b32 %0,{out01,out23}; }"
        : "=r"(r)
        : "r"(n0), "r"(n1), "f"(inv));
    return r;
}
__device__ __forceinline__ int exchange(int i) {
    return i ^ (((i >> 6) & 7) << 2) ^ (((i >> 4) & 3) << 2);
}
template <int Stride> __device__ __forceinline__ void stage(float (&v)[64]) {
    float next[64];
#pragma unroll
    for (int p = 0; p < 32; p++) {
        constexpr int stride = Stride;
        int i = p * 2, base = (i / (4 * stride)) * (4 * stride) + i % stride,
            slot = (i / stride) % 4;
        float2 a = {v[base], v[base + 1]}, b = {v[base + stride], v[base + stride + 1]},
               c = {v[base + 2 * stride], v[base + 2 * stride + 1]},
               d = {v[base + 3 * stride], v[base + 3 * stride + 1]};
        if (slot == 3)
            a = neg(a);
        if (slot == 2)
            b = neg(b);
        if (slot == 1)
            c = neg(c);
        if (slot == 0)
            d = neg(d);
        auto y = h4(a, b, c, d);
        next[i] = y.x;
        next[i + 1] = y.y;
    }
#pragma unroll
    for (int i = 0; i < 64; i++)
        v[i] = next[i];
}
// One CTA owns a complete row. TMA first uses 8 KiB of shared memory; after
// all readers finish, the same allocation holds the FP32 exchange. Groups
// exchanged in the last H4 stage are entirely within a warp.
template <bool Diagnostic = false>
__global__ __launch_bounds__(64) void kernel(const __grid_constant__ CUtensorMap map,
                                             int8_t *__restrict__ q, float *__restrict__ scales,
                                             float *diagnostic = nullptr) {
    __shared__ __align__(1024) float shared[4096];
    __shared__ __align__(8) uint64_t barrier;
    __shared__ float peaks[2];
    unsigned sm = unsigned(__cvta_generic_to_shared(shared)),
             bar = unsigned(__cvta_generic_to_shared(&barrier));
    int t = threadIdx.x, row = blockIdx.x;
    if (t == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0],1; fence.mbarrier_init.release.cluster; "
                     "fence.proxy.async.shared::cta;" ::"r"(bar)
                     : "memory");
        asm volatile("mbarrier.arrive.expect_tx.shared.b64 _,[%0],8192;" ::"r"(bar) : "memory");
    }
    __syncthreads();
    if (t == 0)
        asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes "
                     "[%0],[%1,{0,0,%2}],[%3];" ::"r"(sm),
                     "l"(&map), "r"(row), "r"(bar)
                     : "memory");
    asm volatile("{ .reg .pred p; wait_loop: mbarrier.try_wait.parity.shared.b64 p,[%0],0; @!p bra "
                 "wait_loop; }" ::"r"(bar)
                 : "memory");
    __syncthreads();
    float v[64];
#pragma unroll
    for (int chunk = 0; chunk < 8; chunk++) {
        int logical = t * 64 + chunk * 8, physical = logical ^ (((logical >> 6) & 7) << 3);
        auto raw =
            *reinterpret_cast<const uint4 *>(reinterpret_cast<const uint16_t *>(shared) + physical);
        unsigned words[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
        for (int j = 0; j < 4; j++) {
            auto a = decode(words[j]);
            v[chunk * 8 + j * 2] = a.x;
            v[chunk * 8 + j * 2 + 1] = a.y;
        }
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 64; i += 4) {
        float a = v[i], b = v[i + 1], c = v[i + 2], d = v[i + 3];
        auto y = h4({a, a}, {b, b}, {c, sign_flip(c)}, {sign_flip(d), d});
        auto z = h4({a, sign_flip(a)}, {sign_flip(b), b}, {c, c}, {d, d});
        v[i] = y.x;
        v[i + 1] = y.y;
        v[i + 2] = z.x;
        v[i + 3] = z.y;
    }
    stage<4>(v);
    stage<16>(v);
#pragma unroll
    for (int i = 0; i < 64; i += 4)
        *reinterpret_cast<float4 *>(shared + exchange(t * 64 + i)) = {v[i], v[i + 1], v[i + 2],
                                                                      v[i + 3]};
    __syncwarp();
    unsigned held[32];
    float peak = 0;
#pragma unroll
    for (int chunk = 0; chunk < 4; chunk++) {
        float4 gathered[4];
#pragma unroll
        for (int s = 0; s < 4; s++)
            gathered[s] = *reinterpret_cast<const float4 *>(
                shared + exchange((t / 4) * 256 + (t % 4) * 16 + s * 64 + chunk * 4));
#pragma unroll
        for (int pair = 0; pair < 2; pair++) {
            float2 a =
                pair ? float2{gathered[0].z, gathered[0].w} : float2{gathered[0].x, gathered[0].y};
            float2 b =
                pair ? float2{gathered[1].z, gathered[1].w} : float2{gathered[1].x, gathered[1].y};
            float2 c =
                pair ? float2{gathered[2].z, gathered[2].w} : float2{gathered[2].x, gathered[2].y};
            float2 d =
                pair ? float2{gathered[3].z, gathered[3].w} : float2{gathered[3].x, gathered[3].y};
#pragma unroll
            for (int slot = 0; slot < 4; slot++) {
                auto y = h4(slot == 3 ? neg(a) : a, slot == 2 ? neg(b) : b, slot == 1 ? neg(c) : c,
                            slot == 0 ? neg(d) : d);
                int vi = slot * 16 + chunk * 4 + pair * 2;
                held[vi / 2] = numerator(y);
                peak = fmaxf(peak, fabsf(y.x));
                peak = fmaxf(peak, fabsf(y.y));
                if constexpr (Diagnostic) {
                    int col = (t / 4) * 256 + (t % 4) * 16 + slot * 64 + chunk * 4 + pair * 2;
                    diagnostic[int64_t(row) * 4096 + col] = y.x;
                    diagnostic[int64_t(row) * 4096 + col + 1] = y.y;
                }
            }
        }
    }
#pragma unroll
    for (int shift = 1; shift < 32; shift *= 2)
        peak = fmaxf(peak, __shfl_xor_sync(0xffffffff, peak, shift));
    if ((t & 31) == 0)
        peaks[t / 32] = peak;
    __syncthreads();
    peak = fminf(fmaxf(peaks[0], peaks[1]), 3.3895313892515355e38f);
    float scale = fmaxf(__fmul_rn(peak, 1.f / 127.f), 1.e-30f);
    if (t == 0)
        scales[row] = scale;
    float inv;
    asm("{ .reg .b16 d; .reg .f32 b; cvt.rn.bf16.f32 d,%1; cvt.f32.bf16 b,d; rcp.approx.ftz.f32 "
        "%0,b; }"
        : "=f"(inv)
        : "f"(scale));
#pragma unroll
    for (int slot = 0; slot < 4; slot++) {
        uint4 out;
        out.x = encode(held[slot * 8], held[slot * 8 + 1], inv);
        out.y = encode(held[slot * 8 + 2], held[slot * 8 + 3], inv);
        out.z = encode(held[slot * 8 + 4], held[slot * 8 + 5], inv);
        out.w = encode(held[slot * 8 + 6], held[slot * 8 + 7], inv);
        *reinterpret_cast<uint4 *>(q + int64_t(row) * 4096 + (t / 4) * 256 + (t % 4) * 16 +
                                   slot * 64) = out;
    }
}
} // namespace comfy::convrot_bf16
