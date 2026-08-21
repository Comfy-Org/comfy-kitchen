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

// Sol-Attn preprocess, CUDA.
//
// Produces everything the route and exact kernels consume, in the layouts
// sol_layout.cuh defines:
//   qiP  [B,T,H,D] int8   (perm_d)            qs   [B,T,H] f32
//   kiP  [B*H,Tp,D] int8  (perm_key + perm_d) ksb  [B*H,Tp] float2 = (ks, bias)
//   vTi  [B*H,D,Tp] int8  (perm_d on keys)    vsc  [B*H,D] f32     -- see sol_vtranspose.cu
//   kciP [B*H,NPAD,D] int8 (perm_d)           kcs  [B*H,NPAD] f32
//   vcT  [B*H,D,NPAD] bf16                    threshold [B*H,NQ] f32
//
// Q-side tensors (qiP, qs) are sized and indexed by T, NOT Tp; only the K/V
// layouts pad to Tp. The route and exact kernels index per-token state the same
// way, which is what lets the caller's `out` alias o_part.
//
// q/k/v are read through explicit strides (last dim contiguous -- the staging
// loads uint4), so a BHND view goes in without a copy.
//
// Five passes -- K twice, V twice, Q once: K's smoothing mean and V's
// per-channel scale are global reductions that must complete before their
// quantization, and neither folds into one pass without a grid-wide sync.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>

#include "sol_layout.cuh"

namespace {
using namespace sol;

constexpr int HD = HEAD_DIM, BLK = BLOCK;
constexpr int LDQ = HD + 8;          // bf16 elements; x2 must be a multiple of
                                     // 16 -- the staging writes uint4

__device__ __forceinline__ int8_t q8d(float x, float inv) {
    return (int8_t)max(-127, min(127, __float2int_rn(x * inv)));
}

// Block-wide max over 128 threads (one per channel).
__device__ __forceinline__ float block_absmax(float x, float* s) {
    const int d = threadIdx.x;
    s[d] = fabsf(x);
    __syncthreads();
    for (int w = 64; w; w >>= 1) {
        if (d < w) s[d] = fmaxf(s[d], s[d + w]);
        __syncthreads();
    }
    const float r = s[0];
    __syncthreads();
    return r;
}

// ---- pass 1: K and V block reductions, plus V's per-channel |max| ----
// One CUDA block per (bh, pooled block); thread == channel, so each step of the
// token loop is a fully coalesced 256-byte read.
__global__ void prep_reduce_kv(const __nv_bfloat16* __restrict__ k,
                               const __nv_bfloat16* __restrict__ v,
                               float* __restrict__ kc, __nv_bfloat16* __restrict__ vcT,
                               float* __restrict__ vamax,
                               int T, int H, int NTB, int NPAD,
                               int64_t sb, int64_t st, int64_t sh,
                               int64_t vb, int64_t vt, int64_t vh) {
    const int n = blockIdx.x, bh = blockIdx.y;
    const int batch = bh / H, head = bh % H, d = threadIdx.x;
    const size_t o = ((size_t)bh * NPAD + n) * HD + d;
    if (n >= NTB) {
        kc[o] = 0.f;
        vcT[((size_t)bh * HD + d) * NPAD + n] = __float2bfloat16(0.f);
        return;
    }

    const int t0 = n * BLK, len = min(BLK, T - t0);
    float sk = 0.f, sv = 0.f, av = 0.f;
    for (int i = 0; i < len; ++i) {
        const int64_t off  = batch * sb + (int64_t)(t0 + i) * st + head * sh + d;
        const int64_t voff = batch * vb + (int64_t)(t0 + i) * vt + head * vh + d;
        const float kk = __bfloat162float(k[off]);
        const float vv = __bfloat162float(v[voff]);
        sk += kk; sv += vv; av = fmaxf(av, fabsf(vv));
    }
    kc[o] = sk / (float)len;    // block MEAN of K
    // Block SUMS of V, written transposed to skip an f32 staging copy; the
    // per-channel |max| reduces by atomic instead of a per-block array.
    vcT[((size_t)bh * HD + d) * NPAD + n] = __float2bfloat16(sv);
    atomicMax(reinterpret_cast<unsigned int*>(&vamax[(size_t)bh * HD + d]),
              __float_as_uint(av));   // av >= 0, so the bit pattern orders correctly
}

// ---- pass 2: reductions over the pooled tensors (small) ----
__global__ void prep_pooled_stats(const float* __restrict__ kc,
                                  float* __restrict__ kmean, float* __restrict__ vsc,
                                  float* __restrict__ kcvar, int NTB, int NPAD) {
    const int bh = blockIdx.x, d = threadIdx.x;
    float sm = 0.f, ss = 0.f;
    for (int n = 0; n < NTB; ++n) {
        const float x = kc[((size_t)bh * NPAD + n) * HD + d];
        sm += x;
        ss = fmaf(x, x, ss);
    }
    const float m = sm / (float)NTB;
    kmean[(size_t)bh * HD + d] = m;
    // vsc currently holds the atomically-reduced |max| from pass 1.
    vsc[(size_t)bh * HD + d] = fmaxf(vsc[(size_t)bh * HD + d] / 127.0f, 1e-8f);
    kcvar[(size_t)bh * HD + d] = fmaxf(ss / (float)NTB - m * m, 0.f);
}

// ---- pass 3: centre + quantize the pooled keys; transpose the pooled values ----
__global__ void prep_pooled_quant(const float* __restrict__ kc,
                                  const float* __restrict__ kmean,
                                  int8_t* __restrict__ kciP, float* __restrict__ kcs,
                                  int NTB, int NPAD) {
    __shared__ float sred[HD];
    const int n = blockIdx.x, bh = blockIdx.y, d = threadIdx.x;
    const size_t o = ((size_t)bh * NPAD + n) * HD + d;
    const bool live = n < NTB;
    const float x = live ? (kc[o] - kmean[(size_t)bh * HD + d]) : 0.f;
    const float mx = block_absmax(x, sred);
    const float sc = fmaxf(mx / 127.0f, 1e-12f);
    if (d == 0) kcs[(size_t)bh * NPAD + n] = live ? sc : 0.f;
    kciP[((size_t)bh * NPAD + n) * HD + perm_d(d)] = live ? q8d(x, 1.f / sc) : (int8_t)0;
}

// ---- pass 4: quantize Q and derive the routing threshold, from one read ----
// The per-token scale is a reduction over channels and the routing centroid is a
// reduction over tokens, so the tile is staged once and read both ways.
__global__ void prep_q(const __nv_bfloat16* __restrict__ q, const float* __restrict__ kcvar,
                       int8_t* __restrict__ qiP, float* __restrict__ qs,
                       float* __restrict__ thr,
                       int8_t* __restrict__ cen8, float* __restrict__ cens,
                       int T, int Tp, int H, int NQ, float tau, float log2s,
                       int64_t sb, int64_t st, int64_t sh) {
    __shared__ __nv_bfloat16 sQ[BLK * LDQ];
    __shared__ __align__(16) float sred[HD];
    const int qb = blockIdx.x, bh = blockIdx.y;
    const int batch = bh / H, head = bh % H, tid = threadIdx.x;
    const int t0 = qb * BLK, len = min(BLK, T - t0);

    for (int idx = tid; idx < BLK * (HD / 8); idx += HD) {
        const int t = idx / (HD / 8), c8 = (idx % (HD / 8)) * 8;
        uint4 val = make_uint4(0u, 0u, 0u, 0u);
        if (t < len)
            val = *reinterpret_cast<const uint4*>(
                q + batch * sb + (int64_t)(t0 + t) * st + head * sh + c8);
        *reinterpret_cast<uint4*>(sQ + t * LDQ + c8) = val;
    }
    __syncthreads();

    // per-token scale + quantized store; one thread per token, channels in-thread
    for (int t = tid; t < len; t += HD) {
        float a = 0.f;
        for (int d = 0; d < HD; ++d) a = fmaxf(a, fabsf(__bfloat162float(sQ[t * LDQ + d])));
        const float sc = fmaxf(a / 127.0f, 1e-8f);
        qs[((size_t)(batch * T + t0 + t)) * H + head] = sc;
        const size_t base = ((size_t)(batch * T + t0 + t) * H + head) * HD;
        const float inv = 1.f / sc;
        // Build the permuted row in registers, then store it vectorized. Writing
        // qiP[perm_d(d)] directly costs 128 scattered 1-byte stores per token.
        int8_t out[HD];
        #pragma unroll
        for (int d = 0; d < HD; ++d)
            out[perm_d(d)] = q8d(__bfloat162float(sQ[t * LDQ + d]), inv);
        #pragma unroll
        for (int c = 0; c < HD; c += 16)
            *reinterpret_cast<uint4*>(qiP + base + c) = *reinterpret_cast<const uint4*>(out + c);
    }
    __syncthreads();

    // routing threshold: sigma of the proxy row, from the query centroid
    const int d = tid;
    float c = 0.f;
    for (int t = 0; t < len; ++t) c += __bfloat162float(sQ[t * LDQ + d]);
    c /= (float)len;
    sred[d] = c * c * kcvar[(size_t)bh * HD + d];
    __syncthreads();
    for (int w = 64; w; w >>= 1) {
        if (d < w) sred[d] += sred[d + w];
        __syncthreads();
    }
    if (d == 0)
        thr[(size_t)bh * NQ + qb] = tau * sqrtf(sred[0] * log2s * log2s + 1e-6f);

    // Centroid for the routing pass, quantized like a pseudo-row (same
    // perm_d as the pooled keys, so their dot needs no unpermute).
    __syncthreads();
    sred[d] = fabsf(c);
    __syncthreads();
    for (int w = 64; w; w >>= 1) {
        if (d < w) sred[d] = fmaxf(sred[d], sred[d + w]);
        __syncthreads();
    }
    const float csc = fmaxf(sred[0] / 127.0f, 1e-8f);
    __syncthreads();                       // sred is reused as a byte buffer
    char* s8 = reinterpret_cast<char*>(sred);
    s8[perm_d(d)] = (char)q8d(c, 1.f / csc);
    __syncthreads();
    const size_t cbase = ((size_t)bh * NQ + qb) * HD;
    if (d < HD / 16)
        reinterpret_cast<uint4*>(cen8 + cbase)[d] =
            reinterpret_cast<const uint4*>(s8)[d];
    if (d == 0) cens[(size_t)bh * NQ + qb] = csc;
}

// ---- pass 5: centre + quantize K into the permuted layout ----
__global__ void prep_k(const __nv_bfloat16* __restrict__ k, const float* __restrict__ kmean,
                       int8_t* __restrict__ kiP, float2* __restrict__ ksb,
                       const float* __restrict__ kbias,   // [B, T] log2 units, or null
                       int T, int Tp, int H, int NTB,
                       int64_t sb, int64_t st, int64_t sh) {
    __shared__ __nv_bfloat16 sK[BLK * LDQ];
    const int n = blockIdx.x, bh = blockIdx.y;
    const int batch = bh / H, head = bh % H, tid = threadIdx.x;
    const int t0 = n * BLK, len = max(0, min(BLK, T - t0));

    for (int idx = tid; idx < BLK * (HD / 8); idx += HD) {
        const int t = idx / (HD / 8), c8 = (idx % (HD / 8)) * 8;
        uint4 val = make_uint4(0u, 0u, 0u, 0u);
        if (t < len)
            val = *reinterpret_cast<const uint4*>(
                k + batch * sb + (int64_t)(t0 + t) * st + head * sh + c8);
        *reinterpret_cast<uint4*>(sK + t * LDQ + c8) = val;
    }
    __syncthreads();

    // destination row p takes source row perm_key(p); the smem read absorbs it
    for (int p = tid; p < BLK; p += HD) {
        const int s = perm_key(p);
        const bool live = s < len;
        const size_t dst = (size_t)bh * Tp + n * BLK + p;
        float a = 0.f;
        for (int d = 0; d < HD; ++d)
            a = fmaxf(a, fabsf(__bfloat162float(sK[s * LDQ + d]) - kmean[(size_t)bh * HD + d]));
        const float sc = fmaxf(a / 127.0f, 1e-8f);
        // Per-key additive logit bias; only the exact branch reads ksb, so
        // biased blocks must be sink-routed.
        const float bias = (kbias && live) ? kbias[(size_t)batch * T + t0 + s] : 0.f;
        ksb[dst] = make_float2(live ? sc : 0.f, live ? bias : NEG);
        const float inv = 1.f / sc;
        int8_t out[HD];
        #pragma unroll
        for (int d = 0; d < HD; ++d) {
            const float x = __bfloat162float(sK[s * LDQ + d]) - kmean[(size_t)bh * HD + d];
            out[perm_d(d)] = live ? q8d(x, inv) : (int8_t)0;
        }
        #pragma unroll
        for (int c = 0; c < HD; c += 16)
            *reinterpret_cast<uint4*>(kiP + dst * HD + c) = *reinterpret_cast<const uint4*>(out + c);
    }
}

}  // namespace

extern "C" void launch_sol_preprocess(
    const void* q, const void* k, const void* v,
    void* qiP, void* qs, void* kiP, void* ksb, void* kciP, void* kcs,
    void* vcT, void* threshold, void* cen8, void* cens, void* vsc,
    void* scratch,           // B*H*NPAD*HD + 2 * B*H*HD floats
    const void* key_bias,    // [B, T] f32 in log2 units, or nullptr
    int B, int T, int Tp, int H, int NTB, int NPAD, int NQ,
    int64_t qs_b, int64_t qs_t, int64_t qs_h,
    int64_t ks_b, int64_t ks_t, int64_t ks_h,
    int64_t vs_b, int64_t vs_t, int64_t vs_h,
    float tau, float scale_log2, cudaStream_t stream)
{
    float* s = (float*)scratch;
    const size_t pooled = (size_t)B * H * NPAD * HD;
    float* kc = s;
    float* kmean = s + pooled;
    float* kcvar = kmean + (size_t)B * H * HD;

    // vsc accumulates V's per-channel |max| by atomicMax in pass 1, so start at 0.
    cudaMemsetAsync(vsc, 0, (size_t)B * H * HD * sizeof(float), stream);
    prep_reduce_kv<<<dim3(NPAD, B * H), HD, 0, stream>>>(
        (const __nv_bfloat16*)k, (const __nv_bfloat16*)v, kc, (__nv_bfloat16*)vcT,
        (float*)vsc, T, H, NTB, NPAD, ks_b, ks_t, ks_h, vs_b, vs_t, vs_h);
    prep_pooled_stats<<<B * H, HD, 0, stream>>>(kc, kmean, (float*)vsc, kcvar, NTB, NPAD);
    prep_pooled_quant<<<dim3(NPAD, B * H), HD, 0, stream>>>(
        kc, kmean, (int8_t*)kciP, (float*)kcs, NTB, NPAD);
    prep_q<<<dim3(NQ, B * H), HD, 0, stream>>>(
        (const __nv_bfloat16*)q, kcvar, (int8_t*)qiP, (float*)qs, (float*)threshold,
        (int8_t*)cen8, (float*)cens,
        T, Tp, H, NQ, tau, scale_log2, qs_b, qs_t, qs_h);
    prep_k<<<dim3(NTB, B * H), HD, 0, stream>>>(
        (const __nv_bfloat16*)k, kmean, (int8_t*)kiP, (float2*)ksb,
        (const float*)key_bias, T, Tp, H, NTB, ks_b, ks_t, ks_h);
}

extern "C" size_t sol_preprocess_scratch_bytes(int B, int H, int NPAD) {
    return ((size_t)B * H * NPAD * HEAD_DIM + (size_t)2 * B * H * HEAD_DIM) * sizeof(float);
}
