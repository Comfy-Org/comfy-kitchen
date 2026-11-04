// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Sol-Attn preprocess: post-rope q/k/v -> the workspace carriers
//   qi   [B,T,H,D] int8                       qs   [B,T,H] f32
//   ki   [B*H,Tp,D] int8                      ksb  [B*H,Tp] float2 = (ks, bias)
//   vT   [B*H,D,Tp] int8                      vsc  [B*H,D] f32  -- sol_attn_vtranspose.hip
//   kci  [B*H,NPAD,D] int8                    kcs  [B*H,NPAD] f32
//   vcT  [B*H,D,NPAD] bf16                    threshold [B*H,NQ] f32
//   cen8 [B*H,NQ,D] int8                      cens [B*H,NQ] f32
// Q-side carriers are indexed by T, K/V-side by Tp. Every carrier is in natural
// channel and key order (see sol_common.h). Inputs are read through explicit
// strides with a contiguous last dim. K's mean and V's scale are global
// reductions, hence the separate passes. `launch_sol_finish` is the chunked
// producer's second half.
#include <hip/hip_runtime.h>

#include <cstdint>

#include "sol_common.h"

namespace comfy::hip_backend::sol {
namespace {

constexpr int HD = kHeadDim, BLK = kBlock;

// Scratch: pooled K block means (sums in the producer path), kmean, kcvar.
struct Scratch {
    float* kc;
    float* kmean;
    float* kcvar;
};

inline Scratch carve_scratch(void* scratch, int B, int H, int NPAD) {
    Scratch s;
    s.kc = static_cast<float*>(scratch);
    s.kmean = s.kc + static_cast<size_t>(B) * H * NPAD * HD;
    s.kcvar = s.kmean + static_cast<size_t>(B) * H * HD;
    return s;
}

// routing threshold: tau sigma of the proxy row, from the centroid's proxy variance
__forceinline__ __device__ float thr_of(float var, float tau, float log2s) {
    return tau * sqrtf(var * log2s * log2s + 1e-6f);
}

// atomicMax over non-negative floats: the bit pattern orders the same way.
__forceinline__ __device__ void atomic_max_pos(float* dst, float v) {
    atomicMax(reinterpret_cast<unsigned int*>(dst), __float_as_uint(v));
}

// ---- pass 1: K/V block reductions plus V's per-channel |max| ----
// One block per (bh, pooled block), thread == channel.
template <typename E>
__global__ __launch_bounds__(HD) void prep_reduce_kv(
    const E* __restrict__ k, const E* __restrict__ v, float* __restrict__ kc,
    __bf16* __restrict__ vcT, float* __restrict__ vamax, const int32_t* __restrict__ blen, int T,
    int H, int NTB, int NPAD, int64_t sb, int64_t st, int64_t sh, int64_t vb, int64_t vt,
    int64_t vh) {
    const int n = blockIdx.x, bh = blockIdx.y;
    const int batch = bh / H, head = bh % H, d = threadIdx.x;
    const size_t o = (static_cast<size_t>(bh) * NPAD + n) * HD + d;
    if (n >= NTB) {
        kc[o] = 0.f;
        vcT[(static_cast<size_t>(bh) * HD + d) * NPAD + n] = static_cast<__bf16>(0.f);
        return;
    }

    const int t0 = n * BLK, len = block_len_of(blen, n, T);
    float sk = 0.f, sv = 0.f, av = 0.f;
    for (int i = 0; i < len; ++i) {
        const int64_t koff = batch * sb + static_cast<int64_t>(t0 + i) * st + head * sh + d;
        const int64_t voff = batch * vb + static_cast<int64_t>(t0 + i) * vt + head * vh + d;
        const float kk = static_cast<float>(k[koff]);
        const float vv = static_cast<float>(v[voff]);
        sk += kk;
        sv += vv;
        av = fmaxf(av, fabsf(vv));
    }
    kc[o] = sk / static_cast<float>(len);  // block MEAN of K
    vcT[(static_cast<size_t>(bh) * HD + d) * NPAD + n] = static_cast<__bf16>(sv);  // block SUM of V
    atomic_max_pos(&vamax[static_cast<size_t>(bh) * HD + d], av);
}

// ---- pass 2: reductions over the pooled tensors ----
// vamax (pass 1's |max|; null in the producer path, whose caller supplies the V
// scale) becomes the per-channel V scale; may alias vsc.
__global__ __launch_bounds__(HD) void prep_pooled_stats(const float* __restrict__ kc,
                                                        float* __restrict__ kmean,
                                                        const float* vamax, float* vsc,
                                                        float* __restrict__ kcvar, int NTB,
                                                        int NPAD) {
    const int bh = blockIdx.x, d = threadIdx.x;
    float sm = 0.f, ss = 0.f;
    for (int n = 0; n < NTB; ++n) {
        const float x = kc[(static_cast<size_t>(bh) * NPAD + n) * HD + d];
        sm += x;
        ss = fmaf(x, x, ss);
    }
    const float m = sm / static_cast<float>(NTB);
    kmean[static_cast<size_t>(bh) * HD + d] = m;
    if (vamax) {
        vsc[static_cast<size_t>(bh) * HD + d] =
            fmaxf(vamax[static_cast<size_t>(bh) * HD + d] / 127.0f, 1e-8f);
    }
    kcvar[static_cast<size_t>(bh) * HD + d] = fmaxf(ss / static_cast<float>(NTB) - m * m, 0.f);
}

// ---- pass 3: centre and quantize the pooled keys ----
__global__ __launch_bounds__(HD) void prep_pooled_quant(const float* __restrict__ kc,
                                                        const float* __restrict__ kmean,
                                                        int8_t* __restrict__ kci,
                                                        float* __restrict__ kcs, int NTB,
                                                        int NPAD) {
    __shared__ float sred[HD];
    const int n = blockIdx.x, bh = blockIdx.y, d = threadIdx.x;
    const size_t o = (static_cast<size_t>(bh) * NPAD + n) * HD + d;
    const bool live = n < NTB;
    const float x = live ? (kc[o] - kmean[static_cast<size_t>(bh) * HD + d]) : 0.f;
    const float sc = fmaxf(block_max128(fabsf(x), sred) / 127.0f, 1e-12f);
    if (d == 0) kcs[static_cast<size_t>(bh) * NPAD + n] = live ? sc : 0.f;
    kci[(static_cast<size_t>(bh) * NPAD + n) * HD + d] = live ? q8(x, 1.f / sc)
                                                              : static_cast<int8_t>(0);
}

// ---- pass 4: quantize Q, the centroid and the routing threshold from one tile ----
template <typename E>
__global__ __launch_bounds__(HD) void prep_q(const E* __restrict__ q,
                                             const float* __restrict__ kcvar,
                                             int8_t* __restrict__ qi, float* __restrict__ qs,
                                             float* __restrict__ thr, int8_t* __restrict__ cen8,
                                             float* __restrict__ cens,
                                             float* __restrict__ qmean,  // [B*H, NPAD, HD] f32
                                             const int32_t* __restrict__ blen, int T, int H,
                                             int NQ, int NPAD, float tau, float log2s, int64_t sb,
                                             int64_t st, int64_t sh) {
    __shared__ __attribute__((aligned(16))) E sQ[BLK * kLdTile];
    __shared__ __attribute__((aligned(16))) float sred[HD];
    const int qb = blockIdx.x, bh = blockIdx.y;
    const int batch = bh / H, head = bh % H;
    const int t0 = qb * BLK, len = block_len_of(blen, qb, T), nrows = imin(BLK, T - t0);

    stage_tile64(sQ, q + batch * sb + static_cast<int64_t>(t0) * st + head * sh, st, len);
    __syncthreads();
    const size_t tok0 = static_cast<size_t>(batch) * T + t0;
    quant_q_rows(sQ, len, nrows, qi + (tok0 * H + head) * HD, qs + tok0 * H + head, H);
    __syncthreads();
    const size_t qrow = static_cast<size_t>(bh) * NQ + qb;
    const float c = centroid_quant(sQ, len, sred, cen8 + qrow * HD, cens + qrow);
    qmean[(static_cast<size_t>(bh) * NPAD + qb) * HD + threadIdx.x] = c;
    __syncthreads();  // sred held the centroid bytes
    const float var = block_sum128(c * c * kcvar[static_cast<size_t>(bh) * HD + threadIdx.x], sred);
    if (threadIdx.x == 0) thr[qrow] = thr_of(var, tau, log2s);
}

// ---- pass 5: centre and quantize K ----
template <typename E>
__global__ __launch_bounds__(HD) void prep_k(const E* __restrict__ k,
                                             const float* __restrict__ kmean,
                                             int8_t* __restrict__ ki, float2* __restrict__ ksb,
                                             const float* __restrict__ kbias,  // [B, T] log2 units
                                             const int32_t* __restrict__ blen, int T, int Tp,
                                             int H, int64_t sb, int64_t st, int64_t sh) {
    __shared__ __attribute__((aligned(16))) E sK[BLK * kLdTile];
    const int n = blockIdx.x, bh = blockIdx.y;
    const int batch = bh / H, head = bh % H;
    const int t0 = n * BLK, len = block_len_of(blen, n, T);

    stage_tile64(sK, k + batch * sb + static_cast<int64_t>(t0) * st + head * sh, st, len);
    __syncthreads();
    const size_t dst0 = static_cast<size_t>(bh) * Tp + n * BLK;
    quant_k_rows(sK, len, kmean + static_cast<size_t>(bh) * HD,
                 kbias ? kbias + static_cast<size_t>(batch) * T + t0 : nullptr, ki + dst0 * HD,
                 ksb + dst0);
}

// ---- producer-path finish: pooled sums -> means, next-step kmean ----
__global__ __launch_bounds__(HD) void prep_sums_to_means(float* __restrict__ kc,
                                                         float* __restrict__ kmean_next,
                                                         const int32_t* __restrict__ blen, int T,
                                                         int NTB, int NPAD) {
    const int bh = blockIdx.x, d = threadIdx.x;
    float total = 0.f;
    int tokens = 0;
    for (int n = 0; n < NTB; ++n) {
        const size_t o = (static_cast<size_t>(bh) * NPAD + n) * HD + d;
        const float sum = kc[o];
        total += sum;
        const int len = block_len_of(blen, n, T);
        tokens += len;
        kc[o] = sum / static_cast<float>(len);
    }
    kmean_next[static_cast<size_t>(bh) * HD + d] = total / static_cast<float>(tokens);
}

// ---- producer-path threshold from the quantized centroid ----
__global__ __launch_bounds__(HD) void prep_thr_from_cen(const int8_t* __restrict__ cen8,
                                                        const float* __restrict__ cens,
                                                        const float* __restrict__ kcvar,
                                                        float* __restrict__ thr, int NQ, float tau,
                                                        float log2s) {
    __shared__ float sred[HD];
    const int qb = blockIdx.x, bh = blockIdx.y, d = threadIdx.x;
    const float c = static_cast<float>(cen8[(static_cast<size_t>(bh) * NQ + qb) * HD + d]) *
                    cens[static_cast<size_t>(bh) * NQ + qb];
    const float var = block_sum128(c * c * kcvar[static_cast<size_t>(bh) * HD + d], sred);
    if (d == 0) thr[static_cast<size_t>(bh) * NQ + qb] = thr_of(var, tau, log2s);
}

}  // namespace
}  // namespace comfy::hip_backend::sol

using namespace comfy::hip_backend::sol;

// Producer path: block K sums left in scratch -> means, pooled quant, threshold.
// The caller places the V scale it quantized with in the vsc slot.
void launch_sol_finish(void* scratch, void* kci, void* kcs, void* threshold, const void* cen8,
                       const void* cens, void* kmean_next, const void* blen, int B, int T, int H,
                       int NTB, int NPAD, int NQ, float tau, float scale_log2,
                       hipStream_t stream) {
    const Scratch s = carve_scratch(scratch, B, H, NPAD);
    prep_sums_to_means<<<B * H, HD, 0, stream>>>(s.kc, static_cast<float*>(kmean_next),
                                                 static_cast<const int32_t*>(blen), T, NTB, NPAD);
    prep_pooled_stats<<<B * H, HD, 0, stream>>>(s.kc, s.kmean, nullptr, nullptr, s.kcvar, NTB,
                                                NPAD);
    prep_pooled_quant<<<dim3(NPAD, B * H), HD, 0, stream>>>(
        s.kc, s.kmean, static_cast<int8_t*>(kci), static_cast<float*>(kcs), NTB, NPAD);
    prep_thr_from_cen<<<dim3(NQ, B * H), HD, 0, stream>>>(
        static_cast<const int8_t*>(cen8), static_cast<const float*>(cens), s.kcvar,
        static_cast<float*>(threshold), NQ, tau, scale_log2);
}

template <typename E>
static void preprocess_launch(const void* q, const void* k, const void* v, void* qi, void* qs,
                           void* ki, void* ksb, void* kci, void* kcs, void* vcT, void* threshold,
                           void* cen8, void* cens, void* vsc, void* qmean,
                           void* scratch,          // sol_preprocess_scratch_bytes
                           const void* key_bias,   // [B, T] f32 in log2 units, or nullptr
                           const void* blen,       // [NTB] int32 valid tokens per block, or nullptr
                           int B, int T, int Tp, int H, int NTB, int NPAD, int NQ, int64_t qs_b,
                           int64_t qs_t, int64_t qs_h, int64_t ks_b, int64_t ks_t, int64_t ks_h,
                           int64_t vs_b, int64_t vs_t, int64_t vs_h, float tau, float scale_log2,
                           hipStream_t stream) {
    const Scratch s = carve_scratch(scratch, B, H, NPAD);

    // pass 1 accumulates V's |max| into vsc by atomicMax
    sol_check(hipMemsetAsync(vsc, 0, static_cast<size_t>(B) * H * HD * sizeof(float), stream),
              "clearing the V absmax accumulator");
    prep_reduce_kv<E><<<dim3(NPAD, B * H), HD, 0, stream>>>(
        static_cast<const E*>(k), static_cast<const E*>(v), s.kc,
        static_cast<__bf16*>(vcT), static_cast<float*>(vsc), static_cast<const int32_t*>(blen), T,
        H, NTB, NPAD, ks_b, ks_t, ks_h, vs_b, vs_t, vs_h);
    prep_pooled_stats<<<B * H, HD, 0, stream>>>(s.kc, s.kmean, static_cast<const float*>(vsc),
                                                static_cast<float*>(vsc), s.kcvar, NTB, NPAD);
    prep_pooled_quant<<<dim3(NPAD, B * H), HD, 0, stream>>>(
        s.kc, s.kmean, static_cast<int8_t*>(kci), static_cast<float*>(kcs), NTB, NPAD);
    prep_q<E><<<dim3(NQ, B * H), HD, 0, stream>>>(
        static_cast<const E*>(q), s.kcvar, static_cast<int8_t*>(qi), static_cast<float*>(qs),
        static_cast<float*>(threshold), static_cast<int8_t*>(cen8), static_cast<float*>(cens),
        static_cast<float*>(qmean), static_cast<const int32_t*>(blen), T, H, NQ, NPAD, tau,
        scale_log2, qs_b, qs_t, qs_h);
    prep_k<E><<<dim3(NTB, B * H), HD, 0, stream>>>(
        static_cast<const E*>(k), s.kmean, static_cast<int8_t*>(ki),
        static_cast<float2*>(ksb), static_cast<const float*>(key_bias),
        static_cast<const int32_t*>(blen), T, Tp, H, ks_b, ks_t, ks_h);
}

void launch_sol_preprocess(const void* q, const void* k, const void* v, void* qi, void* qs,
                           void* ki, void* ksb, void* kci, void* kcs, void* vcT, void* threshold,
                           void* cen8, void* cens, void* vsc, void* qmean, void* scratch,
                           const void* key_bias, const void* blen, int B, int T, int Tp, int H,
                           int NTB, int NPAD, int NQ, int64_t qs_b, int64_t qs_t, int64_t qs_h,
                           int64_t ks_b, int64_t ks_t, int64_t ks_h, int64_t vs_b, int64_t vs_t,
                           int64_t vs_h, float tau, float scale_log2, int elem,
                           hipStream_t stream) {
    auto fn = elem == kSolFp16 ? preprocess_launch<_Float16> : preprocess_launch<__bf16>;
    fn(q, k, v, qi, qs, ki, ksb, kci, kcs, vcT, threshold, cen8, cens, vsc, qmean, scratch,
       key_bias, blen, B, T, Tp, H, NTB, NPAD, NQ, qs_b, qs_t, qs_h, ks_b, ks_t, ks_h, vs_b,
       vs_t, vs_h, tau, scale_log2, stream);
}

size_t sol_preprocess_scratch_bytes(int B, int H, int NPAD) {
    return (static_cast<size_t>(B) * H * NPAD * kHeadDim +
            static_cast<size_t>(2) * B * H * kHeadDim) *
           sizeof(float);
}
