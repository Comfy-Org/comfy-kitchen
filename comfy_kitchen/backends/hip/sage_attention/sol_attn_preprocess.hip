// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Sol-Attn preprocess: post-rope q/k/v -> the workspace carriers
//   qi   [B,T,H,D] int8                       qs   [B,T,H] f32
//   ki   [B*H,Tp,D] int8                      ksb  [B*H,Tp] float2 = (ks, bias)
//   vT   [B*H][block][D][64] int8 (vt_at)     vsc  [B*H,D] f32  -- sol_attn_vtranspose.hip
//   kci  [B*H,NPAD,D] int8                    kcs  [B*H,NPAD] f32
//   vcT  [B*H,D,NPAD] bf16                    threshold [B*H,NQ] f32
//   cen8 [B*H,NQ,D] int8                      cens [B*H,NQ] f32
// Q-side carriers are indexed by T, K/V-side by Tp. Every carrier is in natural
// channel and key order (see sol_common.h). Inputs are read through explicit
// strides with a contiguous last dim. K's mean and V's scale are global
// reductions, hence the separate passes. `launch_sol_finish` is the chunked
// producer's second half. `ord` ([B*H, T] int32, or null): the carriers are built
// in a per-head row order, position t of head bh holding source row ord[bh*T + t]
// of q, k and v.
#include <hip/hip_runtime.h>

#include <cstdint>

#include "sol_common.h"

namespace comfy::hip_backend::sol {
namespace {

constexpr int HD = kHeadDim, BLK = kBlock;

// Scratch: pooled K block means (sums in the producer path), kmean, kcvar, the mean query.
struct Scratch {
    float* kc;
    float* kmean;
    float* kcvar;
    float* anchor;
};

inline Scratch carve_scratch(void* scratch, int B, int H, int NPAD) {
    Scratch s;
    s.kc = static_cast<float*>(scratch);
    s.kmean = s.kc + static_cast<size_t>(B) * H * NPAD * HD;
    s.kcvar = s.kmean + static_cast<size_t>(B) * H * HD;
    s.anchor = s.kcvar + static_cast<size_t>(B) * H * HD;
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
    int64_t vh, const int32_t* __restrict__ ord) {
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
        const int64_t row = ord ? ord[static_cast<size_t>(bh) * T + t0 + i] : t0 + i;
        const int64_t koff = batch * sb + row * st + head * sh + d;
        const int64_t voff = batch * vb + row * vt + head * vh + d;
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

// ---- per-row summary tail: each key block as one virtual key of the exact kernel ----
// The mean query of each head, from prep_q's block means.
__global__ __launch_bounds__(HD) void prep_anchor(const float* __restrict__ qmean,
                                                  float* __restrict__ anchor, int T, int NQ) {
    const int bh = blockIdx.x, d = threadIdx.x;
    float sum = 0.f;
    for (int n = 0; n < NQ; ++n) {
        sum = fmaf(qmean[(static_cast<size_t>(bh) * NQ + n) * HD + d],
                   static_cast<float>(imin(BLK, T - n * BLK)), sum);
    }
    anchor[static_cast<size_t>(bh) * HD + d] = sum / static_cast<float>(T);
}

// One summary key per block, from the int8 carriers: with w = softmax(a . k_i), a the
// mean query, AK = sum w_i k_i, AV = sum w_i v_i, log2 mass = log2 sum 2^(a . k_i) - a . AK
// (the block's log-sum-exp to first order around a). Summary keys [B*H, NPAD, D], their
// (ks, log2 mass) [B*H, NPAD] and V [B*H, D, NPAD], in the exact kernel's units. One
// workgroup per block, thread == channel.
__global__ __launch_bounds__(HD) void prep_summary(
    const int8_t* __restrict__ ki, const float2* __restrict__ ksb, const int8_t* __restrict__ vT,
    const float* __restrict__ anchor, int8_t* __restrict__ ski, float2* __restrict__ sksb,
    int8_t* __restrict__ svT, int T, int Tp, int NTB, int NPAD, float log2s) {
    __shared__ __attribute__((aligned(16))) int8_t sK[BLK * HD];
    __shared__ float sks[BLK];
    __shared__ float sw[BLK];   // score, then weight
    __shared__ float sred[HD];
    const int n = blockIdx.x, bh = blockIdx.y, d = threadIdx.x;
    const size_t krow = (static_cast<size_t>(bh) * NPAD + n) * HD;
    const size_t vcol = (static_cast<size_t>(bh) * HD + d) * NPAD + n;
    if (n >= NTB) {
        ski[krow + d] = 0;
        svT[vcol] = 0;
        if (d == 0) sksb[static_cast<size_t>(bh) * NPAD + n] = make_float2(0.f, kNeg);
        return;
    }
    const int len = imin(BLK, T - n * BLK);
    const size_t row0 = static_cast<size_t>(bh) * Tp + static_cast<size_t>(n) * BLK;
    for (int idx = d; idx < BLK * (HD / 16); idx += HD) {
        *reinterpret_cast<uint4*>(sK + idx * 16) =
            *reinterpret_cast<const uint4*>(ki + row0 * HD + idx * 16);
    }
    if (d < BLK) sks[d] = ksb[row0 + d].x;
    __syncthreads();
    const float* a = anchor + static_cast<size_t>(bh) * HD;
    // row scores: wave w takes rows w, w + 4, ..; lane l channels l, l + 32, l + 64, l + 96
    const int wave = d >> 5, lane = d & 31;
    for (int p = wave; p < BLK; p += HD / 32) {
        float dot = 0.f;
#pragma unroll
        for (int ch = lane; ch < HD; ch += 32) {
            dot = fmaf(a[ch], static_cast<float>(sK[p * HD + ch]), dot);
        }
#pragma unroll
        for (int off = 16; off; off >>= 1) dot += __shfl_xor(dot, off, 32);
        if (lane == 0) sw[p] = dot * sks[p] * log2s;
    }
    __syncthreads();
    if (d == 0) {
        float m = kNeg, den = 0.f;
        for (int j = 0; j < len; ++j) m = fmaxf(m, sw[j]);
        for (int j = 0; j < BLK; ++j) {   // the dead rows of a ragged block carry no weight
            const float w = j < len ? exp2f(sw[j] - m) : 0.f;
            sw[j] = w;
            den += w;
        }
        sred[0] = m;
        sred[1] = den;
    }
    __syncthreads();
    const float wmax = sred[0], wden = sred[1];
    float sk = 0.f, sv = 0.f;
    for (int p = 0; p < BLK; ++p) sk = fmaf(sw[p] * sks[p], static_cast<float>(sK[p * HD + d]), sk);
    __attribute__((aligned(16))) int8_t vrow[BLK];
#pragma unroll
    for (int c16 = 0; c16 < BLK; c16 += 16) {
        *reinterpret_cast<uint4*>(vrow + c16) = *reinterpret_cast<const uint4*>(
            vT + static_cast<size_t>(bh) * HD * Tp + vt_at(n, d) + c16);
    }
#pragma unroll
    for (int j = 0; j < BLK; ++j) sv = fmaf(sw[j], static_cast<float>(vrow[j]), sv);
    sk /= wden;
    sv /= wden;
    __syncthreads();   // every thread has read (wmax, wden) before the reductions reuse sred
    const float adot = block_sum128(a[d] * sk, sred);
    const float sc = fmaxf(block_max128(fabsf(sk), sred) / 127.0f, 1e-12f);
    ski[krow + d] = q8(sk, 1.f / sc);
    if (d == 0) {
        sksb[static_cast<size_t>(bh) * NPAD + n] =
            make_float2(sc, wmax + log2f(wden) - adot * log2s);
    }
    svT[vcol] = q8(sv, 1.f);
}

// ---- producer path under a row order: the staged raster carriers regrouped per head ----
// One workgroup per block: its 64 rows go through LDS in 16-byte chunks, then thread ==
// channel pools them. q: the rows stay in source order (the exact kernel reads them
// through ord); the block's centroid as prep_q leaves it.
__global__ __launch_bounds__(HD) void reorder_q(const int8_t* __restrict__ qi,
                                                const float* __restrict__ qs,
                                                int8_t* __restrict__ cen8,
                                                float* __restrict__ cens,
                                                float* __restrict__ qmean,
                                                const int32_t* __restrict__ ord, int T, int H,
                                                int NQ) {
    __shared__ __attribute__((aligned(16))) int8_t sR[BLK * HD];
    __shared__ float ssc[BLK];
    __shared__ float sred[HD];
    const int qb = blockIdx.x, bh = blockIdx.y, d = threadIdx.x;
    const int batch = bh / H, head = bh % H;
    const int t0 = qb * BLK, len = imin(BLK, T - t0);
    const int32_t* rows = ord + static_cast<size_t>(bh) * T + t0;
    for (int idx = d; idx < BLK * (HD / 16); idx += HD) {
        const int i = idx / (HD / 16), c16 = (idx % (HD / 16)) * 16;
        uint4 val = make_uint4(0u, 0u, 0u, 0u);
        if (i < len) {
            val = *reinterpret_cast<const uint4*>(
                qi + ((static_cast<size_t>(batch) * T + rows[i]) * H + head) * HD + c16);
        }
        *reinterpret_cast<uint4*>(sR + i * HD + c16) = val;
    }
    if (d < len) ssc[d] = qs[(static_cast<size_t>(batch) * T + rows[d]) * H + head];
    __syncthreads();
    float sum = 0.f;
    for (int i = 0; i < len; ++i) sum = fmaf(static_cast<float>(sR[i * HD + d]), ssc[i], sum);
    const float mean = sum / static_cast<float>(len);
    const size_t qrow = static_cast<size_t>(bh) * NQ + qb;
    qmean[qrow * HD + d] = mean;
    const float csc = fmaxf(block_max128(fabsf(mean), sred) / 127.0f, 1e-8f);
    cen8[qrow * HD + d] = q8(mean, 1.f / csc);
    if (d == 0) cens[qrow] = csc;
}

// k rows + (ks, bias) and the block's K sum.
__global__ __launch_bounds__(HD) void reorder_k(const int8_t* __restrict__ rk,
                                                const float2* __restrict__ rksb,
                                                int8_t* __restrict__ ki, float2* __restrict__ ksb,
                                                float* __restrict__ kc,
                                                const float* __restrict__ kmean,
                                                const int32_t* __restrict__ ord, int T, int Tp,
                                                int NPAD) {
    __shared__ __attribute__((aligned(16))) int8_t sR[BLK * HD];
    __shared__ float ssc[BLK];
    const int n = blockIdx.x, bh = blockIdx.y, d = threadIdx.x;
    const int t0 = n * BLK, len = imin(BLK, T - t0);
    const int32_t* rows = ord + static_cast<size_t>(bh) * T + t0;
    for (int idx = d; idx < BLK * (HD / 16); idx += HD) {
        const int p = idx / (HD / 16), c16 = (idx % (HD / 16)) * 16;
        uint4 val = make_uint4(0u, 0u, 0u, 0u);   // dead rows, as quant_k_rows leaves them
        if (p < len) {
            val = *reinterpret_cast<const uint4*>(rk + (static_cast<size_t>(bh) * Tp + rows[p]) * HD + c16);
        }
        *reinterpret_cast<uint4*>(ki + (static_cast<size_t>(bh) * Tp + t0 + p) * HD + c16) = val;
        *reinterpret_cast<uint4*>(sR + p * HD + c16) = val;
    }
    if (d < BLK) {
        float2 sb = make_float2(0.f, kNeg);
        if (d < len) sb = rksb[static_cast<size_t>(bh) * Tp + rows[d]];
        ksb[static_cast<size_t>(bh) * Tp + t0 + d] = sb;
        ssc[d] = sb.x;
    }
    __syncthreads();
    float sum = 0.f;
    for (int p = 0; p < BLK; ++p) sum = fmaf(static_cast<float>(sR[p * HD + d]), ssc[p], sum);
    kc[(static_cast<size_t>(bh) * NPAD + n) * HD + d] =
        sum + static_cast<float>(len) * kmean[static_cast<size_t>(bh) * HD + d];
}

// V rows into the transposed layout and the block's V sum. vfac: the staged rows were
// quantized with one scale per producer chunk of vfac_rows rows; vfac [chunks, B*H, HD]
// = that scale / vsc brings them to vsc here.
__global__ __launch_bounds__(HD) void reorder_v(const int8_t* __restrict__ rv,
                                                int8_t* __restrict__ vT, __bf16* __restrict__ vcT,
                                                const float* __restrict__ vsc,
                                                const int32_t* __restrict__ ord,
                                                const float* __restrict__ vfac, int vfac_rows,
                                                int T, int Tp, int NPAD) {
    __shared__ __attribute__((aligned(16))) int8_t sR[BLK * HD];
    const int n = blockIdx.x, bh = blockIdx.y, d = threadIdx.x;
    const int t0 = n * BLK, len = imin(BLK, T - t0);
    const int32_t* rows = ord + static_cast<size_t>(bh) * T + t0;
    for (int idx = d; idx < BLK * (HD / 16); idx += HD) {
        const int j = idx / (HD / 16), c16 = (idx % (HD / 16)) * 16;
        uint4 val = make_uint4(0u, 0u, 0u, 0u);
        if (j < len) {
            val = *reinterpret_cast<const uint4*>(rv + (static_cast<size_t>(bh) * Tp + rows[j]) * HD + c16);
        }
        *reinterpret_cast<uint4*>(sR + j * HD + c16) = val;
    }
    __syncthreads();
    __attribute__((aligned(16))) int8_t col[BLK];
    int sum = 0;
#pragma unroll
    for (int j = 0; j < BLK; ++j) {
        int8_t x = sR[j * HD + d];
        if (vfac && j < len) {
            x = static_cast<int8_t>(static_cast<int>(rintf(
                static_cast<float>(x) *
                vfac[(static_cast<size_t>(rows[j] / vfac_rows) * gridDim.y + bh) * HD + d])));
        }
        col[j] = x;
        sum += x;
    }
    int8_t* dst = vT + static_cast<size_t>(bh) * HD * Tp + vt_at(n, d);
#pragma unroll
    for (int c16 = 0; c16 < BLK; c16 += 16) {
        *reinterpret_cast<uint4*>(dst + c16) = *reinterpret_cast<const uint4*>(col + c16);
    }
    vcT[(static_cast<size_t>(bh) * HD + d) * NPAD + n] =
        static_cast<__bf16>(static_cast<float>(sum) * vsc[static_cast<size_t>(bh) * HD + d]);
}

// ---- row-order planner: features = unit rows of (k8 ks + kmean) W, from the staged int8 keys ----
// One wave per block, scored like the exact kernel scores queries: W's 16 directions
// (dir8 [B*H, 16, HD] int8, scales dsc) are the B operand, so lane % 16 holds a direction
// and eight keys per tile. doff [B*H, 16] is the mean key's features. feat: [B*H, rows1, 16] bf16.
#if defined(COMFY_MMA_GFX11)
constexpr int kFeatPad = 16;
#else
constexpr int kFeatPad = 8;
#endif
constexpr int kFeatStride = HD + kFeatPad;

__global__ __launch_bounds__(32) void group_features(const int8_t* __restrict__ ki,
                                                     const float2* __restrict__ ksb,
                                                     const int8_t* __restrict__ dir8,
                                                     const float* __restrict__ dsc,
                                                     const float* __restrict__ doff,
                                                     __bf16* __restrict__ feat, int Tp, int b0,
                                                     int rows1) {
    __shared__ __attribute__((aligned(16))) int8_t sK[BLK * kFeatStride];
    const int lane = threadIdx.x, r = frag_row(lane), bh = blockIdx.y;
    const size_t row0 = static_cast<size_t>(bh) * Tp + static_cast<size_t>(b0 + blockIdx.x) * BLK;
    for (int idx = lane; idx < BLK * (HD / 8); idx += 32) {
        const int p = idx / (HD / 8), c8 = (idx % (HD / 8)) * 8;
        *reinterpret_cast<uint2*>(sK + p * kFeatStride + c8) =
            *reinterpret_cast<const uint2*>(ki + (row0 + p) * HD + c8);
    }
    typename MmaInt8::Frag dfrag[HD / 16];
    const int8_t* drow = dir8 + (static_cast<size_t>(bh) * 16 + r) * HD;
#pragma unroll
    for (int dt = 0; dt < HD / 16; ++dt) dfrag[dt] = MmaInt8::load_aligned(drow, 0, dt * 16, 0, lane);
    const float sc = dsc[static_cast<size_t>(bh) * 16 + r];
    const float off = doff[static_cast<size_t>(bh) * 16 + r];
    __syncthreads();
#pragma unroll
    for (int kt = 0; kt < BLK / 16; ++kt) {
        typename MmaInt8::Acc acc = MmaInt8::zero();
#pragma unroll
        for (int dt = 0; dt < HD / 16; ++dt) {
            acc = MmaInt8::mma(MmaInt8::load_aligned(sK, kt * 16 + r, dt * 16, kFeatStride, lane),
                               dfrag[dt], acc);
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int key = kt * 16 + acc_row(lane, e);
            const float f = fmaf(static_cast<float>(acc[e]) * sc, ksb[row0 + key].x, off);
            float nrm = f * f;
            // a key's 16 features sit on the 16 lanes of its half-wave
#pragma unroll
            for (int o = 1; o < 16; o <<= 1) nrm += __shfl_xor(nrm, o, 32);
            feat[(static_cast<size_t>(bh) * rows1 + static_cast<size_t>(blockIdx.x) * BLK + key) * 16 + r] =
                static_cast<__bf16>(f * rsqrtf(fmaxf(nrm, 1e-24f)));
        }
    }
}

// ---- row-order planner: the last levels of the grouping tree, one node per workgroup ----
// feat [B*H, n + 1, 16] bf16 unit rows; perm [B*H, total] row per position, updated in
// place: a node per `size` (<= kLeafRows) positions, of which the first n hold rows. Four
// levels of aligned windows (1024 .. 128 slots; the node's rows stay a prefix, padding sorts
// last): a wave per window takes its rows' covariance (exact integer sums) and top direction
// (power iteration), every window is sorted along its direction, and its two halves are the
// next level's windows.
constexpr int kLeafRows = 1024, kLeafIters = 16, kLeafThreads = kLeafRows / 2;

__global__ __launch_bounds__(kLeafThreads) void group_leaves(const __bf16* __restrict__ feat,
                                                             int32_t* __restrict__ perm, int n,
                                                             int size, int total) {
    __shared__ int8_t sF[kLeafRows * 16];      // the node's rows x127 by slot; slot 1023 is the padding's
    __shared__ uint32_t sKey[kLeafRows];       // score | slot per position
    __shared__ int32_t sId[kLeafRows];         // row of each slot
    __shared__ int32_t sG[8][16 * 16 + 16];    // per wave: its window's second moments, then sums
    __shared__ float sV[8 * 16];               // split direction per window
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int p0 = blockIdx.x * size, m = imin(n - p0, size);
    if (m <= BLK || m > kLeafRows) return;
    int32_t* out = perm + static_cast<size_t>(blockIdx.y) * total + p0;
    const __bf16* fr = feat + static_cast<size_t>(blockIdx.y) * (n + 1) * 16;
    for (int j = tid; j < kLeafRows; j += kLeafThreads) {
        sKey[j] = j < m ? static_cast<uint32_t>(j) : 0xffffffffu;
        if (j < m) sId[j] = out[j];
    }
    if (tid < 16) sF[1023 * 16 + tid] = 0;
    __syncthreads();
    for (int e = tid; e < m * 16; e += kLeafThreads) {
        const float x = static_cast<float>(fr[static_cast<size_t>(sId[e >> 4]) * 16 + (e & 15)]);
        sF[e] = static_cast<int8_t>(static_cast<int>(rintf(fminf(fmaxf(x, -1.f), 1.f) * 127.f)));
    }
    __syncthreads();
    for (int W = kLeafRows, sh = 10; W > BLK; W >>= 1, --sh) {
        const int a = wave * W, cnt = imin(m - a, W);   // this wave's window and its rows
        const bool split = wave < 8 && cnt > BLK;
        if (split) {
            // lane: second moments of feature row lane / 2, columns 8 * (lane % 2) .. +7
            const int gr = lane >> 1, gc0 = (lane & 1) * 8;
            int32_t acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            int32_t s = 0;
            for (int i = a; i < a + cnt; ++i) {
                const int8_t* f = sF + (sKey[i] & 1023u) * 16;
                const int32_t x = f[gr];
#pragma unroll
                for (int c = 0; c < 8; ++c) acc[c] += x * static_cast<int32_t>(f[gc0 + c]);
                if (lane < 16) s += f[lane];
            }
            int32_t* G = sG[wave];
#pragma unroll
            for (int c = 0; c < 8; ++c) G[gr * 16 + gc0 + c] = acc[c];
            if (lane < 16) G[256 + lane] = s;
        }
        __syncthreads();
        if (split && lane < 16) {   // lane d holds row d of the covariance and v[d]
            const int32_t* G = sG[wave];
            const float md = static_cast<float>(G[256 + lane]) / static_cast<float>(cnt);
            float cv[16], v = 0.f;
#pragma unroll
            for (int e = 0; e < 16; ++e) {
                cv[e] = (static_cast<float>(G[lane * 16 + e]) - md * static_cast<float>(G[256 + e])) *
                        (1.f / (127.f * 127.f));
                v += cv[e];
            }
            for (int it = 0; it < kLeafIters; ++it) {
                float nv = 0.f;
#pragma unroll
                for (int e = 0; e < 16; ++e) nv = fmaf(cv[e], __shfl(v, e, 16), nv);
                float nrm = nv * nv;
#pragma unroll
                for (int o = 1; o < 16; o <<= 1) nrm += __shfl_xor(nrm, o, 16);
                v = nv * rsqrtf(fmaxf(nrm, 1e-30f));
            }
            sV[wave * 16 + lane] = v;
        }
        __syncthreads();
        for (int p = tid; p < m; p += kLeafThreads) {
            const int w = p >> sh;
            const uint32_t slot = sKey[p] & 1023u;
            float dot = 0.f;
            if (imin(m - w * W, W) > BLK) {
#pragma unroll
                for (int d = 0; d < 16; ++d) dot = fmaf(static_cast<float>(sF[slot * 16 + d]), sV[w * 16 + d], dot);
            }
            const int q = static_cast<int>(rintf(dot * (32000.f / 127.f))) + 32768;
            sKey[p] = static_cast<uint32_t>(imin(imax(q, 0), 65535)) << 10 | slot;
        }
        __syncthreads();
        for (int k = 2; k <= W; k <<= 1) {   // bitonic sort of every window, one pair per thread
            for (int j = k >> 1; j > 0; j >>= 1) {
                const int lo = ((tid & ~(j - 1)) << 1) | (tid & (j - 1)), hi = lo | j;
                const uint32_t x = sKey[lo], y = sKey[hi];
                if ((x > y) == (k == W || !(lo & k))) {
                    sKey[lo] = y;
                    sKey[hi] = x;
                }
                __syncthreads();
            }
        }
    }
    for (int p = tid; p < m; p += kLeafThreads) out[p] = sId[sKey[p] & 1023u];
}

// ---- pass 4: quantize Q, the centroid and the routing threshold from one tile ----
template <typename E>
__global__ __launch_bounds__(HD) void prep_q(const E* __restrict__ q,
                                             const float* __restrict__ kcvar,
                                             int8_t* __restrict__ qi, float* __restrict__ qs,
                                             float* __restrict__ thr, int8_t* __restrict__ cen8,
                                             float* __restrict__ cens,
                                             float* __restrict__ qmean,  // [B*H, NQ, HD] f32
                                             const int32_t* __restrict__ blen, int T, int H,
                                             int NQ, float tau, float log2s, int64_t sb,
                                             int64_t st, int64_t sh,
                                             const int32_t* __restrict__ ord) {
    __shared__ __attribute__((aligned(16))) E sQ[BLK * kLdTile];
    __shared__ __attribute__((aligned(16))) float sred[HD];
    const int qb = blockIdx.x, bh = blockIdx.y;
    const int batch = bh / H, head = bh % H;
    const int t0 = qb * BLK, len = block_len_of(blen, qb, T), nrows = imin(BLK, T - t0);

    if (ord) {
        stage_tile64_rows(sQ, q + batch * sb + head * sh, st, ord + static_cast<size_t>(bh) * T + t0,
                          len);
    } else {
        stage_tile64(sQ, q + batch * sb + static_cast<int64_t>(t0) * st + head * sh, st, len);
    }
    __syncthreads();
    const size_t tok0 = static_cast<size_t>(batch) * T + t0;
    quant_q_rows(sQ, len, nrows, qi + (tok0 * H + head) * HD, qs + tok0 * H + head, H);
    __syncthreads();
    const size_t qrow = static_cast<size_t>(bh) * NQ + qb;
    const float c = centroid_quant(sQ, len, sred, cen8 + qrow * HD, cens + qrow);
    qmean[qrow * HD + threadIdx.x] = c;
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
                                             int H, int64_t sb, int64_t st, int64_t sh,
                                             const int32_t* __restrict__ ord) {
    __shared__ __attribute__((aligned(16))) E sK[BLK * kLdTile];
    const int n = blockIdx.x, bh = blockIdx.y;
    const int batch = bh / H, head = bh % H;
    const int t0 = n * BLK, len = block_len_of(blen, n, T);

    if (ord) {
        stage_tile64_rows(sK, k + batch * sb + head * sh, st, ord + static_cast<size_t>(bh) * T + t0,
                          len);
    } else {
        stage_tile64(sK, k + batch * sb + static_cast<int64_t>(t0) * st + head * sh, st, len);
    }
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
                           const void* key_bias,   // [B, Tk] f32 in log2 units, or nullptr
                           const void* blen,       // [NTB] int32 per block, or nullptr; T == Tk then
                           const void* ord,        // [B*H, T] int32 row order, or nullptr; T == Tk then
                           // T queries, Tk keys
                           int B, int T, int Tk, int Tp, int H, int NTB, int NPAD, int NQ,
                           int64_t qs_b,
                           int64_t qs_t, int64_t qs_h, int64_t ks_b, int64_t ks_t, int64_t ks_h,
                           int64_t vs_b, int64_t vs_t, int64_t vs_h, float tau, float scale_log2,
                           hipStream_t stream) {
    const Scratch s = carve_scratch(scratch, B, H, NPAD);

    // pass 1 accumulates V's |max| into vsc by atomicMax
    sol_check(hipMemsetAsync(vsc, 0, static_cast<size_t>(B) * H * HD * sizeof(float), stream),
              "clearing the V absmax accumulator");
    prep_reduce_kv<E><<<dim3(NPAD, B * H), HD, 0, stream>>>(
        static_cast<const E*>(k), static_cast<const E*>(v), s.kc,
        static_cast<__bf16*>(vcT), static_cast<float*>(vsc), static_cast<const int32_t*>(blen), Tk,
        H, NTB, NPAD, ks_b, ks_t, ks_h, vs_b, vs_t, vs_h, static_cast<const int32_t*>(ord));
    prep_pooled_stats<<<B * H, HD, 0, stream>>>(s.kc, s.kmean, static_cast<const float*>(vsc),
                                                static_cast<float*>(vsc), s.kcvar, NTB, NPAD);
    prep_pooled_quant<<<dim3(NPAD, B * H), HD, 0, stream>>>(
        s.kc, s.kmean, static_cast<int8_t*>(kci), static_cast<float*>(kcs), NTB, NPAD);
    prep_q<E><<<dim3(NQ, B * H), HD, 0, stream>>>(
        static_cast<const E*>(q), s.kcvar, static_cast<int8_t*>(qi), static_cast<float*>(qs),
        static_cast<float*>(threshold), static_cast<int8_t*>(cen8), static_cast<float*>(cens),
        static_cast<float*>(qmean), static_cast<const int32_t*>(blen), T, H, NQ, tau,
        scale_log2, qs_b, qs_t, qs_h, static_cast<const int32_t*>(ord));
    prep_k<E><<<dim3(NTB, B * H), HD, 0, stream>>>(
        static_cast<const E*>(k), s.kmean, static_cast<int8_t*>(ki),
        static_cast<float2*>(ksb), static_cast<const float*>(key_bias),
        static_cast<const int32_t*>(blen), Tk, Tp, H, ks_b, ks_t, ks_h,
        static_cast<const int32_t*>(ord));
}

void launch_sol_preprocess(const void* q, const void* k, const void* v, void* qi, void* qs,
                           void* ki, void* ksb, void* kci, void* kcs, void* vcT, void* threshold,
                           void* cen8, void* cens, void* vsc, void* qmean, void* scratch,
                           const void* key_bias, const void* blen, const void* ord, int B, int T,
                           int Tk, int Tp,
                           int H, int NTB, int NPAD, int NQ, int64_t qs_b, int64_t qs_t,
                           int64_t qs_h,
                           int64_t ks_b, int64_t ks_t, int64_t ks_h, int64_t vs_b, int64_t vs_t,
                           int64_t vs_h, float tau, float scale_log2, int elem,
                           hipStream_t stream) {
    auto fn = elem == kSolFp16 ? preprocess_launch<_Float16> : preprocess_launch<__bf16>;
    fn(q, k, v, qi, qs, ki, ksb, kci, kcs, vcT, threshold, cen8, cens, vsc, qmean, scratch,
       key_bias, blen, ord, B, T, Tk, Tp, H, NTB, NPAD, NQ, qs_b, qs_t, qs_h, ks_b, ks_t, ks_h, vs_b,
       vs_t, vs_h, tau, scale_log2, stream);
}

// Needs the int8 carriers and the query block means; q and k of one length T.
void launch_sol_summary(const void* ki, const void* ksb, const void* vT, const void* qmean,
                        void* scratch, void* ski, void* sksb, void* svT, int B, int T, int Tp,
                        int H, int NTB, int NPAD, int NQ, float scale_log2, hipStream_t stream) {
    const Scratch s = carve_scratch(scratch, B, H, NPAD);
    prep_anchor<<<B * H, HD, 0, stream>>>(static_cast<const float*>(qmean), s.anchor, T, NQ);
    prep_summary<<<dim3(NPAD, B * H), HD, 0, stream>>>(
        static_cast<const int8_t*>(ki), static_cast<const float2*>(ksb),
        static_cast<const int8_t*>(vT), s.anchor, static_cast<int8_t*>(ski),
        static_cast<float2*>(sksb), static_cast<int8_t*>(svT), T, Tp, NTB, NPAD, scale_log2);
}

void launch_sol_group_features(const void* ki, const void* ksb, const void* dir8, const void* dsc,
                               const void* doff, void* feat, int BH, int Tp, int b0, int nblocks,
                               int rows1, hipStream_t stream) {
    group_features<<<dim3(nblocks, BH), 32, 0, stream>>>(
        static_cast<const int8_t*>(ki), static_cast<const float2*>(ksb),
        static_cast<const int8_t*>(dir8), static_cast<const float*>(dsc),
        static_cast<const float*>(doff), static_cast<__bf16*>(feat), Tp, b0, rows1);
}

// One node per workgroup, one thread per pair of slots.
void launch_sol_group_leaves(const void* feat, void* perm, int bh, int n, int size, int total,
                             hipStream_t stream) {
    group_leaves<<<dim3(total / size, bh), kLeafThreads, 0, stream>>>(
        static_cast<const __bf16*>(feat), static_cast<int32_t*>(perm), n, size, total);
}

// Staged raster k / v carriers (r*) -> the carriers and pooled statistics of the reordered
// blocks; q carriers (raster) -> the reordered blocks' centroids.
void launch_sol_reorder(const void* qi, const void* qs, const void* rki, const void* rksb,
                        const void* rvRow, void* ki, void* ksb, void* vT, void* vcT,
                        void* scratch, void* cen8, void* cens, void* qmean, const void* ord,
                        const void* kmean, const void* vscale, const void* vfac, int vfac_rows,
                        int B, int T, int Tp, int H, int NTB, int NPAD, int NQ,
                        hipStream_t stream) {
    const Scratch s = carve_scratch(scratch, B, H, NPAD);
    reorder_q<<<dim3(NQ, B * H), HD, 0, stream>>>(
        static_cast<const int8_t*>(qi), static_cast<const float*>(qs), static_cast<int8_t*>(cen8),
        static_cast<float*>(cens), static_cast<float*>(qmean), static_cast<const int32_t*>(ord),
        T, H, NQ);
    reorder_k<<<dim3(NTB, B * H), HD, 0, stream>>>(
        static_cast<const int8_t*>(rki), static_cast<const float2*>(rksb),
        static_cast<int8_t*>(ki), static_cast<float2*>(ksb), s.kc,
        static_cast<const float*>(kmean), static_cast<const int32_t*>(ord), T, Tp, NPAD);
    reorder_v<<<dim3(NTB, B * H), HD, 0, stream>>>(
        static_cast<const int8_t*>(rvRow), static_cast<int8_t*>(vT), static_cast<__bf16*>(vcT),
        static_cast<const float*>(vscale), static_cast<const int32_t*>(ord),
        static_cast<const float*>(vfac), vfac_rows, T, Tp, NPAD);
}

size_t sol_preprocess_scratch_bytes(int B, int H, int NPAD) {
    return (static_cast<size_t>(B) * H * NPAD * kHeadDim +
            static_cast<size_t>(3) * B * H * kHeadDim) *
           sizeof(float);
}
