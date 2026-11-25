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

// Sol-Attn preprocess: post-rope q/k/v -> the workspace carriers
//   qiP  [B,T,H,D] int8   (perm_d)            qs   [B,T,H] f32
//   kiP  [B*H,Tp,D] int8  (perm_key + perm_d) ksb  [B*H,Tp] float2 = (ks, bias)
//   vTi  [B*H][block][D][64] int8 (vt_at, perm_d on keys)    vsc  [B*H,D] f32  -- sol_attn_vtranspose.cu
//   kciP [B*H,NPAD,D] int8 (perm_d)           kcs  [B*H,NPAD] f32
//   vcT  [B*H,D,NPAD] bf16                    threshold [B*H,NQ] f32
//   cen8 [B*H,NQ,D] int8  (perm_d)            cens [B*H,NQ] f32
// Q-side carriers are indexed by T, K/V-side by Tp. Inputs are read through
// explicit strides (last dim contiguous). K's mean and V's scale are global
// reductions, hence the separate passes. `launch_sol_finish` is the chunked
// producer's second half (pooled sums -> means, pooled quant, threshold).
// `ord` ([B*H, T] int32, or null): the carriers are built in a per-head row
// order, position t of head bh holding source row ord[bh*T + t] of q, k and v.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>

#include "sol_layout.cuh"

namespace {
using namespace sol;

constexpr int HD = HEAD_DIM, BLK = BLOCK;

// Scratch: pooled K block means (sums in the producer path), kmean, kcvar, the mean query.
struct Scratch { float* kc; float* kmean; float* kcvar; float* anchor; };
inline Scratch carve_scratch(void* scratch, int B, int H, int NPAD) {
    Scratch s;
    s.kc = (float*)scratch;
    s.kmean = s.kc + (size_t)B * H * NPAD * HD;
    s.kcvar = s.kmean + (size_t)B * H * HD;
    s.anchor = s.kcvar + (size_t)B * H * HD;
    return s;
}

// routing threshold: tau sigma of the proxy row, from the centroid's proxy variance
__device__ __forceinline__ float thr_of(float var, float tau, float log2s) {
    return tau * sqrtf(var * log2s * log2s + 1e-6f);
}

// ---- pass 1: K/V block reductions + V's per-channel |max| ----
// One block per (bh, pooled block), thread == channel.
template <typename E>
__global__ void prep_reduce_kv(const E* __restrict__ k,
                               const E* __restrict__ v,
                               float* __restrict__ kc, __nv_bfloat16* __restrict__ vcT,
                               float* __restrict__ vamax, const int32_t* __restrict__ blen,
                               int T, int H, int NTB, int NPAD,
                               int64_t sb, int64_t st, int64_t sh,
                               int64_t vb, int64_t vt, int64_t vh,
                               const int32_t* __restrict__ ord) {
    const int n = blockIdx.x, bh = blockIdx.y;
    const int batch = bh / H, head = bh % H, d = threadIdx.x;
    const size_t o = ((size_t)bh * NPAD + n) * HD + d;
    if (n >= NTB) {
        kc[o] = 0.f;
        vcT[((size_t)bh * HD + d) * NPAD + n] = __float2bfloat16(0.f);
        return;
    }

    const int t0 = n * BLK, len = block_len_of(blen, n, T);
    float sk = 0.f, sv = 0.f, av = 0.f;
    for (int i = 0; i < len; ++i) {
        const int64_t row = ord ? ord[(size_t)bh * T + t0 + i] : t0 + i;
        const int64_t voff = batch * vb + row * vt + head * vh + d;
        const float kk = to_f32(k[batch * sb + row * st + head * sh + d]);
        const float vv = to_f32(v[voff]);
        sk += kk; sv += vv; av = fmaxf(av, fabsf(vv));
    }
    kc[o] = sk / (float)len;    // block MEAN of K
    vcT[((size_t)bh * HD + d) * NPAD + n] = __float2bfloat16(sv);   // block SUM of V
    atomicMax(reinterpret_cast<unsigned int*>(&vamax[(size_t)bh * HD + d]),
              __float_as_uint(av));   // av >= 0, so the bit pattern orders correctly
}

// ---- pass 2: reductions over the pooled tensors ----
// vamax (pass 1's |max|; null in the producer path, whose caller supplies the
// V scale) becomes the per-channel V scale; may alias vsc.
__global__ void prep_pooled_stats(const float* __restrict__ kc,
                                  float* __restrict__ kmean,
                                  const float* vamax, float* vsc,
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
    if (vamax) vsc[(size_t)bh * HD + d] = fmaxf(vamax[(size_t)bh * HD + d] / 127.0f, 1e-8f);
    kcvar[(size_t)bh * HD + d] = fmaxf(ss / (float)NTB - m * m, 0.f);
}

// ---- pass 3: centre + quantize the pooled keys ----
__global__ void prep_pooled_quant(const float* __restrict__ kc,
                                  const float* __restrict__ kmean,
                                  int8_t* __restrict__ kciP, float* __restrict__ kcs,
                                  int NTB, int NPAD) {
    __shared__ float sred[HD];
    const int n = blockIdx.x, bh = blockIdx.y, d = threadIdx.x;
    const size_t o = ((size_t)bh * NPAD + n) * HD + d;
    const bool live = n < NTB;
    const float x = live ? (kc[o] - kmean[(size_t)bh * HD + d]) : 0.f;
    const float sc = fmaxf(block_max128(fabsf(x), sred) / 127.0f, 1e-12f);
    if (d == 0) kcs[(size_t)bh * NPAD + n] = live ? sc : 0.f;
    kciP[((size_t)bh * NPAD + n) * HD + perm_d(d)] = live ? q8(x, 1.f / sc) : (int8_t)0;
}

// ---- per-row summary tail: each key block as one virtual key of the exact kernel ----
// The mean query of each head, from prep_q's block means.
__global__ void prep_anchor(const float* __restrict__ qmean, float* __restrict__ anchor, int T, int NQ) {
    const int bh = blockIdx.x, d = threadIdx.x;
    float sum = 0.f;
    for (int n = 0; n < NQ; ++n)
        sum = fmaf(qmean[((size_t)bh * NQ + n) * HD + d], (float)min(BLK, T - n * BLK), sum);
    anchor[(size_t)bh * HD + d] = sum / (float)T;
}

// One summary key per block, from the int8 carriers: with w = softmax(a . k_i), a the mean query,
// AK = sum w_i k_i, AV = sum w_i v_i, log2 mass = log2 sum 2^(a . k_i) - a . AK (the block's
// log-sum-exp to first order around a). Written in the exact kernel's layouts: slot p of a 64-group
// plays logical key perm_key(p). One CTA per block, thread == channel.
__global__ void prep_summary(const int8_t* __restrict__ kiP, const float2* __restrict__ ksb,
                             const int8_t* __restrict__ vTi, const float* __restrict__ anchor,
                             int8_t* __restrict__ skiP, float2* __restrict__ sksb, int8_t* __restrict__ svT,
                             int T, int Tp, int NTB, int NPAD, float log2s) {
    __shared__ __align__(16) int8_t sK[BLK * HD];   // the block's 64 key rows, as stored
    __shared__ float sks[BLK];                       // their scales
    __shared__ float sw[BLK];                        // per LOGICAL key: score, then weight
    __shared__ float sred[HD];
    const int n = blockIdx.x, bh = blockIdx.y, d = threadIdx.x;
    const size_t krow = ((size_t)bh * NPAD + n) * HD;
    const size_t vcol = (size_t)bh * HD * NPAD + vt_at(n >> 6, d) + perm_d(perm_key(n & 63));
    if (n >= NTB) {
        skiP[krow + d] = 0;
        svT[vcol] = 0;
        if (d == 0) sksb[(size_t)bh * NPAD + n] = make_float2(0.f, NEG);
        return;
    }
    const int len = min(BLK, T - n * BLK), c = perm_d(d);
    const size_t row0 = (size_t)bh * Tp + (size_t)n * BLK;
    for (int idx = d; idx < BLK * (HD / 16); idx += HD)
        *reinterpret_cast<uint4*>(sK + idx * 16) = *reinterpret_cast<const uint4*>(kiP + row0 * HD + idx * 16);
    if (d < BLK) sks[d] = ksb[row0 + d].x;
    __syncthreads();
    const float* a = anchor + (size_t)bh * HD;
    // row scores: warp w takes stored rows w, w + 4, ..; lane l channels l, l + 32, l + 64, l + 96
    const int warp = d >> 5, lane = d & 31;
    for (int p = warp; p < BLK; p += 4) {
        float dot = 0.f;
        #pragma unroll
        for (int ch = lane; ch < HD; ch += 32) dot = fmaf(a[ch], (float)sK[p * HD + perm_d(ch)], dot);
        #pragma unroll
        for (int off = 16; off; off >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off);
        if (lane == 0) sw[perm_key(p)] = dot * sks[p] * log2s;
    }
    __syncthreads();
    if (d == 0) {
        float m = NEG, den = 0.f;
        for (int j = 0; j < len; ++j) m = fmaxf(m, sw[j]);
        for (int j = 0; j < BLK; ++j) {   // the dead rows of a ragged block carry no weight
            const float w = j < len ? exp2f(sw[j] - m) : 0.f;
            sw[j] = w;
            den += w;
        }
        sred[0] = m; sred[1] = den;
    }
    __syncthreads();
    const float wmax = sred[0], wden = sred[1];
    float sk = 0.f, sv = 0.f;
    for (int p = 0; p < BLK; ++p)
        sk = fmaf(sw[perm_key(p)] * sks[p], (float)sK[p * HD + c], sk);
    __align__(16) int8_t vrow[BLK];   // channel d's 64 keys in four loads: byte loads touch the tile 64 times over
    #pragma unroll
    for (int c16 = 0; c16 < BLK; c16 += 16)
        *reinterpret_cast<uint4*>(vrow + c16) =
            *reinterpret_cast<const uint4*>(vTi + (size_t)bh * HD * Tp + vt_at(n, d) + c16);
    #pragma unroll
    for (int j = 0; j < BLK; ++j) sv = fmaf(sw[j], (float)vrow[perm_d(j)], sv);
    sk /= wden; sv /= wden;
    __syncthreads();   // every thread has read (wmax, wden) before the reductions reuse sred
    const float adot = block_sum128(a[d] * sk, sred);
    const float sc = fmaxf(block_max128(fabsf(sk), sred) / 127.0f, 1e-12f);
    skiP[krow + c] = q8(sk, 1.f / sc);
    if (d == 0) sksb[(size_t)bh * NPAD + n] = make_float2(sc, wmax + log2f(wden) - adot * log2s);
    svT[vcol] = q8(sv, 1.f);
}

// ---- producer path under a row order: the staged raster carriers regrouped per head ----
// One CTA per block: its 64 rows go through smem in 16 B chunks, then thread == channel pools them.
// q: the rows stay in source order (the exact kernel reads them through ord); the block's centroid as
// prep_q leaves it.
__global__ void reorder_q(const int8_t* __restrict__ qiP, const float* __restrict__ qs,
                          int8_t* __restrict__ cen8, float* __restrict__ cens, float* __restrict__ qmean,
                          const int32_t* __restrict__ ord, int T, int H, int NQ) {
    __shared__ __align__(16) int8_t sR[BLK * HD];
    __shared__ float ssc[BLK];
    __shared__ float sred[HD];
    const int qb = blockIdx.x, bh = blockIdx.y, d = threadIdx.x;
    const int batch = bh / H, head = bh % H;
    const int t0 = qb * BLK, len = min(BLK, T - t0);
    const int32_t* rows = ord + (size_t)bh * T + t0;
    for (int idx = d; idx < BLK * (HD / 16); idx += HD) {
        const int i = idx / (HD / 16), c16 = (idx % (HD / 16)) * 16;
        uint4 val = make_uint4(0u, 0u, 0u, 0u);
        if (i < len)
            val = *reinterpret_cast<const uint4*>(qiP + (((size_t)batch * T + rows[i]) * H + head) * HD + c16);
        *reinterpret_cast<uint4*>(sR + i * HD + c16) = val;
    }
    if (d < len) ssc[d] = qs[((size_t)batch * T + rows[d]) * H + head];
    __syncthreads();
    const int c = perm_d(d);
    float sum = 0.f;
    for (int i = 0; i < len; ++i) sum = fmaf((float)sR[i * HD + c], ssc[i], sum);
    const float mean = sum / (float)len;
    const size_t qrow = (size_t)bh * NQ + qb;
    qmean[qrow * HD + d] = mean;
    const float csc = fmaxf(block_max128(fabsf(mean), sred) / 127.0f, 1e-8f);
    cen8[qrow * HD + c] = q8(mean, 1.f / csc);
    if (d == 0) cens[qrow] = csc;
}

// k rows + (ks, bias) (stored row p of a block is its logical key perm_key(p)) and the block's K sum.
__global__ void reorder_k(const int8_t* __restrict__ rk, const float2* __restrict__ rksb,
                          int8_t* __restrict__ kiP, float2* __restrict__ ksb, float* __restrict__ kc,
                          const float* __restrict__ kmean, const int32_t* __restrict__ ord,
                          int T, int Tp, int NPAD) {
    __shared__ __align__(16) int8_t sR[BLK * HD];
    __shared__ float ssc[BLK];
    const int n = blockIdx.x, bh = blockIdx.y, d = threadIdx.x;
    const int t0 = n * BLK, len = min(BLK, T - t0);
    const int32_t* rows = ord + (size_t)bh * T + t0;
    for (int idx = d; idx < BLK * (HD / 16); idx += HD) {
        const int p = idx / (HD / 16), c16 = (idx % (HD / 16)) * 16, j = perm_key(p);
        uint4 val = make_uint4(0u, 0u, 0u, 0u);   // dead rows, as quant_k_rows leaves them
        if (j < len) {
            const int s = rows[j];
            val = *reinterpret_cast<const uint4*>(
                rk + ((size_t)bh * Tp + (s & ~63) + perm_key_inv(s & 63)) * HD + c16);
        }
        *reinterpret_cast<uint4*>(kiP + ((size_t)bh * Tp + t0 + p) * HD + c16) = val;
        *reinterpret_cast<uint4*>(sR + p * HD + c16) = val;
    }
    if (d < BLK) {
        const int j = perm_key(d);
        float2 sb = make_float2(0.f, NEG);
        if (j < len) {
            const int s = rows[j];
            sb = rksb[(size_t)bh * Tp + (s & ~63) + perm_key_inv(s & 63)];
        }
        ksb[(size_t)bh * Tp + t0 + d] = sb;
        ssc[d] = sb.x;
    }
    __syncthreads();
    const int c = perm_d(d);
    float sum = 0.f;
    for (int p = 0; p < BLK; ++p) sum = fmaf((float)sR[p * HD + c], ssc[p], sum);
    kc[((size_t)bh * NPAD + n) * HD + d] = sum + (float)len * kmean[(size_t)bh * HD + d];
}

// V rows into the transposed layout (logical key j at position perm_d(j)) and the block's V sum. vfac:
// the staged rows were quantized with one scale per producer chunk of vfac_rows rows; vfac [chunks, B*H,
// HD] = that scale / vsc brings them to vsc here.
__global__ void reorder_v(const int8_t* __restrict__ rv, int8_t* __restrict__ vTi,
                          __nv_bfloat16* __restrict__ vcT, const float* __restrict__ vsc,
                          const int32_t* __restrict__ ord, const float* __restrict__ vfac, int vfac_rows,
                          int T, int Tp, int NPAD) {
    __shared__ __align__(16) int8_t sR[BLK * HD];
    const int n = blockIdx.x, bh = blockIdx.y, d = threadIdx.x;
    const int t0 = n * BLK, len = min(BLK, T - t0);
    const int32_t* rows = ord + (size_t)bh * T + t0;
    for (int idx = d; idx < BLK * (HD / 16); idx += HD) {
        const int j = idx / (HD / 16), c16 = (idx % (HD / 16)) * 16;
        uint4 val = make_uint4(0u, 0u, 0u, 0u);
        if (j < len)
            val = *reinterpret_cast<const uint4*>(rv + ((size_t)bh * Tp + rows[j]) * HD + c16);
        *reinterpret_cast<uint4*>(sR + j * HD + c16) = val;
    }
    __syncthreads();
    __align__(16) int8_t col[BLK];   // channel d's 64 keys, built in registers
    int sum = 0;
    #pragma unroll
    for (int j = 0; j < BLK; ++j) {
        int8_t x = sR[j * HD + d];
        if (vfac && j < len)
            x = (int8_t)__float2int_rn((float)x * vfac[((size_t)(rows[j] / vfac_rows) * gridDim.y + bh) * HD + d]);
        col[perm_d(j)] = x;
        sum += x;
    }
    int8_t* dst = vTi + (size_t)bh * HD * Tp + vt_at(t0 / BLK, d);
    #pragma unroll
    for (int c16 = 0; c16 < BLK; c16 += 16)
        *reinterpret_cast<uint4*>(dst + c16) = *reinterpret_cast<const uint4*>(col + c16);
    vcT[((size_t)bh * HD + d) * NPAD + n] = __float2bfloat16((float)sum * vsc[(size_t)bh * HD + d]);
}

// ---- row-order planner: features = unit rows of (k8 ks + kmean) W, from the staged int8 keys ----
// One warp per block: W's 16 directions (dir8 [B*H, 16, HD] int8, scales dsc) take the queries' place
// in the score tile; doff [B*H, 16] is the mean key's features. feat: [B*H, rows1, 16] bf16.
__global__ void group_features(const int8_t* __restrict__ kiP, const float2* __restrict__ ksb,
                               const int8_t* __restrict__ dir8, const float* __restrict__ dsc,
                               const float* __restrict__ doff, __nv_bfloat16* __restrict__ feat,
                               int Tp, int b0, int rows1) {
#if SOL_SM80
    __shared__ __align__(16) int8_t sK[BLK * TILE_LDK];
    const int lane = threadIdx.x, g = lane >> 2, qd = lane & 3;
    const int bh = blockIdx.y;
    const size_t row0 = (size_t)bh * Tp + (size_t)(b0 + blockIdx.x) * BLK;
    for (int idx = lane; idx < BLK * (HD / 16); idx += 32) {
        const int p = idx / (HD / 16), c16 = idx % (HD / 16);
        *reinterpret_cast<uint4*>(sK + p * TILE_LDK + ((c16 ^ swz_k(p)) << 4)) =
            *reinterpret_cast<const uint4*>(kiP + (row0 + p) * HD + c16 * 16);
    }
    uint32_t qa[TILE_KC][4];   // directions g and g + 8, laid out as the exact kernel lays out two query rows
    const int8_t* d0 = dir8 + ((size_t)bh * 16 + g) * HD;
    #pragma unroll
    for (int kc = 0; kc < TILE_KC; ++kc) {
        const uint2 a0 = *reinterpret_cast<const uint2*>(d0 + kc * 32 + qd * 8);
        const uint2 a1 = *reinterpret_cast<const uint2*>(d0 + 8 * HD + kc * 32 + qd * 8);
        qa[kc][0] = a0.x; qa[kc][2] = a0.y;
        qa[kc][1] = a1.x; qa[kc][3] = a1.y;
    }
    const float sc[2] = {dsc[(size_t)bh * 16 + g], dsc[(size_t)bh * 16 + g + 8]};
    const float off[2] = {doff[(size_t)bh * 16 + g], doff[(size_t)bh * 16 + g + 8]};
    __syncthreads();
    int32_t s_acc[TILE_NKT][4];
    tile_qk(sK, qa, g, qd, s_acc);
    #pragma unroll
    for (int nt = 0; nt < TILE_NKT; ++nt) {
        const int c0 = nt * 8 + qd * 2;   // stored rows c0, c0 + 1 of the block
        const float4 kb4 = *reinterpret_cast<const float4*>(ksb + row0 + c0);
        float f[4], nrm[2] = {0.f, 0.f};
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            f[e] = fmaf((float)s_acc[nt][e] * sc[e >> 1], (e & 1) ? kb4.z : kb4.x, off[e >> 1]);
            nrm[e & 1] = fmaf(f[e], f[e], nrm[e & 1]);
        }
        #pragma unroll
        for (int o = 4; o < 32; o <<= 1) {   // a row's 16 features sit on the 8 lanes sharing qd
            nrm[0] += __shfl_xor_sync(0xffffffffu, nrm[0], o);
            nrm[1] += __shfl_xor_sync(0xffffffffu, nrm[1], o);
        }
        const float inv[2] = {rsqrtf(fmaxf(nrm[0], 1e-24f)), rsqrtf(fmaxf(nrm[1], 1e-24f))};
        __nv_bfloat16* dst = feat + ((size_t)bh * rows1 + (size_t)blockIdx.x * BLK + c0) * 16;
        #pragma unroll
        for (int e = 0; e < 4; ++e)
            dst[(e & 1) * 16 + g + 8 * (e >> 1)] = __float2bfloat16(f[e] * inv[e & 1]);
    }
#endif
}

// ---- row-order planner: the last levels of the grouping tree, one node per thread block ----
// feat [B*H, n + 1, 16] bf16 unit rows; perm [B*H, total] row per position, updated in place: a node per
// `size` (<= LEAF_ROWS) positions, of which the first n hold rows. Four levels of aligned
// windows (1024 .. 128 slots; the node's rows stay a prefix, padding sorts last): a warp per window takes
// its rows' covariance (int8 MMA) and top direction (power iteration), every window is sorted along its
// direction, and its two halves are the next level's windows.
constexpr int LEAF_ROWS = 1024, LEAF_ITERS = 16;

__global__ void group_leaves(const __nv_bfloat16* __restrict__ feat, int32_t* __restrict__ perm,
                             int n, int size, int total) {
#if SOL_SM80
    __shared__ int8_t sF[LEAF_ROWS * 16];      // the node's rows x127 by slot; slot 1023 is the padding's
    __shared__ uint32_t sKey[LEAF_ROWS];       // score | slot per position
    __shared__ int32_t sId[LEAF_ROWS];         // row of each slot
    __shared__ int32_t sG[8][16 * 16 + 16];    // per warp: its window's second moments, then sums
    __shared__ float sV[8 * 16];               // split direction per window
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, qd = lane & 3;
    const int p0 = blockIdx.x * size, m = min(n - p0, size);
    if (m <= BLK || m > LEAF_ROWS) return;
    int32_t* out = perm + (size_t)blockIdx.y * total + p0;
    const __nv_bfloat16* fr = feat + (size_t)blockIdx.y * (n + 1) * 16;
    for (int j = tid; j < LEAF_ROWS; j += LEAF_ROWS / 2) {
        sKey[j] = j < m ? (uint32_t)j : 0xffffffffu;
        if (j < m) sId[j] = out[j];
    }
    if (tid < 16) sF[1023 * 16 + tid] = 0;
    __syncthreads();
    for (int e = tid; e < m * 16; e += LEAF_ROWS / 2) {
        const float x = __bfloat162float(fr[(size_t)sId[e >> 4] * 16 + (e & 15)]);
        sF[e] = (int8_t)__float2int_rn(fminf(fmaxf(x, -1.f), 1.f) * 127.f);
    }
    __syncthreads();
    for (int W = LEAF_ROWS, sh = 10; W > BLK; W >>= 1, --sh) {
        const int a = warp * W, cnt = min(m - a, W);   // this warp's window and its rows
        if (warp < 8 && cnt > BLK) {
            int32_t c[3][4] = {};                      // columns 0-7, columns 8-15, the sums
            const uint32_t ones[2] = {g == 0 ? 0x01010101u : 0u, g == 0 ? 0x01010101u : 0u};
            for (int i0 = a; i0 < a + cnt; i0 += 32) { // A = 32 rows transposed, B = the same registers
                uint32_t fa[4] = {0, 0, 0, 0};
                #pragma unroll
                for (int j = 0; j < 16; ++j) {
                    const uint32_t slot = sKey[i0 + (j >> 3) * 16 + qd * 4 + (j & 3)] & 1023u;
                    fa[j >> 2] |= (uint32_t)(uint8_t)sF[slot * 16 + g + ((j >> 2) & 1) * 8] << (8 * (j & 3));
                }
                const uint32_t b0[2] = {fa[0], fa[2]}, b1[2] = {fa[1], fa[3]};
                mma_s8(c[0], fa, b0);
                mma_s8(c[1], fa, b1);
                mma_s8(c[2], fa, ones);
            }
            int32_t* G = sG[warp];
            #pragma unroll
            for (int x = 0; x < 8; ++x)
                G[(g + 8 * ((x >> 1) & 1)) * 16 + 8 * (x >> 2) + qd * 2 + (x & 1)] = c[x >> 2][x & 3];
            if (qd == 0) { G[256 + g] = c[2][0]; G[256 + g + 8] = c[2][2]; }
            __syncwarp();
            if (lane < 16) {                           // lane d holds row d of the covariance and v[d]
                const float md = (float)G[256 + lane] / (float)cnt;
                float cv[16], v = 0.f;
                #pragma unroll
                for (int e = 0; e < 16; ++e) {
                    cv[e] = ((float)G[lane * 16 + e] - md * (float)G[256 + e]) * (1.f / (127.f * 127.f));
                    v += cv[e];
                }
                for (int it = 0; it < LEAF_ITERS; ++it) {
                    float nv = 0.f;
                    #pragma unroll
                    for (int e = 0; e < 16; ++e) nv = fmaf(cv[e], __shfl_sync(0xffffu, v, e), nv);
                    float nrm = nv * nv;
                    #pragma unroll
                    for (int o = 1; o < 16; o <<= 1) nrm += __shfl_xor_sync(0xffffu, nrm, o);
                    v = nv * rsqrtf(fmaxf(nrm, 1e-30f));
                }
                sV[warp * 16 + lane] = v;
            }
        }
        __syncthreads();
        for (int p = tid; p < m; p += LEAF_ROWS / 2) {
            const int w = p >> sh;
            const uint32_t slot = sKey[p] & 1023u;
            float dot = 0.f;
            if (min(m - w * W, W) > BLK) {
                #pragma unroll
                for (int d = 0; d < 16; ++d) dot = fmaf((float)sF[slot * 16 + d], sV[w * 16 + d], dot);
            }
            sKey[p] = (uint32_t)min(max(__float2int_rn(dot * (32000.f / 127.f)) + 32768, 0), 65535) << 10 | slot;
        }
        __syncthreads();
        for (int k = 2; k <= W; k <<= 1)               // bitonic sort of every window, one pair per thread
            for (int j = k >> 1; j > 0; j >>= 1) {
                const int lo = ((tid & ~(j - 1)) << 1) | (tid & (j - 1)), hi = lo | j;
                const uint32_t x = sKey[lo], y = sKey[hi];
                if ((x > y) == (k == W || !(lo & k))) { sKey[lo] = y; sKey[hi] = x; }
                __syncthreads();
            }
    }
    for (int p = tid; p < m; p += LEAF_ROWS / 2) out[p] = sId[sKey[p] & 1023u];
#endif
}

// ---- pass 4: quantize Q, centroid, routing threshold from one staged tile ----
template <typename E>
__global__ void prep_q(const E* __restrict__ q, const float* __restrict__ kcvar,
                       int8_t* __restrict__ qiP, float* __restrict__ qs,
                       float* __restrict__ thr,
                       int8_t* __restrict__ cen8, float* __restrict__ cens,
                       float* __restrict__ qmean,   // [B*H, NQ, HD] f32 block means
                       const int32_t* __restrict__ blen,
                       int T, int H, int NQ, float tau, float log2s,
                       int64_t sb, int64_t st, int64_t sh,
                       const int32_t* __restrict__ ord) {
    __shared__ __align__(16) E sQ[BLK * LD_TILE];
    __shared__ __align__(16) float sred[HD];
    const int qb = blockIdx.x, bh = blockIdx.y;
    const int batch = bh / H, head = bh % H;
    const int t0 = qb * BLK, len = block_len_of(blen, qb, T), nrows = min(BLK, T - t0);

    if (ord) stage_tile64_rows(sQ, q + batch * sb + head * sh, st, ord + (size_t)bh * T + t0, len);
    else stage_tile64(sQ, q + batch * sb + (int64_t)t0 * st + head * sh, st, len);
    __syncthreads();
    const size_t tok0 = (size_t)batch * T + t0;
    quant_q_rows(sQ, len, nrows, qiP + (tok0 * H + head) * HD, qs + tok0 * H + head, H);
    __syncthreads();
    const size_t qrow = (size_t)bh * NQ + qb;
    const float c = centroid_quant(sQ, len, sred, cen8 + qrow * HD, cens + qrow);
    qmean[qrow * HD + threadIdx.x] = c;
    __syncthreads();                       // sred held the centroid bytes
    const float var = block_sum128(c * c * kcvar[(size_t)bh * HD + threadIdx.x], sred);
    if (threadIdx.x == 0) thr[qrow] = thr_of(var, tau, log2s);
}

// ---- pass 5: centre + quantize K into the permuted layout ----
template <typename E>
__global__ void prep_k(const E* __restrict__ k, const float* __restrict__ kmean,
                       int8_t* __restrict__ kiP, float2* __restrict__ ksb,
                       const float* __restrict__ kbias,   // [B, T] log2 units, or null
                       const int32_t* __restrict__ blen,
                       int T, int Tp, int H,
                       int64_t sb, int64_t st, int64_t sh,
                       const int32_t* __restrict__ ord) {
    __shared__ __align__(16) E sK[BLK * LD_TILE];
    const int n = blockIdx.x, bh = blockIdx.y;
    const int batch = bh / H, head = bh % H;
    const int t0 = n * BLK, len = block_len_of(blen, n, T);

    if (ord) stage_tile64_rows(sK, k + batch * sb + head * sh, st, ord + (size_t)bh * T + t0, len);
    else stage_tile64(sK, k + batch * sb + (int64_t)t0 * st + head * sh, st, len);
    __syncthreads();
    const size_t dst0 = (size_t)bh * Tp + n * BLK;
    quant_k_rows(sK, len, kmean + (size_t)bh * HD,
                 kbias ? kbias + (size_t)batch * T + t0 : nullptr,
                 kiP + dst0 * HD, ksb + dst0);
}

// ---- producer-path finish: pooled sums -> means, next-step kmean ----
__global__ void prep_sums_to_means(float* __restrict__ kc,
                                   float* __restrict__ kmean_next,
                                   const int32_t* __restrict__ blen,
                                   int T, int NTB, int NPAD) {
    const int bh = blockIdx.x, d = threadIdx.x;
    float total = 0.f;
    int tokens = 0;
    for (int n = 0; n < NTB; ++n) {
        const size_t o = ((size_t)bh * NPAD + n) * HD + d;
        const float sum = kc[o];
        total += sum;
        const int len = block_len_of(blen, n, T);
        tokens += len;
        kc[o] = sum / (float)len;
    }
    kmean_next[(size_t)bh * HD + d] = total / (float)tokens;
}

// ---- producer-path threshold from the quantized centroid ----
__global__ void prep_thr_from_cen(const int8_t* __restrict__ cen8,
                                  const float* __restrict__ cens,
                                  const float* __restrict__ kcvar,
                                  float* __restrict__ thr,
                                  int NQ, float tau, float log2s) {
    __shared__ float sred[HD];
    const int qb = blockIdx.x, bh = blockIdx.y, d = threadIdx.x;
    const float c = (float)cen8[((size_t)bh * NQ + qb) * HD + perm_d(d)] *
                    cens[(size_t)bh * NQ + qb];
    const float var = block_sum128(c * c * kcvar[(size_t)bh * HD + d], sred);
    if (d == 0) thr[(size_t)bh * NQ + qb] = thr_of(var, tau, log2s);
}

}  // namespace

// Producer path: block K sums left in scratch -> means, pooled quant,
// threshold. The caller places the V scale it quantized with in the vsc slot.
void launch_sol_finish(
    void* scratch, void* kciP, void* kcs, void* threshold,
    const void* cen8, const void* cens, void* kmean_next, const void* blen,
    int B, int T, int H, int NTB, int NPAD, int NQ,
    float tau, float scale_log2, cudaStream_t stream)
{
    const Scratch s = carve_scratch(scratch, B, H, NPAD);
    prep_sums_to_means<<<B * H, HD, 0, stream>>>(
        s.kc, (float*)kmean_next, (const int32_t*)blen, T, NTB, NPAD);
    prep_pooled_stats<<<B * H, HD, 0, stream>>>(
        s.kc, s.kmean, nullptr, nullptr, s.kcvar, NTB, NPAD);
    prep_pooled_quant<<<dim3(NPAD, B * H), HD, 0, stream>>>(
        s.kc, s.kmean, (int8_t*)kciP, (float*)kcs, NTB, NPAD);
    prep_thr_from_cen<<<dim3(NQ, B * H), HD, 0, stream>>>(
        (const int8_t*)cen8, (const float*)cens, s.kcvar, (float*)threshold,
        NQ, tau, scale_log2);
}

template <typename E>
static void preprocess_launch(
    const void* q, const void* k, const void* v,
    void* qiP, void* qs, void* kiP, void* ksb, void* kciP, void* kcs,
    void* vcT, void* threshold, void* cen8, void* cens, void* vsc, void* qmean,
    void* scratch,           // sol_preprocess_scratch_bytes
    const void* key_bias,    // [B, Tk] f32 in log2 units, or nullptr
    const void* blen,        // [NTB] int32 valid tokens per block, or nullptr (T == Tk then)
    const void* ord,         // [B*H, T] int32 row order, or nullptr (T == Tk then)
    int B, int T, int Tk, int Tp, int H, int NTB, int NPAD, int NQ,   // T queries, Tk keys
    int64_t qs_b, int64_t qs_t, int64_t qs_h,
    int64_t ks_b, int64_t ks_t, int64_t ks_h,
    int64_t vs_b, int64_t vs_t, int64_t vs_h,
    float tau, float scale_log2, cudaStream_t stream)
{
    const Scratch s = carve_scratch(scratch, B, H, NPAD);

    // pass 1 accumulates V's |max| into vsc by atomicMax
    cudaMemsetAsync(vsc, 0, (size_t)B * H * HD * sizeof(float), stream);
    prep_reduce_kv<E><<<dim3(NPAD, B * H), HD, 0, stream>>>(
        (const E*)k, (const E*)v, s.kc, (__nv_bfloat16*)vcT,
        (float*)vsc, (const int32_t*)blen, Tk, H, NTB, NPAD, ks_b, ks_t, ks_h, vs_b, vs_t, vs_h,
        (const int32_t*)ord);
    prep_pooled_stats<<<B * H, HD, 0, stream>>>(
        s.kc, s.kmean, (const float*)vsc, (float*)vsc, s.kcvar, NTB, NPAD);
    prep_pooled_quant<<<dim3(NPAD, B * H), HD, 0, stream>>>(
        s.kc, s.kmean, (int8_t*)kciP, (float*)kcs, NTB, NPAD);
    prep_q<E><<<dim3(NQ, B * H), HD, 0, stream>>>(
        (const E*)q, s.kcvar, (int8_t*)qiP, (float*)qs, (float*)threshold,
        (int8_t*)cen8, (float*)cens, (float*)qmean, (const int32_t*)blen,
        T, H, NQ, tau, scale_log2, qs_b, qs_t, qs_h, (const int32_t*)ord);
    prep_k<E><<<dim3(NTB, B * H), HD, 0, stream>>>(
        (const E*)k, s.kmean, (int8_t*)kiP, (float2*)ksb,
        (const float*)key_bias, (const int32_t*)blen, Tk, Tp, H, ks_b, ks_t, ks_h, (const int32_t*)ord);
}

void launch_sol_preprocess(
    const void* q, const void* k, const void* v,
    void* qiP, void* qs, void* kiP, void* ksb, void* kciP, void* kcs,
    void* vcT, void* threshold, void* cen8, void* cens, void* vsc, void* qmean,
    void* scratch, const void* key_bias, const void* blen, const void* ord,
    int B, int T, int Tk, int Tp, int H, int NTB, int NPAD, int NQ,
    int64_t qs_b, int64_t qs_t, int64_t qs_h,
    int64_t ks_b, int64_t ks_t, int64_t ks_h,
    int64_t vs_b, int64_t vs_t, int64_t vs_h,
    float tau, float scale_log2, int elem, cudaStream_t stream)
{
    auto fn = elem == sol::SOL_FP16 ? preprocess_launch<__half> : preprocess_launch<__nv_bfloat16>;
    fn(q, k, v, qiP, qs, kiP, ksb, kciP, kcs, vcT, threshold, cen8, cens, vsc, qmean,
       scratch, key_bias, blen, ord, B, T, Tk, Tp, H, NTB, NPAD, NQ,
       qs_b, qs_t, qs_h, ks_b, ks_t, ks_h, vs_b, vs_t, vs_h, tau, scale_log2, stream);
}

// Needs the int8 carriers and the query block means; q and k of one length T.
void launch_sol_summary(const void* kiP, const void* ksb, const void* vTi, const void* qmean, void* scratch,
                        void* skiP, void* sksb, void* svT,
                        int B, int T, int Tp, int H, int NTB, int NPAD, int NQ,
                        float scale_log2, cudaStream_t stream) {
    const Scratch s = carve_scratch(scratch, B, H, NPAD);
    prep_anchor<<<B * H, HD, 0, stream>>>((const float*)qmean, s.anchor, T, NQ);
    prep_summary<<<dim3(NPAD, B * H), HD, 0, stream>>>(
        (const int8_t*)kiP, (const float2*)ksb, (const int8_t*)vTi, s.anchor,
        (int8_t*)skiP, (float2*)sksb, (int8_t*)svT, T, Tp, NTB, NPAD, scale_log2);
}

void launch_sol_group_features(const void* kiP, const void* ksb, const void* dir8, const void* dsc,
                               const void* doff, void* feat, int BH, int Tp, int b0, int nblocks, int rows1,
                               cudaStream_t stream) {
    group_features<<<dim3(nblocks, BH), 32, 0, stream>>>(
        (const int8_t*)kiP, (const float2*)ksb, (const int8_t*)dir8, (const float*)dsc, (const float*)doff,
        (__nv_bfloat16*)feat, Tp, b0, rows1);
}

// One node per thread block, one thread per pair of slots.
void launch_sol_group_leaves(const void* feat, void* perm, int bh, int n, int size, int total,
                             cudaStream_t stream) {
    group_leaves<<<dim3(total / size, bh), LEAF_ROWS / 2, 0, stream>>>(
        (const __nv_bfloat16*)feat, (int32_t*)perm, n, size, total);
}

// Staged raster k / v carriers (r*) -> the carriers and pooled statistics of the reordered blocks; q
// carriers (raster) -> the reordered blocks' centroids.
void launch_sol_reorder(const void* qiP, const void* qs, const void* rkiP, const void* rksb, const void* rvRow,
                        void* kiP, void* ksb, void* vTi, void* vcT, void* scratch,
                        void* cen8, void* cens, void* qmean,
                        const void* ord, const void* kmean, const void* vscale,
                        const void* vfac, int vfac_rows,
                        int B, int T, int Tp, int H, int NTB, int NPAD, int NQ, cudaStream_t stream) {
    const Scratch s = carve_scratch(scratch, B, H, NPAD);
    reorder_q<<<dim3(NQ, B * H), HD, 0, stream>>>(
        (const int8_t*)qiP, (const float*)qs,
        (int8_t*)cen8, (float*)cens, (float*)qmean, (const int32_t*)ord, T, H, NQ);
    reorder_k<<<dim3(NTB, B * H), HD, 0, stream>>>(
        (const int8_t*)rkiP, (const float2*)rksb, (int8_t*)kiP, (float2*)ksb, s.kc,
        (const float*)kmean, (const int32_t*)ord, T, Tp, NPAD);
    reorder_v<<<dim3(NTB, B * H), HD, 0, stream>>>(
        (const int8_t*)rvRow, (int8_t*)vTi, (__nv_bfloat16*)vcT, (const float*)vscale,
        (const int32_t*)ord, (const float*)vfac, vfac_rows, T, Tp, NPAD);
}

size_t sol_preprocess_scratch_bytes(int B, int H, int NPAD) {
    return ((size_t)B * H * NPAD * HEAD_DIM + (size_t)3 * B * H * HEAD_DIM) * sizeof(float);
}
