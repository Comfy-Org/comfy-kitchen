// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// GatedDeltaNet decode for S <= 8 tokens: the recurrent [DK, DV] fp32 state lives
// in shared memory for all S steps, one block per (batch, head), one thread per
// value column. Snapshots after steps 0..S-2 serve the speculative-decode rollback.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "float_utils.cuh"
#include "dtype_dispatch.cuh"

namespace {

template <typename T> __device__ __forceinline__ float to_f(T v);
template <> __device__ __forceinline__ float to_f<float>(float v) { return v; }
template <> __device__ __forceinline__ float to_f<__half>(__half v) { return __half2float(v); }
template <> __device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <typename T> __device__ __forceinline__ float round_to(float v) { return to_f<T>(comfy::from_float<T>(v)); }

// block-wide sum; every thread must call it (blockDim.x a multiple of 32, red[blockDim.x / 32])
__device__ __forceinline__ float block_sum(float v, float* red) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, o);
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0)
        red[wid] = v;
    __syncthreads();
    float s = 0.0f;
    const int nw = blockDim.x >> 5;
    for (int i = 0; i < nw; ++i)
        s += red[i];
    return s;
}

// gate projections, gate math, q/k normalization, S delta-rule steps and the gated RMSNorm in one block
template <typename T, int DK, int MAXS>
__global__ void gated_delta_decode_fused_kernel(
    const T* __restrict__ mixed_qkv,   // [B, C, S] conv+silu output
    const T* __restrict__ x,           // [B, S, Hd]
    const T* __restrict__ w_a,         // [Hv, Hd]
    const T* __restrict__ w_b,         // [Hv, Hd]
    const float* __restrict__ dt_bias, // [Hv]
    const float* __restrict__ g_decay, // [Hv]
    float* __restrict__ state,         // [B, Hv, DK, DV] updated in place
    T* __restrict__ out,               // [B, S, Hv, DV]
    float* __restrict__ snapshots,     // [S-1, B, Hv, DK, DV] or nullptr
    const T* __restrict__ z,           // [B, S, Hv*DV] norm gate
    const T* __restrict__ norm_w,      // [DV]
    float eps,
    int B, int Hv, int Hk, int S, int DV, int C, int Hd, int key_dim, float scale)
{
    extern __shared__ float sm[];      // DK*DV state | MAXS*DK q rows | MAXS*DK k rows | 4*MAXS*nw reduce scratch
    float* sq = sm + DK * DV;
    float* sk = sq + MAXS * DK;
    float* red = sk + MAXS * DK;
    const int b = static_cast<int>(blockIdx.x) / Hv;
    const int h = static_cast<int>(blockIdx.x) % Hv;
    const int t = threadIdx.x;
    const int lane = t & 31, wid = t >> 5, nw = static_cast<int>(blockDim.x) >> 5;
    const int hk = h / (Hv / Hk);
    const int64_t state_off = (static_cast<int64_t>(b) * Hv + h) * DK * DV;
    const int64_t qbase = static_cast<int64_t>(b) * C + static_cast<int64_t>(hk) * DK;
    const int64_t kbase = qbase + key_dim;
    const int64_t vbase = static_cast<int64_t>(b) * C + 2 * static_cast<int64_t>(key_dim) + static_cast<int64_t>(h) * DV;

    // each thread owns one state column for the whole kernel: no barriers in the state passes
    if (t < DV) {
        #pragma unroll 8
        for (int kk = 0; kk < DK; ++kk)
            sm[kk * DV + t] = state[state_off + kk * DV + t];
    }

    float da[MAXS], db[MAXS];
    #pragma unroll
    for (int s = 0; s < MAXS; ++s) {
        da[s] = 0.0f;
        db[s] = 0.0f;
    }
    {
        const T* __restrict__ wa = w_a + static_cast<int64_t>(h) * Hd;
        const T* __restrict__ wb = w_b + static_cast<int64_t>(h) * Hd;
        const T* __restrict__ xb = x + static_cast<int64_t>(b) * S * Hd;
        for (int i = t; i < Hd; i += blockDim.x) {
            const float wav = to_f<T>(wa[i]);
            const float wbv = to_f<T>(wb[i]);
            #pragma unroll
            for (int s = 0; s < MAXS; ++s) {
                if (s < S) {
                    const float xv = to_f<T>(xb[static_cast<int64_t>(s) * Hd + i]);
                    da[s] = fmaf(xv, wav, da[s]);
                    db[s] = fmaf(xv, wbv, db[s]);
                }
            }
        }
    }
    float qv[MAXS], kv[MAXS], qs[MAXS], ks[MAXS];
    #pragma unroll
    for (int s = 0; s < MAXS; ++s) {
        qv[s] = 0.0f;
        kv[s] = 0.0f;
        if (s < S && t < DK) {
            qv[s] = to_f<T>(mixed_qkv[(qbase + t) * S + s]);
            kv[s] = to_f<T>(mixed_qkv[(kbase + t) * S + s]);
        }
        qs[s] = qv[s] * qv[s];
        ks[s] = kv[s] * kv[s];
    }
    // one block-wide reduction of all 4*S partial sums
    #pragma unroll
    for (int s = 0; s < MAXS; ++s) {
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            da[s] += __shfl_xor_sync(0xffffffffu, da[s], o);
            db[s] += __shfl_xor_sync(0xffffffffu, db[s], o);
            qs[s] += __shfl_xor_sync(0xffffffffu, qs[s], o);
            ks[s] += __shfl_xor_sync(0xffffffffu, ks[s], o);
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (int s = 0; s < MAXS; ++s) {
            if (s < S) {
                red[(4 * s + 0) * nw + wid] = da[s];
                red[(4 * s + 1) * nw + wid] = db[s];
                red[(4 * s + 2) * nw + wid] = qs[s];
                red[(4 * s + 3) * nw + wid] = ks[s];
            }
        }
    }
    __syncthreads();
    float beta[MAXS], gs[MAXS];
    #pragma unroll
    for (int s = 0; s < MAXS; ++s) {
        if (s < S) {
            float sa = 0.0f, sb = 0.0f, sqs = 0.0f, sks = 0.0f;
            for (int w = 0; w < nw; ++w) {
                sa += red[(4 * s + 0) * nw + w];
                sb += red[(4 * s + 1) * nw + w];
                sqs += red[(4 * s + 2) * nw + w];
                sks += red[(4 * s + 3) * nw + w];
            }
            const float qn = fmaxf(sqrtf(sqs), 1e-12f);
            const float kn = fmaxf(sqrtf(sks), 1e-12f);
            if (t < DK) {
                sq[s * DK + t] = qv[s] / qn * scale;
                sk[s * DK + t] = kv[s] / kn;
            }
            // matches the eager chain: bf16 projection outputs, bf16 sigmoid, fp32 softplus/exp
            beta[s] = round_to<T>(1.0f / (1.0f + expf(-round_to<T>(sb))));
            const float aa = round_to<T>(sa) + dt_bias[h];
            const float sp = aa > 20.0f ? aa : log1pf(expf(aa));
            gs[s] = expf(g_decay[h] * sp);
        }
    }
    __syncthreads();

    #pragma unroll 1
    for (int s = 0; s < S; ++s) {
        const float* __restrict__ skr = sk + s * DK;
        const float* __restrict__ sqr = sq + s * DK;
        float o_out = 0.0f;
        if (t < DV) {
            const float g_s = gs[s], b_s = beta[s];
            float kvm = 0.0f;
            #pragma unroll 4
            for (int kk = 0; kk < DK; ++kk) {
                const float sv = sm[kk * DV + t] * g_s;
                sm[kk * DV + t] = sv;
                kvm = fmaf(skr[kk], sv, kvm);
            }
            const float vv = to_f<T>(mixed_qkv[(vbase + t) * S + s]);
            const float delta = (vv - kvm) * b_s;
            float o = 0.0f;
            #pragma unroll 4
            for (int kk = 0; kk < DK; ++kk) {
                const float sv = fmaf(skr[kk], delta, sm[kk * DV + t]);
                sm[kk * DV + t] = sv;
                o = fmaf(sqr[kk], sv, o);
            }
            o_out = round_to<T>(o);
        }
        // torch rms_norm (fp32 compute, one rounding) times a bf16 silu gate
        const float ss = block_sum(o_out * o_out, red);
        if (t < DV) {
            const float rstd = rsqrtf(ss / static_cast<float>(DV) + eps);
            const float y = round_to<T>(o_out * rstd * to_f<T>(norm_w[t]));
            const int64_t orow = ((static_cast<int64_t>(b) * S + s) * Hv + h) * DV + t;
            const float zz = to_f<T>(z[orow]);
            const float gate = round_to<T>(zz / (1.0f + expf(-zz)));
            out[orow] = comfy::from_float<T>(round_to<T>(y * gate));
            if (snapshots != nullptr && s < S - 1) {
                float* snap = snapshots + ((static_cast<int64_t>(s) * B + b) * Hv + h) * DK * DV;
                #pragma unroll 8
                for (int kk = 0; kk < DK; ++kk)
                    snap[kk * DV + t] = sm[kk * DV + t];
            }
        }
    }
    if (t < DV) {
        #pragma unroll 8
        for (int kk = 0; kk < DK; ++kk)
            state[state_off + kk * DV + t] = sm[kk * DV + t];
    }
}

// depthwise causal conv step: one thread per (batch, channel) owns its window, so the in-place state update cannot race
template <typename T>
__global__ void deltanet_conv_step_kernel(
    const T* __restrict__ proj,        // [B, S, C] projection output
    T* __restrict__ conv_state,        // [B, C, KS-1] updated in place
    const T* __restrict__ conv_w,      // [C, KS] depthwise taps
    const T* __restrict__ conv_b,      // [C] or nullptr
    T* __restrict__ conv_out,          // [B, C, S] silu(conv)
    T* __restrict__ conv_snaps,        // [S-1, B, C, KS-1] or nullptr
    int B, int C, int S, int KS)
{
    const int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= B * C)
        return;
    const int b = idx / C, c = idx - b * C;
    constexpr int MAXW = 16;
    float win[MAXW];
    const int L = KS - 1;
    #pragma unroll
    for (int j = 0; j < MAXW; ++j) {
        if (j < L)
            win[j] = to_f<T>(conv_state[(static_cast<int64_t>(b) * C + c) * L + j]);
        else if (j < L + S)
            win[j] = to_f<T>(proj[(static_cast<int64_t>(b) * S + (j - L)) * C + c]);
        else
            win[j] = 0.0f;
    }
    float w[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j)
        w[j] = j < KS ? to_f<T>(conv_w[static_cast<int64_t>(c) * KS + j]) : 0.0f;
    const float bias = conv_b != nullptr ? to_f<T>(conv_b[c]) : 0.0f;
    #pragma unroll
    for (int s = 0; s < 8; ++s) {
        if (s < S) {
            float acc = 0.0f;
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                if (j < KS)
                    acc = fmaf(w[j], win[s + j], acc);
            const float y = round_to<T>(acc + bias);
            conv_out[(static_cast<int64_t>(b) * C + c) * S + s] = comfy::from_float<T>(y / (1.0f + expf(-y)));
            if (conv_snaps != nullptr && s < S - 1) {
                T* snap = conv_snaps + ((static_cast<int64_t>(s) * B + b) * C + c) * L;
                #pragma unroll
                for (int j = 0; j < MAXW; ++j)
                    if (j < L)
                        snap[j] = comfy::from_float<T>(win[s + 1 + j]);
            }
        }
    }
    #pragma unroll
    for (int j = 0; j < MAXW; ++j)
        if (j < L)
            conv_state[(static_cast<int64_t>(b) * C + c) * L + j] = comfy::from_float<T>(win[S + j]);
}

// opt the kernel into the device's full dynamic shared memory once per (device, kernel)
bool fused_shmem_ok(const void* fn, int fn_slot, size_t shmem) {
    constexpr int MAX_DEV = 16, SLOTS = 3;
    static int max_shmem[MAX_DEV] = {};
    static bool attr_set[MAX_DEV][SLOTS] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= MAX_DEV)
        return false;
    if (max_shmem[dev] == 0
            && cudaDeviceGetAttribute(&max_shmem[dev], cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess)
        return false;
    if (shmem > static_cast<size_t>(max_shmem[dev]))
        return false;
    if (!attr_set[dev][fn_slot]) {
        if (cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem[dev]) != cudaSuccess)
            return false;
        attr_set[dev][fn_slot] = true;
    }
    return true;
}

}  // namespace

extern "C" bool launch_gated_delta_decode_fused(
    const void* mixed_qkv, const void* x, const void* w_a, const void* w_b,
    const void* dt_bias, const void* g_decay, void* state, void* out, void* snapshots,
    const void* z, const void* norm_w, float eps,
    int64_t B, int64_t Hv, int64_t Hk, int64_t S, int64_t DK, int64_t DV, int64_t C, int64_t Hd,
    int64_t key_dim, float scale, int dtype_code, cudaStream_t stream)
{
    constexpr int MAXS = 8;
    if (DK != 128 || DV <= 0 || DV > 512 || DV % 32 != 0 || S < 1 || S > MAXS || Hk <= 0 || Hv % Hk != 0
            || dtype_code < 0 || dtype_code > 2)
        return false;
    const int threads = DV > 128 ? static_cast<int>(DV) : 128;
    const size_t shmem = (static_cast<size_t>(DK) * DV + 2 * MAXS * DK + 4 * MAXS * (threads / 32)) * sizeof(float);
    return DISPATCH_FP_DTYPE(dtype_code, T, [&] {
        auto kfn = gated_delta_decode_fused_kernel<T, 128, MAXS>;
        if (!fused_shmem_ok(reinterpret_cast<const void*>(kfn), dtype_code, shmem))
            return false;
        kfn<<<static_cast<unsigned>(B * Hv), threads, shmem, stream>>>(
            static_cast<const T*>(mixed_qkv), static_cast<const T*>(x),
            static_cast<const T*>(w_a), static_cast<const T*>(w_b),
            static_cast<const float*>(dt_bias), static_cast<const float*>(g_decay),
            static_cast<float*>(state), static_cast<T*>(out), static_cast<float*>(snapshots),
            static_cast<const T*>(z), static_cast<const T*>(norm_w), eps,
            static_cast<int>(B), static_cast<int>(Hv), static_cast<int>(Hk), static_cast<int>(S),
            static_cast<int>(DV), static_cast<int>(C), static_cast<int>(Hd),
            static_cast<int>(key_dim), scale);
        return cudaGetLastError() == cudaSuccess;
    });
}

extern "C" bool launch_deltanet_conv_step(
    const void* proj, void* conv_state, const void* conv_w, const void* conv_b,
    void* conv_out, void* conv_snaps,
    int64_t B, int64_t C, int64_t S, int64_t KS, int dtype_code, cudaStream_t stream)
{
    if (S < 1 || S > 8 || KS < 2 || KS > 8 || dtype_code < 0 || dtype_code > 2)
        return false;
    const int threads = 256;
    const unsigned grid = static_cast<unsigned>((B * C + threads - 1) / threads);
    return DISPATCH_FP_DTYPE(dtype_code, T, [&] {
        deltanet_conv_step_kernel<T><<<grid, threads, 0, stream>>>(
            static_cast<const T*>(proj), static_cast<T*>(conv_state), static_cast<const T*>(conv_w),
            static_cast<const T*>(conv_b), static_cast<T*>(conv_out), static_cast<T*>(conv_snaps),
            static_cast<int>(B), static_cast<int>(C), static_cast<int>(S), static_cast<int>(KS));
        return cudaGetLastError() == cudaSuccess;
    });
}
