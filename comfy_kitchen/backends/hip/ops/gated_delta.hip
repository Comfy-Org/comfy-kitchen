// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// GatedDeltaNet decode for S <= 8 tokens: gate projections, gate math, q/k
// normalization, the delta-rule steps and the gated RMSNorm in one block per
// (batch, value head). Snapshots after steps 0..S-2 serve the speculative-decode
// rollback.
//
// The recurrent state is [DK, DV] fp32, which at the shape this serves (DK = DV =
// 128) is 64 KiB, exactly the whole RDNA per-workgroup LDS budget. So the state
// lives in registers instead: R threads split the DK axis of one value column,
// each holding DK / R of it for the whole kernel, and only the two DK reductions
// per step meet in LDS. That also drops the state's global traffic to one read and
// one write regardless of S.
//
// Thread index is r * DV + v, so a wave spans consecutive value columns of one DK
// partition: the state loads, the snapshots and the writeback all coalesce, and
// the partition reduction is the only cross-thread step.
#include <stdexcept>

#include "../mma.h"
#include "../rope_math.h"

namespace comfy::hip_backend {

constexpr int kDeltaMaxSteps = 8;
constexpr int kDeltaKeyDim = 128;  // the only DK the launcher accepts, as in CUDA
constexpr int kDeltaConvMaxWindow = 16;

template <typename T>
__forceinline__ __device__ float to_f(T v) {
    return static_cast<float>(v);
}

template <>
__forceinline__ __device__ float to_f<__half>(__half v) {
    return __half2float(v);
}

// Block-wide sum; every thread must call it. red holds one float per wave.
__forceinline__ __device__ float delta_block_sum(float v, float* red, int waves) {
#pragma unroll
    for (int off = kWave / 2; off > 0; off >>= 1) {
        v += __shfl_xor(v, off, kWave);
    }
    __syncthreads();
    if ((threadIdx.x & (kWave - 1)) == 0) {
        red[threadIdx.x / kWave] = v;
    }
    __syncthreads();
    float total = 0.0f;
    for (int i = 0; i < waves; ++i) {
        total += red[i];
    }
    return total;
}

// KPT is the DK slice one thread owns. It has to be a compile-time constant: the
// state array is indexed only by fully unrolled loops, which is what keeps it in
// registers instead of scratch.
template <typename T, int KPT>
__global__ void gated_delta_decode_fused_kernel(
    const T* __restrict__ mixed_qkv,    // [B, C, S] conv + silu output
    const T* __restrict__ x,            // [B, S, Hd]
    const T* __restrict__ w_a,          // [Hv, Hd]
    const T* __restrict__ w_b,          // [Hv, Hd]
    const float* __restrict__ dt_bias,  // [Hv]
    const float* __restrict__ g_decay,  // [Hv]
    float* __restrict__ state,          // [B, Hv, DK, DV], updated in place
    T* __restrict__ out,                // [B, S, Hv, DV]
    float* __restrict__ snapshots,      // [S-1, B, Hv, DK, DV] or null
    const T* __restrict__ z,            // [B, S, Hv*DV] norm gate
    const T* __restrict__ norm_w,       // [DV]
    float eps, int B, int Hv, int Hk, int S, int DV, int C, int Hd, int key_dim, float scale) {

    constexpr int DK = kDeltaKeyDim;
    constexpr int R = DK / KPT;
    const int waves = static_cast<int>(blockDim.x) / kWave;

    extern __shared__ float sm[];
    float* sq = sm;             // S * DK
    float* sk = sq + S * DK;    // S * DK
    float* part = sk + S * DK;  // R * DV, the DK partition reduction
    float* red = part + R * DV; // 4 * S * waves, block reductions

    const int t = static_cast<int>(threadIdx.x);
    const int v = t % DV;
    const int r = t / DV;
    const int kk0 = r * KPT;
    const int b = static_cast<int>(blockIdx.x) / Hv;
    const int h = static_cast<int>(blockIdx.x) % Hv;
    const int hk = h / (Hv / Hk);
    const int64_t state_off = (static_cast<int64_t>(b) * Hv + h) * DK * DV;
    const int64_t qbase = static_cast<int64_t>(b) * C + static_cast<int64_t>(hk) * DK;
    const int64_t kbase = qbase + key_dim;
    const int64_t vbase = static_cast<int64_t>(b) * C + 2 * static_cast<int64_t>(key_dim) +
                          static_cast<int64_t>(h) * DV;

    float st[KPT];
#pragma unroll
    for (int j = 0; j < KPT; ++j) {
        st[j] = state[state_off + static_cast<int64_t>(kk0 + j) * DV + v];
    }

    float da[kDeltaMaxSteps], db[kDeltaMaxSteps];
#pragma unroll
    for (int s = 0; s < kDeltaMaxSteps; ++s) {
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
            for (int s = 0; s < kDeltaMaxSteps; ++s) {
                if (s < S) {
                    const float xv = to_f<T>(xb[static_cast<int64_t>(s) * Hd + i]);
                    da[s] = fmaf(xv, wav, da[s]);
                    db[s] = fmaf(xv, wbv, db[s]);
                }
            }
        }
    }

    // The raw rows go straight to LDS and are normalized in place once the norms
    // are known, so only the squared partials stay in registers: at KPT = 64 the
    // state alone is most of the budget.
    float qs[kDeltaMaxSteps], ks[kDeltaMaxSteps];
#pragma unroll
    for (int s = 0; s < kDeltaMaxSteps; ++s) {
        float q = 0.0f, k = 0.0f;
        if (s < S && t < DK) {
            q = to_f<T>(mixed_qkv[(qbase + t) * S + s]);
            k = to_f<T>(mixed_qkv[(kbase + t) * S + s]);
            sq[s * DK + t] = q;
            sk[s * DK + t] = k;
        }
        qs[s] = q * q;
        ks[s] = k * k;
    }
    // One block-wide pass folds all 4 * S partial sums together.
#pragma unroll
    for (int s = 0; s < kDeltaMaxSteps; ++s) {
#pragma unroll
        for (int off = kWave / 2; off > 0; off >>= 1) {
            da[s] += __shfl_xor(da[s], off, kWave);
            db[s] += __shfl_xor(db[s], off, kWave);
            qs[s] += __shfl_xor(qs[s], off, kWave);
            ks[s] += __shfl_xor(ks[s], off, kWave);
        }
    }
    if ((t & (kWave - 1)) == 0) {
        const int wid = t / kWave;
#pragma unroll
        for (int s = 0; s < kDeltaMaxSteps; ++s) {
            if (s < S) {
                red[(4 * s + 0) * waves + wid] = da[s];
                red[(4 * s + 1) * waves + wid] = db[s];
                red[(4 * s + 2) * waves + wid] = qs[s];
                red[(4 * s + 3) * waves + wid] = ks[s];
            }
        }
    }
    __syncthreads();

    float beta[kDeltaMaxSteps], gs[kDeltaMaxSteps];
#pragma unroll
    for (int s = 0; s < kDeltaMaxSteps; ++s) {
        beta[s] = 0.0f;
        gs[s] = 0.0f;
        if (s < S) {
            float sa = 0.0f, sb = 0.0f, sqs = 0.0f, sks = 0.0f;
            for (int w = 0; w < waves; ++w) {
                sa += red[(4 * s + 0) * waves + w];
                sb += red[(4 * s + 1) * waves + w];
                sqs += red[(4 * s + 2) * waves + w];
                sks += red[(4 * s + 3) * waves + w];
            }
            const float qn = fmaxf(sqrtf(sqs), 1e-12f);
            const float kn = fmaxf(sqrtf(sks), 1e-12f);
            if (t < DK) {
                sq[s * DK + t] = sq[s * DK + t] / qn * scale;
                sk[s * DK + t] = sk[s * DK + t] / kn;
            }
            // Matches the eager chain: the projection outputs and the sigmoid round
            // to the activation dtype, softplus and exp stay in fp32.
            beta[s] = round_to<T>(1.0f / (1.0f + expf(-round_to<T>(sb))));
            const float aa = round_to<T>(sa) + dt_bias[h];
            const float sp = aa > 20.0f ? aa : log1pf(expf(aa));
            gs[s] = expf(g_decay[h] * sp);
        }
    }
    __syncthreads();

    for (int s = 0; s < S; ++s) {
        const float* __restrict__ skr = sk + s * DK + kk0;
        const float* __restrict__ sqr = sq + s * DK + kk0;
        const float g_s = gs[s];

        float kvm = 0.0f;
#pragma unroll
        for (int j = 0; j < KPT; ++j) {
            st[j] *= g_s;
            kvm = fmaf(skr[j], st[j], kvm);
        }
        part[r * DV + v] = kvm;
        __syncthreads();
        float kv_mem = 0.0f;
        for (int rr = 0; rr < R; ++rr) {
            kv_mem += part[rr * DV + v];
        }

        const float vv = to_f<T>(mixed_qkv[(vbase + v) * S + s]);
        const float delta = (vv - kv_mem) * beta[s];
        float o = 0.0f;
#pragma unroll
        for (int j = 0; j < KPT; ++j) {
            st[j] = fmaf(skr[j], delta, st[j]);
            o = fmaf(sqr[j], st[j], o);
        }
        __syncthreads();
        part[r * DV + v] = o;
        __syncthreads();
        float o_sum = 0.0f;
        for (int rr = 0; rr < R; ++rr) {
            o_sum += part[rr * DV + v];
        }

        // torch rms_norm (fp32 compute, one rounding) times a silu gate in T.
        const float o_out = (r == 0) ? round_to<T>(o_sum) : 0.0f;
        const float ss = delta_block_sum(o_out * o_out, red, waves);
        if (r == 0) {
            const float rstd = rsqrtf(ss / static_cast<float>(DV) + eps);
            const float y = round_to<T>(o_out * rstd * to_f<T>(norm_w[v]));
            const int64_t orow = ((static_cast<int64_t>(b) * S + s) * Hv + h) * DV + v;
            const float zz = to_f<T>(z[orow]);
            const float gate = round_to<T>(zz / (1.0f + expf(-zz)));
            rope_store<T>(out, orow, y * gate);
        }
        if (snapshots != nullptr && s < S - 1) {
            float* snap = snapshots + ((static_cast<int64_t>(s) * B + b) * Hv + h) * DK * DV;
#pragma unroll
            for (int j = 0; j < KPT; ++j) {
                snap[static_cast<int64_t>(kk0 + j) * DV + v] = st[j];
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < KPT; ++j) {
        state[state_off + static_cast<int64_t>(kk0 + j) * DV + v] = st[j];
    }
}

// Depthwise causal conv step: one thread per (batch, channel) owns its whole
// window, so the in-place state update cannot race.
template <typename T>
__global__ void deltanet_conv_step_kernel(
    const T* __restrict__ proj,    // [B, S, C] projection output
    T* __restrict__ conv_state,    // [B, C, KS-1], updated in place
    const T* __restrict__ conv_w,  // [C, KS] depthwise taps
    const T* __restrict__ conv_b,  // [C] or null
    T* __restrict__ conv_out,      // [B, C, S] silu(conv)
    T* __restrict__ conv_snaps,    // [S-1, B, C, KS-1] or null
    int B, int C, int S, int KS) {

    const int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= B * C) {
        return;
    }
    const int b = idx / C;
    const int c = idx - b * C;
    const int L = KS - 1;

    float win[kDeltaConvMaxWindow];
#pragma unroll
    for (int j = 0; j < kDeltaConvMaxWindow; ++j) {
        if (j < L) {
            win[j] = to_f<T>(conv_state[(static_cast<int64_t>(b) * C + c) * L + j]);
        } else if (j < L + S) {
            win[j] = to_f<T>(proj[(static_cast<int64_t>(b) * S + (j - L)) * C + c]);
        } else {
            win[j] = 0.0f;
        }
    }
    float w[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        w[j] = j < KS ? to_f<T>(conv_w[static_cast<int64_t>(c) * KS + j]) : 0.0f;
    }
    const float bias = conv_b != nullptr ? to_f<T>(conv_b[c]) : 0.0f;

#pragma unroll
    for (int s = 0; s < kDeltaMaxSteps; ++s) {
        if (s < S) {
            float acc = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                if (j < KS) {
                    acc = fmaf(w[j], win[s + j], acc);
                }
            }
            const float y = round_to<T>(acc + bias);
            rope_store<T>(conv_out, (static_cast<int64_t>(b) * C + c) * S + s,
                          y / (1.0f + expf(-y)));
            if (conv_snaps != nullptr && s < S - 1) {
                T* snap = conv_snaps + ((static_cast<int64_t>(s) * B + b) * C + c) * L;
#pragma unroll
                for (int j = 0; j < kDeltaConvMaxWindow; ++j) {
                    if (j < L) {
                        rope_store<T>(snap, j, win[s + 1 + j]);
                    }
                }
            }
        }
    }
#pragma unroll
    for (int j = 0; j < kDeltaConvMaxWindow; ++j) {
        if (j < L) {
            rope_store<T>(conv_state, (static_cast<int64_t>(b) * C + c) * L + j, win[S + j]);
        }
    }
}

constexpr int kDeltaSlotMax = 8;  // token slots per deferred side buffer, as in CUDA

// Deferred-commit conv step: one thread per (batch, channel). The window after the
// first pending tokens of the previous step is committed first, then the current S
// tokens are convolved on top of it in sequence.
template <typename T>
__global__ void deltanet_conv_deferred_kernel(
    const T* __restrict__ proj,    // [B, S, C] projection output
    T* __restrict__ proj_buf,      // [2, B, kDeltaSlotMax, C] projections of the previous / current step
    T* __restrict__ conv_state,    // [B, C, KS-1] committed window, updated in place
    const T* __restrict__ conv_w,  // [C, KS] depthwise taps
    const T* __restrict__ conv_b,  // [C] or null
    T* __restrict__ qkv_buf,       // [2, B, C, kDeltaSlotMax] silu(conv) of the previous / current step
    const int* __restrict__ ctl,   // {pending, parity}
    int B, int C, int S, int KS) {

    const int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= B * C) {
        return;
    }
    const int b = idx / C;
    const int c = idx - b * C;
    const int pending = ctl[0], par = ctl[1];
    const int L = KS - 1;
    const int64_t proj_slab = static_cast<int64_t>(B) * kDeltaSlotMax * C;
    const T* __restrict__ prev_proj =
        proj_buf + (1 - par) * proj_slab + (static_cast<int64_t>(b) * kDeltaSlotMax) * C + c;
    T* __restrict__ cur_proj =
        proj_buf + par * proj_slab + (static_cast<int64_t>(b) * kDeltaSlotMax) * C + c;
    T* __restrict__ state_row = conv_state + (static_cast<int64_t>(b) * C + c) * L;

    float win[kDeltaConvMaxWindow];
#pragma unroll
    for (int j = 0; j < kDeltaConvMaxWindow; ++j) {
        if (j < L) {
            // committed window = the last L of [conv_state, prev_proj[0..pending)]
            const int i = pending + j;
            win[j] = i < L ? to_f<T>(state_row[i])
                           : to_f<T>(prev_proj[static_cast<int64_t>(i - L) * C]);
        } else if (j < L + S) {
            win[j] = to_f<T>(proj[(static_cast<int64_t>(b) * S + (j - L)) * C + c]);
        } else {
            win[j] = 0.0f;
        }
    }
#pragma unroll
    for (int j = 0; j < kDeltaConvMaxWindow; ++j) {
        if (j < L) {
            rope_store<T>(state_row, j, win[j]);
        }
    }
    // taps in distance order: w[d] multiplies the token d steps back from the row
    float w[8];
#pragma unroll
    for (int d = 0; d < 8; ++d) {
        w[d] = d < KS ? to_f<T>(conv_w[static_cast<int64_t>(c) * KS + (KS - 1 - d)]) : 0.0f;
    }
    const float bias = conv_b != nullptr ? to_f<T>(conv_b[c]) : 0.0f;
    T* __restrict__ cur_qkv = qkv_buf + par * (static_cast<int64_t>(B) * C * kDeltaSlotMax) +
                              (static_cast<int64_t>(b) * C + c) * kDeltaSlotMax;
#pragma unroll
    for (int s = 0; s < kDeltaSlotMax; ++s) {
        if (s < S) {
            // Sum oldest tap first, preserving the direct path's FMA order.
            float acc = 0.0f;
#pragma unroll
            for (int d = 7; d >= 0; --d) {
                if (d < KS) {
                    acc = fmaf(w[d], win[L + s - d], acc);
                }
            }
            const float y = round_to<T>(acc + bias);
            rope_store<T>(cur_qkv, s, y / (1.0f + expf(-y)));
            rope_store<T>(cur_proj, static_cast<int64_t>(s) * C, win[L + s]);
        }
    }
}

// Deferred-commit delta-rule steps for one (batch, value head), laid out like the
// snapshot kernel above (thread r * DV + v owns a KPT slice of one value column).
// The pending accepted tokens of the previous step are replayed from the side
// buffers (their gates and q/k norms were saved there) and committed to global
// state; the S current chain tokens then only produce outputs.
template <typename T, int KPT>
__global__ void gated_delta_decode_deferred_kernel(
    const T* __restrict__ x,            // [B, S, Hd]
    const T* __restrict__ w_a,          // [Hv, Hd]
    const T* __restrict__ w_b,          // [Hv, Hd]
    const float* __restrict__ dt_bias,  // [Hv]
    const float* __restrict__ g_decay,  // [Hv]
    const T* __restrict__ qkv_buf,      // [2, B, C, kDeltaSlotMax] conv+silu output, previous / current step
    float* __restrict__ gates_buf,      // [2, B, kDeltaSlotMax, Hv, 2]: decay g, beta
    float* __restrict__ sumsq_buf,      // [2, B, kDeltaSlotMax, Hk, 2]: sum q^2, sum k^2
    const int* __restrict__ ctl,        // {pending, parity}
    float* __restrict__ state,          // [B, Hv, DK, DV] committed state, updated in place
    T* __restrict__ out,                // [B, S, Hv, DV]
    const T* __restrict__ z,            // [B, S, Hv*DV] norm gate
    const T* __restrict__ norm_w,       // [DV]
    float eps, int B, int Hv, int Hk, int S, int DV, int C, int Hd, int key_dim, float scale) {

    constexpr int DK = kDeltaKeyDim;
    constexpr int R = DK / KPT;
    const int waves = static_cast<int>(blockDim.x) / kWave;
    const int maxt = kDeltaSlotMax + S;  // replay slots + current rows

    extern __shared__ float sm[];
    float* sq = sm;                   // maxt * DK, normalized and scaled
    float* sk = sq + maxt * DK;       // maxt * DK, normalized
    float* sv = sk + maxt * DK;       // maxt * DV
    float* sg = sv + maxt * DV;       // maxt * 2: decay g, beta
    float* ssum = sg + maxt * 2;      // maxt * 2: sum q^2, sum k^2
    float* part = ssum + maxt * 2;    // R * DV, the DK partition reduction
    float* red = part + R * DV;       // 4 * S * waves, block reductions

    const int t = static_cast<int>(threadIdx.x);
    const int v = t % DV;
    const int r = t / DV;
    const int kk0 = r * KPT;
    const int b = static_cast<int>(blockIdx.x) / Hv;
    const int h = static_cast<int>(blockIdx.x) % Hv;
    const int gqa = Hv / Hk;
    const int hk = h / gqa;
    const int pending = ctl[0], par = ctl[1];
    const int64_t qkv_slab = static_cast<int64_t>(B) * C * kDeltaSlotMax;
    const T* __restrict__ prev_qkv =
        qkv_buf + (1 - par) * qkv_slab + static_cast<int64_t>(b) * C * kDeltaSlotMax;
    const T* __restrict__ cur_qkv =
        qkv_buf + par * qkv_slab + static_cast<int64_t>(b) * C * kDeltaSlotMax;
    const int64_t state_off = (static_cast<int64_t>(b) * Hv + h) * DK * DV;

    float st[KPT];
#pragma unroll
    for (int j = 0; j < KPT; ++j) {
        st[j] = state[state_off + static_cast<int64_t>(kk0 + j) * DV + v];
    }

    // gate projections and q/k sums of squares of the current rows, one block-wide pass
    float da[kDeltaSlotMax], db[kDeltaSlotMax], qs[kDeltaSlotMax], ks[kDeltaSlotMax];
#pragma unroll
    for (int s = 0; s < kDeltaSlotMax; ++s) {
        da[s] = 0.0f;
        db[s] = 0.0f;
        qs[s] = 0.0f;
        ks[s] = 0.0f;
    }
    {
        const T* __restrict__ wa = w_a + static_cast<int64_t>(h) * Hd;
        const T* __restrict__ wb = w_b + static_cast<int64_t>(h) * Hd;
        const T* __restrict__ xb = x + static_cast<int64_t>(b) * S * Hd;
        for (int i = t; i < Hd; i += blockDim.x) {
            const float wav = to_f<T>(wa[i]);
            const float wbv = to_f<T>(wb[i]);
#pragma unroll
            for (int s = 0; s < kDeltaSlotMax; ++s) {
                if (s < S) {
                    const float xv = to_f<T>(xb[static_cast<int64_t>(s) * Hd + i]);
                    da[s] = fmaf(xv, wav, da[s]);
                    db[s] = fmaf(xv, wbv, db[s]);
                }
            }
        }
        if (t < DK) {
            const int64_t qrow = static_cast<int64_t>(hk) * DK + t;
#pragma unroll
            for (int s = 0; s < kDeltaSlotMax; ++s) {
                if (s < S) {
                    const float q = to_f<T>(cur_qkv[qrow * kDeltaSlotMax + s]);
                    const float k = to_f<T>(cur_qkv[(qrow + key_dim) * kDeltaSlotMax + s]);
                    qs[s] = q * q;
                    ks[s] = k * k;
                }
            }
        }
    }
#pragma unroll
    for (int s = 0; s < kDeltaSlotMax; ++s) {
#pragma unroll
        for (int off = kWave / 2; off > 0; off >>= 1) {
            da[s] += __shfl_xor(da[s], off, kWave);
            db[s] += __shfl_xor(db[s], off, kWave);
            qs[s] += __shfl_xor(qs[s], off, kWave);
            ks[s] += __shfl_xor(ks[s], off, kWave);
        }
    }
    if ((t & (kWave - 1)) == 0) {
        const int wid = t / kWave;
#pragma unroll
        for (int s = 0; s < kDeltaSlotMax; ++s) {
            if (s < S) {
                red[(4 * s + 0) * waves + wid] = da[s];
                red[(4 * s + 1) * waves + wid] = db[s];
                red[(4 * s + 2) * waves + wid] = qs[s];
                red[(4 * s + 3) * waves + wid] = ks[s];
            }
        }
    }
    // gates and norms of the previous step's pending tokens come from the side buffers
    if (t < kDeltaSlotMax && t < pending) {
        const float* g = gates_buf +
                         (((1 - par) * static_cast<int64_t>(B) + b) * kDeltaSlotMax + t) * Hv * 2 +
                         h * 2;
        const float* n = sumsq_buf +
                         (((1 - par) * static_cast<int64_t>(B) + b) * kDeltaSlotMax + t) * Hk * 2 +
                         hk * 2;
        sg[t * 2 + 0] = g[0];
        sg[t * 2 + 1] = g[1];
        ssum[t * 2 + 0] = n[0];
        ssum[t * 2 + 1] = n[1];
    }
    __syncthreads();
    if (t >= kDeltaSlotMax && t < kDeltaSlotMax + S) {
        const int s = t - kDeltaSlotMax;
        float sa = 0.0f, sb = 0.0f, sqs = 0.0f, sks = 0.0f;
        for (int w = 0; w < waves; ++w) {
            sa += red[(4 * s + 0) * waves + w];
            sb += red[(4 * s + 1) * waves + w];
            sqs += red[(4 * s + 2) * waves + w];
            sks += red[(4 * s + 3) * waves + w];
        }
        // Matches the eager chain: the projection outputs and the sigmoid round to
        // the activation dtype, softplus and exp stay in fp32.
        const float beta = round_to<T>(1.0f / (1.0f + expf(-round_to<T>(sb))));
        const float aa = round_to<T>(sa) + dt_bias[h];
        const float sp = aa > 20.0f ? aa : log1pf(expf(aa));
        const float gd = expf(g_decay[h] * sp);
        sg[(pending + s) * 2 + 0] = gd;
        sg[(pending + s) * 2 + 1] = beta;
        ssum[(pending + s) * 2 + 0] = sqs;
        ssum[(pending + s) * 2 + 1] = sks;
        float* g = gates_buf + ((par * static_cast<int64_t>(B) + b) * kDeltaSlotMax + s) * Hv * 2 + h * 2;
        g[0] = gd;
        g[1] = beta;
        if (h % gqa == 0) {
            float* n = sumsq_buf + ((par * static_cast<int64_t>(B) + b) * kDeltaSlotMax + s) * Hk * 2 + hk * 2;
            n[0] = sqs;
            n[1] = sks;
        }
    }
    __syncthreads();
    // normalized q (t < DK) and k (DK <= t < 2 DK) rows, then v, for every replay slot and
    // current row
    if (t < 2 * DK) {
        const bool is_q = t < DK;
        const int tt = is_q ? t : t - DK;
        const int64_t src = (is_q ? 0 : key_dim) + static_cast<int64_t>(hk) * DK + tt;
        float* dst = is_q ? sq : sk;
        for (int i = 0; i < pending + S; ++i) {
            const T* buf = i < pending ? prev_qkv + i : cur_qkv + (i - pending);
            const float n = fmaxf(sqrtf(ssum[i * 2 + (is_q ? 0 : 1)]), 1e-12f);
            const float val = to_f<T>(buf[src * kDeltaSlotMax]);
            dst[i * DK + tt] = is_q ? val / n * scale : val / n;
        }
    }
    {
        const int64_t vsrc = 2 * static_cast<int64_t>(key_dim) + static_cast<int64_t>(h) * DV;
        for (int e = t; e < (pending + S) * DV; e += blockDim.x) {
            const int i = e / DV, cc = e - i * DV;
            const T* buf = i < pending ? prev_qkv + i : cur_qkv + (i - pending);
            sv[i * DV + cc] = to_f<T>(buf[(vsrc + cc) * kDeltaSlotMax]);
        }
    }
    __syncthreads();

    // one delta-rule step over shared row i; emit writes output row s
    auto step = [&](int i, bool emit, int s) {
        const float* __restrict__ skr = sk + i * DK + kk0;
        const float* __restrict__ sqr = sq + i * DK + kk0;
        const float g_s = sg[i * 2 + 0], b_s = sg[i * 2 + 1];
        float sgs[KPT];
        float kvm = 0.0f;
#pragma unroll
        for (int j = 0; j < KPT; ++j) {
            sgs[j] = st[j] * g_s;
            kvm = fmaf(skr[j], sgs[j], kvm);
        }
        part[r * DV + v] = kvm;
        __syncthreads();
        float kv_mem = 0.0f;
        for (int rr = 0; rr < R; ++rr) {
            kv_mem += part[rr * DV + v];
        }
        const float delta = (sv[i * DV + v] - kv_mem) * b_s;
        float o = 0.0f;
#pragma unroll
        for (int j = 0; j < KPT; ++j) {
            const float n = fmaf(skr[j], delta, sgs[j]);
            st[j] = n;
            o = fmaf(sqr[j], n, o);
        }
        __syncthreads();
        if (emit) {
            part[r * DV + v] = o;
            __syncthreads();
            float o_sum = 0.0f;
            for (int rr = 0; rr < R; ++rr) {
                o_sum += part[rr * DV + v];
            }
            // torch rms_norm (fp32 compute, one rounding) times a silu gate in T
            const float o_out = (r == 0) ? round_to<T>(o_sum) : 0.0f;
            const float ss = delta_block_sum(o_out * o_out, red, waves);
            if (r == 0) {
                const float rstd = rsqrtf(ss / static_cast<float>(DV) + eps);
                const float y = round_to<T>(o_out * rstd * to_f<T>(norm_w[v]));
                const int64_t orow = ((static_cast<int64_t>(b) * S + s) * Hv + h) * DV + v;
                const float zz = to_f<T>(z[orow]);
                const float gate = round_to<T>(zz / (1.0f + expf(-zz)));
                rope_store<T>(out, orow, y * gate);
            }
            __syncthreads();
        }
    };
    // replay the accepted tokens of the previous step and commit
    for (int i = 0; i < pending; ++i) {
        step(i, false, 0);
    }
#pragma unroll
    for (int j = 0; j < KPT; ++j) {
        state[state_off + static_cast<int64_t>(kk0 + j) * DV + v] = st[j];
    }
    // current tokens: outputs only, the state stays uncommitted until the next step
    for (int s = 0; s < S; ++s) {
        step(pending + s, true, s);
    }
}

inline size_t delta_deferred_shared_bytes(int S, int DV, int kpt, int threads) {
    const int r = kDeltaKeyDim / kpt;
    const int maxt = kDeltaSlotMax + S;
    return (2 * static_cast<size_t>(maxt) * kDeltaKeyDim + static_cast<size_t>(maxt) * DV +
            4 * static_cast<size_t>(maxt) + static_cast<size_t>(r) * DV +
            4 * static_cast<size_t>(S) * (threads / kWave)) *
           sizeof(float);
}

inline int delta_device_lds_limit() {
    int device = 0;
    int lds = 0;
    if (hipGetDevice(&device) != hipSuccess ||
        hipDeviceGetAttribute(&lds, hipDeviceAttributeMaxSharedMemoryPerBlock, device) !=
            hipSuccess) {
        return 0;
    }
    return lds;
}

// One thread per value column would need DK floats of state each, so the DK axis
// is split R ways. A narrower slice means more threads, and the workgroup size is
// the cap, which is why a wide DV has to take a coarser slice.
inline int delta_key_slice(int DV) {
    for (int kpt : {16, 32, 64}) {
        if (static_cast<int64_t>(DV) * (kDeltaKeyDim / kpt) <= 1024) {
            return kpt;
        }
    }
    return 0;
}

inline size_t delta_shared_bytes(int S, int DV, int kpt, int threads) {
    const int r = kDeltaKeyDim / kpt;
    return (2 * static_cast<size_t>(S) * kDeltaKeyDim + static_cast<size_t>(r) * DV +
            4 * static_cast<size_t>(S) * (threads / kWave)) *
           sizeof(float);
}

}  // namespace comfy::hip_backend

// Returns false for a shape outside the kernel's envelope so the caller can fall
// back rather than having to predict it.
extern "C" bool launch_gated_delta_decode_fused_kernel(
    const void* mixed_qkv, const void* x, const void* w_a, const void* w_b, const void* dt_bias,
    const void* g_decay, void* state, void* out, void* snapshots, const void* z,
    const void* norm_w, float eps, int B, int Hv, int Hk, int S, int DK, int DV, int C, int Hd,
    int key_dim, float scale, int dtype_code, hipStream_t stream) {

    using namespace comfy::hip_backend;
    if (DK != kDeltaKeyDim || DV <= 0 || DV > 512 || DV % kWave != 0 || S < 1 ||
        S > kDeltaMaxSteps || Hk <= 0 || Hv <= 0 || Hv % Hk != 0 || B <= 0 || dtype_code < 0 ||
        dtype_code > 2) {
        return false;
    }
    const int kpt = delta_key_slice(DV);
    if (kpt == 0) {
        return false;
    }
    const int threads = DV * (kDeltaKeyDim / kpt);
    const size_t shared = delta_shared_bytes(S, DV, kpt, threads);
    const int lds_limit = delta_device_lds_limit();
    if (lds_limit <= 0 || shared > static_cast<size_t>(lds_limit)) {
        return false;
    }

#define CK_DELTA_LAUNCH(T, KPT)                                                                  \
    gated_delta_decode_fused_kernel<T, KPT>                                                      \
        <<<static_cast<unsigned int>(B) * Hv, threads, shared, stream>>>(                        \
            static_cast<const T*>(mixed_qkv), static_cast<const T*>(x),                          \
            static_cast<const T*>(w_a), static_cast<const T*>(w_b),                              \
            static_cast<const float*>(dt_bias), static_cast<const float*>(g_decay),              \
            static_cast<float*>(state), static_cast<T*>(out), static_cast<float*>(snapshots),    \
            static_cast<const T*>(z), static_cast<const T*>(norm_w), eps, B, Hv, Hk, S, DV, C,   \
            Hd, key_dim, scale)
#define CK_DELTA_DISPATCH(T)     \
    if (kpt == 16) {             \
        CK_DELTA_LAUNCH(T, 16);  \
    } else if (kpt == 32) {      \
        CK_DELTA_LAUNCH(T, 32);  \
    } else {                     \
        CK_DELTA_LAUNCH(T, 64);  \
    }

    if (dtype_code == 0) {
        CK_DELTA_DISPATCH(float);
    } else if (dtype_code == 1) {
        CK_DELTA_DISPATCH(__half);
    } else {
        CK_DELTA_DISPATCH(__bf16);
    }
#undef CK_DELTA_DISPATCH
#undef CK_DELTA_LAUNCH
    return true;
}

extern "C" bool launch_deltanet_conv_step_kernel(
    const void* proj, void* conv_state, const void* conv_w, const void* conv_b, void* conv_out,
    void* conv_snaps, int B, int C, int S, int KS, int dtype_code, hipStream_t stream) {

    using namespace comfy::hip_backend;
    if (S < 1 || S > kDeltaMaxSteps || KS < 2 || KS > 8 || B <= 0 || C <= 0 || dtype_code < 0 ||
        dtype_code > 2) {
        return false;
    }
    // The window is held in registers, so the carried state plus this step's tokens
    // has to fit it.
    if (KS - 1 + S > kDeltaConvMaxWindow) {
        return false;
    }

    constexpr int threads = 256;
    const unsigned int blocks =
        static_cast<unsigned int>((static_cast<int64_t>(B) * C + threads - 1) / threads);

#define CK_CONV_STEP_LAUNCH(T)                                                \
    deltanet_conv_step_kernel<T><<<blocks, threads, 0, stream>>>(             \
        static_cast<const T*>(proj), static_cast<T*>(conv_state),             \
        static_cast<const T*>(conv_w), static_cast<const T*>(conv_b),         \
        static_cast<T*>(conv_out), static_cast<T*>(conv_snaps), B, C, S, KS)

    if (dtype_code == 0) {
        CK_CONV_STEP_LAUNCH(float);
    } else if (dtype_code == 1) {
        CK_CONV_STEP_LAUNCH(__half);
    } else {
        CK_CONV_STEP_LAUNCH(__bf16);
    }
#undef CK_CONV_STEP_LAUNCH
    return true;
}

extern "C" bool launch_gated_delta_decode_deferred_kernel(
    const void* x, const void* w_a, const void* w_b, const void* dt_bias, const void* g_decay,
    const void* qkv_buf, void* gates_buf, void* sumsq_buf, const void* ctl, void* state,
    void* out, const void* z, const void* norm_w, float eps, int B, int Hv, int Hk, int S, int DK,
    int DV, int C, int Hd, int key_dim, float scale, int dtype_code, hipStream_t stream) {

    using namespace comfy::hip_backend;
    // DV = DK = 128 is the shape the side buffers and Core serve; KPT = 16 then puts
    // the state in 1024 threads as the snapshot kernel does
    if (DK != kDeltaKeyDim || DV != kDeltaKeyDim || S < 1 || S > kDeltaSlotMax || Hk <= 0 ||
        Hv <= 0 || Hv % Hk != 0 || B <= 0 || dtype_code < 0 || dtype_code > 2) {
        return false;
    }
    constexpr int kpt = 16;
    const int threads = DV * (kDeltaKeyDim / kpt);
    const size_t shared = delta_deferred_shared_bytes(S, DV, kpt, threads);
    const int lds_limit = delta_device_lds_limit();
    if (lds_limit <= 0 || shared > static_cast<size_t>(lds_limit)) {
        return false;
    }

#define CK_DELTA_DEFERRED_LAUNCH(T)                                                            \
    gated_delta_decode_deferred_kernel<T, kpt>                                                 \
        <<<static_cast<unsigned int>(B) * Hv, threads, shared, stream>>>(                      \
            static_cast<const T*>(x), static_cast<const T*>(w_a), static_cast<const T*>(w_b),  \
            static_cast<const float*>(dt_bias), static_cast<const float*>(g_decay),            \
            static_cast<const T*>(qkv_buf), static_cast<float*>(gates_buf),                    \
            static_cast<float*>(sumsq_buf), static_cast<const int*>(ctl),                      \
            static_cast<float*>(state), static_cast<T*>(out), static_cast<const T*>(z),        \
            static_cast<const T*>(norm_w), eps, B, Hv, Hk, S, DV, C, Hd, key_dim, scale)

    if (dtype_code == 0) {
        CK_DELTA_DEFERRED_LAUNCH(float);
    } else if (dtype_code == 1) {
        CK_DELTA_DEFERRED_LAUNCH(__half);
    } else {
        CK_DELTA_DEFERRED_LAUNCH(__bf16);
    }
#undef CK_DELTA_DEFERRED_LAUNCH
    return true;
}

extern "C" bool launch_deltanet_conv_deferred_kernel(
    const void* proj, void* proj_buf, void* conv_state, const void* conv_w, const void* conv_b,
    void* qkv_buf, const void* ctl, int B, int C, int S, int KS, int dtype_code,
    hipStream_t stream) {

    using namespace comfy::hip_backend;
    if (S < 1 || S > kDeltaSlotMax || KS < 2 || KS > 8 || B <= 0 || C <= 0 || dtype_code < 0 ||
        dtype_code > 2 || KS - 1 + S > kDeltaConvMaxWindow) {
        return false;
    }
    constexpr int threads = 256;
    const unsigned int blocks =
        static_cast<unsigned int>((static_cast<int64_t>(B) * C + threads - 1) / threads);

#define CK_CONV_DEFERRED_LAUNCH(T)                                                      \
    deltanet_conv_deferred_kernel<T><<<blocks, threads, 0, stream>>>(                   \
        static_cast<const T*>(proj), static_cast<T*>(proj_buf), static_cast<T*>(conv_state), \
        static_cast<const T*>(conv_w), static_cast<const T*>(conv_b), static_cast<T*>(qkv_buf), \
        static_cast<const int*>(ctl), B, C, S, KS)

    if (dtype_code == 0) {
        CK_CONV_DEFERRED_LAUNCH(float);
    } else if (dtype_code == 1) {
        CK_CONV_DEFERRED_LAUNCH(__half);
    } else {
        CK_CONV_DEFERRED_LAUNCH(__bf16);
    }
#undef CK_CONV_DEFERRED_LAUNCH
    return true;
}
