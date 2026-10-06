// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// GatedDeltaNet decode for S <= 8 tokens with deferred commit for speculative
// decoding: the recurrent [DK, DV] fp32 state lives in registers on a 4-block
// cluster per (batch, head); the accepted tokens of the previous verify step are
// replayed from small side buffers before the state is written once.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cooperative_groups.h>
#include "../prefetch_ring.h"

__device__ PrefetchRingState* g_gated_delta_prefetch_ring = nullptr;

#include "float_utils.cuh"
#include "dtype_dispatch.cuh"

namespace {

template <typename T> __device__ __forceinline__ float to_f(T v);
template <> __device__ __forceinline__ float to_f<float>(float v) { return v; }
template <> __device__ __forceinline__ float to_f<__half>(__half v) { return __half2float(v); }
template <> __device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <typename T> __device__ __forceinline__ float round_to(float v) { return to_f<T>(comfy::from_float<T>(v)); }

// Two consecutive elements of a row (Hd even) in one 32-bit or 64-bit load.
template <typename T> __device__ __forceinline__ float2 load2(const T* p);
template <> __device__ __forceinline__ float2 load2<float>(const float* p) { return *reinterpret_cast<const float2*>(p); }
template <> __device__ __forceinline__ float2 load2<__half>(const __half* p) { return __half22float2(*reinterpret_cast<const __half2*>(p)); }
template <> __device__ __forceinline__ float2 load2<__nv_bfloat16>(const __nv_bfloat16* p) { return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(p)); }

constexpr int kSlotMax = 8;   // token slots per side buffer (S <= 8)
constexpr int kCtlInts = 2 + 3 * kSlotMax;   // {pending, parity, slot[8], parent[8], prog[8]}

// Deferred-commit DeltaNet decode. A verify step of S draft tokens never writes
// state speculatively: the conv/qkv output, projections, gates and q/k sums of
// squares of its tokens go to a double-buffered side buffer (parity ctl[1]), and
// the next step first replays the ctl[0] tokens of the previous step that the
// verifier accepted, commits the state once, then runs its own S tokens. The
// replayed arithmetic is the same per-token expressions as the direct path, so
// the committed state is bit-identical to having run the accepted tokens alone.
//
// ctl = {pending, parity, slot[kSlotMax], parent[kSlotMax]}: slot[i] is the side
// buffer slot of replay token i (the accepted rows of the previous step in path
// order) and parent[r] the row the current row r extends (-1: the committed
// prefix). A chain is parent[r] = r - 1; a verify tree is any forest of rows whose
// parent index is smaller than their own.

// depthwise causal conv step: one thread per (batch, channel) owns its window.
// conv_state holds the committed window; the window after the pending accepted
// tokens of the previous step is committed here, then the current S tokens are
// convolved on top of it, each over its own ancestor rows.
template <typename T>
__global__ void deltanet_conv_deferred_kernel(
    const T* __restrict__ proj,        // [B, S, C] projection output, row stride ldp
    T* __restrict__ proj_buf,          // [2, B, kSlotMax, C] projections of the previous / current step
    T* __restrict__ conv_state,        // [B, C, KS-1] committed window, updated in place
    const T* __restrict__ conv_w,      // [C, KS] depthwise taps
    const T* __restrict__ conv_b,      // [C] or nullptr
    T* __restrict__ qkv_buf,           // [2, B, C, kSlotMax] silu(conv) of the previous / current step
    const int* __restrict__ ctl,       // {pending, parity, slot[8], parent[8]}
    int B, int C, int S, int KS, int ldp)
{
    const int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= B * C)
        return;
    const int b = idx / C, c = idx - b * C;
    const int pending = ctl[0], par = ctl[1];
    const int* __restrict__ slot = ctl + 2;
    const int* __restrict__ parent = ctl + 2 + kSlotMax;
    const int L = KS - 1;
    constexpr int MAXW = 16;   // (KS - 1) + S <= 7 + 8
    const int64_t proj_slab = static_cast<int64_t>(B) * kSlotMax * C;
    const T* __restrict__ prev_proj = proj_buf + (1 - par) * proj_slab + (static_cast<int64_t>(b) * kSlotMax) * C + c;
    T* __restrict__ cur_proj = proj_buf + par * proj_slab + (static_cast<int64_t>(b) * kSlotMax) * C + c;
    T* __restrict__ state_row = conv_state + (static_cast<int64_t>(b) * C + c) * L;
    float win[MAXW];
    #pragma unroll
    for (int j = 0; j < MAXW; ++j) {
        if (j < L) {
            // committed window = the last L of [conv_state, prev_proj[slot[0..pending)]]
            const int i = pending + j;
            win[j] = i < L ? to_f<T>(state_row[i]) : to_f<T>(prev_proj[static_cast<int64_t>(slot[i - L]) * C]);
        } else if (j < L + S) {
            win[j] = to_f<T>(proj[(static_cast<int64_t>(b) * S + (j - L)) * ldp + c]);
        } else {
            win[j] = 0.0f;
        }
    }
    #pragma unroll
    for (int j = 0; j < MAXW; ++j)
        if (j < L)
            state_row[j] = comfy::from_float<T>(win[j]);
    // taps in distance order: w[d] multiplies the token d steps back from the row
    float w[8];
    #pragma unroll
    for (int d = 0; d < 8; ++d)
        w[d] = d < KS ? to_f<T>(conv_w[static_cast<int64_t>(c) * KS + (KS - 1 - d)]) : 0.0f;
    const float bias = conv_b != nullptr ? to_f<T>(conv_b[c]) : 0.0f;
    T* __restrict__ cur_qkv = qkv_buf + par * (static_cast<int64_t>(B) * C * kSlotMax) + (static_cast<int64_t>(b) * C + c) * kSlotMax;
    #pragma unroll
    for (int s = 0; s < kSlotMax; ++s) {
        if (s < S) {
            // row s convolves over its own ancestors: tap 0 is its own projection, the older
            // taps walk parent[] and then fall back to the committed window. The sum runs
            // oldest tap first, as a contiguous window does.
            float tap[8];
            tap[0] = win[L + s];
            int p = s, back = 0;
            #pragma unroll
            for (int d = 1; d < 8; ++d) {
                if (d < KS) {
                    p = p >= 0 ? parent[p] : -1;
                    tap[d] = p >= 0 ? win[L + p] : win[L - 1 - back++];
                }
            }
            float acc = 0.0f;
            #pragma unroll
            for (int d = 7; d >= 0; --d)
                if (d < KS)
                    acc = fmaf(w[d], tap[d], acc);
            const float y = round_to<T>(acc + bias);
            cur_qkv[s] = comfy::from_float<T>(y / (1.0f + expf(-y)));
            cur_proj[static_cast<int64_t>(s) * C] = comfy::from_float<T>(win[L + s]);
        }
    }
}

// Delta-rule steps and the gated RMSNorm for one (batch, head) on a cluster of
// 4 blocks x 256 threads. Block r owns value columns [32r, 32r+32); a warp owns 4
// columns and a lane one column x a 16-row kk slice held in registers, so the
// per-token column sum over kk is 16 fmas and 3 shuffles with no block barrier.
// The prologue folds the a/b gate projections (threads 0..127, the same 128-thread
// reduction for every block of the cluster) and normalizes q/k for the pending
// replay tokens and the S current tokens into shared memory, laid out so a lane's
// 16-row slice is 4 conflict-free 128-bit reads. The RMSNorm sum of squares is
// the only cross-block value: each block pushes its partial into every sibling's
// shared memory, one cluster barrier, then sums in fixed rank order.
// Tree: the S current tokens form a verify tree (parent[r] < r). ctl's program runs the
// rows in depth-first order; each entry says whether this row's update commits into the
// running state: an argmax-chain row commits, then the leaves hanging off it run against
// that state with commit = false. The state written to global memory is still only the
// committed prefix: no current row ever reaches it.
// the attribute itself is rejected by the pre-sm_90 device passes of a multi-arch build
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
#define CLUSTER_DIMS(cl)
#else
#define CLUSTER_DIMS(cl) __cluster_dims__(cl, 1, 1)
#endif

template <typename T, int DK, int DV, int S, bool Tree>
__global__ void CLUSTER_DIMS(4) __launch_bounds__(256, 2)
gated_delta_decode_deferred_kernel(
    const T* __restrict__ x,           // [B, S, Hd]
    const T* __restrict__ w_a,         // [Hv, Hd]
    const T* __restrict__ w_b,         // [Hv, Hd]
    const float* __restrict__ dt_bias, // [Hv]
    const float* __restrict__ g_decay, // [Hv]
    const T* __restrict__ qkv_buf,     // [2, B, C, kSlotMax] conv+silu output, previous / current step
    float* __restrict__ gates_buf,     // [2, B, kSlotMax, Hv, 2]: decay g, beta
    float* __restrict__ sumsq_buf,     // [2, B, kSlotMax, Hk, 2]: sum q^2, sum k^2
    const int* __restrict__ ctl,       // {pending, parity, slot[8], parent[8], prog[8]}
    float* __restrict__ state,         // [B, Hv, DK, DV] committed state, updated in place
    T* __restrict__ out,               // [B, S, Hv, DV]
    const T* __restrict__ z,           // [B, S, Hv*DV] norm gate, row stride ldz
    const T* __restrict__ norm_w,      // [DV]
    float eps,
    int B, int Hv, int Hk, int C, int Hd, int key_dim, float scale, int ldz)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    constexpr int NT = 256, NW = NT / 32, CL = 4;
    constexpr int COLS = DV / CL;           // columns per block
    constexpr int RG = 8;                   // row groups per warp (lane >> 2)
    constexpr int KPL = DK / RG;            // rows per lane
    constexpr int MAXT = kSlotMax + S;      // replay slots + current tokens
    constexpr int GNT = 128, GNW = GNT / 32;   // gate projection threads / warps
    static_assert(COLS == NW * 4 && KPL == 16 && DK == 128, "kernel is shaped for DK = DV = 128");
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    const int rank = static_cast<int>(cluster.block_rank());
    __shared__ __align__(16) float sk[MAXT * DK];   // [i][j][rg][e]: row = 16 rg + 4 j + e
    __shared__ __align__(16) float sq[MAXT * DK];
    __shared__ float sv[MAXT][COLS];         // v for this block's columns
    __shared__ float sg[MAXT][2];            // decay g, beta per token
    __shared__ float ssum[MAXT][2];          // sum q^2, sum k^2 per token
    __shared__ float red[4 * S][GNW];
    __shared__ float oc[S][COLS];            // bf16-rounded column outputs
    __shared__ float xch[CL][S];             // sibling sums of squares, by rank
    const int head = static_cast<int>(blockIdx.x) / CL;
    const int b = head / Hv;
    const int h = head % Hv;
    const int t = threadIdx.x;
    const int lane = t & 31, wid = t >> 5;
    const int col = lane & 3, rg = lane >> 2;
    const int gc = rank * COLS + wid * 4 + col;   // this lane's value column
    const int gqa = Hv / Hk;
    const int hk = h / gqa;
    const int pending = ctl[0], par = ctl[1];
    const int* __restrict__ prev_slots = ctl + 2;       // side-buffer slots of the replay tokens
    const int* __restrict__ prog = ctl + 2 + 2 * kSlotMax;
    auto prev_slot = [&](int i) { return prev_slots[i]; };
    const int64_t qkv_slab = static_cast<int64_t>(B) * C * kSlotMax;
    const T* __restrict__ prev_qkv = qkv_buf + (1 - par) * qkv_slab + static_cast<int64_t>(b) * C * kSlotMax;
    const T* __restrict__ cur_qkv = qkv_buf + par * qkv_slab + static_cast<int64_t>(b) * C * kSlotMax;
    const int64_t state_off = (static_cast<int64_t>(b) * Hv + h) * DK * DV;
    const int64_t tile_off = static_cast<int64_t>(rg * KPL) * DV + gc;   // this lane's first state element within the head

    float st[KPL];
    #pragma unroll
    for (int r = 0; r < KPL; ++r)
        st[r] = state[state_off + tile_off + r * DV];
    // ring consumption: this block's [DK, COLS] fp32 tile of the state, when the host listed it
    if (t == 0 && g_gated_delta_prefetch_ring != nullptr && (g_gated_delta_prefetch_ring->credits & PREFETCH_RING_CREDIT_DELTA))
        prefetch_ring_consume_device(g_gated_delta_prefetch_ring, static_cast<uint64_t>(DK) * COLS * sizeof(float));

    // gate projections of the current tokens (bf16 projection outputs, bf16 sigmoid,
    // fp32 softplus/exp, as in the eager chain) and the q/k sums of squares
    if (t < GNT) {
        float acc[4 * S];
        #pragma unroll
        for (int i = 0; i < 4 * S; ++i)
            acc[i] = 0.0f;
        const T* __restrict__ wa = w_a + static_cast<int64_t>(h) * Hd;
        const T* __restrict__ wb = w_b + static_cast<int64_t>(h) * Hd;
        const T* __restrict__ xb = x + static_cast<int64_t>(b) * S * Hd;
        #pragma unroll 4
        for (int i = 2 * t; i < Hd; i += 2 * GNT) {
            const float2 wav = load2<T>(wa + i);
            const float2 wbv = load2<T>(wb + i);
            #pragma unroll
            for (int s = 0; s < S; ++s) {
                const float2 xv = load2<T>(xb + static_cast<int64_t>(s) * Hd + i);
                acc[s] = fmaf(xv.x, wav.x, acc[s]);
                acc[S + s] = fmaf(xv.x, wbv.x, acc[S + s]);
                acc[s] = fmaf(xv.y, wav.y, acc[s]);
                acc[S + s] = fmaf(xv.y, wbv.y, acc[S + s]);
            }
        }
        const int64_t qrow = static_cast<int64_t>(hk) * DK + t;
        #pragma unroll
        for (int s = 0; s < S; ++s) {
            const float qv = to_f<T>(cur_qkv[qrow * kSlotMax + s]);
            const float kv = to_f<T>(cur_qkv[(qrow + key_dim) * kSlotMax + s]);
            acc[2 * S + s] = qv * qv;
            acc[3 * S + s] = kv * kv;
        }
        #pragma unroll
        for (int i = 0; i < 4 * S; ++i) {
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                acc[i] += __shfl_xor_sync(0xffffffffu, acc[i], o);
            if (lane == 0)
                red[i][wid] = acc[i];
        }
    }
    // gates and sums of squares of the previous step's pending tokens
    if (t >= GNT && t - GNT < kSlotMax) {
        const int i = t - GNT;
        if (i < pending) {
            const float2 g = *reinterpret_cast<const float2*>(
                gates_buf + (((1 - par) * static_cast<int64_t>(B) + b) * kSlotMax + prev_slot(i)) * Hv * 2 + h * 2);
            const float2 n = *reinterpret_cast<const float2*>(
                sumsq_buf + (((1 - par) * static_cast<int64_t>(B) + b) * kSlotMax + prev_slot(i)) * Hk * 2 + hk * 2);
            sg[i][0] = g.x;
            sg[i][1] = g.y;
            ssum[i][0] = n.x;
            ssum[i][1] = n.y;
        }
    }
    __syncthreads();
    if (t < 2 * S) {
        // thread s: gate pair of token s; thread S + s: norm pair of token s
        const int s = t < S ? t : t - S;
        const int base = t < S ? 0 : 2 * S;
        float u = 0.0f, w = 0.0f;
        #pragma unroll
        for (int i = 0; i < GNW; ++i) {
            u += red[base + s][i];
            w += red[base + S + s][i];
        }
        if (t < S) {
            const float beta = round_to<T>(1.0f / (1.0f + expf(-round_to<T>(w))));
            const float aa = round_to<T>(u) + dt_bias[h];
            const float sp = aa > 20.0f ? aa : log1pf(expf(aa));
            const float gd = expf(g_decay[h] * sp);
            sg[pending + s][0] = gd;
            sg[pending + s][1] = beta;
            if (rank == 0) {
                float* g = gates_buf + ((par * static_cast<int64_t>(B) + b) * kSlotMax + s) * Hv * 2 + h * 2;
                g[0] = gd;
                g[1] = beta;
            }
        } else {
            ssum[pending + s][0] = u;
            ssum[pending + s][1] = w;
            if (rank == 0 && h % gqa == 0) {
                float* n = sumsq_buf + ((par * static_cast<int64_t>(B) + b) * kSlotMax + s) * Hk * 2 + hk * 2;
                n[0] = u;
                n[1] = w;
            }
        }
    }
    __syncthreads();
    {
        // thread t normalizes row 16 rg + 4 j + e of k (t < 128) or q, written at
        // shared index i*DK + t so the store is conflict-free
        const int tt = t & (DK - 1);
        const int row = 16 * ((tt >> 2) & 7) + 4 * (tt >> 5) + (tt & 3);
        const bool is_q = t >= DK;
        const int64_t src = (is_q ? 0 : key_dim) + static_cast<int64_t>(hk) * DK + row;
        float* dst = is_q ? sq : sk;
        #pragma unroll
        for (int i = 0; i < MAXT; ++i) {
            if (i < pending + S) {
                const T* buf = i < pending ? prev_qkv + prev_slot(i) : cur_qkv + (i - pending);
                const float ss = ssum[i][is_q ? 0 : 1];
                const float n = fmaxf(sqrtf(ss), 1e-12f);
                const float v = to_f<T>(buf[src * kSlotMax]);
                dst[i * DK + tt] = is_q ? v / n * scale : v / n;
            }
        }
        const int64_t vsrc = 2 * static_cast<int64_t>(key_dim) + static_cast<int64_t>(h) * DV + rank * COLS;
        for (int e = t; e < MAXT * COLS; e += NT) {
            const int i = e / COLS, cc = e - i * COLS;
            if (i < pending + S) {
                const T* buf = i < pending ? prev_qkv + prev_slot(i) : cur_qkv + (i - pending);
                sv[i][cc] = to_f<T>(buf[(vsrc + cc) * kSlotMax]);
            }
        }
    }
    __syncthreads();

    // commit=false: a sibling's output from the same pre-state, leaving st untouched
    auto step = [&](int i, bool emit, int s, bool commit = true) {
        const float4* __restrict__ skr = reinterpret_cast<const float4*>(sk + i * DK) + rg;
        const float4* __restrict__ sqr = reinterpret_cast<const float4*>(sq + i * DK) + rg;
        const float g_s = sg[i][0], b_s = sg[i][1];
        float k[KPL], q[KPL], sgs[KPL];
        #pragma unroll
        for (int j = 0; j < KPL / 4; ++j) {
            const float4 k4 = skr[j * RG], q4 = sqr[j * RG];
            k[4 * j + 0] = k4.x; k[4 * j + 1] = k4.y; k[4 * j + 2] = k4.z; k[4 * j + 3] = k4.w;
            q[4 * j + 0] = q4.x; q[4 * j + 1] = q4.y; q[4 * j + 2] = q4.z; q[4 * j + 3] = q4.w;
        }
        float kvm = 0.0f;
        #pragma unroll
        for (int r = 0; r < KPL; ++r) {
            sgs[r] = st[r] * g_s;
            kvm = fmaf(k[r], sgs[r], kvm);
        }
        kvm += __shfl_xor_sync(0xffffffffu, kvm, 4);
        kvm += __shfl_xor_sync(0xffffffffu, kvm, 8);
        kvm += __shfl_xor_sync(0xffffffffu, kvm, 16);
        const float delta = (sv[i][wid * 4 + col] - kvm) * b_s;
        float o = 0.0f;
        #pragma unroll
        for (int r = 0; r < KPL; ++r) {
            const float n = fmaf(k[r], delta, sgs[r]);
            if (commit) st[r] = n;
            o = fmaf(q[r], n, o);
        }
        if (emit) {
            o += __shfl_xor_sync(0xffffffffu, o, 4);
            o += __shfl_xor_sync(0xffffffffu, o, 8);
            o += __shfl_xor_sync(0xffffffffu, o, 16);
            if (rg == 0)
                oc[s][wid * 4 + col] = round_to<T>(o);
        }
    };
    // replay the accepted tokens of the previous step and commit
    for (int i = 0; i < pending; ++i)
        step(i, false, 0);
    #pragma unroll
    for (int r = 0; r < KPL; ++r)
        state[state_off + tile_off + r * DV] = st[r];
    // current tokens: outputs only, state stays uncommitted until the next step
    if constexpr (Tree) {
        // prog[e] = row | commit << 5, in depth-first order
        for (int e = 0; e < S; ++e) {
            const int p = prog[e];
            step(pending + (p & 15), true, p & 15, (p & 32) != 0);
        }
    } else {
        #pragma unroll
        for (int s = 0; s < S; ++s)
            step(pending + s, true, s);
    }
    __syncthreads();

    // warp s finishes token s: block sum of squares to every sibling, then the norm
    float o_out = 0.0f;
    if (wid < S) {
        o_out = oc[wid][lane];
        float ss = o_out * o_out;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            ss += __shfl_xor_sync(0xffffffffu, ss, o);
        if (lane < CL)
            cluster.map_shared_rank(&xch[rank][wid], lane)[0] = ss;
    }
    cluster.sync();
    if (wid < S) {
        // torch rms_norm (fp32 compute, one rounding) times a bf16 silu gate
        float ss = 0.0f;
        #pragma unroll
        for (int r = 0; r < CL; ++r)
            ss += xch[r][wid];
        const float rstd = rsqrtf(ss / static_cast<float>(DV) + eps);
        const int oc_col = rank * COLS + lane;
        const float y = round_to<T>(o_out * rstd * to_f<T>(norm_w[oc_col]));
        const int64_t orow = ((static_cast<int64_t>(b) * S + wid) * Hv + h) * DV + oc_col;
        const float zz = to_f<T>(z[(static_cast<int64_t>(b) * S + wid) * ldz + h * DV + oc_col]);
        const float gate = round_to<T>(zz / (1.0f + expf(-zz)));
        out[orow] = comfy::from_float<T>(round_to<T>(y * gate));
    }
#endif
}

}  // namespace

extern "C" void set_gated_delta_prefetch_ring_state(PrefetchRingState* state) {
    cudaMemcpyToSymbol(g_gated_delta_prefetch_ring, &state, sizeof(state));
}

extern "C" bool launch_gated_delta_decode_deferred(
    const void* x, const void* w_a, const void* w_b, const void* dt_bias, const void* g_decay,
    const void* qkv_buf, void* gates_buf, void* sumsq_buf, const void* ctl,
    void* state, void* out, const void* z, const void* norm_w, float eps,
    int64_t B, int64_t Hv, int64_t Hk, int64_t S, int64_t DK, int64_t DV, int64_t C, int64_t Hd,
    int64_t key_dim, float scale, int64_t ldz, int dtype_code, int64_t tree, cudaStream_t stream)
{
    if (DK != 128 || DV != 128 || S < 1 || S > kSlotMax || Hk <= 0 || Hv % Hk != 0 || Hd % 8 != 0
            || dtype_code < 0 || dtype_code > 2)
        return false;
    int device = 0, cc_major = 0;
    if (cudaGetDevice(&device) != cudaSuccess
            || cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess
            || cc_major < 9)
        return false;
    return DISPATCH_FP_DTYPE(dtype_code, T, [&] {
        auto launch = [&]<int Steps, bool IsTree = false>() {
            gated_delta_decode_deferred_kernel<T, 128, 128, Steps, IsTree><<<static_cast<unsigned>(B * Hv * 4), 256, 0, stream>>>(
                static_cast<const T*>(x), static_cast<const T*>(w_a), static_cast<const T*>(w_b),
                static_cast<const float*>(dt_bias), static_cast<const float*>(g_decay),
                static_cast<const T*>(qkv_buf), static_cast<float*>(gates_buf), static_cast<float*>(sumsq_buf),
                static_cast<const int*>(ctl), static_cast<float*>(state), static_cast<T*>(out),
                static_cast<const T*>(z), static_cast<const T*>(norm_w), eps,
                static_cast<int>(B), static_cast<int>(Hv), static_cast<int>(Hk),
                static_cast<int>(C), static_cast<int>(Hd), static_cast<int>(key_dim), scale, static_cast<int>(ldz));
        };
        if (tree != 0) {
            // tree verify: the ctl program decides the schedule, S only sizes the kernel
            switch (S) {
                case 3: launch.template operator()<3, true>(); break;
                case 4: launch.template operator()<4, true>(); break;
                case 5: launch.template operator()<5, true>(); break;
                case 6: launch.template operator()<6, true>(); break;
                case 7: launch.template operator()<7, true>(); break;
                case 8: launch.template operator()<8, true>(); break;
                default: return false;
            }
            return cudaGetLastError() == cudaSuccess;
        }
        switch (S) {
            case 1: launch.template operator()<1>(); break;
            case 2: launch.template operator()<2>(); break;
            case 3: launch.template operator()<3>(); break;
            case 4: launch.template operator()<4>(); break;
            case 5: launch.template operator()<5>(); break;
            case 6: launch.template operator()<6>(); break;
            case 7: launch.template operator()<7>(); break;
            default: launch.template operator()<8>(); break;
        }
        return cudaGetLastError() == cudaSuccess;
    });
}

extern "C" bool launch_deltanet_conv_deferred(
    const void* proj, void* proj_buf, void* conv_state, const void* conv_w, const void* conv_b,
    void* qkv_buf, const void* ctl,
    int64_t B, int64_t C, int64_t S, int64_t KS, int64_t ldp, int dtype_code, cudaStream_t stream)
{
    if (S < 1 || S > kSlotMax || KS < 2 || KS > 8 || dtype_code < 0 || dtype_code > 2)
        return false;
    const int threads = 256;
    const unsigned grid = static_cast<unsigned>((B * C + threads - 1) / threads);
    return DISPATCH_FP_DTYPE(dtype_code, T, [&] {
        deltanet_conv_deferred_kernel<T><<<grid, threads, 0, stream>>>(
            static_cast<const T*>(proj), static_cast<T*>(proj_buf), static_cast<T*>(conv_state),
            static_cast<const T*>(conv_w), static_cast<const T*>(conv_b), static_cast<T*>(qkv_buf),
            static_cast<const int*>(ctl),
            static_cast<int>(B), static_cast<int>(C), static_cast<int>(S), static_cast<int>(KS), static_cast<int>(ldp));
        return cudaGetLastError() == cudaSuccess;
    });
}
