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

// Fused 3D neighborhood attention (NATTEN na3d semantics, dilation 1) and
// window attention over a stack of key planes.
//
// na3d: per non-causal axis each query attends a window of exactly
// kernel_size positions centered on it, shifted inward at grid boundaries;
// per causal axis it attends the min(i+1, kernel_size) nearest previous
// positions.
//
// window_attention: the Tq query planes are the last of the Tk key planes and
// each attends its temporal_window most recent key planes. Every key plane has
// its own (kh, kw) spatial window, shifted inward at the frame border (clamp)
// or cut off by it. K/V may have fewer heads than Q (grouped-query attention).
//
// Flash-attention-2 style: scores, softmax stats, P and the accumulator stay
// in registers; K and V tiles are double-buffered in shared memory with
// cp.async, V fragments come transposed out of the row-major tile.
//  - Row kernel: one block covers 16..64 queries along W at a fixed (t, h),
//    16 per warp. T/H windows are scalar per block, so only W needs
//    per-element masking, and n8 key tiles outside a warp's span are skipped.
//  - Patch kernel, for four query heads sharing a kv head: one warp per head
//    over the same 4x4 query patch, all reading one staged key list, the
//    flattened union of the patch's windows.
//
// Requires sm80+ (bf16 mma / f32 accumulators). head_dim in {16,32,48,64,128}.
// Layout: q/out are contiguous (B, Tq, H, W, NH, HD), k/v (B, Tk, H, W, KVH, HD).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

constexpr int MAX_WINDOWS = 8; // distinct per-plane spatial windows
constexpr float NEG_INF = -3.0e38f;
constexpr int NO_KEY = -(1 << 20); // coordinate of a padding key, outside every window

// Keys per staged tile; 32 at head_dim 128 keeps the K/V double buffer under 48 KB.
__host__ __device__ constexpr int keys_for(int hd) { return hd > 64 ? 32 : 64; }

__host__ __device__ inline int imin(int a, int b) { return a < b ? a : b; }
__host__ __device__ inline int imax(int a, int b) { return a > b ? a : b; }

// Window [lo, hi) of index i along one axis. A non-causal window is shifted
// inward at the border when clamp is set (NATTEN), otherwise cut off by it.
__device__ inline void axis_window(int i, int k, int len, bool causal, bool clamp, int& lo, int& hi) {
    if (causal) {
        lo = imax(i - k + 1, 0);
        hi = i + 1;
    } else if (clamp) {
        lo = imin(imax(i - k / 2, 0), len - k);
        hi = lo + k;
    } else {
        lo = imax(i - k / 2, 0);
        hi = imin(i - k / 2 + k, len);
    }
}

template <typename T> struct MmaTraits;

template <> struct MmaTraits<__half> {
    static __device__ inline void mma(float* d, const uint32_t* a, const uint32_t* b) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
    static __device__ inline uint32_t pack(float lo, float hi) {
        __half2 p = __floats2half2_rn(lo, hi);
        return *reinterpret_cast<uint32_t*>(&p);
    }
};

template <> struct MmaTraits<__nv_bfloat16> {
    static __device__ inline void mma(float* d, const uint32_t* a, const uint32_t* b) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
    static __device__ inline uint32_t pack(float lo, float hi) {
        __nv_bfloat162 p = __floats2bfloat162_rn(lo, hi);
        return *reinterpret_cast<uint32_t*>(&p);
    }
};

// 16-byte global -> shared copy, zero-filled when !valid.
__device__ inline void cp_async16(void* smem, const void* gmem, bool valid) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(s), "l"(gmem), "r"(valid ? 16 : 0));
}
__device__ inline void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ inline void cp_async_wait_one() { asm volatile("cp.async.wait_group 1;\n" ::); }

__device__ inline void ldmatrix_x4_trans(uint32_t* r, const void* smem) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}

// Query plane t attends the key planes in the T-axis window of t + t_offset
// over tk planes; key plane p uses spatial window (kh, kw)[n_windows == 1 ? 0 : p].
struct Params {
    const void* k;
    const void* v;
    int tq, tk, h_size, w_size, num_heads, group;  // group: q heads per kv head
    int64_t q_b, q_t, q_h, q_w, q_n;               // q / out strides
    int64_t k_b, k_t, k_h, k_w, k_n;               // k / v strides
    int kt, t_offset;
    bool causal_t, causal_h, causal_w, clamp;
    int n_windows;
    int kh[MAX_WINDOWS], kw[MAX_WINDOWS];
    float scale;
};

__device__ inline int window_index(const Params& p, int t) { return p.n_windows == 1 ? 0 : t; }

// K and V of key plane t for one batch entry and kv head.
template <typename T>
__device__ inline void plane_kv(const Params& p, int t, int batch, int kv_head, const T*& kp, const T*& vp) {
    const int64_t off = (int64_t)batch * p.k_b + (int64_t)t * p.k_t + (int64_t)kv_head * p.k_n;
    kp = static_cast<const T*>(p.k) + off;
    vp = static_cast<const T*>(p.v) + off;
}

// mma.m16n8k16 fragment maps (PTX ISA), lane = g*4 + q (g = lane>>2, q = lane&3):
//   A (16x16):  a0=(g, 2q..2q+1) a1=(g+8, 2q..2q+1) a2=(g, 2q+8..2q+9) a3=(g+8, 2q+8..2q+9)
//   B (16x8):   b0=(k=2q..2q+1, n=g) b1=(k=2q+8..2q+9, n=g)   (packed along k)
//   C (16x8):   c0=(g, 2q) c1=(g, 2q+1) c2=(g+8, 2q) c3=(g+8, 2q+1)   (f32)

// Q operand fragments of this thread's rows g (row0) and g+8 (row1), loaded
// once and register-resident for the whole kernel.
template <typename T, int HD>
__device__ inline void load_q(const T* __restrict__ q, int64_t row0, int64_t row1, int qd, uint32_t (&qa)[HD / 16][4]) {
    #pragma unroll
    for (int kc = 0; kc < HD / 16; ++kc) {
        const int c0 = kc * 16 + qd * 2;
        qa[kc][0] = *reinterpret_cast<const uint32_t*>(q + row0 + c0);
        qa[kc][1] = *reinterpret_cast<const uint32_t*>(q + row1 + c0);
        qa[kc][2] = *reinterpret_cast<const uint32_t*>(q + row0 + c0 + 8);
        qa[kc][3] = *reinterpret_cast<const uint32_t*>(q + row1 + c0 + 8);
    }
}

// One staged key tile for one warp: S = Q.K^T over the active n8 tiles, masked
// online softmax, O += P.V. visible(row, key) says whether this thread's row
// (0: g, 1: g+8) sees key `key` of the tile.
template <typename T, int HD, int BK, typename Visible>
__device__ inline void attend_tile(const T* tK, const T* tV, const uint32_t (&qa)[HD / 16][4], const bool (&active)[BK / 8],
                                   const Visible& visible, float scale, float (&m_r)[2], float (&l_r)[2],
                                   float (&o_acc)[HD / 8][4], int lane) {
    constexpr int TILES = BK / 8;
    constexpr int NT = HD / 8;
    constexpr int LD = HD + 8;
    const int g = lane >> 2;
    const int qd = lane & 3;

    float s_acc[TILES][4];
    #pragma unroll
    for (int nt = 0; nt < TILES; ++nt) {
        s_acc[nt][0] = 0.f; s_acc[nt][1] = 0.f; s_acc[nt][2] = 0.f; s_acc[nt][3] = 0.f;
        if (active[nt]) {
            const T* krow = tK + (nt * 8 + g) * LD;
            #pragma unroll
            for (int kc = 0; kc < HD / 16; ++kc) {
                uint32_t kb[2];
                kb[0] = *reinterpret_cast<const uint32_t*>(krow + kc * 16 + qd * 2);
                kb[1] = *reinterpret_cast<const uint32_t*>(krow + kc * 16 + qd * 2 + 8);
                MmaTraits<T>::mma(s_acc[nt], qa[kc], kb);
            }
        }
    }

    // Masked online softmax, entirely in registers. This thread's score
    // columns are nt*8 + 2*qd (+1) of the tile for rows g / g+8.
    float m_new[2] = {m_r[0], m_r[1]};
    #pragma unroll
    for (int nt = 0; nt < TILES; ++nt) {
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int row = e >> 1;          // c0,c1 -> row g; c2,c3 -> row g+8
            const bool vis = active[nt] && visible(row, nt * 8 + qd * 2 + (e & 1));
            const float s = vis ? s_acc[nt][e] * scale : NEG_INF;
            s_acc[nt][e] = s;
            m_new[row] = fmaxf(m_new[row], s);
        }
    }
    // Row max across the quad (a row's columns live in 4 lanes).
    #pragma unroll
    for (int off = 1; off <= 2; off <<= 1) {
        m_new[0] = fmaxf(m_new[0], __shfl_xor_sync(0xffffffffu, m_new[0], off));
        m_new[1] = fmaxf(m_new[1], __shfl_xor_sync(0xffffffffu, m_new[1], off));
    }
    const float alpha0 = __expf(m_r[0] - m_new[0]);
    const float alpha1 = __expf(m_r[1] - m_new[1]);
    m_r[0] = m_new[0];
    m_r[1] = m_new[1];
    float l_add[2] = {0.f, 0.f};
    #pragma unroll
    for (int nt = 0; nt < TILES; ++nt) {
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int row = e >> 1;
            const float pv = (s_acc[nt][e] <= NEG_INF) ? 0.f : __expf(s_acc[nt][e] - m_new[row]);
            s_acc[nt][e] = pv;
            l_add[row] += pv;
        }
    }
    #pragma unroll
    for (int off = 1; off <= 2; off <<= 1) {
        l_add[0] += __shfl_xor_sync(0xffffffffu, l_add[0], off);
        l_add[1] += __shfl_xor_sync(0xffffffffu, l_add[1], off);
    }
    l_r[0] = l_r[0] * alpha0 + l_add[0];
    l_r[1] = l_r[1] * alpha1 + l_add[1];

    // Rescale output accumulators (c0,c1 -> row g; c2,c3 -> g+8).
    #pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        o_acc[nt][0] *= alpha0; o_acc[nt][1] *= alpha0;
        o_acc[nt][2] *= alpha1; o_acc[nt][3] *= alpha1;
    }

    // O += P.V: P is a register repack of the score tile, V fragments come
    // transposed out of the row-major tile. k-step kk covers keys kk*16..+15
    // = score tiles 2kk, 2kk+1; the lane picks ldmatrix matrix lane>>3.
    const int v_key = (lane & 7) + ((lane >> 3) & 1) * 8;
    const int v_dim = (lane >> 4) * 8;
    #pragma unroll
    for (int kk = 0; kk < TILES / 2; ++kk) {
        if (!active[2 * kk] && !active[2 * kk + 1])
            continue;
        uint32_t pa[4];
        pa[0] = MmaTraits<T>::pack(s_acc[2 * kk][0], s_acc[2 * kk][1]);
        pa[1] = MmaTraits<T>::pack(s_acc[2 * kk][2], s_acc[2 * kk][3]);
        pa[2] = MmaTraits<T>::pack(s_acc[2 * kk + 1][0], s_acc[2 * kk + 1][1]);
        pa[3] = MmaTraits<T>::pack(s_acc[2 * kk + 1][2], s_acc[2 * kk + 1][3]);
        const T* vrow = tV + (kk * 16 + v_key) * LD + v_dim;
        #pragma unroll
        for (int np = 0; np < NT / 2; ++np) {
            uint32_t vb[4];
            ldmatrix_x4_trans(vb, vrow + np * 16);
            MmaTraits<T>::mma(o_acc[2 * np], pa, vb);
            MmaTraits<T>::mma(o_acc[2 * np + 1], pa, vb + 2);
        }
    }
}

// Epilogue: divide by l and store this thread's elements of rows g and g+8.
template <typename T, int HD>
__device__ inline void store_rows(T* __restrict__ out, const int64_t (&row)[2], const bool (&valid)[2],
                                  const float (&o_acc)[HD / 8][4], const float (&l_r)[2], int qd) {
    #pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
        if (!valid[rr])
            continue;
        const float inv_l = 1.f / fmaxf(l_r[rr], 1e-30f);
        #pragma unroll
        for (int nt = 0; nt < HD / 8; ++nt) {
            const int c = nt * 8 + qd * 2;
            *reinterpret_cast<uint32_t*>(out + row[rr] + c) =
                MmaTraits<T>::pack(o_acc[nt][rr * 2] * inv_l, o_acc[nt][rr * 2 + 1] * inv_l);
        }
    }
}

// ---------------------------------------------------------------------------
// Row kernel
// ---------------------------------------------------------------------------

// Position of the key tile sweep: key plane t, key row h, tile c along W.
struct Sweep {
    int t, h, c, h_hi, w_lo, w_hi;
};

// Key rows and the W range the block's queries w_first..w_last touch in plane t.
__device__ inline void enter_plane(const Params& p, int t, int h_q, int w_first, int w_last, Sweep& s) {
    const int window = window_index(p, t);
    int unused;
    axis_window(h_q, p.kh[window], p.h_size, p.causal_h, p.clamp, s.h, s.h_hi);
    axis_window(w_first, p.kw[window], p.w_size, p.causal_w, p.clamp, s.w_lo, unused);
    axis_window(w_last, p.kw[window], p.w_size, p.causal_w, p.clamp, unused, s.w_hi);
    s.t = t;
    s.c = 0;
}

template <int BK>
__device__ inline bool advance(const Params& p, int t_hi, int h_q, int w_first, int w_last, Sweep& s) {
    if (s.w_lo + (s.c + 1) * BK < s.w_hi) {
        ++s.c;
        return true;
    }
    s.c = 0;
    if (++s.h < s.h_hi)
        return true;
    if (s.t + 1 < t_hi) {
        enter_plane(p, s.t + 1, h_q, w_first, w_last, s);
        return true;
    }
    return false;
}

template <typename T, int HD, int BK, int NTHREADS>
__device__ inline void load_tile(T* sK, T* sV, const Params& p, int batch, int kv_head, const Sweep& s, int tid) {
    constexpr int LD = HD + 8;
    constexpr int VECS = HD / 8;
    const T* k;
    const T* v;
    plane_kv(p, s.t, batch, kv_head, k, v);
    const int wk0 = s.w_lo + s.c * BK;
    const int n_valid = imin(BK, s.w_hi - wk0);
    const int64_t row = (int64_t)s.h * p.k_h + (int64_t)wk0 * p.k_w;
    for (int idx = tid; idx < BK * VECS; idx += NTHREADS) {
        const int r = idx / VECS, c8 = idx % VECS;
        const bool valid = r < n_valid;
        const int64_t off = valid ? row + (int64_t)r * p.k_w + c8 * 8 : 0;
        cp_async16(sK + r * LD + c8 * 8, k + off, valid);
        cp_async16(sV + r * LD + c8 * 8, v + off, valid);
    }
}

template <typename T, int HD, int WQ>
__global__ void __launch_bounds__(WQ * 32) na3d_kernel(
    const T* __restrict__ q,
    T* __restrict__ out,
    const Params p)
{
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    constexpr int BQ = WQ * 16;    // queries per block along W
    // Narrow blocks sweep about 16 + kw keys per row: a 32-key tile covers
    // that, a longer one would stage mostly padding.
    constexpr int BK = WQ == 4 ? keys_for(HD) : 32;
    constexpr int TILES = BK / 8;  // n8 score tiles per key tile
    constexpr int NT = HD / 8;
    constexpr int LD = HD + 8;     // sK / sV row stride (elements)
    constexpr int NTHREADS = WQ * 32;

    __shared__ alignas(16) T sK[2][BK * LD];
    __shared__ alignas(16) T sV[2][BK * LD];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;    // fragment row within m16 tiles
    const int qd = lane & 3;    // quad thread: column pairs

    const int w_size = p.w_size;
    const int t_q = blockIdx.y / p.h_size;
    const int h_q = blockIdx.y % p.h_size;
    const int batch = blockIdx.z / p.num_heads;
    const int head = blockIdx.z % p.num_heads;
    const int64_t q_base = (int64_t)batch * p.q_b + (int64_t)head * p.q_n;
    const int kv_head = head / p.group;

    const int w0 = blockIdx.x * BQ;
    const int w_first = imin(w0, w_size - 1);
    const int w_last = imin(w0 + BQ - 1, w_size - 1);
    const int warp_first = imin(w0 + warp * 16, w_size - 1);
    const int warp_last = imin(w0 + warp * 16 + 15, w_size - 1);

    // This thread's two query rows (block-local r_blk, r_blk + 8). Out-of-range
    // rows clamp to a real row and are never stored back.
    const int r_blk = warp * 16 + g;
    const int wq_r[2] = {imin(w0 + r_blk, w_size - 1), imin(w0 + r_blk + 8, w_size - 1)};
    const int64_t qrow = q_base + (int64_t)t_q * p.q_t + (int64_t)h_q * p.q_h;
    const int64_t rows[2] = {qrow + (int64_t)wq_r[0] * p.q_w, qrow + (int64_t)wq_r[1] * p.q_w};

    uint32_t qa[HD / 16][4];
    load_q<T, HD>(q, rows[0], rows[1], qd, qa);

    float o_acc[NT][4];
    #pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        o_acc[nt][0] = 0.f; o_acc[nt][1] = 0.f; o_acc[nt][2] = 0.f; o_acc[nt][3] = 0.f;
    }
    float m_r[2] = {NEG_INF, NEG_INF};
    float l_r[2] = {0.f, 0.f};

    int t_lo, t_hi;
    axis_window(t_q + p.t_offset, p.kt, p.tk, p.causal_t, true, t_lo, t_hi);

    // The load sweep runs one tile ahead of the compute sweep.
    Sweep ld, cp;
    enter_plane(p, t_lo, h_q, w_first, w_last, ld);
    cp = ld;
    load_tile<T, HD, BK, NTHREADS>(sK[0], sV[0], p, batch, kv_head, ld, tid);
    cp_async_commit();
    bool more = advance<BK>(p, t_hi, h_q, w_first, w_last, ld);

    // Per-thread and per-warp W windows of the compute sweep's plane.
    int plane = -1;
    int ws_r[2], we_r[2], warp_lo, warp_hi;
    int buf = 0;
    while (true) {
        if (more)
            load_tile<T, HD, BK, NTHREADS>(sK[buf ^ 1], sV[buf ^ 1], p, batch, kv_head, ld, tid);
        cp_async_commit();
        cp_async_wait_one();
        __syncthreads();

        if (cp.t != plane) {
            plane = cp.t;
            const int kw = p.kw[window_index(p, plane)];
            int unused;
            axis_window(wq_r[0], kw, w_size, p.causal_w, p.clamp, ws_r[0], we_r[0]);
            axis_window(wq_r[1], kw, w_size, p.causal_w, p.clamp, ws_r[1], we_r[1]);
            axis_window(warp_first, kw, w_size, p.causal_w, p.clamp, warp_lo, unused);
            axis_window(warp_last, kw, w_size, p.causal_w, p.clamp, unused, warp_hi);
        }
        const int wk0 = cp.w_lo + cp.c * BK;
        const int n_valid_k = imin(BK, cp.w_hi - wk0);

        // Skip fully masked tiles. No barrier inside the guard -- the one
        // closing the loop is outside, so warps still rendezvous.
        if (wk0 + BK > warp_lo && wk0 < warp_hi) {
            bool active[TILES];
            #pragma unroll
            for (int nt = 0; nt < TILES; ++nt) {
                const int n0 = wk0 + nt * 8;
                active[nt] = nt * 8 < n_valid_k && n0 < warp_hi && n0 + 8 > warp_lo;
            }
            const auto visible = [&](int row, int c) {
                const int wk = wk0 + c;
                return wk >= ws_r[row] && wk < we_r[row] && c < n_valid_k;
            };
            attend_tile<T, HD, BK>(sK[buf], sV[buf], qa, active, visible, p.scale, m_r, l_r, o_acc, lane);
        }
        __syncthreads();
        if (!advance<BK>(p, t_hi, h_q, w_first, w_last, cp))
            break;
        if (more)
            more = advance<BK>(p, t_hi, h_q, w_first, w_last, ld);
        buf ^= 1;
    }

    const bool valid[2] = {w0 + r_blk < w_size, w0 + r_blk + 8 < w_size};
    store_rows<T, HD>(out, rows, valid, o_acc, l_r, qd);
#endif  // __CUDA_ARCH__ >= 800 (bf16 mma; dispatch constraints require sm80+)
}

// ---------------------------------------------------------------------------
// Patch kernel
// ---------------------------------------------------------------------------

// Position of the key tile sweep: key plane t, first key j of the tile in the
// flattened (y_lo..y_hi) x (x_lo..x_hi) rectangle of the patch's windows.
struct PatchSweep {
    int t, j, count, cols, y_lo, x_lo;
};

__device__ inline void enter_patch_plane(const Params& p, int t, int y_first, int y_last, int x_first, int x_last, PatchSweep& s) {
    const int window = window_index(p, t);
    int y_hi, x_hi, unused;
    axis_window(y_first, p.kh[window], p.h_size, p.causal_h, p.clamp, s.y_lo, unused);
    axis_window(y_last, p.kh[window], p.h_size, p.causal_h, p.clamp, unused, y_hi);
    axis_window(x_first, p.kw[window], p.w_size, p.causal_w, p.clamp, s.x_lo, unused);
    axis_window(x_last, p.kw[window], p.w_size, p.causal_w, p.clamp, unused, x_hi);
    s.t = t;
    s.j = 0;
    s.cols = x_hi - s.x_lo;
    s.count = (y_hi - s.y_lo) * s.cols;
}

template <int BK>
__device__ inline bool advance_patch(const Params& p, int t_hi, int y_first, int y_last, int x_first, int x_last, PatchSweep& s) {
    s.j += BK;
    if (s.j < s.count)
        return true;
    if (s.t + 1 < t_hi) {
        enter_patch_plane(p, s.t + 1, y_first, y_last, x_first, x_last, s);
        return true;
    }
    return false;
}

// Stages the K/V rows and the coordinates of keys s.j..s.j+BK-1 of the rectangle.
template <typename T, int HD, int BK, int NTHREADS>
__device__ inline void load_patch_tile(T* sK, T* sV, int* sY, int* sX, const Params& p, int batch, int kv_head,
                                       const PatchSweep& s, int tid) {
    constexpr int LD = HD + 8;
    constexpr int VECS = HD / 8;
    const T* k;
    const T* v;
    plane_kv(p, s.t, batch, kv_head, k, v);
    for (int idx = tid; idx < BK * VECS; idx += NTHREADS) {
        const int r = idx / VECS, c8 = idx % VECS;
        const int key = s.j + r;
        const bool valid = key < s.count;
        const int ky = s.y_lo + key / s.cols;
        const int kx = s.x_lo + key % s.cols;
        const int64_t off = valid ? (int64_t)ky * p.k_h + (int64_t)kx * p.k_w + c8 * 8 : 0;
        cp_async16(sK + r * LD + c8 * 8, k + off, valid);
        cp_async16(sV + r * LD + c8 * 8, v + off, valid);
        if (c8 == 0) {
            sY[r] = valid ? ky : NO_KEY;
            sX[r] = valid ? kx : NO_KEY;
        }
    }
}

template <typename T, int HD>
__global__ void __launch_bounds__(128) window_patch_kernel(
    const T* __restrict__ q,
    T* __restrict__ out,
    const Params p)
{
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    constexpr int BK = keys_for(HD);
    constexpr int TILES = BK / 8;
    constexpr int NT = HD / 8;
    constexpr int LD = HD + 8;
    constexpr int NTHREADS = 128;

    __shared__ alignas(16) T sK[2][BK * LD];
    __shared__ alignas(16) T sV[2][BK * LD];
    __shared__ int sY[2][BK];
    __shared__ int sX[2][BK];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;  // query head within the block's group of four
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int qd = lane & 3;

    const int h_size = p.h_size, w_size = p.w_size;
    const int patch_rows = (h_size + 3) / 4;
    const int t_q = blockIdx.y / patch_rows;
    const int y0 = (blockIdx.y % patch_rows) * 4;
    const int x0 = blockIdx.x * 4;
    const int head_blocks = p.num_heads / 4;
    const int batch = blockIdx.z / head_blocks;
    const int head0 = (blockIdx.z % head_blocks) * 4;
    const int64_t q_base = (int64_t)batch * p.q_b + (int64_t)(head0 + warp) * p.q_n;
    const int kv_head = head0 / p.group;

    const int y_first = imin(y0, h_size - 1), y_last = imin(y0 + 3, h_size - 1);
    const int x_first = imin(x0, w_size - 1), x_last = imin(x0 + 3, w_size - 1);

    // Row g of the m16 tile is patch query (g / 4, g % 4), row g+8 is two
    // patch rows below. Out-of-range queries clamp to a real one, never stored.
    const int qy[2] = {imin(y0 + g / 4, h_size - 1), imin(y0 + g / 4 + 2, h_size - 1)};
    const int qx = imin(x0 + g % 4, w_size - 1);
    const int64_t qrow = q_base + (int64_t)t_q * p.q_t + (int64_t)qx * p.q_w;
    const int64_t rows[2] = {qrow + (int64_t)qy[0] * p.q_h, qrow + (int64_t)qy[1] * p.q_h};

    uint32_t qa[HD / 16][4];
    load_q<T, HD>(q, rows[0], rows[1], qd, qa);

    float o_acc[NT][4];
    #pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        o_acc[nt][0] = 0.f; o_acc[nt][1] = 0.f; o_acc[nt][2] = 0.f; o_acc[nt][3] = 0.f;
    }
    float m_r[2] = {NEG_INF, NEG_INF};
    float l_r[2] = {0.f, 0.f};

    int t_lo, t_hi;
    axis_window(t_q + p.t_offset, p.kt, p.tk, p.causal_t, true, t_lo, t_hi);

    PatchSweep ld, cp;
    enter_patch_plane(p, t_lo, y_first, y_last, x_first, x_last, ld);
    cp = ld;
    load_patch_tile<T, HD, BK, NTHREADS>(sK[0], sV[0], sY[0], sX[0], p, batch, kv_head, ld, tid);
    cp_async_commit();
    bool more = advance_patch<BK>(p, t_hi, y_first, y_last, x_first, x_last, ld);

    // Per-thread windows of the compute sweep's plane.
    int plane = -1;
    int ylo[2], yhi[2], xlo, xhi;
    int buf = 0;
    while (true) {
        if (more)
            load_patch_tile<T, HD, BK, NTHREADS>(sK[buf ^ 1], sV[buf ^ 1], sY[buf ^ 1], sX[buf ^ 1], p, batch, kv_head, ld, tid);
        cp_async_commit();
        cp_async_wait_one();
        __syncthreads();

        if (cp.t != plane) {
            plane = cp.t;
            const int window = window_index(p, plane);
            axis_window(qy[0], p.kh[window], h_size, p.causal_h, p.clamp, ylo[0], yhi[0]);
            axis_window(qy[1], p.kh[window], h_size, p.causal_h, p.clamp, ylo[1], yhi[1]);
            axis_window(qx, p.kw[window], w_size, p.causal_w, p.clamp, xlo, xhi);
        }
        // Every key of the rectangle is in some patch query's window, so only
        // the padding past the plane's last key is inactive.
        const int n_valid_k = imin(BK, cp.count - cp.j);
        bool active[TILES];
        #pragma unroll
        for (int nt = 0; nt < TILES; ++nt)
            active[nt] = nt * 8 < n_valid_k;
        const int* tY = sY[buf];
        const int* tX = sX[buf];
        const auto visible = [&](int row, int c) {
            const int ky = tY[c], kx = tX[c];
            return ky >= ylo[row] && ky < yhi[row] && kx >= xlo && kx < xhi;
        };
        attend_tile<T, HD, BK>(sK[buf], sV[buf], qa, active, visible, p.scale, m_r, l_r, o_acc, lane);
        __syncthreads();
        if (!advance_patch<BK>(p, t_hi, y_first, y_last, x_first, x_last, cp))
            break;
        if (more)
            more = advance_patch<BK>(p, t_hi, y_first, y_last, x_first, x_last, ld);
        buf ^= 1;
    }

    const bool in_x = x0 + g % 4 < w_size;
    const bool valid[2] = {in_x && y0 + g / 4 < h_size, in_x && y0 + g / 4 + 2 < h_size};
    store_rows<T, HD>(out, rows, valid, o_acc, l_r, qd);
#endif  // __CUDA_ARCH__ >= 800 (bf16 mma; dispatch constraints require sm80+)
}

// ---------------------------------------------------------------------------
// Launch
// ---------------------------------------------------------------------------

// Widest tile covering a short axis, narrowest past that for small windows --
// extra width only adds tiles most warps skip (measured on sm89, W 16..256,
// kw 5..17). Wide windows share each staged key row across four warps.
inline int choose_wq(int w_size, int kw) {
    if (w_size <= 16) return 1;
    if (w_size <= 32) return 2;
    return kw > 17 ? 4 : 1;
}

#define NA3D_HEAD_DIMS(LAUNCH)                                                              \
    switch (head_dim) {                                                                     \
        case 16: LAUNCH(16); break;                                                         \
        case 32: LAUNCH(32); break;                                                         \
        case 48: LAUNCH(48); break;                                                         \
        case 64: LAUNCH(64); break;                                                         \
        case 128: LAUNCH(128); break;                                                       \
        default:                                                                            \
            throw std::runtime_error(                                                       \
                "na3d supports head_dim 16/32/48/64/128, got " + std::to_string(head_dim)); \
    }

template <typename T, int WQ>
void launch_rows(const void* q, void* out, const Params& p, int batch, int head_dim, cudaStream_t stream)
{
    const dim3 grid((p.w_size + WQ * 16 - 1) / (WQ * 16), p.tq * p.h_size, batch * p.num_heads);
#define NA3D_LAUNCH(HD_)                                                                    \
    na3d_kernel<T, HD_, WQ><<<grid, WQ * 32, 0, stream>>>(                                  \
        reinterpret_cast<const T*>(q), reinterpret_cast<T*>(out), p)
    NA3D_HEAD_DIMS(NA3D_LAUNCH)
#undef NA3D_LAUNCH
}

template <typename T>
void launch_patches(const void* q, void* out, const Params& p, int batch, int head_dim, cudaStream_t stream)
{
    const dim3 grid((p.w_size + 3) / 4, p.tq * ((p.h_size + 3) / 4), batch * (p.num_heads / 4));
#define NA3D_LAUNCH(HD_)                                                                    \
    window_patch_kernel<T, HD_><<<grid, 128, 0, stream>>>(                                  \
        reinterpret_cast<const T*>(q), reinterpret_cast<T*>(out), p)
    NA3D_HEAD_DIMS(NA3D_LAUNCH)
#undef NA3D_LAUNCH
}

template <typename T>
void launch_typed(const void* q, void* out, const Params& p, int batch, int head_dim, cudaStream_t stream)
{
    // Four query heads sharing a kv head stage each key tile once for all four.
    if (p.group % 4 == 0) {
        launch_patches<T>(q, out, p, batch, head_dim, stream);
        return;
    }
    int kw = p.kw[0];
    for (int i = 1; i < p.n_windows; ++i)
        kw = imax(kw, p.kw[i]);
    // Only the widths choose_wq can return are instantiated.
    switch (choose_wq(p.w_size, kw)) {
        case 4: launch_rows<T, 4>(q, out, p, batch, head_dim, stream); break;
        case 2: launch_rows<T, 2>(q, out, p, batch, head_dim, stream); break;
        default: launch_rows<T, 1>(q, out, p, batch, head_dim, stream); break;
    }
}

void launch(const void* q, void* out, const Params& p, int batch, int head_dim, int dtype_code, cudaStream_t stream)
{
    if (dtype_code == 1) {
        launch_typed<__half>(q, out, p, batch, head_dim, stream);
    } else if (dtype_code == 2) {
        launch_typed<__nv_bfloat16>(q, out, p, batch, head_dim, stream);
    } else {
        throw std::runtime_error(
            "na3d supports FP16/BF16 only, got dtype code " + std::to_string(dtype_code));
    }
}

// Contiguous (B, T, H, W, NH, HD) strides.
void set_strides(int t_size, int h_size, int w_size, int num_heads, int head_dim,
                 int64_t& s_b, int64_t& s_t, int64_t& s_h, int64_t& s_w, int64_t& s_n)
{
    s_n = head_dim;
    s_w = (int64_t)num_heads * head_dim;
    s_h = (int64_t)w_size * s_w;
    s_t = (int64_t)h_size * s_h;
    s_b = (int64_t)t_size * s_t;
}

}  // namespace

extern "C" void launch_na3d_kernel(
    const void* q, const void* k, const void* v, void* out,
    int batch, int t_size, int h_size, int w_size, int num_heads, int head_dim,
    int kt, int kh, int kw,
    int causal_t, int causal_h, int causal_w,
    float scale, int dtype_code, cudaStream_t stream)
{
    Params p{};
    p.k = k;
    p.v = v;
    p.tq = p.tk = t_size;
    p.h_size = h_size;
    p.w_size = w_size;
    p.num_heads = num_heads;
    p.group = 1;
    set_strides(t_size, h_size, w_size, num_heads, head_dim, p.q_b, p.q_t, p.q_h, p.q_w, p.q_n);
    set_strides(t_size, h_size, w_size, num_heads, head_dim, p.k_b, p.k_t, p.k_h, p.k_w, p.k_n);
    p.causal_t = causal_t != 0;
    p.causal_h = causal_h != 0;
    p.causal_w = causal_w != 0;
    p.clamp = true;
    // Non-causal kernels clamp to the axis length (NATTEN would reject this).
    p.kt = p.causal_t ? kt : imin(kt, t_size);
    p.n_windows = 1;
    p.kh[0] = p.causal_h ? kh : imin(kh, h_size);
    p.kw[0] = p.causal_w ? kw : imin(kw, w_size);
    p.scale = scale;
    launch(q, out, p, batch, head_dim, dtype_code, stream);
}

extern "C" void launch_window_attention_kernel(
    const void* q, const void* k, const void* v, void* out,
    int batch, int tq, int tk, int h_size, int w_size, int num_heads, int kv_heads, int head_dim,
    const int* kh, const int* kw, int n_windows, int temporal_window, int clamp,
    float scale, int dtype_code, cudaStream_t stream)
{
    Params p{};
    p.k = k;
    p.v = v;
    p.tq = tq;
    p.tk = tk;
    p.h_size = h_size;
    p.w_size = w_size;
    p.num_heads = num_heads;
    p.group = num_heads / kv_heads;
    set_strides(tq, h_size, w_size, num_heads, head_dim, p.q_b, p.q_t, p.q_h, p.q_w, p.q_n);
    set_strides(tk, h_size, w_size, kv_heads, head_dim, p.k_b, p.k_t, p.k_h, p.k_w, p.k_n);
    p.kt = temporal_window;
    p.t_offset = tk - tq;
    p.causal_t = true;
    p.clamp = clamp != 0;
    p.n_windows = n_windows;
    for (int i = 0; i < n_windows; ++i) {
        // A clamped window larger than the frame covers the whole axis.
        p.kh[i] = p.clamp ? imin(kh[i], h_size) : kh[i];
        p.kw[i] = p.clamp ? imin(kw[i], w_size) : kw[i];
    }
    p.scale = scale;
    launch(q, out, p, batch, head_dim, dtype_code, stream);
}
