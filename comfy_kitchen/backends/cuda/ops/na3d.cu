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

// Fused 3D neighborhood attention (NATTEN na3d semantics, dilation 1).
//
// Per non-causal axis each query attends a window of exactly kernel_size
// positions centered on it, shifted inward at grid boundaries; per causal
// axis it attends the min(i+1, kernel_size) nearest previous positions.
//
// Flash-attention-2 style: one block covers BQ queries along W at a fixed
// (t, h), 16 rows per warp. Scores, softmax stats, P and the accumulator stay
// in registers; shared memory only stages K and V^T. T/H windows are scalar
// per block, so only W needs per-element masking.
//
// Requires sm80+ (bf16 mma / f32 accumulators). head_dim in {16,32,48,64}.
// Layout: q/k/v/out are contiguous (B, T, H, W, NH, HD).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

constexpr int BK = 32;         // keys per inner tile (4 n8 mma tiles)
constexpr float NEG_INF = -3.0e38f;

// BQ (queries per block along W) is a template parameter, 16 rows per warp.
__host__ __device__ constexpr int warps_for(int bq) { return bq / 16; }
__host__ __device__ constexpr int threads_for(int bq) { return warps_for(bq) * 32; }

__host__ __device__ inline int imin(int a, int b) { return a < b ? a : b; }
__host__ __device__ inline int imax(int a, int b) { return a > b ? a : b; }

__device__ inline void axis_window(int i, int k, int len, bool causal, int& lo, int& hi) {
    if (causal) {
        lo = imax(i - k + 1, 0);
        hi = i + 1;
    } else {
        lo = imin(imax(i - k / 2, 0), len - k);
        hi = lo + k;
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

// mma.m16n8k16 fragment maps (PTX ISA), lane = g*4 + q (g = lane>>2, q = lane&3):
//   A (16x16):  a0=(g, 2q..2q+1) a1=(g+8, 2q..2q+1) a2=(g, 2q+8..2q+9) a3=(g+8, 2q+8..2q+9)
//   B (16x8):   b0=(k=2q..2q+1, n=g) b1=(k=2q+8..2q+9, n=g)   (packed along k)
//   C (16x8):   c0=(g, 2q) c1=(g, 2q+1) c2=(g+8, 2q) c3=(g+8, 2q+1)   (f32)

template <typename T, int HD, int BQ>
__global__ void __launch_bounds__(threads_for(BQ)) na3d_kernel(
    const T* __restrict__ q,
    const T* __restrict__ k,
    const T* __restrict__ v,
    T* __restrict__ out,
    int t_size, int h_size, int w_size, int num_heads,
    int64_t s_b, int64_t s_t, int64_t s_h, int64_t s_w, int64_t s_n,
    int kt, int kh, int kw,
    bool causal_t, bool causal_h, bool causal_w,
    float scale)
{
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    constexpr int KC = HD / 16;   // k-chunks for S = Q.K^T
    constexpr int NT = HD / 8;    // n8 output tiles for O += P.V
    constexpr int LDK = HD + 8;   // sK row stride (elements)
    constexpr int LDV = BK + 8;   // sVt row stride (elements)
    constexpr int NTHREADS = threads_for(BQ);

    __shared__ T sK[BK * LDK];
    __shared__ T sVt[HD * LDV];  // transposed V: sVt[col][key]

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;    // fragment row within m16 tiles
    const int qd = lane & 3;    // quad thread: column pairs

    const int t_q = blockIdx.y / h_size;
    const int h_q = blockIdx.y % h_size;
    const int64_t base = (int64_t)(blockIdx.z / num_heads) * s_b + (int64_t)(blockIdx.z % num_heads) * s_n;

    const int w0 = blockIdx.x * BQ;
    const int n_valid_q = imin(BQ, w_size - w0);

    int t_lo, t_hi, h_lo, h_hi;
    axis_window(t_q, kt, t_size, causal_t, t_lo, t_hi);
    axis_window(h_q, kh, h_size, causal_h, h_lo, h_hi);

    // This thread's two query rows (block-local r_blk, r_blk + 8).
    const int r_blk = warp * 16 + g;
    int wq_r[2], ws_r[2], we_r[2];
    {
        wq_r[0] = imin(w0 + r_blk, w_size - 1);
        wq_r[1] = imin(w0 + r_blk + 8, w_size - 1);
        int lo, hi;
        axis_window(wq_r[0], kw, w_size, causal_w, lo, hi); ws_r[0] = lo; we_r[0] = hi;
        axis_window(wq_r[1], kw, w_size, causal_w, lo, hi); ws_r[1] = lo; we_r[1] = hi;
    }
    // Block sweep range: window start of the first query, end of the last valid.
    int w_lo, w_hi;
    {
        int lo0, hi0, lo1, hi1;
        axis_window(imin(w0, w_size - 1), kw, w_size, causal_w, lo0, hi0);
        axis_window(imin(w0 + BQ - 1, w_size - 1), kw, w_size, causal_w, lo1, hi1);
        w_lo = lo0;
        w_hi = hi1;
    }
    // Key span of this warp's 16 queries: ~16 + kw against the block's BQ + kw.
    int warp_lo, warp_hi;
    {
        int lo0, hi0, lo1, hi1;
        axis_window(imin(w0 + warp * 16, w_size - 1), kw, w_size, causal_w, lo0, hi0);
        axis_window(imin(w0 + warp * 16 + 15, w_size - 1), kw, w_size, causal_w, lo1, hi1);
        warp_lo = lo0;
        warp_hi = hi1;
    }

    // Q operand fragments: loaded once, register-resident for the whole
    // kernel. Out-of-range rows clamp to a real row (never stored back).
    uint32_t qa[KC][4];
    {
        const int64_t qrow = base + (int64_t)t_q * s_t + (int64_t)h_q * s_h;
        const int64_t row0 = qrow + (int64_t)wq_r[0] * s_w;
        const int64_t row1 = qrow + (int64_t)wq_r[1] * s_w;
        #pragma unroll
        for (int kc = 0; kc < KC; ++kc) {
            const int c0 = kc * 16 + qd * 2;
            qa[kc][0] = *reinterpret_cast<const uint32_t*>(q + row0 + c0);
            qa[kc][1] = *reinterpret_cast<const uint32_t*>(q + row1 + c0);
            qa[kc][2] = *reinterpret_cast<const uint32_t*>(q + row0 + c0 + 8);
            qa[kc][3] = *reinterpret_cast<const uint32_t*>(q + row1 + c0 + 8);
        }
    }

    float o_acc[NT][4];
    #pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        o_acc[nt][0] = 0.f; o_acc[nt][1] = 0.f; o_acc[nt][2] = 0.f; o_acc[nt][3] = 0.f;
    }
    float m_r[2] = {NEG_INF, NEG_INF};
    float l_r[2] = {0.f, 0.f};

    for (int tk = t_lo; tk < t_hi; ++tk) {
        for (int hk = h_lo; hk < h_hi; ++hk) {
            const int64_t plane = base + (int64_t)tk * s_t + (int64_t)hk * s_h;
            for (int wk0 = w_lo; wk0 < w_hi; wk0 += BK) {
                const int n_valid_k = imin(BK, w_hi - wk0);

                // Stage K (row-major) and V (transposed) tiles cooperatively.
                {
                    constexpr int VECS = HD / 8;
                    for (int idx = tid; idx < BK * VECS; idx += NTHREADS) {
                        const int r = idx / VECS, c8 = idx % VECS;
                        uint4 kv = make_uint4(0u, 0u, 0u, 0u);
                        uint4 vv = make_uint4(0u, 0u, 0u, 0u);
                        if (r < n_valid_k) {
                            const int64_t off = plane + (int64_t)(wk0 + r) * s_w + c8 * 8;
                            kv = *reinterpret_cast<const uint4*>(k + off);
                            vv = *reinterpret_cast<const uint4*>(v + off);
                        }
                        *reinterpret_cast<uint4*>(sK + r * LDK + c8 * 8) = kv;
                        const T* ve = reinterpret_cast<const T*>(&vv);
                        #pragma unroll
                        for (int e = 0; e < 8; ++e)
                            sVt[(c8 * 8 + e) * LDV + r] = ve[e];
                    }
                }
                __syncthreads();

                // Skip fully masked tiles. No barrier inside the guard -- the
                // one closing the loop is outside, so warps still rendezvous.
                if (wk0 + BK > warp_lo && wk0 < warp_hi) {

                    // S = Q.K^T: 4 n8 tiles of 16x8, accumulated in registers.
                    float s_acc[4][4];
                    #pragma unroll
                    for (int nt = 0; nt < 4; ++nt) {
                        s_acc[nt][0] = 0.f; s_acc[nt][1] = 0.f; s_acc[nt][2] = 0.f; s_acc[nt][3] = 0.f;
                        const T* krow = sK + (nt * 8 + g) * LDK;
                        #pragma unroll
                        for (int kc = 0; kc < KC; ++kc) {
                            uint32_t kb[2];
                            kb[0] = *reinterpret_cast<const uint32_t*>(krow + kc * 16 + qd * 2);
                            kb[1] = *reinterpret_cast<const uint32_t*>(krow + kc * 16 + qd * 2 + 8);
                            MmaTraits<T>::mma(s_acc[nt], qa[kc], kb);
                        }
                    }

                    // Masked online softmax, entirely in registers. This thread's
                    // score columns are wk0 + nt*8 + 2*qd (+1) for rows g / g+8.
                    float p_val[4][4];
                    float m_new[2] = {m_r[0], m_r[1]};
                    #pragma unroll
                    for (int nt = 0; nt < 4; ++nt) {
                        const int c0 = wk0 + nt * 8 + qd * 2;
                        #pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            const int row = e >> 1;          // c0,c1 -> row g; c2,c3 -> row g+8
                            const int wk = c0 + (e & 1);
                            const bool vis = wk >= ws_r[row] && wk < we_r[row] && (wk - wk0) < n_valid_k;
                            const float s = vis ? s_acc[nt][e] * scale : NEG_INF;
                            p_val[nt][e] = s;
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
                    for (int nt = 0; nt < 4; ++nt) {
                        #pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            const int row = e >> 1;
                            const float p = (p_val[nt][e] <= NEG_INF) ? 0.f : __expf(p_val[nt][e] - m_new[row]);
                            p_val[nt][e] = p;
                            l_add[row] += p;
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

                    // P operand fragments: pure register repack of the score tile.
                    // k-step kk covers keys kk*16..+15 = score tiles 2kk, 2kk+1.
                    uint32_t pa[2][4];
                    #pragma unroll
                    for (int kk = 0; kk < 2; ++kk) {
                        pa[kk][0] = MmaTraits<T>::pack(p_val[2 * kk][0], p_val[2 * kk][1]);
                        pa[kk][1] = MmaTraits<T>::pack(p_val[2 * kk][2], p_val[2 * kk][3]);
                        pa[kk][2] = MmaTraits<T>::pack(p_val[2 * kk + 1][0], p_val[2 * kk + 1][1]);
                        pa[kk][3] = MmaTraits<T>::pack(p_val[2 * kk + 1][2], p_val[2 * kk + 1][3]);
                    }

                    // O += P.V via transposed-V fragments (32-bit loads along keys).
                    #pragma unroll
                    for (int nt = 0; nt < NT; ++nt) {
                        const T* vcol = sVt + (nt * 8 + g) * LDV;
                        #pragma unroll
                        for (int kk = 0; kk < 2; ++kk) {
                            uint32_t vb[2];
                            vb[0] = *reinterpret_cast<const uint32_t*>(vcol + kk * 16 + qd * 2);
                            vb[1] = *reinterpret_cast<const uint32_t*>(vcol + kk * 16 + qd * 2 + 8);
                            MmaTraits<T>::mma(o_acc[nt], pa[kk], vb);
                        }
                    }

                }  // warp-active tile
                __syncthreads();
            }
        }
    }

    // Epilogue: divide by l and store this thread's (row, col) elements.
    const float inv_l0 = 1.f / fmaxf(l_r[0], 1e-30f);
    const float inv_l1 = 1.f / fmaxf(l_r[1], 1e-30f);
    const int64_t orow = base + (int64_t)t_q * s_t + (int64_t)h_q * s_h;
    #pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
        const int r = r_blk + rr * 8;
        if (r >= n_valid_q)
            continue;
        const float inv_l = rr ? inv_l1 : inv_l0;
        T* orow_p = out + orow + (int64_t)(w0 + r) * s_w;
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const int c = nt * 8 + qd * 2;
            *reinterpret_cast<uint32_t*>(orow_p + c) =
                MmaTraits<T>::pack(o_acc[nt][rr * 2] * inv_l, o_acc[nt][rr * 2 + 1] * inv_l);
        }
    }
#endif  // __CUDA_ARCH__ >= 800 (bf16 mma; dispatch constraints require sm80+)
}

// Widest tile covering a short axis, narrowest past that -- extra width only
// adds tiles most warps skip. Measured on sm89, W 16..256, kw 5..17.
__host__ __device__ inline int choose_bq(int w_size) {
    if (w_size <= 16) return 16;
    if (w_size <= 32) return 32;
    return 16;
}

template <typename T, int BQ>
void launch_na3d_bq(
    const void* q, const void* k, const void* v, void* out,
    int t_size, int h_size, int w_size, int num_heads, int head_dim,
    int64_t s_b, int64_t s_t, int64_t s_h, int64_t s_w, int64_t s_n,
    int kt, int kh, int kw,
    bool causal_t, bool causal_h, bool causal_w,
    float scale, int batch, cudaStream_t stream)
{
    const dim3 grid((w_size + BQ - 1) / BQ, t_size * h_size, batch * num_heads);

#define NA3D_LAUNCH(HD_)                                                                    \
    na3d_kernel<T, HD_, BQ><<<grid, threads_for(BQ), 0, stream>>>(                          \
        reinterpret_cast<const T*>(q), reinterpret_cast<const T*>(k),                       \
        reinterpret_cast<const T*>(v), reinterpret_cast<T*>(out),                           \
        t_size, h_size, w_size, num_heads, s_b, s_t, s_h, s_w, s_n,                         \
        kt, kh, kw, causal_t, causal_h, causal_w, scale)

    switch (head_dim) {
        case 16: NA3D_LAUNCH(16); break;
        case 32: NA3D_LAUNCH(32); break;
        case 48: NA3D_LAUNCH(48); break;
        case 64: NA3D_LAUNCH(64); break;
        default:
            throw std::runtime_error(
                "na3d supports head_dim 16/32/48/64, got " + std::to_string(head_dim));
    }
#undef NA3D_LAUNCH
}

template <typename T>
void launch_na3d_typed(
    const void* q, const void* k, const void* v, void* out,
    int t_size, int h_size, int w_size, int num_heads, int head_dim,
    int64_t s_b, int64_t s_t, int64_t s_h, int64_t s_w, int64_t s_n,
    int kt, int kh, int kw,
    bool causal_t, bool causal_h, bool causal_w,
    float scale, int batch, cudaStream_t stream)
{
#define NA3D_DISPATCH_BQ(BQ_)                                                               \
    launch_na3d_bq<T, BQ_>(q, k, v, out, t_size, h_size, w_size, num_heads, head_dim,       \
                           s_b, s_t, s_h, s_w, s_n, kt, kh, kw,                             \
                           causal_t, causal_h, causal_w, scale, batch, stream)

    // Only the widths choose_bq can return are instantiated.
    switch (choose_bq(w_size)) {
        case 32: NA3D_DISPATCH_BQ(32); break;
        default: NA3D_DISPATCH_BQ(16); break;
    }
#undef NA3D_DISPATCH_BQ
}

}  // namespace

extern "C" void launch_na3d_kernel(
    const void* q, const void* k, const void* v, void* out,
    int batch, int t_size, int h_size, int w_size, int num_heads, int head_dim,
    int kt, int kh, int kw,
    int causal_t, int causal_h, int causal_w,
    float scale, int dtype_code, cudaStream_t stream)
{
    // Contiguous (B, T, H, W, NH, HD) strides.
    const int64_t s_n = head_dim;
    const int64_t s_w = (int64_t)num_heads * head_dim;
    const int64_t s_h = (int64_t)w_size * s_w;
    const int64_t s_t = (int64_t)h_size * s_h;
    const int64_t s_b = (int64_t)t_size * s_t;

    // Non-causal kernels clamp to the axis length (NATTEN would reject this).
    const int ckt = causal_t ? kt : imin(kt, t_size);
    const int ckh = causal_h ? kh : imin(kh, h_size);
    const int ckw = causal_w ? kw : imin(kw, w_size);

    if (dtype_code == 1) {
        launch_na3d_typed<__half>(q, k, v, out, t_size, h_size, w_size, num_heads, head_dim,
                                  s_b, s_t, s_h, s_w, s_n, ckt, ckh, ckw,
                                  causal_t != 0, causal_h != 0, causal_w != 0, scale, batch, stream);
    } else if (dtype_code == 2) {
        launch_na3d_typed<__nv_bfloat16>(q, k, v, out, t_size, h_size, w_size, num_heads, head_dim,
                                         s_b, s_t, s_h, s_w, s_n, ckt, ckh, ckw,
                                         causal_t != 0, causal_h != 0, causal_w != 0, scale, batch, stream);
    } else {
        throw std::runtime_error(
            "na3d supports FP16/BF16 only, got dtype code " + std::to_string(dtype_code));
    }
}
