// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Fused 3D neighborhood attention (NATTEN na3d semantics, dilation 1).
//
// Per non-causal axis each query attends a window of exactly kernel_size
// positions centered on it, shifted inward at grid boundaries; per causal axis it
// attends the min(i+1, kernel_size) nearest previous positions.
//
// Flash-attention-2 style on the 16x16x16 WMMA: one wave covers 16 queries along
// W at a fixed (t, h) and sweeps that block's key span 16 keys at a time. Scores,
// softmax state and the output accumulator stay in registers; LDS holds only the
// transposed V tile and the probability tile.
//
// The T and H windows are constant across a block, so only W is masked per
// element. head_dim in {16, 32, 48, 64}; q/k/v/out are contiguous
// (B, T, H, W, NH, HD).
#include <stdexcept>
#include <string>

#include "../mma.h"

namespace comfy::hip_backend {

// One wave, 16 queries. A wider block would only add key tiles that most of it
// then skips, since the window a query sees is narrow by construction.
constexpr int kNaQueries = 16;
constexpr int kNaKeys = 16;
constexpr int kNaThreads = 32;
// LDS rows are padded to 9 dwords. LDS has 32 four-byte banks and gcd(9, 32) = 1,
// so the 16 row-strided lanes of a fragment read reach 16 distinct banks; an
// 8-dword or 12-dword row collapses them onto 4 or 8.
constexpr int kNaRowStride = 18;
constexpr float kNaNegInf = -3.0e38f;

namespace {

__forceinline__ __device__ int na_min(int a, int b) { return a < b ? a : b; }
__forceinline__ __device__ int na_max(int a, int b) { return a > b ? a : b; }

__forceinline__ __device__ void axis_window(int i, int k, int len, bool causal, int& lo, int& hi) {
    if (causal) {
        lo = na_max(i - k + 1, 0);
        hi = i + 1;
    } else {
        lo = na_min(na_max(i - k / 2, 0), len - k);
        hi = lo + k;
    }
}

// Reduce across the 16 lanes holding one accumulator row. They are a half-wave, so
// the offsets stop at 8 and never cross into the other half's rows.
__forceinline__ __device__ float row_reduce_max(float v) {
    #pragma unroll
    for (int off = 1; off <= 8; off <<= 1) {
        v = fmaxf(v, __shfl_xor(v, off, kWave));
    }
    return v;
}

__forceinline__ __device__ float row_reduce_sum(float v) {
    #pragma unroll
    for (int off = 1; off <= 8; off <<= 1) {
        v += __shfl_xor(v, off, kWave);
    }
    return v;
}

template <typename Mma, int HD>
__global__ __launch_bounds__(kNaThreads) void na3d_kernel(
    const typename Mma::Elem* __restrict__ q,
    const typename Mma::Elem* __restrict__ k,
    const typename Mma::Elem* __restrict__ v,
    typename Mma::Elem* __restrict__ out,
    int t_size, int h_size, int w_size, int num_heads,
    int64_t s_b, int64_t s_t, int64_t s_h, int64_t s_w, int64_t s_n,
    int kt, int kh, int kw,
    bool causal_t, bool causal_h, bool causal_w,
    float scale) {

    using Elem = typename Mma::Elem;
    constexpr int KC = HD / 16;   // k-chunks of S = Q.K^T
    constexpr int NT = HD / 16;   // output column tiles of O += P.V
    constexpr int LD = kNaRowStride;

    __shared__ Elem sVt[HD * LD];         // transposed V: sVt[column][key]
    __shared__ Elem sP[kNaQueries * LD];  // probabilities: sP[query][key]

    const int lane = threadIdx.x;
    const int r = frag_row(lane);  // fragment row, and this lane's score column
    const int t_q = blockIdx.y / h_size;
    const int h_q = blockIdx.y % h_size;
    const int64_t base = static_cast<int64_t>(blockIdx.z / num_heads) * s_b +
                         static_cast<int64_t>(blockIdx.z % num_heads) * s_n;

    const int w0 = blockIdx.x * kNaQueries;
    const int n_valid_q = na_min(kNaQueries, w_size - w0);

    int t_lo, t_hi, h_lo, h_hi;
    axis_window(t_q, kt, t_size, causal_t, t_lo, t_hi);
    axis_window(h_q, kh, h_size, causal_h, h_lo, h_hi);

    // The eight accumulator rows this lane owns, and each one's W window.
    int ws[8], we[8];
    #pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int wq = na_min(w0 + acc_row(lane, e), w_size - 1);
        axis_window(wq, kw, w_size, causal_w, ws[e], we[e]);
    }

    // Key span of the block: the first query's window start through the last
    // valid query's window end.
    int w_lo, w_hi;
    {
        int lo0, hi0, lo1, hi1;
        axis_window(na_min(w0, w_size - 1), kw, w_size, causal_w, lo0, hi0);
        axis_window(na_min(w0 + n_valid_q - 1, w_size - 1), kw, w_size, causal_w, lo1, hi1);
        w_lo = lo0;
        w_hi = hi1;
    }

    // Loaded once, register-resident for the whole kernel. A row past the grid
    // clamps to a real one; it is never stored back.
    typename Mma::Frag qa[KC];
    {
        const int64_t qrow = base + static_cast<int64_t>(t_q) * s_t +
                             static_cast<int64_t>(h_q) * s_h +
                             static_cast<int64_t>(na_min(w0 + r, w_size - 1)) * s_w;
        #pragma unroll
        for (int kc = 0; kc < KC; ++kc) {
            qa[kc] = load_frag_16bit<Mma>(q + qrow + kc * 16, lane);
        }
    }

    typename Mma::Acc o_acc[NT];
    #pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        o_acc[nt] = Mma::zero();
    }
    float m_r[8], l_r[8];
    #pragma unroll
    for (int e = 0; e < 8; ++e) {
        m_r[e] = kNaNegInf;
        l_r[e] = 0.0f;
    }

    for (int tk = t_lo; tk < t_hi; ++tk) {
        for (int hk = h_lo; hk < h_hi; ++hk) {
            const int64_t plane = base + static_cast<int64_t>(tk) * s_t +
                                  static_cast<int64_t>(hk) * s_h;
            for (int wk0 = w_lo; wk0 < w_hi; wk0 += kNaKeys) {
                const int n_valid_k = na_min(kNaKeys, w_hi - wk0);

                // Stage V transposed; only the LDS write is scattered.
                {
                    constexpr int kVecs = HD / 8;
                    for (int idx = lane; idx < kNaKeys * kVecs; idx += kNaThreads) {
                        const int kk = idx / kVecs;
                        const int c8 = idx % kVecs;
                        Elem vals[8];
                        if (kk < n_valid_k) {
                            const Elem* src = v + plane +
                                              static_cast<int64_t>(wk0 + kk) * s_w + c8 * 8;
                            #pragma unroll
                            for (int e = 0; e < 8; ++e) {
                                vals[e] = src[e];
                            }
                        } else {
                            #pragma unroll
                            for (int e = 0; e < 8; ++e) {
                                vals[e] = static_cast<Elem>(0.0f);
                            }
                        }
                        #pragma unroll
                        for (int e = 0; e < 8; ++e) {
                            sVt[(c8 * 8 + e) * LD + kk] = vals[e];
                        }
                    }
                }

                // The key row clamps into range; a column past n_valid_k is masked
                // below, so whatever it read never survives.
                typename Mma::Acc s_acc = Mma::zero();
                {
                    const int64_t krow =
                        plane + static_cast<int64_t>(na_min(wk0 + r, w_size - 1)) * s_w;
                    #pragma unroll
                    for (int kc = 0; kc < KC; ++kc) {
                        s_acc = Mma::mma(qa[kc], load_frag_16bit<Mma>(k + krow + kc * 16, lane),
                                         s_acc);
                    }
                }

                // Masked online softmax. This lane holds score column wk0 + r for
                // each of its accumulator rows.
                const int wk = wk0 + r;
                const bool in_tile = (wk - wk0) < n_valid_k;
                float p_val[8], m_new[8];
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const bool vis = in_tile && wk >= ws[e] && wk < we[e];
                    p_val[e] = vis ? Mma::get(s_acc, e) * scale : kNaNegInf;
                    m_new[e] = row_reduce_max(fmaxf(m_r[e], p_val[e]));
                }
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const float alpha = __expf(m_r[e] - m_new[e]);
                    m_r[e] = m_new[e];
                    p_val[e] = (p_val[e] <= kNaNegInf) ? 0.0f : __expf(p_val[e] - m_new[e]);
                    l_r[e] = l_r[e] * alpha + row_reduce_sum(p_val[e]);
                    #pragma unroll
                    for (int nt = 0; nt < NT; ++nt) {
                        o_acc[nt][e] *= alpha;
                    }
                }

                // Through LDS because the next A operand wants a whole row per lane
                // and the accumulator gave this lane a column. The barrier below
                // publishes the V transpose staged above as well.
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    sP[acc_row(lane, e) * LD + acc_col(lane)] = static_cast<Elem>(p_val[e]);
                }
                __syncthreads();

                const typename Mma::Frag pa = load_frag_16bit<Mma>(sP + r * LD, lane);
                #pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    o_acc[nt] = Mma::mma(
                        pa, load_frag_16bit<Mma>(sVt + (nt * 16 + r) * LD, lane), o_acc[nt]);
                }
                __syncthreads();
            }
        }
    }

    const int64_t orow = base + static_cast<int64_t>(t_q) * s_t + static_cast<int64_t>(h_q) * s_h;
    #pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int row = acc_row(lane, e);
        if (row >= n_valid_q) {
            continue;
        }
        const float inv_l = 1.0f / fmaxf(l_r[e], 1e-30f);
        Elem* dst = out + orow + static_cast<int64_t>(w0 + row) * s_w + acc_col(lane);
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            dst[nt * 16] = static_cast<Elem>(Mma::get(o_acc[nt], e) * inv_l);
        }
    }
}

template <typename Mma>
void launch_na3d_typed(
    const void* q, const void* k, const void* v, void* out,
    int batch, int t_size, int h_size, int w_size, int num_heads, int head_dim,
    int64_t s_b, int64_t s_t, int64_t s_h, int64_t s_w, int64_t s_n,
    int kt, int kh, int kw,
    bool causal_t, bool causal_h, bool causal_w,
    float scale, hipStream_t stream) {

    using Elem = typename Mma::Elem;
    const dim3 grid((w_size + kNaQueries - 1) / kNaQueries, t_size * h_size, batch * num_heads);

#define COMFY_NA3D_LAUNCH(HD_)                                                                 \
    na3d_kernel<Mma, HD_><<<grid, kNaThreads, 0, stream>>>(                                    \
        reinterpret_cast<const Elem*>(q), reinterpret_cast<const Elem*>(k),                    \
        reinterpret_cast<const Elem*>(v), reinterpret_cast<Elem*>(out),                        \
        t_size, h_size, w_size, num_heads, s_b, s_t, s_h, s_w, s_n,                            \
        kt, kh, kw, causal_t, causal_h, causal_w, scale)

    switch (head_dim) {
        case 16: COMFY_NA3D_LAUNCH(16); break;
        case 32: COMFY_NA3D_LAUNCH(32); break;
        case 48: COMFY_NA3D_LAUNCH(48); break;
        case 64: COMFY_NA3D_LAUNCH(64); break;
        default:
            throw std::runtime_error("na3d supports head_dim 16/32/48/64, got " +
                                     std::to_string(head_dim));
    }
#undef COMFY_NA3D_LAUNCH
}

}  // namespace
}  // namespace comfy::hip_backend

// dtype_code is a DTYPE_TO_CODE value: 1 float16, 2 bfloat16.
extern "C" void launch_na3d_kernel(
    const void* q, const void* k, const void* v, void* out,
    int batch, int t_size, int h_size, int w_size, int num_heads, int head_dim,
    int kt, int kh, int kw,
    int causal_t, int causal_h, int causal_w,
    float scale, int dtype_code, hipStream_t stream) {

    using namespace comfy::hip_backend;
    if (batch <= 0 || t_size <= 0 || h_size <= 0 || w_size <= 0 || num_heads <= 0) {
        throw std::runtime_error("na3d: every extent must be positive");
    }

    // Contiguous (B, T, H, W, NH, HD) strides.
    const int64_t s_n = head_dim;
    const int64_t s_w = static_cast<int64_t>(num_heads) * head_dim;
    const int64_t s_h = static_cast<int64_t>(w_size) * s_w;
    const int64_t s_t = static_cast<int64_t>(h_size) * s_h;
    const int64_t s_b = static_cast<int64_t>(t_size) * s_t;

    // Non-causal kernels clamp to the axis length, which NATTEN would reject.
    const int ckt = causal_t ? kt : (kt < t_size ? kt : t_size);
    const int ckh = causal_h ? kh : (kh < h_size ? kh : h_size);
    const int ckw = causal_w ? kw : (kw < w_size ? kw : w_size);

    if (dtype_code == 1) {
        launch_na3d_typed<MmaF16>(q, k, v, out, batch, t_size, h_size, w_size, num_heads,
                                  head_dim, s_b, s_t, s_h, s_w, s_n, ckt, ckh, ckw,
                                  causal_t != 0, causal_h != 0, causal_w != 0, scale, stream);
    } else if (dtype_code == 2) {
        launch_na3d_typed<MmaBf16>(q, k, v, out, batch, t_size, h_size, w_size, num_heads,
                                   head_dim, s_b, s_t, s_h, s_w, s_n, ckt, ckh, ckw,
                                   causal_t != 0, causal_h != 0, causal_w != 0, scale, stream);
    } else {
        throw std::runtime_error("na3d supports FP16/BF16 only, got dtype code " +
                                 std::to_string(dtype_code));
    }
}
