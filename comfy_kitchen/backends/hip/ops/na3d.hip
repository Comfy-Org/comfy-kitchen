// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Fused 3D neighborhood attention (NATTEN na3d semantics, dilation 1) and
// window attention over a stack of key planes.
//
// na3d: per non-causal axis each query attends a window of exactly kernel_size
// positions centered on it, shifted inward at grid boundaries; per causal axis it
// attends the min(i+1, kernel_size) nearest previous positions.
//
// window_attention: the Tq query planes are the last of the Tk key planes and
// each attends its temporal_window most recent key planes. Every key plane has
// its own (kh, kw) spatial window, shifted inward at the frame border (clamp)
// or cut off by it. k/v may have fewer heads than q (grouped-query attention).
//
// Flash-attention-2 style on the 16x16x16 WMMA: one wave covers 16 queries along
// W at a fixed (t, h) and sweeps that block's key span 16 keys at a time. Scores,
// softmax state and the output accumulator stay in registers; LDS holds only the
// transposed V tile and the probability tile.
//
// The T and H windows are constant across a block, so only W is masked per
// element. head_dim in {16, 32, 48, 64, 128}; q/k/v/out are contiguous
// (B, T, H, W, heads, HD).
#include <stdexcept>
#include <string>

#include "../launchers.h"
#include "../mma.h"

namespace comfy::hip_backend {

// One wave, 16 queries. A wider block would only add key tiles that most of it
// then skips, since the window a query sees is narrow by construction.
constexpr int kNaQueries = 16;
constexpr int kNaKeys = 16;
constexpr int kNaThreads = 32;
constexpr int kNaMaxWindows = 8;
// LDS rows are padded to 9 dwords. LDS has 32 four-byte banks and gcd(9, 32) = 1,
// so the 16 row-strided lanes of a fragment read reach 16 distinct banks; an
// 8-dword or 12-dword row collapses them onto 4 or 8.
constexpr int kNaRowStride = 18;
constexpr float kNaNegInf = -3.0e38f;

namespace {

__forceinline__ __host__ __device__ int na_min(int a, int b) { return a < b ? a : b; }
__forceinline__ __device__ int na_max(int a, int b) { return a > b ? a : b; }

// Window [lo, hi) of index i along one axis. A non-causal window is shifted
// inward at the border when clamp is set (NATTEN), otherwise cut off by it.
__forceinline__ __device__ void axis_window(int i, int k, int len, bool causal, bool clamp,
                                            int& lo, int& hi) {
    if (causal) {
        lo = na_max(i - k + 1, 0);
        hi = i + 1;
    } else if (clamp) {
        lo = na_min(na_max(i - k / 2, 0), len - k);
        hi = lo + k;
    } else {
        lo = na_max(i - k / 2, 0);
        hi = na_min(i - k / 2 + k, len);
    }
}

// Query plane t attends the key planes in the T-axis window of t + t_offset over
// tk planes; key plane p uses spatial window (kh, kw)[n_windows == 1 ? 0 : p].
struct NaParams {
    int tk, h_size, w_size, num_heads, group;  // group: q heads per kv head
    int64_t q_b, q_t, q_h, q_w, q_n;           // q / out strides
    int64_t k_b, k_t, k_h, k_w, k_n;           // k / v strides
    int kt, t_offset;
    bool causal_t, causal_h, causal_w, clamp;
    int n_windows;
    int kh[kNaMaxWindows], kw[kNaMaxWindows];
    float scale;
};

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
    const NaParams p) {

    using Elem = typename Mma::Elem;
    constexpr int KC = HD / 16;   // k-chunks of S = Q.K^T
    constexpr int NT = HD / 16;   // output column tiles of O += P.V
    constexpr int LD = kNaRowStride;

    __shared__ Elem sVt[HD * LD];         // transposed V: sVt[column][key]
    __shared__ Elem sP[kNaQueries * LD];  // probabilities: sP[query][key]

    const int lane = threadIdx.x;
    const int r = frag_row(lane);  // fragment row, and this lane's score column
    const int h_size = p.h_size;
    const int w_size = p.w_size;
    const int t_q = blockIdx.y / h_size;
    const int h_q = blockIdx.y % h_size;
    const int b = blockIdx.z / p.num_heads;
    const int n = blockIdx.z % p.num_heads;
    const int64_t q_base = static_cast<int64_t>(b) * p.q_b + static_cast<int64_t>(n) * p.q_n;
    const int64_t k_base =
        static_cast<int64_t>(b) * p.k_b + static_cast<int64_t>(n / p.group) * p.k_n;

    const int w0 = blockIdx.x * kNaQueries;
    const int n_valid_q = na_min(kNaQueries, w_size - w0);
    const int w_first = na_min(w0, w_size - 1);
    const int w_last = na_min(w0 + n_valid_q - 1, w_size - 1);

    int t_lo, t_hi;
    axis_window(t_q + p.t_offset, p.kt, p.tk, p.causal_t, true, t_lo, t_hi);

    // Loaded once, register-resident for the whole kernel. A row past the grid
    // clamps to a real one; it is never stored back.
    typename Mma::Frag qa[KC];
    {
        const int64_t qrow = q_base + static_cast<int64_t>(t_q) * p.q_t +
                             static_cast<int64_t>(h_q) * p.q_h +
                             static_cast<int64_t>(na_min(w0 + r, w_size - 1)) * p.q_w;
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
        const int window = p.n_windows == 1 ? 0 : tk;
        const int kh = p.kh[window];
        const int kw = p.kw[window];

        int h_lo, h_hi;
        axis_window(h_q, kh, h_size, p.causal_h, p.clamp, h_lo, h_hi);

        // The eight accumulator rows this lane owns, and each one's W window.
        int ws[8], we[8];
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int wq = na_min(w0 + acc_row(lane, e), w_size - 1);
            axis_window(wq, kw, w_size, p.causal_w, p.clamp, ws[e], we[e]);
        }

        // Key span of the block: the first query's window start through the last
        // valid query's window end.
        int w_lo, w_hi, unused;
        axis_window(w_first, kw, w_size, p.causal_w, p.clamp, w_lo, unused);
        axis_window(w_last, kw, w_size, p.causal_w, p.clamp, unused, w_hi);

        for (int hk = h_lo; hk < h_hi; ++hk) {
            const int64_t plane = k_base + static_cast<int64_t>(tk) * p.k_t +
                                  static_cast<int64_t>(hk) * p.k_h;
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
                                              static_cast<int64_t>(wk0 + kk) * p.k_w + c8 * 8;
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
                        plane + static_cast<int64_t>(na_min(wk0 + r, w_size - 1)) * p.k_w;
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
                    p_val[e] = vis ? Mma::get(s_acc, e) * p.scale : kNaNegInf;
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

    const int64_t orow = q_base + static_cast<int64_t>(t_q) * p.q_t +
                         static_cast<int64_t>(h_q) * p.q_h;
    #pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int row = acc_row(lane, e);
        if (row >= n_valid_q) {
            continue;
        }
        const float inv_l = 1.0f / fmaxf(l_r[e], 1e-30f);
        Elem* dst = out + orow + static_cast<int64_t>(w0 + row) * p.q_w + acc_col(lane);
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            dst[nt * 16] = static_cast<Elem>(Mma::get(o_acc[nt], e) * inv_l);
        }
    }
}

template <typename Mma>
void launch_na3d_typed(const void* q, const void* k, const void* v, void* out, const NaParams& p,
                       int batch, int tq, int head_dim, hipStream_t stream) {
    using Elem = typename Mma::Elem;
    const dim3 grid((p.w_size + kNaQueries - 1) / kNaQueries, tq * p.h_size,
                    batch * p.num_heads);

#define COMFY_NA3D_LAUNCH(HD_)                                                                 \
    na3d_kernel<Mma, HD_><<<grid, kNaThreads, 0, stream>>>(                                    \
        reinterpret_cast<const Elem*>(q), reinterpret_cast<const Elem*>(k),                    \
        reinterpret_cast<const Elem*>(v), reinterpret_cast<Elem*>(out), p)

    switch (head_dim) {
        case 16: COMFY_NA3D_LAUNCH(16); break;
        case 32: COMFY_NA3D_LAUNCH(32); break;
        case 48: COMFY_NA3D_LAUNCH(48); break;
        case 64: COMFY_NA3D_LAUNCH(64); break;
        case 128: COMFY_NA3D_LAUNCH(128); break;
        default:
            throw std::runtime_error("na3d supports head_dim 16/32/48/64/128, got " +
                                     std::to_string(head_dim));
    }
#undef COMFY_NA3D_LAUNCH
}

// Contiguous (B, T, H, W, heads, HD) strides.
void set_strides(int t_size, int h_size, int w_size, int heads, int head_dim, int64_t& s_b,
                 int64_t& s_t, int64_t& s_h, int64_t& s_w, int64_t& s_n) {
    s_n = head_dim;
    s_w = static_cast<int64_t>(heads) * head_dim;
    s_h = static_cast<int64_t>(w_size) * s_w;
    s_t = static_cast<int64_t>(h_size) * s_h;
    s_b = static_cast<int64_t>(t_size) * s_t;
}

// dtype_code is a DTYPE_TO_CODE value: 1 float16, 2 bfloat16.
void launch(const void* q, const void* k, const void* v, void* out, const NaParams& p,
            int batch, int tq, int head_dim, int dtype_code, hipStream_t stream) {
    if (dtype_code == 1) {
        launch_na3d_typed<MmaF16>(q, k, v, out, p, batch, tq, head_dim, stream);
    } else if (dtype_code == 2) {
        launch_na3d_typed<MmaBf16>(q, k, v, out, p, batch, tq, head_dim, stream);
    } else {
        throw std::runtime_error("na3d supports FP16/BF16 only, got dtype code " +
                                 std::to_string(dtype_code));
    }
}

}  // namespace
}  // namespace comfy::hip_backend

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

    NaParams p{};
    p.tk = t_size;
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
    // Non-causal kernels clamp to the axis length, which NATTEN would reject.
    p.kt = p.causal_t ? kt : na_min(kt, t_size);
    p.n_windows = 1;
    p.kh[0] = p.causal_h ? kh : na_min(kh, h_size);
    p.kw[0] = p.causal_w ? kw : na_min(kw, w_size);
    p.scale = scale;
    launch(q, k, v, out, p, batch, t_size, head_dim, dtype_code, stream);
}

extern "C" void launch_window_attention_kernel(
    const void* q, const void* k, const void* v, void* out,
    int batch, int tq, int tk, int h_size, int w_size, int num_heads, int kv_heads, int head_dim,
    const int* kh, const int* kw, int n_windows, int temporal_window, int clamp,
    float scale, int dtype_code, hipStream_t stream) {

    using namespace comfy::hip_backend;
    if (batch <= 0 || tq <= 0 || tk < tq || h_size <= 0 || w_size <= 0 || kv_heads <= 0 ||
        num_heads % kv_heads != 0 || temporal_window <= 0) {
        throw std::runtime_error("window_attention: invalid extents");
    }
    if (n_windows < 1 || n_windows > kNaMaxWindows || (n_windows != 1 && n_windows != tk)) {
        throw std::runtime_error("window_attention: needs 1 window or one per key plane, at most " +
                                 std::to_string(kNaMaxWindows));
    }

    NaParams p{};
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
        if (kh[i] <= 0 || kw[i] <= 0) {
            throw std::runtime_error("window_attention: windows must be positive");
        }
        // A clamped window larger than the frame covers the whole axis.
        p.kh[i] = p.clamp ? na_min(kh[i], h_size) : kh[i];
        p.kw[i] = p.clamp ? na_min(kw[i], w_size) : kw[i];
    }
    p.scale = scale;
    launch(q, k, v, out, p, batch, tq, head_dim, dtype_code, stream);
}
