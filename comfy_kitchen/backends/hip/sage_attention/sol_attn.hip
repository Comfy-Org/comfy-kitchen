// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Sol-Attn (arXiv 2607.24027) orchestration over one caller-allocated workspace:
//   preprocess  quantize Q/K/V, pool K/V per block, routing threshold
//   vtranspose  INT8 V^T for the exact stage's PV operand
//   route       choose the routed blocks; approximate tail from the centroids
//   exact       walk each routed list, resuming route's online softmax
// Entry paths: `launch_sol_attn` (post-rope q/k/v) or the chunked producer
// (`sol_producer_begin/chunk`, then `launch_sol_attn_core`). Tensors are
// (B, T, H, 128) bf16 or fp16 (the chunked producer takes bf16 only) and the
// kernels need WMMA, so gfx11 and gfx12 only.
//
// The workspace carve-up matches the CUDA backend's sol_attn.cu slot for slot, so
// the Python layer reads the same plan. What the slots CONTAIN differs by the two
// permutations the CUDA layout applies and this one does not; see sol_common.h.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "../launchers.h"
#include "sol_common.h"

// Sibling-TU launchers. C++ linkage on purpose: a drifted prototype fails to link
// instead of corrupting at runtime.
void launch_sol_preprocess(const void*, const void*, const void*, void*, void*, void*, void*,
                           void*, void*, void*, void*, void*, void*, void*, void*, void*,
                           const void*, const void*, int, int, int, int, int, int, int, int64_t,
                           int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                           float, float, int, hipStream_t);
size_t sol_preprocess_scratch_bytes(int, int, int);
void launch_sol_producer(const void*, const void*, const void*, const void*, const void*,
                         const void*, void*, void*, void*, void*, void*, void*, void*, void*,
                         void*, void*, void*, const void*, float, int, int, int, int, int, int,
                         int, int, hipStream_t);
void launch_sol_finish(void*, void*, void*, void*, const void*, const void*, void*, const void*,
                       int, int, int, int, int, int, float, float, hipStream_t);
void launch_sol_vtranspose(const void*, const void*, void*, int, int, int, int, int64_t, int64_t,
                           int64_t, int, hipStream_t);
void launch_sol_route(const void*, const void*, const void*, const void*, const void*, const void*,
                      const void*, void*, void*, void*, void*, void*, const void*, int, int, int,
                      int, int, int, int, int, int, int, int, float, hipStream_t);
void launch_sol_exact(const void*, const void*, const void*, const void*, const void*,
                      const void*, const void*, const void*, const void*, const void*,
                      const void*, void*, int, int, int, int, int, int, float, int, hipStream_t);

namespace {

constexpr int HD = comfy::hip_backend::sol::kHeadDim;
constexpr int BLK = comfy::hip_backend::sol::kBlock;

inline size_t align16(size_t n) { return (n + 15u) & ~static_cast<size_t>(15u); }

// Workspace carve-up, a pure function of the shape; exported by sol_attn_plan.
struct Plan {
    int Tp, NTB, NPAD, NQ;
    size_t qiP, qs, kiP, ksb, vTi, vsc, kciP, kcs, vcT, thr, cen8, cens;
    size_t idx, cnt, oPart, mPart, lPart, statsV, qmean, scratch, total;

    Plan(int B, int T, int H) {
        NTB = (T + BLK - 1) / BLK;
        Tp = NTB * BLK;
        NPAD = ((NTB + 63) / 64) * 64;  // route reads pooled keys in 64-block groups
        NQ = NTB;
        const size_t bh = static_cast<size_t>(B) * H, tok = static_cast<size_t>(B) * T * H;
        size_t o = 0;
        auto take = [&](size_t bytes) {
            const size_t s = o;
            o = align16(o + bytes);
            return s;
        };
        qiP = take(tok * HD);
        qs = take(tok * sizeof(float));
        kiP = take(bh * Tp * HD);
        ksb = take(bh * Tp * 2 * sizeof(float));
        vTi = take(bh * HD * Tp);
        vsc = take(bh * HD * sizeof(float));
        kciP = take(bh * NPAD * HD);
        kcs = take(bh * NPAD * sizeof(float));
        vcT = take(bh * HD * NPAD * sizeof(uint16_t));
        thr = take(bh * NQ * sizeof(float));
        idx = take(bh * NQ * NTB * sizeof(uint16_t));  // the only T^2 term
        cnt = take(bh * NQ * sizeof(int32_t));
        cen8 = take(bh * NQ * HD);
        cens = take(bh * NQ * sizeof(float));
        // route -> exact handover: one (o, m, l) per (b, h, query block)
        oPart = take(bh * NQ * HD * sizeof(uint16_t));
        mPart = take(bh * NQ * sizeof(float));
        lPart = take(bh * NQ * sizeof(float));
        statsV = take(bh * HD * sizeof(float));         // producer vamax accumulator
        qmean = take(bh * NPAD * HD * sizeof(float));   // f32 query-block means (coarse branch)
        scratch = take(sol_preprocess_scratch_bytes(B, H, NPAD));
        total = o;
    }
};

void validate_shape(int batch, int seq_len, int num_heads) {
    if ((seq_len + BLK - 1) / BLK > 65535)
        throw std::runtime_error("sol_attn: seq_len too long for 16-bit block ids");
    // gridDim.y for every stage.
    if (static_cast<int64_t>(batch) * num_heads > 65535)
        throw std::runtime_error("sol_attn: batch * num_heads exceeds the 65535 grid limit");
}

// route + exact over a populated workspace. The hipGetLastError covers every stage
// launched before it; without it a rejected launch leaves route's handover values
// in `out`.
void run_route_exact(const Plan& p, char* w, const void* ext_threshold, void* out,
                     const void* blen, int tail, int batch, int seq_len, int num_heads,
                     int sink_start, int sink_end, int sink_q_start, int sink_q_end,
                     float scale_log2, int elem, hipStream_t stream) {
    // top-k mode: the caller supplies the per-query-block threshold
    const void* thr = ext_threshold ? ext_threshold : static_cast<const void*>(w + p.thr);
    launch_sol_route(w + p.cen8, w + p.cens, w + p.kciP, w + p.kcs, w + p.vcT, w + p.vsc, thr,
                     w + p.idx, w + p.cnt, w + p.oPart, w + p.mPart, w + p.lPart, blen, tail,
                     batch, seq_len, num_heads, p.NTB, p.NPAD, p.NQ, sink_start, sink_end,
                     sink_q_start, sink_q_end, scale_log2, stream);
    launch_sol_exact(w + p.qiP, w + p.qs, w + p.kiP, w + p.ksb, w + p.vTi, w + p.vsc, w + p.idx,
                     w + p.cnt, w + p.oPart, w + p.mPart, w + p.lPart, out, batch, seq_len, p.Tp,
                     num_heads, p.NQ, p.NTB, scale_log2, elem, stream);
    const hipError_t err = hipGetLastError();
    if (err != hipSuccess)
        throw std::runtime_error(std::string("sol_attn: kernel launch failed: ") +
                                 hipGetErrorString(err));
}

}  // namespace

using comfy::hip_backend::sol::sol_check;

// Plan dims and slot offsets, in the order sol_attn_plan_names lists them.
extern "C" const char* const sol_attn_plan_names[] = {
    "Tp",   "NTB",  "NPAD", "NQ",     "qiP",   "qs",    "kiP",     "ksb",   "vTi",
    "vsc",  "kciP", "kcs",  "vcT",    "thr",   "cen8",  "cens",    "idx",   "cnt",
    "oPart", "mPart", "lPart", "statsV", "qmean", "scratch", "total", nullptr,
};

// Writes the values into out[0..count) and returns count; writes nothing if
// cap < count, so a caller with a fixed buffer can detect a grown Plan.
extern "C" int sol_attn_plan(int batch, int seq_len, int num_heads, int64_t* out, int cap) {
    const Plan p(batch, seq_len, num_heads);
    const int64_t v[] = {
        p.Tp, p.NTB, p.NPAD, p.NQ,
        static_cast<int64_t>(p.qiP), static_cast<int64_t>(p.qs), static_cast<int64_t>(p.kiP),
        static_cast<int64_t>(p.ksb), static_cast<int64_t>(p.vTi), static_cast<int64_t>(p.vsc),
        static_cast<int64_t>(p.kciP), static_cast<int64_t>(p.kcs), static_cast<int64_t>(p.vcT),
        static_cast<int64_t>(p.thr), static_cast<int64_t>(p.cen8), static_cast<int64_t>(p.cens),
        static_cast<int64_t>(p.idx), static_cast<int64_t>(p.cnt), static_cast<int64_t>(p.oPart),
        static_cast<int64_t>(p.mPart), static_cast<int64_t>(p.lPart),
        static_cast<int64_t>(p.statsV), static_cast<int64_t>(p.qmean),
        static_cast<int64_t>(p.scratch), static_cast<int64_t>(p.total),
    };
    const int count = static_cast<int>(sizeof(v) / sizeof(v[0]));
    if (cap >= count)
        for (int i = 0; i < count; ++i) out[i] = v[i];
    return count;
}

// ---- chunked producer path ----
extern "C" void sol_producer_begin(void* workspace, int batch, int seq_len, int num_heads,
                                   hipStream_t stream) {
    validate_shape(batch, seq_len, num_heads);
    const Plan p(batch, seq_len, num_heads);
    char* w = reinterpret_cast<char*>(workspace);
    const size_t bh = static_cast<size_t>(batch) * num_heads;
    // route reads vcT's pad columns; statsV is an atomicMax accumulator
    sol_check(hipMemsetAsync(w + p.vcT, 0, bh * HD * p.NPAD * sizeof(uint16_t), stream),
              "clearing the pooled V slot");
    sol_check(hipMemsetAsync(w + p.statsV, 0, bh * HD * sizeof(float), stream),
              "clearing the V absmax accumulator");
}

extern "C" void sol_producer_chunk(void* workspace, const void* qkv, const void* fab,
                                   const void* qw, const void* kw, const void* kmean,
                                   const void* vscale, const void* blen, float rope_eps,
                                   int rot_dim, int t0, int M, int batch, int seq_len,
                                   int num_heads, hipStream_t stream) {
    const Plan p(batch, seq_len, num_heads);
    char* w = reinterpret_cast<char*>(workspace);
    launch_sol_producer(qkv, fab, qw, kw, kmean, vscale, w + p.qiP, w + p.qs, w + p.kiP,
                        w + p.ksb, w + p.vTi, w + p.vcT, w + p.scratch, w + p.cen8, w + p.cens,
                        w + p.qmean, w + p.statsV, blen, rope_eps, rot_dim, t0, M, seq_len, p.Tp,
                        num_heads, p.NPAD, p.NQ, stream);
}

// vscale: the [B*H, HD] f32 scale the producer quantized V with.
// kmean_next / vamax_out ([B*H, HD] f32) receive this step's statistics.
extern "C" void launch_sol_attn_core(void* workspace, void* out, const void* vscale,
                                     void* kmean_next, void* vamax_out, const void* blen, int tail,
                                     int batch, int seq_len, int num_heads, float tau, float scale,
                                     const void* ext_threshold, int sink_start, int sink_end,
                                     int sink_q_start, int sink_q_end, hipStream_t stream) {
    validate_shape(batch, seq_len, num_heads);
    const Plan p(batch, seq_len, num_heads);
    char* w = reinterpret_cast<char*>(workspace);
    const float scale_log2 = scale * 1.4426950408889634f;
    const size_t stats_bytes = static_cast<size_t>(batch) * num_heads * HD * sizeof(float);
    sol_check(hipMemcpyAsync(w + p.vsc, vscale, stats_bytes, hipMemcpyDeviceToDevice, stream),
              "staging the V scale");
    launch_sol_finish(w + p.scratch, w + p.kciP, w + p.kcs, w + p.thr, w + p.cen8, w + p.cens,
                      kmean_next, blen, batch, seq_len, num_heads, p.NTB, p.NPAD, p.NQ, tau,
                      scale_log2, stream);
    run_route_exact(p, w, ext_threshold, out, blen, tail, batch, seq_len, num_heads, sink_start,
                    sink_end, sink_q_start, sink_q_end, scale_log2,
                    comfy::hip_backend::sol::kSolBf16, stream);
    sol_check(hipMemcpyAsync(vamax_out, w + p.statsV, stats_bytes, hipMemcpyDeviceToDevice, stream),
              "reading back the V absmax");
}

extern "C" void launch_sol_attn(const void* q, const void* k, const void* v, void* out,
                                void* workspace, int batch, int seq_len, int num_heads,
                                int head_dim, int elem, float tau, float scale,
                                const void* key_bias,
                                const void* ext_threshold, const void* blen, int tail,
                                int sink_start, int sink_end, int sink_q_start, int sink_q_end,
                                int64_t qs_b, int64_t qs_t, int64_t qs_h, int64_t ks_b,
                                int64_t ks_t, int64_t ks_h, int64_t vs_b, int64_t vs_t,
                                int64_t vs_h, hipStream_t stream) {
    if (head_dim != HD)
        throw std::runtime_error("sol_attn supports head_dim 128, got " + std::to_string(head_dim));
    validate_shape(batch, seq_len, num_heads);

    const Plan p(batch, seq_len, num_heads);
    char* w = reinterpret_cast<char*>(workspace);
    const float scale_log2 = scale * 1.4426950408889634f;
    launch_sol_preprocess(q, k, v, w + p.qiP, w + p.qs, w + p.kiP, w + p.ksb, w + p.kciP,
                          w + p.kcs, w + p.vcT, w + p.thr, w + p.cen8, w + p.cens, w + p.vsc,
                          w + p.qmean, w + p.scratch, key_bias, blen, batch, seq_len, p.Tp,
                          num_heads, p.NTB, p.NPAD, p.NQ, qs_b, qs_t, qs_h, ks_b, ks_t, ks_h,
                          vs_b, vs_t, vs_h, tau, scale_log2, elem, stream);
    launch_sol_vtranspose(v, w + p.vsc, w + p.vTi, batch, seq_len, p.Tp, num_heads, vs_b, vs_t,
                          vs_h, elem, stream);
    run_route_exact(p, w, ext_threshold, out, blen, tail, batch, seq_len, num_heads, sink_start,
                    sink_end, sink_q_start, sink_q_end, scale_log2, elem, stream);
}
