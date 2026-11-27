// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Sol-Attn (arXiv 2607.24027) orchestration over one caller-allocated workspace:
//   preprocess  quantize Q/K/V, pool K/V per block, routing threshold
//   vtranspose  INT8 V^T for the exact stage's PV operand
//   route       choose the routed blocks; approximate tail from the centroids
//   token       token_aug > 0 only: list the top unrouted tokens per centroid and
//               replace route's pooled tail for the blocks they came from
//   exact       walk each routed list, then the token list, resuming the softmax
// Entry paths: `launch_sol_attn` (post-rope q/k/v) or the chunked producer
// (`sol_producer_begin/chunk`, then `launch_sol_attn_core`). Tensors are
// (B, T, H, 128) bf16 or fp16 (the chunked producer takes bf16 only) and the
// kernels need WMMA, so gfx11 and gfx12 only. On the q/k/v path k and v may be
// longer than q (seq_len queries, key_len keys): the query-side stages take one,
// the key-side stages the other.
//
// The workspace carve-up matches the CUDA backend's sol_attn.cu slot for slot, so
// the Python layer reads the same plan. What the slots CONTAIN differs by the two
// permutations the CUDA layout applies and this one does not; see sol_common.h.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "../launchers.h"
#include "sol_common.h"

// Sibling-TU launchers. C++ linkage on purpose: a drifted prototype fails to link
// instead of corrupting at runtime.
void launch_sol_preprocess(const void*, const void*, const void*, void*, void*, void*, void*,
                           void*, void*, void*, void*, void*, void*, void*, void*, void*,
                           const void*, const void*, const void*, int, int, int, int, int, int,
                           int, int,
                           int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                           int64_t, float, float, int, hipStream_t);
size_t sol_preprocess_scratch_bytes(int, int, int);
void launch_sol_producer(const void*, const void*, const void*, const void*, const void*,
                         const void*, void*, void*, void*, void*, void*, void*, void*, void*,
                         void*, void*, void*, void*, const void*, float, int, bool, int, int, int,
                         int, int, int, int, hipStream_t);
void launch_sol_finish(void*, void*, void*, void*, const void*, const void*, void*, const void*,
                       int, int, int, int, int, int, float, float, hipStream_t);
void launch_sol_vtranspose(const void*, const void*, void*, void*, int, int, int, int, int64_t,
                           int64_t, int64_t, int, hipStream_t, const void*);
void launch_sol_route(const void*, const void*, const void*, const void*, const void*, const void*,
                      const void*, void*, void*, void*, void*, void*, void*, void*, const void*,
                      int, int, int, int, int, int, int, int, int, int, int, int, int, float,
                      hipStream_t, int);
void launch_sol_topk_threshold(const void*, void*, int, int, int, int, int, int, hipStream_t);
void launch_sol_exact(const void*, const void*, const void*, const void*, const void*,
                      const void*, const void*, const void*, const void*, const void*,
                      const void*, const void*, const void*, const void*, int, void*, int, int,
                      int, int, int, int, float, int, hipStream_t, const void*, const void*,
                      const void*, const void*, const void*, const void*, int);
void launch_sol_summary(const void*, const void*, const void*, const void*, void*, void*, void*,
                        void*, int, int, int, int, int, int, int, float, hipStream_t);
void launch_sol_group_features(const void*, const void*, const void*, const void*, const void*,
                               void*, int, int, int, int, int, hipStream_t);
void launch_sol_group_leaves(const void*, void*, int, int, int, int, hipStream_t);
void launch_sol_reorder(const void*, const void*, const void*, const void*, const void*, void*,
                        void*, void*, void*, void*, void*, void*, void*, const void*, const void*,
                        const void*, const void*, int, int, int, int, int, int, int, int,
                        hipStream_t);
void launch_sol_token(const void*, const void*, const void*, void*, void*, void*, void*,
                      const void*, const void*, const void*, void*, void*, void*, void*, void*,
                      void*, void*, void*, void*, void*, void*, int, int, int, int, int, int, int,
                      int, int, float, hipStream_t);
int sol_token_splits(int, int, int);

namespace {

constexpr int HD = comfy::hip_backend::sol::kHeadDim;
constexpr int BLK = comfy::hip_backend::sol::kBlock;

// Slots start on 128-byte lines, as in the CUDA plan, so a staged tile never straddles one.
inline size_t align_slot(size_t n) { return (n + 127u) & ~static_cast<size_t>(127u); }

// Workspace carve-up, a function of the shape and the token budget; exported by
// sol_attn_plan.
struct Plan {
    int Tp, NTB, NPAD, NQ;   // padded key tokens, key blocks (and padded), query blocks
    size_t qiP, qs, kiP, ksb, vTi, vsc, kciP, kcs, vcT, thr, cen8, cens;
    size_t idx, cnt, oPart, mPart, lPart, statsV, qmean, vRow, tokIdx, tokCnt, tokRef, tokHist;
    size_t tokBits, tokTiles, tokTileCnt, gCen8, gCens, gRef, gBits;
    size_t tokPartO, tokPartM, tokPartL, skiP, sksb, svT, scratch;
    size_t rkiP, rksb, rvRow, total;   // raster k / v carriers staged for sol_attn_reorder

    Plan(int B, int T, int Tk, int H, int n_tok, bool reorder = false) {
        NTB = (Tk + BLK - 1) / BLK;
        Tp = NTB * BLK;
        NPAD = ((NTB + 63) / 64) * 64;  // route reads pooled keys in 64-block groups
        NQ = (T + BLK - 1) / BLK;
        const size_t bh = static_cast<size_t>(B) * H, tok = static_cast<size_t>(B) * T * H;
        size_t o = 0;
        auto take = [&](size_t bytes) {
            const size_t s = o;
            o = align_slot(o + bytes);
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
        qmean = take(bh * NQ * HD * sizeof(float));   // f32 query-block means (coarse, token)
        // token routing (empty at n_tok 0)
        const size_t on = n_tok > 0 ? 1 : 0;
        const size_t NG = (NQ + comfy::hip_backend::sol::kTokGroup - 1) /
                          comfy::hip_backend::sol::kTokGroup;
        const size_t W = (NTB + 31) / 32;
        vRow = take(on * bh * Tp * HD);
        tokRef = take(on * bh * NQ * sizeof(float));   // per query block (route)
        tokBits = take(bh * NQ * W * sizeof(uint32_t));   // also the summary tail's mask
        gCen8 = take(on * bh * NG * HD);               // per group
        gCens = take(on * bh * NG * sizeof(float));
        gRef = take(on * bh * NG * sizeof(float));
        gBits = take(on * bh * NG * W * sizeof(uint32_t));
        tokIdx = take(on * bh * NG * n_tok * sizeof(uint32_t));
        tokCnt = take(on * bh * NG * sizeof(int32_t));
        tokHist = take(on * bh * NG * comfy::hip_backend::sol::kTokHistBins * sizeof(uint32_t));
        const size_t groups = static_cast<size_t>((NG + 63) / 64) * bh;   // workgroups of 64 rows
        tokTiles = take(on * groups * NTB * sizeof(uint16_t));
        tokTileCnt = take(on * groups * sizeof(int32_t));
        const size_t parts =
            static_cast<size_t>(sol_token_splits(B, H, static_cast<int>(NG))) * bh * NG;
        tokPartO = take(on * parts * HD * sizeof(uint16_t));   // bf16 partial o
        tokPartM = take(on * parts * sizeof(float));
        tokPartL = take(on * parts * sizeof(float));
        // summary tail: one virtual key per key block
        skiP = take(bh * NPAD * HD);
        sksb = take(bh * NPAD * 2 * sizeof(float));
        svT = take(bh * HD * NPAD);
        scratch = take(sol_preprocess_scratch_bytes(B, H, NPAD));
        // last: no other offset may depend on `reorder`
        const size_t ro = reorder ? 1 : 0;
        rkiP = take(ro * bh * Tp * HD);
        rksb = take(ro * bh * Tp * 2 * sizeof(float));
        rvRow = take(ro * bh * Tp * HD);
        total = o;
    }
};

void validate_token_aug(int n_tok) {
    if (n_tok < 0 || n_tok > comfy::hip_backend::sol::kNTokMax || n_tok % BLK)
        throw std::runtime_error("sol_attn: token_aug must be a multiple of 64 up to " +
                                 std::to_string(comfy::hip_backend::sol::kNTokMax) + ", got " +
                                 std::to_string(n_tok));
}

void validate_shape(int batch, int seq_len, int key_len, int num_heads) {
    // a zero extent is a zero divisor in the Plan
    if (batch <= 0 || seq_len <= 0 || key_len <= 0 || num_heads <= 0)
        throw std::runtime_error(
            "sol_attn: batch, the query and key lengths and num_heads must be positive");
    if ((std::max(seq_len, key_len) + BLK - 1) / BLK > 65535)
        throw std::runtime_error("sol_attn: sequence too long for 16-bit block ids");
    // gridDim.y for every stage.
    if (static_cast<int64_t>(batch) * num_heads > 65535)
        throw std::runtime_error("sol_attn: batch * num_heads exceeds the 65535 grid limit");
}

// Throws if a launch since the last check was rejected, so the error is reported by
// the op that caused it.
void check_launch(const char* op) {
    const hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string(op) + ": kernel launch failed: " +
                                 hipGetErrorString(err));
    }
}

// route + exact over a populated workspace. The launch check covers every stage
// launched before it; without it a rejected launch leaves route's handover values
// in `out`.
void run_route_exact(const Plan& p, char* w, void* out, const void* blen, int tail, int batch,
                     int seq_len, int key_len, int num_heads, int sink_start, int sink_end,
                     int sink_q_start, int sink_q_end, int q_offset, float scale_log2, int elem,
                     int n_tok, hipStream_t stream,
                     const void* ord,   // row order of the carriers (output goes back through it), or null
                     bool q_raster,     // under ord: the q carriers are in source row order (producer path)
                     int topk_count) {  // >= 0: top-k routing with this many key blocks beyond the sinks
    if (tail < 0 || tail > 2)
        throw std::runtime_error("sol_attn: tail must be 0, 1 or 2, got " + std::to_string(tail));
    if (ord && (n_tok || blen || key_len != seq_len))
        throw std::runtime_error("sol_attn: a row order takes no token_aug, block_len or longer k");
    // tail 2: every unrouted key block reaches each row as one summary key, in the exact
    // kernel's ordered instantiation only
    const bool summary = tail == 2;
    if (summary && !ord)
        throw std::runtime_error("sol_attn: the per-row tail (tail 2) needs a row order");
    if (summary) {
        launch_sol_summary(w + p.kiP, w + p.ksb, w + p.vTi, w + p.qmean, w + p.scratch,
                           w + p.skiP, w + p.sksb, w + p.svT, batch, seq_len, p.Tp, num_heads,
                           p.NTB, p.NPAD, p.NQ, scale_log2, stream);
    }
    // always-exact band, relative to the query block: the key blocks holding its
    // own tokens (query i is key q_offset + i) and the 64 tokens on either side;
    // under a row order only its own block
    const int diag_lo = ord ? 0 : q_offset / BLK - 1;
    const int diag_span = ord ? 0 : 2 + (q_offset % BLK != 0);
    // top-k: a scores pass of route, then each query block's k-th largest score
    // replaces its tau threshold
    const void* thr = w + p.thr;
    if (topk_count >= 0) {
        launch_sol_route(w + p.cen8, w + p.cens, w + p.kciP, w + p.kcs, w + p.vcT, w + p.vsc, thr,
                         w + p.idx, w + p.cnt, w + p.oPart, w + p.mPart, w + p.lPart, nullptr,
                         nullptr, blen, tail, batch, key_len, num_heads, p.NTB, p.NPAD, p.NQ,
                         sink_start, sink_end, sink_q_start, sink_q_end, diag_lo, diag_span,
                         scale_log2, stream, 1);
        launch_sol_topk_threshold(w + p.idx, w + p.thr, batch * num_heads, p.NTB, p.NQ,
                                  topk_count, sink_start, sink_end, stream);
    }
    // with token routing (or the summary tail) route also flags the unrouted blocks
    launch_sol_route(w + p.cen8, w + p.cens, w + p.kciP, w + p.kcs, w + p.vcT, w + p.vsc, thr,
                     w + p.idx, w + p.cnt, w + p.oPart, w + p.mPart, w + p.lPart,
                     n_tok ? w + p.tokRef : nullptr, n_tok || summary ? w + p.tokBits : nullptr,
                     blen, tail, batch, key_len, num_heads, p.NTB, p.NPAD, p.NQ, sink_start,
                     sink_end, sink_q_start, sink_q_end, diag_lo, diag_span, scale_log2, stream,
                     0);
    if (n_tok)
        launch_sol_token(w + p.qmean, w + p.tokRef, w + p.tokBits, w + p.gCen8, w + p.gCens,
                         w + p.gRef, w + p.gBits, w + p.kiP, w + p.ksb, w + p.vTi, w + p.tokTiles,
                         w + p.tokTileCnt, w + p.tokHist, w + p.tokIdx, w + p.tokCnt,
                         w + p.tokPartO, w + p.tokPartM, w + p.tokPartL, w + p.oPart, w + p.mPart,
                         w + p.lPart, batch, p.Tp, num_heads, p.NQ, p.NTB, n_tok, tail,
                         sink_q_start, sink_q_end, scale_log2, stream);
    launch_sol_exact(w + p.qiP, w + p.qs, w + p.kiP, w + p.ksb, w + p.vTi, w + p.vsc, w + p.idx,
                     w + p.cnt, w + p.oPart, w + p.mPart, w + p.lPart, w + p.vRow, w + p.tokIdx,
                     w + p.tokCnt, n_tok, out, batch, seq_len, p.Tp, num_heads, p.NQ, p.NTB,
                     scale_log2, elem, stream, ord, q_raster ? ord : nullptr,
                     summary ? w + p.skiP : nullptr, summary ? w + p.sksb : nullptr,
                     summary ? w + p.svT : nullptr, summary ? w + p.tokBits : nullptr, p.NPAD);
    check_launch("sol_attn");
}

}  // namespace

using comfy::hip_backend::sol::sol_check;

// Plan dims and slot offsets, in the order sol_attn_plan_names lists them.
extern "C" const char* const sol_attn_plan_names[] = {
    "Tp",      "NTB",      "NPAD",       "NQ",       "qiP",      "qs",       "kiP",
    "ksb",     "vTi",      "vsc",        "kciP",     "kcs",      "vcT",      "thr",
    "cen8",    "cens",     "idx",        "cnt",      "oPart",    "mPart",    "lPart",
    "statsV",  "qmean",    "vRow",       "tokIdx",   "tokCnt",   "tokRef",   "tokHist",
    "tokBits", "tokTiles", "tokTileCnt", "gCen8",    "gCens",    "gRef",     "gBits",
    "tokPartO", "tokPartM", "tokPartL",  "skiP",     "sksb",     "svT",      "scratch",
    "rkiP",    "rksb",     "rvRow",      "total",    nullptr,
};

// Writes the values into out[0..count) and returns count; writes nothing if
// cap < count, so a caller with a fixed buffer can detect a grown Plan.
extern "C" int sol_attn_plan(int batch, int seq_len, int key_len, int num_heads, int n_tok,
                             int64_t* out, int cap, bool reorder) {
    validate_shape(batch, seq_len, key_len, num_heads);
    validate_token_aug(n_tok);
    const Plan p(batch, seq_len, key_len, num_heads, n_tok, reorder);
    const int64_t v[] = {
        p.Tp, p.NTB, p.NPAD, p.NQ,
        static_cast<int64_t>(p.qiP), static_cast<int64_t>(p.qs), static_cast<int64_t>(p.kiP),
        static_cast<int64_t>(p.ksb), static_cast<int64_t>(p.vTi), static_cast<int64_t>(p.vsc),
        static_cast<int64_t>(p.kciP), static_cast<int64_t>(p.kcs), static_cast<int64_t>(p.vcT),
        static_cast<int64_t>(p.thr), static_cast<int64_t>(p.cen8), static_cast<int64_t>(p.cens),
        static_cast<int64_t>(p.idx), static_cast<int64_t>(p.cnt), static_cast<int64_t>(p.oPart),
        static_cast<int64_t>(p.mPart), static_cast<int64_t>(p.lPart),
        static_cast<int64_t>(p.statsV), static_cast<int64_t>(p.qmean),
        static_cast<int64_t>(p.vRow), static_cast<int64_t>(p.tokIdx),
        static_cast<int64_t>(p.tokCnt), static_cast<int64_t>(p.tokRef),
        static_cast<int64_t>(p.tokHist), static_cast<int64_t>(p.tokBits),
        static_cast<int64_t>(p.tokTiles), static_cast<int64_t>(p.tokTileCnt),
        static_cast<int64_t>(p.gCen8), static_cast<int64_t>(p.gCens),
        static_cast<int64_t>(p.gRef), static_cast<int64_t>(p.gBits),
        static_cast<int64_t>(p.tokPartO), static_cast<int64_t>(p.tokPartM),
        static_cast<int64_t>(p.tokPartL), static_cast<int64_t>(p.skiP),
        static_cast<int64_t>(p.sksb), static_cast<int64_t>(p.svT),
        static_cast<int64_t>(p.scratch), static_cast<int64_t>(p.rkiP),
        static_cast<int64_t>(p.rksb), static_cast<int64_t>(p.rvRow),
        static_cast<int64_t>(p.total),
    };
    const int count = static_cast<int>(sizeof(v) / sizeof(v[0]));
    if (cap >= count)
        for (int i = 0; i < count; ++i) out[i] = v[i];
    return count;
}

// ---- chunked producer path ----
extern "C" void sol_producer_begin(void* workspace, int batch, int seq_len, int num_heads,
                                   int n_tok, hipStream_t stream) {
    validate_shape(batch, seq_len, seq_len, num_heads);
    validate_token_aug(n_tok);
    const Plan p(batch, seq_len, seq_len, num_heads, n_tok);
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
                                   int rot_dim, bool split_half, int t0, int M, int batch,
                                   int seq_len,
                                   int num_heads, int n_tok, hipStream_t stream,
                                   bool reorder) {   // sol_attn_reorder follows: k / V rows go to the staging slots
    validate_token_aug(n_tok);
    const Plan p(batch, seq_len, seq_len, num_heads, n_tok, reorder);
    char* w = reinterpret_cast<char*>(workspace);
    launch_sol_producer(qkv, fab, qw, kw, kmean, vscale, w + p.qiP, w + p.qs,
                        w + (reorder ? p.rkiP : p.kiP), w + (reorder ? p.rksb : p.ksb),
                        reorder ? nullptr : w + p.vTi,
                        reorder ? w + p.rvRow : (n_tok ? w + p.vRow : nullptr), w + p.vcT,
                        w + p.scratch, w + p.cen8, w + p.cens,
                        w + p.qmean, w + p.statsV, blen, rope_eps, rot_dim, split_half, t0, M,
                        seq_len, p.Tp, num_heads, p.NPAD, p.NQ, stream);
}

// vscale: the [B*H, HD] f32 scale the producer quantized V with.
// kmean_next / vamax_out ([B*H, HD] f32) receive this step's statistics.
extern "C" void launch_sol_attn_core(void* workspace, void* out, const void* vscale,
                                     void* kmean_next, void* vamax_out, const void* blen, int tail,
                                     int batch, int seq_len, int num_heads, float tau, float scale,
                                     int sink_start, int sink_end, int sink_q_start,
                                     int sink_q_end, int n_tok, hipStream_t stream,
                                     const void* order,   // [B*H, T] int32 row order of the carriers, or nullptr
                                     int topk_count) {    // >= 0: top-k with this many blocks beyond the sinks
    validate_shape(batch, seq_len, seq_len, num_heads);
    validate_token_aug(n_tok);
    const Plan p(batch, seq_len, seq_len, num_heads, n_tok);
    char* w = reinterpret_cast<char*>(workspace);
    const float scale_log2 = scale * 1.4426950408889634f;
    const size_t stats_bytes = static_cast<size_t>(batch) * num_heads * HD * sizeof(float);
    sol_check(hipMemcpyAsync(w + p.vsc, vscale, stats_bytes, hipMemcpyDeviceToDevice, stream),
              "staging the V scale");
    launch_sol_finish(w + p.scratch, w + p.kciP, w + p.kcs, w + p.thr, w + p.cen8, w + p.cens,
                      kmean_next, blen, batch, seq_len, num_heads, p.NTB, p.NPAD, p.NQ, tau,
                      scale_log2, stream);
    run_route_exact(p, w, out, blen, tail, batch, seq_len, seq_len, num_heads, sink_start,
                    sink_end, sink_q_start, sink_q_end, 0, scale_log2,
                    comfy::hip_backend::sol::kSolBf16, n_tok, stream, order, true, topk_count);
    sol_check(hipMemcpyAsync(vamax_out, w + p.statsV, stats_bytes, hipMemcpyDeviceToDevice, stream),
              "reading back the V absmax");
}

extern "C" void launch_sol_attn(const void* q, const void* k, const void* v, void* out,
                                void* workspace, int batch, int seq_len, int key_len,
                                int num_heads, int head_dim, int elem, float tau, float scale,
                                const void* key_bias, const void* blen, int tail,
                                int sink_start, int sink_end, int sink_q_start, int sink_q_end,
                                int q_offset, int64_t qs_b, int64_t qs_t, int64_t qs_h,
                                int64_t ks_b,
                                int64_t ks_t, int64_t ks_h, int64_t vs_b, int64_t vs_t,
                                int64_t vs_h, int n_tok, hipStream_t stream,
                                const void* order,   // [B*H, T] int32 per-head row order, or nullptr
                                int topk_count) {    // >= 0: top-k with this many blocks beyond the sinks
    if (head_dim != HD)
        throw std::runtime_error("sol_attn supports head_dim 128, got " + std::to_string(head_dim));
    validate_shape(batch, seq_len, key_len, num_heads);
    validate_token_aug(n_tok);

    const Plan p(batch, seq_len, key_len, num_heads, n_tok);
    char* w = reinterpret_cast<char*>(workspace);
    const float scale_log2 = scale * 1.4426950408889634f;
    launch_sol_preprocess(q, k, v, w + p.qiP, w + p.qs, w + p.kiP, w + p.ksb, w + p.kciP,
                          w + p.kcs, w + p.vcT, w + p.thr, w + p.cen8, w + p.cens, w + p.vsc,
                          w + p.qmean, w + p.scratch, key_bias, blen, order, batch, seq_len,
                          key_len,
                          p.Tp, num_heads, p.NTB, p.NPAD, p.NQ, qs_b, qs_t, qs_h, ks_b, ks_t,
                          ks_h, vs_b, vs_t, vs_h, tau, scale_log2, elem, stream);
    launch_sol_vtranspose(v, w + p.vsc, w + p.vTi, n_tok ? w + p.vRow : nullptr, batch, key_len,
                          p.Tp, num_heads, vs_b, vs_t, vs_h, elem, stream, order);
    run_route_exact(p, w, out, blen, tail, batch, seq_len, key_len, num_heads, sink_start,
                    sink_end, sink_q_start, sink_q_end, q_offset, scale_log2, elem, n_tok, stream,
                    order, false, topk_count);
}

// Row-order planner: grouping features of the staged keys of blocks [b0, b0 + nblocks)
// (group_features).
extern "C" void sol_group_features(void* workspace, const void* dir8, const void* dsc,
                                   const void* doff, void* feat, int batch, int seq_len,
                                   int num_heads, int b0, int nblocks, hipStream_t stream) {
    validate_shape(batch, seq_len, seq_len, num_heads);
    const Plan p(batch, seq_len, seq_len, num_heads, 0, true);
    if (b0 < 0 || nblocks <= 0 || (b0 + nblocks) * BLK > seq_len)
        throw std::runtime_error("sol_group_features: blocks must be whole blocks of the sequence");
    char* w = reinterpret_cast<char*>(workspace);
    launch_sol_group_features(w + p.rkiP, w + p.rksb, dir8, dsc, doff, feat, batch * num_heads,
                              p.Tp, b0, nblocks, nblocks * BLK + 1, stream);
    check_launch("sol_group_features");
}

// Row-order planner: the last levels of the grouping tree, in place on perm (group_leaves).
extern "C" void sol_group_leaves(const void* feat, void* perm, int bh, int n, int size, int total,
                                 hipStream_t stream) {
    if (bh <= 0 || bh > 65535)   // gridDim.y
        throw std::runtime_error("sol_group_leaves: batch * num_heads must be in [1, 65535], got " +
                                 std::to_string(bh));
    launch_sol_group_leaves(feat, perm, bh, n, size, total, stream);
    check_launch("sol_group_leaves");
}

// The staged raster carriers -> the kernels' carriers and block statistics in `order`,
// for launch_sol_attn_core with the same order. kmean / vscale: what the producer
// quantized with.
extern "C" void sol_attn_reorder(void* workspace, const void* order, const void* kmean,
                                 const void* vscale, int batch, int seq_len, int num_heads,
                                 hipStream_t stream,
                                 const void* vfac,   // per-chunk V scale / vscale [chunks, B*H, HD], or null
                                 int vfac_rows) {    // rows per producer chunk
    validate_shape(batch, seq_len, seq_len, num_heads);
    const Plan p(batch, seq_len, seq_len, num_heads, 0, true);
    char* w = reinterpret_cast<char*>(workspace);
    launch_sol_reorder(w + p.qiP, w + p.qs, w + p.rkiP, w + p.rksb, w + p.rvRow, w + p.kiP,
                       w + p.ksb, w + p.vTi, w + p.vcT, w + p.scratch, w + p.cen8, w + p.cens,
                       w + p.qmean, order, kmean, vscale, vfac, vfac_rows, batch, seq_len, p.Tp,
                       num_heads, p.NTB, p.NPAD, p.NQ, stream);
    check_launch("sol_attn_reorder");
}
