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

// Sol-Attn (arXiv 2607.24027) orchestration over one caller-allocated workspace:
//   preprocess  quantize Q/K/V, pool K/V per block, routing threshold
//   vtranspose  INT8 V^T for the exact stage's PV operand
//   route       choose the routed blocks; approximate tail from the centroids
//   exact       walk each routed list, resuming route's online softmax
// Entry paths: `launch_sol_attn` (post-rope q/k/v) or the chunked producer
// (`sol_producer_begin/chunk`, then `launch_sol_attn_core`). sm_80+, tuned
// for sm_120; tensors are (B, T, H, 128) bf16 or fp16 (the chunked producer
// takes bf16 only).

#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "sol_layout.cuh"

// Sibling-TU launchers. C++ linkage on purpose: a drifted prototype fails to
// link instead of corrupting at runtime.
void launch_sol_preprocess(const void*, const void*, const void*, void*, void*, void*,
                           void*, void*, void*, void*, void*, void*, void*, void*, void*,
                           void*, const void*, const void*,
                           int, int, int, int, int, int, int,
                           int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                           int64_t, int64_t, int64_t, float, float, int, cudaStream_t);
size_t sol_preprocess_scratch_bytes(int, int, int);
void launch_sol_producer(const void*, const void*, const void*, const void*,
                         const void*, const void*, void*, void*, void*, void*,
                         void*, void*, void*, void*, void*, void*, void*, void*,
                         const void*, float, int, int, int, int, int, int, int, int,
                         cudaStream_t);
void launch_sol_finish(void*, void*, void*, void*, const void*, const void*, void*,
                       const void*, int, int, int, int, int, int, float, float,
                       cudaStream_t);
void launch_sol_vtranspose(const void*, const void*, void*, void*, int, int, int, int,
                           int64_t, int64_t, int64_t, int, cudaStream_t);
void launch_sol_route(const void*, const void*, const void*, const void*, const void*,
                      const void*, const void*, void*, void*, void*, void*, void*, void*,
                      void*, const void*, int,
                      int, int, int, int, int, int, int, int, int, int,
                      float, cudaStream_t);
void launch_sol_exact(const void*, const void*, const void*, const void*, const void*,
                      const void*, const void*, const void*, const void*, const void*,
                      const void*, const void*, const void*, const void*, int, void*,
                      int, int, int, int, int, int, float, int, cudaStream_t);
void launch_sol_token(const void*, const void*, const void*, void*, void*, void*, void*,
                      const void*, const void*, const void*, void*, void*,
                      void*, void*, void*, void*, void*, void*, void*, void*, void*,
                      int, int, int, int, int, int, int, int, int, int, float, cudaStream_t);
int sol_token_splits(int, int, int);

namespace {

constexpr int HD = sol::HEAD_DIM;
constexpr int BLK = sol::BLOCK;

inline size_t align16(size_t n) { return (n + 15u) & ~(size_t)15u; }

// Workspace carve-up, a function of the shape and the token budget; exported by sol_attn_plan.
struct Plan {
    int Tp, NTB, NPAD, NQ;
    size_t qiP, qs, kiP, ksb, vTi, vsc, kciP, kcs, vcT, thr, cen8, cens;
    size_t idx, cnt, oPart, mPart, lPart, statsV, qmean, vRow, tokIdx, tokCnt, tokRef, tokHist, tokBits;
    size_t tokTiles, tokTileCnt, gCen8, gCens, gRef, gBits;
    size_t tokPartO, tokPartM, tokPartL, scratch, total;

    Plan(int B, int T, int H, int n_tok) {
        NTB = (T + BLK - 1) / BLK;
        Tp = NTB * BLK;
        NPAD = ((NTB + 63) / 64) * 64;      // route reads pooled keys in 64-block groups
        NQ = NTB;
        const size_t bh = (size_t)B * H, tok = (size_t)B * T * H;
        size_t o = 0;
        auto take = [&](size_t bytes) { const size_t s = o; o = align16(o + bytes); return s; };
        qiP     = take(tok * HD);
        qs      = take(tok * sizeof(float));
        kiP     = take(bh * Tp * HD);
        ksb     = take(bh * Tp * 2 * sizeof(float));
        vTi     = take(bh * HD * Tp);
        vsc     = take(bh * HD * sizeof(float));
        kciP    = take(bh * NPAD * HD);
        kcs     = take(bh * NPAD * sizeof(float));
        vcT     = take(bh * HD * NPAD * sizeof(uint16_t));
        thr     = take(bh * NQ * sizeof(float));
        idx     = take(bh * NQ * NTB * sizeof(uint16_t));   // the only T^2 term
        cnt     = take(bh * NQ * sizeof(int32_t));
        cen8    = take(bh * NQ * HD);
        cens    = take(bh * NQ * sizeof(float));
        // route -> exact handover: one (o, m, l) per (b, h, query block)
        oPart   = take(bh * NQ * HD * sizeof(uint16_t));
        mPart   = take(bh * NQ * sizeof(float));
        lPart   = take(bh * NQ * sizeof(float));
        statsV  = take(bh * HD * sizeof(float));   // producer vamax accumulator
        qmean   = take(bh * NPAD * HD * sizeof(float));   // f32 query-block means (coarse branch)
        // token routing (empty at n_tok 0)
        const size_t on = n_tok > 0 ? 1 : 0;
        const size_t NG = (NQ + sol::TOK_GROUP - 1) / sol::TOK_GROUP, W = (NTB + 31) / 32;
        vRow    = take(on * bh * Tp * HD);
        tokRef  = take(on * bh * NQ * sizeof(float));             // per query block (route)
        tokBits = take(on * bh * NQ * W * sizeof(uint32_t));
        gCen8   = take(on * bh * NG * HD);                         // per group
        gCens   = take(on * bh * NG * sizeof(float));
        gRef    = take(on * bh * NG * sizeof(float));
        gBits   = take(on * bh * NG * W * sizeof(uint32_t));
        tokIdx  = take(on * bh * NG * n_tok * sizeof(uint32_t));
        tokCnt  = take(on * bh * NG * sizeof(int32_t));
        tokHist = take(on * bh * NG * sol::TOK_HIST_BINS * sizeof(uint32_t));
        const size_t groups = (size_t)((NG + BLK - 1) / BLK) * bh;   // CTAs of 64 rows
        tokTiles = take(on * groups * NTB * sizeof(uint16_t));
        tokTileCnt = take(on * groups * sizeof(int32_t));
        const size_t parts = (size_t)sol_token_splits(B, H, (int)NG) * bh * NG;
        tokPartO = take(on * parts * HD * sizeof(uint16_t));   // bf16 partial o
        tokPartM = take(on * parts * sizeof(float));
        tokPartL = take(on * parts * sizeof(float));
        scratch = take(sol_preprocess_scratch_bytes(B, H, NPAD));
        total = o;
    }
};

void validate_token_aug(int n_tok) {
    if (n_tok < 0 || n_tok > sol::NTOK_MAX || n_tok % BLK)
        throw std::runtime_error("sol_attn: token_aug must be a multiple of 64 up to "
                                 + std::to_string(sol::NTOK_MAX) + ", got " + std::to_string(n_tok));
}

void validate_shape(int batch, int seq_len, int num_heads) {
    if ((seq_len + BLK - 1) / BLK > 65535)
        throw std::runtime_error("sol_attn: seq_len too long for 16-bit block ids");
    // gridDim.y for every stage.
    if ((int64_t)batch * num_heads > 65535)
        throw std::runtime_error("sol_attn: batch * num_heads exceeds the 65535 grid limit");
}

// route + exact over a populated workspace. The cudaGetLastError covers every
// stage launched before it; without it a rejected launch leaves route's
// handover values in `out`.
void run_route_exact(const Plan& p, char* w, const void* ext_threshold, void* out,
                     const void* blen, int tail,
                     int batch, int seq_len, int num_heads,
                     int sink_start, int sink_end, int sink_q_start, int sink_q_end,
                     float scale_log2, int elem, int n_tok, cudaStream_t stream)
{
    // top-k mode: the caller supplies the per-query-block threshold
    const void* thr = ext_threshold ? ext_threshold : (const void*)(w + p.thr);
    // with token routing route also flags the unrouted blocks and leaves their tail to the token stage
    launch_sol_route(w + p.cen8, w + p.cens, w + p.kciP, w + p.kcs, w + p.vcT, w + p.vsc,
                     thr, w + p.idx, w + p.cnt, w + p.oPart, w + p.mPart, w + p.lPart,
                     n_tok ? w + p.tokRef : nullptr, n_tok ? w + p.tokBits : nullptr,
                     blen, tail, batch, seq_len, num_heads, p.NTB, p.NPAD, p.NQ,
                     sink_start, sink_end, sink_q_start, sink_q_end, scale_log2, stream);
    if (n_tok)
        launch_sol_token(w + p.qmean, w + p.tokRef, w + p.tokBits,
                         w + p.gCen8, w + p.gCens, w + p.gRef, w + p.gBits,
                         w + p.kiP, w + p.ksb, w + p.vTi, w + p.tokTiles, w + p.tokTileCnt,
                         w + p.tokHist, w + p.tokIdx, w + p.tokCnt,
                         w + p.tokPartO, w + p.tokPartM, w + p.tokPartL,
                         w + p.oPart, w + p.mPart, w + p.lPart,
                         batch, p.Tp, num_heads, p.NQ, p.NPAD, p.NTB, n_tok, tail,
                         sink_q_start, sink_q_end, scale_log2, stream);
    launch_sol_exact(w + p.qiP, w + p.qs, w + p.kiP, w + p.ksb, w + p.vTi, w + p.vsc,
                     w + p.idx, w + p.cnt, w + p.oPart, w + p.mPart, w + p.lPart,
                     w + p.vRow, w + p.tokIdx, w + p.tokCnt, n_tok, out,
                     batch, seq_len, p.Tp, num_heads, p.NQ, p.NTB,
                     scale_log2, elem, stream);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("sol_attn: kernel launch failed: ")
                                 + cudaGetErrorString(err));
}

}  // namespace

// Plan dims and slot offsets, in the order sol_attn_plan_names lists them.
extern "C" const char* const sol_attn_plan_names[] = {
    "Tp", "NTB", "NPAD", "NQ",
    "qiP", "qs", "kiP", "ksb", "vTi", "vsc", "kciP", "kcs", "vcT", "thr", "cen8", "cens",
    "idx", "cnt", "oPart", "mPart", "lPart", "statsV", "qmean",
    "vRow", "tokIdx", "tokCnt", "tokRef", "tokHist", "tokBits", "tokTiles", "tokTileCnt",
    "gCen8", "gCens", "gRef", "gBits", "tokPartO", "tokPartM", "tokPartL",
    "scratch", "total", nullptr,
};
// Writes the values into out[0..count) and returns count; writes nothing if
// cap < count, so a caller with a fixed buffer can detect a grown Plan.
extern "C" int sol_attn_plan(int batch, int seq_len, int num_heads, int n_tok, int64_t* out, int cap) {
    validate_token_aug(n_tok);
    const Plan p(batch, seq_len, num_heads, n_tok);
    const int64_t v[] = {
        p.Tp, p.NTB, p.NPAD, p.NQ,
        (int64_t)p.qiP, (int64_t)p.qs, (int64_t)p.kiP, (int64_t)p.ksb, (int64_t)p.vTi,
        (int64_t)p.vsc, (int64_t)p.kciP, (int64_t)p.kcs, (int64_t)p.vcT, (int64_t)p.thr,
        (int64_t)p.cen8, (int64_t)p.cens, (int64_t)p.idx, (int64_t)p.cnt, (int64_t)p.oPart,
        (int64_t)p.mPart, (int64_t)p.lPart, (int64_t)p.statsV, (int64_t)p.qmean,
        (int64_t)p.vRow, (int64_t)p.tokIdx, (int64_t)p.tokCnt, (int64_t)p.tokRef, (int64_t)p.tokHist,
        (int64_t)p.tokBits, (int64_t)p.tokTiles, (int64_t)p.tokTileCnt,
        (int64_t)p.gCen8, (int64_t)p.gCens, (int64_t)p.gRef, (int64_t)p.gBits,
        (int64_t)p.tokPartO, (int64_t)p.tokPartM, (int64_t)p.tokPartL,
        (int64_t)p.scratch, (int64_t)p.total,
    };
    const int count = (int)(sizeof(v) / sizeof(v[0]));
    if (cap >= count)
        for (int i = 0; i < count; ++i) out[i] = v[i];
    return count;
}

// ---- chunked producer path ----
extern "C" void sol_producer_begin(void* workspace, int batch, int seq_len,
                                   int num_heads, int n_tok, cudaStream_t stream) {
    validate_shape(batch, seq_len, num_heads);
    validate_token_aug(n_tok);
    const Plan p(batch, seq_len, num_heads, n_tok);
    char* w = reinterpret_cast<char*>(workspace);
    const size_t bh = (size_t)batch * num_heads;
    // route reads vcT's pad columns; statsV is an atomicMax accumulator
    cudaMemsetAsync(w + p.vcT, 0, bh * HD * p.NPAD * sizeof(uint16_t), stream);
    cudaMemsetAsync(w + p.statsV, 0, bh * HD * sizeof(float), stream);
}

extern "C" void sol_producer_chunk(
    void* workspace, const void* qkv, const void* fab,
    const void* qw, const void* kw, const void* kmean, const void* vscale,
    const void* blen, float rope_eps, int rot_dim, int t0, int M,
    int batch, int seq_len, int num_heads, int n_tok, cudaStream_t stream)
{
    validate_token_aug(n_tok);
    const Plan p(batch, seq_len, num_heads, n_tok);
    char* w = reinterpret_cast<char*>(workspace);
    launch_sol_producer(qkv, fab, qw, kw, kmean, vscale,
                        w + p.qiP, w + p.qs, w + p.kiP, w + p.ksb,
                        w + p.vTi, n_tok ? w + p.vRow : nullptr, w + p.vcT, w + p.scratch,
                        w + p.cen8, w + p.cens, w + p.qmean, w + p.statsV,
                        blen, rope_eps, rot_dim, t0, M, seq_len, p.Tp, num_heads,
                        p.NPAD, p.NQ, stream);
}

// vscale: the [B*H, HD] f32 scale the producer quantized V with.
// kmean_next / vamax_out ([B*H, HD] f32) receive this step's statistics.
extern "C" void launch_sol_attn_core(
    void* workspace, void* out, const void* vscale, void* kmean_next, void* vamax_out,
    const void* blen, int tail,
    int batch, int seq_len, int num_heads,
    float tau, float scale, const void* ext_threshold,
    int sink_start, int sink_end, int sink_q_start, int sink_q_end,
    int n_tok, cudaStream_t stream)
{
    validate_shape(batch, seq_len, num_heads);
    validate_token_aug(n_tok);
    const Plan p(batch, seq_len, num_heads, n_tok);
    char* w = reinterpret_cast<char*>(workspace);
    const float scale_log2 = scale * 1.4426950408889634f;
    const size_t stats_bytes = (size_t)batch * num_heads * HD * sizeof(float);
    cudaMemcpyAsync(w + p.vsc, vscale, stats_bytes, cudaMemcpyDeviceToDevice, stream);
    launch_sol_finish(w + p.scratch, w + p.kciP, w + p.kcs, w + p.thr,
                      w + p.cen8, w + p.cens, kmean_next, blen,
                      batch, seq_len, num_heads, p.NTB, p.NPAD, p.NQ,
                      tau, scale_log2, stream);
    run_route_exact(p, w, ext_threshold, out, blen, tail, batch, seq_len, num_heads,
                    sink_start, sink_end, sink_q_start, sink_q_end, scale_log2,
                    sol::SOL_BF16, n_tok, stream);
    cudaMemcpyAsync(vamax_out, w + p.statsV, stats_bytes, cudaMemcpyDeviceToDevice, stream);
}

extern "C" void launch_sol_attn(
    const void* q, const void* k, const void* v, void* out, void* workspace,
    int batch, int seq_len, int num_heads, int head_dim, int elem,
    float tau, float scale, const void* key_bias,
    const void* ext_threshold, const void* blen, int tail,
    int sink_start, int sink_end, int sink_q_start, int sink_q_end,
    int64_t qs_b, int64_t qs_t, int64_t qs_h,
    int64_t ks_b, int64_t ks_t, int64_t ks_h,
    int64_t vs_b, int64_t vs_t, int64_t vs_h,
    int n_tok, cudaStream_t stream)
{
    if (head_dim != HD)
        throw std::runtime_error("sol_attn supports head_dim 128, got " + std::to_string(head_dim));
    validate_shape(batch, seq_len, num_heads);
    validate_token_aug(n_tok);

    const Plan p(batch, seq_len, num_heads, n_tok);
    char* w = reinterpret_cast<char*>(workspace);
    const float scale_log2 = scale * 1.4426950408889634f;
    launch_sol_preprocess(q, k, v, w + p.qiP, w + p.qs, w + p.kiP, w + p.ksb,
                          w + p.kciP, w + p.kcs, w + p.vcT, w + p.thr,
                          w + p.cen8, w + p.cens, w + p.vsc, w + p.qmean, w + p.scratch,
                          key_bias, blen,
                          batch, seq_len, p.Tp, num_heads, p.NTB, p.NPAD, p.NQ,
                          qs_b, qs_t, qs_h, ks_b, ks_t, ks_h, vs_b, vs_t, vs_h,
                          tau, scale_log2, elem, stream);
    launch_sol_vtranspose(v, w + p.vsc, w + p.vTi, n_tok ? w + p.vRow : nullptr, batch, seq_len, p.Tp, num_heads,
                          vs_b, vs_t, vs_h, elem, stream);
    run_route_exact(p, w, ext_threshold, out, blen, tail, batch, seq_len, num_heads,
                    sink_start, sink_end, sink_q_start, sink_q_end, scale_log2, elem, n_tok, stream);
}
