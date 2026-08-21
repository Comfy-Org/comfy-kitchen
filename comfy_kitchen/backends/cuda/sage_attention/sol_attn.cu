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

// Sol-Attn: training-free sparse attention for video diffusion (arXiv 2607.24027).
// Four stages orchestrated over one caller-provided workspace:
//
//   preprocess  quantize Q/K/V, pool K/V per block, derive the routing threshold
//   vtranspose  INT8 V^T for the exact stage's PV operand
//   route       decide the routed blocks; carry the approximate correction
//   exact       walk each routed list, resuming route's online softmax
//
// Layout contract (permutations, swizzles, MMA choice): sol_layout.cuh.
// Requires sm_80+ (tuned for sm_120); q/k/v/out are (B, T, H, 128) bf16.

#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "sol_layout.cuh"

extern "C" {
void launch_sol_preprocess(const void*, const void*, const void*, void*, void*, void*,
                           void*, void*, void*, void*, void*, void*, void*, void*, void*,
                           const void*,
                           int, int, int, int, int, int, int,
                           int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                           int64_t, int64_t, int64_t, float, float, cudaStream_t);
size_t sol_preprocess_scratch_bytes(int, int, int);
void launch_sol_vtranspose(const void*, const void*, void*, int, int, int, int,
                           int64_t, int64_t, int64_t, cudaStream_t);
void launch_sol_route(const void*, const void*, const void*, const void*, const void*,
                      const void*, const void*, void*, void*, void*, void*, void*,
                      int, int, int, int, int, int, int, int, int, int, int,
                      float, cudaStream_t);
void launch_sol_route_perrow(const void*, const void*, const void*, const void*,
                      const void*, const void*, const void*, void*, void*, void*,
                      void*, void*, int, int, int, int, int, int, int, int, int,
                      int, int, float, cudaStream_t);
void launch_sol_exact(const void*, const void*, const void*, const void*, const void*,
                      const void*, const void*, const void*, const void*, const void*,
                      const void*, void*, int, int, int, int, int, int, float, int,
                      cudaStream_t);
}

namespace {

constexpr int HD = sol::HEAD_DIM;
constexpr int BLK = sol::BLOCK;

inline size_t align16(size_t n) { return (n + 15u) & ~(size_t)15u; }

// Workspace carve-up; a pure function of the shape (see sol_attn_workspace_bytes).
struct Plan {
    int Tp, NTB, NPAD, NQ, MAXB;
    size_t qiP, qs, kiP, ksb, vTi, vsc, kciP, kcs, vcT, thr, cen8, cens;
    size_t idx, cnt, oPart, mPart, lPart, mPartRow, lPartRow, scratch, total;

    Plan(int B, int T, int H, int max_blk) {
        NTB = (T + BLK - 1) / BLK;
        Tp = NTB * BLK;
        NPAD = ((NTB + 63) / 64) * 64;      // route reads pooled keys in 64-block groups
        NQ = NTB;
        // The cap is a QUALITY trade: blocks past it are dropped, and a sink_q
        // query block routes all NTB blocks, so any cap truncates those. 0 = off.
        MAXB = (max_blk > 0 && max_blk < NTB) ? max_blk : NTB;
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
        // uint16 block ids (< 65536 while T < ~4.2M); the only T^2 term, so
        // halving it matters at video lengths.
        idx     = take(bh * NQ * MAXB * sizeof(uint16_t));
        cnt     = take(bh * NQ * sizeof(int32_t));
        cen8    = take(bh * NQ * HD);
        cens    = take(bh * NQ * sizeof(float));
        // Handover state: one (o, m, l) per (b, h, query block) -- the
        // centroid tail is shared by all rows of a block.
        oPart   = take(bh * NQ * HD * sizeof(uint16_t));
        mPart   = take(bh * NQ * sizeof(float));
        lPart   = take(bh * NQ * sizeof(float));
        // Per-row m/l for the centroid_tail=false fallback (its o_part
        // aliases `out`). Always reserved so the size ignores the flag.
        mPartRow = take(tok * sizeof(float));
        lPartRow = take(tok * sizeof(float));
        scratch = take(sol_preprocess_scratch_bytes(B, H, NPAD));
        total = o;
    }
};

}  // namespace

extern "C" size_t sol_attn_workspace_bytes(int batch, int seq_len, int num_heads, int max_blocks) {
    return Plan(batch, seq_len, num_heads, max_blocks).total;
}

extern "C" void launch_sol_attn(
    const void* q, const void* k, const void* v, void* out, void* workspace,
    int batch, int seq_len, int num_heads, int head_dim, int max_blocks,
    float tau, float scale, int centroid_tail, const void* key_bias,
    int sink_start, int sink_end, int sink_q_start, int sink_q_end,
    int64_t qs_b, int64_t qs_t, int64_t qs_h,
    int64_t ks_b, int64_t ks_t, int64_t ks_h,
    int64_t vs_b, int64_t vs_t, int64_t vs_h,
    cudaStream_t stream)
{
    if (head_dim != HD)
        throw std::runtime_error("sol_attn supports head_dim 128, got " + std::to_string(head_dim));
    if ((seq_len + BLK - 1) / BLK > 65535)
        throw std::runtime_error("sol_attn: seq_len too long for 16-bit block ids");
    // gridDim.y for every stage.
    if ((int64_t)batch * num_heads > 65535)
        throw std::runtime_error("sol_attn: batch * num_heads exceeds the 65535 grid limit");

    const Plan p(batch, seq_len, num_heads, max_blocks);
    char* w = reinterpret_cast<char*>(workspace);
    const float scale_log2 = scale * 1.4426950408889634f;

    launch_sol_preprocess(q, k, v, w + p.qiP, w + p.qs, w + p.kiP, w + p.ksb,
                          w + p.kciP, w + p.kcs, w + p.vcT, w + p.thr,
                          w + p.cen8, w + p.cens, w + p.vsc, w + p.scratch,
                          key_bias,
                          batch, seq_len, p.Tp, num_heads, p.NTB, p.NPAD, p.NQ,
                          qs_b, qs_t, qs_h, ks_b, ks_t, ks_h, vs_b, vs_t, vs_h,
                          tau, scale_log2, stream);
    launch_sol_vtranspose(v, w + p.vsc, w + p.vTi, batch, seq_len, p.Tp, num_heads,
                          vs_b, vs_t, vs_h, stream);
    if (centroid_tail) {
        launch_sol_route(w + p.cen8, w + p.cens, w + p.kciP, w + p.kcs, w + p.vcT, w + p.vsc,
                         w + p.thr, w + p.idx, w + p.cnt, w + p.oPart, w + p.mPart, w + p.lPart,
                         batch, seq_len, num_heads, p.NTB, p.NPAD, p.NQ, p.MAXB,
                         sink_start, sink_end, sink_q_start, sink_q_end, scale_log2, stream);
        launch_sol_exact(w + p.qiP, w + p.qs, w + p.kiP, w + p.ksb, w + p.vTi, w + p.vsc,
                         w + p.idx, w + p.cnt, w + p.oPart, w + p.mPart, w + p.lPart, out,
                         batch, seq_len, p.Tp, num_heads, p.NQ, p.MAXB,
                         scale_log2, 1, stream);
    } else {
        // Per-row tail: the pre-centroid behaviour, kept selectable for
        // quality A/B. Its handover writes straight into `out`, which the
        // exact stage reads back in its prologue and overwrites.
        launch_sol_route_perrow(w + p.qiP, w + p.qs, w + p.kciP, w + p.kcs, w + p.vcT,
                         w + p.vsc, w + p.thr, w + p.idx, w + p.cnt, out,
                         w + p.mPartRow, w + p.lPartRow,
                         batch, seq_len, num_heads, p.NTB, p.NPAD, p.NQ, p.MAXB,
                         sink_start, sink_end, sink_q_start, sink_q_end, scale_log2, stream);
        launch_sol_exact(w + p.qiP, w + p.qs, w + p.kiP, w + p.ksb, w + p.vTi, w + p.vsc,
                         w + p.idx, w + p.cnt, out, w + p.mPartRow, w + p.lPartRow, out,
                         batch, seq_len, p.Tp, num_heads, p.NQ, p.MAXB,
                         scale_log2, 0, stream);
    }

    // A rejected launch would silently leave route's handover values in `out`.
    // One check covers all four stages: the first failure latches until read.
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("sol_attn: kernel launch failed: ")
                                 + cudaGetErrorString(err));
}
