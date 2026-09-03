// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Launchers called from more than one translation unit. They are extern "C", so a
// declaration that drifts from its definition still links and then corrupts the
// call frame; declaring them once where the definition can see it makes the
// compiler check instead.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

extern "C" {

// Fused 3D neighborhood attention over contiguous (B, T, H, W, NH, HD) tensors.
// dtype_code is a DTYPE_TO_CODE value: 1 float16, 2 bfloat16. See ops/na3d.hip.
void launch_na3d_kernel(const void* q, const void* k, const void* v, void* out, int batch,
                        int t_size, int h_size, int w_size, int num_heads, int head_dim, int kt,
                        int kh, int kw, int causal_t, int causal_h, int causal_w, float scale,
                        int dtype_code, hipStream_t stream);

// BF16 decode attention over a fixed-capacity KV cache. query_length is the GQA
// group count folded into the query sequence dimension by the Python layer, and
// head_dim is fixed at 128. out_accum and lse_accum are read only when
// num_splits > 1. See ops/flash_decode.hip.
void launch_flash_decode(const void* q, const void* k, const void* v, const int* kv_lengths,
                         void* out, float* softmax_lse, float* out_accum, float* lse_accum,
                         int batch, int query_length, int heads, int kv_capacity, int num_splits,
                         int64_t q_batch_stride, int64_t q_row_stride, int64_t q_head_stride,
                         int64_t k_batch_stride, int64_t k_row_stride, int64_t k_head_stride,
                         hipStream_t stream);

// ldc is c's row stride, so a caller writing an N-column slice of a wider output
// passes that output's width; a whole GEMM passes N.
void launch_int8_gemm_kernel(const void* a, const void* b, void* c, const void* scale_a,
                             const void* scale_b, int scale_b_stride, const void* bias,
                             int bias_code, int M, int N, int K, int ldc, int out_code,
                             hipStream_t stream);

// scale_code is a DTYPE_TO_CODE value: 0 float32, 5 e4m3 (passed as raw bytes).
// codebook is 16 floats, or null for the uniform levels.
void launch_dequant_int4_grouped_to_int8_kernel(const void* qw, const void* s_rel, int scale_code,
                                                const void* codebook, void* out, int64_t n,
                                                int64_t k, int group_size, hipStream_t stream);

// in_dtype_code is a DTYPE_TO_CODE value: 0 float32, 1 float16, 2 bfloat16.
// s_rel is written as raw e4m3 bytes; seed is ignored unless stochastic is set.
void launch_quantize_w4a8_convrot_kernel(const void* rotated, const void* codebook, void* packed,
                                         void* s_rel, void* s_channel, int64_t n, int64_t k,
                                         int in_dtype_code, bool stochastic, uint64_t seed,
                                         hipStream_t stream);

// Widest K the fused requantize can take on the current device, 0 if unknown.
int w4a8_requant_max_k_kernel();

void launch_w4a8_int8_gemm_chunked_kernel(const void* xq, const void* qw, const void* s_rel,
                                          int scale_code, const void* codebook,
                                          const void* s_channel, const void* xs, const void* bias,
                                          int bias_code, void* workspace, void* out, int M, int N,
                                          int K, int group_size, int chunk_cols, int out_code,
                                          hipStream_t stream);

// Sol-Attn sparse attention -- see sage_attention/sol_attn.hip. The whole pipeline
// runs over one caller-allocated workspace whose carve-up sol_attn_plan reports.
extern const char* const sol_attn_plan_names[];  // null-terminated
int sol_attn_plan(int batch, int seq_len, int num_heads, int64_t* out, int cap);
void sol_producer_begin(void* workspace, int batch, int seq_len, int num_heads,
                        hipStream_t stream);
void sol_producer_chunk(void* workspace, const void* qkv, const void* fab, const void* qw,
                        const void* kw, const void* kmean, const void* vscale, const void* blen,
                        float rope_eps, int rot_dim, int t0, int M, int batch, int seq_len,
                        int num_heads, hipStream_t stream);
void launch_sol_attn_core(void* workspace, void* out, const void* vscale, void* kmean_next,
                          void* vamax_out, const void* blen, int tail, int batch, int seq_len,
                          int num_heads, float tau, float scale, const void* ext_threshold,
                          int sink_start, int sink_end, int sink_q_start, int sink_q_end,
                          hipStream_t stream);
void launch_sol_attn(const void* q, const void* k, const void* v, void* out, void* workspace,
                     int batch, int seq_len, int num_heads, int head_dim, int elem, float tau, float scale,
                     const void* key_bias, const void* ext_threshold, const void* blen, int tail,
                     int sink_start, int sink_end, int sink_q_start, int sink_q_end, int64_t qs_b,
                     int64_t qs_t, int64_t qs_h, int64_t ks_b, int64_t ks_t, int64_t ks_h,
                     int64_t vs_b, int64_t vs_t, int64_t vs_h, hipStream_t stream);

}  // extern "C"
