// SPDX-FileCopyrightText: Copyright (c) 2024 SageAttention team.
// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

// The host-side entry points of the RDNA2 SageAttention port, expressed as plain
// pointers and extents.
//
// These used to take torch::stable::Tensor and allocate their own outputs, which
// is why the port needed a CMake project and a second .pyd of its own: the main
// HIP extension has no torch at all, by design -- it binds with nanobind's
// nb::ndarray<> and every output is allocated on the Python side. Declaring the
// entry points over plain data is what lets this kernel join HIP_SOURCES, and
// ../dlpack_bindings_gfx103x.cpp does the marshalling instead.
//
// Nothing here validates. The shapes, strides and mask geometry are the caller's
// to get right, exactly as for every other kernel in this backend; the checks that
// do exist (the mask's rank, its contiguity where the kernel assumes it, and its
// tile coverage) live in sage_attn_gfx103x.hip next to the launcher that depends
// on them.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace sageattn_gfx103x {

// The granularity the Q/K quantizer writes its per-block scales at: one per this
// many keys. These fix the shapes of the scale buffers, so both the launcher and
// whoever allocates those buffers need them, and they are the MIN_BLK_Q /
// MIN_BLK_K in sage_attn_gfx103x.hip under different names.
constexpr int64_t kQuantGroupQ = 32;
constexpr int64_t kQuantGroupK = 16;

// Quantize Q and K to int8 with a per-row scale and a per-block scale.
//
// The two outputs' shapes are not derivable here, which is why the caller
// allocates them: q_int8/k_int8 are [B, heads, len, head_dim] int8, and q_scale/
// k_scale are [B, heads, groups] float32 with groups = ceil(len / block). The
// block size is a launch decision below, and on the skip_q path q_int8 and q_scale
// are written not at all.
struct QuantArgs {
    const void* q;
    const void* k;
    // The per-(b,h,d) K mean the quantizer subtracts before narrowing, or null.
    // Unlike the WMMA quantizer in quant_qk_int8.hip this is optional and this is
    // the only caller of it, so there is no case for folding the two together.
    const void* key_mean;
    int8_t* q_int8;
    int8_t* k_int8;
    float* q_scale;
    float* k_scale;
    int64_t batch, q_heads, kv_heads, q_len, kv_len, head_dim;
    int64_t q_stride_b, q_stride_n, q_stride_h;
    int64_t k_stride_b, k_stride_n, k_stride_h;
    int64_t q_groups, k_groups;
    double sm_scale;
    // 1 when q/k are bfloat16, 0 for float16.
    int src_bf16;
    // Nonzero leaves Q alone, for the caller that will quantize it inside the
    // attention kernel instead. q_int8 and q_scale are then unused.
    int skip_q;
};

// The attention itself.
//
// Operands keep the caller's own layouts, read in place: Q and K int8, V fp16 in
// [B, heads, seq, head_dim], output fp16 or bf16. Strides are in elements, as
// the kernels index them.
//
// Q arrives either already packed (q_is_packed, q_fp unused) or as an fp16/bf16
// source the kernel quantizes itself (has_q_fp). Which of the two runs is decided
// below rather than here, because it also depends on the KV length.
struct AttnArgs {
    const int8_t* q;
    const int8_t* k;
    const __half* v;
    void* out;
    const float* q_scale;
    const float* k_scale;
    // The fp16/bf16 Q source when has_q_fp, else unused.
    const void* q_fp;
    int64_t batch, q_heads, kv_heads, qo_len, kv_len, head_dim;
    int64_t q_stride_b, q_stride_n, q_stride_h;
    int64_t q_fp_stride_b, q_fp_stride_n, q_fp_stride_h;
    int64_t k_stride_b, k_stride_n, k_stride_h;
    int64_t v_stride_b, v_stride_n, v_stride_h;
    // The output's own batch / sequence / head strides, unresolved against the
    // layout: layout_is_hnd picks which of the last two is which.
    int64_t o_stride_b, o_stride_seq, o_stride_head;
    int layout_is_hnd;
    int64_t qs_stride_b, qs_stride_h, ks_stride_b, ks_stride_h;
    double sm_scale;
    int causal;
    int q_is_packed;
    int has_q_fp;
    // 1 when q_fp is bfloat16, 0 for float16.
    int q_fp_bf16;
    int out_bf16;

    // Mask. mask is unused when mode is kNone. Raw masks are indexed by the
    // caller's strides; every prepared form is contiguous, which is what
    // mask_contiguous asserts before the kernel is allowed to compute a row
    // address arithmetically. shape/stride are the mask's own, up to rank 5.
    const void* mask;
    int mask_mode, mask_dtype, mask_ndim, mask_contiguous;
    int64_t mask_shape[4];
    int64_t mask_stride[4];
};

void gfx103x_launch_quant_qk(const QuantArgs& a, hipStream_t stream);
void gfx103x_launch_attn(const AttnArgs& a, hipStream_t stream);

}  // namespace sageattn_gfx103x
