# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton flash-attention forward on ``[B, S, H, head_dim]`` tensors.

Non-causal, no mask. One program owns ``block_m`` query rows of one (batch, head)
and streams the keys in ``block_n`` tiles with an online base-2 softmax. The
softmax scale (times log2(e)) is applied to the fp32 ``q @ k^T`` tile inside the
kernel, so q is read as is and never rescaled in its own dtype.
"""

import math

import torch

import triton
import triton.language as tl

_BLOCK_M, _BLOCK_N, _NUM_WARPS, _NUM_STAGES = 128, 64, 8, 2
_LOG2E = 1.4426950408889634


@triton.jit
def _flash_attention_fwd(
    q_ptr, k_ptr, v_ptr, o_ptr, qk_scale, seq_q, seq_k,
    stride_qb, stride_qs, stride_qh,
    stride_kb, stride_ks, stride_kh,
    stride_vb, stride_vs, stride_vh,
    stride_ob, stride_os, stride_oh,
    block_m: tl.constexpr, block_n: tl.constexpr, head_dim: tl.constexpr,
):
    out_dtype = o_ptr.dtype.element_ty
    start_m = tl.program_id(0)
    h = tl.program_id(1).to(tl.int64)
    b = tl.program_id(2).to(tl.int64)
    offs_m = start_m * block_m + tl.arange(0, block_m)
    offs_n = tl.arange(0, block_n)
    offs_d = tl.arange(0, head_dim)
    q = tl.load(q_ptr + b * stride_qb + h * stride_qh + offs_m[:, None].to(tl.int64) * stride_qs + offs_d[None, :],
                mask=offs_m[:, None] < seq_q, other=0.0)
    m_i = tl.full([block_m], float("-inf"), tl.float32)
    l_i = tl.zeros([block_m], tl.float32)
    acc = tl.zeros([block_m, head_dim], tl.float32)
    kb = k_ptr + b * stride_kb + h * stride_kh
    vb = v_ptr + b * stride_vb + h * stride_vh
    for start_n in range(0, seq_k, block_n):
        rn = (start_n + offs_n).to(tl.int64)
        nm = rn < seq_k
        k = tl.load(kb + rn[None, :] * stride_ks + offs_d[:, None], mask=nm[None, :], other=0.0)
        # softmax scale on the fp32 scores: q is not prescaled
        qk = tl.where(nm[None, :], tl.dot(q, k) * qk_scale, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(vb + rn[:, None] * stride_vs + offs_d[None, :], mask=nm[:, None], other=0.0)
        acc += tl.dot(p.to(out_dtype), v)
        m_i = m_ij
    acc = acc / l_i[:, None]
    tl.store(o_ptr + b * stride_ob + h * stride_oh + offs_m[:, None].to(tl.int64) * stride_os + offs_d[None, :],
             acc.to(out_dtype), mask=offs_m[:, None] < seq_q)


def flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    # q [B, S_q, H, D], k / v [B, S_k, H, D]; each with a unit stride on D (call rule)
    batch, seq_q, heads, head_dim = q.shape
    seq_k = k.shape[1]
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    o = torch.empty((batch, seq_q, heads, head_dim), device=q.device, dtype=q.dtype)
    grid = (triton.cdiv(seq_q, _BLOCK_M), heads, batch)
    _flash_attention_fwd[grid](
        q, k, v, o, scale * _LOG2E, seq_q, seq_k,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        block_m=_BLOCK_M, block_n=_BLOCK_N, head_dim=head_dim,
        num_warps=_NUM_WARPS, num_stages=_NUM_STAGES,
    )
    return o
