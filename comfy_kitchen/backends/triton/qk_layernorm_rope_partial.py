# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton joint text + video attention prep: per-head QK LayerNorm + RoPE on the video tokens.

One launch handles q and k (grid axis 1 picks the tensor). Each program owns ``block_r``
consecutive ``[head_dim]`` rows of the contiguous ``[B, S, heads * head_dim]`` projection
(row r = token (r // heads) % S, head r % heads) and, in fp32 registers:

    LayerNorm(head_dim, affine, eps): mean, biased variance, rsqrt, weight and bias
    rows of video tokens (token >= text_len), with t = token - text_len:
        out[2j]     = y[2j]     * cos[t, 2j]     - y[2j + 1] * sin[t, 2j]
        out[2j + 1] = y[2j + 1] * cos[t, 2j + 1] + y[2j]     * sin[t, 2j + 1]
    rows of text tokens: out = y

rounded once to the output dtype and written in the input layout, so the normalized tensors,
the video-token copies and the concatenation are never written out.

Launch configuration (same kernel body), picked per GPU from the measured shapes:
  CUDA: block_r 32 rows, 4 warps (RTX 5090)
  HIP:  block_r 64 rows, 8 warps (Radeon 8060S)
"""

import torch

import triton
import triton.language as tl

_HIP = bool(torch.version.hip)  # picks the launch configuration


@triton.jit
def _qk_layernorm_rope_partial_kernel(
    q,
    k,
    out_q,
    out_k,
    wq,
    bq,
    wk,
    bk,
    cos,
    sin,
    n_rows,
    seq,
    text_len,
    eps,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_r: tl.constexpr,
):
    pid = tl.program_id(0)
    which = tl.program_id(1)
    rows = pid.to(tl.int64) * block_r + tl.arange(0, block_r)
    rmask = rows < n_rows
    offs_d = tl.arange(0, head_dim)
    if which == 0:
        src = q
        dst = out_q
        w_ptr = wq
        b_ptr = bq
    else:
        src = k
        dst = out_k
        w_ptr = wk
        b_ptr = bk
    ptrs = rows[:, None] * head_dim + offs_d[None, :]
    x = tl.load(src + ptrs, mask=rmask[:, None], other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) / head_dim
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, axis=1) / head_dim
    rstd = tl.math.rsqrt(var + eps)
    w = tl.load(w_ptr + offs_d).to(tl.float32)
    b = tl.load(b_ptr + offs_d).to(tl.float32)
    y = xc * rstd[:, None] * w[None, :] + b[None, :]

    tok = (rows // heads) % seq
    is_vid = tok >= text_len
    vrow = tl.where(is_vid, tok - text_len, 0)
    tmask = rmask & is_vid
    tptr = vrow[:, None] * head_dim + offs_d[None, :]
    c = tl.load(cos + tptr, mask=tmask[:, None], other=1.0)
    s = tl.load(sin + tptr, mask=tmask[:, None], other=0.0)
    ye, yo = tl.split(tl.reshape(y, (block_r, head_dim // 2, 2)))
    ce, co = tl.split(tl.reshape(c, (block_r, head_dim // 2, 2)))
    se, so = tl.split(tl.reshape(s, (block_r, head_dim // 2, 2)))
    oe = ye * ce - yo * se
    oo = yo * co + ye * so
    r = tl.interleave(oe, oo)
    out = tl.where(is_vid[:, None], r, y)
    tl.store(dst + ptrs, out.to(dst.dtype.element_ty), mask=rmask[:, None])


def _launch_config() -> tuple[int, int]:
    # (block_r, num_warps)
    if _HIP:
        return 64, 8
    return 32, 4


def qk_layernorm_rope_partial(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    q_bias: torch.Tensor,
    k_weight: torch.Tensor,
    k_bias: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    text_len: int,
    heads: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq, channels = q.shape
    head_dim = channels // heads
    out_q = torch.empty_like(q, memory_format=torch.contiguous_format)
    out_k = torch.empty_like(k, memory_format=torch.contiguous_format)
    n_rows = batch_size * seq * heads
    block_r, num_warps = _launch_config()
    grid = (triton.cdiv(n_rows, block_r), 2)
    _qk_layernorm_rope_partial_kernel[grid](
        q,
        k,
        out_q,
        out_k,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        cos,
        sin,
        n_rows,
        seq,
        text_len,
        float(eps),
        heads=heads,
        head_dim=head_dim,
        block_r=block_r,
        num_warps=num_warps,
    )
    return out_q, out_k
