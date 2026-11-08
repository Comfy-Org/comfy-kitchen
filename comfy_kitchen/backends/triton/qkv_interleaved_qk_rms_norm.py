# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton Hunyuan3D-2.1 DiT attention prep: per-head q / k / v relayout + QK RMSNorm.

The concatenation ``cat((q, k, v), -1)`` (self-attention) or ``cat((k, v), -1)``
(cross-attention) viewed as ``[B, S, heads, n * head_dim]`` and split into ``head_dim``
blocks puts block j of a concatenated row in source ``j // heads`` at column
``(j % heads) * head_dim``; head h takes blocks ``n * h + i`` (i = 0 q, 1 k, 2 v for
self-attention; i = 0 k, 1 v for cross-attention, where q is the plain per-head view).

One program owns ``rows`` consecutive tokens with all heads (an ``[heads, head_dim]`` tile
per output) and reads each block straight from the projection it lives in, so the
concatenation and the per-head copies are never written out. q and k rows are
RMS-normalized in fp32 (``w * (rsqrt(mean(x * x) + eps) * x)``, the sum of squares in the
order of torch's vectorized rms kernel for ``head_dim`` 128: four values per lane with
fma, then a pairwise tree over the lanes) and rounded once to the output dtype; v is
copied. Cross-attention launches the q programs and the k / v programs in one grid.

Launch configuration (same kernel bodies), picked per GPU from the measured shapes:
  CUDA: self-attention 1 token per program, 8 warps; cross-attention 4 tokens, 8 warps
  HIP:  1 token per program, 8 warps for both
"""

import torch

import triton
import triton.language as tl
from triton.language.extra import libdevice

_HIP = bool(torch.version.hip)  # picks the launch configuration


@triton.jit
def _rms(x, w, eps, heads: tl.constexpr, head_dim: tl.constexpr):
    # lane l of the torch kernel holds elements 4l .. 4l + 3 and accumulates them with fma
    # in order; the lanes are then combined pairwise (shuffle-down tree), / head_dim,
    # rsqrt(. + eps), w * (rstd * x)
    xf = x.to(tl.float32)
    lanes: tl.constexpr = head_dim // 4
    a, b = tl.split(tl.reshape(xf, (heads, lanes, 2, 2)))
    x0, x2 = tl.split(a)
    x1, x3 = tl.split(b)
    s = x0 * x0
    s = tl.fma(x1, x1, s)
    s = tl.fma(x2, x2, s)
    s = tl.fma(x3, x3, s)
    half: tl.constexpr = lanes // 2
    s = tl.sum(tl.reshape(s, (heads, 2, half)), axis=1)
    if half >= 2:
        s = tl.sum(tl.reshape(s, (heads, 2, half // 2)), axis=1)
    if half >= 4:
        s = tl.sum(tl.reshape(s, (heads, 2, half // 4)), axis=1)
    if half >= 8:
        s = tl.sum(tl.reshape(s, (heads, 2, half // 8)), axis=1)
    if half >= 16:
        s = tl.sum(tl.reshape(s, (heads, 2, half // 16)), axis=1)
    mean_sq = tl.reshape(s, (heads,)) / head_dim
    rstd = libdevice.rsqrt(mean_sq + eps)
    return w * (rstd[:, None] * xf)


@triton.jit
def _load_block(
    a, b, c, base, j, n_src: tl.constexpr, heads: tl.constexpr, head_dim: tl.constexpr, d, mask
):
    # block j ([heads, 1] per head) of the concatenated row: source j // heads, column
    # (j % heads) * head_dim
    src = j // heads
    off = base + (j % heads) * head_dim + d
    x = tl.load(a + off, mask=mask & (src == 0), other=0.0)
    x += tl.load(b + off, mask=mask & (src == 1), other=0.0)
    if n_src == 3:
        x += tl.load(c + off, mask=mask & (src == 2), other=0.0)
    return x


@triton.jit
def _qkv_self_kernel(
    q,
    k,
    v,
    q_weight,
    k_weight,
    out_q,
    out_k,
    out_v,
    n_tokens,
    eps,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    rows: tl.constexpr,
):
    pid = tl.program_id(0)
    h = tl.arange(0, heads)[:, None]
    d = tl.arange(0, head_dim)[None, :]
    wq = tl.load(q_weight + d).to(tl.float32)
    wk = tl.load(k_weight + d).to(tl.float32)
    for i in tl.static_range(rows):
        row = pid * rows + i
        mask = row < n_tokens
        base = row.to(tl.int64) * (heads * head_dim)
        o = base + h * head_dim + d
        xq = _load_block(q, k, v, base, h * 3 + 0, 3, heads, head_dim, d, mask)
        tl.store(
            out_q + o, _rms(xq, wq, eps, heads, head_dim).to(out_q.dtype.element_ty), mask=mask
        )
        xk = _load_block(q, k, v, base, h * 3 + 1, 3, heads, head_dim, d, mask)
        tl.store(
            out_k + o, _rms(xk, wk, eps, heads, head_dim).to(out_k.dtype.element_ty), mask=mask
        )
        xv = _load_block(q, k, v, base, h * 3 + 2, 3, heads, head_dim, d, mask)
        tl.store(out_v + o, xv, mask=mask)


@triton.jit
def _qkv_cross_kernel(
    q,
    k,
    v,
    q_weight,
    k_weight,
    out_q,
    out_k,
    out_v,
    n_q_tokens,
    n_kv_tokens,
    n_q_programs,
    eps,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    rows: tl.constexpr,
):
    pid = tl.program_id(0)
    h = tl.arange(0, heads)[:, None]
    d = tl.arange(0, head_dim)[None, :]
    if pid < n_q_programs:
        wq = tl.load(q_weight + d).to(tl.float32)
        for i in tl.static_range(rows):
            row = pid * rows + i
            mask = row < n_q_tokens
            o = row.to(tl.int64) * (heads * head_dim) + h * head_dim + d
            xq = tl.load(q + o, mask=mask, other=0.0)
            tl.store(
                out_q + o, _rms(xq, wq, eps, heads, head_dim).to(out_q.dtype.element_ty), mask=mask
            )
    else:
        wk = tl.load(k_weight + d).to(tl.float32)
        for i in tl.static_range(rows):
            row = (pid - n_q_programs) * rows + i
            mask = row < n_kv_tokens
            base = row.to(tl.int64) * (heads * head_dim)
            o = base + h * head_dim + d
            xk = _load_block(k, v, k, base, h * 2 + 0, 2, heads, head_dim, d, mask)
            tl.store(
                out_k + o, _rms(xk, wk, eps, heads, head_dim).to(out_k.dtype.element_ty), mask=mask
            )
            xv = _load_block(k, v, k, base, h * 2 + 1, 2, heads, head_dim, d, mask)
            tl.store(out_v + o, xv, mask=mask)


def _launch_config(q_interleaved: bool) -> tuple[int, int]:
    # (tokens per program, num_warps)
    if _HIP:
        return 1, 8
    return (1, 8) if q_interleaved else (4, 8)


def qkv_interleaved_qk_rms_norm(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    heads: int,
    eps: float = 1e-6,
    q_interleaved: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, seq, channels = q.shape
    seq_kv = k.shape[1]
    head_dim = channels // heads
    out_q = torch.empty_like(q, memory_format=torch.contiguous_format)
    out_k = torch.empty_like(k, memory_format=torch.contiguous_format)
    out_v = torch.empty_like(v, memory_format=torch.contiguous_format)
    rows, num_warps = _launch_config(q_interleaved)
    if q_interleaved:
        grid = (triton.cdiv(batch_size * seq, rows),)
        _qkv_self_kernel[grid](
            q,
            k,
            v,
            q_weight,
            k_weight,
            out_q,
            out_k,
            out_v,
            batch_size * seq,
            float(eps),
            heads=heads,
            head_dim=head_dim,
            rows=rows,
            num_warps=num_warps,
        )
    else:
        n_q_programs = triton.cdiv(batch_size * seq, rows)
        grid = (n_q_programs + triton.cdiv(batch_size * seq_kv, rows),)
        _qkv_cross_kernel[grid](
            q,
            k,
            v,
            q_weight,
            k_weight,
            out_q,
            out_k,
            out_v,
            batch_size * seq,
            batch_size * seq_kv,
            n_q_programs,
            float(eps),
            heads=heads,
            head_dim=head_dim,
            rows=rows,
            num_warps=num_warps,
        )
    return out_q, out_k, out_v
