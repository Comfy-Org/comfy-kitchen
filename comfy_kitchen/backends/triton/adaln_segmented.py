# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton segmented AdaLN and segmented gated residual for x ``[B, L, D]``.

One program owns one token row and holds the whole row in registers (block_d =
next_power_of_2(D)), so x is read once: the norm statistics, the normalize, the
optional RMSNorm weight and the modulation of the row's segment are computed in fp32
and rounded once on the store. The gated residual reads x, the residual and the gate
row of the token's segment and stores ``residual + x * gate``, in place when the
output is the residual. The per-segment modulation rows are read through their own
batch strides, so the ``[2, B, D]`` views the models produce are not copied.

Launch: num_warps 8 for block_d >= 4096, else 4, for both kernels. Of the configs
tried on the RTX 5090 and the Radeon 8060S (this one, a two-pass loop over 1024 /
2048-column chunks, and autotuning num_warps 4 / 8 / 16), it was the fastest or within
3% of the fastest at every captured Qwen-Image, Krea2 and Flux.2 shape (D 3072, 4096,
6144). The gate kernel is not autotuned: autotuning re-runs an in-place launch.
"""

import torch

import triton
import triton.language as tl


@triton.jit
def _adaln_segmented_kernel(
    x_ptr,
    y_ptr,
    w_ptr,
    sc0_ptr,
    sc1_ptr,
    sh0_ptr,
    sh1_ptr,
    seq_len,
    d,
    split,
    stride_xb,
    stride_xl,
    stride_sc0b,
    stride_sc1b,
    stride_sh0b,
    stride_sh1b,
    eps,
    w_off,
    block_d: tl.constexpr,
    rms: tl.constexpr,
    has_w: tl.constexpr,
):
    row = tl.program_id(0)
    b = row // seq_len
    tok = row - b * seq_len
    cols = tl.arange(0, block_d)
    mask = cols < d
    x_off = b.to(tl.int64) * stride_xb + tok.to(tl.int64) * stride_xl
    x = tl.load(x_ptr + x_off + cols, mask=mask, other=0.0).to(tl.float32)
    if rms:
        rstd = tl.rsqrt(tl.sum(x * x, axis=0) / d + eps)
        xn = x * rstd
    else:
        mean = tl.sum(x, axis=0) / d
        xc = tl.where(mask, x - mean, 0.0)
        rstd = tl.rsqrt(tl.sum(xc * xc, axis=0) / d + eps)
        xn = xc * rstd
    if has_w:
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32) + w_off
        xn = xn * w
    if tok < split:
        sc = tl.load(sc0_ptr + b * stride_sc0b + cols, mask=mask, other=0.0).to(tl.float32)
        sh = tl.load(sh0_ptr + b * stride_sh0b + cols, mask=mask, other=0.0).to(tl.float32)
    else:
        sc = tl.load(sc1_ptr + b * stride_sc1b + cols, mask=mask, other=0.0).to(tl.float32)
        sh = tl.load(sh1_ptr + b * stride_sh1b + cols, mask=mask, other=0.0).to(tl.float32)
    out = xn * (1.0 + sc) + sh
    tl.store(y_ptr + row.to(tl.int64) * d + cols, out.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _gated_residual_segmented_kernel(
    x_ptr,
    r_ptr,
    o_ptr,
    g0_ptr,
    g1_ptr,
    seq_len,
    d,
    split,
    stride_xb,
    stride_xl,
    stride_rb,
    stride_rl,
    stride_ob,
    stride_ol,
    stride_g0b,
    stride_g1b,
    block_d: tl.constexpr,
):
    row = tl.program_id(0)
    b = row // seq_len
    tok = row - b * seq_len
    cols = tl.arange(0, block_d)
    mask = cols < d
    b64 = b.to(tl.int64)
    t64 = tok.to(tl.int64)
    if tok < split:
        g = tl.load(g0_ptr + b * stride_g0b + cols, mask=mask, other=0.0).to(tl.float32)
    else:
        g = tl.load(g1_ptr + b * stride_g1b + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + b64 * stride_xb + t64 * stride_xl + cols, mask=mask, other=0.0)
    r = tl.load(r_ptr + b64 * stride_rb + t64 * stride_rl + cols, mask=mask, other=0.0)
    out = r.to(tl.float32) + x.to(tl.float32) * g
    tl.store(
        o_ptr + b64 * stride_ob + t64 * stride_ol + cols, out.to(o_ptr.dtype.element_ty), mask=mask
    )


def _num_warps(block_d: int) -> int:
    return 8 if block_d >= 4096 else 4


def _segment_rows(t: torch.Tensor, batch: int):
    # [2, B, D] / [2, 1, D] -> the two [B, D] row views and their batch strides (0 broadcasts)
    if t.stride(-1) != 1:
        t = t.contiguous()
    rows0, rows1 = t[0], t[1]
    if t.shape[1] == batch and batch > 1:
        return rows0, rows1, rows0.stride(0), rows1.stride(0)
    return rows0, rows1, 0, 0


def _rows(t: torch.Tensor) -> torch.Tensor:
    return t if t.stride(-1) == 1 else t.contiguous()


def adaln_segmented(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    split: int,
    eps: float = 1e-6,
    rms_weight: torch.Tensor | None = None,
    rms_weight_offset: float = 0.0,
) -> torch.Tensor:
    x = _rows(x)
    batch, seq_len, d = x.shape
    sc0, sc1, s_sc0, s_sc1 = _segment_rows(scale, batch)
    sh0, sh1, s_sh0, s_sh1 = _segment_rows(shift, batch)
    has_w = rms_weight is not None
    w = rms_weight.reshape(-1).contiguous() if has_w else x
    y = torch.empty((batch, seq_len, d), dtype=x.dtype, device=x.device)
    block_d = triton.next_power_of_2(d)
    _adaln_segmented_kernel[(batch * seq_len,)](
        x,
        y,
        w,
        sc0,
        sc1,
        sh0,
        sh1,
        seq_len,
        d,
        split,
        x.stride(0),
        x.stride(1),
        s_sc0,
        s_sc1,
        s_sh0,
        s_sh1,
        float(eps),
        float(rms_weight_offset),
        block_d=block_d,
        rms=has_w,
        has_w=has_w,
        num_warps=_num_warps(block_d),
    )
    return y


def _gated_residual_segmented(residual, x, gate, split, out):
    batch, seq_len, d = x.shape
    g0, g1, s_g0, s_g1 = _segment_rows(gate, batch)
    block_d = triton.next_power_of_2(d)
    _gated_residual_segmented_kernel[(batch * seq_len,)](
        x,
        residual,
        out,
        g0,
        g1,
        seq_len,
        d,
        split,
        x.stride(0),
        x.stride(1),
        residual.stride(0),
        residual.stride(1),
        out.stride(0),
        out.stride(1),
        s_g0,
        s_g1,
        block_d=block_d,
        num_warps=_num_warps(block_d),
    )


def gated_residual_segmented(
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor,
    split: int,
) -> torch.Tensor:
    residual, x = _rows(residual), _rows(x)
    out = torch.empty(residual.shape, dtype=residual.dtype, device=residual.device)
    _gated_residual_segmented(residual, x, gate, split, out)
    return out


def gated_residual_segmented_(
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor,
    split: int,
) -> None:
    # the call rule only sends a residual with unit stride on D here, so it is written in place
    _gated_residual_segmented(residual, _rows(x), gate, split, residual)
