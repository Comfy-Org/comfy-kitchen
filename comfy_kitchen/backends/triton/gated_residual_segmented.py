# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton segmented gated residual for x ``[B, L, D]``.

One program owns one token row and holds the whole row in registers (block_d =
next_power_of_2(D)). It reads x, the residual and the gate row of the token's segment
and stores ``residual + x * gate``, computed in fp32 and rounded once, in place when
the output is the residual. The per-segment gate rows are read through their own
batch strides, so the ``[2, B, D]`` views the models produce are not copied.

Launch: num_warps 8 for block_d >= 4096, else 4. The kernel is not autotuned: autotuning
re-runs an in-place launch.
"""

import torch

import triton
import triton.language as tl


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
