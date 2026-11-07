# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton SnakeBeta (log-scale alpha / beta), optionally after a residual add, one pass.

x (and residual) ``[B, C, T]`` contiguous, alpha / beta ``[C]``. One program owns ``block``
time steps of one (batch, channel) row: it computes ``exp(alpha[c])`` and
``1 / (exp(beta[c]) + 1e-9)`` once in registers, then per element

    s   = x + residual                      (residual given; stored when the sum is returned)
    out = s + 1 / (exp(beta) + 1e-9) * sin(s * exp(alpha)) ** 2

in fp32, rounded once to the output dtype. The stored sum is the fp32 sum rounded to the dtype,
which for fp16 inputs is the correctly rounded fp16 sum (fp32's 24 bits >= 2 x 11 + 2), so
it equals the eager add bit for bit; the activation uses the fp32 sum. tl.sin / tl.exp only.

One launch configuration on every GPU: grid (cdiv(T, 2048), B * C), block 2048, 4 warps. Of the
three configurations measured at the Oobleck decoder's shapes (this one, block 8192 with 8 warps,
and a flat 1D grid of block 4096 with 8 warps), it was within 1% of the fastest on an RTX 5090 at
every shape but one (11% behind the flat grid at [1, 1024, 30000]) and within 7% of the fastest
on a Radeon 8060S.
"""

import torch

import triton
import triton.language as tl

_BLOCK = 2048
_NUM_WARPS = 4


@triton.jit
def _snake_beta_kernel(
    x_ptr,
    res_ptr,
    alpha_ptr,
    beta_ptr,
    sum_ptr,
    out_ptr,
    channels,
    steps,
    has_res: tl.constexpr,
    write_sum: tl.constexpr,
    block: tl.constexpr,
):
    pid_t = tl.program_id(0)
    row = tl.program_id(1)
    c = row % channels
    a = tl.exp(tl.load(alpha_ptr + c).to(tl.float32))
    # 0.000000001: SnakeBeta.no_div_by_zero
    inv_b = 1.0 / (tl.exp(tl.load(beta_ptr + c).to(tl.float32)) + 0.000000001)
    offs_t = pid_t * block + tl.arange(0, block)
    mask = offs_t < steps
    offs = row.to(tl.int64) * steps + offs_t
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    if has_res:
        x = x + tl.load(res_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        if write_sum:
            tl.store(sum_ptr + offs, x.to(sum_ptr.dtype.element_ty), mask=mask)
    s = tl.sin(x * a)
    y = x + inv_b * (s * s)
    tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


def _launch(x, alpha, beta, residual, write_sum):
    b, c, t = x.shape
    out = torch.empty_like(x)
    s = torch.empty_like(x) if write_sum else out
    res = x if residual is None else residual
    grid = (triton.cdiv(t, _BLOCK), b * c)
    _snake_beta_kernel[grid](
        x,
        res,
        alpha,
        beta,
        s,
        out,
        c,
        t,
        has_res=residual is not None,
        write_sum=write_sum,
        block=_BLOCK,
        num_warps=_NUM_WARPS,
    )
    return s, out


def snake_beta(
    x: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    return _launch(x, alpha, beta, residual, False)[1]


def snake_beta_with_sum(
    x: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _launch(x, alpha, beta, residual, True)
