# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton channel L2 norm * scale * weight + SiLU for ``[B, C, *spatial]``.

One program owns ``block_p`` spatial positions of one batch item. It reads the
channels twice in ``block_c`` chunks: once for the fp32 sum of squares, once to
normalize, scale, apply SiLU and store, so the normalized tensor is never
written out between the norm and the activation. Other layouts are made
contiguous first; the output is always contiguous.
"""

import torch

import triton
import triton.language as tl

_BLOCK_P, _BLOCK_C, _NUM_WARPS = 64, 32, 4


@triton.jit
def _l2_norm_silu_kernel(x_ptr, w_ptr, y_ptr, channels, positions, scale, eps,
                         block_p: tl.constexpr, block_c: tl.constexpr):
    # y = silu(x / max(||x||_C, eps) * scale * w), per (batch, position)
    pid = tl.program_id(0)
    base = tl.program_id(1).to(tl.int64) * channels * positions
    p = pid * block_p + tl.arange(0, block_p)
    pm = p < positions
    ss = tl.zeros([block_p], tl.float32)
    for c0 in range(0, channels, block_c):
        c = c0 + tl.arange(0, block_c)
        x = tl.load(x_ptr + base + c[:, None].to(tl.int64) * positions + p[None, :],
                    mask=(c[:, None] < channels) & pm[None, :], other=0.0).to(tl.float32)
        ss += tl.sum(x * x, 0)
    inv = 1.0 / tl.maximum(tl.sqrt(ss), eps)
    for c0 in range(0, channels, block_c):
        c = c0 + tl.arange(0, block_c)
        msk = (c[:, None] < channels) & pm[None, :]
        x = tl.load(x_ptr + base + c[:, None].to(tl.int64) * positions + p[None, :], mask=msk,
                    other=0.0).to(tl.float32)
        g = tl.load(w_ptr + c, mask=c < channels, other=0.0).to(tl.float32)
        v = x * inv[None, :] * scale * g[:, None]
        v = v / (1.0 + tl.exp(-v))
        tl.store(y_ptr + base + c[:, None].to(tl.int64) * positions + p[None, :],
                 v.to(y_ptr.dtype.element_ty), mask=msk)


def l2_norm_silu(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    x = x.contiguous()
    batch, channels = x.shape[0], x.shape[1]
    positions = x.numel() // (batch * channels)
    y = torch.empty_like(x)
    # the reference rounds the weight to x's dtype (gamma.to(x)) before it multiplies
    w = weight.reshape(-1).to(x.dtype).contiguous()
    grid = (triton.cdiv(positions, _BLOCK_P), batch)
    _l2_norm_silu_kernel[grid](x, w, y, channels, positions, float(scale), eps,
                               block_p=_BLOCK_P, block_c=_BLOCK_C, num_warps=_NUM_WARPS)
    return y
