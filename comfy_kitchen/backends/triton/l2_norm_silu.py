# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton channel L2 norm * scale * weight + SiLU for ``[B, C, *spatial]``.

One program owns ``block_p`` spatial positions of one batch item. It reads the
channels twice in ``block_c`` chunks: once for the fp32 sum of squares, once to
normalize, scale, apply SiLU and store, so the normalized tensor is never
written out between the norm and the activation. x is read through a batch,
channel and position stride, so channel-first and channels-last x (the spatial
dims flattening to one position stride) are read in place; other layouts are
made contiguous first. The output is always contiguous channel-first.
"""

import functools

import torch

import triton
import triton.language as tl

_BLOCK_P, _BLOCK_C, _NUM_WARPS = 64, 32, 4


@triton.jit
def _l2_norm_silu_kernel(x_ptr, w_ptr, y_ptr, channels, positions, stride_b, stride_c, stride_p,
                         scale, eps, block_p: tl.constexpr, block_c: tl.constexpr):
    # y = silu(x / max(||x||_C, eps) * scale * w), per (batch, position)
    pid = tl.program_id(0)
    batch = tl.program_id(1).to(tl.int64)
    x_base = batch * stride_b
    y_base = batch * channels * positions
    p = pid * block_p + tl.arange(0, block_p)
    pm = p < positions
    xp = p.to(tl.int64) * stride_p
    ss = tl.zeros([block_p], tl.float32)
    for c0 in range(0, channels, block_c):
        c = c0 + tl.arange(0, block_c)
        x = tl.load(x_ptr + x_base + c[:, None].to(tl.int64) * stride_c + xp[None, :],
                    mask=(c[:, None] < channels) & pm[None, :], other=0.0).to(tl.float32)
        ss += tl.sum(x * x, 0)
    inv = 1.0 / tl.maximum(tl.sqrt(ss), eps)
    for c0 in range(0, channels, block_c):
        c = c0 + tl.arange(0, block_c)
        msk = (c[:, None] < channels) & pm[None, :]
        x = tl.load(x_ptr + x_base + c[:, None].to(tl.int64) * stride_c + xp[None, :], mask=msk,
                    other=0.0).to(tl.float32)
        g = tl.load(w_ptr + c, mask=c < channels, other=0.0).to(tl.float32)
        v = x * inv[None, :] * scale * g[:, None]
        v = v / (1.0 + tl.exp(-v))
        tl.store(y_ptr + y_base + c[:, None].to(tl.int64) * positions + p[None, :],
                 v.to(y_ptr.dtype.element_ty), mask=msk)


@functools.cache
def _eps_in(dtype: torch.dtype, eps: float) -> float:
    # F.normalize clamps the norm, a tensor of x's dtype, so eps is rounded to that dtype:
    # the default 1e-12 is 0 in fp16, and an all-zero channel vector gives 0 / 0 = NaN there.
    return float(torch.tensor(eps, dtype=dtype))


def _position_stride(x: torch.Tensor) -> int | None:
    # One stride for the row-major flattened spatial index, or None when the spatial dims
    # do not flatten to one (size-1 dims are free): channel-first contiguous gives 1,
    # channels_last / channels_last_3d give C.
    dims = [(n, s) for n, s in zip(x.shape[2:], x.stride()[2:], strict=True) if n != 1]
    if not dims:
        return 1
    expected = dims[-1][1]
    for n, s in reversed(dims):
        if s != expected:
            return None
        expected = s * n
    return dims[-1][1]


def l2_norm_silu(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    stride_p = _position_stride(x)
    if stride_p is None:
        x = x.contiguous()
        stride_p = 1
    batch, channels = x.shape[0], x.shape[1]
    positions = x.numel() // (batch * channels)
    y = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    # the reference rounds the weight to x's dtype (gamma.to(x)) before it multiplies
    w = weight.reshape(-1).to(x.dtype).contiguous()
    grid = (triton.cdiv(positions, _BLOCK_P), batch)
    _l2_norm_silu_kernel[grid](x, w, y, channels, positions, x.stride(0), x.stride(1), stride_p,
                               float(scale), _eps_in(x.dtype, eps),
                               block_p=_BLOCK_P, block_c=_BLOCK_C, num_warps=_NUM_WARPS)
    return y
