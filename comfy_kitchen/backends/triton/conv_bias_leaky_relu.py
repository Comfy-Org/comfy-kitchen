# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton conv bias add + LeakyReLU, one in-place pass over the conv output.

One kernel per layout of the conv output x ``[N, C, H, W]``:

    contiguous (NCHW)  grid (cdiv(H*W, block), N*C): each program owns ``block`` pixels of one
                       (n, c) plane and adds that channel's bias (a scalar load)
    channels_last      grid (cdiv(N*H*W, block_p), 1): each program owns a ``[block_p, block_c]``
                       tile of block_p pixels by all C channels (block_c = next power of two of
                       C) and adds the bias vector

Each element is read once and written once in place: ``v = (x + b)`` in fp32 rounded to x's
dtype (what the stock in-place ``add_`` stores), then ``where(v > 0, v, v * slope)`` in fp32
rounded to x's dtype (what the stock LeakyReLU kernel computes), so the result is equal to the
stock chain's. Launch configuration: about 1024 elements per program, 4 warps, on CUDA and HIP.
"""

import torch

import triton
import triton.language as tl

_ELEMENTS = 1024
_NUM_WARPS = 4


@triton.jit
def _conv_bias_leaky_relu_nchw_kernel(x_ptr, b_ptr, channels, hw, slope, block: tl.constexpr):
    col = tl.program_id(0) * block + tl.arange(0, block)
    row = tl.program_id(1)  # n * C + c
    mask = col < hw
    offs = row.to(tl.int64) * hw + col
    b = tl.load(b_ptr + row % channels).to(tl.float32)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    v = (x + b).to(x_ptr.dtype.element_ty).to(tl.float32)
    out = tl.where(v > 0, v, v * slope)
    tl.store(x_ptr + offs, out.to(x_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _conv_bias_leaky_relu_cl_kernel(
    x_ptr, b_ptr, pixels, channels, slope, block_p: tl.constexpr, block_c: tl.constexpr
):
    p = tl.program_id(0) * block_p + tl.arange(0, block_p)
    c = tl.arange(0, block_c)
    mask_c = c < channels
    mask = (p < pixels)[:, None] & mask_c[None, :]
    offs = p.to(tl.int64)[:, None] * channels + c[None, :]
    b = tl.load(b_ptr + c, mask=mask_c).to(tl.float32)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    v = (x + b[None, :]).to(x_ptr.dtype.element_ty).to(tl.float32)
    out = tl.where(v > 0, v, v * slope)
    tl.store(x_ptr + offs, out.to(x_ptr.dtype.element_ty), mask=mask)


def conv_bias_leaky_relu_(
    x: torch.Tensor, bias: torch.Tensor, negative_slope: float = 0.2
) -> torch.Tensor:
    n, c, h, w = x.shape
    slope = float(negative_slope)
    if x.is_contiguous():
        hw = h * w
        block = min(_ELEMENTS, triton.next_power_of_2(hw))
        grid = (triton.cdiv(hw, block), n * c)
        _conv_bias_leaky_relu_nchw_kernel[grid](
            x, bias, c, hw, slope, block=block, num_warps=_NUM_WARPS
        )
        return x
    pixels = n * h * w
    block_c = triton.next_power_of_2(c)
    block_p = max(1, _ELEMENTS // block_c)
    grid = (triton.cdiv(pixels, block_p),)
    _conv_bias_leaky_relu_cl_kernel[grid](
        x, bias, pixels, c, slope, block_p=block_p, block_c=block_c, num_warps=_NUM_WARPS
    )
    return x
