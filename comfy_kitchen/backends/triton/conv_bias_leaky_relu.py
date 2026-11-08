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

The kernels are launched through the compiled kernel object, compiled once per
specialization (the pointer alignments and the integer arguments that Triton specializes on),
which keeps the host cost per call low for the small pyramid levels; when Triton has no
``knobs`` module, or a launch hook is set, the regular ``kernel[grid](...)`` launch is used.
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


_COMPILED = {}


def _int_key(v):
    # what Triton specializes an integer argument on
    return (v == 1, v % 16 == 0)


def _launch(kernel, key, grid, args, constexprs):
    knobs = getattr(triton, "knobs", None)
    hooks = None if knobs is None else knobs.runtime
    if hooks is None or hooks.launch_enter_hook is not None or hooks.launch_exit_hook is not None:
        kernel[grid](*args, **constexprs, num_warps=_NUM_WARPS)
        return
    compiled = _COMPILED.get(key)
    if compiled is None:
        compiled = kernel.warmup(*args, grid=grid, **constexprs, num_warps=_NUM_WARPS)
        compiled._init_handles()
        _COMPILED[key] = compiled
    driver = triton.runtime.driver.active
    stream = driver.get_current_stream(driver.get_current_device())
    compiled.run(
        grid[0],
        grid[1],
        1,
        stream,
        compiled.function,
        compiled.packed_metadata,
        None,
        None,
        None,
        *args,
        *constexprs.values(),
    )


def conv_bias_leaky_relu_(
    x: torch.Tensor, bias: torch.Tensor, negative_slope: float = 0.2
) -> torch.Tensor:
    n, c, h, w = x.shape
    slope = float(negative_slope)
    aligned = (x.data_ptr() % 16 == 0, bias.data_ptr() % 16 == 0)
    common = (x.device.index, x.dtype, aligned)
    if x.is_contiguous():
        hw = h * w
        block = min(_ELEMENTS, triton.next_power_of_2(hw))
        grid = (triton.cdiv(hw, block), n * c)
        key = ("nchw", block, *common, _int_key(c), _int_key(hw))
        _launch(
            _conv_bias_leaky_relu_nchw_kernel,
            key,
            grid,
            (x, bias, c, hw, slope),
            {"block": block},
        )
        return x
    pixels = n * h * w
    block_c = triton.next_power_of_2(c)
    block_p = max(1, _ELEMENTS // block_c)
    grid = (triton.cdiv(pixels, block_p), 1)
    key = ("cl", block_p, block_c, *common, _int_key(pixels), _int_key(c))
    _launch(
        _conv_bias_leaky_relu_cl_kernel,
        key,
        grid,
        (x, bias, pixels, c, slope),
        {"block_p": block_p, "block_c": block_c},
    )
    return x
