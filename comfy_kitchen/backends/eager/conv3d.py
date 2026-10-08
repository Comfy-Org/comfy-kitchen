# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor
from torch.nn import functional

from comfy_kitchen.registry import registry


def fp16_conv3d(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    residual: Tensor | None,
    stride: list[int],
    padding: list[int],
) -> Tensor:
    """3D conv with zero padding plus an optional residual add (torch's
    accumulate mode; the CUDA backend accumulates in fp16)."""
    out = functional.conv3d(x, weight, bias, stride=stride, padding=padding)
    out = out if residual is None else out + residual
    return out.contiguous(memory_format=torch.channels_last_3d)  # like the CUDA backend and the fake


def fp16_conv3d_out(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    residual: Tensor | None,
    stride: list[int],
    padding: list[int],
    out: Tensor,
) -> None:
    """fp16_conv3d written into ``out``, an NDHWC-ordered tensor or view of the output shape."""
    out.copy_(fp16_conv3d(x, weight, bias, residual, stride, padding))


def _out_shape(x, weight, stride, padding):
    n, _, d, h, w = x.shape
    k, _, t, r, s = weight.shape
    (sd, sh, sw), (pd, ph, pw) = stride, padding
    return (n, k, (d + 2 * pd - t) // sd + 1, (h + 2 * ph - r) // sh + 1,
            (w + 2 * pw - s) // sw + 1)


@torch.library.custom_op("comfy_kitchen::fp16_conv3d_out", mutates_args=("out",))
def _op_fp16_conv3d_out(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    residual: torch.Tensor | None,
    stride: list[int],
    padding: list[int],
    out: torch.Tensor,
) -> None:
    # copy_ would broadcast or cast into a mismatched buffer instead of failing
    if min(stride) >= 1:
        shape = _out_shape(x, weight, stride, padding)
        if out.shape != shape or out.dtype != x.dtype or out.device != x.device:
            raise ValueError(f"fp16_conv3d: out must be {shape} {x.dtype} on {x.device}")
    kwargs = {"x": x, "weight": weight, "bias": bias, "residual": residual, "stride": stride,
              "padding": padding, "out": out}
    impl = registry.get_implementation("fp16_conv3d_out", kwargs=kwargs)
    impl(**kwargs)


@_op_fp16_conv3d_out.register_fake
def _op_fp16_conv3d_out_fake(x, weight, bias, residual, stride, padding, out):
    return None


@torch.library.custom_op("comfy_kitchen::fp16_conv3d", mutates_args=())
def _op_fp16_conv3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    residual: torch.Tensor | None,
    stride: list[int],
    padding: list[int],
) -> torch.Tensor:
    kwargs = {"x": x, "weight": weight, "bias": bias, "residual": residual, "stride": stride,
              "padding": padding}
    impl = registry.get_implementation("fp16_conv3d", kwargs=kwargs)
    return impl(**kwargs)


@_op_fp16_conv3d.register_fake
def _op_fp16_conv3d_fake(x, weight, bias, residual, stride, padding):
    shape = _out_shape(x, weight, stride, padding)
    return torch.empty(shape, dtype=x.dtype, device=x.device, memory_format=torch.channels_last_3d)
