# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""3D convolution with "same" spatial padding and no temporal padding, optional nearest x2 upsample.

``y = conv3d(up(x), weight, bias, padding=(0, kh // 2, kw // 2))`` with ``up`` the nearest x2
upsample of H and W when ``upsample == 2`` (T unchanged). This is the conv of the Wan VAE
``CausalConv3d`` (the caller pads T causally) and, with a ``[Co, Ci, 1, 3, 3]`` weight and
``upsample=2``, its ``Upsample(nearest-exact, x2) + Conv2d(3, padding=1)`` frame upsampler.
"""

import torch
from torch.nn import functional

from comfy_kitchen.registry import registry


def conv3d_hw_same(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    upsample: int = 1,
) -> torch.Tensor:
    """Reference: nearest x2 upsample (upsample=2), then ``F.conv3d`` with (0, kh//2, kw//2) padding."""
    if upsample == 2:
        x = functional.interpolate(x, scale_factor=(1.0, 2.0, 2.0), mode="nearest")
    return functional.conv3d(x, weight, bias, padding=(0, weight.shape[3] // 2, weight.shape[4] // 2))


def _check(x, weight, bias, upsample):
    if x.dim() != 5 or weight.dim() != 5 or weight.shape[1] != x.shape[1]:
        raise ValueError(f"conv3d_hw_same: x [B, Ci, T, H, W] and weight [Co, Ci, kt, kh, kw], got "
                         f"{tuple(x.shape)} and {tuple(weight.shape)}")
    if weight.shape[3] % 2 == 0 or weight.shape[4] % 2 == 0:
        raise ValueError("conv3d_hw_same: kh and kw must be odd")
    if weight.shape[2] > x.shape[2]:
        raise ValueError(f"conv3d_hw_same: kt {weight.shape[2]} > T {x.shape[2]}")
    if upsample not in (1, 2) or (upsample == 2 and tuple(weight.shape[2:]) != (1, 3, 3)):
        raise ValueError("conv3d_hw_same: upsample is 1, or 2 with a [Co, Ci, 1, 3, 3] weight")
    if bias is not None and bias.numel() != weight.shape[0]:
        raise ValueError("conv3d_hw_same: bias needs Co elements")


@torch.library.custom_op("comfy_kitchen::conv3d_hw_same", mutates_args=())
def _op_conv3d_hw_same(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    upsample: int = 1,
) -> torch.Tensor:
    _check(x, weight, bias, upsample)
    kwargs = {"x": x, "weight": weight, "bias": bias, "upsample": upsample}
    impl = registry.get_implementation("conv3d_hw_same", kwargs=kwargs)
    return impl(**kwargs)


@_op_conv3d_hw_same.register_fake
def _op_conv3d_hw_same_fake(x, weight, bias=None, upsample=1):
    b, _, t, h, w = x.shape
    return x.new_empty((b, weight.shape[0], t - weight.shape[2] + 1, h * upsample, w * upsample))
