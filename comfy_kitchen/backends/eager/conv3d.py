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
) -> Tensor:
    """3D conv with zero padding plus an optional residual add (torch's
    accumulate mode; the CUDA backend accumulates in fp16)."""
    out = functional.conv3d(x, weight, bias, stride=stride)
    out = out if residual is None else out + residual
    return out.contiguous(memory_format=torch.channels_last_3d)  # like the CUDA backend and the fake


@torch.library.custom_op("comfy_kitchen::fp16_conv3d", mutates_args=())
def _op_fp16_conv3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    residual: torch.Tensor | None,
    stride: list[int],
) -> torch.Tensor:
    kwargs = {"x": x, "weight": weight, "bias": bias, "residual": residual, "stride": stride}
    impl = registry.get_implementation("fp16_conv3d", kwargs=kwargs)
    return impl(**kwargs)


@_op_fp16_conv3d.register_fake
def _op_fp16_conv3d_fake(x, weight, bias, residual, stride):
    n, _, d, h, w = x.shape
    k, _, t, r, s = weight.shape
    sd, sh, sw = stride
    shape = (n, k, (d - t) // sd + 1, (h - r) // sh + 1, (w - s) // sw + 1)
    return torch.empty(shape, dtype=x.dtype, device=x.device, memory_format=torch.channels_last_3d)
