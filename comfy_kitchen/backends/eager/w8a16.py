# SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""W8A16 linear: unscaled FP8 weight, 16-bit activations. Eager reference.

Checkpoints such as ``qwen_image_fp8_e4m3fn`` or ``flux1-dev-fp8`` store their
linear weights as plain ``float8_e4m3fn`` with no scale: the weight value is
the FP8 value itself. Without FP8 matrix units in use, the model runs in bf16
(or fp16) and every forward casts each weight to that dtype before a 16-bit
GEMM::

    y = linear(x, weight.to(x.dtype), bias.to(x.dtype))

This module is that reference. The cast is exact (e4m3 values are all
representable in bf16 and fp16), so a backend that converts the FP8 weight in
registers multiplies exactly the same operands.
"""

import torch
from torch import Tensor
from torch.nn import functional

from comfy_kitchen.registry import registry


def linear_w8a16(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
) -> Tensor:
    """Reference: ``linear(x, weight.to(x.dtype), bias.to(x.dtype))``."""
    return functional.linear(
        x,
        weight.to(x.dtype),
        bias.to(x.dtype) if bias is not None else None,
    )


# =============================================================================
# torch.library Custom Op Definitions
# =============================================================================


@torch.library.custom_op("comfy_kitchen::linear_w8a16", mutates_args=())
def _op_linear_w8a16(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    kwargs = {"x": x, "weight": weight, "bias": bias}
    impl = registry.get_implementation("linear_w8a16", kwargs=kwargs)
    return impl(**kwargs)


@_op_linear_w8a16.register_fake
def _op_linear_w8a16_fake(x, weight, bias=None):
    return x.new_empty((*x.shape[:-1], weight.shape[0]))
