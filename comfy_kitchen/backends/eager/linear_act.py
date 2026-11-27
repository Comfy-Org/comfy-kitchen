# SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Linear layers with a fused activation epilogue: eager references."""

import torch
from torch import Tensor
from torch.nn import functional

from comfy_kitchen.registry import registry


def linear_geglu(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    approximate: str = "none",
) -> Tensor:
    """Reference ``h * gelu(gate)``, ``h, gate = linear(x, weight, bias).chunk(2, -1)``."""
    h, gate = functional.linear(x, weight, bias).chunk(2, dim=-1)
    return h * functional.gelu(gate, approximate=approximate)


def linear_swiglu(
    x: Tensor,
    w1: Tensor,
    w3: Tensor,
    b1: Tensor | None = None,
    b3: Tensor | None = None,
) -> Tensor:
    """Reference SwiGLU: ``silu(linear(x, w1, b1)) * linear(x, w3, b3)``."""
    return functional.silu(functional.linear(x, w1, b1)) * functional.linear(x, w3, b3)


# =============================================================================
# torch.library Custom Op Definitions
# =============================================================================


@torch.library.custom_op("comfy_kitchen::linear_geglu", mutates_args=())
def _op_linear_geglu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    approximate: str = "none",
) -> torch.Tensor:
    kwargs = {"x": x, "weight": weight, "bias": bias, "approximate": approximate}
    impl = registry.get_implementation("linear_geglu", kwargs=kwargs)
    return impl(**kwargs)


@_op_linear_geglu.register_fake
def _op_linear_geglu_fake(x, weight, bias=None, approximate="none"):
    return x.new_empty((*x.shape[:-1], weight.shape[0] // 2))


@torch.library.custom_op("comfy_kitchen::linear_swiglu", mutates_args=())
def _op_linear_swiglu(
    x: torch.Tensor,
    w1: torch.Tensor,
    w3: torch.Tensor,
    b1: torch.Tensor | None = None,
    b3: torch.Tensor | None = None,
) -> torch.Tensor:
    kwargs = {"x": x, "w1": w1, "w3": w3, "b1": b1, "b3": b3}
    impl = registry.get_implementation("linear_swiglu", kwargs=kwargs)
    return impl(**kwargs)


@_op_linear_swiglu.register_fake
def _op_linear_swiglu_fake(x, w1, w3, b1=None, b3=None):
    return x.new_empty((*x.shape[:-1], w1.shape[0]))
