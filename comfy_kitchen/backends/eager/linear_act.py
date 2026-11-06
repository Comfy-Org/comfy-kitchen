# SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Linear layers with a fused activation epilogue: eager references."""

import torch
from torch import Tensor
from torch.nn import functional

from comfy_kitchen.registry import registry


def linear_gelu(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    approximate: str = "tanh",
) -> Tensor:
    """Reference ``gelu(linear(x, weight, bias), approximate=approximate)``."""
    return functional.gelu(functional.linear(x, weight, bias), approximate=approximate)


# =============================================================================
# torch.library Custom Op Definitions
# =============================================================================


@torch.library.custom_op("comfy_kitchen::linear_gelu", mutates_args=())
def _op_linear_gelu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    approximate: str = "tanh",
) -> torch.Tensor:
    kwargs = {"x": x, "weight": weight, "bias": bias, "approximate": approximate}
    impl = registry.get_implementation("linear_gelu", kwargs=kwargs)
    return impl(**kwargs)


@_op_linear_gelu.register_fake
def _op_linear_gelu_fake(x, weight, bias=None, approximate="tanh"):
    return x.new_empty((*x.shape[:-1], weight.shape[0]))
