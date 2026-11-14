# SPDX-License-Identifier: Apache-2.0
"""The gated input activations (swiglu, situ) on Apple GPUs.

The other activations are a single torch op already and are left to eager by
the call rule.
"""
from collections.abc import Mapping

import torch

from comfy_kitchen.constraints import ValidationResult

from . import kernels


def input_act(
    x: torch.Tensor,
    input_act: str,
    act_weight: torch.Tensor | None = None,
    act_eps: float = 0.0,
) -> torch.Tensor:
    return kernels.gated(x, input_act)


def validate_input_act(kwargs: Mapping[str, object]) -> ValidationResult:
    reason = kernels.gated_unsupported(kwargs.get("x"), kwargs.get("input_act"))
    return ValidationResult.ok() if reason is None else ValidationResult.fail("input_act", reason)
