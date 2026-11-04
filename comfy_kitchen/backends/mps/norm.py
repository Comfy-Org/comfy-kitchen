# SPDX-License-Identifier: Apache-2.0
"""AdaLN (LayerNorm or RMSNorm, then * (1 + scale) + shift) on Apple GPUs."""
from collections.abc import Mapping

from torch import Tensor

from comfy_kitchen.constraints import ValidationResult

from . import kernels


def adaln(x: Tensor, scale: Tensor, shift: Tensor, eps: float = 1e-6) -> Tensor:
    return kernels.adaln(x, scale, shift, eps)


def rms_adaln(x: Tensor, scale: Tensor, shift: Tensor, eps: float = 1e-6) -> Tensor:
    return kernels.adaln(x, scale, shift, eps, rms=True)


def validate_adaln(kwargs: Mapping[str, object]) -> ValidationResult:
    reason = kernels.adaln_unsupported(kwargs.get("x"), kwargs.get("scale"), kwargs.get("shift"))
    return ValidationResult.ok() if reason is None else ValidationResult.fail("x", reason)
