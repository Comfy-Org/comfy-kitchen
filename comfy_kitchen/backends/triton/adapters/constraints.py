# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dispatch constraints of the triton adapter ops: the eager ops' rules plus a per-tensor-device rule."""

import torch

from comfy_kitchen.backends.eager.adapters import (
    adapter_constraints as eager_constraints,
)
from comfy_kitchen.constraints import FunctionConstraints, ValidationResult

from .tuning import device_info

_DEVICES = frozenset({"cuda", "xpu"})
_MIN_CAPABILITY = (7, 0)
_BF16_CAPABILITY = (8, 0)


def _device_rule(kwargs) -> ValidationResult:
    """One device for all tensors, sm_70+ (sm_80+ for bf16) on NVIDIA."""
    tensors = [v for v in kwargs.values() if isinstance(v, torch.Tensor)]
    if not tensors:
        return ValidationResult.ok()
    first = tensors[0]
    for t in tensors[1:]:
        if t.device != first.device:
            return ValidationResult.fail("__device__", "tensors on different devices")
    dev = device_info(first.device)
    if dev.capability is None:
        return ValidationResult.ok()
    if dev.capability < _MIN_CAPABILITY:
        return ValidationResult.fail("__hardware__", f"{dev.arch} below sm_70")
    if dev.capability < _BF16_CAPABILITY and any(
        t.dtype == torch.bfloat16 for t in tensors
    ):
        return ValidationResult.fail(
            "__hardware__", f"bf16 tl.dot needs sm_80, have {dev.arch}"
        )
    return ValidationResult.ok()


def adapter_constraints() -> dict:
    """The eager ops' param rules on CUDA/XPU, plus the device rule."""
    return {
        name: FunctionConstraints(
            params=fc.params,
            default_devices=_DEVICES,
            call_rules=(_device_rule,),
        )
        for name, fc in eager_constraints(_DEVICES).items()
    }
