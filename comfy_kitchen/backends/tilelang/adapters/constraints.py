# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dispatch constraints of the tilelang adapter ops: NVIDIA sm_70+, fp16 activations and merge factors."""

import torch

from comfy_kitchen.backends.eager.adapters import (
    adapter_constraints as eager_constraints,
)
from comfy_kitchen.constraints import FunctionConstraints, ValidationResult

_DEVICES = frozenset({"cuda"})
_MIN_CAPABILITY = (7, 0)
# Cast to fp16 like ComfyUI casts them; merge factors stay strict.
_ANY_FLOAT = frozenset({"base", "weight", "blocks", "rescale"})
_BYPASS_FACTORS = frozenset(
    {"up", "down", "w1", "w2", "w1_up", "w1_down", "w2_up", "w2_down"}
)


def _device_rule(op: str):
    lenient = _ANY_FLOAT | (_BYPASS_FACTORS if "bypass" in op else frozenset())

    def rule(kwargs) -> ValidationResult:
        tensors = {k: v for k, v in kwargs.items() if isinstance(v, torch.Tensor)}
        if not tensors:
            return ValidationResult.ok()
        devices = {t.device for t in tensors.values()}
        if len(devices) > 1:
            return ValidationResult.fail("__device__", "tensors on different devices")
        device = next(iter(devices))
        if device.type != "cuda" or getattr(torch.version, "hip", None):
            return ValidationResult.fail(
                "__device__", "tilelang adapters run on NVIDIA CUDA"
            )
        cap = torch.cuda.get_device_capability(device)
        if cap < _MIN_CAPABILITY:
            return ValidationResult.fail(
                "__hardware__", f"sm_{cap[0]}{cap[1]}: tilelang needs sm_70+"
            )
        for name, t in tensors.items():
            if name in lenient:
                if not t.dtype.is_floating_point:
                    return ValidationResult.fail(
                        name, f"{t.dtype} is not a float dtype"
                    )
            elif t.dtype != torch.float16:
                return ValidationResult.fail(
                    name, f"{t.dtype}: tilelang kernels take fp16"
                )
        return ValidationResult.ok()

    return rule


def adapter_constraints(names) -> dict:
    """The eager ops' param rules for the named ops on CUDA, plus the device rule."""
    eager = eager_constraints(_DEVICES)
    return {
        name: FunctionConstraints(
            params=eager[name].params,
            default_devices=_DEVICES,
            call_rules=(_device_rule(name),),
        )
        for name in names
    }
