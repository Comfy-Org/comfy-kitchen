# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Eager adapter ops, their differentiable custom-op definitions, and their dispatch constraints."""

import torch

from comfy_kitchen.constraints import ExactDims, FunctionConstraints, ParamConstraint

from .decompose import (
    channel_scale,
    channel_scale_backward,
    weight_decompose,
    weight_decompose_backward,
)
from .loha import (
    loha_bypass,
    loha_bypass_backward,
    loha_merge,
    loha_merge_backward,
)
from .lokr import (
    lokr_bypass,
    lokr_bypass_backward,
    lokr_merge,
    lokr_merge_backward,
)
from .lora import (
    lora_bypass,
    lora_bypass_backward,
    lora_merge,
    lora_merge_backward,
)
from .oft import oft_apply, oft_apply_backward

__all__ = [
    "lora_bypass",
    "lora_bypass_backward",
    "lora_merge",
    "lora_merge_backward",
    "loha_bypass",
    "loha_bypass_backward",
    "loha_merge",
    "loha_merge_backward",
    "lokr_bypass",
    "lokr_bypass_backward",
    "lokr_merge",
    "lokr_merge_backward",
    "oft_apply",
    "oft_apply_backward",
    "weight_decompose",
    "weight_decompose_backward",
    "channel_scale",
    "channel_scale_backward",
    "adapter_constraints",
]

# Floating-point tensor arguments of every adapter op.
_TENSORS = {
    "lora_bypass": ("x", "up", "down", "base"),
    "lora_bypass_backward": ("grad", "x", "up", "down"),
    "lora_merge": ("weight", "up", "down"),
    "lora_merge_backward": ("grad", "up", "down"),
    "loha_bypass": ("x", "w1_up", "w1_down", "w2_up", "w2_down", "base"),
    "loha_bypass_backward": ("grad", "x", "w1_up", "w1_down", "w2_up", "w2_down"),
    "loha_merge": ("weight", "w1_up", "w1_down", "w2_up", "w2_down"),
    "loha_merge_backward": ("grad", "w1_up", "w1_down", "w2_up", "w2_down"),
    "lokr_bypass": ("x", "w1", "w2", "base"),
    "lokr_bypass_backward": ("grad", "x", "w1", "w2"),
    "lokr_merge": ("weight", "w1", "w2"),
    "lokr_merge_backward": ("grad", "w1", "w2"),
    "oft_apply": ("blocks", "x", "rescale"),
    "oft_apply_backward": ("grad", "blocks", "x", "rescale"),
    "weight_decompose": ("weight", "dora_scale", "base", "norm_weight", "delta"),
    "weight_decompose_backward": (
        "grad",
        "weight",
        "dora_scale",
        "norms",
        "base",
        "norm_weight",
        "delta",
    ),
    "channel_scale": ("x", "w"),
    "channel_scale_backward": ("grad", "x", "w"),
}

# 2-D factor arguments.
_MATRICES = frozenset(
    {"up", "down", "w1_up", "w1_down", "w2_up", "w2_down", "w1", "w2"}
)


def adapter_constraints(devices: frozenset[str]) -> dict:
    """Constraints for the adapter ops, for _build_constraints to out.update()."""
    floats = frozenset({torch.float32, torch.float16, torch.bfloat16})
    out = {}
    for name, tensors in _TENSORS.items():
        params = {}
        for arg in tensors:
            rules = (ExactDims(2),) if arg in _MATRICES else ()
            if arg == "blocks":
                rules = (ExactDims(3),)
            params[arg] = ParamConstraint(dtypes=floats, shape_rules=rules)
        out[name] = FunctionConstraints(params=params, default_devices=devices)
    return out
