# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Triton kernels for the weight-adapter ops: lora, loha, lokr, oft, and decompose (DoRA, IA3)."""

from .constraints import adapter_constraints
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
