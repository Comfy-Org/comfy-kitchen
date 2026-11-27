# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TileLang adapter kernels for pre-Ampere NVIDIA tensor cores (sm_70, sm_75).

Triton lowers tl.dot to tensor cores from sm_80 only; on a V100 or a Turing
card its dots run as scalar FMA, 10-300x below cuBLAS in fp16. TileLang emits
the Volta and Turing MMA instructions, so the dot-bound ops get a fused
kernel there too. Configs come from the shared tuner (tables, opt-in runtime
tuning, heuristics) under their own kernel names.
"""

from .constraints import adapter_constraints
from .loha import loha_bypass, loha_bypass_backward, loha_merge, loha_merge_backward
from .lokr import lokr_bypass, lokr_bypass_backward
from .lora import lora_bypass, lora_bypass_backward, lora_merge, lora_merge_backward
from .oft import oft_apply

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
    "oft_apply",
    "adapter_constraints",
]
