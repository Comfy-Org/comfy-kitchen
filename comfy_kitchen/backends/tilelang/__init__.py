# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TileLang backend: fp16 adapter kernels for NVIDIA sm_70+, autotuned against triton and torch."""

import torch

from comfy_kitchen.registry import registry

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
]


def _nvidia_present() -> bool:
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None):
        return False
    return any(
        torch.cuda.get_device_capability(i) >= (7, 0)
        for i in range(torch.cuda.device_count())
    )


_ERROR = None if _nvidia_present() else "no NVIDIA CUDA device sm_70+"
if _ERROR is None:
    try:
        from .adapters import (
            adapter_constraints,
            loha_bypass,
            loha_bypass_backward,
            loha_merge,
            loha_merge_backward,
            lokr_bypass,
            lokr_bypass_backward,
            lora_bypass,
            lora_bypass_backward,
            lora_merge,
            lora_merge_backward,
            oft_apply,
        )
    except ImportError as e:
        _ERROR = f"ImportError: {e!s}"


def _register():
    if _ERROR is not None:
        registry.mark_unavailable("tilelang", _ERROR)
        return
    registry.register(
        name="tilelang",
        module=__import__(__name__, fromlist=__all__),
        capabilities=adapter_constraints(__all__),
    )


_register()
