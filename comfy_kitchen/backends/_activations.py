# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Elementwise activations that a quantizer can absorb on the way in.

An MLP's ``linear(act(proj(x)))`` otherwise writes act's output to HBM and reads
it straight back to quantize it. Backends that can fold the activation into the
quantizer do so; the rest apply it here first and then quantize unchanged, so
every path produces the same result.
"""

import torch

# Must match the kActNone / kActGeluTanh enum in backends/cuda/ops/int8_linear.cu
# the CUDA backend passes these codes straight to the kernel.
INPUT_ACT_TO_CODE: dict[str | None, int] = {
    None: 0,
    "none": 0,
    "gelu_tanh": 1,
}


def input_act_code(input_act: str | None) -> int:
    """Kernel code for `input_act`, rejecting anything unsupported."""
    try:
        return INPUT_ACT_TO_CODE[input_act]
    except KeyError:
        raise ValueError(_unsupported(input_act)) from None


def apply_input_act(x: torch.Tensor, input_act: str | None) -> torch.Tensor:
    """Apply the pre-quantization activation eagerly.

    Used by backends and shapes the fused quantizer does not cover, so the
    fallback result matches the fused one.
    """
    if input_act in (None, "none"):
        return x
    if input_act == "gelu_tanh":
        return torch.nn.functional.gelu(x, approximate="tanh")
    raise ValueError(_unsupported(input_act))


def _unsupported(input_act) -> str:
    known = sorted(k for k in INPUT_ACT_TO_CODE if k is not None)
    return f"unsupported input_act: {input_act!r} (expected one of {known})"
