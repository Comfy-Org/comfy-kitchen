# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Activations that a quantizer can absorb on the way in.

An MLP's ``linear(act(proj(x)))`` otherwise writes act's output to HBM and reads
it straight back to quantize it. Backends that can fold the activation into the
quantizer do so; the rest apply it here first and then quantize unchanged, so
every path produces the same result.

``swiglu`` is the gated pair variant: the input carries ``[gate | up]`` halves
on the last dim and the activation ``silu(gate) * up`` halves it, so the
quantized row is half as wide as the raw input row.

``rms_norm`` is the row-wise ``x * rsqrt(mean(x^2) + eps) * w``; it carries a
K-element weight and an eps alongside the code.
"""

import torch

# Must match the enum in backends/cuda/input_act_codes.h; the CUDA backend
# passes these codes straight to the kernel.
INPUT_ACT_TO_CODE: dict[str | None, int] = {
    None: 0,
    "none": 0,
    "gelu_tanh": 1,
    "swiglu": 2,
    "rms_norm": 3,
}


def input_act_code(input_act: str | None) -> int:
    """Kernel code for `input_act`, rejecting anything unsupported."""
    try:
        return INPUT_ACT_TO_CODE[input_act]
    except KeyError:
        raise ValueError(_unsupported(input_act)) from None


def input_act_width(input_act: str | None) -> int:
    """How many input columns produce one activated column (1, or 2 for swiglu)."""
    return 2 if input_act == "swiglu" else 1


def apply_input_act(
    x: torch.Tensor,
    input_act: str | None,
    act_weight: torch.Tensor | None = None,
    act_eps: float = 0.0,
) -> torch.Tensor:
    """Apply the pre-quantization activation eagerly.

    Used by backends and shapes the fused quantizer does not cover, so the
    fallback result matches the fused one.
    """
    if input_act in (None, "none"):
        return x
    if input_act == "gelu_tanh":
        return torch.nn.functional.gelu(x, approximate="tanh")
    if input_act == "swiglu":
        gate, up = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate).mul_(up)
    if input_act == "rms_norm":
        if act_weight is None:
            raise ValueError("input_act 'rms_norm' requires act_weight")
        return torch.nn.functional.rms_norm(
            x, (x.shape[-1],), weight=act_weight.to(x.dtype), eps=act_eps)
    raise ValueError(_unsupported(input_act))


def apply_residual(
    out: torch.Tensor,
    residual: torch.Tensor | None,
    residual_scale: torch.Tensor | None,
) -> torch.Tensor:
    """``residual + residual_scale * out`` in out's dtype (addcmul's type
    promotion would otherwise widen it)."""
    if residual is None:
        return out
    if residual_scale is None:
        raise ValueError("residual requires residual_scale")
    return torch.addcmul(residual.to(out.dtype), out, residual_scale.to(out.dtype))


def _unsupported(input_act) -> str:
    known = sorted(k for k in INPUT_ACT_TO_CODE if k is not None)
    return f"unsupported input_act: {input_act!r} (expected one of {known})"
