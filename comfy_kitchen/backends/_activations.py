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

``situ`` is MOSS-Image's soft-capped gated pair on ``[hidden | gate]`` halves:
``4*tanh(gate/4)*sigmoid(gate) * 25*tanh(hidden/25)``, also halving the row.

``rms_norm`` is the row-wise ``x * rsqrt(mean(x^2) + eps) * w``; it carries a
K-element weight and an eps alongside the code.

``rms_norm_modulated`` evaluates RMSNorm and ``norm * (1 + scale) + shift``
in FP32. Its weight is a packed FP32 [3, K] tensor: norm weight, scale, shift.
There is no intermediate cast back to the input dtype before rotation.
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
    "situ": 4,
    "rms_norm_modulated": 5,
}


def input_act_code(input_act: str | None) -> int:
    """Kernel code for `input_act`, rejecting anything unsupported."""
    try:
        return INPUT_ACT_TO_CODE[input_act]
    except KeyError:
        raise ValueError(_unsupported(input_act)) from None


def input_act_width(input_act: str | None) -> int:
    """How many input columns produce one activated column (1, or 2 for the gated pairs)."""
    return 2 if input_act in ("swiglu", "situ") else 1


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
    if input_act == "situ":
        # 4*tanh(gate/4)*sigmoid(gate) * 25*tanh(hidden/25), evaluated in the
        # same order and dtype as MOSS-Image's module; in place to avoid a
        # fresh M x K temporary per op.
        hidden, gate = x.chunk(2, dim=-1)
        gated = gate.div(4.0).tanh_().mul_(4.0).mul_(torch.sigmoid(gate))
        return hidden.div(25.0).tanh_().mul_(25.0).mul_(gated)
    if input_act == "rms_norm":
        if act_weight is None:
            raise ValueError("input_act 'rms_norm' requires act_weight")
        return torch.nn.functional.rms_norm(
            x, (x.shape[-1],), weight=act_weight.to(x.dtype), eps=act_eps)
    if input_act == "rms_norm_modulated":
        if act_weight is None or act_weight.shape != (3, x.shape[-1]) or act_weight.dtype != torch.float32:
            raise ValueError("rms_norm_modulated requires FP32 act_weight with shape [3, K]")
        weight, scale, shift = act_weight.unbind(0)
        normalized = torch.nn.functional.rms_norm(x.float(), (x.shape[-1],), weight=weight, eps=act_eps)
        return normalized * (1 + scale) + shift
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
