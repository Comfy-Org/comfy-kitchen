# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AdaLN modulation and gated residual with two modulation segments along the tokens.

DiTs that condition reference-image tokens on timestep zero (``index_timestep_zero``)
modulate tokens ``[0, split)`` with one set of scale / shift / gate rows and tokens
``[split, L)`` with another. ComfyUI runs this as a norm followed by per-segment slices
updated in place:

- Flux / Flux.2 ``apply_mod`` with ``modulation_dims`` (comfy/ldm/flux/layers.py):
  ``t = layer_norm(x); t[:, a:b] *= 1 + scale[:, i]; t[:, a:b] += shift[:, i]`` and
  ``x += apply_mod(out, gate, None, modulation_dims)``.
- Krea2 ``SingleStreamBlock.forward`` (comfy/ldm/krea2/model.py): fp32 ``RMSNorm`` with
  weight ``scale + 1``, rounded to x's dtype, then ``mul_(1 + scale).add_(shift)`` per
  segment; ``attn[:, a:b].mul_(gate)`` per segment, then ``x + attn``.
- Qwen-Image ``_modulate`` / ``_apply_gate`` (comfy/ldm/qwen_image/model.py):
  ``addcmul(shift, layer_norm(x)[:, a:b], 1 + scale)`` per segment, ``torch.cat``, and
  ``y + cat(x[:, a:b] * gate)``.

The reference below is the Flux / Krea2 order: the product ``t * (1 + scale)`` is rounded
to x's dtype before the shift is added. Qwen-Image's ``addcmul`` rounds once, so it can
differ from the reference by one rounding step. The gated residual is the same sequence
(``x * gate`` rounded, then added to the residual) in all three models.

The modulation is ``[2, B, D]`` (``[2, 1, D]`` broadcasts over the batch): row 0 for the
tokens before ``split``, row 1 for the rest. The three models produce it as a view:
Flux ``scale.transpose(0, 1)`` of ``[B, 2, D]``, Qwen-Image and Krea2
``scale.unflatten(0, (2, -1))`` of ``[2B, D]``.
"""

import torch
from torch.nn import functional

from comfy_kitchen.registry import registry


def _norm(x, eps, rms_weight, rms_weight_offset):
    if rms_weight is None:
        return functional.layer_norm(x, x.shape[-1:], eps=eps)
    # Krea2 RMSNorm: fp32 norm with an fp32 weight, rounded to x's dtype after the norm
    weight = rms_weight.to(torch.float32) + rms_weight_offset
    return functional.rms_norm(x.float(), x.shape[-1:], weight=weight, eps=eps).to(x.dtype)


def adaln_segmented(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    split: int,
    eps: float = 1e-6,
    rms_weight: torch.Tensor | None = None,
    rms_weight_offset: float = 0.0,
) -> torch.Tensor:
    """Reference: the norm, then ``*= 1 + scale[i]`` and ``+= shift[i]`` on each segment."""
    out = _norm(x, eps, rms_weight, rms_weight_offset)
    for i, seg in enumerate((slice(0, split), slice(split, x.shape[1]))):
        out[:, seg].mul_(1 + scale[i].unsqueeze(1)).add_(shift[i].unsqueeze(1))
    return out


def gated_residual_segmented_(
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor,
    split: int,
) -> None:
    """Reference: ``residual[:, seg] += x[:, seg] * gate[i]`` on each segment."""
    for i, seg in enumerate((slice(0, split), slice(split, x.shape[1]))):
        residual[:, seg] += x[:, seg] * gate[i].unsqueeze(1)


def gated_residual_segmented(
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor,
    split: int,
) -> torch.Tensor:
    """Reference: ``residual + cat(x[:, seg] * gate[i])``, as a new tensor."""
    out = residual.clone(memory_format=torch.contiguous_format)
    gated_residual_segmented_(out, x, gate, split)
    return out


def _check_modulation(name, t, x):
    batch, _, dim = x.shape
    if t.dim() != 3 or t.shape[0] != 2 or t.shape[1] not in (1, batch) or t.shape[2] != dim:
        raise ValueError(
            f"{name} must be [2, {batch}, {dim}] or [2, 1, {dim}], got {tuple(t.shape)}"
        )


def _check(op, x, split, **modulation):
    if x.dim() != 3:
        raise ValueError(f"{op}: x must be [B, L, D], got {tuple(x.shape)}")
    if not 0 <= split <= x.shape[1]:
        raise ValueError(f"{op}: split must be in [0, {x.shape[1]}], got {split}")
    for name, t in modulation.items():
        _check_modulation(f"{op}: {name}", t, x)


@torch.library.custom_op("comfy_kitchen::adaln_segmented", mutates_args=())
def _op_adaln_segmented(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    split: int,
    eps: float,
    rms_weight: torch.Tensor | None,
    rms_weight_offset: float,
) -> torch.Tensor:
    _check("adaln_segmented", x, split, scale=scale, shift=shift)
    if rms_weight is not None and rms_weight.numel() != x.shape[-1]:
        raise ValueError(
            f"adaln_segmented: rms_weight needs {x.shape[-1]} elements, got {rms_weight.numel()}"
        )
    kwargs = {
        "x": x,
        "scale": scale,
        "shift": shift,
        "split": split,
        "eps": eps,
        "rms_weight": rms_weight,
        "rms_weight_offset": rms_weight_offset,
    }
    impl = registry.get_implementation("adaln_segmented", kwargs=kwargs)
    return impl(**kwargs)


@_op_adaln_segmented.register_fake
def _op_adaln_segmented_fake(x, scale, shift, split, eps, rms_weight, rms_weight_offset):
    return torch.empty_like(x, memory_format=torch.contiguous_format)


@torch.library.custom_op("comfy_kitchen::gated_residual_segmented", mutates_args=())
def _op_gated_residual_segmented(
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor,
    split: int,
) -> torch.Tensor:
    _check("gated_residual_segmented", x, split, gate=gate)
    if residual.shape != x.shape:
        raise ValueError("gated_residual_segmented: residual and x must have the same shape")
    kwargs = {"residual": residual, "x": x, "gate": gate, "split": split}
    impl = registry.get_implementation("gated_residual_segmented", kwargs=kwargs)
    return impl(**kwargs)


@_op_gated_residual_segmented.register_fake
def _op_gated_residual_segmented_fake(residual, x, gate, split):
    return torch.empty_like(residual, memory_format=torch.contiguous_format)


@torch.library.custom_op("comfy_kitchen::gated_residual_segmented_", mutates_args={"residual"})
def _op_gated_residual_segmented_(
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor,
    split: int,
) -> None:
    _check("gated_residual_segmented_", x, split, gate=gate)
    if residual.shape != x.shape:
        raise ValueError("gated_residual_segmented_: residual and x must have the same shape")
    kwargs = {"residual": residual, "x": x, "gate": gate, "split": split}
    registry.get_implementation("gated_residual_segmented_", kwargs=kwargs)(**kwargs)


@_op_gated_residual_segmented_.register_fake
def _op_gated_residual_segmented_fake_(residual, x, gate, split):
    return None
