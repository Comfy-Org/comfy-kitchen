# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gated residual with two gate segments along the tokens.

DiTs that condition reference-image tokens on timestep zero (``index_timestep_zero``)
gate tokens ``[0, split)`` with one gate row and tokens ``[split, L)`` with another.
ComfyUI runs this as per-segment slices updated in place:

- Flux / Flux.2 ``apply_mod`` with ``modulation_dims`` (comfy/ldm/flux/layers.py):
  ``x += apply_mod(out, gate, None, modulation_dims)``.
- Krea2 ``SingleStreamBlock.forward`` (comfy/ldm/krea2/model.py):
  ``attn[:, a:b].mul_(gate)`` per segment, then ``x + attn``.
- Qwen-Image ``_apply_gate`` (comfy/ldm/qwen_image/model.py):
  ``y + cat(x[:, a:b] * gate)``.

The gated residual is the same sequence (``x * gate`` rounded, then added to the
residual) in all three models.

The gate is ``[2, B, D]`` (``[2, 1, D]`` broadcasts over the batch): row 0 for the
tokens before ``split``, row 1 for the rest. The three models produce it as a view:
Flux ``gate.transpose(0, 1)`` of ``[B, 2, D]``, Qwen-Image
``gate.unflatten(0, (2, -1))`` of ``[2B, D]``, Krea2 ``gate.squeeze(1).unflatten(0, (2, -1))``
of ``[2B, 1, D]``.
"""

import torch

from comfy_kitchen.registry import registry


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
