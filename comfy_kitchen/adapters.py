# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public API of the differentiable weight-adapter ops: bypass in x's dtype, merge in the weight's."""

import functools

import torch

from comfy_kitchen.backends.eager.adapters.common import dispatch
from comfy_kitchen.backends.eager.adapters.decompose import GIVEN, MERGED, WEIGHT

__all__ = [
    "lora_bypass",
    "lora_merge",
    "loha_bypass",
    "loha_merge",
    "lokr_bypass",
    "lokr_merge",
    "oft_apply",
    "weight_decompose",
    "channel_scale",
]


@functools.cache
def _arg_names(name: str) -> tuple[str, ...]:
    return tuple(
        a.name for a in getattr(torch.ops.comfy_kitchen, name).default._schema.arguments
    )


def _call(name: str, *args):
    """The op through torch.library under autograd or torch.compile, else the registry's backend directly."""
    graph = torch.is_grad_enabled() and any(
        isinstance(a, torch.Tensor) and a.requires_grad for a in args
    )
    if graph or torch.compiler.is_compiling():
        return getattr(torch.ops.comfy_kitchen, name)(*args)
    return dispatch(name, **dict(zip(_arg_names(name), args, strict=True)))


def _autocast(device: torch.device, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """tensors cast to the autocast dtype when autocast is on."""
    if not torch.is_autocast_enabled(device.type):
        return tensors
    dtype = torch.get_autocast_dtype(device.type)
    return tuple(t.to(dtype) for t in tensors)


def lora_bypass(
    x: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    scale: float = 1.0,
    base: torch.Tensor | None = None,
) -> torch.Tensor:
    """LoRA bypass: [base +] scale * (x @ down^T) @ up^T.

    Args: x (..., in), up (out, rank), down (rank, in), base (..., out) or None.
    """
    x, up, down = _autocast(x.device, x, up, down)
    return _call("lora_bypass", x, up, down, float(scale), base)


def lora_merge(
    weight: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """LoRA merge: weight + scale * (up @ down); weight is (out, ...) with in flattened behind."""
    up, down = _autocast(weight.device, up, down)
    return _call("lora_merge", weight, up, down, float(scale))


def loha_bypass(
    x: torch.Tensor,
    w1_up: torch.Tensor,
    w1_down: torch.Tensor,
    w2_up: torch.Tensor,
    w2_down: torch.Tensor,
    scale: float = 1.0,
    base: torch.Tensor | None = None,
) -> torch.Tensor:
    """LoHa bypass: [base +] scale * x @ ((w1_up@w1_down) * (w2_up@w2_down))^T."""
    x, w1_up, w1_down, w2_up, w2_down = _autocast(
        x.device, x, w1_up, w1_down, w2_up, w2_down
    )
    return _call("loha_bypass", x, w1_up, w1_down, w2_up, w2_down, float(scale), base)


def loha_merge(
    weight: torch.Tensor,
    w1_up: torch.Tensor,
    w1_down: torch.Tensor,
    w2_up: torch.Tensor,
    w2_down: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """LoHa merge: weight + scale * (w1_up@w1_down) * (w2_up@w2_down)."""
    w1_up, w1_down, w2_up, w2_down = _autocast(
        weight.device, w1_up, w1_down, w2_up, w2_down
    )
    return _call("loha_merge", weight, w1_up, w1_down, w2_up, w2_down, float(scale))


def lokr_bypass(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    scale: float = 1.0,
    base: torch.Tensor | None = None,
) -> torch.Tensor:
    """LoKr bypass: [base +] scale * x @ kron(w1, w2)^T; w1 (a, b), w2 (c, d), layer (a*c, b*d)."""
    x, w1, w2 = _autocast(x.device, x, w1, w2)
    return _call("lokr_bypass", x, w1, w2, float(scale), base)


def lokr_merge(
    weight: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """LoKr merge: weight + scale * kron(w1, w2)."""
    w1, w2 = _autocast(weight.device, w1, w2)
    return _call("lokr_merge", weight, w1, w2, float(scale))


def oft_apply(
    blocks: torch.Tensor,
    x: torch.Tensor,
    rescale: torch.Tensor | None = None,
    constraint: float = 0.0,
    multiplier: float = 1.0,
    shift: bool = False,
    channel_axis: int = -1,
) -> torch.Tensor:
    """OFT: rescale * R'^T x [- x] along channel_axis, R' = multiplier * cayley(skew(blocks)) + (1 - multiplier) * I.

    Args: blocks (k, s, s); constraint caps the skew part's Frobenius norm (0: none);
    channel_axis 0 for a weight, -1 for linear and 1 for conv activations.
    """
    return _call(
        "oft_apply",
        blocks,
        x,
        rescale,
        float(constraint),
        float(multiplier),
        bool(shift),
        int(channel_axis),
    )


def _role(
    t: torch.Tensor | None, weight: torch.Tensor
) -> tuple[int, torch.Tensor | None]:
    """(role, tensor to pass) for a base or norm_weight argument."""
    if t is None:
        return MERGED, None
    if t is weight:
        return WEIGHT, None
    return GIVEN, t


def weight_decompose(
    weight: torch.Tensor,
    dora_scale: torch.Tensor,
    multiplier: float = 1.0,
    on_output: bool = True,
    base: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    delta: torch.Tensor | None = None,
) -> torch.Tensor:
    """DoRA: B + multiplier * (M * d / (||N_row|| + eps) - B) on the merged weight M = weight + delta.

    Args: rows are output rows when on_output else input columns; dora_scale (d) has
    one value per row; base (B) and norm_weight (N) default to M, and passing
    weight itself as either reads it once.
    """
    base_role, base = _role(base, weight)
    norm_role, norm_weight = _role(norm_weight, weight)
    return _call(
        "weight_decompose",
        weight,
        dora_scale,
        float(multiplier),
        bool(on_output),
        base,
        norm_weight,
        delta,
        base_role,
        norm_role,
    )[0]


def channel_scale(
    x: torch.Tensor,
    w: torch.Tensor,
    channel_axis: int = -1,
    alpha: float = 1.0,
    gamma: float = 1.0,
) -> torch.Tensor:
    """IA3: x * (alpha + gamma * w[c]); alpha=1 merges, alpha=0 gives the delta."""
    return _call("channel_scale", x, w, int(channel_axis), float(alpha), float(gamma))
