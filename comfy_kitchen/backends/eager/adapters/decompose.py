# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Eager DoRA out = B + m * (M * d / (||N_row|| + eps) - B), M = W + delta, and IA3 channel scale, with their custom-op definitions."""

import torch
from torch import Tensor

from .common import absent, dispatch, present

# What base (B) and norm_weight (N) are: the merged weight M, the weight W itself, or a given tensor.
MERGED, WEIGHT, GIVEN = 0, 1, 2


def role_tensor(
    role: int, given: Tensor | None, weight: Tensor, merged: Tensor
) -> Tensor:
    """The tensor a base or norm role names."""
    if role == MERGED:
        return merged
    return weight if role == WEIGHT else given


def norm_view(v: Tensor, ndim: int, on_output: bool) -> Tensor:
    """A per-row vector shaped to broadcast against the weight."""
    if on_output:
        return v.reshape(-1, *([1] * (ndim - 1)))
    return v.reshape(1, -1, *([1] * (ndim - 2)))


def row_norms(w: Tensor, on_output: bool) -> Tensor:
    """fp32 L2 norm per output row, or per input column."""
    axis = 0 if on_output else 1
    dims = tuple(i for i in range(w.dim()) if i != axis)
    return torch.linalg.vector_norm(w, dim=dims, dtype=torch.float32)


def weight_decompose(
    weight: Tensor,
    dora_scale: Tensor,
    multiplier: float = 1.0,
    on_output: bool = True,
    base: Tensor | None = None,
    norm_weight: Tensor | None = None,
    delta: Tensor | None = None,
    base_role: int = MERGED,
    norm_role: int = MERGED,
) -> tuple[Tensor, Tensor]:
    """-> (B + m * (M * d / n - B) in W's dtype, the fp32 norms n); M = W + delta."""
    eps = torch.finfo(weight.dtype).eps
    merged = weight if delta is None else weight.float() + delta.float()
    src = role_tensor(norm_role, norm_weight, weight, merged)
    norms = row_norms(src, on_output) + eps
    coef = norm_view(
        multiplier * dora_scale.reshape(-1).float() / norms, weight.dim(), on_output
    )
    b = role_tensor(base_role, base, weight, merged)

    # out = M * coef' in one pass when B is M or drops out (m = 1).
    if b is merged or multiplier == 1.0:
        if b is merged:
            coef = coef + (1.0 - multiplier)
        return torch.mul(merged, coef, out=torch.empty_like(weight)), norms

    out = merged * coef
    out.add_(b, alpha=1.0 - multiplier)
    return out.to(weight.dtype), norms


def weight_decompose_backward(
    grad: Tensor,
    weight: Tensor,
    dora_scale: Tensor,
    norms: Tensor,
    multiplier: float = 1.0,
    on_output: bool = True,
    base: Tensor | None = None,
    norm_weight: Tensor | None = None,
    delta: Tensor | None = None,
    base_role: int = MERGED,
    norm_role: int = MERGED,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """(g_weight, g_dora_scale, g_base, g_norm_weight, g_delta); empty for an input not given."""
    w = weight.float()
    merged = w if delta is None else w + delta.float()
    g = grad.reshape(weight.shape).float()
    nd = w.dim()
    axis = 0 if on_output else 1
    dims = tuple(i for i in range(nd) if i != axis)
    d = dora_scale.reshape(-1).float()

    # Per row: g_d, and dn = gradient into the norm source (0 for a zero-norm row).
    rowdot = (g * merged).sum(dims)
    raw = norms - torch.finfo(weight.dtype).eps
    dn = -multiplier * rowdot * d / (norms * norms)
    dn = torch.where(raw > 0, dn / raw.clamp(min=1e-30), torch.zeros_like(dn))
    gd = (multiplier * rowdot / norms).reshape(dora_scale.shape).to(dora_scale.dtype)

    # Gradients through M directly, through B, and through N.
    g_m = g * norm_view(multiplier * d / norms, nd, on_output)
    g_b = (1.0 - multiplier) * g
    src = role_tensor(norm_role, norm_weight, w, merged).float()
    g_n = src * norm_view(dn, nd, on_output)

    # Route each into the tensor its role names; W also receives everything M does.
    if base_role == MERGED:
        g_m = g_m + g_b
    if norm_role == MERGED:
        g_m = g_m + g_n
    g_w = g_m
    if base_role == WEIGHT:
        g_w = g_w + g_b
    if norm_role == WEIGHT:
        g_w = g_w + g_n

    gb = g_b.to(base.dtype) if base_role == GIVEN else absent(weight)
    gn = g_n.to(norm_weight.dtype) if norm_role == GIVEN else absent(weight)
    gdelta = g_m.to(delta.dtype) if delta is not None else absent(weight)
    return g_w.to(weight.dtype), gd, gb, gn, gdelta


def _channel_view(x: Tensor, channel_axis: int) -> list[int]:
    shape = [1] * x.dim()
    shape[channel_axis] = -1
    return shape


def channel_scale(
    x: Tensor,
    w: Tensor,
    channel_axis: int = -1,
    alpha: float = 1.0,
    gamma: float = 1.0,
) -> Tensor:
    """x * (alpha + gamma * w[c])"""
    scale = alpha + gamma * w.reshape(_channel_view(x, channel_axis)).float()
    return torch.mul(x, scale, out=torch.empty_like(x))


def channel_scale_backward(
    grad: Tensor,
    x: Tensor,
    w: Tensor,
    channel_axis: int = -1,
    alpha: float = 1.0,
    gamma: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """(grad_x, grad_w) for :func:`channel_scale`."""
    view = _channel_view(x, channel_axis)
    gx = (grad.float() * (alpha + gamma * w.reshape(view).float())).to(x.dtype)
    dims = [d for d in range(x.dim()) if d != channel_axis % x.dim()]
    gw = (grad.float() * x.float()).sum(dims) * gamma
    return gx, gw.reshape(w.shape).to(w.dtype)


# torch.library Custom Op Definitions


@torch.library.custom_op("comfy_kitchen::weight_decompose", mutates_args=())
def _op_weight_decompose(
    weight: torch.Tensor,
    dora_scale: torch.Tensor,
    multiplier: float,
    on_output: bool,
    base: torch.Tensor | None,
    norm_weight: torch.Tensor | None,
    delta: torch.Tensor | None,
    base_role: int,
    norm_role: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return dispatch(
        "weight_decompose",
        weight=weight,
        dora_scale=dora_scale,
        multiplier=multiplier,
        on_output=on_output,
        base=base,
        norm_weight=norm_weight,
        delta=delta,
        base_role=base_role,
        norm_role=norm_role,
    )


@_op_weight_decompose.register_fake
def _op_weight_decompose_fake(
    weight,
    dora_scale,
    multiplier,
    on_output,
    base,
    norm_weight,
    delta,
    base_role,
    norm_role,
):
    rows = weight.shape[0] if on_output else weight.shape[1]
    return torch.empty_like(weight), weight.new_empty(rows, dtype=torch.float32)


@torch.library.custom_op("comfy_kitchen::weight_decompose_backward", mutates_args=())
def _op_weight_decompose_backward(
    grad: torch.Tensor,
    weight: torch.Tensor,
    dora_scale: torch.Tensor,
    norms: torch.Tensor,
    multiplier: float,
    on_output: bool,
    base: torch.Tensor | None,
    norm_weight: torch.Tensor | None,
    delta: torch.Tensor | None,
    base_role: int,
    norm_role: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return dispatch(
        "weight_decompose_backward",
        grad=grad,
        weight=weight,
        dora_scale=dora_scale,
        norms=norms,
        multiplier=multiplier,
        on_output=on_output,
        base=base,
        norm_weight=norm_weight,
        delta=delta,
        base_role=base_role,
        norm_role=norm_role,
    )


@_op_weight_decompose_backward.register_fake
def _op_weight_decompose_backward_fake(
    grad,
    weight,
    dora_scale,
    norms,
    multiplier,
    on_output,
    base,
    norm_weight,
    delta,
    base_role,
    norm_role,
):
    return (
        torch.empty_like(weight),
        torch.empty_like(dora_scale),
        weight.new_empty(0) if base is None else torch.empty_like(base),
        weight.new_empty(0) if norm_weight is None else torch.empty_like(norm_weight),
        weight.new_empty(0) if delta is None else torch.empty_like(delta),
    )


def _decompose_setup(ctx, inputs, output):
    (
        weight,
        dora_scale,
        multiplier,
        on_output,
        base,
        norm_weight,
        delta,
        b_role,
        n_role,
    ) = inputs
    ctx.save_for_backward(weight, dora_scale, output[1], base, norm_weight, delta)
    ctx.args = (multiplier, on_output, b_role, n_role)


def _decompose_backward(ctx, grad, grad_norms):
    weight, dora_scale, norms, base, norm_weight, delta = ctx.saved_tensors
    multiplier, on_output, b_role, n_role = ctx.args
    gw, gd, gb, gn, gdelta = torch.ops.comfy_kitchen.weight_decompose_backward(
        grad.contiguous(),
        weight,
        dora_scale,
        norms,
        multiplier,
        on_output,
        base,
        norm_weight,
        delta,
        b_role,
        n_role,
    )
    return gw, gd, None, None, present(gb), present(gn), present(gdelta), None, None


_op_weight_decompose.register_autograd(
    _decompose_backward, setup_context=_decompose_setup
)


@torch.library.custom_op("comfy_kitchen::channel_scale", mutates_args=())
def _op_channel_scale(
    x: torch.Tensor,
    w: torch.Tensor,
    channel_axis: int,
    alpha: float,
    gamma: float,
) -> torch.Tensor:
    return dispatch(
        "channel_scale", x=x, w=w, channel_axis=channel_axis, alpha=alpha, gamma=gamma
    )


@_op_channel_scale.register_fake
def _op_channel_scale_fake(x, w, channel_axis, alpha, gamma):
    return torch.empty_like(x)


@torch.library.custom_op("comfy_kitchen::channel_scale_backward", mutates_args=())
def _op_channel_scale_backward(
    grad: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    channel_axis: int,
    alpha: float,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return dispatch(
        "channel_scale_backward",
        grad=grad,
        x=x,
        w=w,
        channel_axis=channel_axis,
        alpha=alpha,
        gamma=gamma,
    )


@_op_channel_scale_backward.register_fake
def _op_channel_scale_backward_fake(grad, x, w, channel_axis, alpha, gamma):
    return torch.empty_like(x), torch.empty_like(w)


def _scale_setup(ctx, inputs, output):
    x, w, channel_axis, alpha, gamma = inputs
    ctx.save_for_backward(x, w)
    ctx.args = (channel_axis, alpha, gamma)


def _scale_backward(ctx, grad):
    x, w = ctx.saved_tensors
    gx, gw = torch.ops.comfy_kitchen.channel_scale_backward(
        grad.contiguous(), x, w, *ctx.args
    )
    return gx, gw, None, None, None


_op_channel_scale.register_autograd(_scale_backward, setup_context=_scale_setup)
