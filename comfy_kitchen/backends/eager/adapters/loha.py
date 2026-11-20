# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Eager LoHa ops and their custom-op definitions: DeltaW = (w1_up @ w1_down) * (w2_up @ w2_down)."""

import torch
from torch import Tensor
from torch.nn import functional

from .common import add_base, bypass_dtype, dispatch, factor_dtype


def _delta(w1_up, w1_down, w2_up, w2_down, dt):
    """DeltaW in dtype dt."""
    return (w1_up.to(dt) @ w1_down.to(dt)) * (w2_up.to(dt) @ w2_down.to(dt))


def _factor_grads(gw, w1_up, w1_down, w2_up, w2_down, dt):
    """Factor gradients of DeltaW given its gradient gw, already scaled."""
    a1, b1, a2, b2 = (t.to(dt) for t in (w1_up, w1_down, w2_up, w2_down))
    e1 = gw * (a2 @ b2)
    e2 = gw * (a1 @ b1)
    return (
        (e1 @ b1.t()).to(w1_up.dtype),
        (a1.t() @ e1).to(w1_down.dtype),
        (e2 @ b2.t()).to(w2_up.dtype),
        (a2.t() @ e2).to(w2_down.dtype),
    )


def loha_bypass(
    x: Tensor,
    w1_up: Tensor,
    w1_down: Tensor,
    w2_up: Tensor,
    w2_down: Tensor,
    scale: float = 1.0,
    base: Tensor | None = None,
) -> Tensor:
    """[base +] scale * x @ DeltaW^T, the scale on w1_up and the base added after the GEMM."""
    dt = x.dtype
    delta = _delta(w1_up.to(dt) * scale, w1_down, w2_up, w2_down, dt)
    y = functional.linear(x, delta)
    if base is not None and base.dtype == dt:
        return y.add_(base)
    return add_base(y, base)


def loha_bypass_backward(
    grad: Tensor,
    x: Tensor,
    w1_up: Tensor,
    w1_down: Tensor,
    w2_up: Tensor,
    w2_down: Tensor,
    scale: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """(grad_x, g_w1_up, g_w1_down, g_w2_up, g_w2_down) for :func:`loha_bypass`."""
    dt = x.dtype
    x2 = x.reshape(-1, x.shape[-1])
    g2 = grad.reshape(-1, grad.shape[-1]).to(dt)
    delta = _delta(w1_up, w1_down, w2_up, w2_down, dt)
    grad_x = ((g2 @ delta) * scale).reshape(x.shape)
    gw = (g2.t() @ x2) * scale
    return (grad_x, *_factor_grads(gw, w1_up, w1_down, w2_up, w2_down, dt))


def loha_merge(
    weight: Tensor,
    w1_up: Tensor,
    w1_down: Tensor,
    w2_up: Tensor,
    w2_down: Tensor,
    scale: float = 1.0,
) -> Tensor:
    """weight + scale * DeltaW, in the weight's dtype."""
    dt = factor_dtype(w1_up, w1_down, w2_up, w2_down)
    delta = _delta(w1_up, w1_down, w2_up, w2_down, dt)
    return (weight + scale * delta.reshape(weight.shape)).to(weight.dtype)


def loha_merge_backward(
    grad: Tensor,
    w1_up: Tensor,
    w1_down: Tensor,
    w2_up: Tensor,
    w2_down: Tensor,
    scale: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """(g_w1_up, g_w1_down, g_w2_up, g_w2_down) for :func:`loha_merge`."""
    dt = factor_dtype(w1_up, w1_down, w2_up, w2_down)
    gw = grad.reshape(w1_up.shape[0], -1).to(dt) * scale
    return _factor_grads(gw, w1_up, w1_down, w2_up, w2_down, dt)


# torch.library Custom Op Definitions


@torch.library.custom_op("comfy_kitchen::loha_bypass", mutates_args=())
def _op_loha_bypass(
    x: torch.Tensor,
    w1_up: torch.Tensor,
    w1_down: torch.Tensor,
    w2_up: torch.Tensor,
    w2_down: torch.Tensor,
    scale: float,
    base: torch.Tensor | None,
) -> torch.Tensor:
    return dispatch(
        "loha_bypass",
        x=x,
        w1_up=w1_up,
        w1_down=w1_down,
        w2_up=w2_up,
        w2_down=w2_down,
        scale=scale,
        base=base,
    )


@_op_loha_bypass.register_fake
def _op_loha_bypass_fake(x, w1_up, w1_down, w2_up, w2_down, scale, base):
    return x.new_empty((*x.shape[:-1], w1_up.shape[0]), dtype=bypass_dtype(x, base))


@torch.library.custom_op("comfy_kitchen::loha_bypass_backward", mutates_args=())
def _op_loha_bypass_backward(
    grad: torch.Tensor,
    x: torch.Tensor,
    w1_up: torch.Tensor,
    w1_down: torch.Tensor,
    w2_up: torch.Tensor,
    w2_down: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return dispatch(
        "loha_bypass_backward",
        grad=grad,
        x=x,
        w1_up=w1_up,
        w1_down=w1_down,
        w2_up=w2_up,
        w2_down=w2_down,
        scale=scale,
    )


@_op_loha_bypass_backward.register_fake
def _op_loha_bypass_backward_fake(grad, x, w1_up, w1_down, w2_up, w2_down, scale):
    return (
        torch.empty_like(x),
        torch.empty_like(w1_up),
        torch.empty_like(w1_down),
        torch.empty_like(w2_up),
        torch.empty_like(w2_down),
    )


def _bypass_setup(ctx, inputs, output):
    x, w1_up, w1_down, w2_up, w2_down, scale, base = inputs
    ctx.save_for_backward(x, w1_up, w1_down, w2_up, w2_down)
    ctx.scale = scale
    ctx.has_base = base is not None


def _bypass_backward(ctx, grad):
    grads = torch.ops.comfy_kitchen.loha_bypass_backward(
        grad.contiguous(), *ctx.saved_tensors, ctx.scale
    )
    return (*grads, None, grad if ctx.has_base else None)


_op_loha_bypass.register_autograd(_bypass_backward, setup_context=_bypass_setup)


@torch.library.custom_op("comfy_kitchen::loha_merge", mutates_args=())
def _op_loha_merge(
    weight: torch.Tensor,
    w1_up: torch.Tensor,
    w1_down: torch.Tensor,
    w2_up: torch.Tensor,
    w2_down: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    return dispatch(
        "loha_merge",
        weight=weight,
        w1_up=w1_up,
        w1_down=w1_down,
        w2_up=w2_up,
        w2_down=w2_down,
        scale=scale,
    )


@_op_loha_merge.register_fake
def _op_loha_merge_fake(weight, w1_up, w1_down, w2_up, w2_down, scale):
    return torch.empty_like(weight)


@torch.library.custom_op("comfy_kitchen::loha_merge_backward", mutates_args=())
def _op_loha_merge_backward(
    grad: torch.Tensor,
    w1_up: torch.Tensor,
    w1_down: torch.Tensor,
    w2_up: torch.Tensor,
    w2_down: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return dispatch(
        "loha_merge_backward",
        grad=grad,
        w1_up=w1_up,
        w1_down=w1_down,
        w2_up=w2_up,
        w2_down=w2_down,
        scale=scale,
    )


@_op_loha_merge_backward.register_fake
def _op_loha_merge_backward_fake(grad, w1_up, w1_down, w2_up, w2_down, scale):
    return (
        torch.empty_like(w1_up),
        torch.empty_like(w1_down),
        torch.empty_like(w2_up),
        torch.empty_like(w2_down),
    )


def _merge_setup(ctx, inputs, output):
    _weight, w1_up, w1_down, w2_up, w2_down, scale = inputs
    ctx.save_for_backward(w1_up, w1_down, w2_up, w2_down)
    ctx.scale = scale


def _merge_backward(ctx, grad):
    grads = torch.ops.comfy_kitchen.loha_merge_backward(
        grad.contiguous(), *ctx.saved_tensors, ctx.scale
    )
    # The base weight passes its gradient straight through the sum.
    return (grad, *grads, None)


_op_loha_merge.register_autograd(_merge_backward, setup_context=_merge_setup)
