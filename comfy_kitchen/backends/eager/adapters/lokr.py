# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Eager LoKr ops and their custom-op definitions: DeltaW = kron(w1, w2), bypass as vec(w1 @ X_t @ w2^T)."""

import torch
from torch import Tensor

from .common import add_base, bypass_dtype, dispatch, factor_dtype


def lokr_bypass(
    x: Tensor,
    w1: Tensor,
    w2: Tensor,
    scale: float = 1.0,
    base: Tensor | None = None,
) -> Tensor:
    """[base +] scale * x @ kron(w1, w2)^T without building the product, the scale on w1."""
    dt = x.dtype
    a, b = w1.shape
    c = w2.shape[0]
    lead = x.shape[:-1]
    x3 = x.reshape(-1, b, x.shape[-1] // b)
    mid = x3 @ w2.to(dt).t()  # (t, b, c)
    u = w1.to(dt) * scale if scale != 1.0 else w1.to(dt)
    if base is not None and base.dtype == dt:
        t = x3.shape[0]
        out = torch.baddbmm(base.reshape(t, a, c), u.expand(t, a, b), mid)
        return out.reshape(base.shape)
    return add_base(torch.matmul(u, mid).reshape(*lead, a * c), base)


def lokr_bypass_backward(
    grad: Tensor, x: Tensor, w1: Tensor, w2: Tensor, scale: float = 1.0
) -> tuple[Tensor, Tensor, Tensor]:
    """(grad_x, grad_w1, grad_w2) for :func:`lokr_bypass`."""
    dt = x.dtype
    a, b = w1.shape
    c, d = w2.shape
    u, v = w1.to(dt), w2.to(dt)
    x3 = x.reshape(-1, b, d)
    g3 = grad.reshape(-1, a, c).to(dt)
    mid = x3 @ v.t()  # (t, b, c)
    n = torch.matmul(u.t(), g3)  # (t, b, c)
    grad_x = ((n @ v) * scale).reshape(x.shape)
    grad_w1 = torch.einsum("tac,tbc->ab", g3, mid) * scale
    grad_w2 = torch.einsum("tbc,tbd->cd", n, x3) * scale
    return grad_x, grad_w1.to(w1.dtype), grad_w2.to(w2.dtype)


def lokr_merge(weight: Tensor, w1: Tensor, w2: Tensor, scale: float = 1.0) -> Tensor:
    """weight + scale * kron(w1, w2), in the weight's dtype."""
    dt = factor_dtype(w1, w2)
    delta = torch.kron(w1.to(dt), w2.to(dt))
    return (weight + scale * delta.reshape(weight.shape)).to(weight.dtype)


def lokr_merge_backward(
    grad: Tensor, w1: Tensor, w2: Tensor, scale: float = 1.0
) -> tuple[Tensor, Tensor]:
    """(grad_w1, grad_w2) for :func:`lokr_merge`."""
    dt = factor_dtype(w1, w2)
    a, b = w1.shape
    c, d = w2.shape
    g4 = grad.reshape(a, c, b, d).to(dt) * scale
    return (
        torch.einsum("acbd,cd->ab", g4, w2.to(dt)).to(w1.dtype),
        torch.einsum("acbd,ab->cd", g4, w1.to(dt)).to(w2.dtype),
    )


# torch.library Custom Op Definitions


@torch.library.custom_op("comfy_kitchen::lokr_bypass", mutates_args=())
def _op_lokr_bypass(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    scale: float,
    base: torch.Tensor | None,
) -> torch.Tensor:
    return dispatch("lokr_bypass", x=x, w1=w1, w2=w2, scale=scale, base=base)


@_op_lokr_bypass.register_fake
def _op_lokr_bypass_fake(x, w1, w2, scale, base):
    return x.new_empty(
        (*x.shape[:-1], w1.shape[0] * w2.shape[0]), dtype=bypass_dtype(x, base)
    )


@torch.library.custom_op("comfy_kitchen::lokr_bypass_backward", mutates_args=())
def _op_lokr_bypass_backward(
    grad: torch.Tensor,
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return dispatch("lokr_bypass_backward", grad=grad, x=x, w1=w1, w2=w2, scale=scale)


@_op_lokr_bypass_backward.register_fake
def _op_lokr_bypass_backward_fake(grad, x, w1, w2, scale):
    return torch.empty_like(x), torch.empty_like(w1), torch.empty_like(w2)


def _bypass_setup(ctx, inputs, output):
    x, w1, w2, scale, base = inputs
    ctx.save_for_backward(x, w1, w2)
    ctx.scale = scale
    ctx.has_base = base is not None


def _bypass_backward(ctx, grad):
    gx, gw1, gw2 = torch.ops.comfy_kitchen.lokr_bypass_backward(
        grad.contiguous(), *ctx.saved_tensors, ctx.scale
    )
    return gx, gw1, gw2, None, grad if ctx.has_base else None


_op_lokr_bypass.register_autograd(_bypass_backward, setup_context=_bypass_setup)


@torch.library.custom_op("comfy_kitchen::lokr_merge", mutates_args=())
def _op_lokr_merge(
    weight: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, scale: float
) -> torch.Tensor:
    return dispatch("lokr_merge", weight=weight, w1=w1, w2=w2, scale=scale)


@_op_lokr_merge.register_fake
def _op_lokr_merge_fake(weight, w1, w2, scale):
    return torch.empty_like(weight)


@torch.library.custom_op("comfy_kitchen::lokr_merge_backward", mutates_args=())
def _op_lokr_merge_backward(
    grad: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    return dispatch("lokr_merge_backward", grad=grad, w1=w1, w2=w2, scale=scale)


@_op_lokr_merge_backward.register_fake
def _op_lokr_merge_backward_fake(grad, w1, w2, scale):
    return torch.empty_like(w1), torch.empty_like(w2)


def _merge_setup(ctx, inputs, output):
    _weight, w1, w2, scale = inputs
    ctx.save_for_backward(w1, w2)
    ctx.scale = scale


def _merge_backward(ctx, grad):
    gw1, gw2 = torch.ops.comfy_kitchen.lokr_merge_backward(
        grad.contiguous(), *ctx.saved_tensors, ctx.scale
    )
    return grad, gw1, gw2, None


_op_lokr_merge.register_autograd(_merge_backward, setup_context=_merge_setup)
