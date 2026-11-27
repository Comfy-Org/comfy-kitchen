# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Eager LoRA ops and their custom-op definitions: bypass in x's dtype, merge in the factors' widest dtype."""

import torch
from torch import Tensor
from torch.nn import functional

from .common import add_base, bypass_dtype, dispatch, factor_dtype


def lora_bypass(
    x: Tensor,
    up: Tensor,
    down: Tensor,
    scale: float = 1.0,
    base: Tensor | None = None,
) -> Tensor:
    """[base +] scale * (x @ down^T) @ up^T, the scale on up and a same-dtype base as the GEMM addend."""
    dt = x.dtype
    h = functional.linear(x, down.to(dt))
    u = up.to(dt) * scale if scale != 1.0 else up.to(dt)
    if base is not None and base.dtype == dt:
        o = up.shape[0]
        y = torch.addmm(base.reshape(-1, o), h.reshape(-1, h.shape[-1]), u.t())
        return y.reshape(base.shape)
    return add_base(functional.linear(h, u), base)


def lora_bypass_backward(
    grad: Tensor, x: Tensor, up: Tensor, down: Tensor, scale: float = 1.0
) -> tuple[Tensor, Tensor, Tensor]:
    """(grad_x, grad_up, grad_down) for :func:`lora_bypass`."""
    dt = x.dtype
    x2 = x.reshape(-1, x.shape[-1])
    g2 = grad.reshape(-1, grad.shape[-1]).to(dt)
    u, d = up.to(dt), down.to(dt)
    h = x2 @ d.t()
    q = g2 @ u
    grad_x = ((q @ d) * scale).reshape(x.shape)
    grad_up = (g2.t() @ h) * scale
    grad_down = (q.t() @ x2) * scale
    return grad_x, grad_up.to(up.dtype), grad_down.to(down.dtype)


def lora_merge(weight: Tensor, up: Tensor, down: Tensor, scale: float = 1.0) -> Tensor:
    """weight + scale * (up @ down), in the weight's dtype."""
    dt = factor_dtype(up, down)
    delta = up.to(dt) @ down.to(dt)
    return (weight + scale * delta.reshape(weight.shape)).to(weight.dtype)


def lora_merge_backward(
    grad: Tensor, up: Tensor, down: Tensor, scale: float = 1.0
) -> tuple[Tensor, Tensor]:
    """(grad_up, grad_down) for :func:`lora_merge`; grad_weight is grad itself."""
    dt = factor_dtype(up, down)
    g2 = grad.reshape(up.shape[0], -1).to(dt)
    u, d = up.to(dt), down.to(dt)
    return (
        ((g2 @ d.t()) * scale).to(up.dtype),
        ((u.t() @ g2) * scale).to(down.dtype),
    )


# torch.library Custom Op Definitions


@torch.library.custom_op("comfy_kitchen::lora_bypass", mutates_args=())
def _op_lora_bypass(
    x: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    scale: float,
    base: torch.Tensor | None,
) -> torch.Tensor:
    return dispatch("lora_bypass", x=x, up=up, down=down, scale=scale, base=base)


@_op_lora_bypass.register_fake
def _op_lora_bypass_fake(x, up, down, scale, base):
    return x.new_empty((*x.shape[:-1], up.shape[0]), dtype=bypass_dtype(x, base))


@torch.library.custom_op("comfy_kitchen::lora_bypass_backward", mutates_args=())
def _op_lora_bypass_backward(
    grad: torch.Tensor,
    x: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return dispatch(
        "lora_bypass_backward", grad=grad, x=x, up=up, down=down, scale=scale
    )


@_op_lora_bypass_backward.register_fake
def _op_lora_bypass_backward_fake(grad, x, up, down, scale):
    return torch.empty_like(x), torch.empty_like(up), torch.empty_like(down)


def _bypass_setup(ctx, inputs, output):
    x, up, down, scale, base = inputs
    ctx.save_for_backward(x, up, down)
    ctx.scale = scale
    ctx.has_base = base is not None


def _bypass_backward(ctx, grad):
    x, up, down = ctx.saved_tensors
    gx, gu, gd = torch.ops.comfy_kitchen.lora_bypass_backward(
        grad.contiguous(), x, up, down, ctx.scale
    )
    # A fused base passes its gradient straight through the sum.
    return gx, gu, gd, None, grad if ctx.has_base else None


_op_lora_bypass.register_autograd(_bypass_backward, setup_context=_bypass_setup)


@torch.library.custom_op("comfy_kitchen::lora_merge", mutates_args=())
def _op_lora_merge(
    weight: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    return dispatch("lora_merge", weight=weight, up=up, down=down, scale=scale)


@_op_lora_merge.register_fake
def _op_lora_merge_fake(weight, up, down, scale):
    return torch.empty_like(weight)


@torch.library.custom_op("comfy_kitchen::lora_merge_backward", mutates_args=())
def _op_lora_merge_backward(
    grad: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return dispatch("lora_merge_backward", grad=grad, up=up, down=down, scale=scale)


@_op_lora_merge_backward.register_fake
def _op_lora_merge_backward_fake(grad, up, down, scale):
    return torch.empty_like(up), torch.empty_like(down)


def _merge_setup(ctx, inputs, output):
    _weight, up, down, scale = inputs
    ctx.save_for_backward(up, down)
    ctx.scale = scale


def _merge_backward(ctx, grad):
    up, down = ctx.saved_tensors
    gu, gd = torch.ops.comfy_kitchen.lora_merge_backward(
        grad.contiguous(), up, down, ctx.scale
    )
    # The base weight passes its gradient straight through the sum.
    return grad, gu, gd, None


_op_lora_merge.register_autograd(_merge_backward, setup_context=_merge_setup)
