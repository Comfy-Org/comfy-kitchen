# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Eager OFT ops and their custom-op definitions: out = rescale * R'^T x [- x] along channel_axis."""

import torch
from torch import Tensor

from .common import absent, dispatch, present

_NORM_EPS = 1e-8


def skew(
    blocks: Tensor, constraint: float
) -> tuple[Tensor, Tensor | float, Tensor | None]:
    """(q0, c, q_norm) in fp32: skew part q = c*q0, its norm capped at constraint over all blocks."""
    b = blocks.float()
    q0 = b - b.transpose(-1, -2)
    if constraint <= 0:
        return q0, 1.0, None
    q_norm = torch.linalg.vector_norm(q0) + _NORM_EPS
    c = torch.where(q_norm > constraint, constraint / q_norm, torch.ones_like(q_norm))
    return q0, c, q_norm


def cayley(blocks: Tensor, constraint: float):
    """(q0, c, q_norm, M, R) in fp32: M = (I - q)^-1, R = (I + q) M."""
    q0, c, q_norm = skew(blocks, constraint)
    q = q0 * c
    eye = torch.eye(q.shape[-1], device=q.device, dtype=q.dtype)
    # inv_ex: no device sync (I - q is never singular for skew q).
    m = torch.linalg.inv_ex(eye - q).inverse
    return q0, c, q_norm, m, (eye + q) @ m


def rotation(blocks: Tensor, constraint: float, multiplier: float) -> Tensor:
    """R' = m*(I + q)(I - q)^-1 + (1-m)*I in fp32, per block."""
    *_, r = cayley(blocks, constraint)
    eye = torch.eye(r.shape[-1], device=r.device, dtype=r.dtype)
    return multiplier * r + (1.0 - multiplier) * eye


def chain_to_blocks(
    grad_rot: Tensor,
    r: Tensor,
    m: Tensor,
    q0: Tensor,
    c: Tensor | float,
    q_norm: Tensor | None,
    constraint: float,
    multiplier: float,
) -> Tensor:
    """The blocks' gradient (fp32) given R''s: dR = (I + R) dq M, then skew and norm."""
    eye = torch.eye(r.shape[-1], device=r.device, dtype=r.dtype)
    g_q = torch.bmm(
        torch.bmm((eye + r).transpose(-1, -2), multiplier * grad_rot.float()),
        m.transpose(-1, -2),
    )
    g_q0 = g_q * c
    if q_norm is not None:
        # c = constraint / ||q0|| past the constraint.
        raw = q_norm - _NORM_EPS
        active = q_norm > constraint
        dc = torch.where(
            active, -constraint / (q_norm * q_norm * raw), torch.zeros_like(raw)
        )
        g_q0 = g_q0 + (g_q * q0).sum() * dc * q0
    return g_q0 - g_q0.transpose(-1, -2)


def rotation_backward(
    blocks: Tensor, constraint: float, multiplier: float, grad_rot: Tensor
) -> Tensor:
    """The blocks' gradient given R''s."""
    q0, c, q_norm, m, r = cayley(blocks, constraint)
    g = chain_to_blocks(grad_rot, r, m, q0, c, q_norm, constraint, multiplier)
    return g.to(blocks.dtype)


def _grouped(t: Tensor, channel_axis: int, k: int, s: int) -> Tensor:
    tc = t.movedim(channel_axis, -1).float()
    return tc.reshape(*tc.shape[:-1], k, s)


def oft_apply(
    blocks: Tensor,
    x: Tensor,
    rescale: Tensor | None = None,
    constraint: float = 0.0,
    multiplier: float = 1.0,
    shift: bool = False,
    channel_axis: int = -1,
) -> Tensor:
    """rescale * R'^T x [- x] along channel_axis, computed in fp32."""
    k, s, _ = blocks.shape
    r = rotation(blocks, constraint, multiplier)
    xg = _grouped(x, channel_axis, k, s)
    out = torch.einsum("knm,...kn->...km", r, xg)
    if rescale is not None:
        out = out * rescale.float().reshape(k, s)
    if shift:
        out = out - xg
    out = out.reshape(*out.shape[:-2], k * s)
    return out.movedim(-1, channel_axis).to(x.dtype)


def oft_apply_backward(
    grad: Tensor,
    blocks: Tensor,
    x: Tensor,
    rescale: Tensor | None = None,
    constraint: float = 0.0,
    multiplier: float = 1.0,
    shift: bool = False,
    channel_axis: int = -1,
) -> tuple[Tensor, Tensor, Tensor]:
    """(grad_x, grad_blocks, grad_rescale); grad_rescale is empty without rescale."""
    k, s, _ = blocks.shape
    r = rotation(blocks, constraint, multiplier)
    xg = _grouped(x, channel_axis, k, s)
    gg = _grouped(grad, channel_axis, k, s)
    batch = tuple(range(xg.dim() - 2))
    g_res = absent(x)
    gy = gg
    if rescale is not None:
        y = torch.einsum("knm,...kn->...km", r, xg)
        g_res = (gg * y).sum(batch).reshape(rescale.shape).to(rescale.dtype)
        gy = gg * rescale.float().reshape(k, s)
    gx = torch.einsum("knm,...km->...kn", r, gy)
    if shift:
        gx = gx - gg
    g_rot = torch.einsum("...kn,...km->knm", xg, gy)
    gx = gx.reshape(*gx.shape[:-2], k * s).movedim(-1, channel_axis).to(x.dtype)
    return gx, rotation_backward(blocks, constraint, multiplier, g_rot), g_res


# torch.library Custom Op Definitions


@torch.library.custom_op("comfy_kitchen::oft_apply", mutates_args=())
def _op_oft_apply(
    blocks: torch.Tensor,
    x: torch.Tensor,
    rescale: torch.Tensor | None,
    constraint: float,
    multiplier: float,
    shift: bool,
    channel_axis: int,
) -> torch.Tensor:
    return dispatch(
        "oft_apply",
        blocks=blocks,
        x=x,
        rescale=rescale,
        constraint=constraint,
        multiplier=multiplier,
        shift=shift,
        channel_axis=channel_axis,
    )


@_op_oft_apply.register_fake
def _op_oft_apply_fake(blocks, x, rescale, constraint, multiplier, shift, channel_axis):
    return torch.empty_like(x)


@torch.library.custom_op("comfy_kitchen::oft_apply_backward", mutates_args=())
def _op_oft_apply_backward(
    grad: torch.Tensor,
    blocks: torch.Tensor,
    x: torch.Tensor,
    rescale: torch.Tensor | None,
    constraint: float,
    multiplier: float,
    shift: bool,
    channel_axis: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return dispatch(
        "oft_apply_backward",
        grad=grad,
        blocks=blocks,
        x=x,
        rescale=rescale,
        constraint=constraint,
        multiplier=multiplier,
        shift=shift,
        channel_axis=channel_axis,
    )


@_op_oft_apply_backward.register_fake
def _op_oft_apply_backward_fake(
    grad, blocks, x, rescale, constraint, multiplier, shift, channel_axis
):
    return (
        torch.empty_like(x),
        torch.empty_like(blocks),
        x.new_empty(0) if rescale is None else torch.empty_like(rescale),
    )


def _apply_setup(ctx, inputs, output):
    blocks, x, rescale, constraint, multiplier, shift, channel_axis = inputs
    ctx.save_for_backward(blocks, x, rescale)
    ctx.args = (constraint, multiplier, shift, channel_axis)


def _apply_backward(ctx, grad):
    blocks, x, rescale = ctx.saved_tensors
    gx, gb, gres = torch.ops.comfy_kitchen.oft_apply_backward(
        grad.contiguous(), blocks, x, rescale, *ctx.args
    )
    return gb, gx, present(gres), None, None, None, None


_op_oft_apply.register_autograd(_apply_backward, setup_context=_apply_setup)
