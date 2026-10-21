# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor
from torch.nn import functional

from comfy_kitchen.registry import registry


def group_norm_stats3d(x: Tensor, num_groups: int, eps: float) -> Tensor:
    """Per-frame GroupNorm statistics of x [B, C, T, H, W]: float32 [B*T, num_groups, 2]
    holding (mean, rstd), the form group_norm_silu_pad3d takes as ``stats``."""
    frames = x.permute(0, 2, 1, 3, 4).reshape(x.shape[0] * x.shape[2], num_groups, -1).float()
    var, mean = torch.var_mean(frames, dim=-1, unbiased=False)
    return torch.stack((mean, torch.rsqrt(var + eps)), dim=-1)


def group_norm_silu_pad3d(
    x: Tensor,
    weight: Tensor | None,
    bias: Tensor | None,
    num_groups: int,
    eps: float,
    pad: list[int],
    silu: bool,
    zero_pad: bool = False,
    stats: Tensor | None = None,
) -> Tensor:
    """Per-frame GroupNorm, optional SiLU, then (l, r, t, b) spatial and `front`
    zero frames of padding. weight=None skips the norm. The spatial border reflects
    unless zero_pad, which models whose convolutions pad with zeros need instead.
    ``stats`` (from group_norm_stats3d) normalizes with given per-frame statistics, so
    x may be a row strip of the frame they describe."""
    if min(pad) < 0:
        raise ValueError("group_norm_silu_pad3d: padding must be non-negative")
    orig = x
    b, c, t, h, w = x.shape
    if weight is not None:
        # group_norm on CUDA rejects mixed dtypes; the affine params may be fp32
        weight = weight.to(x.dtype)
        bias = None if bias is None else bias.to(x.dtype)
        y = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        if stats is None:
            y = functional.group_norm(y, num_groups, weight, bias, eps)
        else:
            if stats.shape != (b * t, num_groups, 2):  # a short stats tensor would broadcast silently
                raise ValueError(f"group_norm_silu_pad3d: stats must be [{b * t}, {num_groups}, 2]")
            # per-(frame, channel) scale and shift in fp32, rounded once like group_norm's output
            scale = stats[..., 1].repeat_interleave(c // num_groups, dim=1) * weight.float()
            shift = -stats[..., 0].repeat_interleave(c // num_groups, dim=1) * scale
            if bias is not None:
                shift = shift + bias.float()
            y = torch.addcmul(shift[:, :, None, None], y.float(), scale[:, :, None, None]).to(x.dtype)
        x = y.view(b, t, c, h, w).permute(0, 2, 1, 3, 4)
    if silu:
        x = functional.silu(x)
    left, right, top, bottom, front = pad
    if left or right or top or bottom:
        x = functional.pad(x, (left, right, top, bottom, 0, 0),
                           mode="constant" if zero_pad else "reflect")
    if front:
        x = functional.pad(x, (0, 0, 0, 0, front, 0))
    # channels_last_3d like the CUDA backend (the registered fake promises it);
    # a custom op's output must not alias its input (nothing-to-do call)
    out = x.contiguous(memory_format=torch.channels_last_3d)
    return out if out is not orig else out.clone()


def group_norm_silu_pad3d_out(
    x: Tensor,
    weight: Tensor | None,
    bias: Tensor | None,
    num_groups: int,
    eps: float,
    pad: list[int],
    silu: bool,
    zero_pad: bool,
    out: Tensor,
    stats: Tensor | None = None,
) -> None:
    """group_norm_silu_pad3d written into ``out`` (the padded shape, NDHWC-ordered)."""
    out.copy_(group_norm_silu_pad3d(x, weight, bias, num_groups, eps, pad, silu, zero_pad, stats))


@torch.library.custom_op("comfy_kitchen::group_norm_stats3d", mutates_args=())
def _op_group_norm_stats3d(x: torch.Tensor, num_groups: int, eps: float) -> torch.Tensor:
    kwargs = {"x": x, "num_groups": num_groups, "eps": eps}
    impl = registry.get_implementation("group_norm_stats3d", kwargs=kwargs)
    return impl(**kwargs)


@_op_group_norm_stats3d.register_fake
def _op_group_norm_stats3d_fake(x, num_groups, eps):
    return torch.empty((x.shape[0] * x.shape[2], num_groups, 2), dtype=torch.float32, device=x.device)


@torch.library.custom_op("comfy_kitchen::group_norm_silu_pad3d_out", mutates_args=("out",))
def _op_group_norm_silu_pad3d_out(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    num_groups: int,
    eps: float,
    pad: list[int],
    silu: bool,
    zero_pad: bool,
    out: torch.Tensor,
    stats: torch.Tensor | None,
) -> None:
    # copy_ would broadcast or cast into a mismatched buffer instead of failing
    if min(pad) >= 0:
        b, c, t, h, w = x.shape
        left, right, top, bottom, front = pad
        shape = (b, c, t + front, h + top + bottom, w + left + right)
        if out.shape != shape or out.dtype != x.dtype or out.device != x.device:
            raise ValueError(f"group_norm_silu_pad3d: out must be {shape} {x.dtype} on {x.device}")
    kwargs = {"x": x, "weight": weight, "bias": bias, "num_groups": num_groups, "eps": eps,
              "pad": pad, "silu": silu, "zero_pad": zero_pad, "stats": stats, "out": out}
    impl = registry.get_implementation("group_norm_silu_pad3d_out", kwargs=kwargs)
    impl(**kwargs)


@_op_group_norm_silu_pad3d_out.register_fake
def _op_group_norm_silu_pad3d_out_fake(x, weight, bias, num_groups, eps, pad, silu, zero_pad, out, stats):
    return None


@torch.library.custom_op("comfy_kitchen::group_norm_silu_pad3d", mutates_args=())
def _op_group_norm_silu_pad3d(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    num_groups: int,
    eps: float,
    pad: list[int],
    silu: bool,
    zero_pad: bool,
    stats: torch.Tensor | None,
) -> torch.Tensor:
    kwargs = {"x": x, "weight": weight, "bias": bias, "num_groups": num_groups,
              "eps": eps, "pad": pad, "silu": silu, "zero_pad": zero_pad, "stats": stats}
    impl = registry.get_implementation("group_norm_silu_pad3d", kwargs=kwargs)
    return impl(**kwargs)


@_op_group_norm_silu_pad3d.register_fake
def _op_group_norm_silu_pad3d_fake(x, weight, bias, num_groups, eps, pad, silu, zero_pad, stats):
    b, c, t, h, w = x.shape
    left, right, top, bottom, front = pad
    return torch.empty((b, c, t + front, h + top + bottom, w + left + right),
                       dtype=x.dtype, device=x.device, memory_format=torch.channels_last_3d)
