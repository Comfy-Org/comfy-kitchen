# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bilinear backward warp of images by optical flows, concatenated over channels.

FILM frame interpolation (``comfy_extras/frame_interpolation_models/film_net.py``) warps a
feature map by a flow in pixel units with ``_warp_core`` (``grid_x`` / ``grid_y`` are the
``linspace(-(1 - 1 / n), 1 - 1 / n, n)`` pixel centres FILMNet caches per level):

    dx = flow[:, 0].float() / (W * 0.5)
    dy = flow[:, 1].float() / (H * 0.5)
    grid = stack([grid_x[None, None, :] + dx, grid_y[None, :, None] + dy], dim=3)
    warped = grid_sample(image.float(), grid, "bilinear", "border", align_corners=False)
    warped = warped.to(image.dtype)

and concatenates the result with other tensors before the next conv. Two forms:

- ``PyramidFlowEstimator`` step: ``cat([prefix, warp(image, flow)], dim=1)`` (the
  ``FlowEstimator`` input, ``prefix`` = the other frame's features), one image, no scales;
- ``FILMNet.forward_multi_timestep`` fusion level: with ``scales = [dt, 1 - dt]`` the flows
  are first scaled as ``multiply_pyramid`` does, ``f * s[:, None, None, None]``, and the
  scaled flows are appended after the warped images:
  ``cat([warp(image0, f0 * s0), warp(image1, f1 * s1), f0 * s0, f1 * s1], dim=1)``.
"""

import torch
from torch.nn import functional

from comfy_kitchen.registry import registry


def _warp_core(image, flow, grid_x, grid_y):
    dtype = image.dtype
    height, width = flow.shape[2], flow.shape[3]
    dx = flow[:, 0].float() / (width * 0.5)
    dy = flow[:, 1].float() / (height * 0.5)
    grid = torch.stack([grid_x[None, None, :] + dx, grid_y[None, :, None] + dy], dim=3)
    return functional.grid_sample(
        image.float(), grid, mode="bilinear", padding_mode="border", align_corners=False
    ).to(dtype)


def flow_warp_cat(
    images: list[torch.Tensor],
    flows: list[torch.Tensor],
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
    prefix: torch.Tensor | None = None,
    scales: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    """Reference: ``multiply_pyramid``'s scaling, FILM's ``_warp_core`` and ``torch.cat``, in
    ComfyUI's order."""
    flows = list(flows)
    if scales is not None:
        flows = [f * s[:, None, None, None] for f, s in zip(flows, scales, strict=True)]
    warped = [_warp_core(i, f, grid_x, grid_y) for i, f in zip(images, flows, strict=True)]
    parts = [] if prefix is None else [prefix]
    parts += warped
    if scales is not None:
        parts += flows
    return torch.cat(parts, dim=1)


def _check(images, flows, grid_x, grid_y, prefix, scales):
    name = "flow_warp_cat"
    if len(images) == 0 or len(images) != len(flows):
        raise ValueError(f"{name}: images and flows must be non-empty lists of the same length")
    if scales is not None and len(scales) != len(flows):
        raise ValueError(f"{name}: scales must have one entry per flow")
    if grid_x.dim() != 1 or grid_y.dim() != 1:
        raise ValueError(f"{name}: grid_x and grid_y must be 1D")
    f0 = flows[0]
    for image, flow in zip(images, flows, strict=True):
        if image.dim() != 4 or flow.dim() != 4 or flow.shape[1] != 2:
            raise ValueError(
                f"{name}: images must be [B, C, H, W] and flows [B, 2, H, W], got "
                f"{tuple(image.shape)} and {tuple(flow.shape)}"
            )
        if flow.shape[0] != f0.shape[0] or flow.shape[2:] != f0.shape[2:]:
            raise ValueError(f"{name}: flows must have the same batch, height and width")
        if image.shape[0] != f0.shape[0]:
            raise ValueError(f"{name}: images must have the flows' batch")
    if prefix is not None and (
        prefix.dim() != 4 or prefix.shape[0] != f0.shape[0] or prefix.shape[2:] != f0.shape[2:]
    ):
        raise ValueError(
            f"{name}: prefix must be [B, C, H, W] with the flows' B, H, W, got {tuple(prefix.shape)}"
        )
    if scales is not None:
        for s in scales:
            if s.dim() != 1:
                raise ValueError(f"{name}: scales must be 1D (one value per batch item or one)")


@torch.library.custom_op("comfy_kitchen::flow_warp_cat", mutates_args=())
def _op_flow_warp_cat(
    images: list[torch.Tensor],
    flows: list[torch.Tensor],
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
    prefix: torch.Tensor | None,
    scales: list[torch.Tensor],
) -> torch.Tensor:
    # an empty scales list means no scales: "Tensor[]?" is a custom op schema type only from
    # torch 2.15
    scales = scales or None
    _check(images, flows, grid_x, grid_y, prefix, scales)
    kwargs = {
        "images": images,
        "flows": flows,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "prefix": prefix,
        "scales": scales,
    }
    impl = registry.get_implementation("flow_warp_cat", kwargs=kwargs)
    return impl(**kwargs)


@_op_flow_warp_cat.register_fake
def _op_flow_warp_cat_fake(images, flows, grid_x, grid_y, prefix, scales):
    scales = scales or None
    _check(images, flows, grid_x, grid_y, prefix, scales)
    batch, _, height, width = flows[0].shape
    channels = sum(i.shape[1] for i in images)
    dtypes = [i.dtype for i in images]
    if prefix is not None:
        channels += prefix.shape[1]
        dtypes.append(prefix.dtype)
    if scales is not None:
        channels += 2 * len(flows)
        dtypes += [
            torch.promote_types(f.dtype, s.dtype) for f, s in zip(flows, scales, strict=True)
        ]
    dtype = dtypes[0]
    for d in dtypes[1:]:
        dtype = torch.promote_types(dtype, d)
    return flows[0].new_empty((batch, channels, height, width), dtype=dtype)
