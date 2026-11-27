# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional temporal frame interleave, frames to batch, then nearest-exact 2x upsample.

The upsample body of the Wan VAE ``Resample`` (``upsample2d`` / ``upsample3d``), from the
``time_conv`` output up to the input of the ``Conv2d`` that follows the upsample:
with ``interleave`` the ``[B, 2C, T, H, W]`` time_conv output is split into its two channel
halves and their frames are interleaved into ``[B, C, 2T, H, W]``; the frames are then moved
into the batch (``'b c t h w -> (b t) c h w'``) and upsampled 2x with ``nearest-exact``.
For a scale of 2 every output pixel is a copy of one input pixel.
"""

import torch
from torch.nn import functional

from comfy_kitchen.registry import registry


def temporal_interleave_upsample_nearest2x(
    x: torch.Tensor,
    interleave: bool = False,
) -> torch.Tensor:
    """Reference: the reshape / stack / rearrange / ``nn.Upsample`` sequence of the Wan VAE."""
    b, c, t, h, w = x.size()
    if interleave:
        c = c // 2
        x = x.reshape(b, 2, c, t, h, w)
        x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
        x = x.reshape(b, c, t * 2, h, w)
    t = x.shape[2]
    # rearrange(x, 'b c t h w -> (b t) c h w')
    x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    # nn.Upsample(scale_factor=(2., 2.), mode='nearest-exact')
    x = functional.interpolate(x, scale_factor=(2.0, 2.0), mode="nearest-exact")
    return x.contiguous()


def _out_shape(x: torch.Tensor, interleave: bool) -> tuple[int, int, int, int]:
    b, c, t, h, w = x.shape
    if interleave:
        c, t = c // 2, t * 2
    return b * t, c, 2 * h, 2 * w


def _check(x, interleave):
    if x.dim() != 5:
        raise ValueError(
            f"temporal_interleave_upsample_nearest2x: x must be [B, C, T, H, W], got {tuple(x.shape)}"
        )
    if interleave and x.shape[1] % 2:
        raise ValueError(
            "temporal_interleave_upsample_nearest2x: interleave needs an even channel count, "
            f"got {x.shape[1]}"
        )


@torch.library.custom_op("comfy_kitchen::temporal_interleave_upsample_nearest2x", mutates_args=())
def _op_temporal_interleave_upsample_nearest2x(
    x: torch.Tensor,
    interleave: bool = False,
) -> torch.Tensor:
    _check(x, interleave)
    kwargs = {"x": x, "interleave": interleave}
    impl = registry.get_implementation("temporal_interleave_upsample_nearest2x", kwargs=kwargs)
    return impl(**kwargs)


@_op_temporal_interleave_upsample_nearest2x.register_fake
def _op_temporal_interleave_upsample_nearest2x_fake(x, interleave=False):
    _check(x, interleave)
    return x.new_empty(_out_shape(x, interleave))
