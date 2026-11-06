# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""3D depth-to-space of a channel-repeated shortcut, added to a residual.

``residual + depth_to_space3d(repeat_interleave(x, repeats, dim=1))``: the upsampling
shortcut of the video VAE decoders. x ``[B, C, T, H, W]`` has its channels repeated
``repeats`` times, then each group of ``factor_t * factor_s * factor_s`` channels becomes a
``factor_t x factor_s x factor_s`` block of output positions, giving
``[B, C * repeats / (factor_t * factor_s**2), T * factor_t, H * factor_s, W * factor_s]``.

- Wan 2.2 VAE (``comfy/ldm/wan/vae2_2.py``): ``DupUp3D`` with channels ordered
  ``(c, ft, fs, fs)``; ``first_chunk`` drops the first ``factor_t - 1`` output frames, and
  ``Up_ResidualBlock`` adds the result to the main branch, out of place for video and into
  the main branch in row strips (``strip_apply``) for images.
- HunyuanVideo 1.5 VAE (``comfy/ldm/hunyuan_video/vae_refiner.py``): ``UpSmpl`` with channels
  ordered ``(ft, fs, fs, c)`` (``factors_first``), where the residual is the packed conv output
  and goes through the same depth-to-space without the repeat (``residual_packed``).

This backend runs those ComfyUI sequences as they are.
"""

import torch

from comfy_kitchen.registry import registry

# comfy/ldm/wan/vae2_2.py: strips of rows bound the size of the shortcut temporary
STRIP_ELEMS = 2**24


def _depth_to_space3d(x, factor_t, factor_s, repeats, first_chunk, factors_first):
    if repeats is not None:
        x = x.repeat_interleave(repeats, dim=1)
    b, c, t, h, w = x.shape
    channels = c // (factor_t * factor_s * factor_s)
    if factors_first:
        # UpSmpl.forward
        x = x.reshape(b, factor_t, factor_s, factor_s, channels, t, h, w)
        x = x.permute(0, 4, 5, 1, 6, 2, 7, 3)
        x = x.reshape(b, channels, t * factor_t, h * factor_s, w * factor_s)
    else:
        # DupUp3D.forward
        x = x.view(b, channels, factor_t, factor_s, factor_s, t, h, w)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(b, channels, x.size(2) * factor_t, x.size(4) * factor_s, x.size(6) * factor_s)
    if first_chunk:
        x = x[:, :, factor_t - 1 :, :, :]
    return x


def depth_to_space3d_residual_add(
    x: torch.Tensor,
    residual: torch.Tensor,
    factor_t: int,
    factor_s: int,
    repeats: int = 1,
    first_chunk: bool = False,
    factors_first: bool = False,
    residual_packed: bool = False,
) -> torch.Tensor:
    """Reference: ``x_main + DupUp3D(x)`` (Wan 2.2) or the ``UpSmpl`` tail ``h + x`` (HunyuanVideo 1.5)."""
    shortcut = _depth_to_space3d(x, factor_t, factor_s, repeats, first_chunk, factors_first)
    if residual_packed:
        residual = _depth_to_space3d(residual, factor_t, factor_s, None, first_chunk, factors_first)
    return residual + shortcut


def depth_to_space3d_residual_add_(
    x: torch.Tensor,
    residual: torch.Tensor,
    factor_t: int,
    factor_s: int,
    repeats: int = 1,
    first_chunk: bool = False,
) -> torch.Tensor:
    """Reference: ``strip_apply(DupUp3D, x, scale=factor_s, halo=0, out=residual)`` of the Wan 2.2
    ``Up_ResidualBlock`` image path, which adds the shortcut into residual one row strip at a time."""
    # max(1, ...) only keeps empty inputs from dividing by zero
    n = max(1, -(-x.numel() * factor_s * factor_s // STRIP_ELEMS))
    size = x.shape[-2]
    step = max(1, -(-size // n))
    for a in range(0, size, step):
        b = min(size, a + step)
        y = _depth_to_space3d(
            x.narrow(-2, a, b - a), factor_t, factor_s, repeats, first_chunk, False
        )
        residual.narrow(-2, a * factor_s, (b - a) * factor_s).add_(y)
    return residual


def _out_shape(x, factor_t, factor_s, repeats, first_chunk):
    b, c, t, h, w = x.shape
    channels = c * repeats // (factor_t * factor_s * factor_s)
    return (
        b,
        channels,
        t * factor_t - (factor_t - 1 if first_chunk else 0),
        h * factor_s,
        w * factor_s,
    )


def _check(x, residual, factor_t, factor_s, repeats, first_chunk, residual_packed=False):
    name = "depth_to_space3d_residual_add"
    if x.dim() != 5 or residual.dim() != 5:
        raise ValueError(
            f"{name}: x and residual must be 5-D, got {tuple(x.shape)} and {tuple(residual.shape)}"
        )
    if factor_t < 1 or factor_s < 1 or repeats < 1:
        raise ValueError(f"{name}: factors and repeats must be >= 1")
    if (x.shape[1] * repeats) % (factor_t * factor_s * factor_s):
        raise ValueError(
            f"{name}: {x.shape[1]} * {repeats} channels do not split into "
            f"{factor_t} x {factor_s} x {factor_s} blocks"
        )
    if first_chunk and x.shape[2] == 0:
        raise ValueError(f"{name}: first_chunk needs at least one frame")
    if residual_packed:
        want = (x.shape[0], x.shape[1] * repeats, *x.shape[2:])
    else:
        want = _out_shape(x, factor_t, factor_s, repeats, first_chunk)
    if tuple(residual.shape) != want:
        raise ValueError(f"{name}: residual must be {want}, got {tuple(residual.shape)}")


@torch.library.custom_op("comfy_kitchen::depth_to_space3d_residual_add", mutates_args=())
def _op_depth_to_space3d_residual_add(
    x: torch.Tensor,
    residual: torch.Tensor,
    factor_t: int,
    factor_s: int,
    repeats: int,
    first_chunk: bool,
    factors_first: bool,
    residual_packed: bool,
) -> torch.Tensor:
    _check(x, residual, factor_t, factor_s, repeats, first_chunk, residual_packed)
    kwargs = {
        "x": x,
        "residual": residual,
        "factor_t": factor_t,
        "factor_s": factor_s,
        "repeats": repeats,
        "first_chunk": first_chunk,
        "factors_first": factors_first,
        "residual_packed": residual_packed,
    }
    impl = registry.get_implementation("depth_to_space3d_residual_add", kwargs=kwargs)
    return impl(**kwargs)


@_op_depth_to_space3d_residual_add.register_fake
def _op_depth_to_space3d_residual_add_fake(
    x, residual, factor_t, factor_s, repeats, first_chunk, factors_first, residual_packed
):
    # the eager add promotes mixed dtypes
    dtype = torch.promote_types(x.dtype, residual.dtype)
    if residual_packed:
        shape = _out_shape(x, factor_t, factor_s, repeats, first_chunk)
        return torch.empty(shape, dtype=dtype, device=residual.device)
    return torch.empty_like(residual, dtype=dtype)


@torch.library.custom_op(
    "comfy_kitchen::depth_to_space3d_residual_add_", mutates_args=("residual",)
)
def _op_depth_to_space3d_residual_add_(
    x: torch.Tensor,
    residual: torch.Tensor,
    factor_t: int,
    factor_s: int,
    repeats: int,
    first_chunk: bool,
) -> None:
    _check(x, residual, factor_t, factor_s, repeats, first_chunk)
    kwargs = {
        "x": x,
        "residual": residual,
        "factor_t": factor_t,
        "factor_s": factor_s,
        "repeats": repeats,
        "first_chunk": first_chunk,
    }
    registry.get_implementation("depth_to_space3d_residual_add_", kwargs=kwargs)(**kwargs)


@_op_depth_to_space3d_residual_add_.register_fake
def _op_depth_to_space3d_residual_add_fake_(x, residual, factor_t, factor_s, repeats, first_chunk):
    return None
