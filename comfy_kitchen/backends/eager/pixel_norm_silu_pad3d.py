# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PixelNorm over channels, SiLU, then the causal temporal padding of the next conv.

The LTX VAE ``ResnetBlock3D`` norm -> SiLU -> ``CausalConv3d`` sequence (and the decoder's
``conv_norm_out`` -> ``conv_act`` -> ``conv_out``), from the PixelNorm input up to the
``Conv3d`` call, with ``timestep_conditioning`` off:

    h = silu(x / sqrt(mean(x**2, dim=1, keepdim=True) + eps))
    y = cat([prefix, h[:, :, :1] repeated n_front times, h, h[:, :, -1:] repeated n_back times],
            dim=2)
    y = F.pad(y, (1, 1, 1, 1, 0, 0), mode="reflect")    # pad_mode "reflect" only

``prefix`` is the temporal cache of ``CausalConv3d``: frames of the conv input of the
previous chunk, which are already normalized and activated, so they are copied as they are.
Without a cache ``CausalConv3d`` repeats the first frame (``n_front``); on the last chunk of a
non-causal conv it repeats the last frame (``n_back``). pad_mode "zeros" gives the conv input
(the conv keeps its implicit zero padding); "reflect" gives the input already reflect-padded by
1 in H and W, which is what ``Conv3d`` with ``padding_mode="reflect"`` builds before a conv with
padding 0 (LTXV 0.9).
"""

import torch
from torch.nn import functional

from comfy_kitchen.registry import registry

_PAD_MODES = ("zeros", "reflect")
_REFLECT_PAD = 1  # CausalConv3d, kernel 3: padding (0, 1, 1)


def pixel_norm_silu_pad3d(
    x: torch.Tensor,
    prefix: torch.Tensor | None = None,
    n_front: int = 1,
    n_back: int = 0,
    pad_mode: str = "zeros",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Reference: ``PixelNorm``, ``nn.SiLU``, the ``CausalConv3d`` repeat / cat and the
    reflect ``F.pad`` of ``Conv3d``, in that order."""
    h = x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + eps)
    h = functional.silu(h)
    pieces = [] if prefix is None else [prefix]
    if n_front:
        pieces.append(h[:, :, :1, :, :].repeat((1, 1, n_front, 1, 1)))
    pieces.append(h)
    if n_back:
        pieces.append(h[:, :, -1:, :, :].repeat((1, 1, n_back, 1, 1)))
    out = torch.cat(pieces, dim=2)
    if pad_mode == "reflect":
        p = _REFLECT_PAD
        out = functional.pad(out, (p, p, p, p, 0, 0), mode="reflect")
    return out.contiguous()


def _out_shape(x, prefix, n_front, n_back, pad_mode):
    b, c, t, h, w = x.shape
    p = _REFLECT_PAD if pad_mode == "reflect" else 0
    t_pre = 0 if prefix is None else prefix.shape[2]
    return b, c, t_pre + n_front + t + n_back, h + 2 * p, w + 2 * p


def _check(x, prefix, n_front, n_back, pad_mode):
    name = "pixel_norm_silu_pad3d"
    if x.dim() != 5:
        raise ValueError(f"{name}: x must be [B, C, T, H, W], got {tuple(x.shape)}")
    if x.shape[2] < 1:
        raise ValueError(f"{name}: x needs at least one frame, got {tuple(x.shape)}")
    if n_front < 0 or n_back < 0:
        raise ValueError(f"{name}: n_front and n_back must be >= 0, got {n_front}, {n_back}")
    if pad_mode not in _PAD_MODES:
        raise ValueError(f"{name}: pad_mode must be one of {_PAD_MODES}, got {pad_mode!r}")
    if pad_mode == "reflect" and min(x.shape[3], x.shape[4]) <= _REFLECT_PAD:
        raise ValueError(f"{name}: reflect padding needs H and W > {_REFLECT_PAD}")
    if prefix is not None:
        if prefix.dim() != 5 or prefix.shape[:2] != x.shape[:2] or prefix.shape[3:] != x.shape[3:]:
            raise ValueError(
                f"{name}: prefix must be [B, C, T_prefix, H, W] of x {tuple(x.shape)}, "
                f"got {tuple(prefix.shape)}"
            )
        if prefix.dtype != x.dtype or prefix.device != x.device:
            raise ValueError(f"{name}: prefix must have x's dtype and device")


@torch.library.custom_op("comfy_kitchen::pixel_norm_silu_pad3d", mutates_args=())
def _op_pixel_norm_silu_pad3d(
    x: torch.Tensor,
    prefix: torch.Tensor | None = None,
    n_front: int = 1,
    n_back: int = 0,
    pad_mode: str = "zeros",
    eps: float = 1e-8,
) -> torch.Tensor:
    _check(x, prefix, n_front, n_back, pad_mode)
    kwargs = {
        "x": x,
        "prefix": prefix,
        "n_front": n_front,
        "n_back": n_back,
        "pad_mode": pad_mode,
        "eps": eps,
    }
    impl = registry.get_implementation("pixel_norm_silu_pad3d", kwargs=kwargs)
    return impl(**kwargs)


@_op_pixel_norm_silu_pad3d.register_fake
def _op_pixel_norm_silu_pad3d_fake(x, prefix=None, n_front=1, n_back=0, pad_mode="zeros", eps=1e-8):
    _check(x, prefix, n_front, n_back, pad_mode)
    return x.new_empty(_out_shape(x, prefix, n_front, n_back, pad_mode))
