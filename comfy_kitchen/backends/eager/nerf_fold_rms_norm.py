# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Patch-to-image fold of the NeRF head output, then RMSNorm over channels.

The Chroma Radiance NeRF head tail (``comfy/ldm/chroma_radiance/model.py``
``forward_tiled_nerf`` / ``forward_nerf``, then ``NerfFinalLayerConv.norm`` in ``layers.py``),
up to the final 3x3 conv:

    img_dct = cat(tiles, dim=0)                                   # [B*L, P*P, C]
    img_dct = img_dct.transpose(1, 2).reshape(B, L, -1).transpose(1, 2)   # [B, C*P*P, L]
    img = fold(img_dct, output_size=(H, W), kernel_size=P, stride=P)       # [B, C, H, W]
    out = rms_norm(img.movedim(1, -1), (C,), weight, eps).movedim(-1, 1)

``tiles`` are the NeRF block outputs of consecutive row ranges (one tensor when the head is
not tiled); row ``b * L + l`` is patch ``l`` of batch item ``b``, pixel ``py * P + px`` of a
patch is the image pixel ``(lh * P + py, lw * P + px)`` with ``l = lh * (W // P) + lw``. The
output has the layout the stock path hands to the conv: shape ``[B, C, H, W]`` with the
channels innermost in memory (strides ``[H*W*C, 1, W*C, C]``). ``eps=None`` is
``RMSNorm``'s default (``F.rms_norm`` with eps None, the machine epsilon of the compute type).
"""

import torch
from torch.nn import functional

from comfy_kitchen.registry import registry


def nerf_fold_rms_norm(
    tiles: list[torch.Tensor],
    weight: torch.Tensor,
    batch: int,
    height: int,
    width: int,
    patch_size: int,
    eps: float | None = None,
) -> torch.Tensor:
    """Reference: ``torch.cat``, the transpose / reshape / transpose, ``F.fold`` and
    ``F.rms_norm`` on ``movedim(1, -1)``, in ComfyUI's order."""
    img_dct = torch.cat(list(tiles), dim=0)
    num_patches = img_dct.shape[0] // batch
    img_dct = img_dct.transpose(1, 2)
    img_dct = img_dct.reshape(batch, num_patches, -1)
    img_dct = img_dct.transpose(1, 2)
    img_dct = functional.fold(
        img_dct, output_size=(height, width), kernel_size=patch_size, stride=patch_size
    )
    x = img_dct.movedim(1, -1)
    return functional.rms_norm(x, (x.shape[-1],), weight, eps).movedim(-1, 1)


def _check(tiles, weight, batch, height, width, patch_size):
    name = "nerf_fold_rms_norm"
    if len(tiles) == 0:
        raise ValueError(f"{name}: tiles must not be empty")
    t0 = tiles[0]
    for t in tiles:
        if t.dim() != 3 or t.shape[1:] != t0.shape[1:]:
            raise ValueError(
                f"{name}: tiles must be [rows, P*P, C] with the same P*P and C, "
                f"got {[tuple(u.shape) for u in tiles]}"
            )
        if t.dtype != t0.dtype or t.device != t0.device:
            raise ValueError(f"{name}: tiles must have the same dtype and device")
    if patch_size < 1 or t0.shape[1] != patch_size * patch_size:
        raise ValueError(
            f"{name}: tiles must hold patch_size**2 = {patch_size * patch_size} pixels per row, "
            f"got {t0.shape[1]}"
        )
    if weight.dim() != 1 or weight.shape[0] != t0.shape[2]:
        raise ValueError(f"{name}: weight must be [C] = [{t0.shape[2]}], got {tuple(weight.shape)}")
    if weight.device != t0.device:
        raise ValueError(f"{name}: weight must be on the tiles' device")
    if batch < 1 or height % patch_size or width % patch_size:
        raise ValueError(
            f"{name}: batch must be >= 1 and height, width multiples of patch_size, "
            f"got {batch}, {height}, {width}, {patch_size}"
        )
    rows = sum(t.shape[0] for t in tiles)
    patches = (height // patch_size) * (width // patch_size)
    if rows != batch * patches:
        raise ValueError(
            f"{name}: tiles hold {rows} rows, expected batch * (H // P) * (W // P) = "
            f"{batch * patches}"
        )


@torch.library.custom_op("comfy_kitchen::nerf_fold_rms_norm", mutates_args=())
def _op_nerf_fold_rms_norm(
    tiles: list[torch.Tensor],
    weight: torch.Tensor,
    batch: int,
    height: int,
    width: int,
    patch_size: int,
    eps: float | None = None,
) -> torch.Tensor:
    _check(tiles, weight, batch, height, width, patch_size)
    kwargs = {
        "tiles": tiles,
        "weight": weight,
        "batch": batch,
        "height": height,
        "width": width,
        "patch_size": patch_size,
        "eps": eps,
    }
    impl = registry.get_implementation("nerf_fold_rms_norm", kwargs=kwargs)
    return impl(**kwargs)


@_op_nerf_fold_rms_norm.register_fake
def _op_nerf_fold_rms_norm_fake(tiles, weight, batch, height, width, patch_size, eps=None):
    _check(tiles, weight, batch, height, width, patch_size)
    channels = tiles[0].shape[2]
    return tiles[0].new_empty((batch, height, width, channels)).permute(0, 3, 1, 2)
