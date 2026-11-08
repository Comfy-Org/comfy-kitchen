# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-group channel sums of a CHW image, the groups given by an index per pixel.

ComfyUI's ``sum_indexed_values`` (``comfy_extras/void_noise_warp.py``), called once per frame
by ``regaussianize`` in the VOID warped-noise node with the index matrix of ``unique_pixels``:

    u = int(index_matrix.max().item()) + 1
    flat = rearrange(image, "c h w -> (h w) c")
    out = torch.zeros((u, C), dtype=image.dtype, device=image.device)
    out.index_add_(0, index_matrix.view(-1), flat)

``image`` is ``[C, H, W]``, ``index_matrix`` ``[H, W]`` int32 / int64 with values >= 0; the
output is ``[U, C]`` of the image's dtype with ``U = index_matrix.max() + 1``, row ``g`` the sum
of the pixels whose index is ``g`` (zero for an index no pixel has).
"""

import torch

from comfy_kitchen.registry import registry


def sum_indexed_values(image: torch.Tensor, index_matrix: torch.Tensor) -> torch.Tensor:
    """Reference: ``max().item()``, ``torch.zeros`` and ``index_add_``, in ComfyUI's order."""
    _, h, w = image.shape
    u = int(index_matrix.max().item()) + 1
    # rearrange(image, "c h w -> (h w) c")
    flat = image.permute(1, 2, 0).reshape(h * w, -1)
    out = torch.zeros((u, flat.shape[1]), dtype=flat.dtype, device=flat.device)
    out.index_add_(0, index_matrix.view(-1), flat)
    return out


def _check(image, index_matrix):
    name = "sum_indexed_values"
    if image.dim() != 3:
        raise ValueError(f"{name}: image must be [C, H, W], got {tuple(image.shape)}")
    if index_matrix.dim() != 2 or index_matrix.shape != image.shape[1:]:
        raise ValueError(
            f"{name}: index_matrix must be [H, W] = {tuple(image.shape[1:])}, "
            f"got {tuple(index_matrix.shape)}"
        )
    if index_matrix.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name}: index_matrix must be int32 or int64, got {index_matrix.dtype}")
    if index_matrix.device != image.device:
        raise ValueError(f"{name}: index_matrix must be on the image's device")
    if index_matrix.numel() == 0:
        raise ValueError(f"{name}: the image must have at least one pixel")


@torch.library.custom_op("comfy_kitchen::sum_indexed_values", mutates_args=())
def _op_sum_indexed_values(image: torch.Tensor, index_matrix: torch.Tensor) -> torch.Tensor:
    _check(image, index_matrix)
    kwargs = {"image": image, "index_matrix": index_matrix}
    impl = registry.get_implementation("sum_indexed_values", kwargs=kwargs)
    return impl(**kwargs)


@_op_sum_indexed_values.register_fake
def _op_sum_indexed_values_fake(image, index_matrix):
    _check(image, index_matrix)
    # U = index_matrix.max() + 1 depends on the data
    groups = torch.library.get_ctx().new_dynamic_size(min=1)
    return image.new_empty((groups, image.shape[0]))
