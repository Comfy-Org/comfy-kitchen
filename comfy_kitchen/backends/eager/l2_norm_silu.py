# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Channel L2 normalization, scale and weight, then SiLU.

``silu(x / max(||x||_2, eps) * scale * weight)`` with the norm taken over the
channel dim 1 of a channel-first ``[B, C, *spatial]`` tensor: the
``F.normalize(x, dim=1) * scale * gamma`` RMS norm of the Wan VAE (scale is
``sqrt(C)``) followed by the SiLU that comes after it in every residual block.
"""

import torch
from torch.nn import functional

from comfy_kitchen.registry import registry


def l2_norm_silu(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Reference: ``F.normalize`` over dim 1, times scale and weight, then ``F.silu``."""
    w = weight.to(x.dtype).reshape(x.shape[1], *([1] * (x.dim() - 2)))
    return functional.silu(functional.normalize(x, dim=1, eps=eps) * scale * w).contiguous()


def _check(x, weight):
    if x.dim() < 3:
        raise ValueError(f"l2_norm_silu: x must be [B, C, *spatial], got {tuple(x.shape)}")
    if weight.numel() != x.shape[1]:
        raise ValueError(f"l2_norm_silu: weight needs {x.shape[1]} elements, got {weight.numel()}")


@torch.library.custom_op("comfy_kitchen::l2_norm_silu", mutates_args=())
def _op_l2_norm_silu(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    _check(x, weight)
    kwargs = {"x": x, "weight": weight, "scale": scale, "eps": eps}
    impl = registry.get_implementation("l2_norm_silu", kwargs=kwargs)
    return impl(**kwargs)


@_op_l2_norm_silu.register_fake
def _op_l2_norm_silu_fake(x, weight, scale, eps=1e-12):
    return torch.empty_like(x, memory_format=torch.contiguous_format)
