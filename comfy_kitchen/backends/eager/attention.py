# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dense non-causal attention on ``[B, S, H, head_dim]`` tensors (no mask).

The eager backend is ``F.scaled_dot_product_attention`` on the head-major
transpose; the Triton backend is a flash-attention forward that applies the
softmax scale inside the kernel, so q is never prescaled.
"""

import torch
from torch.nn import functional

from comfy_kitchen.registry import registry


def flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Reference: SDPA over [B, H, S, head_dim], returned as [B, S_q, H, head_dim]."""
    out = functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=scale)
    return out.transpose(1, 2).contiguous()


def _check(q, k, v):
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("flash_attn: q, k, v must be [B, S, H, head_dim]")
    if k.shape != v.shape:
        raise ValueError(f"flash_attn: k {tuple(k.shape)} and v {tuple(v.shape)} must match")
    if q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]:
        raise ValueError(f"flash_attn: q {tuple(q.shape)} and k {tuple(k.shape)} must share "
                         "batch, heads and head_dim")


@torch.library.custom_op("comfy_kitchen::flash_attn", mutates_args=())
def _op_flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    _check(q, k, v)
    kwargs = {"q": q, "k": k, "v": v, "scale": scale}
    impl = registry.get_implementation("flash_attn", kwargs=kwargs)
    return impl(**kwargs)


@_op_flash_attn.register_fake
def _op_flash_attn_fake(q, k, v, scale=None):
    return torch.empty_like(q, memory_format=torch.contiguous_format)
