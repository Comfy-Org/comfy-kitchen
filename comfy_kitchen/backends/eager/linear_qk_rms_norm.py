# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused QKV projection with per-head q/k RMSNorm (joint-attention pre-attention).

``y = x @ weight.T + bias`` is split into q, k, v of ``head_dim``-wide heads; q and k get
an RMSNorm over each head with their own scale, v is passed through.
"""

import torch
from torch import Tensor
from torch.nn import functional

from comfy_kitchen.registry import registry


def _qkv_shape(x: Tensor, weight: Tensor, head_dim: int) -> tuple[int, int, int, int]:
    b, s, _ = x.shape
    dim = weight.shape[0] // 3
    return b, s, dim // head_dim, dim


def linear_qk_rms_norm(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    q_scale: Tensor,
    k_scale: Tensor,
    head_dim: int,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Reference: ``F.linear`` rounded to x's dtype, split, then ``F.rms_norm`` on q and k."""
    b, s, h, _ = _qkv_shape(x, weight, head_dim)
    y = functional.linear(x, weight, bias)
    q, k, v = y.view(b, s, 3, h, head_dim).unbind(2)
    q = functional.rms_norm(q, (head_dim,), q_scale.to(x.dtype), eps)
    k = functional.rms_norm(k, (head_dim,), k_scale.to(x.dtype), eps)
    return q, k, v.contiguous()


def _check(x, weight, bias, q_scale, k_scale, head_dim):
    if x.dim() != 3 or weight.dim() != 2 or weight.shape[1] != x.shape[2]:
        raise ValueError(f"linear_qk_rms_norm: x [B, S, K] and weight [3*D, K], got {tuple(x.shape)} "
                         f"and {tuple(weight.shape)}")
    if weight.shape[0] % 3 or (weight.shape[0] // 3) % head_dim:
        raise ValueError(f"linear_qk_rms_norm: weight rows {weight.shape[0]} must be 3 * H * head_dim")
    if q_scale.numel() != head_dim or k_scale.numel() != head_dim:
        raise ValueError("linear_qk_rms_norm: q_scale and k_scale need head_dim elements")
    if bias is not None and bias.numel() != weight.shape[0]:
        raise ValueError("linear_qk_rms_norm: bias needs 3 * D elements")


@torch.library.custom_op("comfy_kitchen::linear_qk_rms_norm", mutates_args=())
def _op_linear_qk_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    head_dim: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _check(x, weight, bias, q_scale, k_scale, head_dim)
    kwargs = {"x": x, "weight": weight, "bias": bias, "q_scale": q_scale, "k_scale": k_scale,
              "head_dim": head_dim, "eps": eps}
    impl = registry.get_implementation("linear_qk_rms_norm", kwargs=kwargs)
    q, k, v = impl(**kwargs)
    # custom ops must not return views of each other or of inputs
    return q.clone() if q._base is not None else q, k.clone() if k._base is not None else k, \
        v.clone() if v._base is not None else v


@_op_linear_qk_rms_norm.register_fake
def _op_linear_qk_rms_norm_fake(x, weight, bias, q_scale, k_scale, head_dim, eps):
    b, s, h, _ = _qkv_shape(x, weight, head_dim)
    return tuple(x.new_empty((b, s, h, head_dim)) for _ in range(3))
