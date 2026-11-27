# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hunyuan3D VAE cross-attention prep: per-head QK LayerNorm with k read from the fused kv.

The Hunyuan3D shape VAE geometry decoder (``comfy/ldm/hunyuan3d/vae.py``
``QKVMultiheadCrossAttention.forward``) gets q = ``c_q(x)`` and the fused kv = ``c_kv(data)``,
whose ``[2 * head_dim]`` block per head holds k then v:

    _, n_ctx, _ = q.shape
    bs, n_data, width = kv.shape
    attn_ch = width // self.heads // 2
    q = q.view(bs, n_ctx, self.heads, -1)
    kv = kv.view(bs, n_data, self.heads, -1)
    k, v = torch.split(kv, attn_ch, dim=-1)
    q = self.q_norm(q)
    k = self.k_norm(k)
    -> permute and F.scaled_dot_product_attention(q, k, v)

The norms are ``F.layer_norm`` over ``head_dim`` with the weight and bias already in the
inputs' dtype (ComfyUI's LayerNorm casts them to the input's dtype). This op returns the
normalized q ``[bs, n_ctx, heads, head_dim]`` and k ``[bs, n_data, heads, head_dim]``; v
stays the strided view ``kv.view(bs, n_data, heads, 2 * head_dim)[..., head_dim:]``
(no data movement) and is not part of the op.
"""

import torch
from torch.nn import functional

from comfy_kitchen.registry import registry


def qk_layernorm_kv_interleaved(
    q: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    q_bias: torch.Tensor,
    k_weight: torch.Tensor,
    k_bias: torch.Tensor,
    heads: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference: the view, ``torch.split`` and ``F.layer_norm`` sequence, in ComfyUI's order."""
    _, n_ctx, _ = q.shape
    bs, n_data, width = kv.shape
    attn_ch = width // heads // 2
    q = q.view(bs, n_ctx, heads, -1)
    kv = kv.view(bs, n_data, heads, -1)
    k, _ = torch.split(kv, attn_ch, dim=-1)
    q = functional.layer_norm(q, (q.shape[-1],), q_weight, q_bias, eps)
    k = functional.layer_norm(k, (k.shape[-1],), k_weight, k_bias, eps)
    return q, k


def _check(q, kv, q_weight, q_bias, k_weight, k_bias, heads):
    name = "qk_layernorm_kv_interleaved"
    if q.dim() != 3 or kv.dim() != 3:
        raise ValueError(
            f"{name}: q must be [B, S_q, heads * head_dim] and kv [B, S_kv, 2 * heads * head_dim], "
            f"got {tuple(q.shape)} and {tuple(kv.shape)}"
        )
    if kv.dtype != q.dtype or kv.device != q.device:
        raise ValueError(f"{name}: q and kv must have the same dtype and device")
    channels = q.shape[2]
    if heads < 1 or channels % heads:
        raise ValueError(f"{name}: channels {channels} must split into {heads} heads")
    if kv.shape[0] != q.shape[0] or kv.shape[2] != 2 * channels:
        raise ValueError(
            f"{name}: kv must be [B, S_kv, 2 * {channels}] with q's B, got {tuple(kv.shape)}"
        )
    head_dim = channels // heads
    for t in (q_weight, q_bias, k_weight, k_bias):
        if t.dim() != 1 or t.shape[0] != head_dim:
            raise ValueError(
                f"{name}: weights and biases must be [head_dim] = [{head_dim}], "
                f"got {tuple(t.shape)}"
            )
        if t.dtype != q.dtype or t.device != q.device:
            raise ValueError(f"{name}: weights and biases must have q's dtype and device")


@torch.library.custom_op("comfy_kitchen::qk_layernorm_kv_interleaved", mutates_args=())
def _op_qk_layernorm_kv_interleaved(
    q: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    q_bias: torch.Tensor,
    k_weight: torch.Tensor,
    k_bias: torch.Tensor,
    heads: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    _check(q, kv, q_weight, q_bias, k_weight, k_bias, heads)
    kwargs = {
        "q": q,
        "kv": kv,
        "q_weight": q_weight,
        "q_bias": q_bias,
        "k_weight": k_weight,
        "k_bias": k_bias,
        "heads": heads,
        "eps": eps,
    }
    impl = registry.get_implementation("qk_layernorm_kv_interleaved", kwargs=kwargs)
    return impl(**kwargs)


@_op_qk_layernorm_kv_interleaved.register_fake
def _op_qk_layernorm_kv_interleaved_fake(
    q, kv, q_weight, q_bias, k_weight, k_bias, heads, eps=1e-6
):
    _check(q, kv, q_weight, q_bias, k_weight, k_bias, heads)
    bs, n_ctx, channels = q.shape
    head_dim = channels // heads
    return (
        q.new_empty((bs, n_ctx, heads, head_dim)),
        q.new_empty((bs, kv.shape[1], heads, head_dim)),
    )
