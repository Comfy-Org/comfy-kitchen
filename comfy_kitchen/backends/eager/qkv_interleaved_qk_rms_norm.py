# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hunyuan3D-2.1 DiT attention prep: per-head q / k / v relayout of the projections + QK RMSNorm.

The Hunyuan3D-2.1 DiT (``comfy/ldm/hunyuan3dv2_1/hunyuandit.py``) concatenates the
projection outputs and splits the result per head, so head h of q, k and v is not the h-th
``[head_dim]`` slice of the q, k and v projections but an interleaved block of the
concatenation. Self-attention (``Attention.forward``), with q, k, v = ``to_q(x)``,
``to_k(x)``, ``to_v(x)``:

    qkv_combined = torch.cat((query, key, value), dim=-1)
    split_size = qkv_combined.shape[-1] // self.num_heads // 3
    qkv = qkv_combined.view(B, -1, self.num_heads, split_size * 3)
    query, key, value = torch.split(qkv, split_size, dim=-1)
    query = query.reshape(B, N, self.num_heads, self.head_dim)
    key = key.reshape(B, N, self.num_heads, self.head_dim)
    value = value.reshape(B, N, self.num_heads * self.head_dim)
    query = self.q_norm(query)
    key = self.k_norm(key)
    -> query.reshape(B, N, -1), key.reshape(B, N, -1), value

Cross-attention (``CrossAttention.forward``), with q = ``to_q(x)``, k, v = ``to_k(y)``,
``to_v(y)``: only k and v are concatenated and split per head, q is viewed per head:

    kv = torch.cat((k, v), dim=-1)
    split_size = kv.shape[-1] // self.num_heads // 2
    kv = kv.view(b, -1, self.num_heads, split_size * 2)
    k, v = torch.split(kv, split_size, dim=-1)
    q = q.view(b, s1, self.num_heads, self.head_dim)
    k = k.view(b, s2, self.num_heads, self.head_dim)
    v = v.reshape(b, s2, self.num_heads * self.head_dim)
    q = self.q_norm(q)
    k = self.k_norm(k)
    -> q.reshape(b, s1, -1), k.reshape(b, s2, -1), v

The norms are ``F.rms_norm`` over ``head_dim`` with the weight already in the projections'
dtype (ComfyUI's RMSNorm casts it to the input's dtype).
"""

import torch
from torch.nn import functional

from comfy_kitchen.registry import registry


def qkv_interleaved_qk_rms_norm(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    heads: int,
    eps: float = 1e-6,
    q_interleaved: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference: the cat, view, split, reshape and ``F.rms_norm`` sequence, in ComfyUI's
    order (``Attention.forward`` if ``q_interleaved`` else ``CrossAttention.forward``)."""
    if q_interleaved:
        B, N, _ = q.shape
        head_dim = q.shape[-1] // heads
        qkv_combined = torch.cat((q, k, v), dim=-1)
        split_size = qkv_combined.shape[-1] // heads // 3
        qkv = qkv_combined.view(B, -1, heads, split_size * 3)
        query, key, value = torch.split(qkv, split_size, dim=-1)
        query = query.reshape(B, N, heads, head_dim)
        key = key.reshape(B, N, heads, head_dim)
        value = value.reshape(B, N, heads * head_dim)
        query = functional.rms_norm(query, (head_dim,), q_weight, eps)
        key = functional.rms_norm(key, (head_dim,), k_weight, eps)
        return query.reshape(B, N, -1), key.reshape(B, N, -1), value

    b, s1, _ = q.shape
    s2 = k.shape[1]
    head_dim = q.shape[-1] // heads
    kv = torch.cat((k, v), dim=-1)
    split_size = kv.shape[-1] // heads // 2
    kv = kv.view(b, -1, heads, split_size * 2)
    k, v = torch.split(kv, split_size, dim=-1)
    q = q.view(b, s1, heads, head_dim)
    k = k.view(b, s2, heads, head_dim)
    v = v.reshape(b, s2, heads * head_dim)
    q = functional.rms_norm(q, (head_dim,), q_weight, eps)
    k = functional.rms_norm(k, (head_dim,), k_weight, eps)
    return q.reshape(b, s1, -1), k.reshape(b, s2, -1), v


def _check(q, k, v, q_weight, k_weight, heads, q_interleaved):
    name = "qkv_interleaved_qk_rms_norm"
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError(f"{name}: q, k and v must be [B, S, heads * head_dim]")
    if k.shape != v.shape:
        raise ValueError(
            f"{name}: k and v must have one shape, got {tuple(k.shape)} and {tuple(v.shape)}"
        )
    if q_interleaved and q.shape != k.shape:
        raise ValueError(
            f"{name}: q, k and v must have one shape when q is interleaved, "
            f"got {tuple(q.shape)} and {tuple(k.shape)}"
        )
    if k.shape[0] != q.shape[0] or k.shape[2] != q.shape[2]:
        raise ValueError(
            f"{name}: k and v must be [B, S_kv, heads * head_dim] with q's B and channels, "
            f"got {tuple(q.shape)} and {tuple(k.shape)}"
        )
    for t in (k, v):
        if t.dtype != q.dtype or t.device != q.device:
            raise ValueError(f"{name}: q, k and v must have the same dtype and device")
    channels = q.shape[2]
    if heads < 1 or channels % heads:
        raise ValueError(f"{name}: channels {channels} must split into {heads} heads")
    head_dim = channels // heads
    for t in (q_weight, k_weight):
        if t.dim() != 1 or t.shape[0] != head_dim:
            raise ValueError(
                f"{name}: weights must be [head_dim] = [{head_dim}], got {tuple(t.shape)}"
            )
        if t.dtype != q.dtype or t.device != q.device:
            raise ValueError(f"{name}: weights must have q's dtype and device")


@torch.library.custom_op("comfy_kitchen::qkv_interleaved_qk_rms_norm", mutates_args=())
def _op_qkv_interleaved_qk_rms_norm(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    heads: int,
    eps: float = 1e-6,
    q_interleaved: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _check(q, k, v, q_weight, k_weight, heads, q_interleaved)
    kwargs = {
        "q": q,
        "k": k,
        "v": v,
        "q_weight": q_weight,
        "k_weight": k_weight,
        "heads": heads,
        "eps": eps,
        "q_interleaved": q_interleaved,
    }
    impl = registry.get_implementation("qkv_interleaved_qk_rms_norm", kwargs=kwargs)
    return impl(**kwargs)


@_op_qkv_interleaved_qk_rms_norm.register_fake
def _op_qkv_interleaved_qk_rms_norm_fake(
    q, k, v, q_weight, k_weight, heads, eps=1e-6, q_interleaved=True
):
    _check(q, k, v, q_weight, k_weight, heads, q_interleaved)
    return q.new_empty(q.shape), k.new_empty(k.shape), v.new_empty(v.shape)
