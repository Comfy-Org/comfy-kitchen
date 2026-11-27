# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dual-stream attention prep: per-head QK RMSNorm, txt / img concat and RoPE, plus the v concat.

The Qwen-Image joint attention (``comfy/ldm/qwen_image/model.py`` ``Attention.forward``),
from the six projection outputs to the rotated joint q / k and the joint v:

    img_query = img_q.view(B, L_img, heads, -1).transpose(1, 2).contiguous()   # same for k
    img_value = img_v.view(B, L_img, heads, -1).transpose(1, 2)
    txt_query = txt_q.view(B, L_txt, heads, -1).transpose(1, 2).contiguous()   # same for k
    txt_value = txt_v.view(B, L_txt, heads, -1).transpose(1, 2)
    img_query = norm_q(img_query)            # RMSNorm over head_dim, one weight per stream
    img_key = norm_k(img_key)                # and per q / k
    txt_query = norm_added_q(txt_query)
    txt_key = norm_added_k(txt_key)
    joint_query = cat([txt_query, img_query], dim=2)
    joint_key = cat([txt_key, img_key], dim=2)
    joint_value = cat([txt_value, img_value], dim=2)
    joint_query = apply_rope1(joint_query, freqs_cis)
    joint_key = apply_rope1(joint_key, freqs_cis)

The projections are ``[B, L, heads * head_dim]``; the outputs are contiguous
``[B, heads, L_txt + L_img, head_dim]`` with the text rows first. ``freqs_cis`` is the
``[B or 1, 1, L_txt + L_img, head_dim / 2, 2, 2]`` rotation table of the interleaved RoPE
(pair k = elements 2k, 2k + 1). The norms are ``F.rms_norm`` with the weights already in the
projections' dtype (ComfyUI's RMSNorm casts them to the input's dtype); the rotation is
comfy-kitchen's eager ``apply_rope1`` (``f[..., 0] * x0``, then ``addcmul_`` of
``f[..., 1] * x1``, in the table's dtype), the rounding the HIP ``apply_rope1`` kernel
reproduces (``backends/hip/rope_math.h``).
"""

import torch
from torch.nn import functional

from comfy_kitchen.registry import registry

from .rope import apply_rope1


def joint_qk_rms_rope_cat(
    img_q: torch.Tensor,
    img_k: torch.Tensor,
    img_v: torch.Tensor,
    txt_q: torch.Tensor,
    txt_k: torch.Tensor,
    txt_v: torch.Tensor,
    img_q_weight: torch.Tensor,
    img_k_weight: torch.Tensor,
    txt_q_weight: torch.Tensor,
    txt_k_weight: torch.Tensor,
    freqs_cis: torch.Tensor,
    heads: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference: the view / transpose / contiguous, ``F.rms_norm``, ``torch.cat`` and
    ``apply_rope1`` sequence, in ComfyUI's order."""
    batch_size, seq_img, seq_txt = img_q.shape[0], img_q.shape[1], txt_q.shape[1]
    img_query = img_q.view(batch_size, seq_img, heads, -1).transpose(1, 2).contiguous()
    img_key = img_k.view(batch_size, seq_img, heads, -1).transpose(1, 2).contiguous()
    img_value = img_v.view(batch_size, seq_img, heads, -1).transpose(1, 2)
    txt_query = txt_q.view(batch_size, seq_txt, heads, -1).transpose(1, 2).contiguous()
    txt_key = txt_k.view(batch_size, seq_txt, heads, -1).transpose(1, 2).contiguous()
    txt_value = txt_v.view(batch_size, seq_txt, heads, -1).transpose(1, 2)

    head_dim = (img_query.shape[-1],)
    img_query = functional.rms_norm(img_query, head_dim, img_q_weight, eps)
    img_key = functional.rms_norm(img_key, head_dim, img_k_weight, eps)
    txt_query = functional.rms_norm(txt_query, head_dim, txt_q_weight, eps)
    txt_key = functional.rms_norm(txt_key, head_dim, txt_k_weight, eps)

    joint_query = torch.cat([txt_query, img_query], dim=2)
    joint_key = torch.cat([txt_key, img_key], dim=2)
    joint_value = torch.cat([txt_value, img_value], dim=2)

    joint_query = apply_rope1(joint_query, freqs_cis)
    joint_key = apply_rope1(joint_key, freqs_cis)
    return joint_query, joint_key, joint_value


def _check(
    img_q,
    img_k,
    img_v,
    txt_q,
    txt_k,
    txt_v,
    img_q_weight,
    img_k_weight,
    txt_q_weight,
    txt_k_weight,
    freqs_cis,
    heads,
):
    name = "joint_qk_rms_rope_cat"
    if img_q.dim() != 3 or txt_q.dim() != 3:
        raise ValueError(
            f"{name}: projections must be [B, L, heads * head_dim], "
            f"got {tuple(img_q.shape)} and {tuple(txt_q.shape)}"
        )
    for t in (img_k, img_v):
        if t.shape != img_q.shape:
            raise ValueError(
                f"{name}: img_k and img_v must have img_q's shape {tuple(img_q.shape)}"
            )
    for t in (txt_k, txt_v):
        if t.shape != txt_q.shape:
            raise ValueError(
                f"{name}: txt_k and txt_v must have txt_q's shape {tuple(txt_q.shape)}"
            )
    if txt_q.shape[0] != img_q.shape[0] or txt_q.shape[2] != img_q.shape[2]:
        raise ValueError(
            f"{name}: txt and img projections must agree on B and the channel count, "
            f"got {tuple(txt_q.shape)} and {tuple(img_q.shape)}"
        )
    channels = img_q.shape[2]
    if heads < 1 or channels % heads or (channels // heads) % 2:
        raise ValueError(
            f"{name}: channels {channels} must split into {heads} heads of an even head_dim"
        )
    head_dim = channels // heads
    projections = (img_q, img_k, img_v, txt_q, txt_k, txt_v)
    for t in projections:
        if t.dtype != img_q.dtype or t.device != img_q.device:
            raise ValueError(f"{name}: projections must have the same dtype and device")
    for w in (img_q_weight, img_k_weight, txt_q_weight, txt_k_weight):
        if w.dim() != 1 or w.shape[0] != head_dim:
            raise ValueError(
                f"{name}: weights must be [head_dim] = [{head_dim}], got {tuple(w.shape)}"
            )
        if w.dtype != img_q.dtype or w.device != img_q.device:
            raise ValueError(f"{name}: weights must have the projections' dtype and device")
    seq_total = txt_q.shape[1] + img_q.shape[1]
    expected = (seq_total, head_dim // 2, 2, 2)
    if (
        freqs_cis.dim() != 6
        or freqs_cis.shape[0] not in (1, img_q.shape[0])
        or freqs_cis.shape[1] != 1
        or tuple(freqs_cis.shape[2:]) != expected
    ):
        raise ValueError(
            f"{name}: freqs_cis must be [B or 1, 1, L_txt + L_img, head_dim / 2, 2, 2] = "
            f"[{img_q.shape[0]} or 1, 1, {', '.join(map(str, expected))}], "
            f"got {tuple(freqs_cis.shape)}"
        )
    if not freqs_cis.is_floating_point() or freqs_cis.device != img_q.device:
        raise ValueError(f"{name}: freqs_cis must be floating point on the projections' device")


@torch.library.custom_op("comfy_kitchen::joint_qk_rms_rope_cat", mutates_args=())
def _op_joint_qk_rms_rope_cat(
    img_q: torch.Tensor,
    img_k: torch.Tensor,
    img_v: torch.Tensor,
    txt_q: torch.Tensor,
    txt_k: torch.Tensor,
    txt_v: torch.Tensor,
    img_q_weight: torch.Tensor,
    img_k_weight: torch.Tensor,
    txt_q_weight: torch.Tensor,
    txt_k_weight: torch.Tensor,
    freqs_cis: torch.Tensor,
    heads: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _check(
        img_q,
        img_k,
        img_v,
        txt_q,
        txt_k,
        txt_v,
        img_q_weight,
        img_k_weight,
        txt_q_weight,
        txt_k_weight,
        freqs_cis,
        heads,
    )
    kwargs = {
        "img_q": img_q,
        "img_k": img_k,
        "img_v": img_v,
        "txt_q": txt_q,
        "txt_k": txt_k,
        "txt_v": txt_v,
        "img_q_weight": img_q_weight,
        "img_k_weight": img_k_weight,
        "txt_q_weight": txt_q_weight,
        "txt_k_weight": txt_k_weight,
        "freqs_cis": freqs_cis,
        "heads": heads,
        "eps": eps,
    }
    impl = registry.get_implementation("joint_qk_rms_rope_cat", kwargs=kwargs)
    return impl(**kwargs)


@_op_joint_qk_rms_rope_cat.register_fake
def _op_joint_qk_rms_rope_cat_fake(
    img_q,
    img_k,
    img_v,
    txt_q,
    txt_k,
    txt_v,
    img_q_weight,
    img_k_weight,
    txt_q_weight,
    txt_k_weight,
    freqs_cis,
    heads,
    eps=1e-6,
):
    _check(
        img_q,
        img_k,
        img_v,
        txt_q,
        txt_k,
        txt_v,
        img_q_weight,
        img_k_weight,
        txt_q_weight,
        txt_k_weight,
        freqs_cis,
        heads,
    )
    batch_size, channels = img_q.shape[0], img_q.shape[2]
    shape = (batch_size, heads, txt_q.shape[1] + img_q.shape[1], channels // heads)
    return img_q.new_empty(shape), img_q.new_empty(shape), img_q.new_empty(shape)
