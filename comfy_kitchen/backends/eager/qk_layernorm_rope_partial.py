# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Joint text + video attention prep: per-head QK LayerNorm, then RoPE on the video tokens only.

The CogVideoX block (``comfy/ldm/cogvideo/model.py`` ``CogVideoXBlock.forward``, used by
CogVideoX and VOID), from the q / k projection outputs to the attention call:

    q = self.q(qkv_input).view(b, s, n, d)            # same for k
    q = self.norm_q(q).view(b, s, n, d)               # LayerNorm over head_dim, affine
    q_img = q[:, text_seq_length:].transpose(1, 2)    # [B, heads, S - text, head_dim]
    q_img = apply_rotary_emb(q_img, (cos, sin))
    q = torch.cat([q[:, :text_seq_length], q_img.transpose(1, 2)], dim=1)
    q = q.reshape(b, s, n * d)

with ``apply_rotary_emb`` the CogVideoX interleaved rotation (pair j = elements 2j, 2j + 1)
from the ``[S - text, head_dim]`` cos / sin tables:

    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
    (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

The text tokens (the first ``text_len`` rows) are only normalized. The norms are
``F.layer_norm`` with the weight and bias already in the projections' dtype (ComfyUI's
LayerNorm casts them to the input's dtype).
"""

import torch
from torch.nn import functional

from comfy_kitchen.registry import registry


def _apply_rotary_emb(x, freqs_cos_sin):
    # comfy/ldm/cogvideo/model.py apply_rotary_emb, as it is
    cos, sin = freqs_cos_sin
    cos = cos[None, None, :, :].to(x.device)
    sin = sin[None, None, :, :].to(x.device)

    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)

    return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)


def qk_layernorm_rope_partial(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    q_bias: torch.Tensor,
    k_weight: torch.Tensor,
    k_bias: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    text_len: int,
    heads: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference: the view, ``F.layer_norm``, slice / transpose, ``apply_rotary_emb``,
    ``torch.cat`` and reshape sequence, in ComfyUI's order."""
    b, s, c = q.shape
    n = heads
    d = c // heads
    q = q.view(b, s, n, d)
    k = k.view(b, s, n, d)

    q = functional.layer_norm(q, (d,), q_weight, q_bias, eps).view(b, s, n, d)
    k = functional.layer_norm(k, (d,), k_weight, k_bias, eps).view(b, s, n, d)

    q_img = q[:, text_len:].transpose(1, 2)
    k_img = k[:, text_len:].transpose(1, 2)
    q_img = _apply_rotary_emb(q_img, (cos, sin))
    k_img = _apply_rotary_emb(k_img, (cos, sin))
    q = torch.cat([q[:, :text_len], q_img.transpose(1, 2)], dim=1)
    k = torch.cat([k[:, :text_len], k_img.transpose(1, 2)], dim=1)
    return q.reshape(b, s, n * d), k.reshape(b, s, n * d)


def _check(q, k, q_weight, q_bias, k_weight, k_bias, cos, sin, text_len, heads):
    name = "qk_layernorm_rope_partial"
    if q.dim() != 3 or k.shape != q.shape:
        raise ValueError(
            f"{name}: q and k must be [B, S, heads * head_dim] of one shape, "
            f"got {tuple(q.shape)} and {tuple(k.shape)}"
        )
    if k.dtype != q.dtype or k.device != q.device:
        raise ValueError(f"{name}: q and k must have the same dtype and device")
    channels = q.shape[2]
    if heads < 1 or channels % heads or (channels // heads) % 2:
        raise ValueError(
            f"{name}: channels {channels} must split into {heads} heads of an even head_dim"
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
    seq = q.shape[1]
    # at least one video token: the stock rotation cannot reshape an empty video slice
    if not 0 <= text_len < seq:
        raise ValueError(f"{name}: text_len {text_len} must be in [0, {seq})")
    expected = (seq - text_len, head_dim)
    for t in (cos, sin):
        if tuple(t.shape) != expected:
            raise ValueError(
                f"{name}: cos and sin must be [S - text_len, head_dim] = "
                f"[{', '.join(map(str, expected))}], got {tuple(t.shape)}"
            )
        if not t.is_floating_point() or t.device != q.device:
            raise ValueError(f"{name}: cos and sin must be floating point on q's device")


@torch.library.custom_op("comfy_kitchen::qk_layernorm_rope_partial", mutates_args=())
def _op_qk_layernorm_rope_partial(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    q_bias: torch.Tensor,
    k_weight: torch.Tensor,
    k_bias: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    text_len: int,
    heads: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    _check(q, k, q_weight, q_bias, k_weight, k_bias, cos, sin, text_len, heads)
    kwargs = {
        "q": q,
        "k": k,
        "q_weight": q_weight,
        "q_bias": q_bias,
        "k_weight": k_weight,
        "k_bias": k_bias,
        "cos": cos,
        "sin": sin,
        "text_len": text_len,
        "heads": heads,
        "eps": eps,
    }
    impl = registry.get_implementation("qk_layernorm_rope_partial", kwargs=kwargs)
    return impl(**kwargs)


@_op_qk_layernorm_rope_partial.register_fake
def _op_qk_layernorm_rope_partial_fake(
    q, k, q_weight, q_bias, k_weight, k_bias, cos, sin, text_len, heads, eps=1e-6
):
    _check(q, k, q_weight, q_bias, k_weight, k_bias, cos, sin, text_len, heads)
    return q.new_empty(q.shape), k.new_empty(k.shape)
