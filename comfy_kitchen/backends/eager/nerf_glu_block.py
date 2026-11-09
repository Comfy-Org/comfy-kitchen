# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chroma Radiance NeRF GLU block: a per-patch GLU MLP whose weights are generated per patch.

``NerfGLUBlock.forward`` in ComfyUI (``comfy/ldm/chroma_radiance/layers.py``) after its
``param_generator`` Linear. x ``[B, N, H]`` holds the N pixels of each of B patches;
mlp_params ``[B, 3 * H * D]`` (``D = H * mlp_ratio``) holds one generated MLP per patch:

    gate, value, w2 = mlp_params.chunk(3, dim=-1)
    gate  = normalize(gate.view(B, H, D), dim=-2)     # unit-norm columns
    value = normalize(value.view(B, H, D), dim=-2)
    w2    = normalize(w2.view(B, D, H), dim=-2)
    out   = bmm(silu(bmm(rms_norm(x), gate)) * bmm(rms_norm(x), value), w2) + x

``rms_norm`` is the block's ``RMSNorm`` over H with weight norm_weight; ComfyUI builds it with
``eps=None``, which ``F.rms_norm`` resolves itself.

This backend runs that sequence as it is.
"""

import torch
from torch.nn import functional

from comfy_kitchen.registry import registry


def nerf_glu_block(
    x: torch.Tensor,
    mlp_params: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float | None = None,
) -> torch.Tensor:
    """Reference: ``NerfGLUBlock.forward`` from the ``chunk`` of the generated parameters on."""
    batch_size, _, hidden_size_x = x.shape
    hidden = mlp_params.shape[-1] // (3 * hidden_size_x)
    fc1_gate_params, fc1_value_params, fc2_params = mlp_params.chunk(3, dim=-1)
    fc1_gate = fc1_gate_params.view(batch_size, hidden_size_x, hidden)
    fc1_value = fc1_value_params.view(batch_size, hidden_size_x, hidden)
    fc2 = fc2_params.view(batch_size, hidden, hidden_size_x)
    fc1_gate = functional.normalize(fc1_gate, dim=-2)
    fc1_value = functional.normalize(fc1_value, dim=-2)
    fc2 = functional.normalize(fc2, dim=-2)
    res_x = x
    x = functional.rms_norm(x, (hidden_size_x,), norm_weight, eps)
    x = torch.bmm(functional.silu(torch.bmm(x, fc1_gate)) * torch.bmm(x, fc1_value), fc2)
    return x + res_x


def _check(x, mlp_params, norm_weight):
    name = "nerf_glu_block"
    if x.dim() != 3 or mlp_params.dim() != 2 or norm_weight.dim() != 1:
        raise ValueError(
            f"{name}: x must be [B, N, H], mlp_params [B, 3 * H * D] and norm_weight [H], got "
            f"{tuple(x.shape)}, {tuple(mlp_params.shape)} and {tuple(norm_weight.shape)}"
        )
    batch_size, _, hidden_size_x = x.shape
    if hidden_size_x == 0 or mlp_params.shape[0] != batch_size:
        raise ValueError(
            f"{name}: mlp_params must have one row per patch of x {tuple(x.shape)}, "
            f"got {tuple(mlp_params.shape)}"
        )
    if mlp_params.shape[1] == 0 or mlp_params.shape[1] % (3 * hidden_size_x):
        raise ValueError(
            f"{name}: mlp_params rows must hold 3 * {hidden_size_x} * D values, "
            f"got {mlp_params.shape[1]}"
        )
    if norm_weight.shape[0] != hidden_size_x:
        raise ValueError(
            f"{name}: norm_weight must be [{hidden_size_x}], got {tuple(norm_weight.shape)}"
        )
    if mlp_params.dtype != x.dtype or norm_weight.dtype != x.dtype:
        raise ValueError(f"{name}: mlp_params and norm_weight must have x's dtype")


@torch.library.custom_op("comfy_kitchen::nerf_glu_block", mutates_args=())
def _op_nerf_glu_block(
    x: torch.Tensor,
    mlp_params: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float | None = None,
) -> torch.Tensor:
    _check(x, mlp_params, norm_weight)
    kwargs = {"x": x, "mlp_params": mlp_params, "norm_weight": norm_weight, "eps": eps}
    impl = registry.get_implementation("nerf_glu_block", kwargs=kwargs)
    return impl(**kwargs)


@_op_nerf_glu_block.register_fake
def _op_nerf_glu_block_fake(x, mlp_params, norm_weight, eps=None):
    _check(x, mlp_params, norm_weight)
    return x.new_empty(x.shape)
