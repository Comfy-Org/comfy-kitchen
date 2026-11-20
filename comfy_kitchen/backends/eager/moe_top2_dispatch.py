# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Top-2 MoE routing and expert-major dispatch of the token rows.

The HiDream ``MoEGate`` -> ``MOEFeedForwardSwiGLU`` dispatch side (``comfy/ldm/hidream/model.py``),
from the router logits up to the expert calls:

    scores = logits.softmax(dim=-1)
    topk_weight, topk_idx = torch.topk(scores, k=2, dim=-1, sorted=False)
    flat_topk_idx = topk_idx.view(-1)
    x = x.repeat_interleave(2, dim=0)
    ... expert i runs on x[flat_topk_idx == i], its output is written to y[flat_topk_idx == i]

Instead of the per-expert boolean-mask gathers this returns the rows of all experts in one
expert-major buffer, ``x_sorted = cat([x.repeat_interleave(2, dim=0)[flat_topk_idx == i] for i])``
(rows of expert i in their token order, ``counts[i]`` of them), and ``slot`` [2N], the row of
x_sorted that holds dispatched row ``t * 2 + k``. With ``out`` the expert outputs in x_sorted's
order, ``out.index_select(0, slot)`` is the ``y`` the stock loop builds.
"""

import torch

from comfy_kitchen.registry import registry

TOP_K = 2


def moe_top2_dispatch(
    x: torch.Tensor, logits: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference: the stock softmax, ``topk(k=2, sorted=False)``, ``repeat_interleave`` and the
    per-expert ``flat_topk_idx == i`` masks, in that order."""
    num_experts = logits.shape[-1]
    scores = logits.softmax(dim=-1)
    topk_weight, topk_idx = torch.topk(scores, k=TOP_K, dim=-1, sorted=False)
    flat_topk_idx = topk_idx.view(-1)
    x = x.repeat_interleave(TOP_K, dim=0)
    masks = [flat_topk_idx == i for i in range(num_experts)]
    x_sorted = torch.cat([x[mask] for mask in masks])
    counts = torch.stack([mask.sum() for mask in masks])
    slot = torch.empty_like(flat_topk_idx)
    start = 0
    for mask, n in zip(masks, counts.tolist(), strict=True):
        slot[mask] = torch.arange(start, start + n, device=slot.device)
        start += n
    return topk_weight, topk_idx, x_sorted, slot, counts


def _check(x, logits):
    name = "moe_top2_dispatch"
    if x.dim() != 2:
        raise ValueError(f"{name}: x must be [N, D], got {tuple(x.shape)}")
    if logits.dim() != 2 or logits.shape[0] != x.shape[0]:
        raise ValueError(
            f"{name}: logits must be [N, E] for x {tuple(x.shape)}, got {tuple(logits.shape)}"
        )
    if logits.shape[1] < TOP_K:
        raise ValueError(f"{name}: needs at least {TOP_K} experts, got {logits.shape[1]}")
    if logits.device != x.device:
        raise ValueError(f"{name}: x and logits must be on the same device")


@torch.library.custom_op("comfy_kitchen::moe_top2_dispatch", mutates_args=())
def _op_moe_top2_dispatch(
    x: torch.Tensor, logits: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _check(x, logits)
    kwargs = {"x": x, "logits": logits}
    impl = registry.get_implementation("moe_top2_dispatch", kwargs=kwargs)
    return impl(**kwargs)


@_op_moe_top2_dispatch.register_fake
def _op_moe_top2_dispatch_fake(x, logits):
    _check(x, logits)
    n, d = x.shape
    index = torch.int64
    return (
        logits.new_empty((n, TOP_K)),
        torch.empty((n, TOP_K), dtype=index, device=x.device),
        x.new_empty((n * TOP_K, d)),
        torch.empty((n * TOP_K,), dtype=index, device=x.device),
        torch.empty((logits.shape[1],), dtype=index, device=x.device),
    )
