# SPDX-License-Identifier: Apache-2.0
# Adapted from anemoi-project/anemoi revision 4e85afba741bdeaf2d9486cab19cb76d3e7985a4.
"""Draft full-precision reference, routing reference, and custom-op boundary.

Precision phases share full-precision arithmetic here, as in Kitchen's Sol
reference. Native quantization must additionally be checked against donor
kernels.
"""

import math
from dataclasses import dataclass

import torch

from ...draft import materialize_layout, resolve_options
from ...registry import registry


def _pool(packed, counts, block, *, maximum=False, second=False):
    batch, heads, tokens, dim = packed.shape
    x = packed.view(batch, heads, tokens // block, block, dim).float()
    valid = torch.arange(block, device=x.device).view(1, 1, 1, block, 1) < counts.view(
        1, 1, -1, 1, 1
    )
    if maximum:
        # Donor max pooling excludes padded zero slots.
        return x.masked_fill(~valid, float("-inf")).amax(-2).half()
    if second:
        x = x.square()
    return (x.masked_fill(~valid, 0).sum(-2) / counts.view(1, 1, -1, 1)).half()


def _pack(x, layout, prefix):
    video = x[:, prefix:].permute(0, 2, 1, 3).index_select(2, layout.indices).half()
    return video.masked_fill(~layout.slot_valid.view(1, 1, -1, 1), 0)


def _k_tail_probability(qm, km, kv, counts, tail_count):
    batch, heads, blocks, dim = km.shape
    keys = kv.view(batch, heads, blocks, 64, dim)
    distances = (keys.float() - km.float().unsqueeze(-2)).square().sum(-1)
    valid = torch.arange(64, device=kv.device) < counts.view(1, 1, -1, 1)
    distances.masked_fill_(~valid, float("-inf"))
    indices = distances.argsort(dim=-1, descending=True, stable=True)[..., :tail_count]
    # Count-one R2 descriptors repeat the first extreme in the donor.
    if tail_count == 2:
        indices[..., 1] = torch.where(counts.view(1, 1, -1) == 1, indices[..., 0], indices[..., 1])
    extreme = keys.gather(-2, indices.unsqueeze(-1).expand(-1, -1, -1, -1, dim))
    descriptors = torch.cat((km.unsqueeze(-2), extreme), -2).flatten(-3, -2)
    logits = (qm.float() @ descriptors.float().transpose(-1, -2) * dim**-0.5).half().float()
    logits = logits.view(batch, heads, blocks, blocks, tail_count + 1)
    mean, tails = logits[..., 0], logits[..., 1:]
    n = counts.view(1, 1, 1, blocks)
    bulk_count = n - tail_count
    bulk = (n * mean - tails.sum(-1)) / bulk_count.clamp_min(1)
    weighted = bulk + bulk_count.clamp_min(1).float().log()
    weighted = weighted.masked_fill(bulk_count <= 0, float("-inf"))
    log_n = torch.logsumexp(torch.cat((weighted.unsqueeze(-1), tails), -1), -1)
    log_n = torch.where(n <= 1, mean, log_n)
    return torch.softmax(log_n, -1).half()


def draft_attention(
    q,
    k,
    v,
    *,
    video_shape,
    prefix_tokens=0,
    sparsity_ratio=0.8,
    query_block_size=64,
    nvfp4_ratio=0.0,
    int8_ratio=1.0,
    mxfp8_ratio=0.0,
    fp16_ratio=0.0,
    prefix_kv_precision="auto",
    prefix_query_precision="auto",
    draftmap_proxy="mean",
    diag_jensen=False,
    maxpool_weight=0.0,
    enable_anchors=False,
    smooth_k=False,
    nvfp4_scales=(1.0, 1.0, 1.0),
):
    options = resolve_options(
        architecture="reference",
        query_block_size=query_block_size,
        nvfp4_ratio=nvfp4_ratio,
        int8_ratio=int8_ratio,
        mxfp8_ratio=mxfp8_ratio,
        fp16_ratio=fp16_ratio,
        prefix_kv_precision=prefix_kv_precision,
        prefix_query_precision=prefix_query_precision,
        draftmap_proxy=draftmap_proxy,
        diag_jensen=diag_jensen,
        maxpool_weight=maxpool_weight,
        enable_anchors=enable_anchors,
        smooth_k=smooth_k,
        nvfp4_scales=nvfp4_scales,
        prefix_tokens=prefix_tokens,
    )
    layout = materialize_layout(q.device, video_shape, query_block_size, enable_anchors)
    qv, kv, vv = (_pack(x, layout, prefix_tokens) for x in (q, k, v))
    qm, km = (_pool(x, layout.counts, query_block_size) for x in (qv, kv))
    qs = _pool(qv, layout.counts, query_block_size, second=True) if diag_jensen else None
    ks = _pool(kv, layout.counts, query_block_size, second=True) if diag_jensen else None
    probability = (
        _k_tail_probability(qm, km, kv, layout.counts, int(draftmap_proxy[-1]))
        if draftmap_proxy != "mean"
        else draft_probability(qm, km, qs, ks)
    )
    if options.maxpool_weight:
        qmax, kmax = (_pool(x, layout.counts, query_block_size, maximum=True) for x in (qv, kv))
        # cuBLAS writes each proxy's scaled logits to half before row softmax;
        # the native kernel mixes the two distributions and rounds once.
        scale = q.shape[-1] ** -0.5
        mean_logits = (
            (torch.matmul(qm.float(), km.float().transpose(-1, -2)) * scale).half().float()
        )
        max_logits = (
            (torch.matmul(qmax.float(), kmax.float().transpose(-1, -2)) * scale).half().float()
        )
        probability = (
            torch.softmax(mean_logits, -1)
            .lerp(torch.softmax(max_logits, -1), options.maxpool_weight)
            .half()
        )
    route = route_probability(
        probability,
        sparsity_ratio=sparsity_ratio,
        nvfp4_ratio=nvfp4_ratio,
        middle_ratio=int8_ratio + mxfp8_ratio,
        fp16_ratio=fp16_ratio,
        anchors=layout.anchors,
        anchor_count=layout.anchor_count,
    )
    counts = route.nvfp4_counts + route.middle_counts + route.fp16_counts
    block_mask = torch.zeros_like(route.block_ids, dtype=torch.bool)
    keep = torch.arange(route.block_ids.shape[-1], device=q.device) < counts.unsqueeze(-1)
    block_mask.scatter_(-1, route.block_ids.long(), keep)
    mask = block_mask.repeat_interleave(query_block_size, -2).repeat_interleave(
        query_block_size, -1
    )
    mask &= layout.slot_valid.view(1, 1, 1, -1)
    if prefix_tokens:
        kp, vp = (x[:, :prefix_tokens].permute(0, 2, 1, 3).half() for x in (k, v))
        kv, vv = torch.cat((kp, kv), 2), torch.cat((vp, vv), 2)
        mask = torch.cat(
            (
                torch.ones((*mask.shape[:-1], prefix_tokens), dtype=torch.bool, device=q.device),
                mask,
            ),
            -1,
        )
    logits = torch.matmul(qv.float(), kv.float().transpose(-1, -2)) * q.shape[-1] ** -0.5
    has_keys = mask.any(-1, keepdim=True)
    logits.masked_fill_(~mask, torch.finfo(torch.float32).min)
    probs = torch.softmax(logits, -1).masked_fill_(~has_keys, 0)
    result = (probs @ vv.float()).to(q.dtype).index_select(2, layout.inverse).permute(0, 2, 1, 3)
    if prefix_tokens:
        prefix = (
            torch.nn.functional.scaled_dot_product_attention(
                q[:, :prefix_tokens].permute(0, 2, 1, 3).float(),
                k.permute(0, 2, 1, 3).float(),
                v.permute(0, 2, 1, 3).float(),
            )
            .to(q.dtype)
            .permute(0, 2, 1, 3)
        )
        result = torch.cat((prefix, result), 1)
    return result.contiguous()


def route_counts(items, sparsity_ratio, nvfp4_ratio, middle_ratio, fp16_ratio):
    """Hamilton allocation of retained pairs, returned as FP16/middle/NVFP4."""
    if type(items) is not int or items <= 0:
        raise ValueError("items must be a positive integer")
    values = (sparsity_ratio, nvfp4_ratio, middle_ratio, fp16_ratio)
    if any(
        isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
        for x in values
    ):
        raise ValueError("route ratios must be finite real scalars")
    if not 0 <= sparsity_ratio < 1:
        raise ValueError("sparsity_ratio must be in [0, 1)")
    if min(nvfp4_ratio, middle_ratio, fp16_ratio) < 0 or not math.isclose(
        nvfp4_ratio + middle_ratio + fp16_ratio, 1.0, rel_tol=0, abs_tol=1e-6
    ):
        raise ValueError("precision ratios must be nonnegative and sum to one")
    retained = min(items, max(1, math.floor((1 - sparsity_ratio) * items + 0.5)))
    quotas = (retained * fp16_ratio, retained * middle_ratio, retained * nvfp4_ratio)
    counts = [math.floor(x) for x in quotas]
    order = sorted(range(3), key=lambda i: (-(quotas[i] - counts[i]), i))
    for i in order[: retained - sum(counts)]:
        counts[i] += 1
    return tuple(counts)


def draft_probability(q_pool, k_pool, q_second=None, k_second=None):
    if q_pool.shape != k_pool.shape or q_pool.ndim != 4:
        raise ValueError("Q/K pool must have matching [B,H,R,D] shapes")
    dim = q_pool.shape[-1]
    scores = torch.matmul(q_pool.float(), k_pool.float().transpose(-2, -1)) * dim**-0.5
    if (q_second is None) != (k_second is None):
        raise ValueError("Jensen correction requires both second moments")
    if q_second is not None:
        if q_second.shape != q_pool.shape or k_second.shape != k_pool.shape:
            raise ValueError("second moments must match pool shapes")
        variance = torch.matmul(q_second.float(), k_second.float().transpose(-2, -1))
        variance.sub_(
            torch.matmul(q_pool.float().square(), k_pool.float().square().transpose(-2, -1))
        )
        scores.add_(variance.div_(dim).clamp_min_(0), alpha=0.5)
    return torch.softmax(scores, dim=-1).to(torch.float16)


@dataclass(frozen=True, slots=True)
class DraftRoute:
    block_ids: torch.Tensor
    nvfp4_counts: torch.Tensor
    middle_counts: torch.Tensor
    fp16_counts: torch.Tensor


def route_probability(
    probability,
    *,
    sparsity_ratio,
    nvfp4_ratio=0.0,
    middle_ratio=1.0,
    fp16_ratio=0.0,
    anchors=None,
    anchor_count=0,
):
    if probability.ndim != 4 or probability.shape[-1] != probability.shape[-2]:
        raise ValueError("probability must be square [B,H,R,R]")
    batch, heads, rows, columns = probability.shape
    if type(anchor_count) is not int or not 0 <= anchor_count <= rows * columns:
        raise ValueError("invalid anchor_count")
    if anchors is None:
        if anchor_count:
            raise ValueError("anchor_count must be zero without anchors")
    elif (
        anchors.shape != (rows, columns)
        or anchors.dtype != torch.bool
        or anchors.device != probability.device
    ):
        raise ValueError("anchors must be a bool matrix matching the DraftMap device and geometry")
    n16, n8, n4 = route_counts(
        rows * columns, sparsity_ratio, nvfp4_ratio, middle_ratio, fp16_ratio
    )
    retained = n16 + n8 + n4
    lowest_code, lowest_count = (1, n4) if n4 else ((2, n8) if n8 else (3, n16))
    if anchors is not None and anchor_count > lowest_count:
        raise ValueError("anchor_count exceeds the lowest-precision budget")
    selected = probability.reshape(batch, heads, -1).argsort(dim=-1, descending=True, stable=True)[
        ..., :retained
    ]
    precision = torch.zeros(
        (batch, heads, rows * columns), device=probability.device, dtype=torch.uint8
    )
    for begin, end, code in ((0, n16, 3), (n16, n16 + n8, 2), (n16 + n8, retained, 1)):
        precision.scatter_(-1, selected[..., begin:end], code)
    if anchors is not None:
        mask = anchors.reshape(1, 1, -1).expand(batch, heads, -1)
        missing = mask & precision.eq(0)
        low_ids = selected[..., retained - lowest_count :]
        ordinary_reversed = (~mask.gather(-1, low_ids)).flip(-1)
        evict = (
            ordinary_reversed & (ordinary_reversed.cumsum(-1) <= missing.sum(-1, keepdim=True))
        ).flip(-1)
        precision.scatter_(-1, low_ids, precision.gather(-1, low_ids).masked_fill_(evict, 0))
        precision.masked_fill_(missing, lowest_code)
    precision = precision.view(batch, heads, rows, columns)
    counts = tuple((precision == code).sum(-1, dtype=torch.int32) for code in (1, 2, 3))
    column_ids = torch.arange(columns, device=probability.device, dtype=torch.int32).view(
        1, 1, 1, -1
    )
    phase_order = torch.where(precision == 0, 3, precision.to(torch.int32) - 1)
    ids = (phase_order * columns + column_ids).argsort(-1).to(torch.int32)
    return DraftRoute(ids.contiguous(), *counts)


@torch.library.custom_op("comfy_kitchen::draft_attention", mutates_args=())
def _op_draft_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    video_shape: list[int],
    prefix_tokens: int = 0,
    sparsity_ratio: float = 0.8,
    query_block_size: int = 64,
    nvfp4_ratio: float = 0.0,
    int8_ratio: float = 1.0,
    mxfp8_ratio: float = 0.0,
    fp16_ratio: float = 0.0,
    prefix_kv_precision: str = "auto",
    prefix_query_precision: str = "auto",
    draftmap_proxy: str = "mean",
    diag_jensen: bool = False,
    maxpool_weight: float = 0.0,
    enable_anchors: bool = False,
    smooth_k: bool = False,
    nvfp4_scales: list[float] | None = None,
) -> torch.Tensor:
    kwargs = {
        "q": q,
        "k": k,
        "v": v,
        "video_shape": video_shape,
        "prefix_tokens": prefix_tokens,
        "sparsity_ratio": sparsity_ratio,
        "query_block_size": query_block_size,
        "nvfp4_ratio": nvfp4_ratio,
        "int8_ratio": int8_ratio,
        "mxfp8_ratio": mxfp8_ratio,
        "fp16_ratio": fp16_ratio,
        "prefix_kv_precision": prefix_kv_precision,
        "prefix_query_precision": prefix_query_precision,
        "draftmap_proxy": draftmap_proxy,
        "diag_jensen": diag_jensen,
        "maxpool_weight": maxpool_weight,
        "enable_anchors": enable_anchors,
        "smooth_k": smooth_k,
        "nvfp4_scales": (1.0, 1.0, 1.0) if nvfp4_scales is None else tuple(nvfp4_scales),
    }
    implementation = registry.get_implementation("draft_attention", kwargs=kwargs)
    return implementation(**kwargs)


@_op_draft_attention.register_fake
def _op_draft_attention_fake(
    q,
    k,
    v,
    video_shape,
    prefix_tokens=0,
    sparsity_ratio=0.8,
    query_block_size=64,
    nvfp4_ratio=0.0,
    int8_ratio=1.0,
    mxfp8_ratio=0.0,
    fp16_ratio=0.0,
    prefix_kv_precision="auto",
    prefix_query_precision="auto",
    draftmap_proxy="mean",
    diag_jensen=False,
    maxpool_weight=0.0,
    enable_anchors=False,
    smooth_k=False,
    nvfp4_scales=None,
):
    return torch.empty(q.shape, dtype=q.dtype, device=q.device)
