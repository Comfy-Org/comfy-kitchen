# SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Block-sparse Sol-Attn for MPS.

Implements Sol-Attn routing + exact/pooled attention without materializing an
O(T²) score tensor. Routing uses the cheap N×N block proxy; each query block
then attends only its exact key tokens plus one pooled summary logit per
non-exact key block. Pooled logits are block-shared (matching the reference),
so the per-block working set is O(BLOCK · (k·BLOCK + N)) rather than O(T²).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from comfy_kitchen.backends.eager.sol_attn import (
    BLOCK,
    _LOG2E,
    _block_lengths,
    _normalize_key_bias,
    _pool,
    _sink_count,
    _topk_count,
    _valid_rows,
    add_coarse_,
    coarse_output,
)


def sol_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tau: float = 1.0,
    scale: float | None = None,
    sink_blocks: list[int] | None = None,
    sink_q: list[int] | None = None,
    key_bias: torch.Tensor | None = None,
    topk_ratio: float = 0.0,
    tail: bool = True,
    block_len: torch.Tensor | None = None,
    coarse_gate: torch.Tensor | None = None,
    token_aug: int = 0,
) -> torch.Tensor:
    del token_aug
    if q.device.type != "mps":
        raise RuntimeError("metal sol_attn requires mps tensors")
    b, t, h, d = q.shape
    n = (t + BLOCK - 1) // BLOCK
    if scale is None:
        scale = d**-0.5
    log2s = scale * _LOG2E
    sink_kv0, sink_kv1 = (sink_blocks or [0, 0])[:2]
    sink_q0, sink_q1 = (sink_q or [0, 0])[:2]

    fq, fk, fv = q.float(), k.float(), v.float()
    lengths = _block_lengths(t, n, q.device, block_len)
    kc = _pool(fk, n, "mean", lengths)
    vc = _pool(fv, n, "sum", lengths)
    k_mean = kc.mean(dim=1, keepdim=True)
    kcc = kc - k_mean
    kc_var = kcc.pow(2).mean(dim=1)
    centroid = _pool(fq, n, "mean", lengths)

    # Block-block proxy: (B, H, N, N). Equal to eager's token-averaged colmean
    # for full blocks; live-row pooling already matches block_len handling.
    qbh = centroid.permute(0, 2, 1, 3)
    kbh = kcc.permute(0, 2, 1, 3)
    colmean = torch.matmul(qbh, kbh.transpose(-1, -2)) * log2s

    idx = torch.arange(n, device=q.device)
    if topk_ratio:
        ranked = colmean.clone()
        ranked[..., sink_kv0:sink_kv1] = float("-inf")
        kk = _topk_count(n - _sink_count(n, sink_kv0, sink_kv1), topk_ratio)
        if kk:
            thr = ranked.topk(kk, dim=-1).values[..., -1:]
            exact = ranked >= thr
        else:
            exact = torch.zeros_like(ranked, dtype=torch.bool)
    else:
        var = (centroid.pow(2) * kc_var.unsqueeze(1)).sum(-1)
        thr = tau * torch.sqrt(var * log2s * log2s + 1e-6)
        exact = colmean > thr.permute(0, 2, 1).unsqueeze(-1)

    exact = exact | ((idx.view(1, -1) - idx.view(-1, 1)).abs() <= 1).view(1, 1, n, n)
    exact = exact | ((idx >= sink_kv0) & (idx < sink_kv1)).view(1, 1, 1, n)
    exact = exact | ((idx >= sink_q0) & (idx < sink_q1)).view(1, 1, n, 1)

    live = _valid_rows(t, lengths)
    pad_t = n * BLOCK - t
    if pad_t:
        k_pad = F.pad(fk - k_mean, (0, 0, 0, 0, 0, pad_t))
        v_pad = F.pad(fv, (0, 0, 0, 0, 0, pad_t))
        live_pad = F.pad(live, (0, pad_t))
        q_pad = F.pad(fq, (0, 0, 0, 0, 0, pad_t))
    else:
        k_pad = fk - k_mean
        v_pad = fv
        live_pad = live
        q_pad = fq

    k_blocks = k_pad.view(b, n, BLOCK, h, d)
    v_blocks = v_pad.view(b, n, BLOCK, h, d)
    q_blocks = q_pad.view(b, n, BLOCK, h, d)
    live_blocks = live_pad.view(n, BLOCK)

    kb = None
    if key_bias is not None:
        kb = _normalize_key_bias(key_bias, b, t, q.device)
        kb_pad = F.pad(kb, (0, pad_t)) if pad_t else kb
        kb_blocks = kb_pad.view(kb_pad.shape[0], n, BLOCK)
    else:
        kb_blocks = None

    neg = torch.finfo(torch.float32).min
    out = q.new_empty(b, t, h, d)

    for qb in range(n):
        q_live = live_blocks[qb]
        nq = int(q_live.sum().item())
        if nq == 0:
            continue
        q_tok = q_blocks[:, qb, :nq]  # (B, Nq, H, D)
        qh = q_tok.permute(0, 2, 1, 3)  # (B, H, Nq, D)

        exact_qb = exact[:, :, qb, :]  # (B, H, N)
        # A key block may be exact for some heads and pooled for others: keep it
        # in both lists and mask per-(b,h) so neither branch is dropped.
        want_exact = exact_qb.any(dim=0).any(dim=0)
        want_pool = (~exact_qb).any(dim=0).any(dim=0) if tail else want_exact.new_zeros(n)
        exact_idx = want_exact.nonzero(as_tuple=False).flatten()
        pool_idx = want_pool.nonzero(as_tuple=False).flatten()

        logits_parts = []
        v_parts = []
        den_extra = []  # length multipliers for pooled slots (1 for exact tokens)

        if exact_idx.numel():
            e = exact_idx.numel()
            ek = k_blocks.index_select(1, exact_idx).permute(0, 3, 1, 2, 4).reshape(
                b, h, e * BLOCK, d
            )
            ev = v_blocks.index_select(1, exact_idx).permute(0, 3, 1, 2, 4).reshape(
                b, h, e * BLOCK, d
            )
            elive = live_blocks.index_select(0, exact_idx).reshape(-1)
            scores = torch.matmul(qh, ek.transpose(-1, -2)) * log2s  # (B,H,Nq,E*BLOCK)
            allow = exact_qb.index_select(-1, exact_idx)  # (B,H,E)
            allow_tok = allow.unsqueeze(-1).expand(b, h, e, BLOCK).reshape(b, h, e * BLOCK)
            scores = scores.masked_fill(~allow_tok.unsqueeze(2), neg)
            scores = scores.masked_fill(~elive.view(1, 1, 1, -1), neg)
            if kb_blocks is not None:
                kb_exact = kb_blocks.index_select(1, exact_idx).reshape(kb_blocks.shape[0], -1)
                scores = scores + (kb_exact * _LOG2E).view(kb_exact.shape[0], 1, 1, -1)
            logits_parts.append(scores)
            v_parts.append(ev)
            den_extra.append(scores.new_ones(e * BLOCK))

        if pool_idx.numel():
            # Block-shared pooled logits: same value for every query token in qb.
            p_scores = colmean[:, :, qb : qb + 1, :].index_select(-1, pool_idx)
            p_scores = p_scores.expand(b, h, nq, pool_idx.numel()).clone()
            pool_allow = ~exact_qb.index_select(-1, pool_idx)  # True = this head pools
            p_scores = p_scores.masked_fill(~pool_allow.unsqueeze(2), neg)
            pv = vc.index_select(1, pool_idx).permute(0, 2, 1, 3)  # (B,H,P,D) SUM values
            logits_parts.append(p_scores)
            v_parts.append(pv)
            den_extra.append(lengths.index_select(0, pool_idx).to(dtype=p_scores.dtype))

        if not logits_parts:
            continue

        logits = torch.cat(logits_parts, dim=-1)
        vv = torch.cat(v_parts, dim=-2)
        denscale = torch.cat(den_extra, dim=0)

        m = logits.amax(dim=-1, keepdim=True)
        p = torch.exp2(logits - m)
        p = p.masked_fill(logits <= neg / 2, 0.0)
        num = torch.matmul(p, vv)
        den = (p * denscale.view(1, 1, 1, -1)).sum(dim=-1).clamp_min(1e-30)
        o = (num / den.unsqueeze(-1)).permute(0, 2, 1, 3).contiguous()
        t0 = qb * BLOCK
        out[:, t0 : t0 + nq] = o.to(dtype=v.dtype)

    out = out.contiguous()
    if coarse_gate is not None:
        flat = lambda p: p.permute(0, 2, 1, 3).reshape(b * h, n, d)  # noqa: E731
        oc = coarse_output(
            flat(centroid),
            flat(kc),
            flat(vc / lengths.view(1, -1, 1, 1)),
            scale,
        )
        add_coarse_(out, oc, coarse_gate)
    return out
