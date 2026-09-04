"""Sol-Attn: training-free sparse attention for video diffusion (arXiv 2607.24027).

Reference implementation: each query block attends a routed subset of key
blocks exactly; every other block contributes one pooled term (summary key,
summed values), so the whole sequence stays in the softmax denominator.

Defines the algorithm, not the CUDA kernel's arithmetic (full precision here
vs INT8 there -- expect cos ~0.999, not bitwise), and materialises the scores
densely, so memory is O(T^2).
"""

import torch

from ...registry import registry

BLOCK = 64
_LOG2E = 1.4426950408889634
# Past this the caller almost certainly wanted a fused backend and got
# silently downgraded; a clear error beats an allocator failure.
_MAX_SCORE_BYTES = 4 * 2**30


def _normalize_key_bias(key_bias, batch, t, device):
    """Reduce SDPA-mask-like forms -- (T,), (B, T), (B|1, 1, 1, T), bool or
    float -- to (B, T) float log-bias. Rejects head/query-varying masks and
    wrong-device tensors (a host pointer would poison the CUDA context).
    """
    if key_bias.device != device:
        raise ValueError(
            f"sol_attn: key_bias must be on {device}, got {key_bias.device}")
    if key_bias.dim() == 4:
        if key_bias.shape[1] != 1 or key_bias.shape[2] != 1:
            raise ValueError(
                "sol_attn: key_bias must be key-only; a mask varying over "
                f"heads or queries ({tuple(key_bias.shape)}) cannot be "
                "expressed by this op -- use a dense attention for those calls")
        key_bias = key_bias[:, 0, 0, :]
    if key_bias.dim() == 1:
        key_bias = key_bias.unsqueeze(0)
    if key_bias.dim() != 2 or key_bias.shape[-1] != t or key_bias.shape[0] not in (1, batch):
        raise ValueError(
            f"sol_attn: key_bias must be (T,), (B, T) or (B, 1, 1, T), got "
            f"{tuple(key_bias.shape)} for T={t}, B={batch}")
    if key_bias.dtype == torch.bool:
        key_bias = torch.where(key_bias, 0.0, float("-inf"))
    return key_bias.float()


def _block_lengths(t: int, n: int, device, block_len=None) -> torch.Tensor:
    """Live tokens per block: the caller's table (zero-padded tiles) or 64
    everywhere except a ragged last block."""
    tail = t - (n - 1) * BLOCK
    if block_len is not None:   # clamped like the kernels (block_len_of)
        lengths = block_len.to(device=device, dtype=torch.float32).clamp(1, BLOCK)
        if tail < BLOCK:
            lengths = lengths.clone()
            lengths[-1] = lengths[-1].clamp(max=tail)
        return lengths
    lengths = torch.full((n,), float(BLOCK), device=device)
    if tail < BLOCK:
        lengths[-1] = float(tail)
    return lengths


def _sink_count(n, s0, s1):
    """Sink blocks inside [0, n): they are always exact, so the top-k budget
    counts the others only."""
    return max(0, min(s1, n) - min(s0, n))


def _valid_rows(t: int, lengths: torch.Tensor) -> torch.Tensor:
    """(T,) bool: token t is live iff it sits in the first len rows of its block."""
    pos = torch.arange(t, device=lengths.device)
    return (pos % BLOCK) < lengths[pos // BLOCK]


def coarse_output(qm, km, vm, scale):
    """VSA coarse branch: dense attention over the block means. qm/km/vm are
    ``[BH, N, D]`` fp32; returns ``[BH, N, D]`` fp32."""
    s = torch.bmm(qm, km.transpose(1, 2)) * scale
    return torch.bmm(torch.softmax(s, dim=-1), vm)


def add_coarse_(out, oc, gate):
    """out += gate * oc[block(t)], in place (one rounding: addcmul_ promotes
    internally). out/gate ``(B, T, H, D)``; oc ``[BH, N, D]`` fp32."""
    b, t, h, d = out.shape
    oc = oc.view(b, h, -1, d).permute(0, 2, 1, 3)   # (B, N, H, D)
    gate = gate.contiguous()
    nfull = t // BLOCK
    if nfull:
        out[:, :nfull * BLOCK].view(b, nfull, BLOCK, h, d).addcmul_(
            gate[:, :nfull * BLOCK].view(b, nfull, BLOCK, h, d), oc[:, :nfull, None])
    if t % BLOCK:
        out[:, nfull * BLOCK:].addcmul_(gate[:, nfull * BLOCK:], oc[:, nfull:nfull + 1])
    return out


def _topk_count(n: int, ratio: float) -> int:
    """Key blocks a query block keeps under top-k: ratio * n, clamped to
    [1, n-1]; 0 when n <= 1 (nothing to select beyond the forced blocks)."""
    return max(0, min(n - 1, max(1, round(ratio * n))))


def _pool(x: torch.Tensor, n_blocks: int, reduce: str, lengths=None) -> torch.Tensor:
    """(B, T, H, D) -> (B, N, H, D), block mean or sum over the live rows."""
    b, t, h, d = x.shape
    if lengths is None:
        lengths = _block_lengths(t, n_blocks, x.device)
    x = x * _valid_rows(t, lengths).view(1, -1, 1, 1)
    pad = n_blocks * BLOCK - t
    if pad:
        x = torch.cat([x, x.new_zeros(b, pad, h, d)], dim=1)
    blocks = x.reshape(b, n_blocks, BLOCK, h, d)
    if reduce == "sum":
        return blocks.sum(dim=2)
    return blocks.sum(dim=2) / lengths.view(1, -1, 1, 1)


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
    token_aug: int = 0,   # ignored: the reference is exact arithmetic over the same
                          # block selection, so there is no token routing stage to model
) -> torch.Tensor:
    """Sol-Attn over ``(B, T, H, D)`` tensors. See the module docstring.

    ``topk_ratio`` > 0 selects SLA-style per-query-block top-k instead of the
    tau threshold (sinks and diagonal still forced exact); tau is ignored.
    ``tail=False`` drops the pooled term (softmax over routed blocks only),
    ``block_len`` marks the live rows at the front of each 64-block (values
    clamped to [1, rows in the block]; dead rows are never keys and their
    output rows are unspecified), and ``coarse_gate`` adds VSA's gated coarse
    branch: ``gate * softmax(q_mean k_mean^T * scale) v_mean`` per block.
    """
    b, t, h, d = q.shape
    n = (t + BLOCK - 1) // BLOCK
    if scale is None:
        scale = d ** -0.5
    log2s = scale * _LOG2E
    sink_kv0, sink_kv1 = (sink_blocks or [0, 0])[:2]
    sink_q0, sink_q1 = (sink_q or [0, 0])[:2]

    score_bytes = b * h * t * t * 4
    if score_bytes > _MAX_SCORE_BYTES:
        raise RuntimeError(
            f"sol_attn: the eager reference is O(T^2) and would need "
            f"{score_bytes / 2**30:.1f} GiB for the score tensor at "
            f"(B={b}, H={h}, T={t}). It was selected because no fused backend "
            f"accepted these inputs -- the fused backends take bfloat16 or float16 "
            f"with head_dim 128, and got {q.dtype} on {q.device.type}."
        )

    fq, fk, fv = q.float(), k.float(), v.float()
    lengths = _block_lengths(t, n, q.device, block_len)
    kc = _pool(fk, n, "mean", lengths)              # (B, N, H, D) summary keys
    vc = _pool(fv, n, "sum", lengths)               # (B, N, H, D) summed values

    # centring K shifts every score in a row by a constant: softmax-invariant
    k_mean = kc.mean(dim=1, keepdim=True)           # (B, 1, H, D)
    kcc = kc - k_mean
    kc_var = kcc.pow(2).mean(dim=1)                 # (B, H, D)

    centroid = _pool(fq, n, "mean", lengths)                        # (B, N, H, D)

    qh = fq.permute(0, 2, 1, 3)                                     # (B, H, T, D)
    kh = (fk - k_mean).permute(0, 2, 1, 3)
    vh = fv.permute(0, 2, 1, 3)
    kch = kcc.permute(0, 2, 1, 3)                                   # (B, H, N, D)
    vch = vc.permute(0, 2, 1, 3)

    s_tok = (qh @ kh.transpose(-1, -2)) * log2s                     # (B, H, T, T)
    if block_len is not None:
        s_tok = s_tok.masked_fill(~_valid_rows(t, lengths).view(1, 1, 1, t),
                                  torch.finfo(s_tok.dtype).min)
    if key_bias is not None:
        # Per-key logit bias (natural log). Exact branch only: biased blocks
        # must be sink-covered, the pooled tail cannot see per-token bias.
        kb = _normalize_key_bias(key_bias, b, t, q.device)
        s_tok = s_tok + (kb * _LOG2E).reshape(-1, 1, 1, t)
    s_blk = (qh @ kch.transpose(-1, -2)) * log2s                    # (B, H, T, N)

    # routed = column mean over the query block clears the threshold; the
    # diagonal +-1, sink blocks and sink_q rows are always exact
    qblk = torch.arange(t, device=q.device) // BLOCK
    if block_len is not None:   # dead query rows must not enter the block's mean
        s_blk = s_blk * _valid_rows(t, lengths).view(1, 1, t, 1)
    colmean = torch.zeros(b, h, n, n, device=q.device, dtype=s_blk.dtype)
    colmean.scatter_add_(2, qblk.view(1, 1, t, 1).expand(b, h, t, n), s_blk)
    colmean = colmean / lengths.view(1, 1, n, 1)
    idx = torch.arange(n, device=q.device)
    if topk_ratio:
        # sink blocks are always exact, so they neither count toward nor consume the budget
        ranked = colmean.clone()
        ranked[..., sink_kv0:sink_kv1] = float("-inf")
        kk = _topk_count(n - _sink_count(n, sink_kv0, sink_kv1), topk_ratio)
        # >= the k-th score: a tied group at the boundary is kept, not dropped
        if kk:
            row_thr = ranked.topk(kk, dim=-1).values[..., -1:]
            exact = ranked >= row_thr
        else:
            exact = torch.zeros_like(ranked, dtype=torch.bool)
    else:
        # tau sigma of the proxy row, from the query-block centroid
        var = (centroid.pow(2) * kc_var.unsqueeze(1)).sum(-1)      # (B, N, H)
        thr = tau * torch.sqrt(var * log2s * log2s + 1e-6)
        exact = colmean > thr.permute(0, 2, 1).unsqueeze(-1)            # (B, H, NQ, N)
    exact |= ((idx.view(1, -1) - idx.view(-1, 1)).abs() <= 1).view(1, 1, n, n)
    exact |= ((idx >= sink_kv0) & (idx < sink_kv1)).view(1, 1, 1, n)
    exact |= ((idx >= sink_q0) & (idx < sink_q1)).view(1, 1, n, 1)

    ex_tok = exact.gather(2, qblk.view(1, 1, t, 1).expand(b, h, t, n))   # (B,H,T,N)
    keep_tok = ex_tok.repeat_interleave(BLOCK, dim=-1)[..., :t]
    neg = torch.finfo(s_tok.dtype).min
    s_tok = s_tok.masked_fill(~keep_tok, neg)
    # every row shares its query block's tail (colmean IS the centroid score)
    s_blk = colmean.gather(2, qblk.view(1, 1, t, 1).expand(b, h, t, n))
    s_blk = s_blk.masked_fill(ex_tok, neg)
    if not tail:
        s_blk = torch.full_like(s_blk, neg)

    # one softmax over both branches; a pooled term weighs its block length (vc is a sum)
    logits = torch.cat([s_tok, s_blk], dim=-1)
    p = torch.exp2(logits - logits.amax(dim=-1, keepdim=True))
    p = p.masked_fill(logits <= neg, 0.0)
    num = p[..., :t] @ vh + p[..., t:] @ vch
    den = p[..., :t].sum(-1) + (p[..., t:] * lengths.view(1, 1, 1, n)).sum(-1)
    out = (num / den.clamp_min(1e-30).unsqueeze(-1)).permute(0, 2, 1, 3).to(v.dtype)
    # contiguous, matching the CUDA backend and register_fake
    out = out.contiguous()
    if coarse_gate is not None:
        flat = lambda p: p.permute(0, 2, 1, 3).reshape(b * h, n, d)  # noqa: E731
        oc = coarse_output(flat(centroid), flat(kc), flat(vc / lengths.view(1, -1, 1, 1)), scale)
        add_coarse_(out, oc, coarse_gate)
    return out


@torch.library.custom_op("comfy_kitchen::sol_attn", mutates_args=())
def _op_sol_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tau: float,
    scale: float | None,
    sink_blocks: list[int],
    sink_q: list[int],
    key_bias: torch.Tensor | None,
    topk_ratio: float,
    tail: bool,
    block_len: torch.Tensor | None,
    coarse_gate: torch.Tensor | None,
    token_aug: int = 0,
) -> torch.Tensor:
    kwargs = {
        "q": q, "k": k, "v": v, "tau": tau, "scale": scale,
        "sink_blocks": sink_blocks, "sink_q": sink_q,
        "key_bias": key_bias, "topk_ratio": topk_ratio,
        "tail": tail, "block_len": block_len, "coarse_gate": coarse_gate,
        "token_aug": token_aug,
    }
    impl = registry.get_implementation("sol_attn", kwargs=kwargs)
    return impl(**kwargs)


@_op_sol_attn.register_fake
def _op_sol_attn_fake(q, k, v, tau, scale, sink_blocks, sink_q,
                      key_bias, topk_ratio, tail, block_len, coarse_gate, token_aug=0):
    # contiguous, NOT empty_like(v): both real implementations return contiguous
    return torch.empty(v.shape, dtype=v.dtype, device=v.device)
