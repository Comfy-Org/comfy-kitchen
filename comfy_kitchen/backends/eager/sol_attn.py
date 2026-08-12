"""Sol-Attn: training-free sparse attention for video diffusion (arXiv 2607.24027).

Reference implementation: each query block attends a routed subset of key
blocks exactly; every other block contributes one pooled term (summary key,
summed values), so the whole sequence stays in the softmax denominator.

Defines the algorithm, not the CUDA kernel's arithmetic (full precision here
vs INT8 there -- expect cos ~0.999, not bitwise), and materialises the scores
densely, so memory is O(T^2).
"""

import functools
import inspect

import torch

from ...registry import registry

BLOCK = 64
_LOG2E = 1.4426950408889634
# Past this the caller almost certainly wanted a fused backend and got
# silently downgraded; a clear error beats an allocator failure.
_MAX_SCORE_BYTES = 4 * 2**30


def _normalize_key_bias(key_bias, batch, t, device):
    """Accept SDPA-mask-like forms and reduce them to (B, T) float log-bias.

    Shapes: (T,), (B, T), or broadcastable 4-D (B-or-1, 1, 1, T) -- the
    key-only slice of the standard attn_mask contract. Bool masks follow SDPA
    semantics: True = attend (bias 0), False = masked (bias -inf). Anything
    varying over heads or queries is not expressible here and is rejected.

    The device check is not optional politeness: a host pointer handed to the
    CUDA preprocess is an asynchronous illegal-memory-access that poisons the
    context on some LATER call.
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


def _pool(x: torch.Tensor, n_blocks: int, reduce: str) -> torch.Tensor:
    """(B, T, H, D) -> (B, N, H, D), block mean or sum, ragged tail handled."""
    b, t, h, d = x.shape
    pad = n_blocks * BLOCK - t
    if pad:
        x = torch.cat([x, x.new_zeros(b, pad, h, d)], dim=1)
    blocks = x.reshape(b, n_blocks, BLOCK, h, d)
    if reduce == "sum":
        return blocks.sum(dim=2)
    lengths = x.new_full((n_blocks,), float(BLOCK))
    if pad:
        lengths[-1] = float(BLOCK - pad)
    return blocks.sum(dim=2) / lengths.view(1, -1, 1, 1)


def sol_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tau: float = 1.0,
    scale: float | None = None,
    sink_blocks: list[int] | None = None,
    sink_q: list[int] | None = None,
    centroid_tail: bool = True,
    key_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sol-Attn over ``(B, T, H, D)`` tensors. See the module docstring."""
    b, t, h, d = q.shape
    n = (t + BLOCK - 1) // BLOCK
    if scale is None:
        scale = d ** -0.5
    log2s = scale * _LOG2E
    sink_kv0, sink_kv1 = (sink_blocks or [0, 0])[:2]
    sink_q0, sink_q1 = (sink_q or [0, 0])[:2]

    # Anything routed here by accident (fp16/fp32 input at video length) would
    # otherwise die in the allocator with nothing pointing at the cause.
    score_bytes = b * h * t * t * 4
    if score_bytes > _MAX_SCORE_BYTES:
        raise RuntimeError(
            f"sol_attn: the eager reference is O(T^2) and would need "
            f"{score_bytes / 2**30:.1f} GiB for the score tensor at "
            f"(B={b}, H={h}, T={t}). It was selected because no fused backend "
            f"accepted these inputs -- the CUDA backend requires bfloat16 on CUDA "
            f"with head_dim 128, and got {q.dtype} on {q.device.type}."
        )

    fq, fk, fv = q.float(), k.float(), v.float()
    kc = _pool(fk, n, "mean")                       # (B, N, H, D) summary keys
    vc = _pool(fv, n, "sum")                        # (B, N, H, D) summed values

    # Centring K by the pooled mean shifts every score in a row by a constant,
    # leaving the softmax unchanged; it only shrinks the INT8 dynamic range.
    k_mean = kc.mean(dim=1, keepdim=True)           # (B, 1, H, D)
    kcc = kc - k_mean
    kc_var = kcc.pow(2).mean(dim=1)                 # (B, H, D)

    # Routing threshold: tau sigma of the proxy row, from the query-block centroid.
    lengths = fq.new_full((n,), float(BLOCK))
    if n * BLOCK - t:
        lengths[-1] = float(t - (n - 1) * BLOCK)
    centroid = _pool(fq, n, "mean")                                 # (B, N, H, D)
    var = (centroid.pow(2) * kc_var.unsqueeze(1)).sum(-1)          # (B, N, H)
    thr = tau * torch.sqrt(var * log2s * log2s + 1e-6)

    qh = fq.permute(0, 2, 1, 3)                                     # (B, H, T, D)
    kh = (fk - k_mean.squeeze(1).unsqueeze(1)).permute(0, 2, 1, 3)
    vh = fv.permute(0, 2, 1, 3)
    kch = kcc.permute(0, 2, 1, 3)                                   # (B, H, N, D)
    vch = vc.permute(0, 2, 1, 3)

    s_tok = (qh @ kh.transpose(-1, -2)) * log2s                     # (B, H, T, T)
    if key_bias is not None:
        # Additive logit bias per KEY (natural-log units, like an SDPA mask or
        # LTX's log(strength)). Applied to the exact branch only: the pooled
        # tail cannot see per-token bias, so blocks holding nonzero bias must
        # be covered by sink_blocks (routed exact) for the bias to be honoured.
        kb = _normalize_key_bias(key_bias, b, t, q.device)
        s_tok = s_tok + (kb * _LOG2E).reshape(-1, 1, 1, t)
    s_blk = (qh @ kch.transpose(-1, -2)) * log2s                    # (B, H, T, N)

    # A block is routed if its mean score over the query block clears the
    # threshold; the diagonal and its neighbours, and any sink block, are always
    # exact, and every block is exact for a query inside the sink range.
    qblk = torch.arange(t, device=q.device) // BLOCK
    colmean = torch.zeros(b, h, n, n, device=q.device, dtype=s_blk.dtype)
    colmean.scatter_add_(2, qblk.view(1, 1, t, 1).expand(b, h, t, n), s_blk)
    colmean = colmean / lengths.view(1, 1, n, 1)
    idx = torch.arange(n, device=q.device)
    exact = colmean > thr.permute(0, 2, 1).unsqueeze(-1)            # (B, H, NQ, N)
    exact |= ((idx.view(1, -1) - idx.view(-1, 1)).abs() <= 1).view(1, 1, n, n)
    exact |= ((idx >= sink_kv0) & (idx < sink_kv1)).view(1, 1, 1, n)
    exact |= ((idx >= sink_q0) & (idx < sink_q1)).view(1, 1, n, 1)
    valid_blk = (idx * BLOCK < t).view(1, 1, 1, n)
    exact &= valid_blk

    ex_tok = exact.gather(2, qblk.view(1, 1, t, 1).expand(b, h, t, n))   # (B,H,T,N)
    keep_tok = ex_tok.repeat_interleave(BLOCK, dim=-1)[..., :t]
    neg = torch.finfo(s_tok.dtype).min
    s_tok = s_tok.masked_fill(~keep_tok, neg)
    if centroid_tail:
        # The tail is evaluated at the query-block centroid -- colmean IS the
        # centroid score, the same quantity the routing decision thresholds --
        # so every row of a query block shares its tail. Costs ~5e-4 cosine vs
        # the per-row tail; buys the fused routing pass a 64x smaller problem.
        s_blk = colmean.gather(2, qblk.view(1, 1, t, 1).expand(b, h, t, n))
    s_blk = s_blk.masked_fill(ex_tok | ~valid_blk, neg)

    # One softmax over both branches. A pooled term carries its block's length in
    # the denominator because vc is a sum, not a mean.
    logits = torch.cat([s_tok, s_blk], dim=-1)
    p = torch.exp2(logits - logits.amax(dim=-1, keepdim=True))
    p = p.masked_fill(logits <= neg, 0.0)
    num = p[..., :t] @ vh + p[..., t:] @ vch
    den = p[..., :t].sum(-1) + (p[..., t:] * lengths.view(1, 1, 1, n)).sum(-1)
    out = (num / den.clamp_min(1e-30).unsqueeze(-1)).permute(0, 2, 1, 3).to(v.dtype)
    # Contiguous to match the CUDA backend and register_fake (torch.compile
    # plans downstream ops against the fake's strides).
    return out.contiguous()


@functools.cache
def _accepts_max_blocks(impl) -> bool:
    """Whether this backend's ``sol_attn`` takes the routed-block cap.

    Asked of the signature, not the module name, so it survives registry
    refactors; cached because it sits in the per-call path.
    """
    try:
        return "max_blocks" in inspect.signature(impl).parameters
    except (TypeError, ValueError):
        return False


@torch.library.custom_op("comfy_kitchen::sol_attn", mutates_args=())
def _op_sol_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tau: float,
    scale: float | None,
    sink_blocks: list[int],
    sink_q: list[int],
    max_blocks: int,
    centroid_tail: bool,
    key_bias: torch.Tensor | None,
) -> torch.Tensor:
    kwargs = {
        "q": q, "k": k, "v": v, "tau": tau, "scale": scale,
        "sink_blocks": sink_blocks, "sink_q": sink_q,
        "centroid_tail": centroid_tail, "key_bias": key_bias,
    }
    impl = registry.get_implementation("sol_attn", kwargs=kwargs)
    if _accepts_max_blocks(impl):
        kwargs["max_blocks"] = max_blocks
    return impl(**kwargs)


@_op_sol_attn.register_fake
def _op_sol_attn_fake(q, k, v, tau, scale, sink_blocks, sink_q, max_blocks,
                      centroid_tail, key_bias):
    # Contiguous, NOT empty_like(v): both real implementations return
    # contiguous, and for a strided v, empty_like would promise a layout
    # downstream compiled ops never get.
    return torch.empty(v.shape, dtype=v.dtype, device=v.device)
