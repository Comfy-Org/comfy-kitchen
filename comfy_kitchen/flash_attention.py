from __future__ import annotations

import math

import torch

from .allocation import allocation_context
from .backends import cuda as _cuda_backend

if getattr(torch.version, "hip", None):
    from .backends import hip as _hip_backend
else:
    _hip_backend = None

_MINIMUM_CAPABILITY = (8, 0)

# Split-KV partials are folded by the last-arriving split CTA inside the decode kernel
# (one launch) rather than by a second combine launch. Per-device arrival counters; the kernel
# rearms them, so the buffer only needs zeroing once. False restores the two-launch path.
# Calls must not overlap on a device. Keep the buffer alive: graphs capture its address.
fused_combine = True
_COMBINE_COUNTERS = 4096
_combine_counters: dict[torch.device, torch.Tensor] = {}


def _counters(device: torch.device, needed: int, num_splits: int) -> torch.Tensor:
    if not fused_combine or num_splits == 1 or needed > _COMBINE_COUNTERS:
        return _cuda_backend._empty_cuda_tensor(device, torch.int32)
    counters = _combine_counters.get(device)
    if counters is None:
        with allocation_context():
            counters = _combine_counters[device] = torch.zeros(
                _COMBINE_COUNTERS, dtype=torch.int32, device=device)
    return counters


def is_available(device: torch.device | None = None) -> bool:
    """Return whether flash attention decode is available on the requested device."""
    if device is not None and device.type != "cuda":
        return False
    if not torch.cuda.is_available():
        return False
    if _hip_backend is not None:
        # torch.cuda is the ROCm API here, and get_device_capability reports
        # something SM-shaped for a gfx part, so the compute capability test
        # below would wave AMD hardware through to a CUDA extension that never
        # loaded. Ask the HIP backend instead, which answers for the process
        # rather than for one device: its arch gates take the intersection over
        # every visible device, the way int8 attention and the op registry do.
        return _hip_backend.flash_attention_decode_is_available()
    if (
        not _cuda_backend._EXT_AVAILABLE
        or _cuda_backend._C is None
        or not hasattr(_cuda_backend._C, "flash_attention_decode")
    ):
        return False
    return torch.cuda.get_device_capability(device) >= _MINIMUM_CAPABILITY


def _num_splits(batch_heads: int, kv_capacity: int, multiprocessors: int, block_n: int) -> int:
    blocks = (kv_capacity + block_n - 1) // block_n
    max_splits = min(32, multiprocessors * 2, blocks)
    best = 0.0
    efficiencies = []
    for splits in range(1, max_splits + 1):
        eligible = splits == 1 or math.ceil(blocks / splits) != math.ceil(
            blocks / (splits - 1)
        )
        efficiency = batch_heads * splits / (multiprocessors * 2)
        efficiency = efficiency / math.ceil(efficiency) if eligible else 0.0
        efficiencies.append(efficiency)
        best = max(best, efficiency)
    return next(
        splits
        for splits, efficiency in enumerate(efficiencies, 1)
        if efficiency >= 0.85 * best
    )


def flash_attention_decode(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kv_lengths: torch.Tensor
) -> torch.Tensor:
    """Decode BF16 [batch, 1, heads, head_dim] queries over a fixed KV cache.

    CUDA and HIP support head dimensions 128 and 256.
    """
    batch, _, query_heads, head_dim = q.shape
    _, kv_capacity, kv_heads, _ = k.shape
    if not is_available(q.device):
        raise RuntimeError(
            "flash_attention_decode requires the HIP extension on an AMD device "
            "with bf16 support (RDNA3 or newer)"
            if _hip_backend is not None
            else "flash_attention_decode requires the CUDA extension on SM80 or newer"
        )

    groups = query_heads // kv_heads
    query = (
        q.reshape(batch, kv_heads, groups, head_dim)
        .transpose(1, 2)
        .reshape(batch * groups, kv_heads, head_dim)
    )
    output = torch.empty_like(query)
    num_splits = _num_splits(
        batch * kv_heads,
        kv_capacity,
        torch.cuda.get_device_properties(q.device).multi_processor_count,
        64 if head_dim == 256 and _hip_backend is None else 128,
    )
    softmax_lse = torch.empty(batch * kv_heads * groups, dtype=torch.float32, device=q.device)
    if num_splits > 1:
        softmax_lse_accum = torch.empty(
            num_splits, batch, kv_heads, groups, dtype=torch.float32, device=q.device
        )
        output_accum = torch.empty(
            num_splits,
            batch,
            kv_heads,
            groups,
            head_dim,
            dtype=torch.float32,
            device=q.device,
        )
    else:
        softmax_lse_accum = output_accum = softmax_lse[:0]
    if _hip_backend is not None:
        _hip_backend.flash_decode(
            query, k, v, kv_lengths, output, softmax_lse, softmax_lse_accum, output_accum,
            num_splits,
        )
    else:
        _cuda_backend._C.flash_attention_decode(
            *map(
                _cuda_backend._wrap_for_dlpack,
                (query, k, v, kv_lengths, output, softmax_lse, softmax_lse_accum, output_accum,
                 _counters(q.device, batch * kv_heads, num_splits)),
            ),
            num_splits,
            torch.cuda.current_stream(q.device).cuda_stream,
        )
    return output.view(batch, groups, kv_heads, head_dim).transpose(1, 2).reshape_as(q)


def flash_attention_decode_gqa_is_available(
    device: torch.device | int | None = None, dtype: torch.dtype = torch.bfloat16
) -> bool:
    """Return whether flash_attention_decode_gqa can run on ``device`` for ``dtype``
    inputs: the CUDA extension on SM80 or newer, BF16 only."""
    if dtype != torch.bfloat16 or _hip_backend is not None:
        return False
    if isinstance(device, int):
        device = torch.device("cuda", device)
    return is_available(device) and hasattr(_cuda_backend._C, "flash_attention_decode_gqa")


def flash_attention_decode_gqa(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kv_lengths: torch.Tensor, return_lse: bool = False,
    causal: bool = True,
) -> torch.Tensor:
    """GQA decode attention for BF16 q [B, H, S, 256] over k/v [B, Hk, capacity, 256].

    Causal: query row j of batch b attends cache slots < kv_lengths[b] - S + j + 1 (the
    speculative verify staircase; S == 1 is plain decode). Otherwise every row attends slots <
    kv_lengths[b], which is how a verify tree reads its committed prefix. Returns
    [B, S, H*256], with the fp32 log-sum-exp of the scaled scores [B, H, S] as a second value
    when return_lse."""
    batch, heads, query_length, head_dim = q.shape
    _, kv_heads, kv_capacity, _ = k.shape
    if head_dim != 256:
        raise ValueError(f"flash_attention_decode_gqa supports head_dim 256, got {head_dim}")
    if not flash_attention_decode_gqa_is_available(q.device, q.dtype):
        raise RuntimeError(
            "flash_attention_decode_gqa requires BF16 inputs and the CUDA extension on SM80 or newer"
        )
    output = torch.empty((batch, query_length, heads * head_dim), dtype=q.dtype, device=q.device)
    # S == 1 folds the group heads into query rows (one pass over K/V per kv head), so the
    # kernel then runs kv_heads CTAs per split rather than heads.
    num_splits = _num_splits(
        batch * (kv_heads if query_length == 1 else heads),
        kv_capacity,
        torch.cuda.get_device_properties(q.device).multi_processor_count,
        64,
    )
    rows = batch * heads * query_length
    softmax_lse = torch.empty(rows, dtype=torch.float32, device=q.device)
    if num_splits > 1:
        softmax_lse_accum = torch.empty(num_splits * rows, dtype=torch.float32, device=q.device)
        output_accum = torch.empty(num_splits * rows * head_dim, dtype=torch.float32, device=q.device)
    else:
        softmax_lse_accum = output_accum = softmax_lse[:0]
    _cuda_backend._C.flash_attention_decode_gqa(
        *map(
            _cuda_backend._wrap_for_dlpack,
            (q, k, v, kv_lengths, output, softmax_lse, softmax_lse_accum, output_accum,
             _counters(q.device, batch * (kv_heads if query_length == 1 else heads), num_splits)),
        ),
        num_splits,
        causal,
        torch.cuda.current_stream(q.device).cuda_stream,
    )
    if return_lse:
        return output, softmax_lse.view(batch, heads, query_length)
    return output


def flash_attention_decode_tree_merge(
    out: torch.Tensor, lse: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    mask: torch.Tensor, merged: torch.Tensor,
) -> torch.Tensor:
    """Fold a verify tree's own rows into a prefix-only decode result.

    ``out`` [B, S, H*256] and ``lse`` [B, H, S] come from flash_attention_decode_gqa over the
    committed prefix with causal=False; q [B, H, S, 256] and k/v [B, Hk, S, 256] are the verify
    rows' own rotated query/key/value, and mask [S] int32 the rows each row attends inside the
    step (bit t = row t, including itself). Writes and returns ``merged`` [B, S, H*256]."""
    if not flash_attention_decode_gqa_is_available(q.device, q.dtype):
        raise RuntimeError(
            "flash_attention_decode_tree_merge requires BF16 inputs and the CUDA extension on SM80 or newer"
        )
    _cuda_backend._C.flash_attention_decode_tree_merge(
        *map(_cuda_backend._wrap_for_dlpack, (out, lse, q, k, v, mask, merged)),
        torch.cuda.current_stream(q.device).cuda_stream,
    )
    return merged
