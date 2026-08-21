from __future__ import annotations

import math

import torch

from .backends import cuda as _cuda_backend

if getattr(torch.version, "hip", None):
    from .backends import hip as _hip_backend
else:
    _hip_backend = None

_MINIMUM_CAPABILITY = (8, 0)


def is_available(device: torch.device | int | None = None) -> bool:
    """Return whether flash attention decode is available on this GPU."""
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


def _num_splits(batch_heads: int, kv_capacity: int, multiprocessors: int) -> int:
    blocks = (kv_capacity + 127) // 128
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
    """Decode attention for BF16 [batch, length, heads, 128] tensors."""
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
                (query, k, v, kv_lengths, output, softmax_lse, softmax_lse_accum, output_accum),
            ),
            num_splits,
            torch.cuda.current_stream(q.device).cuda_stream,
        )
    return output.view(batch, groups, kv_heads, head_dim).transpose(1, 2).reshape_as(q)
