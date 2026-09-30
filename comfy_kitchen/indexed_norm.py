# SPDX-License-Identifier: Apache-2.0
"""Weighted RMSNorm, indexed modulation and ConvRot256 INT8 quantization."""

import math

import torch
from torch.nn import functional

from .backends import cuda
from .backends.eager.quantization import quantize_int8_convrot_weight


def _validate(x, weight, shift, scale, rows, eps):
    if x.ndim != 2 or x.shape[1] == 0 or x.shape[1] % 256:
        raise ValueError("x must be [M,K] with positive K divisible by 256")
    m, k = x.shape
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16 or weight.shape != (k,):
        raise ValueError("x and weight [K] must be BF16")
    if shift.ndim != 2 or shift.shape[1] != k or scale.shape != shift.shape:
        raise ValueError("shift and scale must be matching [G,K] tables")
    if shift.dtype not in (torch.float32, torch.bfloat16) or scale.dtype != shift.dtype:
        raise ValueError("modulation tables must share FP32 or BF16 dtype")
    if rows.shape != (m,) or rows.dtype != torch.int32:
        raise ValueError("rows must be INT32 [M]")
    if any(t.device != x.device for t in (weight, shift, scale, rows)):
        raise ValueError("all tensors must share a device")
    if any(t.requires_grad for t in (x, weight, shift, scale)):
        raise ValueError("indexed_norm_convrot is inference-only")
    if (
        not math.isfinite(eps)
        or not torch.finfo(torch.float32).tiny <= eps <= torch.finfo(torch.float32).max
    ):
        raise ValueError("eps must be finite positive normal FP32")


def _modulation(table, rows):
    if table.shape[0] == 0:
        return table.new_zeros((rows.numel(), table.shape[1]), dtype=torch.bfloat16)
    selected = table[rows.clamp(0, table.shape[0] - 1).long()].to(torch.bfloat16)
    valid = (rows >= 0) & (rows < table.shape[0])
    return torch.where(valid[:, None], selected, 0)


def _fallback(x, weight, shift, scale, rows, eps):
    h = functional.rms_norm(x, (x.shape[1],), weight, eps)
    h = h * (1.0 + _modulation(scale, rows))
    h = h + _modulation(shift, rows)
    if x.is_cuda and not torch.version.hip and cuda._C is not None:
        return cuda.quantize_int8_rowwise_convrot64(h.contiguous(), 256)
    q, scales = quantize_int8_convrot_weight(h, 256)
    return q.contiguous(), scales.reshape(-1, 1).contiguous()


@torch.library.custom_op("comfy_kitchen::indexed_norm_convrot", mutates_args=())
def _op(
    x: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    rows: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate(x, weight, shift, scale, rows, eps)
    m, k = x.shape
    q = torch.empty((m, k), device=x.device, dtype=torch.int8)
    qs = torch.empty((m, 1), device=x.device, dtype=torch.float32)
    if m == 0:
        return q, qs
    # The native reduction reproduces this Torch CUDA RMSNorm tree. Other
    # versions keep their native RMSNorm rather than silently changing it.
    if (
        x.is_cuda
        and not torch.version.hip
        and cuda._C is not None
        and hasattr(cuda._C, "indexed_norm_convrot")
        and torch.version.git_version == "7661cd9c6b841b62b7f411aa52ec51f05457263b"
        and k == 5376
        and x.stride(1) == 1
        and shift.stride(1) == 1
        and scale.stride(1) == 1
        and weight.is_contiguous()
        and rows.is_contiguous()
    ):
        with torch.cuda.device(x.device):
            used = cuda._C.indexed_norm_convrot(
                *(cuda._wrap_for_dlpack(t) for t in (x, weight, shift, scale, rows, q, qs)),
                eps,
                torch.cuda.current_stream(x.device).cuda_stream,
            )
        if used:
            return q, qs
    return _fallback(x, weight, shift, scale, rows, eps)


@_op.register_fake
def _fake(x, weight, shift, scale, rows, eps):
    _validate(x, weight, shift, scale, rows, eps)
    return (
        x.new_empty(x.shape, dtype=torch.int8),
        x.new_empty((x.shape[0], 1), dtype=torch.float32),
    )


def indexed_norm_convrot(x, weight, shift, scale, rows, *, eps=1e-5):
    """Normalize, modulate, then quantize BF16 rows with ConvRot256.

    Each intermediate rounds to BF16: ``h = rms_norm(x, weight, eps)``;
    ``h *= bf16(1 + bf16(scale[rows]))``; ``h += bf16(shift[rows])``.
    Returns contiguous INT8 [M,K] and FP32 [M,1]. Invalid indices and empty
    tables select zero shift/scale (identity modulation), without CPU sync.
    Inputs are unmodified. Inference only. CUDA specialization: SM120, K5376,
    and the qualified Torch RMSNorm reduction; other inputs use the ordinary
    RMSNorm/modulation/quantization chain. No cross-backend bitwise guarantee.
    """
    return _op(x, weight, shift, scale, rows, eps)
