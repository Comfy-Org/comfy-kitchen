# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org. All rights reserved.
"""Block-sparse SOL attention for long-sequence BF16 self-attention."""

from __future__ import annotations

import math

import torch

from .backends import cuda as _cuda_backend

_BLOCK_SIZE = 64
_HEAD_DIM = 128
# The exact-block list temporarily lives in one 64 x 128 output tile.
_MAX_ROUTED_BLOCKS = _BLOCK_SIZE * _HEAD_DIM
_MINIMUM_CAPABILITY = (8, 0)


def is_available(device: torch.device | int | None = None) -> bool:
    """Return whether BF16 SOL attention can run on the selected CUDA device."""
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None):
        return False
    return (
        torch.cuda.get_device_capability(device) >= _MINIMUM_CAPABILITY
        and _cuda_backend._EXT_AVAILABLE
        and hasattr(_cuda_backend._C, "sol_attention_bf16")
    )


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    tau: float,
    scale: float | None,
) -> tuple[float, float]:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape [batch, heads, sequence, head_dim]")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(
            "SOL attention is self-attention and requires matching q, k, and v shapes, "
            f"got {tuple(q.shape)}, {tuple(k.shape)}, and {tuple(v.shape)}"
        )
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("SOL attention requires bfloat16 q, k, and v")
    if not q.is_cuda or q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same CUDA device")
    if not is_available(q.device):
        raise RuntimeError(
            "SOL attention requires the comfy-kitchen CUDA extension on SM80 or newer"
        )

    batch, heads, sequence_length, head_dim = q.shape
    if batch <= 0 or heads <= 0 or sequence_length <= 0:
        raise ValueError("batch, heads, and sequence length must be positive")
    if head_dim != _HEAD_DIM:
        raise ValueError(f"SOL attention requires head_dim {_HEAD_DIM}, got {head_dim}")
    if sequence_length % _BLOCK_SIZE:
        raise ValueError(
            f"SOL attention requires sequence length divisible by {_BLOCK_SIZE}, "
            f"got {sequence_length}"
        )
    if sequence_length // _BLOCK_SIZE > _MAX_ROUTED_BLOCKS:
        raise ValueError(
            f"SOL attention supports at most {_MAX_ROUTED_BLOCKS * _BLOCK_SIZE} tokens, "
            f"got {sequence_length}"
        )
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("the last dimension of q, k, and v must be contiguous")

    tau_value = float(tau)
    if not math.isfinite(tau_value):
        raise ValueError(f"tau must be finite, got {tau_value}")
    scale_value = head_dim**-0.5 if scale is None else float(scale)
    if not math.isfinite(scale_value):
        raise ValueError(f"scale must be finite, got {scale_value}")
    return tau_value, scale_value


def _sol_attention_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    tau: float,
    scale: float | None,
) -> torch.Tensor:
    tau_value, scale_value = _validate_inputs(q, k, v, tau=tau, scale=scale)

    # The CUDA kernel folds batch and heads. Preserve the friendly BHND API and
    # materialize only when a caller supplies a strided attention view.
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    batch, heads, sequence_length, head_dim = q.shape
    blocks = sequence_length // _BLOCK_SIZE
    output = torch.empty_like(q)
    kc = torch.empty(batch, heads, blocks, head_dim, dtype=torch.bfloat16, device=q.device)
    vc = torch.empty_like(kc)
    key_mean = torch.empty(batch, heads, head_dim, dtype=torch.float32, device=q.device)
    key_variance = torch.empty_like(key_mean)
    threshold = torch.empty(batch, heads, blocks, dtype=torch.float32, device=q.device)

    wrap = _cuda_backend._wrap_for_dlpack
    stream = torch.cuda.current_stream(q.device).cuda_stream
    _cuda_backend._C.sol_attention_bf16(
        wrap(q),
        wrap(k),
        wrap(v),
        wrap(output),
        wrap(kc),
        wrap(vc),
        wrap(key_mean),
        wrap(key_variance),
        wrap(threshold),
        tau_value,
        scale_value,
        stream,
    )
    return output


@torch.library.custom_op("comfy_kitchen::sol_attention", mutates_args=())
def _op_sol_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tau: float,
    scale: float | None,
) -> torch.Tensor:
    return _sol_attention_cuda(q, k, v, tau=tau, scale=scale)


@_op_sol_attention.register_fake
def _op_sol_attention_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tau: float,
    scale: float | None,
) -> torch.Tensor:
    return q.new_empty(q.shape)


def sol_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    tau: float = 1.0,
    scale: float | None = None,
) -> torch.Tensor:
    """Compute approximate block-sparse SOL self-attention.

    Inputs and output use [batch, heads, sequence, head_dim] layout. SOL
    summarizes K/V in 64-token blocks, routes locally important blocks through
    exact BF16 attention, and approximates the remaining blocks from their
    summaries. tau controls sparsity: larger values route fewer exact blocks.
    This first implementation supports BF16, head dimension 128, equal Q/K/V
    shapes, and sequence lengths divisible by 64.
    """
    return torch.ops.comfy_kitchen.sol_attention(q, k, v, tau, scale)
