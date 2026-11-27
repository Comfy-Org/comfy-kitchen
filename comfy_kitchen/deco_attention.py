# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Opt-in DeCo NVFP4 self-attention for Blackwell inference."""

from __future__ import annotations

import functools

import torch
from torch.nn import functional

from .allocation import allocation_context
from .backends import cuda
from .registry import registry


# Read the compiled shape catalog once. Ordinary Python guards let Dynamo
# specialize on the actual dimensions without calling into nanobind while tracing.
_SHAPES = tuple(getattr(cuda._C, "deco_attention_shapes", ()))
_ARCHITECTURES = tuple(getattr(cuda._C, "deco_attention_architectures", ()))


def _has_kernel(sequence: int, heads: int) -> bool:
    # Short-circuit on the head count before testing symbolic sequence lengths.
    # Tuple membership creates a large compound guard in dynamic torch.compile.
    for supported_sequence, supported_heads in _SHAPES:
        if heads == supported_heads and sequence == supported_sequence:
            return True
    return False


@functools.cache
def _workspace_size(sequence: int, heads: int, device: int) -> int:
    return cuda._C.deco_attention_workspace_size(sequence, heads, device)


def is_available(device: torch.device | str | None = None) -> bool:
    """Whether this build has a native DeCo kernel for the selected device."""
    device = torch.device("cuda" if device is None else device)
    if device.type != "cuda" or not torch.cuda.is_available() or not registry.is_available("cuda"):
        return False
    return torch.cuda.get_device_capability(device) in _ARCHITECTURES


def _single_writer(name: str, sequence: int, heads: int) -> bool:
    if "RTX PRO 2000" in name:
        return (sequence, heads) in ((8704, 24), (22563, 56), (27280, 24))
    if "RTX 5080" in name:
        return (sequence, heads) != (16481, 56)
    return not ("GB10" in name and (sequence, heads) == (32768, 1))


@functools.cache
def _rotation(device: torch.device) -> torch.Tensor:
    # DeCo uses the Sylvester matrix with signed rows and no normalization.
    matrix = torch.ones((1, 1), device="cpu")
    while matrix.shape[0] < 128:
        top = torch.cat((matrix, matrix), dim=1)
        bottom = torch.cat((matrix, -matrix), dim=1)
        matrix = torch.cat((top, bottom), dim=0)
    generator = torch.Generator(device="cpu").manual_seed(0)
    signs = 2 * torch.randint(2, (128, 1), device="cpu", generator=generator) - 1
    return (signs * matrix).to(device=device, dtype=torch.bfloat16)


@functools.cache
def _sampling_constants(sequence: int, device: torch.device, tagged: bool):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "Warm up deco_attention once with this shape and tag mode before CUDA graph capture"
        )
    if tagged:
        mask = (1 << 30) - 1
        x = (torch.arange(sequence, dtype=torch.int64, device=device) + 0x9E3779B9) & mask
        x = ((x ^ (x >> 16)) * 0x85EBCA6B) & mask
        x = ((x ^ (x >> 13)) * 0xC2B2AE35) & mask
        table = (x ^ (x >> 16)).argsort().to(torch.int32)
    else:
        generator = torch.Generator(device=device).manual_seed(0)
        perm = torch.randperm(sequence, device=device, generator=generator)
        table = perm[: max(64, sequence // 256)].to(torch.int32)
    rotation = _rotation(device)
    # Publish immutable device data after initialization, including to streams
    # that may already be capturing. This wait is paid only on a cold entry.
    torch.cuda.current_stream(device).synchronize()
    return table, rotation


@torch.library.custom_op("comfy_kitchen::deco_attention", mutates_args=(), device_types="cuda")
def _deco_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, token_tags: torch.Tensor | None
) -> torch.Tensor:
    _, heads, sequence, dim = q.shape
    with torch.cuda.device(q.device):
        table, rotation = _sampling_constants(sequence, q.device, token_tags is not None)
        with allocation_context():
            out = torch.empty((1, sequence, heads, dim), dtype=q.dtype, device=q.device)
            workspace = torch.empty(
                _workspace_size(sequence, heads, q.device.index), dtype=torch.uint8, device=q.device
            )
        rows = table
        stream = torch.cuda.current_stream(q.device).cuda_stream
        if token_tags is not None:
            tags = token_tags.resolve_neg().reshape(-1)
            if tags.data_ptr() % tags.element_size():
                tags = tags.clone()
            rows = torch.empty(max(64, sequence // 256), dtype=torch.int32, device=q.device)
            cuda._C.deco_attention_sample_rows(
                *(cuda._wrap_for_dlpack(t) for t in (tags, table, rows, workspace)), stream
            )
        # DLPack exposes raw storage. Materialize lazy signs and align external buffers.
        q, k, v = (
            t.clone(memory_format=torch.contiguous_format) if t.data_ptr() % 16 else t
            for t in (q.resolve_neg(), k.resolve_neg(), v.resolve_neg())
        )
        tensors = (q, k, v, rows, rotation, out, workspace)
        cuda._C.deco_attention_forward(
            *(cuda._wrap_for_dlpack(t) for t in tensors),
            _single_writer(torch.cuda.get_device_name(q.device), sequence, heads),
            stream,
        )
        return out.transpose(1, 2)


@_deco_attention.register_fake
def _fake(q, k, v, token_tags):
    batch, heads, sequence, dim = q.shape
    return q.new_empty((batch, sequence, heads, dim)).transpose(1, 2)


def deco_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: float | None = None,
    enable_gqa: bool = False,
    token_tags: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply SDPA to BHND tensors, using rank-32 compensated NVFP4 on supported inference shapes.

    token_tags optionally supplies one integer per token ([N] or [1,N]) on the input device.
    Zero marks video/image, positive marks other modalities, and negative excludes a token
    from basis sampling. Tags do not mask attention. Other SDPA shapes and modes use PyTorch.
    """
    if token_tags is not None and (
        q.ndim < 2
        or token_tags.shape not in ((q.shape[-2],), (1, q.shape[-2]))
        or token_tags.device != q.device
        or token_tags.dtype not in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64)
    ):
        raise ValueError("token_tags must contain one integer per query token on the query device")
    if (
        q.is_cuda
        and _ARCHITECTURES
        and registry.is_available("cuda")
        and attn_mask is None
        and dropout_p == 0.0
        and not is_causal
        and not enable_gqa
        and (scale is None or scale in (128**-0.5, 1.0 / 128**0.5))
        and q.device == k.device == v.device
        and q.dtype == k.dtype == v.dtype == torch.bfloat16
        and not (
            torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad)
        )
        and all(t.layout == torch.strided and not t.is_nested for t in (q, k, v))
        and q.ndim == 4
        and q.shape[0] == 1
        and q.shape[-1] == 128
        and _has_kernel(q.shape[2], q.shape[1])
        and q.shape == k.shape == v.shape
        and all(
            t.stride(-1) == 1
            and t.stride(2) > 0
            and t.stride(2) % 8 == 0
            and (q.shape[1] == 1 or (t.stride(1) > 0 and t.stride(1) % 8 == 0))
            for t in (q, k, v)
        )
        and torch.cuda.get_device_capability(q.device) in _ARCHITECTURES
    ):
        return _deco_attention(q, k, v, token_tags)
    return functional.scaled_dot_product_attention(
        q, k, v, attn_mask, dropout_p, is_causal, scale=scale, enable_gqa=enable_gqa
    )
