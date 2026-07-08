# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Eager ConvRot W4A4 kernels."""
from __future__ import annotations

import math

import torch

from .svdquant import _INT4_GROUP_SIZE, _INT4_MAX, _pack_int4_row_major, _unpack_int4_row_major

_HADAMARD_CACHE = {}


def _build_hadamard(
    size: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    cache_key = (size, str(device), dtype)
    if cache_key in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[cache_key]

    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")

    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=dtype,
        device=device,
    )

    h = h4
    current_size = 4
    while current_size < size:
        h = torch.kron(h, h4)
        current_size *= 4

    h_normalized = h / (size**0.5)
    _HADAMARD_CACHE[cache_key] = h_normalized
    return h_normalized


def _rotate_weight(
    weight: torch.Tensor,
    h: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    out_f, in_f = weight.shape
    if in_f % group_size != 0:
        raise ValueError(f"in_features {in_f} not divisible by group_size {group_size}")
    n_groups = in_f // group_size

    weight_grouped = weight.reshape(out_f, n_groups, group_size)
    h_t = h.T.to(dtype=weight.dtype, device=weight.device)
    weight_rotated = torch.matmul(weight_grouped, h_t)
    return weight_rotated.reshape(out_f, in_f)


def _rotate_activation(
    x: torch.Tensor,
    h: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    orig_shape = x.shape
    features = orig_shape[-1]
    if features % group_size != 0:
        raise ValueError(f"features {features} not divisible by group_size {group_size}")
    n_groups = features // group_size

    x_grouped = x.reshape(-1, n_groups, group_size)
    h = h.to(dtype=x.dtype, device=x.device)
    x_rotated = torch.matmul(x_grouped, h)
    return x_rotated.reshape(orig_shape)


def validate_w4a4_shape(tensor: torch.Tensor, convrot_groupsize: int, quant_group_size: int) -> None:
    if tensor.dim() != 2:
        raise ValueError(f"ConvRot W4A4 expects a 2D tensor, got shape {tuple(tensor.shape)}")
    k = tensor.shape[-1]
    if k % convrot_groupsize != 0:
        raise ValueError(f"in_features {k} not divisible by convrot_groupsize {convrot_groupsize}")
    if k % quant_group_size != 0:
        raise ValueError(f"in_features {k} not divisible by quant_group_size {quant_group_size}")


def _int4_stochastic_rng(x: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)
    return torch.rand(
        x.shape,
        dtype=x.dtype,
        layout=x.layout,
        device=x.device,
        generator=generator,
    )


def _round_int4(scaled: torch.Tensor, stochastic_rounding: int | None = 0) -> torch.Tensor:
    if stochastic_rounding is not None and stochastic_rounding > 0:
        rng = _int4_stochastic_rng(scaled, stochastic_rounding)
        scaled.add_(rng)
        return scaled.floor_().clamp_(-_INT4_MAX, _INT4_MAX).to(torch.int8)
    return scaled.round_().clamp_(-_INT4_MAX, _INT4_MAX).to(torch.int8)


def quantize_signed_int4_rowwise(
    x: torch.Tensor,
    stochastic_rounding: int | None = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, _ = x.shape
    absmax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    scales = absmax / _INT4_MAX
    q = _round_int4(x / scales, stochastic_rounding=stochastic_rounding)
    return _pack_int4_row_major(q), scales.reshape(rows).to(torch.float32)


def prepare_int4_weight_for_int8_linear(weight: torch.Tensor) -> torch.Tensor:
    """Unpack a packed signed INT4 weight matrix into INT8 values."""
    if weight.dim() != 2:
        raise ValueError("prepared INT4 fallback weight expects a 2D tensor")
    return _unpack_int4_row_major(weight).to(torch.int8)


def quantize_convrot_w4a4_weight(
    weight: torch.Tensor,
    convrot_groupsize: int = 256,
    quant_group_size: int = _INT4_GROUP_SIZE,
    stochastic_rounding: int | None = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate a weight matrix with ConvRot and pack it as signed W4."""
    if quant_group_size != _INT4_GROUP_SIZE:
        raise ValueError(f"int4 MMA kernel requires quant_group_size {_INT4_GROUP_SIZE}")
    validate_w4a4_shape(weight, convrot_groupsize, quant_group_size)
    h = _build_hadamard(convrot_groupsize, device=weight.device, dtype=weight.dtype)
    weight_rot = _rotate_weight(weight, h, convrot_groupsize)
    return quantize_signed_int4_rowwise(weight_rot, stochastic_rounding=stochastic_rounding)


def dequantize_convrot_w4a4_weight(
    qdata: torch.Tensor,
    scales: torch.Tensor,
    convrot_groupsize: int = 256,
    quant_group_size: int = _INT4_GROUP_SIZE,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize packed ConvRot W4A4 weights and rotate back to original basis."""
    if quant_group_size != _INT4_GROUP_SIZE:
        raise ValueError(f"int4 MMA kernel requires quant_group_size {_INT4_GROUP_SIZE}")
    w_int = _unpack_int4_row_major(qdata).to(torch.float32)
    w_rot = w_int * scales.to(device=qdata.device, dtype=torch.float32).reshape(-1, 1)
    h = _build_hadamard(convrot_groupsize, device=qdata.device, dtype=torch.float32)
    return _rotate_weight(w_rot.float(), h, convrot_groupsize).to(output_dtype)


def int4_linear(
    x_qdata: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    x_int = _unpack_int4_row_major(x_qdata).to(dtype=out_dtype)
    w_int = _unpack_int4_row_major(weight).to(dtype=out_dtype)
    out = x_int @ w_int.t()
    out = out * x_scale.reshape(-1, 1).to(device=out.device, dtype=out_dtype)
    out = out * weight_scale.reshape(1, -1).to(device=out.device, dtype=out_dtype)
    if bias is not None:
        out = out + bias.to(device=out.device, dtype=out_dtype).reshape(1, -1)
    return out


def convrot_w4a4_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    wscales: torch.Tensor,
    bias: torch.Tensor | None = None,
    convrot_groupsize: int = 256,
    quant_group_size: int = _INT4_GROUP_SIZE,
) -> torch.Tensor:
    """Compute ``x @ W.T + bias`` using the eager ConvRot W4A4 path."""
    if quant_group_size != _INT4_GROUP_SIZE:
        raise ValueError(f"int4 MMA kernel requires quant_group_size {_INT4_GROUP_SIZE}")
    if x.shape[-1] != qweight.shape[-1] * 2:
        raise ValueError(f"Input K={x.shape[-1]} does not match qweight K={qweight.shape[-1] * 2}")
    if x.shape[-1] % convrot_groupsize != 0:
        raise ValueError(f"Input K={x.shape[-1]} not divisible by convrot_groupsize {convrot_groupsize}")

    orig_shape = x.shape
    x2d = x.reshape(-1, orig_shape[-1]).contiguous()
    h = _build_hadamard(convrot_groupsize, device=x2d.device, dtype=x2d.dtype)
    x_rot = _rotate_activation(x2d, h, convrot_groupsize).contiguous()
    qact, x_scale = quantize_signed_int4_rowwise(x_rot)
    out = int4_linear(qact, qweight, x_scale, wscales, bias, x.dtype)
    return out[: x2d.shape[0]].reshape(*orig_shape[:-1], qweight.shape[0])
