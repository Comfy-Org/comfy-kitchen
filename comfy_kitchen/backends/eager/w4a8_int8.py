# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Eager W4A8 weight preparation, dequantization, and linear operations."""

from __future__ import annotations

import torch

from .quantization import int8_linear, rotate_int8_convrot_weight

# Lloyd-Max-optimal 16 levels for a group-normalized Gaussian. ConvRot makes every layer's
# rotated groups Gaussian, so this one table matches a per-tensor fit and skips the k-means.
_FIXED_LUT = (
    -0.980602, -0.794529, -0.638165, -0.500986, -0.377321, -0.263187, -0.155210, -0.050720,
    0.052541, 0.156985, 0.265284, 0.379533, 0.502636, 0.638953, 0.794876, 0.980671,
)
# Fit per-tensor instead when the rotated groups stay heavy-tailed. Gaussian layers sit near
# -0.5; the threshold only trips on genuinely non-Gaussian ones (real models never do).
_W4A8_GATE_KURTOSIS = -0.1
_KURTOSIS_SAMPLE = 1 << 19


def _codebook_for(normalized: torch.Tensor) -> torch.Tensor:
    """Frozen LUT unless the group-normalized rotated weight is heavy-tailed, then fit."""
    x = normalized.detach().flatten()
    if x.numel() > _KURTOSIS_SAMPLE:
        gen = torch.Generator(device=x.device).manual_seed(0)
        x = x[torch.randint(0, x.numel(), (_KURTOSIS_SAMPLE,), device=x.device, generator=gen)]
    x = x.float()
    excess_kurtosis = ((x - x.mean()) / (x.std() + 1e-9)).pow(4).mean() - 3.0
    if excess_kurtosis.item() <= _W4A8_GATE_KURTOSIS:
        return torch.tensor(_FIXED_LUT, device=normalized.device, dtype=torch.float32)
    return _fit_codebook(normalized)


def _assign_codes(normalized: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Find nearest codebook indices without a trailing 16-wide temporary."""
    best = (normalized - codebook[0]).abs()
    indices = torch.zeros_like(normalized, dtype=torch.int32)
    for index in range(1, codebook.numel()):
        distance = (normalized - codebook[index]).abs()
        closer = distance < best
        best = torch.where(closer, distance, best)
        indices = torch.where(closer, index, indices)
    return indices


def _assign_grid(
    weight: torch.Tensor,
    levels: torch.Tensor,
    s_channel: torch.Tensor,
) -> torch.Tensor:
    """Find the nearest decoded INT8 level for each grouped weight."""
    target = weight / s_channel.view(-1, 1, 1)
    best = (target - levels[..., 0:1].expand_as(weight)).abs()
    indices = torch.zeros_like(weight, dtype=torch.int32)
    for index in range(1, 16):
        distance = (target - levels[..., index : index + 1].expand_as(weight)).abs()
        closer = distance < best
        best = torch.where(closer, distance, best)
        indices = torch.where(closer, index, indices)
    return indices


def _fit_codebook(
    normalized: torch.Tensor,
    levels: int = 16,
    iterations: int = 25,
    sample_size: int = 300000,
) -> torch.Tensor:
    """Fit a data-free Lloyd-Max codebook to normalized rotated weights."""
    samples = normalized.flatten()
    if samples.numel() > sample_size:
        generator = torch.Generator(device=samples.device).manual_seed(0)
        indices = torch.randint(
            0,
            samples.numel(),
            (sample_size,),
            device=samples.device,
            generator=generator,
        )
        samples = samples[indices]
    samples = samples.float()
    codebook = torch.quantile(
        samples,
        torch.linspace(0, 1, levels, device=samples.device),
    )
    for _ in range(iterations):
        assignments = (samples.unsqueeze(-1) - codebook).abs().argmin(-1)
        updated = codebook.clone()
        for index in range(levels):
            selected = assignments == index
            if selected.any():
                updated[index] = samples[selected].mean()
        codebook = updated
    return codebook.contiguous()


def validate_w4a8_weight_shape(
    weight: torch.Tensor,
    group_size: int,
    convrot_groupsize: int,
) -> None:
    """Validate a floating weight before W4A8 preparation."""
    if weight.dim() != 2:
        raise ValueError(f"W4A8 weight must be 2D, got shape {tuple(weight.shape)}")
    k = weight.shape[1]
    if (
        k % 16 != 0
        or k % group_size != 0
        or k % convrot_groupsize != 0
        or group_size < 4
        or (16 % group_size != 0 and group_size % 16 != 0)
    ):
        raise ValueError(
            f"K={k} must be divisible by 16, group_size={group_size}, and "
            f"convrot_groupsize={convrot_groupsize}; group_size must be >=4 "
            f"and divide 16 or be a multiple of 16"
        )


def _quantize_rotated_w4a8_int8_weight(
    weight: torch.Tensor,
    group_size: int = 16,
    symmetric: bool = True,
    scale_dtype: torch.dtype = torch.float8_e4m3fn,
    codebook: bool = True,
    codebook_override: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Quantize an already ConvRot-rotated weight into W4A8 storage.

    ``codebook_override`` lets a chunked caller pin the same table across chunks; when None
    it is chosen from this (sub)tensor via _codebook_for.
    """
    if scale_dtype not in (torch.float32, torch.float8_e4m3fn):
        raise ValueError(f"scale_dtype must be float32 or float8_e4m3fn, got {scale_dtype}")

    original_dtype = weight.dtype
    n, k = weight.shape
    groups = k // group_size
    grouped_weight = weight.float().view(n, groups, group_size)

    codebook_tensor = None
    if symmetric and codebook:
        group_scale = grouped_weight.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        normalized = grouped_weight / group_scale
        codebook_tensor = (
            codebook_override
            if codebook_override is not None
            else _codebook_for(normalized)
        )
        quantized = _assign_codes(normalized, codebook_tensor)
        for _ in range(3):
            quantized_codebook = codebook_tensor[quantized]
            group_scale = (
                (grouped_weight * quantized_codebook).sum(-1, keepdim=True)
                / (quantized_codebook * quantized_codebook).sum(-1, keepdim=True).clamp(min=1e-8)
            ).clamp(min=1e-8)
            quantized = _assign_codes(grouped_weight / group_scale, codebook_tensor)
        unsigned = quantized.to(torch.int32).view(n, k)
        shifted_weight = codebook_tensor[quantized] * group_scale
        correction = None
    elif symmetric:
        group_scale = (grouped_weight.abs().amax(dim=-1, keepdim=True) / 7.0).clamp(min=1e-8)
        signed = (grouped_weight / group_scale).round().clamp(-8, 7).to(torch.int32)
        unsigned = (signed + 8).view(n, k)
        shifted_weight = signed * group_scale
        correction = None
    else:
        minimum = grouped_weight.amin(dim=-1, keepdim=True)
        group_scale = ((grouped_weight.amax(dim=-1, keepdim=True) - minimum) / 15.0).clamp(min=1e-8)
        unsigned = (
            ((grouped_weight - minimum) / group_scale)
            .round()
            .clamp(0, 15)
            .to(torch.int32)
            .view(n, k)
        )
        shifted_weight = (unsigned.view(n, groups, group_size) - 8) * group_scale
        correction = (8.0 * group_scale + minimum).squeeze(-1).t().contiguous().to(original_dtype)

    s_channel = (shifted_weight.abs().amax(dim=(1, 2)) / 127.0).clamp(min=1e-8)
    s_rel = (group_scale.squeeze(-1) / s_channel.unsqueeze(1)).float().contiguous()
    if scale_dtype != torch.float32:
        s_rel = s_rel.to(scale_dtype).contiguous()
    if codebook_tensor is not None:
        levels = (
            (codebook_tensor.view(1, 1, 16) * s_rel.float().unsqueeze(-1))
            .round_()
            .clamp_(-127, 127)
        )
        unsigned = _assign_grid(grouped_weight, levels, s_channel).view(n, k)

    packed = (
        ((unsigned[:, 0::2] & 0xF) | ((unsigned[:, 1::2] & 0xF) << 4)).to(torch.int8).contiguous()
    )
    return (
        packed,
        s_rel,
        s_channel.float().contiguous(),
        correction,
        codebook_tensor,
    )


# Rows to rotate+pack per chunk: caps the fp32 working set that otherwise scales with the
# whole tensor and OOMs on large layers (e.g. the LoRA requantize path). A 1-row floor keeps
# wide weights within budget (a fixed row floor would blow it for large k).
_QUANT_ROW_ELEM_BUDGET = 1 << 22  # ~4M elems/chunk
# Elements sampled to decide the codebook. Independent of the chunk size so the chosen table
# is a function of the whole tensor, never of how it happens to be chunked.
_CODEBOOK_SAMPLE_ELEMS = 1 << 22


def _decide_codebook(
    weight: torch.Tensor,
    rotate_fn,
    group_size: int,
    convrot_groupsize: int,
) -> torch.Tensor:
    """Choose one codebook for the whole tensor from a bounded, deterministic row sample that
    spans every chunk (evenly strided rows), so the choice never depends on the chunk size."""
    n, k = weight.shape
    max_rows = max(1, _CODEBOOK_SAMPLE_ELEMS // max(k, 1))
    if n > max_rows:
        idx = torch.linspace(0, n - 1, max_rows, device=weight.device).long()
        sample = weight.index_select(0, idx)
    else:
        sample = weight
    rotated = rotate_fn(sample.contiguous(), convrot_groupsize)
    grouped = rotated.float().view(sample.shape[0], k // group_size, group_size)
    normalized = grouped / grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    return _codebook_for(normalized)


def _quantize_w4a8_chunked(
    weight: torch.Tensor,
    rotate_fn,
    group_size: int,
    convrot_groupsize: int,
    symmetric: bool,
    scale_dtype: torch.dtype,
    codebook: bool,
    codebook_override: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Rotate and pack a weight in row chunks to cap peak memory.

    One codebook is chosen for the whole tensor -- from a caller ``codebook_override`` (e.g. a
    LoRA requantize reusing the prior table), else from a whole-tensor-spanning sample -- and
    shared across every chunk, so the packed result does not depend on the chunk size. Row math
    is otherwise independent.
    """
    n, k = weight.shape
    block = max(1, _QUANT_ROW_ELEM_BUDGET // max(k, 1))
    override = None
    if symmetric and codebook:
        override = (
            codebook_override.to(device=weight.device, dtype=torch.float32)
            if codebook_override is not None
            else _decide_codebook(weight, rotate_fn, group_size, convrot_groupsize)
        )

    if n <= block:
        return _quantize_rotated_w4a8_int8_weight(
            rotate_fn(weight.contiguous(), convrot_groupsize),
            group_size=group_size,
            symmetric=symmetric,
            scale_dtype=scale_dtype,
            codebook=codebook,
            codebook_override=override,
        )

    packed_parts, srel_parts, sch_parts, corr_parts = [], [], [], []
    codebook_tensor = None
    for r0 in range(0, n, block):
        rot = rotate_fn(weight[r0 : r0 + block].contiguous(), convrot_groupsize)
        packed, s_rel, s_channel, correction, codebook_tensor = (
            _quantize_rotated_w4a8_int8_weight(
                rot,
                group_size=group_size,
                symmetric=symmetric,
                scale_dtype=scale_dtype,
                codebook=codebook,
                codebook_override=override,
            )
        )
        packed_parts.append(packed)
        srel_parts.append(s_rel)
        sch_parts.append(s_channel)
        if correction is not None:
            corr_parts.append(correction)
        del rot

    return (
        torch.cat(packed_parts, dim=0),
        torch.cat(srel_parts, dim=0),
        torch.cat(sch_parts, dim=0),
        torch.cat(corr_parts, dim=1) if corr_parts else None,
        codebook_tensor,
    )


def quantize_w4a8_int8_weight(
    weight: torch.Tensor,
    group_size: int = 16,
    convrot_groupsize: int = 256,
    symmetric: bool = True,
    scale_dtype: torch.dtype = torch.float8_e4m3fn,
    codebook: bool = True,
    codebook_tensor: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Rotate and prepare a floating weight for grouped W4A8 storage."""
    validate_w4a8_weight_shape(weight, group_size, convrot_groupsize)
    return _quantize_w4a8_chunked(
        weight,
        rotate_int8_convrot_weight,
        group_size=group_size,
        convrot_groupsize=convrot_groupsize,
        symmetric=symmetric,
        scale_dtype=scale_dtype,
        codebook=codebook,
        codebook_override=codebook_tensor,
    )


def _dequant_int4_grouped_to_int8(
    qdata: torch.Tensor,
    s_rel: torch.Tensor,
    codebook: torch.Tensor | None,
    group_size: int,
) -> torch.Tensor:
    """Decode grouped INT4 storage to the INT8 grid consumed by the GEMM."""
    n, k_half = qdata.shape
    k = k_half * 2
    groups = k // group_size
    packed = qdata.to(torch.int32) & 0xFF
    quantized = torch.empty(n, k, dtype=torch.int32, device=qdata.device)
    quantized[:, 0::2] = packed & 0xF
    quantized[:, 1::2] = (packed >> 4) & 0xF
    if codebook is not None:
        values = codebook.to(device=qdata.device, dtype=torch.float32)[quantized]
    else:
        values = quantized.float() - 8.0
    values = values.view(n, groups, group_size) * s_rel.float().unsqueeze(-1)
    return values.view(n, k).round().clamp_(-127, 127).to(torch.int8)


def validate_w4a8_operands(
    qdata: torch.Tensor,
    s_rel: torch.Tensor,
    s_channel: torch.Tensor,
    codebook: torch.Tensor | None,
    correction: torch.Tensor | None,
    group_size: int,
    convrot_groupsize: int,
) -> None:
    """Validate packed W4A8 tensors shared by dequantize and linear."""
    if qdata.dim() != 2 or qdata.dtype != torch.int8:
        raise ValueError("packed weight must be a 2D int8 tensor")
    n, k_half = qdata.shape
    k = k_half * 2
    if (
        group_size < 4
        or k % 16 != 0
        or k % group_size != 0
        or k % convrot_groupsize != 0
        or (16 % group_size != 0 and group_size % 16 != 0)
    ):
        raise ValueError(
            f"K={k} must be divisible by 16, group_size={group_size}, and "
            f"convrot_groupsize={convrot_groupsize}; group_size must be >=4 "
            f"and divide 16 or be a multiple of 16"
        )
    groups = k // group_size
    expected_scale_shape = (n, groups)
    if tuple(s_rel.shape) != expected_scale_shape:
        raise ValueError(f"s_rel must have shape {expected_scale_shape}, got {tuple(s_rel.shape)}")
    if tuple(s_channel.shape) != (n,):
        raise ValueError(f"s_channel must have shape {(n,)}, got {tuple(s_channel.shape)}")
    if correction is not None and tuple(correction.shape) != (groups, n):
        raise ValueError(f"correction must have shape {(groups, n)}, got {tuple(correction.shape)}")
    if codebook is not None and tuple(codebook.shape) != (16,):
        raise ValueError(f"codebook must have shape (16,), got {tuple(codebook.shape)}")


def _dequantize_w4a8_int8_weight_from_int8(
    int8_weight: torch.Tensor,
    s_channel: torch.Tensor,
    correction: torch.Tensor | None,
    group_size: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Apply channel scaling and correction to decoded grouped INT8 weights."""
    n, k = int8_weight.shape
    groups = k // group_size
    weight_rotated = int8_weight.float().view(n, groups, group_size)
    weight_rotated = weight_rotated * s_channel.float().view(n, 1, 1)
    if correction is not None:
        weight_rotated = weight_rotated + correction.t().unsqueeze(-1).float()
    return weight_rotated.view(n, k).to(output_dtype)


def dequantize_w4a8_int8_weight(
    qdata: torch.Tensor,
    s_rel: torch.Tensor,
    s_channel: torch.Tensor,
    codebook: torch.Tensor | None = None,
    correction: torch.Tensor | None = None,
    group_size: int = 16,
    convrot_groupsize: int = 256,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Decode W4A8 storage into its physical [N, K] floating weight."""
    validate_w4a8_operands(
        qdata,
        s_rel,
        s_channel,
        codebook,
        correction,
        group_size,
        convrot_groupsize,
    )
    int8_weight = _dequant_int4_grouped_to_int8(
        qdata,
        s_rel,
        codebook,
        group_size,
    )
    weight_rotated = _dequantize_w4a8_int8_weight_from_int8(
        int8_weight,
        s_channel,
        correction,
        group_size,
        output_dtype,
    )
    return rotate_int8_convrot_weight(weight_rotated, convrot_groupsize).to(output_dtype)


def w4a8_int8_linear(
    x: torch.Tensor,
    qdata: torch.Tensor,
    s_rel: torch.Tensor,
    s_channel: torch.Tensor,
    codebook: torch.Tensor | None = None,
    correction: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    group_size: int = 16,
    convrot_groupsize: int = 256,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Compute x @ W.T + bias using portable W4A8 operations."""
    validate_w4a8_operands(
        qdata,
        s_rel,
        s_channel,
        codebook,
        correction,
        group_size,
        convrot_groupsize,
    )
    if x.shape[-1] != qdata.shape[-1] * 2:
        raise ValueError(f"Input K={x.shape[-1]} does not match qdata K={qdata.shape[-1] * 2}")
    if correction is not None:
        weight = dequantize_w4a8_int8_weight(
            qdata,
            s_rel,
            s_channel,
            codebook=codebook,
            correction=correction,
            group_size=group_size,
            convrot_groupsize=convrot_groupsize,
            output_dtype=x.dtype,
        )
        return torch.nn.functional.linear(x, weight, bias).to(out_dtype)

    int8_weight = _dequant_int4_grouped_to_int8(
        qdata,
        s_rel,
        codebook,
        group_size,
    )
    return int8_linear(
        x,
        int8_weight,
        s_channel,
        bias=bias,
        out_dtype=out_dtype,
        convrot=True,
        convrot_groupsize=convrot_groupsize,
    )
