# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Eager W3A8 / W4A8 / W5A8 / W6A8 weight preparation, dequantization, and linear operations.

All bit widths share one storage contract: ``qdata`` is an int8 [N, K*bits/8] tensor built from
bit planes, and every consumer reads the row width and infers the bit depth:
  4 bits: a K/2-byte nibble plane (even col = low nibble).
  6 bits: the nibble plane, then a K/4-byte 2-bit plane of each code's top 2 bits (col c in
          byte c//4 at bit 2*(c%4)).
  5 bits: the nibble plane, then a K/8-byte 1-bit plane of each code's top bit (col c in byte
          c//8 at bit c%8).
  3 bits: a K/4-byte 2-bit plane of each code's low 2 bits (byte c//4, bit 2*(c%4)), then the
          K/8-byte 1-bit plane of the top bit.
Levels: 4-bit codes index a 16-entry codebook (or the uniform q-8 grid), 3-bit codes an 8-entry
Lloyd-Max codebook, 5- and 6-bit codes are uniform (c-16, c-32) with no codebook.
"""

from __future__ import annotations

import torch

from comfy_kitchen.backends._activations import apply_input_act, apply_residual
from comfy_kitchen.tensor.w4a8_packed import _pack_codes, _unpack_codes, default_w4a8_codebook, unpack_w4a8_mma_weight

from .quantization import int8_linear, rotate_int8_convrot_weight

# Fit per-tensor instead when the rotated groups stay heavy-tailed. Gaussian layers sit near
# -0.5; the threshold only trips on genuinely non-Gaussian ones (real models never do).
_W4A8_GATE_KURTOSIS = -0.1
_KURTOSIS_SAMPLE = 1 << 19
# ALS group-scale refinement passes. Iteration 1 captures nearly all the gain (relL2 ~0.0706);
# further passes add < 0.001, so 2 is the quality/speed knee (each pass is a gather + reduction
# + reassign over the whole tensor, and requantize reruns this per layer under VRAM offload).
_ALS_ITERS = 2
# W6A8 codes are uniform: -31..31 stored as q+32 in [1, 63]. A 64-level Lloyd-Max table
# measured WORSE than uniform at this rate (both in fp32 and through the int8 grid), so the
# 6-bit path has no codebook and the kernels decode (c - 32) * s_rel.
_W6_LEVELS = 31
# W5A8 follows the same uniform recipe one bit down: -15..15 stored as q+16 in [1, 31].
_W5_LEVELS = 15
_UNIFORM_LEVELS = {5: _W5_LEVELS, 6: _W6_LEVELS}
# W3A8 codes index an 8-entry Lloyd-Max table (a uniform 8-level grid measured ~7% worse on
# rotated weights): the frozen _FIXED_LUT8 under the same kurtosis gate as the 16-entry table
# (a per-tensor fit lands on the same levels), stored like the 4-bit codebook.
_W3_CODEBOOK_LEVELS = 8
_CODEBOOK_LEVELS = {3: _W3_CODEBOOK_LEVELS, 4: 16}
_SUPPORTED_BITS = (3, 4, 5, 6)
# Grid-aware group-scale search (W6A8, quantize-time only): the ALS scale and its two
# stored-dtype neighbours (one e4m3 ulp = 12.5% either side; +-3% for fp32 scales), keeping
# per group the one whose decoded int8 levels reconstruct the group best. That reaches the
# int8 rounding floor at g16; a wider search measured no gain.
_SCALE_SEARCH_FP32_STEP = 0.03
_E4M3_MAX_FINITE_BITS = 0x7E  # 0x7F is NaN in e4m3fn


def _check_plane_layout(k: int, group_size: int, bits: int) -> None:
    """Multi-plane rows (3, 5, 6 bits) need every plane 8-byte aligned and a 16-col decode
    vector that never spans groups: K % 32 at 6 bits (planes of K/2 and K/4 bytes), K % 64
    at 3 and 5 bits (a K/8-byte 1-bit plane), group_size a multiple of 16. Finer groups buy
    nothing at these widths, so they are not a supported layout."""
    k_align = 32 if bits == 6 else 64
    if k % k_align != 0 or group_size < 16 or group_size % 16 != 0:
        raise ValueError(
            f"{bits}-bit storage needs K divisible by {k_align} and group_size a multiple of 16, "
            f"got K={k}, group_size={group_size}"
        )


def _w4a8_geometry(
    qdata: torch.Tensor, s_rel: torch.Tensor, group_size: int
) -> tuple[int, int, int]:
    """(n, k, bits) of a packed weight: K from the scale grid, bits from the packed width.
    The one place the Python side infers and validates the storage contract."""
    n, cols = qdata.shape
    k = s_rel.shape[1] * group_size
    bits = (cols * 8) // k if k > 0 else 0
    if k <= 0 or bits not in _SUPPORTED_BITS or k * bits != cols * 8:
        raise ValueError(f"packed row width {cols} does not match K={k} at 3, 4, 5 or 6 bits")
    if bits != 4:
        _check_plane_layout(k, group_size, bits)
    return n, k, bits


def _codebook_for(normalized: torch.Tensor, levels: int = 16) -> torch.Tensor:
    """Frozen LUT (16 levels at 4 bits, 8 at 3) unless the group-normalized rotated weight is
    heavy-tailed, then fit per tensor."""
    x = normalized.detach().flatten()
    if x.numel() > _KURTOSIS_SAMPLE:
        gen = torch.Generator(device=x.device).manual_seed(0)
        x = x[torch.randint(0, x.numel(), (_KURTOSIS_SAMPLE,), device=x.device, generator=gen)]
    x = x.float()
    excess_kurtosis = ((x - x.mean()) / (x.std() + 1e-9)).pow(4).mean() - 3.0
    if excess_kurtosis.item() <= _W4A8_GATE_KURTOSIS:
        return default_w4a8_codebook(levels).to(normalized.device)
    return _fit_codebook(normalized, levels=levels)


def _assign_codes(normalized: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Nearest codebook index via binary search on the sorted codebook.

    The codebook is monotonic, so the nearest level is one of the two neighbors of the
    insertion point -- one searchsorted + a neighbor compare, vs a 16-pass linear scan.
    Bit-identical to the scan (ties resolve to the lower index in both) and ~4x faster,
    which matters because the ALS refinement calls this several times per requantize.
    """
    last = codebook.numel() - 1
    pos = torch.searchsorted(codebook, normalized.contiguous())
    lo = (pos - 1).clamp(0, last)
    hi = pos.clamp(0, last)
    dlo = (normalized - codebook[lo]).abs_()
    dhi = (normalized - codebook[hi]).abs_()
    return torch.where(dhi < dlo, hi, lo).to(torch.int32)


def _stochastic_round(x: torch.Tensor, seed: int) -> torch.Tensor:
    """Round to nearest integer with probability 1-frac, up with probability frac.

    floor(x + U[0,1)) is stochastic rounding: unbiased in expectation, so sub-quantum deltas
    (e.g. a small LoRA merged into a low-bit weight) survive instead of being rounded away.
    """
    gen = torch.Generator(device=x.device).manual_seed(int(seed))
    noise = torch.rand(x.shape, generator=gen, device=x.device, dtype=x.dtype)
    return torch.floor(x + noise)


def _round(x: torch.Tensor, stochastic_rounding: int) -> torch.Tensor:
    return _stochastic_round(x, stochastic_rounding) if stochastic_rounding > 0 else x.round()


def _assign_grid(
    weight: torch.Tensor,
    levels: torch.Tensor,
    s_channel: torch.Tensor,
    stochastic_rounding: int = 0,
    target: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pick a decoded INT8 codebook level for each grouped weight.

    Nearest by default; with ``stochastic_rounding`` > 0, round stochastically between the two
    bracketing (sorted) codebook levels so a merged LoRA delta below the int4 step is preserved.
    ``target`` is ``weight / s_channel`` if the caller already has it.
    """
    n, groups, gsize = weight.shape
    last = levels.shape[-1] - 1
    lv = levels.reshape(n * groups, last + 1).contiguous()  # per-group sorted levels
    if target is None:
        target = weight / s_channel.view(-1, 1, 1)
    tg = target.reshape(n * groups, gsize).contiguous()
    pos = torch.searchsorted(lv, tg)  # insertion point in the sorted levels
    if stochastic_rounding > 0:
        # SR = pick the upper level when the weight exceeds a uniform-random point in its
        # bracket. Reruns per layer during inference under VRAM offload, so keep it lean.
        lo = (pos - 1).clamp_(0, last - 1)  # lower bracket so hi = lo+1 stays in range
        level_lo = torch.gather(lv, 1, lo)
        width = torch.gather(lv, 1, lo + 1).sub_(level_lo)
        gen = torch.Generator(device=tg.device).manual_seed(int(stochastic_rounding))
        thr = torch.rand(tg.shape, generator=gen, device=tg.device, dtype=tg.dtype)
        thr.mul_(width).add_(level_lo)  # uniform point in [level[lo], level[lo+1])
        idx = (lo + (tg > thr)).clamp_(0, last)
        return idx.to(torch.int32).reshape(n, groups, gsize)
    lo = (pos - 1).clamp(0, last)
    hi = pos.clamp(0, last)
    dlo = tg.sub(torch.gather(lv, 1, lo)).abs_()
    dhi = tg.sub(torch.gather(lv, 1, hi)).abs_()
    return torch.where(dhi < dlo, hi, lo).to(torch.int32).reshape(n, groups, gsize)


def _grid_levels(level_table: torch.Tensor, s_rel: torch.Tensor) -> torch.Tensor:
    """The int8 values each level of every group decodes to: round(clamp(level * s_rel))."""
    return (level_table.view(1, 1, -1) * s_rel.float().unsqueeze(-1)).round_().clamp_(-127, 127)


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
    bits: int = 4,
) -> None:
    """Validate a floating weight (and the requested layout) before W4A8/W6A8 preparation."""
    if weight.dim() != 2:
        raise ValueError(f"W4A8 weight must be 2D, got shape {tuple(weight.shape)}")
    if bits not in _SUPPORTED_BITS:
        raise ValueError(f"bits must be 3, 4, 5 or 6, got {bits}")
    k = weight.shape[1]
    if bits != 4:
        _check_plane_layout(k, group_size, bits)
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
    stochastic_rounding: int = 0,
    bits: int = 4,
    scale_search: bool = True,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Quantize an already ConvRot-rotated weight into W3A8 / W4A8 / W5A8 / W6A8 storage.

    ``codebook_override`` lets a chunked caller pin the same table across chunks; when None
    it is chosen from this (sub)tensor via _codebook_for. ``stochastic_rounding`` > 0 seeds
    stochastic rounding of the final level assignment (the LoRA-merge path). ``bits`` 5 and 6
    are symmetric, uniform and codebook-free; 3 always uses an 8-entry codebook (frozen
    unless heavy-tailed, like 4);
    ``scale_search`` (5- and 6-bit only) runs the grid-aware group-scale search, which the
    requantize path turns off.
    """
    if scale_dtype not in (torch.float32, torch.float8_e4m3fn):
        raise ValueError(f"scale_dtype must be float32 or float8_e4m3fn, got {scale_dtype}")

    original_dtype = weight.dtype
    n, k = weight.shape
    groups = k // group_size
    grouped_weight = weight.float().view(n, groups, group_size)

    codebook_tensor = None
    level_table = None  # levels the grid assignment decodes against (codebook or uniform)
    uniform = bits in _UNIFORM_LEVELS
    if (uniform or bits == 3) and not symmetric:
        raise ValueError(f"{bits}-bit storage is symmetric only")
    if uniform or bits == 3 or (symmetric and codebook):
        # One ALS loop for both level tables. Uniform levels are the integers -L..L (L = 15
        # at 5 bits, 31 at 6), so the stored s_rel is the per-level step the kernels multiply
        # (c - L - 1) by and the group scale starts at amax / L; a codebook is normalized, so
        # its scale starts at amax.
        amax = grouped_weight.abs().amax(dim=-1, keepdim=True)
        if uniform:
            lmax = _UNIFORM_LEVELS[bits]
            level_table = torch.arange(-lmax, lmax + 1, device=weight.device, dtype=torch.float32)
            group_scale = (amax / lmax).clamp(min=1e-8)

            def assign(normalized):  # nearest integer level: round, no search needed
                return (normalized.round_().clamp_(-lmax, lmax) + lmax).to(torch.int32)

            quantized = assign(grouped_weight / group_scale)
        else:
            group_scale = amax.clamp(min=1e-8)
            normalized = grouped_weight / group_scale
            codebook_tensor = (
                codebook_override if codebook_override is not None
                else _codebook_for(normalized, _CODEBOOK_LEVELS[bits])
            )
            level_table = codebook_tensor

            def assign(normalized):
                return _assign_codes(normalized, level_table)

            quantized = assign(normalized)
        for _ in range(_ALS_ITERS):
            quantized_levels = level_table[quantized]
            group_scale = (
                (grouped_weight * quantized_levels).sum(-1, keepdim=True)
                / (quantized_levels * quantized_levels).sum(-1, keepdim=True).clamp(min=1e-8)
            ).clamp(min=1e-8)
            quantized = assign(grouped_weight / group_scale)
        shifted_weight = level_table[quantized] * group_scale
        correction = None
    elif symmetric:
        group_scale = (grouped_weight.abs().amax(dim=-1, keepdim=True) / 7.0).clamp(min=1e-8)
        signed = _round(grouped_weight / group_scale, stochastic_rounding).clamp(-8, 7).to(torch.int32)
        unsigned = (signed + 8).view(n, k)
        shifted_weight = signed * group_scale
        correction = None
    else:
        minimum = grouped_weight.amin(dim=-1, keepdim=True)
        group_scale = ((grouped_weight.amax(dim=-1, keepdim=True) - minimum) / 15.0).clamp(min=1e-8)
        unsigned = (
            _round((grouped_weight - minimum) / group_scale, stochastic_rounding)
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
    if level_table is not None:
        searched = uniform and scale_search
        if searched:
            s_rel, unsigned = _search_group_scale(grouped_weight, s_rel, s_channel, level_table)
        if not searched or stochastic_rounding > 0:  # SR needs its own (seeded) assignment
            levels = _grid_levels(level_table, s_rel)
            unsigned = _assign_grid(grouped_weight, levels, s_channel, stochastic_rounding)
        if uniform:
            unsigned.add_(1)  # level index 0..2L -> stored code q+L+1 (1..31 / 1..63)
        unsigned = unsigned.view(n, k)

    packed = _pack_codes(unsigned, bits)
    return (
        packed,
        s_rel,
        s_channel.float().contiguous(),
        correction,
        codebook_tensor,
    )


def _search_group_scale(
    grouped_weight: torch.Tensor,
    s_rel: torch.Tensor,
    s_channel: torch.Tensor,
    level_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per group, the stored-dtype neighbour of the ALS scale whose decoded int8 levels fit
    the group best (through-grid squared error). Returns the chosen scales in ``s_rel``'s
    dtype and the matching level assignment, so the caller need not assign again."""
    target = grouped_weight / s_channel.view(-1, 1, 1)

    def score(cand):
        levels = _grid_levels(level_table, cand)
        idx = _assign_grid(grouped_weight, levels, s_channel, target=target)
        err = torch.gather(levels, 2, idx.long()).sub_(target).pow_(2).sum(-1)
        return cand, err, idx

    if s_rel.dtype == torch.float8_e4m3fn:
        # exact representable neighbours: +-1 on the e4m3 bit pattern (s_rel > 0)
        raw = s_rel.view(torch.uint8)
        candidates = [
            s_rel.float(),
            (raw - 1).clamp_(min=1).view(torch.float8_e4m3fn).float(),
            (raw + 1).clamp_(max=_E4M3_MAX_FINITE_BITS).view(torch.float8_e4m3fn).float(),
        ]
    else:
        base = s_rel.float()
        candidates = [base, base * (1 - _SCALE_SEARCH_FP32_STEP), base * (1 + _SCALE_SEARCH_FP32_STEP)]
    best_scale, best_err, best_idx = score(candidates[0])
    for cand in candidates[1:]:
        cand, err, idx = score(cand)
        better = err < best_err
        torch.where(better, err, best_err, out=best_err)
        torch.where(better, cand, best_scale, out=best_scale)
        torch.where(better.unsqueeze(-1), idx, best_idx, out=best_idx)
    return best_scale.to(s_rel.dtype).contiguous(), best_idx


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
    levels: int = 16,
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
    return _codebook_for(normalized, levels)


def _quantize_w4a8_chunked(
    weight: torch.Tensor,
    rotate_fn,
    group_size: int,
    convrot_groupsize: int,
    symmetric: bool,
    scale_dtype: torch.dtype,
    codebook: bool,
    codebook_override: torch.Tensor | None = None,
    stochastic_rounding: int = 0,
    bits: int = 4,
    scale_search: bool = True,
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
    is otherwise independent. ``stochastic_rounding`` > 0 seeds SR of the final assignment
    (the LoRA-merge path); the seed is offset by the row start so chunks decorrelate yet stay
    deterministic for a fixed chunk size.
    """
    n, k = weight.shape
    block = max(1, _QUANT_ROW_ELEM_BUDGET // max(k, 1))
    override = None
    if bits in _UNIFORM_LEVELS:
        codebook = False  # uniform only; a 5/6-bit table is never fit or stored
    if bits == 3:
        codebook = True  # always the 8-entry table (frozen unless heavy-tailed, like 4-bit)
    if symmetric and codebook:
        override = (
            codebook_override.to(device=weight.device, dtype=torch.float32)
            if codebook_override is not None
            else _decide_codebook(weight, rotate_fn, group_size, convrot_groupsize, _CODEBOOK_LEVELS[bits])
        )

    if n <= block:
        return _quantize_rotated_w4a8_int8_weight(
            rotate_fn(weight.contiguous(), convrot_groupsize),
            group_size=group_size,
            symmetric=symmetric,
            scale_dtype=scale_dtype,
            codebook=codebook,
            codebook_override=override,
            stochastic_rounding=stochastic_rounding,
            bits=bits,
            scale_search=scale_search,
        )

    packed_parts, srel_parts, sch_parts, corr_parts = [], [], [], []
    codebook_tensor = None
    for r0 in range(0, n, block):
        rot = rotate_fn(weight[r0 : r0 + block].contiguous(), convrot_groupsize)
        chunk_sr = stochastic_rounding + r0 if stochastic_rounding > 0 else 0
        packed, s_rel, s_channel, correction, codebook_tensor = (
            _quantize_rotated_w4a8_int8_weight(
                rot,
                group_size=group_size,
                symmetric=symmetric,
                scale_dtype=scale_dtype,
                codebook=codebook,
                codebook_override=override,
                stochastic_rounding=chunk_sr,
                bits=bits,
                scale_search=scale_search,
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
    stochastic_rounding: int = 0,
    bits: int = 4,
    scale_search: bool = True,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Rotate and prepare a floating weight for grouped W3A8 / W4A8 / W5A8 / W6A8 storage."""
    validate_w4a8_weight_shape(weight, group_size, convrot_groupsize, bits)
    return _quantize_w4a8_chunked(
        weight,
        rotate_int8_convrot_weight,
        group_size=group_size,
        convrot_groupsize=convrot_groupsize,
        symmetric=symmetric,
        scale_dtype=scale_dtype,
        codebook=codebook,
        codebook_override=codebook_tensor,
        stochastic_rounding=stochastic_rounding,
        bits=bits,
        scale_search=scale_search,
    )


def _dequant_int4_grouped_to_int8(
    qdata: torch.Tensor,
    s_rel: torch.Tensor,
    codebook: torch.Tensor | None,
    group_size: int,
) -> torch.Tensor:
    """Decode grouped INT3/INT4/INT5/INT6 storage to the INT8 grid consumed by the GEMM."""
    n, k, bits = _w4a8_geometry(qdata, s_rel, group_size)
    groups = k // group_size
    quantized = _unpack_codes(qdata, k, bits)
    if codebook is not None:
        values = codebook.to(device=qdata.device, dtype=torch.float32)[quantized]
    else:
        values = quantized.float() - float(1 << (bits - 1))  # uniform, zero point mid-range
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
) -> tuple[int, int, int]:
    """Validate packed W4A8/W6A8 tensors shared by dequantize and linear; returns (n, k, bits)."""
    if qdata.dim() != 2 or qdata.dtype != torch.int8:
        raise ValueError("packed weight must be a 2D int8 tensor")
    if s_rel.dim() != 2:
        raise ValueError("s_rel must be 2D [N, K/group_size]")
    if group_size < 4 or (16 % group_size != 0 and group_size % 16 != 0):
        raise ValueError(f"group_size={group_size} must be >=4 and divide 16 or be a multiple of 16")
    n, k, bits = _w4a8_geometry(qdata, s_rel, group_size)  # k = groups * group_size
    if bits in _UNIFORM_LEVELS and (codebook is not None or correction is not None):
        raise ValueError(f"{bits}-bit storage is uniform and symmetric: no codebook or correction")
    if bits == 3 and (codebook is None or correction is not None):
        raise ValueError("3-bit storage needs its 8-entry codebook and has no correction")
    if k % 16 != 0 or k % convrot_groupsize != 0:
        raise ValueError(
            f"K={k} must be divisible by 16 and convrot_groupsize={convrot_groupsize}"
        )
    groups = k // group_size
    expected_scale_shape = (n, groups)
    if tuple(s_rel.shape) != expected_scale_shape:
        raise ValueError(f"s_rel must have shape {expected_scale_shape}, got {tuple(s_rel.shape)}")
    if tuple(s_channel.shape) != (n,):
        raise ValueError(f"s_channel must have shape {(n,)}, got {tuple(s_channel.shape)}")
    if correction is not None and tuple(correction.shape) != (groups, n):
        raise ValueError(f"correction must have shape {(groups, n)}, got {tuple(correction.shape)}")
    if codebook is not None and tuple(codebook.shape) != (_CODEBOOK_LEVELS.get(bits, 16),):
        raise ValueError(f"codebook must have shape ({_CODEBOOK_LEVELS.get(bits, 16)},), got {tuple(codebook.shape)}")
    return n, k, bits


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
    input_act: str | None = None,
    input_act_weight: torch.Tensor | None = None,
    input_act_eps: float = 0.0,
    residual: torch.Tensor | None = None,
    residual_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute x @ W.T + bias (then residual + residual_scale * that) using portable W4A8 operations."""
    if residual is not None:
        return apply_residual(
            w4a8_int8_linear(
                x, qdata, s_rel, s_channel, codebook, correction, bias, group_size,
                convrot_groupsize, out_dtype, input_act, input_act_weight,
                input_act_eps,
            ),
            residual,
            residual_scale,
        )
    x = apply_input_act(x, input_act, input_act_weight, input_act_eps)
    _n, k, _bits = validate_w4a8_operands(
        qdata,
        s_rel,
        s_channel,
        codebook,
        correction,
        group_size,
        convrot_groupsize,
    )
    if x.shape[-1] != k:
        raise ValueError(f"Input K={x.shape[-1]} does not match qdata K={k}")
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


def w4a8_packed_linear(
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
    input_act: str | None = None,
    input_act_weight: torch.Tensor | None = None,
    input_act_eps: float = 0.0,
    residual: torch.Tensor | None = None,
    residual_scale: torch.Tensor | None = None,
    decode_lut: torch.Tensor | None = None,  # the CUDA stream's table; eager decodes through codebook
) -> torch.Tensor:
    """Portable fallback for resident packed weights."""
    x = apply_input_act(x, input_act, input_act_weight, input_act_eps)
    qdata, s_rel = unpack_w4a8_mma_weight(qdata, s_channel.numel(), x.shape[-1])
    return w4a8_int8_linear(
        x, qdata, s_rel, s_channel, codebook, correction, bias, group_size,
        convrot_groupsize, out_dtype, residual=residual, residual_scale=residual_scale,
    )
