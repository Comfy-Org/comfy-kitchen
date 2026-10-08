# SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RoPE and RMS-RoPE implementations for Huawei Ascend NPUs."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch_npu

from comfy_kitchen._rope_utils import (
    check_rope_inplace,
    detect_rms_rope_bnhd,
    trim_rope_freqs,
)
from comfy_kitchen.constraints import ValidationResult


def _rotary_coefficients(
    freqs_cis: torch.Tensor, *, split_half: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert Comfy Kitchen's 2x2 matrices to RotaryMul coefficients."""
    m00 = freqs_cis[..., 0, 0]
    m01 = freqs_cis[..., 0, 1]
    m10 = freqs_cis[..., 1, 0]
    m11 = freqs_cis[..., 1, 1]
    if split_half:
        return torch.cat((m00, m11), dim=-1), torch.cat((-m01, m10), dim=-1)
    return (
        torch.stack((m00, m11), dim=-1).flatten(-2),
        torch.stack((-m01, m10), dim=-1).flatten(-2),
    )


def _apply_rope1(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    *,
    split_half: bool,
) -> torch.Tensor:
    freqs_cis = trim_rope_freqs(x, freqs_cis)
    r1, r2 = _rotary_coefficients(freqs_cis, split_half=split_half)
    rotary_mode = "half" if split_half else "interleave"
    output = torch_npu.npu_rotary_mul(x.to(freqs_cis.dtype), r1, r2, rotary_mode)
    return output.to(x.dtype)


def apply_rope1(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    return _apply_rope1(x, freqs_cis, split_half=False)


def apply_rope1_(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    check_rope_inplace(x, readonly=(freqs_cis,))
    x.copy_(apply_rope1(x, freqs_cis))
    return x


def apply_rope(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return apply_rope1(xq, freqs_cis), apply_rope1(xk, freqs_cis)


def apply_rope_(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    check_rope_inplace(xq, xk, readonly=(freqs_cis,))
    q_out, k_out = apply_rope(xq, xk, freqs_cis)
    xq.copy_(q_out)
    xk.copy_(k_out)
    return xq, xk


def apply_rope_split_half1(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    return _apply_rope1(x, freqs_cis, split_half=True)


def apply_rope_split_half1_(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    check_rope_inplace(x, readonly=(freqs_cis,))
    x.copy_(apply_rope_split_half1(x, freqs_cis))
    return x


def apply_rope_split_half(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        apply_rope_split_half1(xq, freqs_cis),
        apply_rope_split_half1(xk, freqs_cis),
    )


def apply_rope_split_half_(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    check_rope_inplace(xq, xk, readonly=(freqs_cis,))
    q_out, k_out = apply_rope_split_half(xq, xk, freqs_cis)
    xq.copy_(q_out)
    xk.copy_(k_out)
    return xq, xk


def _rms_rope1(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float,
    *,
    split_half: bool,
    rot_dim: int = 0,
) -> torch.Tensor:
    normalized = torch_npu.npu_rms_norm(x, scale, epsilon)[0]
    if rot_dim and rot_dim != x.shape[-1]:
        rotated = _apply_rope1(normalized[..., :rot_dim], freqs_cis, split_half=split_half)
        return torch.cat((rotated, normalized[..., rot_dim:]), dim=-1)
    return _apply_rope1(normalized, freqs_cis, split_half=split_half)


def rms_rope1(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    return _rms_rope1(x, freqs_cis, scale, epsilon, split_half=False)


def rms_rope1_(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    check_rope_inplace(x, readonly=(freqs_cis, scale))
    x.copy_(rms_rope1(x, freqs_cis, scale, epsilon))
    return x


def rms_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor | None = None,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    k_scale = q_scale if k_scale is None else k_scale
    return (
        rms_rope1(q, freqs_cis, q_scale, epsilon),
        rms_rope1(k, freqs_cis, k_scale, epsilon),
    )


def rms_rope_(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor | None = None,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    k_scale = q_scale if k_scale is None else k_scale
    check_rope_inplace(q, k, readonly=(freqs_cis, q_scale, k_scale))
    q_out, k_out = rms_rope(q, k, freqs_cis, q_scale, k_scale, epsilon)
    q.copy_(q_out)
    k.copy_(k_out)
    return q, k


def rms_rope_split_half1(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    return _rms_rope1(x, freqs_cis, scale, epsilon, split_half=True)


def rms_rope_split_half1_(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    check_rope_inplace(x, readonly=(freqs_cis, scale))
    x.copy_(rms_rope_split_half1(x, freqs_cis, scale, epsilon))
    return x


def rms_rope_split_half(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor | None = None,
    epsilon: float = 1e-6,
    rot_dim: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    k_scale = q_scale if k_scale is None else k_scale
    return (
        _rms_rope1(
            q,
            freqs_cis,
            q_scale,
            epsilon,
            split_half=True,
            rot_dim=rot_dim,
        ),
        _rms_rope1(
            k,
            freqs_cis,
            k_scale,
            epsilon,
            split_half=True,
            rot_dim=rot_dim,
        ),
    )


def rms_rope_split_half_(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor | None = None,
    epsilon: float = 1e-6,
    rot_dim: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    k_scale = q_scale if k_scale is None else k_scale
    check_rope_inplace(q, k, readonly=(freqs_cis, q_scale, k_scale))
    q_out, k_out = rms_rope_split_half(q, k, freqs_cis, q_scale, k_scale, epsilon, rot_dim)
    q.copy_(q_out)
    k.copy_(k_out)
    return q, k


def _validate_rope_tensor(
    x: object,
    freqs_cis: object,
    *,
    split_half: bool,
    rot_dim: int = 0,
) -> ValidationResult:
    if not isinstance(x, torch.Tensor) or not isinstance(freqs_cis, torch.Tensor):
        return ValidationResult.ok()
    if x.numel() == 0:
        return ValidationResult.fail("x", "empty tensors are not supported by npu_rotary_mul")
    if x.stride(-1) != 1:
        return ValidationResult.fail("x", "last dimension must be contiguous")

    freqs_cis = trim_rope_freqs(x, freqs_cis)
    bnhd = detect_rms_rope_bnhd(x, freqs_cis, rot_dim=rot_dim)
    if bnhd is None:
        return ValidationResult.fail(
            "freqs_cis", "shape must broadcast to a supported four-dimensional RoPE input"
        )

    batch = x.shape[0]
    heads = x.shape[2] if bnhd else x.shape[1]
    rotary_dim = rot_dim or x.shape[-1]
    if rotary_dim >= 896:
        return ValidationResult.fail("x", "rotary dimension must be less than 896")
    if batch >= 1000 or heads >= 1000:
        return ValidationResult.fail("x", "batch and head counts must be less than 1000")
    if not split_half:
        if batch * heads >= 1000:
            return ValidationResult.fail(
                "x", "interleaved RoPE requires batch times head count below 1000"
            )
        if freqs_cis.shape[0] != 1:
            return ValidationResult.fail(
                "freqs_cis", "interleaved npu_rotary_mul requires batch-broadcast frequencies"
            )
    return ValidationResult.ok()


def _validate_pair(
    kwargs: Mapping[str, object],
    *,
    q_name: str,
    k_name: str,
    split_half: bool,
    rot_dim: int = 0,
) -> ValidationResult:
    freqs_cis = kwargs.get("freqs_cis")
    for name in (q_name, k_name):
        result = _validate_rope_tensor(
            kwargs.get(name), freqs_cis, split_half=split_half, rot_dim=rot_dim
        )
        if not result.success:
            result.failed_param = name if result.failed_param == "x" else result.failed_param
            return result
    return ValidationResult.ok()


def validate_apply_rope1(kwargs: Mapping[str, object]) -> ValidationResult:
    return _validate_rope_tensor(kwargs.get("x"), kwargs.get("freqs_cis"), split_half=False)


def validate_apply_rope(kwargs: Mapping[str, object]) -> ValidationResult:
    return _validate_pair(kwargs, q_name="xq", k_name="xk", split_half=False)


def validate_apply_rope_split_half1(kwargs: Mapping[str, object]) -> ValidationResult:
    return _validate_rope_tensor(kwargs.get("x"), kwargs.get("freqs_cis"), split_half=True)


def validate_apply_rope_split_half(kwargs: Mapping[str, object]) -> ValidationResult:
    return _validate_pair(kwargs, q_name="xq", k_name="xk", split_half=True)


def _validate_scale(scale: object, x: object, name: str) -> ValidationResult:
    if not isinstance(scale, torch.Tensor) or not isinstance(x, torch.Tensor):
        return ValidationResult.ok()
    if scale.ndim != 1 or scale.numel() != x.shape[-1]:
        return ValidationResult.fail(name, "must be one-dimensional and match head dimension")
    return ValidationResult.ok()


def validate_rms_rope1(kwargs: Mapping[str, object]) -> ValidationResult:
    result = validate_apply_rope1(kwargs)
    if not result.success:
        return result
    return _validate_scale(kwargs.get("scale"), kwargs.get("x"), "scale")


def validate_rms_rope(kwargs: Mapping[str, object]) -> ValidationResult:
    result = _validate_pair(kwargs, q_name="q", k_name="k", split_half=False)
    if not result.success:
        return result
    result = _validate_scale(kwargs.get("q_scale"), kwargs.get("q"), "q_scale")
    if not result.success:
        return result
    k_scale = kwargs.get("k_scale")
    if k_scale is None:
        k_scale = kwargs.get("q_scale")
    return _validate_scale(k_scale, kwargs.get("k"), "k_scale")


def validate_rms_rope_split_half1(kwargs: Mapping[str, object]) -> ValidationResult:
    result = _validate_rope_tensor(kwargs.get("x"), kwargs.get("freqs_cis"), split_half=True)
    if not result.success:
        return result
    return _validate_scale(kwargs.get("scale"), kwargs.get("x"), "scale")


def validate_rms_rope_split_half(kwargs: Mapping[str, object]) -> ValidationResult:
    rot_dim = int(kwargs.get("rot_dim") or 0)
    result = _validate_pair(
        kwargs,
        q_name="q",
        k_name="k",
        split_half=True,
        rot_dim=rot_dim,
    )
    if not result.success:
        return result
    result = _validate_scale(kwargs.get("q_scale"), kwargs.get("q"), "q_scale")
    if not result.success:
        return result
    k_scale = kwargs.get("k_scale")
    if k_scale is None:
        k_scale = kwargs.get("q_scale")
    return _validate_scale(k_scale, kwargs.get("k"), "k_scale")
