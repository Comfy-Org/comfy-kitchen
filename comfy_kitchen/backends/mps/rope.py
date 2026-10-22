# SPDX-License-Identifier: Apache-2.0
"""RoPE and RMSNorm+RoPE on Apple GPUs, one Metal kernel for every variant.

Same signatures and results as the eager backend. The call rules at the bottom
decline whatever the kernel does not take, so the registry falls through to eager.
"""
from collections.abc import Mapping

import torch

from comfy_kitchen._rope_utils import check_rope_inplace
from comfy_kitchen.constraints import ValidationResult

from . import kernels


def _rope1(x, freqs_cis, *, split_half, scale=None, epsilon=1e-6, rot_dim=0, inplace=False):
    return kernels.rope(x, freqs_cis, split_half=split_half, weight=scale, eps=epsilon,
                        rot_dim=rot_dim, inplace=inplace)


# plain RoPE

def apply_rope1(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    return _rope1(x, freqs_cis, split_half=False)


def apply_rope(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    return apply_rope1(xq, freqs_cis), apply_rope1(xk, freqs_cis)


def apply_rope1_(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    check_rope_inplace(x, readonly=(freqs_cis,))
    return _rope1(x, freqs_cis, split_half=False, inplace=True)


def apply_rope_(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    check_rope_inplace(xq, xk, readonly=(freqs_cis,))
    _rope1(xq, freqs_cis, split_half=False, inplace=True)
    _rope1(xk, freqs_cis, split_half=False, inplace=True)
    return xq, xk


def apply_rope_split_half1(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    return _rope1(x, freqs_cis, split_half=True)


def apply_rope_split_half(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    return apply_rope_split_half1(xq, freqs_cis), apply_rope_split_half1(xk, freqs_cis)


def apply_rope_split_half1_(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    check_rope_inplace(x, readonly=(freqs_cis,))
    return _rope1(x, freqs_cis, split_half=True, inplace=True)


def apply_rope_split_half_(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    check_rope_inplace(xq, xk, readonly=(freqs_cis,))
    _rope1(xq, freqs_cis, split_half=True, inplace=True)
    _rope1(xk, freqs_cis, split_half=True, inplace=True)
    return xq, xk


# RMSNorm + RoPE

def rms_rope1(x: torch.Tensor, freqs_cis: torch.Tensor, scale: torch.Tensor, epsilon: float = 1e-6):
    return _rope1(x, freqs_cis, split_half=False, scale=scale, epsilon=epsilon)


def rms_rope(q, k, freqs_cis, q_scale, k_scale=None, epsilon: float = 1e-6):
    k_scale = q_scale if k_scale is None else k_scale
    return rms_rope1(q, freqs_cis, q_scale, epsilon), rms_rope1(k, freqs_cis, k_scale, epsilon)


def rms_rope1_(x: torch.Tensor, freqs_cis: torch.Tensor, scale: torch.Tensor, epsilon: float = 1e-6):
    check_rope_inplace(x, readonly=(freqs_cis, scale))
    return _rope1(x, freqs_cis, split_half=False, scale=scale, epsilon=epsilon, inplace=True)


def rms_rope_(q, k, freqs_cis, q_scale, k_scale=None, epsilon: float = 1e-6):
    k_scale = q_scale if k_scale is None else k_scale
    check_rope_inplace(q, k, readonly=(freqs_cis, q_scale, k_scale))
    _rope1(q, freqs_cis, split_half=False, scale=q_scale, epsilon=epsilon, inplace=True)
    _rope1(k, freqs_cis, split_half=False, scale=k_scale, epsilon=epsilon, inplace=True)
    return q, k


def rms_rope_split_half1(x, freqs_cis, scale, epsilon: float = 1e-6):
    return _rope1(x, freqs_cis, split_half=True, scale=scale, epsilon=epsilon)


def rms_rope_split_half(q, k, freqs_cis, q_scale, k_scale=None, epsilon: float = 1e-6, rot_dim: int = 0):
    k_scale = q_scale if k_scale is None else k_scale
    return (_rope1(q, freqs_cis, split_half=True, scale=q_scale, epsilon=epsilon, rot_dim=rot_dim),
            _rope1(k, freqs_cis, split_half=True, scale=k_scale, epsilon=epsilon, rot_dim=rot_dim))


def rms_rope_split_half1_(x, freqs_cis, scale, epsilon: float = 1e-6):
    check_rope_inplace(x, readonly=(freqs_cis, scale))
    return _rope1(x, freqs_cis, split_half=True, scale=scale, epsilon=epsilon, inplace=True)


def rms_rope_split_half_(q, k, freqs_cis, q_scale, k_scale=None, epsilon: float = 1e-6, rot_dim: int = 0):
    k_scale = q_scale if k_scale is None else k_scale
    check_rope_inplace(q, k, readonly=(freqs_cis, q_scale, k_scale))
    _rope1(q, freqs_cis, split_half=True, scale=q_scale, epsilon=epsilon, rot_dim=rot_dim, inplace=True)
    _rope1(k, freqs_cis, split_half=True, scale=k_scale, epsilon=epsilon, rot_dim=rot_dim, inplace=True)
    return q, k


# call rules

def _check(name, x, freqs, scale=None, rot_dim=0) -> ValidationResult:
    reason = kernels.rope_unsupported(x, freqs, scale, rot_dim)
    return ValidationResult.ok() if reason is None else ValidationResult.fail(name, reason)


def _rule(*tensors, scales=(), rot=False):
    """Call rule over the named tensor arguments (paired with their scale arguments)."""
    def rule(kwargs: Mapping[str, object]) -> ValidationResult:
        rot_dim = kwargs.get("rot_dim", 0) if rot else 0
        for i, name in enumerate(tensors):
            scale = None
            if scales:
                # only the second scale (k_scale) may be None, meaning "same as the first"
                scale = kwargs.get(scales[i])
                if scale is None and i > 0:
                    scale = kwargs.get(scales[0])
                if scale is None:
                    return ValidationResult.fail(scales[i], "scale is required")
            result = _check(name, kwargs.get(name), kwargs.get("freqs_cis"), scale, rot_dim)
            if not result.success:
                return result
        return ValidationResult.ok()
    return rule


validate_apply_rope1 = _rule("x")
validate_apply_rope = _rule("xq", "xk")
validate_rms_rope1 = _rule("x", scales=("scale",))
validate_rms_rope = _rule("q", "k", scales=("q_scale", "k_scale"))
validate_rms_rope_split_half = _rule("q", "k", scales=("q_scale", "k_scale"), rot=True)
