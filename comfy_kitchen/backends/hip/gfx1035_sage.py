# SPDX-FileCopyrightText: Copyright (c) 2024 SageAttention team.
# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""gfx1035 (RDNA2) SageAttention: INT8 QK^T + FP16 P·V on VALU.

RDNA2 has no matrix cores, so the main HIP backend's sage_attention/*.hip
sources (built on WMMA) do not build or run for gfx103x at all. This module
loads the ported RDNA2 kernels and exposes the same call shape as the external
SageAttention package, so callers can treat it as a drop-in backend.

The gate in :func:`is_available` is deliberately architecture-strict: the
extension is compiled for gfx103x only, and this module refuses to run on any
other target rather than silently producing wrong results.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import sys
from collections.abc import Sequence

import torch

from . import _visible_gfx_arches, _gfx_arch

# Supported head dimensions for the ported kernels.
_SUPPORTED_HEAD_DIMS = (64, 128)

_module = None
_import_error: Exception | None = None
_ARCH_LOADED: str | None = None


def _detect_gfx1035_arch(device: torch.device | int | None = None) -> str | None:
    """Return the gfx103x architecture name of ``device``, or None."""
    arch = _gfx_arch(device)
    return arch if isinstance(arch, str) and arch.startswith("gfx103") else None


def _load_module() -> None:
    """Import the compiled extension; caches success and the first failure."""
    global _module, _import_error, _ARCH_LOADED
    if _module is not None or _import_error is not None:
        return

    arch = _detect_gfx1035_arch()
    if arch is None:
        _import_error = RuntimeError(
            "gfx1035 sage attention: no RDNA2 (gfx103x) device is visible; "
            "this backend is compiled for RDNA2 only."
        )
        return

    directory = os.path.dirname(__file__)
    suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    path = None
    for name in sorted(os.listdir(directory)):
        if name.startswith("_qattn_gfx1035.") and name.endswith(suffixes):
            path = os.path.join(directory, name)
            break

    if path is None:
        _import_error = RuntimeError(
            "gfx1035 sage attention: extension not built (no _qattn_gfx1035 module "
            "in backends/hip). Rebuild with COMFY_HIP_ARCHS=gfx1035."
        )
        return

    try:
        spec = importlib.util.spec_from_file_location(
            "comfy_kitchen.backends.hip._qattn_gfx1035", path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"no extension loader for {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["comfy_kitchen.backends.hip._qattn_gfx1035"] = module
        spec.loader.exec_module(module)
    except Exception as e:  # a broken extension must not break import
        if sys.modules.get("comfy_kitchen.backends.hip._qattn_gfx1035") is not None:
            del sys.modules["comfy_kitchen.backends.hip._qattn_gfx1035"]
        _import_error = e
        return

    _module = module
    _ARCH_LOADED = arch


def is_available(device: torch.device | int | None = None) -> bool:
    """Whether the gfx1035 SageAttention path can run for this device."""
    if not torch.cuda.is_available() or not getattr(torch.version, "hip", None):
        return False
    if _detect_gfx1035_arch(device) is None:
        return False
    _load_module()
    return _module is not None


def import_error() -> str | None:
    """Why the extension is unavailable, for error messages."""
    _load_module()
    return None if _module is not None else str(_import_error)


def _ops():
    if _module is None:
        raise RuntimeError(f"gfx1035 sage attention unavailable: {_import_error}")
    return torch.ops.comfy_kitchen_qattn_gfx1035


def sageattn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: float | None = None,
    smooth_k: bool = False,
) -> torch.Tensor:
    """SDPA with INT8 Q·K^T and FP16 P·V on RDNA2.

    Shapes and layout semantics match SageAttention's ``sageattn``:
    ``tensor_layout="HND"`` takes ``[batch, heads, seq, head_dim]`` and
    ``"NHD"`` takes ``[batch, seq, heads, head_dim]``. Only head_dim 64 and 128
    are supported, matching the ported kernels.
    """
    if not is_available(q.device):
        raise RuntimeError(
            f"gfx1035 sage attention is unavailable: {import_error()}"
        )

    dtype = q.dtype
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"inputs must be fp16, bf16 or fp32, got {dtype}")
    if not (q.dtype == k.dtype == v.dtype):
        raise TypeError("q, k, v must share one dtype")
    if not (q.device == k.device == v.device):
        raise ValueError("q, k, v must be on the same device")
    if not q.is_cuda:
        raise ValueError("q, k, v must be on a CUDA/HIP device")

    head_dim = q.size(-1)
    if head_dim not in _SUPPORTED_HEAD_DIMS:
        raise ValueError(f"head_dim must be 64 or 128, got {head_dim}")
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.stride(-1) != 1:
            raise ValueError(f"{name} must be contiguous along head_dim")

    input_dtype = dtype
    if dtype == torch.float32:
        q, k, v = (t.half() for t in (q, k, v))
        dtype = torch.float16

    ops = _ops()
    layout_code = 1 if tensor_layout == "HND" else 0

    if sm_scale is None:
        sm_scale = head_dim ** -0.5

    # The quantization and attention kernels read q and k with 16-byte vector
    # loads whose offsets are computed from the strides. Non-contiguous inputs
    # (from permute/slice) can read past the intended row into the padding
    # row left by the slice, producing wrong quantized values. Make them
    # contiguous; the copy is one kernel and cheaper than a wrong result.
    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()
    if not v.is_contiguous():
        v = v.contiguous()

    # Write-back uses 16-byte vector stores, so q's strides must be multiples of
    # 8 halfs. Non-contiguous q (from permute/slice) would misalign them.
    for dim in (0, 1, 2):
        if q.stride(dim) % 8 != 0:
            raise ValueError(
                "native backend requires q strides that are multiples of 8 halfs "
                "(16-byte aligned write-back). Use contiguous tensors. "
                f"Got strides={q.stride()}."
            )

    o = torch.empty_like(q)

    kv_len = k.size(2) if tensor_layout == "HND" else k.size(1)
    kv_heads_n = k.size(1) if tensor_layout == "HND" else k.size(2)

    # gfx103x reads native [B,H,N,D] V directly; no transpose is needed, and the
    # v_scale argument is a placeholder the kernel does not read. The kernel
    # itself is FP16-only, so bf16 V is converted here (a single cast kernel,
    # which the direct-fp16 path avoids by folding the cast into its transpose).
    if tensor_layout == "NHD":
        v16 = v if v.dtype == torch.float16 else v.half()
        if v16.is_contiguous():
            b_, n_, h_, d_ = v16.shape
            v_for_attn = v16.as_strided(
                (b_, h_, n_, d_), (n_ * h_ * d_, d_, h_ * d_, 1)
            )
        else:
            v_for_attn = v16.permute(0, 2, 1, 3).contiguous()
    else:
        v_for_attn = v if v.dtype == torch.float16 else v.half()
    v_scale = torch.empty(
        q.size(0), kv_heads_n, (kv_len + 31) // 32,
        device=q.device, dtype=torch.float32,
    )

    if smooth_k:
        k_mean = ops.mean_seq(k, layout_code)
    else:
        k_mean = torch.empty(0, device=q.device, dtype=q.dtype)

    # In-kernel Q quantization avoids a Q round-trip for short KV, where the
    # prepass dominates. Only safe for contiguous, non-causal q.
    skip_inq = bool(
        head_dim in _SUPPORTED_HEAD_DIMS
        and not is_causal
        and kv_len <= 1024
        and q.is_contiguous()
    )
    q_fp = q
    if skip_inq and tensor_layout == "NHD":
        b_, s_, h_, d_ = q.shape
        q_fp = q.as_strided((b_, h_, s_, d_), (s_ * h_ * d_, d_, h_ * d_, 1))

    q_int8, q_scale, k_int8, k_scale = ops.quant_qk_int8(
        q, k, k_mean, layout_code, float(sm_scale), int(skip_inq)
    )
    ops.qk_int8_sv_bf16_attn_t(
        q_int8, k_int8, v_for_attn, o,
        q_scale, k_scale, v_scale,
        layout_code, int(is_causal), float(sm_scale),
        q_fp if skip_inq else torch.empty(0, device=q.device, dtype=q.dtype),
    )

    if input_dtype == torch.float32:
        o = o.to(torch.float32)
    return o
