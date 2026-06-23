# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# AWQ W4A16 (4-bit weight, fp16/bf16 activation) layout for tensor dispatch.

"""AWQ W4A16 quantization layout for modulation-style linears.

Each quantized linear stores:
  qweight:        (N, K // 2)  int8       packed uint4 (kitchen-native row-major)
                                          bits 0..3 -> column 2j   (uint4 [0, 15])
                                          bits 4..7 -> column 2j+1 (uint4 [0, 15])
  scale=wscales:  (K // G, N)  bf16/fp16  per-group fp scales
  wzeros:         (K // G, N)  bf16/fp16  per-group fp zero points

Dequantization (per element):
    W[n, k] = (qweight[n, k] - 8) * wscales[k // G, n] + wzeros[k // G, n]

Targets the modulation linears (``img_mod.1`` / ``txt_mod.1``) in
Qwen-Image-Edit and similar topologies — small batch, called once per block,
where W4A16 GEMV stays fp16/bf16-accurate while the int4 weights cut both
checkpoint size and resident VRAM by ~4x vs the bf16-dequantized fallback.

Dispatch goes through ``ck.gemv_awq_w4a16``, which has an eager pure-PyTorch
implementation registered as a ``torch.library`` custom op; a CUDA backend
can be added later without changing this layout.
"""
from __future__ import annotations

import dataclasses
import logging
import os
from dataclasses import dataclass
from functools import lru_cache

import torch

import comfy_kitchen as ck

from .base import (
    BaseLayoutParams,
    QuantizedLayout,
    QuantizedTensor,
    dequantize_args,
    register_layout_op,
)

logger = logging.getLogger(__name__)

_DEFAULT_GROUP_SIZE = 64
_FALSE_ENV_VALUES = {"0", "false", "no", "off"}


def _env_enabled(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in _FALSE_ENV_VALUES


def _registry_backend_override() -> str | None:
    return getattr(ck.registry._thread_local, "backend_override", None)


@lru_cache(maxsize=1)
def _cuda_backend_module():
    try:
        from comfy_kitchen.backends import cuda as cuda_backend
    except Exception:
        return None
    if not getattr(cuda_backend, "_EXT_AVAILABLE", False):
        return None
    return cuda_backend


def _direct_cuda_backend(input_tensor: torch.Tensor, qdata: torch.Tensor):
    """Return CUDA backend when AWQ can bypass registry dispatch safely."""
    if not _env_enabled("COMFY_KITCHEN_AWQ_DIRECT_CUDA", default=True):
        return None

    override = _registry_backend_override()
    if override is not None and override != "cuda":
        return None

    if not input_tensor.is_cuda or not qdata.is_cuda:
        return None
    if input_tensor.device != qdata.device:
        return None

    return _cuda_backend_module()


class TensorCoreAWQW4A16Layout(QuantizedLayout):
    """AWQ W4A16 weight quantization with per-group fp scales + zeros.

    Note:
        Offline-quantized only — ``quantize()`` raises NotImplementedError.
        Use the upstream AWQ / DeepCompressor calibration pipeline to produce
        the pre-quantized tensors. ``from_state_dict`` is the loading path.
    """

    # Eager dispatch: any device that runs torch + bf16/fp16 works.
    # CUDA fast path TBD; will gate min_compute_capability there.
    MIN_SM_VERSION = None

    # Activation stays fp; the kernel does not pre-quantize x.
    QUANTIZES_INPUT = False

    @dataclass(frozen=True)
    class Params(BaseLayoutParams):
        """AWQ W4A16 parameters.

        Inherits ``scale`` (= wscales), ``orig_dtype``, ``orig_shape`` from
        BaseLayoutParams. Adds ``zeros`` (per-group fp zero points) and
        ``group_size`` (the K-dim quantization group size — typically 64,
        matching the wscales / wzeros leading shape).
        """
        zeros: torch.Tensor
        group_size: int = _DEFAULT_GROUP_SIZE
        transposed: bool = False

        def _tensor_fields(self) -> list[str]:
            return ["scale", "zeros"]

        def _validate_tensor_fields(self):
            # wscales / wzeros are per-group and stay in compute dtype — no coerce.
            return

    @classmethod
    def quantize(cls, tensor: torch.Tensor, **kwargs):
        raise NotImplementedError(
            "AWQ W4A16 requires offline calibration. Load pre-quantized "
            "tensors via `from_state_dict` instead."
        )

    @classmethod
    def dequantize(cls, qdata: torch.Tensor, params: Params) -> torch.Tensor:
        """Reconstruct the bf16/fp16 weight matrix W of shape ``(N, K)``.

        Used by the fallback path when the dispatch handler can't run a
        quantized matmul (e.g. RHS is not transposed, or operand types don't
        match). Stays in fp accumulation on the eager backend.
        """
        n, k_half = qdata.shape
        k = k_half * 2
        g = params.group_size
        # Unpack two uint4 nibbles per byte: bits 0..3 -> col 2j, bits 4..7 -> col 2j+1
        q_i32 = qdata.to(torch.int32)
        lo = (q_i32 & 0x0F).to(torch.int32)
        hi = ((q_i32 >> 4) & 0x0F).to(torch.int32)
        w_uint = torch.empty(n, k, dtype=torch.int32, device=qdata.device)
        w_uint[:, 0::2] = lo
        w_uint[:, 1::2] = hi
        scales = params.scale.t().unsqueeze(-1)   # (n, k/g, 1)
        zeros  = params.zeros.t().unsqueeze(-1)
        groups = w_uint.view(n, k // g, g).to(params.orig_dtype)
        w = ((groups - 8.0) * scales + zeros).view(n, k)
        return w

    @classmethod
    def get_plain_tensors(cls, qtensor: QuantizedTensor):
        p = qtensor._params
        return qtensor._qdata, p.scale, p.zeros

    @classmethod
    def state_dict_tensors(cls, qdata: torch.Tensor, params: Params) -> dict[str, torch.Tensor]:
        return {
            "": qdata,
            "_scale": params.scale,
            "_zeros": params.zeros,
        }


# ==================== Linear Dispatch ====================

def _awq_forward(
    input_tensor: torch.Tensor,
    weight_qt: QuantizedTensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Compute y = x @ W^T + bias via AWQ W4A16 GEMV."""
    qdata, wscales, wzeros = TensorCoreAWQW4A16Layout.get_plain_tensors(weight_qt)
    group_size = int(getattr(weight_qt._params, "group_size", _DEFAULT_GROUP_SIZE))

    cuda_backend = _direct_cuda_backend(input_tensor, qdata)
    if cuda_backend is not None:
        return cuda_backend.gemv_awq_w4a16(
            input_tensor, qdata, wscales, wzeros,
            bias=bias, group_size=group_size,
        )

    return ck.gemv_awq_w4a16(
        input_tensor, qdata, wscales, wzeros,
        bias=bias, group_size=group_size,
    )


@register_layout_op(torch.ops.aten.t.default, TensorCoreAWQW4A16Layout)
def _handle_awq_t(qt, args, kwargs):
    """Zero-copy logical transpose — flip the ``transposed`` flag."""
    input_tensor = args[0]
    if not isinstance(input_tensor, QuantizedTensor):
        return torch.ops.aten.t.default(*args, **kwargs)

    old = input_tensor._params
    new_params = dataclasses.replace(
        old,
        orig_shape=(old.orig_shape[1], old.orig_shape[0]),
        transposed=not old.transposed,
    )
    return QuantizedTensor(input_tensor._qdata, "TensorCoreAWQW4A16Layout", new_params)


def _resolve_awq_rhs(rhs: QuantizedTensor) -> QuantizedTensor:
    """Return rhs unchanged if it is logically transposed (represents W^T)."""
    if not rhs._params.transposed:
        raise RuntimeError(
            "AWQ W4A16 GEMM expects the RHS to be W.T (stored W). "
            "Use F.linear(x, W) or mm(x, W.t())."
        )
    return rhs


@register_layout_op(torch.ops.aten.linear.default, TensorCoreAWQW4A16Layout)
def _handle_awq_linear(qt, args, kwargs):
    """Direct F.linear(input, W, bias) → AWQ GEMV."""
    input_tensor, weight = args[0], args[1]
    bias = args[2] if len(args) > 2 else None

    if not isinstance(weight, QuantizedTensor):
        return torch.nn.functional.linear(*dequantize_args((input_tensor, weight, bias)))
    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()
    if weight._params.transposed:
        return torch.nn.functional.linear(input_tensor, weight.dequantize(), bias)
    return _awq_forward(input_tensor, weight, bias)


@register_layout_op(torch.ops.aten.mm.default, TensorCoreAWQW4A16Layout)
def _handle_awq_mm(qt, args, kwargs):
    """``mm(x, W.t())`` — F.linear's decomposition for tensor subclass weights."""
    a, b = args[0], args[1]
    if not isinstance(b, QuantizedTensor):
        return torch.mm(*dequantize_args((a, b)))
    if isinstance(a, QuantizedTensor):
        a = a.dequantize()
    b = _resolve_awq_rhs(b)
    return _awq_forward(a, b, bias=None)


@register_layout_op(torch.ops.aten.addmm.default, TensorCoreAWQW4A16Layout)
def _handle_awq_addmm(qt, args, kwargs):
    """``addmm(bias, x, W.t())``."""
    bias, a, b = args[0], args[1], args[2]
    if not isinstance(b, QuantizedTensor):
        return torch.addmm(*dequantize_args((bias, a, b)))
    if isinstance(a, QuantizedTensor):
        a = a.dequantize()
    b = _resolve_awq_rhs(b)
    return _awq_forward(a, b, bias=bias)
