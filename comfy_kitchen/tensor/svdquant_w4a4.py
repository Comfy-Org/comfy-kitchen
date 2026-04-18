# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# SVDQuant W4A4 (int4 weight, int4 activation, SVD low-rank correction) layout.

"""SVDQuant W4A4 quantization layout for tensor cores.

Each quantized linear stores:
  qweight:       (N, K // 2)  int8        packed W4 residual
  scale=wscales: (K // 64, N) bf16/fp16   per-group weight scales
  proj_down:     (K, R)       bf16/fp16   SVD down projection (V^T)
  proj_up:       (N, R)       bf16/fp16   SVD up projection (U)
  smooth_factor: (K,)         bf16/fp16   input-side smoothing

LoRA-style proj_down / proj_up recover the outlier-heavy singular directions
that pure 4-bit quantization cannot represent; the dispatched kernel fuses
activation quantization + low-rank correction + int4 matmul into a single call.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

import comfy_kitchen as ck

from .base import BaseLayoutParams, QuantizedLayout, dequantize_args, register_layout_op

if TYPE_CHECKING:
    from .base import QuantizedTensor

logger = logging.getLogger(__name__)

_INT4_GROUP_SIZE = 64
_GELU_UNSIGNED_SHIFT = 0.171875


class TensorCoreSVDQuantW4A4Layout(QuantizedLayout):
    """SVDQuant W4A4 weight quantization with low-rank correction.

    Note:
        Offline-quantized only — `quantize()` raises NotImplementedError because
        SVDQuant factorization requires calibration (smooth_factor, proj_down,
        proj_up) that must be computed from activation statistics. Use the
        DeepCompressor pipeline to produce the pre-quantized tensors.
    """

    # SVDQuant ships kernels for SM >= 8.0 (Ampere); kitchen dispatches to the
    # nunchaku backend which enforces this via its own constraint system.
    MIN_SM_VERSION = (8, 0)

    # Activation quantization is fused inside the kernel — do not pre-wrap
    # the input with QuantizedTensor.from_float(). Consumers (e.g. ComfyUI's
    # mixed_precision_ops.Linear) should read this flag before attempting to
    # quantize an incoming float activation.
    QUANTIZES_INPUT = False

    @dataclass(frozen=True)
    class Params(BaseLayoutParams):
        """SVDQuant W4A4 parameters.

        Inherits `scale` (= wscales), `orig_dtype`, `orig_shape` from
        BaseLayoutParams. Adds the three tensors that parameterize the
        low-rank correction and input smoothing, plus a logical-transpose flag
        used by the aten.t / aten.mm dispatch path.
        """
        proj_down: torch.Tensor
        proj_up: torch.Tensor
        smooth_factor: torch.Tensor
        act_unsigned: bool = False
        transposed: bool = False

        def _tensor_fields(self) -> list[str]:
            return ["scale", "proj_down", "proj_up", "smooth_factor"]

        def _validate_tensor_fields(self):
            # Unlike per-tensor scale layouts, wscales is per-group and stays
            # in the model compute dtype (bf16 / fp16) — do not coerce.
            return

    @classmethod
    def quantize(
        cls,
        tensor: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, Params]:
        raise NotImplementedError(
            "SVDQuant W4A4 requires offline calibration (DeepCompressor). "
            "Load pre-quantized tensors via `from_state_dict` instead."
        )

    @classmethod
    def dequantize(cls, qdata: torch.Tensor, params: Params) -> torch.Tensor:
        """Reconstruct the effective weight W_eff such that plain ``x @ W_eff.T + bias``
        reproduces the SVDQuant kernel output to bf16 precision.

        Uses the kitchen kernel itself with an identity input — bit-exact with the
        kernel's tile-interleaved packed layout without requiring us to replicate
        that layout in Python.
        """
        out_features, _ = qdata.shape
        in_features = params.orig_shape[1]
        device = qdata.device
        dtype = params.orig_dtype

        eye = torch.eye(in_features, dtype=dtype, device=device)
        q_x, ascales, lora_act = ck.quantize_svdquant_w4a4(
            eye, smooth=params.smooth_factor, lora_down=params.proj_down,
        )
        w_eff = ck.scaled_mm_svdquant_w4a4(
            act=q_x, wgt=qdata, ascales=ascales, wscales=params.scale,
            lora_act_in=lora_act, lora_up=params.proj_up, bias=None,
        )[:in_features]
        return w_eff.t().contiguous()

    @classmethod
    def get_plain_tensors(
        cls, qtensor: QuantizedTensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        p = qtensor._params
        return qtensor._qdata, p.scale, p.smooth_factor, p.proj_down, p.proj_up

    @classmethod
    def state_dict_tensors(cls, qdata: torch.Tensor, params: Params) -> dict[str, torch.Tensor]:
        """Serialization mapping.

        Suffixes compose onto the owning Parameter's key (typically `*.weight`),
        producing for example `transformer_blocks.0.attn.to_q.weight`,
        `...weight_scale`, `...weight_proj_down`, etc.
        """
        return {
            "": qdata,
            "_scale": params.scale,
            "_proj_down": params.proj_down,
            "_proj_up": params.proj_up,
            "_smooth_factor": params.smooth_factor,
        }


# ==================== Linear Dispatch ====================

def _w4a4_forward(
    input_tensor: torch.Tensor,
    weight_qt: "QuantizedTensor",
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Compute y = x @ W^T + bias via nunchaku's fused int4 kernel."""
    qdata, wscales, smooth, proj_down, proj_up = TensorCoreSVDQuantW4A4Layout.get_plain_tensors(weight_qt)
    act_unsigned = bool(getattr(weight_qt._params, "act_unsigned", False))

    orig_shape = input_tensor.shape
    x2d = input_tensor.reshape(-1, orig_shape[-1])
    M = x2d.shape[0]

    q_x, ascales, lora_act = ck.quantize_svdquant_w4a4(
        x2d,
        smooth=smooth,
        lora_down=proj_down,
        act_unsigned=act_unsigned,
        shift_value=_GELU_UNSIGNED_SHIFT if act_unsigned else 0.0,
    )
    out = ck.scaled_mm_svdquant_w4a4(
        act=q_x, wgt=qdata, ascales=ascales, wscales=wscales,
        lora_act_in=lora_act, lora_up=proj_up, bias=bias,
        act_unsigned=act_unsigned,
    )
    out_features = qdata.shape[0]
    return out[:M].reshape(*orig_shape[:-1], out_features)


@register_layout_op(torch.ops.aten.t.default, TensorCoreSVDQuantW4A4Layout)
def _handle_w4a4_t(qt, args, kwargs):
    """Zero-copy logical transpose — flip the ``transposed`` flag.

    Lets ``F.linear(x, W)`` decompose into ``x @ W.t()`` without reordering any
    storage; ``mm`` / ``addmm`` handlers below unwind the flag.
    """
    import dataclasses

    from .base import QuantizedTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, QuantizedTensor):
        return torch.ops.aten.t.default(*args, **kwargs)

    old = input_tensor._params
    new_params = dataclasses.replace(
        old,
        orig_shape=(old.orig_shape[1], old.orig_shape[0]),
        transposed=not old.transposed,
    )
    return QuantizedTensor(input_tensor._qdata, "TensorCoreSVDQuantW4A4Layout", new_params)


def _resolve_svdquant_rhs(rhs: "QuantizedTensor") -> "QuantizedTensor":
    """Return rhs unchanged if it is logically transposed (represents W^T)."""
    if not rhs._params.transposed:
        raise RuntimeError(
            "SVDQuant W4A4 GEMM expects the RHS to be W.T (stored W). "
            "Use F.linear(x, W) or mm(x, W.t())."
        )
    return rhs


@register_layout_op(torch.ops.aten.linear.default, TensorCoreSVDQuantW4A4Layout)
def _handle_w4a4_linear(qt, args, kwargs):
    """Direct F.linear(input, W, bias) → kitchen kernel."""
    from .base import QuantizedTensor

    input_tensor, weight = args[0], args[1]
    bias = args[2] if len(args) > 2 else None

    if not isinstance(weight, QuantizedTensor):
        return torch.nn.functional.linear(*dequantize_args((input_tensor, weight, bias)))
    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()
    if weight._params.transposed:
        return torch.nn.functional.linear(input_tensor, weight.dequantize(), bias)
    return _w4a4_forward(input_tensor, weight, bias)


@register_layout_op(torch.ops.aten.mm.default, TensorCoreSVDQuantW4A4Layout)
def _handle_w4a4_mm(qt, args, kwargs):
    """Handle ``mm(x, W.t())`` — the decomposition F.linear takes when the weight
    is a non-default tensor subclass.
    """
    from .base import QuantizedTensor

    a, b = args[0], args[1]
    if not isinstance(b, QuantizedTensor):
        return torch.mm(*dequantize_args((a, b)))
    if isinstance(a, QuantizedTensor):
        a = a.dequantize()
    b = _resolve_svdquant_rhs(b)
    return _w4a4_forward(a, b, bias=None)


@register_layout_op(torch.ops.aten.addmm.default, TensorCoreSVDQuantW4A4Layout)
def _handle_w4a4_addmm(qt, args, kwargs):
    """Handle ``addmm(bias, x, W.t())``."""
    from .base import QuantizedTensor

    bias, a, b = args[0], args[1], args[2]
    if not isinstance(b, QuantizedTensor):
        return torch.addmm(*dequantize_args((bias, a, b)))
    if isinstance(a, QuantizedTensor):
        a = a.dequantize()
    b = _resolve_svdquant_rhs(b)
    return _w4a4_forward(a, b, bias=bias)
