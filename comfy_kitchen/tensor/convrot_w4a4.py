# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ConvRot W4A4 quantization layout.

This is the plain ConvRot path from the paper adapted to kitchen's existing
int4 tensor-core GEMM: offline regular-Hadamard weight rotation, online
regular-Hadamard activation rotation, symmetric int4 quantization, and int4
MMA with per-group scales.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch

from comfy_kitchen.backends.eager.svdquant import _INT4_GROUP_SIZE
from comfy_kitchen.registry import registry

from .base import (
    BaseLayoutParams,
    QuantizedLayout,
    QuantizedTensor,
    dequantize_args,
    register_layout_op,
)


def quantize_convrot_w4a4_weight(
    weight: torch.Tensor,
    convrot_groupsize: int = 256,
    quant_group_size: int = _INT4_GROUP_SIZE,
    stochastic_rounding: int | None = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate a weight matrix with ConvRot and pack it as signed W4."""
    impl = registry.get_implementation(
        "quantize_convrot_w4a4_weight",
        kwargs={
            "weight": weight,
            "convrot_groupsize": convrot_groupsize,
            "quant_group_size": quant_group_size,
            "stochastic_rounding": stochastic_rounding,
        },
    )
    return impl(
        weight,
        convrot_groupsize=convrot_groupsize,
        quant_group_size=quant_group_size,
        stochastic_rounding=stochastic_rounding,
    )


def dequantize_convrot_w4a4_weight(
    qdata: torch.Tensor,
    scales: torch.Tensor,
    convrot_groupsize: int = 256,
    quant_group_size: int = _INT4_GROUP_SIZE,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize packed ConvRot W4A4 weights and rotate back to original basis."""
    impl = registry.get_implementation(
        "dequantize_convrot_w4a4_weight",
        kwargs={
            "qdata": qdata,
            "scales": scales,
            "convrot_groupsize": convrot_groupsize,
            "quant_group_size": quant_group_size,
            "output_dtype": output_dtype,
        },
    )
    return impl(
        qdata,
        scales,
        convrot_groupsize=convrot_groupsize,
        quant_group_size=quant_group_size,
        output_dtype=output_dtype,
    )


def convrot_w4a4_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    wscales: torch.Tensor,
    bias: torch.Tensor | None = None,
    convrot_groupsize: int = 256,
    quant_group_size: int = _INT4_GROUP_SIZE,
    linear_dtype: str = "int4",
) -> torch.Tensor:
    """Compute ``x @ W.T + bias`` using ConvRot W4A4 int4 MMA."""
    if linear_dtype not in {"int4", "int8"}:
        raise ValueError(f"ConvRot W4A4 linear_dtype must be 'int4' or 'int8', got {linear_dtype!r}")
    impl = registry.get_implementation(
        "convrot_w4a4_linear",
        kwargs={
            "x": x,
            "qweight": qweight,
            "wscales": wscales,
            "bias": bias,
            "convrot_groupsize": convrot_groupsize,
            "quant_group_size": quant_group_size,
            "linear_dtype": linear_dtype,
        },
    )
    return impl(
        x,
        qweight,
        wscales,
        bias=bias,
        convrot_groupsize=convrot_groupsize,
        quant_group_size=quant_group_size,
        linear_dtype=linear_dtype,
    )


class TensorCoreConvRotW4A4Layout(QuantizedLayout):
    """ConvRot W4A4 weight layout using kitchen's int4 tensor-core GEMM."""

    MIN_SM_VERSION = (7, 5)
    QUANTIZES_INPUT = False

    @dataclass(frozen=True)
    class Params(BaseLayoutParams):
        convrot_groupsize: int = 256
        quant_group_size: int = _INT4_GROUP_SIZE
        linear_dtype: str = "int4"
        transposed: bool = False

        def __post_init__(self):
            super().__post_init__()
            if self.linear_dtype not in {"int4", "int8"}:
                raise ValueError(f"ConvRot W4A4 linear_dtype must be 'int4' or 'int8', got {self.linear_dtype!r}")

        def _tensor_fields(self) -> list[str]:
            return ["scale"]

        def _validate_tensor_fields(self):
            return

    @classmethod
    def quantize(
        cls,
        tensor: torch.Tensor,
        convrot_groupsize: int = 256,
        quant_group_size: int = _INT4_GROUP_SIZE,
        stochastic_rounding: int | None = 0,
        linear_dtype: str = "int4",
        **kwargs,
    ) -> tuple[torch.Tensor, Params]:
        if linear_dtype not in {"int4", "int8"}:
            raise ValueError(f"ConvRot W4A4 linear_dtype must be 'int4' or 'int8', got {linear_dtype!r}")
        qdata, scales = quantize_convrot_w4a4_weight(
            tensor,
            convrot_groupsize,
            quant_group_size,
            stochastic_rounding=stochastic_rounding,
        )
        params = cls.Params(
            scale=scales,
            orig_dtype=tensor.dtype,
            orig_shape=tuple(tensor.shape),
            convrot_groupsize=convrot_groupsize,
            quant_group_size=quant_group_size,
            linear_dtype=linear_dtype,
        )
        return qdata, params

    @classmethod
    def dequantize(cls, qdata: torch.Tensor, params: Params) -> torch.Tensor:
        return dequantize_convrot_w4a4_weight(
            qdata,
            params.scale,
            params.convrot_groupsize,
            params.quant_group_size,
            params.orig_dtype,
        )

    @classmethod
    def get_plain_tensors(cls, qtensor: QuantizedTensor) -> tuple[torch.Tensor, torch.Tensor]:
        return qtensor._qdata, qtensor._params.scale

    @classmethod
    def state_dict_tensors(cls, qdata: torch.Tensor, params: Params) -> dict[str, torch.Tensor]:
        return {"": qdata, "_scale": params.scale}

    @classmethod
    def requantize_kwargs(cls, qtensor: QuantizedTensor) -> dict[str, object]:
        params = qtensor._params
        return {
            "convrot_groupsize": params.convrot_groupsize,
            "quant_group_size": params.quant_group_size,
            "linear_dtype": params.linear_dtype,
        }


@register_layout_op(torch.ops.aten.t.default, TensorCoreConvRotW4A4Layout)
def _handle_convrot_w4a4_t(qt, args, kwargs):
    input_tensor = args[0]
    if not isinstance(input_tensor, QuantizedTensor):
        return torch.ops.aten.t.default(*args, **kwargs)
    old = input_tensor._params
    new_params = dataclasses.replace(
        old,
        orig_shape=(old.orig_shape[1], old.orig_shape[0]),
        transposed=not old.transposed,
    )
    return QuantizedTensor(input_tensor._qdata, "TensorCoreConvRotW4A4Layout", new_params)


def _resolve_convrot_w4a4_rhs(rhs: QuantizedTensor) -> QuantizedTensor:
    if not rhs._params.transposed:
        raise RuntimeError("ConvRot W4A4 GEMM expects RHS W.T. Use F.linear(x, W) or mm(x, W.t()).")
    return rhs


def _convrot_w4a4_forward(input_tensor: torch.Tensor, weight: QuantizedTensor, bias: torch.Tensor | None):
    qweight, wscales = TensorCoreConvRotW4A4Layout.get_plain_tensors(weight)
    params = weight._params
    return convrot_w4a4_linear(
        input_tensor,
        qweight,
        wscales,
        bias=bias,
        convrot_groupsize=params.convrot_groupsize,
        quant_group_size=params.quant_group_size,
        linear_dtype=params.linear_dtype,
    )


@register_layout_op(torch.ops.aten.linear.default, TensorCoreConvRotW4A4Layout)
def _handle_convrot_w4a4_linear(qt, args, kwargs):
    input_tensor, weight = args[0], args[1]
    bias = args[2] if len(args) > 2 else None
    if not isinstance(weight, QuantizedTensor):
        return torch.nn.functional.linear(*dequantize_args((input_tensor, weight, bias)))
    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()
    if weight._params.transposed:
        return torch.nn.functional.linear(input_tensor, weight.dequantize(), bias)
    return _convrot_w4a4_forward(input_tensor, weight, bias)


@register_layout_op(torch.ops.aten.mm.default, TensorCoreConvRotW4A4Layout)
def _handle_convrot_w4a4_mm(qt, args, kwargs):
    a, b = args[0], args[1]
    if not isinstance(b, QuantizedTensor):
        return torch.mm(*dequantize_args((a, b)))
    if isinstance(a, QuantizedTensor):
        a = a.dequantize()
    b = _resolve_convrot_w4a4_rhs(b)
    return _convrot_w4a4_forward(a, b, bias=None)


@register_layout_op(torch.ops.aten.addmm.default, TensorCoreConvRotW4A4Layout)
def _handle_convrot_w4a4_addmm(qt, args, kwargs):
    bias, a, b = args[0], args[1], args[2]
    if not isinstance(b, QuantizedTensor):
        return torch.addmm(*dequantize_args((bias, a, b)))
    if isinstance(a, QuantizedTensor):
        a = a.dequantize()
    b = _resolve_convrot_w4a4_rhs(b)
    return _convrot_w4a4_forward(a, b, bias=bias)
