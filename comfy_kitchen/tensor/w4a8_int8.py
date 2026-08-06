# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Grouped W4A8 weights executed through the shared INT8 linear operation.

Weights are ConvRot-rotated and quantized per ``group_size``. The default uses
a symmetric Lloyd-Max codebook and FP8 group scales. The tensor layout owns
only metadata and operator routing; eager, CUDA, and Triton backends own the
operation implementations.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch

from comfy_kitchen.registry import registry

from .base import (
    BaseLayoutParams,
    QuantizedLayout,
    QuantizedTensor,
    dequantize_args,
    register_layout_op,
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
    """Rotate and prepare a floating weight for grouped W4A8 storage.

    ``codebook_tensor`` reuses a previously decided table (e.g. on LoRA requantize) so the
    codebook decision -- kurtosis probe and any k-means -- is skipped.
    """
    if scale_dtype not in (torch.float32, torch.float8_e4m3fn):
        raise ValueError(f"scale_dtype must be float32 or float8_e4m3fn, got {scale_dtype}")
    kwargs = {
        "weight": weight,
        "group_size": group_size,
        "convrot_groupsize": convrot_groupsize,
        "symmetric": symmetric,
        "scale_dtype": scale_dtype,
        "codebook": codebook,
        "codebook_tensor": codebook_tensor,
    }
    impl = registry.get_implementation("quantize_w4a8_int8_weight", kwargs=kwargs)
    return impl(**kwargs)


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
    """Dequantize a packed W4A8 weight into its original basis."""
    kwargs = {
        "qdata": qdata,
        "s_rel": s_rel,
        "s_channel": s_channel,
        "codebook": codebook,
        "correction": correction,
        "group_size": group_size,
        "convrot_groupsize": convrot_groupsize,
        "output_dtype": output_dtype,
    }
    impl = registry.get_implementation("dequantize_w4a8_int8_weight", kwargs=kwargs)
    return impl(**kwargs)


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
    """Compute ``x @ W.T + bias`` with the selected W4A8 backend."""
    kwargs = {
        "x": x,
        "qdata": qdata,
        "s_rel": s_rel,
        "s_channel": s_channel,
        "codebook": codebook,
        "correction": correction,
        "bias": bias,
        "group_size": group_size,
        "convrot_groupsize": convrot_groupsize,
        "out_dtype": out_dtype,
    }
    impl = registry.get_implementation("w4a8_int8_linear", kwargs=kwargs)
    return impl(**kwargs)


class AsymW4A8Int8Layout(QuantizedLayout):
    """Grouped W4A8 weights run through the selected INT8 backend."""

    MIN_SM_VERSION = (8, 0)
    QUANTIZES_INPUT = False

    @dataclass(frozen=True)
    class Params(BaseLayoutParams):
        # scale holds s_rel: per-group relative scale [N, K // group_size].
        s_channel: torch.Tensor | None = None
        correction: torch.Tensor | None = None
        codebook: torch.Tensor | None = None
        group_size: int = 16
        convrot_groupsize: int = 256
        transposed: bool = False

        def _tensor_fields(self) -> list[str]:
            fields = ["scale", "s_channel"]
            if self.correction is not None:
                fields.append("correction")
            if self.codebook is not None:
                fields.append("codebook")
            return fields

        def _validate_tensor_fields(self):
            if len(self.orig_shape) != 2:
                raise ValueError(f"orig_shape must be 2D, got {self.orig_shape}")
            n, k = self.orig_shape
            if self.transposed:
                n, k = k, n
            if (
                self.group_size < 4
                or k % 16 != 0
                or k % self.group_size != 0
                or k % self.convrot_groupsize != 0
                or (16 % self.group_size != 0 and self.group_size % 16 != 0)
            ):
                raise ValueError(
                    f"K={k} must be divisible by 16, group_size={self.group_size}, and "
                    f"convrot_groupsize={self.convrot_groupsize}; group_size must be >=4 "
                    f"and divide 16 or be a multiple of 16"
                )
            groups = k // self.group_size
            expected_scale_shape = (n, groups)
            if tuple(self.scale.shape) != expected_scale_shape:
                raise ValueError(
                    f"scale must have shape {expected_scale_shape}, got {tuple(self.scale.shape)}"
                )
            if self.s_channel is None or tuple(self.s_channel.shape) != (n,):
                actual_shape = None if self.s_channel is None else tuple(self.s_channel.shape)
                raise ValueError(f"s_channel must have shape {(n,)}, got {actual_shape}")
            if self.correction is not None and tuple(self.correction.shape) != (groups, n):
                raise ValueError(
                    f"correction must have shape {(groups, n)}, got {tuple(self.correction.shape)}"
                )
            if self.codebook is not None and tuple(self.codebook.shape) != (16,):
                raise ValueError(
                    f"codebook must have shape (16,), got {tuple(self.codebook.shape)}"
                )

    @classmethod
    def quantize(
        cls,
        tensor: torch.Tensor,
        group_size: int = 16,
        convrot_groupsize: int = 256,
        symmetric: bool = True,
        scale_dtype: torch.dtype = torch.float8_e4m3fn,
        codebook: bool = True,
        codebook_tensor: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Params]:
        qdata, s_rel, s_channel, correction, codebook_tensor = quantize_w4a8_int8_weight(
            tensor,
            group_size=group_size,
            convrot_groupsize=convrot_groupsize,
            symmetric=symmetric,
            scale_dtype=scale_dtype,
            codebook=codebook,
            codebook_tensor=codebook_tensor,
        )
        params = cls.Params(
            scale=s_rel,
            s_channel=s_channel,
            correction=correction,
            codebook=codebook_tensor,
            orig_dtype=tensor.dtype,
            orig_shape=tuple(tensor.shape),
            group_size=group_size,
            convrot_groupsize=convrot_groupsize,
        )
        return qdata, params

    @classmethod
    def dequantize(cls, qdata: torch.Tensor, params: Params) -> torch.Tensor:
        return dequantize_w4a8_int8_weight(
            qdata,
            params.scale,
            params.s_channel,
            codebook=params.codebook,
            correction=params.correction,
            group_size=params.group_size,
            convrot_groupsize=params.convrot_groupsize,
            output_dtype=params.orig_dtype,
        )

    @classmethod
    def get_plain_tensors(
        cls,
        qtensor: QuantizedTensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        params = qtensor._params
        return (
            qtensor._qdata,
            params.scale,
            params.s_channel,
            params.correction,
            params.codebook,
        )

    @classmethod
    def state_dict_tensors(cls, qdata: torch.Tensor, params: Params) -> dict[str, torch.Tensor]:
        out = {
            "": qdata,
            "_s_rel": params.scale,
            "_s_channel": params.s_channel,
        }
        if params.correction is not None:
            out["_correction"] = params.correction
        if params.codebook is not None:
            out["_codebook"] = params.codebook
        return out

    @classmethod
    def requantize_kwargs(cls, qtensor: QuantizedTensor) -> dict[str, object]:
        params = qtensor._params
        return {
            "group_size": params.group_size,
            "convrot_groupsize": params.convrot_groupsize,
            "symmetric": params.correction is None,
            "codebook": params.codebook is not None,
            "codebook_tensor": params.codebook,
            "scale_dtype": params.scale.dtype,
        }


def _is_w4a8(weight: object) -> bool:
    return isinstance(weight, QuantizedTensor) and weight._layout_cls == "AsymW4A8Int8Layout"


def _resolve_w4a8_rhs(rhs: QuantizedTensor) -> QuantizedTensor:
    if not rhs._params.transposed:
        raise RuntimeError("W4A8 GEMM expects RHS W.T. Use F.linear(x, W) or mm(x, W.t()).")
    return rhs


def _w4a8_int8_forward(
    input_tensor: torch.Tensor,
    weight: QuantizedTensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    qdata, s_rel, s_channel, correction, codebook = AsymW4A8Int8Layout.get_plain_tensors(weight)
    params = weight._params
    return w4a8_int8_linear(
        input_tensor,
        qdata,
        s_rel,
        s_channel,
        codebook=codebook,
        correction=correction,
        bias=bias,
        group_size=params.group_size,
        convrot_groupsize=params.convrot_groupsize,
        out_dtype=out_dtype,
    )


@register_layout_op(torch.ops.aten.t.default, AsymW4A8Int8Layout)
def _handle_w4a8int8_t(qt, args, kwargs):
    input_tensor = args[0]
    if not isinstance(input_tensor, QuantizedTensor):
        return torch.ops.aten.t.default(*args, **kwargs)
    old = input_tensor._params
    new_params = dataclasses.replace(
        old,
        orig_shape=(old.orig_shape[1], old.orig_shape[0]),
        transposed=not old.transposed,
    )
    return QuantizedTensor(input_tensor._qdata, "AsymW4A8Int8Layout", new_params)


@register_layout_op(torch.ops.aten.linear.default, AsymW4A8Int8Layout)
def _handle_w4a8int8_linear(qt, args, kwargs):
    input_tensor, weight = args[0], args[1]
    bias = args[2] if len(args) > 2 else None
    if not _is_w4a8(weight):
        return torch.nn.functional.linear(*dequantize_args((input_tensor, weight, bias)))
    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()
    if weight._params.transposed:
        return torch.nn.functional.linear(input_tensor, weight.dequantize(), bias)
    out_dtype = kwargs.get("out_dtype", weight._params.orig_dtype)
    return _w4a8_int8_forward(input_tensor, weight, bias, out_dtype)


@register_layout_op(torch.ops.aten.mm.default, AsymW4A8Int8Layout)
def _handle_w4a8int8_mm(qt, args, kwargs):
    input_tensor, weight = args[0], args[1]
    if not _is_w4a8(weight):
        return torch.mm(*dequantize_args((input_tensor, weight)))
    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()
    weight = _resolve_w4a8_rhs(weight)
    return _w4a8_int8_forward(
        input_tensor,
        weight,
        bias=None,
        out_dtype=weight._params.orig_dtype,
    )


@register_layout_op(torch.ops.aten.addmm.default, AsymW4A8Int8Layout)
def _handle_w4a8int8_addmm(qt, args, kwargs):
    bias, input_tensor, weight = args[0], args[1], args[2]
    scaled = kwargs.get("beta", 1) != 1 or kwargs.get("alpha", 1) != 1
    matrix_addend = isinstance(bias, torch.Tensor) and bias.dim() != 1
    if not _is_w4a8(weight) or scaled or matrix_addend:
        return torch.addmm(*dequantize_args(args), **dequantize_args(kwargs))
    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()
    weight = _resolve_w4a8_rhs(weight)
    return _w4a8_int8_forward(
        input_tensor,
        weight,
        bias=bias,
        out_dtype=weight._params.orig_dtype,
    )
