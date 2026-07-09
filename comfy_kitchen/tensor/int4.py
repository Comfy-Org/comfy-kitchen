# SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tensor-wise INT4 quantization layout (int4_tensorwise, W4A4 + optional ConvRot).

Downward extension of :mod:`comfy_kitchen.tensor.int8` (``TensorWiseINT8Layout``):
per-output-channel signed INT4 weights packed 2-per-byte (low nibble first, int8
container ``[N, K/2]``) with a per-row float32 scale ``[N, 1]``; activations are
rotated online (when ConvRot) and dynamically quantized per-token to INT4 inside
``int4_linear``. Emission contract is shared with the svdquant kernels: symmetric
``[-7, 7]``, ``scale = absmax / 7``.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass

import torch

from comfy_kitchen.registry import registry

from .base import (
    BaseLayoutParams,
    QuantizedLayout,
    QuantizedTensor,
    dequantize_args,
    get_cuda_capability,
    register_layout_op,
)

logger = logging.getLogger(__name__)

_INT4_DEQUANT_DTYPE_TO_CODE = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
}


def _dtype_code(dtype: torch.dtype) -> int:
    code = _INT4_DEQUANT_DTYPE_TO_CODE.get(dtype)
    if code is None:
        raise ValueError(f"Unsupported INT4 output dtype: {dtype}")
    return code


class TensorWiseINT4Layout(QuantizedLayout):
    """Per-row INT4 quantization (W4A4), packed storage, optional ConvRot.

    - Weights: per-output-channel fp32 scale, packed int8 container [N, K/2]
    - Activations: per-token scales + online rotation, dynamic inside int4_linear

    Uses the int8 tensor-core matmul path (INT4 codes are valid INT8 operands,
    INT32 accumulation is exact). Native INT4 tensor cores exist only on
    SM 7.5-8.9; newer architectures execute the same math via the INT8 pipeline.

    Note:
        Requires SM >= 8.0: the fused CUDA path is built on cp.async, which
        Turing lacks. Native INT4 tensor cores exist on SM 7.5-8.9; SM >= 9.0
        executes the same math via the INT8 pipeline (correct, int8-class speed).
    """

    MIN_SM_VERSION = (8, 0)

    @dataclass(frozen=True)
    class Params(BaseLayoutParams):
        """Tensor-wise INT4 layout parameters.

        Inherits scale, orig_dtype, orig_shape from BaseLayoutParams.
        ``orig_shape`` is the LOGICAL [N, K] shape; qdata is packed [N, K/2].
        """

        is_weight: bool = True
        convrot: bool = False
        convrot_groupsize: int = 256
        transposed: bool = False

        def _tensor_fields(self) -> list[str]:
            return ["scale"]

        def _validate_tensor_fields(self):
            pass

    @classmethod
    def quantize(
        cls,
        tensor: torch.Tensor,
        scale: torch.Tensor | float | str | None = None,
        stochastic_rounding: int | None = 0,
        inplace_ops: bool = False,
        is_weight: bool = True,
        per_channel: bool = True,
        convrot: bool = False,
        convrot_groupsize: int = 256,
        **kwargs,
    ) -> tuple[torch.Tensor, Params]:
        """Quantize a tensor to packed per-row INT4.

        Args:
            tensor: Input tensor to quantize (last dim even; ConvRot additionally
                requires last dim % convrot_groupsize == 0).
            scale: Ignored (scales are always recomputed per-row, absmax/7).
            stochastic_rounding: Seed for stochastic rounding. Disabled when <= 0.
            inplace_ops: Kept for ComfyUI compatibility; quantization does not mutate input.
            is_weight: If True this is a weight (ConvRot eligible).
            per_channel: Kept for interface compatibility; INT4 is always per-row.
            convrot: If True, apply orthogonal group-wise Hadamard rotation to weight.
            convrot_groupsize: Group size for Hadamard rotation.
            **kwargs: Additional arguments (ignored).

        Returns:
            Tuple of (packed_int4_data, params).
        """
        orig_dtype = tensor.dtype
        orig_shape = tuple(tensor.shape)

        if convrot and not is_weight:
            raise ValueError("convrot is only supported when is_weight is True")

        if convrot:
            impl = registry.get_implementation(
                "quantize_int4_convrot_weight",
                kwargs={"weight": tensor, "group_size": convrot_groupsize, "stochastic_rounding": stochastic_rounding},
            )
            qdata, qscale = impl(tensor, convrot_groupsize, stochastic_rounding=stochastic_rounding)
        else:
            impl = registry.get_implementation(
                "quantize_int4_rowwise",
                kwargs={"x": tensor, "stochastic_rounding": stochastic_rounding},
            )
            qdata, qscale = impl(tensor, stochastic_rounding=stochastic_rounding)

        params = cls.Params(
            scale=qscale,
            orig_dtype=orig_dtype,
            orig_shape=orig_shape,
            is_weight=is_weight,
            convrot=convrot,
            convrot_groupsize=convrot_groupsize,
        )
        return qdata, params

    @classmethod
    def dequantize(cls, qdata: torch.Tensor, params: Params) -> torch.Tensor:
        """Dequantize packed INT4 data back to original dtype (un-rotating ConvRot)."""
        output_dtype_code = _INT4_DEQUANT_DTYPE_TO_CODE.get(params.orig_dtype, 0)
        if getattr(params, "convrot", False):
            result = torch.ops.comfy_kitchen.dequantize_int4_convrot_weight_dtype(
                qdata, params.scale, params.convrot_groupsize, output_dtype_code
            )
        else:
            result = torch.ops.comfy_kitchen.dequantize_int4_simple_dtype(qdata, params.scale, output_dtype_code)
        return result.to(params.orig_dtype)

    @classmethod
    def get_plain_tensors(cls, qtensor: QuantizedTensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract raw tensors for computation: (packed_qdata, scale)."""
        return qtensor._qdata, qtensor._params.scale

    @classmethod
    def state_dict_tensors(cls, qdata: torch.Tensor, params: Params) -> dict[str, torch.Tensor]:
        """Return key suffix → tensor mapping for serialization."""
        return {
            "": qdata,
            "_scale": params.scale,
        }

    @classmethod
    def requantize_kwargs(cls, qtensor: QuantizedTensor) -> dict[str, object]:
        """Return INT4 quantization options needed to preserve this layout.

        Critical for the LoRA-offload requantize path: a ConvRot weight must be
        re-rotated before requantization (see the int8 convrot lesson,
        ComfyUI #14642).
        """
        params = qtensor._params
        return {
            "is_weight": getattr(params, "is_weight", True),
            "per_channel": True,
            "convrot": getattr(params, "convrot", False),
            "convrot_groupsize": getattr(params, "convrot_groupsize", 256),
        }

    @classmethod
    def get_logical_shape_from_storage(cls, storage_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Compute the logical shape from packed storage by reversing 2-per-byte packing.

        int4_tensorwise never pads (K must be even), so ``padded_shape`` equals the
        logical [N, K] and ``is_padded`` stays False — same hook as the NVFP4 layout.
        """
        return (storage_shape[0], storage_shape[1] * 2)

    @classmethod
    def supports_fast_matmul(cls) -> bool:
        """Check if the int8-pipeline matmul used by int4_linear is available."""
        capability = get_cuda_capability()
        if capability is None:
            return False
        return capability >= cls.MIN_SM_VERSION


# =============================================================================
# INT4 Tensor-wise Operations
# =============================================================================


@register_layout_op(torch.ops.aten.t.default, TensorWiseINT4Layout)
def _handle_int4_transpose(qt, args, kwargs):
    """Handle transpose as a logical flag flip for INT4 tensors."""
    input_tensor = args[0]
    if not isinstance(input_tensor, QuantizedTensor):
        return torch.ops.aten.t.default(*args, **kwargs)

    old = input_tensor._params
    new_params = dataclasses.replace(
        old,
        orig_shape=(old.orig_shape[1], old.orig_shape[0]),
        transposed=not old.transposed,
    )
    return QuantizedTensor(input_tensor._qdata, "TensorWiseINT4Layout", new_params)


def _int4_weight_linear(input_tensor, weight, bias, out_dtype):
    weight_qdata, weight_scale = TensorWiseINT4Layout.get_plain_tensors(weight)
    convrot = getattr(weight._params, "convrot", False)
    convrot_groupsize = getattr(weight._params, "convrot_groupsize", 256)
    return torch.ops.comfy_kitchen.int4_linear(
        input_tensor.contiguous(),
        weight_qdata.contiguous(),
        weight_scale,
        bias,
        _dtype_code(out_dtype),
        convrot,
        convrot_groupsize,
    )


@register_layout_op(torch.ops.aten.linear.default, TensorWiseINT4Layout)
def _handle_int4_linear_tensorwise(qt, args, kwargs):
    """INT4 linear for tensor-wise layout: output = input @ weight.T + bias."""
    input_tensor = args[0]
    weight = args[1]
    bias = args[2] if len(args) > 2 else None

    if not isinstance(weight, QuantizedTensor) or weight._layout_cls != "TensorWiseINT4Layout":
        return torch.nn.functional.linear(*dequantize_args(args), **dequantize_args(kwargs))
    if getattr(weight._params, "transposed", False):
        return torch.nn.functional.linear(*dequantize_args(args), **dequantize_args(kwargs))

    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()

    out_dtype = kwargs.get("out_dtype", input_tensor.dtype)
    return _int4_weight_linear(input_tensor, weight, bias, out_dtype)


@register_layout_op(torch.ops.aten.mm.default, TensorWiseINT4Layout)
def _handle_int4_mm_tensorwise(qt, args, kwargs):
    """INT4 matrix multiplication for tensor-wise layout: output = a @ b."""
    input_tensor = args[0]
    weight = args[1]

    if not isinstance(weight, QuantizedTensor) or weight._layout_cls != "TensorWiseINT4Layout":
        return torch.mm(*dequantize_args(args), **dequantize_args(kwargs))

    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()

    if not getattr(weight._params, "transposed", False):
        # Per-row scales belong to the rows of the logical RHS, not output
        # columns, so a directly-quantized RHS cannot ride int4_linear.
        return torch.mm(*dequantize_args(args), **dequantize_args(kwargs))

    out_dtype = kwargs.get("out_dtype", input_tensor.dtype)
    return _int4_weight_linear(input_tensor, weight, None, out_dtype)


@register_layout_op(torch.ops.aten.addmm.default, TensorWiseINT4Layout)
def _handle_int4_addmm_tensorwise(qt, args, kwargs):
    """INT4 addmm for tensor-wise layout: output = bias + input @ weight."""
    bias = args[0]
    input_tensor = args[1]
    weight = args[2]

    if not isinstance(weight, QuantizedTensor) or weight._layout_cls != "TensorWiseINT4Layout":
        return torch.addmm(*dequantize_args(args), **dequantize_args(kwargs))

    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()

    if not getattr(weight._params, "transposed", False):
        return torch.addmm(*dequantize_args(args), **dequantize_args(kwargs))

    out_dtype = kwargs.get("out_dtype", input_tensor.dtype)
    return _int4_weight_linear(input_tensor, weight, bias, out_dtype)
