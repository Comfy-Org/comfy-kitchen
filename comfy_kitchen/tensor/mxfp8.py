"""MXFP8 (Microscaling FP8) block quantization layout for tensor cores."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

import comfy_kitchen as ck
from comfy_kitchen.float_utils import roundup

from .base import BaseLayoutParams, QuantizedLayout, dequantize_args, register_layout_op

if TYPE_CHECKING:
    from .base import QuantizedTensor

logger = logging.getLogger(__name__)


class TensorCoreMXFP8Layout(QuantizedLayout):
    """MXFP8 block quantization with E8M0 (power-of-2) block scaling.

    MXFP8 uses:
    - FP8 E4M3 data (no packing)
    - E8M0 block scales (pure power-of-2 exponents)
    - Block size of 32

    Auto-pads to 32x32 alignment for cuBLAS compatibility.

    Note:
        Requires SM >= 10.0 (Blackwell) for hardware-accelerated matmul.
        Shape operations (view, reshape) are not supported.
    """

    MIN_SM_VERSION = (10, 0)

    @dataclass(frozen=True)
    class Params(BaseLayoutParams):
        """MXFP8 layout parameters.

        Inherits scale, orig_dtype, orig_shape from BaseLayoutParams.
        scale contains the E8M0 per-block scaling factors.
        """
        transposed: bool = False

        def _tensor_fields(self) -> list[str]:
            return ["scale"]

        def _validate_tensor_fields(self):
            pass

    @classmethod
    def quantize(
        cls,
        tensor: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, Params]:
        if tensor.dim() != 2:
            raise ValueError(f"MXFP8 requires 2D tensor, got {tensor.dim()}D")

        orig_dtype = tensor.dtype
        orig_shape = tuple(tensor.shape)

        padded_shape = cls.get_padded_shape(orig_shape)
        needs_padding = padded_shape != orig_shape

        qdata, block_scale = ck.quantize_mxfp8(tensor, pad_32x=needs_padding)

        params = cls.Params(
            scale=block_scale,
            orig_dtype=orig_dtype,
            orig_shape=orig_shape,
        )
        return qdata, params

    @classmethod
    def dequantize(cls, qdata: torch.Tensor, params: Params) -> torch.Tensor:
        return ck.dequantize_mxfp8(qdata, params.scale, params.orig_dtype)

    @classmethod
    def get_plain_tensors(cls, qtensor: QuantizedTensor) -> tuple[torch.Tensor, torch.Tensor]:
        return qtensor._qdata, qtensor._params.scale

    @classmethod
    def state_dict_tensors(cls, qdata: torch.Tensor, params: Params) -> dict[str, torch.Tensor]:
        return {"": qdata, "_scale": params.scale}

    @classmethod
    def get_padded_shape(cls, orig_shape: tuple[int, ...]) -> tuple[int, ...]:
        if len(orig_shape) != 2:
            raise ValueError(f"MXFP8 requires 2D shape, got {len(orig_shape)}D")
        rows, cols = orig_shape
        return (roundup(rows, 32), roundup(cols, 32))

    @classmethod
    def get_storage_shape(cls, orig_shape: tuple[int, ...]) -> tuple[int, ...]:
        return cls.get_padded_shape(orig_shape)

    @classmethod
    def get_logical_shape_from_storage(cls, storage_shape: tuple[int, ...]) -> tuple[int, ...]:
        return storage_shape


def _mxfp8_scaled_mm(
    a_qdata: torch.Tensor,
    b_qdata: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """MXFP8 scaled matmul: computes a @ b.T + bias (linear semantics)."""
    return ck.scaled_mm_mxfp8(
        a_qdata, b_qdata,
        block_scale_a=scale_a,
        block_scale_b=scale_b,
        bias=bias,
        out_dtype=out_dtype,
    )


def _slice_to_original_shape(result: torch.Tensor, orig_m: int, orig_n: int) -> torch.Tensor:
    """Slice padded matmul output back to original dimensions."""
    if result.shape[0] != orig_m or result.shape[1] != orig_n:
        return result[:orig_m, :orig_n]
    return result


@register_layout_op(torch.ops.aten.t.default, TensorCoreMXFP8Layout)
def _handle_mxfp8_transpose(qt, args, kwargs):
    """Handle transpose as a logical flag flip for MXFP8."""
    from .base import QuantizedTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, QuantizedTensor):
        return torch.ops.aten.t.default(*args, **kwargs)

    old_shape = input_tensor._params.orig_shape
    new_params = TensorCoreMXFP8Layout.Params(
        orig_dtype=input_tensor._params.orig_dtype,
        orig_shape=(old_shape[1], old_shape[0]),
        scale=input_tensor._params.scale,
        transposed=not input_tensor._params.transposed,
    )
    return QuantizedTensor(input_tensor._qdata, "TensorCoreMXFP8Layout", new_params)


@register_layout_op(torch.ops.aten.mm.default, TensorCoreMXFP8Layout)
def _handle_mxfp8_mm(qt, args, kwargs):
    """MXFP8 mm: requires b to be logically transposed (from .t() call)."""
    from .base import QuantizedTensor

    a, b = args[0], args[1]

    if not (isinstance(a, QuantizedTensor) and isinstance(b, QuantizedTensor)):
        return torch.mm(*dequantize_args(args))
    if a._qdata.dim() != 2:
        return torch.mm(*dequantize_args(args))

    a_transposed = getattr(a._params, "transposed", False)
    b_transposed = getattr(b._params, "transposed", False)

    if a_transposed or not b_transposed:
        return torch.mm(*dequantize_args(args))

    a_qdata, scale_a = TensorCoreMXFP8Layout.get_plain_tensors(a)
    b_qdata, scale_b = TensorCoreMXFP8Layout.get_plain_tensors(b)
    out_dtype = kwargs.get("out_dtype", a._params.orig_dtype)

    try:
        result = _mxfp8_scaled_mm(a_qdata, b_qdata, scale_a, scale_b, out_dtype=out_dtype)
        return _slice_to_original_shape(result, a._params.orig_shape[0], b._params.orig_shape[1])
    except (RuntimeError, TypeError) as e:
        logger.warning(f"MXFP8 mm failed: {e}")
        return torch.mm(*dequantize_args(args))


@register_layout_op(torch.ops.aten.addmm.default, TensorCoreMXFP8Layout)
def _handle_mxfp8_addmm(qt, args, kwargs):
    """MXFP8 addmm: bias + input @ weight.T (decomposed from F.linear with bias)."""
    from .base import QuantizedTensor

    bias, mat1, mat2 = args[0], args[1], args[2]

    if not (isinstance(mat1, QuantizedTensor) and isinstance(mat2, QuantizedTensor)):
        return torch.addmm(*dequantize_args((bias, mat1, mat2)))
    if mat1._qdata.dim() != 2:
        return torch.addmm(*dequantize_args((bias, mat1, mat2)))

    input_transposed = getattr(mat1._params, "transposed", False)
    weight_transposed = getattr(mat2._params, "transposed", False)

    if input_transposed or not weight_transposed:
        return torch.addmm(*dequantize_args((bias, mat1, mat2)))

    input_qdata, scale_a = TensorCoreMXFP8Layout.get_plain_tensors(mat1)
    weight_qdata, scale_b = TensorCoreMXFP8Layout.get_plain_tensors(mat2)
    out_dtype = mat1._params.orig_dtype

    try:
        result = _mxfp8_scaled_mm(input_qdata, weight_qdata, scale_a, scale_b, bias, out_dtype)
        orig_m = mat1._params.orig_shape[0]
        orig_n = mat2._params.orig_shape[1]
        return _slice_to_original_shape(result, orig_m, orig_n)
    except (RuntimeError, TypeError) as e:
        logger.warning(f"MXFP8 addmm failed: {e}")
        return torch.addmm(*dequantize_args((bias, mat1, mat2)))


@register_layout_op(torch.ops.aten.linear.default, TensorCoreMXFP8Layout)
def _handle_mxfp8_linear(qt, args, kwargs):
    """MXFP8 linear: input @ weight.T + bias."""
    from .base import QuantizedTensor

    input_tensor, weight = args[0], args[1]
    bias = args[2] if len(args) > 2 else None

    if not (isinstance(input_tensor, QuantizedTensor) and isinstance(weight, QuantizedTensor)):
        return torch.nn.functional.linear(*dequantize_args((input_tensor, weight, bias)))
    if input_tensor._qdata.dim() != 2:
        return torch.nn.functional.linear(*dequantize_args((input_tensor, weight, bias)))

    if getattr(input_tensor._params, "transposed", False) or getattr(weight._params, "transposed", False):
        return torch.nn.functional.linear(*dequantize_args((input_tensor, weight, bias)))

    input_qdata, scale_a = TensorCoreMXFP8Layout.get_plain_tensors(input_tensor)
    weight_qdata, scale_b = TensorCoreMXFP8Layout.get_plain_tensors(weight)
    out_dtype = kwargs.get("out_dtype", input_tensor._params.orig_dtype)

    try:
        result = _mxfp8_scaled_mm(input_qdata, weight_qdata, scale_a, scale_b, bias, out_dtype)
        return _slice_to_original_shape(result, input_tensor._params.orig_shape[0], weight._params.orig_shape[0])
    except (RuntimeError, TypeError) as e:
        logger.warning(f"MXFP8 linear failed: {e}")
        return torch.nn.functional.linear(*dequantize_args((input_tensor, weight, bias)))
