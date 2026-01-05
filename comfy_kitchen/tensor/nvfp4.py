"""NVFP4 (E2M1) block quantization layout for tensor cores."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

import comfy_kitchen as ck
from comfy_kitchen.float_utils import F4_E2M1_MAX, F8_E4M3_MAX, roundup

from .base import BaseLayoutParams, QuantizedLayout, dequantize_args, register_layout_op

if TYPE_CHECKING:
    from .base import QuantizedTensor

logger = logging.getLogger(__name__)


class TensorCoreNVFP4Layout(QuantizedLayout):
    """NVFP4 E2M1 block quantization with per-tensor and block scaling.
    Auto-pads to 16x16 alignment

    Note:
        Requires SM >= 10.0 (Blackwell) for hardware-accelerated matmul.
        Shape operations (view, reshape, transpose) are not supported due to
        packed format and block scales - they fall back to dequantization.
    """

    MIN_SM_VERSION = (10, 0)

    @dataclass(frozen=True)
    class Params(BaseLayoutParams):
        """NVFP4 layout parameters.

        Inherits scale, orig_dtype, orig_shape from BaseLayoutParams.
        Adds block_scale for per-block scaling factors.
        """
        block_scale: torch.Tensor
        transposed: bool = False

        def _tensor_fields(self) -> list[str]:
            """Override to include block_scale in tensor operations."""
            return ["scale", "block_scale"]

    @classmethod
    def quantize(
        cls,
        tensor: torch.Tensor,
        scale: torch.Tensor | float | str | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Params]:
        if tensor.dim() != 2:
            raise ValueError(f"NVFP4 requires 2D tensor, got {tensor.dim()}D")

        orig_dtype = tensor.dtype
        orig_shape = tuple(tensor.shape)

        if scale is None or scale == "recalculate":
            scale = torch.amax(tensor.abs()) / (F8_E4M3_MAX * F4_E2M1_MAX)

        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale)
        scale = scale.to(device=tensor.device, dtype=torch.float32)

        padded_shape = cls.get_padded_shape(orig_shape)
        needs_padding = padded_shape != orig_shape

        qdata, block_scale = ck.quantize_nvfp4(tensor, scale, pad_16x=needs_padding)

        params = cls.Params(
            scale=scale,
            orig_dtype=orig_dtype,
            orig_shape=orig_shape,
            block_scale=block_scale,
        )
        return qdata, params

    @classmethod
    def dequantize(cls, qdata: torch.Tensor, params: Params) -> torch.Tensor:
        return ck.dequantize_nvfp4(qdata, params.scale, params.block_scale, params.orig_dtype)

    @classmethod
    def get_plain_tensors(
        cls, qtensor: QuantizedTensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return qtensor._qdata, qtensor._params.scale, qtensor._params.block_scale

    @classmethod
    def state_dict_tensors(cls, qdata: torch.Tensor, params: Params) -> dict[str, torch.Tensor]:
        """Return key suffix → tensor mapping for serialization."""
        return {
            "": qdata,
            "_scale": params.block_scale,
            "_scale_2": params.scale,
        }

    @classmethod
    def get_padded_shape(cls, orig_shape: tuple[int, ...]) -> tuple[int, ...]:
        if len(orig_shape) != 2:
            raise ValueError(f"NVFP4 requires 2D shape, got {len(orig_shape)}D")
        rows, cols = orig_shape
        return (roundup(rows, 16), roundup(cols, 16))

    @classmethod
    def get_storage_shape(cls, orig_shape: tuple[int, ...]) -> tuple[int, ...]:
        padded = cls.get_padded_shape(orig_shape)
        return (padded[0], padded[1] // 2)

    @classmethod
    def get_logical_shape_from_storage(cls, storage_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Compute logical (padded) shape from storage shape by reversing packing."""
        return (storage_shape[0], storage_shape[1] * 2)


# ==================== NVFP4 Transpose Operation ====================
# Transpose is a no-op that tracks logical transposition via a flag.

@register_layout_op(torch.ops.aten.t.default, TensorCoreNVFP4Layout)
def _handle_nvfp4_transpose(qt, args, kwargs):
    """Handle transpose as a logical no-op for NVFP4.
    """
    from .base import QuantizedTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, QuantizedTensor):
        return torch.ops.aten.t.default(*args, **kwargs)

    old_shape = input_tensor._params.orig_shape
    new_shape = (old_shape[1], old_shape[0])

    new_params = TensorCoreNVFP4Layout.Params(
        scale=input_tensor._params.scale,
        orig_dtype=input_tensor._params.orig_dtype,
        orig_shape=new_shape,
        block_scale=input_tensor._params.block_scale,
        transposed=not input_tensor._params.transposed,
    )
    return QuantizedTensor(input_tensor._qdata, "TensorCoreNVFP4Layout", new_params)


# ==================== NVFP4 Matmul Operations ====================

def _slice_to_original_shape(
    result: torch.Tensor,
    orig_m: int,
    orig_n: int,
) -> torch.Tensor:
    """Slice padded matmul output back to original dimensions."""
    if result.shape[0] != orig_m or result.shape[1] != orig_n:
        return result[:orig_m, :orig_n]
    return result


@register_layout_op(torch.ops.aten.mm.default, TensorCoreNVFP4Layout)
def _handle_nvfp4_mm(qt, args, kwargs):
    """NVFP4 matrix multiplication: output = a @ b

    When b is logically transposed (from a prior .t() call), this works directly
    with scaled_mm_nvfp4 since that kernel computes a @ b_phys.T, which equals
    a @ b_logical when b_logical = b_phys.T.

    This handles the common torch.compile decomposition: linear(x, w) → mm(x, w.t())
    """
    from .base import QuantizedTensor

    a, b = args[0], args[1]

    # Fast path: both operands are NVFP4 QuantizedTensors
    if not (isinstance(a, QuantizedTensor) and isinstance(b, QuantizedTensor)):
        return torch.mm(*dequantize_args(args))

    # NVFP4 only supports 2D tensors
    if a._qdata.dim() != 2:
        return torch.mm(*dequantize_args(args))

    a_transposed = getattr(a._params, "transposed", False)
    b_transposed = getattr(b._params, "transposed", False)

    if a_transposed or not b_transposed:
        # Can't handle these cases with current kernel, fallback
        logger.debug("NVFP4 mm: unsupported transpose configuration, falling back to dequantize")
        return torch.mm(*dequantize_args(args))

    a_qdata, scale_a, block_scale_a = TensorCoreNVFP4Layout.get_plain_tensors(a)
    b_qdata, scale_b, block_scale_b = TensorCoreNVFP4Layout.get_plain_tensors(b)
    out_dtype = kwargs.get("out_dtype", a._params.orig_dtype)

    try:
        result = ck.scaled_mm_nvfp4(
            a_qdata,
            b_qdata,
            tensor_scale_a=scale_a,
            tensor_scale_b=scale_b,
            block_scale_a=block_scale_a,
            block_scale_b=block_scale_b,
            out_dtype=out_dtype,
        )

        orig_m = a._params.orig_shape[0]
        orig_n = b._params.orig_shape[1]
        return _slice_to_original_shape(result, orig_m, orig_n)

    except (RuntimeError, TypeError) as e:
        logger.warning(f"NVFP4 mm failed: {e}, falling back to dequantization")
        return torch.mm(*dequantize_args(args))


@register_layout_op(torch.ops.aten.linear.default, TensorCoreNVFP4Layout)
def _handle_nvfp4_linear(qt, args, kwargs):
    """NVFP4 linear: output = input @ weight.T + bias

    Uses ck.scaled_mm_nvfp4 for hardware-accelerated NVFP4 matmul.
    Output is sliced to original (non-padded) shape.
    """
    from .base import QuantizedTensor

    input_tensor, weight = args[0], args[1]
    bias = args[2] if len(args) > 2 else None

    # Fast path: both operands are NVFP4 QuantizedTensors
    if not (isinstance(input_tensor, QuantizedTensor) and isinstance(weight, QuantizedTensor)):
        return torch.nn.functional.linear(*dequantize_args((input_tensor, weight, bias)))

    # NVFP4 only supports 2D tensors
    if input_tensor._qdata.dim() != 2:
        return torch.nn.functional.linear(*dequantize_args((input_tensor, weight, bias)))

    input_transposed = getattr(input_tensor._params, "transposed", False)
    weight_transposed = getattr(weight._params, "transposed", False)
    if input_transposed or weight_transposed:
        logger.debug("NVFP4 linear: unsupported transpose configuration, falling back to dequantize")
        return torch.nn.functional.linear(*dequantize_args((input_tensor, weight, bias)))

    input_qdata, scale_a, block_scale_a = TensorCoreNVFP4Layout.get_plain_tensors(input_tensor)
    weight_qdata, scale_b, block_scale_b = TensorCoreNVFP4Layout.get_plain_tensors(weight)
    out_dtype = kwargs.get("out_dtype", input_tensor._params.orig_dtype)

    try:
        # scaled_mm_nvfp4 computes (a @ b.T) * scale, which is linear semantics
        result = ck.scaled_mm_nvfp4(
            input_qdata,
            weight_qdata,
            tensor_scale_a=scale_a,
            tensor_scale_b=scale_b,
            block_scale_a=block_scale_a,
            block_scale_b=block_scale_b,
            bias=bias,
            out_dtype=out_dtype,
        )

        # Slice output to original (non-padded) shape
        orig_m = input_tensor._params.orig_shape[0]
        orig_n = weight._params.orig_shape[0]  # weight is (out_features, in_features)
        return _slice_to_original_shape(result, orig_m, orig_n)

    except (RuntimeError, TypeError) as e:
        logger.warning(f"NVFP4 scaled_mm failed: {e}, falling back to dequantization")
        return torch.nn.functional.linear(*dequantize_args((input_tensor, weight, bias)))
