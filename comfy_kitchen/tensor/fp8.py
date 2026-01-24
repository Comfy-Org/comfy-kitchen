"""FP8 (E4M3) quantization layout for tensor cores."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

import comfy_kitchen as ck
from comfy_kitchen.scaled_mm_v2 import scaled_mm_v2

from .base import BaseLayoutParams, QuantizedLayout, dequantize_args, register_layout_op

if TYPE_CHECKING:
    from .base import QuantizedTensor

logger = logging.getLogger(__name__)


class TensorCoreFP8Layout(QuantizedLayout):
    """FP8 E4M3 quantization with per-tensor scaling.

    Example:
        >>> x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        >>> qt = QuantizedTensor.from_float(x, TensorCoreFP8Layout)
        >>> qt.shape
        torch.Size([128, 256])
        >>> dq = qt.dequantize()
        >>> torch.allclose(dq, x, rtol=0.1)
        True

    Note:
        Requires SM >= 8.9 (Ada Lovelace) for hardware-accelerated matmul.
    """

    MIN_SM_VERSION = (8, 9)

    @dataclass(frozen=True)
    class Params(BaseLayoutParams):
        def _validate_tensor_fields(self):
            if isinstance(self.scale, torch.Tensor):
                object.__setattr__(self, "scale", self.scale.to(dtype=torch.float32, non_blocking=True))

    @classmethod
    def quantize(
        cls,
        tensor: torch.Tensor,
        scale: torch.Tensor | float | str | None = None,
        dtype: torch.dtype = torch.float8_e4m3fn,
        **kwargs,
    ) -> tuple[torch.Tensor, Params]:
        orig_dtype = tensor.dtype
        orig_shape = tuple(tensor.shape)

        if scale is None or scale == "recalculate":
            scale = torch.amax(tensor.abs()) / torch.finfo(dtype).max

        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale)
        scale = scale.to(device=tensor.device, dtype=torch.float32)

        qdata = ck.quantize_per_tensor_fp8(tensor, scale, dtype)

        return qdata, cls.Params(scale=scale, orig_dtype=orig_dtype, orig_shape=orig_shape)

    @classmethod
    def dequantize(cls, qdata: torch.Tensor, params: Params) -> torch.Tensor:
        return ck.dequantize_per_tensor_fp8(qdata, params.scale, params.orig_dtype)

    @classmethod
    def get_plain_tensors(cls, qtensor: QuantizedTensor) -> tuple[torch.Tensor, torch.Tensor]:
        return qtensor._qdata, qtensor._params.scale

    @classmethod
    def state_dict_tensors(cls, qdata: torch.Tensor, params: Params) -> dict[str, torch.Tensor]:
        """Return key suffix → tensor mapping for serialization."""
        return {
            "": qdata,
            "_scale": params.scale,
        }


# ==================== Helper Utilities ====================

def _fp8_scaled_mm(
    input_qdata: torch.Tensor,
    weight_qdata: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return scaled_mm_v2(
        input_qdata.contiguous(),
        weight_qdata,
        scale_a=scale_a,
        scale_b=scale_b,
        bias=bias,
        out_dtype=out_dtype,
    )


def _make_fp8_shape_handler(aten_op):
    """Factory for shape-changing operations (view, reshape, t, etc.).

    These ops work directly on FP8 since it's not packed (1:1 element mapping).
    The aten_op is applied to _qdata and the result is wrapped in a new QuantizedTensor.
    """
    from .base import QuantizedTensor

    def handler(qt, args, kwargs):
        input_tensor = args[0]
        if not isinstance(input_tensor, QuantizedTensor):
            return aten_op(*args, **kwargs)

        # Apply aten op to quantized data
        new_qdata = aten_op(input_tensor._qdata, *args[1:], **kwargs)

        # Create new params with updated shape (scale is shared, not cloned)
        new_params = TensorCoreFP8Layout.Params(
            scale=input_tensor._params.scale,
            orig_dtype=input_tensor._params.orig_dtype,
            orig_shape=tuple(new_qdata.shape),
        )
        return QuantizedTensor(new_qdata, "TensorCoreFP8Layout", new_params)

    return handler


# ==================== FP8 Matmul Operations ====================

@register_layout_op(torch.ops.aten.linear.default, TensorCoreFP8Layout)
def _handle_fp8_linear(qt, args, kwargs):
    """FP8 linear: output = input @ weight.T + bias

    Uses torch._scaled_mm for hardware-accelerated FP8 matmul when both
    input and weight are FP8 QuantizedTensors.
    """
    from .base import QuantizedTensor

    input_tensor, weight = args[0], args[1]
    bias = args[2] if len(args) > 2 else None

    # Fast path: both operands are FP8 QuantizedTensors
    if not (isinstance(input_tensor, QuantizedTensor) and isinstance(weight, QuantizedTensor)):
        return torch.nn.functional.linear(*dequantize_args((input_tensor, weight, bias)))

    input_qdata, scale_a = TensorCoreFP8Layout.get_plain_tensors(input_tensor)
    weight_qdata, scale_b = TensorCoreFP8Layout.get_plain_tensors(weight)
    out_dtype = kwargs.get("out_dtype", input_tensor._params.orig_dtype)

    # Transpose weight for linear: output = input @ weight.T
    weight_t = weight_qdata.t()

    try:
        output = _fp8_scaled_mm(input_qdata, weight_t, scale_a, scale_b, bias, out_dtype)

        # Wrap FP8 output in QuantizedTensor
        if output.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            output_params = TensorCoreFP8Layout.Params(
                scale=scale_a * scale_b,
                orig_dtype=input_tensor._params.orig_dtype,
                orig_shape=tuple(output.shape),
            )
            return QuantizedTensor(output, "TensorCoreFP8Layout", output_params)
        return output

    except (RuntimeError, TypeError) as e:
        logger.warning(f"FP8 _scaled_mm failed: {e}, falling back to dequantization")
        return torch.nn.functional.linear(*dequantize_args((input_tensor, weight, bias)))


@register_layout_op(torch.ops.aten.mm.default, TensorCoreFP8Layout)
def _handle_fp8_mm(qt, args, kwargs):
    """FP8 matrix multiplication: output = a @ b"""
    from .base import QuantizedTensor

    a, b = args[0], args[1]

    if not (isinstance(a, QuantizedTensor) and isinstance(b, QuantizedTensor)):
        return torch.mm(*dequantize_args(args))

    a_qdata, scale_a = TensorCoreFP8Layout.get_plain_tensors(a)
    b_qdata, scale_b = TensorCoreFP8Layout.get_plain_tensors(b)
    out_dtype = kwargs.get("out_dtype", a._params.orig_dtype)

    try:
        return _fp8_scaled_mm(a_qdata, b_qdata, scale_a, scale_b, out_dtype=out_dtype)
    except (RuntimeError, TypeError):
        return torch.mm(*dequantize_args(args))


@register_layout_op(torch.ops.aten.addmm.default, TensorCoreFP8Layout)
def _handle_fp8_addmm(qt, args, kwargs):
    """FP8 addmm: output = bias + input @ weight"""
    from .base import QuantizedTensor

    bias, input_tensor, weight = args[0], args[1], args[2]

    if not (isinstance(input_tensor, QuantizedTensor) and isinstance(weight, QuantizedTensor)):
        return torch.addmm(*dequantize_args(args))

    input_qdata, scale_a = TensorCoreFP8Layout.get_plain_tensors(input_tensor)
    weight_qdata, scale_b = TensorCoreFP8Layout.get_plain_tensors(weight)
    out_dtype = kwargs.get("out_dtype", input_tensor._params.orig_dtype)

    try:
        return _fp8_scaled_mm(input_qdata, weight_qdata, scale_a, scale_b, bias, out_dtype)
    except (RuntimeError, TypeError):
        return torch.addmm(*dequantize_args(args))


# ==================== FP8 Shape Operations ====================
# These preserve quantization since FP8 is not packed (1:1 element mapping)

for _aten_op in (
    torch.ops.aten.view.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten.t.default,
):
    register_layout_op(_aten_op, TensorCoreFP8Layout)(_make_fp8_shape_handler(_aten_op))
