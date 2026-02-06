"""FP8 (E4M3) quantization layout for tensor cores."""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

import comfy_kitchen as ck

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
    """Core FP8 scaled matrix multiplication using torch._scaled_mm."""
    output = torch._scaled_mm(
        input_qdata.contiguous(),
        weight_qdata,
        bias=bias,
        scale_a=scale_a,
        scale_b=scale_b,
        out_dtype=out_dtype,
    )
    # Handle tuple return for older PyTorch versions
    return output[0] if isinstance(output, tuple) else output


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


# ==================== Distributed Operations ====================
# Required c10d ops : c10d allgather, c10d wait, c10d broadcast (this is for broadcast_rank0=True)
# Required aten ops : slice, split, new_zeros, as_strided, cat, alias

@register_layout_op(torch.ops._c10d_functional.all_gather_into_tensor.default, TensorCoreFP8Layout)
def _handle_all_gather(qt, args, kwargs):
    from .base import QuantizedTensor

    input_tensor = None
    input_idx = None
    for i, arg in enumerate(args):
        if isinstance(arg, QuantizedTensor):
            input_tensor = arg
            input_idx = i

    assert input_tensor is not None

    qdata = input_tensor._qdata
    layout_cls = input_tensor._layout_cls
    params = input_tensor._params

    qdata_bytes = qdata.contiguous().view(torch.uint8)

    new_args = list(args)
    new_args[input_idx] = qdata_bytes

    gathered_bytes = torch.ops._c10d_functional.all_gather_into_tensor.default(
        *new_args, **kwargs
    )

    gathered_qdata = gathered_bytes.view(qdata.dtype)
    gathered_params = replace(params, orig_shape=tuple(gathered_qdata.shape))
    return QuantizedTensor(gathered_qdata, layout_cls, gathered_params)


@register_layout_op(torch.ops._c10d_functional.wait_tensor.default, TensorCoreFP8Layout)
def _handle_wait_tensor(qt, args, kwargs):
    from .base import QuantizedTensor

    qtensor = args[0]

    waited_bytes = torch.ops._c10d_functional.wait_tensor.default(
        qtensor._qdata.view(torch.uint8),
        *args[1:],
        **kwargs,
    )

    waited_qdata = waited_bytes.view(qtensor._qdata.dtype)
    waited_params = replace(qtensor._params, orig_shape=tuple(waited_qdata.shape))
    return QuantizedTensor(waited_qdata, qtensor._layout_cls, waited_params)


@register_layout_op(torch.ops.c10d.broadcast_.default, TensorCoreFP8Layout)
def _handle_broadcast(qt, args, kwargs):
    from .base import QuantizedTensor
    import torch

    tensor_list = args[0]

    input_tensor = None
    input_idx = None
    for idx, t in enumerate(tensor_list):
        if isinstance(t, QuantizedTensor):
            input_tensor = t
            input_idx = idx
            break

    if input_tensor is None:
        return torch.ops.c10d.broadcast_.default(*args, **kwargs)

    qdata = input_tensor._qdata.contiguous()
    qdata_bytes = qdata.view(torch.uint8)

    new_tensor_list = list(tensor_list)
    new_tensor_list[input_idx] = qdata_bytes

    new_args = list(args)
    new_args[0] = new_tensor_list

    broadcasted = torch.ops.c10d.broadcast_.default(
        *new_args,
        **kwargs,
    )

    if isinstance(broadcasted, tuple):
        tensor_list_out, work = broadcasted
    else:
        tensor_list_out = broadcasted
        work = None

    broadcasted_qdata = tensor_list_out[input_idx].view(qdata.dtype)

    new_out_list = list(tensor_list_out)
    new_out_list[input_idx] = _wrap_fp8_tensor(
        input_tensor,
        broadcasted_qdata,
    )

    if work is not None:
        return new_out_list, work
    else:
        return new_out_list


def _wrap_fp8_tensor(qtensor, qdata):
    from .base import QuantizedTensor

    new_params = TensorCoreFP8Layout.Params(
        scale=qtensor._params.scale,
        orig_dtype=qtensor._params.orig_dtype,
        orig_shape=tuple(qdata.shape),
    )
    return QuantizedTensor(qdata, qtensor._layout_cls, new_params)


@register_layout_op(torch.ops.c10d.scatter_.default, TensorCoreFP8Layout)
def _handle_scatter(qt, args, kwargs):
    from .base import QuantizedTensor

    output_tensors = args[0]
    input_tensors = args[1]

    quantized_outputs: list[tuple[int, QuantizedTensor]] = []
    new_output_tensors = list(output_tensors)
    for idx, tensor in enumerate(output_tensors):
        if isinstance(tensor, QuantizedTensor):
            quantized_outputs.append((idx, tensor))
            new_output_tensors[idx] = tensor._qdata.contiguous().view(torch.uint8)

    has_quantized_input = False
    new_input_tensors: list[list[torch.Tensor]] = []

    def process_input_list(entry):
        nonlocal has_quantized_input
        processed: list[torch.Tensor] = []
        for t in entry:
            if isinstance(t, QuantizedTensor):
                has_quantized_input = True
                processed.append(t._qdata.contiguous().view(torch.uint8))
            else:
                processed.append(t)
        return processed

    for entry in input_tensors:
        if isinstance(entry, (list, tuple)):
            new_input_tensors.append(process_input_list(entry))
        else:
            new_input_tensors.append(entry)  # type: ignore[arg-type]

    if not quantized_outputs and not has_quantized_input:
        return torch.ops.c10d.scatter_.default(*args, **kwargs)

    new_args = [new_output_tensors, new_input_tensors, *args[2:]]
    result = torch.ops.c10d.scatter_.default(*new_args, **kwargs)

    if isinstance(result, tuple):
        output_list, work = result
    else:
        output_list = result
        work = None

    output_list = list(output_list)
    for idx, original in quantized_outputs:
        qdata = output_list[idx].view(original._qdata.dtype)
        output_list[idx] = _wrap_fp8_tensor(original, qdata)

    if work is not None:
        return output_list, work
    return output_list


@register_layout_op(torch.ops.aten.slice.Tensor, TensorCoreFP8Layout)
def _handle_fp8_slice_tensor(qt, args, kwargs):
    from .base import QuantizedTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, QuantizedTensor):
        return torch.ops.aten.slice.Tensor(*args, **kwargs)

    sliced_qdata = torch.ops.aten.slice.Tensor(input_tensor._qdata, *args[1:], **kwargs)
    return _wrap_fp8_tensor(input_tensor, sliced_qdata)


@register_layout_op(torch.ops.aten.split.Tensor, TensorCoreFP8Layout)
def _handle_fp8_split(qt, args, kwargs):
    from .base import QuantizedTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, QuantizedTensor):
        return torch.ops.aten.split.Tensor(*args, **kwargs)

    qdata_chunks = torch.ops.aten.split.Tensor(input_tensor._qdata, *args[1:], **kwargs)
    wrapped_chunks = tuple(
        _wrap_fp8_tensor(input_tensor, chunk)
        for chunk in qdata_chunks
    )
    return wrapped_chunks


@register_layout_op(torch.ops.aten.cat.default, TensorCoreFP8Layout)
def _handle_fp8_cat(qt, args, kwargs):
    from .base import QuantizedTensor

    tensors = args[0]
    if not isinstance(tensors, (list, tuple)) or not tensors:
        return torch.ops.aten.cat.default(*args, **kwargs)

    qdata_list = []
    first_qtensor = None
    for item in tensors:
        if not isinstance(item, QuantizedTensor):
            return torch.ops.aten.cat.default(*args, **kwargs)
        qdata_list.append(item._qdata)
        if first_qtensor is None:
            first_qtensor = item

    assert first_qtensor is not None
    concatenated_qdata = torch.ops.aten.cat.default(qdata_list, *args[1:], **kwargs)
    return _wrap_fp8_tensor(first_qtensor, concatenated_qdata)


# ==================== FP8 Shape Operations ====================
# These preserve quantization since FP8 is not packed (1:1 element mapping)

for _aten_op in (
    torch.ops.aten.view.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten.t.default,
    torch.ops.aten.as_strided.default,
    torch.ops.aten.new_zeros.default,
    torch.ops.aten.alias.default
):
    register_layout_op(_aten_op, TensorCoreFP8Layout)(_make_fp8_shape_handler(_aten_op))
