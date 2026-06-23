"""NVFP4 (E2M1) block quantization layout for tensor cores."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

import comfy_kitchen as ck
from comfy_kitchen.float_utils import F4_E2M1_MAX, F8_E4M3_MAX, from_blocked, roundup, to_blocked

from .base import BaseLayoutParams, QuantizedLayout, dequantize_args, register_layout_op

if TYPE_CHECKING:
    from .base import QuantizedTensor

logger = logging.getLogger(__name__)


class TensorCoreNVFP4Layout(QuantizedLayout):
    """NVFP4 E2M1 block quantization with per-tensor and block scaling.
    Auto-pads to 16x16 alignment

    Note:
        Requires SM >= 10.0 (Blackwell) for hardware-accelerated matmul.
        View-like operations remain limited because NVFP4 uses packed values plus
        blocked scales, but FSDP row-wise alias/slice/split are supported.
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

        def _validate_tensor_fields(self):
            if isinstance(self.scale, torch.Tensor):
                object.__setattr__(
                    self, "scale", self.scale.to(dtype=torch.float32, non_blocking=True)
                )

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

    @classmethod
    def pre_all_gather(cls, qtensor: QuantizedTensor, mesh):
        qdata = qtensor._qdata.contiguous()
        block_scale = qtensor._params.block_scale.contiguous()
        metadata = {
            "scale": qtensor._params.scale,
            "orig_dtype": qtensor._params.orig_dtype,
            "orig_shape": qtensor._params.orig_shape,
            "transposed": qtensor._params.transposed,
            "qdata_shape": tuple(qdata.shape),
        }
        return (qdata, block_scale), metadata

    @classmethod
    def post_all_gather(
        cls,
        qtensor: QuantizedTensor,
        all_gather_outputs: tuple[torch.Tensor, ...],
        metadata,
        param_dtype: torch.dtype,
        *,
        out: QuantizedTensor | None = None,
    ):
        from .base import QuantizedTensor

        gathered_qdata, gathered_block_scale = all_gather_outputs
        orig_shape = _scaled_rowwise_orig_shape(qtensor, gathered_qdata, metadata.get("orig_shape"))
        params = cls.Params(
            scale=metadata["scale"],
            orig_dtype=metadata.get("orig_dtype", param_dtype),
            orig_shape=orig_shape,
            block_scale=gathered_block_scale,
            transposed=metadata.get("transposed", False),
        )

        if out is not None:
            out._qdata = gathered_qdata
            out._params = params
            return out, (gathered_qdata, gathered_block_scale)
        return QuantizedTensor(gathered_qdata, qtensor._layout_cls, params), (
            gathered_qdata,
            gathered_block_scale,
        )


class _CompositeWork:
    def __init__(self, *works):
        self._works = [work for work in works if work is not None]

    def wait(self):
        result = None
        for work in self._works:
            result = work.wait()
        return result

    def is_completed(self):
        return all(getattr(work, "is_completed", lambda: True)() for work in self._works)


def _extract_collective_result(result):
    if isinstance(result, tuple):
        return result[0], result[1]
    return result, None


def _block_scale_unblocked_shape(qtensor) -> tuple[int, int]:
    storage_shape = tuple(qtensor._qdata.shape)
    logical_cols = TensorCoreNVFP4Layout.get_logical_shape_from_storage(storage_shape)[1]
    return storage_shape[0], logical_cols // 16


def _unblock_block_scale(qtensor, block_scale: torch.Tensor | None = None) -> torch.Tensor:
    block_scale = qtensor._params.block_scale if block_scale is None else block_scale
    num_rows, num_cols = _block_scale_unblocked_shape(qtensor)
    return from_blocked(block_scale, num_rows=num_rows, num_cols=num_cols)


def _reblock_block_scale(scale_rows: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return to_blocked(scale_rows.to(dtype=dtype), flatten=False)


def _scaled_rowwise_orig_shape(
    qtensor, new_qdata: torch.Tensor, orig_shape=None
) -> tuple[int, ...]:
    if orig_shape is not None:
        return tuple(orig_shape)
    orig_shape = qtensor._params.orig_shape
    if getattr(qtensor._params, "transposed", False):
        return tuple(orig_shape)
    if len(orig_shape) != 2 or qtensor._qdata.dim() != 2 or new_qdata.dim() != 2:
        return tuple(orig_shape)

    old_logical_rows = int(orig_shape[0])
    old_storage_rows = int(qtensor._qdata.shape[0])
    new_storage_rows = int(new_qdata.shape[0])
    if old_storage_rows == 0:
        new_logical_rows = new_storage_rows
    else:
        new_logical_rows = (
            old_logical_rows * new_storage_rows + old_storage_rows - 1
        ) // old_storage_rows
    return (new_logical_rows, int(orig_shape[1]))


def _wrap_nvfp4_tensor(
    qtensor,
    qdata: torch.Tensor,
    *,
    block_scale: torch.Tensor | None = None,
    orig_shape: tuple[int, ...] | None = None,
    transposed: bool | None = None,
):
    from .base import QuantizedTensor

    new_params = TensorCoreNVFP4Layout.Params(
        scale=qtensor._params.scale,
        orig_dtype=qtensor._params.orig_dtype,
        orig_shape=_scaled_rowwise_orig_shape(qtensor, qdata, orig_shape),
        block_scale=qtensor._params.block_scale if block_scale is None else block_scale,
        transposed=qtensor._params.transposed if transposed is None else transposed,
    )
    return QuantizedTensor(qdata, qtensor._layout_cls, new_params)


def _normalize_slice_args(size: int, start, end, step) -> tuple[int, int, int]:
    step = 1 if step is None else step
    if step != 1:
        raise NotImplementedError("NVFP4 only supports slice step=1")
    start = 0 if start is None else start
    end = size if end is None else end
    if start < 0:
        start += size
    if end < 0:
        end += size
    start = max(0, min(start, size))
    end = max(start, min(end, size))
    return start, end, step


def _logical_rows_to_storage_rows(qtensor, start: int, end: int) -> tuple[int, int]:
    logical_rows = int(qtensor._params.orig_shape[0])
    storage_rows = int(qtensor._qdata.shape[0])
    if logical_rows <= 0:
        return 0, 0
    start_storage = (start * storage_rows) // logical_rows
    end_storage = (end * storage_rows + logical_rows - 1) // logical_rows
    return start_storage, end_storage


def _slice_rows_nvfp4(input_tensor, start, end):
    start_storage, end_storage = _logical_rows_to_storage_rows(input_tensor, start, end)
    sliced_qdata = torch.ops.aten.slice.Tensor(
        input_tensor._qdata, 0, start_storage, end_storage, 1
    )
    block_scale_rows = _unblock_block_scale(input_tensor)
    sliced_block_scale = block_scale_rows[start_storage:end_storage]
    reblocked_scale = _reblock_block_scale(
        sliced_block_scale, input_tensor._params.block_scale.dtype
    )
    new_orig_shape = (end - start, int(input_tensor._params.orig_shape[1]))
    return _wrap_nvfp4_tensor(
        input_tensor, sliced_qdata, block_scale=reblocked_scale, orig_shape=new_orig_shape
    )


# ==================== Distributed Operations ====================


@register_layout_op(
    torch.ops._c10d_functional.all_gather_into_tensor.default, TensorCoreNVFP4Layout
)
def _handle_all_gather(qt, args, kwargs):
    from .base import QuantizedTensor

    input_tensor = None
    input_idx = None
    for i, arg in enumerate(args):
        if isinstance(arg, QuantizedTensor):
            input_tensor = arg
            input_idx = i
            break

    assert input_tensor is not None
    assert input_idx is not None

    qdata_bytes = input_tensor._qdata.contiguous().view(torch.uint8)
    block_scale_bytes = input_tensor._params.block_scale.contiguous().view(torch.uint8)

    q_args = list(args)
    q_args[input_idx] = qdata_bytes
    gathered_qdata_bytes = torch.ops._c10d_functional.all_gather_into_tensor.default(
        *q_args, **kwargs
    )

    b_args = list(args)
    b_args[input_idx] = block_scale_bytes
    gathered_block_scale_bytes = torch.ops._c10d_functional.all_gather_into_tensor.default(
        *b_args, **kwargs
    )

    gathered_qdata = gathered_qdata_bytes.view(input_tensor._qdata.dtype)
    gathered_block_scale = gathered_block_scale_bytes.view(input_tensor._params.block_scale.dtype)
    return _wrap_nvfp4_tensor(input_tensor, gathered_qdata, block_scale=gathered_block_scale)


@register_layout_op(torch.ops._c10d_functional.wait_tensor.default, TensorCoreNVFP4Layout)
def _handle_wait_tensor(qt, args, kwargs):
    qtensor = args[0]

    waited_qdata_bytes = torch.ops._c10d_functional.wait_tensor.default(
        qtensor._qdata.view(torch.uint8),
        *args[1:],
        **kwargs,
    )
    waited_block_scale_bytes = torch.ops._c10d_functional.wait_tensor.default(
        qtensor._params.block_scale.contiguous().view(torch.uint8),
        *args[1:],
        **kwargs,
    )

    waited_qdata = waited_qdata_bytes.view(qtensor._qdata.dtype)
    waited_block_scale = waited_block_scale_bytes.view(qtensor._params.block_scale.dtype)
    return _wrap_nvfp4_tensor(qtensor, waited_qdata, block_scale=waited_block_scale)


@register_layout_op(torch.ops.c10d.broadcast_.default, TensorCoreNVFP4Layout)
def _handle_broadcast(qt, args, kwargs):
    from .base import QuantizedTensor

    tensor_list = args[0]
    quantized_entries = [
        (idx, tensor)
        for idx, tensor in enumerate(tensor_list)
        if isinstance(tensor, QuantizedTensor)
    ]
    if not quantized_entries:
        return torch.ops.c10d.broadcast_.default(*args, **kwargs)

    q_tensor_list = list(tensor_list)
    b_tensor_list = list(tensor_list)
    for idx, tensor in quantized_entries:
        q_tensor_list[idx] = tensor._qdata.contiguous().view(torch.uint8)
        b_tensor_list[idx] = tensor._params.block_scale.contiguous().view(torch.uint8)

    q_result = torch.ops.c10d.broadcast_.default(q_tensor_list, *args[1:], **kwargs)
    b_result = torch.ops.c10d.broadcast_.default(b_tensor_list, *args[1:], **kwargs)
    q_list, q_work = _extract_collective_result(q_result)
    b_list, b_work = _extract_collective_result(b_result)

    output_list = list(q_list)
    for idx, original in quantized_entries:
        output_list[idx] = _wrap_nvfp4_tensor(
            original,
            q_list[idx].view(original._qdata.dtype),
            block_scale=b_list[idx].view(original._params.block_scale.dtype),
            orig_shape=original._params.orig_shape,
        )

    if q_work is not None or b_work is not None:
        return output_list, _CompositeWork(q_work, b_work)
    return output_list


@register_layout_op(torch.ops.c10d.scatter_.default, TensorCoreNVFP4Layout)
def _handle_scatter(qt, args, kwargs):
    from .base import QuantizedTensor

    output_tensors = args[0]
    input_tensors = args[1]

    quantized_outputs = []
    new_q_outputs = list(output_tensors)
    new_b_outputs = list(output_tensors)
    for idx, tensor in enumerate(output_tensors):
        if isinstance(tensor, QuantizedTensor):
            quantized_outputs.append((idx, tensor))
            new_q_outputs[idx] = tensor._qdata.contiguous().view(torch.uint8)
            new_b_outputs[idx] = tensor._params.block_scale.contiguous().view(torch.uint8)

    has_quantized_input = False
    q_inputs = []
    b_inputs = []
    for entry in input_tensors:
        if isinstance(entry, (list, tuple)):
            q_entry = []
            b_entry = []
            for tensor in entry:
                if isinstance(tensor, QuantizedTensor):
                    has_quantized_input = True
                    q_entry.append(tensor._qdata.contiguous().view(torch.uint8))
                    b_entry.append(tensor._params.block_scale.contiguous().view(torch.uint8))
                else:
                    q_entry.append(tensor)
                    b_entry.append(tensor)
            q_inputs.append(q_entry)
            b_inputs.append(b_entry)
        else:
            q_inputs.append(entry)
            b_inputs.append(entry)

    if not quantized_outputs and not has_quantized_input:
        return torch.ops.c10d.scatter_.default(*args, **kwargs)

    q_result = torch.ops.c10d.scatter_.default(new_q_outputs, q_inputs, *args[2:], **kwargs)
    b_result = torch.ops.c10d.scatter_.default(new_b_outputs, b_inputs, *args[2:], **kwargs)
    q_list, q_work = _extract_collective_result(q_result)
    b_list, b_work = _extract_collective_result(b_result)

    output_list = list(q_list)
    for idx, original in quantized_outputs:
        output_list[idx] = _wrap_nvfp4_tensor(
            original,
            q_list[idx].view(original._qdata.dtype),
            block_scale=b_list[idx].view(original._params.block_scale.dtype),
            orig_shape=original._params.orig_shape,
        )

    if q_work is not None or b_work is not None:
        return output_list, _CompositeWork(q_work, b_work)
    return output_list


# ==================== NVFP4 Shape Operations ====================


@register_layout_op(torch.ops.aten.alias.default, TensorCoreNVFP4Layout)
def _handle_nvfp4_alias(qt, args, kwargs):
    from .base import QuantizedTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, QuantizedTensor):
        return torch.ops.aten.alias.default(*args, **kwargs)

    aliased_qdata = torch.ops.aten.alias.default(input_tensor._qdata)
    aliased_block_scale = torch.ops.aten.alias.default(input_tensor._params.block_scale)
    return _wrap_nvfp4_tensor(
        input_tensor,
        aliased_qdata,
        block_scale=aliased_block_scale,
        orig_shape=input_tensor._params.orig_shape,
        transposed=input_tensor._params.transposed,
    )


@register_layout_op(torch.ops.aten.slice.Tensor, TensorCoreNVFP4Layout)
def _handle_nvfp4_slice(qt, args, kwargs):
    from .base import QuantizedTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, QuantizedTensor):
        return torch.ops.aten.slice.Tensor(*args, **kwargs)
    if getattr(input_tensor._params, "transposed", False):
        return torch.ops.aten.slice.Tensor(*dequantize_args(args), **kwargs)

    dim = args[1] if len(args) > 1 else 0
    start = args[2] if len(args) > 2 else None
    end = args[3] if len(args) > 3 else None
    step = args[4] if len(args) > 4 else None
    dim = dim if dim >= 0 else dim + len(input_tensor._params.orig_shape)

    if dim == 0:
        start, end, _ = _normalize_slice_args(
            int(input_tensor._params.orig_shape[0]), start, end, step
        )
        return _slice_rows_nvfp4(input_tensor, start, end)

    if dim == 1:
        start, end, step = _normalize_slice_args(
            int(input_tensor._params.orig_shape[1]), start, end, step
        )
        if start == 0 and end == int(input_tensor._params.orig_shape[1]) and step == 1:
            return _handle_nvfp4_alias(qt, (input_tensor,), {})
        return torch.ops.aten.slice.Tensor(*dequantize_args(args), **kwargs)

    return torch.ops.aten.slice.Tensor(*dequantize_args(args), **kwargs)


@register_layout_op(torch.ops.aten.split.Tensor, TensorCoreNVFP4Layout)
def _handle_nvfp4_split(qt, args, kwargs):
    from .base import QuantizedTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, QuantizedTensor):
        return torch.ops.aten.split.Tensor(*args, **kwargs)
    if getattr(input_tensor._params, "transposed", False):
        return torch.ops.aten.split.Tensor(*dequantize_args(args), **kwargs)

    split_size = args[1]
    dim = kwargs.get("dim", args[2] if len(args) > 2 else 0)
    dim = dim if dim >= 0 else dim + len(input_tensor._params.orig_shape)
    if dim != 0:
        return torch.ops.aten.split.Tensor(*dequantize_args(args), **kwargs)

    logical_rows = int(input_tensor._params.orig_shape[0])
    if isinstance(split_size, int):
        chunks = []
        for start in range(0, logical_rows, split_size):
            end = min(start + split_size, logical_rows)
            chunks.append(_slice_rows_nvfp4(input_tensor, start, end))
        return tuple(chunks)
    return torch.ops.aten.split.Tensor(*dequantize_args(args), **kwargs)


@register_layout_op(torch.ops.aten.cat.default, TensorCoreNVFP4Layout)
def _handle_nvfp4_cat(qt, args, kwargs):
    from .base import QuantizedTensor

    tensors = args[0]
    dim = kwargs.get("dim", args[1] if len(args) > 1 else 0)
    if dim != 0 or not isinstance(tensors, (list, tuple)) or not tensors:
        return torch.ops.aten.cat.default(*dequantize_args(args), **kwargs)
    if not all(isinstance(tensor, QuantizedTensor) for tensor in tensors):
        return torch.ops.aten.cat.default(*dequantize_args(args), **kwargs)

    first = tensors[0]
    if any(getattr(tensor._params, "transposed", False) for tensor in tensors):
        return torch.ops.aten.cat.default(*dequantize_args(args), **kwargs)
    if any(tensor._params.orig_shape[1] != first._params.orig_shape[1] for tensor in tensors):
        return torch.ops.aten.cat.default(*dequantize_args(args), **kwargs)

    qdata = torch.ops.aten.cat.default([tensor._qdata for tensor in tensors], 0)
    block_scale_rows = [_unblock_block_scale(tensor) for tensor in tensors]
    block_scale = _reblock_block_scale(
        torch.ops.aten.cat.default(block_scale_rows, 0), first._params.block_scale.dtype
    )
    orig_rows = sum(int(tensor._params.orig_shape[0]) for tensor in tensors)
    return _wrap_nvfp4_tensor(
        first,
        qdata,
        block_scale=block_scale,
        orig_shape=(orig_rows, int(first._params.orig_shape[1])),
    )


@register_layout_op(torch.ops.aten.new_zeros.default, TensorCoreNVFP4Layout)
def _handle_new_zeros(qt, args, kwargs):
    input_tensor = args[0]
    size = tuple(args[1]) if len(args) > 1 else tuple(input_tensor._params.orig_shape)
    if len(size) != 2:
        return torch.ops.aten.new_zeros.default(*dequantize_args(args), **kwargs)

    device = kwargs.get("device", input_tensor._qdata.device)
    storage_shape = TensorCoreNVFP4Layout.get_storage_shape(size)
    qdata = torch.zeros(storage_shape, device=device, dtype=input_tensor._qdata.dtype)
    block_cols = TensorCoreNVFP4Layout.get_padded_shape(size)[1] // 16
    block_scale_rows = torch.zeros(
        (storage_shape[0], block_cols),
        device=device,
        dtype=input_tensor._params.block_scale.dtype,
    )
    block_scale = _reblock_block_scale(block_scale_rows, input_tensor._params.block_scale.dtype)
    return _wrap_nvfp4_tensor(input_tensor, qdata, block_scale=block_scale, orig_shape=size)


# ==================== NVFP4 Transpose Operation ====================
# Transpose is a no-op that tracks logical transposition via a flag.

@register_layout_op(torch.ops.aten.t.default, TensorCoreNVFP4Layout)
def _handle_nvfp4_transpose(qt, args, kwargs):
    """Handle transpose as a logical no-op for NVFP4."""
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


def _linear_dequantize_fallback(input_tensor, weight, bias):
    input_dense, weight_dense, bias_dense = dequantize_args((input_tensor, weight, bias))
    assert isinstance(input_dense, torch.Tensor)
    assert isinstance(weight_dense, torch.Tensor)
    return torch.nn.functional.linear(input_dense, weight_dense, bias_dense)


@register_layout_op(torch.ops.aten.mm.default, TensorCoreNVFP4Layout)
def _handle_nvfp4_mm(qt, args, kwargs):
    """NVFP4 matrix multiplication: output = a @ b

    When b is logically transposed (from a prior .t() call), this works directly
    with scaled_mm_nvfp4 since that kernel computes a @ b_phys.T, which equals
    a @ b_logical when b_logical = b_phys.T.

    This handles the common torch.compile decomposition: linear(x, w) -> mm(x, w.t())
    """
    from .base import QuantizedTensor

    a, b = args[0], args[1]

    if not (isinstance(a, QuantizedTensor) and isinstance(b, QuantizedTensor)):
        return torch.mm(*dequantize_args(args))

    if a._qdata.dim() != 2:
        return torch.mm(*dequantize_args(args))

    a_transposed = getattr(a._params, "transposed", False)
    b_transposed = getattr(b._params, "transposed", False)

    if a_transposed or not b_transposed:
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

    if not (isinstance(input_tensor, QuantizedTensor) and isinstance(weight, QuantizedTensor)):
        return _linear_dequantize_fallback(input_tensor, weight, bias)

    if input_tensor._qdata.dim() != 2:
        return _linear_dequantize_fallback(input_tensor, weight, bias)

    input_transposed = getattr(input_tensor._params, "transposed", False)
    weight_transposed = getattr(weight._params, "transposed", False)
    if input_transposed or weight_transposed:
        logger.debug(
            "NVFP4 linear: unsupported transpose configuration, falling back to dequantize"
        )
        return _linear_dequantize_fallback(input_tensor, weight, bias)

    input_qdata, scale_a, block_scale_a = TensorCoreNVFP4Layout.get_plain_tensors(input_tensor)
    weight_qdata, scale_b, block_scale_b = TensorCoreNVFP4Layout.get_plain_tensors(weight)
    out_dtype = kwargs.get("out_dtype", input_tensor._params.orig_dtype)

    try:
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

        orig_m = input_tensor._params.orig_shape[0]
        orig_n = weight._params.orig_shape[0]
        return _slice_to_original_shape(result, orig_m, orig_n)

    except (RuntimeError, TypeError) as e:
        logger.warning(f"NVFP4 scaled_mm failed: {e}, falling back to dequantization")
        return _linear_dequantize_fallback(input_tensor, weight, bias)
