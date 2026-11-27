"""NVFP4 (E2M1) block quantization layout for tensor cores."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

import comfy_kitchen as ck
from comfy_kitchen.float_utils import F4_E2M1_MAX, F8_E4M3_MAX, roundup

from .base import (
    BaseLayoutParams,
    QuantizedLayout,
    QuantizedTensor,
    dequantize_args,
    nvfp4_mm_has_fast_path,
    register_layout_op,
)

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

        def _validate_tensor_fields(self):
            if isinstance(self.scale, torch.Tensor):
                object.__setattr__(self, "scale", self.scale.to(dtype=torch.float32, non_blocking=True))

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
        """Return key suffix -> tensor mapping for serialization."""
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

def _bias_for_kernel(bias: torch.Tensor | None, logical_n: int, padded_n: int) -> torch.Tensor | None:
    """The bias in the length the kernel's epilogue indexes, which is the padded N.

    ``ck.scaled_mm_nvfp4`` derives N from the packed operand's row count, and the
    weight is stored padded to a multiple of 16, so an ``nn.Linear`` whose
    out_features is 20 hands over a 20-element bias for a kernel that will index 32
    of them. ``_bias_operand`` raises ValueError on the mismatch, and ValueError is
    not in these handlers' except clause -- so the call escaped as an exception
    instead of falling back, which is a regression: without the fused handler the
    same call returns the dequantized answer.

    Zero-extending is what makes it correct rather than merely non-crashing. The
    padded columns of the output are discarded by ``_slice_to_original_shape``, so
    whatever sits there cannot reach the result; zero is simply the value that has
    to be a defined one so the epilogue's load stays in bounds.

    The common case, N already a multiple of 16, returns the tensor untouched --
    an ``nn.Linear`` in a quantized checkpoint almost always is, and this runs on
    every forward.

    **The return value has two meanings, and a caller must not collapse them.**
    ``None`` means "no bias" when ``bias`` was None and "this bias is not usable"
    otherwise, and the caller has to be able to tell those apart -- so it must keep
    the result in its own variable and only reject a bias that was not None to begin
    with. Assigning this over the caller's ``bias`` is what made every bias-free
    linear fall back to dequantize-and-matmul, and made the exception fallback hand
    the padded bias to a weight with ``orig_n`` rows. Only this function may return
    None for both reasons; every caller must keep them apart.

    A bias that is not 1-D is rejected here rather than left to ``_bias_operand``,
    which raises ValueError, and ValueError is not in the handlers' except clause --
    so it would escape instead of falling back. ``F.linear`` accepts a ``(1, N)``
    bias, so this is reachable.
    """
    if bias is None:
        return None
    if bias.dim() != 1 or bias.numel() != logical_n:
        # A broadcast addend, a bias for a different operand, or a rank the epilogue
        # cannot index. Not ours, and not something to pass on.
        return None
    return torch.nn.functional.pad(bias, (0, padded_n - logical_n))


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

    This handles the common torch.compile decomposition: linear(x, w) -> mm(x, w.t())
    """
    a, b = args[0], args[1]

    # Fast path: both operands are NVFP4 QuantizedTensors
    if not (isinstance(a, QuantizedTensor) and isinstance(b, QuantizedTensor)):
        return torch.mm(*dequantize_args(args))

    # NVFP4 only supports 2D tensors. Both operands -- `b._params.orig_shape[1]` below
    # is an index into a 2-tuple, so a 1-D b would raise IndexError rather than the
    # RuntimeError eager gives for the same call.
    if a._qdata.dim() != 2 or b._qdata.dim() != 2:
        return torch.mm(*dequantize_args(args))

    # The fused scaled GEMM this dispatches to does not exist on every part torch
    # reports as "cuda": PyTorch's own _scaled_mm is SM100-only for NVFP4, while
    # this fork's HIP backend has a WMMA kernel for it on any part with matrix
    # cores. Decline here rather than let the try/except below catch it: under
    # torch.compile the custom op is traced into the graph, so the exception would
    # escape instead (see nvfp4_mm_has_fast_path).
    if not nvfp4_mm_has_fast_path(a._qdata.device.type):
        logger.debug("NVFP4 mm: no fused scaled GEMM here, falling back to dequantize")
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


@register_layout_op(torch.ops.aten.addmm.default, TensorCoreNVFP4Layout)
def _handle_nvfp4_addmm(qt, args, kwargs):
    """NVFP4 addmm: output = bias + input @ weight, the form linear takes.

    This is not an extra entry point so much as the one that runs. `F.linear` on
    a 2D input with a bias does not dispatch to `aten.linear`: ATen folds it into
    `addmm` whenever it can, so every biased `nn.Linear` -- which is every
    Linear in the model being benchmarked -- arrives here and never reaches
    _handle_nvfp4_linear. Without this handler the NVFP4 fast path is
    unreachable in practice, not merely unused.

    mat2 is the already-transposed weight, as in _handle_nvfp4_mm, and the
    kernel's ``a @ b.T`` is what that means.
    """
    bias, mat1, mat2 = args[0], args[1], args[2]

    # Every fallback carries kwargs through. This op's scale arguments are not in
    # them, but beta/alpha and out_dtype are, and dropping them changes the result
    # rather than just its spelling.
    def fallback():
        return torch.addmm(*dequantize_args(args), **dequantize_args(kwargs))

    if not (isinstance(mat1, QuantizedTensor) and isinstance(mat2, QuantizedTensor)):
        return fallback()
    # Both operands. Checking only mat1 left a 1-D mat2 to reach `mat2.shape[1]`,
    # which is an IndexError -- and eager's answer there is a RuntimeError saying
    # "Expected 2D tensor", so the caller sees the wrong exception type from the
    # wrong library.
    if mat1._qdata.dim() != 2 or mat2._qdata.dim() != 2:
        return fallback()

    # beta/alpha have no place in a scaled GEMM: the kernel applies exactly one
    # product scale and adds the bias once. A broadcast addend rather than a
    # per-column bias is the same kind of request -- the epilogue takes a 1D
    # bias of length N -- and both fall back.
    if kwargs.get("beta", 1) != 1 or kwargs.get("alpha", 1) != 1:
        return fallback()
    if not (isinstance(bias, torch.Tensor) and bias.dim() == 1):
        return fallback()

    # See _handle_nvfp4_mm: the gate has to be a Python decision dynamo can
    # trace, not a caught exception.
    if not nvfp4_mm_has_fast_path(mat1._qdata.device.type):
        logger.debug("NVFP4 addmm: no fused scaled GEMM here, falling back to dequantize")
        return fallback()

    if getattr(mat1._params, "transposed", False) or not getattr(mat2._params, "transposed", False):
        logger.debug("NVFP4 addmm: unsupported transpose configuration, falling back to dequantize")
        return fallback()

    input_qdata, scale_a, block_scale_a = TensorCoreNVFP4Layout.get_plain_tensors(mat1)
    weight_qdata, scale_b, block_scale_b = TensorCoreNVFP4Layout.get_plain_tensors(mat2)
    out_dtype = kwargs.get("out_dtype", mat1._params.orig_dtype)

    # mat2.shape[1] is the unpadded N; the kernel indexes the padded one. As in
    # _handle_nvfp4_linear, `bias` stays the caller's for every fallback and
    # `kernel_bias` is the only thing the kernel sees -- otherwise a bias-free addmm
    # reads as an unusable bias and silently loses the fused path.
    kernel_bias = _bias_for_kernel(bias, mat2.shape[1], weight_qdata.shape[0])
    if bias is not None and kernel_bias is None:
        return fallback()

    try:
        result = ck.scaled_mm_nvfp4(
            input_qdata,
            weight_qdata,
            tensor_scale_a=scale_a,
            tensor_scale_b=scale_b,
            block_scale_a=block_scale_a,
            block_scale_b=block_scale_b,
            bias=kernel_bias,
            out_dtype=out_dtype,
        )
        return _slice_to_original_shape(
            result, mat1._params.orig_shape[0], mat2._params.orig_shape[1]
        )
    except (RuntimeError, TypeError) as e:
        logger.warning(f"NVFP4 addmm failed: {e}, falling back to dequantization")
        return fallback()


@register_layout_op(torch.ops.aten.linear.default, TensorCoreNVFP4Layout)
def _handle_nvfp4_linear(qt, args, kwargs):
    """NVFP4 linear: output = input @ weight.T + bias

    Uses ck.scaled_mm_nvfp4 for hardware-accelerated NVFP4 matmul.
    Output is sliced to original (non-padded) shape.
    """
    input_tensor, weight = args[0], args[1]
    bias = args[2] if len(args) > 2 else None

    # Fast path: both operands are NVFP4 QuantizedTensors
    if not (isinstance(input_tensor, QuantizedTensor) and isinstance(weight, QuantizedTensor)):
        return torch.nn.functional.linear(*dequantize_args((input_tensor, weight, bias)))

    # NVFP4 only supports 2D tensors. Both operands: checking only the input left a
    # 1-D weight to reach an indexing expression that raises IndexError, where
    # eager raises a RuntimeError about the rank.
    if input_tensor._qdata.dim() != 2 or weight._qdata.dim() != 2:
        return torch.nn.functional.linear(*dequantize_args((input_tensor, weight, bias)))

    # See _handle_nvfp4_mm: PyTorch's own NVFP4 GEMM is NVIDIA-only, so the gate has
    # to accept this fork's HIP kernel, and it still has to be a Python decision
    # dynamo can trace rather than a caught exception.
    if not nvfp4_mm_has_fast_path(input_tensor._qdata.device.type):
        logger.debug("NVFP4 linear: no fused scaled GEMM here, falling back to dequantize")
        return torch.nn.functional.linear(*dequantize_args((input_tensor, weight, bias)))

    input_transposed = getattr(input_tensor._params, "transposed", False)
    weight_transposed = getattr(weight._params, "transposed", False)
    if input_transposed or weight_transposed:
        logger.debug("NVFP4 linear: unsupported transpose configuration, falling back to dequantize")
        return torch.nn.functional.linear(*dequantize_args((input_tensor, weight, bias)))

    input_qdata, scale_a, block_scale_a = TensorCoreNVFP4Layout.get_plain_tensors(input_tensor)
    weight_qdata, scale_b, block_scale_b = TensorCoreNVFP4Layout.get_plain_tensors(weight)
    out_dtype = kwargs.get("out_dtype", input_tensor._params.orig_dtype)

    # weight is (out_features, in_features), so its row count is N; the kernel
    # indexes the padded one. Two variables, deliberately: `bias` stays the caller's
    # and every fallback below must use it, while `kernel_bias` is the padded form
    # only the fused kernel accepts. Collapsing them into one is what made a
    # bias-free linear fall back (kernel_bias is None, which read as "unusable"),
    # and made the except path hand the padded bias to a weight with orig_n rows.
    kernel_bias = _bias_for_kernel(bias, weight.shape[0], weight_qdata.shape[0])
    if bias is not None and kernel_bias is None:
        return torch.nn.functional.linear(*dequantize_args((input_tensor, weight, bias)))

    try:
        # scaled_mm_nvfp4 computes (a @ b.T) * scale, which is linear semantics
        result = ck.scaled_mm_nvfp4(
            input_qdata,
            weight_qdata,
            tensor_scale_a=scale_a,
            tensor_scale_b=scale_b,
            block_scale_a=block_scale_a,
            block_scale_b=block_scale_b,
            bias=kernel_bias,
            out_dtype=out_dtype,
        )

        # Slice output to original (non-padded) shape
        orig_m = input_tensor._params.orig_shape[0]
        orig_n = weight._params.orig_shape[0]  # weight is (out_features, in_features)
        return _slice_to_original_shape(result, orig_m, orig_n)

    except (RuntimeError, TypeError) as e:
        logger.warning(f"NVFP4 scaled_mm failed: {e}, falling back to dequantization")
        return torch.nn.functional.linear(*dequantize_args((input_tensor, weight, bias)))
