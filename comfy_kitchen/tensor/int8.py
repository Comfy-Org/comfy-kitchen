# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tensor-wise INT8 quantization layout.

This provides a QuantizedTensor layout for tensor-wise INT8 quantization.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass

import torch

from .base import (
    BaseLayoutParams,
    QuantizedLayout,
    QuantizedTensor,
    dequantize_args,
    register_layout_op,
)
from .int8_utils import _build_hadamard, _rotate_weight

logger = logging.getLogger(__name__)


def _dtype_code(dtype: torch.dtype) -> int:
    if dtype == torch.float32:
        return 0
    if dtype == torch.float16:
        return 1
    if dtype == torch.bfloat16:
        return 2
    raise ValueError(f"Unsupported INT8 output dtype: {dtype}")


class TensorWiseINT8Layout(QuantizedLayout):
    """Tensor-wise INT8 quantization (from dxqb/OneTrainer).

    Simpler approach than block-wise:
    - Weights: Single scale per tensor
    - Activations: Per-row scales (dynamic quantization)

    Uses torch._int_mm/cuBLASLt IMMA for fast matmul.

    Example:
        >>> w = torch.randn(512, 4096, device="cuda", dtype=torch.bfloat16)
        >>> qt = QuantizedTensor.from_float(w, "TensorWiseINT8Layout")
        >>> qt.shape
        torch.Size([512, 4096])

    Note:
        Requires SM >= 7.5 (Turing) for INT8 tensor core support.
    """

    MIN_SM_VERSION = (7, 5)

    @dataclass(frozen=True)
    class Params(BaseLayoutParams):
        """Tensor-wise INT8 layout parameters.

        Inherits scale, orig_dtype, orig_shape from BaseLayoutParams.
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
        is_weight: bool = True,
        per_channel: bool = False,
        convrot: bool = False,
        convrot_groupsize: int = 256,
        **kwargs,
    ) -> tuple[torch.Tensor, Params]:
        """Quantize a tensor to INT8 with tensorwise or rowwise scaling.

        Args:
            tensor: Input tensor to quantize.
            is_weight: If True, use tensorwise or per-channel scale. If False, use per-row.
            per_channel: If True and is_weight, use per-channel (row-wise) scaling.
            convrot: If True, apply orthogonal group-wise Hadamard rotation to weight.
            convrot_groupsize: Group size for Hadamard rotation.
            **kwargs: Additional arguments (ignored).

        Returns:
            Tuple of (quantized_data, params).
        """
        orig_dtype = tensor.dtype
        orig_shape = tuple(tensor.shape)

        if convrot:
            if not is_weight:
                raise ValueError("convrot is only supported when is_weight is True")
            if not per_channel:
                raise ValueError("convrot is only supported when per_channel is True")

            h = _build_hadamard(convrot_groupsize, device=tensor.device, dtype=tensor.dtype)
            tensor = _rotate_weight(tensor, h, convrot_groupsize)

        if is_weight:
            if per_channel:
                qdata, scale = torch.ops.comfy_kitchen.quantize_int8_rowwise(tensor)
            else:
                # Tensorwise: single absmax scale — no triton kernel, eager fast enough.
                abs_max = tensor.abs().max()
                scale = (abs_max.float() / 127.0).clamp(min=1e-30)
                qdata = (tensor.float() / scale).round().clamp(-128.0, 127.0).to(torch.int8)
        else:
            # Rowwise: route through registry (triton -> eager).
            qdata, scale = torch.ops.comfy_kitchen.quantize_int8_rowwise(tensor)

        params = cls.Params(
            scale=scale,
            orig_dtype=orig_dtype,
            orig_shape=orig_shape,
            is_weight=is_weight,
            convrot=convrot,
            convrot_groupsize=convrot_groupsize,
        )
        return qdata, params

    @classmethod
    def dequantize(cls, qdata: torch.Tensor, params: Params) -> torch.Tensor:
        """Dequantize INT8 data back to original dtype.

        Args:
            qdata: Quantized INT8 data.
            params: Layout parameters including scale.

        Returns:
            Dequantized tensor.
        """
        result = qdata.float() * params.scale
        if getattr(params, "convrot", False):
            h = _build_hadamard(params.convrot_groupsize, device=qdata.device, dtype=result.dtype)
            result = _rotate_weight(result, h, params.convrot_groupsize)
        return result.to(params.orig_dtype)

    @classmethod
    def get_plain_tensors(cls, qtensor: QuantizedTensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract raw tensors for computation.

        Args:
            qtensor: Quantized tensor.

        Returns:
            Tuple of (quantized_data, scale).
        """
        return qtensor._qdata, qtensor._params.scale

    @classmethod
    def state_dict_tensors(cls, qdata: torch.Tensor, params: Params) -> dict[str, torch.Tensor]:
        """Return key suffix → tensor mapping for serialization.

        Args:
            qdata: Quantized data.
            params: Layout parameters.

        Returns:
            Dictionary mapping suffix to tensor.
        """
        return {
            "": qdata,
            "_scale": params.scale,
        }

    @classmethod
    def supports_fast_matmul(cls) -> bool:
        """Check if fast INT8 matmul is available."""
        if not torch.cuda.is_available():
            return False
        sm_major, sm_minor = torch.cuda.get_device_capability()
        return (sm_major, sm_minor) >= cls.MIN_SM_VERSION


# =============================================================================
# INT8 Tensor-wise Operations
# =============================================================================


@register_layout_op(torch.ops.aten.t.default, TensorWiseINT8Layout)
def _handle_int8_transpose(qt, args, kwargs):
    """Handle transpose as a logical flag flip for INT8 tensors."""
    input_tensor = args[0]
    if not isinstance(input_tensor, QuantizedTensor):
        return torch.ops.aten.t.default(*args, **kwargs)

    old = input_tensor._params
    new_params = dataclasses.replace(
        old,
        orig_shape=(old.orig_shape[1], old.orig_shape[0]),
        transposed=not old.transposed,
    )
    return QuantizedTensor(input_tensor._qdata, "TensorWiseINT8Layout", new_params)


@register_layout_op(torch.ops.aten.linear.default, TensorWiseINT8Layout)
def _handle_int8_linear_tensorwise(qt, args, kwargs):
    """INT8 linear for tensor-wise layout: output = input @ weight.T + bias."""
    input_tensor = args[0]
    weight = args[1]
    bias = args[2] if len(args) > 2 else None

    # Fast path: weight is a TensorWiseINT8Layout QuantizedTensor
    if not isinstance(weight, QuantizedTensor) or weight._layout_cls != "TensorWiseINT8Layout":
        return torch.nn.functional.linear(*dequantize_args(args), **dequantize_args(kwargs))
    if getattr(weight._params, "transposed", False):
        return torch.nn.functional.linear(*dequantize_args(args), **dequantize_args(kwargs))

    weight_qdata, weight_scale = TensorWiseINT8Layout.get_plain_tensors(weight)
    out_dtype = kwargs.get("out_dtype", weight._params.orig_dtype)

    # If input is already quantized, dequantize it (TensorWise needs dynamic row-wise quant)
    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()

    convrot = getattr(weight._params, "convrot", False)
    convrot_groupsize = getattr(weight._params, "convrot_groupsize", 256)

    return torch.ops.comfy_kitchen.int8_linear(
        input_tensor.contiguous(),
        weight_qdata.contiguous(),
        weight_scale,
        bias,
        _dtype_code(out_dtype),
        convrot,
        convrot_groupsize,
    )


@register_layout_op(torch.ops.aten.mm.default, TensorWiseINT8Layout)
def _handle_int8_mm_tensorwise(qt, args, kwargs):
    """INT8 matrix multiplication for tensor-wise layout: output = a @ b."""
    input_tensor = args[0]
    weight = args[1]

    # Usually mm is called with weight as the second argument
    if not isinstance(weight, QuantizedTensor) or weight._layout_cls != "TensorWiseINT8Layout":
        return torch.mm(*dequantize_args(args), **dequantize_args(kwargs))

    weight_qdata, weight_scale = TensorWiseINT8Layout.get_plain_tensors(weight)
    out_dtype = kwargs.get("out_dtype", weight._params.orig_dtype)

    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()

    convrot = getattr(weight._params, "convrot", False)
    convrot_groupsize = getattr(weight._params, "convrot_groupsize", 256)

    if getattr(weight._params, "transposed", False):
        # Common decomposition: linear(x, W) -> mm(x, W.t()). Storage is still
        # W [N, K], and logical RHS is W.T [K, N].
        int8_weight = weight_qdata.contiguous()
    elif weight_scale.numel() == 1 and not convrot:
        # A directly quantized RHS [K, N] with a scalar scale can be represented
        # as the [N, K] weight expected by int8_linear.
        int8_weight = weight_qdata.t().contiguous()
    else:
        # Per-row scales belong to the rows of the logical RHS, not output
        # columns, so transposing qdata alone would apply the wrong scales.
        return torch.mm(*dequantize_args(args), **dequantize_args(kwargs))

    return torch.ops.comfy_kitchen.int8_linear(
        input_tensor.contiguous(),
        int8_weight,
        weight_scale,
        None,
        _dtype_code(out_dtype),
        convrot,
        convrot_groupsize,
    )


@register_layout_op(torch.ops.aten.addmm.default, TensorWiseINT8Layout)
def _handle_int8_addmm_tensorwise(qt, args, kwargs):
    """INT8 addmm for tensor-wise layout: output = bias + input @ weight."""
    bias = args[0]
    input_tensor = args[1]
    weight = args[2]

    if not isinstance(weight, QuantizedTensor) or weight._layout_cls != "TensorWiseINT8Layout":
        return torch.addmm(*dequantize_args(args), **dequantize_args(kwargs))

    weight_qdata, weight_scale = TensorWiseINT8Layout.get_plain_tensors(weight)
    out_dtype = kwargs.get("out_dtype", weight._params.orig_dtype)

    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()

    convrot = getattr(weight._params, "convrot", False)
    convrot_groupsize = getattr(weight._params, "convrot_groupsize", 256)

    if getattr(weight._params, "transposed", False):
        int8_weight = weight_qdata.contiguous()
    elif weight_scale.numel() == 1 and not convrot:
        int8_weight = weight_qdata.t().contiguous()
    else:
        return torch.addmm(*dequantize_args(args), **dequantize_args(kwargs))

    return torch.ops.comfy_kitchen.int8_linear(
        input_tensor.contiguous(),
        int8_weight,
        weight_scale,
        bias,
        _dtype_code(out_dtype),
        convrot,
        convrot_groupsize,
    )
