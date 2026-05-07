# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tensor-wise INT8 quantization layout.

This provides a QuantizedTensor layout for tensor-wise INT8 quantization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .base import BaseLayoutParams, QuantizedLayout, dequantize_args, register_layout_op

if TYPE_CHECKING:
    from .base import QuantizedTensor

logger = logging.getLogger(__name__)


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

        def _tensor_fields(self) -> list[str]:
            return ["scale"]

        def _validate_tensor_fields(self):
            pass

    @classmethod
    def quantize(
        cls,
        tensor: torch.Tensor,
        is_weight: bool = True,
        **kwargs,
    ) -> tuple[torch.Tensor, Params]:
        """Quantize a tensor to INT8 with tensorwise or rowwise scaling.

        Args:
            tensor: Input tensor to quantize.
            is_weight: If True, use tensorwise scale. If False, use per-row.
            **kwargs: Additional arguments (ignored).

        Returns:
            Tuple of (quantized_data, params).
        """
        orig_dtype = tensor.dtype
        orig_shape = tuple(tensor.shape)

        if is_weight:
            # Tensorwise: single absmax scale — no triton kernel, eager fast enough.
            from comfy_kitchen.backends.eager.quantization import quantize_int8_tensorwise

            qdata, scale = quantize_int8_tensorwise(tensor)
        else:
            # Rowwise: route through registry (triton -> eager).
            qdata, scale = torch.ops.comfy_kitchen.quantize_int8_rowwise(tensor)

        params = cls.Params(
            scale=scale,
            orig_dtype=orig_dtype,
            orig_shape=orig_shape,
            is_weight=is_weight,
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
        from comfy_kitchen.backends.eager.quantization import dequantize_int8_simple

        result = dequantize_int8_simple(qdata, params.scale)
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


@register_layout_op(torch.ops.aten.linear.default, TensorWiseINT8Layout)
def _handle_int8_linear_tensorwise(qt, args, kwargs):
    """INT8 linear for tensor-wise layout: output = input @ weight.T + bias."""
    from .base import QuantizedTensor, dequantize_args

    input_tensor = args[0]
    weight = args[1]
    bias = args[2] if len(args) > 2 else None

    # Fast path: weight is a TensorWiseINT8Layout QuantizedTensor
    if not isinstance(weight, QuantizedTensor) or weight._layout_cls != "TensorWiseINT8Layout":
        return torch.nn.functional.linear(*dequantize_args(args), **dequantize_args(kwargs))

    weight_qdata, weight_scale = TensorWiseINT8Layout.get_plain_tensors(weight)
    out_dtype = kwargs.get("out_dtype", weight._params.orig_dtype)

    # If input is already quantized, dequantize it (TensorWise needs dynamic row-wise quant)
    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()

    # Try Triton kernel first
    try:
        from comfy_kitchen.backends.triton.quantization import int8_linear

        return int8_linear(
            input_tensor.contiguous(), weight_qdata.contiguous(), weight_scale, bias, out_dtype
        )
    except Exception as e:
        import traceback

        err_msg = (
            f"Triton INT8 scaled_mm failed: {e}\n{traceback.format_exc()}\nfalling back to eager"
        )
        print(err_msg)  # Force print to stdout
        logger.debug(err_msg)

    # Fallback to eager backend
    try:
        from comfy_kitchen.backends.eager.quantization import int8_linear

        return int8_linear(
            input_tensor.contiguous(), weight_qdata.contiguous(), weight_scale, bias, out_dtype
        )
    except Exception as e:
        import traceback

        err_msg = f"Eager INT8 scaled_mm failed: {e}\n{traceback.format_exc()}\nfalling back to dequantization"
        print(err_msg)  # Force print to stdout
        logger.debug(err_msg)

    # Final fallback
    return torch.nn.functional.linear(*dequantize_args((input_tensor, weight, bias)))


@register_layout_op(torch.ops.aten.mm.default, TensorWiseINT8Layout)
def _handle_int8_mm_tensorwise(qt, args, kwargs):
    """INT8 matrix multiplication for tensor-wise layout: output = a @ b."""
    from .base import QuantizedTensor, dequantize_args

    input_tensor = args[0]
    weight = args[1]

    # Usually mm is called with weight as the second argument
    if not isinstance(weight, QuantizedTensor) or weight._layout_cls != "TensorWiseINT8Layout":
        return torch.mm(*dequantize_args(args), **dequantize_args(kwargs))

    weight_qdata, weight_scale = TensorWiseINT8Layout.get_plain_tensors(weight)
    out_dtype = kwargs.get("out_dtype", weight._params.orig_dtype)

    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()

    # mm expects b to NOT be transposed, but our kernels expect (N, K)
    # For mm, weight is (K, N), so we need to transpose it to (N, K)
    try:
        from comfy_kitchen.backends.triton.quantization import int8_linear

        return int8_linear(
            input_tensor, weight_qdata.t().contiguous(), weight_scale, None, out_dtype
        )
    except Exception as e:
        import traceback

        err_msg = f"Triton INT8 mm failed: {e}\n{traceback.format_exc()}\nfalling back to eager"
        print(err_msg)
        logger.debug(err_msg)

    try:
        from comfy_kitchen.backends.eager.quantization import int8_linear

        return int8_linear(
            input_tensor, weight_qdata.t().contiguous(), weight_scale, None, out_dtype
        )
    except Exception as e:
        import traceback

        err_msg = (
            f"Eager INT8 mm failed: {e}\n{traceback.format_exc()}\nfalling back to dequantization"
        )
        print(err_msg)
        logger.debug(err_msg)

    return torch.mm(*dequantize_args(args), **dequantize_args(kwargs))


@register_layout_op(torch.ops.aten.addmm.default, TensorWiseINT8Layout)
def _handle_int8_addmm_tensorwise(qt, args, kwargs):
    """INT8 addmm for tensor-wise layout: output = bias + input @ weight."""
    from .base import QuantizedTensor, dequantize_args

    bias = args[0]
    input_tensor = args[1]
    weight = args[2]

    if not isinstance(weight, QuantizedTensor) or weight._layout_cls != "TensorWiseINT8Layout":
        return torch.addmm(*dequantize_args(args), **dequantize_args(kwargs))

    weight_qdata, weight_scale = TensorWiseINT8Layout.get_plain_tensors(weight)
    out_dtype = kwargs.get("out_dtype", weight._params.orig_dtype)

    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()

    try:
        from comfy_kitchen.backends.triton.quantization import int8_linear

        return int8_linear(
            input_tensor, weight_qdata.t().contiguous(), weight_scale, bias, out_dtype
        )
    except (ImportError, RuntimeError):
        pass

    try:
        from comfy_kitchen.backends.eager.quantization import int8_linear

        return int8_linear(
            input_tensor, weight_qdata.t().contiguous(), weight_scale, bias, out_dtype
        )
    except (ImportError, RuntimeError):
        pass

    return torch.addmm(*dequantize_args(args), **dequantize_args(kwargs))
