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


_HADAMARD_CACHE = {}


def _build_hadamard(
    size: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a normalized REGULAR orthogonal Hadamard matrix (ConvRot).

    Size must be a power of 4 (e.g., 4, 16, 64, 256, 1024...).
    Uses Kronecker construction to avoid the all-1s column of Sylvester Hadamard.
    """
    import math

    cache_key = (size, str(device), dtype)
    if cache_key in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[cache_key]

    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")

    # Base H4 matrix
    H4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=dtype,
        device=device,
    )

    H = H4
    current_size = 4
    while current_size < size:
        H = torch.kron(H, H4)
        current_size *= 4

    H_normalized = H / (size**0.5)
    _HADAMARD_CACHE[cache_key] = H_normalized
    return H_normalized


def _rotate_weight(
    weight: torch.Tensor,
    H: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Rotate weight matrix offline: W_rot = W @ H_block^T."""
    out_f, in_f = weight.shape
    if in_f % group_size != 0:
        raise ValueError(f"in_features {in_f} not divisible by group_size {group_size}")
    n_groups = in_f // group_size

    W_grouped = weight.reshape(out_f, n_groups, group_size)
    H_t = H.T.to(dtype=weight.dtype, device=weight.device)
    W_rot = torch.matmul(W_grouped, H_t)
    return W_rot.reshape(out_f, in_f)


def _rotate_activation(
    x: torch.Tensor,
    H: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Rotate activation online using Fast Hadamard Transform (FHT) — O(N log N)."""
    orig_shape = x.shape
    features = orig_shape[-1]
    if features % group_size != 0:
        raise ValueError(f"features {features} not divisible by group_size {group_size}")
    n_groups = features // group_size

    # Reshape x to (..., n_groups, group_size)
    x_grouped = x.reshape(-1, n_groups, group_size)

    h = 1
    while h < group_size:
        # Reshape to isolate the size-4 dimension for this step
        x_grouped = x_grouped.reshape(-1, group_size // (4 * h), 4, h)

        x0 = x_grouped[:, :, 0, :]
        x1 = x_grouped[:, :, 1, :]
        x2 = x_grouped[:, :, 2, :]
        x3 = x_grouped[:, :, 3, :]

        s01 = x0 + x1
        d01 = x0 - x1
        s23 = x2 + x3
        d23 = x2 - x3

        y0 = s01 + d23
        y1 = s01 - d23
        y2 = d01 + s23
        y3 = s23 - d01

        # Stack back along the size-4 dimension
        x_grouped = torch.stack([y0, y1, y2, y3], dim=2)
        h *= 4

    return x_grouped.reshape(orig_shape) / (group_size ** 0.5)


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

            H = _build_hadamard(convrot_groupsize, device=tensor.device, dtype=tensor.dtype)
            tensor = _rotate_weight(tensor, H, convrot_groupsize)

        if is_weight:
            if per_channel:
                qdata, scale = torch.ops.comfy_kitchen.quantize_int8_rowwise(tensor)
            else:
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
        from comfy_kitchen.backends.eager.quantization import dequantize_int8_simple

        result = dequantize_int8_simple(qdata, params.scale)
        if getattr(params, "convrot", False):
            H = _build_hadamard(params.convrot_groupsize, device=qdata.device, dtype=result.dtype)
            result = _rotate_weight(result, H, params.convrot_groupsize)
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
    import comfy_kitchen as ck

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

    convrot = getattr(weight._params, "convrot", False)
    convrot_groupsize = getattr(weight._params, "convrot_groupsize", 256)

    return ck.int8_linear(
        input_tensor.contiguous(),
        weight_qdata.contiguous(),
        weight_scale,
        bias,
        out_dtype,
        convrot=convrot,
        convrot_groupsize=convrot_groupsize,
    )


@register_layout_op(torch.ops.aten.mm.default, TensorWiseINT8Layout)
def _handle_int8_mm_tensorwise(qt, args, kwargs):
    """INT8 matrix multiplication for tensor-wise layout: output = a @ b."""
    from .base import QuantizedTensor, dequantize_args
    import comfy_kitchen as ck

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

    # mm expects b to NOT be transposed, but our kernels expect (N, K)
    # For mm, weight is (K, N), so we need to transpose it to (N, K)
    return ck.int8_linear(
        input_tensor.contiguous(),
        weight_qdata.t().contiguous(),
        weight_scale,
        None,
        out_dtype,
        convrot=convrot,
        convrot_groupsize=convrot_groupsize,
    )


@register_layout_op(torch.ops.aten.addmm.default, TensorWiseINT8Layout)
def _handle_int8_addmm_tensorwise(qt, args, kwargs):
    """INT8 addmm for tensor-wise layout: output = bias + input @ weight."""
    from .base import QuantizedTensor, dequantize_args
    import comfy_kitchen as ck

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

    return ck.int8_linear(
        input_tensor.contiguous(),
        weight_qdata.t().contiguous(),
        weight_scale,
        bias,
        out_dtype,
        convrot=convrot,
        convrot_groupsize=convrot_groupsize,
    )
