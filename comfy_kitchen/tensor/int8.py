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

from comfy_kitchen.exceptions import NoCapableBackendError
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

_INT8_DEQUANT_DTYPE_TO_CODE = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
}


def _dtype_code(dtype: torch.dtype) -> int:
    try:
        return _INT8_DEQUANT_DTYPE_TO_CODE[dtype]
    except KeyError:
        raise ValueError(f"Unsupported INT8 output dtype: {dtype}") from None


def _has_native_fusion(func_name: str, kwargs: dict) -> bool:
    """Whether ``func_name`` would dispatch to a native kernel for ``kwargs``.

    The eager fused ops only re-compose the unfused ops, so without a native
    kernel the caller's unfused path is at least as fast. An explicit backend
    override (e.g. ``use_backend("eager")`` in tests) is honored.
    """
    if getattr(registry._thread_local, "backend_override", None) is not None:
        return True
    try:
        return registry.get_capable_backend(func_name, kwargs) != "eager"
    except NoCapableBackendError:
        return False


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

    @classmethod
    def quantize(
        cls,
        tensor: torch.Tensor,
        scale: torch.Tensor | float | str | None = None,
        stochastic_rounding: int | None = 0,
        inplace_ops: bool = False,
        is_weight: bool = True,
        per_channel: bool = False,
        convrot: bool = False,
        convrot_groupsize: int = 256,
        **kwargs,
    ) -> tuple[torch.Tensor, Params]:
        """Quantize a tensor to INT8 with tensorwise or rowwise scaling.

        Args:
            tensor: Input tensor to quantize.
            scale: Optional tensorwise scale. "recalculate" recomputes from tensor absmax.
            stochastic_rounding: Seed for stochastic rounding. Disabled when <= 0.
            inplace_ops: Kept for ComfyUI compatibility. INT8 quantization does not mutate input.
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

        if convrot:
            impl = registry.get_implementation(
                "quantize_int8_convrot_weight",
                kwargs={"weight": tensor, "group_size": convrot_groupsize, "stochastic_rounding": stochastic_rounding},
            )
            qdata, qscale = impl(tensor, convrot_groupsize, stochastic_rounding=stochastic_rounding)
        elif is_weight:
            if per_channel:
                impl = registry.get_implementation(
                    "quantize_int8_rowwise",
                    kwargs={"x": tensor, "stochastic_rounding": stochastic_rounding},
                )
                qdata, qscale = impl(tensor, stochastic_rounding=stochastic_rounding)
            else:
                impl = registry.get_implementation(
                    "quantize_int8_tensorwise",
                    kwargs={"x": tensor, "scale": scale, "stochastic_rounding": stochastic_rounding},
                )
                qdata, qscale = impl(tensor, scale=scale, stochastic_rounding=stochastic_rounding)
        else:
            impl = registry.get_implementation(
                "quantize_int8_rowwise",
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
        """Dequantize INT8 data back to original dtype.

        Args:
            qdata: Quantized INT8 data.
            params: Layout parameters including scale.

        Returns:
            Dequantized tensor.
        """
        output_dtype_code = _INT8_DEQUANT_DTYPE_TO_CODE.get(params.orig_dtype, 0)
        if getattr(params, "convrot", False):
            result = torch.ops.comfy_kitchen.dequantize_int8_convrot_weight_dtype(
                qdata, params.scale, params.convrot_groupsize, output_dtype_code
            )
        else:
            result = torch.ops.comfy_kitchen.dequantize_int8_simple_dtype(qdata, params.scale, output_dtype_code)
        return result.to(params.orig_dtype)

    @classmethod
    def dequantize_embedding(cls, qdata: torch.Tensor, params: Params, indices: torch.Tensor) -> torch.Tensor:
        """Gather rows from an INT8 embedding table and dequantize only those rows.

        Embedding counterpart of ``dequantize``, which would materialize the whole ``[vocab, dim]``
        table to read a few rows. Un-rotates if ``params.convrot``. Returns ``[*indices.shape, dim]``.
        """
        output_dtype_code = _INT8_DEQUANT_DTYPE_TO_CODE.get(params.orig_dtype, 0)
        group_size = params.convrot_groupsize if getattr(params, "convrot", False) else 0
        result = torch.ops.comfy_kitchen.dequantize_int8_embedding(
            qdata, params.scale, indices, group_size, output_dtype_code
        )
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
    def _fusion_operand(cls, weight: QuantizedTensor):
        """Resolve row-major storage needed by compound INT8 operations."""
        if not isinstance(weight, QuantizedTensor) or weight._layout_cls != cls.__name__:
            return None
        params = weight._params
        if getattr(params, "transposed", False):
            return None
        return (
            weight._qdata.contiguous(),
            params.scale,
            bool(getattr(params, "convrot", False)),
            int(getattr(params, "convrot_groupsize", 256)),
        )

    @classmethod
    def _fusion_operands(cls, weights):
        """Resolve a compatible group of projections or return ``None``."""
        operands = tuple(
            cls._fusion_operand(weight) for weight in weights
        )
        if any(operand is None for operand in operands):
            return None
        qdata, _, convrot, group_size = operands[0]
        if any(
            operand[2:4] != (convrot, group_size)
            or operand[0].shape != qdata.shape
            for operand in operands[1:]
        ):
            return None
        return operands

    @classmethod
    def fused_rms_modulated(
        cls,
        x: torch.Tensor,
        weight: QuantizedTensor,
        bias: torch.Tensor | None,
        norm_weight: torch.Tensor,
        norm_eps: float,
        modulation_scale: torch.Tensor,
    ):
        # The fused kernels apply one modulation row to every token; a batched
        # (e.g. CFG, B > 1) modulation falls back to the unfused composition.
        width = x.shape[-1]
        if modulation_scale.numel() != width or norm_weight.numel() != width:
            return NotImplemented
        operand = cls._fusion_operand(weight)
        if operand is None:
            return NotImplemented
        qdata, scale, convrot, group_size = operand
        if not _has_native_fusion("int8_linear_rms_modulated", {
            "x": x, "norm_weight": norm_weight,
            "modulation_scale": modulation_scale, "weight": qdata,
            "out_dtype": x.dtype,
        }):
            return NotImplemented
        from comfy_kitchen import int8_linear_rms_modulated

        return int8_linear_rms_modulated(
            x, norm_weight, norm_eps, modulation_scale, qdata, scale,
            bias, x.dtype, convrot=convrot,
            convrot_groupsize=group_size,
        )

    @classmethod
    def fused_swiglu_ffn(
        cls,
        x: torch.Tensor,
        gate_weight: QuantizedTensor,
        up_weight: QuantizedTensor,
        down_weight: QuantizedTensor,
        gate_bias: torch.Tensor | None,
        up_bias: torch.Tensor | None,
        down_bias: torch.Tensor | None,
        *,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
        modulation_scale: torch.Tensor | None = None,
    ):
        # Modulation is only fused together with the RMSNorm, and only as one
        # row shared by every token (see fused_rms_modulated).
        width = x.shape[-1]
        if (norm_weight is None) != (modulation_scale is None):
            return NotImplemented
        if norm_weight is not None and (
            norm_weight.numel() != width or modulation_scale.numel() != width
        ):
            return NotImplemented
        pair = cls._fusion_operands((gate_weight, up_weight))
        down = cls._fusion_operand(down_weight)
        if pair is None or down is None:
            return NotImplemented
        first, second = pair
        qgate, gate_scale, convrot, group_size = first
        qup, up_scale, _, _ = second
        pair_op = (
            "int8_linear_pair" if norm_weight is None
            else "int8_linear_pair_rms_modulated"
        )
        if not (
            _has_native_fusion(pair_op, {
                "x": x, "norm_weight": norm_weight,
                "modulation_scale": modulation_scale,
                "weight0": qgate, "weight1": qup, "out_dtype": x.dtype,
            })
            and _has_native_fusion("int8_linear_swiglu_split", {
                "weight": down[0], "out_dtype": x.dtype,
            })
        ):
            return NotImplemented
        if norm_weight is None:
            from comfy_kitchen import int8_linear_pair

            gate, up = int8_linear_pair(
                x, qgate, qup, gate_scale, up_scale, gate_bias, up_bias,
                x.dtype, convrot=convrot, convrot_groupsize=group_size,
            )
        else:
            from comfy_kitchen import int8_linear_pair_rms_modulated

            gate, up = int8_linear_pair_rms_modulated(
                x, norm_weight, norm_eps, modulation_scale,
                qgate, qup, gate_scale, up_scale, gate_bias, up_bias,
                x.dtype, convrot=convrot, convrot_groupsize=group_size,
            )

        qdown, down_scale, down_convrot, down_group = down
        from comfy_kitchen import int8_linear_swiglu_split

        return int8_linear_swiglu_split(
            gate, up, qdown, down_scale, down_bias, gate.dtype,
            convrot=down_convrot, convrot_groupsize=down_group,
        )

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
    def requantize_kwargs(cls, qtensor: QuantizedTensor) -> dict[str, object]:
        """Return INT8 quantization options needed to preserve this layout."""
        params = qtensor._params
        is_weight = getattr(params, "is_weight", True)
        convrot = getattr(params, "convrot", False)
        return {
            "is_weight": is_weight,
            "per_channel": bool(is_weight and (convrot or params.scale.dim() > 0)),
            "convrot": convrot,
            "convrot_groupsize": getattr(params, "convrot_groupsize", 256),
        }

    @classmethod
    def supports_fast_matmul(cls) -> bool:
        """Check if fast INT8 matmul is available."""
        capability = get_cuda_capability()
        if capability is None:
            return False
        return capability >= cls.MIN_SM_VERSION


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

    # If input is already quantized, dequantize it (TensorWise needs dynamic row-wise quant)
    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()

    weight_qdata, weight_scale = TensorWiseINT8Layout.get_plain_tensors(weight)
    out_dtype = kwargs.get("out_dtype", input_tensor.dtype)

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

    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()

    weight_qdata, weight_scale = TensorWiseINT8Layout.get_plain_tensors(weight)
    out_dtype = kwargs.get("out_dtype", input_tensor.dtype)

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

    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()

    weight_qdata, weight_scale = TensorWiseINT8Layout.get_plain_tensors(weight)
    out_dtype = kwargs.get("out_dtype", input_tensor.dtype)

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
