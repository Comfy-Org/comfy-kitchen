# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tensor-wise INT8 quantization layout.

This provides a QuantizedTensor layout for tensor-wise INT8 quantization.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
from dataclasses import dataclass

import torch

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

_WMMA_WEIGHT_PACK_LOCK = threading.RLock()

_INT8_DEQUANT_DTYPE_TO_CODE = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
}


def _dtype_code(dtype: torch.dtype) -> int:
    if dtype == torch.float32:
        return 0
    if dtype == torch.float16:
        return 1
    if dtype == torch.bfloat16:
        return 2
    raise ValueError(f"Unsupported INT8 output dtype: {dtype}")


def _uses_nonduplicated_wmma(device: torch.device) -> bool:
    try:
        arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
    except (AttributeError, RuntimeError):
        return False
    return arch.startswith("gfx12")


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
        # Runtime-only physical storage layout. Zero means checkpoint-compatible
        # row-major [N, K]. Nonzero values mean qdata is packed as
        # [N / tile_n, K / tile_k, tile_n, tile_k] and then flattened back to
        # the original 2-D storage shape. Serialization always unpacks it.
        wmma_tile_n: int = 0
        wmma_tile_k: int = 0

        def _tensor_fields(self) -> list[str]:
            return ["scale"]

        def _validate_tensor_fields(self):
            if (self.wmma_tile_n == 0) != (self.wmma_tile_k == 0):
                raise ValueError(
                    "wmma_tile_n and wmma_tile_k must both be zero or nonzero"
                )

    @staticmethod
    def _unpack_wmma_weight(qdata: torch.Tensor, params: Params) -> torch.Tensor:
        """Return a row-major view/copy of runtime-tiled weight storage."""
        tile_n = getattr(params, "wmma_tile_n", 0)
        tile_k = getattr(params, "wmma_tile_k", 0)
        if tile_n == 0:
            return qdata
        if len(params.orig_shape) != 2 or getattr(params, "transposed", False):
            raise ValueError("WMMA-tiled INT8 storage requires a non-transposed 2-D weight")
        n, k = params.orig_shape
        if n % tile_n or k % tile_k or qdata.numel() != n * k:
            raise ValueError(
                f"Invalid WMMA-tiled INT8 storage: shape={params.orig_shape}, "
                f"tile=({tile_n}, {tile_k}), numel={qdata.numel()}"
            )
        return (
            qdata.reshape(n // tile_n, k // tile_k, tile_n, tile_k)
            .permute(0, 2, 1, 3)
            .contiguous()
            .reshape(n, k)
        )

    @classmethod
    def pack_wmma_weight_(
        cls, qtensor: QuantizedTensor, tile_k: int, tile_n: int = 128
    ) -> QuantizedTensor:
        """Pack one weight in place for static-tile GEMM loaders.

        The wrapper and logical shape remain unchanged. Only its private INT8
        storage is replaced, so the old row-major allocation can be released
        rather than retaining a second multi-gigabyte model copy.
        """
        if not isinstance(qtensor, QuantizedTensor) or qtensor._layout_cls != cls.__name__:
            raise TypeError("pack_wmma_weight_ requires a TensorWiseINT8Layout tensor")
        params = qtensor._params
        if not getattr(params, "is_weight", True):
            raise ValueError("Only INT8 weights can use WMMA-tiled storage")
        if getattr(params, "transposed", False) or len(params.orig_shape) != 2:
            raise ValueError("WMMA-tiled storage requires a non-transposed 2-D weight")
        n, k = params.orig_shape
        if tile_n <= 0 or tile_k <= 0 or n % tile_n or k % tile_k:
            raise ValueError(
                f"Weight shape {(n, k)} is not divisible by tile {(tile_n, tile_k)}"
            )
        with _WMMA_WEIGHT_PACK_LOCK:
            # Another inference thread may have packed this wrapper while the
            # caller was checking eligibility.
            params = qtensor._params
            old_tile_n = getattr(params, "wmma_tile_n", 0)
            old_tile_k = getattr(params, "wmma_tile_k", 0)
            if (old_tile_n, old_tile_k) == (tile_n, tile_k):
                return qtensor
            row_major = cls._unpack_wmma_weight(qtensor._qdata, params)
            packed = (
                row_major.reshape(n // tile_n, tile_n, k // tile_k, tile_k)
                .permute(0, 2, 1, 3)
                .contiguous()
                .reshape_as(qtensor._qdata)
            )
            # Publish the new layout only after its asynchronous device copy is
            # complete. This makes first-use packing safe across inference
            # threads and streams; subsequent calls do not synchronize.
            if packed.device.type == "cuda":
                ready = torch.cuda.Event()
                ready.record(torch.cuda.current_stream(packed.device))
                ready.synchronize()
            if qtensor._qdata.untyped_storage().resizable():
                qtensor._qdata = packed
            else:
                qtensor._qdata.copy_(packed)
            qtensor._params = dataclasses.replace(
                params, wmma_tile_n=tile_n, wmma_tile_k=tile_k
            )
        return qtensor

    @classmethod
    def prepare_wmma_weight_(
        cls, qtensor: QuantizedTensor, input_tensor: torch.Tensor
    ) -> int:
        """Lazily select a measured tile layout for one inference weight.

        The policy is deliberately dimensional. Wide, square, and very
        deep-K projections use a 64-byte K tile; moderately deep-K
        projections use a 128-byte tile. The native storage contract is
        architecture-neutral; automatic selection requires the
        non-duplicated gfx12 WMMA operand layout.

        Returns the physical K tile, or zero when row-major storage is kept.
        """
        if not isinstance(qtensor, QuantizedTensor) or qtensor._layout_cls != cls.__name__:
            return 0
        params = qtensor._params
        existing = getattr(params, "wmma_tile_k", 0)
        if existing:
            return existing
        if getattr(params, "transposed", False) or not getattr(params, "is_weight", True):
            return 0
        if input_tensor.dtype != torch.bfloat16 or input_tensor.device.type != "cuda":
            return 0
        if input_tensor.requires_grad or torch.is_grad_enabled():
            return 0
        try:
            if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
                return 0
        except (AttributeError, RuntimeError):
            return 0
        if (not _uses_nonduplicated_wmma(input_tensor.device)
                or qtensor._qdata.device != input_tensor.device):
            return 0
        if len(params.orig_shape) != 2 or input_tensor.ndim == 0:
            return 0
        n, k = params.orig_shape
        if input_tensor.shape[-1] != k:
            return 0
        # These projection families use row-major kernels. Packing either weight
        # as tiled-B selects an incompatible HIP path.
        if (n, k) in ((3840, 10240), (10240, 3840)):
            return 0
        m = input_tensor.numel() // k
        if m < 512 or n % 128 or k % 256:
            return 0
        tile_k = (
            128
            if (n, k) == (3840, 3840) or n < k < 4 * n
            else 64
        )
        cls.pack_wmma_weight_(qtensor, tile_k=tile_k, tile_n=128)
        return tile_k

    @classmethod
    def wmma_weight_is_supported(
        cls, qtensor: QuantizedTensor, input_tensor: torch.Tensor
    ) -> bool:
        """Whether this call can consume an already tiled physical weight."""
        if not isinstance(qtensor, QuantizedTensor) or qtensor._layout_cls != cls.__name__:
            return False
        params = qtensor._params
        tile_n = getattr(params, "wmma_tile_n", 0)
        tile_k = getattr(params, "wmma_tile_k", 0)
        if tile_n != 128 or tile_k not in (64, 128):
            return False
        if input_tensor.dtype != torch.bfloat16 or input_tensor.device.type != "cuda":
            return False
        if qtensor._qdata.device != input_tensor.device or input_tensor.ndim == 0:
            return False
        if (not _uses_nonduplicated_wmma(input_tensor.device)
                or len(params.orig_shape) != 2):
            return False
        n, k = params.orig_shape
        m = input_tensor.numel() // input_tensor.shape[-1]
        return (
            input_tensor.shape[-1] == k
            and (n, k) not in ((3840, 10240), (10240, 3840))
            and m >= 96
            and n % tile_n == 0
            and k % tile_k == 0
        )

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
        qdata = cls._unpack_wmma_weight(qdata, params)
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
        qdata = cls._unpack_wmma_weight(qdata, params)
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
    def _fusion_operand(
        cls,
        weight: QuantizedTensor,
        x: torch.Tensor,
        *,
        prepare: bool = False,
    ):
        """Resolve private storage needed by compound INT8 operations."""
        if not isinstance(weight, QuantizedTensor) or weight._layout_cls != cls.__name__:
            return None
        params = weight._params
        if getattr(params, "transposed", False):
            return None
        if prepare:
            cls.prepare_wmma_weight_(weight, x)
            params = weight._params
        tile_k = int(getattr(params, "wmma_tile_k", 0))
        if tile_k and not cls.wmma_weight_is_supported(weight, x):
            return None
        qdata, scale = cls.get_plain_tensors(weight)
        return (
            qdata.contiguous(),
            scale,
            bool(getattr(params, "convrot", False)),
            int(getattr(params, "convrot_groupsize", 256)),
            tile_k,
            weight,
        )


    @classmethod
    def _fusion_operands(cls, weights, x: torch.Tensor):
        """Resolve a compatible group of projections or return ``None``."""
        operands = tuple(
            cls._fusion_operand(weight, x, prepare=True) for weight in weights
        )
        if any(operand is None for operand in operands):
            return None
        qdata, _, convrot, group_size, tile_k, _ = operands[0]
        if any(
            operand[2:5] != (convrot, group_size, tile_k)
            or operand[0].shape != qdata.shape
            for operand in operands[1:]
        ):
            return None
        return operands

    @classmethod
    def is_fusion_weight(cls, weight: torch.Tensor) -> bool:
        """Whether ``weight`` has the logical layout compound ops require."""
        return (
            isinstance(weight, QuantizedTensor)
            and weight._layout_cls == cls.__name__
            and not getattr(weight._params, "transposed", False)
        )

    @classmethod
    def fused_linear(
        cls,
        x: torch.Tensor,
        weight: QuantizedTensor,
        bias: torch.Tensor | None,
        *,
        input_act: str | None = None,
        residual: torch.Tensor | None = None,
        gate: torch.Tensor | None = None,
    ):
        """Run a compound linear when this layout can preserve its contract."""
        operand = cls._fusion_operand(weight, x, prepare=True)
        if operand is None:
            return NotImplemented
        qdata, scale, convrot, group_size, tile_k, packed_weight = operand
        source = x
        if input_act == "gelu_tanh":
            source = torch.nn.functional.gelu(x, approximate="tanh")
        elif input_act == "swiglu":
            gate_source, up_source = x.chunk(2, dim=-1)
            source = torch.nn.functional.silu(gate_source).mul_(up_source)
        elif input_act is not None:
            return NotImplemented

        if tile_k:
            if residual is None:
                return torch.nn.functional.linear(source, packed_weight, bias)
            if gate is None:
                return NotImplemented
            from comfy_kitchen import int8_linear_gated_residual

            rows = source.numel() // source.shape[-1]
            return int8_linear_gated_residual(
                source, qdata, scale, residual, gate, bias, x.dtype,
                convrot=convrot, convrot_groupsize=group_size,
                weight_tile_k=tile_k, dual_m=rows >= 512,
            )

        from comfy_kitchen import int8_linear

        output = int8_linear(
            x, qdata, scale, bias, x.dtype, convrot=convrot,
            convrot_groupsize=group_size, input_act=input_act,
        )
        if residual is not None:
            if gate is None:
                return NotImplemented
            output = torch.addcmul(residual, gate, output)
        return output

    @classmethod
    def fused_pair(
        cls,
        x: torch.Tensor,
        weight0: QuantizedTensor,
        weight1: QuantizedTensor,
        bias0: torch.Tensor | None,
        bias1: torch.Tensor | None,
        *,
        modulation_scale: torch.Tensor | None = None,
        modulation_shift: torch.Tensor | None = None,
    ):
        operands = cls._fusion_operands((weight0, weight1), x)
        if operands is None:
            return NotImplemented
        first, second = operands
        qdata0, scale0, convrot, group_size, tile_k, _ = first
        qdata1, scale1, _, _, _, _ = second
        if modulation_scale is None:
            from comfy_kitchen import int8_linear_pair

            return int8_linear_pair(
                x, qdata0, qdata1, scale0, scale1, bias0, bias1, x.dtype,
                convrot=convrot, convrot_groupsize=group_size,
                weight_tile_k=tile_k,
            )
        if modulation_shift is None:
            return NotImplemented
        from comfy_kitchen import int8_linear_pair_modulated

        return int8_linear_pair_modulated(
            x, modulation_scale, qdata0, qdata1, scale0, scale1,
            bias0, bias1, x.dtype, convrot=convrot,
            convrot_groupsize=group_size,
            modulation_shift=modulation_shift, weight_tile_k=tile_k,
        )

    @classmethod
    def fused_affine(
        cls,
        x: torch.Tensor,
        weight: QuantizedTensor,
        bias: torch.Tensor | None,
        modulation_scale: torch.Tensor,
        modulation_shift: torch.Tensor,
    ):
        operand = cls._fusion_operand(weight, x, prepare=True)
        if operand is None:
            return NotImplemented
        qdata, scale, convrot, group_size, tile_k, _ = operand
        from comfy_kitchen import int8_linear_modulated

        return int8_linear_modulated(
            x, modulation_scale, qdata, scale, bias, x.dtype,
            convrot=convrot, convrot_groupsize=group_size,
            modulation_shift=modulation_shift, weight_tile_k=tile_k,
        )

    @classmethod
    def fused_triple_modulated(
        cls,
        x: torch.Tensor,
        weights: tuple[QuantizedTensor, QuantizedTensor, QuantizedTensor],
        biases: tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None],
        modulation_scale: torch.Tensor,
        modulation_shift: torch.Tensor,
    ):
        operands = cls._fusion_operands(weights, x)
        if operands is None:
            return NotImplemented
        qdata0, scale0, convrot, group_size, tile_k, _ = operands[0]
        from comfy_kitchen import int8_linear_triple_modulated

        return int8_linear_triple_modulated(
            x, modulation_scale,
            qdata0, operands[1][0], operands[2][0],
            scale0, operands[1][1], operands[2][1],
            *biases, x.dtype, convrot=convrot,
            convrot_groupsize=group_size,
            modulation_shift=modulation_shift, weight_tile_k=tile_k,
        )

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
        operand = cls._fusion_operand(weight, x, prepare=True)
        if operand is None:
            return NotImplemented
        qdata, scale, convrot, group_size, tile_k, _ = operand
        from comfy_kitchen import int8_linear_rms_modulated

        return int8_linear_rms_modulated(
            x, norm_weight, norm_eps, modulation_scale, qdata, scale,
            bias, x.dtype, convrot=convrot,
            convrot_groupsize=group_size, weight_tiled_b=bool(tile_k),
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
        pair = cls._fusion_operands((gate_weight, up_weight), x)
        if pair is None:
            return NotImplemented
        first, second = pair
        qgate, gate_scale, convrot, group_size, tile_k, _ = first
        qup, up_scale, _, _, _, _ = second
        if norm_weight is None:
            from comfy_kitchen import int8_linear_pair

            gate, up = int8_linear_pair(
                x, qgate, qup, gate_scale, up_scale, gate_bias, up_bias,
                x.dtype, convrot=convrot, convrot_groupsize=group_size,
                weight_tile_k=tile_k,
            )
        else:
            if modulation_scale is None or tile_k:
                return NotImplemented
            from comfy_kitchen import int8_linear_pair_rms_modulated

            gate, up = int8_linear_pair_rms_modulated(
                x, norm_weight, norm_eps, modulation_scale,
                qgate, qup, gate_scale, up_scale, gate_bias, up_bias,
                x.dtype, convrot=convrot, convrot_groupsize=group_size,
            )

        down = cls._fusion_operand(down_weight, gate, prepare=True)
        if down is None:
            return NotImplemented
        qdown, down_scale, down_convrot, down_group, down_tile_k, _ = down
        from comfy_kitchen import int8_linear_swiglu_split

        return int8_linear_swiglu_split(
            gate, up, qdown, down_scale, down_bias, gate.dtype,
            convrot=down_convrot, convrot_groupsize=down_group,
            weight_tiled_b=bool(down_tile_k), weight_tile_k=down_tile_k,
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
            "": cls._unpack_wmma_weight(qdata, params),
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

    if getattr(input_tensor._params, "wmma_tile_n", 0):
        return input_tensor.dequantize().t()

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

    TensorWiseINT8Layout.prepare_wmma_weight_(weight, input_tensor)
    tile_n = getattr(weight._params, "wmma_tile_n", 0)
    tile_k = getattr(weight._params, "wmma_tile_k", 0)
    m = input_tensor.numel() // input_tensor.shape[-1]
    tiled_supported = TensorWiseINT8Layout.wmma_weight_is_supported(
        weight, input_tensor
    )
    if tile_n and not tiled_supported:
        return torch.nn.functional.linear(*dequantize_args(args), **dequantize_args(kwargs))

    weight_qdata, weight_scale = TensorWiseINT8Layout.get_plain_tensors(weight)
    out_dtype = kwargs.get("out_dtype", input_tensor.dtype)

    convrot = getattr(weight._params, "convrot", False)
    convrot_groupsize = getattr(weight._params, "convrot_groupsize", 256)

    op = (
        torch.ops.comfy_kitchen.int8_linear_tiled_b
        if tiled_supported
        else torch.ops.comfy_kitchen.int8_linear
    )
    op_args = [
        input_tensor.contiguous(), weight_qdata.contiguous(), weight_scale,
        bias, _dtype_code(out_dtype), convrot, convrot_groupsize,
    ]
    if tiled_supported:
        op_args.extend((tile_k, m >= 512))
    return op(*op_args)


@register_layout_op(torch.ops.aten.mm.default, TensorWiseINT8Layout)
def _handle_int8_mm_tensorwise(qt, args, kwargs):
    """INT8 matrix multiplication for tensor-wise layout: output = a @ b."""
    input_tensor = args[0]
    weight = args[1]

    # Usually mm is called with weight as the second argument
    if not isinstance(weight, QuantizedTensor) or weight._layout_cls != "TensorWiseINT8Layout":
        return torch.mm(*dequantize_args(args), **dequantize_args(kwargs))
    if getattr(weight._params, "wmma_tile_n", 0):
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
    if getattr(weight._params, "wmma_tile_n", 0):
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
