# SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Huawei Ascend NPU backend.

The backend is intentionally optional: importing :mod:`comfy_kitchen` must not
require torch-npu on CPU, CUDA, HIP, or XPU installations.
"""

from __future__ import annotations

import torch

from comfy_kitchen.backends._activations import (
    apply_input_act as _apply_input_act,
)
from comfy_kitchen.backends._activations import (
    input_act_width as _input_act_width,
)
from comfy_kitchen.constraints import (
    ExactDims,
    FunctionConstraints,
    MinDims,
    ParamConstraint,
    ValidationResult,
)
from comfy_kitchen.registry import registry
from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation

__all__ = [
    "dequantize_int8_simple",
    "dequantize_int8_simple_dtype",
    "quantize_int8_rowwise",
    "quantize_int8_tensorwise",
    "int8_linear",
]

_ASCEND_AVAILABLE = False
_ASCEND_QUANT_MATMUL_AVAILABLE = False
_ASCEND_ERROR: str | None = None

try:
    import torch_npu

    if not torch.npu.is_available():
        _ASCEND_ERROR = "torch-npu is installed, but no Huawei Ascend device is available"
    elif not hasattr(torch_npu, "npu_dynamic_quant"):
        _ASCEND_ERROR = "torch-npu does not provide npu_dynamic_quant"
    elif not hasattr(torch_npu, "npu_quantize"):
        _ASCEND_ERROR = "torch-npu does not provide npu_quantize"
    else:
        _ASCEND_AVAILABLE = True
        _ASCEND_QUANT_MATMUL_AVAILABLE = hasattr(torch_npu, "npu_quant_matmul")
except ImportError as exc:
    _ASCEND_ERROR = f"torch-npu is not installed: {exc}"
except Exception as exc:
    _ASCEND_ERROR = f"torch-npu initialization failed: {exc}"


_DTYPE_CODE_TO_DTYPE = {
    0: torch.float32,
    1: torch.float16,
    2: torch.bfloat16,
}


def _validate_deterministic_quantization(kwargs) -> ValidationResult:
    stochastic_rounding = kwargs.get("stochastic_rounding")
    if stochastic_rounding is not None and stochastic_rounding > 0:
        return ValidationResult.fail(
            "stochastic_rounding", "not supported by the Ascend quantization operators"
        )
    return ValidationResult.ok()


def _validate_tensorwise_scale(kwargs) -> ValidationResult:
    result = _validate_deterministic_quantization(kwargs)
    if not result.success:
        return result

    scale = kwargs.get("scale")
    if isinstance(scale, str) and scale != "recalculate":
        return ValidationResult.fail("scale", "string value must be 'recalculate'")
    if isinstance(scale, torch.Tensor) and scale.numel() != 1:
        return ValidationResult.fail("scale", "must contain exactly one element")
    return ValidationResult.ok()


def _validate_output_dtype(kwargs) -> ValidationResult:
    output_dtype_code = kwargs.get("output_dtype_code")
    if output_dtype_code not in _DTYPE_CODE_TO_DTYPE:
        return ValidationResult.fail(
            "output_dtype_code", "must select float32, float16, or bfloat16"
        )
    return ValidationResult.ok()


def _safe_scale(scale: torch.Tensor) -> torch.Tensor:
    # npu_dynamic_quant returns zero for an all-zero row. Comfy Kitchen's
    # quantization contract uses a small positive scale instead.
    return scale.clamp_min_(1e-30)


def _validate_int8_linear(kwargs) -> ValidationResult:
    x = kwargs.get("x")
    weight = kwargs.get("weight")
    weight_scale = kwargs.get("weight_scale")
    bias = kwargs.get("bias")
    input_act = kwargs.get("input_act")

    if input_act not in (None, "none", "gelu_tanh", "swiglu"):
        return ValidationResult.fail("input_act", f"unsupported value {input_act!r}")
    if not isinstance(x, torch.Tensor) or not isinstance(weight, torch.Tensor):
        return ValidationResult.ok()
    if x.numel() == 0 or weight.numel() == 0:
        return ValidationResult.fail("x", "empty tensors are not supported by npu_quant_matmul")

    input_features = x.shape[-1]
    width = _input_act_width(input_act)
    if input_features % width != 0:
        return ValidationResult.fail(
            "x", f"last dimension {input_features} is not divisible by activation width {width}"
        )
    activated_features = input_features // width
    if activated_features != weight.shape[-1]:
        return ValidationResult.fail(
            "weight",
            f"input features {activated_features} do not match weight features {weight.shape[-1]}",
        )
    if weight_scale is not None and weight_scale.numel() not in (1, weight.shape[0]):
        return ValidationResult.fail(
            "weight_scale",
            f"must be scalar or contain {weight.shape[0]} per-output-channel values",
        )
    if bias is not None and bias.numel() != weight.shape[0]:
        return ValidationResult.fail(
            "bias", f"must contain {weight.shape[0]} output-channel values"
        )

    if kwargs.get("convrot"):
        group_size = kwargs.get("convrot_groupsize", 256)
        is_power_of_four = (
            isinstance(group_size, int)
            and group_size >= 4
            and (group_size & (group_size - 1)) == 0
            and (group_size.bit_length() - 1) % 2 == 0
        )
        if not is_power_of_four:
            return ValidationResult.fail("convrot_groupsize", "must be a positive power of four")
        if activated_features % group_size != 0:
            return ValidationResult.fail(
                "convrot_groupsize",
                f"{group_size} does not divide input features {activated_features}",
            )
    return ValidationResult.ok()


def quantize_int8_tensorwise(
    x: torch.Tensor,
    scale: torch.Tensor | float | str | None = None,
    stochastic_rounding: int | None = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize an Ascend tensor using one float32 scale for the whole tensor."""
    del stochastic_rounding

    if scale is None or (isinstance(scale, str) and scale == "recalculate"):
        abs_max = x.abs().max()
        output_scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    else:
        output_scale = torch.as_tensor(scale, dtype=torch.float32, device=x.device)
    quantized = torch_npu.npu_quantize(
        x,
        output_scale.reshape(1),
        zero_points=None,
        dtype=torch.qint8,
        axis=-1,
        div_mode=True,
    )
    return quantized, output_scale


def quantize_int8_rowwise(
    x: torch.Tensor,
    stochastic_rounding: int | None = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize an Ascend tensor per row using ``npu_dynamic_quant``."""
    del stochastic_rounding
    quantized, scale = torch_npu.npu_dynamic_quant(x)
    return quantized, _safe_scale(scale).unsqueeze(-1)


def dequantize_int8_simple(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize INT8 data on Ascend without transferring it to the host."""
    return q.float() * scale


def dequantize_int8_simple_dtype(
    q: torch.Tensor, scale: torch.Tensor, output_dtype_code: int
) -> torch.Tensor:
    """Dequantize INT8 data to the requested floating-point dtype."""
    return dequantize_int8_simple(q, scale).to(_DTYPE_CODE_TO_DTYPE[output_dtype_code])


def int8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    convrot: bool = False,
    convrot_groupsize: int = 256,
    input_act: str | None = None,
) -> torch.Tensor:
    """Run dynamically quantized INT8 linear on Ascend NPU.

    Comfy Kitchen stores weights as ``[N, K]``. ``npu_quant_matmul`` accepts
    the resulting non-contiguous ``[K, N]`` transpose view directly, avoiding
    a full weight copy on every forward pass.
    """
    orig_shape = x.shape
    x = _apply_input_act(x, input_act)
    if x.shape[-1] != weight.shape[-1]:
        raise ValueError(
            "Input and weight inner dimensions must match, "
            f"got {x.shape[-1]} and {weight.shape[-1]}"
        )

    weight = weight.to(device=x.device).contiguous()
    weight_scale = weight_scale.to(device=x.device, dtype=torch.float32).reshape(-1)
    if weight_scale.numel() not in (1, weight.shape[0]):
        raise ValueError(
            "INT8 weight scale must be scalar or per-output-channel, "
            f"got {tuple(weight_scale.shape)} for weight shape {tuple(weight.shape)}"
        )

    if convrot:
        if x.shape[-1] % convrot_groupsize != 0:
            raise ValueError(
                f"ConvRot group size {convrot_groupsize} does not divide "
                f"input features {x.shape[-1]}"
            )
        hadamard = _build_hadamard(convrot_groupsize, device=x.device, dtype=x.dtype)
        x = _rotate_activation(x, hadamard, convrot_groupsize)

    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    quantized_x, activation_scale = torch_npu.npu_dynamic_quant(x_2d)
    npu_bias = None
    if bias is not None:
        npu_bias = bias.to(device=x.device, dtype=out_dtype).reshape(-1).contiguous()

    result = torch_npu.npu_quant_matmul(
        quantized_x,
        weight.t(),
        weight_scale.contiguous(),
        pertoken_scale=activation_scale.reshape(-1).contiguous(),
        bias=npu_bias,
        output_dtype=out_dtype,
    )
    return result.reshape(*orig_shape[:-1], weight.shape[0])


def _build_constraints() -> dict[str, FunctionConstraints]:
    ascend_devices = frozenset({"npu"})
    ascend_floats = frozenset({torch.float16, torch.bfloat16})
    scale_values = frozenset({torch.float16, torch.bfloat16, torch.float32, float, int, str})

    return {
        "quantize_int8_tensorwise": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=ascend_floats),
                "scale": ParamConstraint(dtypes=scale_values),
                "stochastic_rounding": ParamConstraint(dtypes=frozenset({int})),
            },
            default_devices=ascend_devices,
            call_rules=(_validate_tensorwise_scale,),
        ),
        "quantize_int8_rowwise": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=ascend_floats, shape_rules=(MinDims(2),)),
                "stochastic_rounding": ParamConstraint(dtypes=frozenset({int})),
            },
            default_devices=ascend_devices,
            call_rules=(_validate_deterministic_quantization,),
        ),
        "dequantize_int8_simple": FunctionConstraints(
            params={
                "q": ParamConstraint(dtypes=frozenset({torch.int8})),
                "scale": ParamConstraint(dtypes=frozenset({torch.float32})),
            },
            default_devices=ascend_devices,
        ),
        "dequantize_int8_simple_dtype": FunctionConstraints(
            params={
                "q": ParamConstraint(dtypes=frozenset({torch.int8})),
                "scale": ParamConstraint(dtypes=frozenset({torch.float32})),
                "output_dtype_code": ParamConstraint(dtypes=frozenset({int})),
            },
            default_devices=ascend_devices,
            call_rules=(_validate_output_dtype,),
        ),
        "int8_linear": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=ascend_floats, shape_rules=(MinDims(2),)),
                "weight": ParamConstraint(
                    dtypes=frozenset({torch.int8}), shape_rules=(ExactDims(2),)
                ),
                "weight_scale": ParamConstraint(dtypes=frozenset({torch.float32})),
                "bias": ParamConstraint(
                    dtypes=frozenset({torch.float16, torch.bfloat16, torch.float32})
                ),
                "out_dtype": ParamConstraint(dtypes=ascend_floats),
                "convrot": ParamConstraint(dtypes=frozenset({bool})),
                "convrot_groupsize": ParamConstraint(dtypes=frozenset({int})),
                "input_act": ParamConstraint(dtypes=frozenset({str})),
            },
            default_devices=ascend_devices,
            call_rules=(_validate_int8_linear,),
        ),
    }


if _ASCEND_AVAILABLE:
    capabilities = _build_constraints()
    if not _ASCEND_QUANT_MATMUL_AVAILABLE:
        capabilities.pop("int8_linear")
    registry.register(
        name="ascend",
        module=__import__(__name__, fromlist=__all__),
        capabilities=capabilities,
    )
else:
    registry.mark_unavailable("ascend", _ASCEND_ERROR or "Huawei Ascend backend is unavailable")
