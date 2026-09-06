# SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Huawei Ascend NPU backend.

The backend is intentionally optional: importing :mod:`comfy_kitchen` must not
require torch-npu on CPU, CUDA, HIP, or XPU installations.
"""

from __future__ import annotations

import torch

from comfy_kitchen.constraints import (
    ExactDims,
    FunctionConstraints,
    MinDims,
    ParamConstraint,
    ValidationResult,
)
from comfy_kitchen.registry import registry

__all__ = [
    "dequantize_int8_simple",
    "dequantize_int8_simple_dtype",
    "quantize_int8_rowwise",
    "quantize_int8_tensorwise",
]

_ASCEND_DEVICE_AVAILABLE = False
_ASCEND_ERROR: str | None = None


def _operator_has_parameter(operator: object, parameter: str) -> bool:
    """Return whether the default torch operator schema contains a parameter."""
    try:
        arguments = operator.default._schema.arguments  # type: ignore[attr-defined]
    except (AttributeError, RuntimeError):
        return False
    return any(argument.name == parameter for argument in arguments)


try:
    import torch_npu

    if not torch.npu.is_available():
        _ASCEND_ERROR = "torch-npu is installed, but no Huawei Ascend device is available"
    else:
        _ASCEND_DEVICE_AVAILABLE = True
except ImportError as exc:
    _ASCEND_ERROR = f"torch-npu is not installed: {exc}"
except Exception as exc:
    _ASCEND_ERROR = f"torch-npu initialization failed: {exc}"


_ASCEND_QUANT_AVAILABLE = (
    _ASCEND_DEVICE_AVAILABLE
    and hasattr(torch_npu, "npu_dynamic_quant")
    and hasattr(torch_npu, "npu_quantize")
    and _operator_has_parameter(torch_npu.npu_quantize, "div_mode")
)
_ASCEND_ROPE_AVAILABLE = (
    _ASCEND_DEVICE_AVAILABLE
    and hasattr(torch_npu, "npu_rotary_mul")
    and _operator_has_parameter(torch_npu.npu_rotary_mul, "rotary_mode")
)
_ASCEND_RMS_ROPE_AVAILABLE = _ASCEND_ROPE_AVAILABLE and hasattr(torch_npu, "npu_rms_norm")

if _ASCEND_ROPE_AVAILABLE:
    from .rope import (
        apply_rope,
        apply_rope1,
        apply_rope1_,
        apply_rope_,
        apply_rope_split_half,
        apply_rope_split_half1,
        apply_rope_split_half1_,
        apply_rope_split_half_,
        validate_apply_rope,
        validate_apply_rope1,
        validate_apply_rope_split_half,
        validate_apply_rope_split_half1,
    )

    __all__ += [
        "apply_rope",
        "apply_rope1",
        "apply_rope1_",
        "apply_rope_",
        "apply_rope_split_half",
        "apply_rope_split_half1",
        "apply_rope_split_half1_",
        "apply_rope_split_half_",
    ]

if _ASCEND_RMS_ROPE_AVAILABLE:
    from .rope import (
        rms_rope,
        rms_rope1,
        rms_rope1_,
        rms_rope_,
        rms_rope_split_half,
        rms_rope_split_half1,
        rms_rope_split_half1_,
        rms_rope_split_half_,
        validate_rms_rope,
        validate_rms_rope1,
        validate_rms_rope_split_half,
        validate_rms_rope_split_half1,
    )

    __all__ += [
        "rms_rope",
        "rms_rope1",
        "rms_rope1_",
        "rms_rope_",
        "rms_rope_split_half",
        "rms_rope_split_half1",
        "rms_rope_split_half1_",
        "rms_rope_split_half_",
    ]


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


def _build_constraints() -> dict[str, FunctionConstraints]:
    ascend_devices = frozenset({"npu"})
    ascend_floats = frozenset({torch.float16, torch.bfloat16})
    scale_values = frozenset({torch.float16, torch.bfloat16, torch.float32, float, int, str})

    constraints = {
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
    }

    if _ASCEND_QUANT_AVAILABLE:
        constraints.update(
            {
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
            }
        )

    rope_tensors = {
        "freqs_cis": ParamConstraint(
            dtypes=frozenset({torch.float16, torch.bfloat16, torch.float32}),
            shape_rules=(ExactDims(6),),
        )
    }
    if _ASCEND_ROPE_AVAILABLE:
        constraints.update(
            {
                "apply_rope1": FunctionConstraints(
                    params={
                        "x": ParamConstraint(dtypes=ascend_floats, shape_rules=(ExactDims(4),)),
                        **rope_tensors,
                    },
                    default_devices=ascend_devices,
                    call_rules=(validate_apply_rope1,),
                ),
                "apply_rope": FunctionConstraints(
                    params={
                        "xq": ParamConstraint(dtypes=ascend_floats, shape_rules=(ExactDims(4),)),
                        "xk": ParamConstraint(dtypes=ascend_floats, shape_rules=(ExactDims(4),)),
                        **rope_tensors,
                    },
                    default_devices=ascend_devices,
                    call_rules=(validate_apply_rope,),
                ),
                "apply_rope_split_half1": FunctionConstraints(
                    params={
                        "x": ParamConstraint(dtypes=ascend_floats, shape_rules=(ExactDims(4),)),
                        **rope_tensors,
                    },
                    default_devices=ascend_devices,
                    call_rules=(validate_apply_rope_split_half1,),
                ),
                "apply_rope_split_half": FunctionConstraints(
                    params={
                        "xq": ParamConstraint(dtypes=ascend_floats, shape_rules=(ExactDims(4),)),
                        "xk": ParamConstraint(dtypes=ascend_floats, shape_rules=(ExactDims(4),)),
                        **rope_tensors,
                    },
                    default_devices=ascend_devices,
                    call_rules=(validate_apply_rope_split_half,),
                ),
            }
        )

    if _ASCEND_RMS_ROPE_AVAILABLE:
        scale_constraint = ParamConstraint(
            dtypes=frozenset({torch.float16, torch.bfloat16, torch.float32}),
            shape_rules=(ExactDims(1),),
        )
        constraints.update(
            {
                "rms_rope1": FunctionConstraints(
                    params={
                        "x": ParamConstraint(dtypes=ascend_floats, shape_rules=(ExactDims(4),)),
                        **rope_tensors,
                        "scale": scale_constraint,
                    },
                    default_devices=ascend_devices,
                    call_rules=(validate_rms_rope1,),
                ),
                "rms_rope": FunctionConstraints(
                    params={
                        "q": ParamConstraint(dtypes=ascend_floats, shape_rules=(ExactDims(4),)),
                        "k": ParamConstraint(dtypes=ascend_floats, shape_rules=(ExactDims(4),)),
                        **rope_tensors,
                        "q_scale": scale_constraint,
                        "k_scale": scale_constraint,
                    },
                    default_devices=ascend_devices,
                    call_rules=(validate_rms_rope,),
                ),
                "rms_rope_split_half1": FunctionConstraints(
                    params={
                        "x": ParamConstraint(dtypes=ascend_floats, shape_rules=(ExactDims(4),)),
                        **rope_tensors,
                        "scale": scale_constraint,
                    },
                    default_devices=ascend_devices,
                    call_rules=(validate_rms_rope_split_half1,),
                ),
                "rms_rope_split_half": FunctionConstraints(
                    params={
                        "q": ParamConstraint(dtypes=ascend_floats, shape_rules=(ExactDims(4),)),
                        "k": ParamConstraint(dtypes=ascend_floats, shape_rules=(ExactDims(4),)),
                        **rope_tensors,
                        "q_scale": scale_constraint,
                        "k_scale": scale_constraint,
                        "rot_dim": ParamConstraint(dtypes=frozenset({int})),
                    },
                    default_devices=ascend_devices,
                    call_rules=(validate_rms_rope_split_half,),
                ),
            }
        )

    for inplace_name, functional_name in {
        "apply_rope_": "apply_rope",
        "apply_rope1_": "apply_rope1",
        "apply_rope_split_half_": "apply_rope_split_half",
        "apply_rope_split_half1_": "apply_rope_split_half1",
        "rms_rope_": "rms_rope",
        "rms_rope1_": "rms_rope1",
        "rms_rope_split_half_": "rms_rope_split_half",
        "rms_rope_split_half1_": "rms_rope_split_half1",
    }.items():
        if functional_name in constraints:
            constraints[inplace_name] = constraints[functional_name]
    return constraints


if _ASCEND_DEVICE_AVAILABLE:
    registry.register(
        name="ascend",
        module=__import__(__name__, fromlist=__all__),
        capabilities=_build_constraints(),
    )
else:
    registry.mark_unavailable("ascend", _ASCEND_ERROR or "Huawei Ascend backend is unavailable")
