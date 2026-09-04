# SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Huawei Ascend NPU backend.

The backend is intentionally optional: importing :mod:`comfy_kitchen` must not
require torch-npu on CPU, CUDA, HIP, or XPU installations.
"""

from __future__ import annotations

import torch

from comfy_kitchen.constraints import (
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

_ASCEND_AVAILABLE = False
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
    }


if _ASCEND_AVAILABLE:
    registry.register(
        name="ascend",
        module=__import__(__name__, fromlist=__all__),
        capabilities=_build_constraints(),
    )
else:
    registry.mark_unavailable("ascend", _ASCEND_ERROR or "Huawei Ascend backend is unavailable")
