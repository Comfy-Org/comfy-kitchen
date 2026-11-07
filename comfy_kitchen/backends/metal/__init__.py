# SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Apple Metal backend for Comfy Kitchen.

Hardware-sympathetic path:
  - ``int8_linear`` → Apple ``torch._weight_int8pack_mm`` (native MPS GEMM)
  - RoPE / AdaLN / RMS / INT8 quant → custom Metal kernels (simdgroup + float2/float4)

Optional: importing :mod:`comfy_kitchen` must not require MPS off Mac.
"""

from __future__ import annotations

import logging

import torch

from comfy_kitchen.backends._activations import INPUT_ACT_TO_CODE as _INPUT_ACT_TO_CODE
from comfy_kitchen.backends._activations import (
    apply_input_act as _apply_input_act,
)
from comfy_kitchen.backends._activations import (
    apply_residual as _apply_residual,
)
from comfy_kitchen.backends._activations import (
    input_act_width as _input_act_width,
)
from comfy_kitchen._rope_utils import check_rope_inplace, trim_rope_freqs
from comfy_kitchen.backends.eager.convrot_w4a4 import (
    dequantize_convrot_w4a4_weight as _eager_dequantize_convrot_w4a4_weight,
)
from comfy_kitchen.backends.eager.convrot_w4a4 import validate_w4a4_shape
from comfy_kitchen.backends.eager.na import na3d as _eager_na3d
from comfy_kitchen.backends.eager.na import window_attention as _eager_window_attention
from comfy_kitchen.backends.eager.quantization import DTYPE_CODE_TO_DTYPE, DTYPE_TO_CODE
from comfy_kitchen.backends.eager.sol_attn import sol_attn as _eager_sol_attn
from comfy_kitchen.backends.eager.svdquant import (
    _INT4_GROUP_SIZE,
    _ceil_div,
    _tile_packed_lora_up_to_natural,
    _tile_packed_scales_to_natural,
    _tile_packed_weight_to_row_major,
)
from comfy_kitchen.backends.metal.mx_nv_fp import (
    dequantize_mxfp8,
    dequantize_nvfp4,
    quantize_mxfp8,
    quantize_nvfp4,
    scaled_mm_mxfp8,
    scaled_mm_nvfp4,
)
from comfy_kitchen.constraints import (
    ExactDims,
    FunctionConstraints,
    MinDims,
    ParamConstraint,
    ValidationResult,
    na3d_common_call_rule,
    window_attention_common_call_rule,
)
from comfy_kitchen.registry import registry
from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation, _rotate_weight

_logger = logging.getLogger("comfy_kitchen.metal")

__all__ = [
    "int8_linear",
    "apply_rope1",
    "apply_rope",
    "apply_rope1_",
    "apply_rope_",
    "apply_rope_split_half1",
    "apply_rope_split_half",
    "apply_rope_split_half1_",
    "apply_rope_split_half_",
    "rms_rope1",
    "rms_rope",
    "rms_rope_split_half1",
    "rms_rope_split_half",
    "rms_rope1_",
    "rms_rope_",
    "rms_rope_split_half1_",
    "rms_rope_split_half_",
    "adaln",
    "rms_adaln",
    "quantize_int8_rowwise",
    "quantize_int8_tensorwise",
    "dequantize_int8_simple",
    "dequantize_int8_simple_dtype",
    "quantize_per_tensor_fp8",
    "dequantize_per_tensor_fp8",
    "stochastic_rounding_fp8",
    "quantize_svdquant_w4a4",
    "scaled_mm_svdquant_w4a4",
    "na3d",
    "window_attention",
    "quantize_and_rotate_rowwise",
    "quantize_int8_convrot_weight",
    "rotate_int8_convrot_weight",
    "dequantize_int8_convrot_weight",
    "dequantize_int8_convrot_weight_dtype",
    "fp16_linear",
    "rms_gated_residual",
    "quantize_convrot_w4a4_weight",
    "dequantize_convrot_w4a4_weight",
    "convrot_w4a4_linear",
    "quantize_mxfp8",
    "dequantize_mxfp8",
    "scaled_mm_mxfp8",
    "quantize_nvfp4",
    "dequantize_nvfp4",
    "scaled_mm_nvfp4",
    "sol_attn",
    "gemv_awq_w4a16",
]

_METAL_DEVICE_AVAILABLE = False
_METAL_INT8PACK_AVAILABLE = False
_KERNELS = None
_KERNELS_ERROR: str | None = None
_METAL_ERROR: str | None = None

try:
    if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
        _METAL_ERROR = "torch.backends.mps is not available"
    else:
        _METAL_DEVICE_AVAILABLE = True
        _METAL_INT8PACK_AVAILABLE = hasattr(torch, "_weight_int8pack_mm")
        if not _METAL_INT8PACK_AVAILABLE:
            _METAL_ERROR = "torch._weight_int8pack_mm is not available"
except Exception as exc:
    _METAL_ERROR = f"MPS initialization failed: {exc}"


def _kernels():
    global _KERNELS, _KERNELS_ERROR
    if _KERNELS is not None:
        return _KERNELS
    if _KERNELS_ERROR is not None:
        return None
    try:
        from . import _build

        _KERNELS = _build.get_ops()
        if not _KERNELS.metal_available():
            _KERNELS_ERROR = "Metal device unavailable"
            _KERNELS = None
            return None
        return _KERNELS
    except Exception as e:
        _KERNELS_ERROR = str(e)
        _logger.info("metal kernels not loaded: %s", e)
        return None


def _mps_fp(t: torch.Tensor) -> bool:
    return (
        isinstance(t, torch.Tensor)
        and t.device.type == "mps"
        and t.dtype in (torch.float16, torch.bfloat16)
    )


_METAL_INT_MM_AVAILABLE = hasattr(torch, "_int_mm")


def _validate_int8_linear(kwargs) -> ValidationResult:
    x = kwargs.get("x")
    weight = kwargs.get("weight")
    weight_scale = kwargs.get("weight_scale")
    bias = kwargs.get("bias")
    input_act = kwargs.get("input_act")

    if not (_METAL_INT_MM_AVAILABLE or _METAL_INT8PACK_AVAILABLE):
        return ValidationResult.fail("x", "no MPS int8 matmul available")
    if _kernels() is None and not _METAL_INT8PACK_AVAILABLE:
        return ValidationResult.fail("x", "metal kernels unavailable for W8A8")
    if input_act not in _INPUT_ACT_TO_CODE:
        return ValidationResult.fail("input_act", f"unsupported value {input_act!r}")
    if not isinstance(x, torch.Tensor) or not isinstance(weight, torch.Tensor):
        return ValidationResult.ok()
    if x.device.type != "mps" or weight.device.type != "mps":
        return ValidationResult.fail("x", "metal int8_linear requires mps tensors")
    if x.dtype not in (torch.float16, torch.bfloat16):
        return ValidationResult.fail("x", "metal int8_linear requires fp16/bf16 activations")
    if x.numel() == 0 or weight.numel() == 0:
        return ValidationResult.fail("x", "empty tensors are not supported")

    input_features = x.shape[-1]
    if input_act == "rms_norm":
        norm_weight = kwargs.get("input_act_weight")
        if not isinstance(norm_weight, torch.Tensor) or norm_weight.shape != (input_features,):
            return ValidationResult.fail(
                "input_act_weight", "rms_norm requires a one-dimensional input-channel weight"
            )
    if kwargs.get("residual") is not None and kwargs.get("residual_scale") is None:
        return ValidationResult.fail("residual_scale", "required when residual is provided")

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
    # W8A8 via _int_mm has no 32-alignment requirement. The pack_mm fast path
    # still needs N/K % 32, but validation must accept the general W8A8 shapes.
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


def int8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.float16,
    convrot: bool = False,
    convrot_groupsize: int = 256,
    input_act: str | None = None,
    input_act_weight: torch.Tensor | None = None,
    input_act_eps: float = 0.0,
    residual: torch.Tensor | None = None,
    residual_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dynamic W8A8 linear on MPS.

    Prefer Metal row-wise INT8 quant + ``torch._int_mm`` (matches eager). When
    kernels are unavailable but dims are 32-aligned, fall back to Apple
    ``_weight_int8pack_mm`` (W8A16).
    """
    orig_shape = x.shape
    x = _apply_input_act(x, input_act, input_act_weight, input_act_eps)
    if x.shape[-1] != weight.shape[-1]:
        raise ValueError(
            f"Input and weight inner dimensions must match, got {x.shape[-1]} and {weight.shape[-1]}"
        )

    weight = weight.to(device=x.device).contiguous()
    if convrot:
        if x.shape[-1] % convrot_groupsize != 0:
            raise ValueError(
                f"ConvRot group size {convrot_groupsize} does not divide input features {x.shape[-1]}"
            )
        hadamard = _build_hadamard(convrot_groupsize, device=x.device, dtype=x.dtype)
        x = _rotate_activation(x, hadamard, convrot_groupsize)

    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    w_scale = weight_scale.to(device=x.device, dtype=torch.float32).reshape(-1)
    if w_scale.numel() == 1:
        w_scale = w_scale.expand(weight.shape[0])
    elif w_scale.numel() != weight.shape[0]:
        raise ValueError(
            "INT8 weight scale must be scalar or per-output-channel, "
            f"got {tuple(weight_scale.shape)} for weight shape {tuple(weight.shape)}"
        )

    ops = _kernels()
    use_w8a8 = ops is not None and _METAL_INT_MM_AVAILABLE and _mps_fp(x_2d)
    if use_w8a8:
        x_8, x_scale = ops.quantize_int8_rowwise(x_2d)
        x_8 = x_8.to(torch.int8)
        acc = torch._int_mm(x_8, weight.t().contiguous())
        if acc.device.type != "mps":
            raise RuntimeError(
                "metal int8_linear: torch._int_mm left MPS (CPU fallback); "
                "refusing silent host bounce"
            )
        result = (acc.float() * (x_scale.to(torch.float32) * w_scale.reshape(1, -1))).to(out_dtype)
    else:
        if not _METAL_INT8PACK_AVAILABLE:
            raise RuntimeError("metal int8_linear: no int8 matmul path")
        if x_2d.dtype != torch.float16:
            x_2d = x_2d.to(dtype=torch.float16)
        if weight.shape[0] % 32 != 0 or weight.shape[1] % 32 != 0:
            raise RuntimeError("metal pack_mm fallback requires N/K divisible by 32")
        scale = w_scale.to(dtype=torch.float16).contiguous()
        result = torch._weight_int8pack_mm(x_2d, weight, scale)
        if result.dtype != out_dtype:
            result = result.to(dtype=out_dtype)

    if bias is not None:
        result = result + bias.to(device=result.device, dtype=result.dtype).reshape(1, -1)
    result = result.reshape(*orig_shape[:-1], weight.shape[0])
    return _apply_residual(result, residual, residual_scale)


def _freqs_for_metal(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    freqs_cis = trim_rope_freqs(x, freqs_cis)
    if freqs_cis.dtype != torch.float32:
        freqs_cis = freqs_cis.float()
    return freqs_cis


def _metal_rope_ok(x: torch.Tensor) -> bool:
    """Metal RoPE needs 4D even-D with contiguous head_dim (BHND or BNHD)."""
    return x.ndim == 4 and x.shape[-1] % 2 == 0 and x.stride(-1) == 1


def apply_rope1(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    ops = _kernels()
    if ops is None or not _mps_fp(x) or not _metal_rope_ok(x):
        raise RuntimeError("metal apply_rope1 declined")
    y = x.clone()
    ops.apply_rope1_(y, _freqs_for_metal(y, freqs_cis))
    return y


def apply_rope(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    return apply_rope1(xq, freqs_cis), apply_rope1(xk, freqs_cis)


def apply_rope1_(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    ops = _kernels()
    if ops is None or not _mps_fp(x) or not _metal_rope_ok(x):
        raise RuntimeError("metal apply_rope1_ declined")
    check_rope_inplace(x, readonly=(freqs_cis,))
    ops.apply_rope1_(x, _freqs_for_metal(x, freqs_cis))
    return x


def apply_rope_(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    apply_rope1_(xq, freqs_cis)
    apply_rope1_(xk, freqs_cis)
    return xq, xk


def apply_rope_split_half1(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    ops = _kernels()
    if ops is None or not _mps_fp(x) or not _metal_rope_ok(x):
        raise RuntimeError("metal apply_rope_split_half1 declined")
    y = x.clone()
    ops.apply_rope_split_(y, _freqs_for_metal(y, freqs_cis))
    return y


def apply_rope_split_half(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    return apply_rope_split_half1(xq, freqs_cis), apply_rope_split_half1(xk, freqs_cis)


def apply_rope_split_half1_(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    ops = _kernels()
    if ops is None or not _mps_fp(x) or not _metal_rope_ok(x):
        raise RuntimeError("metal apply_rope_split_half1_ declined")
    check_rope_inplace(x, readonly=(freqs_cis,))
    ops.apply_rope_split_(x, _freqs_for_metal(x, freqs_cis))
    return x


def apply_rope_split_half_(xq, xk, freqs_cis):
    apply_rope_split_half1_(xq, freqs_cis)
    apply_rope_split_half1_(xk, freqs_cis)
    return xq, xk


def _metal_rope_layout_ok(x: torch.Tensor, freqs_cis: torch.Tensor, rot_dim: int = 0) -> bool:
    """Accept contiguous BHND or BNHD. Metal kernels index both via strides."""
    del freqs_cis, rot_dim
    return _metal_rope_ok(x) and x.is_contiguous()


def _rms_rope_fused(
    x,
    freqs_cis,
    scale,
    epsilon,
    *,
    split_half: bool,
    rot_dim: int = 0,
    inplace: bool = False,
):
    ops = _kernels()
    if ops is None or not _mps_fp(x) or not _metal_rope_layout_ok(x, freqs_cis, rot_dim):
        raise RuntimeError("metal rms_rope declined")
    if scale.dtype != x.dtype or scale.device != x.device:
        scale = scale.to(device=x.device, dtype=x.dtype)
    if not x.is_contiguous():
        if inplace:
            raise RuntimeError("metal rms_rope inplace requires contiguous x")
        x = x.contiguous()
    return ops.rms_rope_fused(
        x,
        _freqs_for_metal(x, freqs_cis),
        scale.contiguous(),
        float(epsilon),
        bool(split_half),
        int(rot_dim),
        bool(inplace),
    )


def rms_rope1(x, freqs_cis, scale, epsilon: float = 1e-6):
    return _rms_rope_fused(x, freqs_cis, scale, epsilon, split_half=False)


def rms_rope(q, k, freqs_cis, q_scale, k_scale=None, epsilon: float = 1e-6):
    if k_scale is None:
        k_scale = q_scale
    return (
        rms_rope1(q, freqs_cis, q_scale, epsilon),
        rms_rope1(k, freqs_cis, k_scale, epsilon),
    )


def rms_rope_split_half1(x, freqs_cis, scale, epsilon: float = 1e-6, rot_dim: int = 0):
    return _rms_rope_fused(
        x, freqs_cis, scale, epsilon, split_half=True, rot_dim=rot_dim
    )


def rms_rope_split_half(q, k, freqs_cis, q_scale, k_scale=None, epsilon: float = 1e-6, rot_dim: int = 0):
    if k_scale is None:
        k_scale = q_scale
    return (
        rms_rope_split_half1(q, freqs_cis, q_scale, epsilon, rot_dim=rot_dim),
        rms_rope_split_half1(k, freqs_cis, k_scale, epsilon, rot_dim=rot_dim),
    )


def rms_rope1_(x, freqs_cis, scale, epsilon: float = 1e-6):
    check_rope_inplace(x, readonly=(freqs_cis, scale))
    return _rms_rope_fused(x, freqs_cis, scale, epsilon, split_half=False, inplace=True)


def rms_rope_(q, k, freqs_cis, q_scale, k_scale=None, epsilon: float = 1e-6):
    if k_scale is None:
        k_scale = q_scale
    check_rope_inplace(q, k, readonly=(freqs_cis, q_scale, k_scale))
    rms_rope1_(q, freqs_cis, q_scale, epsilon)
    rms_rope1_(k, freqs_cis, k_scale, epsilon)
    return q, k


def rms_rope_split_half1_(x, freqs_cis, scale, epsilon: float = 1e-6, rot_dim: int = 0):
    check_rope_inplace(x, readonly=(freqs_cis, scale))
    return _rms_rope_fused(
        x, freqs_cis, scale, epsilon, split_half=True, rot_dim=rot_dim, inplace=True
    )


def rms_rope_split_half_(q, k, freqs_cis, q_scale, k_scale=None, epsilon: float = 1e-6, rot_dim: int = 0):
    if k_scale is None:
        k_scale = q_scale
    check_rope_inplace(q, k, readonly=(freqs_cis, q_scale, k_scale))
    rms_rope_split_half1_(q, freqs_cis, q_scale, epsilon, rot_dim=rot_dim)
    rms_rope_split_half1_(k, freqs_cis, k_scale, epsilon, rot_dim=rot_dim)
    return q, k


def adaln(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    ops = _kernels()
    if ops is None or not _mps_fp(x):
        raise RuntimeError("metal adaln declined")
    return ops.adaln(x.contiguous(), scale.contiguous(), shift.contiguous(), float(eps), False)


def rms_adaln(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    ops = _kernels()
    if ops is None or not _mps_fp(x):
        raise RuntimeError("metal rms_adaln declined")
    return ops.adaln(x.contiguous(), scale.contiguous(), shift.contiguous(), float(eps), True)


def quantize_int8_rowwise(x: torch.Tensor, stochastic_rounding: int | None = 0):
    ops = _kernels()
    if ops is None or not _mps_fp(x) or (stochastic_rounding is not None and stochastic_rounding > 0):
        raise RuntimeError("metal quantize_int8_rowwise declined")
    q, scale = ops.quantize_int8_rowwise(x.contiguous())
    return q.to(torch.int8), scale


def dequantize_int8_simple(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    ops = _kernels()
    if ops is None or q.device.type != "mps":
        raise RuntimeError("metal dequantize_int8_simple declined")
    return ops.dequantize_int8_simple(q.contiguous(), scale.contiguous())


def quantize_int8_tensorwise(
    x: torch.Tensor,
    scale: torch.Tensor | float | str | None = None,
    stochastic_rounding: int | None = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    ops = _kernels()
    if ops is None or not _mps_fp(x) or (stochastic_rounding is not None and int(stochastic_rounding) > 0):
        raise RuntimeError("metal quantize_int8_tensorwise declined")
    x_c = x.contiguous()
    if scale is None or (isinstance(scale, str) and scale == "recalculate"):
        abs_max = x_c.abs().amax()
        scale_t = (abs_max.float() / 127.0).clamp(min=1e-30)
    elif not isinstance(scale, torch.Tensor):
        scale_t = torch.tensor(scale, device=x.device, dtype=torch.float32)
    else:
        scale_t = scale.to(device=x.device, dtype=torch.float32).reshape(()).contiguous()
    q = ops.quantize_int8_tensorwise(x_c, scale_t)
    return q.to(torch.int8), scale_t


def dequantize_int8_simple_dtype(
    q: torch.Tensor, scale: torch.Tensor, output_dtype_code: int
) -> torch.Tensor:
    ops = _kernels()
    if ops is None or q.device.type != "mps":
        raise RuntimeError("metal dequantize_int8_simple_dtype declined")
    if int(output_dtype_code) not in DTYPE_CODE_TO_DTYPE:
        raise RuntimeError("metal dequantize_int8_simple_dtype unsupported dtype")
    return ops.dequantize_int8_simple_dtype(
        q.contiguous(), scale.contiguous(), int(output_dtype_code)
    )


def stochastic_rounding_fp8(
    x: torch.Tensor,
    rng: torch.Tensor,
    output_type: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    ops = _kernels()
    if ops is None or x.device.type != "mps":
        raise RuntimeError("metal stochastic_rounding_fp8 declined")
    if output_type not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise RuntimeError("metal stochastic_rounding_fp8 unsupported dtype")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise RuntimeError("metal stochastic_rounding_fp8 requires fp16/bf16/fp32")
    if rng.device != x.device or rng.shape != x.shape:
        raise ValueError("rng must match x device and shape")
    if not rng.is_contiguous():
        raise ValueError("rng must be contiguous: the kernel writes the result into it")
    return ops.stochastic_round_fp8(
        x.contiguous(),
        rng,
        int(DTYPE_TO_CODE[output_type]),
    )


def na3d(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: list[int],
    is_causal: list[bool] | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """Neighborhood attention on MPS with batched SDPA (eager g_max path)."""
    if q.device.type != "mps":
        raise RuntimeError("metal na3d requires mps tensors")
    return _eager_na3d(q, k, v, kernel_size, is_causal, scale)


def window_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kh: list[int],
    kw: list[int],
    temporal_window: int,
    clamp: bool,
    scale: float | None = None,
) -> torch.Tensor:
    if q.device.type != "mps":
        raise RuntimeError("metal window_attention requires mps tensors")
    return _eager_window_attention(q, k, v, kh, kw, temporal_window, clamp, scale)


def quantize_and_rotate_rowwise(
    x: torch.Tensor,
    H: torch.Tensor,
    group_size: int,
    stochastic_rounding: int | None = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    ops = _kernels()
    if ops is None or not _mps_fp(x) or (stochastic_rounding is not None and int(stochastic_rounding) > 0):
        raise RuntimeError("metal quantize_and_rotate_rowwise declined")
    if x.shape[-1] % group_size != 0:
        raise RuntimeError("metal quantize_and_rotate_rowwise: group_size must divide features")
    x_rot = _rotate_activation(x.contiguous(), H, group_size)
    q, scale = ops.quantize_int8_rowwise(x_rot.contiguous())
    return q.to(torch.int8), scale


def rotate_int8_convrot_weight(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    if not _mps_fp(weight) or weight.dim() != 2:
        raise RuntimeError("metal rotate_int8_convrot_weight declined")
    if weight.shape[-1] % group_size != 0:
        raise RuntimeError("metal rotate_int8_convrot_weight: group_size must divide K")
    h = _build_hadamard(group_size, device=weight.device, dtype=weight.dtype)
    return _rotate_weight(weight.contiguous(), h, group_size)


def quantize_int8_convrot_weight(
    weight: torch.Tensor,
    group_size: int,
    stochastic_rounding: int | None = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    ops = _kernels()
    if ops is None or not _mps_fp(weight) or (stochastic_rounding is not None and int(stochastic_rounding) > 0):
        raise RuntimeError("metal quantize_int8_convrot_weight declined")
    rotated = rotate_int8_convrot_weight(weight, group_size)
    q, scale = ops.quantize_int8_rowwise(rotated.contiguous())
    return q.to(torch.int8), scale


def dequantize_int8_convrot_weight(
    q: torch.Tensor, scale: torch.Tensor, group_size: int
) -> torch.Tensor:
    ops = _kernels()
    if ops is None or q.device.type != "mps" or q.dim() != 2:
        raise RuntimeError("metal dequantize_int8_convrot_weight declined")
    dequant = ops.dequantize_int8_simple(q.contiguous(), scale.contiguous())
    h = _build_hadamard(group_size, device=q.device, dtype=torch.float32)
    return _rotate_weight(dequant, h, group_size)


def dequantize_int8_convrot_weight_dtype(
    q: torch.Tensor, scale: torch.Tensor, group_size: int, output_dtype_code: int
) -> torch.Tensor:
    out = dequantize_int8_convrot_weight(q, scale, group_size)
    return out.to(DTYPE_CODE_TO_DTYPE[int(output_dtype_code)])


def fp16_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    residual_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    if x.device.type != "mps" or weight.device.type != "mps":
        raise RuntimeError("metal fp16_linear requires mps tensors")
    if x.dtype not in (torch.float16, torch.bfloat16) or weight.dtype != x.dtype:
        raise RuntimeError("metal fp16_linear requires matching fp16/bf16")
    out = torch.nn.functional.linear(x, weight, bias)
    return _apply_residual(out, residual, residual_scale)


def rms_gated_residual(
    activation: torch.Tensor,
    norm_weight: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor,
    eps: float = 1.0e-5,
) -> torch.Tensor:
    ops = _kernels()
    if ops is None or not _mps_fp(activation):
        raise RuntimeError("metal rms_gated_residual declined")
    if activation.shape != residual.shape:
        raise RuntimeError("metal rms_gated_residual requires matching activation/residual shapes")
    if norm_weight.numel() != activation.shape[-1] or gate.numel() != activation.shape[-1]:
        raise RuntimeError("metal rms_gated_residual requires width-sized weight/gate")
    return ops.rms_gated_residual(
        activation.contiguous(),
        norm_weight.contiguous(),
        residual.contiguous(),
        gate.contiguous(),
        float(eps),
    )


def quantize_convrot_w4a4_weight(
    weight: torch.Tensor,
    convrot_groupsize: int = 256,
    quant_group_size: int = _INT4_GROUP_SIZE,
    stochastic_rounding: int | None = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    ops = _kernels()
    if (
        ops is None
        or not _mps_fp(weight)
        or (stochastic_rounding is not None and int(stochastic_rounding) > 0)
    ):
        raise RuntimeError("metal quantize_convrot_w4a4_weight declined")
    if quant_group_size != _INT4_GROUP_SIZE:
        raise RuntimeError("metal convrot w4a4 requires quant_group_size 64")
    validate_w4a4_shape(weight, convrot_groupsize, quant_group_size)
    h = _build_hadamard(convrot_groupsize, device=weight.device, dtype=weight.dtype)
    weight_rot = _rotate_weight(weight.contiguous(), h, convrot_groupsize)
    packed, scales = ops.quantize_int4_rowwise(weight_rot.contiguous())
    return packed.to(torch.int8), scales


def dequantize_convrot_w4a4_weight(
    qdata: torch.Tensor,
    scales: torch.Tensor,
    convrot_groupsize: int = 256,
    quant_group_size: int = _INT4_GROUP_SIZE,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if qdata.device.type != "mps":
        raise RuntimeError("metal dequantize_convrot_w4a4_weight declined")
    return _eager_dequantize_convrot_w4a4_weight(
        qdata, scales, convrot_groupsize, quant_group_size, output_dtype
    )


def convrot_w4a4_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    wscales: torch.Tensor,
    bias: torch.Tensor | None = None,
    convrot_groupsize: int = 256,
    quant_group_size: int = _INT4_GROUP_SIZE,
    linear_dtype: str = "int4",
) -> torch.Tensor:
    ops = _kernels()
    if ops is None or not _mps_fp(x) or qweight.device.type != "mps":
        raise RuntimeError("metal convrot_w4a4_linear declined")
    if linear_dtype != "int4":
        raise RuntimeError("metal convrot_w4a4_linear supports linear_dtype='int4' only")
    if quant_group_size != _INT4_GROUP_SIZE:
        raise RuntimeError("metal convrot w4a4 requires quant_group_size 64")
    if x.shape[-1] != qweight.shape[-1] * 2:
        raise RuntimeError("metal convrot_w4a4_linear K mismatch")
    if x.shape[-1] % convrot_groupsize != 0:
        raise RuntimeError("metal convrot_w4a4_linear: K not divisible by convrot_groupsize")

    orig_shape = x.shape
    x2d = x.reshape(-1, orig_shape[-1]).contiguous()
    h = _build_hadamard(convrot_groupsize, device=x2d.device, dtype=x2d.dtype)
    x_rot = _rotate_activation(x2d, h, convrot_groupsize).contiguous()
    qact, x_scale = ops.quantize_int4_rowwise(x_rot)
    x_fp = ops.dequantize_int4_rowwise(qact, x_scale)
    w_fp = ops.dequantize_int4_rowwise(
        qweight.contiguous(),
        wscales.to(device=x.device, dtype=torch.float32).reshape(-1).contiguous(),
    )
    out = x_fp.to(dtype=x.dtype) @ w_fp.to(dtype=x.dtype).t()
    if bias is not None:
        out = out + bias.to(device=out.device, dtype=out.dtype).reshape(1, -1)
    return out[: x2d.shape[0]].reshape(*orig_shape[:-1], qweight.shape[0])


def sol_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tau: float = 1.0,
    scale: float | None = None,
    sink_blocks: list[int] | None = None,
    sink_q: list[int] | None = None,
    key_bias: torch.Tensor | None = None,
    topk_ratio: float = 0.0,
    tail: bool = True,
    block_len: torch.Tensor | None = None,
    coarse_gate: torch.Tensor | None = None,
    token_aug: int = 0,
) -> torch.Tensor:
    if q.device.type != "mps":
        raise RuntimeError("metal sol_attn requires mps tensors")
    return _eager_sol_attn(
        q, k, v, tau, scale, sink_blocks, sink_q, key_bias,
        topk_ratio, tail, block_len, coarse_gate, token_aug,
    )


def gemv_awq_w4a16(
    x: torch.Tensor,
    qweight: torch.Tensor,
    wscales: torch.Tensor,
    wzeros: torch.Tensor,
    bias: torch.Tensor | None = None,
    group_size: int = 64,
) -> torch.Tensor:
    ops = _kernels()
    if ops is None or not _mps_fp(x) or qweight.device.type != "mps":
        raise RuntimeError("metal gemv_awq_w4a16 requires mps + metallib")
    if x.dtype != torch.float16 or wscales.dtype != torch.float16:
        # Kernel is fp16; cast scales/activations, keep math identical.
        x = x.to(dtype=torch.float16)
        wscales = wscales.to(dtype=torch.float16)
        wzeros = wzeros.to(dtype=torch.float16)
        if bias is not None:
            bias = bias.to(dtype=torch.float16)
    return ops.gemv_awq_w4a16_fused(
        x.contiguous(),
        qweight.contiguous(),
        wscales.contiguous(),
        wzeros.contiguous(),
        bias,
        int(group_size),
    )


def quantize_per_tensor_fp8(
    x: torch.Tensor,
    scale: torch.Tensor,
    output_type: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    ops = _kernels()
    if ops is None or x.device.type != "mps":
        raise RuntimeError("metal quantize_per_tensor_fp8 declined")
    if output_type not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise RuntimeError("metal quantize_per_tensor_fp8 unsupported dtype")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise RuntimeError("metal quantize_per_tensor_fp8 requires fp16/bf16/fp32")
    return ops.quantize_per_tensor_fp8(
        x.contiguous(),
        scale.to(device=x.device, dtype=torch.float32).reshape(()).contiguous(),
        int(DTYPE_TO_CODE[output_type]),
    )


def dequantize_per_tensor_fp8(
    x: torch.Tensor,
    scale: torch.Tensor,
    output_type: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    ops = _kernels()
    if ops is None or x.device.type != "mps":
        raise RuntimeError("metal dequantize_per_tensor_fp8 declined")
    if x.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise RuntimeError("metal dequantize_per_tensor_fp8 requires fp8 input")
    if output_type not in (torch.float16, torch.bfloat16, torch.float32):
        raise RuntimeError("metal dequantize_per_tensor_fp8 unsupported output")
    return ops.dequantize_per_tensor_fp8(
        x.contiguous(),
        scale.to(device=x.device, dtype=torch.float32).reshape(()).contiguous(),
        int(DTYPE_TO_CODE[output_type]),
    )


def quantize_svdquant_w4a4(
    x: torch.Tensor,
    smooth: torch.Tensor,
    lora_down: torch.Tensor,
    pad_size: int = 256,
    act_unsigned: bool = False,
    lora_x: torch.Tensor | None = None,
):
    ops = _kernels()
    if ops is None or not _mps_fp(x):
        raise RuntimeError("metal quantize_svdquant_w4a4 declined")
    if x.dim() != 2 or x.shape[-1] % _INT4_GROUP_SIZE != 0:
        raise RuntimeError("metal svdquant quantize requires 2D x with K % 64 == 0")
    m, k = x.shape
    m_pad = _ceil_div(m, pad_size) * pad_size
    x_c = x.contiguous()
    smooth_c = smooth.to(device=x.device, dtype=x.dtype).contiguous()
    if m_pad > m:
        x_pad = torch.nn.functional.pad(x_c, (0, 0, 0, m_pad - m))
    else:
        x_pad = x_c
    q_packed, ascales = ops.svdquant_quantize_w4a4(x_pad, smooth_c, bool(act_unsigned))
    lora_src = lora_x if lora_x is not None else x
    lora_src = lora_src.to(device=x.device, dtype=x.dtype).contiguous()
    lora_down = lora_down.to(device=x.device, dtype=x.dtype).contiguous()
    lora_act = torch.zeros(m_pad, lora_down.shape[1], device=x.device, dtype=torch.float32)
    if m > 0:
        lora_act[:m] = (lora_src[:m].float() @ lora_down.float())
    return q_packed.to(torch.int8), ascales, lora_act


def scaled_mm_svdquant_w4a4(
    act: torch.Tensor,
    wgt: torch.Tensor,
    ascales: torch.Tensor,
    wscales: torch.Tensor,
    lora_act_in: torch.Tensor,
    lora_up: torch.Tensor,
    bias: torch.Tensor | None = None,
    act_unsigned: bool = False,
) -> torch.Tensor:
    ops = _kernels()
    if ops is None or act.device.type != "mps":
        raise RuntimeError("metal scaled_mm_svdquant_w4a4 declined")
    wgt = _tile_packed_weight_to_row_major(wgt).contiguous()
    wscales = _tile_packed_scales_to_natural(wscales).contiguous()
    lora_up = _tile_packed_lora_up_to_natural(lora_up).contiguous()
    if wscales.dtype != torch.float16:
        wscales = wscales.to(dtype=torch.float16)
    ascales_h = ascales.to(dtype=torch.float16).contiguous()
    # Fused on-the-fly int4 dequant × GEMM (no full fp16 A/W in DRAM).
    out = ops.scaled_mm_svdquant_fused(
        act.contiguous().view(torch.uint8),
        wgt.contiguous().view(torch.uint8),
        ascales_h,
        wscales,
        bool(act_unsigned),
    )
    lora = lora_act_in.to(dtype=torch.float32) @ lora_up.to(dtype=torch.float32).t()
    out = out + lora.to(dtype=out.dtype)
    if bias is not None:
        out = out + bias.to(device=out.device, dtype=out.dtype)
    return out


def _validate_mps_fp_tensor(name: str):
    def _rule(kwargs) -> ValidationResult:
        if _kernels() is None:
            return ValidationResult.fail(name, "metal kernels unavailable")
        t = kwargs.get(name)
        if isinstance(t, torch.Tensor) and not _mps_fp(t):
            return ValidationResult.fail(name, "metal requires mps fp16/bf16")
        return ValidationResult.ok()

    return _rule


def _validate_mps_fp_device(name: str):
    """MPS fp16/bf16 check that does not require the metallib extension."""

    def _rule(kwargs) -> ValidationResult:
        t = kwargs.get(name)
        if isinstance(t, torch.Tensor) and not _mps_fp(t):
            return ValidationResult.fail(name, "metal requires mps fp16/bf16")
        return ValidationResult.ok()

    return _rule


def _validate_quantize_rowwise(kwargs) -> ValidationResult:
    if _kernels() is None:
        return ValidationResult.fail("x", "metal kernels unavailable")
    x = kwargs.get("x")
    if isinstance(x, torch.Tensor) and not _mps_fp(x):
        return ValidationResult.fail("x", "metal requires mps fp16/bf16")
    sr = kwargs.get("stochastic_rounding", 0)
    if sr is not None and int(sr) > 0:
        return ValidationResult.fail("stochastic_rounding", "metal path is deterministic only")
    return ValidationResult.ok()


def _validate_quantize_tensorwise(kwargs) -> ValidationResult:
    return _validate_quantize_rowwise(kwargs)


def _validate_dequant_dtype(kwargs) -> ValidationResult:
    if _kernels() is None:
        return ValidationResult.fail("q", "metal kernels unavailable")
    q = kwargs.get("q")
    if isinstance(q, torch.Tensor) and q.device.type != "mps":
        return ValidationResult.fail("q", "metal requires mps")
    code = kwargs.get("output_dtype_code")
    if code is not None and int(code) not in (0, 1, 2):
        return ValidationResult.fail("output_dtype_code", "metal supports fp32/fp16/bf16 only")
    return ValidationResult.ok()


def _validate_stochastic_fp8(kwargs) -> ValidationResult:
    if _kernels() is None:
        return ValidationResult.fail("x", "metal kernels unavailable")
    x = kwargs.get("x")
    rng = kwargs.get("rng")
    if isinstance(x, torch.Tensor) and x.device.type != "mps":
        return ValidationResult.fail("x", "metal requires mps")
    if isinstance(x, torch.Tensor) and x.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        return ValidationResult.fail("x", "metal requires fp16/bf16/fp32")
    if isinstance(rng, torch.Tensor) and not rng.is_contiguous():
        return ValidationResult.fail("rng", "must be contiguous")
    return ValidationResult.ok()


def _validate_mps_qkv(name: str = "q"):
    def _rule(kwargs) -> ValidationResult:
        t = kwargs.get(name)
        if isinstance(t, torch.Tensor) and t.device.type != "mps":
            return ValidationResult.fail(name, "metal requires mps")
        if isinstance(t, torch.Tensor) and t.dtype not in (torch.float16, torch.bfloat16):
            return ValidationResult.fail(name, "metal requires fp16/bf16")
        return ValidationResult.ok()

    return _rule


def _validate_rms_rope_bhnd(kwargs) -> ValidationResult:
    if _kernels() is None:
        return ValidationResult.fail("x", "metal kernels unavailable")
    x = kwargs.get("x", kwargs.get("q"))
    freqs = kwargs.get("freqs_cis")
    rot_dim = int(kwargs.get("rot_dim", 0) or 0)
    if isinstance(x, torch.Tensor) and isinstance(freqs, torch.Tensor):
        if not _metal_rope_layout_ok(x, freqs, rot_dim):
            return ValidationResult.fail(
                "x", "metal rms_rope needs contiguous 4D even-D BHND or BNHD"
            )
    return ValidationResult.ok()


def _build_constraints() -> dict[str, FunctionConstraints]:
    metal_devices = frozenset({"mps"})
    metal_floats = frozenset({torch.float16, torch.bfloat16})

    constraints: dict[str, FunctionConstraints] = {}

    if _METAL_INT_MM_AVAILABLE or _METAL_INT8PACK_AVAILABLE:
        constraints["int8_linear"] = FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=metal_floats, shape_rules=(MinDims(2),)),
                "weight": ParamConstraint(
                    dtypes=frozenset({torch.int8}), shape_rules=(ExactDims(2),)
                ),
                "weight_scale": ParamConstraint(
                    dtypes=frozenset({torch.float16, torch.bfloat16, torch.float32})
                ),
                "bias": ParamConstraint(
                    dtypes=frozenset({torch.float16, torch.bfloat16, torch.float32})
                ),
                "out_dtype": ParamConstraint(dtypes=frozenset({torch.float16, torch.bfloat16})),
                "convrot": ParamConstraint(dtypes=frozenset({bool})),
                "convrot_groupsize": ParamConstraint(dtypes=frozenset({int})),
                "input_act": ParamConstraint(dtypes=frozenset({str})),
                "input_act_weight": ParamConstraint(
                    dtypes=frozenset({torch.float16, torch.bfloat16, torch.float32})
                ),
                "input_act_eps": ParamConstraint(dtypes=frozenset({float, int})),
                "residual": ParamConstraint(
                    dtypes=frozenset({torch.float16, torch.bfloat16, torch.float32})
                ),
                "residual_scale": ParamConstraint(
                    dtypes=frozenset({torch.float16, torch.bfloat16, torch.float32})
                ),
            },
            default_devices=metal_devices,
            call_rules=(_validate_int8_linear,),
        )

    # Kernel-backed ops: registered only when the metallib extension loads.
    if _kernels() is not None:
        rope_tensors = {
            "freqs_cis": ParamConstraint(
                dtypes=frozenset({torch.float16, torch.bfloat16, torch.float32}),
            )
        }
        constraints.update(
            {
                "apply_rope1": FunctionConstraints(
                    params={
                        "x": ParamConstraint(dtypes=metal_floats, shape_rules=(ExactDims(4),)),
                        **rope_tensors,
                    },
                    default_devices=metal_devices,
                    call_rules=(_validate_mps_fp_tensor("x"),),
                ),
                "apply_rope": FunctionConstraints(
                    params={
                        "xq": ParamConstraint(dtypes=metal_floats, shape_rules=(ExactDims(4),)),
                        "xk": ParamConstraint(dtypes=metal_floats, shape_rules=(ExactDims(4),)),
                        **rope_tensors,
                    },
                    default_devices=metal_devices,
                ),
                "apply_rope_split_half1": FunctionConstraints(
                    params={
                        "x": ParamConstraint(dtypes=metal_floats, shape_rules=(ExactDims(4),)),
                        **rope_tensors,
                    },
                    default_devices=metal_devices,
                ),
                "apply_rope_split_half": FunctionConstraints(
                    params={
                        "xq": ParamConstraint(dtypes=metal_floats, shape_rules=(ExactDims(4),)),
                        "xk": ParamConstraint(dtypes=metal_floats, shape_rules=(ExactDims(4),)),
                        **rope_tensors,
                    },
                    default_devices=metal_devices,
                ),
                "rms_rope1": FunctionConstraints(
                    params={
                        "x": ParamConstraint(dtypes=metal_floats, shape_rules=(ExactDims(4),)),
                        **rope_tensors,
                        "scale": ParamConstraint(dtypes=metal_floats, shape_rules=(ExactDims(1),)),
                    },
                    default_devices=metal_devices,
                    call_rules=(_validate_rms_rope_bhnd,),
                ),
                "rms_rope": FunctionConstraints(
                    params={
                        "q": ParamConstraint(dtypes=metal_floats, shape_rules=(ExactDims(4),)),
                        "k": ParamConstraint(dtypes=metal_floats, shape_rules=(ExactDims(4),)),
                        **rope_tensors,
                        "q_scale": ParamConstraint(dtypes=metal_floats, shape_rules=(ExactDims(1),)),
                        "k_scale": ParamConstraint(dtypes=metal_floats, shape_rules=(ExactDims(1),)),
                    },
                    default_devices=metal_devices,
                    call_rules=(_validate_rms_rope_bhnd,),
                ),
                "rms_rope_split_half1": FunctionConstraints(
                    params={
                        "x": ParamConstraint(dtypes=metal_floats, shape_rules=(ExactDims(4),)),
                        **rope_tensors,
                        "scale": ParamConstraint(dtypes=metal_floats, shape_rules=(ExactDims(1),)),
                    },
                    default_devices=metal_devices,
                    call_rules=(_validate_rms_rope_bhnd,),
                ),
                "rms_rope_split_half": FunctionConstraints(
                    params={
                        "q": ParamConstraint(dtypes=metal_floats, shape_rules=(ExactDims(4),)),
                        "k": ParamConstraint(dtypes=metal_floats, shape_rules=(ExactDims(4),)),
                        **rope_tensors,
                        "q_scale": ParamConstraint(dtypes=metal_floats, shape_rules=(ExactDims(1),)),
                        "k_scale": ParamConstraint(dtypes=metal_floats, shape_rules=(ExactDims(1),)),
                        "rot_dim": ParamConstraint(dtypes=frozenset({int})),
                    },
                    default_devices=metal_devices,
                    call_rules=(_validate_rms_rope_bhnd,),
                ),
                "adaln": FunctionConstraints(
                    params={
                        "x": ParamConstraint(dtypes=metal_floats),
                        "scale": ParamConstraint(dtypes=metal_floats),
                        "shift": ParamConstraint(dtypes=metal_floats),
                    },
                    default_devices=metal_devices,
                ),
                "rms_adaln": FunctionConstraints(
                    params={
                        "x": ParamConstraint(dtypes=metal_floats),
                        "scale": ParamConstraint(dtypes=metal_floats),
                        "shift": ParamConstraint(dtypes=metal_floats),
                    },
                    default_devices=metal_devices,
                ),
                "quantize_int8_rowwise": FunctionConstraints(
                    params={
                        "x": ParamConstraint(dtypes=metal_floats, shape_rules=(MinDims(2),)),
                        "stochastic_rounding": ParamConstraint(dtypes=frozenset({int})),
                    },
                    default_devices=metal_devices,
                    call_rules=(_validate_quantize_rowwise,),
                ),
                "quantize_int8_tensorwise": FunctionConstraints(
                    params={
                        "x": ParamConstraint(dtypes=metal_floats),
                        "scale": ParamConstraint(
                            dtypes=frozenset(
                                {torch.float16, torch.bfloat16, torch.float32, float, str}
                            )
                        ),
                        "stochastic_rounding": ParamConstraint(dtypes=frozenset({int})),
                    },
                    default_devices=metal_devices,
                    call_rules=(_validate_quantize_tensorwise,),
                ),
                "dequantize_int8_simple": FunctionConstraints(
                    params={
                        "q": ParamConstraint(dtypes=frozenset({torch.int8})),
                        "scale": ParamConstraint(dtypes=frozenset({torch.float32})),
                    },
                    default_devices=metal_devices,
                ),
                "dequantize_int8_simple_dtype": FunctionConstraints(
                    params={
                        "q": ParamConstraint(dtypes=frozenset({torch.int8})),
                        "scale": ParamConstraint(
                            dtypes=frozenset({torch.float16, torch.bfloat16, torch.float32})
                        ),
                        "output_dtype_code": ParamConstraint(dtypes=frozenset({int})),
                    },
                    default_devices=metal_devices,
                    call_rules=(_validate_dequant_dtype,),
                ),
                "stochastic_rounding_fp8": FunctionConstraints(
                    params={
                        "x": ParamConstraint(
                            dtypes=frozenset({torch.float16, torch.bfloat16, torch.float32})
                        ),
                        "rng": ParamConstraint(dtypes=frozenset({torch.uint8})),
                        "output_type": ParamConstraint(
                            dtypes=frozenset({torch.float8_e4m3fn, torch.float8_e5m2})
                        ),
                    },
                    default_devices=metal_devices,
                    call_rules=(_validate_stochastic_fp8,),
                ),
                "na3d": FunctionConstraints(
                    params={
                        "q": ParamConstraint(dtypes=metal_floats, shape_rules=(ExactDims(6),)),
                        "k": ParamConstraint(dtypes=metal_floats, shape_rules=(ExactDims(6),)),
                        "v": ParamConstraint(dtypes=metal_floats, shape_rules=(ExactDims(6),)),
                    },
                    default_devices=metal_devices,
                    call_rules=(na3d_common_call_rule, _validate_mps_qkv("q")),
                ),
                "window_attention": FunctionConstraints(
                    params={
                        "q": ParamConstraint(dtypes=metal_floats, shape_rules=(ExactDims(6),)),
                        "k": ParamConstraint(dtypes=metal_floats, shape_rules=(ExactDims(6),)),
                        "v": ParamConstraint(dtypes=metal_floats, shape_rules=(ExactDims(6),)),
                    },
                    default_devices=metal_devices,
                    call_rules=(window_attention_common_call_rule, _validate_mps_qkv("q")),
                ),
                "quantize_and_rotate_rowwise": FunctionConstraints(
                    params={
                        "x": ParamConstraint(dtypes=metal_floats, shape_rules=(MinDims(2),)),
                        "H": ParamConstraint(dtypes=metal_floats, shape_rules=(ExactDims(2),)),
                        "group_size": ParamConstraint(dtypes=frozenset({int})),
                        "stochastic_rounding": ParamConstraint(dtypes=frozenset({int})),
                    },
                    default_devices=metal_devices,
                    call_rules=(_validate_quantize_rowwise,),
                ),
                "quantize_int8_convrot_weight": FunctionConstraints(
                    params={
                        "weight": ParamConstraint(dtypes=metal_floats, shape_rules=(ExactDims(2),)),
                        "group_size": ParamConstraint(dtypes=frozenset({int})),
                        "stochastic_rounding": ParamConstraint(dtypes=frozenset({int})),
                    },
                    default_devices=metal_devices,
                    call_rules=(_validate_mps_fp_tensor("weight"),),
                ),
                "rotate_int8_convrot_weight": FunctionConstraints(
                    params={
                        "weight": ParamConstraint(dtypes=metal_floats, shape_rules=(ExactDims(2),)),
                        "group_size": ParamConstraint(dtypes=frozenset({int})),
                    },
                    default_devices=metal_devices,
                    call_rules=(_validate_mps_fp_tensor("weight"),),
                ),
                "dequantize_int8_convrot_weight": FunctionConstraints(
                    params={
                        "q": ParamConstraint(dtypes=frozenset({torch.int8}), shape_rules=(ExactDims(2),)),
                        "scale": ParamConstraint(
                            dtypes=frozenset({torch.float16, torch.bfloat16, torch.float32})
                        ),
                        "group_size": ParamConstraint(dtypes=frozenset({int})),
                    },
                    default_devices=metal_devices,
                ),
                "dequantize_int8_convrot_weight_dtype": FunctionConstraints(
                    params={
                        "q": ParamConstraint(dtypes=frozenset({torch.int8}), shape_rules=(ExactDims(2),)),
                        "scale": ParamConstraint(
                            dtypes=frozenset({torch.float16, torch.bfloat16, torch.float32})
                        ),
                        "group_size": ParamConstraint(dtypes=frozenset({int})),
                        "output_dtype_code": ParamConstraint(dtypes=frozenset({int})),
                    },
                    default_devices=metal_devices,
                    call_rules=(_validate_dequant_dtype,),
                ),
                "fp16_linear": FunctionConstraints(
                    params={
                        "x": ParamConstraint(dtypes=metal_floats, shape_rules=(MinDims(2),)),
                        "weight": ParamConstraint(dtypes=metal_floats, shape_rules=(ExactDims(2),)),
                        "bias": ParamConstraint(dtypes=metal_floats),
                        "residual": ParamConstraint(dtypes=metal_floats),
                        "residual_scale": ParamConstraint(dtypes=metal_floats),
                    },
                    default_devices=metal_devices,
                    call_rules=(_validate_mps_fp_tensor("x"),),
                ),
                "rms_gated_residual": FunctionConstraints(
                    params={
                        "activation": ParamConstraint(dtypes=metal_floats),
                        "norm_weight": ParamConstraint(dtypes=metal_floats),
                        "residual": ParamConstraint(dtypes=metal_floats),
                        "gate": ParamConstraint(dtypes=metal_floats),
                        "eps": ParamConstraint(dtypes=frozenset({float})),
                    },
                    default_devices=metal_devices,
                    call_rules=(_validate_mps_fp_tensor("activation"),),
                ),
                "quantize_convrot_w4a4_weight": FunctionConstraints(
                    params={
                        "weight": ParamConstraint(dtypes=metal_floats, shape_rules=(ExactDims(2),)),
                        "convrot_groupsize": ParamConstraint(dtypes=frozenset({int})),
                        "quant_group_size": ParamConstraint(dtypes=frozenset({int})),
                        "stochastic_rounding": ParamConstraint(dtypes=frozenset({int})),
                    },
                    default_devices=metal_devices,
                    call_rules=(_validate_mps_fp_tensor("weight"),),
                ),
                "dequantize_convrot_w4a4_weight": FunctionConstraints(
                    params={
                        "qdata": ParamConstraint(
                            dtypes=frozenset({torch.int8}), shape_rules=(ExactDims(2),)
                        ),
                        "scales": ParamConstraint(
                            dtypes=frozenset({torch.float16, torch.bfloat16, torch.float32}),
                            shape_rules=(ExactDims(1),),
                        ),
                        "convrot_groupsize": ParamConstraint(dtypes=frozenset({int})),
                        "quant_group_size": ParamConstraint(dtypes=frozenset({int})),
                        "output_dtype": ParamConstraint(dtypes=metal_floats),
                    },
                    default_devices=metal_devices,
                ),
                "convrot_w4a4_linear": FunctionConstraints(
                    params={
                        "x": ParamConstraint(dtypes=metal_floats),
                        "qweight": ParamConstraint(
                            dtypes=frozenset({torch.int8}), shape_rules=(ExactDims(2),)
                        ),
                        "wscales": ParamConstraint(
                            dtypes=frozenset({torch.float16, torch.bfloat16, torch.float32}),
                            shape_rules=(ExactDims(1),),
                        ),
                        "bias": ParamConstraint(dtypes=metal_floats),
                        "convrot_groupsize": ParamConstraint(dtypes=frozenset({int})),
                        "quant_group_size": ParamConstraint(dtypes=frozenset({int})),
                        "linear_dtype": ParamConstraint(dtypes=frozenset({str})),
                    },
                    default_devices=metal_devices,
                    call_rules=(_validate_mps_fp_tensor("x"),),
                ),
                "quantize_per_tensor_fp8": FunctionConstraints(
                    params={
                        "x": ParamConstraint(
                            dtypes=frozenset({torch.float16, torch.bfloat16, torch.float32})
                        ),
                        "scale": ParamConstraint(dtypes=frozenset({torch.float32})),
                        "output_type": ParamConstraint(
                            dtypes=frozenset({torch.float8_e4m3fn, torch.float8_e5m2})
                        ),
                    },
                    default_devices=metal_devices,
                ),
                "dequantize_per_tensor_fp8": FunctionConstraints(
                    params={
                        "x": ParamConstraint(
                            dtypes=frozenset({torch.float8_e4m3fn, torch.float8_e5m2})
                        ),
                        "scale": ParamConstraint(
                            dtypes=frozenset({torch.float16, torch.bfloat16, torch.float32})
                        ),
                        "output_type": ParamConstraint(
                            dtypes=frozenset({torch.float16, torch.bfloat16, torch.float32})
                        ),
                    },
                    default_devices=metal_devices,
                ),
                "quantize_svdquant_w4a4": FunctionConstraints(
                    params={
                        "x": ParamConstraint(dtypes=metal_floats, shape_rules=(ExactDims(2),)),
                        "smooth": ParamConstraint(dtypes=metal_floats, shape_rules=(ExactDims(1),)),
                        "lora_down": ParamConstraint(
                            dtypes=metal_floats, shape_rules=(ExactDims(2),)
                        ),
                        "pad_size": ParamConstraint(dtypes=frozenset({int})),
                        "act_unsigned": ParamConstraint(dtypes=frozenset({bool})),
                        "lora_x": ParamConstraint(dtypes=metal_floats, shape_rules=(ExactDims(2),)),
                    },
                    default_devices=metal_devices,
                ),
                "scaled_mm_svdquant_w4a4": FunctionConstraints(
                    params={
                        "act": ParamConstraint(
                            dtypes=frozenset({torch.int8}), shape_rules=(ExactDims(2),)
                        ),
                        "wgt": ParamConstraint(dtypes=frozenset({torch.int8})),
                        "ascales": ParamConstraint(
                            dtypes=metal_floats, shape_rules=(ExactDims(2),)
                        ),
                        "wscales": ParamConstraint(dtypes=metal_floats),
                        "lora_act_in": ParamConstraint(
                            dtypes=frozenset(
                                {torch.float16, torch.bfloat16, torch.float32}
                            ),
                            shape_rules=(ExactDims(2),),
                        ),
                        "lora_up": ParamConstraint(dtypes=metal_floats),
                        "bias": ParamConstraint(dtypes=metal_floats),
                        "act_unsigned": ParamConstraint(dtypes=frozenset({bool})),
                    },
                    default_devices=metal_devices,
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

    # Eager-on-MPS wrappers: no metallib required (dequant + MPS GEMM / SDPA).
    constraints.update(
        {
            "quantize_nvfp4": FunctionConstraints(
                params={
                    "x": ParamConstraint(dtypes=metal_floats, shape_rules=(ExactDims(2),)),
                    "per_tensor_scale": ParamConstraint(dtypes=frozenset({torch.float32})),
                },
                default_devices=metal_devices,
                call_rules=(_validate_mps_fp_device("x"),),
            ),
            "dequantize_nvfp4": FunctionConstraints(
                params={
                    "qx": ParamConstraint(
                        dtypes=frozenset({torch.uint8}), shape_rules=(ExactDims(2),)
                    ),
                    "per_tensor_scale": ParamConstraint(dtypes=frozenset({torch.float32})),
                    "block_scales": ParamConstraint(
                        dtypes=frozenset({torch.float8_e4m3fn})
                    ),
                    "output_type": ParamConstraint(dtypes=metal_floats),
                },
                default_devices=metal_devices,
            ),
            "scaled_mm_nvfp4": FunctionConstraints(
                params={
                    "a": ParamConstraint(
                        dtypes=frozenset({torch.uint8}), shape_rules=(ExactDims(2),)
                    ),
                    "b": ParamConstraint(
                        dtypes=frozenset({torch.uint8}), shape_rules=(ExactDims(2),)
                    ),
                    "tensor_scale_a": ParamConstraint(dtypes=frozenset({torch.float32})),
                    "tensor_scale_b": ParamConstraint(dtypes=frozenset({torch.float32})),
                    "block_scale_a": ParamConstraint(
                        dtypes=frozenset({torch.float8_e4m3fn})
                    ),
                    "block_scale_b": ParamConstraint(
                        dtypes=frozenset({torch.float8_e4m3fn})
                    ),
                    "out_dtype": ParamConstraint(dtypes=metal_floats),
                },
                default_devices=metal_devices,
            ),
            "sol_attn": FunctionConstraints(
                params={
                    "q": ParamConstraint(dtypes=metal_floats, shape_rules=(ExactDims(4),)),
                    "k": ParamConstraint(dtypes=metal_floats, shape_rules=(ExactDims(4),)),
                    "v": ParamConstraint(dtypes=metal_floats, shape_rules=(ExactDims(4),)),
                },
                default_devices=metal_devices,
                call_rules=(_validate_mps_qkv("q"),),
            ),
        }
    )
    if _kernels() is not None:
        constraints["gemv_awq_w4a16"] = FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=metal_floats),
                "qweight": ParamConstraint(
                    dtypes=frozenset({torch.int8}), shape_rules=(ExactDims(2),)
                ),
                "wscales": ParamConstraint(dtypes=metal_floats, shape_rules=(ExactDims(2),)),
                "wzeros": ParamConstraint(dtypes=metal_floats, shape_rules=(ExactDims(2),)),
            },
            default_devices=metal_devices,
            call_rules=(_validate_mps_fp_tensor("x"),),
        )
    if hasattr(torch, "float8_e8m0fnu"):
        constraints.update(
            {
                "quantize_mxfp8": FunctionConstraints(
                    params={
                        "x": ParamConstraint(
                            dtypes=metal_floats, shape_rules=(ExactDims(2),)
                        ),
                    },
                    default_devices=metal_devices,
                    call_rules=(_validate_mps_fp_device("x"),),
                ),
                "dequantize_mxfp8": FunctionConstraints(
                    params={
                        "qx": ParamConstraint(
                            dtypes=frozenset({torch.float8_e4m3fn}),
                            shape_rules=(ExactDims(2),),
                        ),
                        "block_scales": ParamConstraint(
                            dtypes=frozenset({torch.float8_e8m0fnu}),
                        ),
                        "output_type": ParamConstraint(dtypes=metal_floats),
                    },
                    default_devices=metal_devices,
                ),
                "scaled_mm_mxfp8": FunctionConstraints(
                    params={
                        "a": ParamConstraint(
                            dtypes=frozenset({torch.float8_e4m3fn}),
                            shape_rules=(ExactDims(2),),
                        ),
                        "b": ParamConstraint(
                            dtypes=frozenset({torch.float8_e4m3fn}),
                            shape_rules=(ExactDims(2),),
                        ),
                        "block_scale_a": ParamConstraint(
                            dtypes=frozenset({torch.float8_e8m0fnu}),
                        ),
                        "block_scale_b": ParamConstraint(
                            dtypes=frozenset({torch.float8_e8m0fnu}),
                        ),
                        "out_dtype": ParamConstraint(dtypes=metal_floats),
                    },
                    default_devices=metal_devices,
                ),
            }
        )

    return constraints


if _METAL_DEVICE_AVAILABLE:
    _caps = _build_constraints()
    if _caps:
        registry.register(
            name="metal",
            module=__import__(__name__, fromlist=__all__),
            capabilities=_caps,
        )
    else:
        registry.mark_unavailable("metal", _METAL_ERROR or "no metal capabilities")
else:
    registry.mark_unavailable("metal", _METAL_ERROR or "Apple Metal backend is unavailable")
