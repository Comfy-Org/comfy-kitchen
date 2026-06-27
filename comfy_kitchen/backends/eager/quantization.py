# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Portions of this code are derived from PyTorch AO:
#   https://github.com/pytorch/ao/blob/main/torchao/prototype/mx_formats/nvfp4_tensor.py
#   Copyright (c) Meta Platforms, Inc. and affiliates.
#   Licensed under the BSD 3-Clause License (see NOTICE file for details)

import torch

from comfy_kitchen.float_utils import (
    E8M0_BIAS,
    F4_E2M1_MAX,
    F8_E4M3_MAX,
    F8_E5M2_MAX,
    _f32_to_floatx_unpacked,
    _float8_round,
    e8m0_to_f32,
    from_blocked,
    pack_uint4,
    roundup,
    to_blocked,
)
from comfy_kitchen.registry import registry
from comfy_kitchen.scaled_mm_v2 import ScalingType, SwizzleType, scaled_mm_v2
from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation

# =============================================================================
# Dtype Code Mappings (shared between custom ops and backends)
# =============================================================================

# Maps dtype codes to torch dtypes (matches CUDA backend conventions)
DTYPE_CODE_TO_DTYPE = {
    0: torch.float32,
    1: torch.float16,
    2: torch.bfloat16,
    5: torch.float8_e4m3fn,
    6: torch.float8_e5m2,
}

DTYPE_TO_CODE = {v: k for k, v in DTYPE_CODE_TO_DTYPE.items()}

def quantize_per_tensor_fp8(
    x: torch.Tensor, scale: torch.Tensor, output_type: torch.dtype = torch.float8_e4m3fn
) -> torch.Tensor:
    if output_type == torch.float8_e4m3fn:
        lp_max = F8_E4M3_MAX
    elif output_type == torch.float8_e5m2:
        lp_max = F8_E5M2_MAX
    else:
        raise ValueError(
            f"Unsupported output_type: {output_type}. Expected torch.float8_e4m3fn or torch.float8_e5m2"
        )

    temp = x * (1.0 / scale).to(x.dtype)
    temp = torch.clamp(temp, -lp_max, lp_max, out=temp)
    return temp.to(output_type)

def dequantize_per_tensor_fp8(
    x: torch.Tensor, scale: torch.Tensor, output_type: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    dq_tensor = x.to(dtype=output_type) * scale.to(dtype=output_type)
    return dq_tensor


def calc_mantissa(abs_x, exponent, normal_mask, MANTISSA_BITS, EXPONENT_BIAS, rng):  # noqa: N803
    mantissa_scaled = torch.where(
        normal_mask,
        (abs_x / (2.0 ** (exponent - EXPONENT_BIAS)) - 1.0) * (2**MANTISSA_BITS),
        (abs_x / (2.0 ** (-EXPONENT_BIAS + 1 - MANTISSA_BITS))),
    )

    mantissa_scaled += rng.to(dtype=mantissa_scaled.dtype) * (1.0 / 256.0)
    return mantissa_scaled.floor() / (2**MANTISSA_BITS)


def stochastic_rounding_fp8(
    x: torch.Tensor,
    rng: torch.Tensor,
    output_type: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    if output_type == torch.float8_e4m3fn:
        EXPONENT_BITS, MANTISSA_BITS, EXPONENT_BIAS = 4, 3, 7  # noqa: N806
    elif output_type == torch.float8_e5m2:
        EXPONENT_BITS, MANTISSA_BITS, EXPONENT_BIAS = 5, 2, 15  # noqa: N806
    else:
        raise ValueError(
            f"Unsupported output_type: {output_type}. Expected torch.float8_e4m3fn "
            "or torch.float8_e5m2"
        )

    x = x.half()
    sign = torch.sign(x)
    abs_x = x.abs()
    sign = torch.where(abs_x == 0, 0, sign)

    exponent = torch.clamp(
        torch.floor(torch.log2(abs_x)) + EXPONENT_BIAS,
        0,
        2**EXPONENT_BITS - 1,
    )
    normal_mask = ~(exponent == 0)

    abs_x[:] = calc_mantissa(abs_x, exponent, normal_mask, MANTISSA_BITS, EXPONENT_BIAS, rng)

    sign *= torch.where(
        normal_mask,
        (2.0 ** (exponent - EXPONENT_BIAS)) * (1.0 + abs_x),
        (2.0 ** (-EXPONENT_BIAS + 1)) * abs_x,
    )

    info = torch.finfo(output_type)
    torch.clamp(sign, min=info.min, max=info.max, out=sign)
    return sign.to(output_type)


def quantize_nvfp4(
    x: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    epsilon: float = 0.0,
    pad_16x: bool = False,
    hi_first: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_shape = x.shape

    # Handle padding
    if pad_16x:
        rows, cols = x.shape
        padded_rows = roundup(rows, 16)
        padded_cols = roundup(cols, 16)
        if padded_rows != rows or padded_cols != cols:
            x = torch.nn.functional.pad(x, (0, padded_cols - cols, 0, padded_rows - rows))
            # Note: We update orig_shape because the output tensor logic below assumes x.shape matches
            # what we want to produce. If we pad here, we want the padded output.
            orig_shape = x.shape

    block_size = 16

    x = x.reshape(orig_shape[0], -1, block_size)
    max_abs = torch.amax(torch.abs(x), dim=-1)
    block_scale = max_abs.to(torch.float32) / F4_E2M1_MAX
    scaled_block_scales = block_scale / per_tensor_scale
    scaled_block_scales_fp8 = torch.clamp(scaled_block_scales, max=F8_E4M3_MAX)
    scaled_block_scales_fp32 = _float8_round(scaled_block_scales_fp8)
    total_scale = per_tensor_scale * scaled_block_scales_fp32

    # Handle zero blocks (from padding): avoid 0/0 NaN
    zero_scale_mask = (total_scale == 0)
    total_scale_safe = torch.where(zero_scale_mask, torch.ones_like(total_scale), total_scale)

    data_scaled = x.float() / total_scale_safe.unsqueeze(-1)
    data_scaled = torch.where(zero_scale_mask.unsqueeze(-1), torch.zeros_like(data_scaled), data_scaled)

    out_scales = scaled_block_scales_fp8

    data_scaled = torch.clamp(data_scaled, -F4_E2M1_MAX, F4_E2M1_MAX)
    data_scaled = data_scaled.view(orig_shape)

    data_lp = _f32_to_floatx_unpacked(data_scaled, 2, 1)
    data_lp = pack_uint4(data_lp, hi_first=hi_first)
    blocked_scales = to_blocked(out_scales.to(torch.float8_e4m3fn), flatten=False)
    return data_lp, blocked_scales


E2M1_LUT = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
]).unsqueeze(1)

E2M1_LUT_CACHE = {}


def dequantize_nvfp4(
    qx: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    block_scales: torch.Tensor,
    output_type: torch.dtype = torch.bfloat16,
    hi_first: bool = True,
) -> torch.Tensor:
    lut = E2M1_LUT_CACHE.get((qx.device, output_type))
    if lut is None:
        lut = E2M1_LUT.to(qx.device, output_type)
        E2M1_LUT_CACHE[(qx.device, output_type)] = lut

    lo = qx & 0x0F
    hi = qx >> 4
    if hi_first:
        out = torch.stack([hi, lo], dim=-1).view(*qx.shape[:-1], -1)
    else:
        out = torch.stack([lo, hi], dim=-1).view(*qx.shape[:-1], -1)
    out = torch.nn.functional.embedding(out.int(), lut).squeeze(-1)

    # Get original shape (packed tensor has half the columns)
    orig_shape = out.shape
    block_size = 16

    # Reshape to blocks for scaling
    out = out.reshape(orig_shape[0], -1, block_size)

    # Unswizzle block_scales from cuBLAS tiled layout
    # The scales are in (RoundUp(num_rows, 128), RoundUp(num_cols//16, 4)) format
    # We need to extract the actual scales for our data shape
    num_blocks_per_row = orig_shape[1] // block_size

    # Use from_blocked to unswizzle the tiled layout
    block_scales_unswizzled = from_blocked(
        block_scales,
        num_rows=orig_shape[0],
        num_cols=num_blocks_per_row
    )

    # Compute total decode scale: per_tensor_scale * block_scale_fp8
    total_scale = per_tensor_scale.to(output_type) * block_scales_unswizzled.to(output_type)

    # Apply scaling to dequantize
    data_dequantized = out * total_scale.unsqueeze(-1)

    # Reshape back to original shape and convert to output type
    result = data_dequantized.view(orig_shape).to(output_type)

    return result


def scaled_mm_nvfp4(
    a: torch.Tensor,
    b: torch.Tensor,
    tensor_scale_a: torch.Tensor,
    tensor_scale_b: torch.Tensor,
    block_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    alpha: torch.Tensor | None = None,
) -> torch.Tensor:
    result = scaled_mm_v2(
        a.view(torch.float4_e2m1fn_x2),
        b.view(torch.float4_e2m1fn_x2).t(),
        scale_a=[block_scale_a.view(-1), tensor_scale_a],
        scale_b=[block_scale_b.view(-1), tensor_scale_b],
        bias=bias,
        out_dtype=out_dtype,
        scale_recipe_a = [ScalingType.BlockWise1x16, ScalingType.TensorWise],
        scale_recipe_b = [ScalingType.BlockWise1x16, ScalingType.TensorWise],
        swizzle_a = [SwizzleType.SWIZZLE_32_4_4, SwizzleType.NO_SWIZZLE],
        swizzle_b = [SwizzleType.SWIZZLE_32_4_4, SwizzleType.NO_SWIZZLE],
    )

    return result


# =============================================================================
# MXFP8 Quantization Functions
# MXFP8 uses FP8 data with E8M0 block scales and block size 32
# =============================================================================

MXFP8_BLOCK_SIZE = 32


def quantize_mxfp8(
    x: torch.Tensor,
    pad_32x: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to MXFP8 format with block-wise E8M0 scaling.

    MXFP8 uses block size 32 with power-of-2 (E8M0) block scales.

    Args:
        x: Input tensor (2D, shape M x K)
        output_type: FP8 output dtype (float8_e4m3fn)
        pad_32x: If True, pad dimensions to be divisible by 32

    Returns:
        Tuple of (quantized_fp8_tensor, block_scales_e8m0)
        - quantized_fp8_tensor: FP8 data of shape (M, K) or padded shape
        - block_scales_e8m0: E8M0 scales of shape (M, K//32) in swizzled layout
    """
    orig_shape = x.shape
    assert x.dim() == 2, "Input must be 2D"

    # Handle padding
    if pad_32x:
        rows, cols = x.shape
        padded_rows = roundup(rows, 32)
        padded_cols = roundup(cols, 32)
        if padded_rows != rows or padded_cols != cols:
            x = torch.nn.functional.pad(x, (0, padded_cols - cols, 0, padded_rows - rows))
            orig_shape = x.shape
    else:
        assert x.shape[1] % MXFP8_BLOCK_SIZE == 0, f"K dimension must be divisible by {MXFP8_BLOCK_SIZE}"

    rows, cols = orig_shape
    num_blocks = cols // MXFP8_BLOCK_SIZE
    x_blocked = x.reshape(rows, num_blocks, MXFP8_BLOCK_SIZE)
    max_abs = torch.amax(torch.abs(x_blocked), dim=-1)

    scale_needed = max_abs.float() / F8_E4M3_MAX
    scale_needed = torch.clamp(scale_needed, min=2**(-127))  # Min E8M0 value

    # Convert to E8M0 exponent (round up to ensure values fit)
    log2_scale = torch.log2(scale_needed)
    exp_biased = torch.ceil(log2_scale).to(torch.int32) + E8M0_BIAS
    exp_biased = torch.clamp(exp_biased, 0, 254)  # Valid E8M0 range [0, 254]

    block_scales_e8m0 = exp_biased.to(torch.uint8)
    block_scales_f32 = e8m0_to_f32(block_scales_e8m0)

    # Handle zero blocks
    zero_mask = (max_abs == 0)
    block_scales_f32 = torch.where(zero_mask, torch.ones_like(block_scales_f32), block_scales_f32)

    # Quantize: scale down by block scale, then clamp and convert to FP8
    data_scaled = x_blocked.float() / block_scales_f32.unsqueeze(-1)
    data_scaled = torch.where(zero_mask.unsqueeze(-1), torch.zeros_like(data_scaled), data_scaled)

    # Clamp to FP8 range and convert
    data_scaled = torch.clamp(data_scaled, -F8_E4M3_MAX, F8_E4M3_MAX)
    data_fp8 = data_scaled.reshape(orig_shape).to(torch.float8_e4m3fn)

    # Handle zero blocks in scales
    block_scales_e8m0 = torch.where(zero_mask, torch.zeros_like(block_scales_e8m0), block_scales_e8m0)

    # Convert scales to swizzled layout for cuBLAS compatibility
    # For MXFP8 with block size 32, we have num_blocks = K/32
    blocked_scales = to_blocked(block_scales_e8m0, flatten=False)

    return data_fp8, blocked_scales.view(torch.float8_e8m0fnu)


def dequantize_mxfp8(
    qx: torch.Tensor,
    block_scales: torch.Tensor,
    output_type: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize tensor from MXFP8 format.

    Args:
        qx: Quantized FP8 tensor (float8_e4m3fn or float8_e5m2)
        block_scales: E8M0 block scales in swizzled layout (float8_e8m0fnu)
        output_type: Target output dtype

    Returns:
        Dequantized tensor
    """
    orig_shape = qx.shape
    rows, cols = orig_shape
    num_blocks = cols // MXFP8_BLOCK_SIZE

    # Unswizzle block_scales from cuBLAS tiled layout
    block_scales_uint8 = block_scales.view(torch.uint8)
    block_scales_unswizzled = from_blocked(
        block_scales_uint8,
        num_rows=rows,
        num_cols=num_blocks
    )

    # Convert E8M0 scales to float32
    block_scales_f32 = e8m0_to_f32(block_scales_unswizzled)

    # Reshape FP8 data for block-wise dequantization
    data_f32 = qx.to(torch.float32).reshape(rows, num_blocks, MXFP8_BLOCK_SIZE)

    # Apply block scales
    data_dequantized = data_f32 * block_scales_f32.unsqueeze(-1)

    # Reshape and convert to output type
    return data_dequantized.reshape(orig_shape).to(output_type)


def scaled_mm_mxfp8(
    a: torch.Tensor,
    b: torch.Tensor,
    block_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None
) -> torch.Tensor:
    """MXFP8 matrix multiplication using block-wise E8M0 scales.

    Computes: output = a @ b.T + bias (linear semantics)

    Args:
        a: FP8 matrix A of shape (M, K)
        b: FP8 matrix B of shape (N, K) - will be transposed internally
        block_scale_a: E8M0 block scales for A in swizzled format
        block_scale_b: E8M0 block scales for B in swizzled format
        bias: Optional bias vector of shape (N,)
        out_dtype: Output dtype

    Scales are expected to be in swizzled (SWIZZLE_32_4_4) format from quantize_mxfp8.
    """
    result = scaled_mm_v2(
        a,
        b.t(),  # Transpose b for linear semantics: a @ b.T
        scale_a=block_scale_a,
        scale_b=block_scale_b,
        bias=bias,
        out_dtype=out_dtype,
        scale_recipe_a=ScalingType.BlockWise1x32,
        scale_recipe_b=ScalingType.BlockWise1x32,
        swizzle_a=SwizzleType.SWIZZLE_32_4_4,
        swizzle_b=SwizzleType.SWIZZLE_32_4_4,
    )

    return result

# =============================================================================
# torch.library Custom Op Definitions
# These are the entry points for torch.compile. They dispatch to the best
# available backend via the registry.
# =============================================================================


@torch.library.custom_op("comfy_kitchen::quantize_fp8", mutates_args=())
def _op_quantize_fp8(
    x: torch.Tensor,
    scale: torch.Tensor,
    output_dtype_code: int,
) -> torch.Tensor:
    """Quantize tensor to FP8 format with per-tensor scaling.

    Args:
        x: Input tensor
        scale: Scale tensor (scalar)
        output_dtype_code: FP8 dtype code (5=float8_e4m3fn, 6=float8_e5m2)

    Returns:
        Quantized FP8 tensor
    """
    output_dtype = DTYPE_CODE_TO_DTYPE[output_dtype_code]
    kwargs = {"x": x, "scale": scale, "output_type": output_dtype}
    impl = registry.get_implementation("quantize_per_tensor_fp8", kwargs=kwargs)
    return impl(**kwargs)


@_op_quantize_fp8.register_fake
def _op_quantize_fp8_fake(x, scale, output_dtype_code):
    output_dtype = DTYPE_CODE_TO_DTYPE[output_dtype_code]
    return torch.empty_like(x, dtype=output_dtype)


@torch.library.custom_op("comfy_kitchen::dequantize_fp8", mutates_args=())
def _op_dequantize_fp8(
    x: torch.Tensor,
    scale: torch.Tensor,
    output_dtype_code: int,
) -> torch.Tensor:
    """Dequantize tensor from FP8 format with per-tensor scaling.

    Args:
        x: Input FP8 tensor (float8_e4m3fn or float8_e5m2)
        scale: Scale tensor (scalar)
        output_dtype_code: Target dtype code (0=float32, 1=float16, 2=bfloat16)

    Returns:
        Dequantized tensor in specified output format
    """
    output_dtype = DTYPE_CODE_TO_DTYPE[output_dtype_code]
    kwargs = {"x": x, "scale": scale, "output_type": output_dtype}
    impl = registry.get_implementation("dequantize_per_tensor_fp8", kwargs=kwargs)
    return impl(**kwargs)


@_op_dequantize_fp8.register_fake
def _op_dequantize_fp8_fake(x, scale, output_dtype_code):
    output_dtype = DTYPE_CODE_TO_DTYPE[output_dtype_code]
    return torch.empty_like(x, dtype=output_dtype)


@torch.library.custom_op("comfy_kitchen::quantize_nvfp4", mutates_args=())
def _op_quantize_nvfp4(
    x: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    epsilon: float,
    pad_16x: bool,
    hi_first: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to NVFP4 format with block-wise scaling.

    Args:
        x: Input tensor (2D)
        per_tensor_scale: Global scale factor
        epsilon: Epsilon for numerical stability
        pad_16x: If True, pad dimensions to be divisible by 16
        hi_first: If True, the even-indexed element is packed into the high nibble (default).
                  If False, the even-indexed element goes into the low nibble.

    Returns:
        Tuple of (quantized_tensor, block_scales)
    """
    kwargs = {"x": x, "per_tensor_scale": per_tensor_scale, "epsilon": epsilon, "pad_16x": pad_16x, "hi_first": hi_first}
    impl = registry.get_implementation("quantize_nvfp4", kwargs=kwargs)
    return impl(**kwargs)


@_op_quantize_nvfp4.register_fake
def _op_quantize_nvfp4_fake(x, per_tensor_scale, epsilon, pad_16x, hi_first):
    rows, cols = x.shape

    if pad_16x:
        rows = roundup(rows, 16)
        cols = roundup(cols, 16)

    # Packed output: 2 FP4 values per uint8
    qdata = torch.empty((rows, cols // 2), dtype=torch.uint8, device=x.device)

    # Block scales: cuBLAS tiled layout
    scale_rows = roundup(rows, 128)
    scale_cols = roundup(cols // 16, 4)
    block_scales = torch.empty((scale_rows, scale_cols), dtype=torch.float8_e4m3fn, device=x.device)

    return qdata, block_scales


@torch.library.custom_op("comfy_kitchen::dequantize_nvfp4", mutates_args=())
def _op_dequantize_nvfp4(
    qx: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    block_scales: torch.Tensor,
    output_dtype_code: int,
    hi_first: bool,
) -> torch.Tensor:
    """Dequantize tensor from NVFP4 format with block-wise scaling.

    Args:
        qx: Quantized FP4 tensor (packed as uint8)
        per_tensor_scale: Global scale factor
        block_scales: Block scales in swizzled layout (float8_e4m3fn)
        output_dtype_code: Target dtype code (0=float32, 1=float16, 2=bfloat16)
        hi_first: If True, the high nibble is the even-indexed element (default).
                  If False, the low nibble is the even-indexed element.

    Returns:
        Dequantized tensor in specified output format
    """
    output_dtype = DTYPE_CODE_TO_DTYPE[output_dtype_code]
    kwargs = {"qx": qx, "per_tensor_scale": per_tensor_scale, "block_scales": block_scales, "output_type": output_dtype, "hi_first": hi_first}
    impl = registry.get_implementation("dequantize_nvfp4", kwargs=kwargs)
    return impl(**kwargs)


@_op_dequantize_nvfp4.register_fake
def _op_dequantize_nvfp4_fake(qx, per_tensor_scale, block_scales, output_dtype_code, hi_first):
    output_dtype = DTYPE_CODE_TO_DTYPE[output_dtype_code]
    # Unpacked shape: cols * 2 (since 2 FP4 values per uint8)
    rows, cols_packed = qx.shape
    return torch.empty((rows, cols_packed * 2), dtype=output_dtype, device=qx.device)


@torch.library.custom_op("comfy_kitchen::scaled_mm_nvfp4", mutates_args=())
def _op_scaled_mm_nvfp4(
    a: torch.Tensor,
    b: torch.Tensor,
    tensor_scale_a: torch.Tensor,
    tensor_scale_b: torch.Tensor,
    block_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    bias: torch.Tensor | None,
    output_dtype_code: int,
    alpha: torch.Tensor | None,
) -> torch.Tensor:
    """Matrix multiplication with NVFP4 quantized inputs.

    Computes: y = (a @ b.T) * (tensor_scale_a * tensor_scale_b) + bias

    Args:
        a: Quantized matrix A (M, K//2) in uint8 format
        b: Quantized matrix B (N, K//2) in uint8 format
        tensor_scale_a: Global scale for A
        tensor_scale_b: Global scale for B
        block_scale_a: Block-wise scales for A
        block_scale_b: Block-wise scales for B
        bias: Optional bias vector
        output_dtype_code: Output dtype code (0=float32, 1=float16, 2=bfloat16)
        alpha: Output scale (tensor_scale_a * tensor_scale_b)

    Returns:
        Result tensor of shape (M, N)
    """
    out_dtype = DTYPE_CODE_TO_DTYPE[output_dtype_code]
    kwargs = {
        "a": a, "b": b,
        "tensor_scale_a": tensor_scale_a, "tensor_scale_b": tensor_scale_b,
        "block_scale_a": block_scale_a, "block_scale_b": block_scale_b,
        "bias": bias, "out_dtype": out_dtype, "alpha": alpha,
    }
    impl = registry.get_implementation("scaled_mm_nvfp4", kwargs=kwargs)
    return impl(**kwargs)


@_op_scaled_mm_nvfp4.register_fake
def _op_scaled_mm_nvfp4_fake(
    a, b, tensor_scale_a, tensor_scale_b,
    block_scale_a, block_scale_b, bias, output_dtype_code, alpha
):
    out_dtype = DTYPE_CODE_TO_DTYPE[output_dtype_code]
    m = a.shape[0]
    n = b.shape[0]
    return torch.empty((m, n), dtype=out_dtype, device=a.device)


# =============================================================================
# MXFP8 Custom Ops
# =============================================================================

@torch.library.custom_op("comfy_kitchen::quantize_mxfp8", mutates_args=())
def _op_quantize_mxfp8(
    x: torch.Tensor,
    pad_32x: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to MXFP8 format with block-wise E8M0 scaling.

    MXFP8 uses block size 32 with power-of-2 (E8M0) block scales.

    Args:
        x: Input tensor (2D, shape M x K)
        pad_32x: If True, pad dimensions to be divisible by 32

    Returns:
        Tuple of (quantized_fp8_tensor, block_scales_e8m0)
    """
    kwargs = {"x": x, "pad_32x": pad_32x}
    impl = registry.get_implementation("quantize_mxfp8", kwargs=kwargs)
    return impl(**kwargs)


@_op_quantize_mxfp8.register_fake
def _op_quantize_mxfp8_fake(x, pad_32x):
    rows, cols = x.shape

    if pad_32x:
        rows = roundup(rows, 32)
        cols = roundup(cols, 32)

    # FP8 output (not packed)
    qdata = torch.empty((rows, cols), dtype=torch.float8_e4m3fn, device=x.device)

    # Block scales: E8M0 in cuBLAS tiled layout
    # Block size 32, so num_blocks = cols // 32
    num_blocks = cols // 32
    scale_rows = roundup(rows, 128)
    scale_cols = roundup(num_blocks, 4)
    block_scales = torch.empty((scale_rows, scale_cols), dtype=torch.float8_e8m0fnu, device=x.device)

    return qdata, block_scales


@torch.library.custom_op("comfy_kitchen::dequantize_mxfp8", mutates_args=())
def _op_dequantize_mxfp8(
    qx: torch.Tensor,
    block_scales: torch.Tensor,
    output_dtype_code: int,
) -> torch.Tensor:
    """Dequantize tensor from MXFP8 format.

    Args:
        qx: Quantized FP8 tensor (float8_e4m3fn)
        block_scales: E8M0 block scales in swizzled layout (float8_e8m0fnu)
        output_dtype_code: Target dtype code (0=float32, 1=float16, 2=bfloat16)

    Returns:
        Dequantized tensor in specified output format
    """
    output_dtype = DTYPE_CODE_TO_DTYPE[output_dtype_code]
    kwargs = {"qx": qx, "block_scales": block_scales, "output_type": output_dtype}
    impl = registry.get_implementation("dequantize_mxfp8", kwargs=kwargs)
    return impl(**kwargs)


@_op_dequantize_mxfp8.register_fake
def _op_dequantize_mxfp8_fake(qx, block_scales, output_dtype_code):
    output_dtype = DTYPE_CODE_TO_DTYPE[output_dtype_code]
    return torch.empty_like(qx, dtype=output_dtype)


@torch.library.custom_op("comfy_kitchen::scaled_mm_mxfp8", mutates_args=())
def _op_scaled_mm_mxfp8(
    a: torch.Tensor,
    b: torch.Tensor,
    block_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    bias: torch.Tensor | None,
    output_dtype_code: int,
) -> torch.Tensor:
    """Matrix multiplication with MXFP8 quantized inputs.

    Computes: y = a @ b.T + bias

    Args:
        a: Quantized FP8 matrix A (M, K)
        b: Quantized FP8 matrix B (N, K)
        block_scale_a: E8M0 block scales for A in swizzled layout
        block_scale_b: E8M0 block scales for B in swizzled layout
        bias: Optional bias vector
        output_dtype_code: Output dtype code (0=float32, 1=float16, 2=bfloat16)

    Returns:
        Result tensor of shape (M, N)
    """
    out_dtype = DTYPE_CODE_TO_DTYPE[output_dtype_code]
    kwargs = {
        "a": a, "b": b,
        "block_scale_a": block_scale_a, "block_scale_b": block_scale_b,
        "bias": bias, "out_dtype": out_dtype,
    }
    impl = registry.get_implementation("scaled_mm_mxfp8", kwargs=kwargs)
    return impl(**kwargs)


@_op_scaled_mm_mxfp8.register_fake
def _op_scaled_mm_mxfp8_fake(
    a, b, block_scale_a, block_scale_b, bias, output_dtype_code
):
    out_dtype = DTYPE_CODE_TO_DTYPE[output_dtype_code]
    m = a.shape[0]
    n = b.shape[0]
    return torch.empty((m, n), dtype=out_dtype, device=a.device)
# =============================================================================
# INT8 Tensor-wise Quantization (from dxqb/OneTrainer)
# =============================================================================
# Simpler approach: single scale per tensor + per-row activation scaling.
# Uses torch._int_mm for cuBLASLt acceleration on CUDA.


_turing_device_cache: dict[int, bool] = {}


def _cuda_device_is_turing(device_index: int) -> bool:
    cached = _turing_device_cache.get(device_index)
    if cached is not None:
        return cached
    try:
        is_turing = torch.cuda.get_device_capability(device_index) == (7, 5)
    except RuntimeError:
        is_turing = False
    _turing_device_cache[device_index] = is_turing
    return is_turing


def _int8_mm_n_alignment(tensor: torch.Tensor) -> int:
    # Turing cuBLASLt INT8 rejects some skinny-N shapes, e.g. N=17.
    return 32 if tensor.is_cuda and _cuda_device_is_turing(tensor.get_device()) else 8


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _int8_matmul_accumulate(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Multiply INT8 matrices and return INT32 accumulators."""
    def fast_int8_mm(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        if hasattr(torch, "int8_mm"):
            return torch.int8_mm(lhs, rhs)
        return torch._int_mm(lhs, rhs)

    orig_m = a.size(0)
    orig_n = b.size(1)
    k = a.size(1)
    if orig_m == 0 or k == 0:
        return fast_int8_mm(a, b)

    padded_m = _round_up(max(orig_m, 32), 32) if a.is_cuda else orig_m
    if padded_m != orig_m:
        row_padding = torch.zeros((padded_m - orig_m, k), device=a.device, dtype=a.dtype)
        a = torch.cat((a, row_padding), dim=0)

    padded_k = ((k + 7) // 8) * 8
    if padded_k != k:
        a_padding = torch.zeros((a.size(0), padded_k - k), device=a.device, dtype=a.dtype)
        b_padding = torch.zeros((padded_k - k, b.size(1)), device=b.device, dtype=b.dtype)
        a = torch.cat((a, a_padding), dim=1)
        b = torch.cat((b, b_padding), dim=0)

    padded_n = _round_up(orig_n, _int8_mm_n_alignment(a))
    if padded_n != orig_n:
        b_padding = torch.zeros((b.size(0), padded_n - orig_n), device=b.device, dtype=b.dtype)
        b = torch.cat((b, b_padding), dim=1)

    result = fast_int8_mm(a, b)
    if result.size(0) != orig_m or result.size(1) != orig_n:
        result = result[:orig_m, :orig_n]
    return result


def mm_int8(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """INT8 matrix multiplication: C[M,N] = A[M,K] @ B[K,N].

    Uses torch._int_mm (cuBLASLt on CUDA). Output is int32.

    Args:
        a: INT8 tensor [M, K].
        b: INT8 tensor [K, N].

    Returns:
        INT32 tensor [M, N] with accumulated dot products.
    """
    assert a.dtype == torch.int8 and b.dtype == torch.int8
    assert a.dim() == 2 and b.dim() == 2
    assert a.size(1) == b.size(0), f"K mismatch: {a.size(1)} vs {b.size(0)}"
    return _int8_matmul_accumulate(a, b)


def quantize_int8_tensorwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to INT8 with single tensorwise scale.

    Args:
        x: Input tensor of any shape.

    Returns:
        Tuple of (quantized_int8, scale):
            - quantized_int8: INT8 tensor with same shape
            - scale: Scalar float32 tensor
    """
    abs_max = x.abs().max()
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    q = (x.float() / scale).round().clamp(-128.0, 127.0).to(torch.int8)
    return q, scale


def quantize_int8_rowwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to INT8 with per-row scales (for activations).

    Args:
        x: Input tensor [..., K] where quantization is per-row.

    Returns:
        Tuple of (quantized_int8, scales):
            - quantized_int8: INT8 tensor with same shape
            - scales: Float32 tensor [..., 1] with per-row scales
    """
    abs_max = x.abs().amax(dim=-1, keepdim=True)
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    q = (x.float() / scale).round().clamp(-128.0, 127.0).to(torch.int8)
    return q, scale


def quantize_and_rotate_rowwise(x: torch.Tensor, h: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused online activation rotation + row-wise quantization (Eager fallback).

    Args:
        x: Input unrotated activation tensor [M, K].
        h: Pre-built normalized Hadamard matrix [group_size, group_size].
        group_size: ConvRot group size.

    Returns:
        Tuple of (rotated_quantized_x_int8, row_scales).
    """
    x_rot = _rotate_activation(x, h, group_size)
    return quantize_int8_rowwise(x_rot)


def dequantize_int8_simple(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize INT8 tensor with scale.

    Args:
        q: Quantized INT8 tensor.
        scale: Scale tensor (scalar or broadcastable).

    Returns:
        Dequantized float tensor.
    """
    return q.float() * scale


def int8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    convrot: bool = False,
    convrot_groupsize: int = 256,
) -> torch.Tensor:
    """INT8 linear layer using torch.int8_mm with memory-efficient scaling.

    Quantizes x dynamically per-row, uses tensorwise weight scale.
    Processes scaling in chunks to avoid materializing large float32 tensors.

    Args:
        x: Input tensor [..., K].
        weight: INT8 weight tensor [N, K].
        weight_scale: Scalar weight scale.
        bias: Optional bias [N].
        out_dtype: Output dtype.
        convrot: If True, apply online activation rotation.
        convrot_groupsize: Group size for Hadamard rotation.

    Returns:
        Result tensor [..., N].
    """
    if x.shape[-1] != weight.shape[-1]:
        raise ValueError(
            f"Input and weight inner dimensions must match, got {x.shape[-1]} and {weight.shape[-1]}"
        )

    weight = weight.to(device=x.device).contiguous()
    weight_scale = weight_scale.to(device=x.device, dtype=torch.float32).reshape(-1)
    if weight_scale.numel() not in (1, weight.shape[0]):
        raise ValueError(
            f"INT8 weight scale must be scalar or per-output-channel, got {tuple(weight_scale.shape)} "
            f"for weight shape {tuple(weight.shape)}"
        )

    if convrot:
        if x.shape[-1] % convrot_groupsize != 0:
            raise ValueError(
                f"ConvRot group size {convrot_groupsize} does not divide input features {x.shape[-1]}"
            )
        h = _build_hadamard(convrot_groupsize, device=x.device, dtype=x.dtype)
        x = _rotate_activation(x, h, convrot_groupsize)

    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])

    # Quantize input per-row
    x_8, x_scale = quantize_int8_rowwise(x_2d)

    # Compute: x_8 @ weight.T using fast int8 matmul when shape constraints allow it.
    # weight is [N, K], we need [K, N] for matmul so transpose
    result = _int8_matmul_accumulate(x_8, weight.T.contiguous())

    # Scale back efficiently: result * (weight_scale * x_scale)
    # Process in chunks to avoid materializing large float32 tensor
    # which causes OOM for large models

    m, n = result.shape
    chunk_size = max(1, min(m, 256 * 1024 * 1024 // (n * 4)))  # Estimate safe chunk size

    weight_scale = weight_scale.reshape(1, -1)
    scaled_parts = []
    for i in range(0, m, chunk_size):
        end_i = min(i + chunk_size, m)
        chunk = result[i:end_i].float()

        # Apply scales: chunk * (weight_scale * x_scale[i:end_i])
        chunk_scales = x_scale[i:end_i].to(device=chunk.device, dtype=torch.float32) * weight_scale
        chunk_scaled = chunk * chunk_scales

        # Convert to output dtype immediately to free memory
        chunk_scaled = chunk_scaled.to(out_dtype)
        scaled_parts.append(chunk_scaled)

    result = torch.cat(scaled_parts, dim=0)

    if bias is not None:
        result = result + bias.to(device=result.device, dtype=result.dtype).reshape(1, -1)

    return result.reshape(*orig_shape[:-1], weight.shape[0])


# =============================================================================
# torch.library Custom Op Definitions — INT8 Tensor-wise
# =============================================================================


@torch.library.custom_op("comfy_kitchen::quantize_int8_tensorwise", mutates_args=())
def _op_quantize_int8_tensorwise(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    kwargs = {"x": x}
    impl = registry.get_implementation("quantize_int8_tensorwise", kwargs=kwargs)
    return impl(**kwargs)


@_op_quantize_int8_tensorwise.register_fake
def _op_quantize_int8_tensorwise_fake(x):
    q = torch.empty_like(x, dtype=torch.int8)
    scale = torch.empty((), dtype=torch.float32, device=x.device)
    return q, scale


@torch.library.custom_op("comfy_kitchen::quantize_int8_rowwise", mutates_args=())
def _op_quantize_int8_rowwise(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    kwargs = {"x": x}
    impl = registry.get_implementation("quantize_int8_rowwise", kwargs=kwargs)
    return impl(**kwargs)


@_op_quantize_int8_rowwise.register_fake
def _op_quantize_int8_rowwise_fake(x):
    q = torch.empty_like(x, dtype=torch.int8)
    scale = torch.empty(*x.shape[:-1], 1, dtype=torch.float32, device=x.device)
    return q, scale


@torch.library.custom_op("comfy_kitchen::dequantize_int8_simple", mutates_args=())
def _op_dequantize_int8_simple(
    q: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    kwargs = {"q": q, "scale": scale}
    impl = registry.get_implementation("dequantize_int8_simple", kwargs=kwargs)
    return impl(**kwargs)


@_op_dequantize_int8_simple.register_fake
def _op_dequantize_int8_simple_fake(q, scale):
    return torch.empty_like(q, dtype=torch.float32)


@torch.library.custom_op("comfy_kitchen::int8_linear", mutates_args=())
def _op_int8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    output_dtype_code: int,
    convrot: bool = False,
    convrot_groupsize: int = 256,
) -> torch.Tensor:
    out_dtype = DTYPE_CODE_TO_DTYPE[output_dtype_code]
    kwargs = {
        "x": x,
        "weight": weight,
        "weight_scale": weight_scale,
        "bias": bias,
        "out_dtype": out_dtype,
        "convrot": convrot,
        "convrot_groupsize": convrot_groupsize,
    }
    impl = registry.get_implementation("int8_linear", kwargs=kwargs)
    return impl(**kwargs)


@_op_int8_linear.register_fake
def _op_int8_linear_fake(x, weight, weight_scale, bias, output_dtype_code, convrot=False, convrot_groupsize=256):
    out_dtype = DTYPE_CODE_TO_DTYPE[output_dtype_code]
    return torch.empty(*x.shape[:-1], weight.shape[0], dtype=out_dtype, device=x.device)
