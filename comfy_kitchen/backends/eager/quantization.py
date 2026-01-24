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
from comfy_kitchen.scaled_mm_v2 import ScalingType, SwizzleType, scaled_mm_v2

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

def quantize_nvfp4(
    x: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    epsilon: float = 0.0,
    pad_16x: bool = False,
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
    data_lp = pack_uint4(data_lp)
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
) -> torch.Tensor:
    lut = E2M1_LUT_CACHE.get((qx.device, output_type))
    if lut is None:
        lut = E2M1_LUT.to(qx.device, output_type)
        E2M1_LUT_CACHE[(qx.device, output_type)] = lut

    lo = qx & 0x0F
    hi = qx >> 4
    out = torch.stack([hi, lo], dim=-1).view(*qx.shape[:-1], -1)
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
    from comfy_kitchen.registry import registry

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
    from comfy_kitchen.registry import registry

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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to NVFP4 format with block-wise scaling.

    Args:
        x: Input tensor (2D)
        per_tensor_scale: Global scale factor
        epsilon: Epsilon for numerical stability
        pad_16x: If True, pad dimensions to be divisible by 16

    Returns:
        Tuple of (quantized_tensor, block_scales)
    """
    from comfy_kitchen.registry import registry

    kwargs = {"x": x, "per_tensor_scale": per_tensor_scale, "epsilon": epsilon, "pad_16x": pad_16x}
    impl = registry.get_implementation("quantize_nvfp4", kwargs=kwargs)
    return impl(**kwargs)


@_op_quantize_nvfp4.register_fake
def _op_quantize_nvfp4_fake(x, per_tensor_scale, epsilon, pad_16x):
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
) -> torch.Tensor:
    """Dequantize tensor from NVFP4 format with block-wise scaling.

    Args:
        qx: Quantized FP4 tensor (packed as uint8)
        per_tensor_scale: Global scale factor
        block_scales: Block scales in swizzled layout (float8_e4m3fn)
        output_dtype_code: Target dtype code (0=float32, 1=float16, 2=bfloat16)

    Returns:
        Dequantized tensor in specified output format
    """
    from comfy_kitchen.registry import registry

    output_dtype = DTYPE_CODE_TO_DTYPE[output_dtype_code]
    kwargs = {"qx": qx, "per_tensor_scale": per_tensor_scale, "block_scales": block_scales, "output_type": output_dtype}
    impl = registry.get_implementation("dequantize_nvfp4", kwargs=kwargs)
    return impl(**kwargs)


@_op_dequantize_nvfp4.register_fake
def _op_dequantize_nvfp4_fake(qx, per_tensor_scale, block_scales, output_dtype_code):
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
    from comfy_kitchen.registry import registry

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
    from comfy_kitchen.registry import registry

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
    from comfy_kitchen.registry import registry

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
    from comfy_kitchen.registry import registry

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
