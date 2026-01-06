# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Portions of this code are derived from PyTorch AO:
#   https://github.com/pytorch/ao/blob/main/torchao/prototype/mx_formats/nvfp4_tensor.py
#   Copyright (c) Meta Platforms, Inc. and affiliates.
#   Licensed under the BSD 3-Clause License (see NOTICE file for details)

import torch

from comfy_kitchen.float_utils import (
    F4_E2M1_MAX,
    F8_E4M3_MAX,
    F8_E5M2_MAX,
    _f32_to_floatx_unpacked,
    _float8_round,
    _floatx_unpacked_to_f32,
    from_blocked,
    pack_uint4,
    roundup,
    to_blocked,
    unpack_uint4,
)

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

    q_tensor = torch.clamp(x * (1.0 / scale).to(x.dtype), -lp_max, lp_max).to(output_type)
    return q_tensor

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


def dequantize_nvfp4(
    qx: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    block_scales: torch.Tensor,
    output_type: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    # Unpack FP4 data
    data_unpacked = unpack_uint4(qx)  # Returns unpacked uint8 values

    # Convert unpacked FP4 to float32
    # The unpacked values are in E2M1 format (2 exponent bits, 1 mantissa bit)
    data_f32 = _floatx_unpacked_to_f32(data_unpacked, ebits=2, mbits=1)

    # Get original shape (packed tensor has half the columns)
    orig_shape = data_f32.shape
    block_size = 16

    # Reshape to blocks for scaling
    data_reshaped = data_f32.reshape(orig_shape[0], -1, block_size)

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
    total_scale = per_tensor_scale * block_scales_unswizzled.to(torch.float32)

    # Apply scaling to dequantize
    data_dequantized = data_reshaped * total_scale.unsqueeze(-1)

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
    if alpha is None:
        alpha = tensor_scale_a * tensor_scale_b
    should_add_bias_separately = bias is not None

    result = torch._scaled_mm(
        a.view(torch.float4_e2m1fn_x2),
        b.view(torch.float4_e2m1fn_x2).t(),
        block_scale_a.view(-1),
        block_scale_b.view(-1),
        bias=None if should_add_bias_separately else bias,
        out_dtype=out_dtype,
        # scale_result=alpha,  # Not supported yet
    )
    result = result * alpha.to(out_dtype)

    if should_add_bias_separately:
        result = result + bias

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
