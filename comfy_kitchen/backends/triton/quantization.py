# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

import triton
import triton.language as tl
from comfy_kitchen.float_utils import (
    F8_E4M3_MAX,
    F8_E5M2_MAX,
    ceil_div,
)


@triton.jit
def quantize_fp8_kernel_tl(
    x_ptr,
    output_ptr,
    scale_ptr,
    lp_max,
    n_elements,
    block_size: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < n_elements

    # Load scale value from device tensor
    scale = tl.load(scale_ptr)

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    scaled = x.to(tl.float32) / scale

    clamped = tl.maximum(tl.minimum(scaled, lp_max), -lp_max)
    tl.store(output_ptr + offsets, clamped, mask=mask)


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

    if not x.is_contiguous():
        x = x.contiguous()

    orig_shape = x.shape
    x_flat = x.flatten()
    n_elements = x_flat.numel()

    output = torch.empty_like(x_flat, dtype=output_type)


    if n_elements < 32768:  # < 32K elements
        block_size = 128
    elif n_elements < 131072:  # < 128K elements
        block_size = 256
    elif n_elements < 524288:  # < 512K elements
        block_size = 512
    else:
        block_size = 1024

    grid = (triton.cdiv(n_elements, block_size),)

    quantize_fp8_kernel_tl[grid](
        x_flat,
        output,
        scale,
        lp_max,
        n_elements,
        block_size=block_size,
    )

    output = output.view(orig_shape)

    return output


@triton.jit
def dequantize_fp8_kernel_tl(
    x_ptr,
    output_ptr,
    scale_ptr,
    n_elements,
    block_size: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < n_elements

    # Load scale value from device tensor
    scale = tl.load(scale_ptr)

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    dequantized = x.to(tl.float32) * scale

    tl.store(output_ptr + offsets, dequantized, mask=mask)


def dequantize_per_tensor_fp8(
    x: torch.Tensor, scale: torch.Tensor, output_type: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    if not x.is_contiguous():
        x = x.contiguous()

    orig_shape = x.shape
    x_flat = x.flatten()
    n_elements = x_flat.numel()

    output = torch.empty_like(x_flat, dtype=output_type)

    if n_elements < 32768:  # < 32K elements
        block_size = 128
    elif n_elements < 131072:  # < 128K elements
        block_size = 256
    elif n_elements < 524288:  # < 512K elements
        block_size = 512
    else:
        block_size = 1024

    grid = (triton.cdiv(n_elements, block_size),)

    dequantize_fp8_kernel_tl[grid](
        x_flat,
        output,
        scale,
        n_elements,
        block_size=block_size,
    )

    output = output.view(orig_shape)

    return output


@triton.jit
def _compute_swizzled_scale_offset(
    in_row,
    in_col,
    n_col_blocks,
    padded_scale_cols,
):
    """Compute the swizzled offset for a scale at logical position (in_row, in_col).

    This implements the cuBLAS blocked layout transformation (to_blocked).
    Used by both quantize (write) and dequantize (read) kernels.
    """
    # Compute which 128x4 block this element belongs to
    row_block = in_row // 128
    col_block = in_col // 4

    # Position within the 128x4 block
    in_block_row = in_row % 128
    in_block_col = in_col % 4

    # Map through the swizzle transformations
    sub_block = in_block_row // 32
    fine_row = in_block_row % 32

    combined_block = row_block * n_col_blocks + col_block
    intermediate_col = sub_block * 4 + in_block_col

    # Flatten intermediate and compute linear index
    linear_idx = combined_block * 512 + fine_row * 16 + intermediate_col

    # Convert to final 2D position and compute offset
    out_row = linear_idx // padded_scale_cols
    out_col = linear_idx % padded_scale_cols
    return out_row * padded_scale_cols + out_col


@triton.jit
def quantize_nvfp4_kernel_tl(
    x_ptr,
    packed_output_ptr,
    swizzled_scales_ptr,
    per_tensor_scale_ptr,
    m,
    n,
    num_blocks,
    scale_rows,
    scale_cols,
    padded_scale_rows,
    padded_scale_cols,
    block_size: tl.constexpr,
    blocks_per_program: tl.constexpr,
):
    """Single Triton kernel for NVFP4 quantization with packing and scale swizzling.

    Performs all operations in one kernel:
    1. Computes block-wise scales
    2. Quantizes and packs data to FP4
    3. Applies to_blocked swizzle pattern to scales

    Optimized with:
    - Vectorized processing of multiple blocks per thread
    - Efficient packing using interleave operations
    - Coalesced memory accesses

    Args:
        x_ptr: Input tensor pointer (m x n)
        packed_output_ptr: Output packed FP4 data (m x n//2)
        swizzled_scales_ptr: Output swizzled FP8 block scales (padded_scale_rows x padded_scale_cols)
        per_tensor_scale_ptr: Pointer to global scaling factor tensor
        m: Number of rows in input
        n: Number of columns in input (must be divisible by block_size)
        num_blocks: Number of blocks per row (n // block_size)
        scale_rows: Unpadded scale rows (m)
        scale_cols: Unpadded scale cols (num_blocks)
        padded_scale_rows: Padded scale rows for swizzle
        padded_scale_cols: Padded scale cols for swizzle
        block_size: Size of each quantization block (typically 16)
        blocks_per_program: Number of blocks to process per program
    """
    # Get program IDs - each program processes blocks_per_program blocks
    pid_m = tl.program_id(axis=0)
    pid_n_base = tl.program_id(axis=1) * blocks_per_program

    # Load per-tensor scale value from device tensor
    per_tensor_scale = tl.load(per_tensor_scale_ptr)

    # Process multiple blocks per program
    for block_offset in range(blocks_per_program):
        pid_n = pid_n_base + block_offset

        # Skip if beyond num_blocks
        if pid_n < num_blocks:
            # Calculate offsets for the input data block
            offs_n = pid_n * block_size + tl.arange(0, block_size)
            mask = offs_n < n
            x_offs = pid_m * n + offs_n

            # Load input data block
            x = tl.load(x_ptr + x_offs, mask=mask, other=0.0).to(tl.float32)

            # Compute block-wise absolute maximum
            x_abs = tl.abs(x)
            max_abs = tl.max(x_abs, axis=0)

            # Calculate block scale: block_scale = max_abs / F4_E2M1_MAX (6.0)
            block_scale = max_abs / 6.0

            # Scale block scale to FP8
            scaled_block_scale = block_scale / per_tensor_scale
            # Clamp to [F8_E4M3_EPS, F8_E4M3_MAX] = [0.125, 448.0]
            scaled_block_scale = tl.maximum(scaled_block_scale, 0.125)
            scaled_block_scale = tl.minimum(scaled_block_scale, 448.0)

            # Round to FP8 precision
            scaled_block_scale_fp8 = scaled_block_scale.to(tl.float8e4nv)

            # Compute swizzled position and store scale
            n_col_blocks = tl.cdiv(scale_cols, 4)
            swizzled_offs = _compute_swizzled_scale_offset(
                pid_m, pid_n, n_col_blocks, padded_scale_cols
            )
            if pid_m < scale_rows and pid_n < scale_cols:
                tl.store(swizzled_scales_ptr + swizzled_offs, scaled_block_scale_fp8)

            # Calculate total scale for data quantization
            scaled_block_scale_fp32 = scaled_block_scale_fp8.to(tl.float32)
            total_scale = per_tensor_scale * scaled_block_scale_fp32
            total_scale = tl.where(total_scale < 1e-10, 1.0, total_scale)

            # Scale and clamp data
            data_scaled = x / total_scale
            data_scaled = tl.maximum(data_scaled, -6.0)  # -F4_E2M1_MAX
            data_scaled = tl.minimum(data_scaled, 6.0)   # F4_E2M1_MAX

            # Quantize to FP4 values and pack - optimized version
            # Convert all values to FP4 representation
            sign_all = tl.where(data_scaled < 0, 1, 0)
            abs_all = tl.abs(data_scaled)

            # Map all to FP4 bit pattern (E2M1)
            q_all = tl.where(abs_all <= 0.25, 0,
                     tl.where(abs_all < 0.75, 1,
                     tl.where(abs_all <= 1.25, 2,
                     tl.where(abs_all < 1.75, 3,
                     tl.where(abs_all <= 2.5, 4,
                     tl.where(abs_all < 3.5, 5,
                     tl.where(abs_all <= 5.0, 6, 7)))))))

            # Add sign bits to all values
            fp4_all = (sign_all.to(tl.int32) << 3) | q_all.to(tl.int32)

            # Pack consecutive pairs of FP4 values
            # fp4_all has 16 elements: [v0, v1, v2, v3, ..., v15]
            # We want to pack: (v0,v1), (v2,v3), ..., (v14,v15)
            pair_idx = tl.arange(0, block_size // 2)
            even_idx = pair_idx * 2
            odd_idx = pair_idx * 2 + 1

            # Extract even and odd elements using one-hot selection
            indices = tl.arange(0, block_size)
            fp4_even = tl.sum(tl.where(indices == even_idx[:, None], fp4_all, 0), axis=1)
            fp4_odd = tl.sum(tl.where(indices == odd_idx[:, None], fp4_all, 0), axis=1)

            # Pack two 4-bit values into one uint8
            packed_bytes = ((fp4_even << 4) | fp4_odd).to(tl.uint8)

            # Store packed bytes
            out_offs = pid_m * (n // 2) + pid_n * (block_size // 2) + pair_idx
            out_mask = (pid_n * block_size + even_idx) < n
            tl.store(packed_output_ptr + out_offs, packed_bytes, mask=out_mask)


def quantize_nvfp4(
    x: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    epsilon: float = 0.0,
    pad_16x: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Note: epsilon is accepted for API compatibility but not currently used
    orig_shape = x.shape

    # Handle padding
    if pad_16x:
        rows, cols = x.shape
        pad_rows = (rows + 15) // 16 * 16 - rows
        pad_cols = (cols + 15) // 16 * 16 - cols
        if pad_rows > 0 or pad_cols > 0:
            x = torch.nn.functional.pad(x, (0, pad_cols, 0, pad_rows))
            # Note: We update orig_shape because the output tensor logic below assumes x.shape matches
            # what we want to produce. If we pad here, we want the padded output.
            orig_shape = x.shape

    block_size = 16

    # Reshape for block processing
    x = x.reshape(orig_shape[0], -1, block_size)
    m, num_blocks, _ = x.shape
    n = num_blocks * block_size

    # Flatten to 2D for kernel processing
    x_2d = x.reshape(m, n).contiguous()

    # Calculate swizzled scale dimensions
    scale_rows = m
    scale_cols = num_blocks
    n_row_blocks = ceil_div(scale_rows, 128)
    n_col_blocks = ceil_div(scale_cols, 4)
    padded_scale_rows = n_row_blocks * 128
    padded_scale_cols = n_col_blocks * 4

    # Allocate output tensors
    packed_output = torch.empty((m, n // 2), dtype=torch.uint8, device=x.device)
    # Use zeros for scales to avoid garbage in padded regions
    swizzled_scales = torch.zeros(
        (padded_scale_rows, padded_scale_cols),
        dtype=torch.float8_e4m3fn,
        device=x.device
    )

    # Determine blocks per program based on tensor size for better occupancy
    total_blocks = m * num_blocks
    if total_blocks < 1024:
        blocks_per_program = 1
    elif total_blocks < 4096:
        blocks_per_program = 2
    else:
        blocks_per_program = 4

    # Launch single kernel that does everything
    grid = (m, triton.cdiv(num_blocks, blocks_per_program))

    quantize_nvfp4_kernel_tl[grid](
        x_2d,
        packed_output,
        swizzled_scales,
        per_tensor_scale,
        m,
        n,
        num_blocks,
        scale_rows,
        scale_cols,
        padded_scale_rows,
        padded_scale_cols,
        block_size=block_size,
        blocks_per_program=blocks_per_program,
    )

    # Reshape packed output to original shape (with last dim halved)
    packed_shape = list(orig_shape)
    packed_shape[-1] = packed_shape[-1] // 2
    packed_output = packed_output.reshape(packed_shape)

    return packed_output, swizzled_scales


@triton.jit
def dequantize_nvfp4_kernel_tl(
    packed_ptr,
    scale_ptr,
    global_scale_ptr,
    output_ptr,
    n,
    scale_cols,
    n_col_blocks,
    padded_scale_cols,
    block_size: tl.constexpr,
    tile_size: tl.constexpr,
):
    """Dequantizes FP4 packed data using per-block scaling factors.

    Args:
        packed_ptr (tl.pointer): Pointer to packed uint8 tensor (m x n//2)
        scale_ptr (tl.pointer): Pointer to swizzled per-block scale tensor
        output_ptr (tl.pointer): Pointer to output tensor (m x n)
        global_scale_ptr (tl.pointer): Pointer to global scale tensor
        n (int): Number of columns in unpacked tensor
        scale_cols (int): Number of scale columns (n // block_size)
        n_col_blocks (int): Number of 4-column blocks in scales
        padded_scale_cols (int): Padded scale columns (n_col_blocks * 4)
        block_size (tl.constexpr): Size of each FP4 quantization block
        tile_size (tl.constexpr): Size of the processing tile (in packed elements)
    """
    # Get program ID for processing packed elements
    pid = tl.program_id(0)

    # Calculate packed element offsets (each packed element contains 2 FP4 values)
    packed_start = pid * tile_size
    packed_offs = packed_start + tl.arange(0, tile_size)

    # Calculate 2D coordinates for packed data
    packed_row_idx = packed_offs // (n // 2)
    packed_col_idx = packed_offs % (n // 2)

    # Create mask for packed data bounds checking
    packed_mask = packed_col_idx < (n // 2)

    # Load global scale
    global_scale = tl.load(global_scale_ptr)

    # Load packed data
    packed_data = tl.load(packed_ptr + packed_offs, mask=packed_mask, other=0)

    # Unpack packed FP4 values (uint8) to float16x2
    x_f16x2_packed = tl.inline_asm_elementwise(
        asm="""
        {
            .reg .b8 byte0, byte1, byte2, byte3;
            mov.b32 {byte0, byte1, byte2, byte3}, $4;
            cvt.rn.f16x2.e2m1x2 $0, byte0;
            cvt.rn.f16x2.e2m1x2 $1, byte1;
            cvt.rn.f16x2.e2m1x2 $2, byte2;
            cvt.rn.f16x2.e2m1x2 $3, byte3;
        }
        """,
        constraints="=r,=r,=r,=r,r",
        args=[packed_data],
        dtype=tl.uint32,
        is_pure=True,
        pack=4,
    )
    val_low = (
        (x_f16x2_packed & 0xFFFF).cast(tl.uint16).cast(tl.float16, bitcast=True).cast(tl.float32)
    )
    val_high = (
        (x_f16x2_packed >> 16).cast(tl.uint16).cast(tl.float16, bitcast=True).cast(tl.float32)
    )

    # Calculate output positions for both values
    out_col_low = packed_col_idx * 2
    out_col_high = packed_col_idx * 2 + 1
    out_offs_low = packed_row_idx * n + out_col_low
    out_offs_high = packed_row_idx * n + out_col_high

    # Calculate block indices for scaling (logical positions in scale tensor)
    block_col_low = out_col_low // block_size
    block_col_high = out_col_high // block_size

    # Compute swizzled offsets for scale lookups
    scale_offs_low = _compute_swizzled_scale_offset(
        packed_row_idx, block_col_low, n_col_blocks, padded_scale_cols
    )
    scale_offs_high = _compute_swizzled_scale_offset(
        packed_row_idx, block_col_high, n_col_blocks, padded_scale_cols
    )

    # Load scaling factors from swizzled positions
    scale_low = tl.load(
        scale_ptr + scale_offs_low,
        mask=packed_mask & (block_col_low < scale_cols),
        other=1.0,
    )
    scale_high = tl.load(
        scale_ptr + scale_offs_high,
        mask=packed_mask & (block_col_high < scale_cols),
        other=1.0,
    )

    # Apply scaling
    # Note: Due to packing order ((even << 4) | odd) and PTX cvt.rn.f16x2.e2m1x2:
    # - val_low (from low nibble) contains odd-indexed values
    # - val_high (from high nibble) contains even-indexed values
    result_even = val_high * scale_low.to(tl.float32) * global_scale
    result_odd = val_low * scale_high.to(tl.float32) * global_scale

    # Store results
    out_mask_low = packed_mask & (out_col_low < n)
    out_mask_high = packed_mask & (out_col_high < n)

    tl.store(output_ptr + out_offs_low, result_even, mask=out_mask_low)
    tl.store(output_ptr + out_offs_high, result_odd, mask=out_mask_high)


def dequantize_nvfp4(
    qx: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    block_scales: torch.Tensor,
    output_type: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    # Triton backend: fused kernel with inline SM100 cvt.rn.f16x2.e2m1x2 instruction
    block_size = 16
    tile_size = 128

    packed_n = qx.shape[-1]
    n = packed_n * 2
    scale_cols = n // block_size

    # Compute swizzle layout parameters
    n_col_blocks = ceil_div(scale_cols, 4)
    padded_scale_cols = n_col_blocks * 4

    # Create output tensor with proper shape handling
    output_shape = list(qx.shape)
    output_shape[-1] = n
    output = torch.empty(output_shape, dtype=output_type, device=qx.device)

    # Calculate total number of elements and grid size
    def grid(meta):
        return (triton.cdiv(qx.numel(), meta["tile_size"]),)

    dequantize_nvfp4_kernel_tl[grid](
        qx,
        block_scales,
        per_tensor_scale,
        output,
        n,
        scale_cols,
        n_col_blocks,
        padded_scale_cols,
        block_size=block_size,
        tile_size=tile_size,
    )

    return output
