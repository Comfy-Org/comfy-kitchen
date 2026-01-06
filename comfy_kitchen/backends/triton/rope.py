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


@triton.jit
def apply_rope_kernel(
    xq_ptr,
    xk_ptr,
    freqs_ptr,
    xq_out_ptr,
    xk_out_ptr,
    batch,
    dim1,
    dim2,
    head_dim,
    freqs_batch,
    freqs_dim1,
    freqs_dim2,
    stride_x_batch,
    stride_x_dim1,
    stride_x_dim2,
    stride_x_dim,
    stride_freqs_batch,
    stride_freqs_dim1,
    stride_freqs_dim2,
    stride_freqs_dim,
    stride_freqs_rot,
    stride_freqs_pair,
    compute_dtype: tl.constexpr,
    block_size: tl.constexpr,
):
    """Triton kernel for RoPE (Rotary Position Embedding) with flexible layout.

    Args:
        xq_ptr: Query tensor pointer (batch, dim1, dim2, head_dim)
        xk_ptr: Key tensor pointer (batch, dim1, dim2, head_dim)
        freqs_ptr: Frequency tensor pointer (batch, dim1, dim2, head_dim//2, 2, 2)
        xq_out_ptr: Output query tensor pointer
        xk_out_ptr: Output key tensor pointer
        batch, dim1, dim2, head_dim: Tensor dimensions (dim1/dim2 can be heads/seq in any order)
        freqs_batch, freqs_dim1, freqs_dim2: Frequency tensor dimensions (1 for broadcasting)
        stride_*: Stride information for memory access (enables layout flexibility)
        compute_dtype: Data type for computation (from freqs_cis)
        block_size: Number of elements to process per block
    """
    # Get program ID - each program processes a chunk of the output
    pid = tl.program_id(0)

    # Calculate total number of pairs to process
    n_pairs = head_dim // 2
    total_elements = batch * dim1 * dim2 * n_pairs

    # Calculate the starting index for this program
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < total_elements

    # Decompose linear index into (batch_idx, dim1_idx, dim2_idx, pair_idx)
    pair_idx = offsets % n_pairs
    temp = offsets // n_pairs
    dim2_idx = temp % dim2
    temp = temp // dim2
    dim1_idx = temp % dim1
    batch_idx = temp // dim1

    # Calculate indices for the two elements in each pair
    dim_idx_0 = pair_idx * 2
    dim_idx_1 = pair_idx * 2 + 1

    # Calculate offsets for xq and xk using strides (layout-agnostic)
    x_offset_0 = (batch_idx * stride_x_batch +
                   dim1_idx * stride_x_dim1 +
                   dim2_idx * stride_x_dim2 +
                   dim_idx_0 * stride_x_dim)
    x_offset_1 = (batch_idx * stride_x_batch +
                   dim1_idx * stride_x_dim1 +
                   dim2_idx * stride_x_dim2 +
                   dim_idx_1 * stride_x_dim)

    # Handle broadcasting for freqs_cis (all spatial dimensions)
    freqs_batch_idx = tl.where(freqs_batch == 1, 0, batch_idx)
    freqs_dim1_idx = tl.where(freqs_dim1 == 1, 0, dim1_idx)
    freqs_dim2_idx = tl.where(freqs_dim2 == 1, 0, dim2_idx)

    # Calculate offsets for freqs_cis using its strides
    freqs_base = (freqs_batch_idx * stride_freqs_batch +
                  freqs_dim1_idx * stride_freqs_dim1 +
                  freqs_dim2_idx * stride_freqs_dim2 +
                  pair_idx * stride_freqs_dim)

    # Load rotation matrix elements (shared for both xq and xk)
    freqs_00_offset = freqs_base + 0 * stride_freqs_rot + 0 * stride_freqs_pair
    freqs_01_offset = freqs_base + 0 * stride_freqs_rot + 1 * stride_freqs_pair
    freqs_10_offset = freqs_base + 1 * stride_freqs_rot + 0 * stride_freqs_pair
    freqs_11_offset = freqs_base + 1 * stride_freqs_rot + 1 * stride_freqs_pair

    freqs_00 = tl.load(freqs_ptr + freqs_00_offset, mask=mask, other=0.0)
    freqs_01 = tl.load(freqs_ptr + freqs_01_offset, mask=mask, other=0.0)
    freqs_10 = tl.load(freqs_ptr + freqs_10_offset, mask=mask, other=0.0)
    freqs_11 = tl.load(freqs_ptr + freqs_11_offset, mask=mask, other=0.0)

    _apply_freq_tile(xq_ptr, xq_out_ptr, mask, freqs_00, freqs_01, freqs_10, freqs_11, x_offset_0, x_offset_1, compute_dtype)
    if xk_ptr is not None:
        _apply_freq_tile(xk_ptr, xk_out_ptr, mask, freqs_00, freqs_01, freqs_10, freqs_11, x_offset_0, x_offset_1, compute_dtype)

@triton.jit
def _apply_freq_tile(x_ptr, x_out_ptr, mask, freqs_00, freqs_01, freqs_10, freqs_11, x_offset_0, x_offset_1, compute_dtype):
    # Load xq values and cast to computation dtype
    x_0 = tl.load(x_ptr + x_offset_0, mask=mask, other=0.0).to(compute_dtype)
    x_1 = tl.load(x_ptr + x_offset_1, mask=mask, other=0.0).to(compute_dtype)

    # Apply rotation to xq
    xq_out_0 = freqs_00 * x_0 + freqs_01 * x_1
    xq_out_1 = freqs_10 * x_0 + freqs_11 * x_1

    # Store results
    tl.store(x_out_ptr + x_offset_0, xq_out_0, mask=mask)
    tl.store(x_out_ptr + x_offset_1, xq_out_1, mask=mask)


def _apply_rope(x1: torch.Tensor, freqs_cis: torch.Tensor, x2: torch.Tensor = None):
    batch, dim1, dim2, head_dim = x1.shape
    freqs_batch, freqs_dim1, freqs_dim2 = freqs_cis.shape[0], freqs_cis.shape[1], freqs_cis.shape[2]

    # Ensure inputs are contiguous
    if not x1.is_contiguous():
        x1 = x1.contiguous()
    if not freqs_cis.is_contiguous():
        freqs_cis = freqs_cis.contiguous()

    x1_out = torch.empty_like(x1)
    x2_out = None

    # Calculate total number of pairs to process
    n_pairs = head_dim // 2
    total_elements = batch * dim1 * dim2 * n_pairs

    # Choose block size based on tensor size
    if total_elements < 4096:
        block_size = 256
    elif total_elements < 32768:
        block_size = 512
    else:
        block_size = 1024

    # Calculate grid size
    grid = (triton.cdiv(total_elements, block_size),)

    # Get strides - these automatically adapt to the layout (BHND or BNHD)
    stride_x_batch, stride_x_dim1, stride_x_dim2, stride_x_dim = x1.stride()
    stride_freqs = freqs_cis.stride()

    # Map dtype to Triton dtype
    dtype_map = {
        torch.float32: tl.float32,
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
    }
    compute_dtype = dtype_map.get(freqs_cis.dtype, tl.float32)

    if x2 is not None:
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        x2_out = torch.empty_like(x2)

    apply_rope_kernel[grid](
        x1,
        x2,
        freqs_cis,
        x1_out,
        x2_out,
        batch,
        dim1,
        dim2,
        head_dim,
        freqs_batch,
        freqs_dim1,
        freqs_dim2,
        stride_x_batch,
        stride_x_dim1,
        stride_x_dim2,
        stride_x_dim,
        stride_freqs[0],  # batch
        stride_freqs[1],  # dim1
        stride_freqs[2],  # dim2
        stride_freqs[3],  # dim (pairs)
        stride_freqs[4],  # rotation component (2)
        stride_freqs[5],  # pair element (2)
        compute_dtype=compute_dtype,
        block_size=block_size,
    )

    return x1_out, x2_out


def apply_rope1(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    x_out, _ = _apply_rope(x, freqs_cis)
    return x_out


def apply_rope(
        xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return _apply_rope(xq, freqs_cis, xk)
