# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

import triton
import triton.language as tl
from comfy_kitchen.backends._modulation import adaln_prep_modulation


@triton.jit
def _adaln_fwd_kernel(
    x_ptr, scale_ptr, shift_ptr, y_ptr,
    d,
    scale_group, shift_group,
    stride_xr, stride_sr, stride_shr, stride_yr,
    eps,
    block_d: tl.constexpr,
    dtype: tl.constexpr,
):
    row = tl.program_id(0)
    x_base  = row * stride_xr
    # scale/shift are passed in distinct-row form and broadcast over contiguous
    # blocks of `*_group` output rows, mirroring the CUDA kernel (row / group).
    s_base  = (row // scale_group) * stride_sr
    sh_base = (row // shift_group) * stride_shr
    y_base  = row * stride_yr

    # Pass 1: mean + variance in one pass (accumulate sum and sum-of-squares),
    # so a single reduction yields both — matching the fused CUDA kernel.
    sum_acc   = tl.zeros([block_d], dtype=tl.float32)
    sumsq_acc = tl.zeros([block_d], dtype=tl.float32)
    for off in range(0, d, block_d):
        cols = off + tl.arange(0, block_d)
        mask = cols < d
        x = tl.load(x_ptr + x_base + cols, mask=mask, other=0.0).to(tl.float32)
        sum_acc   += x
        sumsq_acc += x * x
    mean = tl.sum(sum_acc) / d
    # var = E[x^2] - mean^2, clamped against tiny negative rounding for
    # (near-)constant rows before the rsqrt.
    var  = tl.sum(sumsq_acc) / d - mean * mean
    var  = tl.maximum(var, 0.0)
    rstd = tl.rsqrt(var + eps)

    # Pass 2: normalize + modulate.
    for off in range(0, d, block_d):
        cols = off + tl.arange(0, block_d)
        mask = cols < d
        x  = tl.load(x_ptr + x_base + cols, mask=mask, other=0.0).to(tl.float32)
        sc = tl.load(scale_ptr + s_base + cols, mask=mask, other=0.0).to(tl.float32)
        sh = tl.load(shift_ptr + sh_base + cols, mask=mask, other=0.0).to(tl.float32)
        out = (x - mean) * rstd * (1.0 + sc) + sh
        tl.store(y_ptr + y_base + cols, out.to(dtype), mask=mask)


def adaln(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    orig_shape = x.shape
    d = x.shape[-1]
    n = x.numel() // d

    x_flat = x.reshape(n, d)
    if not x_flat.is_contiguous():
        x_flat = x_flat.contiguous()

    # Broadcast scale/shift to distinct rows + a per-row group, avoiding the
    # expand+copy materialization (matches the CUDA backend).
    scale_flat, scale_group = adaln_prep_modulation(scale, x, n, d)
    shift_flat, shift_group = adaln_prep_modulation(shift, x, n, d)

    out = torch.empty_like(x_flat)
    block_d = min(triton.next_power_of_2(d), 4096)

    dtype_map = {
        torch.float32: tl.float32,
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
    }
    dtype = dtype_map.get(x.dtype, tl.bfloat16)

    _adaln_fwd_kernel[(n,)](
        x_flat, scale_flat, shift_flat, out,
        d,
        scale_group, shift_group,
        x_flat.stride(0), scale_flat.stride(0), shift_flat.stride(0), out.stride(0),
        eps,
        block_d=block_d,
        dtype=dtype,
    )
    return out.reshape(orig_shape)
