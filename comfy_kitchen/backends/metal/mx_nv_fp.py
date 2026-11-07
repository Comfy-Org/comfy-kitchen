# SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MXFP8 + NVFP4 on MPS — TensorOps MXFP8 matmul and fused NVFP4 GEMM."""

from __future__ import annotations

import torch

from comfy_kitchen.backends.eager.quantization import (
    dequantize_mxfp8 as _eager_dequantize_mxfp8,
)
from comfy_kitchen.backends.eager.quantization import (
    dequantize_nvfp4 as _eager_dequantize_nvfp4,
)
from comfy_kitchen.backends.eager.quantization import (
    quantize_mxfp8 as _eager_quantize_mxfp8,
)
from comfy_kitchen.backends.eager.quantization import (
    quantize_nvfp4 as _eager_quantize_nvfp4,
)
from comfy_kitchen.float_utils import from_blocked


def _require_mps(*tensors: torch.Tensor, name: str = "metal") -> None:
    for t in tensors:
        if isinstance(t, torch.Tensor) and t.device.type != "mps":
            raise RuntimeError(f"{name} requires mps tensors")


def _ops():
    from comfy_kitchen.backends.metal._build import get_ops

    return get_ops()


def _unswizzle_mx_scales(block_scales: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    num_blocks = cols // 32
    return from_blocked(
        block_scales.view(torch.uint8),
        num_rows=rows,
        num_cols=num_blocks,
    ).contiguous()


def _unswizzle_nv_scales(block_scales: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    num_blocks = cols // 16
    return from_blocked(
        block_scales,
        num_rows=rows,
        num_cols=num_blocks,
    ).contiguous()


def quantize_mxfp8(x: torch.Tensor, pad_32x: bool = False):
    _require_mps(x, name="metal quantize_mxfp8")
    return _eager_quantize_mxfp8(x, pad_32x=pad_32x)


def dequantize_mxfp8(
    qx: torch.Tensor,
    block_scales: torch.Tensor,
    output_type: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    _require_mps(qx, block_scales, name="metal dequantize_mxfp8")
    return _eager_dequantize_mxfp8(qx, block_scales, output_type=output_type)


def scaled_mm_mxfp8(
    a: torch.Tensor,
    b: torch.Tensor,
    block_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    _require_mps(a, b, block_scale_a, block_scale_b, name="metal scaled_mm_mxfp8")
    od = out_dtype or torch.float16
    if od != torch.float16:
        raise RuntimeError("metal scaled_mm_mxfp8 requires out_dtype=float16")
    if a.shape[1] % 32 != 0:
        raise RuntimeError("metal scaled_mm_mxfp8 requires K % 32 == 0")

    ops = _ops()
    if not ops.mxfp8_tensorops_available():
        raise RuntimeError("metal scaled_mm_mxfp8 TensorOps kernel unavailable")

    sa = _unswizzle_mx_scales(block_scale_a, a.shape[0], a.shape[1])
    sb = _unswizzle_mx_scales(block_scale_b, b.shape[0], b.shape[1])
    a_u8 = a.view(torch.uint8) if a.dtype != torch.uint8 else a
    b_u8 = b.view(torch.uint8) if b.dtype != torch.uint8 else b
    out = ops.scaled_mm_mxfp8_tensorops(
        a_u8.contiguous(),
        b_u8.contiguous(),
        sa,
        sb,
    )
    if bias is not None:
        out = out + bias.to(device=out.device, dtype=out.dtype)
    return out


def quantize_nvfp4(
    x: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    epsilon: float = 0.0,
    pad_16x: bool = False,
    hi_first: bool = True,
):
    _require_mps(x, per_tensor_scale, name="metal quantize_nvfp4")
    return _eager_quantize_nvfp4(
        x, per_tensor_scale, epsilon=epsilon, pad_16x=pad_16x, hi_first=hi_first
    )


def dequantize_nvfp4(
    qx: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    block_scales: torch.Tensor,
    output_type: torch.dtype = torch.bfloat16,
    hi_first: bool = True,
) -> torch.Tensor:
    _require_mps(qx, per_tensor_scale, block_scales, name="metal dequantize_nvfp4")
    return _eager_dequantize_nvfp4(
        qx, per_tensor_scale, block_scales, output_type=output_type, hi_first=hi_first
    )


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
    """Fused NVFP4 GEMM: on-the-fly E2M1 unpack + scale inside the Metal K-loop."""
    _require_mps(
        a, b, tensor_scale_a, tensor_scale_b, block_scale_a, block_scale_b,
        name="metal scaled_mm_nvfp4",
    )
    od = out_dtype or torch.float16
    if od != torch.float16:
        raise RuntimeError("metal scaled_mm_nvfp4 requires out_dtype=float16")

    k = a.shape[1] * 2
    sa = _unswizzle_nv_scales(block_scale_a, a.shape[0], k).view(torch.uint8).contiguous()
    sb = _unswizzle_nv_scales(block_scale_b, b.shape[0], k).view(torch.uint8).contiguous()
    tsa = float(tensor_scale_a.reshape(-1)[0].item())
    tsb = float(tensor_scale_b.reshape(-1)[0].item())

    out = _ops().scaled_mm_nvfp4_fused(
        a.contiguous().view(torch.uint8),
        b.contiguous().view(torch.uint8),
        sa,
        sb,
        tsa,
        tsb,
        True,  # kitchen default hi_first
    )
    if alpha is not None:
        out = out * alpha.to(device=out.device, dtype=out.dtype)
    if bias is not None:
        out = out + bias.to(device=out.device, dtype=out.dtype)
    return out
