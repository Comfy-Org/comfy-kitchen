# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton AsymW4A8Int8 linear (AMD/CUDA) -- the accelerated path where the compiled
CUDA backend is unavailable. A fused kernel dequantizes int4->grouped int8 in one launch
(matching CUDA), feeding the Triton INT8 GEMM.
"""
from __future__ import annotations

import torch

import triton
import triton.language as tl
from comfy_kitchen.backends.eager.w4a8_int8 import (
    validate_w4a8_operands,
)
from comfy_kitchen.backends.eager.w4a8_int8 import (
    w4a8_int8_linear as eager_w4a8_int8_linear,
)
from triton.language.extra import libdevice

from .quantization import int8_linear


@triton.jit
def _dequant_int4_grouped_to_int8_kernel(
    qdata_ptr,   # [n, k/2] packed uint4 (int8 storage): even col=low nibble, odd=high
    srel_ptr,    # [n, groups] fp32 per-group scale (fp8 decoded to fp32 by the wrapper)
    cb_ptr,      # [16] fp32 codebook (unused when has_cb is False)
    out_ptr,     # [n, k] int8 output
    n, k, group_size,
    stride_qn, stride_qk,
    stride_sn, stride_sg,
    stride_on, stride_ok,
    has_cb: tl.constexpr,
    block_n: tl.constexpr,
    block_kh: tl.constexpr,  # packed byte-columns per tile (each -> 2 output columns)
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    k_half = k // 2

    rows = pid_n * block_n + tl.arange(0, block_n)          # [block_n]
    bcols = pid_k * block_kh + tl.arange(0, block_kh)       # [block_kh] packed byte cols
    m = (rows[:, None] < n) & (bcols[None, :] < k_half)     # [block_n, block_kh]

    byte = tl.load(qdata_ptr + rows[:, None] * stride_qn + bcols[None, :] * stride_qk,
                   mask=m, other=0).to(tl.int32) & 0xFF
    low = byte & 0xF
    high = (byte >> 4) & 0xF

    # even group_size => a byte's two nibbles share a group -> one scale load per byte
    grp = (2 * bcols) // group_size                         # [block_kh]
    s = tl.load(srel_ptr + rows[:, None] * stride_sn + grp[None, :] * stride_sg,
                mask=m, other=0.0)                          # [block_n, block_kh] fp32

    if has_cb:
        v_low = tl.load(cb_ptr + low)                       # gather (indices always 0..15)
        v_high = tl.load(cb_ptr + high)
    else:
        v_low = (low - 8).to(tl.float32)                    # symmetric uniform levels
        v_high = (high - 8).to(tl.float32)

    q_low = tl.clamp(libdevice.rint(v_low * s), -127.0, 127.0).to(tl.int8)
    q_high = tl.clamp(libdevice.rint(v_high * s), -127.0, 127.0).to(tl.int8)

    tl.store(out_ptr + rows[:, None] * stride_on + (2 * bcols)[None, :] * stride_ok, q_low, mask=m)
    tl.store(out_ptr + rows[:, None] * stride_on + (2 * bcols + 1)[None, :] * stride_ok, q_high, mask=m)


def _dequant_int4_grouped_to_int8(
    qdata: torch.Tensor,
    s_rel: torch.Tensor,
    codebook: torch.Tensor | None,
    group_size: int,
) -> torch.Tensor:
    """Fused Triton int4 -> grouped int8: round(clamp(level(q) * s_rel, -127, 127))."""
    n, k_half = qdata.shape
    k = k_half * 2
    qi = qdata.contiguous()
    srel_f = s_rel.float().contiguous()  # decode fp8 scale to fp32 (small tensor)
    has_cb = codebook is not None
    cb = (codebook.float().contiguous() if has_cb
          else torch.empty(16, device=qdata.device, dtype=torch.float32))
    out = torch.empty(n, k, dtype=torch.int8, device=qdata.device)

    block_n, block_kh = 32, 128
    grid = (triton.cdiv(n, block_n), triton.cdiv(k_half, block_kh))
    _dequant_int4_grouped_to_int8_kernel[grid](
        qi, srel_f, cb, out,
        n, k, group_size,
        qi.stride(0), qi.stride(1),
        srel_f.stride(0), srel_f.stride(1),
        out.stride(0), out.stride(1),
        has_cb=has_cb, block_n=block_n, block_kh=block_kh,
    )
    return out


def w4a8_int8_linear(
    x: torch.Tensor,
    qdata: torch.Tensor,
    s_rel: torch.Tensor,
    s_channel: torch.Tensor,
    codebook: torch.Tensor | None = None,
    correction: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    group_size: int = 16,
    convrot_groupsize: int = 256,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """``x @ W.T + bias`` for AsymW4A8Int8 via the fused Triton dequant + INT8 GEMM."""
    validate_w4a8_operands(
        qdata,
        s_rel,
        s_channel,
        codebook,
        correction,
        group_size,
        convrot_groupsize,
    )
    if x.shape[-1] != qdata.shape[-1] * 2:
        raise ValueError(
            f"Input K={x.shape[-1]} does not match qdata K={qdata.shape[-1] * 2}"
        )
    if correction is not None:
        return eager_w4a8_int8_linear(
            x,
            qdata,
            s_rel,
            s_channel,
            codebook=codebook,
            correction=correction,
            bias=bias,
            group_size=group_size,
            convrot_groupsize=convrot_groupsize,
            out_dtype=out_dtype,
        )

    int8_w = _dequant_int4_grouped_to_int8(qdata, s_rel, codebook, group_size)
    return int8_linear(
        x, int8_w, s_channel, bias=bias, out_dtype=out_dtype,
        convrot=True, convrot_groupsize=convrot_groupsize,
    )
