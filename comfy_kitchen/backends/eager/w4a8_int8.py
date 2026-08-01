# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Eager (pure-torch) AsymW4A8Int8 path -- portable fallback for any torch device
(CPU/CUDA/ROCm). Bit-exact int4->int8 dequant feeding the shared ``int8_linear``.
"""
from __future__ import annotations

import torch

from .quantization import int8_linear


def _dequant_int4_grouped_to_int8(
    qdata: torch.Tensor,      # [N, K/2] packed uint4 (even col=low nibble)
    s_rel: torch.Tensor,      # [N, K/group] per-group scale (fp8 or fp32)
    codebook: torch.Tensor | None,  # [16] levels, or None for uniform (q-8)
    group_size: int,
) -> torch.Tensor:
    """int4 -> grouped int8: round(clamp(level(q) * s_rel, -127, 127)). Per-group scale
    folded in, per-channel scale left for the GEMM epilogue. Bit-exact with CUDA."""
    n, k_half = qdata.shape
    k = k_half * 2
    b = qdata.to(torch.int32) & 0xFF
    q = torch.empty(n, k, dtype=torch.int32, device=qdata.device)
    q[:, 0::2] = b & 0xF
    q[:, 1::2] = (b >> 4) & 0xF
    srel = s_rel.float().repeat_interleave(group_size, dim=1)  # [N, K]
    if codebook is not None:
        vals = codebook.to(device=qdata.device, dtype=torch.float32)[q]  # direct [0,15] index
    else:
        vals = q.float() - 8.0  # symmetric uniform levels -8..7
    return (vals * srel).round().clamp_(-127, 127).to(torch.int8)


def w4a8_int8_linear(
    x: torch.Tensor,
    qdata: torch.Tensor,
    s_rel: torch.Tensor,
    s_channel: torch.Tensor,
    codebook: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    group_size: int = 16,
    convrot_groupsize: int = 256,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """``x @ W.T + bias`` for AsymW4A8Int8 (symmetric): dequant int4->int8, then the
    shared INT8 linear (ConvRot activation rotation + per-channel weight scale)."""
    int8_w = _dequant_int4_grouped_to_int8(qdata, s_rel, codebook, group_size)
    return int8_linear(
        x, int8_w, s_channel, bias=bias, out_dtype=out_dtype,
        convrot=True, convrot_groupsize=convrot_groupsize,
    )
