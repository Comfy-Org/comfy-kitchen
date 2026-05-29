# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# AWQ W4A16 (4-bit weight, fp16/bf16 activation): eager pure-PyTorch
# reference implementation plus torch.library dispatch.
#
# Kitchen-native storage layout (independent of the AWQ paper's CUDA layout):
#
#   qweight  (N, K // 2)       int8  — two unsigned int4 values per byte
#                                     bits 0..3 -> column 2j   (range [0, 15])
#                                     bits 4..7 -> column 2j+1 (range [0, 15])
#   wscales  (K // G, N)       same fp dtype as compute
#   wzeros   (K // G, N)       same fp dtype as compute — fp zero points
#   bias     (N,)              fp   (optional)
#
# Forward math (per-group asymmetric):
#   W[n, k] = (qweight[n, k] - 8) * wscales[k // G, n] + wzeros[k // G, n]
#   y       = x @ W.T + bias

import torch

_DEFAULT_GROUP = 64


def _unpack_uint4_row_major(packed: torch.Tensor) -> torch.Tensor:
    """(..., K//2) int8 storing two uint4 per byte -> (..., K) int8 in [0, 15]."""
    x32 = packed.to(torch.int32)
    lo = x32 & 0x0F
    hi = (x32 >> 4) & 0x0F
    stacked = torch.stack([lo, hi], dim=-1)
    return stacked.reshape(*packed.shape[:-1], -1).to(torch.int8)


def gemv_awq_w4a16(
    x: torch.Tensor,
    qweight: torch.Tensor,
    wscales: torch.Tensor,
    wzeros: torch.Tensor,
    bias: torch.Tensor | None = None,
    group_size: int = _DEFAULT_GROUP,
) -> torch.Tensor:
    """AWQ W4A16 GEMV / small-batch GEMM.

    Args:
        x: (M, K) or (K,) fp16/bf16 input.
        qweight: (N, K // 2) int8 packed (kitchen-native row-major uint4).
        wscales: (K // group_size, N) fp.
        wzeros: (K // group_size, N) fp.
        bias: (N,) fp or None.
        group_size: quantization group size (default 64).

    Returns:
        Output with the input's leading shape and trailing dim N.
    """
    orig_shape = x.shape
    x2d = x.reshape(-1, orig_shape[-1])
    _m, k = x2d.shape
    if k % group_size != 0:
        raise ValueError(f"K={k} not divisible by group_size={group_size}")

    n, k_half = qweight.shape
    if k_half * 2 != k:
        raise ValueError(f"qweight K//2={k_half} inconsistent with x K={k}")

    compute_dtype = wscales.dtype

    # Dequantize: (qweight - 8) * wscales + wzeros
    w_uint = _unpack_uint4_row_major(qweight).to(compute_dtype)  # (n, k)
    w_groups = w_uint.view(n, k // group_size, group_size)
    # wscales / wzeros are (K/G, N) — transpose to (n, k/g) for broadcasting
    scales_ng = wscales.t().unsqueeze(-1)  # (n, k/g, 1)
    zeros_ng = wzeros.t().unsqueeze(-1)
    w_fp = ((w_groups - 8.0) * scales_ng + zeros_ng).view(n, k)

    out = x2d.to(compute_dtype) @ w_fp.t()
    if bias is not None:
        out = out + bias
    return out.reshape(*orig_shape[:-1], n)


# =============================================================================
# torch.library Custom Op Dispatch
# =============================================================================


@torch.library.custom_op("comfy_kitchen::gemv_awq_w4a16", mutates_args=())
def _op_gemv_awq_w4a16(
    x: torch.Tensor,
    qweight: torch.Tensor,
    wscales: torch.Tensor,
    wzeros: torch.Tensor,
    bias: torch.Tensor | None = None,
    group_size: int = _DEFAULT_GROUP,
) -> torch.Tensor:
    from comfy_kitchen.registry import registry

    kwargs = {
        "x": x, "qweight": qweight, "wscales": wscales, "wzeros": wzeros,
        "bias": bias, "group_size": group_size,
    }
    impl = registry.get_implementation("gemv_awq_w4a16", kwargs=kwargs)
    return impl(**kwargs)


@_op_gemv_awq_w4a16.register_fake
def _op_gemv_awq_w4a16_fake(x, qweight, wscales, wzeros, bias=None, group_size=64):
    out_shape = (*x.shape[:-1], wscales.shape[1])
    return torch.empty(out_shape, dtype=wscales.dtype, device=x.device)
