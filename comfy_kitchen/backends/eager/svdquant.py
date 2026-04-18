# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# SVDQuant W4A4 (4-bit weight, 4-bit activation, SVD low-rank correction):
# eager pure-PyTorch reference implementations plus torch.library dispatch.
#
# Kitchen-native storage layout (independent of any third-party kernel):
#
#   qweight       (N, K // 2)        int8  — two signed int4 values per byte
#                                            bits 0..3 -> column 2j   (range [-8, 7])
#                                            bits 4..7 -> column 2j+1 (range [-8, 7])
#   wscales       (K // 64, N)       same fp dtype as compute
#   proj_down     (K, R)             fp
#   proj_up       (N, R)             fp
#   smooth_factor (K,)               fp
#   bias          (N,)               fp   (optional)
#
# Activation quantization produces:
#   q_act   (M_pad, K // 2)          int8 (same packing as qweight)
#   ascales (K // 64, M_pad)         fp
#   lora_act (M_pad, R)              fp32
#
# Forward math (matches SmoothQuant convention — activation is DIVIDED by
# smooth to reduce per-channel outliers; the calibrated residual weight
# absorbs those outliers during offline quantization):
#
#   out = (x / smooth) @ (W_int4 * wscales_group).T
#       + x @ proj_down @ proj_up.T
#       + bias

import math

import torch
import torch.nn.functional as F

_INT4_GROUP_SIZE = 64
_INT4_MAX = 7  # signed int4 [-8, 7], symmetric scale uses 7
_UINT4_MAX = 15  # unsigned int4 [0, 15], used by fused GELU -> fc2 path
_GELU_UNSIGNED_SHIFT = 0.171875  # matches nunchaku's SHIFT_GELU


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _pack_int4_row_major(values: torch.Tensor) -> torch.Tensor:
    """Pack (..., K) signed int4 values (int8 dtype, range [-8, 7]) into
    (..., K // 2) int8 with two nibbles per byte (low = even column).
    """
    if values.shape[-1] % 2 != 0:
        raise ValueError(f"last dim must be even, got {values.shape[-1]}")
    lo = values[..., 0::2].to(torch.int32) & 0x0F
    hi = values[..., 1::2].to(torch.int32) & 0x0F
    return (lo | (hi << 4)).to(torch.int8)


def _unpack_int4_row_major(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of _pack_int4_row_major. Returns int8 in [-8, 7]."""
    x32 = packed.to(torch.int32)
    lo = x32 & 0x0F
    hi = (x32 >> 4) & 0x0F
    lo = torch.where(lo >= 8, lo - 16, lo)
    hi = torch.where(hi >= 8, hi - 16, hi)
    stacked = torch.stack([lo, hi], dim=-1)
    return stacked.reshape(*packed.shape[:-1], -1).to(torch.int8)


def _unpack_uint4_row_major(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of _pack_int4_row_major for unsigned nibble payloads."""
    x32 = packed.to(torch.int32)
    lo = x32 & 0x0F
    hi = (x32 >> 4) & 0x0F
    stacked = torch.stack([lo, hi], dim=-1)
    return stacked.reshape(*packed.shape[:-1], -1).to(torch.int8)


def quantize_svdquant_w4a4(
    x: torch.Tensor,
    smooth: torch.Tensor,
    lora_down: torch.Tensor,
    pad_size: int = 256,
    act_unsigned: bool = False,
    shift_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize activations to int4 with fused smoothing and LoRA down projection.

    Args:
        x: (M, K) bf16/fp16 input.
        smooth: (K,) per-channel smoothing factor applied before quantization.
        lora_down: (K, R) low-rank down projection weight (fp32 accumulation).
        pad_size: pad M to a multiple of this (default 256) to match downstream kernels.
        act_unsigned: if True, quantize into uint4 [0, 15] instead of signed int4.
        shift_value: additive shift applied before smoothing. Qwen Image MLP fc2
            uses 0.171875 after GELU to enable unsigned activation quantization.

    Returns:
        q_x: (M_pad, K // 2) int8 packed (2 int4 per byte, same layout as weight).
        ascales: (K // 64, M_pad) same dtype as x — per-row per-group scales.
        lora_act: (M_pad, R) fp32 LoRA activations.
    """
    if x.dim() != 2:
        raise ValueError(f"expected 2D input, got shape {tuple(x.shape)}")
    M, K = x.shape
    rank = lora_down.shape[1]
    group = _INT4_GROUP_SIZE
    if K % group != 0:
        raise ValueError(f"K={K} not divisible by group_size={group}")
    M_pad = _ceil_div(M, pad_size) * pad_size

    # LoRA down: computed on the un-smoothed input (matches SVDQuant convention)
    lora_act = x.float() @ lora_down.float()  # (M, R)

    # Smooth (divide) + per-row per-group int4 quantization.
    # SmoothQuant: outliers are moved from activations to weights at calibration.
    # At inference, activations divide by smooth so they quantize cleanly.
    x_smooth = (x + shift_value) / smooth
    groups = x_smooth.view(M, K // group, group)
    absmax = groups.abs().amax(dim=-1).clamp(min=1e-10)  # (M, K/G)
    qmax = _UINT4_MAX if act_unsigned else _INT4_MAX
    qmin = 0 if act_unsigned else (-_INT4_MAX - 1)
    scales = absmax / qmax
    q_vals = (groups / scales.unsqueeze(-1)).round().clamp(qmin, qmax).to(torch.int8)
    q_vals = q_vals.reshape(M, K)
    q_packed = _pack_int4_row_major(q_vals)  # (M, K // 2)

    if M_pad > M:
        pad = M_pad - M
        q_packed = F.pad(q_packed, (0, 0, 0, pad))
        scales = F.pad(scales, (0, 0, 0, pad))
        lora_act = F.pad(lora_act, (0, 0, 0, pad))

    ascales = scales.t().contiguous().to(x.dtype)  # (K/G, M_pad)
    return q_packed, ascales, lora_act


def scaled_mm_svdquant_w4a4(
    act: torch.Tensor,
    wgt: torch.Tensor,
    ascales: torch.Tensor,
    wscales: torch.Tensor,
    lora_act_in: torch.Tensor,
    lora_up: torch.Tensor,
    bias: torch.Tensor | None = None,
    act_unsigned: bool = False,
) -> torch.Tensor:
    """SVDQuant W4A4 GEMM with fused LoRA up projection and optional bias.

    Args:
        act: (M, K // 2) int8 packed activations from quantize_svdquant_w4a4.
        wgt: (N, K // 2) int8 packed weights (kitchen-native layout).
        ascales: (K // 64, M) per-row per-group activation scales.
        wscales: (K // 64, N) per-group weight scales.
        lora_act_in: (M, R) fp32 LoRA down-projection activations.
        lora_up: (N, R) fp LoRA up-projection weight.
        bias: (N,) fp bias or None.
        act_unsigned: if True, activations are stored as uint4 in [0, 15] with
            an implicit -8 offset (used after GELU when activations are all positive).

    Returns:
        out: (M, N) fp in the dtype of wscales/lora_up.
    """
    M, K_half = act.shape
    N = wgt.shape[0]
    K = K_half * 2
    group = _INT4_GROUP_SIZE
    compute_dtype = wscales.dtype

    # --- weight dequantization ---
    wgt_int = _unpack_int4_row_major(wgt).to(compute_dtype)  # (N, K)
    wgt_g = wgt_int.view(N, K // group, group)
    wscales_bng = wscales.t().unsqueeze(-1)  # (N, K/G, 1)
    wgt_fp = (wgt_g * wscales_bng).view(N, K)

    # --- activation dequantization ---
    if act_unsigned:
        act_int = _unpack_uint4_row_major(act)
    else:
        act_int = _unpack_int4_row_major(act)
    act_int = act_int.to(compute_dtype).view(M, K // group, group)
    ascales_mng = ascales.t().unsqueeze(-1)  # (M, K/G, 1)
    act_fp = (act_int * ascales_mng).view(M, K)

    out = act_fp @ wgt_fp.t()  # (M, N)

    # LoRA up branch (in fp32 for accumulation stability)
    lora_contribution = lora_act_in.float() @ lora_up.float().t()  # (M, N)
    out = out + lora_contribution.to(out.dtype)

    if bias is not None:
        out = out + bias
    return out


# =============================================================================
# torch.library Custom Op Dispatch
# =============================================================================
#
# The custom ops live in eager because eager is always imported and acts as
# the dispatcher host — consistent with rope.py. The actual implementation is
# chosen at call time by the registry based on backend priority and constraints.


@torch.library.custom_op("comfy_kitchen::quantize_svdquant_w4a4", mutates_args=())
def _op_quantize_svdquant_w4a4(
    x: torch.Tensor,
    smooth: torch.Tensor,
    lora_down: torch.Tensor,
    pad_size: int = 256,
    act_unsigned: bool = False,
    shift_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from comfy_kitchen.registry import registry

    kwargs = {
        "x": x,
        "smooth": smooth,
        "lora_down": lora_down,
        "pad_size": pad_size,
        "act_unsigned": act_unsigned,
        "shift_value": shift_value,
    }
    impl = registry.get_implementation("quantize_svdquant_w4a4", kwargs=kwargs)
    return impl(**kwargs)


@_op_quantize_svdquant_w4a4.register_fake
def _op_quantize_svdquant_w4a4_fake(
    x, smooth, lora_down, pad_size=256, act_unsigned=False, shift_value=0.0,
):
    M, K = x.shape
    R = lora_down.shape[1]
    M_pad = _ceil_div(M, pad_size) * pad_size
    q_x = torch.empty(M_pad, K // 2, dtype=torch.int8, device=x.device)
    ascales = torch.empty(K // _INT4_GROUP_SIZE, M_pad, dtype=x.dtype, device=x.device)
    lora_act = torch.empty(M_pad, R, dtype=torch.float32, device=x.device)
    return q_x, ascales, lora_act


@torch.library.custom_op("comfy_kitchen::scaled_mm_svdquant_w4a4", mutates_args=())
def _op_scaled_mm_svdquant_w4a4(
    act: torch.Tensor,
    wgt: torch.Tensor,
    ascales: torch.Tensor,
    wscales: torch.Tensor,
    lora_act_in: torch.Tensor,
    lora_up: torch.Tensor,
    bias: torch.Tensor | None = None,
    act_unsigned: bool = False,
) -> torch.Tensor:
    from comfy_kitchen.registry import registry

    kwargs = {
        "act": act, "wgt": wgt, "ascales": ascales, "wscales": wscales,
        "lora_act_in": lora_act_in, "lora_up": lora_up,
        "bias": bias, "act_unsigned": act_unsigned,
    }
    impl = registry.get_implementation("scaled_mm_svdquant_w4a4", kwargs=kwargs)
    return impl(**kwargs)


@_op_scaled_mm_svdquant_w4a4.register_fake
def _op_scaled_mm_svdquant_w4a4_fake(
    act, wgt, ascales, wscales, lora_act_in, lora_up, bias=None, act_unsigned=False,
):
    M = act.shape[0]
    N = wgt.shape[0]
    return torch.empty(M, N, dtype=lora_up.dtype, device=act.device)
