# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# SVDQuant W4A4 (4-bit weight, 4-bit activation, SVD low-rank correction):
# eager pure-PyTorch reference implementations plus torch.library dispatch.
#
# Two related but distinct int4 ranges appear in this module:
#
#   (a) Storage / nibble encoding   = the 4-bit two's-complement field can hold
#                                     any value in [-8, 7] (signed) or [0, 15]
#                                     (unsigned). Pack/unpack codecs handle the
#                                     full range because that's what the physical
#                                     bits can represent.
#   (b) Quantizer emission range    = the clamp the quantizer actually applies.
#                                     For absmax-symmetric quantization with
#                                     scale = max/7 (nunchaku / kitchen CUDA
#                                     contract), signed emission is [-7, 7] —
#                                     we skip -8 so the dequant range is
#                                     symmetric about zero. Unsigned emission
#                                     is [0, 15] with scale = max/15.
#
# Kitchen-native storage layout (independent of any third-party kernel):
#
#   qweight       (N, K // 2)        int8  — two int4 values per byte
#                                            bits 0..3 -> column 2j   (nibble)
#                                            bits 4..7 -> column 2j+1 (nibble)
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


import torch
import torch.nn.functional as F  # noqa: N812

from comfy_kitchen.registry import registry

_INT4_GROUP_SIZE = 64
_TILE_PACKED_BLOCK_N = 128
_TILE_PACKED_INTERLEAVE = 4
# Quantizer emission ranges (not the same as the 4-bit storage range — see
# module header). Symmetric absmax quantization uses scale = max/_INT4_MAX and
# clamps to [-_INT4_MAX, +_INT4_MAX]; -8 is representable by the nibble but is
# not emitted because it would break the dequant symmetry used by nunchaku's
# W4A4 kernel contract.
_INT4_MAX = 7   # signed quantizer range: [-7, 7], scale = max/7
_UINT4_MAX = 15 # unsigned quantizer range: [0, 15], scale = max/15 (post-GELU+shift fc2)
_GELU_UNSIGNED_SHIFT = 0.171875  # matches nunchaku's SHIFT_GELU constant


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _pack_int4_row_major(values: torch.Tensor) -> torch.Tensor:
    """Pack (..., K) int4 values into (..., K // 2) int8 (low = even column).

    Storage-level codec: handles the full 4-bit field; the caller decides which
    quantizer emission range to use. Inputs may contain any value that fits in
    a nibble (signed [-8, 7] or unsigned [0, 15]); values outside get masked
    via the & 0x0F below.
    """
    if values.shape[-1] % 2 != 0:
        raise ValueError(f"last dim must be even, got {values.shape[-1]}")
    lo = values[..., 0::2].to(torch.int32) & 0x0F
    hi = values[..., 1::2].to(torch.int32) & 0x0F
    return (lo | (hi << 4)).to(torch.int8)


def _unpack_int4_row_major(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of _pack_int4_row_major with signed-nibble interpretation.

    Storage-level codec: returns int8 across the full signed nibble range
    [-8, 7]. This is wider than the quantizer's emission range [-7, 7] by
    design — the codec must accept any bit pattern that could land in the
    nibble.
    """
    x32 = packed.to(torch.int32)
    lo = x32 & 0x0F
    hi = (x32 >> 4) & 0x0F
    lo = torch.where(lo >= 8, lo - 16, lo)
    hi = torch.where(hi >= 8, hi - 16, hi)
    stacked = torch.stack([lo, hi], dim=-1)
    return stacked.reshape(*packed.shape[:-1], -1).to(torch.int8)


def _unpack_uint4_row_major(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of _pack_int4_row_major with unsigned-nibble interpretation.

    Storage-level codec: returns int8 across the full unsigned nibble range
    [0, 15] (used by the u4.s4 MMA path — post-GELU+shift fc2 activations).
    """
    x32 = packed.to(torch.int32)
    lo = x32 & 0x0F
    hi = (x32 >> 4) & 0x0F
    stacked = torch.stack([lo, hi], dim=-1)
    return stacked.reshape(*packed.shape[:-1], -1).to(torch.int8)


def _tile_packed_weight_to_row_major(packed: torch.Tensor) -> torch.Tensor:
    """Convert kitchen_tile_packed_w4a4 weight to natural (N, K//2)."""
    if packed.dim() != 4:
        return packed
    n_tiles, k_groups, n_quads, bytes_per_quad = packed.shape
    if n_quads != _TILE_PACKED_BLOCK_N // _TILE_PACKED_INTERLEAVE:
        raise ValueError(f"unexpected tile-packed N quads: {n_quads}")
    if bytes_per_quad != _TILE_PACKED_INTERLEAVE * (_INT4_GROUP_SIZE // 2):
        raise ValueError(f"unexpected tile-packed byte axis: {bytes_per_quad}")
    return packed.view(
        n_tiles, k_groups, n_quads, _TILE_PACKED_INTERLEAVE, _INT4_GROUP_SIZE // 2,
    ).permute(0, 2, 3, 1, 4).contiguous().view(
        n_tiles * _TILE_PACKED_BLOCK_N, k_groups * (_INT4_GROUP_SIZE // 2),
    )


def _tile_packed_scales_to_natural(wscales: torch.Tensor) -> torch.Tensor:
    """Convert tile-packed wscales (N/128, K/G, 128) to natural (K/G, N)."""
    if wscales.dim() != 3:
        return wscales
    return wscales.permute(1, 0, 2).contiguous().view(
        wscales.shape[1], wscales.shape[0] * _TILE_PACKED_BLOCK_N,
    )


def _tile_packed_lora_up_to_natural(lora_up: torch.Tensor) -> torch.Tensor:
    """Convert tile-packed proj_up (N/128, R, 128) to natural (N, R)."""
    if lora_up.dim() != 3:
        return lora_up
    return lora_up.permute(0, 2, 1).contiguous().view(
        lora_up.shape[0] * _TILE_PACKED_BLOCK_N, lora_up.shape[1],
    )


def quantize_svdquant_w4a4(
    x: torch.Tensor,
    smooth: torch.Tensor,
    lora_down: torch.Tensor,
    pad_size: int = 256,
    act_unsigned: bool = False,
    lora_x: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize activations to int4 with smoothing + separate LoRA down projection.

    Args:
        x: (M, K) bf16/fp16 — main-path input (may be shifted by caller for unsigned).
        smooth: (K,) per-channel smoothing factor applied before quantization.
        lora_down: (K, R) low-rank down projection weight. The eager reference
            here runs the matmul in fp32 for numerical stability; the CUDA
            backend takes a lower-precision bf16 matmul path for memory/launch
            savings. Do not use this eager result as a bitwise oracle for the
            CUDA output — it is a high-precision reference, not a backend parity
            target.
        pad_size: pad M to a multiple of this (default 256) to match downstream kernels.
        act_unsigned: if True, quantize into uint4 [0, 15] (scale=max/15) instead of
            signed int4 [-7, 7] (scale=max/7). Selects MMA grid for downstream u4.s4.
        lora_x: (M, K) bf16/fp16 — input for LoRA matmul. Defaults to x. Pass
            separately when caller pre-shifts x (SVDQuant LoRA is mathematically
            defined on pre-quantization, pre-shift activations).

    Returns:
        q_x: (M_pad, K // 2) int8 packed (2 int4 per byte, same layout as weight).
        ascales: (K // 64, M_pad) same dtype as x — per-row per-group scales.
        lora_act: (M_pad, R) fp32 LoRA activations.
    """
    if x.dim() != 2:
        raise ValueError(f"expected 2D input, got shape {tuple(x.shape)}")
    m, k = x.shape
    lora_down.shape[1]
    group = _INT4_GROUP_SIZE
    if k % group != 0:
        raise ValueError(f"K={k} not divisible by group_size={group}")
    m_pad = _ceil_div(m, pad_size) * pad_size

    # LoRA down uses un-shifted, un-smoothed activation (SVDQuant invariant).
    lora_src = lora_x if lora_x is not None else x
    lora_act = lora_src.float() @ lora_down.float()  # (m, r)

    # Smooth (divide) + per-row per-group int4 quantization.
    # SmoothQuant: outliers are moved from activations to weights at calibration.
    # At inference, activations divide by smooth so they quantize cleanly.
    x_smooth = x / smooth
    groups = x_smooth.view(m, k // group, group)
    absmax = groups.abs().amax(dim=-1).clamp(min=1e-10)  # (m, k/g)
    # Quantizer emission range: symmetric signed [-7, 7] or unsigned [0, 15].
    # Signed clamp intentionally does NOT reach -8 even though the nibble can
    # represent it — absmax/7 scaling assumes ±qmax symmetry and emitting -8
    # would break dequant parity with nunchaku and kitchen CUDA.
    qmax = _UINT4_MAX if act_unsigned else _INT4_MAX
    qmin = 0 if act_unsigned else -_INT4_MAX
    scales = absmax / qmax
    q_vals = (groups / scales.unsqueeze(-1)).round().clamp(qmin, qmax).to(torch.int8)
    q_vals = q_vals.reshape(m, k)
    q_packed = _pack_int4_row_major(q_vals)  # (m, k // 2)

    if m_pad > m:
        pad = m_pad - m
        q_packed = F.pad(q_packed, (0, 0, 0, pad))
        scales = F.pad(scales, (0, 0, 0, pad))
        lora_act = F.pad(lora_act, (0, 0, 0, pad))

    ascales = scales.t().contiguous().to(x.dtype)  # (k/g, m_pad)
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
    """SVDQuant W4A4 int4 GEMM + LoRA up + optional bias (eager reference impl).

    Semantic mirror of the CUDA path; used as a high-precision reference and on
    non-CUDA devices. Numerical notes:

    * Quantization grid and emission range match the CUDA kernel exactly
      (signed absmax/7, unsigned absmax/15 — see module header).
    * The LoRA-up branch here accumulates in fp32 for stability. The CUDA
      wrapper takes a bf16 addmm_ path for memory/launch-count savings and
      drops the fp32 lora_act_in precision. This eager output is therefore a
      high-precision reference, not a bit-parity oracle for the CUDA backend.
      Tests that compare the two paths should use tolerances consistent with
      bf16 matmul (~8e-3 abs).

    Args:
        act: (M, K // 2) int8 packed activations from quantize_svdquant_w4a4.
            Bit-pattern interpretation depends on act_unsigned (signed [-7,7]
            vs unsigned [0,15]).
        wgt: (N, K // 2) int8 packed weights (kitchen natural row-major) or
            (N/128, K/64, 32, 128) kitchen_tile_packed_w4a4.
        ascales: (K // 64, M) per-row per-group activation scales.
        wscales: (K // 64, N) natural or (N/128, K/64, 128) tile-packed.
        lora_act_in: (M, R) fp32 LoRA down-projection activations.
        lora_up: (N, R) natural or (N/128, R, 128) tile-packed.
        bias: (N,) bias or None.
        act_unsigned: if True, interpret packed activations as unsigned [0, 15]
            (matches the u4.s4 MMA path used for post-GELU+shift fc2 layers).

    Returns:
        out: (M, N) in the dtype of wscales/lora_up.
    """
    wgt = _tile_packed_weight_to_row_major(wgt)
    wscales = _tile_packed_scales_to_natural(wscales)
    lora_up = _tile_packed_lora_up_to_natural(lora_up)

    m, k_half = act.shape
    n = wgt.shape[0]
    k = k_half * 2
    group = _INT4_GROUP_SIZE
    compute_dtype = wscales.dtype

    # --- weight dequantization ---
    wgt_int = _unpack_int4_row_major(wgt).to(compute_dtype)  # (n, k)
    wgt_g = wgt_int.view(n, k // group, group)
    wscales_bng = wscales.t().unsqueeze(-1)  # (n, k/g, 1)
    wgt_fp = (wgt_g * wscales_bng).view(n, k)

    # --- activation dequantization ---
    act_int = _unpack_uint4_row_major(act) if act_unsigned else _unpack_int4_row_major(act)
    act_int = act_int.to(compute_dtype).view(m, k // group, group)
    ascales_mng = ascales.t().unsqueeze(-1)  # (m, k/g, 1)
    act_fp = (act_int * ascales_mng).view(m, k)

    out = act_fp @ wgt_fp.t()  # (m, n)

    # LoRA up branch (in fp32 for accumulation stability)
    lora_contribution = lora_act_in.float() @ lora_up.float().t()  # (m, n)
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
    lora_x: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kwargs = {
        "x": x,
        "smooth": smooth,
        "lora_down": lora_down,
        "pad_size": pad_size,
        "act_unsigned": act_unsigned,
        "lora_x": lora_x,
    }
    impl = registry.get_implementation("quantize_svdquant_w4a4", kwargs=kwargs)
    return impl(**kwargs)


@_op_quantize_svdquant_w4a4.register_fake
def _op_quantize_svdquant_w4a4_fake(
    x, smooth, lora_down, pad_size=256, act_unsigned=False, lora_x=None,
):
    m, k = x.shape
    r = lora_down.shape[1]
    m_pad = _ceil_div(m, pad_size) * pad_size
    q_x = torch.empty(m_pad, k // 2, dtype=torch.int8, device=x.device)
    ascales = torch.empty(k // _INT4_GROUP_SIZE, m_pad, dtype=x.dtype, device=x.device)
    lora_act = torch.empty(m_pad, r, dtype=torch.float32, device=x.device)
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
    m = act.shape[0]
    n = wgt.shape[0] * _TILE_PACKED_BLOCK_N if wgt.dim() == 4 else wgt.shape[0]
    return torch.empty(m, n, dtype=lora_up.dtype, device=act.device)
