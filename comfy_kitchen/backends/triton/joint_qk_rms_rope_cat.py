# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton dual-stream attention prep: QK RMSNorm + txt / img concat + RoPE, and the v concat.

One launch reads the six projection outputs once, through their ``[B, L, heads * head_dim]``
rows (last dim contiguous), and writes the contiguous joint ``[B, heads, L_txt + L_img,
head_dim]`` q, k and v, text rows first (joint row l < L_txt is text row l, the rest image row
l - L_txt). Each program owns ``block_l`` joint rows of one batch item, for one head or for
all heads (see the launch configuration below):

    q, k: per-head RMSNorm with the stream's weight, then the interleaved RoPE
          (pair j = elements 2j, 2j + 1; out[2j + i] = f[j, i, 0] * x[2j] + f[j, i, 1] * x[2j + 1])
          from the [.., head_dim / 2, 2, 2] table, in one of two rounding modes:
          stock rounding: the unfused path's rounding, so q and k equal the unfused chain
                (F.rms_norm, then apply_rope1 on the HIP backend) bit for bit: the normalized
                value rounded to bf16 as F.rms_norm stores it (same summation order), and
                the rotation rounded as the HIP apply_rope1 kernel and eager do
          otherwise: in fp32, rounded once to the output dtype
    v:    copied

so the heads-first copies, the normalized tensors and the unrotated concatenations are never
written out.

Stock rounding is used only on a ROCm build where the HIP backend is registered with
apply_rope1 (it is then first in the backend priority, so stock apply_rope1 runs it). On CUDA
builds, and on ROCm builds without the HIP backend (an arch it does not support, or the
extension not built), stock apply_rope1 runs another backend with another rounding, and q and
k are rounded once. Decided on the first call and kept for the process.

Launch configuration (same kernel body), picked per build:
  CUDA: grid (L tiles, heads, B), block_l 32, 4 warps
  HIP:  grid (L tiles, 1, B), block_l 16, 4 warps, loop over the heads inside the program so
        the rows' RoPE table tile is loaded once
"""

import torch

import triton
import triton.language as tl

from comfy_kitchen.registry import registry

_HIP = bool(torch.version.hip)  # picks the launch configuration
_stock_rounding: bool | None = None  # set by _rounds_like_stock on the first call


def _rounds_like_stock() -> bool:
    # Stock rounding iff stock apply_rope1 runs on the HIP backend: a ROCm build with the HIP
    # backend registered for apply_rope1 (comfy_kitchen then puts it first in the priority).
    # Read once; the backends are registered when comfy_kitchen is imported.
    global _stock_rounding
    if _stock_rounding is None:
        hip = registry.list_backends().get("hip", {})
        _stock_rounding = bool(
            _HIP and registry.is_available("hip") and "apply_rope1" in hip.get("capabilities", ())
        )
    return _stock_rounding


@triton.jit
def _round_bf16(v):
    return v.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _fold(s, block_l: tl.constexpr, half: tl.constexpr):
    # s[:, t] + s[:, t + half], the shuffle-down step of a warp reduction
    lo, hi = tl.split(tl.permute(tl.reshape(s, (block_l, 2, half)), (0, 2, 1)))
    return hi + lo


@triton.jit
def _sum_squares(x, block_l: tl.constexpr, head_dim: tl.constexpr):
    # the summation order of ATen's vectorized layer-norm reduction (vectorized_layer_norm_kernel),
    # which F.rms_norm runs: lane t adds the squares of elements 4t .. 4t + 3 in turn, then
    # the lanes reduce by shuffle-down
    lanes: tl.constexpr = head_dim // 4
    e, o = tl.split(tl.reshape(x * x, (block_l, lanes, 2, 2)))
    e0, e1 = tl.split(e)
    o0, o1 = tl.split(o)
    s = ((e0 + o0) + e1) + o1
    if lanes >= 32:
        s = _fold(s, block_l, 16)
    if lanes >= 16:
        s = _fold(s, block_l, 8)
    if lanes >= 8:
        s = _fold(s, block_l, 4)
    s = _fold(s, block_l, 2)
    s = _fold(s, block_l, 1)
    return tl.reshape(s, (block_l,))


@triton.jit
def _rope_pair(fa, xa, fb, xb, stock_rounding: tl.constexpr):
    if stock_rounding:
        # HIP apply_rope1 and eager: the first product rounded, addcmul_ fuses the second;
        # x and f are bf16 values here, so each product is exact in fp32
        return _round_bf16(fa * xa) + fb * xb
    return fa * xa + fb * xb


@triton.jit
def _qk_row(
    src_t,
    src_i,
    row_t,
    row_i,
    is_txt,
    lmask,
    offs_d,
    w_t,
    w_i,
    f00,
    f01,
    f10,
    f11,
    eps,
    head_dim: tl.constexpr,
    block_l: tl.constexpr,
    stock_rounding: tl.constexpr,
):
    ptr = tl.where(
        is_txt[:, None],
        src_t + row_t[:, None] + offs_d[None, :],
        src_i + row_i[:, None] + offs_d[None, :],
    )
    x = tl.load(ptr, mask=lmask[:, None], other=0.0).to(tl.float32)
    if stock_rounding:
        ss = _sum_squares(x, block_l, head_dim) / head_dim
    else:
        ss = tl.sum(x * x, axis=1) / head_dim
    inv = tl.math.rsqrt(ss + eps)
    w = tl.where(is_txt[:, None], w_t[None, :], w_i[None, :])
    y = x * inv[:, None] * w
    if stock_rounding:
        # F.rms_norm's single rounding of w * (x * rstd) to bf16: the unfused path stores it
        y = _round_bf16(y)
    ye, yo = tl.split(tl.reshape(y, (block_l, head_dim // 2, 2)))
    oe = _rope_pair(f00, ye, f01, yo, stock_rounding)
    oo = _rope_pair(f10, ye, f11, yo, stock_rounding)
    return tl.interleave(oe, oo)


@triton.jit
def _joint_qk_rms_rope_cat_kernel(
    qi,
    ki,
    vi,
    qt,
    kt,
    vt,
    wq,
    wk,
    waq,
    wak,
    fr,
    oq,
    ok,
    ov,
    seq_img,
    seq_txt,
    seq_total,
    s_ib,
    s_il,
    s_tb,
    s_tl,
    f_sb,
    eps,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_l: tl.constexpr,
    head_loop: tl.constexpr,
    stock_rounding: tl.constexpr,
):
    pid_l = tl.program_id(0)
    pid_h = tl.program_id(1)
    b = tl.program_id(2).to(tl.int64)
    offs_l = pid_l * block_l + tl.arange(0, block_l)
    lmask = offs_l < seq_total
    is_txt = offs_l < seq_txt
    offs_d = tl.arange(0, head_dim)
    l64 = offs_l.to(tl.int64)
    row_t = b * s_tb + l64 * s_tl
    row_i = b * s_ib + (l64 - seq_txt) * s_il

    offs_f = tl.arange(0, 2 * head_dim)
    f = tl.load(
        fr + b * f_sb + l64[:, None] * (2 * head_dim) + offs_f[None, :],
        mask=lmask[:, None],
        other=0.0,
    ).to(tl.float32)
    fj0, fj1 = tl.split(tl.reshape(f, (block_l, head_dim // 2, 2, 2)))
    f00, f10 = tl.split(fj0)
    f01, f11 = tl.split(fj1)

    wq_ = tl.load(wq + offs_d).to(tl.float32)
    wk_ = tl.load(wk + offs_d).to(tl.float32)
    waq_ = tl.load(waq + offs_d).to(tl.float32)
    wak_ = tl.load(wak + offs_d).to(tl.float32)

    out_row = b * heads * seq_total * head_dim + l64[:, None] * head_dim + offs_d[None, :]
    if head_loop:
        for h in range(heads):
            hd = h * head_dim
            hoff = h * seq_total * head_dim
            yq = _qk_row(
                qt + hd,
                qi + hd,
                row_t,
                row_i,
                is_txt,
                lmask,
                offs_d,
                waq_,
                wq_,
                f00,
                f01,
                f10,
                f11,
                eps,
                head_dim,
                block_l,
                stock_rounding,
            )
            tl.store(oq + hoff + out_row, yq.to(oq.dtype.element_ty), mask=lmask[:, None])
            yk = _qk_row(
                kt + hd,
                ki + hd,
                row_t,
                row_i,
                is_txt,
                lmask,
                offs_d,
                wak_,
                wk_,
                f00,
                f01,
                f10,
                f11,
                eps,
                head_dim,
                block_l,
                stock_rounding,
            )
            tl.store(ok + hoff + out_row, yk.to(ok.dtype.element_ty), mask=lmask[:, None])
            pv = tl.where(
                is_txt[:, None],
                vt + hd + row_t[:, None] + offs_d[None, :],
                vi + hd + row_i[:, None] + offs_d[None, :],
            )
            v = tl.load(pv, mask=lmask[:, None], other=0.0)
            tl.store(ov + hoff + out_row, v, mask=lmask[:, None])
    else:
        hd = pid_h * head_dim
        hoff = pid_h.to(tl.int64) * seq_total * head_dim
        yq = _qk_row(
            qt + hd,
            qi + hd,
            row_t,
            row_i,
            is_txt,
            lmask,
            offs_d,
            waq_,
            wq_,
            f00,
            f01,
            f10,
            f11,
            eps,
            head_dim,
            block_l,
            stock_rounding,
        )
        tl.store(oq + hoff + out_row, yq.to(oq.dtype.element_ty), mask=lmask[:, None])
        yk = _qk_row(
            kt + hd,
            ki + hd,
            row_t,
            row_i,
            is_txt,
            lmask,
            offs_d,
            wak_,
            wk_,
            f00,
            f01,
            f10,
            f11,
            eps,
            head_dim,
            block_l,
            stock_rounding,
        )
        tl.store(ok + hoff + out_row, yk.to(ok.dtype.element_ty), mask=lmask[:, None])
        pv = tl.where(
            is_txt[:, None],
            vt + hd + row_t[:, None] + offs_d[None, :],
            vi + hd + row_i[:, None] + offs_d[None, :],
        )
        v = tl.load(pv, mask=lmask[:, None], other=0.0)
        tl.store(ov + hoff + out_row, v, mask=lmask[:, None])


def _launch_config() -> tuple[int, int, bool]:
    # (block_l, num_warps, head_loop)
    if _HIP:
        return 16, 4, True
    return 32, 4, False


def joint_qk_rms_rope_cat(
    img_q: torch.Tensor,
    img_k: torch.Tensor,
    img_v: torch.Tensor,
    txt_q: torch.Tensor,
    txt_k: torch.Tensor,
    txt_v: torch.Tensor,
    img_q_weight: torch.Tensor,
    img_k_weight: torch.Tensor,
    txt_q_weight: torch.Tensor,
    txt_k_weight: torch.Tensor,
    freqs_cis: torch.Tensor,
    heads: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, seq_img, channels = img_q.shape
    seq_txt = txt_q.shape[1]
    head_dim = channels // heads
    seq_total = seq_txt + seq_img
    shape = (batch_size, heads, seq_total, head_dim)
    out_q = torch.empty(shape, device=img_q.device, dtype=img_q.dtype)
    out_k = torch.empty_like(out_q)
    out_v = torch.empty_like(out_q)
    f_sb = 0 if freqs_cis.shape[0] == 1 else freqs_cis.stride(0)
    block_l, num_warps, head_loop = _launch_config()
    grid = (triton.cdiv(seq_total, block_l), 1 if head_loop else heads, batch_size)
    _joint_qk_rms_rope_cat_kernel[grid](
        img_q,
        img_k,
        img_v,
        txt_q,
        txt_k,
        txt_v,
        img_q_weight,
        img_k_weight,
        txt_q_weight,
        txt_k_weight,
        freqs_cis,
        out_q,
        out_k,
        out_v,
        seq_img,
        seq_txt,
        seq_total,
        img_q.stride(0),
        img_q.stride(1),
        txt_q.stride(0),
        txt_q.stride(1),
        f_sb,
        float(eps),
        heads=heads,
        head_dim=head_dim,
        block_l=block_l,
        head_loop=head_loop,
        stock_rounding=_rounds_like_stock(),
        num_warps=num_warps,
    )
    return out_q, out_k, out_v
