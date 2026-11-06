# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton PixelNorm + SiLU + causal temporal padding (+ reflect pad), one pass.

Writes the contiguous ``[B, C, To, Ho, Wo]`` conv input straight from x, each output pixel
from its source position:

    output frame fo < Tp      -> prefix frame fo (the already activated temporal cache), copied
    otherwise                 -> x frame clamp(fo - Tp - n_front, 0, T - 1), normalized + SiLU
    output position (hp, wp)  -> reflect(hp - pad, H), reflect(wp - pad, W)   (pad 0 for zeros)

so the repeated frames and the reflected border recompute the norm of their source pixel
instead of reading back a written value, and the normalized tensor, the repeated frames and
the unpadded concatenation are never written out. One program owns ``block_p`` output
positions of one output frame of one batch item; the sum of squares over C and the norm are in
fp32, rounded once to x's dtype at the store. x and prefix are read through their strides and
storage offsets (decoder time chunks and the temporal cache are views).

Launch configuration (same kernel body), picked per GPU from the measured shapes:
  CUDA: one pass, all channels in registers, block_c = next_pow2(C),
        block_p = max(8, 8192 // block_c), 4 warps for block_c <= 256 else 8 (RTX 5090)
  HIP:  two passes over C, 8 warps, (block_c, block_p) = (64, 64) for C >= 512 else (32, 128)
        (Radeon 8060S)
"""

import torch

import triton
import triton.language as tl

_REFLECT_PAD = 1  # CausalConv3d, kernel 3: padding (0, 1, 1)
_HIP = bool(torch.version.hip)  # picks the launch configuration


@triton.jit
def _pixel_norm_silu_pad3d_kernel(
    x_ptr,
    pre_ptr,
    out_ptr,
    channels,
    frames,
    height,
    width,
    pre_frames,
    n_front,
    out_frames,
    out_height,
    out_width,
    pad,
    sxb,
    sxc,
    sxt,
    sxh,
    sxw,
    spb,
    spc,
    spt,
    sph,
    spw,
    eps,
    single: tl.constexpr,
    block_c: tl.constexpr,
    block_p: tl.constexpr,
):
    pid_p = tl.program_id(0)
    fo = tl.program_id(1)
    b = tl.program_id(2).to(tl.int64)
    out_hw = out_height * out_width
    q = pid_p * block_p + tl.arange(0, block_p)
    qm = q < out_hw
    hs = q // out_width - pad
    ws = q % out_width - pad
    # reflect (identity when pad == 0)
    hs = tl.where(hs < 0, -hs, hs)
    hs = tl.where(hs >= height, 2 * height - 2 - hs, hs)
    ws = tl.where(ws < 0, -ws, ws)
    ws = tl.where(ws >= width, 2 * width - 2 - ws, ws)
    hs = tl.where(qm, hs, 0)
    ws = tl.where(qm, ws, 0)
    out_cs = out_frames.to(tl.int64) * out_hw
    o_base = b * channels * out_cs + fo.to(tl.int64) * out_hw + q

    if fo < pre_frames:
        p_sp = hs.to(tl.int64) * sph + ws.to(tl.int64) * spw
        p_base = b * spb + fo.to(tl.int64) * spt
        for c0 in range(0, channels, block_c):
            c = c0 + tl.arange(0, block_c)
            m = (c[:, None] < channels) & qm[None, :]
            v = tl.load(
                pre_ptr + p_base + c[:, None].to(tl.int64) * spc + p_sp[None, :], mask=m, other=0.0
            )
            tl.store(out_ptr + o_base[None, :] + c[:, None].to(tl.int64) * out_cs, v, mask=m)
    else:
        t = fo - pre_frames - n_front
        t = tl.minimum(tl.maximum(t, 0), frames - 1)
        x_sp = hs.to(tl.int64) * sxh + ws.to(tl.int64) * sxw
        x_base = b * sxb + t.to(tl.int64) * sxt
        if single:
            c = tl.arange(0, block_c)
            m = (c[:, None] < channels) & qm[None, :]
            xv = tl.load(
                x_ptr + x_base + c[:, None].to(tl.int64) * sxc + x_sp[None, :], mask=m, other=0.0
            ).to(tl.float32)
            r = tl.sqrt(tl.sum(xv * xv, 0) / channels + eps)
            y = xv / r[None, :]
            y = y / (1.0 + tl.exp(-y))
            tl.store(
                out_ptr + o_base[None, :] + c[:, None].to(tl.int64) * out_cs,
                y.to(out_ptr.dtype.element_ty),
                mask=m,
            )
        else:
            ss = tl.zeros([block_p], tl.float32)
            for c0 in range(0, channels, block_c):
                c = c0 + tl.arange(0, block_c)
                m = (c[:, None] < channels) & qm[None, :]
                xv = tl.load(
                    x_ptr + x_base + c[:, None].to(tl.int64) * sxc + x_sp[None, :],
                    mask=m,
                    other=0.0,
                ).to(tl.float32)
                ss += tl.sum(xv * xv, 0)
            r = tl.sqrt(ss / channels + eps)
            for c0 in range(0, channels, block_c):
                c = c0 + tl.arange(0, block_c)
                m = (c[:, None] < channels) & qm[None, :]
                xv = tl.load(
                    x_ptr + x_base + c[:, None].to(tl.int64) * sxc + x_sp[None, :],
                    mask=m,
                    other=0.0,
                ).to(tl.float32)
                y = xv / r[None, :]
                y = y / (1.0 + tl.exp(-y))
                tl.store(
                    out_ptr + o_base[None, :] + c[:, None].to(tl.int64) * out_cs,
                    y.to(out_ptr.dtype.element_ty),
                    mask=m,
                )


def _launch_config(channels: int) -> tuple[bool, int, int, int]:
    # (single pass, block_c, block_p, num_warps)
    if _HIP:
        block_c, block_p = (64, 64) if channels >= 512 else (32, 128)
        return False, block_c, block_p, 8
    block_c = triton.next_power_of_2(channels)
    return True, block_c, max(8, 8192 // block_c), 4 if block_c <= 256 else 8


def pixel_norm_silu_pad3d(
    x: torch.Tensor,
    prefix: torch.Tensor | None = None,
    n_front: int = 1,
    n_back: int = 0,
    pad_mode: str = "zeros",
    eps: float = 1e-8,
) -> torch.Tensor:
    pad = _REFLECT_PAD if pad_mode == "reflect" else 0
    b, c, t, h, w = x.shape
    if prefix is None:
        pre, pre_frames, pre_strides = x, 0, (0, 0, 0, 0, 0)
    else:
        pre, pre_frames, pre_strides = prefix, prefix.shape[2], prefix.stride()
    out_frames = pre_frames + n_front + t + n_back
    out_height, out_width = h + 2 * pad, w + 2 * pad
    out = torch.empty((b, c, out_frames, out_height, out_width), dtype=x.dtype, device=x.device)
    single, block_c, block_p, num_warps = _launch_config(c)
    grid = (triton.cdiv(out_height * out_width, block_p), out_frames, b)
    _pixel_norm_silu_pad3d_kernel[grid](
        x,
        pre,
        out,
        c,
        t,
        h,
        w,
        pre_frames,
        n_front,
        out_frames,
        out_height,
        out_width,
        pad,
        *x.stride(),
        *pre_strides,
        float(eps),
        single=single,
        block_c=block_c,
        block_p=block_p,
        num_warps=num_warps,
    )
    return out
