# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton bilinear backward warp written straight into its channel slice of the concat.

One kernel warps fp16 images by fp16 flows (pixel units) with FILM's ``_warp_core`` math in
fp32 and stores each warped value, rounded once to fp16, at its channel of a preallocated
contiguous ``[1, C_out, H, W]`` buffer, so the grid, the fp32 image and warp copies and the
separate concat are never written out. A 2D tile is ``block_c`` channels x ``block_p`` pixels;
the four taps are read through the image strides (channels-last and contiguous inputs alike).

- ``PyramidFlowEstimator`` form (one image, ``prefix``): extra channel blocks of the same
  launch copy ``prefix`` into channels ``[0, C_prefix)``, the warp goes to the channels after.
- Fusion-level form (two images, ``scales``): grid axis 2 selects the direction, so both warps
  are one launch; each flow is first scaled by its 1-element scale and rounded to fp16 like the
  stock fp16 multiply, and the scaled flows are also written after the warped channels.

Arithmetic follows ATen's CUDA path: ``dx = f * (1 / (W * 0.5))`` (the division by a CPU
scalar is a multiply by its fp32 reciprocal), the grid add, the unnormalize
``fma(g + 1, W, -1) * 0.5``, the border clip, ``floor``, ATen's bilinear weights and the
nw, ne, sw, se accumulation with fma and its in-bounds checks. The kernel is compiled with
``enable_fp_fusion=False`` so that only these explicit fma are fused; on CUDA ``dx`` / ``dy``
use ``libdevice.mul_rn``, because a plain multiply can still be contracted into the grid add.
Launch configuration on CUDA and HIP: 128 pixels x 32 channels with 8
warps for the ``prefix`` form, 128 pixels x 16 channels with 4 warps for the fusion form.
"""

import functools

import torch

import triton
import triton.language as tl

try:
    from triton.language.extra import libdevice
except ImportError:  # pragma: no cover
    libdevice = None

_MUL_RN = bool(torch.version.cuda) and libdevice is not None and hasattr(libdevice, "mul_rn")
_BLOCK_P = 128
_PREFIX_CFG = (32, 8)  # block_c, num_warps
_SCALED_CFG = (16, 4)


@triton.jit
def _warp_tile(
    img,
    flow,
    gx_ptr,
    gy_ptr,
    scale_ptr,
    out,
    offs_p,
    pmask,
    h,
    w,
    pid_c,
    channels,
    height,
    width,
    hw,
    s_ic,
    s_ih,
    s_iw,
    s_fc,
    s_fh,
    s_fw,
    out_c_off,
    flow_out_off,
    inv_hw,
    inv_hh,
    has_scale: tl.constexpr,
    mul_rn: tl.constexpr,
    block_c: tl.constexpr,
):
    f0 = tl.load(flow + h * s_fh + w * s_fw, mask=pmask, other=0.0)
    f1 = tl.load(flow + s_fc + h * s_fh + w * s_fw, mask=pmask, other=0.0)
    if has_scale:
        s = tl.load(scale_ptr).to(tl.float32)
        f0 = (f0.to(tl.float32) * s).to(tl.float16)
        f1 = (f1.to(tl.float32) * s).to(tl.float16)
        if pid_c == 0:
            tl.store(out + flow_out_off * hw + offs_p, f0, mask=pmask)
            tl.store(out + (flow_out_off + 1) * hw + offs_p, f1, mask=pmask)
    if mul_rn:
        # a rounded multiply, as ATen's separate kernels give; a plain one can be contracted
        # into the grid add even with enable_fp_fusion=False
        dx = libdevice.mul_rn(f0.to(tl.float32), inv_hw)
        dy = libdevice.mul_rn(f1.to(tl.float32), inv_hh)
    else:
        dx = f0.to(tl.float32) * inv_hw
        dy = f1.to(tl.float32) * inv_hh
    gxv = tl.load(gx_ptr + w, mask=pmask, other=0.0) + dx
    gyv = tl.load(gy_ptr + h, mask=pmask, other=0.0) + dy
    wf = width.to(tl.float32)
    hf = height.to(tl.float32)
    ix = tl.fma(gxv + 1.0, wf, -1.0) * 0.5
    iy = tl.fma(gyv + 1.0, hf, -1.0) * 0.5
    ix = tl.minimum(tl.maximum(ix, 0.0), wf - 1.0)
    iy = tl.minimum(tl.maximum(iy, 0.0), hf - 1.0)
    x0f = tl.floor(ix)
    y0f = tl.floor(iy)
    x1f = x0f + 1.0
    y1f = y0f + 1.0
    nw = (x1f - ix) * (y1f - iy)
    ne = (ix - x0f) * (y1f - iy)
    sw = (x1f - ix) * (iy - y0f)
    se = (ix - x0f) * (iy - y0f)
    x0 = x0f.to(tl.int32)
    y0 = y0f.to(tl.int32)
    vx = (x0 + 1) < width
    vy = (y0 + 1) < height
    c = pid_c * block_c + tl.arange(0, block_c)
    m = (c < channels)[:, None] & pmask[None, :]
    base = img + c[:, None] * s_ic
    p00 = y0 * s_ih + x0 * s_iw
    v00 = tl.load(base + p00[None, :], mask=m, other=0.0).to(tl.float32)
    v01 = tl.load(base + (p00 + s_iw)[None, :], mask=m & vx[None, :], other=0.0).to(tl.float32)
    v10 = tl.load(base + (p00 + s_ih)[None, :], mask=m & vy[None, :], other=0.0).to(tl.float32)
    v11 = tl.load(base + (p00 + s_ih + s_iw)[None, :], mask=m & (vx & vy)[None, :], other=0.0).to(
        tl.float32
    )
    acc = v00 * nw[None, :]
    acc = tl.where(vx[None, :], tl.fma(v01, ne[None, :], acc), acc)
    acc = tl.where(vy[None, :], tl.fma(v10, sw[None, :], acc), acc)
    acc = tl.where((vx & vy)[None, :], tl.fma(v11, se[None, :], acc), acc)
    tl.store(out + (out_c_off + c)[:, None] * hw + offs_p[None, :], acc.to(tl.float16), mask=m)


@triton.jit
def _copy_tile(
    src, out, offs_p, pmask, h, w, cb, c_src, hw, s_sc, s_sh, s_sw, block_c: tl.constexpr
):
    c = cb * block_c + tl.arange(0, block_c)
    m = (c < c_src)[:, None] & pmask[None, :]
    v = tl.load(src + c[:, None] * s_sc + (h * s_sh + w * s_sw)[None, :], mask=m)
    tl.store(out + c[:, None] * hw + offs_p[None, :], v, mask=m)


@triton.jit
def _flow_warp_cat_kernel(
    img0,
    flow0,
    scale0,
    img1,
    flow1,
    scale1,
    gx_ptr,
    gy_ptr,
    out,
    src,
    channels,
    height,
    width,
    hw,
    c_src,
    s_ic,
    s_ih,
    s_iw,
    s_fc,
    s_fh,
    s_fw,
    s_sc,
    s_sh,
    s_sw,
    out_c_off0,
    flow_out_off0,
    out_c_off1,
    flow_out_off1,
    inv_hw,
    inv_hh,
    n_warp_cb,
    has_scale: tl.constexpr,
    has_copy: tl.constexpr,
    two_dir: tl.constexpr,
    mul_rn: tl.constexpr,
    block_p: tl.constexpr,
    block_c: tl.constexpr,
):
    pid_p = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_p = pid_p * block_p + tl.arange(0, block_p)
    pmask = offs_p < hw
    h = offs_p // width
    w = offs_p - h * width
    if has_copy:
        if pid_c >= n_warp_cb:
            _copy_tile(
                src,
                out,
                offs_p,
                pmask,
                h,
                w,
                pid_c - n_warp_cb,
                c_src,
                hw,
                s_sc,
                s_sh,
                s_sw,
                block_c,
            )
        else:
            _warp_tile(
                img0,
                flow0,
                gx_ptr,
                gy_ptr,
                scale0,
                out,
                offs_p,
                pmask,
                h,
                w,
                pid_c,
                channels,
                height,
                width,
                hw,
                s_ic,
                s_ih,
                s_iw,
                s_fc,
                s_fh,
                s_fw,
                out_c_off0,
                flow_out_off0,
                inv_hw,
                inv_hh,
                has_scale,
                mul_rn,
                block_c,
            )
    elif two_dir:
        if tl.program_id(2) == 0:
            _warp_tile(
                img0,
                flow0,
                gx_ptr,
                gy_ptr,
                scale0,
                out,
                offs_p,
                pmask,
                h,
                w,
                pid_c,
                channels,
                height,
                width,
                hw,
                s_ic,
                s_ih,
                s_iw,
                s_fc,
                s_fh,
                s_fw,
                out_c_off0,
                flow_out_off0,
                inv_hw,
                inv_hh,
                has_scale,
                mul_rn,
                block_c,
            )
        else:
            _warp_tile(
                img1,
                flow1,
                gx_ptr,
                gy_ptr,
                scale1,
                out,
                offs_p,
                pmask,
                h,
                w,
                pid_c,
                channels,
                height,
                width,
                hw,
                s_ic,
                s_ih,
                s_iw,
                s_fc,
                s_fh,
                s_fw,
                out_c_off1,
                flow_out_off1,
                inv_hw,
                inv_hh,
                has_scale,
                mul_rn,
                block_c,
            )
    else:
        _warp_tile(
            img0,
            flow0,
            gx_ptr,
            gy_ptr,
            scale0,
            out,
            offs_p,
            pmask,
            h,
            w,
            pid_c,
            channels,
            height,
            width,
            hw,
            s_ic,
            s_ih,
            s_iw,
            s_fc,
            s_fh,
            s_fw,
            out_c_off0,
            flow_out_off0,
            inv_hw,
            inv_hh,
            has_scale,
            mul_rn,
            block_c,
        )


@functools.cache
def _inv_half(n: int) -> float:
    # ATen divides by the CPU scalar n * 0.5 as a multiply by its fp32 reciprocal
    one = torch.tensor(1.0, dtype=torch.float32)
    return (one / torch.tensor(n * 0.5, dtype=torch.float32)).item()


def flow_warp_cat(
    images: list[torch.Tensor],
    flows: list[torch.Tensor],
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
    prefix: torch.Tensor | None = None,
    scales: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    img0, flow0 = images[0], flows[0]
    _, channels, height, width = img0.shape
    hw = height * width
    inv_hw, inv_hh = _inv_half(width), _inv_half(height)
    if scales is None:
        # PyramidFlowEstimator form: cat([prefix, warp(image, flow)])
        block_c, num_warps = _PREFIX_CFG
        c_pre = prefix.shape[1]
        out = torch.empty(
            (1, c_pre + channels, height, width), dtype=img0.dtype, device=img0.device
        )
        n_warp_cb = triton.cdiv(channels, block_c)
        grid = (triton.cdiv(hw, _BLOCK_P), n_warp_cb + triton.cdiv(c_pre, block_c))
        _flow_warp_cat_kernel[grid](
            img0,
            flow0,
            flow0,
            img0,
            flow0,
            flow0,
            grid_x,
            grid_y,
            out,
            prefix,
            channels,
            height,
            width,
            hw,
            c_pre,
            img0.stride(1),
            img0.stride(2),
            img0.stride(3),
            flow0.stride(1),
            flow0.stride(2),
            flow0.stride(3),
            prefix.stride(1),
            prefix.stride(2),
            prefix.stride(3),
            c_pre,
            0,
            c_pre,
            0,
            inv_hw,
            inv_hh,
            n_warp_cb,
            has_scale=False,
            has_copy=True,
            two_dir=False,
            mul_rn=_MUL_RN,
            block_p=_BLOCK_P,
            block_c=block_c,
            num_warps=num_warps,
            enable_fp_fusion=False,
        )
        return out
    # fusion-level form: cat([warp(i0, f0 * s0), warp(i1, f1 * s1), f0 * s0, f1 * s1])
    img1, flow1 = images[1], flows[1]
    block_c, num_warps = _SCALED_CFG
    out = torch.empty((1, 2 * channels + 4, height, width), dtype=img0.dtype, device=img0.device)
    n_warp_cb = triton.cdiv(channels, block_c)
    grid = (triton.cdiv(hw, _BLOCK_P), n_warp_cb, 2)
    _flow_warp_cat_kernel[grid](
        img0,
        flow0,
        scales[0],
        img1,
        flow1,
        scales[1],
        grid_x,
        grid_y,
        out,
        img0,
        channels,
        height,
        width,
        hw,
        0,
        img0.stride(1),
        img0.stride(2),
        img0.stride(3),
        flow0.stride(1),
        flow0.stride(2),
        flow0.stride(3),
        0,
        0,
        0,
        0,
        2 * channels,
        channels,
        2 * channels + 2,
        inv_hw,
        inv_hh,
        n_warp_cb,
        has_scale=True,
        has_copy=False,
        two_dir=True,
        mul_rn=_MUL_RN,
        block_p=_BLOCK_P,
        block_c=block_c,
        num_warps=num_warps,
        enable_fp_fusion=False,
    )
    return out
