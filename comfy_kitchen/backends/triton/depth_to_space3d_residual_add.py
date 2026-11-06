# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton depth_to_space3d(repeat_interleave(x)) + residual as one gather-add.

Each program owns ``block_r`` output rows (one ``(b, c, t, h)`` each) by ``block_w`` output
columns. For every output element it computes the source channel and position of the
rearranged shortcut, loads it and the residual element, adds them in fp32 and rounds once to
the output dtype, which is what the eager fp16 / bf16 / fp32 add does, so the result is
bit-identical to the eager backend. The repeated and rearranged shortcut is never written
out. All operands are read and written through their strides.

- ``(c, ft, fs, fs)`` order (Wan 2.2 ``DupUp3D``): residual has the output shape; the output
  is a new tensor with residual's layout, or residual itself for the in-place form.
- ``(ft, fs, fs, c)`` order (HunyuanVideo 1.5 ``UpSmpl``, factor_s 2): residual is packed and
  rearranged by the same index map; the output is a new contiguous tensor.

The tile shape and warp count are autotuned per output size (key ``n_wo``, ``n_rows`` and the
factors) over 7 tile shapes x num_warps {4, 8}: on every Wan 2.2, Qwen-Image 2.1 and
HunyuanVideo 1.5 shape measured it came within 4% of the fixed tiles it was compared with
(1 x next_pow2(width), and 4 x 256 for the Wan 2.2 order) on an RTX 5090 and within 10%
on a Radeon 8060S (gfx1151).
"""

import torch

import triton
import triton.language as tl


@triton.jit
def _d2s3d_residual_add_kernel(
    x_ptr,
    main_ptr,
    out_ptr,
    n_rows,
    n_oc,
    n_to,
    n_ho,
    n_wo,
    t_off,
    sxb,
    sxc,
    sxt,
    sxh,
    sxw,
    smb,
    smc,
    smt,
    smh,
    smw,
    sob,
    soc,
    sot,
    soh,
    sow,
    ft: tl.constexpr,
    fs: tl.constexpr,
    rep: tl.constexpr,
    block_r: tl.constexpr,
    block_w: tl.constexpr,
):
    # out[b, oc, to, ho, wo] = main[b, oc, to, ho, wo] + x[b, ci, t, h, w] with
    #   t, it = divmod(to + t_off, ft); h, ih = divmod(ho, fs); w, iw = divmod(wo, fs)
    #   ci = (((oc * ft + it) * fs + ih) * fs + iw) // rep
    pid_r = tl.program_id(0)
    pid_w = tl.program_id(1)
    rows = pid_r.to(tl.int64) * block_r + tl.arange(0, block_r)
    cols = pid_w * block_w + tl.arange(0, block_w)
    rmask = rows < n_rows
    cmask = cols < n_wo

    # row -> (b, oc, to, ho)
    ho = rows % n_ho
    r2 = rows // n_ho
    to = r2 % n_to
    r3 = r2 // n_to
    oc = r3 % n_oc
    b = r3 // n_oc
    t_full = to + t_off
    t = t_full // ft
    it = t_full % ft
    h = ho // fs
    ih = ho % fs
    ce_row = ((oc * ft + it) * fs + ih) * fs  # [block_r]
    # col -> (w, iw)
    w = cols // fs
    iw = cols % fs
    ci = (ce_row[:, None] + iw[None, :]) // rep  # [block_r, block_w]

    x_off = (b * sxb + t * sxt + h * sxh)[:, None] + ci * sxc + (w * sxw)[None, :]
    m_off = (b * smb + oc * smc + to * smt + ho * smh)[:, None] + (cols * smw)[None, :]
    o_off = (b * sob + oc * soc + to * sot + ho * soh)[:, None] + (cols * sow)[None, :]
    mask = rmask[:, None] & cmask[None, :]

    xv = tl.load(x_ptr + x_off, mask=mask, other=0.0)
    mv = tl.load(main_ptr + m_off, mask=mask, other=0.0)
    y = mv.to(tl.float32) + xv.to(tl.float32)
    tl.store(out_ptr + o_off, y.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _pixel_shuffle3d_add_kernel(
    h_ptr,
    x_ptr,
    out_ptr,
    n_rows,
    nc,
    n_to,
    n_ho,
    n_wo,
    shb,
    shc,
    sht,
    shh,
    shw,
    sxb,
    sxc,
    sxt,
    sxh,
    sxw,
    sob,
    soc,
    sot,
    soh,
    sow,
    r1: tl.constexpr,
    rep: tl.constexpr,
    block_r: tl.constexpr,
    block_w: tl.constexpr,
):
    # out[b, oc, to, ho, wo] = h[b, c, t, hh, w] + x[b, c // rep, t, hh, w] with
    #   t, it = divmod(to, r1); hh, ih = divmod(ho, 2); w, iw = divmod(wo, 2)
    #   c = ((it * 2 + ih) * 2 + iw) * nc + oc
    pid_r = tl.program_id(0)
    pid_w = tl.program_id(1)
    rows = pid_r.to(tl.int64) * block_r + tl.arange(0, block_r)
    cols = pid_w * block_w + tl.arange(0, block_w)
    rmask = rows < n_rows
    cmask = cols < n_wo

    ho = rows % n_ho
    r2 = rows // n_ho
    to = r2 % n_to
    r3 = r2 // n_to
    oc = r3 % nc
    b = r3 // nc
    t = to // r1
    it = to % r1
    hh = ho // 2
    ih = ho % 2
    c_row = (it * 2 + ih) * 2 * nc + oc  # [block_r]
    w = cols // 2
    iw = cols % 2
    c = c_row[:, None] + (iw * nc)[None, :]  # [block_r, block_w]
    ci = c // rep

    hw_off = (w * shw)[None, :]
    h_off = (b * shb + t * sht + hh * shh)[:, None] + c * shc + hw_off
    x_off = (b * sxb + t * sxt + hh * sxh)[:, None] + ci * sxc + (w * sxw)[None, :]
    o_off = (b * sob + oc * soc + to * sot + ho * soh)[:, None] + (cols * sow)[None, :]
    mask = rmask[:, None] & cmask[None, :]

    hv = tl.load(h_ptr + h_off, mask=mask, other=0.0)
    xv = tl.load(x_ptr + x_off, mask=mask, other=0.0)
    y = hv.to(tl.float32) + xv.to(tl.float32)
    tl.store(out_ptr + o_off, y.to(out_ptr.dtype.element_ty), mask=mask)


_CONFIGS = [
    triton.Config({"block_r": r, "block_w": w}, num_warps=nw)
    for (r, w) in [(1, 1024), (2, 512), (4, 256), (8, 128), (2, 1024), (4, 512), (8, 256)]
    for nw in (4, 8)
]

# restore_value: the in-place form writes into its residual, so the autotuner's benchmark runs
# must not add the shortcut more than once
_d2s3d_residual_add = triton.autotune(
    configs=_CONFIGS,
    key=["n_wo", "n_rows", "ft", "fs", "rep"],
    restore_value=["out_ptr"],
)(_d2s3d_residual_add_kernel)

_pixel_shuffle3d_add = triton.autotune(configs=_CONFIGS, key=["n_wo", "n_rows", "r1", "rep"])(
    _pixel_shuffle3d_add_kernel
)


def _launch_d2s3d(x, residual, out, factor_t, factor_s, repeats, first_chunk):
    b, oc, to, ho, wo = out.shape
    n_rows = b * oc * to * ho
    t_off = factor_t - 1 if first_chunk else 0
    grid = lambda meta: (triton.cdiv(n_rows, meta["block_r"]), triton.cdiv(wo, meta["block_w"]))  # noqa: E731
    _d2s3d_residual_add[grid](
        x,
        residual,
        out,
        n_rows,
        oc,
        to,
        ho,
        wo,
        t_off,
        *x.stride(),
        *residual.stride(),
        *out.stride(),
        ft=factor_t,
        fs=factor_s,
        rep=repeats,
    )
    return out


def depth_to_space3d_residual_add(
    x: torch.Tensor,
    residual: torch.Tensor,
    factor_t: int,
    factor_s: int,
    repeats: int = 1,
    first_chunk: bool = False,
    factors_first: bool = False,
    residual_packed: bool = False,
) -> torch.Tensor:
    if not factors_first:
        # empty_like keeps residual's layout, as the eager add does for these inputs
        out = torch.empty_like(residual)
        return _launch_d2s3d(x, residual, out, factor_t, factor_s, repeats, first_chunk)
    # (ft, 2, 2, c) order with a packed residual, factor_s 2 (the call rule declines the rest)
    b, c, t, h, w = residual.shape
    nc = c // (factor_t * 4)
    out = torch.empty(
        (b, nc, t * factor_t, 2 * h, 2 * w), dtype=residual.dtype, device=residual.device
    )
    _, _, to, ho, wo = out.shape
    n_rows = b * nc * to * ho
    grid = lambda meta: (triton.cdiv(n_rows, meta["block_r"]), triton.cdiv(wo, meta["block_w"]))  # noqa: E731
    _pixel_shuffle3d_add[grid](
        residual,
        x,
        out,
        n_rows,
        nc,
        to,
        ho,
        wo,
        *residual.stride(),
        *x.stride(),
        *out.stride(),
        r1=factor_t,
        rep=repeats,
    )
    return out


def depth_to_space3d_residual_add_(
    x: torch.Tensor,
    residual: torch.Tensor,
    factor_t: int,
    factor_s: int,
    repeats: int = 1,
    first_chunk: bool = False,
) -> torch.Tensor:
    # one launch over the whole residual; the eager form's row strips only bound its temporary
    return _launch_d2s3d(x, residual, residual, factor_t, factor_s, repeats, first_chunk)
