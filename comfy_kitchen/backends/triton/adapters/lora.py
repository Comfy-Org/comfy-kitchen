# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LoRA kernels (linear layout): bypass [base +] scale*(x@down^T)@up^T, merge w + scale*(up@down)."""

import torch
import triton
import triton.language as tl

from comfy_kitchen.backends.eager.adapters import lora as eager

from .common import (
    copy_of,
    dot_precision,
    dtype_key,
    grid_configs,
    guarded,
    launch,
    load2d,
    measurer,
    pad16,
    token_bucket,
)
from .tuning import TORCH, device_info, select
from .tuning.model import Work, cdiv, rank, shared_reads, staged, tiled_reads

# Largest padded rank the fused bypass holds in registers.
MAX_RANK = 128


@triton.jit
def _bypass_fwd_kernel(
    x_ptr,
    d_ptr,
    u_ptr,
    b_ptr,
    y_ptr,
    n_tok,
    n_out,
    n_in,
    rank,
    sxt,
    sxi,
    sdr,
    sdi,
    suo,
    sur,
    sbt,
    sbo,
    syt,
    syo,
    scale,
    has_base: tl.constexpr,
    prec: tl.constexpr,
    block_r: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    split_o: tl.constexpr,
):
    """y = [base +] scale * (x @ down^T) @ up^T for one token tile and one slice of O.

    Grid: (token tiles, split_o output slices).
    """
    # Token rows and rank columns this program owns.
    pid_t = tl.program_id(0)
    pid_o = tl.program_id(1)
    rt = pid_t * block_m + tl.arange(0, block_m)
    rr = tl.arange(0, block_r)
    mt = rt < n_tok
    mr = rr < rank

    # h (tokens, rank) in fp32 registers: x tile @ down^T over the input axis.
    h = tl.zeros((block_m, block_r), tl.float32)
    for i0 in range(0, n_in, block_k):
        ri = i0 + tl.arange(0, block_k)
        mi = ri < n_in
        xv = load2d(x_ptr, rt, ri, sxt, sxi, mt, mi)
        dv = load2d(d_ptr, ri, rr, sdi, sdr, mi, mr)
        h = tl.dot(xv, dv, h, input_precision=prec)
    # h in x's dtype, as the two-Linear chain rounds it.
    hs = h.to(x_ptr.dtype.element_ty)

    # Output columns [lo, hi) of this program's slice, streamed block_n at a time.
    per = tl.cdiv(tl.cdiv(n_out, block_n), split_o) * block_n
    lo = pid_o * per
    hi = tl.minimum(lo + per, n_out)
    for o0 in range(lo, hi, block_n):
        ro = o0 + tl.arange(0, block_n)
        mo = ro < n_out

        # y chunk = scale * h @ up^T [+ base], stored once.
        uv = load2d(u_ptr, rr, ro, sur, suo, mr, mo)
        y = tl.dot(hs, uv, input_precision=prec) * scale
        if has_base:
            y += load2d(b_ptr, rt, ro, sbt, sbo, mt, mo).to(tl.float32)
        tl.store(
            y_ptr + rt[:, None] * syt + ro[None, :] * syo,
            y.to(y_ptr.dtype.element_ty),
            mask=mt[:, None] & mo[None, :],
        )


@triton.jit
def _bypass_bwd_kernel(
    x_ptr,
    d_ptr,
    u_ptr,
    g_ptr,
    gx_ptr,
    h_ptr,
    q_ptr,
    n_tok,
    n_out,
    n_in,
    rank,
    sxt,
    sxi,
    sdr,
    sdi,
    suo,
    sur,
    sgt,
    sgo,
    sgxt,
    sgxi,
    scale,
    prec: tl.constexpr,
    block_r: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    """gx = scale * (g @ up) @ down per token tile, storing h and q for the GEMMs.

    Grid: (token tiles,).
    """
    pid = tl.program_id(0)
    rt = pid * block_m + tl.arange(0, block_m)
    rr = tl.arange(0, block_r)
    mt = rt < n_tok
    mr = rr < rank
    tr_mask = mt[:, None] & mr[None, :]

    # h = x @ down^T (tokens, rank), stored for the g_up GEMM.
    h = tl.zeros((block_m, block_r), tl.float32)
    for i0 in range(0, n_in, block_k):
        ri = i0 + tl.arange(0, block_k)
        mi = ri < n_in
        xv = load2d(x_ptr, rt, ri, sxt, sxi, mt, mi)
        dv = load2d(d_ptr, ri, rr, sdi, sdr, mi, mr)
        h = tl.dot(xv, dv, h, input_precision=prec)
    tl.store(
        h_ptr + rt[:, None] * rank + rr[None, :],
        h.to(h_ptr.dtype.element_ty),
        mask=tr_mask,
    )

    # q = g @ up (tokens, rank), stored for the g_down GEMM.
    q = tl.zeros((block_m, block_r), tl.float32)
    for o0 in range(0, n_out, block_n):
        ro = o0 + tl.arange(0, block_n)
        mo = ro < n_out
        gv = load2d(g_ptr, rt, ro, sgt, sgo, mt, mo)
        uv = load2d(u_ptr, ro, rr, suo, sur, mo, mr)
        q = tl.dot(gv, uv, q, input_precision=prec)
    qs = q.to(q_ptr.dtype.element_ty)
    tl.store(q_ptr + rt[:, None] * rank + rr[None, :], qs, mask=tr_mask)

    # gx chunk = scale * q @ down over the input axis.
    for i0 in range(0, n_in, block_k):
        ri = i0 + tl.arange(0, block_k)
        mi = ri < n_in
        dv = load2d(d_ptr, rr, ri, sdr, sdi, mr, mi)
        gxv = tl.dot(qs, dv, input_precision=prec) * scale
        tl.store(
            gx_ptr + rt[:, None] * sgxt + ri[None, :] * sgxi,
            gxv.to(gx_ptr.dtype.element_ty),
            mask=mt[:, None] & mi[None, :],
        )


@triton.jit
def _merge_fwd_kernel(
    u_ptr,
    d_ptr,
    w_ptr,
    out_ptr,
    n_out,
    n_in,
    rank,
    suo,
    sur,
    sdr,
    sdi,
    swo,
    swi,
    soo,
    soi,
    scale,
    add_base: tl.constexpr,
    prec: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_r: tl.constexpr,
):
    """out = [w +] scale * (up @ down), one program per output tile.

    Grid: (output row tiles, input column tiles).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * block_m + tl.arange(0, block_m)
    rn = pid_n * block_n + tl.arange(0, block_n)
    mm = rm < n_out
    mn = rn < n_in

    # acc (block_m, block_n) = up tile @ down tile, reduced over rank.
    acc = tl.zeros((block_m, block_n), tl.float32)
    for r0 in range(0, rank, block_r):
        rk = r0 + tl.arange(0, block_r)
        mk = rk < rank
        uv = load2d(u_ptr, rm, rk, suo, sur, mm, mk)
        dv = load2d(d_ptr, rk, rn, sdr, sdi, mk, mn)
        acc = tl.dot(uv, dv, acc, input_precision=prec)

    # Epilogue: scale, add the weight tile, store.
    out = acc * scale
    if add_base:
        out += load2d(w_ptr, rm, rn, swo, swi, mm, mn).to(tl.float32)
    tl.store(
        out_ptr + rm[:, None] * soo + rn[None, :] * soi,
        out.to(out_ptr.dtype.element_ty),
        mask=mm[:, None] & mn[None, :],
    )


@triton.jit
def _merge_bwd_kernel(
    g_ptr,
    u_ptr,
    d_ptr,
    gu_ptr,
    gd_ptr,
    n_out,
    n_in,
    rank,
    sgo,
    sgi,
    suo,
    sur,
    sdr,
    sdi,
    n_up,
    n_down,
    prec: tl.constexpr,
    block_r: tl.constexpr,
    block_m: tl.constexpr,
    block_k: tl.constexpr,
):
    """fp32 partials of g_up (first n_up*S programs) and g_down (the rest), slab s per split s."""
    pid = tl.program_id(0)
    splits = tl.num_programs(0) // (n_up + n_down)
    rr = tl.arange(0, block_r)
    mr = rr < rank
    ct = u_ptr.dtype.element_ty

    # g_up role: rows of g_up (block_m, rank) = g @ down^T over split `part` of the input axis.
    if pid < n_up * splits:
        part = pid // n_up
        rm = (pid % n_up) * block_m + tl.arange(0, block_m)
        mm = rm < n_out
        per = tl.cdiv(tl.cdiv(n_in, block_k), splits) * block_k
        lo = part * per
        hi = tl.minimum(lo + per, n_in)
        acc = tl.zeros((block_m, block_r), tl.float32)
        for i0 in range(lo, hi, block_k):
            ri = i0 + tl.arange(0, block_k)
            mi = ri < hi
            gv = load2d(g_ptr, rm, ri, sgo, sgi, mm, mi).to(ct)
            dv = load2d(d_ptr, ri, rr, sdi, sdr, mi, mr)
            acc = tl.dot(gv, dv, acc, input_precision=prec)
        tl.store(
            gu_ptr + (part * n_out + rm)[:, None] * rank + rr[None, :],
            acc,
            mask=mm[:, None] & mr[None, :],
        )
    # g_down role: columns of g_down (rank, block_m) = up^T @ g over split `part_d` of the output axis.
    else:
        q = pid - n_up * splits
        part_d = q // n_down
        rn = (q % n_down) * block_m + tl.arange(0, block_m)
        mn = rn < n_in
        per_d = tl.cdiv(tl.cdiv(n_out, block_k), splits) * block_k
        lo_d = part_d * per_d
        hi_d = tl.minimum(lo_d + per_d, n_out)
        acc_d = tl.zeros((block_r, block_m), tl.float32)
        for o0 in range(lo_d, hi_d, block_k):
            ro = o0 + tl.arange(0, block_k)
            mo = ro < hi_d
            gt = load2d(g_ptr, ro, rn, sgo, sgi, mo, mn).to(ct)
            ut = load2d(u_ptr, rr, ro, sur, suo, mr, mo)
            acc_d = tl.dot(ut, gt, acc_d, input_precision=prec)
        tl.store(
            gd_ptr + (part_d * rank + rr)[:, None] * n_in + rn[None, :],
            acc_d,
            mask=mr[:, None] & mn[None, :],
        )


@triton.jit
def _merge_bwd_tile_kernel(
    g_ptr,
    u_ptr,
    d_ptr,
    gu_ptr,
    gd_ptr,
    n_out,
    n_in,
    rank,
    sgo,
    sgi,
    suo,
    sur,
    sdr,
    sdi,
    prec: tl.constexpr,
    block_r: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    """fp32 partials of g_up (slab per split) and g_down (slab per row tile) from one read of g.

    Grid: (output row tiles, splits of the input axis).
    """
    pid_m = tl.program_id(0)
    pid_s = tl.program_id(1)
    splits = tl.num_programs(1)
    rm = pid_m * block_m + tl.arange(0, block_m)
    rr = tl.arange(0, block_r)
    mm = rm < n_out
    mr = rr < rank
    ct = u_ptr.dtype.element_ty

    # up^T rows of this row tile, held for every g tile; acc = this split's g_up partial.
    ut = load2d(u_ptr, rr, rm, sur, suo, mr, mm)
    acc = tl.zeros((block_m, block_r), tl.float32)

    # Input columns [lo, hi) of this split, one g tile at a time.
    per = tl.cdiv(tl.cdiv(n_in, block_n), splits) * block_n
    lo = pid_s * per
    hi = tl.minimum(lo + per, n_in)
    for i0 in range(lo, hi, block_n):
        ri = i0 + tl.arange(0, block_n)
        mi = ri < hi
        gv = load2d(g_ptr, rm, ri, sgo, sgi, mm, mi).to(ct)
        dv = load2d(d_ptr, ri, rr, sdi, sdr, mi, mr)

        # g_up += g @ down^T; g_down tile = up^T @ g, stored per row tile.
        acc = tl.dot(gv, dv, acc, input_precision=prec)
        gd = tl.dot(ut, gv, input_precision=prec)
        tl.store(
            gd_ptr + (pid_m * rank + rr)[:, None] * n_in + ri[None, :],
            gd,
            mask=mr[:, None] & mi[None, :],
        )
    tl.store(
        gu_ptr + (pid_s * n_out + rm)[:, None] * rank + rr[None, :],
        acc,
        mask=mm[:, None] & mr[None, :],
    )


def _plan_bypass_fwd(dev, t, o, i, r, e, has_base):
    """Bypass forward configs, model-ranked, then torch."""
    rp = pad16(r)
    if rp > MAX_RANK:
        return [TORCH]
    space = grid_configs(
        dev,
        BLOCK_M=(16, 32, 64, 128),
        BLOCK_N=(32, 64, 128),
        BLOCK_K=(32, 64, 128),
        SPLIT_O=(1, 2, 4, 8),
        num_warps=(2, 4, 8),
    )

    def work(c):
        bm, bn, bk, split = c["BLOCK_M"], c["BLOCK_N"], c["BLOCK_K"], c["SPLIT_O"]
        tiles = cdiv(t, bm)
        return Work(
            grid=tiles * split,
            warps=c["num_warps"],
            acc_fp32=bm * rp + bm * bn,
            operand_bytes=0,
            smem_stage=max((bm * bk + bk * rp) * e, rp * bn * e),
            stages=c["num_stages"],
            hbm_bytes=t * i * e * split
            + shared_reads(dev, (i + o) * rp * e, tiles * split)
            + t * o * e * (2 if has_base else 1),
            mma_flops=2.0 * tiles * bm * rp * (i * split + o),
            # every program reads all of down and its 1/split of up
            l2_bytes=t * i * e * split
            + tiles * split * (i + cdiv(o, split)) * rp * e
            + t * o * e * (2 if has_base else 1),
            # loads in flight: loop 1 an x tile per stage; loop 2 a base tile (an up tile without base)
            phases=(
                (t * i * e * split, staged(dev, bm * bk * e, c["num_stages"])),
                (
                    t * o * e * (2 if has_base else 1),
                    (bm * bn if has_base else bn * rp) * e,
                ),
            ),
        )

    return [*rank(dev, space, work), TORCH]


def _plan_bypass_bwd(dev, t, o, i, r, e):
    rp = pad16(r)
    if rp > MAX_RANK:
        return [TORCH]
    space = grid_configs(
        dev,
        BLOCK_M=(16, 32, 64, 128),
        BLOCK_N=(32, 64, 128),
        BLOCK_K=(32, 64, 128),
        num_warps=(2, 4, 8),
    )

    def work(c):
        bm, bn, bk = c["BLOCK_M"], c["BLOCK_N"], c["BLOCK_K"]
        tiles = cdiv(t, bm)
        return Work(
            grid=tiles,
            warps=c["num_warps"],
            acc_fp32=2 * bm * rp + bm * bk,
            operand_bytes=0,
            smem_stage=max((bm * bk + bk * rp) * e, (bm * bn + bn * rp) * e),
            stages=c["num_stages"],
            hbm_bytes=(2 * t * i + t * o + 2 * t * r) * e
            + shared_reads(dev, (i + o) * rp * e, tiles),
            mma_flops=2.0 * tiles * bm * rp * (2 * i + o),
            # every token tile reads both factors whole
            l2_bytes=(2 * t * i + t * o + 2 * t * r) * e + tiles * (i + o) * rp * e,
        )

    return [*rank(dev, space, work), TORCH]


def _plan_merge(dev, o, i, r, e_w, e_f):
    rp = pad16(r)
    brs = tuple(sorted({min(rp, 16), min(rp, 32), min(rp, 64)}))
    space = grid_configs(
        dev,
        BLOCK_M=(32, 64, 128),
        BLOCK_N=(32, 64, 128, 256),
        BLOCK_R=brs,
        num_warps=(2, 4, 8),
    )

    def work(c):
        bm, bn, br = c["BLOCK_M"], c["BLOCK_N"], c["BLOCK_R"]
        tm, tn = cdiv(o, bm), cdiv(i, bn)
        return Work(
            grid=tm * tn,
            warps=c["num_warps"],
            acc_fp32=bm * bn,
            operand_bytes=0,
            smem_stage=(bm + bn) * br * e_f,
            stages=c["num_stages"],
            hbm_bytes=2 * o * i * e_w
            + shared_reads(dev, (o + i) * rp * e_f, max(tm, tn)),
            mma_flops=2.0 * tm * bm * tn * bn * rp,
            l2_bytes=2 * o * i * e_w + tiled_reads(o, i, rp, bm, bn, e_f),
        )

    return [*rank(dev, space, work), TORCH]


def _plan_merge_bwd(dev, o, i, r, e_g, e_f):
    rp = pad16(r)
    if rp > MAX_RANK:
        return [TORCH]
    roles = grid_configs(
        dev,
        LAYOUT=("roles",),
        BLOCK_M=(32, 64, 128),
        BLOCK_K=(32, 64, 128),
        SPLIT=(1, 2, 4, 8, 16),
        num_warps=(2, 4, 8),
    )
    tile = grid_configs(
        dev,
        LAYOUT=("tile",),
        BLOCK_M=(32, 64, 128),
        BLOCK_N=(32, 64, 128),
        SPLIT=(1, 2, 4, 8, 16),
        num_warps=(2, 4, 8),
    )

    def work(c):
        bm, split = c["BLOCK_M"], c["SPLIT"]
        weights = shared_reads(dev, (o + i) * rp * e_f, cdiv(o, bm) * split)
        if c["LAYOUT"] == "roles":
            bk = c["BLOCK_K"]
            return Work(
                grid=(cdiv(o, bm) + cdiv(i, bm)) * split,
                warps=c["num_warps"],
                acc_fp32=bm * rp,
                operand_bytes=0,
                smem_stage=(bm * bk + bk * rp) * e_f,
                stages=c["num_stages"],
                hbm_bytes=2 * o * i * e_g + 2 * split * (o + i) * rp * 4 + weights,
                mma_flops=4.0 * o * i * rp,
                # g_up programs read down's columns, g_down programs up's rows
                l2_bytes=2 * o * i * e_g
                + (cdiv(o, bm) * i + cdiv(i, bm) * o) * rp * e_f
                + 2 * split * (o + i) * rp * 4,
            )
        bn = c["BLOCK_N"]
        tm = cdiv(o, bm)
        return Work(
            grid=tm * split,
            warps=c["num_warps"],
            acc_fp32=bm * rp + rp * bn,
            operand_bytes=0,
            smem_stage=(bm * bn + bn * rp + rp * bm) * e_f,
            stages=c["num_stages"],
            hbm_bytes=o * i * e_g + 2 * (split * o + tm * i) * rp * 4 + weights,
            mma_flops=4.0 * o * i * rp,
            # each (row tile, split) reads its slice of down and the tile's rows of up
            l2_bytes=o * i * e_g
            + (tm * i + tm * split * bm) * rp * e_f
            + 2 * (split * o + tm * i) * rp * 4,
        )

    return [*rank(dev, roles + tile, work), TORCH]


def _bypass_fwd_run(cfg, flat, u, d, scale, base2, out_dtype):
    if cfg == TORCH:
        return eager.lora_bypass(flat, u, d, scale, base2)
    t, i = flat.shape
    o, r = u.shape
    out = torch.empty(t, o, device=flat.device, dtype=out_dtype)
    b = base2 if base2 is not None else out
    grid = (triton.cdiv(t, cfg["BLOCK_M"]), cfg["SPLIT_O"])
    _bypass_fwd_kernel[grid](
        flat,
        d,
        u,
        b,
        out,
        t,
        o,
        i,
        r,
        *flat.stride(),
        *d.stride(),
        *u.stride(),
        *b.stride(),
        *out.stride(),
        scale,
        has_base=base2 is not None,
        prec=dot_precision(device_info(flat.device), flat.dtype),
        block_r=pad16(r),
        block_m=cfg["BLOCK_M"],
        block_n=cfg["BLOCK_N"],
        block_k=cfg["BLOCK_K"],
        split_o=cfg["SPLIT_O"],
        **launch(cfg),
    )
    return out


def _bypass_bwd_run(cfg, g2, flat, u, d, scale):
    """(gx, g_up, g_down) in the compute dtype, all operands already in it."""
    if cfg == TORCH:
        return eager.lora_bypass_backward(g2, flat, u, d, scale)
    t, i = flat.shape
    o, r = u.shape
    gx = torch.empty_like(flat)
    h = torch.empty(t, r, device=flat.device, dtype=flat.dtype)
    q = torch.empty_like(h)
    _bypass_bwd_kernel[(triton.cdiv(t, cfg["BLOCK_M"]),)](
        flat,
        d,
        u,
        g2,
        gx,
        h,
        q,
        t,
        o,
        i,
        r,
        *flat.stride(),
        *d.stride(),
        *u.stride(),
        *g2.stride(),
        *gx.stride(),
        scale,
        prec=dot_precision(device_info(flat.device), flat.dtype),
        block_r=pad16(r),
        block_m=cfg["BLOCK_M"],
        block_n=cfg["BLOCK_N"],
        block_k=cfg["BLOCK_K"],
        **launch(cfg),
    )
    gu = torch.mm(g2.t(), h).mul_(scale)
    gd = torch.mm(q.t(), flat).mul_(scale)
    return gx, gu, gd


def _merge_run(cfg, weight2, u, d, scale, add_base=True, out_dtype=None):
    o, r = u.shape
    i = d.shape[1]
    out = torch.empty(o, i, device=u.device, dtype=out_dtype or weight2.dtype)
    if cfg == TORCH:
        delta = (u @ d).mul_(scale)
        return out.copy_(weight2 + delta if add_base else delta)
    grid = (triton.cdiv(o, cfg["BLOCK_M"]), triton.cdiv(i, cfg["BLOCK_N"]))
    w = weight2 if add_base else out
    _merge_fwd_kernel[grid](
        u,
        d,
        w,
        out,
        o,
        i,
        r,
        *u.stride(),
        *d.stride(),
        *w.stride(),
        *out.stride(),
        scale,
        add_base=add_base,
        prec=dot_precision(device_info(u.device), u.dtype),
        block_m=cfg["BLOCK_M"],
        block_n=cfg["BLOCK_N"],
        block_r=cfg["BLOCK_R"],
        **launch(cfg),
    )
    return out


def _merge_bwd_run(cfg, g2, u, d, scale):
    """(g_up, g_down) in the factors' dtype from the (O, I) grad."""
    if cfg == TORCH:
        g = g2.to(u.dtype)
        return torch.mm(g, d.t()).mul_(scale), torch.mm(u.t(), g).mul_(scale)
    o, r = u.shape
    i = d.shape[1]
    prec = dot_precision(device_info(u.device), u.dtype)
    bm = cfg["BLOCK_M"]
    if cfg["LAYOUT"] == "tile":
        n_m = triton.cdiv(o, bm)
        split = max(1, min(cfg["SPLIT"], triton.cdiv(i, cfg["BLOCK_N"])))
        gu = torch.empty(split, o, r, device=u.device, dtype=torch.float32)
        gd = torch.empty(n_m, r, i, device=u.device, dtype=torch.float32)
        _merge_bwd_tile_kernel[(n_m, split)](
            g2,
            u,
            d,
            gu,
            gd,
            o,
            i,
            r,
            *g2.stride(),
            *u.stride(),
            *d.stride(),
            prec=prec,
            block_r=pad16(r),
            block_m=bm,
            block_n=cfg["BLOCK_N"],
            **launch(cfg),
        )
    else:
        n_up, n_down = triton.cdiv(o, bm), triton.cdiv(i, bm)
        split = max(1, min(cfg["SPLIT"], triton.cdiv(min(o, i), cfg["BLOCK_K"])))
        gu = torch.empty(split, o, r, device=u.device, dtype=torch.float32)
        gd = torch.empty(split, r, i, device=u.device, dtype=torch.float32)
        _merge_bwd_kernel[((n_up + n_down) * split,)](
            g2,
            u,
            d,
            gu,
            gd,
            o,
            i,
            r,
            *g2.stride(),
            *u.stride(),
            *d.stride(),
            n_up,
            n_down,
            prec=prec,
            block_r=pad16(r),
            block_m=bm,
            block_k=cfg["BLOCK_K"],
            **launch(cfg),
        )
    return gu.sum(0).mul_(scale).to(u.dtype), gd.sum(0).mul_(scale).to(d.dtype)


def lora_bypass(
    x: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    scale: float = 1.0,
    base: torch.Tensor | None = None,
) -> torch.Tensor:
    """[base +] scale * (x @ down^T) @ up^T for a linear adapter; x is (..., in)."""
    dev = device_info(x.device)
    lead = x.shape[:-1]
    flat = x.reshape(-1, x.shape[-1])
    t, i = flat.shape
    o, r = up.shape
    u, d = up.to(x.dtype), down.to(x.dtype)
    base2 = base.reshape(-1, o) if base is not None else None
    out_dtype = x.dtype if base is None else base.dtype
    key = (
        "lora_bypass",
        (dtype_key(x.dtype), base is not None),
        (token_bucket(t), o, i, r),
    )

    def inputs():
        return copy_of(flat), copy_of(base2)

    cfg = select(
        dev,
        key,
        lambda: _plan_bypass_fwd(dev, t, o, i, r, x.element_size(), base is not None),
        measurer(
            dev,
            (t * (i + o * (2 if base2 is not None else 1))) * x.element_size(),
            inputs,
            lambda c, s: lambda: _bypass_fwd_run(c, s[0], u, d, scale, s[1], out_dtype),
        ),
    )
    out = guarded(
        dev, key, cfg, lambda c: _bypass_fwd_run(c, flat, u, d, scale, base2, out_dtype)
    )
    return out.reshape(*lead, o)


def lora_bypass_backward(
    grad: torch.Tensor,
    x: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(grad_x, grad_up, grad_down) for :func:`lora_bypass`."""
    dev = device_info(x.device)
    flat = x.reshape(-1, x.shape[-1])
    t, i = flat.shape
    o, r = up.shape
    u, d = up.to(x.dtype), down.to(x.dtype)
    g2 = grad.reshape(-1, o).to(x.dtype)
    key = ("lora_bypass_backward", (dtype_key(x.dtype),), (token_bucket(t), o, i, r))

    def inputs():
        return copy_of(g2), copy_of(flat)

    cfg = select(
        dev,
        key,
        lambda: _plan_bypass_bwd(dev, t, o, i, r, x.element_size()),
        measurer(
            dev,
            2 * t * (i + o) * x.element_size(),
            inputs,
            lambda c, s: lambda: _bypass_bwd_run(c, s[0], s[1], u, d, scale),
        ),
    )
    gx, gu, gd = guarded(
        dev, key, cfg, lambda c: _bypass_bwd_run(c, g2, flat, u, d, scale)
    )
    return gx.reshape(x.shape), gu.to(up.dtype), gd.to(down.dtype)


def lora_merge(
    weight: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """weight + scale * (up @ down), in the weight's dtype and shape."""
    dev = device_info(weight.device)
    ct = torch.promote_types(up.dtype, down.dtype)
    u, d = up.to(ct), down.to(ct)
    o, r = u.shape
    i = d.shape[1]
    w2 = weight.reshape(o, -1)
    key = ("lora_merge", (dtype_key(weight.dtype), dtype_key(ct)), (o, i, r))
    cfg = select(
        dev,
        key,
        lambda: _plan_merge(dev, o, i, r, weight.element_size(), u.element_size()),
        measurer(
            dev,
            2 * o * i * weight.element_size(),
            lambda: (copy_of(w2),),
            lambda c, s: lambda: _merge_run(c, s[0], u, d, scale),
        ),
    )
    return guarded(dev, key, cfg, lambda c: _merge_run(c, w2, u, d, scale)).reshape(
        weight.shape
    )


def lora_merge_backward(
    grad: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(grad_up, grad_down) for :func:`lora_merge`; grad_weight is grad itself."""
    dev = device_info(up.device)
    ct = torch.promote_types(up.dtype, down.dtype)
    u, d = up.to(ct), down.to(ct)
    o, r = u.shape
    i = d.shape[1]
    g2 = grad.reshape(o, -1)
    key = ("lora_merge_backward", (dtype_key(grad.dtype), dtype_key(ct)), (o, i, r))
    cfg = select(
        dev,
        key,
        lambda: _plan_merge_bwd(dev, o, i, r, grad.element_size(), u.element_size()),
        measurer(
            dev,
            o * i * grad.element_size(),
            lambda: (copy_of(g2),),
            lambda c, s: lambda: _merge_bwd_run(c, s[0], u, d, scale),
        ),
    )
    gu, gd = guarded(dev, key, cfg, lambda c: _merge_bwd_run(c, g2, u, d, scale))
    return gu.to(up.dtype), gd.to(down.dtype)
