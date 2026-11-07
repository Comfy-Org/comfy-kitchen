# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OFT kernels: out = rescale * R'^T x [- x] along a channel axis, x viewed as (outer, C, inner)."""

import torch
import triton
import triton.language as tl

from comfy_kitchen.backends.eager.adapters import oft as eager
from comfy_kitchen.backends.eager.adapters.common import absent

from .common import (
    copy_of,
    dot_precision,
    dtype_key,
    grid_configs,
    guarded,
    launch,
    measurer,
    pad16,
    token_bucket,
)
from .tuning import TORCH, device_info, select
from .tuning.model import Work, cdiv, rank

# Largest padded block the in-kernel Cayley handles; larger blocks use torch.
MAX_CAYLEY = 64


@triton.jit
def _rotation_kernel(
    b_ptr,
    c_ptr,
    res_ptr,
    rf_ptr,
    r_ptr,
    m_ptr,
    size,
    sbk,
    sbi,
    sbj,
    multiplier,
    constrain: tl.constexpr,
    rescale_on: tl.constexpr,
    shift: tl.constexpr,
    write_rm: tl.constexpr,
    ps: tl.constexpr,
):
    """Rf = rescale * R'^T - shift*I for one block, and R, M when asked.

    Grid: (blocks,); one (ps, ps) block per program, all in registers.
    """
    pid = tl.program_id(0)
    ri = tl.arange(0, ps)
    rj = tl.arange(0, ps)
    mi = ri < size

    # q = B - B^T, optionally scaled by the constraint factor.
    blk = tl.load(
        b_ptr + pid * sbk + ri[:, None] * sbi + rj[None, :] * sbj,
        mask=mi[:, None] & mi[None, :],
        other=0.0,
    ).to(tl.float32)
    q = blk - tl.trans(blk)
    if constrain:
        q = q * tl.load(c_ptr)

    # Gauss-Jordan on [I - q | I]: after column j, m holds (I - q)^-1 = M.
    eye = (ri[:, None] == rj[None, :]).to(tl.float32)
    a = eye - q
    m = eye
    for j in tl.static_range(ps):
        is_row = ri[:, None] == j
        pr_a = tl.sum(tl.where(is_row, a, 0.0), axis=0)
        pr_m = tl.sum(tl.where(is_row, m, 0.0), axis=0)
        pj = tl.sum(tl.where(rj == j, pr_a, 0.0))
        fac = tl.sum(tl.where(rj[None, :] == j, a, 0.0), axis=1)
        adj = tl.where(is_row, 0.0, fac[:, None])
        a = a - adj * (pr_a / pj)[None, :]
        m = m - adj * (pr_m / pj)[None, :]
        a = tl.where(is_row, pr_a[None, :] / pj, a)
        m = tl.where(is_row, pr_m[None, :] / pj, m)

    # R = (I + q) M; Rf = rescale * (m*R + (1-m)*I)^T [- I].
    r = tl.dot(eye + q, m, input_precision="ieee")
    rf = tl.trans(multiplier * r + (1.0 - multiplier) * eye)
    if rescale_on:
        res = tl.load(res_ptr + pid * size + ri, mask=mi, other=1.0).to(tl.float32)
        rf = res[:, None] * rf
    if shift:
        rf = rf - eye

    # Store padded Rf, and R, M for the backward.
    tl.store(rf_ptr + pid * ps * ps + ri[:, None] * ps + rj[None, :], rf)
    if write_rm:
        mask = mi[:, None] & mi[None, :]
        off = pid * size * size + ri[:, None] * size + rj[None, :]
        tl.store(r_ptr + off, r, mask=mask)
        tl.store(m_ptr + off, m, mask=mask)


@triton.jit
def _cols_fwd_kernel(
    rf_ptr,
    x_ptr,
    out_ptr,
    size,
    inner,
    n_cols,
    sxo,
    sxc,
    sxi,
    soo,
    soc,
    soi,
    prec: tl.constexpr,
    dot16: tl.constexpr,
    ps: tl.constexpr,
    block_n: tl.constexpr,
):
    """out = Rf x for one block over one slice of the (outer, inner) columns.

    Grid: (blocks, column groups).
    """
    pid_k = tl.program_id(0)
    pid_g = tl.program_id(1)
    groups = tl.num_programs(1)
    ri = tl.arange(0, ps)
    rj = tl.arange(0, ps)
    mi = ri < size
    ch = pid_k * size + ri

    # This block's Rf, held for every column tile.
    rf = tl.load(rf_ptr + pid_k * ps * ps + ri[:, None] * ps + rj[None, :])
    if dot16:
        rf = rf.to(x_ptr.dtype.element_ty)

    # Columns [lo, hi) of this group; column n is (outer n // inner, inner n % inner).
    span = tl.cdiv(tl.cdiv(n_cols, block_n), groups) * block_n
    lo = pid_g * span
    hi = tl.minimum(lo + span, n_cols)
    for c0 in range(lo, hi, block_n):
        rn = c0 + tl.arange(0, block_n)
        o = rn // inner
        i = rn % inner
        m = mi[:, None] & (rn < hi)[None, :]
        x = tl.load(
            x_ptr + o[None, :] * sxo + ch[:, None] * sxc + i[None, :] * sxi,
            mask=m,
            other=0.0,
        )
        if not dot16:
            x = x.to(tl.float32)
        out = tl.dot(rf, x, input_precision=prec)
        tl.store(
            out_ptr + o[None, :] * soo + ch[:, None] * soc + i[None, :] * soi,
            out.to(out_ptr.dtype.element_ty),
            mask=m,
        )


@triton.jit
def _cols_bwd_kernel(
    rf_ptr,
    x_ptr,
    g_ptr,
    gx_ptr,
    part_ptr,
    size,
    inner,
    n_cols,
    sxo,
    sxc,
    sxi,
    sgo,
    sgc,
    sgi,
    sdo,
    sdc,
    sdi,
    prec: tl.constexpr,
    same_dtype: tl.constexpr,
    dot16: tl.constexpr,
    ps: tl.constexpr,
    block_n: tl.constexpr,
):
    """gx = Rf^T g over one column slice, and its partial of g(Rf) = g x^T.

    Grid: (blocks, column groups); partials land at part[block, group].
    """
    pid_k = tl.program_id(0)
    pid_g = tl.program_id(1)
    groups = tl.num_programs(1)
    ri = tl.arange(0, ps)
    rj = tl.arange(0, ps)
    mi = ri < size
    ch = pid_k * size + ri

    # This block's Rf^T, held for every column tile.
    rft = tl.trans(tl.load(rf_ptr + pid_k * ps * ps + ri[:, None] * ps + rj[None, :]))
    if dot16:
        rft = rft.to(g_ptr.dtype.element_ty)

    # grf (ps, ps) = this group's sum of g x^T.
    grf = tl.zeros((ps, ps), tl.float32)
    span = tl.cdiv(tl.cdiv(n_cols, block_n), groups) * block_n
    lo = pid_g * span
    hi = tl.minimum(lo + span, n_cols)
    for c0 in range(lo, hi, block_n):
        rn = c0 + tl.arange(0, block_n)
        o = rn // inner
        i = rn % inner
        m = mi[:, None] & (rn < hi)[None, :]
        g = tl.load(
            g_ptr + o[None, :] * sgo + ch[:, None] * sgc + i[None, :] * sgi,
            mask=m,
            other=0.0,
        )
        x = tl.load(
            x_ptr + o[None, :] * sxo + ch[:, None] * sxc + i[None, :] * sxi,
            mask=m,
            other=0.0,
        )
        gr = g if dot16 else g.to(tl.float32)
        tl.store(
            gx_ptr + o[None, :] * sdo + ch[:, None] * sdc + i[None, :] * sdi,
            tl.dot(rft, gr, input_precision=prec).to(gx_ptr.dtype.element_ty),
            mask=m,
        )
        if same_dtype:
            grf = tl.dot(g, tl.trans(x), grf, input_precision="ieee")
        else:
            grf = tl.dot(
                g.to(tl.float32),
                tl.trans(x.to(tl.float32)),
                grf,
                input_precision="ieee",
            )
    tl.store(
        part_ptr + ((pid_k * groups + pid_g) * ps + ri[:, None]) * ps + rj[None, :], grf
    )


@triton.jit
def _block_diag(
    rf_ptr,
    pid_c,
    n_blk,
    size,
    per_tile: tl.constexpr,
    ps: tl.constexpr,
    block_ch: tl.constexpr,
):
    """Rf of per_tile consecutive blocks on the diagonal of a (block_ch, block_ch) tile."""
    j = tl.arange(0, block_ch)
    blk_local = j // size
    within = j % size
    blk = pid_c * per_tile + blk_local
    valid = (blk_local < per_tile) & (blk < n_blk)
    same = blk_local[:, None] == blk_local[None, :]
    off = blk[:, None] * ps * ps + within[:, None] * ps + within[None, :]
    rbd = tl.load(rf_ptr + off, mask=same & valid[:, None] & valid[None, :], other=0.0)
    return rbd, blk * size + within, valid


@triton.jit
def _fma_offsets(rr, bb, vv, size, inner, so, sc, si):
    """Offsets of the (rows, blocks, channel-in-block) tile of an (outer, C, inner) tensor."""
    o = rr // inner
    i = rr % inner
    ch = bb[:, None] * size + vv[None, :]
    return (o * so + i * si)[:, None, None] + ch[None, :, :] * sc


@triton.jit
def _fma_fwd_kernel(
    rf_ptr,
    x_ptr,
    out_ptr,
    n_blk,
    size,
    inner,
    n_rows,
    sxo,
    sxc,
    sxi,
    soo,
    soc,
    soi,
    ps: tl.constexpr,
    sp: tl.constexpr,
    block_t: tl.constexpr,
    block_b: tl.constexpr,
):
    """out[r, b, u] = sum_v Rf[b, u, v] x[r, b, v] for a (row tile, block tile), as vector FMAs.

    Grid: (row tiles, block tiles); sp is the block size padded to a power of two.
    """
    pid_r = tl.program_id(0)
    pid_b = tl.program_id(1)
    rr = pid_r * block_t + tl.arange(0, block_t)
    bb = pid_b * block_b + tl.arange(0, block_b)
    uu = tl.arange(0, sp)
    mb = bb < n_blk
    mu = uu < size
    m3 = (rr < n_rows)[:, None, None] & (mb[:, None] & mu[None, :])[None, :, :]

    # x tile (rows, blocks, v).
    x = tl.load(
        x_ptr + _fma_offsets(rr, bb, uu, size, inner, sxo, sxc, sxi), mask=m3, other=0.0
    ).to(tl.float32)

    # Per output channel u: out[r, b, u] = sum over v of x[r, b, v] * Rf[b, u, v], kept in x's layout.
    out = tl.zeros((block_t, block_b, sp), tl.float32)
    for u in tl.static_range(sp):
        ru = tl.load(
            rf_ptr + (bb * ps * ps)[:, None] + u * ps + uu[None, :],
            mask=mb[:, None] & mu[None, :],
            other=0.0,
        )
        out_u = tl.sum(x * ru[None, :, :], axis=2)
        out = tl.where(uu[None, None, :] == u, out_u[:, :, None], out)

    tl.store(
        out_ptr + _fma_offsets(rr, bb, uu, size, inner, soo, soc, soi),
        out.to(out_ptr.dtype.element_ty),
        mask=m3,
    )


@triton.jit
def _fma_bwd_kernel(
    rf_ptr,
    x_ptr,
    g_ptr,
    gx_ptr,
    part_ptr,
    n_blk,
    size,
    inner,
    n_rows,
    sxo,
    sxc,
    sxi,
    sgo,
    sgc,
    sgi,
    sdo,
    sdc,
    sdi,
    ps: tl.constexpr,
    sp: tl.constexpr,
    block_t: tl.constexpr,
    block_b: tl.constexpr,
):
    """gx[r, b, v] = sum_u g[r, b, u] Rf[b, u, v], and this split's sum over its rows of g[r, b, u] x[r, b, v].

    Grid: (row splits, block tiles); a split walks its share of the rows block_t at a time.
    """
    pid_s = tl.program_id(0)
    pid_b = tl.program_id(1)
    splits = tl.num_programs(0)
    bb = pid_b * block_b + tl.arange(0, block_b)
    uu = tl.arange(0, sp)
    mb = bb < n_blk
    mu = uu < size
    span = tl.cdiv(tl.cdiv(n_rows, block_t), splits) * block_t
    lo = pid_s * span
    hi = tl.minimum(lo + span, n_rows)
    rf = tl.load(
        rf_ptr
        + (bb * ps * ps)[:, None, None]
        + uu[None, :, None] * ps
        + uu[None, None, :],
        mask=mb[:, None, None] & mu[None, :, None] & mu[None, None, :],
        other=0.0,
    )

    # part[b, u, v] accumulates over the split's rows.
    part = tl.zeros((block_b, sp, sp), tl.float32)
    for r0 in range(lo, hi, block_t):
        rr = r0 + tl.arange(0, block_t)
        m3 = (rr < hi)[:, None, None] & (mb[:, None] & mu[None, :])[None, :, :]
        x = tl.load(
            x_ptr + _fma_offsets(rr, bb, uu, size, inner, sxo, sxc, sxi),
            mask=m3,
            other=0.0,
        ).to(tl.float32)
        g = tl.load(
            g_ptr + _fma_offsets(rr, bb, uu, size, inner, sgo, sgc, sgi),
            mask=m3,
            other=0.0,
        ).to(tl.float32)

        # gx[r, b, v] = sum over u of g[r, b, u] * Rf[b, u, v]; part[b, u, v] += sum over r of g * x.
        gx = tl.sum(g[:, :, :, None] * rf[None, :, :, :], axis=2)
        part += tl.sum(g[:, :, :, None] * x[:, :, None, :], axis=0)
        tl.store(
            gx_ptr + _fma_offsets(rr, bb, uu, size, inner, sdo, sdc, sdi),
            gx.to(gx_ptr.dtype.element_ty),
            mask=m3,
        )

    tl.store(
        part_ptr
        + ((pid_s * n_blk + bb) * ps * ps)[:, None, None]
        + uu[None, :, None] * ps
        + uu[None, None, :],
        part,
        mask=mb[:, None, None] & mu[None, :, None] & mu[None, None, :],
    )


@triton.jit
def _rows_fwd_kernel(
    rf_ptr,
    x_ptr,
    out_ptr,
    n_blk,
    size,
    inner,
    n_rows,
    sxo,
    sxc,
    sxi,
    soo,
    soc,
    soi,
    prec: tl.constexpr,
    dot16: tl.constexpr,
    ps: tl.constexpr,
    per_tile: tl.constexpr,
    block_ch: tl.constexpr,
    block_t: tl.constexpr,
    row_tiles: tl.constexpr,
):
    """out[r, ch] = sum Rf_bd[ch, ch'] x[r, ch'] for row_tiles row tiles, one channel tile.

    Grid: (row tile groups, channel tiles).
    """
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)

    # Block-diagonal Rf of this channel tile, transposed, held for every row tile.
    rbd, ch, valid = _block_diag(rf_ptr, pid_c, n_blk, size, per_tile, ps, block_ch)
    rbd_t = tl.trans(rbd)
    if dot16:
        rbd_t = rbd_t.to(x_ptr.dtype.element_ty)

    # Each row tile: out tile = x tile @ Rf_bd^T.
    for t in range(row_tiles):
        rr = (pid_r * row_tiles + t) * block_t + tl.arange(0, block_t)
        o = rr // inner
        i = rr % inner
        m = (rr < n_rows)[:, None] & valid[None, :]
        x = tl.load(
            x_ptr + o[:, None] * sxo + ch[None, :] * sxc + i[:, None] * sxi,
            mask=m,
            other=0.0,
        )
        if not dot16:
            x = x.to(tl.float32)
        out = tl.dot(x, rbd_t, input_precision=prec)
        tl.store(
            out_ptr + o[:, None] * soo + ch[None, :] * soc + i[:, None] * soi,
            out.to(out_ptr.dtype.element_ty),
            mask=m,
        )


@triton.jit
def _rows_bwd_kernel(
    rf_ptr,
    x_ptr,
    g_ptr,
    gx_ptr,
    part_ptr,
    n_blk,
    size,
    inner,
    n_rows,
    sxo,
    sxc,
    sxi,
    sgo,
    sgc,
    sgi,
    sdo,
    sdc,
    sdi,
    prec: tl.constexpr,
    same_dtype: tl.constexpr,
    dot16: tl.constexpr,
    ps: tl.constexpr,
    per_tile: tl.constexpr,
    block_ch: tl.constexpr,
    block_t: tl.constexpr,
):
    """gx = g Rf_bd over this program's rows, and its partial of g(Rf_bd) = g^T x.

    Grid: (row splits, channel tiles).
    """
    pid_s = tl.program_id(0)
    pid_c = tl.program_id(1)
    splits = tl.num_programs(0)
    n_ct = tl.num_programs(1)

    # Block-diagonal Rf of this channel tile, held for every row tile.
    rbd, ch, valid = _block_diag(rf_ptr, pid_c, n_blk, size, per_tile, ps, block_ch)
    if dot16:
        rbd = rbd.to(g_ptr.dtype.element_ty)

    # acc (block_ch, block_ch) = this split's sum of g^T x over rows [lo, hi).
    acc = tl.zeros((block_ch, block_ch), tl.float32)
    span = tl.cdiv(tl.cdiv(n_rows, block_t), splits) * block_t
    lo = pid_s * span
    hi = tl.minimum(lo + span, n_rows)
    for r0 in range(lo, hi, block_t):
        rr = r0 + tl.arange(0, block_t)
        o = rr // inner
        i = rr % inner
        m = (rr < hi)[:, None] & valid[None, :]
        g = tl.load(
            g_ptr + o[:, None] * sgo + ch[None, :] * sgc + i[:, None] * sgi,
            mask=m,
            other=0.0,
        )
        x = tl.load(
            x_ptr + o[:, None] * sxo + ch[None, :] * sxc + i[:, None] * sxi,
            mask=m,
            other=0.0,
        )
        gr = g if dot16 else g.to(tl.float32)
        tl.store(
            gx_ptr + o[:, None] * sdo + ch[None, :] * sdc + i[:, None] * sdi,
            tl.dot(gr, rbd, input_precision=prec).to(gx_ptr.dtype.element_ty),
            mask=m,
        )
        if same_dtype:
            acc = tl.dot(tl.trans(g), x, acc, input_precision="ieee")
        else:
            acc = tl.dot(
                tl.trans(g.to(tl.float32)),
                x.to(tl.float32),
                acc,
                input_precision="ieee",
            )
    a = tl.arange(0, block_ch)
    tl.store(
        part_ptr
        + ((pid_s * n_ct + pid_c) * block_ch + a[:, None]) * block_ch
        + a[None, :],
        acc,
    )


def _layout(x: torch.Tensor, channel_axis: int) -> tuple[torch.Tensor, int, int, int]:
    """x as (outer, C, inner) -> (view, outer, C, inner)."""
    axis = channel_axis % x.dim()
    outer = 1
    for size in x.shape[:axis]:
        outer *= size
    inner = 1
    for size in x.shape[axis + 1 :]:
        inner *= size
    c = x.shape[axis]
    return x.reshape(outer, c, inner), outer, c, inner


def _apply_precision(dev, dtype):
    """Apply-dot precision: tf32 for 16-bit outputs where available, else torch's fp32 matmul precision."""
    if dtype != torch.float32 and dev.tf32:
        return "tf32"
    return dot_precision(dev, torch.float32)


def _rotations(dev, blocks, rescale, constraint, multiplier, shift, need_rm):
    """(Rf (k, PS, PS) fp32, R, M, q0, c, q_norm); R and M only when need_rm."""
    k, s, _ = blocks.shape
    ps = pad16(s)
    if ps > MAX_CAYLEY:
        return _rotations_torch(blocks, rescale, constraint, multiplier, shift)
    q0, c, q_norm = (
        eager.skew(blocks, constraint)
        if (need_rm or constraint > 0)
        else (None, 1.0, None)
    )
    flags = (need_rm, torch.is_tensor(c), rescale is not None, shift)
    key = ("oft_rotation", flags, (k, ps))
    args = (blocks, rescale, c, multiplier, shift, need_rm)
    cfg = select(
        dev,
        key,
        lambda: _plan_rotation(dev, k, ps, need_rm),
        measurer(
            dev,
            k * ps * ps * 4,
            lambda: (),
            # compared on the s x s blocks; the padding is never read
            lambda cf, _: lambda: _rotation_run(cf, *args, constraint)[0][:, :s, :s],
        ),
    )
    rf, r, m = guarded(dev, key, cfg, lambda cf: _rotation_run(cf, *args, constraint))
    return rf, r, m, q0, c, q_norm


def _rotations_torch(blocks, rescale, constraint, multiplier, shift):
    """(Rf (k, PS, PS) fp32, R, M, q0, c, q_norm) from torch's batched inverse."""
    k, s, _ = blocks.shape
    ps = pad16(s)
    q0, c, q_norm, m, r = eager.cayley(blocks, constraint)
    eye = torch.eye(s, device=r.device, dtype=r.dtype)
    rf = (multiplier * r + (1.0 - multiplier) * eye).transpose(-1, -2)
    if rescale is not None:
        rf = rescale.float().reshape(k, s, 1) * rf
    if shift:
        rf = rf - eye
    padded = torch.zeros(k, ps, ps, device=r.device, dtype=torch.float32)
    padded[:, :s, :s] = rf
    return padded, r, m, q0, c, q_norm


def _plan_rotation(dev, k, ps, need_rm):
    """Warp counts of the per-block Gauss-Jordan, model-ranked, then torch's batched inverse."""
    # XPU Triton miscompiles the 64-wide variant below 8 warps.
    warps = (8,) if dev.kind == "xpu" and ps >= 64 else (2, 4, 8, 16)
    space = [{"num_warps": w, "num_stages": 1} for w in warps]

    def work(cf):
        # one program per block holds the augmented [I - q | I + q] and pivots ps times
        return Work(
            grid=k,
            warps=cf["num_warps"],
            acc_fp32=2 * ps * ps,
            operand_bytes=ps * ps * 4,
            smem_stage=0,
            stages=1,
            hbm_bytes=k * ps * ps * 4 * (4 if need_rm else 2),
            vector_flops=4.0 * k * ps**3,
        )

    return [*rank(dev, space, work), TORCH]


def _rotation_run(cfg, blocks, rescale, c, multiplier, shift, need_rm, constraint):
    """(Rf, R, M) from the in-kernel Gauss-Jordan at cfg's warps, or torch's batched inverse."""
    if cfg == TORCH:
        rf, r, m, *_ = _rotations_torch(blocks, rescale, constraint, multiplier, shift)
        return rf, r, m
    k, s, _ = blocks.shape
    ps = pad16(s)
    rf = torch.empty(k, ps, ps, device=blocks.device, dtype=torch.float32)
    r = m = None
    if need_rm:
        r = torch.empty(k, s, s, device=blocks.device, dtype=torch.float32)
        m = torch.empty_like(r)
    cval = c if torch.is_tensor(c) else rf
    res = rescale.reshape(-1) if rescale is not None else rf
    _rotation_kernel[(k,)](
        blocks,
        cval,
        res,
        rf,
        r if need_rm else rf,
        m if need_rm else rf,
        s,
        *blocks.stride(),
        multiplier,
        constrain=torch.is_tensor(c),
        rescale_on=rescale is not None,
        shift=shift,
        write_rm=need_rm,
        ps=ps,
        **launch(cfg),
    )
    return rf, r, m


def _dot16_axis(dev, dtype):
    """DOT16 values: Rf applied in the activations' 16-bit dtype, offered on matrix hardware only."""
    return (False, True) if dtype.itemsize == 2 and dev.mma else (False,)


# DRAM sector size, bytes.
_SECTOR = 32
# Largest fp32 (rows, blocks, s, s) product an fma program holds.
_FMA_TILE = 8192
# Largest padded block size of the fma layout (Triton rewrites larger 4-D products into a dot it cannot lower).
_FMA_MAX_BLOCK = 8


def _oft_work(dev, cfg, k, s, outer, inner, e, backward):
    """Work of one apply config; the backward reads x and g and writes gx."""
    ps = pad16(s)
    c = k * s
    rows = outer * inner
    passes = 3 if backward else 2
    if cfg["LAYOUT"] == "fma":
        bt, bb, sp = cfg["BT"], cfg["BB"], triton.next_power_of_2(s)
        n_bt = cdiv(k, bb)
        # contiguous bytes per access: a row's channel span (channels-last) or a column's rows
        run = bb * s * e if inner == 1 else min(bt, inner) * e
        strided = max(1.0, _SECTOR / run)
        if backward:
            splits = max(1, min(cdiv(rows, bt), cdiv(cfg["OCC"] * dev.units, n_bt)))
            grid, partial = splits * n_bt, splits * k * ps * ps * 4 * 2
        else:
            grid, partial = cdiv(rows, bt) * n_bt, 0
        return Work(
            grid=grid,
            warps=cfg["num_warps"],
            acc_fp32=bt * bb * sp * sp + (bb * sp * sp if backward else 0),
            operand_bytes=bb * sp * sp * 4 + passes * bt * bb * sp * e,
            smem_stage=0,
            stages=1,
            hbm_bytes=passes * rows * c * e * strided + k * ps * ps * 4 + partial,
            vector_flops=2.0 * rows * c * s * (2 if backward else 1),
        )
    # tf32 counted at half the 16-bit dot rate.
    dot = 2.0 * (1 if cfg["DOT16"] or e == 4 else 2) * (2 if backward else 1)
    if cfg["LAYOUT"] == "cols":
        bn = cfg["BLOCK_N"]
        groups = max(1, min(cdiv(rows, bn), cdiv(cfg["OCC"] * dev.units, k)))
        strided = _SECTOR / e if inner == 1 else 1.0
        return Work(
            grid=k * groups,
            warps=cfg["num_warps"],
            acc_fp32=ps * bn + (ps * ps if backward else 0),
            operand_bytes=ps * ps * 4 + passes * ps * bn * e,
            smem_stage=0,
            stages=1,
            hbm_bytes=passes * rows * c * e * strided + k * groups * ps * ps * 4,
            mma_flops=dot * rows * c * ps,
        )
    bch, bt = cfg["BCH"], cfg["BT"]
    g = bch // s
    n_ct = cdiv(k, g)
    if backward:
        splits = max(1, min(cdiv(rows, bt), cdiv(cfg["OCC"] * dev.units, n_ct)))
        grid, partial = splits * n_ct, splits * n_ct * bch * bch * 4 * 2
    else:
        grid, partial = cdiv(rows, bt * cfg["RT"]) * n_ct, 0
    strided = _SECTOR / e if inner > 1 else 1.0
    return Work(
        grid=grid,
        warps=cfg["num_warps"],
        acc_fp32=bt * bch + (bch * bch if backward else 0),
        operand_bytes=bch * bch * 4 + passes * bt * bch * e,
        smem_stage=0,
        stages=1,
        hbm_bytes=passes * rows * c * e * strided + grid * bch * bch * 4 + partial,
        mma_flops=dot * rows * c * bch,
    )


def _plan(dev, k, s, outer, inner, dtype, backward):
    """Apply configs of both layouts, model-ranked, then torch."""
    ps = pad16(s)
    rows = []
    for bch in (16, 32, 64, 128):
        if bch < ps or bch // s == 0:
            continue
        if backward:
            rows += grid_configs(
                dev,
                LAYOUT=("rows",),
                BCH=(bch,),
                BT=(32, 64),
                OCC=(2, 4, 8),
                DOT16=_dot16_axis(dev, dtype),
                num_warps=(4, 8),
                num_stages=(1,),
            )
        else:
            rows += grid_configs(
                dev,
                LAYOUT=("rows",),
                BCH=(bch,),
                BT=(32, 64, 128),
                RT=(1, 4, 16),
                DOT16=_dot16_axis(dev, dtype),
                num_warps=(4, 8),
                num_stages=(1,),
            )
    cols = grid_configs(
        dev,
        LAYOUT=("cols",),
        BLOCK_N=(32, 64, 128),
        OCC=(2, 4, 8),
        DOT16=_dot16_axis(dev, dtype),
        num_warps=(4, 8),
        num_stages=(1,),
    )
    # fma: a (rows, blocks, s, s) fp32 product of at most _FMA_TILE elements per program.
    sp = triton.next_power_of_2(s)
    fma = [
        cfg
        for cfg in grid_configs(
            dev,
            LAYOUT=("fma",),
            BT=(16, 32, 64),
            BB=(1, 2, 4, 8, 16, 32, 64),
            **({"OCC": (2, 4, 8)} if backward else {}),
            DOT16=(False,),
            num_warps=(4, 8),
            num_stages=(1,),
        )
        if sp <= _FMA_MAX_BLOCK
        and cfg["BT"] * cfg["BB"] * sp * sp <= _FMA_TILE
        and cfg["BB"] <= triton.next_power_of_2(k)
    ]
    e = dtype.itemsize
    return [
        *rank(
            dev,
            rows + cols + fma,
            lambda cf: _oft_work(dev, cf, k, s, outer, inner, e, backward),
        ),
        TORCH,
    ]


def _groups(dev, k, cols, cfg):
    return max(
        1,
        min(triton.cdiv(cols, cfg["BLOCK_N"]), triton.cdiv(cfg["OCC"] * dev.units, k)),
    )


def _blocks_view(t: torch.Tensor, k: int, channel_axis: int) -> torch.Tensor:
    """t with its channel axis last, split into (..., k blocks, s) in fp32."""
    tm = t.movedim(channel_axis, -1).float()
    return tm.reshape(*tm.shape[:-1], k, tm.shape[-1] // k)


def torch_apply(rf: torch.Tensor, x: torch.Tensor, channel_axis: int) -> torch.Tensor:
    """out[..., b, u] = sum_v Rf[b, u, v] x[..., b, v] along channel_axis, in torch (fp32, then x's dtype)."""
    k = rf.shape[0]
    xb = _blocks_view(x, k, channel_axis)
    s = xb.shape[-1]
    out = torch.einsum("...bv,buv->...bu", xb, rf[:, :s, :s])
    return out.reshape(*xb.shape[:-2], k * s).movedim(-1, channel_axis).to(x.dtype)


def _torch_apply_bwd(rf, x, g, channel_axis):
    """(gx, g(Rf) (k, s, s)) of torch_apply in torch."""
    k = rf.shape[0]
    xb = _blocks_view(x, k, channel_axis)
    gb = _blocks_view(g, k, channel_axis)
    s = xb.shape[-1]
    gx = torch.einsum("...bu,buv->...bv", gb, rf[:, :s, :s])
    grf = torch.einsum("...bu,...bv->buv", gb, xb)
    gx = gx.reshape(*xb.shape[:-2], k * s).movedim(-1, channel_axis).to(x.dtype)
    return gx, grf


def _fwd_run(
    cfg, dev, blocks, x, rescale, constraint, multiplier, shift, channel_axis, rf
):
    """The apply under cfg with the folded rotation rf built once by the caller."""
    if cfg == TORCH:
        return torch_apply(rf, x, channel_axis)
    k, s, _ = blocks.shape
    xv, outer, c, inner = _layout(x, channel_axis)
    out = torch.empty((outer, c, inner), device=x.device, dtype=x.dtype)
    prec = _apply_precision(dev, x.dtype)
    if cfg["LAYOUT"] == "fma":
        rows = outer * inner
        grid = (triton.cdiv(rows, cfg["BT"]), triton.cdiv(k, cfg["BB"]))
        _fma_fwd_kernel[grid](
            rf,
            xv,
            out,
            k,
            s,
            inner,
            rows,
            *xv.stride(),
            *out.stride(),
            ps=pad16(s),
            sp=triton.next_power_of_2(s),
            block_t=cfg["BT"],
            block_b=cfg["BB"],
            **launch(cfg),
        )
    elif cfg["LAYOUT"] == "cols":
        cols = outer * inner
        _cols_fwd_kernel[(k, _groups(dev, k, cols, cfg))](
            rf,
            xv,
            out,
            s,
            inner,
            cols,
            *xv.stride(),
            *out.stride(),
            prec=prec,
            dot16=cfg["DOT16"],
            ps=pad16(s),
            block_n=cfg["BLOCK_N"],
            **launch(cfg),
        )
    else:
        g = cfg["BCH"] // s
        rows = outer * inner
        grid = (triton.cdiv(rows, cfg["BT"] * cfg["RT"]), triton.cdiv(k, g))
        _rows_fwd_kernel[grid](
            rf,
            xv,
            out,
            k,
            s,
            inner,
            rows,
            *xv.stride(),
            *out.stride(),
            prec=prec,
            dot16=cfg["DOT16"],
            ps=pad16(s),
            per_tile=g,
            block_ch=cfg["BCH"],
            block_t=cfg["BT"],
            row_tiles=cfg["RT"],
            **launch(cfg),
        )
    return out.reshape(x.shape)


def _key(name, x, blocks, rescale, shift, channel_axis):
    _, outer, _c, inner = _layout(x, channel_axis)
    k, s, _ = blocks.shape
    cat = (dtype_key(x.dtype), rescale is not None, shift, inner == 1)
    return name, cat, (token_bucket(outer * inner), k, s)


def oft_apply(
    blocks: torch.Tensor,
    x: torch.Tensor,
    rescale: torch.Tensor | None = None,
    constraint: float = 0.0,
    multiplier: float = 1.0,
    shift: bool = False,
    channel_axis: int = -1,
) -> torch.Tensor:
    """rescale * R'^T x [- x] along channel_axis."""
    dev = device_info(x.device)
    rf, *_ = _rotations(dev, blocks, rescale, constraint, multiplier, shift, False)
    return apply_rotation(
        dev, blocks, x, rescale, constraint, multiplier, shift, channel_axis, rf
    )


def apply_rotation(
    dev, blocks, x, rescale, constraint, multiplier, shift, channel_axis, rf
):
    """oft_apply with the folded rotation rf (k, PS, PS) already built."""
    k, s, _ = blocks.shape
    _, outer, _c, inner = _layout(x, channel_axis)
    args = (rescale, constraint, multiplier, shift, channel_axis, rf)
    key = _key("oft_apply", x, blocks, rescale, shift, channel_axis)
    cfg = select(
        dev,
        key,
        lambda: _plan(dev, k, s, outer, inner, x.dtype, False),
        measurer(
            dev,
            2 * x.numel() * x.element_size(),
            lambda: (copy_of(x),),
            lambda c, st: lambda: _fwd_run(c, dev, blocks, st[0], *args),
        ),
    )
    return guarded(dev, key, cfg, lambda c: _fwd_run(c, dev, blocks, x, *args))


def _diag_blocks(part: torch.Tensor, k: int, s: int, g: int) -> torch.Tensor:
    """The (k, s, s) diagonal blocks of per-channel-tile (n_ct, BCH, BCH) sums."""
    blk = torch.arange(k, device=part.device)
    tile = blk // g
    rows = (blk % g)[:, None] * s + torch.arange(s, device=part.device)
    return part[tile[:, None, None], rows[:, :, None], rows[:, None, :]]


def _bwd_kernel_run(cfg, dev, rf, x, g, channel_axis):
    """(gx, g(Rf) (k, s, s)) for the folded rotation."""
    k, ps, _ = rf.shape
    xv, outer, c, inner = _layout(x, channel_axis)
    s = c // k
    gv = g.reshape(outer, c, inner)
    gx = torch.empty((outer, c, inner), device=x.device, dtype=x.dtype)
    prec = _apply_precision(dev, x.dtype)
    if cfg["LAYOUT"] == "fma":
        rows = outer * inner
        n_bt = triton.cdiv(k, cfg["BB"])
        splits = max(
            1,
            min(
                triton.cdiv(rows, cfg["BT"]), triton.cdiv(cfg["OCC"] * dev.units, n_bt)
            ),
        )
        part = torch.zeros(splits, k, ps, ps, device=x.device, dtype=torch.float32)
        _fma_bwd_kernel[(splits, n_bt)](
            rf,
            xv,
            gv,
            gx,
            part,
            k,
            s,
            inner,
            rows,
            *xv.stride(),
            *gv.stride(),
            *gx.stride(),
            ps=ps,
            sp=triton.next_power_of_2(s),
            block_t=cfg["BT"],
            block_b=cfg["BB"],
            **launch(cfg),
        )
        return gx.reshape(x.shape), part.sum(0)[:, :s, :s]
    if cfg["LAYOUT"] == "cols":
        cols = outer * inner
        groups = _groups(dev, k, cols, cfg)
        part = torch.empty(k, groups, ps, ps, device=x.device, dtype=torch.float32)
        _cols_bwd_kernel[(k, groups)](
            rf,
            xv,
            gv,
            gx,
            part,
            s,
            inner,
            cols,
            *xv.stride(),
            *gv.stride(),
            *gx.stride(),
            prec=prec,
            same_dtype=gv.dtype == xv.dtype,
            dot16=cfg["DOT16"],
            ps=ps,
            block_n=cfg["BLOCK_N"],
            **launch(cfg),
        )
        return gx.reshape(x.shape), part.sum(1)[:, :s, :s]
    gper = cfg["BCH"] // s
    n_ct = triton.cdiv(k, gper)
    rows = outer * inner
    splits = max(
        1, min(triton.cdiv(rows, cfg["BT"]), triton.cdiv(cfg["OCC"] * dev.units, n_ct))
    )
    part = torch.empty(
        splits, n_ct, cfg["BCH"], cfg["BCH"], device=x.device, dtype=torch.float32
    )
    _rows_bwd_kernel[(splits, n_ct)](
        rf,
        xv,
        gv,
        gx,
        part,
        k,
        s,
        inner,
        rows,
        *xv.stride(),
        *gv.stride(),
        *gx.stride(),
        prec=prec,
        same_dtype=gv.dtype == xv.dtype,
        dot16=cfg["DOT16"],
        ps=ps,
        per_tile=gper,
        block_ch=cfg["BCH"],
        block_t=cfg["BT"],
        **launch(cfg),
    )
    return gx.reshape(x.shape), _diag_blocks(part.sum(0), k, s, gper)


def oft_apply_backward(
    grad: torch.Tensor,
    blocks: torch.Tensor,
    x: torch.Tensor,
    rescale: torch.Tensor | None = None,
    constraint: float = 0.0,
    multiplier: float = 1.0,
    shift: bool = False,
    channel_axis: int = -1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(grad_x, grad_blocks, grad_rescale); grad_rescale is empty without rescale."""
    dev = device_info(x.device)
    k, s, _ = blocks.shape
    _, outer, _c, inner = _layout(x, channel_axis)
    g = grad.reshape(x.shape)
    rf, r, m, q0, c, q_norm = _rotations(
        dev, blocks, rescale, constraint, multiplier, shift, True
    )

    def run(cfg, x_, g_):
        if cfg == TORCH:
            gx, grf = _torch_apply_bwd(rf, x_, g_, channel_axis)
        else:
            gx, grf = _bwd_kernel_run(cfg, dev, rf, x_, g_, channel_axis)
        eye = torch.eye(s, device=r.device, dtype=r.dtype)
        rp_t = (multiplier * r + (1.0 - multiplier) * eye).transpose(-1, -2)
        if rescale is not None:
            gres = (grf * rp_t).sum(-1).reshape(rescale.shape).to(rescale.dtype)
            grf = rescale.float().reshape(k, s, 1) * grf
        else:
            gres = absent(x)
        gb = eager.chain_to_blocks(
            grf.transpose(-1, -2), r, m, q0, c, q_norm, constraint, multiplier
        )
        return gx, gb.to(blocks.dtype), gres

    key = _key("oft_apply_backward", x, blocks, rescale, shift, channel_axis)
    cfg = select(
        dev,
        key,
        lambda: _plan(dev, k, s, outer, inner, x.dtype, True),
        measurer(
            dev,
            3 * x.numel() * x.element_size(),
            lambda: (copy_of(x), copy_of(g)),
            lambda c, st: lambda: run(c, st[0], st[1]),
        ),
    )
    return guarded(dev, key, cfg, lambda c: run(c, x, g))
