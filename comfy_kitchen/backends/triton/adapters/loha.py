# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LoHa kernels: DeltaW = (w1_up@w1_down) * (w2_up@w2_down), bypass and merge."""

import torch
import triton
import triton.language as tl

from comfy_kitchen.backends.eager.adapters import loha as eager

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
from .tuning.model import Work, cdiv, rank, shared_reads, tiled_reads

# Largest padded rank the regen bypass and the merge backward hold in registers.
MAX_RANK = 128
# Bypass forms: DeltaW then one addmm with the base, or DeltaW then a matmul and an add.
MATERIALIZE = {"impl": "materialize"}
MATERIALIZE_ADD = {"impl": "materialize_add"}


@triton.jit
def _merge_fwd_kernel(
    a1_ptr,
    b1_ptr,
    a2_ptr,
    b2_ptr,
    w_ptr,
    out_ptr,
    n_out,
    n_in,
    rank,
    sa1o,
    sa1r,
    sb1r,
    sb1i,
    sa2o,
    sa2r,
    sb2r,
    sb2i,
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
    """out = [w +] scale * (a1@b1) * (a2@b2), one program per output tile."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * block_m + tl.arange(0, block_m)
    rn = pid_n * block_n + tl.arange(0, block_n)
    mm = rm < n_out
    mn = rn < n_in

    # acc1 = w1_up @ w1_down tile, acc2 = w2_up @ w2_down tile, reduced over rank.
    acc1 = tl.zeros((block_m, block_n), tl.float32)
    acc2 = tl.zeros((block_m, block_n), tl.float32)
    for r0 in range(0, rank, block_r):
        rk = r0 + tl.arange(0, block_r)
        mk = rk < rank
        a1 = load2d(a1_ptr, rm, rk, sa1o, sa1r, mm, mk)
        b1 = load2d(b1_ptr, rk, rn, sb1r, sb1i, mk, mn)
        a2 = load2d(a2_ptr, rm, rk, sa2o, sa2r, mm, mk)
        b2 = load2d(b2_ptr, rk, rn, sb2r, sb2i, mk, mn)
        acc1 = tl.dot(a1, b1, acc1, input_precision=prec)
        acc2 = tl.dot(a2, b2, acc2, input_precision=prec)

    # Epilogue: hadamard, scale, add the weight tile, store.
    out = acc1 * acc2 * scale
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
    a1_ptr,
    b1_ptr,
    a2_ptr,
    b2_ptr,
    ga1_ptr,
    gb1_ptr,
    ga2_ptr,
    gb2_ptr,
    n_out,
    n_in,
    rank,
    sgo,
    sgi,
    sa1o,
    sa1r,
    sb1r,
    sb1i,
    sa2o,
    sa2r,
    sb2r,
    sb2i,
    scale,
    n_rows,
    n_cols,
    prec: tl.constexpr,
    block_r: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    """fp32 partials of the four gradients: row-side (first n_rows*S programs), then column-side."""
    pid = tl.program_id(0)
    splits = tl.num_programs(0) // (n_rows + n_cols)
    rr = tl.arange(0, block_r)
    mr = rr < rank
    ct = a1_ptr.dtype.element_ty

    # Row role: ga1, ga2 (block_m, rank) for one row tile over split `part` of the columns.
    if pid < n_rows * splits:
        part = pid // n_rows
        rm = (pid % n_rows) * block_m + tl.arange(0, block_m)
        mm = rm < n_out
        amask = mm[:, None] & mr[None, :]
        a1 = load2d(a1_ptr, rm, rr, sa1o, sa1r, mm, mr)
        a2 = load2d(a2_ptr, rm, rr, sa2o, sa2r, mm, mr)
        ga1 = tl.zeros((block_m, block_r), tl.float32)
        ga2 = tl.zeros((block_m, block_r), tl.float32)
        per = tl.cdiv(tl.cdiv(n_in, block_n), splits) * block_n
        lo = part * per
        hi = tl.minimum(lo + per, n_in)
        for i0 in range(lo, hi, block_n):
            rn = i0 + tl.arange(0, block_n)
            mn = rn < hi
            b1 = load2d(b1_ptr, rr, rn, sb1r, sb1i, mr, mn)
            b2 = load2d(b2_ptr, rr, rn, sb2r, sb2i, mr, mn)
            g = load2d(g_ptr, rm, rn, sgo, sgi, mm, mn).to(tl.float32) * scale

            # e1 = g * (w2_up @ w2_down), e2 = g * (w1_up @ w1_down) on this tile.
            e1 = (g * tl.dot(a2, b2, input_precision=prec)).to(ct)
            e2 = (g * tl.dot(a1, b1, input_precision=prec)).to(ct)

            # ga1 += e1 @ b1^T, ga2 += e2 @ b2^T.
            ga1 = tl.dot(e1, tl.trans(b1), ga1, input_precision=prec)
            ga2 = tl.dot(e2, tl.trans(b2), ga2, input_precision=prec)

        off = (part * n_out + rm)[:, None] * rank + rr[None, :]
        tl.store(ga1_ptr + off, ga1, mask=amask)
        tl.store(ga2_ptr + off, ga2, mask=amask)
    # Column role: gb1, gb2 (rank, block_n) for one column tile over split `part_c` of the rows.
    else:
        q = pid - n_rows * splits
        part_c = q // n_cols
        rn = (q % n_cols) * block_n + tl.arange(0, block_n)
        mn = rn < n_in
        bmask = mr[:, None] & mn[None, :]
        b1 = load2d(b1_ptr, rr, rn, sb1r, sb1i, mr, mn)
        b2 = load2d(b2_ptr, rr, rn, sb2r, sb2i, mr, mn)
        gb1 = tl.zeros((block_r, block_n), tl.float32)
        gb2 = tl.zeros((block_r, block_n), tl.float32)
        per_c = tl.cdiv(tl.cdiv(n_out, block_m), splits) * block_m
        lo_c = part_c * per_c
        hi_c = tl.minimum(lo_c + per_c, n_out)
        for o0 in range(lo_c, hi_c, block_m):
            rm = o0 + tl.arange(0, block_m)
            mm = rm < hi_c
            a1 = load2d(a1_ptr, rm, rr, sa1o, sa1r, mm, mr)
            a2 = load2d(a2_ptr, rm, rr, sa2o, sa2r, mm, mr)
            g = load2d(g_ptr, rm, rn, sgo, sgi, mm, mn).to(tl.float32) * scale

            # gb1 += a1^T @ (g * P2), gb2 += a2^T @ (g * P1).
            e1 = (g * tl.dot(a2, b2, input_precision=prec)).to(ct)
            e2 = (g * tl.dot(a1, b1, input_precision=prec)).to(ct)
            gb1 = tl.dot(tl.trans(a1), e1, gb1, input_precision=prec)
            gb2 = tl.dot(tl.trans(a2), e2, gb2, input_precision=prec)

        off_c = (part_c * rank + rr)[:, None] * n_in + rn[None, :]
        tl.store(gb1_ptr + off_c, gb1, mask=bmask)
        tl.store(gb2_ptr + off_c, gb2, mask=bmask)


@triton.jit
def _merge_bwd_tile_kernel(
    g_ptr,
    a1_ptr,
    b1_ptr,
    a2_ptr,
    b2_ptr,
    ga1_ptr,
    gb1_ptr,
    ga2_ptr,
    gb2_ptr,
    n_out,
    n_in,
    rank,
    sgo,
    sgi,
    sa1o,
    sa1r,
    sb1r,
    sb1i,
    sa2o,
    sa2r,
    sb2r,
    sb2i,
    scale,
    prec: tl.constexpr,
    block_r: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    """fp32 partials of the four gradients from one read of g: ga per split, gb per row tile.

    Grid: (output row tiles, splits of the input axis).
    """
    pid_m = tl.program_id(0)
    pid_s = tl.program_id(1)
    splits = tl.num_programs(1)
    rm = pid_m * block_m + tl.arange(0, block_m)
    rr = tl.arange(0, block_r)
    mm = rm < n_out
    mr = rr < rank
    ct = a1_ptr.dtype.element_ty

    # Up-factor rows of this row tile; ga1, ga2 accumulate this split's partials.
    a1 = load2d(a1_ptr, rm, rr, sa1o, sa1r, mm, mr)
    a2 = load2d(a2_ptr, rm, rr, sa2o, sa2r, mm, mr)
    ga1 = tl.zeros((block_m, block_r), tl.float32)
    ga2 = tl.zeros((block_m, block_r), tl.float32)

    # Input columns [lo, hi) of this split, one g tile at a time.
    per = tl.cdiv(tl.cdiv(n_in, block_n), splits) * block_n
    lo = pid_s * per
    hi = tl.minimum(lo + per, n_in)
    for i0 in range(lo, hi, block_n):
        rn = i0 + tl.arange(0, block_n)
        mn = rn < hi
        bmask = mr[:, None] & mn[None, :]
        b1 = load2d(b1_ptr, rr, rn, sb1r, sb1i, mr, mn)
        b2 = load2d(b2_ptr, rr, rn, sb2r, sb2i, mr, mn)
        g = load2d(g_ptr, rm, rn, sgo, sgi, mm, mn).to(tl.float32) * scale

        # e1 = g * P2, e2 = g * P1 for this tile, built once for all four gradients.
        e1 = (g * tl.dot(a2, b2, input_precision=prec)).to(ct)
        e2 = (g * tl.dot(a1, b1, input_precision=prec)).to(ct)

        # ga += e @ b^T; gb tile = a^T @ e, stored per row tile.
        ga1 = tl.dot(e1, tl.trans(b1), ga1, input_precision=prec)
        ga2 = tl.dot(e2, tl.trans(b2), ga2, input_precision=prec)
        off_b = (pid_m * rank + rr)[:, None] * n_in + rn[None, :]
        tl.store(
            gb1_ptr + off_b, tl.dot(tl.trans(a1), e1, input_precision=prec), mask=bmask
        )
        tl.store(
            gb2_ptr + off_b, tl.dot(tl.trans(a2), e2, input_precision=prec), mask=bmask
        )
    off_a = (pid_s * n_out + rm)[:, None] * rank + rr[None, :]
    amask = mm[:, None] & mr[None, :]
    tl.store(ga1_ptr + off_a, ga1, mask=amask)
    tl.store(ga2_ptr + off_a, ga2, mask=amask)


@triton.jit
def _bypass_regen_kernel(
    x_ptr,
    a1_ptr,
    b1_ptr,
    a2_ptr,
    b2_ptr,
    b_ptr,
    y_ptr,
    n_tok,
    n_out,
    n_in,
    rank,
    sxt,
    sxi,
    sa1o,
    sa1r,
    sb1r,
    sb1i,
    sa2o,
    sa2r,
    sb2r,
    sb2i,
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
):
    """y = [base +] scale * x @ DeltaW^T, the DeltaW tile generated per K step.

    Grid: (token tiles, output column tiles).
    """
    pid_t = tl.program_id(0)
    pid_n = tl.program_id(1)
    rt = pid_t * block_m + tl.arange(0, block_m)
    rn = pid_n * block_n + tl.arange(0, block_n)
    rr = tl.arange(0, block_r)
    mt = rt < n_tok
    mn = rn < n_out
    mr = rr < rank

    # Up-factor rows of this output tile, held for the whole K loop.
    a1 = load2d(a1_ptr, rn, rr, sa1o, sa1r, mn, mr)
    a2 = load2d(a2_ptr, rn, rr, sa2o, sa2r, mn, mr)

    # acc (tokens, outputs) = x @ DeltaW^T over the input axis.
    acc = tl.zeros((block_m, block_n), tl.float32)
    for k0 in range(0, n_in, block_k):
        rk = k0 + tl.arange(0, block_k)
        mk = rk < n_in
        x = load2d(x_ptr, rt, rk, sxt, sxi, mt, mk)
        b1 = load2d(b1_ptr, rr, rk, sb1r, sb1i, mr, mk)
        b2 = load2d(b2_ptr, rr, rk, sb2r, sb2i, mr, mk)

        # DeltaW tile (outputs, k) = (a1 @ b1) * (a2 @ b2), then acc += x @ tile^T.
        w = tl.dot(a1, b1, input_precision=prec) * tl.dot(a2, b2, input_precision=prec)
        acc = tl.dot(x, tl.trans(w.to(x.dtype)), acc, input_precision=prec)

    # Epilogue: scale, add base, store.
    y = acc * scale
    if has_base:
        y += load2d(b_ptr, rt, rn, sbt, sbo, mt, mn).to(tl.float32)
    tl.store(
        y_ptr + rt[:, None] * syt + rn[None, :] * syo,
        y.to(y_ptr.dtype.element_ty),
        mask=mt[:, None] & mn[None, :],
    )


def _plan_merge(dev, o, i, r, e_w, e_f):
    rp = pad16(r)
    brs = tuple(sorted({min(rp, 16), min(rp, 32), min(rp, 64)}))
    space = grid_configs(
        dev,
        BLOCK_M=(32, 64, 128),
        BLOCK_N=(32, 64, 128),
        BLOCK_R=brs,
        num_warps=(2, 4, 8),
    )

    def work(c):
        bm, bn, br = c["BLOCK_M"], c["BLOCK_N"], c["BLOCK_R"]
        tm, tn = cdiv(o, bm), cdiv(i, bn)
        return Work(
            grid=tm * tn,
            warps=c["num_warps"],
            acc_fp32=2 * bm * bn,
            operand_bytes=0,
            smem_stage=2 * (bm + bn) * br * e_f,
            stages=c["num_stages"],
            hbm_bytes=2 * o * i * e_w
            + shared_reads(dev, 2 * (o + i) * rp * e_f, max(tm, tn)),
            mma_flops=4.0 * tm * bm * tn * bn * rp,
            vector_flops=3.0 * o * i,
            l2_bytes=2 * o * i * e_w + 2 * tiled_reads(o, i, rp, bm, bn, e_f),
        )

    return [*rank(dev, space, work), TORCH]


def _plan_merge_bwd(dev, o, i, r, e_g, e_f):
    rp = pad16(r)
    if rp > MAX_RANK:
        return [TORCH]
    # XPU Triton miscompiles "roles" below 8 warps when the rank tile is partly masked.
    roles_warps = (8,) if dev.kind == "xpu" and rp != r else (2, 4, 8)
    roles = grid_configs(
        dev,
        LAYOUT=("roles",),
        BLOCK_M=(32, 64),
        BLOCK_N=(32, 64, 128),
        SPLIT=(1, 2, 4, 8, 16),
        num_warps=roles_warps,
    )
    tile = grid_configs(
        dev,
        LAYOUT=("tile",),
        BLOCK_M=(32, 64),
        BLOCK_N=(32, 64, 128),
        SPLIT=(1, 2, 4, 8, 16),
        num_warps=(2, 4, 8),
    )

    def work(c):
        bm, bn, split = c["BLOCK_M"], c["BLOCK_N"], c["SPLIT"]
        weights = shared_reads(dev, 2 * (o + i) * rp * e_f, cdiv(o, bm) * split)
        tile_smem = (2 * bm * rp + 2 * rp * bn + 2 * bm * bn) * e_f
        if c["LAYOUT"] == "roles":
            return Work(
                grid=(cdiv(o, bm) + cdiv(i, bn)) * split,
                warps=c["num_warps"],
                acc_fp32=2 * (bm * rp + bm * bn),
                operand_bytes=0,
                smem_stage=tile_smem,
                stages=c["num_stages"],
                hbm_bytes=2 * o * i * e_g + 2 * 2 * split * (o + i) * rp * 4 + weights,
                mma_flops=16.0 * o * i * rp,
                vector_flops=6.0 * o * i,
                # row roles read both ups' rows and both downs' split; column roles the transpose
                l2_bytes=2 * o * i * e_g
                + 2
                * (o * split + cdiv(o, bm) * i + i * split + cdiv(i, bn) * o)
                * rp
                * e_f
                + 2 * 2 * split * (o + i) * rp * 4,
            )
        tm = cdiv(o, bm)
        return Work(
            grid=tm * split,
            warps=c["num_warps"],
            acc_fp32=2 * (bm * rp + bm * bn + rp * bn),
            operand_bytes=0,
            smem_stage=tile_smem,
            stages=c["num_stages"],
            hbm_bytes=o * i * e_g + 2 * 2 * (split * o + tm * i) * rp * 4 + weights,
            mma_flops=12.0 * o * i * rp,
            vector_flops=3.0 * o * i,
            # each (row tile, split) reads both ups' rows and both downs' split
            l2_bytes=o * i * e_g
            + 2 * (tm * split * bm + tm * i) * rp * e_f
            + 2 * 2 * (split * o + tm * i) * rp * 4,
        )

    return [*rank(dev, roles + tile, work), TORCH]


def _plan_bypass(dev, t, o, i, r, e, has_base):
    """Regen configs model-ranked, then the materialize forms and torch."""
    whole = [MATERIALIZE, MATERIALIZE_ADD, TORCH]
    rp = pad16(r)
    if rp > MAX_RANK:
        return whole
    space = grid_configs(
        dev,
        BLOCK_M=(32, 64, 128),
        BLOCK_N=(32, 64, 128),
        BLOCK_K=(32, 64),
        num_warps=(2, 4, 8),
    )

    def work(c):
        bm, bn, bk = c["BLOCK_M"], c["BLOCK_N"], c["BLOCK_K"]
        tt, tn = cdiv(t, bm), cdiv(o, bn)
        return Work(
            grid=tt * tn,
            warps=c["num_warps"],
            acc_fp32=bm * bn + 2 * bn * bk,
            operand_bytes=0,
            smem_stage=(bm * bk + 2 * (bn + bk) * rp) * e,
            stages=c["num_stages"],
            hbm_bytes=t * i * e * tn
            + t * o * e * (2 if has_base else 1)
            + shared_reads(dev, 2 * (o + i) * rp * e, tt * tn),
            mma_flops=2.0 * tt * bm * tn * bn * i + 4.0 * tt * tn * bn * i * rp,
            vector_flops=tt * tn * bn * i,
            # each tile reads its x rows, both ups' rows and both downs whole
            l2_bytes=tt * tn * (bm * i + 2 * bn * rp + 2 * rp * i) * e
            + t * o * e * (2 if has_base else 1),
        )

    return [*rank(dev, space, work), *whole]


def _merge_run(cfg, w2, factors, scale, add_base=True, out_dtype=None):
    a1, b1, a2, b2 = factors
    o, r = a1.shape
    i = b1.shape[1]
    out = torch.empty(o, i, device=a1.device, dtype=out_dtype or w2.dtype)
    if cfg == TORCH:
        delta = ((a1 @ b1) * (a2 @ b2)).mul_(scale)
        return out.copy_(w2 + delta if add_base else delta)
    w = w2 if add_base else out
    grid = (triton.cdiv(o, cfg["BLOCK_M"]), triton.cdiv(i, cfg["BLOCK_N"]))
    _merge_fwd_kernel[grid](
        a1,
        b1,
        a2,
        b2,
        w,
        out,
        o,
        i,
        r,
        *a1.stride(),
        *b1.stride(),
        *a2.stride(),
        *b2.stride(),
        *w.stride(),
        *out.stride(),
        scale,
        add_base=add_base,
        prec=dot_precision(device_info(a1.device), a1.dtype),
        block_m=cfg["BLOCK_M"],
        block_n=cfg["BLOCK_N"],
        block_r=cfg["BLOCK_R"],
        **launch(cfg),
    )
    return out


def _merge_bwd_run(cfg, g2, factors, scale):
    a1, b1, a2, b2 = factors
    if cfg == TORCH:
        gw = g2.to(a1.dtype) * scale
        e1 = gw * (a2 @ b2)
        e2 = gw * (a1 @ b1)
        return e1 @ b1.t(), a1.t() @ e1, e2 @ b2.t(), a2.t() @ e2
    o, r = a1.shape
    i = b1.shape[1]
    bm, bn = cfg["BLOCK_M"], cfg["BLOCK_N"]
    prec = dot_precision(device_info(a1.device), a1.dtype)
    n_rows = triton.cdiv(o, bm)
    strides = (*g2.stride(), *a1.stride(), *b1.stride(), *a2.stride(), *b2.stride())
    if cfg["LAYOUT"] == "tile":
        split = max(1, min(cfg["SPLIT"], triton.cdiv(i, bn)))
        ga = [
            torch.empty(split, o, r, device=a1.device, dtype=torch.float32)
            for _ in range(2)
        ]
        gb = [
            torch.empty(n_rows, r, i, device=a1.device, dtype=torch.float32)
            for _ in range(2)
        ]
        _merge_bwd_tile_kernel[(n_rows, split)](
            g2,
            a1,
            b1,
            a2,
            b2,
            ga[0],
            gb[0],
            ga[1],
            gb[1],
            o,
            i,
            r,
            *strides,
            scale,
            prec=prec,
            block_r=pad16(r),
            block_m=bm,
            block_n=bn,
            **launch(cfg),
        )
    else:
        n_cols = triton.cdiv(i, bn)
        split = max(1, min(cfg["SPLIT"], triton.cdiv(o, bm), triton.cdiv(i, bn)))
        ga = [
            torch.empty(split, o, r, device=a1.device, dtype=torch.float32)
            for _ in range(2)
        ]
        gb = [
            torch.empty(split, r, i, device=a1.device, dtype=torch.float32)
            for _ in range(2)
        ]
        _merge_bwd_kernel[((n_rows + n_cols) * split,)](
            g2,
            a1,
            b1,
            a2,
            b2,
            ga[0],
            gb[0],
            ga[1],
            gb[1],
            o,
            i,
            r,
            *strides,
            scale,
            n_rows,
            n_cols,
            prec=prec,
            block_r=pad16(r),
            block_m=bm,
            block_n=bn,
            **launch(cfg),
        )
    return tuple(
        p.sum(0).to(f.dtype)
        for p, f in zip((ga[0], gb[0], ga[1], gb[1]), factors, strict=True)
    )


def _merge_config(dev, w_dtype, ct, o, i, r, factors, scale):
    key = ("loha_merge", (dtype_key(w_dtype), dtype_key(ct)), (o, i, r))
    return key, select(
        dev,
        key,
        lambda: _plan_merge(
            dev, o, i, r, torch.finfo(w_dtype).bits // 8, factors[0].element_size()
        ),
        measurer(
            dev,
            2 * o * i * torch.finfo(w_dtype).bits // 8,
            lambda: (torch.full((o, i), 0.5, device=factors[0].device, dtype=w_dtype),),
            lambda c, s: lambda: _merge_run(c, s[0], factors, scale),
        ),
    )


def _materialize(dev, factors, scale):
    """scale * DeltaW in the factors' dtype, through the merge kernel."""
    o, r = factors[0].shape
    i = factors[1].shape[1]
    ct = factors[0].dtype
    key, cfg = _merge_config(dev, ct, ct, o, i, r, factors, scale)
    return guarded(
        dev,
        key,
        cfg,
        lambda c: _merge_run(c, None, factors, scale, add_base=False, out_dtype=ct),
    )


def _bypass_run(cfg, dev, flat, factors, scale, base2, out_dtype):
    if cfg == TORCH:
        return eager.loha_bypass(flat, *factors, scale, base2)
    t, i = flat.shape
    o, r = factors[0].shape
    if cfg in (MATERIALIZE, MATERIALIZE_ADD):
        delta = _materialize(dev, factors, scale)
        same = base2 is not None and base2.dtype == flat.dtype
        if same and cfg == MATERIALIZE:
            return torch.addmm(base2, flat, delta.t())
        y = flat @ delta.t()
        if base2 is None:
            return y
        return y.add_(base2) if same else (base2 + y).to(out_dtype)
    a1, b1, a2, b2 = factors
    out = torch.empty(t, o, device=flat.device, dtype=out_dtype)
    b = base2 if base2 is not None else out
    grid = (triton.cdiv(t, cfg["BLOCK_M"]), triton.cdiv(o, cfg["BLOCK_N"]))
    _bypass_regen_kernel[grid](
        flat,
        a1,
        b1,
        a2,
        b2,
        b,
        out,
        t,
        o,
        i,
        r,
        *flat.stride(),
        *a1.stride(),
        *b1.stride(),
        *a2.stride(),
        *b2.stride(),
        *b.stride(),
        *out.stride(),
        scale,
        has_base=base2 is not None,
        prec=dot_precision(dev, flat.dtype),
        block_r=pad16(r),
        block_m=cfg["BLOCK_M"],
        block_n=cfg["BLOCK_N"],
        block_k=cfg["BLOCK_K"],
        **launch(cfg),
    )
    return out


def _merge_bwd_config(dev, g_dtype, ct, o, i, r, factors, scale):
    key = ("loha_merge_backward", (dtype_key(g_dtype), dtype_key(ct)), (o, i, r))
    return key, select(
        dev,
        key,
        lambda: _plan_merge_bwd(
            dev, o, i, r, torch.finfo(g_dtype).bits // 8, factors[0].element_size()
        ),
        measurer(
            dev,
            o * i * torch.finfo(g_dtype).bits // 8,
            lambda: (torch.full((o, i), 0.5, device=factors[0].device, dtype=g_dtype),),
            lambda c, s: lambda: _merge_bwd_run(c, s[0], factors, scale),
        ),
    )


def _compute_dtype(params) -> torch.dtype:
    ct = params[0].dtype
    for p in params[1:]:
        ct = torch.promote_types(ct, p.dtype)
    return ct


def loha_bypass(
    x: torch.Tensor,
    w1_up: torch.Tensor,
    w1_down: torch.Tensor,
    w2_up: torch.Tensor,
    w2_down: torch.Tensor,
    scale: float = 1.0,
    base: torch.Tensor | None = None,
) -> torch.Tensor:
    """[base +] scale * x @ ((w1_up@w1_down) * (w2_up@w2_down))^T; x is (..., in)."""
    dev = device_info(x.device)
    lead = x.shape[:-1]
    flat = x.reshape(-1, x.shape[-1])
    t, i = flat.shape
    factors = tuple(f.to(x.dtype) for f in (w1_up, w1_down, w2_up, w2_down))
    o, r = factors[0].shape
    base2 = base.reshape(-1, o) if base is not None else None
    out_dtype = x.dtype if base is None else base.dtype
    key = (
        "loha_bypass",
        (dtype_key(x.dtype), base is not None),
        (token_bucket(t), o, i, r),
    )

    def inputs():
        return copy_of(flat), copy_of(base2)

    cfg = select(
        dev,
        key,
        lambda: _plan_bypass(dev, t, o, i, r, x.element_size(), base is not None),
        measurer(
            dev,
            t * (i + o) * x.element_size(),
            inputs,
            lambda c, s: lambda: _bypass_run(
                c, dev, s[0], factors, scale, s[1], out_dtype
            ),
        ),
    )
    out = guarded(
        dev,
        key,
        cfg,
        lambda c: _bypass_run(c, dev, flat, factors, scale, base2, out_dtype),
    )
    return out.reshape(*lead, o)


def loha_bypass_backward(
    grad: torch.Tensor,
    x: torch.Tensor,
    w1_up: torch.Tensor,
    w1_down: torch.Tensor,
    w2_up: torch.Tensor,
    w2_down: torch.Tensor,
    scale: float = 1.0,
) -> tuple[torch.Tensor, ...]:
    """(grad_x, g_w1_up, g_w1_down, g_w2_up, g_w2_down) for :func:`loha_bypass`."""
    dev = device_info(x.device)
    flat = x.reshape(-1, x.shape[-1])
    i = flat.shape[1]
    params = (w1_up, w1_down, w2_up, w2_down)
    factors = tuple(f.to(x.dtype) for f in params)
    o, r = factors[0].shape
    g2 = grad.reshape(-1, o).to(x.dtype)
    delta = _materialize(dev, factors, scale)
    gx = (g2 @ delta).reshape(x.shape)
    gw = g2.t() @ flat
    key, cfg = _merge_bwd_config(dev, gw.dtype, x.dtype, o, i, r, factors, scale)
    grads = guarded(dev, key, cfg, lambda c: _merge_bwd_run(c, gw, factors, scale))
    return (gx, *(g.to(p.dtype) for g, p in zip(grads, params, strict=True)))


def loha_merge(
    weight: torch.Tensor,
    w1_up: torch.Tensor,
    w1_down: torch.Tensor,
    w2_up: torch.Tensor,
    w2_down: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """weight + scale * (w1_up@w1_down) * (w2_up@w2_down), weight's dtype and shape."""
    dev = device_info(weight.device)
    params = (w1_up, w1_down, w2_up, w2_down)
    ct = _compute_dtype(params)
    factors = tuple(p.to(ct) for p in params)
    o, r = factors[0].shape
    i = factors[1].shape[1]
    w2 = weight.reshape(o, -1)
    key, cfg = _merge_config(dev, weight.dtype, ct, o, i, r, factors, scale)
    return guarded(dev, key, cfg, lambda c: _merge_run(c, w2, factors, scale)).reshape(
        weight.shape
    )


def loha_merge_backward(
    grad: torch.Tensor,
    w1_up: torch.Tensor,
    w1_down: torch.Tensor,
    w2_up: torch.Tensor,
    w2_down: torch.Tensor,
    scale: float = 1.0,
) -> tuple[torch.Tensor, ...]:
    """(g_w1_up, g_w1_down, g_w2_up, g_w2_down) for :func:`loha_merge`."""
    dev = device_info(grad.device)
    params = (w1_up, w1_down, w2_up, w2_down)
    ct = _compute_dtype(params)
    factors = tuple(p.to(ct) for p in params)
    o, r = factors[0].shape
    i = factors[1].shape[1]
    g2 = grad.reshape(o, -1)
    key, cfg = _merge_bwd_config(dev, grad.dtype, ct, o, i, r, factors, scale)
    grads = guarded(dev, key, cfg, lambda c: _merge_bwd_run(c, g2, factors, scale))
    return tuple(g.to(p.dtype) for g, p in zip(grads, params, strict=True))
