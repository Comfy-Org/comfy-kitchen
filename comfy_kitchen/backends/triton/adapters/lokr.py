# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LoKr kernels: DeltaW[u*C+c, v*D+d] = w1[u,v] * w2[c,d], bypass as Y_t = w1 @ X_t @ w2^T, and merge."""

import torch
import triton
import triton.language as tl

from comfy_kitchen.backends.eager.adapters import lokr as eager

from .common import (
    copy_of,
    dot_precision,
    dtype_key,
    grid_configs,
    guarded,
    launch,
    load2d,
    measurer,
    stage_axis,
    token_bucket,
)
from .tuning import TORCH, device_info, select
from .tuning.model import Work, cdiv, rank, shared_reads

# Largest fp32 register tile per program (128 x 128).
_TILE_BUDGET = 16384


@triton.jit
def _bypass_kernel(
    x_ptr,
    w1_ptr,
    w2_ptr,
    b_ptr,
    y_ptr,
    n_tok,
    n_a,
    n_b,
    n_c,
    n_d,
    sxt,
    sxi,
    s1a,
    s1b,
    s2c,
    s2d,
    sbt,
    sbo,
    syt,
    syo,
    scale,
    has_base: tl.constexpr,
    prec: tl.constexpr,
    block_t: tl.constexpr,
    block_v: tl.constexpr,
    block_d: tl.constexpr,
    block_c: tl.constexpr,
    block_a: tl.constexpr,
):
    """Y[t, u*n_c+c] = [base +] scale * sum_v w1[u,v] sum_d X[t, v*n_d+d] w2[c,d], w1 as a block-diagonal tile.

    Grid: (token tiles, w1 row tiles, w2 row tiles).
    """
    pid_t = tl.program_id(0)
    pid_a = tl.program_id(1)
    pid_c = tl.program_id(2)

    # mid rows are (token, v), output rows (token, u).
    kv = tl.arange(0, block_t * block_v)
    tok = pid_t * block_t + kv // block_v
    kt = kv // block_v
    vl = kv % block_v
    ru2 = tl.arange(0, block_t * block_a)
    ut = ru2 // block_a
    ru = pid_a * block_a + ru2 % block_a
    rc = pid_c * block_c + tl.arange(0, block_c)
    mtok = tok < n_tok
    mc = rc < n_c
    mu = ru < n_a
    ct = x_ptr.dtype.element_ty

    # acc rows (token, u), columns c: the output tile, reduced over w1's columns v.
    acc = tl.zeros((block_t * block_a, block_c), tl.float32)
    for v0 in range(0, n_b, block_v):
        rv = v0 + vl
        mv = rv < n_b

        # mid rows (token, v), columns c: M = X @ w2^T, reduced over d.
        mid = tl.zeros((block_t * block_v, block_c), tl.float32)
        for d0 in range(0, n_d, block_d):
            rd = d0 + tl.arange(0, block_d)
            md = rd < n_d
            xv = tl.load(
                x_ptr + tok[:, None] * sxt + (rv[:, None] * n_d + rd[None, :]) * sxi,
                mask=(mtok & mv)[:, None] & md[None, :],
                other=0.0,
            )
            w2t = load2d(w2_ptr, rd, rc, s2d, s2c, md, mc)
            mid = tl.dot(xv, w2t, mid, input_precision=prec)
        # w1 on the token-diagonal blocks, zero elsewhere.
        w1blk = tl.load(
            w1_ptr + ru[:, None] * s1a + rv[None, :] * s1b,
            mask=(ut[:, None] == kt[None, :]) & mu[:, None] & mv[None, :],
            other=0.0,
        )
        # mid in x's dtype, as eager's chain rounds it.
        acc = tl.dot(w1blk.to(ct), mid.to(ct), acc, input_precision=prec)

    # Epilogue: scale, add base, store rows (token, u) at columns u*n_c + c.
    tok2 = pid_t * block_t + ut
    omask = ((tok2 < n_tok) & mu)[:, None] & mc[None, :]
    col = ru[:, None] * n_c + rc[None, :]
    y = acc * scale
    if has_base:
        base = tl.load(b_ptr + tok2[:, None] * sbt + col * sbo, mask=omask, other=0.0)
        y += base.to(tl.float32)
    tl.store(
        y_ptr + tok2[:, None] * syt + col * syo,
        y.to(y_ptr.dtype.element_ty),
        mask=omask,
    )


@triton.jit
def _bypass_batch_kernel(
    x_ptr,
    w1_ptr,
    w2_ptr,
    b_ptr,
    y_ptr,
    n_tok,
    n_a,
    n_b,
    n_c,
    n_d,
    sxt,
    sxi,
    s1a,
    s1b,
    s2c,
    s2d,
    sbt,
    sbo,
    syt,
    syo,
    scale,
    has_base: tl.constexpr,
    prec: tl.constexpr,
    block_t: tl.constexpr,
    block_v: tl.constexpr,
    block_d: tl.constexpr,
    block_c: tl.constexpr,
    block_a: tl.constexpr,
):
    """The same product as _bypass_kernel with tokens as the batch of 3-D dots.

    Grid: (token tiles, w1 row tiles, w2 row tiles).
    """
    pid_t = tl.program_id(0)
    pid_a = tl.program_id(1)
    pid_c = tl.program_id(2)
    tok = pid_t * block_t + tl.arange(0, block_t)
    ru = pid_a * block_a + tl.arange(0, block_a)
    rc = pid_c * block_c + tl.arange(0, block_c)
    mtok = tok < n_tok
    mu = ru < n_a
    mc = rc < n_c
    ct = x_ptr.dtype.element_ty

    # acc (token, u, c): the output tile, reduced over w1's columns v.
    acc = tl.zeros((block_t, block_a, block_c), tl.float32)
    for v0 in range(0, n_b, block_v):
        rv = v0 + tl.arange(0, block_v)
        mv = rv < n_b

        # mid (token, v, c) = X_t @ w2^T per token, reduced over d.
        mid = tl.zeros((block_t, block_v, block_c), tl.float32)
        for d0 in range(0, n_d, block_d):
            rd = d0 + tl.arange(0, block_d)
            md = rd < n_d
            xv = tl.load(
                x_ptr
                + tok[:, None, None] * sxt
                + (rv[None, :, None] * n_d + rd[None, None, :]) * sxi,
                mask=mtok[:, None, None] & mv[None, :, None] & md[None, None, :],
                other=0.0,
            )
            w2t = load2d(w2_ptr, rd, rc, s2d, s2c, md, mc)
            mid = tl.dot(
                xv,
                tl.broadcast_to(w2t[None, :, :], (block_t, block_d, block_c)),
                mid,
                input_precision=prec,
            )

        # acc += w1 tile @ mid per token, mid rounded to x's dtype.
        w1 = load2d(w1_ptr, ru, rv, s1a, s1b, mu, mv)
        w1b = tl.broadcast_to(w1.to(ct)[None, :, :], (block_t, block_a, block_v))
        acc = tl.dot(w1b, mid.to(ct), acc, input_precision=prec)

    # Epilogue: scale, add base, store (token, u*n_c + c).
    omask = mtok[:, None, None] & mu[None, :, None] & mc[None, None, :]
    col = ru[None, :, None] * n_c + rc[None, None, :]
    y = acc * scale
    if has_base:
        base = tl.load(
            b_ptr + tok[:, None, None] * sbt + col * sbo, mask=omask, other=0.0
        )
        y += base.to(tl.float32)
    tl.store(
        y_ptr + tok[:, None, None] * syt + col * syo,
        y.to(y_ptr.dtype.element_ty),
        mask=omask,
    )


@triton.jit
def _merge_fwd_kernel(
    w1_ptr,
    w2_ptr,
    w_ptr,
    out_ptr,
    n_out,
    n_in,
    n_c,
    n_d,
    s1u,
    s1v,
    s2c,
    s2d,
    swo,
    swi,
    soo,
    soi,
    scale,
    add_base: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    """out = [w +] scale * kron(w1, w2), one program per output tile."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    ro = pid_m * block_m + tl.arange(0, block_m)
    ri = pid_n * block_n + tl.arange(0, block_n)
    mo = ro < n_out
    mi = ri < n_in
    omask = mo[:, None] & mi[None, :]

    # Element (u*n_c + c, v*n_d + d) = w1[u, v] * w2[c, d], gathered per element.
    u, c = ro // n_c, ro % n_c
    v, d = ri // n_d, ri % n_d
    w1 = load2d(w1_ptr, u, v, s1u, s1v, mo, mi)
    w2 = load2d(w2_ptr, c, d, s2c, s2d, mo, mi)

    # Epilogue: scale, add the weight tile, store.
    out = w1.to(tl.float32) * w2.to(tl.float32) * scale
    if add_base:
        out += load2d(w_ptr, ro, ri, swo, swi, mo, mi).to(tl.float32)
    tl.store(
        out_ptr + ro[:, None] * soo + ri[None, :] * soi,
        out.to(out_ptr.dtype.element_ty),
        mask=omask,
    )


@triton.jit
def _merge_bwd_kernel(
    g_ptr,
    w1_ptr,
    w2_ptr,
    gw1_ptr,
    gw2_ptr,
    n_a,
    n_b,
    n_c,
    n_d,
    sgo,
    sgi,
    s1u,
    s1v,
    s2c,
    s2d,
    scale,
    tiles_c,
    tiles_d,
    split: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    """g_w1[u, v] per program (first n_a*n_b), then fp32 g_w2 tile partials per split."""
    pid = tl.program_id(0)

    # g_w1[u, v] = scale * sum over the (c, d) block of g * w2.
    if pid < n_a * n_b:
        u = pid // n_b
        v = pid % n_b
        acc = tl.zeros((block_m, block_n), tl.float32)
        for c0 in range(0, n_c, block_m):
            rc = c0 + tl.arange(0, block_m)
            for d0 in range(0, n_d, block_n):
                rd = d0 + tl.arange(0, block_n)
                m = (rc < n_c)[:, None] & (rd < n_d)[None, :]
                g = tl.load(
                    g_ptr
                    + (u * n_c + rc)[:, None] * sgo
                    + (v * n_d + rd)[None, :] * sgi,
                    mask=m,
                    other=0.0,
                )
                w2 = load2d(w2_ptr, rc, rd, s2c, s2d, rc < n_c, rd < n_d)
                acc += g.to(tl.float32) * w2.to(tl.float32)
        tl.store(
            gw1_ptr + u * n_b + v, (tl.sum(acc) * scale).to(gw1_ptr.dtype.element_ty)
        )
    # g_w2 tile (c, d) partial = sum over this split's (u, v) blocks of w1[u, v] * g block.
    else:
        job = pid - n_a * n_b
        tile = job % (tiles_c * tiles_d)
        part = job // (tiles_c * tiles_d)
        rc = (tile // tiles_d) * block_m + tl.arange(0, block_m)
        rd = (tile % tiles_d) * block_n + tl.arange(0, block_n)
        m = (rc < n_c)[:, None] & (rd < n_d)[None, :]
        acc = tl.zeros((block_m, block_n), tl.float32)
        per = tl.cdiv(n_a * n_b, split)
        for uv in range(part * per, tl.minimum(part * per + per, n_a * n_b)):
            u = uv // n_b
            v = uv % n_b
            w1 = tl.load(w1_ptr + u * s1u + v * s1v).to(tl.float32)
            g = tl.load(
                g_ptr + (u * n_c + rc)[:, None] * sgo + (v * n_d + rd)[None, :] * sgi,
                mask=m,
                other=0.0,
            )
            acc += w1 * g.to(tl.float32)
        tl.store(
            gw2_ptr + (part * n_c + rc)[:, None] * n_d + rd[None, :],
            acc * scale,
            mask=m,
        )


def _bypass_legal(cfg):
    bt, bv, bc, ba, bd = cfg["BT"], cfg["BV"], cfg["BC"], cfg["BA"], cfg["BD"]
    if cfg["LAYOUT"] == "batch":
        if min(bv, ba, bc, bd) < 16:
            return False
        tiles = (bt * bv * bc, bt * bv * bd, bt * ba * bc, bt * ba * bv)
    else:
        if bt * bv < 16 or bt * ba < 16:
            return False
        tiles = (bt * bv * bc, bt * bv * bd, bt * ba * bc, bt * ba * bt * bv)
    return max(tiles) <= _TILE_BUDGET


def _factor_blocks(n):
    return tuple(
        sorted({min(triton.next_power_of_2(n), v) for v in (4, 8, 16, 32, 64, 128)})
    )


def _plan_bypass(dev, t, a, b, c, d, e, has_base):
    """Both layouts' legal configs, model-ranked, then torch."""
    stages = tuple(sorted({1, *stage_axis(dev)}))
    space = [
        cfg
        for cfg in grid_configs(
            dev,
            LAYOUT=("diag", "batch"),
            BT=(1, 2, 4, 8, 16),
            BV=_factor_blocks(b),
            BD=(16, 32, 64),
            BC=(16, 32, 64, 128),
            BA=_factor_blocks(a),
            num_warps=(2, 4, 8),
            num_stages=stages,
        )
        if _bypass_legal(cfg)
    ]

    def work(cf):
        bt, bv, bd, bc, ba = cf["BT"], cf["BV"], cf["BD"], cf["BC"], cf["BA"]
        tt, ta, tc, tv = cdiv(t, bt), cdiv(a, ba), cdiv(c, bc), cdiv(b, bv)
        mid = 2.0 * tt * bt * tv * bv * cdiv(d, bd) * bd * tc * bc * ta
        if cf["LAYOUT"] == "diag":
            w1dot = 2.0 * tt * ta * tc * tv * (bt * ba) * (bt * bv) * bc
            w1reg = bt * ba * bt * bv
        else:
            w1dot = 2.0 * tt * ta * tc * tv * bt * ba * bv * bc
            w1reg = ba * bv
        return Work(
            grid=tt * ta * tc,
            warps=cf["num_warps"],
            acc_fp32=bt * bv * bc + bt * ba * bc,
            operand_bytes=w1reg * e,
            smem_stage=(bt * bv * bd + bd * bc) * e,
            stages=cf["num_stages"],
            hbm_bytes=t * b * d * e * ta * tc
            + t * a * c * e * (2 if has_base else 1)
            + shared_reads(dev, (a * b + c * d) * e, tt * ta * tc),
            mma_flops=mid + w1dot,
            # each program reads its x rows, w2's rows once per v block, and its w1 rows
            l2_bytes=t * b * d * e * ta * tc
            + tt * ta * tc * (tv * bc * d + ba * b) * e
            + t * a * c * e * (2 if has_base else 1),
        )

    return [*rank(dev, space, work), TORCH]


def _bypass_run(cfg, dev, flat, w1, w2, scale, base2, out_dtype):
    """Y = [base +] scale * vec(w1 @ X_t @ w2^T); w1/w2 may be strided views."""
    if cfg == TORCH:
        return eager.lokr_bypass(flat, w1.contiguous(), w2.contiguous(), scale, base2)
    a, b = w1.shape
    c, d = w2.shape
    t = flat.shape[0]
    out = torch.empty(t, a * c, device=flat.device, dtype=out_dtype)
    bb = base2 if base2 is not None else out
    grid = (
        triton.cdiv(t, cfg["BT"]),
        triton.cdiv(a, cfg["BA"]),
        triton.cdiv(c, cfg["BC"]),
    )
    kernel = _bypass_batch_kernel if cfg["LAYOUT"] == "batch" else _bypass_kernel
    kernel[grid](
        flat,
        w1,
        w2,
        bb,
        out,
        t,
        a,
        b,
        c,
        d,
        *flat.stride(),
        *w1.stride(),
        *w2.stride(),
        *bb.stride(),
        *out.stride(),
        scale,
        has_base=base2 is not None,
        prec=dot_precision(dev, flat.dtype),
        block_t=cfg["BT"],
        block_v=cfg["BV"],
        block_d=cfg["BD"],
        block_c=cfg["BC"],
        block_a=cfg["BA"],
        **launch(cfg),
    )
    return out


def _bypass_config(dev, name, flat, w1, w2, scale, has_base, out_dtype):
    a, b = w1.shape
    c, d = w2.shape
    t = flat.shape[0]
    key = (name, (dtype_key(flat.dtype), has_base), (token_bucket(t), a, b, c, d))

    def inputs():
        base = (
            torch.full((t, a * c), 0.5, device=flat.device, dtype=out_dtype)
            if has_base
            else None
        )
        return copy_of(flat), base

    return key, select(
        dev,
        key,
        lambda: _plan_bypass(dev, t, a, b, c, d, flat.element_size(), has_base),
        measurer(
            dev,
            t * (b * d + a * c) * flat.element_size(),
            inputs,
            lambda cf, s: lambda: _bypass_run(
                cf, dev, s[0], w1, w2, scale, s[1], out_dtype
            ),
        ),
    )


def _plan_merge(dev, a, b, c, d, e_w, e_f):
    o, i = a * c, b * d
    space = grid_configs(
        dev,
        BLOCK_M=(8, 16, 32, 64),
        BLOCK_N=(32, 64, 128, 256),
        num_warps=(1, 2, 4, 8),
        num_stages=(1,),
    )

    def work(cf):
        bm, bn = cf["BLOCK_M"], cf["BLOCK_N"]
        return Work(
            grid=cdiv(o, bm) * cdiv(i, bn),
            warps=cf["num_warps"],
            acc_fp32=0,
            operand_bytes=bm * bn * (2 * e_w + 2 * e_f),
            smem_stage=0,
            stages=1,
            hbm_bytes=2 * o * i * e_w
            + shared_reads(dev, (a * b + c * d) * e_f, cdiv(o, bm) * cdiv(i, bn)),
            vector_flops=2.0 * o * i,
            # every output element gathers one w1 and one w2 value
            l2_bytes=2 * o * i * (e_w + e_f),
        )

    return [*rank(dev, space, work), TORCH]


def _merge_run(cfg, w2d, f1, f2, scale):
    a, b = f1.shape
    c, d = f2.shape
    o, i = a * c, b * d
    out = torch.empty_like(w2d)
    if cfg == TORCH:
        return out.copy_(w2d + scale * torch.kron(f1, f2))
    grid = (triton.cdiv(o, cfg["BLOCK_M"]), triton.cdiv(i, cfg["BLOCK_N"]))
    _merge_fwd_kernel[grid](
        f1,
        f2,
        w2d,
        out,
        o,
        i,
        c,
        d,
        *f1.stride(),
        *f2.stride(),
        *w2d.stride(),
        *out.stride(),
        scale,
        add_base=True,
        block_m=cfg["BLOCK_M"],
        block_n=cfg["BLOCK_N"],
        **launch(cfg),
    )
    return out


def _plan_merge_bwd(dev, a, b, c, d, e_g, e_f):
    o, i = a * c, b * d
    space = grid_configs(
        dev,
        BLOCK_M=(16, 32, 64),
        BLOCK_N=(32, 64, 128),
        SPLIT=(1, 4, 16, 64),
        num_warps=(2, 4, 8),
        num_stages=(1,),
    )

    def work(cf):
        bm, bn, split = cf["BLOCK_M"], cf["BLOCK_N"], min(cf["SPLIT"], a * b)
        tiles = cdiv(c, bm) * cdiv(d, bn)
        return Work(
            grid=a * b + tiles * split,
            warps=cf["num_warps"],
            acc_fp32=bm * bn,
            operand_bytes=bm * bn * (e_g + e_f),
            smem_stage=0,
            stages=1,
            hbm_bytes=2 * o * i * e_g
            + 2 * split * c * d * 4
            + shared_reads(dev, (a * b + c * d) * e_f, a * b),
            vector_flops=4.0 * o * i,
            # the w1 programs read w2 whole each; the w2 programs gather w1 per element
            l2_bytes=2 * o * i * (e_g + e_f) + 2 * split * c * d * 4,
        )

    return [*rank(dev, space, work), TORCH]


def _merge_bwd_run(cfg, g2, f1, f2, scale):
    a, b = f1.shape
    c, d = f2.shape
    if cfg == TORCH:
        g4 = g2.reshape(a, c, b, d).to(f1.dtype)
        return (
            torch.einsum("acbd,cd->ab", g4, f2).mul_(scale),
            torch.einsum("acbd,ab->cd", g4, f1).mul_(scale),
        )
    gw1 = torch.empty_like(f1)
    split = min(cfg["SPLIT"], a * b)
    part = torch.empty(split, c, d, device=f2.device, dtype=torch.float32)
    nc, nd = triton.cdiv(c, cfg["BLOCK_M"]), triton.cdiv(d, cfg["BLOCK_N"])
    _merge_bwd_kernel[(a * b + nc * nd * split,)](
        g2,
        f1,
        f2,
        gw1,
        part,
        a,
        b,
        c,
        d,
        *g2.stride(),
        *f1.stride(),
        *f2.stride(),
        scale,
        nc,
        nd,
        split=split,
        block_m=cfg["BLOCK_M"],
        block_n=cfg["BLOCK_N"],
        **launch(cfg),
    )
    return gw1, part.sum(0)


def lokr_bypass(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    scale: float = 1.0,
    base: torch.Tensor | None = None,
) -> torch.Tensor:
    """[base +] scale * x @ kron(w1, w2)^T, without building the product."""
    dev = device_info(x.device)
    lead = x.shape[:-1]
    flat = x.reshape(-1, x.shape[-1])
    f1, f2 = w1.to(x.dtype), w2.to(x.dtype)
    o = f1.shape[0] * f2.shape[0]
    base2 = base.reshape(-1, o) if base is not None else None
    out_dtype = x.dtype if base is None else base.dtype
    key, cfg = _bypass_config(
        dev, "lokr_bypass", flat, f1, f2, scale, base is not None, out_dtype
    )
    out = guarded(
        dev,
        key,
        cfg,
        lambda c: _bypass_run(c, dev, flat, f1, f2, scale, base2, out_dtype),
    )
    return out.reshape(*lead, o)


def lokr_bypass_backward(
    grad: torch.Tensor,
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(grad_x, grad_w1, grad_w2) for :func:`lokr_bypass`."""
    dev = device_info(x.device)
    f1, f2 = w1.to(x.dtype), w2.to(x.dtype)
    a, b = f1.shape
    c, d = f2.shape
    g2 = grad.reshape(-1, a * c).to(x.dtype)
    f1t, f2t = f1.t(), f2.t()
    key, cfg = _bypass_config(
        dev, "lokr_bypass_dx", g2, f1t, f2t, scale, False, x.dtype
    )
    gx = guarded(
        dev, key, cfg, lambda c: _bypass_run(c, dev, g2, f1t, f2t, scale, None, x.dtype)
    )
    x3 = x.reshape(-1, b, d)
    g3 = g2.reshape(-1, a, c)
    mid = x3 @ f2.t()
    n = torch.matmul(f1.t(), g3)
    gw1 = torch.einsum("tac,tbc->ab", g3, mid).mul_(scale)
    gw2 = torch.einsum("tbc,tbd->cd", n, x3).mul_(scale)
    return gx.reshape(x.shape), gw1.to(w1.dtype), gw2.to(w2.dtype)


def lokr_merge(
    weight: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """weight + scale * kron(w1, w2), in the weight's dtype and shape."""
    dev = device_info(weight.device)
    ct = torch.promote_types(w1.dtype, w2.dtype)
    f1, f2 = w1.to(ct), w2.to(ct)
    a, b = f1.shape
    c, d = f2.shape
    w2d = weight.reshape(a * c, -1)
    key = ("lokr_merge", (dtype_key(weight.dtype), dtype_key(ct)), (a, b, c, d))
    cfg = select(
        dev,
        key,
        lambda: _plan_merge(dev, a, b, c, d, weight.element_size(), f1.element_size()),
        measurer(
            dev,
            2 * w2d.numel() * weight.element_size(),
            lambda: (copy_of(w2d),),
            lambda cf, s: lambda: _merge_run(cf, s[0], f1, f2, scale),
        ),
    )
    return guarded(dev, key, cfg, lambda c: _merge_run(c, w2d, f1, f2, scale)).reshape(
        weight.shape
    )


def lokr_merge_backward(
    grad: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(grad_w1, grad_w2) for :func:`lokr_merge`."""
    dev = device_info(grad.device)
    ct = torch.promote_types(w1.dtype, w2.dtype)
    f1, f2 = w1.to(ct), w2.to(ct)
    a, b = f1.shape
    c, d = f2.shape
    g2 = grad.reshape(a * c, -1)
    key = ("lokr_merge_backward", (dtype_key(grad.dtype), dtype_key(ct)), (a, b, c, d))
    cfg = select(
        dev,
        key,
        lambda: _plan_merge_bwd(
            dev, a, b, c, d, grad.element_size(), f1.element_size()
        ),
        measurer(
            dev,
            g2.numel() * grad.element_size(),
            lambda: (copy_of(g2),),
            lambda cf, s: lambda: _merge_bwd_run(cf, s[0], f1, f2, scale),
        ),
    )
    gw1, gw2 = guarded(dev, key, cfg, lambda c: _merge_bwd_run(c, g2, f1, f2, scale))
    return gw1.to(w1.dtype), gw2.to(w2.dtype)
