# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LoRA in TileLang for pre-Ampere tensor cores (fp16 in, fp32 accumulate).

bypass: a program owns BM tokens. h = x_tile @ down^T accumulates over the
input axis in an fp32 fragment, is rounded to fp16 in shared memory (as the
two-Linear chain rounds it), then feeds y = [base +] scale * h @ up^T over
the program's 1/SPLIT of the output axis. The rank pads to 16 with zero rows.
Its backward's gx is the same kernel on (up^T, down^T); h, q and the factor
gradients are cuBLAS GEMMs.

merge: out = [w +] scale * up @ down, one program per output tile, the base
add in the epilogue. Its backward reads each grad tile once for both factor
gradients ("tile"): g_up's rows accumulate over the program's columns, g_down
is stored per row tile; fp32 partials summed in torch. Volta's MMA path has
no transposed A operand, so up^T is staged instead of transposing up.
"""

import tilelang.language as T  # noqa: N812 -- TileLang's own spelling
import torch
from torch.nn import functional

from comfy_kitchen.backends.eager.adapters import lora as eager
from comfy_kitchen.backends.triton.adapters.common import (
    copy_of,
    dtype_key,
    token_bucket,
)
from comfy_kitchen.backends.triton.adapters.tuning import TORCH, device_info, select
from comfy_kitchen.backends.triton.adapters.tuning.model import (
    Work,
    rank,
    shared_reads,
    tiled_reads,
)

from comfy_kitchen.backends.triton.adapters import lora as triton_lora

from .common import (
    TRITON,
    measurer,
    compiled,
    fits,
    guarded,
    pad_to,
    padded_cols,
    padded_rows,
    partitions,
    whole_ops,
)

F16 = "float16"
F32 = "float32"


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _bypass_prim(o, i, rp, bm, bn, bk, threads, split, has_base):
    """y = [base +] scale * (x @ down^T) @ up^T; a program owns bm tokens and 1/split of the outputs."""
    t = T.symbolic("t")
    per = _cdiv(_cdiv(o, bn), split)

    @T.prim_func
    def with_base(
        x: T.Tensor((t, i), F16),
        down: T.Tensor((rp, i), F16),
        up: T.Tensor((o, rp), F16),
        base: T.Tensor((t, o), F16),
        y: T.Tensor((t, o), F16),
        scale: T.float32,
    ):
        # Grid: (output slices bs, token tiles bt).
        with T.Kernel(split, T.ceildiv(t, bm), threads=threads) as (bs, bt):
            x_s = T.alloc_shared((bm, bk), F16)
            d_s = T.alloc_shared((rp, bk), F16)
            h_f = T.alloc_fragment((bm, rp), F32)
            h_s = T.alloc_shared((bm, rp), F16)
            u_s = T.alloc_shared((bn, rp), F16)
            y_f = T.alloc_fragment((bm, bn), F32)
            b_f = T.alloc_fragment((bm, bn), F16)

            # h = x tile @ down^T over the input axis, then fp16 in shared memory.
            T.clear(h_f)
            for k in T.Pipelined(T.ceildiv(i, bk), num_stages=1):
                T.copy(x[bt * bm, k * bk], x_s)
                T.copy(down[0, k * bk], d_s)
                T.gemm(x_s, d_s, h_f, transpose_B=True)
            T.copy(h_f, h_s)

            # Each bn-wide output chunk of this slice: y = base + scale * h @ up^T.
            for n in T.serial(per):
                if (bs * per + n) * bn < o:
                    T.copy(up[(bs * per + n) * bn, 0], u_s)
                    T.clear(y_f)
                    T.gemm(h_s, u_s, y_f, transpose_B=True)
                    T.copy(base[bt * bm, (bs * per + n) * bn], b_f)
                    for a, b in T.Parallel(bm, bn):
                        y_f[a, b] = b_f[a, b] + scale * y_f[a, b]
                    T.copy(y_f, y[bt * bm, (bs * per + n) * bn])

    @T.prim_func
    def plain(
        x: T.Tensor((t, i), F16),
        down: T.Tensor((rp, i), F16),
        up: T.Tensor((o, rp), F16),
        y: T.Tensor((t, o), F16),
        scale: T.float32,
    ):
        with T.Kernel(split, T.ceildiv(t, bm), threads=threads) as (bs, bt):
            x_s = T.alloc_shared((bm, bk), F16)
            d_s = T.alloc_shared((rp, bk), F16)
            h_f = T.alloc_fragment((bm, rp), F32)
            h_s = T.alloc_shared((bm, rp), F16)
            u_s = T.alloc_shared((bn, rp), F16)
            y_f = T.alloc_fragment((bm, bn), F32)

            # h = x tile @ down^T over the input axis, then fp16 in shared memory.
            T.clear(h_f)
            for k in T.Pipelined(T.ceildiv(i, bk), num_stages=1):
                T.copy(x[bt * bm, k * bk], x_s)
                T.copy(down[0, k * bk], d_s)
                T.gemm(x_s, d_s, h_f, transpose_B=True)
            T.copy(h_f, h_s)

            # Each bn-wide output chunk of this slice: y = scale * h @ up^T.
            for n in T.serial(per):
                if (bs * per + n) * bn < o:
                    T.copy(up[(bs * per + n) * bn, 0], u_s)
                    T.clear(y_f)
                    T.gemm(h_s, u_s, y_f, transpose_B=True)
                    for a, b in T.Parallel(bm, bn):
                        y_f[a, b] = scale * y_f[a, b]
                    T.copy(y_f, y[bt * bm, (bs * per + n) * bn])

    return with_base if has_base else plain


def _merge_prim(o, i, rp, bm, bn, threads, has_base):
    """out = [w +] scale * up @ down; grid (column tiles bx, row tiles by), one GEMM over rank per tile."""

    @T.prim_func
    def with_base(
        up: T.Tensor((o, rp), F16),
        down: T.Tensor((rp, i), F16),
        w: T.Tensor((o, i), F16),
        out: T.Tensor((o, i), F16),
        scale: T.float32,
    ):
        with T.Kernel(T.ceildiv(i, bn), T.ceildiv(o, bm), threads=threads) as (bx, by):
            u_s = T.alloc_shared((bm, rp), F16)
            d_s = T.alloc_shared((rp, bn), F16)
            acc = T.alloc_fragment((bm, bn), F32)
            w_f = T.alloc_fragment((bm, bn), F16)
            T.clear(acc)
            T.copy(up[by * bm, 0], u_s)
            T.copy(down[0, bx * bn], d_s)
            T.gemm(u_s, d_s, acc)
            T.copy(w[by * bm, bx * bn], w_f)
            for a, b in T.Parallel(bm, bn):
                acc[a, b] = w_f[a, b] + scale * acc[a, b]
            T.copy(acc, out[by * bm, bx * bn])

    @T.prim_func
    def plain(
        up: T.Tensor((o, rp), F16),
        down: T.Tensor((rp, i), F16),
        out: T.Tensor((o, i), F16),
        scale: T.float32,
    ):
        with T.Kernel(T.ceildiv(i, bn), T.ceildiv(o, bm), threads=threads) as (bx, by):
            u_s = T.alloc_shared((bm, rp), F16)
            d_s = T.alloc_shared((rp, bn), F16)
            acc = T.alloc_fragment((bm, bn), F32)
            T.clear(acc)
            T.copy(up[by * bm, 0], u_s)
            T.copy(down[0, bx * bn], d_s)
            T.gemm(u_s, d_s, acc)
            for a, b in T.Parallel(bm, bn):
                acc[a, b] = scale * acc[a, b]
            T.copy(acc, out[by * bm, bx * bn])

    return with_base if has_base else plain


def _merge_bwd_prim(o, i, rp, bm, bn, threads, split):
    """fp32 partials: gu[split] = g @ down^T per row tile, gd[row tile] = up^T @ g; grid (splits, row tiles)."""
    n_m = _cdiv(o, bm)
    per = _cdiv(_cdiv(i, bn), split)

    @T.prim_func
    def main(
        g: T.Tensor((o, i), F16),
        up_t: T.Tensor((rp, o), F16),
        down: T.Tensor((rp, i), F16),
        gu: T.Tensor((split, o, rp), F32),
        gd: T.Tensor((n_m, rp, i), F32),
    ):
        with T.Kernel(split, n_m, threads=threads) as (bs, bm_i):
            g_s = T.alloc_shared((bm, bn), F16)
            d_s = T.alloc_shared((rp, bn), F16)
            ut_s = T.alloc_shared((rp, bm), F16)
            gu_f = T.alloc_fragment((bm, rp), F32)
            gd_f = T.alloc_fragment((rp, bn), F32)

            # up^T of this row tile, held for every g tile; gu_f accumulates this split.
            T.clear(gu_f)
            T.copy(up_t[0, bm_i * bm], ut_s)

            # Each g tile of this split: gu += g @ down^T, gd tile = up^T @ g.
            for n in T.serial(per):
                if (bs * per + n) * bn < i:
                    T.copy(g[bm_i * bm, (bs * per + n) * bn], g_s)
                    T.copy(down[0, (bs * per + n) * bn], d_s)
                    T.gemm(g_s, d_s, gu_f, transpose_B=True)
                    T.clear(gd_f)
                    T.gemm(ut_s, g_s, gd_f)
                    T.copy(gd_f, gd[bm_i, 0, (bs * per + n) * bn])
            T.copy(gu_f, gu[bs, bm_i * bm, 0])

    return main


def _bypass_smem(cfg, rp):
    bm, bn, bk = cfg["BM"], cfg["BN"], cfg["BK"]
    return bm * bk + rp * bk + bm * rp + bn * rp


def _plan_bypass(dev, t, o, i, r, has_base):
    """Legal bypass configs, model-ranked, then torch."""
    rp = pad_to(r)
    if rp > 128:
        return whole_ops(dev)
    space = [
        {"BM": bm, "BN": bn, "BK": bk, "threads": th, "SPLIT": s}
        for bm in (32, 64, 128)
        for bn in (64, 128)
        for bk in (32, 64)
        for th in (128, 256)
        for s in (1, 2, 4, 8)
    ]
    space = [
        c
        for c in space
        if c["SPLIT"] <= _cdiv(o, c["BN"]) and _bypass_legal(dev, c, rp)
    ]

    def work(c):
        bm, bn, split = c["BM"], c["BN"], c["SPLIT"]
        tiles = _cdiv(t, bm)
        # Every output slice rebuilds h from x.
        return Work(
            grid=tiles * split,
            warps=c["threads"] // 32,
            acc_fp32=bm * rp + bm * bn,
            operand_bytes=0,
            smem_stage=2 * _bypass_smem(c, rp),
            stages=1,
            hbm_bytes=2 * (t * i * split + t * o * (2 if has_base else 1))
            + shared_reads(dev, 2 * (i + o) * rp, tiles * split),
            mma_flops=2.0 * tiles * bm * rp * (i * split + o),
            # every program reads all of down and its 1/split of up
            l2_bytes=2
            * (
                t * i * split
                + tiles * split * (i + _cdiv(o, split)) * rp
                + t * o * (2 if has_base else 1)
            ),
        )

    return [*rank(dev, space, work), *whole_ops(dev)]


def _bypass_legal(dev, cfg, rp):
    bm = cfg["BM"]
    return fits(dev, _bypass_smem(cfg, rp)) and partitions(
        cfg["threads"], (bm, rp), (bm, cfg["BN"])
    )


def _bypass_run(cfg, flat, up, down, scale, base2, out_dtype):
    """[base +] scale * (flat @ down^T) @ up^T; flat, up, down fp16."""
    if cfg == TORCH:
        return eager.lora_bypass(flat, up, down, scale, base2)
    if cfg == TRITON:
        return triton_lora.lora_bypass(flat, up, down, scale, base2).to(out_dtype)
    t, i = flat.shape
    o, r = up.shape
    rp = pad_to(r)
    d = padded_rows(down, rp)
    u = padded_cols(up, rp)
    fused_base = base2 is not None and base2.dtype == torch.float16
    args = (
        o,
        i,
        rp,
        cfg["BM"],
        cfg["BN"],
        cfg["BK"],
        cfg["threads"],
        cfg["SPLIT"],
        fused_base,
    )
    kernel = compiled(_bypass_prim, flat.device, *args)
    y = torch.empty(t, o, device=flat.device, dtype=torch.float16)
    x = flat.contiguous()
    if fused_base:
        kernel(x, d, u, base2.contiguous(), y, scale)
        return y if out_dtype == torch.float16 else y.to(out_dtype)
    kernel(x, d, u, y, scale)
    if base2 is None:
        return y.to(out_dtype)
    return (base2 + y).to(out_dtype)


def lora_bypass(x, up, down, scale=1.0, base=None):
    """[base +] scale * (x @ down^T) @ up^T for a linear adapter; x is (..., in)."""
    dev = device_info(x.device)
    lead = x.shape[:-1]
    flat = x.reshape(-1, x.shape[-1])
    t, i = flat.shape
    o, r = up.shape
    u, d = up.to(torch.float16), down.to(torch.float16)
    xf = flat.to(torch.float16)
    base2 = base.reshape(-1, o) if base is not None else None
    out_dtype = x.dtype if base is None else base.dtype
    key = (
        "tl_lora_bypass",
        (dtype_key(x.dtype), base is not None),
        (token_bucket(t), o, i, r),
    )
    cfg = select(
        dev,
        key,
        lambda: _plan_bypass(dev, t, o, i, r, base is not None),
        measurer(
            dev,
            t * (i + o * (2 if base2 is not None else 1)) * 2,
            lambda: (copy_of(xf), copy_of(base2)),
            lambda c, s: lambda: _bypass_run(c, s[0], u, d, scale, s[1], out_dtype),
        ),
    )
    out = guarded(
        dev, key, cfg, lambda c: _bypass_run(c, xf, u, d, scale, base2, out_dtype)
    )
    return out.reshape(*lead, o)


def lora_bypass_backward(grad, x, up, down, scale=1.0):
    """(grad_x, grad_up, grad_down) for :func:`lora_bypass`."""
    dev = device_info(x.device)
    flat = x.reshape(-1, x.shape[-1]).to(torch.float16)
    t, i = flat.shape
    o, r = up.shape
    u, d = up.to(torch.float16), down.to(torch.float16)
    g2 = grad.reshape(-1, o).to(torch.float16)
    ut, dt = u.t(), d.t()
    key = ("tl_lora_bypass_dx", (dtype_key(x.dtype),), (token_bucket(t), i, o, r))
    cfg = select(
        dev,
        key,
        lambda: _plan_bypass(dev, t, i, o, r, False),
        measurer(
            dev,
            t * (i + o) * 2,
            lambda: (copy_of(g2),),
            lambda c, s: lambda: _bypass_run(
                c, s[0], dt, ut, scale, None, torch.float16
            ),
        ),
    )
    gx = guarded(
        dev, key, cfg, lambda c: _bypass_run(c, g2, dt, ut, scale, None, torch.float16)
    )
    h = functional.linear(flat, d)
    q = g2 @ u
    gu = torch.mm(g2.t(), h).mul_(scale)
    gd = torch.mm(q.t(), flat).mul_(scale)
    return gx.reshape(x.shape).to(x.dtype), gu.to(up.dtype), gd.to(down.dtype)


def _plan_merge(dev, o, i, r, e_w):
    rp = pad_to(r)
    cfgs = [
        {"BM": bm, "BN": bn, "threads": th}
        for bm in (32, 64, 128)
        for bn in (64, 128, 256)
        for th in (128, 256)
    ]

    def work(c):
        bm, bn = c["BM"], c["BN"]
        tm, tn = _cdiv(o, bm), _cdiv(i, bn)
        return Work(
            grid=tm * tn,
            warps=c["threads"] // 32,
            acc_fp32=bm * bn,
            operand_bytes=0,
            smem_stage=2 * (bm + bn) * rp,
            stages=1,
            hbm_bytes=2 * o * i * e_w
            + shared_reads(dev, 2 * (o + i) * rp, max(tm, tn)),
            mma_flops=2.0 * tm * bm * tn * bn * rp,
            l2_bytes=2 * o * i * e_w + tiled_reads(o, i, rp, bm, bn, 2),
        )

    return [
        *rank(dev, [c for c in cfgs if _merge_legal(dev, c, rp)], work),
        *whole_ops(dev),
    ]


def _merge_legal(dev, cfg, rp):
    return fits(dev, (cfg["BM"] + cfg["BN"]) * rp) and partitions(
        cfg["threads"], (cfg["BM"], cfg["BN"])
    )


def _merge_run(cfg, w2, up, down, scale, out_dtype):
    if cfg == TORCH:
        return eager.lora_merge(w2, up, down, scale).to(out_dtype)
    if cfg == TRITON:
        return triton_lora.lora_merge(w2, up, down, scale).to(out_dtype)
    o, r = up.shape
    i = down.shape[1]
    rp = pad_to(r)
    u = padded_cols(up.to(torch.float16), rp)
    d = padded_rows(down.to(torch.float16), rp)
    fused = w2 is not None and w2.dtype == torch.float16
    kernel = compiled(
        _merge_prim, up.device, o, i, rp, cfg["BM"], cfg["BN"], cfg["threads"], fused
    )
    out = torch.empty(o, i, device=up.device, dtype=torch.float16)
    if fused:
        kernel(u, d, w2.contiguous(), out, scale)
        return out if out_dtype == torch.float16 else out.to(out_dtype)
    kernel(u, d, out, scale)
    return (w2 + out).to(out_dtype) if w2 is not None else out.to(out_dtype)


def lora_merge(weight, up, down, scale=1.0):
    """weight + scale * (up @ down), in the weight's dtype and shape."""
    dev = device_info(weight.device)
    o, r = up.shape
    i = down.shape[1]
    w2 = weight.reshape(o, -1)
    key = ("tl_lora_merge", (dtype_key(weight.dtype), dtype_key(up.dtype)), (o, i, r))
    cfg = select(
        dev,
        key,
        lambda: _plan_merge(dev, o, i, r, weight.element_size()),
        measurer(
            dev,
            2 * o * i * weight.element_size(),
            lambda: (copy_of(w2),),
            lambda c, s: lambda: _merge_run(c, s[0], up, down, scale, weight.dtype),
        ),
    )
    return guarded(
        dev, key, cfg, lambda c: _merge_run(c, w2, up, down, scale, weight.dtype)
    ).reshape(weight.shape)


def _plan_merge_bwd(dev, o, i, r, e_g):
    rp = pad_to(r)
    if rp > 128:
        return whole_ops(dev)
    cfgs = [
        {"BM": bm, "BN": bn, "threads": th, "SPLIT": s}
        for bm in (32, 64, 128)
        for bn in (64, 128)
        for th in (128, 256)
        for s in (1, 2, 4, 8, 16)
    ]
    legal = [
        c
        for c in cfgs
        if fits(dev, c["BM"] * c["BN"] + rp * (c["BN"] + c["BM"]))
        and partitions(c["threads"], (c["BM"], rp), (rp, c["BN"]))
    ]

    def work(c):
        bm, bn, split = c["BM"], c["BN"], c["SPLIT"]
        tm = _cdiv(o, bm)
        # g read once; fp32 partials: g_up per split, g_down per row tile.
        return Work(
            grid=tm * split,
            warps=c["threads"] // 32,
            acc_fp32=bm * rp + rp * bn,
            operand_bytes=0,
            smem_stage=2 * (bm * bn + rp * (bn + bm)),
            stages=1,
            hbm_bytes=o * i * e_g
            + 2 * (split * o + tm * i) * rp * 4
            + shared_reads(dev, 2 * (o + i) * rp, tm * split),
            mma_flops=4.0 * o * i * rp,
            # each (row tile, split) reads its slice of down and the tile's rows of up
            l2_bytes=o * i * e_g
            + 2 * (tm * i + tm * split * bm) * rp
            + 2 * (split * o + tm * i) * rp * 4,
        )

    return [*rank(dev, legal, work), *whole_ops(dev)]


def _merge_bwd_run(cfg, g2, up, down, scale):
    if cfg == TORCH:
        g = g2.to(up.dtype)
        return torch.mm(g, down.t()).mul_(scale), torch.mm(up.t(), g).mul_(scale)
    if cfg == TRITON:
        return triton_lora.lora_merge_backward(g2, up, down, scale)
    o, r = up.shape
    i = down.shape[1]
    rp = pad_to(r)
    bm, bn = cfg["BM"], cfg["BN"]
    split = max(1, min(cfg["SPLIT"], _cdiv(i, bn)))
    ut = padded_rows(up.to(torch.float16).t(), rp)
    d = padded_rows(down.to(torch.float16), rp)
    gu = torch.empty(split, o, rp, device=up.device, dtype=torch.float32)
    gd = torch.empty(_cdiv(o, bm), rp, i, device=up.device, dtype=torch.float32)
    kernel = compiled(
        _merge_bwd_prim, up.device, o, i, rp, bm, bn, cfg["threads"], split
    )
    kernel(g2.to(torch.float16).contiguous(), ut, d, gu, gd)
    return (
        gu.sum(0)[:, :r].mul_(scale).to(up.dtype),
        gd.sum(0)[:r].mul_(scale).to(down.dtype),
    )


def lora_merge_backward(grad, up, down, scale=1.0):
    """(grad_up, grad_down) for :func:`lora_merge`; grad_weight is grad itself."""
    dev = device_info(up.device)
    o, r = up.shape
    i = down.shape[1]
    g2 = grad.reshape(o, -1)
    key = (
        "tl_lora_merge_backward",
        (dtype_key(grad.dtype), dtype_key(up.dtype)),
        (o, i, r),
    )
    cfg = select(
        dev,
        key,
        lambda: _plan_merge_bwd(dev, o, i, r, grad.element_size()),
        measurer(
            dev,
            o * i * grad.element_size(),
            lambda: (copy_of(g2),),
            lambda c, s: lambda: _merge_bwd_run(c, s[0], up, down, scale),
        ),
    )
    return guarded(dev, key, cfg, lambda c: _merge_bwd_run(c, g2, up, down, scale))
