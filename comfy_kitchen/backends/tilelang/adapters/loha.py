# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LoHa in TileLang: DeltaW = (w1_up@w1_down) * (w2_up@w2_down).

merge: one program per output tile builds both rank-r products into fp32
fragments and folds the hadamard, the scale and the base add into the
epilogue. The bypass materializes scale*DeltaW with that kernel and runs one
cuBLAS GEMM, the base its addend; its backward is gx = g @ DeltaW and
gW = g^T @ x (cuBLAS) feeding the merge backward.

merge backward ("tile"): a program owns a row tile and 1/SPLIT of the
columns; each grad tile builds both products once and feeds all four
gradients. The row-side gradients accumulate in fragments over the
program's columns, the column-side ones are stored per row tile; fp32
partials summed in torch. The up factors are staged transposed, since
Volta's MMA path has no transposed A operand.
"""

import tilelang.language as T  # noqa: N812 -- TileLang's own spelling
import torch

from comfy_kitchen.backends.eager.adapters import loha as eager
from comfy_kitchen.backends.triton.adapters.common import dtype_key
from comfy_kitchen.backends.triton.adapters.tuning import TORCH, device_info, select
from comfy_kitchen.backends.triton.adapters.tuning.model import (
    Work,
    rank,
    shared_reads,
    tiled_reads,
)

from comfy_kitchen.backends.triton.adapters import loha as triton_loha

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
MAX_RANK = 128


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _merge_prim(o, i, rp, bm, bn, threads, has_base):
    """out = [w +] scale * (a1 @ b1) * (a2 @ b2); grid (column tiles bx, row tiles by)."""

    # acc1 = a1 @ b1 and acc2 = a2 @ b2 on output tile (by, bx), factors staged in shared memory.
    @T.macro
    def products(a1, b1, a2, b2, bx, by, acc1, acc2):
        a1_s = T.alloc_shared((bm, rp), F16)
        b1_s = T.alloc_shared((rp, bn), F16)
        a2_s = T.alloc_shared((bm, rp), F16)
        b2_s = T.alloc_shared((rp, bn), F16)
        T.copy(a1[by * bm, 0], a1_s)
        T.copy(b1[0, bx * bn], b1_s)
        T.copy(a2[by * bm, 0], a2_s)
        T.copy(b2[0, bx * bn], b2_s)
        T.clear(acc1)
        T.clear(acc2)
        T.gemm(a1_s, b1_s, acc1)
        T.gemm(a2_s, b2_s, acc2)

    @T.prim_func
    def with_base(
        a1: T.Tensor((o, rp), F16),
        b1: T.Tensor((rp, i), F16),
        a2: T.Tensor((o, rp), F16),
        b2: T.Tensor((rp, i), F16),
        w: T.Tensor((o, i), F16),
        out: T.Tensor((o, i), F16),
        scale: T.float32,
    ):
        with T.Kernel(T.ceildiv(i, bn), T.ceildiv(o, bm), threads=threads) as (bx, by):
            acc1 = T.alloc_fragment((bm, bn), F32)
            acc2 = T.alloc_fragment((bm, bn), F32)
            w_f = T.alloc_fragment((bm, bn), F16)
            products(a1, b1, a2, b2, bx, by, acc1, acc2)

            # Epilogue: w + scale * acc1 * acc2, stored.
            T.copy(w[by * bm, bx * bn], w_f)
            for p, q in T.Parallel(bm, bn):
                acc1[p, q] = w_f[p, q] + scale * acc1[p, q] * acc2[p, q]
            T.copy(acc1, out[by * bm, bx * bn])

    @T.prim_func
    def plain(
        a1: T.Tensor((o, rp), F16),
        b1: T.Tensor((rp, i), F16),
        a2: T.Tensor((o, rp), F16),
        b2: T.Tensor((rp, i), F16),
        out: T.Tensor((o, i), F16),
        scale: T.float32,
    ):
        with T.Kernel(T.ceildiv(i, bn), T.ceildiv(o, bm), threads=threads) as (bx, by):
            acc1 = T.alloc_fragment((bm, bn), F32)
            acc2 = T.alloc_fragment((bm, bn), F32)
            products(a1, b1, a2, b2, bx, by, acc1, acc2)
            for p, q in T.Parallel(bm, bn):
                acc1[p, q] = scale * acc1[p, q] * acc2[p, q]
            T.copy(acc1, out[by * bm, bx * bn])

    return with_base if has_base else plain


def _merge_bwd_prim(o, i, rp, bm, bn, threads, split):
    """fp32 partials of the four gradients; grid (splits, row tiles), each g tile read once."""
    n_m = _cdiv(o, bm)
    per = _cdiv(_cdiv(i, bn), split)

    @T.prim_func
    def main(
        g: T.Tensor((o, i), F16),
        a1: T.Tensor((o, rp), F16),
        a1t: T.Tensor((rp, o), F16),
        b1: T.Tensor((rp, i), F16),
        a2: T.Tensor((o, rp), F16),
        a2t: T.Tensor((rp, o), F16),
        b2: T.Tensor((rp, i), F16),
        ga1: T.Tensor((split, o, rp), F32),
        gb1: T.Tensor((n_m, rp, i), F32),
        ga2: T.Tensor((split, o, rp), F32),
        gb2: T.Tensor((n_m, rp, i), F32),
        scale: T.float32,
    ):
        with T.Kernel(split, n_m, threads=threads) as (bs, bmi):
            a1_s = T.alloc_shared((bm, rp), F16)
            a2_s = T.alloc_shared((bm, rp), F16)
            a1t_s = T.alloc_shared((rp, bm), F16)
            a2t_s = T.alloc_shared((rp, bm), F16)
            b1_s = T.alloc_shared((rp, bn), F16)
            b2_s = T.alloc_shared((rp, bn), F16)
            g_f = T.alloc_fragment((bm, bn), F16)
            p1 = T.alloc_fragment((bm, bn), F32)
            p2 = T.alloc_fragment((bm, bn), F32)
            e1_s = T.alloc_shared((bm, bn), F16)
            e2_s = T.alloc_shared((bm, bn), F16)
            ga1_f = T.alloc_fragment((bm, rp), F32)
            ga2_f = T.alloc_fragment((bm, rp), F32)
            gb_f = T.alloc_fragment((rp, bn), F32)

            # Up factors of this row tile (and their transposes), held for every column tile.
            T.copy(a1[bmi * bm, 0], a1_s)
            T.copy(a2[bmi * bm, 0], a2_s)
            T.copy(a1t[0, bmi * bm], a1t_s)
            T.copy(a2t[0, bmi * bm], a2t_s)
            T.clear(ga1_f)
            T.clear(ga2_f)

            # Each column tile of this split.
            for n in T.serial(per):
                if (bs * per + n) * bn < i:
                    col = (bs * per + n) * bn
                    T.copy(b1[0, col], b1_s)
                    T.copy(b2[0, col], b2_s)
                    T.copy(g[bmi * bm, col], g_f)

                    # p1, p2 = the two rank products; e1 = scale*g*p2, e2 = scale*g*p1.
                    T.clear(p1)
                    T.clear(p2)
                    T.gemm(a1_s, b1_s, p1)
                    T.gemm(a2_s, b2_s, p2)
                    for p, q in T.Parallel(bm, bn):
                        e1_s[p, q] = scale * g_f[p, q] * p2[p, q]
                        e2_s[p, q] = scale * g_f[p, q] * p1[p, q]

                    # ga += e @ b^T; gb tile = a^T @ e, stored per row tile.
                    T.gemm(e1_s, b1_s, ga1_f, transpose_B=True)
                    T.gemm(e2_s, b2_s, ga2_f, transpose_B=True)
                    T.clear(gb_f)
                    T.gemm(a1t_s, e1_s, gb_f)
                    T.copy(gb_f, gb1[bmi, 0, col])
                    T.clear(gb_f)
                    T.gemm(a2t_s, e2_s, gb_f)
                    T.copy(gb_f, gb2[bmi, 0, col])
            T.copy(ga1_f, ga1[bs, bmi * bm, 0])
            T.copy(ga2_f, ga2[bs, bmi * bm, 0])

    return main


def _padded(factors, rp):
    a1, b1, a2, b2 = (f.to(torch.float16) for f in factors)
    return (
        padded_cols(a1, rp),
        padded_rows(b1, rp),
        padded_cols(a2, rp),
        padded_rows(b2, rp),
    )


def _merge_smem(cfg, rp):
    return 2 * cfg["BM"] * rp + 2 * rp * cfg["BN"]


def _merge_bwd_smem(cfg, rp):
    bm, bn = cfg["BM"], cfg["BN"]
    return 4 * bm * rp + 2 * rp * bn + 2 * bm * bn


def _plan_merge(dev, o, i, r, e_w):
    rp = pad_to(r)
    if rp > MAX_RANK:
        return whole_ops(dev)
    cfgs = [
        {"BM": bm, "BN": bn, "threads": th}
        for bm in (32, 64, 128)
        for bn in (64, 128)
        for th in (128, 256)
    ]

    def work(c):
        bm, bn = c["BM"], c["BN"]
        tm, tn = _cdiv(o, bm), _cdiv(i, bn)
        return Work(
            grid=tm * tn,
            warps=c["threads"] // 32,
            acc_fp32=2 * bm * bn,
            operand_bytes=0,
            smem_stage=2 * _merge_smem(c, rp),
            stages=1,
            hbm_bytes=2 * o * i * e_w
            + shared_reads(dev, 4 * (o + i) * rp, max(tm, tn)),
            mma_flops=4.0 * tm * bm * tn * bn * rp,
            vector_flops=3.0 * o * i,
            l2_bytes=2 * o * i * e_w + 2 * tiled_reads(o, i, rp, bm, bn, 2),
        )

    return [
        *rank(dev, [c for c in cfgs if _merge_legal(dev, c, rp)], work),
        *whole_ops(dev),
    ]


def _merge_legal(dev, cfg, rp):
    return fits(dev, _merge_smem(cfg, rp)) and partitions(
        cfg["threads"], (cfg["BM"], cfg["BN"])
    )


def _merge_bwd_legal(dev, cfg, rp):
    bm, bn = cfg["BM"], cfg["BN"]
    return fits(dev, _merge_bwd_smem(cfg, rp)) and partitions(
        cfg["threads"], (bm, bn), (bm, rp), (rp, bn)
    )


def _merge_run(cfg, w2, factors, scale, out_dtype):
    if cfg == TORCH:
        if w2 is not None:
            return eager.loha_merge(w2, *factors, scale).to(out_dtype)
        a1, b1, a2, b2 = factors
        return eager._delta(a1.to(out_dtype) * scale, b1, a2, b2, out_dtype)
    if cfg == TRITON:
        o, i = factors[0].shape[0], factors[1].shape[1]
        base = (
            w2
            if w2 is not None
            else torch.zeros(o, i, device=factors[0].device, dtype=out_dtype)
        )
        return triton_loha.loha_merge(base, *factors, scale).to(out_dtype)
    o, r = factors[0].shape
    i = factors[1].shape[1]
    rp = pad_to(r)
    padded = _padded(factors, rp)
    fused = w2 is not None and w2.dtype == torch.float16
    kernel = compiled(
        _merge_prim,
        factors[0].device,
        o,
        i,
        rp,
        cfg["BM"],
        cfg["BN"],
        cfg["threads"],
        fused,
    )
    out = torch.empty(o, i, device=factors[0].device, dtype=torch.float16)
    if fused:
        kernel(*padded, w2.contiguous(), out, scale)
        return out if out_dtype == torch.float16 else out.to(out_dtype)
    kernel(*padded, out, scale)
    return (w2 + out).to(out_dtype) if w2 is not None else out.to(out_dtype)


def _merge_config(dev, w_dtype, o, i, r, factors, scale):
    key = ("tl_loha_merge", (dtype_key(w_dtype),), (o, i, r))
    return key, select(
        dev,
        key,
        lambda: _plan_merge(dev, o, i, r, torch.finfo(w_dtype).bits // 8),
        measurer(
            dev,
            2 * o * i * torch.finfo(w_dtype).bits // 8,
            lambda: (torch.full((o, i), 0.5, device=factors[0].device, dtype=w_dtype),),
            lambda c, s: lambda: _merge_run(c, s[0], factors, scale, w_dtype),
        ),
    )


def _delta(dev, factors, scale):
    """scale * DeltaW in fp16 through the merge kernel."""
    o, r = factors[0].shape
    i = factors[1].shape[1]
    key, cfg = _merge_config(dev, torch.float16, o, i, r, factors, scale)
    return guarded(
        dev, key, cfg, lambda c: _merge_run(c, None, factors, scale, torch.float16)
    )


def loha_merge(weight, w1_up, w1_down, w2_up, w2_down, scale=1.0):
    """weight + scale * (w1_up@w1_down) * (w2_up@w2_down), weight's dtype and shape."""
    dev = device_info(weight.device)
    factors = (w1_up, w1_down, w2_up, w2_down)
    o, r = w1_up.shape
    i = w1_down.shape[1]
    w2 = weight.reshape(o, -1)
    key, cfg = _merge_config(dev, weight.dtype, o, i, r, factors, scale)
    return guarded(
        dev, key, cfg, lambda c: _merge_run(c, w2, factors, scale, weight.dtype)
    ).reshape(weight.shape)


def loha_bypass(x, w1_up, w1_down, w2_up, w2_down, scale=1.0, base=None):
    """[base +] scale * x @ ((w1_up@w1_down) * (w2_up@w2_down))^T; x is (..., in)."""
    dev = device_info(x.device)
    lead = x.shape[:-1]
    flat = x.reshape(-1, x.shape[-1]).to(torch.float16)
    delta = _delta(dev, (w1_up, w1_down, w2_up, w2_down), scale)
    o = delta.shape[0]
    if base is not None and base.dtype == torch.float16:
        out = torch.addmm(base.reshape(-1, o), flat, delta.t())
    else:
        out = flat @ delta.t()
        if base is not None:
            out = base.reshape(-1, o) + out
    return out.to(x.dtype if base is None else base.dtype).reshape(*lead, o)


def _plan_merge_bwd(dev, o, i, r, e_g):
    rp = pad_to(r)
    if rp > MAX_RANK:
        return whole_ops(dev)
    cfgs = [
        {"BM": bm, "BN": bn, "threads": th, "SPLIT": s}
        for bm in (32, 64)
        for bn in (32, 64, 128)
        for th in (128, 256)
        for s in (1, 2, 4, 8, 16)
    ]

    def work(c):
        bm, bn, split = c["BM"], c["BN"], c["SPLIT"]
        tm = _cdiv(o, bm)
        # Products built once per grad tile (12r FLOP per element); fp32
        # partials: row-side per split, column-side per row tile.
        return Work(
            grid=tm * split,
            warps=c["threads"] // 32,
            acc_fp32=2 * (bm * bn + bm * rp + rp * bn),
            operand_bytes=0,
            smem_stage=2 * _merge_bwd_smem(c, rp),
            stages=1,
            hbm_bytes=o * i * e_g
            + 4 * (split * o + tm * i) * rp * 4
            + shared_reads(dev, 4 * (o + i) * rp, tm * split),
            mma_flops=12.0 * o * i * rp,
            vector_flops=3.0 * o * i,
            # each (row tile, split) reads both ups (and transposes) and both downs' split
            l2_bytes=o * i * e_g
            + 2 * (2 * tm * split * bm + tm * i) * rp * 2
            + 4 * (split * o + tm * i) * rp * 4,
        )

    return [
        *rank(dev, [c for c in cfgs if _merge_bwd_legal(dev, c, rp)], work),
        *whole_ops(dev),
    ]


def _merge_bwd_run(cfg, g2, factors, scale):
    if cfg == TORCH:
        return eager.loha_merge_backward(g2, *factors, scale)
    if cfg == TRITON:
        return triton_loha.loha_merge_backward(g2, *factors, scale)
    o, r = factors[0].shape
    i = factors[1].shape[1]
    rp = pad_to(r)
    bm, bn = cfg["BM"], cfg["BN"]
    split = max(1, min(cfg["SPLIT"], _cdiv(i, bn)))
    a1, b1, a2, b2 = _padded(factors, rp)
    dev = factors[0].device
    ga = [torch.empty(split, o, rp, device=dev, dtype=torch.float32) for _ in range(2)]
    gb = [
        torch.empty(_cdiv(o, bm), rp, i, device=dev, dtype=torch.float32)
        for _ in range(2)
    ]
    kernel = compiled(_merge_bwd_prim, dev, o, i, rp, bm, bn, cfg["threads"], split)
    kernel(
        g2.to(torch.float16).contiguous(),
        a1,
        a1.t().contiguous(),
        b1,
        a2,
        a2.t().contiguous(),
        b2,
        ga[0],
        gb[0],
        ga[1],
        gb[1],
        scale,
    )
    grads = (
        ga[0].sum(0)[:, :r],
        gb[0].sum(0)[:r],
        ga[1].sum(0)[:, :r],
        gb[1].sum(0)[:r],
    )
    return tuple(gr.to(f.dtype) for gr, f in zip(grads, factors, strict=True))


def _merge_bwd_config(dev, g_dtype, o, i, r, factors, scale):
    key = ("tl_loha_merge_backward", (dtype_key(g_dtype),), (o, i, r))
    return key, select(
        dev,
        key,
        lambda: _plan_merge_bwd(dev, o, i, r, torch.finfo(g_dtype).bits // 8),
        measurer(
            dev,
            o * i * torch.finfo(g_dtype).bits // 8,
            lambda: (torch.full((o, i), 0.5, device=factors[0].device, dtype=g_dtype),),
            lambda c, s: lambda: _merge_bwd_run(c, s[0], factors, scale),
        ),
    )


def loha_merge_backward(grad, w1_up, w1_down, w2_up, w2_down, scale=1.0):
    """(g_w1_up, g_w1_down, g_w2_up, g_w2_down) for :func:`loha_merge`."""
    dev = device_info(grad.device)
    factors = (w1_up, w1_down, w2_up, w2_down)
    o, r = w1_up.shape
    i = w1_down.shape[1]
    g2 = grad.reshape(o, -1)
    key, cfg = _merge_bwd_config(dev, grad.dtype, o, i, r, factors, scale)
    return guarded(dev, key, cfg, lambda c: _merge_bwd_run(c, g2, factors, scale))


def loha_bypass_backward(grad, x, w1_up, w1_down, w2_up, w2_down, scale=1.0):
    """(grad_x, g_w1_up, g_w1_down, g_w2_up, g_w2_down) for :func:`loha_bypass`."""
    dev = device_info(x.device)
    factors = (w1_up, w1_down, w2_up, w2_down)
    flat = x.reshape(-1, x.shape[-1]).to(torch.float16)
    o, r = w1_up.shape
    i = flat.shape[1]
    g2 = grad.reshape(-1, o).to(torch.float16)
    delta = _delta(dev, factors, scale)
    gx = (g2 @ delta).reshape(x.shape).to(x.dtype)
    gw = g2.t() @ flat
    key, cfg = _merge_bwd_config(dev, gw.dtype, o, i, r, factors, scale)
    return (
        gx,
        *guarded(dev, key, cfg, lambda c: _merge_bwd_run(c, gw, factors, scale)),
    )
