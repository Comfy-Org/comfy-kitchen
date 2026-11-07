# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LoKr bypass in TileLang: Y_t = [B_t +] scale * w1 @ X_t @ w2^T per token.

Both products are plain 2-D GEMMs over views. A contiguous x (t, b*d) is the
(t*b, d) matrix whose rows are (token, v), so for BT whole tokens
M = X @ w2^T is a (BT*b, d) x (d, c) GEMM; y (t, a*c) is the (t*a, c) matrix
with rows (token, u), and Y = blockdiag(w1 x BT) @ M is a (BT*a, BT*b) x
(BT*b, c) GEMM against the block-diagonal tile, built once per call in
torch. BT is the smallest token count making BT*a and BT*b multiples of 16
(the MMA's extent), times a tuned factor. M is rounded to fp16 in shared
memory between the two, as eager's chain rounds it. gx is the same kernel on
(w1^T, w2^T); the factor gradients are cuBLAS GEMMs.
"""

import math

import tilelang.language as T  # noqa: N812 -- TileLang's own spelling
import torch

from comfy_kitchen.backends.eager.adapters import lokr as eager
from comfy_kitchen.backends.triton.adapters.common import (
    copy_of,
    dtype_key,
    token_bucket,
)
from comfy_kitchen.backends.triton.adapters.tuning import TORCH, device_info, select
from comfy_kitchen.backends.triton.adapters.tuning.model import Work, rank, shared_reads

from comfy_kitchen.backends.triton.adapters import lokr as triton_lokr

from .common import (
    MIN_DIM,
    TRITON,
    compiled,
    fits,
    guarded,
    measurer,
    partitions,
    whole_ops,
)

F16 = "float16"
F32 = "float32"
# Rows of either GEMM a program may hold: BT*a and BT*b.
MAX_ROWS = 128


def _bypass_prim(a, b, c, d, bt, bc, bd, threads, has_base):
    """Y = [base +] scale * blockdiag(w1) @ (X @ w2^T); grid (w2 row tiles bx, token tiles by)."""
    ra, rb = bt * a, bt * b
    tb = T.symbolic("tb")
    ta = T.symbolic("ta")

    @T.macro
    def products(xv, w2, w1bd, bx, by, y_f):
        """y_f = blockdiag(w1) @ fp16(X_tile @ w2_tile^T) for program (bx, by)."""
        x_s = T.alloc_shared((rb, bd), F16)
        w2_s = T.alloc_shared((bc, bd), F16)
        m_f = T.alloc_fragment((rb, bc), F32)
        m_s = T.alloc_shared((rb, bc), F16)
        w1_s = T.alloc_shared((ra, rb), F16)

        # M rows (token, v) = X @ w2^T over d, then fp16 in shared memory.
        T.clear(m_f)
        for k in T.Pipelined(T.ceildiv(d, bd), num_stages=1):
            T.copy(xv[by * rb, k * bd], x_s)
            T.copy(w2[bx * bc, k * bd], w2_s)
            T.gemm(x_s, w2_s, m_f, transpose_B=True)
        T.copy(m_f, m_s)

        # Y rows (token, u) = blockdiag(w1) @ M.
        T.copy(w1bd, w1_s)
        T.clear(y_f)
        T.gemm(w1_s, m_s, y_f)

    @T.prim_func
    def with_base(
        xv: T.Tensor((tb, d), F16),
        w2: T.Tensor((c, d), F16),
        w1bd: T.Tensor((ra, rb), F16),
        basev: T.Tensor((ta, c), F16),
        yv: T.Tensor((ta, c), F16),
        scale: T.float32,
    ):
        with T.Kernel(T.ceildiv(c, bc), T.ceildiv(tb, rb), threads=threads) as (bx, by):
            y_f = T.alloc_fragment((ra, bc), F32)
            b_f = T.alloc_fragment((ra, bc), F16)
            products(xv, w2, w1bd, bx, by, y_f)
            T.copy(basev[by * ra, bx * bc], b_f)
            for p, q in T.Parallel(ra, bc):
                y_f[p, q] = b_f[p, q] + scale * y_f[p, q]
            T.copy(y_f, yv[by * ra, bx * bc])

    @T.prim_func
    def plain(
        xv: T.Tensor((tb, d), F16),
        w2: T.Tensor((c, d), F16),
        w1bd: T.Tensor((ra, rb), F16),
        yv: T.Tensor((ta, c), F16),
        scale: T.float32,
    ):
        with T.Kernel(T.ceildiv(c, bc), T.ceildiv(tb, rb), threads=threads) as (bx, by):
            y_f = T.alloc_fragment((ra, bc), F32)
            products(xv, w2, w1bd, bx, by, y_f)
            for p, q in T.Parallel(ra, bc):
                y_f[p, q] = scale * y_f[p, q]
            T.copy(y_f, yv[by * ra, bx * bc])

    return with_base if has_base else plain


def _unit(a, b):
    """Fewest tokens per program making BT*a and BT*b multiples of 16."""
    ua = MIN_DIM // math.gcd(a, MIN_DIM)
    ub = MIN_DIM // math.gcd(b, MIN_DIM)
    return ua * ub // math.gcd(ua, ub)


def _smem(cfg, a, b):
    ra, rb = cfg["BT"] * a, cfg["BT"] * b
    return rb * cfg["BD"] + cfg["BC"] * cfg["BD"] + rb * cfg["BC"] + ra * rb


def _legal(dev, cfg, a, b):
    ra, rb = cfg["BT"] * a, cfg["BT"] * b
    return fits(dev, _smem(cfg, a, b)) and partitions(
        cfg["threads"], (rb, cfg["BC"]), (ra, cfg["BC"])
    )


def _plan_bypass(dev, t, a, b, c, d, has_base):
    """Legal tilings, model-ranked, then torch."""
    unit = _unit(a, b)
    if unit * max(a, b) > MAX_ROWS:
        return whole_ops(dev)
    space = []
    for mult in (1, 2, 4, 8, 16):
        bt = unit * mult
        if bt * max(a, b) > MAX_ROWS:
            break
        for bc in (32, 64, 128):
            for bd in (32, 64):
                for threads in (64, 128, 256):
                    cfg = {"BT": bt, "BC": bc, "BD": bd, "threads": threads}
                    if _legal(dev, cfg, a, b):
                        space.append(cfg)

    def work(cf):
        bt, bc, bd = cf["BT"], cf["BC"], cf["BD"]
        ra, rb = bt * a, bt * b
        tt, tc = math.ceil(t / bt), math.ceil(c / bc)
        # x is read once per column tile of w2; the block-diagonal w1 GEMM
        # costs BT x the dense w1 contraction.
        return Work(
            grid=tt * tc,
            warps=cf["threads"] // 32,
            acc_fp32=rb * bc + ra * bc,
            operand_bytes=0,
            smem_stage=2 * _smem(cf, a, b),
            stages=1,
            hbm_bytes=2 * (t * b * d * tc + t * a * c * (2 if has_base else 1))
            + shared_reads(dev, 2 * (c * d + ra * rb), tt * tc),
            mma_flops=2.0 * tt * tc * (rb * math.ceil(d / bd) * bd * bc + ra * rb * bc),
            # each program reads its x rows, its w2 rows and the block-diagonal w1
            l2_bytes=2
            * (
                t * b * d * tc
                + tt * tc * (bc * d + ra * rb)
                + t * a * c * (2 if has_base else 1)
            ),
        )

    return [*rank(dev, space, work), *whole_ops(dev)]


def _bypass_run(cfg, flat, w1, w2, scale, base2, out_dtype):
    """[base +] scale * vec(w1 @ X_t @ w2^T); flat (t, b*d), w1 (a, b), w2 (c, d), fp16."""
    if cfg == TORCH:
        return eager.lokr_bypass(flat, w1.contiguous(), w2.contiguous(), scale, base2)
    if cfg == TRITON:
        return triton_lokr.lokr_bypass(
            flat, w1.contiguous(), w2.contiguous(), scale, base2
        ).to(out_dtype)
    a, b = w1.shape
    c, d = w2.shape
    t = flat.shape[0]
    bt = cfg["BT"]
    w1bd = torch.block_diag(*([w1] * bt)).contiguous()
    fused = base2 is not None and base2.dtype == torch.float16
    kernel = compiled(
        _bypass_prim,
        flat.device,
        a,
        b,
        c,
        d,
        bt,
        cfg["BC"],
        cfg["BD"],
        cfg["threads"],
        fused,
    )
    xv = flat.contiguous().view(t * b, d)
    y = torch.empty(t * a, c, device=flat.device, dtype=torch.float16)
    if fused:
        kernel(xv, w2.contiguous(), w1bd, base2.contiguous().view(t * a, c), y, scale)
        y = y.view(t, a * c)
        return y if out_dtype == torch.float16 else y.to(out_dtype)
    kernel(xv, w2.contiguous(), w1bd, y, scale)
    y = y.view(t, a * c)
    return y.to(out_dtype) if base2 is None else (base2 + y).to(out_dtype)


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
        lambda: _plan_bypass(dev, t, a, b, c, d, has_base),
        measurer(
            dev,
            t * (b * d + a * c) * 2,
            inputs,
            lambda cf, s: lambda: _bypass_run(cf, s[0], w1, w2, scale, s[1], out_dtype),
        ),
    )


def lokr_bypass(x, w1, w2, scale=1.0, base=None):
    """[base +] scale * x @ kron(w1, w2)^T, without building the product."""
    dev = device_info(x.device)
    lead = x.shape[:-1]
    flat = x.reshape(-1, x.shape[-1]).to(torch.float16)
    f1, f2 = w1.to(torch.float16), w2.to(torch.float16)
    o = f1.shape[0] * f2.shape[0]
    base2 = base.reshape(-1, o) if base is not None else None
    out_dtype = x.dtype if base is None else base.dtype
    key, cfg = _bypass_config(
        dev, "tl_lokr_bypass", flat, f1, f2, scale, base is not None, out_dtype
    )
    out = guarded(
        dev, key, cfg, lambda c: _bypass_run(c, flat, f1, f2, scale, base2, out_dtype)
    )
    return out.reshape(*lead, o)


def lokr_bypass_backward(grad, x, w1, w2, scale=1.0):
    """(grad_x, grad_w1, grad_w2) for :func:`lokr_bypass`."""
    dev = device_info(x.device)
    f1, f2 = w1.to(torch.float16), w2.to(torch.float16)
    a, b = f1.shape
    c, d = f2.shape
    g2 = grad.reshape(-1, a * c).to(torch.float16)
    f1t, f2t = f1.t().contiguous(), f2.t().contiguous()
    key, cfg = _bypass_config(
        dev, "tl_lokr_bypass_dx", g2, f1t, f2t, scale, False, torch.float16
    )
    gx = guarded(
        dev,
        key,
        cfg,
        lambda c_: _bypass_run(c_, g2, f1t, f2t, scale, None, torch.float16),
    )
    x3 = x.reshape(-1, b, d).to(torch.float16)
    g3 = g2.reshape(-1, a, c)
    mid = x3 @ f2.t()
    n = torch.matmul(f1.t(), g3)
    gw1 = torch.einsum("tac,tbc->ab", g3, mid).mul_(scale)
    gw2 = torch.einsum("tbc,tbd->cd", n, x3).mul_(scale)
    return gx.reshape(x.shape).to(x.dtype), gw1.to(w1.dtype), gw2.to(w2.dtype)
