# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DoRA kernels out = B + m*(W*d/||N_row|| - B) on W as (rows, cols) around the norm axis, and IA3 x*(alpha+gamma*w[c])."""

import torch
import triton
import triton.language as tl

from comfy_kitchen.backends.eager.adapters import decompose as eager
from comfy_kitchen.backends.eager.adapters.common import absent

from .common import (
    copy_of,
    dtype_key,
    grid_configs,
    guarded,
    launch,
    measurer,
    token_bucket,
)
from .tuning import TORCH, device_info, select
from .tuning.model import Work, cdiv, rank


@triton.jit
def _offsets(rm, rn, s_row, k2, s1, s2):
    """Element offsets of (row, col) with col split as (col // k2, col % k2)."""
    return rm[:, None] * s_row + (rn // k2)[None, :] * s1 + (rn % k2)[None, :] * s2


@triton.jit
def _merged_tile(w_ptr, dl_ptr, off, m, has_delta: tl.constexpr):
    """(W tile, M = W + delta tile) in fp32."""
    w = tl.load(w_ptr + off, mask=m, other=0.0).to(tl.float32)
    mg = w
    if has_delta:
        mg = w + tl.load(dl_ptr + off, mask=m, other=0.0).to(tl.float32)
    return w, mg


@triton.jit
def _source_tile(
    role: tl.constexpr, ptr, off, m, w_ptr, dl_ptr, has_delta: tl.constexpr
):
    """The tile a base or norm role names: M (0), W (1), or the given tensor at ptr (2)."""
    if role == 2:
        t = tl.load(ptr + off, mask=m, other=0.0).to(tl.float32)
    else:
        w, mg = _merged_tile(w_ptr, dl_ptr, off, m, has_delta)
        t = mg if role == 0 else w
    return t


@triton.jit
def _decompose_fwd_kernel(
    w_ptr,
    dl_ptr,
    b_ptr,
    n_ptr,
    d_ptr,
    norm_ptr,
    y_ptr,
    n_rows,
    n_cols,
    s_row,
    k2,
    s1,
    s2,
    mult,
    eps,
    has_delta: tl.constexpr,
    base_role: tl.constexpr,
    norm_role: tl.constexpr,
    resident: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    """y = B + m * (M * d / n - B) with M = W + delta, n = ||N_row|| + eps, and the norms.

    Grid: (row tiles,); resident holds whole rows of W, M and N in registers.
    """
    pid = tl.program_id(0)
    rm = pid * block_m + tl.arange(0, block_m)
    mm = rm < n_rows

    # Pass 1: n per row, from the whole row in registers or streamed in column chunks.
    if resident:
        rn = tl.arange(0, block_n)
        m = mm[:, None] & (rn < n_cols)[None, :]
        off = _offsets(rm, rn, s_row, k2, s1, s2)
        w, mg = _merged_tile(w_ptr, dl_ptr, off, m, has_delta)
        if norm_role == 2:
            src = tl.load(n_ptr + off, mask=m, other=0.0).to(tl.float32)
        else:
            src = mg if norm_role == 0 else w
        n = tl.sqrt(tl.sum(src * src, axis=1)) + eps
    else:
        acc = tl.zeros((block_m,), tl.float32)
        for c0 in range(0, n_cols, block_n):
            rn = c0 + tl.arange(0, block_n)
            m = mm[:, None] & (rn < n_cols)[None, :]
            off = _offsets(rm, rn, s_row, k2, s1, s2)
            src = _source_tile(norm_role, n_ptr, off, m, w_ptr, dl_ptr, has_delta)
            acc += tl.sum(src * src, axis=1)
        n = tl.sqrt(acc) + eps

    tl.store(norm_ptr + rm, n, mask=mm)
    s = tl.load(d_ptr + rm, mask=mm, other=0.0).to(tl.float32) / n

    # Pass 2: y chunk = B + m * (M * s - B), W and M from registers when resident.
    for c0 in range(0, n_cols, block_n):
        rn = c0 + tl.arange(0, block_n)
        m = mm[:, None] & (rn < n_cols)[None, :]
        off = _offsets(rm, rn, s_row, k2, s1, s2)
        if not resident:
            w, mg = _merged_tile(w_ptr, dl_ptr, off, m, has_delta)
        if base_role == 2:
            b = tl.load(b_ptr + off, mask=m, other=0.0).to(tl.float32)
        else:
            b = mg if base_role == 0 else w

        y = b + mult * (mg * s[:, None] - b)
        tl.store(y_ptr + off, y.to(y_ptr.dtype.element_ty), mask=m)


@triton.jit
def _decompose_bwd_kernel(
    g_ptr,
    w_ptr,
    dl_ptr,
    n_ptr,
    d_ptr,
    norm_ptr,
    gw_ptr,
    gd_ptr,
    gb_ptr,
    gn_ptr,
    gdl_ptr,
    n_rows,
    n_cols,
    s_row,
    k2,
    s1,
    s2,
    mult,
    eps,
    has_delta: tl.constexpr,
    base_role: tl.constexpr,
    norm_role: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    """Gradients of y = B + m*(M*d/n - B), M = W + delta, for W, d, delta, and B and N when given.

    Grid: (row tiles,).
    """
    pid = tl.program_id(0)
    rm = pid * block_m + tl.arange(0, block_m)
    mm = rm < n_rows

    # Pass 1: rowdot = sum over the row of g * M.
    rowdot = tl.zeros((block_m,), tl.float32)
    for c0 in range(0, n_cols, block_n):
        rn = c0 + tl.arange(0, block_n)
        m = mm[:, None] & (rn < n_cols)[None, :]
        off = _offsets(rm, rn, s_row, k2, s1, s2)
        g = tl.load(g_ptr + off, mask=m, other=0.0).to(tl.float32)
        _, mg = _merged_tile(w_ptr, dl_ptr, off, m, has_delta)
        rowdot += tl.sum(g * mg, axis=1)

    # Per-row scalars: g_d, and dn = gradient into the norm source (0 for a zero-norm row).
    d = tl.load(d_ptr + rm, mask=mm, other=0.0).to(tl.float32)
    n = tl.load(norm_ptr + rm, mask=mm, other=1.0)
    s = d / n
    tl.store(gd_ptr + rm, (mult * rowdot / n).to(gd_ptr.dtype.element_ty), mask=mm)
    raw = n - eps
    dn = tl.where(raw > 0, -mult * rowdot * d / (n * n) / raw, 0.0)

    # Pass 2: per chunk, the gradients through M, B and N, routed by role.
    for c0 in range(0, n_cols, block_n):
        rn = c0 + tl.arange(0, block_n)
        m = mm[:, None] & (rn < n_cols)[None, :]
        off = _offsets(rm, rn, s_row, k2, s1, s2)
        g = tl.load(g_ptr + off, mask=m, other=0.0).to(tl.float32)
        gm = g * (mult * s)[:, None]
        gb = (1.0 - mult) * g
        src = _source_tile(norm_role, n_ptr, off, m, w_ptr, dl_ptr, has_delta)
        gn = dn[:, None] * src

        # M collects its own terms; W collects M's plus the roles that name W.
        if base_role == 0:
            gm += gb
        if norm_role == 0:
            gm += gn
        gw = gm
        if base_role == 1:
            gw += gb
        if norm_role == 1:
            gw += gn

        tl.store(gw_ptr + off, gw.to(gw_ptr.dtype.element_ty), mask=m)
        if has_delta:
            tl.store(gdl_ptr + off, gm.to(gdl_ptr.dtype.element_ty), mask=m)
        if base_role == 2:
            tl.store(gb_ptr + off, gb.to(gb_ptr.dtype.element_ty), mask=m)
        if norm_role == 2:
            tl.store(gn_ptr + off, gn.to(gn_ptr.dtype.element_ty), mask=m)


@triton.jit
def _channel_fwd_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    n_rows,
    n_cols,
    n_ch,
    alpha,
    gamma,
    ch_on_cols: tl.constexpr,
    block_r: tl.constexpr,
    block_q: tl.constexpr,
):
    """y = x * (alpha + gamma * w[ch]) over x viewed as (rows, cols): ch is the column
    (channels-last, cols = C) or row % C (channels-first, rows = outer * C, cols = inner).

    Grid: (row tiles, column tiles).
    """
    rr = tl.program_id(0) * block_r + tl.arange(0, block_r)
    qq = tl.program_id(1) * block_q + tl.arange(0, block_q)
    mr = rr < n_rows
    mq = qq < n_cols
    off = rr[:, None] * n_cols + qq[None, :]
    m = mr[:, None] & mq[None, :]

    # Scale per column or per row, loaded once per tile.
    if ch_on_cols:
        w = tl.load(w_ptr + qq, mask=mq, other=0.0).to(tl.float32)[None, :]
    else:
        w = tl.load(w_ptr + rr % n_ch, mask=mr, other=0.0).to(tl.float32)[:, None]

    x = tl.load(x_ptr + off, mask=m, other=0.0).to(tl.float32)
    tl.store(y_ptr + off, (x * (alpha + gamma * w)).to(y_ptr.dtype.element_ty), mask=m)


@triton.jit
def _channel_bwd_kernel(
    g_ptr,
    x_ptr,
    w_ptr,
    gx_ptr,
    part_ptr,
    n_ch,
    inner,
    n_cols,
    alpha,
    gamma,
    block_c: tl.constexpr,
    block_j: tl.constexpr,
):
    """gx, and the partial channel sums of g*x*gamma for one (channel tile, column slice).

    Grid: (channel tiles, column splits); column j is (outer j // inner, inner j % inner).
    """
    pid_c = tl.program_id(0)
    pid_s = tl.program_id(1)
    splits = tl.num_programs(1)
    rc = pid_c * block_c + tl.arange(0, block_c)
    mc = rc < n_ch
    w = tl.load(w_ptr + rc, mask=mc, other=0.0).to(tl.float32)

    # Columns [lo, hi) of this split; acc = per-channel sum of g * x.
    span = tl.cdiv(tl.cdiv(n_cols, block_j), splits) * block_j
    lo = pid_s * span
    hi = tl.minimum(lo + span, n_cols)
    acc = tl.zeros((block_c,), tl.float32)
    for j0 in range(lo, hi, block_j):
        rj = j0 + tl.arange(0, block_j)
        m = mc[:, None] & (rj < hi)[None, :]
        off = (
            (rj // inner)[None, :] * (n_ch * inner)
            + rc[:, None] * inner
            + (rj % inner)[None, :]
        )
        g = tl.load(g_ptr + off, mask=m, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + off, mask=m, other=0.0).to(tl.float32)
        tl.store(
            gx_ptr + off,
            (g * (alpha + gamma * w)[:, None]).to(gx_ptr.dtype.element_ty),
            mask=m,
        )
        acc += tl.sum(g * x, axis=1)
    tl.store(part_ptr + pid_s * n_ch + rc, acc * gamma, mask=mc)


def _axes(w: torch.Tensor, on_output: bool) -> tuple[int, int, int, int, int, int]:
    """(rows, cols, SR, K2, S1, S2) of a contiguous weight around its norm axis."""
    o, i = w.shape[0], w.shape[1]
    k = w.numel() // (o * i)
    if on_output:
        return o, i * k, i * k, 1, 1, 0
    return i, o * k, k, k, i * k, 1


def _role_bytes(role: int, merged: int, e_w: int, e_given: int) -> int:
    """Bytes per element of the tensor a base or norm role names."""
    return {eager.MERGED: merged, eager.WEIGHT: e_w, eager.GIVEN: e_given}[role]


def _plan_fwd(dev, rows, cols, sizes, base_role, norm_role):
    """Forward configs, model-ranked, then torch; sizes = bytes per element of (W, delta or 0, B, N)."""
    e_w, e_d, e_b, e_n = sizes
    merged = e_w + e_d
    once = merged + (e_b if base_role == eager.GIVEN else 0)
    once += e_n if norm_role == eager.GIVEN else 0
    reread = 0 if norm_role == eager.GIVEN else _role_bytes(norm_role, merged, e_w, e_n)
    resident = [
        {"RESIDENT": True, "BLOCK_M": m, "num_warps": w, "num_stages": 1}
        for m in (1, 2, 4, 8)
        for w in (2, 4, 8, 16)
        if m * triton.next_power_of_2(cols) <= 16384
    ]
    streamed = grid_configs(
        dev,
        RESIDENT=(False,),
        BLOCK_M=(1, 2, 4, 8),
        BLOCK_N=(512, 1024, 2048),
        num_warps=(4, 8),
        num_stages=(1,),
    )

    def work(c):
        bm = c["BLOCK_M"]
        tile = bm * (triton.next_power_of_2(cols) if c["RESIDENT"] else c["BLOCK_N"])
        reads = once + (0 if c["RESIDENT"] else reread)
        return Work(
            grid=cdiv(rows, bm),
            warps=c["num_warps"],
            acc_fp32=tile * (2 if e_d else 1),
            operand_bytes=tile * once,
            smem_stage=0,
            stages=1,
            hbm_bytes=rows * cols * (reads + e_w),
            vector_flops=6.0 * rows * cols,
        )

    return [*rank(dev, resident + streamed, work), TORCH]


def _plan_bwd(dev, rows, cols, sizes, base_role, norm_role):
    """Backward configs; sizes = bytes per element of (g, W, delta or 0, B, N)."""
    e_g, e_w, e_d, e_b, e_n = sizes
    merged = e_w + e_d
    reads = 2 * e_g + merged + _role_bytes(norm_role, merged, e_w, e_n)
    writes = e_w + e_d + (e_b if base_role == eager.GIVEN else 0)
    writes += e_n if norm_role == eager.GIVEN else 0
    space = grid_configs(
        dev,
        BLOCK_M=(1, 2, 4, 8),
        BLOCK_N=(512, 1024, 2048),
        num_warps=(4, 8),
        num_stages=(1,),
    )

    def work(c):
        bm, bn = c["BLOCK_M"], c["BLOCK_N"]
        return Work(
            grid=cdiv(rows, bm),
            warps=c["num_warps"],
            acc_fp32=bm * bn,
            operand_bytes=bm * bn * (e_g + merged + e_n),
            smem_stage=0,
            stages=1,
            hbm_bytes=rows * cols * (reads + writes),
            vector_flops=10.0 * rows * cols,
        )

    return [*rank(dev, space, work), TORCH]


def _plan_channel_fwd(dev, rows, cols, e):
    """IA3 forward (row tile, column tile, warps) configs over the (rows, cols) view, model-ranked, then torch."""
    qmax = triton.next_power_of_2(cols)
    space = [
        cf
        for cf in grid_configs(
            dev,
            BR=(1, 2, 4, 8, 16, 32),
            BQ=(64, 128, 256, 512, 1024, 2048, 4096),
            num_warps=(2, 4, 8),
            num_stages=(1,),
        )
        if 512 <= cf["BR"] * cf["BQ"] <= 8192 and cf["BQ"] <= max(64, qmax)
    ]
    n = rows * cols

    def work(cf):
        br, bq = cf["BR"], cf["BQ"]
        # masked lanes past the last column still occupy the tile
        used = min(bq, cols) / bq
        return Work(
            grid=cdiv(rows, br) * cdiv(cols, bq),
            warps=cf["num_warps"],
            acc_fp32=br * bq,
            operand_bytes=br * bq * e,
            smem_stage=0,
            stages=1,
            hbm_bytes=2 * n * e,
            vector_flops=2.0 * n / used,
            # one x tile loaded per program
            phases=((2 * n * e, br * bq * e * used),),
        )

    return [*rank(dev, space, work), TORCH]


def _plan_channel_bwd(dev, outer, c, inner, e):
    if inner == 1:
        space = grid_configs(
            dev,
            BLOCK_C=(32, 64, 128),
            BLOCK_J=(16, 32, 64),
            num_warps=(4, 8),
            num_stages=(1,),
        )
    else:
        space = grid_configs(
            dev,
            BLOCK_C=(1, 2, 4),
            BLOCK_J=(256, 512, 1024),
            num_warps=(4, 8),
            num_stages=(1,),
        )
    cols = outer * inner

    def work(cf):
        bc, bj = cf["BLOCK_C"], cf["BLOCK_J"]
        tiles = cdiv(c, bc)
        splits = max(1, min(cdiv(cols, bj), cdiv(4 * dev.units, tiles)))
        # Read amplification from tiles narrower than a 32-byte sector.
        sector = max(1.0, 32 / (bc * e)) if inner == 1 else max(1.0, 32 / (bj * e))
        return Work(
            grid=tiles * splits,
            warps=cf["num_warps"],
            acc_fp32=bc,
            operand_bytes=3 * bc * bj * e,
            smem_stage=0,
            stages=1,
            hbm_bytes=3 * c * cols * e * sector + 2 * splits * c * 4,
            vector_flops=3.0 * c * cols,
        )

    return [*rank(dev, space, work), TORCH]


def _contig(t: torch.Tensor | None, fallback: torch.Tensor) -> torch.Tensor:
    """t contiguous, or fallback as the pointer a kernel never reads."""
    return t.contiguous() if t is not None else fallback


def _sizes(*tensors: torch.Tensor | None) -> tuple[int, ...]:
    """Bytes per element of each tensor, 0 for an absent one."""
    return tuple(t.element_size() if t is not None else 0 for t in tensors)


def _fwd_run(
    cfg,
    weight,
    dora_scale,
    multiplier,
    on_output,
    base,
    norm_weight,
    delta,
    b_role,
    n_role,
):
    if cfg == TORCH:
        return eager.weight_decompose(
            weight,
            dora_scale,
            multiplier,
            on_output,
            base,
            norm_weight,
            delta,
            b_role,
            n_role,
        )
    w = weight.contiguous()
    rows, cols, sr, k2, s1, s2 = _axes(w, on_output)
    out = torch.empty_like(w)
    norms = torch.empty(rows, device=w.device, dtype=torch.float32)
    block_n = triton.next_power_of_2(cols) if cfg["RESIDENT"] else cfg["BLOCK_N"]
    _decompose_fwd_kernel[(triton.cdiv(rows, cfg["BLOCK_M"]),)](
        w,
        _contig(delta, w),
        _contig(base, w),
        _contig(norm_weight, w),
        dora_scale.reshape(-1),
        norms,
        out,
        rows,
        cols,
        sr,
        k2,
        s1,
        s2,
        multiplier,
        torch.finfo(weight.dtype).eps,
        has_delta=delta is not None,
        base_role=b_role,
        norm_role=n_role,
        resident=cfg["RESIDENT"],
        block_m=cfg["BLOCK_M"],
        block_n=block_n,
        **launch(cfg),
    )
    return out, norms


def weight_decompose(
    weight: torch.Tensor,
    dora_scale: torch.Tensor,
    multiplier: float = 1.0,
    on_output: bool = True,
    base: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    delta: torch.Tensor | None = None,
    base_role: int = eager.MERGED,
    norm_role: int = eager.MERGED,
) -> tuple[torch.Tensor, torch.Tensor]:
    """-> (B + m * (M * d / n - B) in W's dtype, the fp32 norms n); M = W + delta."""
    dev = device_info(weight.device)
    rows, cols, *_ = _axes(weight, on_output)
    sizes = _sizes(weight, delta, base, norm_weight)
    key = (
        "weight_decompose",
        (dtype_key(weight.dtype), on_output, sizes[1], base_role, norm_role),
        (rows, cols),
    )
    args = (
        dora_scale,
        multiplier,
        on_output,
        base,
        norm_weight,
        delta,
        base_role,
        norm_role,
    )
    cfg = select(
        dev,
        key,
        lambda: _plan_fwd(dev, rows, cols, sizes, base_role, norm_role),
        measurer(
            dev,
            2 * weight.numel() * weight.element_size(),
            lambda: (copy_of(weight),),
            lambda c, s: lambda: _fwd_run(c, s[0], *args),
        ),
    )
    return guarded(dev, key, cfg, lambda c: _fwd_run(c, weight, *args))


def _bwd_run(
    cfg,
    grad,
    weight,
    dora_scale,
    norms,
    multiplier,
    on_output,
    base,
    norm_weight,
    delta,
    b_role,
    n_role,
):
    if cfg == TORCH:
        return eager.weight_decompose_backward(
            grad,
            weight,
            dora_scale,
            norms,
            multiplier,
            on_output,
            base,
            norm_weight,
            delta,
            b_role,
            n_role,
        )
    w = weight.contiguous()
    g = grad.reshape(w.shape).contiguous()
    rows, cols, sr, k2, s1, s2 = _axes(w, on_output)
    gw = torch.empty_like(w)
    gd = torch.empty(rows, device=w.device, dtype=dora_scale.dtype)
    # The kernel writes contiguous offsets: a channels_last input still gets a contiguous grad.
    cf = torch.contiguous_format
    gb = (
        torch.empty_like(base, memory_format=cf) if b_role == eager.GIVEN else absent(w)
    )
    gn = (
        torch.empty_like(norm_weight, memory_format=cf)
        if n_role == eager.GIVEN
        else absent(w)
    )
    gdl = torch.empty_like(delta, memory_format=cf) if delta is not None else absent(w)
    _decompose_bwd_kernel[(triton.cdiv(rows, cfg["BLOCK_M"]),)](
        g,
        w,
        _contig(delta, w),
        _contig(norm_weight, w),
        dora_scale.reshape(-1),
        norms,
        gw,
        gd,
        gb if b_role == eager.GIVEN else gw,
        gn if n_role == eager.GIVEN else gw,
        gdl if delta is not None else gw,
        rows,
        cols,
        sr,
        k2,
        s1,
        s2,
        multiplier,
        torch.finfo(weight.dtype).eps,
        has_delta=delta is not None,
        base_role=b_role,
        norm_role=n_role,
        block_m=cfg["BLOCK_M"],
        block_n=cfg["BLOCK_N"],
        **launch(cfg),
    )
    return gw, gd.reshape(dora_scale.shape), gb, gn, gdl


def weight_decompose_backward(
    grad: torch.Tensor,
    weight: torch.Tensor,
    dora_scale: torch.Tensor,
    norms: torch.Tensor,
    multiplier: float = 1.0,
    on_output: bool = True,
    base: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    delta: torch.Tensor | None = None,
    base_role: int = eager.MERGED,
    norm_role: int = eager.MERGED,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(g_weight, g_dora_scale, g_base, g_norm_weight, g_delta); absent inputs get empty grads."""
    dev = device_info(weight.device)
    rows, cols, *_ = _axes(weight, on_output)
    sizes = _sizes(grad, weight, delta, base, norm_weight)
    key = (
        "weight_decompose_backward",
        (dtype_key(weight.dtype), on_output, sizes[2], base_role, norm_role),
        (rows, cols),
    )
    args = (
        dora_scale,
        norms,
        multiplier,
        on_output,
        base,
        norm_weight,
        delta,
        base_role,
        norm_role,
    )
    cfg = select(
        dev,
        key,
        lambda: _plan_bwd(dev, rows, cols, sizes, base_role, norm_role),
        measurer(
            dev,
            3 * weight.numel() * weight.element_size(),
            lambda: (copy_of(grad.reshape(weight.shape)), copy_of(weight)),
            lambda c, s: lambda: _bwd_run(c, s[0], s[1], *args),
        ),
    )
    return guarded(dev, key, cfg, lambda c: _bwd_run(c, grad, weight, *args))


def _view3(x: torch.Tensor, channel_axis: int) -> tuple[torch.Tensor, int, int, int]:
    """(contiguous x, outer, C, inner) around the channel axis."""
    axis = channel_axis % x.dim()
    c = x.shape[axis]
    inner = 1
    for size in x.shape[axis + 1 :]:
        inner *= size
    return x.contiguous(), x.numel() // (c * inner), c, inner


def channel_scale(
    x: torch.Tensor,
    w: torch.Tensor,
    channel_axis: int = -1,
    alpha: float = 1.0,
    gamma: float = 1.0,
) -> torch.Tensor:
    """x * (alpha + gamma * w[c]); alpha=1 is the merged form, 0 the delta."""
    dev = device_info(x.device)
    xc, outer, c, inner = _view3(x, channel_axis)
    wv = w.reshape(-1).contiguous()
    # (rows, cols) view: channels-last puts C on the columns, channels-first on the rows.
    rows, cols = (outer, c) if inner == 1 else (outer * c, inner)
    key = (
        "channel_scale",
        (dtype_key(x.dtype), inner == 1),
        (token_bucket(rows), cols),
    )

    def run(cfg, xx):
        if cfg == TORCH:
            return eager.channel_scale(xx.view(x.shape), w, channel_axis, alpha, gamma)
        y = torch.empty_like(xx)
        grid = (triton.cdiv(rows, cfg["BR"]), triton.cdiv(cols, cfg["BQ"]))
        _channel_fwd_kernel[grid](
            xx,
            wv,
            y,
            rows,
            cols,
            c,
            alpha,
            gamma,
            ch_on_cols=inner == 1,
            block_r=cfg["BR"],
            block_q=cfg["BQ"],
            **launch(cfg),
        )
        return y.view(x.shape)

    cfg = select(
        dev,
        key,
        lambda: _plan_channel_fwd(dev, rows, cols, x.element_size()),
        measurer(
            dev,
            2 * x.numel() * x.element_size(),
            lambda: (copy_of(xc),),
            lambda cf, s: lambda: run(cf, s[0]),
        ),
    )
    return guarded(dev, key, cfg, lambda cf: run(cf, xc))


def _channel_bwd_run(cfg, dev, g, xc, w, outer, c, inner, alpha, gamma):
    if cfg == TORCH:
        return None
    gx = torch.empty_like(xc)
    cols = outer * inner
    tiles = triton.cdiv(c, cfg["BLOCK_C"])
    splits = max(
        1, min(triton.cdiv(cols, cfg["BLOCK_J"]), triton.cdiv(4 * dev.units, tiles))
    )
    part = torch.empty(splits, c, device=xc.device, dtype=torch.float32)
    _channel_bwd_kernel[(tiles, splits)](
        g,
        xc,
        w,
        gx,
        part,
        c,
        inner,
        cols,
        alpha,
        gamma,
        block_c=cfg["BLOCK_C"],
        block_j=cfg["BLOCK_J"],
        **launch(cfg),
    )
    return gx, part.sum(0)


def channel_scale_backward(
    grad: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    channel_axis: int = -1,
    alpha: float = 1.0,
    gamma: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(grad_x, grad_w) for :func:`channel_scale`."""
    dev = device_info(x.device)
    xc, outer, c, inner = _view3(x, channel_axis)
    g = grad.reshape(x.shape).contiguous()
    wv = w.reshape(-1).contiguous()
    key = (
        "channel_scale_backward",
        (dtype_key(x.dtype), inner == 1),
        (token_bucket(outer * inner), c),
    )

    def run(cfg, gg, xx):
        out = _channel_bwd_run(cfg, dev, gg, xx, wv, outer, c, inner, alpha, gamma)
        if out is None:
            return eager.channel_scale_backward(
                gg.view(x.shape), xx.view(x.shape), w, channel_axis, alpha, gamma
            )
        return out

    cfg = select(
        dev,
        key,
        lambda: _plan_channel_bwd(dev, outer, c, inner, x.element_size()),
        measurer(
            dev,
            3 * x.numel() * x.element_size(),
            lambda: (copy_of(g), copy_of(xc)),
            lambda cf, s: lambda: run(cf, s[0], s[1]),
        ),
    )
    gx, gw = guarded(dev, key, cfg, lambda cf: run(cf, g, xc))
    return gx.reshape(x.shape), gw.reshape(w.shape).to(w.dtype)
