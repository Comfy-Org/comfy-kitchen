# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OFT apply in TileLang: out = rescale * R'^T x [- x] along a channel axis.

The folded rotation Rf (k, s, s) comes from the triton backend's in-kernel
Gauss-Jordan (it compiles for sm_70/sm_75: its one dot is an exact fp32
product) and is rounded to fp16, the precision ComfyUI applies it in. A channel tile of
BCH = g*s channels holds g whole blocks, g dividing k; its Rf blocks form one
(BCH, BCH) block-diagonal tile, so the apply is one GEMM per tile:
channels-last x (rows, C) gives out = X @ Rbd^T over (row tile, channel
tile), channels-first x (outer, C, inner) gives out_o = Rbd @ X_o over
(channel tile, column tile, outer).
"""

import math

import tilelang.language as T  # noqa: N812 -- TileLang's own spelling
import torch

from comfy_kitchen.backends.triton.adapters import oft as triton_oft
from comfy_kitchen.backends.triton.adapters.common import (
    copy_of,
    dtype_key,
    token_bucket,
)
from comfy_kitchen.backends.triton.adapters.tuning import TORCH, device_info, select
from comfy_kitchen.backends.triton.adapters.tuning.model import Work, rank

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
MAX_TILE = 128


def _rows_prim(c, bch, bt, threads):
    """Channels-last out tile = x tile @ Rbd; grid (channel tiles ct, row tiles rt)."""
    rows = T.symbolic("rows")

    @T.prim_func
    def main(
        x: T.Tensor((rows, c), F16),
        tiles: T.Tensor((c, bch), F16),
        out: T.Tensor((rows, c), F16),
    ):
        with T.Kernel(c // bch, T.ceildiv(rows, bt), threads=threads) as (ct, rt):
            x_s = T.alloc_shared((bt, bch), F16)
            w_s = T.alloc_shared((bch, bch), F16)
            acc = T.alloc_fragment((bt, bch), F32)
            T.copy(tiles[ct * bch, 0], w_s)
            T.copy(x[rt * bt, ct * bch], x_s)
            T.clear(acc)
            T.gemm(x_s, w_s, acc)
            T.copy(acc, out[rt * bt, ct * bch])

    return main


def _cols_prim(c, bch, bn, threads):
    """Channels-first out tile = Rbd @ x tile; grid (column tiles nt, channel tiles ct, outer o)."""
    outer_c = T.symbolic("outer_c")
    inner = T.symbolic("inner")

    @T.prim_func
    def main(
        x: T.Tensor((outer_c, inner), F16),
        tiles: T.Tensor((c, bch), F16),
        out: T.Tensor((outer_c, inner), F16),
    ):
        with T.Kernel(
            T.ceildiv(inner, bn), c // bch, outer_c // c, threads=threads
        ) as (nt, ct, o):
            x_s = T.alloc_shared((bch, bn), F16)
            w_s = T.alloc_shared((bch, bch), F16)
            acc = T.alloc_fragment((bch, bn), F32)
            T.copy(tiles[ct * bch, 0], w_s)
            T.copy(x[o * c + ct * bch, nt * bn], x_s)
            T.clear(acc)
            T.gemm(w_s, x_s, acc)
            T.copy(acc, out[o * c + ct * bch, nt * bn])

    return main


def _channels(x, channel_axis):
    """(C, inner): the channel count and the product of the dims after it."""
    axis = channel_axis % x.dim()
    inner = 1
    for size in x.shape[axis + 1 :]:
        inner *= size
    return x.shape[axis], inner


def _rotation(blocks, rescale, constraint, multiplier, shift):
    """Rf = rescale * R'^T - shift*I in fp32, (k, PS, PS) padded, from triton's tuned rotation."""
    dev = device_info(blocks.device)
    rf, *_ = triton_oft._rotations(
        dev, blocks, rescale, constraint, multiplier, shift, False
    )
    return rf


def _tiles(rf, g, transpose):
    """(k*s, g*s) fp16: per channel tile the block-diagonal of its g blocks."""
    k, s, _ = rf.shape
    blocks = rf.transpose(-1, -2) if transpose else rf
    grouped = blocks.reshape(k // g, g, s, s)
    eye = torch.eye(g, device=rf.device, dtype=rf.dtype)
    bd = grouped[:, :, :, None, :] * eye[None, :, None, :, None]
    return bd.reshape(k * s, g * s).to(torch.float16).contiguous()


def _groups(k, s):
    """Blocks per channel tile: divisors g of k with g*s a 16-multiple up to MAX_TILE."""
    return [
        g
        for g in range(1, k + 1)
        if k % g == 0 and (g * s) % MIN_DIM == 0 and g * s <= MAX_TILE
    ]


def _plan(dev, k, s, outer, inner):
    """Legal channel tilings, model-ranked, then torch."""
    space = []
    for g in _groups(k, s):
        bch = g * s
        for tile in (32, 64, 128):
            if not fits(dev, bch * tile + bch * bch):
                continue
            for th in (64, 128, 256):
                if partitions(th, (tile, bch) if inner == 1 else (bch, tile)):
                    space.append(
                        {"G": g, "BT": tile, "threads": th}
                        if inner == 1
                        else {"G": g, "BN": tile, "threads": th}
                    )
    c = k * s
    rows = outer * inner

    def work(cf):
        bch = cf["G"] * s
        tile = cf.get("BT", cf.get("BN"))
        # Rows layout: a program per (channel tile, row tile); cols layout
        # per (channel tile, column tile, outer). The block-diagonal tile
        # does bch MACs per element where s are needed.
        grid = (c // bch) * (
            math.ceil(rows / tile) if inner == 1 else math.ceil(inner / tile) * outer
        )
        return Work(
            grid=grid,
            warps=cf["threads"] // 32,
            acc_fp32=bch * tile,
            operand_bytes=0,
            smem_stage=2 * (bch * tile + bch * bch),
            stages=1,
            hbm_bytes=2 * 2 * rows * c + grid * 2 * bch * bch,
            mma_flops=2.0 * rows * c * bch,
        )

    return [*rank(dev, space, work), *whole_ops(dev)]


def _run(cfg, blocks, x, rescale, constraint, multiplier, shift, channel_axis, rf):
    """The apply under cfg with the folded rotation rf (k, PS, PS) built once by the caller."""
    if cfg == TORCH:
        return triton_oft.torch_apply(rf, x, channel_axis)
    if cfg == TRITON:
        dev = device_info(x.device)
        return triton_oft.apply_rotation(
            dev, blocks, x, rescale, constraint, multiplier, shift, channel_axis, rf
        )
    s = blocks.shape[1]
    rf = rf[:, :s, :s]
    c, inner = _channels(x, channel_axis)
    g = cfg["G"]
    xc = x.to(torch.float16).contiguous()
    if inner == 1:
        tiles = _tiles(rf, g, transpose=True)
        x2 = xc.view(-1, c)
        out = torch.empty_like(x2)
        kernel = compiled(_rows_prim, x.device, c, g * s, cfg["BT"], cfg["threads"])
        kernel(x2, tiles, out)
    else:
        tiles = _tiles(rf, g, transpose=False)
        x2 = xc.view(-1, inner)
        out = torch.empty_like(x2)
        kernel = compiled(_cols_prim, x.device, c, g * s, cfg["BN"], cfg["threads"])
        kernel(x2, tiles, out)
    return out.view(x.shape).to(x.dtype)


def oft_apply(
    blocks,
    x,
    rescale=None,
    constraint=0.0,
    multiplier=1.0,
    shift=False,
    channel_axis=-1,
):
    """rescale * R'^T x [- x] along channel_axis."""
    dev = device_info(x.device)
    k, s, _ = blocks.shape
    _, inner = _channels(x, channel_axis)
    rows = x.numel() // (k * s)
    rf = _rotation(blocks, rescale, constraint, multiplier, shift)
    args = (rescale, constraint, multiplier, shift, channel_axis, rf)
    key = (
        "tl_oft_apply",
        (dtype_key(x.dtype), rescale is not None, shift, inner == 1),
        (token_bucket(rows), k, s),
    )
    cfg = select(
        dev,
        key,
        lambda: _plan(dev, k, s, rows // inner, inner),
        measurer(
            dev,
            2 * x.numel() * x.element_size(),
            lambda: (copy_of(x),),
            lambda cf, st: lambda: _run(cf, blocks, st[0], *args),
        ),
    )
    return guarded(dev, key, cfg, lambda cf: _run(cf, blocks, x, *args))
