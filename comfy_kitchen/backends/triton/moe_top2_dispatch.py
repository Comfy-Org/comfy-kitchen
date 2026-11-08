# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton top-2 MoE routing + expert-major dispatch, 4 experts, two kernels.

route   one program per block of token rows: softmax over the 4 logits in fp32, in the
        order of the PyTorch softmax kernel for a last dim of 4 (max, exp(x - max),
        (e0 + e2) + (e1 + e3), IEEE division), rounded to the logits dtype; top 2 of the
        rounded scores, highest first, the smaller expert index first on ties (the order
        torch.topk(k=2, sorted=False) returns), NaN above every number as in torch.topk; the
        two choices always differ. A row with a NaN, a +inf or only -inf logits has 4 NaN
        scores, as in the PyTorch softmax, and gets experts 0 and 1 with NaN weights;
        per-block expert counts, each (token, k) counted once.
gather  one program per block of token rows: the expert bases from the counts of all
        blocks, the slot of every (token, k) inside its expert in token order (the row order
        of x[flat_topk_idx == i]), each x row read once and written to its two slots;
        program 0 writes the expert totals.

Launch configuration (same on CUDA and HIP, measured on an RTX 5090 and a Radeon 8060S):
32 token rows per program, 256 columns per step, 4 warps.
"""

import torch

import triton
import triton.language as tl
from triton.language.extra import libdevice

NUM_EXPERTS = 4
_BLOCK_T = 32  # token rows per program
_BLOCK_D = 256  # columns per copy step
_NUM_WARPS = 4


@triton.jit
def _ranks_above(a, b):
    # torch.topk order: NaN ranks above every number, two NaNs tie
    return (a > b) | ((a != a) & (b == b))


@triton.jit
def _moe_top2_route_kernel(
    logits_ptr,
    weight_ptr,
    idx_ptr,
    block_counts_ptr,
    n_tokens,
    stride_l,
    block_t: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * block_t + tl.arange(0, block_t)
    rm = rows < n_tokens
    base = logits_ptr + rows.to(tl.int64) * stride_l
    l0 = tl.load(base + 0, mask=rm, other=0.0).to(tl.float32)
    l1 = tl.load(base + 1, mask=rm, other=0.0).to(tl.float32)
    l2 = tl.load(base + 2, mask=rm, other=0.0).to(tl.float32)
    l3 = tl.load(base + 3, mask=rm, other=0.0).to(tl.float32)
    m = tl.maximum(tl.maximum(l0, l2), tl.maximum(l1, l3))
    e0 = libdevice.exp(l0 - m)
    e1 = libdevice.exp(l1 - m)
    e2 = libdevice.exp(l2 - m)
    e3 = libdevice.exp(l3 - m)
    s = (e0 + e2) + (e1 + e3)
    out_ty = weight_ptr.dtype.element_ty
    q0 = tl.div_rn(e0, s).to(out_ty).to(tl.float32)
    q1 = tl.div_rn(e1, s).to(out_ty).to(tl.float32)
    q2 = tl.div_rn(e2, s).to(out_ty).to(tl.float32)
    q3 = tl.div_rn(e3, s).to(out_ty).to(tl.float32)
    # first choice: a strict comparison keeps the smaller index on ties
    b1 = q0
    i1 = tl.zeros([block_t], dtype=tl.int32)
    c = _ranks_above(q1, b1)
    b1 = tl.where(c, q1, b1)
    i1 = tl.where(c, 1, i1)
    c = _ranks_above(q2, b1)
    b1 = tl.where(c, q2, b1)
    i1 = tl.where(c, 2, i1)
    c = _ranks_above(q3, b1)
    b1 = tl.where(c, q3, b1)
    i1 = tl.where(c, 3, i1)
    # second choice among the other three, same tie rule: start from the smallest index that
    # is not the first choice and never take the first choice, so the pair is always distinct
    i2 = tl.where(i1 == 0, 1, 0)
    b2 = tl.where(i1 == 0, q1, q0)
    c = (i1 != 1) & _ranks_above(q1, b2)
    b2 = tl.where(c, q1, b2)
    i2 = tl.where(c, 1, i2)
    c = (i1 != 2) & _ranks_above(q2, b2)
    b2 = tl.where(c, q2, b2)
    i2 = tl.where(c, 2, i2)
    c = (i1 != 3) & _ranks_above(q3, b2)
    b2 = tl.where(c, q3, b2)
    i2 = tl.where(c, 3, i2)
    tl.store(weight_ptr + rows * 2, b1.to(out_ty), mask=rm)
    tl.store(weight_ptr + rows * 2 + 1, b2.to(out_ty), mask=rm)
    tl.store(idx_ptr + rows * 2, i1.to(tl.int64), mask=rm)
    tl.store(idx_ptr + rows * 2 + 1, i2.to(tl.int64), mask=rm)
    ex = tl.arange(0, 4)
    # count each (token, k) separately, as the gather kernel does
    hit = (i1[:, None] == ex[None, :]).to(tl.int32) + (i2[:, None] == ex[None, :]).to(tl.int32)
    hit = tl.where(rm[:, None], hit, 0)
    tl.store(block_counts_ptr + pid * 4 + ex, tl.sum(hit, axis=0))


@triton.jit
def _moe_top2_gather_kernel(
    x_ptr,
    idx_ptr,
    block_counts_ptr,
    out_ptr,
    slot_ptr,
    counts_ptr,
    n_tokens,
    dim,
    n_blocks,
    stride_x,
    stride_o,
    block_t: tl.constexpr,
    n_blocks_p2: tl.constexpr,
    block_d: tl.constexpr,
):
    pid = tl.program_id(0)
    ex = tl.arange(0, 4)
    # expert bases from the per-block counts of every block
    bi = tl.arange(0, n_blocks_p2)
    allc = tl.load(
        block_counts_ptr + bi[:, None] * 4 + ex[None, :], mask=(bi < n_blocks)[:, None], other=0
    )
    total = tl.sum(allc, axis=0)
    before = tl.sum(tl.where((bi < pid)[:, None], allc, 0), axis=0)
    base = tl.cumsum(total, axis=0) - total + before  # [4]
    if pid == 0:
        tl.store(counts_ptr + ex, total.to(tl.int64))
    rows = pid * block_t + tl.arange(0, block_t)
    rm = rows < n_tokens
    i0 = tl.load(idx_ptr + rows * 2, mask=rm, other=-1).to(tl.int32)
    i1 = tl.load(idx_ptr + rows * 2 + 1, mask=rm, other=-1).to(tl.int32)
    oh0 = (i0[:, None] == ex[None, :]).to(tl.int32)
    oh1 = (i1[:, None] == ex[None, :]).to(tl.int32)
    hits = oh0 + oh1
    excl = tl.cumsum(hits, axis=0) - hits
    s0 = tl.sum(oh0 * (excl + base[None, :]), axis=1).to(tl.int64)
    s1 = tl.sum(oh1 * (excl + base[None, :]), axis=1).to(tl.int64)
    tl.store(slot_ptr + rows * 2, s0, mask=rm)
    tl.store(slot_ptr + rows * 2 + 1, s1, mask=rm)
    cols = tl.arange(0, block_d)
    src = x_ptr + rows.to(tl.int64)[:, None] * stride_x
    dst0 = out_ptr + s0[:, None] * stride_o
    dst1 = out_ptr + s1[:, None] * stride_o
    for d in range(0, dim, block_d):
        cc = d + cols
        msk = rm[:, None] & (cc < dim)[None, :]
        v = tl.load(src + cc[None, :], mask=msk)
        tl.store(dst0 + cc[None, :], v, mask=msk)
        tl.store(dst1 + cc[None, :], v, mask=msk)


def moe_top2_dispatch(
    x: torch.Tensor, logits: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n, dim = x.shape
    n_blocks = triton.cdiv(n, _BLOCK_T)
    dev = x.device
    weight = torch.empty((n, 2), dtype=logits.dtype, device=dev)
    idx = torch.empty((n, 2), dtype=torch.int64, device=dev)
    block_counts = torch.empty((n_blocks, NUM_EXPERTS), dtype=torch.int32, device=dev)
    _moe_top2_route_kernel[(n_blocks,)](
        logits,
        weight,
        idx,
        block_counts,
        n,
        logits.stride(0),
        block_t=_BLOCK_T,
        num_warps=_NUM_WARPS,
    )
    x_sorted = torch.empty((2 * n, dim), dtype=x.dtype, device=dev)
    slot = torch.empty((2 * n,), dtype=torch.int64, device=dev)
    counts = torch.empty((NUM_EXPERTS,), dtype=torch.int64, device=dev)
    _moe_top2_gather_kernel[(n_blocks,)](
        x,
        idx,
        block_counts,
        x_sorted,
        slot,
        counts,
        n,
        dim,
        n_blocks,
        x.stride(0),
        x_sorted.stride(0),
        block_t=_BLOCK_T,
        n_blocks_p2=triton.next_power_of_2(n_blocks),
        block_d=_BLOCK_D,
        num_warps=_NUM_WARPS,
    )
    return weight, idx, x_sorted, slot, counts
