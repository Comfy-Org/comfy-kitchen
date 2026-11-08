# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton per-group channel sums with one atomic add per run of equal indices.

``index_add_`` adds every pixel to its group's row with an atomic, one per (pixel, channel).
In the VOID warped noise about half of the pixels of a frame share one index (the zero-noise
pixels the warp starts from), so millions of atomics land on the same C addresses. Here each
program takes ``block_p`` consecutive pixels (raster order) with all channels, marks where the
index changes, sums each run of equal consecutive indices with a segmented inclusive scan and
adds the run's sum at its last pixel: one atomic per (run, channel). A pixel whose neighbours
have other indices still costs one atomic per channel, as before.

The partial sums of a group spread over several runs (several programs, or pixels that are not
adjacent) are added in a varying order, so those rows can differ in the last bits from call to
call, as they do with ``index_add_``; a group held by one run is one atomic onto zero. The
image is read as ``[C, H*W]`` (contiguous), the index as ``H*W`` int64. Launch configuration:
1024 pixels per program, 8 warps, on CUDA and HIP; ``max().item()`` and ``torch.zeros`` as in
the eager path.
"""

import torch

import triton
import triton.language as tl

_BLOCK_P = 1024
_NUM_WARPS = 8


@triton.jit
def _seg_combine(f1, v1, f2, v2):
    # (flag, value) segmented sum: flag = 1 at a run start
    return f1 | f2, tl.where(f2 != 0, v2, v1 + v2)


@triton.jit
def _sum_indexed_values_kernel(
    img_ptr,
    idx_ptr,
    out_ptr,
    n,
    channels,
    block_p: tl.constexpr,
    block_c: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_p = pid * block_p + tl.arange(0, block_p)
    offs_c = tl.arange(0, block_c)
    pm = offs_p < n
    cm = offs_c < channels
    idx = tl.load(idx_ptr + offs_p, mask=pm, other=-1)
    prev = tl.load(idx_ptr + offs_p - 1, mask=pm & (offs_p > 0), other=-2)
    nxt = tl.load(idx_ptr + offs_p + 1, mask=(offs_p + 1) < n, other=-3)
    first = tl.arange(0, block_p) == 0
    last = tl.arange(0, block_p) == block_p - 1
    start = (idx != prev) | first
    end = ((idx != nxt) | last) & pm
    # image is [C, N] contiguous: value(p, c) = img[c * N + p]
    m2 = pm[:, None] & cm[None, :]
    v = tl.load(img_ptr + offs_c[None, :].to(tl.int64) * n + offs_p[:, None], mask=m2, other=0.0)
    f = tl.broadcast_to(start.to(tl.int32)[:, None], (block_p, block_c))
    _, s = tl.associative_scan((f, v), 0, _seg_combine)
    dst = out_ptr + idx[:, None] * channels + offs_c[None, :]
    tl.atomic_add(dst, s, mask=end[:, None] & cm[None, :], sem="relaxed")


def sum_indexed_values(image: torch.Tensor, index_matrix: torch.Tensor) -> torch.Tensor:
    channels, h, w = image.shape
    n = h * w
    groups = int(index_matrix.max().item()) + 1
    out = torch.zeros((groups, channels), dtype=image.dtype, device=image.device)
    _sum_indexed_values_kernel[(triton.cdiv(n, _BLOCK_P),)](
        image,
        index_matrix,
        out,
        n,
        channels,
        block_p=_BLOCK_P,
        block_c=triton.next_power_of_2(channels),
        num_warps=_NUM_WARPS,
    )
    return out
