# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton patch-to-image fold + RMSNorm over channels, one pass per tile.

Each program loads ``block_p`` (row, pixel) pairs of one tile, all C channels contiguous,
computes the RMSNorm in fp32 and stores the row of C channels at its image position

    row R, pixel q   -> b = R // L, patch l = R % L, (lh, lw) = divmod(l, W // P),
                        (py, px) = divmod(q, P)
    output offset    = ((b * H + lh * P + py) * W + lw * P + px) * C

of a ``[B, H, W, C]`` buffer, returned as the ``[B, C, H, W]`` view ``permute(0, 3, 1, 2)``:
the shape and strides the stock ``fold`` -> ``movedim`` -> ``rms_norm`` -> ``movedim`` path
gives the conv. The tiles are read in place (one launch per tile), so the concatenated tiles,
the transposed copies and the folded image are never written out.

The sum of squares follows the order of PyTorch's CUDA RMSNorm for C = 64 (each of 16 lanes
accumulates 4 consecutive channels with fma, then a halving tree over the lanes), then
``rsqrt(sum / C + eps)`` and ``weight * (rstd * x)``, rounded once to the tiles' dtype at the
store. Launch configuration: 128 (row, pixel) pairs per program, 8 warps, on CUDA and HIP.
"""

import torch

import triton
import triton.language as tl

_BLOCK_P = 128
_NUM_WARPS = 8


@triton.jit
def _nerf_fold_rms_norm_kernel(
    src,
    w_ptr,
    out,
    row0,
    n_q,
    num_patches,
    patches_w,
    height,
    width,
    eps,
    patch: tl.constexpr,
    pp: tl.constexpr,
    channels: tl.constexpr,
    block_p: tl.constexpr,
):
    pid = tl.program_id(0)
    q = pid * block_p + tl.arange(0, block_p)  # flat (row-in-tile, pixel) index
    mask = q < n_q
    c = tl.arange(0, channels)
    x = tl.load(src + q[:, None] * channels + c[None, :], mask=mask[:, None], other=0.0)
    x = x.to(tl.float32)
    # sum of squares in the CUDA rms_norm order (vec_size 4): "thread" t owns channels
    # 4t..4t+3, sequential fma over them, then a halving tree over t
    nt: tl.constexpr = channels // 4
    x3 = tl.reshape(x, [block_p, nt, 4])
    vi = tl.arange(0, 4)[None, None, :]
    x0 = tl.sum(tl.where(vi == 0, x3, 0.0), axis=2)
    x1 = tl.sum(tl.where(vi == 1, x3, 0.0), axis=2)
    x2 = tl.sum(tl.where(vi == 2, x3, 0.0), axis=2)
    x4 = tl.sum(tl.where(vi == 3, x3, 0.0), axis=2)
    s = x0 * x0
    s = tl.fma(x1, x1, s)
    s = tl.fma(x2, x2, s)
    s = tl.fma(x4, x4, s)
    # warp shfl_down 16 (adds zeros), 8, 4, 2, 1 over 16 live lanes
    tl.static_assert(channels == 64)
    s = tl.sum(tl.reshape(s, [block_p, 2, 8]), axis=1)
    s = tl.sum(tl.reshape(s, [block_p, 2, 4]), axis=1)
    s = tl.sum(tl.reshape(s, [block_p, 2, 2]), axis=1)
    s = tl.sum(tl.reshape(s, [block_p, 2, 1]), axis=1)
    ms = tl.reshape(s, [block_p]) / channels
    rstd = tl.math.rsqrt(ms + eps)
    w = tl.load(w_ptr + c).to(tl.float32)
    y = w[None, :] * (rstd[:, None] * x)
    row = row0 + q // pp
    pix = q % pp
    b = row // num_patches
    p = row % num_patches
    ph = p // patches_w
    pw = p % patches_w
    py = pix // patch
    px = pix % patch
    o = ((b * height + ph * patch + py).to(tl.int64) * width + (pw * patch + px)) * channels
    tl.store(out + o[:, None] + c[None, :], y.to(out.dtype.element_ty), mask=mask[:, None])


def nerf_fold_rms_norm(
    tiles: list[torch.Tensor],
    weight: torch.Tensor,
    batch: int,
    height: int,
    width: int,
    patch_size: int,
    eps: float | None = None,
) -> torch.Tensor:
    t0 = tiles[0]
    channels = t0.shape[2]
    pp = patch_size * patch_size
    eps = torch.finfo(torch.float32).eps if eps is None else float(eps)
    num_patches = sum(t.shape[0] for t in tiles) // batch
    out = torch.empty((batch, height, width, channels), device=t0.device, dtype=t0.dtype)
    row0 = 0
    for t in tiles:
        n_q = t.shape[0] * pp
        _nerf_fold_rms_norm_kernel[(triton.cdiv(n_q, _BLOCK_P),)](
            t,
            weight,
            out,
            row0,
            n_q,
            num_patches,
            width // patch_size,
            height,
            width,
            eps,
            patch=patch_size,
            pp=pp,
            channels=channels,
            block_p=_BLOCK_P,
            num_warps=_NUM_WARPS,
        )
        row0 += t.shape[0]
    return out.permute(0, 3, 1, 2)
