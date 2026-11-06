# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton temporal interleave + frames to batch + nearest-exact 2x upsample, one gather.

Writes the contiguous ``[N, C, 2H, 2W]`` output (``N = B * T'``) straight from x:

    out[n, ch, ho, wo] = x[bb, src_c, src_t, ho // 2, wo // 2],  bb, tt = divmod(n, T')
    interleave: src_c = (tt % 2) * C + ch, src_t = tt // 2   (T' = 2T, x has 2C channels)
    otherwise:  src_c = ch,                src_t = tt        (T' = T)

For a scale of 2, nearest-exact maps an output index ``d`` to ``floor((d + 0.5) / 2) = d // 2``,
so the result is a pure copy, bit-identical to the eager path. One program row handles
``block_r`` source rows of ``block_ws`` pixels: it loads them once from x through x's strides,
duplicates every pixel along the row with ``tl.interleave`` and stores the result to both
output rows. The stack, the rearranged copy and the upsample of the eager path, which each
read and write the full tensor, become this one read of x and one write of the output.
"""

import torch

import triton
import triton.language as tl


@triton.jit
def _src_row_base(rows, channels, frames, height, sxb, sxc, sxt, sxh, interleave: tl.constexpr):
    # offset in x of the first pixel of source row `rows` = ((n * C + ch) * H + hs)
    hs = rows % height
    r2 = rows // height
    ch = r2 % channels
    n = r2 // channels
    bb = n // frames
    tt = n % frames
    if interleave:
        src_c = (tt % 2) * channels + ch
        src_t = tt // 2
    else:
        src_c = ch
        src_t = tt
    return bb * sxb + src_c * sxc + src_t * sxt + hs * sxh


@triton.jit
def _interleave_upsample_nearest2x_kernel(
    x_ptr, out_ptr, n_rows, channels, frames, height, width,
    sxb, sxc, sxt, sxh, sxw,
    interleave: tl.constexpr, block_r: tl.constexpr, block_ws: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_w = tl.program_id(1)
    rows = pid_r.to(tl.int64) * block_r + tl.arange(0, block_r)
    ws = pid_w * block_ws + tl.arange(0, block_ws)
    base = _src_row_base(rows, channels, frames, height, sxb, sxc, sxt, sxh, interleave)
    lmask = (rows < n_rows)[:, None] & (ws < width)[None, :]
    v = tl.load(x_ptr + base[:, None] + (ws * sxw)[None, :], mask=lmask)
    v2 = tl.interleave(v, v)  # [block_r, 2 * block_ws]: every pixel twice along the row
    cols = pid_w * (2 * block_ws) + tl.arange(0, 2 * block_ws)
    width_out = 2 * width
    smask = (rows < n_rows)[:, None] & (cols < width_out)[None, :]
    # source row r = (n * C + ch) * H + hs goes to output rows 2r and 2r + 1
    o_off = ((2 * rows) * width_out)[:, None] + cols[None, :]
    tl.store(out_ptr + o_off, v2, mask=smask)
    tl.store(out_ptr + o_off + width_out, v2, mask=smask)


# (block_r, block_ws) tiles x num_warps, as measured on RTX 5090 and Radeon 8060S
_CONFIGS = [
    triton.Config({"block_r": r, "block_ws": w}, num_warps=nw)
    for (r, w) in [(1, 512), (2, 256), (4, 128), (8, 64), (2, 512), (4, 256), (8, 128)]
    for nw in (4, 8)
]
_kernel = triton.autotune(configs=_CONFIGS, key=["width", "n_rows", "interleave"])(
    _interleave_upsample_nearest2x_kernel
)


def temporal_interleave_upsample_nearest2x(
    x: torch.Tensor,
    interleave: bool = False,
) -> torch.Tensor:
    b, c, t, h, w = x.shape
    if interleave:
        c, t = c // 2, t * 2
    out = torch.empty((b * t, c, 2 * h, 2 * w), dtype=x.dtype, device=x.device)
    n_rows = b * t * c * h

    def grid(meta):
        return (triton.cdiv(n_rows, meta["block_r"]), triton.cdiv(w, meta["block_ws"]))

    _kernel[grid](x, out, n_rows, c, t, h, w, *x.stride(), interleave=bool(interleave))
    return out
