# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton NeRF GLU block: weight normalization, RMSNorm, GLU MLP and residual in one kernel.

One program owns ``block_m`` pixels of one patch (grid ``(N / block_m, B)``) and reads that
patch's generated matrices straight from its mlp_params row:

    x tile [block_m, H] -> RMSNorm in fp32 -> x's dtype (the operand of the first two dots)
    prepass over the output matrix w2 [D, H]: its column norms n2 (over D) in fp32
    for each block of block_h hidden units:
        gate / value columns [H, block_h]: their norms over H in fp32
        g = dot(n, gate) / n_gate, v = dot(n, value) / n_value       (fp32)
        h = silu(g) * v, rounded to x's dtype; acc += dot(h, w2 rows)
    out = acc / n2 + x, one rounding to x's dtype

``F.normalize(dim=-2)`` scales each column of a matrix, so the dots run on the raw generated
weights and the scale is applied in fp32 after the dot: the normalized matrices, the normed x,
the gate / value / hidden activations and the MLP output are never written out. Norms are
clamped to 1e-12 as ``F.normalize`` does.

Launch configuration: block_m 256, block_h 32, 8 warps on every GPU; it was the fastest of
the three measured on an RTX 5090 and on a Radeon 8060S at Chroma Radiance's block
(x [512, 256, 64], mlp_params [512, 49152], bf16).
"""

import torch

import triton
import triton.language as tl

BLOCK_M = 256
BLOCK_H = 32
NUM_WARPS = 8


@triton.jit
def _nerf_glu_block_kernel(
    x_ptr,
    p_ptr,
    w_ptr,
    out_ptr,
    sxb,
    sxn,
    spb,
    sob,
    son,
    eps,
    hidden_x: tl.constexpr,
    hidden: tl.constexpr,
    block_m: tl.constexpr,
    block_h: tl.constexpr,
):
    pid_m = tl.program_id(0)
    b = tl.program_id(1).to(tl.int64)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_k = tl.arange(0, hidden_x)

    x_raw = tl.load(x_ptr + b * sxb + offs_m[:, None] * sxn + offs_k[None, :])
    xf = x_raw.to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(xf * xf, axis=1) / hidden_x + eps)
    nw = tl.load(w_ptr + offs_k).to(tl.float32)
    xn = (xf * rstd[:, None] * nw[None, :]).to(x_raw.dtype)

    g_base = p_ptr + b * spb  # gate [hidden_x, hidden], row-major
    v_base = g_base + hidden_x * hidden  # value [hidden_x, hidden]
    w2_base = g_base + 2 * hidden_x * hidden  # w2 [hidden, hidden_x]

    sq2 = tl.zeros([hidden_x], dtype=tl.float32)
    for hb in range(0, hidden, block_h):
        offs_h = hb + tl.arange(0, block_h)
        w2 = tl.load(w2_base + offs_h[:, None] * hidden_x + offs_k[None, :]).to(tl.float32)
        sq2 += tl.sum(w2 * w2, axis=0)
    n2 = tl.maximum(tl.sqrt(sq2), 1e-12)  # F.normalize clamps the norm to 1e-12

    acc = tl.zeros([block_m, hidden_x], dtype=tl.float32)
    for hb in range(0, hidden, block_h):
        offs_h = hb + tl.arange(0, block_h)
        wg = tl.load(g_base + offs_k[:, None] * hidden + offs_h[None, :])
        wv = tl.load(v_base + offs_k[:, None] * hidden + offs_h[None, :])
        w2 = tl.load(w2_base + offs_h[:, None] * hidden_x + offs_k[None, :])
        wgf = wg.to(tl.float32)
        wvf = wv.to(tl.float32)
        ng = tl.maximum(tl.sqrt(tl.sum(wgf * wgf, axis=0)), 1e-12)
        nv = tl.maximum(tl.sqrt(tl.sum(wvf * wvf, axis=0)), 1e-12)
        g = tl.dot(xn, wg) / ng[None, :]
        v = tl.dot(xn, wv) / nv[None, :]
        h = (g / (1.0 + tl.exp(-g)) * v).to(x_raw.dtype)
        acc += tl.dot(h, w2)

    out = (acc / n2[None, :] + xf).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + b * sob + offs_m[:, None] * son + offs_k[None, :], out)


def nerf_glu_block(
    x: torch.Tensor,
    mlp_params: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float | None = None,
) -> torch.Tensor:
    batch_size, num_x, hidden_x = x.shape
    hidden = mlp_params.shape[1] // (3 * hidden_x)
    if eps is None:
        # what F.rms_norm uses for eps=None on fp16 / bf16 input (it computes in fp32)
        eps = torch.finfo(torch.float32).eps
    out = torch.empty((batch_size, num_x, hidden_x), dtype=x.dtype, device=x.device)
    grid = (num_x // BLOCK_M, batch_size)
    _nerf_glu_block_kernel[grid](
        x,
        mlp_params,
        norm_weight,
        out,
        x.stride(0),
        x.stride(1),
        mlp_params.stride(0),
        out.stride(0),
        out.stride(1),
        float(eps),
        hidden_x=hidden_x,
        hidden=hidden,
        block_m=BLOCK_M,
        block_h=BLOCK_H,
        num_warps=NUM_WARPS,
    )
    return out
