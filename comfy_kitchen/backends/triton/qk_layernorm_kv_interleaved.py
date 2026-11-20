# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton cross-attention prep: per-head QK LayerNorm with k read from the fused kv.

One launch handles q and k: the first programs own ``block_r`` consecutive ``[head_dim]``
rows of the contiguous q, the rest ``block_r`` rows of k, read straight from the fused kv
(row stride ``2 * head_dim``: the first half of each head's block), so the strided-k copy
that ``F.layer_norm`` makes in the stock path is never written out. Each row is LayerNorm-ed
in fp32 registers and rounded once to the output dtype:

    mean / biased variance in the order of torch's vectorized layer_norm kernel for
    head_dim 64: a Welford online update over four values per lane (fma), then the
    shuffle-down combine tree over the lanes
    out = weight * (rsqrt(var + eps) * (x - mean)) + bias   (one fma)

Outputs: q ``[B, S_q, heads, head_dim]`` and k ``[B, S_kv, heads, head_dim]``, contiguous.

The Welford update multiplies by the reciprocal of the running count (1, 2, 3, 4). torch's
CUDA build uses the correctly rounded ``1.0f / count``; its ROCm build
(USE_LAYERNORM_FAST_RECIPROCAL, on by default for ROCm except gfx90a) uses the hardware
approximate reciprocal ``__builtin_amdgcn_rcpf(count)``, which is exact for 1, 2 and 4 and
one ulp below the correctly rounded value for 3. The kernel takes the count-3 reciprocal
that matches the torch build it runs under.

Launch configuration (same kernel body), per backend:
  CUDA: block_r 32 rows, 4 warps
  HIP:  block_r 16 rows, 4 warps
"""

import functools

import torch

import triton
import triton.language as tl

_HIP = bool(torch.version.hip)  # picks the launch configuration and the count-3 reciprocal

_RCP3_IEEE = 0.3333333432674408  # fp32 1.0f / 3 (0x3EAAAAAB)
_RCP3_AMDGCN = 0.3333333134651184  # __builtin_amdgcn_rcpf(3.0f) (0x3EAAAAAA)


@triton.jit
def _welford_combine(m, s, block_r: tl.constexpr, lanes: tl.constexpr, count: tl.constexpr):
    # torch cuWelfordCombine(dataB = own lane t < lanes, dataA = lane t + lanes), both
    # holding `count` values: mean = nA * A.mean + nB * B.mean (nA = nB = 0.5, exact
    # products), sigma2 = (A.sigma2 + B.sigma2) + delta * delta * A.count * nB
    mp = tl.permute(tl.reshape(m, (block_r, 2, lanes)), (0, 2, 1))
    sp = tl.permute(tl.reshape(s, (block_r, 2, lanes)), (0, 2, 1))
    mb, ma = tl.split(mp)
    sb, sa = tl.split(sp)
    delta = mb - ma
    mean = 0.5 * ma + 0.5 * mb
    sigma2 = (sa + sb) + delta * delta * count * 0.5
    return mean, sigma2


@triton.jit
def _qk_layernorm_kv_interleaved_kernel(
    q,
    kv,
    out_q,
    out_k,
    q_weight,
    q_bias,
    k_weight,
    k_bias,
    n_q_rows,
    n_k_rows,
    n_q_programs,
    eps,
    head_dim: tl.constexpr,
    block_r: tl.constexpr,
    rcp3: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_d = tl.arange(0, head_dim)
    if pid < n_q_programs:
        rows = pid.to(tl.int64) * block_r + tl.arange(0, block_r)
        rmask = rows < n_q_rows
        src = q + rows[:, None] * head_dim + offs_d[None, :]
        dst = out_q + rows[:, None] * head_dim + offs_d[None, :]
        w_ptr = q_weight
        b_ptr = q_bias
    else:
        rows = (pid - n_q_programs).to(tl.int64) * block_r + tl.arange(0, block_r)
        rmask = rows < n_k_rows
        src = kv + rows[:, None] * (2 * head_dim) + offs_d[None, :]
        dst = out_k + rows[:, None] * head_dim + offs_d[None, :]
        w_ptr = k_weight
        b_ptr = k_bias
    x = tl.load(src, mask=rmask[:, None], other=0.0).to(tl.float32)
    # lane t owns x[4t .. 4t + 3]: online Welford with the reciprocals 1, 1/2, rcp3, 1/4
    xe, xo = tl.split(tl.reshape(x, (block_r, head_dim // 4, 2, 2)))
    x0, x2 = tl.split(xe)
    x1, x3 = tl.split(xo)
    m = x0
    s = tl.zeros_like(x0)
    d = x1 - m
    m = tl.fma(d, 0.5, m)
    s = tl.fma(d, x1 - m, s)
    d = x2 - m
    m = tl.fma(d, rcp3, m)
    s = tl.fma(d, x2 - m, s)
    d = x3 - m
    m = tl.fma(d, 0.25, m)
    s = tl.fma(d, x3 - m, s)
    # shuffle-down tree over the 16 lanes holding data (the other lanes hold count 0, the
    # identity of the combine): offsets 8, 4, 2, 1
    m, s = _welford_combine(m, s, block_r, 8, 4.0)
    m, s = _welford_combine(m, s, block_r, 4, 8.0)
    m, s = _welford_combine(m, s, block_r, 2, 16.0)
    m, s = _welford_combine(m, s, block_r, 1, 32.0)
    mean = tl.reshape(m, (block_r,))
    var = tl.reshape(s, (block_r,)) / head_dim
    rstd = tl.math.rsqrt(var + eps)
    w = tl.load(w_ptr + offs_d).to(tl.float32)
    b = tl.load(b_ptr + offs_d).to(tl.float32)
    y = tl.fma(w[None, :], rstd[:, None] * (x - mean[:, None]), b[None, :])
    tl.store(dst, y.to(out_q.dtype.element_ty), mask=rmask[:, None])


@functools.cache
def _rcp3(device_index: int) -> float:
    # the count-3 reciprocal of torch's layer_norm Welford update on this device
    if not _HIP:
        return _RCP3_IEEE
    arch = torch.cuda.get_device_properties(device_index).gcnArchName
    return _RCP3_IEEE if arch.split(":")[0] == "gfx90a" else _RCP3_AMDGCN


def _launch_config() -> tuple[int, int]:
    # (block_r, num_warps)
    if _HIP:
        return 16, 4
    return 32, 4


def qk_layernorm_kv_interleaved(
    q: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    q_bias: torch.Tensor,
    k_weight: torch.Tensor,
    k_bias: torch.Tensor,
    heads: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    bs, n_ctx, channels = q.shape
    n_data = kv.shape[1]
    head_dim = channels // heads
    out_q = torch.empty((bs, n_ctx, heads, head_dim), device=q.device, dtype=q.dtype)
    out_k = torch.empty((bs, n_data, heads, head_dim), device=q.device, dtype=q.dtype)
    n_q_rows = bs * n_ctx * heads
    n_k_rows = bs * n_data * heads
    block_r, num_warps = _launch_config()
    n_q_programs = triton.cdiv(n_q_rows, block_r)
    grid = (n_q_programs + triton.cdiv(n_k_rows, block_r),)
    _qk_layernorm_kv_interleaved_kernel[grid](
        q,
        kv,
        out_q,
        out_k,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        n_q_rows,
        n_k_rows,
        n_q_programs,
        float(eps),
        head_dim=head_dim,
        block_r=block_r,
        rcp3=_rcp3(q.device.index if q.device.index is not None else torch.cuda.current_device()),
        num_warps=num_warps,
    )
    return out_q, out_k
