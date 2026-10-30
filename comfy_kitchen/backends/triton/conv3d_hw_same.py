# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton implicit-GEMM conv3d, "same" H/W padding, no T padding, optional nearest x2 upsample.

M = To * H * W output positions by N = Co channels, K = kt * kh * kw * Ci, accumulated in fp32.
The input is read straight from x [1, Ci, T, Hin, Win] (with upsample=2 the nearest x2 upsample
is folded into the load addresses: input row ih // 2, column iw // 2), and the weight from its
own [Co, Ci, kt, kh, kw] layout through its strides, so a sliced weight (the T == 1
``weight[:, :, -1:]`` path) needs no copy. Offsets are int64.
"""

import torch

import triton
import triton.language as tl


@triton.jit
def _conv3d_hw_same_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    ci, t_in, h, w, h_in, w_in, co, m,
    sw_o, sw_i, sw_t, sw_h, sw_w,
    up: tl.constexpr, kt_: tl.constexpr, kh_: tl.constexpr, kw_: tl.constexpr,
    has_bias: tl.constexpr, block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr,
    group_m: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m = tl.cdiv(m, block_m)
    num_n = tl.cdiv(co, block_n)
    gid = pid // (group_m * num_n)
    first = gid * group_m
    gsz = tl.minimum(num_m - first, group_m)
    pm = first + (pid % (group_m * num_n)) % gsz
    pn = (pid % (group_m * num_n)) // gsz
    rm = (pm * block_m + tl.arange(0, block_m)).to(tl.int64)
    rn = (pn * block_n + tl.arange(0, block_n)).to(tl.int64)
    ow = rm % w
    oh = (rm // w) % h
    ot = rm // (w * h)
    hw_in = h_in * w_in
    t_stride = t_in * hw_in          # one input channel; int64 once multiplied by the int64 channel index
    acc = tl.zeros([block_m, block_n], tl.float32)
    for kk in range(kt_ * kh_ * kw_):
        dt = kk // (kh_ * kw_)
        dh = (kk // kw_) % kh_
        dw = kk % kw_
        ih = oh + dh - kh_ // 2
        iw = ow + dw - kw_ // 2
        valid = (rm < m) & (ih >= 0) & (ih < h) & (iw >= 0) & (iw < w)
        base = (ot + dt) * hw_in + (ih // up) * w_in + iw // up
        w_base = w_ptr + dt * sw_t + dh * sw_h + dw * sw_w
        for c0 in range(0, ci, block_k):
            c = (c0 + tl.arange(0, block_k)).to(tl.int64)
            a = tl.load(x_ptr + c[None, :] * t_stride + base[:, None],
                        mask=valid[:, None] & (c[None, :] < ci), other=0.0)
            b = tl.load(w_base + c[:, None] * sw_i + rn[None, :] * sw_o,
                        mask=(c[:, None] < ci) & (rn[None, :] < co), other=0.0)
            acc += tl.dot(a, b)
    if has_bias:
        acc += tl.load(b_ptr + rn, mask=rn < co, other=0.0).to(tl.float32)[None, :]
    tl.store(y_ptr + rn[None, :] * m + rm[:, None], acc.to(y_ptr.dtype.element_ty),
             mask=(rm[:, None] < m) & (rn[None, :] < co))


def _config(m, ci, co, kh, up):
    # (block_m, block_n, block_k, num_warps, num_stages), the per-shape tiles measured on gfx1201
    if kh == 1:
        return 128, 128, 64, 8, 1
    if co <= 64:
        return 128, 64, 32, 8, 2
    if up == 2:
        return (64, 256, 64, 8, 2) if ci >= 1024 else (128, 256, 32, 8, 2)
    if m <= 1024:
        return 64, 64, 64, 4, 2
    if ci >= 1024:
        return 128, 256, 64, 8, 2
    return 256, 128, 32, 8, 2


def conv3d_hw_same(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    upsample: int = 1,
) -> torch.Tensor:
    # x [1, Ci, T, Hin, Win]; weight [Co, Ci, kt, kh, kw] any strides (call rule: batch 1, fp16/bf16)
    x = x.contiguous()
    _, ci, t_in, h_in, w_in = x.shape
    co, _, kt, kh, kw = weight.shape
    h, w = h_in * upsample, w_in * upsample
    t_out = t_in - kt + 1
    m = t_out * h * w
    y = torch.empty((1, co, t_out, h, w), device=x.device, dtype=x.dtype)
    block_m, block_n, block_k, num_warps, num_stages = _config(m, ci, co, kh, upsample)
    grid = (triton.cdiv(m, block_m) * triton.cdiv(co, block_n),)
    _conv3d_hw_same_kernel[grid](
        x, weight, bias if bias is not None else weight, y,
        ci, t_in, h, w, h_in, w_in, co, m,
        *weight.stride(),
        up=upsample, kt_=kt, kh_=kh, kw_=kw, has_bias=bias is not None,
        block_m=block_m, block_n=block_n, block_k=block_k, group_m=8,
        num_warps=num_warps, num_stages=num_stages,
    )
    return y
