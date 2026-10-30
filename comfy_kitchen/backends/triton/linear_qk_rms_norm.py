# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton fused QKV projection + per-head q/k RMSNorm.

One GEMM tile covers ``block_n // head_dim`` whole heads of exactly one of q, k, v (the
config pruner keeps ``block_n`` a multiple of ``head_dim`` that divides D), so the RMSNorm
runs in the epilogue on the fp32 accumulator after ``acc + bias`` is rounded once to the
output dtype, as ``F.linear`` followed by ``F.rms_norm`` rounds.
"""

import torch

import triton
import triton.language as tl


def _prune_configs(configs, named_args, **kwargs):
    dim = named_args["dim"]
    head_dim = named_args.get("head_dim", kwargs.get("head_dim"))   # constexprs arrive in kwargs
    kept = [c for c in configs
            if c.kwargs["block_n"] % head_dim == 0 and dim % c.kwargs["block_n"] == 0]
    return kept


@triton.autotune(
    configs=[
        triton.Config({"block_m": 64, "block_n": 128, "block_k": 64, "group_size_m": 8}, num_stages=2, num_warps=4),
        triton.Config({"block_m": 128, "block_n": 128, "block_k": 32, "group_size_m": 8}, num_stages=2, num_warps=4),
        triton.Config({"block_m": 128, "block_n": 128, "block_k": 64, "group_size_m": 8}, num_stages=2, num_warps=8),
        triton.Config({"block_m": 64, "block_n": 64, "block_k": 64, "group_size_m": 8}, num_stages=2, num_warps=4),
        triton.Config({"block_m": 128, "block_n": 64, "block_k": 32, "group_size_m": 8}, num_stages=3, num_warps=4),
        triton.Config({"block_m": 64, "block_n": 256, "block_k": 32, "group_size_m": 8}, num_stages=2, num_warps=8),
    ],
    key=["m", "dim", "k", "head_dim"],
    prune_configs_by={"early_config_prune": _prune_configs},
)
@triton.jit
def _linear_qk_rms_norm_kernel(
    x_ptr, w_ptr, b_ptr, qs_ptr, ks_ptr, q_ptr, k_ptr, v_ptr,
    m, dim, k, eps,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    head_dim: tl.constexpr,
    has_bias: tl.constexpr,
    block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr,
    group_size_m: tl.constexpr,
):
    out_dtype = q_ptr.dtype.element_ty
    n = 3 * dim
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(m, block_m)
    num_pid_n = n // block_n
    num_pid_in_group = group_size_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * group_size_m
    group_m = min(num_pid_m - first_pid_m, group_size_m)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_m
    pid_n = (pid % num_pid_in_group) // group_m

    rm = pid_m * block_m + tl.arange(0, block_m)
    rn = pid_n * block_n + tl.arange(0, block_n)
    rk = tl.arange(0, block_k)
    m_mask = rm < m

    x_ptrs = x_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk
    w_ptrs = w_ptr + rn[None, :] * stride_wn + rk[:, None] * stride_wk
    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(k, block_k)):
        k_mask = rk < k - k0 * block_k
        a = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=k_mask[:, None], other=0.0)
        acc += tl.dot(a, w)
        x_ptrs += block_k * stride_xk
        w_ptrs += block_k * stride_wk

    if has_bias:
        acc += tl.load(b_ptr + rn).to(tl.float32)[None, :]
    # F.linear rounds the projection to the output dtype before the norm reads it
    y = acc.to(out_dtype).to(tl.float32)

    which = (pid_n * block_n) // dim          # 0 = q, 1 = k, 2 = v: a tile never straddles two
    col = rn - which * dim
    d = col % head_dim
    if which < 2:
        y3 = tl.reshape(y, (block_m, block_n // head_dim, head_dim))
        inv = tl.math.rsqrt(tl.sum(y3 * y3, 2) / head_dim + eps)
        inv = tl.reshape(tl.broadcast_to(inv[:, :, None], (block_m, block_n // head_dim, head_dim)),
                         (block_m, block_n))
        if which == 0:
            scale = tl.load(qs_ptr + d).to(out_dtype).to(tl.float32)
        else:
            scale = tl.load(ks_ptr + d).to(out_dtype).to(tl.float32)
        y = y * inv * scale[None, :]

    dst = rm[:, None] * dim + col[None, :]   # q, k, v are contiguous [M, D]
    if which == 0:
        tl.store(q_ptr + dst, y.to(out_dtype), mask=m_mask[:, None])
    elif which == 1:
        tl.store(k_ptr + dst, y.to(out_dtype), mask=m_mask[:, None])
    else:
        tl.store(v_ptr + dst, y.to(out_dtype), mask=m_mask[:, None])


def linear_qk_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    head_dim: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b, s, kdim = x.shape
    dim = weight.shape[0] // 3
    x2 = x.reshape(b * s, kdim)
    m = b * s
    has_bias = bias is not None
    qs = q_scale.contiguous()
    ks = k_scale.contiguous()
    q, k, v = (torch.empty((b, s, dim // head_dim, head_dim), device=x.device, dtype=x.dtype) for _ in range(3))

    def grid(meta):
        return (triton.cdiv(m, meta["block_m"]) * ((3 * dim) // meta["block_n"]),)

    _linear_qk_rms_norm_kernel[grid](
        x2, weight, bias if has_bias else qs, qs, ks, q, k, v,
        m, dim, kdim, eps,
        x2.stride(0), x2.stride(1),
        weight.stride(0), weight.stride(1),
        head_dim=head_dim, has_bias=has_bias,
    )
    return q, k, v
