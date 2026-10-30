# SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton GEMM with a fused tanh-GELU epilogue.

    out[M, N] = gelu_tanh(x[M, K] @ weight[N, K]^T + bias[N])

fp16 / bf16 in and out, fp32 accumulate. The point is the epilogue: an FFN up
projection followed by a separate GELU writes the (M, N) pre-activation to memory
and reads it straight back; here GELU runs on the accumulator tile before the
single store.

Plain Triton only (tl.dot / tl.exp2, no TMA, no inline asm), so the same kernel
compiles for NVIDIA and AMD GPUs.

Numerics follow the unfused PyTorch ops: the biased accumulator is rounded to the
activation dtype (what F.linear returns), GELU is evaluated in fp32 on that value
and rounded once more, as PyTorch's GELU kernel does with its fp32 opmath.

Requirements (checked by the backend's constraints, otherwise dispatch falls back):
weight rows contiguous, N % N_MULTIPLE == 0, K % K_MULTIPLE == 0. M is masked.
"""

import torch

import triton
import triton.language as tl

# Every autotune config below tiles N and K evenly when these hold.
N_MULTIPLE = 128
K_MULTIPLE = 64
# Below this many rows the GEMM is too small for the fused kernel to be worth
# routing to; dispatch keeps those on eager (cuBLAS / hipBLASLt + GELU).
MIN_M = 256

_CONFIGS = [
    # Tuned on NVIDIA RTX 5090 (sm_120, ~99 KB shared memory per block) for the
    # Wan2.1-1.3B FFN up projection (M 28080, K 1536, N 8960, fp16).
    triton.Config({"BM": 128, "BN": 128, "BK": 64, "GROUP_M": 8}, num_warps=4, num_stages=2),
    # Smaller-footprint configs so the kernel also runs where shared memory / LDS
    # is tighter (e.g. 64 KB on AMD GPUs). Not tuned on any particular GPU.
    triton.Config({"BM": 128, "BN": 128, "BK": 32, "GROUP_M": 8}, num_warps=4, num_stages=2),
    triton.Config({"BM": 64, "BN": 128, "BK": 64, "GROUP_M": 8}, num_warps=4, num_stages=2),
    triton.Config({"BM": 128, "BN": 64, "BK": 32, "GROUP_M": 8}, num_warps=4, num_stages=2),
    triton.Config({"BM": 128, "BN": 128, "BK": 32, "GROUP_M": 8}, num_warps=8, num_stages=2),
]


@triton.jit
def _round_to(v, dt: tl.constexpr):
    # Round fp32 to dt and back. Goes through an integer bitcast because Triton
    # folds a plain v.to(dt).to(fp32) round trip while the value is in registers.
    if dt == tl.bfloat16:
        b = v.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) << 16
        return b.to(tl.float32, bitcast=True)
    else:
        h = v.to(tl.float16).to(tl.int16, bitcast=True)
        return h.to(tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def _gelu_tanh(x):
    # 0.5 * x * (1 + tanh(u)) == x / (1 + exp(-2u)), u = sqrt(2/pi) * (x + 0.044715 x^3),
    # with exp(-2u) = exp2(-2u * log2(e)): one exp2, no libdevice tanh.
    u = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    return x / (1.0 + tl.math.exp2(-2.8853900817779268 * u))


@triton.autotune(configs=_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _linear_gelu_tanh_kernel(
    x_ptr, w_ptr, b_ptr, o_ptr,
    M, N, K,
    stride_xm, stride_wn, stride_om,
    HAS_BIAS: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr,
):
    # Grouped tile order: GROUP_M row tiles sweep the N tiles together for L2 reuse.
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    group = GROUP_M * num_n
    gid = pid // group
    first_m = gid * GROUP_M
    gsz = tl.minimum(num_m - first_m, GROUP_M)
    pm = first_m + (pid % group) % gsz
    pn = (pid % group) // gsz

    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a_p = x_ptr + rm[:, None].to(tl.int64) * stride_xm + rk[None, :]
    b_p = w_ptr + rn[None, :].to(tl.int64) * stride_wn + rk[:, None]
    m_mask = rm[:, None] < M

    acc = tl.zeros([BM, BN], tl.float32)
    for _ in range(0, K, BK):
        a = tl.load(a_p, mask=m_mask, other=0.0)
        b = tl.load(b_p)
        acc = tl.dot(a, b, acc)
        a_p += BK
        b_p += BK

    dt = o_ptr.dtype.element_ty
    if HAS_BIAS:
        acc = acc + tl.load(b_ptr + rn).to(tl.float32)[None, :]
    o = _gelu_tanh(_round_to(acc, dt)).to(dt)
    tl.store(o_ptr + rm[:, None].to(tl.int64) * stride_om + rn[None, :], o, mask=m_mask)


def linear_gelu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    approximate: str = "tanh",
) -> torch.Tensor:
    """gelu(linear(x, weight, bias), approximate="tanh") in one GEMM."""
    if approximate != "tanh":
        raise ValueError(
            f"triton linear_gelu implements approximate='tanh' only, got {approximate!r}"
        )
    k = x.shape[-1]
    n = weight.shape[0]
    x2 = x.reshape(-1, k)
    if x2.stride(1) != 1:
        x2 = x2.contiguous()
    m = x2.shape[0]
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    has_bias = bias is not None

    def grid(meta):
        return (triton.cdiv(m, meta["BM"]) * (n // meta["BN"]),)

    _linear_gelu_tanh_kernel[grid](
        x2, weight, bias if has_bias else weight, out,
        m, n, k,
        x2.stride(0), weight.stride(0), out.stride(0),
        HAS_BIAS=has_bias,
    )
    return out.view(*x.shape[:-1], n)
