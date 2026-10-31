# SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton GEMMs with a fused activation epilogue.

One dual-accumulator GEMM kernel with the activation fused into its epilogue,
selected by ACT:

    linear_geglu:  h, gate = x @ W^T + b (W = [W_h; W_gate], packed by rows)
                   out = h * gelu(gate)
    linear_swiglu: out = silu(x @ W1^T + b1) * (x @ W3^T + b3)

fp16 / bf16 in and out, fp32 accumulate. The point is the epilogue: a gated FFN
up projection writes two (M, N) projections to memory and a separate activation
and multiply read them straight back; here the activation and the product run
on the two accumulator tiles before the single store. Each K step loads one x
tile and feeds it to both accumulators, so x is also read once instead of twice.

Plain Triton only (tl.dot / tl.exp / tl.exp2 / tl.math.erf, no TMA, no inline
asm), so the same kernel compiles for NVIDIA and AMD GPUs.

Numerics follow the unfused PyTorch ops: every value the eager op sequence
materializes (the biased GEMM result, the activation output, the product) is
rounded to the activation dtype at the same point, and the activations are
evaluated in fp32 on the rounded inputs, as PyTorch's kernels do with their fp32
opmath.

Requirements (checked by the backend's constraints, otherwise dispatch falls back):
weight rows contiguous, N % GLU_N_MULTIPLE == 0, K % K_MULTIPLE == 0. M is
masked; so is N whenever the tile width BN of the chosen config does not divide
it, so no autotune config can leave output columns unwritten.
"""

import torch

import triton
import triton.language as tl

# Every autotune config below tiles K evenly when K_MULTIPLE holds. N tiles
# that run past N are masked (EVEN_N below), so BN need not divide N.
GLU_N_MULTIPLE = 64  # N of one half (the output width) for the gated forms
K_MULTIPLE = 64
# Below this many rows the GEMM is too small for the fused kernel to be worth
# routing to; dispatch keeps those on eager (cuBLAS / hipBLASLt + activation).
MIN_M = 256

# Epilogue selector (constexpr).
_ACT_GEGLU_ERF = 0
_ACT_GEGLU_TANH = 1
_ACT_SWIGLU = 2

# The dual-accumulator kernel holds two (BM, BN) accumulators and streams two
# weight tiles per K step, so it uses narrow N tiles. GEGLU and SwiGLU get one
# list each: they run in different dtypes (SDXL GEGLU fp16, Z-Image SwiGLU bf16)
# at different shapes and weight layouts, and a config tuned for one can win the
# other's autotune run without being the fastest there.
_GEGLU_CONFIGS = [
    # Tuned on NVIDIA RTX 5090 (sm_120): SDXL GEGLU at K 1280 (M 2048, N 5120)
    # and K 640 (M 8192, N 2560), fp16.
    triton.Config({"BM": 128, "BN": 64, "BK": 32, "GROUP_M": 8}, num_warps=4, num_stages=3),
    triton.Config({"BM": 64, "BN": 64, "BK": 32, "GROUP_M": 8}, num_warps=4, num_stages=3),
    # Smaller footprint; the pick on AMD Radeon 8060S (gfx1151) at both shapes.
    triton.Config({"BM": 64, "BN": 64, "BK": 64, "GROUP_M": 8}, num_warps=4, num_stages=2),
    # Tuned on AMD Radeon AI PRO R9700 (gfx1201, 64 KB LDS) for the SDXL GEGLU
    # shapes above. There the BM 128 dual tiles need 8 warps: with 4 warps the two
    # fp16 accumulators hit the 256-VGPR limit and spill.
    triton.Config({"BM": 128, "BN": 64, "BK": 64, "GROUP_M": 16}, num_warps=8, num_stages=1),
    triton.Config({"BM": 128, "BN": 64, "BK": 32, "GROUP_M": 16}, num_warps=8, num_stages=3),
]

# SwiGLU, tuned on the Z-Image FFN (M 4160, K 3840, N 10240, bf16). Only configs
# that measured fastest, or within about 1% of the fastest, on one of the GPUs
# below.
_SWIGLU_CONFIGS = [
    # Fastest on NVIDIA RTX 5090 (sm_120); within 1% on AMD Radeon AI PRO R9700.
    triton.Config({"BM": 128, "BN": 64, "BK": 64, "GROUP_M": 32}, num_warps=4, num_stages=2),
    # Fastest on AMD Radeon AI PRO R9700 (gfx1201).
    triton.Config({"BM": 128, "BN": 64, "BK": 64, "GROUP_M": 16}, num_warps=8, num_stages=1),
    # Tuned on AMD Radeon 8060S (gfx1151, 2 MB L2): within 1% of the fastest
    # config measured there, while GROUP_M 32 is about 45% slower on the small
    # L2. Within 1% on the R9700.
    triton.Config({"BM": 128, "BN": 64, "BK": 64, "GROUP_M": 8}, num_warps=4, num_stages=2),
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


@triton.jit
def _gelu_erf(x):
    # x * 0.5 * (1 + erf(x / sqrt(2))), PyTorch's approximate="none" form.
    return x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))


@triton.jit
def _linear_act_kernel(
    x_ptr, w1_ptr, w2_ptr, b1_ptr, b2_ptr, o_ptr,
    M, N, K,
    stride_xm, stride_w1n, stride_w2n, stride_om,
    w2_row0,
    ACT: tl.constexpr, HAS_B1: tl.constexpr, HAS_B2: tl.constexpr,
    EVEN_N: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr,
):
    # Second accumulator: rows w2_row0 .. w2_row0 + N of w2 (and the same
    # elements of b2). GEGLU passes the packed weight as both w1 and w2 with
    # w2_row0 = N, so its halves are read by offset, without per-call views.
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
    if EVEN_N:
        rn_ld = rn
        o_mask = rm[:, None] < M
    else:
        # Last N tile runs past N: wrap its weight / bias reads onto valid rows
        # (no masks in the K loop) and drop those columns at the store.
        rn_ld = rn % N
        o_mask = (rm[:, None] < M) & (rn[None, :] < N)
    rk = tl.arange(0, BK)
    a_p = x_ptr + rm[:, None].to(tl.int64) * stride_xm + rk[None, :]
    b1_p = w1_ptr + rn_ld[None, :].to(tl.int64) * stride_w1n + rk[:, None]
    b2_p = w2_ptr + (rn_ld[None, :] + w2_row0).to(tl.int64) * stride_w2n + rk[:, None]
    m_mask = rm[:, None] < M

    acc1 = tl.zeros([BM, BN], tl.float32)
    acc2 = tl.zeros([BM, BN], tl.float32)
    for _ in range(0, K, BK):
        # All loads first, then the dots: loading the second weight tile after
        # the first dot measured 2-7% slower.
        a = tl.load(a_p, mask=m_mask, other=0.0)
        b1 = tl.load(b1_p)
        b2 = tl.load(b2_p)
        acc1 = tl.dot(a, b1, acc1)
        acc2 = tl.dot(a, b2, acc2)
        a_p += BK
        b1_p += BK
        b2_p += BK

    dt = o_ptr.dtype.element_ty
    if HAS_B1:
        acc1 = acc1 + tl.load(b1_ptr + rn_ld).to(tl.float32)[None, :]
    if HAS_B2:
        acc2 = acc2 + tl.load(b2_ptr + w2_row0 + rn_ld).to(tl.float32)[None, :]
    if ACT == 2:
        # silu(c1) * c3, each eager intermediate rounded.
        c1 = _round_to(acc1, dt)
        c3 = _round_to(acc2, dt)
        s = _round_to(c1 / (1.0 + tl.exp(-c1)), dt)
        o = (s * c3).to(dt)
    else:
        # h * gelu(gate), each eager intermediate rounded.
        h = _round_to(acc1, dt)
        gate = _round_to(acc2, dt)
        if ACT == 0:
            g = _round_to(_gelu_erf(gate), dt)
        else:
            g = _round_to(_gelu_tanh(gate), dt)
        o = (h * g).to(dt)
    tl.store(o_ptr + rm[:, None].to(tl.int64) * stride_om + rn[None, :], o, mask=o_mask)


# EVEN_N follows the BN of whichever config the autotuner runs; when BN divides
# N (every shape the configs were tuned on) the kernel compiles without the N
# wrap and mask, i.e. exactly as before. One autotuner per op (own config list
# and timing cache), both over the same kernel.
def _autotuned(configs):
    return triton.autotune(configs=configs, key=["M", "N", "K", "ACT"])(
        triton.heuristics({"EVEN_N": lambda args: args["N"] % args["BN"] == 0})(_linear_act_kernel)
    )


_linear_geglu_kernel = _autotuned(_GEGLU_CONFIGS)
_linear_swiglu_kernel = _autotuned(_SWIGLU_CONFIGS)


def _launch(kernel, x, w1, w2, w2_row0, b1, b2, n, act):
    k = x.shape[-1]
    x2 = x.reshape(-1, k)
    if x2.stride(1) != 1:
        x2 = x2.contiguous()
    m = x2.shape[0]
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (triton.cdiv(m, meta["BM"]) * triton.cdiv(n, meta["BN"]),)

    kernel[grid](
        x2, w1, w2, w1 if b1 is None else b1, w1 if b2 is None else b2, out,
        m, n, k,
        x2.stride(0), w1.stride(0), w2.stride(0), out.stride(0),
        w2_row0,
        ACT=act, HAS_B1=b1 is not None, HAS_B2=b2 is not None,
    )
    return out.view(*x.shape[:-1], n)


def linear_geglu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    approximate: str = "none",
) -> torch.Tensor:
    """h * gelu(gate), (h, gate) = linear(x, weight, bias).chunk(2, -1), in one GEMM."""
    if approximate not in ("none", "tanh"):
        raise ValueError(f"approximate must be 'none' or 'tanh', got {approximate!r}")
    n = weight.shape[0] // 2
    act = _ACT_GEGLU_ERF if approximate == "none" else _ACT_GEGLU_TANH
    return _launch(_linear_geglu_kernel, x, weight, weight, n, bias, bias, n, act)


def linear_swiglu(
    x: torch.Tensor,
    w1: torch.Tensor,
    w3: torch.Tensor,
    b1: torch.Tensor | None = None,
    b3: torch.Tensor | None = None,
) -> torch.Tensor:
    """silu(linear(x, w1, b1)) * linear(x, w3, b3) in one GEMM."""
    return _launch(_linear_swiglu_kernel, x, w1, w3, 0, b1, b3, w1.shape[0], _ACT_SWIGLU)
