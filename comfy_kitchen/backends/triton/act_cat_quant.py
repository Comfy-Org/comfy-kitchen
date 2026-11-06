# SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Activation + concat + per-tensor FP8 quantization in one pass (triton).

Unfused, a Flux single-stream block runs four kernels in front of its FP8
output projection (silu, mul, cat, quantize) that write and re-read the
activation and the concatenated rows in 16 bit. Here one kernel reads x and h
once and writes only the FP8 data.

Numerics follow the unfused ops step by step, so the result is bit-identical:

- silu: ``x / (1 + exp(-x))`` in fp32, rounded to the input dtype (PyTorch's
  GPU silu); the product with the other half is rounded again. Under ROCm
  exp is written out as hipcc's expf, which PyTorch's ROCm silu calls.
- gelu_tanh: PyTorch's tanh-approximation formula in fp32, rounded once.
- quantize: the per-tensor quantizer of kitchen's compiled backend, then a
  round-to-nearest-even cast. CUDA: ``float(v) / scale`` (Triton's default fp32
  division, which is also kitchen's triton quantizer), clamped max-then-min so
  a NaN gives -448. ROCm: fast-math division, NaN kept (HIP quantizer); the
  one difference there is the sign of a NaN the activation makes out of a
  NaN / inf input, which follows Triton's NaN rules rather than PyTorch's.
"""

import torch

import triton
import triton.language as tl
from comfy_kitchen.float_utils import F8_E4M3_MAX
from triton.language.extra import libdevice

from ..eager.act_cat_quant import ACT_CAT_ACTS

# act -> kernel code
_ACT_CODE = {"swiglu": 0, "swiglu_hidden_gate": 1, "gelu_tanh": 2}

# Rows x columns per program. Each program covers BLOCK_N columns of either the
# copied part or the activated part, never both. Picked from a sweep at the
# Flux.2 Klein shape (M 17408, 3072 + 9216 columns): within 1% of the best on an
# RTX 5090, and about 14% faster than 8 x 512 on a Radeon AI PRO R9700.
BLOCK_M = 2
BLOCK_N = 1024
NUM_WARPS = 8

# Reproduce the platform's compiled code: under ROCm kitchen's HIP quantizer and
# hipcc's expf (PyTorch's silu), otherwise kitchen's CUDA quantizer.
_HIP = bool(getattr(torch.version, "hip", None))


@triton.jit
def _silu_rounded(g, hip: tl.constexpr):
    gf = g.to(tl.float32)
    if hip:
        # HIP's expf(-x) written out (log2(e) = 0x1.715476p+0 + 0x1.4ae0bep-26).
        # libdevice.exp is the same sequence, but with FMA fusion on it measured
        # 1 ulp off (fp16 silu at -2.7246); computing ph as an fma with a +0
        # addend matches hipcc on every fp16 and bf16 value.
        ph = tl.fma(gf, -1.4426950216293335, 0.0)
        pl = tl.fma(gf, -1.925962855864327e-08, tl.fma(gf, -1.4426950216293335, -ph))
        e = libdevice.rint(ph)
        ex = libdevice.ldexp(tl.math.exp2((ph - e) + pl), e.to(tl.int32))
        ex = tl.where(gf > 103.2789306640625, 0.0, ex)  # 0x1.9d1da0p+6
        ex = tl.where(gf < -88.72283935546875, float("inf"), ex)  # -0x1.62e430p+6
    else:
        ex = libdevice.exp(-gf)
    return tl.div_rn(gf, 1.0 + ex).to(g.dtype)


@triton.jit
def _round_product(p, sign, g, hip: tl.constexpr):
    """The fp32 product ``p`` rounded to ``g``'s dtype; ``sign`` holds p's sign bit."""
    if hip and g.dtype == tl.float16:
        # The AMDGPU backend can fold the multiply and the fp16 rounding into
        # one v_fma_mix with a +0 addend, which turns a -0 product into +0 (and
        # so the FP8 byte 0x80 into 0x00). Adding the product's own signed zero
        # keeps that sign and changes no other value.
        p = p + (sign & 0x80000000).to(tl.float32, bitcast=True)
    return p.to(g.dtype)


@triton.jit
def _bits(f):
    return f.to(tl.uint32, bitcast=True)


@triton.jit
def _act_cat_quant_kernel(
    x_ptr, h_ptr, out_ptr, scale_ptr, n_rows,
    stride_x, stride_h, stride_out,
    dx: tl.constexpr, dm: tl.constexpr, act: tl.constexpr, fp8_max: tl.constexpr,
    hip: tl.constexpr, block_m: tl.constexpr, block_n: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * block_m + tl.arange(0, block_m)
    row_mask = rows < n_rows
    rows64 = rows.to(tl.int64)
    nx: tl.constexpr = (dx + block_n - 1) // block_n
    if pid_n < nx:
        cols = pid_n * block_n + tl.arange(0, block_n)
        mask = row_mask[:, None] & (cols < dx)[None, :]
        v = tl.load(x_ptr + rows64[:, None] * stride_x + cols[None, :], mask=mask, other=0.0)
        out_cols = cols
    else:
        cols = (pid_n - nx) * block_n + tl.arange(0, block_n)
        mask = row_mask[:, None] & (cols < dm)[None, :]
        base = h_ptr + rows64[:, None] * stride_h + cols[None, :]
        if act == 0:  # [gate | up]: silu(gate) * up
            g = tl.load(base, mask=mask, other=0.0)
            u = tl.load(base + dm, mask=mask, other=0.0)
            sf, uf = _silu_rounded(g, hip).to(tl.float32), u.to(tl.float32)
            v = _round_product(sf * uf, _bits(sf) ^ _bits(uf), g, hip)
        elif act == 1:  # [hidden | gate]: silu(gate) * hidden
            u = tl.load(base, mask=mask, other=0.0)
            g = tl.load(base + dm, mask=mask, other=0.0)
            sf, uf = _silu_rounded(g, hip).to(tl.float32), u.to(tl.float32)
            v = _round_product(sf * uf, _bits(sf) ^ _bits(uf), g, hip)
        else:  # gelu, tanh approximation
            g = tl.load(base, mask=mask, other=0.0)
            xf = g.to(tl.float32)
            inner = 0.7978845608028654 * (xf + 0.044715 * (xf * xf * xf))
            # sign of x: 1 + tanh is never below +0
            v = _round_product(0.5 * xf * (1.0 + libdevice.tanh(inner)), _bits(xf), g, hip)
        out_cols = dx + cols
    vf = v.to(tl.float32)
    s = tl.load(scale_ptr)
    if hip:
        # kitchen's HIP quantizer: fast-math division, NaN stays NaN
        q = libdevice.fast_dividef(vf, tl.zeros([block_m, block_n], tl.float32) + s)
        c = tl.minimum(tl.maximum(q, -fp8_max), fp8_max)
        c = tl.where(q != q, q, c)
    else:
        # kitchen's CUDA quantizer: fp32 division, fmaxf / fminf clamp (NaN -> -max)
        q = vf / s
        c = tl.minimum(tl.maximum(q, -fp8_max), fp8_max)
    tl.store(out_ptr + rows64[:, None] * stride_out + out_cols[None, :],
             c.to(out_ptr.dtype.element_ty), mask=mask)


def _rows(t: torch.Tensor) -> torch.Tensor:
    """2D (rows, last dim) view with unit column stride; copies only if needed."""
    t2 = t.reshape(-1, t.shape[-1])
    return t2 if t2.stride(-1) == 1 else t2.contiguous()


def act_cat_quantize_fp8(
    x: torch.Tensor,
    h: torch.Tensor,
    scale: torch.Tensor,
    act: str = "swiglu",
    output_type: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    dx = x.shape[-1]
    dm = h.shape[-1] // ACT_CAT_ACTS[act]
    x2, h2 = _rows(x), _rows(h)
    m = x2.shape[0]
    out = torch.empty((m, dx + dm), dtype=output_type, device=x.device)
    if m > 0 and dx + dm > 0:
        grid = (triton.cdiv(m, BLOCK_M), triton.cdiv(dx, BLOCK_N) + triton.cdiv(dm, BLOCK_N))
        _act_cat_quant_kernel[grid](
            x2, h2, out, scale, m,
            x2.stride(0), h2.stride(0), out.stride(0),
            dx=dx, dm=dm, act=_ACT_CODE[act], fp8_max=F8_E4M3_MAX, hip=_HIP,
            block_m=BLOCK_M, block_n=BLOCK_N, num_warps=NUM_WARPS,
            # The activation formulas are matched with FMA contraction on (the
            # default before Triton 3.10, and how the PyTorch kernels are built).
            enable_fp_fusion=True,
        )
    return out.view(*x.shape[:-1], dx + dm)
