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
weight rows contiguous, N % N_MULTIPLE == 0, K % K_MULTIPLE == 0. M is masked;
so is N whenever the tile width BN of the chosen config does not divide it, so
no autotune config can leave output columns unwritten.
"""

import inspect

import torch

import triton
import triton.language as tl

# Every autotune config below tiles K evenly when K_MULTIPLE holds. N tiles
# that run past N are masked (even_n below), so BN need not divide N.
N_MULTIPLE = 128
K_MULTIPLE = 64
# Below this many rows the GEMM is too small for the fused kernel to be worth
# routing to; dispatch keeps those on eager (cuBLAS / hipBLASLt + GELU).
MIN_M = 256

# Each config is tagged with the GPU archs it measured fastest, or within about
# 1% of the fastest, on at the Wan2.1 shape below. On a tagged arch the
# autotuner only times that arch's configs (_prune_to_arch); on any other arch
# it times the whole list.
_SM120 = "sm_120"  # NVIDIA RTX 5090
_GFX1151 = "gfx1151"  # AMD Radeon 8060S
_GFX1201 = "gfx1201"  # AMD Radeon AI PRO R9700

_TAGGED = [
    # Tuned on NVIDIA RTX 5090 (sm_120, ~99 KB shared memory per block) for the
    # Wan2.1-1.3B FFN up projection (M 28080, K 1536, N 8960, fp16). Fastest
    # there; also fastest of the list on AMD Radeon 8060S (gfx1151).
    (triton.Config({"block_m": 128, "block_n": 128, "block_k": 64, "group_size_m": 8}, num_warps=4, num_stages=2),
     (_SM120, _GFX1151)),
    # Smaller-footprint configs so the kernel also runs where shared memory / LDS
    # is tighter (e.g. 64 KB on AMD GPUs). Not tuned on any particular GPU.
    # Within 1% of the fastest on the RTX 5090.
    (triton.Config({"block_m": 128, "block_n": 128, "block_k": 32, "group_size_m": 8}, num_warps=4, num_stages=2),
     (_SM120,)),
    # Within 1% of the fastest on the Radeon 8060S.
    (triton.Config({"block_m": 64, "block_n": 128, "block_k": 64, "group_size_m": 8}, num_warps=4, num_stages=2),
     (_GFX1151,)),
    (triton.Config({"block_m": 128, "block_n": 64, "block_k": 32, "group_size_m": 8}, num_warps=4, num_stages=2),
     ()),
    (triton.Config({"block_m": 128, "block_n": 128, "block_k": 32, "group_size_m": 8}, num_warps=8, num_stages=2),
     ()),
    # Tuned on AMD Radeon AI PRO R9700 (gfx1201, 64 KB LDS) for the Wan2.1 shape
    # above; fastest there. BN 256 does not divide every served N (N % 128 ==
    # 0); partial N tiles are masked.
    (triton.Config({"block_m": 128, "block_n": 256, "block_k": 32, "group_size_m": 8}, num_warps=8, num_stages=1),
     (_GFX1201,)),
]

_CONFIGS = [c for c, _ in _TAGGED]


def _arch_name(target) -> str | None:
    """'sm_<major><minor>' (NVIDIA) or 'gfxNNNN' (AMD, feature suffixes such as
    ':sramecc+' dropped) of a Triton GPUTarget; None for other backends."""
    if target.backend == "cuda":
        return f"sm_{target.arch}"
    if target.backend == "hip":
        return str(target.arch).split(":")[0]
    return None


def _current_arch() -> str | None:
    try:
        return _arch_name(triton.runtime.driver.active.get_current_target())
    except Exception:  # Triton without get_current_target, or no active GPU driver
        return None


def _configs_for_arch(configs, tagged, arch):
    """The configs tagged with arch, in list order; all configs if none is."""
    allowed = [c for c, archs in tagged if arch in archs]
    return [c for c in configs if c in allowed] or configs


def _prune_to_arch(configs, named_args, **kwargs):
    # early_config_prune runs when the autotuner meets a new key, so the arch is
    # that of the device the call runs on.
    return _configs_for_arch(configs, _TAGGED, _current_arch())


def _bench_warm(kernel_call, quantiles, warmup=250, rep=100):
    """Autotune timer (ms at the given quantiles): triton.testing.do_bench's
    loop (warmup / rep in ms, one event pair per call, calls back to back)
    without its cache clear, and with a longer warmup.

    Triton's default timer zero-fills a 256 MB buffer before every timed call,
    so each call starts with cold, dirty caches, while the GEMM runs back to
    back in a model. Warmup: after an idle gap (such as compiling the config
    about to be timed) a GPU can run faster for the first 100-200 ms of load
    (about 5% on the RTX 5090), so 25 ms of warmup times every config in that
    window; 250 ms reaches the back-to-back speed.
    """
    di = triton.runtime.driver.active.get_device_interface()
    kernel_call()
    di.synchronize()
    start, end = di.Event(enable_timing=True), di.Event(enable_timing=True)
    start.record()
    for _ in range(5):
        kernel_call()
    end.record()
    di.synchronize()
    estimate = max(start.elapsed_time(end) / 5, 1e-3)
    for _ in range(max(1, int(warmup / estimate))):
        kernel_call()
    n = max(5, int(rep / estimate))
    starts = [di.Event(enable_timing=True) for _ in range(n)]
    ends = [di.Event(enable_timing=True) for _ in range(n)]
    for s, e in zip(starts, ends, strict=True):
        s.record()
        kernel_call()
        e.record()
    di.synchronize()
    times = torch.tensor([s.elapsed_time(e) for s, e in zip(starts, ends, strict=True)])
    return [torch.quantile(times, q).item() for q in quantiles]


# triton.autotune takes do_bench since Triton 3.2; older versions keep their
# default timer.
_AUTOTUNE_TIMER = (
    {"do_bench": _bench_warm} if "do_bench" in inspect.signature(triton.autotune).parameters else {}
)


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


@triton.autotune(
    configs=_CONFIGS, key=["m", "n", "k"], prune_configs_by={"early_config_prune": _prune_to_arch},
    **_AUTOTUNE_TIMER,
)
# even_n follows the BN of whichever config the autotuner runs; when BN divides
# N the kernel compiles without the N wrap and mask.
@triton.heuristics({"even_n": lambda args: args["n"] % args["block_n"] == 0})
@triton.jit
def _linear_gelu_tanh_kernel(
    x_ptr, w_ptr, b_ptr, o_ptr,
    m, n, k,
    stride_xm, stride_wn, stride_om,
    has_bias: tl.constexpr, even_n: tl.constexpr,
    block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr, group_size_m: tl.constexpr,
):
    # Grouped tile order: group_size_m row tiles sweep the N tiles together for L2 reuse.
    pid = tl.program_id(0)
    num_m = tl.cdiv(m, block_m)
    num_n = tl.cdiv(n, block_n)
    group = group_size_m * num_n
    gid = pid // group
    first_m = gid * group_size_m
    gsz = tl.minimum(num_m - first_m, group_size_m)
    pm = first_m + (pid % group) % gsz
    pn = (pid % group) // gsz

    rm = pm * block_m + tl.arange(0, block_m)
    rn = pn * block_n + tl.arange(0, block_n)
    if even_n:
        rn_ld = rn
        o_mask = rm[:, None] < m
    else:
        # Last N tile runs past N: wrap its weight / bias reads onto valid rows
        # (no masks in the K loop) and drop those columns at the store.
        rn_ld = rn % n
        o_mask = (rm[:, None] < m) & (rn[None, :] < n)
    rk = tl.arange(0, block_k)
    a_p = x_ptr + rm[:, None].to(tl.int64) * stride_xm + rk[None, :]
    b_p = w_ptr + rn_ld[None, :].to(tl.int64) * stride_wn + rk[:, None]
    m_mask = rm[:, None] < m

    acc = tl.zeros([block_m, block_n], tl.float32)
    for _ in range(0, k, block_k):
        a = tl.load(a_p, mask=m_mask, other=0.0)
        b = tl.load(b_p)
        acc = tl.dot(a, b, acc)
        a_p += block_k
        b_p += block_k

    dt = o_ptr.dtype.element_ty
    if has_bias:
        acc = acc + tl.load(b_ptr + rn_ld).to(tl.float32)[None, :]
    o = _gelu_tanh(_round_to(acc, dt)).to(dt)
    tl.store(o_ptr + rm[:, None].to(tl.int64) * stride_om + rn[None, :], o, mask=o_mask)


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
        return (triton.cdiv(m, meta["block_m"]) * triton.cdiv(n, meta["block_n"]),)

    _linear_gelu_tanh_kernel[grid](
        x2, weight, bias if has_bias else weight, out,
        m, n, k,
        x2.stride(0), weight.stride(0), out.stride(0),
        has_bias=has_bias,
    )
    return out.view(*x.shape[:-1], n)
