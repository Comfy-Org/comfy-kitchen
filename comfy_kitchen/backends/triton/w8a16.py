# SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""W8A16 linear (triton): unscaled FP8 weight read directly, 16-bit activations.

    out[M, N] = x[M, K] @ weight[N, K].to(x.dtype)^T (+ bias), fp32 accumulate

The eager path (and ComfyUI's manual-cast path for unscaled ``*_fp8_e4m3fn``
checkpoints) writes a bf16/fp16 copy of the whole weight on every call and then
runs a 16-bit GEMM on it. Here the kernel loads the FP8 bytes (half of the 16-bit
weight traffic) and converts each tile to the activation dtype in registers,
right before ``tl.dot``. e4m3 -> bf16 / fp16 is exact, so the dot operands are
the ones the eager path multiplies; the bias is added to the fp32 accumulator
before the single rounding to the output dtype (as a GEMM bias epilogue does).
Results differ from cuBLAS / hipBLASLt only by the fp32 summation order.

One kernel, two shapes of work, chosen per config (constexpr ``gemv``):

- tiled GEMM: grouped tile order, M and N masked, K stepped by block_k (K % block_k == 0).
- GEMV (a few rows): one program per (row, block_n outputs); [block_n, block_k] FP8 tiles are
  converted to fp32, multiplied by the x chunk and row-summed at the end. Bound
  by reading the FP8 weight (modulation / time-embedding layers, M = 1, and the
  short text streams of CFG's unconditional pass).

Plain Triton only (tl.load / tl.dot / tl.sum and dtype casts; no TMA, tensor
descriptors or inline asm), so the same kernel compiles for NVIDIA and AMD GPUs.

Configs are picked by ``triton.autotune``. Rows <= SMALL_M use one list (GEMV
configs plus small GEMM tiles), more rows another (GEMM tiles), keyed on N, K
and the row count (rounded up to a power of two above SMALL_M, so a new text
length does not trigger a new tuning run). Configs whose block_k does not divide K,
or whose pipelined tiles do not fit the device's shared memory (64 KB of LDS on
gfx1151 vs ~99 KB on an RTX 5090), are pruned before tuning; every list has
a small block_k 32 entry, so K % K_MULTIPLE == 0 always leaves at least one.
"""

import functools
import inspect

import torch

import triton
import triton.language as tl

K_MULTIPLE = 32
# Up to this many rows the small-M list (GEMV + small GEMM tiles) is autotuned.
SMALL_M = 8
# ROCm: the archs this kernel is served on (exact gcnArchName, feature suffixes
# stripped) and, per arch, the most rows it is served for. Every other ROCm arch,
# and more rows, stay on eager (cast + hipBLASLt). Measured on an AMD Radeon 8060S
# (gfx1151): the kernel wins while the GEMM is small (it reads half the weight
# bytes), hipBLASLt's bf16 GEMM once the weight read is amortized. Per shape, the
# kernel's time vs the cast path at 384 / 512 rows: 3072 -> 3072 0.79x / 1.02x,
# 3072 -> 12288 0.64x / 0.79x, 12288 -> 3072 0.46x / 0.62x; so 384 rows, below
# the first measured loss.
HIP_MAX_ROWS = {"gfx1151": 384}
# Fewer weight elements than this stay on eager too: for a weight that small (the
# gfx1151 measurement: 256 -> 3072, 0.8 MB of FP8) the saved weight read is worth
# less than the triton launch path costs on the CPU, and a back-to-back call was
# 1.41x slower than the cast path (768 -> 3072, 2.4 MB: 0.82x, faster).
HIP_MIN_WEIGHT = {"gfx1151": 1 << 21}


def _mm(bm, bn, bk, group_m, num_warps, num_stages):
    return triton.Config(
        {"gemv": False, "block_m": bm, "block_n": bn, "block_k": bk, "group_size_m": group_m},
        num_warps=num_warps, num_stages=num_stages,
    )


def _gemv(bn, bk, num_warps, num_stages=3):
    return triton.Config(
        {"gemv": True, "block_m": 1, "block_n": bn, "block_k": bk, "group_size_m": 1},
        num_warps=num_warps, num_stages=num_stages,
    )


# GEMM tiles for more than SMALL_M rows. Tuned on NVIDIA RTX 5090 (sm_120) at
# the Qwen-Image (img M 6889, text M 262 / 46) and Flux.1 dev (img M 4096 /
# 4352, text M 256) linear shapes.
_LARGE_M_CONFIGS = [
    _mm(128, 128, 32, 8, 8, 5),  # large M, K 3072 (q/k/v/out, FFN up, Flux linear1)
    _mm(128, 128, 128, 4, 8, 3),  # large M, K 12288 (FFN down)
    _mm(128, 128, 128, 8, 8, 3),  # large M, K 15360 (Flux linear2); text M 256 x N 9216
    _mm(128, 64, 64, 8, 4, 3),  # K 64 (Qwen-Image img_in)
    _mm(32, 64, 64, 8, 4, 4),  # N 64 (Qwen-Image proj_out)
    _mm(32, 64, 128, 8, 4, 3),  # text M ~256 at N 3072
    _mm(64, 128, 64, 8, 4, 3),  # text M 262, FFN up
    _mm(64, 64, 64, 8, 4, 3),  # text M 256, FFN up (Flux)
    _mm(16, 64, 128, 8, 4, 3),  # M 9..64 (Qwen-Image 2512 uncond text M 46)
    # Tuned on AMD Radeon 8060S (gfx1151) at the text M 256 / 262 shapes. Of the tiles
    # above, only the 128 x 128 x block_k 128 3-stage ones exceed its 64 KB of LDS
    # (98304 B by _smem_bytes) and are pruned there.
    _mm(64, 128, 64, 8, 4, 2),  # N 3072 / 12288 at K 3072, K 3584
    _mm(128, 64, 64, 8, 4, 2),  # N 9216 / 12288 (Flux)
    _mm(128, 128, 64, 8, 8, 2),  # K 12288 / 4096
]

# Up to SMALL_M rows: GEMV configs (bound by reading the FP8 weight) plus small
# GEMM tiles. Tuned on NVIDIA RTX 5090 (sm_120).
_SMALL_M_CONFIGS = [
    _gemv(8, 512, 8),  # M 1, N 18432 / 6144 (modulation, norm_out)
    _gemv(8, 512, 4),  # M 1, N 3072 / 9216
    _gemv(32, 256, 8),  # M 5, 3072 x 3072
    _gemv(64, 256, 8),  # M 5, K 12288 / 3584
    _gemv(32, 128, 8),  # M 1, K 256 / 768 (time / vector embedding)
    _mm(16, 64, 128, 8, 4, 3),  # M 5, FFN up (N 12288)
    _mm(16, 64, 32, 8, 4, 3),  # any K % 32 == 0
    # Tuned on AMD Radeon 8060S (gfx1151): uncond text M 5, FFN up / down.
    _mm(16, 64, 256, 8, 4, 2),
]


@triton.jit
def _mm_tiles(
    x_ptr, w_ptr, b_ptr, o_ptr, m, n, k, stride_xm, stride_wn, stride_om,
    has_bias: tl.constexpr, block_m: tl.constexpr, block_n: tl.constexpr,
    block_k: tl.constexpr, group_size_m: tl.constexpr,
):
    # Grouped tile order: group_size_m row tiles sweep the N tiles together for L2 reuse.
    pid = tl.program_id(0)
    num_m = tl.cdiv(m, block_m)
    num_n = tl.cdiv(n, block_n)
    group = group_size_m * num_n
    first_m = (pid // group) * group_size_m
    gsz = tl.minimum(num_m - first_m, group_size_m)
    pm = first_m + (pid % group) % gsz
    pn = (pid % group) // gsz

    rm = pm * block_m + tl.arange(0, block_m)
    rn = pn * block_n + tl.arange(0, block_n)
    rk = tl.arange(0, block_k)
    m_mask = rm[:, None] < m
    n_mask = rn[None, :] < n
    a_p = x_ptr + rm[:, None].to(tl.int64) * stride_xm + rk[None, :]
    # [block_k, block_n] view of the row-major [N, K] FP8 weight
    b_p = w_ptr + rn[None, :].to(tl.int64) * stride_wn + rk[:, None]
    dt = o_ptr.dtype.element_ty
    acc = tl.zeros([block_m, block_n], tl.float32)
    for _ in range(0, k, block_k):
        a = tl.load(a_p, mask=m_mask, other=0.0)
        b = tl.load(b_p, mask=n_mask, other=0.0)
        acc = tl.dot(a, b.to(dt), acc)  # exact e4m3 -> bf16 / fp16 in registers
        a_p += block_k
        b_p += block_k
    if has_bias:
        acc = acc + tl.load(b_ptr + rn, mask=rn < n, other=0.0).to(tl.float32)[None, :]
    o_p = o_ptr + rm[:, None].to(tl.int64) * stride_om + rn[None, :]
    tl.store(o_p, acc.to(dt), mask=m_mask & n_mask)


@triton.jit
def _gemv_rows(
    x_ptr, w_ptr, b_ptr, o_ptr, n, k, stride_xm, stride_wn, stride_om,
    has_bias: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr,
):
    pn = tl.program_id(0)
    m = tl.program_id(1)
    rn = pn * block_n + tl.arange(0, block_n)
    rk = tl.arange(0, block_k)
    n_mask = rn < n
    w_p = w_ptr + rn[:, None].to(tl.int64) * stride_wn + rk[None, :]
    x_p = x_ptr + m.to(tl.int64) * stride_xm + rk
    acc = tl.zeros([block_n, block_k], tl.float32)
    for _ in range(0, k, block_k):
        w = tl.load(w_p, mask=n_mask[:, None], other=0.0).to(tl.float32)
        x = tl.load(x_p).to(tl.float32)
        acc += w * x[None, :]
        w_p += block_k
        x_p += block_k
    out = tl.sum(acc, axis=1)
    if has_bias:
        out = out + tl.load(b_ptr + rn, mask=n_mask, other=0.0).to(tl.float32)
    tl.store(o_ptr + m.to(tl.int64) * stride_om + rn, out.to(o_ptr.dtype.element_ty), mask=n_mask)


@triton.jit
def _linear_w8a16_kernel(
    x_ptr, w_ptr, b_ptr, o_ptr, m, n, k, m_bucket,
    stride_xm, stride_wn, stride_om,
    has_bias: tl.constexpr, gemv: tl.constexpr,
    block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr,
    group_size_m: tl.constexpr,
):
    # m_bucket is only an autotune key (row count rounded up); unused here.
    if gemv:
        _gemv_rows(x_ptr, w_ptr, b_ptr, o_ptr, n, k, stride_xm, stride_wn, stride_om,
                   has_bias, block_n, block_k)
    else:
        _mm_tiles(x_ptr, w_ptr, b_ptr, o_ptr, m, n, k, stride_xm, stride_wn, stride_om,
                  has_bias, block_m, block_n, block_k, group_size_m)


@functools.cache
def _max_shared_mem(device_index: int) -> int:
    props = triton.runtime.driver.active.utils.get_device_properties(device_index)
    return props["max_shared_mem"]


def _smem_bytes(config) -> int:
    """Shared memory of a config's pipelined K loop: num_stages - 1 buffered copies of
    the 16-bit x tile and the FP8 weight tile. (That is what Triton reports for the
    configs that do not fit: block_m 128 x block_n 128 x block_k 128 at 3 stages needs
    98304 B.)"""
    kw = config.kwargs
    if kw["gemv"]:
        per_stage = kw["block_n"] * kw["block_k"] + 2 * kw["block_k"]
    else:
        per_stage = 2 * kw["block_m"] * kw["block_k"] + kw["block_k"] * kw["block_n"]
    return max(config.num_stages - 1, 1) * per_stage


def _prune(configs, named_args, **kwargs):
    """Configs that tile K evenly and fit the device's shared memory (LDS)."""
    k = named_args["k"]
    tiled = [c for c in configs if k % c.kwargs["block_k"] == 0]
    limit = _max_shared_mem(named_args["x_ptr"].device.index)
    fit = [c for c in tiled if _smem_bytes(c) <= limit]
    return fit or [min(tiled, key=_smem_bytes)]


_HIP = bool(getattr(torch.version, "hip", None))
_FLUSH_BYTES = 256 * 2**20


def _bench_read_flush(kernel_call, quantiles, reps=25):
    """Autotune timing used under ROCm (ms at the given quantiles).

    Triton's default do_bench evicts the caches by zero-filling a 256 MB buffer
    before every launch. On gfx1151 (an APU with a 32 MB last-level cache) the timed
    kernel then runs behind the write-back of those dirty lines: times come out 2-3x
    high and in a different order than in steady state, and the autotuner picked
    configs 10-38% slower than the fastest at the text (M 256 / 262) shapes. Reading
    the buffer instead evicts the caches without leaving dirty lines; that ranking
    matched the one measured with rotating weight copies (the model's access
    pattern) at every shape checked.
    """
    flush = torch.ones(_FLUSH_BYTES // 4, dtype=torch.int32, device="cuda")
    kernel_call()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        flush.sum()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        kernel_call()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    t = torch.tensor(times)
    return [torch.quantile(t, q).item() for q in quantiles]


# triton.autotune takes do_bench since Triton 3.2; older versions, and CUDA, keep
# Triton's default timer.
_AUTOTUNE_TIMER = (
    {"do_bench": _bench_read_flush}
    if _HIP and "do_bench" in inspect.signature(triton.autotune).parameters
    else {}
)


def _autotuned(configs, key):
    return triton.autotune(
        configs=configs, key=key, prune_configs_by={"early_config_prune": _prune},
        **_AUTOTUNE_TIMER,
    )(_linear_w8a16_kernel)


_small_m_kernel = _autotuned(_SMALL_M_CONFIGS, ["m", "n", "k"])
_large_m_kernel = _autotuned(_LARGE_M_CONFIGS, ["m_bucket", "n", "k"])


def _grid(m, n):
    def grid(meta):
        if meta["gemv"]:
            return (triton.cdiv(n, meta["block_n"]), m)
        return (triton.cdiv(m, meta["block_m"]) * triton.cdiv(n, meta["block_n"]), 1)

    return grid


def linear_w8a16(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """``linear(x, weight.to(x.dtype), bias.to(x.dtype))`` reading the FP8 weight directly."""
    k = x.shape[-1]
    n = weight.shape[0]
    x2 = x.reshape(-1, k)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    m = x2.shape[0]
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    if m == 0 or n == 0:
        return out.view(*x.shape[:-1], n)
    if bias is not None:
        # The reference rounds the bias to x's dtype first; FP8 -> x.dtype is exact
        # and done in the kernel, anything else is rounded here (N elements).
        if bias.dtype not in (x.dtype, torch.float8_e4m3fn):
            bias = bias.to(x.dtype)
        bias = bias.contiguous()
    if m <= SMALL_M:
        kernel, m_bucket = _small_m_kernel, m
    else:
        kernel, m_bucket = _large_m_kernel, triton.next_power_of_2(m)
    kernel[_grid(m, n)](
        x2, weight, weight if bias is None else bias, out, m, n, k, m_bucket,
        x2.stride(0), weight.stride(0), out.stride(0),
        has_bias=bias is not None,
    )
    return out.view(*x.shape[:-1], n)
