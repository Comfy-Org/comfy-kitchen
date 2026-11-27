"""The denominators, measured on this card this run: DRAM bandwidth and matmul rate."""

import functools

import torch
import triton

from .device import API, KIND, arch, l2_bytes, properties, units
from .timing import device_ms


def _stream(n: int) -> dict:
    # Random data: constant buffers compress in memory on some GPUs (Xe2) and move faster than real ones.
    a = torch.empty(n, dtype=torch.float32, device=KIND)
    b = torch.randn(n, dtype=torch.float32, device=KIND)
    c = torch.randn(n, dtype=torch.float32, device=KIND)
    out = torch.empty((), dtype=torch.float32, device=KIND)
    nbytes = n * 4
    # A compute-kernel copy: XPU runs copy_ on a copy engine the timing events do not bracket.
    copy = device_ms(lambda: torch.mul(b, 1.0, out=a), iters=5)[0]
    read = device_ms(lambda: torch.sum(b, dim=0, out=out), iters=5)[0]
    triad = device_ms(lambda: torch.add(b, c, alpha=2.0, out=a), iters=5)[0]
    return {
        "copy": 2 * nbytes / (copy * 1e-3) / 1e9,
        "read": nbytes / (read * 1e-3) / 1e9,
        "triad": 3 * nbytes / (triad * 1e-3) / 1e9,
    }


@functools.cache
def stream_bandwidths(gib: float = 1.0) -> dict:
    """GB/s of copy, read and triad over buffers far larger than L2."""
    res = _stream(int(gib * (1 << 30)) // 4)
    API.empty_cache()
    return res


@functools.cache
def matmul_tflops(dtype: torch.dtype) -> float:
    """Best vendor-GEMM rate over square GEMMs (fp32 accumulate): the achievable ceiling."""
    best = 0.0
    for n in (2048, 4096, 8192):
        a = torch.randn(n, n, device=KIND, dtype=dtype)
        b = torch.randn(n, n, device=KIND, dtype=dtype)
        o = torch.empty(n, n, device=KIND, dtype=dtype)
        ms = device_ms(lambda a=a, b=b, o=o: torch.mm(a, b, out=o), iters=10)[0]
        best = max(best, 2 * n**3 / (ms * 1e-3) / 1e12)
        del a, b, o
    API.empty_cache()
    return best


def device_meta(dtype: torch.dtype) -> dict:
    """Machine facts plus the measured ceilings, so a later plot names its denominators."""
    bw = stream_bandwidths()
    return {
        "device": properties().name,
        "arch": arch(),
        "sms": units(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "dtype": str(dtype).removeprefix("torch."),
        "dram_gbps": max(bw.values()),
        "bw_patterns": bw,
        "mamf_tflops": matmul_tflops(dtype),
        "l2_mib": l2_bytes() / 2**20,
    }
