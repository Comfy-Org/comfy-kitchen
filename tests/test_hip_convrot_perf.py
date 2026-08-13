"""Performance microbenchmarks for HIP ConvRot INT8 quant (fused/global vs eager).

Runs on HIP hardware without external models. Compares the HIP C++ kernel against
the eager Python ``quantize_and_rotate_rowwise`` path on Z-Image Turbo INT8 shapes.
"""

from __future__ import annotations

import os
import statistics
import time

import pytest
import torch

from comfy_kitchen.backends.eager.quantization import (
    quantize_and_rotate_rowwise as eager_quantize_and_rotate_rowwise,
)
from comfy_kitchen.tensor.int8_utils import _build_hadamard

from tests.test_hip_wmma import DEV, _unavailable_reason, needs_wmma

_UNAVAILABLE = _unavailable_reason()

pytestmark = [
    pytest.mark.skipif(_UNAVAILABLE is not None, reason=_UNAVAILABLE or ""),
    pytest.mark.performance,
    pytest.mark.slow,
    needs_wmma,
]

# Z-Image Turbo INT8 activation quant shapes (G=256 bf16).
ZIMAGE_FUSED_SHAPE = (4128, 3840)       # fused LDS path on RDNA4
ZIMAGE_LARGE_K_SHAPE = (4128, 10240)    # large K; often still fused on dGPU (64 KB LDS)
ZIMAGE_SPILL_SHAPE = (4128, 17408)      # K above fused LDS ceiling -> global spill

COL_FUSED_GLOBAL = "Fused/global"
COL_EAGER_CONVROT = "Eager ConvRot"


def _sync(device: torch.device | str = DEV) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _mean_ms(times_s: list[float]) -> float:
    return statistics.mean(times_s) * 1000.0


def _uplift_pct(baseline: float, improved: float, *, lower_is_better: bool = True) -> float:
    if baseline <= 0:
        return 0.0
    if lower_is_better:
        return (baseline - improved) / baseline * 100.0
    return (improved - baseline) / baseline * 100.0


def _speedup(baseline: float, improved: float, *, lower_is_better: bool = True) -> float:
    if improved <= 0:
        return 0.0
    if lower_is_better:
        return baseline / improved
    return improved / baseline


def _print_uplift_table(
    title: str,
    rows: list[dict[str, str]],
    *,
    col_left: str,
    col_right: str,
) -> None:
    headers = ("Benchmark", col_left, col_right, "Uplift")
    col_widths = [
        max(len(headers[0]), *(len(r["name"]) for r in rows)),
        max(len(headers[1]), *(len(r["left"]) for r in rows)),
        max(len(headers[2]), *(len(r["right"]) for r in rows)),
        max(len(headers[3]), *(len(r["uplift"]) for r in rows)),
    ]

    def _row(cells: tuple[str, str, str, str]) -> str:
        return (
            f"| {cells[0]:<{col_widths[0]}} | "
            f"{cells[1]:>{col_widths[1]}} | "
            f"{cells[2]:>{col_widths[2]}} | "
            f"{cells[3]:>{col_widths[3]}} |"
        )

    rule = "|-" + "-|-".join("-" * w for w in col_widths) + "-|"
    print(f"\n## {title}")
    print(_row(headers))
    print(rule)
    for row in rows:
        print(_row((row["name"], row["left"], row["right"], row["uplift"])))


def _bench_hip_convrot_quant(
    hip,
    x: torch.Tensor,
    group_size: int,
    *,
    warmup: int,
    iters: int,
) -> float:
    m, k = x.shape
    q = torch.empty((m, k), dtype=torch.int8, device=x.device)
    scales = torch.empty((m,), dtype=torch.float32, device=x.device)
    spill_rotated = torch.empty((m, k), dtype=x.dtype, device=x.device)
    spill_partials = torch.empty((m, k // 256), dtype=torch.float32, device=x.device)
    for _ in range(warmup):
        hip._C.quantize_int8_convrot(
            hip._dl(x),
            hip._dl(q),
            hip._dl(scales),
            hip._dl(spill_rotated),
            hip._dl(spill_partials),
            m,
            k,
            group_size,
            0,
            hip._stream(x),
        )
    _sync(x.device)
    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        hip._C.quantize_int8_convrot(
            hip._dl(x),
            hip._dl(q),
            hip._dl(scales),
            hip._dl(spill_rotated),
            hip._dl(spill_partials),
            m,
            k,
            group_size,
            0,
            hip._stream(x),
        )
        _sync(x.device)
        times.append(time.perf_counter() - t0)
    return _mean_ms(times)


def _bench_eager_convrot_quant(
    x: torch.Tensor,
    group_size: int,
    *,
    warmup: int,
    iters: int,
) -> float:
    h = _build_hadamard(group_size, device=x.device, dtype=x.dtype)
    for _ in range(warmup):
        eager_quantize_and_rotate_rowwise(x, h, group_size)
    _sync(x.device)
    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        eager_quantize_and_rotate_rowwise(x, h, group_size)
        _sync(x.device)
        times.append(time.perf_counter() - t0)
    return _mean_ms(times)


@pytest.fixture
def hip():
    from comfy_kitchen.backends import hip as hip_backend

    return hip_backend


@pytest.mark.parametrize(
    ("shape", "label"),
    [
        (ZIMAGE_FUSED_SHAPE, "fused_k3840"),
        (ZIMAGE_LARGE_K_SHAPE, "large_k10240"),
        (ZIMAGE_SPILL_SHAPE, "spill_k17408"),
    ],
    ids=["fused_k3840", "large_k10240", "spill_k17408"],
)
def test_convrot_quant_hip_faster_than_eager(hip, shape, label):
    """HIP fused/global ConvRot quant should beat the eager Python path on Z-Image shapes."""
    m, k = shape
    torch.manual_seed(0)
    x = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)

    warmup, iters = 10, 30
    fused_ms = _bench_hip_convrot_quant(hip, x, 256, warmup=warmup, iters=iters)
    eager_ms = _bench_eager_convrot_quant(x, 256, warmup=warmup, iters=iters)
    speedup = _speedup(eager_ms, fused_ms)
    uplift_pct = _uplift_pct(eager_ms, fused_ms)

    _print_uplift_table(
        f"ConvRot quant microbench ({label}, M={m} K={k} G=256)",
        [
            {
                "name": "Quant latency",
                "left": f"{fused_ms:.2f} ms",
                "right": f"{eager_ms:.2f} ms",
                "uplift": f"{uplift_pct:+.1f}% ({speedup:.2f}x)",
            }
        ],
        col_left=COL_FUSED_GLOBAL,
        col_right=COL_EAGER_CONVROT,
    )

    min_speedup = os.environ.get("CONVROT_PERF_MIN_SPEEDUP")
    if min_speedup is not None:
        threshold = float(min_speedup)
        assert speedup >= threshold, (
            f"expected fused/global quant >= {threshold:.2f}x faster than eager ConvRot for "
            f"{label}, got {speedup:.2f}x (fused/global={fused_ms:.2f} ms, "
            f"eager={eager_ms:.2f} ms)"
        )
