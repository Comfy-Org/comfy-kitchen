# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the adapter kernels: tiles, padding, precision, keys, measuring."""

import contextlib
import itertools
import math
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import torch
import triton
import triton.language as tl
from triton.runtime.errors import OutOfResources

from .tuning import TORCH, DeviceInfo, buffer_sets, time_candidates, tuner

LAUNCH_KEYS = ("num_warps", "num_stages")
# Relative Frobenius distance a candidate may sit from the torch composition, per dtype.
_AGREE_TOL = {torch.float32: 2e-3, torch.float16: 1e-2, torch.bfloat16: 4e-2}


@triton.jit
def load2d(ptr, rows, cols, s_row, s_col, m_row, m_col):
    """The (rows, cols) tile at ptr, zero where either mask is off."""
    return tl.load(
        ptr + rows[:, None] * s_row + cols[None, :] * s_col,
        mask=m_row[:, None] & m_col[None, :],
        other=0.0,
    )


def pad16(n: int) -> int:
    """n rounded up to a power of two, at least 16."""
    return max(16, triton.next_power_of_2(max(1, n)))


def token_bucket(t: int) -> int:
    """Token count bucket: power of two up to 4096, then multiples of 4096."""
    if t <= 4096:
        return triton.next_power_of_2(max(1, t))
    return -(-t // 4096) * 4096


def dtype_key(dtype: torch.dtype) -> str:
    """Cache key of a dtype: its bit width."""
    return f"b{torch.finfo(dtype).bits}"


def dot_precision(dev: DeviceInfo, dtype: torch.dtype) -> str:
    """tl.dot input precision: tf32 for 16-bit, torch's matmul precision for fp32."""
    if dtype != torch.float32:
        return "tf32"
    if dev.tf32 and torch.get_float32_matmul_precision() != "highest":
        return "tf32"
    return "ieee"


def launch(cfg: dict) -> dict:
    """The launch options of a config."""
    return {k: cfg[k] for k in LAUNCH_KEYS if k in cfg}


def default_stages(dev: DeviceInfo) -> int:
    """Pipeline depth: 3 on sm_80+, else 2."""
    return 3 if dev.capability is not None and dev.capability >= (8, 0) else 2


def stage_axis(dev: DeviceInfo) -> tuple[int, ...]:
    """Pipeline depths a config may use on this device."""
    return (2, 3, 4) if default_stages(dev) == 3 else (1, 2)


def grid_configs(dev: DeviceInfo, **axes) -> list[dict]:
    """Every combination of the named axes, plus the device's stage axis."""
    if "num_stages" not in axes:
        axes["num_stages"] = stage_axis(dev)
    names = list(axes)
    return [
        dict(zip(names, values, strict=True))
        for values in itertools.product(*axes.values())
    ]


def copy_of(t: torch.Tensor | None) -> torch.Tensor | None:
    """A same-layout copy of t (never fresh random data: tuning leaves the RNG alone)."""
    return None if t is None else torch.empty_like(t).copy_(t)


def guarded(dev: DeviceInfo, key, cfg: dict, run: Callable[[dict], object]):
    """run(cfg), or run(TORCH) from now on for key when cfg does not fit the device."""
    try:
        return run(cfg)
    except OutOfResources:
        tuner.demote(dev, key)
        return run(TORCH)


def _tensors(result) -> list[torch.Tensor]:
    if isinstance(result, torch.Tensor):
        return [result]
    return [t for t in result if isinstance(t, torch.Tensor) and t.numel() > 0]


def disagreement(got, ref) -> str | None:
    """None when got matches the torch composition's ref, else the mismatch."""
    for n, (g, r) in enumerate(zip(_tensors(got), _tensors(ref), strict=True)):
        tol = _AGREE_TOL.get(r.dtype, _AGREE_TOL[torch.bfloat16])
        g, r = g.float(), r.float()
        if not torch.isfinite(g).all():
            return f"output {n} not finite"
        norm = r.norm()
        if norm > 0 and (gap := ((g - r).norm() / norm).item()) > tol:
            return f"output {n} off torch by {gap:.2e} (tolerance {tol:.0e})"
    return None


def agrees(got, ref) -> bool:
    """Whether got matches the torch composition's ref."""
    return disagreement(got, ref) is None


def _warm(run, cfg: dict, inputs: tuple) -> None:
    """Runs cfg once so its kernels are compiled; a failure surfaces again when it is validated."""
    with contextlib.suppress(Exception):
        run(cfg, inputs)()


def measurer(
    dev: DeviceInfo,
    bytes_per_call: int,
    make_inputs: Callable[[], tuple],
    run: Callable[[dict, tuple], Callable[[], object]],
) -> Callable[[list[dict]], list[float]]:
    """The tuner's measure(): per-config seconds, inf for a config that disagrees with torch."""
    state = {}

    def measure(configs: list[dict]) -> list[float]:
        if not state:
            # Input copies and the torch reference; on OOM nothing stays allocated and the tuner falls back.
            try:
                state["sets"] = [
                    make_inputs() for _ in range(buffer_sets(bytes_per_call, dev))
                ]
                state["ref"] = run(TORCH, state["sets"][0])()
            except torch.OutOfMemoryError:
                state.clear()
                raise
        sets = state["sets"]

        # Compile every kernel config of the shortlist concurrently, each worker on the operands' device.
        kernels = [c for c in configs if "impl" not in c]
        if len(kernels) > 1:
            api = getattr(torch, dev.kind)
            api.synchronize(dev.index)

            def warm(c):
                with api.device(dev.index):
                    _warm(run, c, sets[0])
                    api.synchronize(dev.index)

            with ThreadPoolExecutor(len(kernels)) as pool:
                list(pool.map(warm, kernels))

        valid = []
        for c in configs:
            if c == TORCH:
                valid.append(True)
                continue
            try:
                gap = disagreement(run(c, sets[0])(), state["ref"])
            except Exception as e:
                gap = f"{type(e).__name__}: {str(e)[:300]}"
            valid.append(gap is None)
            if gap is not None:
                tuner.note_failure(c, gap)
        makers = [lambda j, c=c: run(c, sets[j]) for c in configs]
        errors: list[str | None] = []
        times = time_candidates(makers, len(sets), dev, errors=errors)
        for c, err in zip(configs, errors, strict=True):
            if err is not None:
                tuner.note_failure(c, err)
        return [t if ok else math.inf for t, ok in zip(times, valid, strict=True)]

    return measure
