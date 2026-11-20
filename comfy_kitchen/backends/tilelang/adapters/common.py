# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared by the TileLang adapter kernels: compile cache, padding, legality checks, whole-op candidates."""

import threading
from collections.abc import Callable

import tilelang
import torch

from comfy_kitchen.backends.triton.adapters.common import measurer
from comfy_kitchen.backends.triton.adapters.tuning import TORCH, tuner

__all__ = ["measurer"]

_CACHE: dict[tuple, object] = {}
_KEY_LOCKS: dict[tuple, threading.Lock] = {}
_LOCK = threading.Lock()

TRITON = {"impl": "triton"}


def whole_ops(dev) -> list[dict]:
    """Whole-op candidates timed against the TileLang configs: triton where tl.dot has MMA, torch."""
    return [TRITON, TORCH] if dev.mma else [TORCH]


# T.gemm's smallest legal extent on every target: ranks and small factors pad to it.
MIN_DIM = 16


def pad_to(n: int, m: int = MIN_DIM) -> int:
    return -(-n // m) * m


def partitions(threads: int, *gemms: tuple[int, int]) -> bool:
    """Whether every (M, N) T.gemm tile splits evenly over the program's warps,
    each owning a multiple of 16 rows and 8 columns (TileLang's MMA partition);
    a config that fails this does not compile."""
    warps = threads // 32
    for m, n in gemms:
        if not any(
            warps % wm == 0 and m % (16 * wm) == 0 and n % (8 * (warps // wm)) == 0
            for wm in range(1, warps + 1)
        ):
            return False
    return True


def fits(dev, fp16_elements: int) -> bool:
    """Whether shared tiles of this many fp16 elements fit one program: a
    launch past the device's opt-in limit fails (64 KiB on Turing, 96 on V100)."""
    return 2 * fp16_elements + 1024 <= dev.smem


def arch(device: torch.device) -> str:
    major, minor = torch.cuda.get_device_capability(device)
    return f"sm_{major}{minor}"


class CompileError(RuntimeError):
    """A config TileLang could not lower for this device."""


def _pass_configs(target_arch: str) -> dict:
    """Warp specialization and TMA lowering only on sm_90, the arch with setmaxnreg and TMA."""
    if target_arch == "sm_90":
        return {}
    return {"tl.disable_warp_specialized": True, "tl.disable_tma_lower": True}


def compiled(builder: Callable, device: torch.device, *args):
    """builder(*args) compiled for this device's arch, once per process; a failed build is remembered."""
    target_arch = arch(device)
    key = (builder.__module__, builder.__qualname__, target_arch, *args)
    kernel = _CACHE.get(key)
    if kernel is None:
        with _LOCK:
            lock = _KEY_LOCKS.setdefault(key, threading.Lock())
        with lock:
            kernel = _CACHE.get(key)
            if kernel is None:
                try:
                    kernel = tilelang.compile(
                        builder(*args),
                        target={"kind": "cuda", "arch": target_arch},
                        pass_configs=_pass_configs(target_arch),
                    )
                except Exception as e:
                    kernel = CompileError(
                        f"{builder.__qualname__}{args}: {type(e).__name__}: {e}"
                    )
                _CACHE[key] = kernel
    if isinstance(kernel, CompileError):
        raise kernel
    return kernel


def guarded(dev, key, cfg: dict, run: Callable[[dict], object]):
    """run(cfg); a config that does not compile here runs the torch composition
    instead, remembered per key."""
    try:
        return run(cfg)
    except CompileError:
        if cfg == TORCH:
            raise
        tuner.demote(dev, key)
        return run(TORCH)


def padded_cols(t: torch.Tensor, cols: int) -> torch.Tensor:
    """t (rows, c) zero-padded to (rows, cols) contiguous; t itself when it fits."""
    if t.shape[1] == cols and t.is_contiguous():
        return t
    out = torch.zeros(t.shape[0], cols, device=t.device, dtype=t.dtype)
    out[:, : t.shape[1]] = t
    return out


def padded_rows(t: torch.Tensor, rows: int) -> torch.Tensor:
    """t (r, cols) zero-padded to (rows, cols) contiguous; t itself when it fits."""
    if t.shape[0] == rows and t.is_contiguous():
        return t
    out = torch.zeros(rows, t.shape[1], device=t.device, dtype=t.dtype)
    out[: t.shape[0]] = t
    return out
