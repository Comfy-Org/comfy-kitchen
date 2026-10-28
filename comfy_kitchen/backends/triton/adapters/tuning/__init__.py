# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config selection for the adapter kernels: an analytical model ranks, a few run.

Per key (kernel, dtype class and flags, shape bucket), once per process:

1. the winner cached on disk for this GPU arch, when there is one;
2. otherwise the kernel's planner orders its legal configs with the
   analytical model (model.py); TILE_SHORTLIST tile shapes spread over the
   tile axes in the model's order, plus the whole-op alternatives (the torch
   composition, LoHa's materialize forms), are timed once on fresh copies of
   the inputs, then LAUNCH_SHORTLIST launch settings of the fastest tile; the
   fastest of all is cached;
3. inside CUDA-graph capture or torch.compile tracing, where nothing can be
   timed, and with COMFY_KITCHEN_ADAPTER_TUNE=off, the model's top config.

The cache lives in COMFY_KITCHEN_ADAPTER_TUNE_CACHE (a directory), default
~/.cache/comfy_kitchen/adapters, one JSON file per arch.
"""

import contextlib
import os
import threading
from collections.abc import Callable
from pathlib import Path

import torch
import triton

from . import model
from .device import DeviceInfo, device_info
from .table import read_rows, write_rows
from .timing import buffer_sets, time_candidates

__all__ = [
    "LAUNCH_SHORTLIST",
    "TILE_SHORTLIST",
    "TORCH",
    "DeviceInfo",
    "buffer_sets",
    "device_info",
    "select",
    "set_mode",
    "time_candidates",
    "tuner",
]

TORCH = {"impl": "torch"}
# Kernel configs timed per key: tile shapes, then launch settings of the fastest tile.
TILE_SHORTLIST = 4
LAUNCH_SHORTLIST = 3
# Config keys that set how a tile is launched, not its shape.
_LAUNCH_KEYS = frozenset({"num_warps", "num_stages", "threads"})

Key = tuple[str, tuple, tuple]


def _freeze(config: dict) -> tuple:
    return tuple(sorted(config.items()))


def _capturing(dev: DeviceInfo) -> bool:
    if dev.kind == "cuda" and torch.cuda.is_current_stream_capturing():
        return True
    return torch.compiler.is_compiling()


def _default_cache_dir() -> Path:
    env = os.environ.get("COMFY_KITCHEN_ADAPTER_TUNE_CACHE")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "comfy_kitchen" / "adapters"


class Tuner:
    def __init__(self):
        self.mode = os.environ.get("COMFY_KITCHEN_ADAPTER_TUNE", "on").lower()
        self.cache_dir = _default_cache_dir()
        self._chosen: dict[tuple[str, Key], dict] = {}
        self._disk: dict[str, dict[Key, dict]] = {}
        # When a list, every tuned key's ranking and timings are appended.
        self.record: list[dict] | None = None
        self._failures: dict[tuple, str] = {}
        # Reentrant: a candidate being timed may itself select (LoHa's
        # materialize path picks a merge config).
        self._lock = threading.RLock()

    def _cache(self, dev: DeviceInfo) -> dict[Key, dict]:
        rows = self._disk.get(dev.arch)
        if rows is None:
            rows = {
                (r["kernel"], tuple(r["cat"]), tuple(r["num"])): r["config"]
                for r in read_rows(self.cache_dir / f"{dev.arch}.json")
            }
            self._disk[dev.arch] = rows
        return rows

    def select(
        self,
        dev: DeviceInfo,
        key: Key,
        plan: Callable[[], list[dict]],
        measure: Callable[[list[dict]], list[float]],
    ) -> dict:
        """The config for key. plan() returns the legal configs, model-ranked,
        with whole-op alternatives (an "impl" key) anywhere in the list."""
        memo = (dev.arch, key)
        config = self._chosen.get(memo)
        if config is not None:
            return config
        with self._lock:
            config = self._chosen.get(memo)
            if config is not None:
                return config
            ranked = plan()
            config = self._cache(dev).get(key)
            if config is not None and _freeze(config) not in {
                _freeze(c) for c in ranked
            }:
                config = None
            if config is None:
                if self.mode == "off" or _capturing(dev):
                    # Not remembered: the key is timed once timing is possible.
                    return _model_pick(ranked)
                config = self.tune(dev, key, ranked, measure)
            self._chosen[memo] = config
            return config

    def tune(self, dev: DeviceInfo, key: Key, ranked: list[dict], measure) -> dict:
        """Time the tile shortlist and the whole-op alternatives, then launch settings of the
        fastest tile -> winner."""
        self._failures = {}
        ordered = [c for c in ranked if "impl" not in c]
        tile_axes = {k for c in ordered for k in c} - _LAUNCH_KEYS
        best_per_tile, seen = [], set()
        for c in ordered:
            tile = _freeze({a: c.get(a) for a in tile_axes})
            if tile not in seen:
                seen.add(tile)
                best_per_tile.append(c)
        tiles = shortlist(best_per_tile, sorted(tile_axes), TILE_SHORTLIST)
        candidates = tiles + [c for c in ranked if "impl" in c]
        times = measure(candidates)

        # Launch settings of the fastest tile, among the configs sharing its shape.
        timed_tiles = [i for i, c in enumerate(candidates) if "impl" not in c]
        if timed_tiles:
            lead = candidates[min(timed_tiles, key=times.__getitem__)]
            same = [
                c
                for c in ordered
                if c not in candidates
                and all(c.get(a) == lead.get(a) for a in tile_axes)
            ]
            launch_axes = sorted({k for c in same for k in c} & _LAUNCH_KEYS)
            variants = shortlist([lead, *same], launch_axes, LAUNCH_SHORTLIST + 1)[1:]
            if variants:
                times += measure(variants)
                candidates += variants

        winner = candidates[min(range(len(candidates)), key=times.__getitem__)]
        if self.record is not None:
            self.record.append(
                {
                    "arch": dev.arch,
                    "key": key,
                    "timed": [[c, t] for c, t in zip(candidates, times, strict=True)],
                    "failures": [[dict(f), why] for f, why in self._failures.items()],
                }
            )
        self._cache(dev)[key] = winner
        self._persist(dev)
        return winner

    def _persist(self, dev: DeviceInfo) -> None:
        rows = [
            {"kernel": k, "cat": list(c), "num": list(n), "config": cfg}
            for (k, c, n), cfg in self._cache(dev).items()
        ]
        # A read-only home leaves the cache in memory for this process only.
        with contextlib.suppress(OSError):
            write_rows(
                self.cache_dir / f"{dev.arch}.json",
                rows,
                {"triton": triton.__version__},
            )

    def note_failure(self, config: dict, reason: str) -> None:
        """A candidate of the key being tuned failed; kept for the record."""
        self._failures[_freeze(config)] = reason

    def demote(self, dev: DeviceInfo, key: Key) -> None:
        """A config this device could not build: the key runs torch from now on."""
        self._chosen[(dev.arch, key)] = TORCH

    def reset(self) -> None:
        with self._lock:
            self._chosen.clear()
            self._disk.clear()


def shortlist(kernels: list[dict], axes: list[str], size: int) -> list[dict]:
    """size configs from the model's order: its best, then in turn per axis the best config
    with a value of that axis not yet picked; the model's order fills what is left."""
    if not kernels:
        return []
    picked = [kernels[0]]
    covered = {a: {kernels[0].get(a)} for a in axes}
    while len(picked) < size:
        added = False
        for axis in axes:
            pick = next(
                (
                    c
                    for c in kernels
                    if c not in picked and c.get(axis) not in covered[axis]
                ),
                None,
            )
            if pick is None:
                continue
            picked.append(pick)
            for a in axes:
                covered[a].add(pick.get(a))
            added = True
            if len(picked) == size:
                return picked
        if not added:
            break
    rest = [c for c in kernels if c not in picked]
    return picked + rest[: size - len(picked)]


def _model_pick(ranked: list[dict]) -> dict:
    for c in ranked:
        if "impl" not in c:
            return c
    return ranked[0] if ranked else TORCH


tuner = Tuner()


def select(dev, key, plan, measure) -> dict:
    return tuner.select(dev, key, plan, measure)


def set_mode(mode: str, cache_dir: str | None = None) -> None:
    """ "on" times the shortlist, "off" takes the model's pick, "safe" the smallest legal config."""
    tuner.mode = "off" if mode == "safe" else mode
    model.ORDER["by"] = "footprint" if mode == "safe" else "time"
    if cache_dir is not None:
        tuner.cache_dir = Path(cache_dir).expanduser()
    tuner.reset()
