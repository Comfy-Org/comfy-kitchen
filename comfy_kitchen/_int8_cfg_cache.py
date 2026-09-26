"""Per-shape INT8 GEMM CUTLASS config cache.

Loaded once at import; consulted by `int8_linear` on the CUDA backend to
override the heuristic with a benchmarked winner when we have one for the
current (M, N, K, out_dtype).

Cache source precedence:
  1. Path in COMFY_KITCHEN_INT8_CFG_CACHE env var, if set.
  2. `a6000_int8_cfg_table.json` inside the package (shipped in wheels) if present.
  3. Otherwise empty; the existing heuristic in C++ wins.

The JSON schema matches what `int8_autotune_sweep.py` writes:

  {
    "device": "NVIDIA RTX A6000",
    "sm_version": "8.6",
    "shapes": {
      "<M>x<N>x<K>": {"best_cfg": int, "best_ms": float, ...},
      ...
    }
  }

Only the entry under "shapes" is consumed; the rest is informational.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import torch

_logger = logging.getLogger("comfy_kitchen.int8_cfg_cache")

# Map from (m, n, k, out_dtype_code) -> cfg index.
_loaded: dict[tuple[int, int, int, int], int] = {}
_load_attempted = False  # true after the first load attempt, hit or miss
_loaded_from: Path | None = None
_cache_sm: str | None = None  # sm_version from the JSON (e.g. "8.6")
_m_min_swept: int | None = None
_m_max_swept: int | None = None
_range_warning_emitted: set[str] = set()  # one-shot per direction


def _default_cache_path() -> Path | None:
    env = os.environ.get("COMFY_KITCHEN_INT8_CFG_CACHE")
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p
        _logger.warning("COMFY_KITCHEN_INT8_CFG_CACHE=%s not found", p)
        return None

    # Look inside the package first (this path ships in wheels), then beside
    # the package (dev checkouts, where int8_autotune_sweep.py writes).
    here = Path(__file__).resolve().parent
    for candidate in (
        here / "a6000_int8_cfg_table.json",
        here.parent / "a6000_int8_cfg_table.json",
    ):
        if candidate.is_file():
            return candidate
    return None


def _load() -> None:
    global _load_attempted, _loaded_from, _cache_sm, _m_min_swept, _m_max_swept
    if _load_attempted:
        return
    _load_attempted = True
    path = _default_cache_path()
    if path is None:
        return
    try:
        with path.open() as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        _logger.warning("failed to load INT8 cfg cache %s: %s", path, e)
        return

    sm = data.get("sm_version", "")
    _cache_sm = sm or None
    shapes = data.get("shapes", {})
    count = 0
    for key, entry in shapes.items():
        if not isinstance(entry, dict):
            continue
        best = entry.get("best_cfg")
        if best is None:
            continue
        try:
            m_s, n_s, k_s = key.split("x")
            m, n, k = int(m_s), int(n_s), int(k_s)
        except ValueError:
            continue
        # The sweep was run against bf16 (out_dtype_code=2). If we ever sweep
        # other dtypes, distinguish in the cache key. For now assume bf16.
        _loaded[(m, n, k, 2)] = int(best)
        count += 1

    _m_min_swept = data.get("m_min_swept")
    _m_max_swept = data.get("m_max_swept")
    _loaded_from = path
    # WARNING level (not INFO) so it's visible by default in ComfyUI startups
    # without needing to configure logging — this is a "yes your setup is
    # actually doing the thing" signal users are likely to want.
    _logger.warning(
        "[int8_cfg_cache] loaded %d entries from %s (sm=%s, swept M in [%s, %s])",
        count,
        path,
        sm,
        _m_min_swept,
        _m_max_swept,
    )


def get_cfg(m: int, n: int, k: int, out_dtype_code: int, device_index: int = 0) -> int | None:
    """Return the cached cfg index for (m, n, k, out_dtype_code), or None.

    Returns None when no cache is loaded, the shape isn't in the cache, or
    the cached file was authored for a different SM than `device_index`.
    Callers should fall back to the existing heuristic on None.
    """
    if not _load_attempted:
        _load()
    if not _loaded:
        return None

    # Only consult the cache on the SM it was benchmarked on — per-shape
    # winners are silicon-specific. Allow the file to omit sm_version; that
    # means "trust me".
    if _cache_sm:
        try:
            cur = torch.cuda.get_device_capability(device_index)
            cur_str = f"{cur[0]}.{cur[1]}"
            if cur_str != _cache_sm:
                return None
        except Exception:
            return None  # can't even ask for the capability; don't guess

    return _loaded.get((m, n, k, out_dtype_code))


def reset() -> None:
    """Drop the loaded cache (for tests)."""
    global _loaded, _load_attempted, _loaded_from, _cache_sm, _range_warning_emitted
    global _m_min_swept, _m_max_swept
    _loaded = {}
    _load_attempted = False
    _loaded_from = None
    _cache_sm = None
    _range_warning_emitted = set()
    _m_min_swept = None
    _m_max_swept = None


def check_m_in_swept_range(m: int) -> None:
    """One-shot log when runtime M falls outside the swept range.

    Doesn't change behavior — the entry either hits the shape-exact cache or
    falls through to the heuristic — but tells users when their workload is
    beyond the tuning set, which is the "you should re-tune" signal.
    """
    if not _load_attempted:
        _load()
    if _m_min_swept is None or _m_max_swept is None:
        return
    if _m_min_swept <= m <= _m_max_swept:
        return
    direction = "below" if m < _m_min_swept else "above"
    if direction in _range_warning_emitted:
        return
    _range_warning_emitted.add(direction)
    _logger.info(
        "int8 GEMM M=%d is %s swept range [%d, %d]; consider re-running "
        "int8_autotune_sweep.py with shapes covering this M (heuristic may "
        "be suboptimal here)",
        m,
        direction,
        _m_min_swept,
        _m_max_swept,
    )
