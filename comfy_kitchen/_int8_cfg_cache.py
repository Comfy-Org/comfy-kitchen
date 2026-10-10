"""Per-shape INT8 GEMM CUTLASS config cache.

Loaded once at import; consulted by `int8_linear` on the CUDA backend to
override the heuristic with a benchmarked winner when we have one for the
current (M, N, K, out_dtype).

Cache source precedence:
  1. Path in COMFY_KITCHEN_INT8_CFG_CACHE env var, if set and the file exists
     (a missing file is logged and the search continues).
  2. `a6000_int8_cfg_table.json` inside the package (shipped in wheels) if present.
  3. Otherwise empty; the existing heuristic in C++ wins.

The JSON schema matches what `samples/int8_autotune_sweep.py` writes:

  {
    "device": "NVIDIA RTX A6000",
    "sm_version": "8.6",
    "multiprocessor_count": 84,
    "shapes": {
      "<M>x<N>x<K>": {"best_cfg": int, "best_ms": float, ...},
      ...
    }
  }

Consumed: "shapes" (the picks), "sm_version" and "multiprocessor_count" (which
devices the picks may serve); the swept M range is derived from the entries.
Everything else is informational.
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
_cache_sms: int | None = None  # multiprocessor_count the table was swept on, if recorded
_device_matches: dict[int, bool] = {}  # per device index: may this table serve it?
_m_min_swept: int | None = None
_m_max_swept: int | None = None
_range_warning_emitted: set[str] = set()  # one-shot per direction


def _default_cache_path() -> Path | None:
    env = os.environ.get("COMFY_KITCHEN_INT8_CFG_CACHE")
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p
        # A missing override must not silently disable the cache: say so and fall
        # through to the packaged table.
        _logger.warning(
            "COMFY_KITCHEN_INT8_CFG_CACHE=%s not found; using the packaged table if present", p
        )

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
    global _load_attempted, _loaded_from, _cache_sm, _cache_sms, _m_min_swept, _m_max_swept
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
    shapes = data.get("shapes") if isinstance(data, dict) else None
    if not isinstance(shapes, dict):
        _logger.warning("INT8 cfg cache %s: not an object with a 'shapes' object; ignoring", path)
        return

    sm = data.get("sm_version", "")
    _cache_sm = sm if isinstance(sm, str) and sm else None
    sms = data.get("multiprocessor_count")
    _cache_sms = sms if type(sms) is int and sms > 0 else None
    count = 0
    skipped = 0
    for key, entry in shapes.items():
        # A malformed entry must never reach the inference path: skip it, don't raise.
        best = entry.get("best_cfg") if isinstance(entry, dict) else None
        if best is None:
            continue
        try:
            m_s, n_s, k_s = key.split("x")
            m, n, k = int(m_s), int(n_s), int(k_s)
            cfg = int(best)
        except (TypeError, ValueError):
            skipped += 1
            continue
        if cfg < 0:
            skipped += 1
            continue
        # The sweep was run against bf16 (out_dtype_code=2). If we ever sweep
        # other dtypes, distinguish in the cache key. For now assume bf16.
        # An index beyond the compiled config list is declined by the extension
        # at launch, and the caller then falls back to the heuristic.
        _loaded[(m, n, k, 2)] = cfg
        count += 1
    if skipped:
        _logger.warning("INT8 cfg cache %s: skipped %d malformed entries", path, skipped)

    # The swept M range comes from the entries just loaded, not from the file's
    # informational m_min_swept/m_max_swept keys: a hand-edited table cannot then
    # put a string or a stale bound into the inference-path comparison.
    if _loaded:
        _m_min_swept = min(key[0] for key in _loaded)
        _m_max_swept = max(key[0] for key in _loaded)
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
    the cached file was authored for different silicon than `device_index`.
    Callers should fall back to the existing heuristic on None.
    """
    if not _load_attempted:
        _load()
    if not _loaded:
        return None
    fits = _device_matches.get(device_index)
    if fits is None:
        fits = _device_matches[device_index] = _table_fits_device(device_index)
    if not fits:
        return None
    return _loaded.get((m, n, k, out_dtype_code))


def _table_fits_device(device_index: int) -> bool:
    """Per-shape winners are silicon-specific: the table only serves the SM
    version it was swept on and, when it recorded one, a device with about the
    same multiprocessor count (every sm86 die from GA106 to GA102 shares the
    version; a 28-SM card must not run picks timed on 84). A file that omits
    these fields means "trust me". Decided once per device index."""
    try:
        cur = torch.cuda.get_device_capability(device_index)
        if _cache_sm and f"{cur[0]}.{cur[1]}" != _cache_sm:
            return False
        if _cache_sms:
            sms = torch.cuda.get_device_properties(device_index).multi_processor_count
            if abs(sms - _cache_sms) > _cache_sms // 10:
                _logger.warning(
                    "[int8_cfg_cache] table was swept on a %d-SM device, cuda:%d has %d SMs; "
                    "not using its per-shape picks there",
                    _cache_sms,
                    device_index,
                    sms,
                )
                return False
    except Exception:
        return False  # can't even ask the device; don't guess
    return True


def reset() -> None:
    """Drop the loaded cache (for tests)."""
    global _loaded, _load_attempted, _loaded_from, _cache_sm, _cache_sms, _device_matches
    global _m_min_swept, _m_max_swept, _range_warning_emitted
    _loaded = {}
    _load_attempted = False
    _loaded_from = None
    _cache_sm = None
    _cache_sms = None
    _device_matches = {}
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
    # WARNING like the load message: visible without logging configuration, and
    # bounded to one line per direction per process.
    _logger.warning(
        "int8 GEMM M=%d is %s swept range [%d, %d]; consider re-running "
        "int8_autotune_sweep.py with shapes covering this M (heuristic may "
        "be suboptimal here)",
        m,
        direction,
        _m_min_swept,
        _m_max_swept,
    )
