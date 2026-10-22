"""In-process autotune of INT8 GEMM CUTLASS configs for sm86 (A6000).

Uses _C.benchmark_cutlass_int8_dequant_config to time every cfg on each
shape. No subprocess, no env vars, no rebuild needed — everything runs in
this process against the already-compiled _C extension.

IMPORTANT — cwd shadowing: do NOT run this from inside the comfy-kitchen
source dir; if you do, Python resolves `import comfy_kitchen` to the source
tree (which has _C = None until setup.py builds it). Run from /tmp or
another neutral cwd so the installed wheel is picked up:

    cd /tmp && CUDA_VISIBLE_DEVICES=<idx> python /path/to/samples/int8_autotune_sweep.py

Writes results to comfy_kitchen/a6000_int8_cfg_table.json — the copy the runtime
loader reads first and the wheel ships, so a sweep is never left out of a build —
with the schema:
  {
    "device": "NVIDIA RTX A6000",
    "sm_version": "8.6",
    "shapes": {
      "<M>x<N>x<K>": {
        "m": ..., "n": ..., "k": ...,
        "per_cfg_ms": {"0": ..., "1": ..., ...},
        "best_cfg": int,
        "best_ms": float
      },
      ...
    }
  }

Usage:
  python int8_autotune_sweep.py                       # full sweep (recommended)
  python int8_autotune_sweep.py --shapes ltx          # only LTX shapes
  python int8_autotune_sweep.py --shapes minimax      # only MiniMax shapes
  python int8_autotune_sweep.py --shapes wan          # only Wan 2.2 14B shapes
  python int8_autotune_sweep.py --shapes mid          # only the mid-M bridge series
  python int8_autotune_sweep.py --cfgs 0 1 12 13      # subset of cfgs
  python int8_autotune_sweep.py --iters 200           # more iters (default 100)
  python int8_autotune_sweep.py --out table.json      # different output path
  python int8_autotune_sweep.py --resume              # skip shapes already in the table

An existing table at --out is merged, not replaced: a partial sweep (one shape
group, --resume) rewrites only the shapes it timed and keeps the rest, and each
entry records its own `swept_at` date. A table swept on another SM, or a --cfgs
subset against an existing table, is refused (name a different --out).

Run with the GPU otherwise idle. Safe to interrupt mid-sweep — the JSON is
checkpointed after each shape; continue with --resume.

Multi-GPU note: pin the target with CUDA_VISIBLE_DEVICES so device 0 below
is the card you actually want, e.g.:
  CUDA_VISIBLE_DEVICES=<idx> python int8_autotune_sweep.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# cwd/source-tree shadowing fix: if a copy of this script is run from the repo
# root (`python /path/to/int8_autotune_sweep.py`), Python puts that directory
# at sys.path[0], which means `import comfy_kitchen` resolves to the SOURCE
# tree (no _C extension). Remove that entry so the installed wheel wins.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
if (_HERE / "comfy_kitchen").is_dir() and (_HERE / "setup.py").exists():
    sys.path = [p for p in sys.path if Path(p).resolve() != _HERE]

import torch  # noqa: E402  (must come after the sys.path fixup above)

from comfy_kitchen.backends.cuda import _C, _wrap_for_dlpack  # noqa: E402

if _C is None:
    print(
        "ERROR: comfy_kitchen.backends.cuda._C is None — the compiled CUDA "
        "extension isn't loaded. Run from site-packages, not the source tree.",
        file=sys.stderr,
    )
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parents[1]  # samples/ -> repo root
# The packaged copy is what _int8_cfg_cache loads first and what the wheel ships.
DEFAULT_OUT = REPO_ROOT / "comfy_kitchen" / "a6000_int8_cfg_table.json"

# ---------------------------------------------------------------------------
# Shapes actually observed in the wild (COMFY_KITCHEN_LOG_INT8_PATH=1 prints them)
#
# N and K are fixed by the model architecture (hidden_dim, qkv_size, ffn_dim).
# M is tokens-in-flight and varies with resolution x frames x CFG.
# ---------------------------------------------------------------------------

LTX_SHAPES = [
    # (M, N, K) — LTX 2.5
    (1024, 4096, 4096),
    (1024, 16384, 4096),
    (1024, 4096, 16384),
    (1024, 2048, 2048),
    (1024, 8192, 2048),
    (1024, 2048, 8192),
    (25900, 4096, 4096),
    (274, 2048, 2048),
    (25900, 2048, 4096),
    (25900, 4096, 2048),
    (25900, 16384, 4096),
    (25900, 4096, 16384),
    (274, 8192, 2048),
    (274, 2048, 8192),
]

MINIMAX_SHAPES = [
    # (M, N, K) — MiniMax H3
    (1797, 6144, 2048),
    (1797, 2048, 2048),
    (1797, 16384, 2048),
    (1797, 2048, 8192),
    (53730, 21504, 5376),  # QKV
    (53730, 5376, 7168),
    (53730, 28672, 5376),  # MLP_up
    (53730, 5376, 14336),  # MLP_down (tall-K)
    (74977, 21504, 5376),  # QKV
    (74977, 5376, 7168),
    (74977, 28672, 5376),  # MLP_up
    (74977, 5376, 14336),  # MLP_down (tall-K)
    (80666, 21504, 5376),  # QKV
    (80666, 5376, 7168),
    (80666, 28672, 5376),  # MLP_up
    (80666, 5376, 14336),  # MLP_down (tall-K)
]

WAN_SHAPES = [
    # (M, N, K) — Wan 2.2 14B: attention q/k/v/o (5120x5120) and FFN up/down
    # (13824x5120, 5120x13824). At 1280x736, M=161920 is 44 latent frames (175 frames)
    # in one pass and M=77280 one 21-latent-frame (81-frame) context window; the
    # cross-attention k/v projections run on the 512 text tokens.
    (161920, 5120, 5120),
    (161920, 13824, 5120),
    (161920, 5120, 13824),
    (77280, 5120, 5120),
    (77280, 13824, 5120),
    (77280, 5120, 13824),
    (512, 5120, 5120),
]

MID_M_SHAPES = [
    # (M, N, K) — mid-M series bridging the LTX and MiniMax bands (2026-08-29 sweep)
    (2048, 4096, 4096),
    (2048, 16384, 4096),
    (4096, 4096, 4096),
    (4096, 16384, 4096),
    (8192, 4096, 4096),
    (8192, 16384, 4096),
    (16384, 4096, 4096),
    (16384, 16384, 4096),
]

# 14 cfgs defined in cutlass_gemm_int8.cu's dispatch_fused_no_bias_config
ALL_CFGS = list(range(14))

BF16_CODE = 2  # matches DTYPE_TO_CODE[torch.bfloat16]; the production path


# ---------------------------------------------------------------------------
# Bench
# ---------------------------------------------------------------------------


def make_tensors(m: int, n: int, k: int, device: torch.device) -> dict:
    """Pre-allocate inputs. Use realistic int8 magnitude (-127..127)."""
    return {
        "x_q": torch.randint(-127, 128, (m, k), dtype=torch.int8, device=device),
        "w": torch.randint(-127, 128, (n, k), dtype=torch.int8, device=device),
        "xs": torch.full((m, 1), 0.01, dtype=torch.float32, device=device),
        "ws": torch.full((n,), 0.01, dtype=torch.float32, device=device),
        "out": torch.empty((m, n), dtype=torch.bfloat16, device=device),
    }


def bench_cfg(
    tensors: dict,
    cfg: int,
    iters: int,
    stream: torch.cuda.Stream,
) -> float | None:
    """Time a cfg on the pre-allocated tensors. Returns avg ms, or None if fails."""
    ms_total = _C.benchmark_cutlass_int8_dequant_config(
        _wrap_for_dlpack(tensors["x_q"]),
        _wrap_for_dlpack(tensors["w"]),
        _wrap_for_dlpack(tensors["xs"]),
        _wrap_for_dlpack(tensors["ws"]),
        _wrap_for_dlpack(tensors["out"]),
        BF16_CODE,
        cfg,
        iters,
        stream.cuda_stream,
    )
    if ms_total < 0:
        return None
    return ms_total / iters


def sweep_shape(
    m: int,
    n: int,
    k: int,
    cfgs: list[int],
    iters: int,
    warmup: int,
    device: torch.device,
) -> dict:
    print(f"\n=== M={m}  N={n}  K={k}  (warmup={warmup}, iters={iters}) ===", flush=True)
    t0 = time.time()
    tensors = make_tensors(m, n, k, device)
    stream = torch.cuda.current_stream(device)

    per_cfg: dict[str, float] = {}
    for cfg in cfgs:
        # Warmup (also catches can_implement / workspace alloc failures)
        warm_ms = bench_cfg(tensors, cfg, warmup, stream)
        if warm_ms is None:
            print(f"  cfg={cfg:2d}  FAIL (can_implement or init)", flush=True)
            continue
        # Timed run
        ms = bench_cfg(tensors, cfg, iters, stream)
        if ms is None:
            print(f"  cfg={cfg:2d}  FAIL (timed pass)", flush=True)
            continue
        per_cfg[str(cfg)] = ms
        print(f"  cfg={cfg:2d}  {ms:.4f} ms", flush=True)

    # Cleanup
    del tensors
    torch.cuda.empty_cache()

    if not per_cfg:
        print("  → no cfg succeeded", flush=True)
        return {
            "m": m,
            "n": n,
            "k": k,
            "per_cfg_ms": {},
            "best_cfg": None,
            "best_ms": None,
        }

    best_cfg_str = min(per_cfg, key=per_cfg.get)
    best_cfg = int(best_cfg_str)
    best_ms = per_cfg[best_cfg_str]
    sorted_cfgs = sorted(per_cfg.items(), key=lambda kv: kv[1])
    runner_up_ms = sorted_cfgs[1][1] if len(sorted_cfgs) > 1 else None
    margin = ((runner_up_ms / best_ms) - 1.0) * 100 if runner_up_ms else None
    margin_str = f"  (next-best +{margin:.1f}% slower)" if margin else ""
    print(f"  → BEST: cfg={best_cfg} @ {best_ms:.4f} ms{margin_str}", flush=True)
    print(f"  elapsed: {time.time() - t0:.1f} s", flush=True)

    return {
        "m": m,
        "n": n,
        "k": k,
        "per_cfg_ms": per_cfg,
        "best_cfg": best_cfg,
        "best_ms": best_ms,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


SHAPE_GROUPS = {
    "ltx": LTX_SHAPES,
    "minimax": MINIMAX_SHAPES,
    "wan": WAN_SHAPES,
    "mid": MID_M_SHAPES,
}


def load_existing_table(path: Path, sm_version: str) -> dict[str, dict]:
    """Entries of an existing table at `path`, to be merged with this sweep.

    A table for another SM, or a file that is not a table, is never merged into or
    overwritten: the caller must name a different --out (or fix the file).
    """
    if not path.is_file():
        return {}
    try:
        with path.open() as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        sys.exit(f"ERROR: cannot read {path} ({e}); fix it or pass --out <new file>.")
    shapes = data.get("shapes") if isinstance(data, dict) else None
    if not isinstance(shapes, dict):
        sys.exit(f"ERROR: {path} is not a sweep table (no 'shapes' object); pass --out <new file>.")
    existing_sm = data.get("sm_version")
    if existing_sm != sm_version:
        sys.exit(
            f"ERROR: {path} was swept on sm {existing_sm}, this device is sm {sm_version}; "
            f"pass --out <new file> instead of merging into it."
        )
    return dict(shapes)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", choices=["all", *SHAPE_GROUPS], default="all")
    ap.add_argument("--cfgs", type=int, nargs="+", default=ALL_CFGS)
    ap.add_argument("--iters", type=int, default=100, help="timed iterations per cfg")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="skip shapes already present in --out")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: no CUDA device")
        sys.exit(1)
    device = torch.device("cuda", args.device)
    props = torch.cuda.get_device_properties(device)
    cap = torch.cuda.get_device_capability(device)
    if cap != (8, 6):
        print(
            f"WARNING: device is sm{cap[0]}{cap[1]} ({props.name}); this script is tuned for sm86"
        )
        print("         results are still meaningful per shape; pass --out to name the table")

    groups = SHAPE_GROUPS if args.shapes == "all" else [args.shapes]
    shapes = [shape for group in groups for shape in SHAPE_GROUPS[group]]

    sm_version = f"{cap[0]}.{cap[1]}"
    # A partial sweep (one shape group, or --resume) must not drop the other groups'
    # entries from an existing table: start from them and replace only what is swept.
    results = load_existing_table(args.out, sm_version)
    if set(args.cfgs) != set(ALL_CFGS) and results:
        sys.exit(
            f"ERROR: --cfgs {args.cfgs} times only part of the palette; its best_cfg cannot "
            f"replace the full entries in {args.out}. Pass --out <new file>."
        )
    if args.resume:
        shapes = [s for s in shapes if f"{s[0]}x{s[1]}x{s[2]}" not in results]

    print(f"Device: {props.name}  sm{cap[0]}{cap[1]}")
    print(
        f"Sweeping {len(shapes)} shapes  x  {len(args.cfgs)} cfgs  @ warmup={args.warmup} iters={args.iters}"
    )
    print(f"  Output: {args.out}  ({len(results)} existing entries kept)")
    print("  GPU should be otherwise idle for reproducibility\n")

    swept_at = time.strftime("%Y-%m-%d")
    for m, n, k in shapes:
        r = sweep_shape(m, n, k, args.cfgs, args.iters, args.warmup, device)
        r["swept_at"] = swept_at
        results[f"{m}x{n}x{k}"] = r
        # checkpoint after each shape
        # Min/max M they've actually benchmarked. The runtime cache loader
        # warns once when ComfyUI dispatches an M outside this range so users
        # know their workload has drifted from the tuning set.
        m_vals = [r["m"] for r in results.values()]
        payload = {
            "device": props.name,
            "sm_version": sm_version,
            # the loader serves the table only to devices with about this many SMs
            "multiprocessor_count": props.multi_processor_count,
            "total_memory_mib": props.total_memory // (1024 * 1024),
            "warmup_iters": args.warmup,
            "timed_iters": args.iters,
            "m_min_swept": min(m_vals) if m_vals else None,
            "m_max_swept": max(m_vals) if m_vals else None,
            "shapes": results,
        }
        # Write-then-rename so a process loading the table mid-sweep never sees a
        # half-written file (the loader is one-shot per process).
        tmp = args.out.with_name(args.out.name + ".tmp")
        with tmp.open("w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, args.out)

    # Summary table at the end
    print("\n=== SUMMARY ===")
    print(f"{'shape':>22}  {'best':>4}  {'best ms':>9}    runner-ups")
    for key, r in results.items():
        if r["best_cfg"] is None:
            print(f"{key:>22}  FAIL")
            continue
        per = r["per_cfg_ms"]
        top3 = sorted(per.items(), key=lambda kv: kv[1])[:3]
        runner_ups = ", ".join(f"cfg{c}={v:.3f}" for c, v in top3[1:])
        print(f"{key:>22}  {r['best_cfg']:>4d}  {r['best_ms']:>9.4f}    {runner_ups}")

    print(f"\nWrote {args.out}")
    print("\nNext: load this from int8_linear via comfy_kitchen/_int8_cfg_cache.py")


if __name__ == "__main__":
    main()
