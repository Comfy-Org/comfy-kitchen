"""In-process autotune of INT8 GEMM CUTLASS configs for sm86 (A6000).

Uses _C.benchmark_cutlass_int8_dequant_config to time every cfg on each
shape. No subprocess, no env vars, no rebuild needed — everything runs in
this process against the already-compiled _C extension.

IMPORTANT — cwd shadowing: do NOT run this from inside the comfy-kitchen
source dir; if you do, Python resolves `import comfy_kitchen` to the source
tree (which has _C = None until setup.py builds it). Run from /tmp or
another neutral cwd so the installed wheel is picked up:

    cd /tmp && CUDA_VISIBLE_DEVICES=<gpu> python /path/to/int8_autotune_sweep.py

Writes results to a6000_int8_cfg_table.json with the schema:
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
  python int8_autotune_sweep.py --cfgs 0 1 12 13      # subset of cfgs
  python int8_autotune_sweep.py --iters 200           # more iters (default 100)
  python int8_autotune_sweep.py --out table.json      # different output path

Run with the GPU otherwise idle. Safe to interrupt mid-sweep — the JSON is
checkpointed after each shape.

Multi-GPU note: pin the target with CUDA_VISIBLE_DEVICES so device 0 below
is the card you actually want, e.g.:
  CUDA_VISIBLE_DEVICES=<idx> python int8_autotune_sweep.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# cwd/source-tree shadowing fix: this script lives inside the comfy-kitchen
# repo. When run with `python /path/to/int8_autotune_sweep.py`, Python puts
# the script's directory at sys.path[0], which means `import comfy_kitchen`
# resolves to the SOURCE tree (no _C extension). Remove that entry so the
# installed wheel wins.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
if (_HERE / "comfy_kitchen").is_dir() and (_HERE / "setup.py").exists():
    # We're inside the repo. Remove this dir from sys.path so the wheel wins.
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

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = REPO_ROOT / "a6000_int8_cfg_table.json"

# ---------------------------------------------------------------------------
# Shapes actually observed in the wild (see probe_cutlass_sm86.py)
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

# Gap-fill sweep — M values in the previously-unmeasured 2048..16383 range,
# crossed with the two most-common (N,K) pairs in the LTX stack. Cost of
# adding these 8 shapes is roughly 30-60s; covers the heuristic boundary.
LTX_GAP_FILL = [
    (2048, 4096, 4096),
    (2048, 16384, 4096),
    (4096, 4096, 4096),
    (4096, 16384, 4096),
    (8192, 4096, 4096),
    (8192, 16384, 4096),
    (16384, 4096, 4096),
    (16384, 16384, 4096),
]

MINIMAX_SHAPES = [
    # (M, N, K) — MiniMax H3
    (74977, 21504, 5376),  # QKV
    (74977, 5376, 7168),
    (74977, 28672, 5376),  # MLP_up
    (74977, 5376, 14336),  # MLP_down (tall-K)
    (80666, 21504, 5376),  # QKV
    (80666, 5376, 7168),
    (80666, 28672, 5376),  # MLP_up
    (80666, 5376, 14336),  # MLP_down (tall-K)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", choices=["all", "ltx", "ltx-gap", "minimax"], default="all")
    ap.add_argument("--cfgs", type=int, nargs="+", default=ALL_CFGS)
    ap.add_argument("--iters", type=int, default=100, help="timed iterations per cfg")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--device", type=int, default=0)
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
        print("         results are still meaningful per-shape; filename is a lie.")

    if args.shapes in ("all", "ltx", "ltx-gap"):
        shapes = list(LTX_SHAPES)
        if args.shapes in ("all", "ltx-gap"):
            shapes += LTX_GAP_FILL
    else:
        shapes = []
    if args.shapes in ("all", "minimax"):
        shapes += MINIMAX_SHAPES

    print(f"Device: {props.name}  sm{cap[0]}{cap[1]}")
    print(
        f"Sweeping {len(shapes)} shapes  x  {len(args.cfgs)} cfgs  @ warmup={args.warmup} iters={args.iters}"
    )
    print(f"  Output: {args.out}")
    print("  GPU should be otherwise idle for reproducibility\n")

    results: dict[str, dict] = {}
    for m, n, k in shapes:
        r = sweep_shape(m, n, k, args.cfgs, args.iters, args.warmup, device)
        results[f"{m}x{n}x{k}"] = r
        # checkpoint after each shape
        # Min/max M they've actually benchmarked. The runtime cache loader
        # warns once when ComfyUI dispatches an M outside this range so users
        # know their workload has drifted from the tuning set.
        m_vals = [r["m"] for r in results.values()]
        with args.out.open("w") as f:
            json.dump(
                {
                    "device": props.name,
                    "sm_version": f"{cap[0]}.{cap[1]}",
                    "total_memory_mib": props.total_memory // (1024 * 1024),
                    "warmup_iters": args.warmup,
                    "timed_iters": args.iters,
                    "m_min_swept": min(m_vals) if m_vals else None,
                    "m_max_swept": max(m_vals) if m_vals else None,
                    "shapes": results,
                },
                f,
                indent=2,
            )

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
