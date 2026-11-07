"""Measure the adapter ops: every arm, every sweep, one JSON per family.

    python -m samples.adapter_bench.run --out samples/adapter_bench/out \
        [--family lora,lokr] [--sweep tokens,rank] [--arms comfy,kitchen] [--quick]

Writes <out>/<arch>/<family>.json after every case, so a crash loses at most
the case it hit. Plot with samples.adapter_bench.plot.plot_all.
"""

import argparse
import gc
import time
import traceback
from pathlib import Path

import torch

from .core.case import measure_case
from .core.ceilings import device_meta
from .core.device import API, capability
from .core.io import save_rows
from .families import FAMILIES

ARMS = ("comfy", "compile", "eager", "kitchen", "lycoris")


def default_dtype() -> torch.dtype:
    """bf16 where the matrix units take it (sm_80+, XPU), fp16 on Volta and Turing."""
    cap = capability()
    return torch.float16 if cap is not None and cap < (8, 0) else torch.bfloat16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("samples/adapter_bench/out"))
    ap.add_argument("--family", default=",".join(FAMILIES))
    ap.add_argument(
        "--sweep", default="", help="comma list; empty runs every sweep of each family"
    )
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--dtype", choices=("bfloat16", "float16"))
    ap.add_argument("--quick", action="store_true", help="every other sweep point")
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype) if args.dtype else default_dtype()
    arms = set(args.arms.split(","))
    meta = device_meta(dtype)
    print(
        f"{meta['device']} {meta['arch']} {meta['dtype']}: DRAM {meta['dram_gbps']:.0f} GB/s "
        f"(best of {meta['bw_patterns']}), matmul {meta['mamf_tflops']:.1f} TFLOP/s",
        flush=True,
    )
    wanted = [s for s in args.sweep.split(",") if s]
    for fam_name in args.family.split(","):
        fam = FAMILIES[fam_name]
        rows: list[dict] = []
        path = args.out / meta["arch"] / f"{fam_name}.json"
        fam_meta = {
            **meta,
            "family": fam_name,
            "arms": sorted(arms),
            "quick": args.quick,
        }
        started = time.time()
        for sweep in fam.SWEEPS:
            if wanted and sweep not in wanted:
                continue
            for build in fam.cases(sweep, dtype, args.quick, arms):
                case = build()
                try:
                    rows += measure_case(case, dtype, meta)
                except torch.OutOfMemoryError as e:
                    print(
                        f"{case.op} {case.case}: out of memory, skipped ({str(e)[:120]})",
                        flush=True,
                    )
                except (
                    Exception
                ):  # the case is lost, the sweep goes on; the log has why
                    print(f"{case.op} {case.case}: failed, skipped", flush=True)
                    traceback.print_exc()
                del case
                gc.collect()
                API.empty_cache()
                fam_meta["seconds"] = time.time() - started
                save_rows(path, fam_meta, rows)
        print(f"{fam_name}: {len(rows)} rows -> {path}", flush=True)


if __name__ == "__main__":
    main()
