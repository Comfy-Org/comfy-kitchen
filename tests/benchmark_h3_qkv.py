# SPDX-License-Identifier: Apache-2.0
import json
import statistics
import torch
from pathlib import Path

from test_h3_qkv import operands
from comfy_kitchen.h3_qkv import _fallback, _op

r = Path(__file__).resolve().parents[1] / "docs" / "benchmarks"
r.mkdir(parents=True, exist_ok=True)

report = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "records": []}
for m in [4096, 4097, 8192, 14850, 32700, 87142, 90461]:
    args = operands(m)
    functions = {}
    outputs = {}
    times = {"stock": [], "fused": []}
    for key, fn in [("stock", _fallback), ("fused", _op)]:
        for _ in range(3):
            fn(*args, 1e-5)
        outputs[key] = fn(*args, 1e-5)
        functions[key] = fn
    equal = {
        name: torch.equal(a, b)
        for name, a, b in zip(
            ["q", "k", "v", "qs", "ks", "vs"], outputs["stock"], outputs["fused"], strict=True
        )
    }
    for rep in range(8):
        for key in ["stock", "fused"] if rep % 2 else ["fused", "stock"]:
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(4):
                functions[key](*args, 1e-5)
            b.record()
            b.synchronize()
            times[key].append(a.elapsed_time(b) / 4)
    med = {key: statistics.median(val) for key, val in times.items()}
    row = {
        "m": m,
        "samples_ms": times,
        "median_ms": med,
        "reduction_percent": 100 * (1 - med["fused"] / med["stock"]),
        "equal": equal,
    }
    report["records"].append(row)
    (r / "h3-qkv-benchmark.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(row), flush=True)
    del outputs, functions, args
    torch.cuda.empty_cache()
