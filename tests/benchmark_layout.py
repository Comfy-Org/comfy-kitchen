# SPDX-License-Identifier: Apache-2.0
import torch
import json
import statistics
from pathlib import Path

import comfy_kitchen as ck

r = Path(__file__).resolve().parent.parent / "docs" / "benchmarks"
report = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "cases": []}
for h, n in [(8, 8192), (56, 14850), (56, 32700), (56, 87142), (56, 90461)]:
    torch.manual_seed(42)
    q = torch.randn(1, h, n, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    packed = ck.prequantize_int8_attention(q, k, v)
    del q, k, v

    def old(packed=packed):
        return ck.int8_attention_from_prequantized(packed).transpose(1, 2).contiguous()

    def new(packed=packed):
        return ck.int8_attention_from_prequantized(packed, output_layout="BSHD")

    a = old()
    b = new()
    assert torch.equal(a, b)
    del a, b
    graphs = {}
    outputs = {}
    for key, fn in [("reorder", old), ("direct", new)]:
        for _ in range(3):
            fn()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            outputs[key] = fn()
        graphs[key] = g
    samples = {key: [] for key in graphs}
    for i in range(8):
        for key in ["reorder", "direct"] if i % 2 else ["direct", "reorder"]:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(4):
                graphs[key].replay()
            end.record()
            end.synchronize()
            samples[key].append(start.elapsed_time(end) / 4)
    med = {key: statistics.median(vals) for key, vals in samples.items()}
    row = {
        "shape": [1, h, n, 128],
        "samples_ms": samples,
        "median_ms": med,
        "reduction_percent": 100 * (1 - med["direct"] / med["reorder"]),
        "equal": torch.equal(outputs["reorder"], outputs["direct"]),
    }
    report["cases"].append(row)
    (r / "layout-benchmark.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(row), flush=True)
    del outputs, graphs, g, packed, old, new, fn
