# SPDX-License-Identifier: Apache-2.0
import torch
import json
import statistics
from pathlib import Path

import comfy_kitchen as ck
from comfy_kitchen.backends import cuda

r = Path(__file__).resolve().parent.parent / "docs" / "benchmarks"
if (
    torch.version.git_version != "7661cd9c6b841b62b7f411aa52ec51f05457263b"
    or torch.cuda.get_device_capability() != (12, 0)
    or cuda._C is None
    or not hasattr(cuda._C, "indexed_norm_convrot")
):
    raise RuntimeError(
        "This fused benchmark requires the qualified Torch revision and SM120 extension"
    )
report = {
    "torch": torch.__version__,
    "torch_git": torch.version.git_version,
    "gpu": torch.cuda.get_device_name(),
    "implementation": "native SM120",
    "cases": [],
}
for m in [131, 8192, 14850, 32700, 87142, 90461]:
    torch.manual_seed(42)
    k = 5376
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(k, device="cuda", dtype=torch.bfloat16)
    shift = torch.randn(6, k, device="cuda", dtype=torch.float32)
    scale = torch.randn_like(shift)
    rows = torch.empty(m, device="cuda", dtype=torch.int32)
    segments = []
    for g in range(6):
        a, b = m * g // 6, m * (g + 1) // 6
        rows[a:b] = g
        segments.append((a, b, g))

    def base(x=x, k=k, w=w, segments=segments, scale=scale, shift=shift):
        y = torch.nn.functional.rms_norm(x, (k,), w, 1e-5)
        for a, b, g in segments:
            y[a:b].mul_(1.0 + scale[g].to(torch.bfloat16)).add_(shift[g].to(torch.bfloat16))
        return cuda.quantize_int8_rowwise_convrot64(y, 256)

    def fast(x=x, w=w, shift=shift, scale=scale, rows=rows):
        return ck.indexed_norm_convrot(x, w, shift, scale, rows)

    probe_q = torch.empty((m, k), dtype=torch.int8, device=x.device)
    probe_s = torch.empty((m, 1), dtype=torch.float32, device=x.device)
    assert cuda._C.indexed_norm_convrot(
        *(cuda._wrap_for_dlpack(t) for t in (x, w, shift, scale, rows, probe_q, probe_s)),
        1e-5,
        torch.cuda.current_stream().cuda_stream,
    )
    del probe_q, probe_s
    a = base()
    b = fast()
    assert all(torch.equal(aa, bb) for aa, bb in zip(a, b, strict=True))
    del a, b
    graphs = {}
    outputs = {}
    for key, fn in [("segmented", base), ("fused", fast)]:
        for _ in range(3):
            fn()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            outputs[key] = fn()
        graphs[key] = g
    samples = {key: [] for key in graphs}
    for i in range(8):
        for key in ["segmented", "fused"] if i % 2 else ["fused", "segmented"]:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(10):
                graphs[key].replay()
            end.record()
            end.synchronize()
            samples[key].append(start.elapsed_time(end) / 10)
    med = {key: statistics.median(vals) for key, vals in samples.items()}
    row = {
        "shape": [m, k],
        "samples_ms": samples,
        "median_ms": med,
        "reduction_percent": 100 * (1 - med["fused"] / med["segmented"]),
        "equal": all(
            torch.equal(a, b) for a, b in zip(outputs["segmented"], outputs["fused"], strict=True)
        ),
    }
    report["cases"].append(row)
    (r / "norm-benchmark.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(row), flush=True)
    del outputs, graphs, g, x, w, shift, scale, rows, base, fast, fn
