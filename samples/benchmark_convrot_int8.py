# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Graph timing for exact BF16 ConvRot + INT8 with inputs exceeding L2.

Run before and after a change with the same toolkit, GPU and power settings:
    python samples/benchmark_convrot_int8.py --rows 4352 139264

Reported bandwidth counts useful bytes. Use Nsight Compute for physical DRAM
throughput and percentage of sustained peak; do not infer it from SM clocks.
"""

import argparse
import json
import random
import statistics

import torch

from comfy_kitchen.backends import cuda


def benchmark(rows):
    cols = 4096
    slots = max(3, 2 * torch.cuda.get_device_properties(0).L2_cache_size // (rows * cols * 2) + 1)
    inputs = [torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16) for _ in range(slots)]
    methods = {
        "convrot": cuda.quantize_int8_rowwise_convrot,
        "convrot64_reference": cuda.quantize_int8_rowwise_convrot64,
    }
    graphs = {}
    # Warm up on a side stream before capture. Both methods preserve the
    # original ordered FP32 rotation and BF16 quantizer recipe.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for fn in methods.values():
            fn(inputs[0], 256)
    torch.cuda.current_stream().wait_stream(stream)
    for name, fn in methods.items():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(4):
                for x in inputs:
                    output, scale = fn(x, 256)
        graph.replay()
        expected_output, expected_scale = methods["convrot64_reference"](inputs[-1], 256)
        assert torch.equal(output, expected_output)
        assert torch.equal(scale.view(torch.int32), expected_scale.view(torch.int32))
        graphs[name] = graph

    def measure(graph):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / (4 * slots)

    samples = {name: [] for name in methods}
    rng = random.Random(391)
    for _ in range(7):
        order = list(graphs)
        rng.shuffle(order)
        for name in order:
            for _ in range(5):
                measure(graphs[name])
            samples[name].append(measure(graphs[name]))
    useful_bytes = rows * cols * 3 + rows * 4
    medians = {name: statistics.median(values) for name, values in samples.items()}
    return {
        "rows": rows,
        "cols": cols,
        "slots": slots,
        "input_pool_bytes": slots * rows * cols * 2,
        "median_ms": medians,
        "samples_ms": samples,
        "useful_TB_per_second": {name: useful_bytes / (ms * 1e9) for name, ms in medians.items()},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", type=int, default=[4352, 139264])
    args = parser.parse_args()
    if any(rows <= 0 for rows in args.rows):
        parser.error("row counts must be positive")
    torch.manual_seed(391)
    with torch.no_grad():
        result = {
            "gpu": torch.cuda.get_device_name(),
            "cuda": torch.version.cuda,
            "rows": [benchmark(rows) for rows in args.rows],
        }
    print(json.dumps(result, indent=2))
