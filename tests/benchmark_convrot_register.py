# SPDX-License-Identifier: Apache-2.0
"""Run after building the baseline and PR wheels in separate environments.

python tests/benchmark_convrot_register.py --output results.json
CUDA Graph timing removes Python launch starvation for small matrices.
"""

import argparse
import json
import statistics

import torch

from comfy_kitchen.backends.cuda import quantize_int8_rowwise_convrot64


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cases": []}
    for k in (5376, 7168):
        for m in (1, 128, 1024, 4096, 32700, 65536, 90461):
            torch.manual_seed(42)
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            for _ in range(10):
                quantize_int8_rowwise_convrot64(x, 256)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = quantize_int8_rowwise_convrot64(x, 256)
            samples = []
            for _ in range(8):
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                for _ in range(100):
                    graph.replay()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) / 100)
            result["cases"].append(
                {"shape": [m, k], "samples_ms": samples, "median_ms": statistics.median(samples)}
            )
            del graph, out, x
    with open(args.output, "w") as output:
        json.dump(result, output, indent=2)


if __name__ == "__main__":
    main()
