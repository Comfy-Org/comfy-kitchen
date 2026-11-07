# SPDX-License-Identifier: Apache-2.0
"""Compare separate main/PR environments; retain outputs for an exact A/B diff.

python tests/benchmark_attention_tma.py --output /tmp/attention-main.pt
python tests/benchmark_attention_tma.py --output /tmp/attention-pr.pt
"""

import argparse
import statistics

import torch

import comfy_kitchen as ck


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cases": []}
    for b, h, hk, lq, lk in [
        (1, 8, 2, 8192, 8193),
        (1, 8, 8, 16384, 16384),
        (1, 42, 42, 32700, 32700),
        (2, 4, 2, 8197, 9217),
    ]:
        torch.manual_seed(42)
        q = torch.randn(b, h, lq, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(b, hk, lk, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        packed = ck.prequantize_int8_attention(q, k, v)
        for _ in range(5):
            out = ck.int8_attention_from_prequantized(packed)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = ck.int8_attention_from_prequantized(packed)
        samples = []
        for _ in range(8):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(10):
                graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / 10)
        result["cases"].append(
            {
                "shape": [b, h, hk, lq, lk],
                "samples_ms": samples,
                "median_ms": statistics.median(samples),
                "output": out.cpu(),
            }
        )
        del graph, out, q, k, v, packed
    torch.save(result, args.output)


if __name__ == "__main__":
    main()
