# SPDX-License-Identifier: Apache-2.0
"""Public API A/B benchmark; outputs and original samples saved as JSON."""

import argparse
import json
import statistics

import torch

import comfy_kitchen as ck
from comfy_kitchen.indexed_gate import _cuda_plain_fallback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cases": []}
    for m, n, k in [
        (1024, 5376, 14336),
        (8192, 5376, 14336),
        (32700, 5376, 14336),
        (65536, 5376, 14336),
        (1024, 7168, 18944),
        (8192, 7168, 18944),
    ]:
        torch.manual_seed(42)
        tensors = (
            torch.randint(-128, 128, (m, k), device="cuda", dtype=torch.int8),
            torch.randint(-128, 128, (n, k), device="cuda", dtype=torch.int8),
            torch.rand(m, device="cuda") * 0.01,
            torch.rand(n, device="cuda") * 0.01,
            torch.randn(5, n, device="cuda", dtype=torch.bfloat16),
            torch.randint(0, 5, (m,), device="cuda", dtype=torch.int32),
            torch.randn(m, n, device="cuda", dtype=torch.bfloat16),
        )
        functions = {"unfused": _cuda_plain_fallback, "fused_api": ck.int8_gemm_indexed_gate}
        outputs = {name: fn(*tensors) for name, fn in functions.items()}
        equal = torch.equal(outputs["unfused"], outputs["fused_api"])
        assert equal
        for _ in range(5):
            for fn in functions.values():
                fn(*tensors)
        samples = {name: [] for name in functions}
        for group in range(8):
            order = list(functions) if group % 2 == 0 else list(reversed(functions))
            for name in order:
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                for _ in range(10):
                    output = functions[name](*tensors)
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end) / 10)
        result["cases"].append(
            {
                "shape": [m, n, k],
                "equal": equal,
                "samples_ms": samples,
                "median_ms": {name: statistics.median(s) for name, s in samples.items()},
            }
        )
        with open(args.output, "w") as output_file:
            json.dump(result, output_file, indent=2)
        del tensors, outputs, output


if __name__ == "__main__":
    main()
