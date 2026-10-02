"""Compare uncached W4A4 Linear with content-validated weight reuse."""

import argparse
import json
import statistics
import time

import torch
import torch_npu  # noqa: F401 -- register the NPU device

from comfy_kitchen.backends import ascend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=4135)
    parser.add_argument("--n", type=int, default=6144)
    parser.add_argument("--k", type=int, default=6144)
    parser.add_argument("--budget-mib", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    if min(args.m, args.n, args.k, args.iterations) <= 0 or args.k % 256:
        parser.error("dimensions/iterations must be positive and K divisible by 256")
    if args.warmup < 0 or args.budget_mib < 0:
        parser.error("warmup and budget must be non-negative")
    torch.npu.set_device(0)
    torch.manual_seed(20260929)
    with torch.inference_mode(), ascend.W4A4WeightCache(args.budget_mib * 1024**2) as cache:
        x = torch.randn(args.m, args.k, device="npu", dtype=torch.float32)
        weight = torch.randint(-128, 128, (args.n, args.k // 2), device="npu", dtype=torch.int8)
        scale = torch.ones(args.n, device="npu")

        def run(reuse):
            return ascend.convrot_w4a4_linear(
                x, weight, scale, weight_cache=cache if reuse else None
            )

        torch.npu.synchronize()
        start = time.perf_counter()
        first = run(True)
        torch.npu.synchronize()
        first_call_ms = (time.perf_counter() - start) * 1000
        assert torch.equal(first, run(False))
        del first
        for _ in range(args.warmup):
            run(False)
            run(True)
        pairs = []
        for i in range(args.iterations):
            pair = {}
            for reuse in (False, True) if i % 2 == 0 else (True, False):
                torch.npu.synchronize()
                start = time.perf_counter()
                result = run(reuse)
                torch.npu.synchronize()
                pair["reuse" if reuse else "baseline"] = (time.perf_counter() - start) * 1000
                del result
            pairs.append(pair)
        print(
            json.dumps(
                {
                    "shape_mnk": [args.m, args.n, args.k],
                    "dtype": "float32",
                    "exact": True,
                    "first_call_ms": first_call_ms,
                    "bytes_used": cache.bytes_used,
                    "hits": cache.hits,
                    "misses": cache.misses,
                    "means_ms": {
                        arm: statistics.mean(p[arm] for p in pairs) for arm in ("baseline", "reuse")
                    },
                    "pairs": pairs,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
