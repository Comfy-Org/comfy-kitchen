"""Compare Ascend W4A4 with and without activation pack/unpack.

Run from the source checkout, for example:
    ASCEND_RT_VISIBLE_DEVICES=4 PYTHONPATH=. python \
        samples/benchmark_ascend_w4a4_unpacked.py --m 4135 --n 6144 --k 6144

The legacy arm restores only activation pack/unpack, not eager matmul. This
script temporarily replaces a private helper and must run in its own process.
"""

import argparse
import json
import statistics
import time
from unittest.mock import patch

import torch
import torch_npu  # noqa: F401

from comfy_kitchen.backends import ascend
from comfy_kitchen.backends.eager.convrot_w4a4 import quantize_signed_int4_rowwise
from comfy_kitchen.backends.eager.svdquant import _unpack_int4_row_major


def legacy_activation_quantization(x):
    packed, scale = quantize_signed_int4_rowwise(x)
    return _unpack_int4_row_major(packed).contiguous(), scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    args = parser.parse_args()
    if min(args.m, args.n, args.k, args.warmup, args.iterations) <= 0 or args.k % 256:
        parser.error("sizes/counts must be positive and K must be a multiple of 256")
    if not torch.npu.is_available() or not ascend._ASCEND_QUANT_MATMUL_AVAILABLE:
        parser.error("an Ascend NPU with npu_quant_matmul is required")

    torch.npu.set_device("npu:0")
    torch.manual_seed(42)
    dtype = getattr(torch, args.dtype)
    inputs = {
        "x": torch.randn(args.m, args.k, device="npu:0", dtype=dtype),
        "qweight": torch.randint(
            -128, 128, (args.n, args.k // 2), device="npu:0", dtype=torch.int8
        ),
        "wscales": torch.rand(args.n, device="npu:0", dtype=torch.float32) / 7,
    }
    helpers = {
        "legacy": legacy_activation_quantization,
        "unpacked": ascend._quantize_signed_int4_rowwise_unpacked,
    }
    outputs = {}
    for arm, helper in helpers.items():
        with patch.object(ascend, "_quantize_signed_int4_rowwise_unpacked", helper):
            outputs[arm] = ascend.convrot_w4a4_linear(**inputs)
            for _ in range(args.warmup):
                ascend.convrot_w4a4_linear(**inputs)
    torch.testing.assert_close(outputs["legacy"], outputs["unpacked"], rtol=0, atol=0)
    del outputs

    timings = {arm: [] for arm in helpers}
    for iteration in range(args.iterations):
        order = list(helpers) if iteration % 2 == 0 else list(reversed(helpers))
        for arm in order:
            with patch.object(ascend, "_quantize_signed_int4_rowwise_unpacked", helpers[arm]):
                torch.npu.synchronize()
                start = time.perf_counter()
                output = ascend.convrot_w4a4_linear(**inputs)
                torch.npu.synchronize()
                timings[arm].append((time.perf_counter() - start) * 1000)
                del output

    print(
        json.dumps(
            {
                "config": vars(args),
                "median_ms": {arm: statistics.median(values) for arm, values in timings.items()},
                "samples_ms": timings,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
