# SPDX-License-Identifier: Apache-2.0
"""Run on both checkouts with the same GPU/build flags; compare median_ms."""

import argparse
import hashlib
import json
import statistics

import torch

from comfy_kitchen.backends import cuda


def benchmark(fn, repeats):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(8):
            fn()
    graph.replay()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / 8)
    return {"median_ms": statistics.median(samples), "samples_ms": samples}


def signature(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[8192, 12288, 14850, 32700, 87142, 90461])
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--repeats", type=int, default=8)
    args = parser.parse_args()
    torch.manual_seed(57)
    report = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "rows": []}
    for dtype in [torch.float16, torch.bfloat16]:
        for n in args.lengths:
            for layout in ["BHND", "BNHD", "QKV"]:
                h, d = args.heads, 128
                if layout == "BHND":
                    v = torch.randn(1, h, n, d, device="cuda", dtype=dtype)
                elif layout == "BNHD":
                    v = torch.randn(1, n, h, d, device="cuda", dtype=dtype).transpose(1, 2)
                else:
                    v = torch.randn(1, n, 3, h, d, device="cuda", dtype=dtype)[:, :, 2].transpose(1, 2)
                padded = (n + 127) // 128 * 128
                out = torch.empty((1, h, d, padded), device="cuda", dtype=torch.int8)
                scale = torch.empty((1, h, d), device="cuda", dtype=torch.float32)

                def quantize(v=v, out=out, scale=scale, padded=padded, dtype=dtype):
                    cuda._C._quant_v_int8(
                        *(cuda._wrap_for_dlpack(t) for t in (v, out, scale)),
                        padded, 1 if dtype == torch.float16 else 2,
                        torch.cuda.current_stream().cuda_stream,
                    )

                row = {"shape": [1, h, n, d], "dtype": str(dtype), "layout": layout}
                row.update(benchmark(quantize, args.repeats))
                row["sha256"] = [signature(out), signature(scale)]
                report["rows"].append(row)
                del quantize, v, out, scale
    with open(args.output, "w") as output:
        json.dump(report, output, indent=2)


if __name__ == "__main__":
    main()
