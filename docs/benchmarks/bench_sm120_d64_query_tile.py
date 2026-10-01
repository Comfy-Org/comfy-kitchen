# SPDX-License-Identifier: Apache-2.0
"""Compare independently built Kitchen checkouts in separate processes.

python docs/benchmarks/bench_sm120_d64_query_tile.py \
    --baseline-dir /path/to/base --candidate-dir /path/to/pr --output results.json

Each worker loads one extension only. Input quantization is outside timing.
Results measure the public prequantized API, not complete VAE/request latency.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys


def worker(args):
    sys.path.insert(0, str(args.worker.resolve()))
    import torch
    import comfy_kitchen as ck

    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This qualification targets SM120")
    torch.set_num_threads(2)
    records = []
    shapes = [(1, 1024), (2, 1025), (4, 1797), (8, 1797), (4, 2048), (4, 2049)]
    with torch.inference_mode():
        for dtype in [torch.float16, torch.bfloat16]:
            for batch, length in shapes:
                torch.manual_seed(41)
                qkv = torch.randn(batch, length, 32, 3, 64, device="cuda", dtype=dtype)
                q, k, v = [qkv[:, :, :, i].transpose(1, 2) for i in range(3)]
                packed = ck.prequantize_int8_attention(q, k, v)
                for _ in range(10):
                    actual = ck.int8_attention_from_prequantized(packed)
                timings = []
                for _ in range(10):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(30):
                        actual = ck.int8_attention_from_prequantized(packed)
                    end.record()
                    end.synchronize()
                    timings.append(start.elapsed_time(end) / 30)
                signature = hashlib.sha256(
                    actual.cpu().contiguous().view(torch.uint8).numpy().tobytes()
                ).hexdigest()
                records.append({
                    "shape": [batch, 32, length, 64], "dtype": str(dtype),
                    "sha256": signature, "samples_ms": timings,
                    "median_ms": statistics.median(timings),
                })
    return {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "rows": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--candidate-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        args.output.write_text(json.dumps(worker(args), indent=2))
        return
    if args.baseline_dir is None or args.candidate_dir is None:
        parser.error("both checkout directories are required")
    runs = []
    for index, arm in enumerate(["baseline", "candidate", "candidate", "baseline"]):
        checkout = args.baseline_dir if arm == "baseline" else args.candidate_dir
        output = args.output.with_name(f"{args.output.stem}-{arm}-{index}.json")
        subprocess.run([
            sys.executable, str(Path(__file__).resolve()), "--worker", str(checkout),
            "--output", str(output),
        ], check=True)
        runs.append({"arm": arm, "data": json.loads(output.read_text())})
    records = []
    for index, base in enumerate(runs[0]["data"]["rows"]):
        peers = [run["data"]["rows"][index] for run in runs]
        if any(row["shape"] != base["shape"] or row["dtype"] != base["dtype"] for row in peers):
            raise AssertionError("worker shape order mismatch")
        if any(row["sha256"] != base["sha256"] for row in peers):
            raise AssertionError((base["shape"], base["dtype"], "output mismatch"))
        medians = {
            arm: statistics.median(
                value for run in runs if run["arm"] == arm
                for value in run["data"]["rows"][index]["samples_ms"]
            ) for arm in ["baseline", "candidate"]
        }
        records.append({
            "shape": base["shape"], "dtype": base["dtype"], "equal": True,
            "median_ms": medians,
            "time_reduction_percent": 100 * (1 - medians["candidate"] / medians["baseline"]),
        })
    args.output.write_text(json.dumps({"runs": runs, "comparison": records}, indent=2))
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
