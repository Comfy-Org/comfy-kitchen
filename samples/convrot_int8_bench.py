import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch

import comfy_kitchen as ck
from comfy_kitchen.backends import hip

SHAPES = {
    "qkv": (21504, 5376),
    "mlp_up": (28672, 5376),
    "mlp_down": (5376, 14336),
    "attn_out": (5376, 7168),
}


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return result


def nonnegative_int(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ck-root", type=Path, required=True)
    parser.add_argument("--shape", choices=["all", *SHAPES], default="all")
    parser.add_argument("--rows", type=positive_int, default=3802)
    parser.add_argument("--warmup", type=nonnegative_int, default=3)
    parser.add_argument("--iterations", type=positive_int, default=11)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


args = parse_args()
ck_root = args.ck_root.resolve()

if ck_root not in Path(ck.__file__).resolve().parents:
    raise RuntimeError(f"Imported comfy_kitchen from {ck.__file__}, expected {ck_root}")
if torch.version.hip is None:
    raise RuntimeError("This benchmark requires a gfx1150-or-newer ROCm device")

arch = torch.cuda.get_device_properties(0).gcnArchName.partition(":")[0]
arch_number = arch[3:] if arch.startswith("gfx") else ""
if not (
    arch_number.isdigit()
    and (arch_number.startswith("115") or arch_number.startswith("12"))
):
    raise RuntimeError("This benchmark requires a gfx115x or gfx12xx ROCm device")


def check_linear():
    torch.manual_seed(0)
    x = torch.randn(16, 1024, device="cuda", dtype=torch.bfloat16)
    weight = torch.randint(-127, 128, (64, 1024), device="cuda", dtype=torch.int8)
    weight_scale = torch.rand(64, device="cuda", dtype=torch.float32) * 0.01
    actual = hip.int8_linear(x, weight, weight_scale, convrot=True, convrot_groupsize=256)
    with ck.use_backend("eager"):
        expected = ck.int8_linear(x, weight, weight_scale, convrot=True, convrot_groupsize=256)
    torch.cuda.synchronize()
    delta = actual.float() - expected.float()
    rel_l2 = torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(expected.float())
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().flatten(), expected.float().flatten(), dim=0
    )
    if not torch.isfinite(actual).all() or rel_l2.item() > 0.02:
        raise AssertionError(f"HIP INT8 relative L2 error {rel_l2.item()} exceeds 0.02")
    print(
        json.dumps(
            {
                "check": "passed",
                "relative_l2": rel_l2.item(),
                "cosine": cosine.item(),
                "max_abs": delta.abs().max().item(),
            }
        )
    )


def measure(fn):
    for _ in range(args.warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return samples


def benchmark(name, n, k):
    m = args.rows
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
    weight_scale = torch.rand(n, device="cuda", dtype=torch.float32) * 0.01
    qact = torch.empty((m, k), device="cuda", dtype=torch.int8)
    act_scale = torch.empty(m, device="cuda", dtype=torch.float32)
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    spill_rotated = None
    spill_partials = None
    if hip._C.convrot_int8_needs_spill(m, k, 2):
        spill_rotated = torch.empty((m, k), device="cuda", dtype=torch.bfloat16)
        spill_partials = torch.empty((m, k // 256), device="cuda", dtype=torch.float32)

    def quantize():
        hip._C.quantize_int8_convrot(
            hip._dl(x),
            hip._dl(qact),
            hip._dl(act_scale),
            None if spill_rotated is None else hip._dl(spill_rotated),
            None if spill_partials is None else hip._dl(spill_partials),
            m,
            k,
            256,
            0,
            hip._stream(x),
        )

    def gemm():
        hip._C.int8_gemm(
            hip._dl(qact),
            hip._dl(weight),
            hip._dl(out),
            hip._dl(act_scale),
            hip._dl(weight_scale),
            1,
            None,
            m,
            n,
            k,
            2,
            hip._stream(x),
        )

    def linear():
        return hip.int8_linear(x, weight, weight_scale, convrot=True, convrot_groupsize=256)

    result = {"shape": name, "m": m, "n": n, "k": k}
    for label, fn in (("quant", quantize), ("gemm", gemm), ("full", linear)):
        samples = measure(fn)
        result[f"{label}_median_ms"] = statistics.median(samples)
        result[f"{label}_samples_ms"] = samples
    print(json.dumps(result), flush=True)
    return result


if args.check:
    check_linear()
else:
    selected = SHAPES.items() if args.shape == "all" else [(args.shape, SHAPES[args.shape])]
    results = []
    for name, (n, k) in selected:
        results.append(benchmark(name, n, k))
        gc.collect()
        torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "backend": str(ck_root),
                "device": torch.cuda.get_device_name(0),
                "arch": torch.cuda.get_device_properties(0).gcnArchName,
                "torch": torch.__version__,
                "results": results,
            }
        )
    )
