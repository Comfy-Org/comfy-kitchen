#!/usr/bin/env python3
"""Repeat-call CUDA/PyTorch interoperability check for comfy-kitchen issue #245.

Run in a *fresh process* using the selected Conda Python after installing the
Kitchen wheel. Use the same command against static and experimental shared
cudart builds. Do not confuse a successful first CUDA synchronization with a
successful second PyTorch GPU launch.
"""
import argparse
import re
import subprocess
import sys
import traceback

import torch

import comfy_kitchen as ck
from comfy_kitchen.backends import cuda

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=("quantize", "sol"), default="quantize")
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    torch.cuda.init()
    device = torch.device("cuda:0")
    print("Torch:", torch.__version__, "torch CUDA:", torch.version.cuda, flush=True)
    print("GPU:", torch.cuda.get_device_name(device), flush=True)
    print("Kitchen extension:", cuda._C.__file__, flush=True)
    print("Current stream:", hex(torch.cuda.current_stream(device).cuda_stream), flush=True)
    libraries = sorted({
        line.split()[-1] for line in open("/proc/self/maps", encoding="utf8")
        if "libcudart.so" in line
    })
    print("Mapped shared CUDART:", libraries, flush=True)
    result = subprocess.run(
        ["readelf", "-d", cuda._C.__file__],
        capture_output=True, text=True, check=True,
    )
    needed = [line.strip() for line in result.stdout.splitlines() if "libcudart" in line]
    runtime_paths = [
        line.strip() for line in result.stdout.splitlines()
        if "(RPATH)" in line or "(RUNPATH)" in line
    ]
    print("Extension CUDART DT_NEEDED:", needed, flush=True)
    print("Extension RPATH/RUNPATH:", runtime_paths, flush=True)
    expected = torch.version.cuda.split(".", 1)[0]
    majors = {
        m.group(1) for line in needed
        for m in re.finditer(r"libcudart\.so\.(\d+)", line)
    }
    if majors != {expected}:
        raise RuntimeError(
            f"Extension CUDART majors={sorted(majors)} but PyTorch CUDA major={expected}. "
            "Refusing to interpret this as a matched-runtime test."
        )
    for line in runtime_paths:
        paths = re.search(r"\[(.*)\]", line)
        if paths is None:
            raise RuntimeError(f"Cannot validate ELF search paths: {line}")
        for path in paths.group(1).split(":"):
            if path.startswith("/"):
                raise RuntimeError(f"Non-portable absolute ELF RPATH/RUNPATH: {path}")
    torch.manual_seed(0)
    if args.operation == "quantize":
        x = torch.randn(64, 256, device=device, dtype=torch.bfloat16)
    else:
        q, k, v = (torch.randn(1, 4096, 2, 128, device=device,
                               dtype=torch.bfloat16) for _ in range(3))
    torch.cuda.synchronize()
    for i in range(args.iterations):
        print(f"Iteration {i+1}: native call", flush=True)
        if args.operation == "quantize":
            cuda.quantize_int8_rowwise(x)
        else:
            ck.sol_attn(q, k, v, tau=4.0, token_aug=0)
        torch.cuda.synchronize()
        print("Native sync: PASS", flush=True)
        tmp = torch.empty(64, device=device)
        print("PyTorch allocate: PASS", flush=True)
        tmp.fill_(1.0)
        torch.cuda.synchronize()
        print("PyTorch fill/sync: PASS", flush=True)
    print("RESULT: PASS", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        print("RESULT: FAIL", flush=True)
        sys.exit(1)
