# -*- coding: utf-8 -*-
"""Post-install verification for comfy-kitchen (HIP backend + int8 GEMM smoke).

Run from the repo root (or anywhere). It deliberately drops the repo directory
from sys.path when present, so ``import comfy_kitchen`` resolves to the
*site-packages* install that install.bat just created — not the source tree,
which contains no compiled extension and would falsely report a broken/slow
install.

Exit code 0 = HIP backend active and int8 GEMM working; 1 = something is off.
"""
import os
import sys
import time

# If this script sits inside the comfy-kitchen source tree, the script's own
# directory is sys.path[0] and would shadow the installed package. Drop it.
_here = os.path.dirname(os.path.abspath(__file__))
if os.path.isdir(os.path.join(_here, "comfy_kitchen")):
    sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _here]

import torch  # noqa: E402

import comfy_kitchen as ck  # noqa: E402
from comfy_kitchen.backends import hip  # noqa: E402

print(f"comfy_kitchen loaded from: {ck.__file__}")

if not getattr(torch.version, "hip", None):
    print("[FAIL] PyTorch is not a ROCm/HIP build")
    sys.exit(1)

if not torch.cuda.is_available():
    print("[FAIL] no CUDA/HIP device visible")
    sys.exit(1)

print(f"device: {torch.cuda.get_device_name(0)}  "
      f"gcn: {torch.cuda.get_device_properties(0).gcnArchName}")

# 1) HIP backend must be loaded AND have WMMA (matrix cores) available.
if not hip.is_available():
    print(f"[FAIL] HIP backend not available (reason: {hip._EXT_ERROR or 'unknown'})")
    print("       -> the install is likely a pure-Python wheel without the HIP extension.")
    sys.exit(1)
if not hip.has_wmma():
    print("[FAIL] HIP backend loaded but the GPU has no matrix cores (RDNA2?), "
          "int8 GEMMs will fall back to triton/eager.")
    sys.exit(1)
print("[OK] HIP backend active (WMMA: True)")

# 2) int8 GEMM smoke + rough speed check (this APU class reaches ~7-8 TFLOPS).
torch.manual_seed(0)
m, k, n = 4096, 2048, 2048
x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
w16 = torch.randn(n, k, device="cuda", dtype=torch.float16)
wmax = w16.abs().amax(dim=1, keepdim=True) + 1e-5
w8 = (w16 / wmax * 127).round().clamp(-128, 127).to(torch.int8)
scale = (wmax.squeeze(1) / 127.0).float()


def bench(fn, iters=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


flops = 2 * m * k * n
t_int8 = bench(lambda: ck.int8_linear(x, w8, scale, None, torch.bfloat16))
t_fp16 = bench(lambda: torch.mm(x.float().half(), w16.t()))
tf_int8 = flops / (t_int8 / 1000) / 1e12
tf_fp16 = flops / (t_fp16 / 1000) / 1e12
print(f"int8_linear {m}x{k}x{n}: {t_int8:.2f} ms  ({tf_int8:.1f} TFLOPS)")
print(f"torch.mm   fp16      : {t_fp16:.2f} ms  ({tf_fp16:.1f} TFLOPS)")
if tf_int8 < 2.0:
    print("[WARN] int8 throughput is very low; the GEMM is likely not on the "
          "compiled HIP kernels (check which backend the registry picked).")
    sys.exit(1)
print("[OK] int8 GEMM is running on the compiled HIP kernels.")
sys.exit(0)
