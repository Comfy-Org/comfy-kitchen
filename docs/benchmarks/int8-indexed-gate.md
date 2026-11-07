# INT8 GEMM indexed gate + residual epilogue

Measured 2026-09-30, one RTX5090 D v2, PyTorch2.12.0+cu130, CUDA13.0.88.
Baseline Kitchen main `19ea55b9ebdaf77942dab36223e1222009d3ce11` (0.2.36).
Pinned CUTLASS `d4b4b494c3c51bf6507e7ab09fbafd1e9fa94f39`.

This is an opt-in prequantized operation, not a change to `int8_linear`.
Baseline is the existing fused-dequant CUTLASS GEMM followed by gate gather
and torch.addcmul. Candidate is the public `int8_gemm_indexed_gate` API,
including validation and fallback overhead. Eight alternating groups of ten
calls, warmed CUDA-event timing, no global clock changes.

| M | N | K | Existing chain ms | New API ms | Less time | Exact BF16 |
|---:|---:|---:|---:|---:|---:|:---:|
| 1024 | 5376 | 14336 | 0.290685 | 0.293189 | -0.86% | True |
| 8192 | 5376 | 14336 | 2.670003 | 2.363296 | 11.49% | True |
| 32700 | 5376 | 14336 | 10.983971 | 8.985152 | 18.20% | True |
| 65536 | 5376 | 14336 | 21.839299 | 17.755728 | 18.70% | True |
| 1024 | 7168 | 18944 | 0.582677 | 0.585597 | -0.50% | True |
| 8192 | 7168 | 18944 | 4.655568 | 4.014264 | 13.77% | True |

The two M1024 shapes use the ordinary GEMM chain (about0.5–0.9% wrapper
overhead in this run). The larger shapes fuse gate loads/residual and avoid
materializing both the BF16 branch and expanded gate. No whole-model speedup
is claimed: consumers must explicitly wire this operation into their model.
This does not duplicate PR#215's N-banded Stream-K scheduling experiment.

Numerical contract: integer dot product → FP32 x-scale then w-scale → BF16
branch → FP32 FMA with BF16 gate/residual → BF16 output. The BF16 intermediate
is intentional. All inputs remain unchanged. Gates are [G,N], row indices
INT32[M]; invalid indices select zero, without a device-to-host synchronization.
Noncontiguous/unsupported shapes use a PyTorch fallback. No backward support.
During graph capture the fused path uses a workspace-free identity schedule;
ordinary inference uses the existing Stream-K implementation.

Tests cover integer-oracle comparison, invalid/empty gates, empty dimensions,
noncontiguous fallback, validation, torch.compile, nondefault streams and CUDA
Graph replay. Indexed gate + existing residual tests: **36 passed, 1 skipped**.

```python
import comfy_kitchen as ck
out = ck.int8_gemm_indexed_gate(a_int8, b_int8, x_scale, w_scale,
                              gate_bf16, row_indices_int32, residual_bf16)
```

```bash
python -m pytest tests/test_int8_indexed_gate.py tests/test_int8_residual.py
python tests/benchmark_int8_indexed_gate.py --output /tmp/gate.json
```

Raw samples: `gate-api-benchmark.json`. Inputs are seeded synthetic shapes
representative of H3; these numbers are not video/audio quality scores.
