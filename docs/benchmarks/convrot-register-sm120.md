# SM120 BF16 register ConvRot256

Measured 2026-09-30 on one RTX 5090 D v2 (SM120), PyTorch 2.12.0+cu130,
CUDA 13.0.88. Baseline: Kitchen `19ea55b9ebdaf77942dab36223e1222009d3ce11`
(0.2.36). Baseline and candidate use the same compiler flags. Other GPUs and
FP16/FP32, stochastic rounding, fused activations and other widths retain the
existing implementation.

Eight alternating groups, CUDA-event timing of warmed CUDA Graph replays,
10–100 replays/group. This isolates the quantize+rotate operator from Python
launch overhead. JSON includes every sample, seeds and correctness cases.

| M | K | Main (ms) | Register (ms) | Less time |
|---:|---:|---:|---:|---:|
| 1024 | 5376 | 0.016394 | 0.008220 | 49.86% |
| 4096 | 5376 | 0.049215 | 0.023842 | 51.55% |
| 32700 | 5376 | 0.489247 | 0.444126 | 9.22% |
| 65536 | 5376 | 0.976363 | 0.891171 | 8.73% |
| 90461 | 5376 | 1.344021 | 1.231091 | 8.40% |
| 1024 | 7168 | 0.020506 | 0.010267 | 49.93% |
| 4096 | 7168 | 0.067724 | 0.028698 | 57.62% |
| 32700 | 7168 | 0.647534 | 0.592686 | 8.47% |
| 65536 | 7168 | 1.292194 | 1.190602 | 7.86% |
| 90461 | 7168 | 1.781528 | 1.643211 | 7.76% |

INT8 values and FP32 scale bits match the retained generic CUDA kernel for both
widths, M=1 through 90,461; zeros, negative zero, tiny/huge inputs, NaN/Inf,
random BF16 bit patterns; nondefault stream and graph replay. Automated tests
force the generic implementation with a contiguous but unaligned BF16 input.
`test_convrot_register.py` + `test_int8_input_act.py`: **80 passed, 3 skipped**.

Public H3 sanity workload: Larry v4 INT8, 768×512, 124 requested frames, seed42,
Euler/beta, 8 steps, CFG1, frozen condition. After one warmup, three full DiT
runs average **15.860485 → 15.815302 s (0.285% less time)**. Both audio/video
latent SHA256 match on every run. This small sequential-run difference is an
observation, not a robust end-to-end speed claim; VAE/export are excluded.
The operator optimization does not change LoRA, steps or quantization format.

Reproduce after building each revision in separate environments:

```bash
python -m pytest tests/test_convrot_register.py tests/test_int8_input_act.py
python tests/benchmark_convrot_register.py --output /tmp/convrot.json
```

See `convrot-graph-benchmark.json` and `full-dit.json`. No private media is
included. The original whole-SpeedKit speedup does not belong to this PR.
