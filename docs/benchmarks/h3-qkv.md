# H3 QKV preparation on SM120

`h3_qkv_prequantize` consumes already ConvRot256-quantized INT8 activations and weights. It returns the existing `PrequantizedInt8Attention` container, leaving dense attention and its output layout unchanged.

The CUDA path combines a 5376→21504 projection, per-head weighted RMSNorm, 96-channel split-half RoPE, Q/K rotation and quantization, and V partial maxima. Nine current-input K rows are projected first to reproduce the existing Kitchen anchor selection; every query/key/value remains present. There is no inter-step cache, sparsity, step skipping, or new quantization policy. The BF16 projection and RMS/RoPE rounding points are retained.

## Scope

- CUDA, SM120, one packed sequence, 56 heads × 128 channels, BF16 RoPE/norm weights, no bias. Inputs follow the public API's documented shapes.
- Native fusion is selected for 8192–200000 rows, 16-byte aligned inputs after contiguous normalization. Smaller inputs keep the existing projection/preparation chain because the fused kernel was slower in measurement. Other CUDA architectures use the fallback. CPU/ROCm are outside this API's declared contract.
- Source builds without CUTLASS or a loadable device image return to the fallback before launching the fused chain. The eager projection fallback processes at most 1024 rows of INT32/FP32 scratch at a time.
- Graph/Inductor use an opaque custom op and fake tensor contract. The fallback avoids the direct CUTLASS launcher while capturing because that launcher's workspace allocation can invalidate capture for large M.
- Finite model values and finite positive normal FP32 epsilon are expected. Existing APIs remain unchanged.

## Reproduction

```bash
PYTHONPATH=. python -m pytest -q tests/test_h3_qkv.py
PYTHONPATH=. python tests/benchmark_h3_qkv.py
```

RTX 5090 D v2, Torch 2.12.0+cu130, CUDA 13.0.88, Kitchen base 19ea55b. Microbenchmark uses 8 alternating measurement groups, 4 eager calls per CUDA event interval. It includes projection and complete Q/K/V preparation, excludes input quantization and dense attention, and compares all six packed tensors byte-for-byte. See `h3-qkv-benchmark.json` for samples. Capture results from an earlier diagnostic are not mixed into this table.


| Rows | Stock ms | Fused/selected ms | Reduction |
|---:|---:|---:|---:|
| 4096 | 2.169664 | 2.149704 | 0.920% |
| 4097 | 2.240808 | 2.243132 | -0.104% |
| 8192 | 4.491612 | 4.173356 | 7.086% |
| 14850 | 8.336608 | 7.397484 | 11.265% |
| 32700 | 18.807447 | 15.913076 | 15.389% |
| 87142 | 50.770336 | 42.275801 | 16.731% |
| 90461 | 52.687628 | 43.912533 | 16.655% |

4096/4097 rows select the fallback; small timing differences there are measurement noise, not native-fusion claims.

## Full sampler experiment

Current ComfyUI 8cfe5e1, Larry v4 INT8 / Euler-beta 8 steps / seed42 / CFG1 / shift12/4, 768×512 / 124 frames / 14850 packed tokens. Four alternating AB/BA formal pairs after warmup; all samples retained.

- Mean complete sampler: **15.803847 → 15.366207 s, 2.7692% reduction**.
- 400 fused preparations per optimized request; all eight formal requests have identical video and audio latent SHA256.
- Peak Torch allocation unchanged at 1833455616 bytes, excluding external aimdo allocation.
- This experiment explicitly selects the same Kitchen INT8 attention in both arms. It is not a proposed generic ComfyUI attention-dispatch implementation. A proper upstream consumer must preserve selected attention and patch contracts.
- No fresh RGB/PCM decode or continuous-service throughput measurement is claimed. Do not add this percentage to other fusions or the historical SpeedKit 6.49% result.

The portable CUDA wrapper initially used separate raw-source kernels; the submitted implementation removes unused planes/helpers and shares the CUTLASS Mma type. It has been rebuilt and retested after this cleanup: **24 tests passed, 1 skipped** in 1.76 seconds (including 87142-row strided-input memory, no-CUTLASS projection, and declined-dispatch scratch lifetime; the two-device test was skipped in the one-GPU container). Existing Comfy Org/NVIDIA source notices are retained; CUTLASS is a build dependency under its own BSD license.


Review follow-up: an actual sdist was built with `python setup.py --no-cuda --no-hip sdist`; all 65 CUDA source entries, including the new runtime header and QKV `.cu` files, were present. CUDA image probes were compiled separately with SM80-only SASS and without CUTLASS: both returned unavailable on SM120 before dispatch, and subsequent CUDA work succeeded. See `h3-qkv-image-probes.json` and `h3-qkv-sdist-check.json`.
