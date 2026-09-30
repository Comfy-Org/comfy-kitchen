# H3 QKV preparation on SM120

`h3_qkv_prequantize` consumes already ConvRot256-quantized INT8 activations and weights. It returns the existing `PrequantizedInt8Attention` container, leaving dense attention and its output layout unchanged.

The CUDA path combines a 5376→21504 projection, per-head weighted RMSNorm, 96-channel split-half RoPE, Q/K rotation and quantization, and V partial maxima. Nine current-input K rows are projected first to reproduce the existing Kitchen anchor selection; every query/key/value remains present. There is no inter-step cache, sparsity, step skipping, or new quantization policy. The BF16 projection and RMS/RoPE rounding points are retained.

## Scope

- CUDA, SM120, one packed sequence, 56 heads × 128 channels, BF16 RoPE/norm weights, no bias. Inputs follow the public API's documented shapes.
- Native fusion is selected for 8192–200000 rows, contiguous and 16-byte aligned inputs. Smaller inputs keep the existing projection/preparation chain because the fused kernel was slower in measurement. Other CUDA architectures/layouts use the fallback. CPU/ROCm are outside this API's declared contract.
- Source builds without CUTLASS return to the fallback before launching the fused chain.
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
| 4096 | 2.142864 | 2.182340 | -1.842% |
| 4097 | 2.232588 | 2.249572 | -0.761% |
| 8192 | 4.491016 | 4.174260 | 7.053% |
| 14850 | 8.352496 | 7.396684 | 11.443% |
| 32700 | 18.809397 | 15.913524 | 15.396% |
| 87142 | 50.810171 | 42.261513 | 16.825% |
| 90461 | 52.681864 | 43.918579 | 16.634% |

4096/4097 rows select the fallback; small timing differences there are measurement noise, not native-fusion claims.

## Full sampler experiment

Current ComfyUI 8cfe5e1, Larry v4 INT8 / Euler-beta 8 steps / seed42 / CFG1 / shift12/4, 768×512 / 124 frames / 14850 packed tokens. Four alternating AB/BA formal pairs after warmup; all samples retained.

- Mean complete sampler: **15.803847 → 15.366207 s, 2.7692% reduction**.
- 400 fused preparations per optimized request; all eight formal requests have identical video and audio latent SHA256.
- Peak Torch allocation unchanged at 1833455616 bytes, excluding external aimdo allocation.
- This experiment explicitly selects the same Kitchen INT8 attention in both arms. It is not a proposed generic ComfyUI attention-dispatch implementation. A proper upstream consumer must preserve selected attention and patch contracts.
- No fresh RGB/PCM decode or continuous-service throughput measurement is claimed. Do not add this percentage to other fusions or the historical SpeedKit 6.49% result.

The portable CUDA wrapper initially used separate raw-source kernels; the submitted implementation removes unused planes/helpers and shares the CUTLASS Mma type. It has been rebuilt and retested after this cleanup: **20 tests passed** in 1.62 seconds (including declined-dispatch scratch lifetime). Existing Comfy Org/NVIDIA source notices are retained; CUTLASS is a build dependency under its own BSD license.
