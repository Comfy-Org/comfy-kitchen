# SM120 V quantization: larger blocks for long D128 inputs

`quant_v_int8_kernel` used 512 threads for every N > 256. On RTX 5090 D v2, 1024 threads improve memory-level parallelism for large D128 inputs. This change instantiates the existing kernel with a different block size; it preserves scale computation, rounding, permutation, padding and all input values.

Dispatch is limited to SM120, FP16/BF16, D=128, N >= 12288, B*H >= 32, and at least 128 MiB of logical V data. Supported layouts are contiguous BHND and token-major BNHD / interleaved QKV with the documented strides. Other cases use the existing schedule. Device capability is cached per host thread/current device, without tensor retention or synchronization.

## Measurements

RTX 5090 D v2 / Torch 2.12.0+cu130 / CUDA 13.0.88, 2026-10-01. Main Python and main attention/V source pinned to `12389a30463c62c93670b049d59bf3fa56c0316d`. Both extensions were linked from the same existing combined-build object set with only the V object replaced. Other operators are identical between arms; this is not a clean rebuild of every main object. All formal sampler processes are independent; no services were changed.

V quantization includes both passes and padding. Each arm uses a warmed graph of eight launches, with eight interleaved AB/BA CUDA-event intervals. Below: BF16, B=1, H=56, D=128; raw FP16/BF16 records are in [sm120-v-quant.json](sm120-v-quant.json).

| N | Layout | Baseline ms | Candidate ms | Time reduction |
|---|---|---:|---:|---:|
| 14850 | BHND | 0.467676 | 0.363188 | 22.34% |
| 14850 | QKV | 0.852880 | 0.521000 | 38.91% |
| 32700 | BHND | 1.173544 | 0.865228 | 26.27% |
| 32700 | QKV | 2.264932 | 1.769694 | 21.87% |
| 87142 | BHND | 3.859126 | 3.015986 | 21.85% |
| 87142 | QKV | 6.950430 | 6.167594 | 11.26% |

The complete public attention call, including Q/K and V quantization, improves 0.50%–2.92% across the six measured cases. This percentage is smaller than the V-only improvement. All outputs match exactly.

Full Larry v4 INT8 / Euler-beta / 8-step sampler, seed 42, CFG 1, shift 12/4, 768x512 / 124 frames / 14,850 packed tokens: **15.688698 -> 15.537895 s, 0.9612% time reduction**. Two processes per arm, each one warmup and two formal requests, process order base/candidate/candidate/base. All eight formal requests have identical video/audio latent hashes and the same 1,833,455,616-byte peak Torch allocation. Profiling is separate from timed runs.

The sampler excludes input conditioning, VAE, export and initial weight loading. This is not serving QPS or an additional gain over the fully fused R85 path: the latter's QKV preparation already produces V through a separate fused implementation.

## Validation

- 38 targeted cases pass; one device-switch case is skipped because only one GPU is visible. Includes FP16/BF16/FP32, layout and threshold boundaries, partial padding, zero/signed zero, tiny/extreme/NaN/Inf values, nondefault streams and CUDA Graph replay.
- Each large output is compared with independent one-head calls that use the previous schedule. INT8 codes and FP32 scale bits match exactly.
- Existing attention suite plus the new tests: **both baseline and candidate have 457 passed, 595 skipped, 2 failed**. The same two inherited zero/negative-scale tests fail on both; they are tracked separately in [#224](https://github.com/Comfy-Org/comfy-kitchen/pull/224). They are not removed or counted as passes.
- A 204-case exploratory schedule sweep preceded the final narrow dispatch. No claim of measured performance on other GPU architectures.

To reproduce after building each checkout with identical flags:

```bash
python -m pytest tests/test_int8_v_quant_schedule.py tests/test_int8_attention.py -q
python benchmarks/bench_v_quant.py --output v-quant.json
```

Compare output hashes and repeated measurements from both checkouts. The benchmark script only produces random public input; no model or business media is distributed.
