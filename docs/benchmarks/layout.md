# Contiguous BSHD output for prequantized INT8 attention

The new keyword `output_layout="BSHD"` preserves quantization and attention
arithmetic, but writes native CUDA results in projection-ready storage.
BHSD remains the default. Non-tile head dimensions trim with a contiguous copy;
HIP currently converts the result after its existing kernel.

SM120, RTX 5090 D v2, Torch 2.12.0+cu130, CUDA 13.0. Baseline is PR #218's
TMA build, followed by transpose/contiguous. Candidate uses the same build with
the BSHD output interface. Eight alternating groups, four CUDA Graph replays per
sample; three warmups per arm. These are **attention + output-layout** timings,
not complete-model or serving-throughput measurements.

| Shape B,H,S,D | BHSD + reorder ms | Direct BSHD ms | Time reduction |
|---|---:|---:|---:|
| [1, 8, 8192, 128] | 0.578232 | 0.551984 | 4.539% |
| [1, 56, 14850, 128] | 12.165948 | 11.821732 | 2.829% |
| [1, 56, 32700, 128] | 57.380583 | 56.607347 | 1.348% |
| [1, 56, 87142, 128] | 402.390869 | 400.229385 | 0.537% |
| [1, 56, 90461, 128] | 433.013412 | 430.830399 | 0.504% |

All five outputs are bitwise equal. 28 tests pass, covering the new layout,
existing TMA cases, batch 2, GQA, padded D96, D64/128/256, masks, FP16/BF16/FP32,
streams and graph replay. Nonpositive-scale accuracy retains #218's independent
FP32 math-SDPA validation. Other GPU architectures and HIP were not available.

Raw samples: [layout-benchmark.json](layout-benchmark.json).
Reproduce with `python tests/benchmark_layout.py` from the repository root and
`python -m pytest tests/test_int8_attention_layout.py tests/test_int8_attention_tma.py`.
This is a dependent follow-up to #218; only the BSHD delta is new.
