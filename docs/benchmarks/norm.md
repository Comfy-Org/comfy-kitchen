# Indexed RMSNorm + modulation + ConvRot256 INT8

Adds `indexed_norm_convrot(x, weight, shift, scale, rows, eps=...)`, returning
INT8 activations and row scales without materializing the two BF16 norm/mod
intermediates. The fused CUDA path is initially SM120/K5376 and is gated on the
qualified PyTorch CUDA RMSNorm reduction. Other versions/layouts use native
RMSNorm plus the existing quantization chain. Invalid indices select zero
shift/scale; empty tables use identity modulation. No cross-backend bitwise
claim. This does not replace `rms_adaln`, which has a different contract.

RTX 5090 D v2, Torch 2.12.0+cu130 (git 7661cd9c6b841b62b7f411aa52ec51f05457263b),
CUDA 13.0. Main baseline 19ea55b. Comparison uses native RMSNorm, six contiguous
modulation segments and the existing ConvRot256 CUDA quantizer, not a full-size
GPU gather as its baseline. Eight alternating groups, ten CUDA Graph replays
per sample; three warmups per arm. **Whole preparation chain**, not full DiT.

| Shape M,K | Original segmented ms | Fused ms | Time reduction |
|---|---:|---:|---:|
| [131, 5376] | 0.043074 | 0.008398 | 80.502% |
| [8192, 5376] | 0.456670 | 0.159211 | 65.137% |
| [14850, 5376] | 0.878498 | 0.273794 | 68.834% |
| [32700, 5376] | 1.979624 | 0.576888 | 70.859% |
| [87142, 5376] | 6.234430 | 1.498915 | 75.957% |
| [90461, 5376] | 6.469998 | 1.555659 | 75.956% |

All Q bytes and FP32 scale values match. 33 tests pass, including CPU and CUDA,
empty and invalid indices, empty M, BF16/FP32 strided tables, magnitude extremes,
noncontiguous fallback, direct native dispatch, streams, graph replay and real
Inductor. Other devices and PyTorch reductions still need native qualification.

Raw samples: [norm-benchmark.json](norm-benchmark.json).
Reproduce: `python tests/benchmark_norm.py` and
`python -m pytest tests/test_indexed_norm.py`.
This is an opt-in API; model integration is required to gain full-model speed.
