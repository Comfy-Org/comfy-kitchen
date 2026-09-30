# SM120 D128 dense TMA pipeline

Measured 2026-09-30 on RTX 5090 D v2, PyTorch 2.12.0+cu130, CUDA13.0.88.
Main is `19ea55b9ebdaf77942dab36223e1222009d3ce11` (Kitchen0.2.36).
Separate baseline/PR extensions; identical pinned dependencies and input seeds.
Eight groups of ten warmed CUDA Graph replays per shape, CUDA-event times.

| B,Hq,Hkv,Lq,Lkv | Main ms | TMA ms | Less time | Exact BF16 |
|---|---:|---:|---:|:---:|
| 1,8,2,8192,8193 | 0.602890 | 0.579014 | 3.96% | True |
| 1,8,8,16384,16384 | 2.144539 | 2.092317 | 2.44% | True |
| 1,42,42,32700,32700 | 42.892517 | 42.624614 | 0.62% | True |
| 2,4,2,8197,9217 | 0.712371 | 0.683634 | 4.03% | True |

The kernel retains the baseline Q/K/V quantization, QK/PV MMA ordering, online
softmax helpers and final-tile mask arithmetic. Changes are a three-slot TMA
K/V ring, a producer warpgroup, cached Q fragments and register redistribution.
The initial route is deliberately narrow: SM120, contiguous BHSD, D128, BF16
output, no mask, CTA128, both sequence lengths>=8192, uint32-safe offsets.
Short/image shapes, other architectures/layouts/dtypes/masks retain main.
The specialized object is built as SM120a only; no device-driver link is added.

Existing attention suite plus new stream/graph checks: 421 passed, 594 skipped,
2 failures. Both failures also reproduce on unmodified main: on this Torch/CUDA
stack, SDPA produces NaN references for scale=0/negative with Lq129/Lkv8193.
Those shapes do not select TMA. They are documented, not hidden by test changes.
The expanded new stream/graph tests also cover zero/negative scale: 6 passed.

Full public H3 DiT smoke test (same conditions as ConvRot report): three warm
runs average **15.860485 → 15.839752 s (0.131% less time)**. Both audio/video
latent hashes match. This is too small to claim a reliable deployment speedup;
the useful evidence is the operator A/B above. No VAE or export timings claimed.

```bash
python -m pytest tests/test_int8_attention_tma.py tests/test_int8_attention.py
python tests/benchmark_attention_tma.py --output /tmp/attention-main.pt
# Repeat in PR environment with another output path; compare tensors with torch.equal.
```

Raw samples: `attention-public-benchmark.json`; full DiT: `full-dit.json`.
