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

For positive attention scales, the kernel retains the baseline Q/K/V
quantization, QK/PV MMA ordering, online-softmax helpers and final-tile mask
arithmetic. Zero/negative scales use a separate specialization that scales
scores before taking the maximum and applying the padding mask. Changes are a three-slot TMA
K/V ring, a producer warpgroup, cached Q fragments and register redistribution.
The initial route is deliberately narrow: SM120, contiguous BHSD, D128, BF16
output, no mask, CTA128, both sequence lengths>=8192, uint32-safe offsets.
Short/image shapes, other architectures/layouts/dtypes/masks retain main.
The specialized object is built as SM120a only; no device-driver link is added.

Independent numerical regressions now compare every result with FP32 PyTorch
SDPA forced to `SDPBackend.MATH`, using the same scale and repeated GQA heads.
Only queries are chunked (512 rows); all keys remain visible. This bounds
reference memory without changing unmasked attention semantics. The tests also
retain repeated execution, nondefault stream, and CUDA Graph assertions.

All **9 new tests pass** on RTX 5090 D v2, covering three shapes (GQA, full
and partial tiles) and default/zero/negative scales. NRMSE ranges are:

| Scale | NRMSE against independent reference | Limit |
|---|---:|---:|
| Default positive | 0.016187–0.016302 | <0.03 |
| Zero | 0.009147–0.009456 | <0.03 |
| Negative | 0.016199–0.016305 | <0.03 |

The added reference uncovered a real inherited numerical issue, beyond the
previously reported SDPA NaNs: both unmodified main and the initial PR returned
all-zero outputs for the two tested negative-scale, partial-tile cases
(NRMSE **1.0**). Scaling after a maximum is not valid for negative scales, and
scaling a padding sentinel reverses its sign. The new TMA specialization fixes
this by scaling first, masking second and reusing main's FP32-score-to-U8
probability helper. Default positive-scale arithmetic remains unchanged.
Independent explicit `softmax(QK * scale) @ V` checks of 128 query rows agree
with the FP32 math SDPA reference (NRMSE below 8e-7).

Combined suite after this fix: **428 passed, 594 skipped, 2 failed**. The two
unchanged failures at Lq129/Lkv8193 remain in the original non-TMA path and
also reproduce on unmodified main. Optimized SDPA supplies NaN references for
zero/negative scale on this Torch/CUDA stack; the negative-scale kernel issue
also exists in that original path. This PR fixes the new TMA route, not the
ordinary kernel. No existing test was skipped or weakened to hide these
failures. Raw before/after accuracy: `attention-reference-regression.json`.

Full public H3 DiT smoke test (same conditions as ConvRot report): three warm
runs average **15.860485 → 15.839752 s (0.131% less time)**. Both audio/video
latent hashes match. This is too small to claim a reliable deployment speedup;
the useful evidence is the operator A/B above. No VAE or export timings claimed.

```bash
python -m pytest tests/test_int8_attention_tma.py tests/test_int8_attention.py
python tests/benchmark_attention_tma.py --output /tmp/attention-main.pt
# Repeat in PR environment with another output path; compare tensors with torch.equal.
```

After the review fix, a fresh extension rebuild and the dedicated CMake target
both pass. A repeat default-positive-scale A/B benchmark gives:

| B,Hq,Hkv,Lq,Lkv | Main ms | Fixed TMA ms | Less time | Exact BF16 |
|---|---:|---:|---:|:---:|
| 1,8,2,8192,8193 | 0.589357 | 0.567477 | 3.71% | True |
| 1,8,8,16384,16384 | 2.130446 | 2.083659 | 2.20% | True |
| 1,42,42,32700,32700 | 42.821834 | 42.503345 | 0.74% | True |
| 2,4,2,8197,9217 | 0.713445 | 0.682398 | 4.35% | True |

Separate warmed processes, eight groups of ten CUDA Graph replays; not an
interleaved timing. Small differences remain sensitive to clocks and run order.
The four output comparisons have zero mismatches. Original timings are retained
above for auditability rather than silently replaced.

Raw original samples: `attention-public-benchmark.json`; original full DiT:
`full-dit.json`; repeat samples: `attention-post-review-benchmark.json`.

Full DiT was also rerun after rebuilding: three warm samples average
**15.834297 s**, with both audio/video latent hashes identical to the
original baseline in every run (including warmup). This is a correctness smoke
check, not a contemporaneous end-to-end performance A/B. Raw samples:
`full-dit-post-review.json`.
