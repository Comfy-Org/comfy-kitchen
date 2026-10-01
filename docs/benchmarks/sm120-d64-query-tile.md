# SM120 D64 query tile

The H3 video VAE has 32 attention heads with 64 dimensions each. A 64-query CTA with 16-query warps lowers live accumulators while retaining the original 128-row quantization scale groups. The initial dispatch is limited to SM120, 1–8 batches, 32 query/key heads, equal 1024–2048 query/key lengths, no mask, and positive attention scale.

## Paired kernel measurements

NVIDIA GeForce RTX 5090 D v2, CUDA 13.0, PyTorch 2.12.0+cu130. Raw events and equality checks are in [sm120-d64-query-tile.json](sm120-d64-query-tile.json).

| Batch, heads, sequence, dim | Output | Original ms | Small tile ms | Time reduction |
|---|---|---:|---:|---:|
| 4, 32, 1797, 64 | float16 | 0.238291 | 0.227432 | 4.56% |
| 1, 32, 1797, 64 | float16 | 0.066618 | 0.062077 | 6.82% |
| 8, 32, 1797, 64 | float16 | 0.497086 | 0.490026 | 1.42% |
| 4, 32, 1797, 64 | bfloat16 | 0.242810 | 0.228682 | 5.82% |
| 1, 32, 1797, 64 | bfloat16 | 0.068507 | 0.063702 | 7.01% |
| 8, 32, 1797, 64 | bfloat16 | 0.503851 | 0.494549 | 1.85% |

## Full decoder scope

A separate integration run held all non-attention extension entry points fixed and decoded the same two 768×1344 joint-AV latents (243 and 260 frames). Each arm was warmed once, then measured in ABBA order. Floating-point RGB and RGB8 SHA256 were equal for all 12 decodes.

| Frames | Base seconds | Small tile seconds | Time reduction |
|---:|---:|---:|---:|
| 243 | 9.929252 | 9.874037 | 0.556% |
| 260 | 10.627293 | 10.590084 | 0.350% |

These decoder timings include dynamic weight loading and GPU→CPU output. They are not online service throughput or an end-to-end generation speedup. Only two representative latents were used for this integration check; no private prompts or media are included.

Cached-Q candidates were also tested. The plain small tile was chosen for its simpler scope and consistent benefit in the qualified workloads. The cache pipeline is unchanged in this PR.

## Correctness suite

- Current-main Python API plus this launcher: 429 passed, 594 skipped, and two pre-existing zero/negative-scale failures.
- The unmodified main launcher reproduces those same two failures; they are handled by #224.
- Combined with #224 and the existing upstream candidates: 527 passed, 594 skipped, no failures.
- The new split-head oracle contributes ten byte-equality cases with interleaved QKV, both 16-bit output types, and tail lengths.
