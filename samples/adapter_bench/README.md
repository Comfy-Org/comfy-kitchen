# Adapter op benchmarks

Measures `comfy_kitchen.adapters` against ComfyUI's own code for the same
operation, one sweep at a time around a fixed real layer (a Flux attention
projection: 3072 x 3072, rank 32, 4096 tokens).

```
python -m samples.adapter_bench.run --out samples/adapter_bench/out [--family lora] [--sweep tokens] [--quick]
python -m samples.adapter_bench.plot.plot_all --dir samples/adapter_bench/out/sm_120
```

## Arms

| arm | what it runs |
|---|---|
| `comfy` | ComfyUI's arithmetic from `comfy/weight_adapter` (h(), g(), calculate_weight(), weight_decompose()), linear case, copied in `arms/comfy.py` |
| `compile` | `torch.compile` of the comfy arm, compiled per case; a case that falls back to eager is recorded as an error, never measured as compile |
| `eager` | kitchen's torch backend through the public API |
| `kitchen` | the public API as shipped (`comfy_kitchen.adapters.*`) |
| `lycoris` | LoKr only: LyCORIS's register-resident bypass kernel, verbatim, with its own plan space timed per case (forward only; factors up to 128; sm_80+, where its tl.dot reaches tensor cores) |

IA3 has no ComfyUI adapter; its baseline arm is plain torch, named `torch`.

## Sweeps

`tokens` 64..16384, `batch` 1..8 x 4096 tokens, `dim` square 640..5120,
`rank` 4..256, LoKr `factor` (balanced, 4..32), OFT `block` 4..128, conv OFT
`conv_spatial` (2 x 640 x HW) and `conv_batch`, and the merge ops' `merge_dim`,
`merge_rank`, `merge_factor`, `merge_block`. A linear layer sees batch x
context only as their product, so the context sweep at batch 1 is `tokens`.

## Method

Follows LyCORIS `scripts/bench/kernels/harness.py` and KohakUwULLM
`docs/performance/benchmarking.md`:

- L2 flushed (a buffer twice the card's L2) before every timed call.
- Arms interleaved round-robin; each keeps its best round, a round being a median.
- Per row: wall (event-timed single call, device plus host issue), host issue
  time, device time (back-to-back batch), `host_share` and `host_bound`
  (host over half the wall: the wall rate is a floor).
- Forward and forward+backward; gradients cleared inside the timed closure,
  and the backward is checked to reach every leaf before it is timed.
- Peak memory above the resident inputs, per arm and phase.
- Error against the operation's definition in fp64: relative L2 and max ULP
  of the run dtype (rms-scaled for GEMM-like ops, elementwise for IA3), for
  the output and the worst gradient.
- Ceilings measured on the card each run: DRAM (best of copy/read/triad over
  1 GiB) and cuBLAS matmul rate; both go into the JSON meta and every figure.
- bf16 on sm_80+, fp16 on Volta and Turing.

Output: `<out>/<arch>/<family>.json` = `{"meta": {...}, "rows": [...]}`, one
row per (case, arm), written after every case.
