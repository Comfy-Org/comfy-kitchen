# Kitchen CUDA W4A4 kernel — historical design plan

> **Status (2026-04)**: Design document kept for reference. Kernels are
> implemented and in production — actual source is in this directory's
> `quantize_svdquant_w4a4.cu` and `scaled_mm_svdquant_w4a4.cu`. File headers in
> those `.cu` files document the current tile layout, pipeline depth, dequant
> chain, and the act_unsigned mode. The phased rollout below describes the
> order work was done, not the present architecture.

**Original goal**: native implementation of `quantize_svdquant_w4a4` and
`scaled_mm_svdquant_w4a4` that consumes **kitchen-native row-major** tensors.
No nunchaku vendor code. sm_80+ (A100, L40, 4090, 5090 all supported).

Performance target: match or approach nunchaku's vendored kernel (1.67 s/iter
on RTX 5090 for Qwen-Image-Edit 20 steps @ 1024×1024).

---

## Layouts (kitchen-native, inputs to the kernels)

All tensors are row-major contiguous; no fragment tile packing.

| tensor | shape | dtype | notes |
|---|---|---|---|
| `x` (input activation) | (M, K) | bf16/fp16 | row-major |
| `smooth` | (K,) | bf16/fp16 | per-channel smoothing |
| `lora_down` | (K, R) | bf16/fp16 | SVDQuant down projection |
| `qweight` | (N, K/2) | int8 | row-major; low nibble = q[n, 2k], high nibble = q[n, 2k+1]; values in [-8, 7] |
| `wscales` | (K/G, N) | bf16/fp16 | per-group weight scale; G=64 for INT4 |
| `lora_up` | (N, R) | bf16/fp16 | SVDQuant up projection |
| `bias` | (N,) | bf16/fp16 | optional |

**Outputs of `quantize_svdquant_w4a4`**:
| tensor | shape | dtype | notes |
|---|---|---|---|
| `q_x` | (M_pad, K/2) | int8 | same packing as qweight |
| `ascales` | (K/G, M_pad) | bf16/fp16 | per-row per-group scale |
| `lora_act` | (M_pad, R) | fp32 | fp32 for accumulation stability |

**Output of `scaled_mm_svdquant_w4a4`**: `out (M, N) bf16/fp16`.

`M_pad = ceil(M / pad_size) * pad_size`, pad_size=256.

---

## Architecture

Two CUDA kernels. Each launches one grid per forward call; no cuBLAS
dependency for the int4 path.

### Kernel A: `quantize_svdquant_w4a4`

Fuses three ops over the activation tensor:
1. `x_smooth = x / smooth` (broadcast along M)
2. per-row per-group absmax → scale = absmax / 7 → int4 quantize + pack
3. `lora_act = x @ lora_down`  (note: uses raw x, NOT smoothed x — matches
   SVDQuant convention where lora_down is calibrated against un-smoothed input)

**Tiling**:
- `BLOCK_M = 32` (output tokens per CTA)
- `BLOCK_N = 128` (input features per CTA, = nunchaku's BLOCK_N to match G=64 × 2)
- grid: `(ceil(M / BLOCK_M), K / BLOCK_N)`
- Each CTA has 4 warps

**Per-CTA work**:
- Load `x[M_offset:M_offset+BLOCK_M, K_offset:K_offset+BLOCK_N]` via cp.async
- Load `smooth[K_offset:K_offset+BLOCK_N]` via cp.async
- Load `lora_down[K_offset:K_offset+BLOCK_N, :]` via cp.async
- Compute `x_smooth = x / smooth` in registers
- Intra-warp reduce: within each 64-wide K group (so BLOCK_N/64 = 2 groups
  per CTA), find absmax across the 64 K elements per row. XOR shfl tree
  within warp.
- Scale = absmax / 7; broadcast to all lanes in group via shfl
- Quantize: `round(x_smooth / scale)` clamp to [-8, 7] → pack 2 int4s per byte
- atomicAdd lora_act = x @ lora_down in fp32 (or use split-K)

**Key challenge**: lora_act output atomicAdd across K blocks. Options:
- (a) Use atomicAdd (simple but slow)
- (b) Allocate lora_act as (K/BLOCK_N, M_pad, R) and split-K reduce later
- (c) Compute LoRA in a separate grid after quantize finishes

Option (a) is simplest. For Qwen R=96 it's manageable.

### Kernel B: `scaled_mm_svdquant_w4a4`

Fuses four ops:
1. int4 GEMM: `q_act (M, K/2) @ qweight^T (K/2, N) → int32 (M, N)` via
   `mma.m16n8k64.s4.s4.s32`
2. Dequant: `out_fp = int32 * ascales[:, None] * wscales[None, :]`
3. LoRA up: `out_fp += lora_act @ lora_up^T`
4. Bias: `out_fp += bias`

**Tiling**:
- `BLOCK_M = 64`, `BLOCK_N = 128`, `BLOCK_K = 256` (4 groups of 64)
- `WARP_M = 32`, `WARP_N = 64` → 2×2 warps per CTA
- MMA m16n8k64: each warp covers (32 rows × 64 cols) per K-stage =
  (WARP_M/16) × (WARP_N/8) = 2×8 = 16 MMA instructions per K-stage

**Per-CTA work**:
- Async load act chunk and wgt chunk into shmem
- ldmatrix.sync.aligned.m8n8.x4 into int4 fragments
- mma.m16n8k64.s4.s4.s32 to accumulate int32 psum
- At end of K-loop: dequant by multiplying with (ascales × wscales) per group,
  convert to fp32 accumulator
- LoRA up: load lora_act (M_pad, R) fragment + lora_up (N, R) fragment,
  do R-rank matmul in bf16, accumulate to fp32 out
- Add bias (N,)
- Write out as bf16/fp16

**Key challenge**: int4 MMA lane mapping is nontrivial. Reference:
- PTX ISA §9.7.14.5.12 (mma.m16n8k64 int4/int1)
- NVIDIA CUTLASS examples/44_multi_gemm_ir for int4 GEMM skeleton

---

## File layout

```
comfy_kitchen/backends/cuda/ops/
  quantize_svdquant_w4a4.cu   # kernel A
  gemm_svdquant_w4a4.cu        # kernel B
  svdquant_utils.cuh           # shared helpers (int4 pack/unpack device-side)
```

Plus:
- `dlpack_bindings.cpp`: add `svdquant_quantize_w4a4` + `svdquant_scaled_mm_w4a4` nb::defs
- `CMakeLists.txt`: add 2 sources to `CUDA_SOURCES`
- `comfy_kitchen/backends/cuda/__init__.py`: Python wrappers + FunctionConstraints registration

---

## Phased development

To de-risk the 30-40 hour effort, break into independently verifiable phases.

### Phase 1: design + shared utilities (~4 hours)
- Write `svdquant_utils.cuh`: device helpers for int4 pack/unpack, warp-level
  absmax reduction
- Build system: empty stubs for kernels A and B, verify CMake builds and
  nanobind finds symbols
- Kitchen `__init__.py` wrappers that raise `NotImplementedError`

### Phase 2: bare-bones scaled_mm (no LoRA, no bias) (~10 hours)
- Minimal int4×int4 → int32 MMA GEMM
- Dequant to bf16 out
- Unit test: compare to pure-PyTorch `unpack(q_act) * ascales @ (unpack(qweight) * wscales)^T`
- Validate on layer 0 of Qwen-Image-Edit kitchen-native checkpoint
- Target: rel_max < 1e-3 (bit-close to manual int32 accumulation)

### Phase 3: bare-bones quantize (no LoRA) (~8 hours)
- Per-row per-group absmax via warp XOR reduction
- int4 packing
- smooth division
- Unit test: compare `q_x, ascales` to pure-PyTorch reference
- Target: rel_max < 1e-3

### Phase 4: fuse LoRA into both (~8 hours)
- Quantize kernel: add `x @ lora_down` with fp32 atomicAdd
- GEMM kernel: add `lora_act @ lora_up^T` accumulated into fp32 output path
- Validate end-to-end: single layer parity < 5% (matches current eager tolerance)

### Phase 5: bias + polish (~3 hours)
- GEMM kernel: final bias add
- Full ComfyUI end-to-end run — target < 3 s/iter

### Phase 6: tuning (optional, ~5 hours)
- Profile via ncu, tune BLOCK_M/N/K, pipelining depth
- Target: within 1.5x of nunchaku vendor (so ~2.5 s/iter)

---

## Risks

| risk | mitigation |
|---|---|
| int4 MMA lane mapping mis-decoded | write a minimal toy test first: feed 16×64 identity matrix, verify output matches scalar reference |
| sm_120 (Blackwell RTX 5090) int4 MMA behavior differs from sm_80/89 | test on 5090 (primary dev GPU), reference behavior on sm_80 via code review |
| atomicAdd on lora_act becomes serialized bottleneck | fallback: split-K allocation with reduction pass |
| cp.async + ldmatrix sequencing bugs (shmem bank conflicts) | use CUTLASS swizzling patterns |
| sm_75 requested by default CUDA archs (75-virtual;80;89;...) — no int4 MMA | skip sm_75 in COMFY_CUDA_ARCHS for svdquant sources, or fall back to eager |

---

## Open questions (to user before starting)

1. **Target CUDA arch list**: current kitchen default is `75-virtual;80;89;90a;100f;120f`.
   int4 MMA needs sm_80+. OK to narrow to `80;89;90a;100f;120f` for svdquant
   sources? (other kitchen ops unaffected)
2. **FP4 path**: do we need NVFP4 support now, or INT4 only? NVFP4 requires
   sm_100 block-scale MMA. Propose INT4-only in phase B; FP4 if/when needed.
3. **Interleaved vs separate quantize and GEMM dispatch**: current eager
   splits these into two custom_ops, so CUDA mirrors that. OK to keep?
4. **Build time tolerance**: naive approach compiles ~30 seconds; template-heavy
   could be minutes. Acceptable?
5. **Testing budget**: after each phase gate, should I run full ComfyUI e2e
   (takes ~2 min) or just unit tests (~10 sec)? Recommend unit+ small e2e.
