# rdna1 backend (gfx1010)

A comfy-kitchen backend for RDNA1 (gfx1010, e.g. RX 5600M / 5700 XT), built for that
one target from its own copy of the HIP sources. It registers only for gfx1010,
declines tensors on any other device, and leaves the `hip` backend and every other
GPU untouched.

## Why `rdna1` is a separate backend rather than another tier in `hip`

- **gfx1010 needs different algorithms.** It has no WMMA, no dot-product instructions, no bf16 arithmetic, and is wave32. The `hip` kernels are built around WMMA tiles, so RDNA1 needs its own GEMMs (`gemm_simt.h`, the packed fp16 FMA GEMM, the implicit-GEMM conv3d) and a software tile policy for attention.
- **`hip` has no tier it fits.** `architectures.json` has `elementwise_only`, `wmma_gfx11` and `wmma_gfx12`. A fourth, non-WMMA tier would thread `#if` branches through every GEMM and attention kernel the supported cards share.
- **Those shared branches have already broken gfx1010 silently.** `#if GFX12 ... #else` code that assumed the else branch meant gfx11 gave wrong masked int8 attention on gfx1010. CI has no RDNA1 hardware, so every HIP port could reintroduce that class of bug.
- **No regression risk for supported cards.** This PR changes no file under `backends/hip`. `rdna1` registers only when a gfx1010 is visible and declines tensors on any other device.
- **The op set differs.** `rdna1` drops flash decode and GatedDeltaNet decode (they need bf16) and adds `gemm_f16_packed` and `transpose_cast`. As a separate backend, the missing ops fall through to eager via the registry instead of needing per-op arch guards inside `hip`.
- **It is easy to drop.** `COMFY_KITCHEN_DISABLE_RDNA1=1` turns it off, and removing it is mostly deleting one directory and its tests, plus the small dispatch shim in `backends/amd.py`.

**Cost:** `rdna1` starts from a copy of the HIP sources, so shared headers (`mma.h`, `hadamard.h`, `sage_common.h` and others) are duplicated, and fixes to them have to be ported by hand.

## Why use it instead of the eager backend

On gfx1010 the eager backend cannot run INT8 models at all: its INT8 matmul is
`torch._int_mm`, which PyTorch sends to hipBLASLt, and ROCm ships no hipBLASLt kernels
for gfx1010. It also has no INT8 attention, so ComfyUI's `--use-ck-attention` refuses
to start, and Triton is no alternative since it miscompiles on this target. This
backend supplies those kernels itself and never calls `torch._int_mm`.

## Hardware constraints

gfx1010 has:

- **no matrix cores (WMMA)**: every GEMM and attention tile is computed in vector
  arithmetic;
- **no dot-product instructions** (`v_dot4`/`sdot4`): int8 products go through
  `v_mad_i32_i24`, or through fp16 FMAs, in which int8 values are exact;
- **no bf16 arithmetic**: bf16 inputs are widened;
- **wave32** natively (built with `-mno-wavefrontsize64`).

PyTorch's ROCm libraries do not cover it fully either: hipBLASLt has no gfx1010
kernels, and MIOpen has no CK grouped-conv library for it.

## What it provides

- Quantized GEMMs (fp8, int8, int4/int6, ConvRot, SVDQuant, AWQ) and the fp16 linear
  on a thread-level, register-blocked GEMM (`gemm_simt.h`); short M packs into 16-
  or 32-row tiles and splits K.
- `fp16_packed_linear` / `fp16_packed_conv3d`: fp16 weights on `v_pk_fma_f16`, partial
  sums folded into fp32 every 32 products, activations of any range scaled by a power
  of two. The conv is an implicit GEMM with no im2col workspace. ConvRot INT8 linears
  with group size 256 run on the same packed GEMM.
- NA3D, Sage INT8 and Sol attention on a software 16x16 tile policy (`mma.h`), plus an
  fp16-FMA Sage kernel for unmasked head_dim 64/128 and `int8_block_sparse_attention`.
- Elementwise kernels: fp8/int8 quantize, RoPE, RMS-RoPE, AdaLN, GroupNorm+SiLU+pad.

Not provided: NA2D, flash decode and the GatedDeltaNet decode (the last two need bf16
arithmetic), and the NVFP4 and MXFP8 layouts. Those fall through to eager. Where the
fused ConvRot quantizer cannot take K, the activation is rotated eagerly and quantized
onto this backend's GEMM.

## Dispatch

The backend registers as `rdna1` when a gfx1010 is visible, and every op's call rule
declines tensors on any other device, so it sits beside `hip` in one process. Where
it registers, priority is `rdna1 -> hip -> eager`; Triton is left out. Calls that
bypass the registry (fp8 `scaled_mm_v2`, Sage attention, `sol_attn_chunked`) resolve
per device through `comfy_kitchen/backends/amd.py`. `COMFY_KITCHEN_DISABLE_RDNA1=1`
removes the backend.

## Build and test

```bash
cd comfy-kitchen
COMFY_HIP_ARCHS="gfx1010" python setup.py build_ext --inplace
HIP_VISIBLE_DEVICES=0 python -m pytest -o addopts="" tests/test_rdna1.py \
    tests/test_fp16_packed_linear.py tests/test_fp16_conv3d.py tests/test_int8_attention.py
```

The INT8 references in `tests/test_rdna1.py` run eager with an exact fp32 matmul in
place of `torch._int_mm`.

## Benchmark: Z-Image Turbo on upstream ComfyUI

Stock upstream ComfyUI on an RX 5600M (6 GB), 2026-10-10. A is the best configuration
the eager backend can run; B is this backend.

| | A: fp8, eager, sub-quadratic attention | B: int8_convrot, rdna1, `--use-ck-attention` |
| --- | --- | --- |
| Cold run (model loading included) | 173.0 s | 105.6 s |
| Warm runs (new seed) | 139.5 s, 143.8 s | 73.9 s, 75.5 s |
| Sampling | 26.7-28.1 s/it | 10.7-10.8 s/it |
| Peak VRAM | 5.2 GB | 3.5 GB |

B samples 2.5x faster and peaks 1.7 GB lower. Both images of a seed show the same
composition.

The int8_convrot models do not run in configuration A: the first sampling step fails
in eager's `int8_linear` with `HIPBLAS_STATUS_INVALID_VALUE` from `torch._int_mm`
(`rocblaslt error: Cannot read "TensileLibrary_lazy_gfx1010.dat"`). fp8 is therefore
the eager baseline.

### Where B's time goes

A B run with the GPU synchronized around each call, per 4-step image:

| Op | Calls | Time | Runs on |
| --- | --- | --- | --- |
| ConvRot INT8 linears (`int8_linear`) | 680 | 33.8 s | rdna1, packed fp16 GEMM |
| Attention (`int8_attention_from_prequantized`) | 136 | 5.8 s | rdna1 |
| RMS-RoPE | 136 | 0.3 s | rdna1 |
| Unquantized linears on the GPU | 24 | under 0.1 s | torch |
| VAE decode | | 22.5 s | CPU (`--cpu-vae`) |

Upstream ComfyUI's own `int8_linear` and attention calls reach these kernels; nothing
in ComfyUI was changed. The VAE decode and, on a cold run, the text encoder run on the
CPU in both configurations, which is why the wall-clock ratio is smaller than the
sampling ratio.

### Setup

- ComfyUI `3a52aae` (upstream master), no custom nodes, Python 3.14.
- torch 2.12.0+rocm10.1.0, torchvision 0.27.0+rocm10.1.0, `amd-torch-device-gfx1010`
  and `rocm-sdk-devel` 10.1.0 from `https://stable.repo.amd.com/rocm/whl-next/`.
- Both: `--disable-dynamic-vram --lowvram --cpu-vae`, `HIP_VISIBLE_DEVICES=0`. A adds
  `--use-quad-cross-attention` and uses comfy-kitchen 0.2.37 from PyPI, where only the
  eager backend is active on gfx1010. B adds `--use-ck-attention` and uses this branch
  built against the same ROCm.
- ComfyUI's Z-Image Turbo template (cfg 1, `res_multistep`/`simple`, AuraFlow shift 3)
  at 4 steps, 1024x1024, batch 1, one fresh server per configuration.
- A: `z-image-turbo_fp8_scaled_e4m3fn_KJ.safetensors` (Kijai/Z-Image_comfy_fp8_scaled)
  and `qwen_3_4b_fp8_mixed.safetensors` (Comfy-Org/z_image_turbo).
- B: `z_image_turbo_int8_convrot.safetensors` (Comfy-Org/z_image_turbo) and
  `qwen3-4b_int8_convrot_fp16emixed.safetensors` (martin-rizzo/Qwen3-4B-INT8-ConvRot-ComfyUI).
- Both: `ae.safetensors` (Comfy-Org/z_image_turbo).

### Caveats

- Upstream's default dynamic VRAM does not load a model on this GPU, so both
  configurations need `--disable-dynamic-vram`.
- `--lowvram` runs the text encoder on the CPU, where this backend declines the
  tensors; the int8_convrot encoder is dequantized by eager there.
- Both configurations keep about 2.2 GB of the diffusion model on the GPU and stream
  the other 3.7 GB from host RAM each step.
- VRAM is device-wide `mem_info_vram_used` sampled at 20 Hz, which includes PyTorch's
  caching allocator; the card holds 20 MB at idle.
- One cold and two warm runs per configuration.
