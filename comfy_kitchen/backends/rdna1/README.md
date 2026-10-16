# rdna1 backend (gfx1010)

A comfy-kitchen backend for RDNA1 (gfx1010, e.g. RX 5600M / 5700 XT), built for that
one target from its own copy of the HIP sources. It registers only for gfx1010,
declines tensors on any other device, and leaves the `hip` backend and every other
GPU untouched.

## Why `rdna1` is a separate backend rather than another tier in `hip`

- **gfx1010 needs different algorithms.** It has no WMMA, no dot-product instructions and no bf16 arithmetic. The `hip` kernels are built around WMMA tiles, so RDNA1 needs its own GEMMs (`gemm_simt.h`, the packed fp16 FMA GEMM, the implicit-GEMM conv3d) and a software tile policy for attention.
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
- **wave32** for the GEMM, conv and attention kernels, which measured 3-11% slower at wave64 on the RX 5600M. The files whose kernels measured faster at wave64 (1.02-1.75x: AdaLN, RoPE, the INT8 and fp8 quantizers, group norm, depthwise conv, the fp16, fp8, W4A8 and AWQ GEMVs, SVDQuant, the Sage quantizers, Sol routing and exact blocks) are built with `-mwavefrontsize64`; the list is in `CMakeLists.txt`.

PyTorch's ROCm libraries do not cover it fully either: hipBLASLt has no gfx1010
kernels, and MIOpen has no CK grouped-conv library for it.

## What it provides

- Quantized GEMMs (fp8, int8, int4/int6, ConvRot, SVDQuant, AWQ) and the fp16 linear
  on a thread-level, register-blocked GEMM (`gemm_simt.h`); short M packs into 16-
  or 32-row tiles and splits K.
- `fp16_packed_linear` / `fp16_packed_conv3d`: fp16 weights on `v_pk_fma_f16`, partial
  sums folded into fp32 every 32 products, activations of any range scaled by a power
  of two. The conv is an implicit GEMM with no im2col workspace. ConvRot INT8 linears
  with group size 256 run on the same packed GEMM. Fetching operands, not arithmetic,
  bounds that GEMM here, so it reads the INT8 weight where it lies and widens it to fp16
  as it stages it (one SDWA conversion per element), and where it takes its 128x128 tile
  the activations are staged in a tiled layout (a tile's stage of K stored contiguously).
- `fp16_packed_conv_transpose3d`: a transposed conv as one stride-1 implicit GEMM per output
  residue (phase), each phase's epilogue writing its positions of the output directly, so
  there is no zero-stuffed input. MIOpen's fp16 transposed conv runs at 0.15-0.2 TFLOPS here;
  this is 13-25x that at audio VAE upsampler shapes and level with MIOpen's fp32 one. Input
  channel counts that are not a multiple of 32 are zero-padded in the fp16 copy of x, for
  this and for `fp16_packed_conv3d`, which runs RGB and latent conv_in shapes 2-12x slower
  than MIOpen; `direct_conv3d` serves those.
- `depthwise_conv3d` / `depthwise_conv_transpose3d`: a gather kernel for one-filter-per-channel
  convs (the vocoders' anti-aliasing FIR filters), 1.6-2.1x MIOpen for the transpose and
  0.8-2.0x for the forward conv.
- `direct_conv3d`: a direct conv in fp32 math that reads x in place, for the convs the
  implicit GEMM serves badly. At 3x3 against MIOpen (time ratio): 3 -> 128 channels over
  512x512 0.42 in fp32 and fp16; 128 -> 3 over 512x512 0.93 in fp32 and 0.17 in fp16, 2.7-3.4x
  the implicit GEMM. MIOpen's fp32 kernels stay ahead past about 64 patch elements and at
  narrow outputs under 512x512 (1.5-2.4x); the implicit GEMM past 8 output channels.
  It also serves 1x1 convs of a channels-first x (up to 512 input channels in fp32, 128 in
  fp16), which the packed conv would first transpose.
- A fused Winograd F(2x2, 3x3) kernel inside `fp16_packed_conv3d`, taken for `[T, 3, 3]`
  filters at spatial stride 1 over a channels-first x with a multiple of 16 channels and
  at least 64 output tiles: 0.5-0.9x the implicit GEMM's time, error 1.5e-3 against its
  8.9e-4. x is first copied channel-blocked (`[N, C/16, D, H, W, 16]`) so patch gathers
  coalesce.
- `fp32_winograd_conv2d` / `fp32_winograd_conv_transpose2d`: MIOpen's own fp32 Winograd RxS
  kernels for gfx10 (`Conv_Winograd_v30_3_1`, f2x3 and f3x2 at stride 1, stride 2 and
  dilation 2; see [miopen_winograd/README.md](miopen_winograd/README.md)), assembled at build
  time and launched from `dlpack_bindings.cpp` without MIOpen. f2x3 serves stride 1 up to 5x5
  filters, f3x2 stride 2 and 7x7. The transposed conv runs them as MIOpen's backward data
  does (filter reversed, K and C swapped). 0.72-1.06x MIOpen's warm time with no per-shape
  search, error ~1e-7; extents must stay under 2^16.
- `q8_0_linear` / `q4_k_linear`: GGUF Q8_0 and Q4_K weights read as their raw blocks by the
  packed GEMM, which applies the per-32-element scales (and Q4_K's offsets, as the offset
  times the row's sum over the 32 elements) as it folds each stage. No fp16 copy of the weight
  is made; shapes the tiled kernel declines (64 rows or fewer, K not a multiple of 256 for
  Q4_K) dequantize to fp16 for `fp16_packed_linear`.

Where MIOpen or rocBLAS stay ahead once warm: 1x1 convs past 256 channels (1.05-1.5x), heads
of a few output channels (1.2-1.4x), small fp16 transposed convs (1.4x), and GEMMs under
about a GFLOP over a short K (the packed linear's floor is ~0.15 ms against rocBLAS's ~0.1).
MIOpen is only that fast after its first call with each shape, which searches for a kernel
(0.2-5 s a shape; gfx1010 ships no find-db), so the RDNA1 ComfyUI fork sends every float conv
here.

- NA3D, Sage INT8 and Sol attention on a software 16x16 tile policy (`mma.h`), plus an
  fp16-FMA Sage kernel for unmasked head_dim 64/128 and `int8_block_sparse_attention`.
- W4A8 linears take the `hip` backend's fused input activation and scaled residual.
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
| Cold run (model loading included) | 173.0 s | 97.6 s |
| Warm runs (new seed) | 139.5 s, 143.8 s | 66.0 s, 65.6 s |
| Sampling | 26.7-28.1 s/it | 9.2-9.3 s/it |
| Peak VRAM | 5.2 GB | 3.6 GB |

B samples 2.9x faster and peaks 1.6 GB lower. Both images of a seed show the same
composition.

The int8_convrot models do not run in configuration A: the first sampling step fails
in eager's `int8_linear` with `HIPBLAS_STATUS_INVALID_VALUE` from `torch._int_mm`
(`rocblaslt error: Cannot read "TensileLibrary_lazy_gfx1010.dat"`). fp8 is therefore
the eager baseline.

### Where B's time goes

A B run with the GPU synchronized around each call, per 4-step image:

| Op | Calls | Time | Runs on |
| --- | --- | --- | --- |
| ConvRot INT8 linears (`int8_linear`) | 680 | 27.6 s | rdna1, packed fp16 GEMM |
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
- The GPU is power-limited under load, so its clock, and with it the sampling rate, moves
  about 5% with temperature: the same build measured 9.2-9.4 s/it from a cool start and
  9.6-9.9 s/it straight after a long run.
