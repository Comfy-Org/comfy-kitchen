# Comfy Kitchen

Fast kernel library for Diffusion inference with multiple compute backends.

## Backend Capabilities Matrix

| Function                    | eager | cuda | triton | hip | ascend |
|-----------------------------|-------|------|--------|-----|-----|
| `quantize_per_tensor_fp8`   | ✓     | ✓    | ✓      | ✓   |     |
| `dequantize_per_tensor_fp8` | ✓     | ✓    | ✓      | ✓   |     |
| `stochastic_rounding_fp8`   | ✓     | ✓    |        | ✓   |     |
| `quantize_nvfp4`            | ✓     | ✓    | ✓      |     |     |
| `dequantize_nvfp4`          | ✓     | ✓    | ✓      |     |     |
| `scaled_mm_nvfp4`           | ✓     | ✓    |        |     |     |
| `quantize_mxfp8`            | ✓     | ✓    | ✓      |     |     |
| `dequantize_mxfp8`          | ✓     |      |        |     |     |
| `scaled_mm_mxfp8`           | ✓     |      |        |     |     |
| `adaln`                     | ✓     | ✓    | ✓      | ✓   |     |
| `rms_adaln`                 | ✓     | ✓    | ✓      | ✓   |     |
| `na3d`                      | ✓     | ✓    | ✓      | ✓   |     |
| `na2d`                      | ✓     | ✓    | ✓      | ✓   |     |
| `sol_attn`                  | ✓     | ✓    |        | ✓   |     |
| `int8_attention`            |       | ✓    |        | ✓   |     |
| `apply_rope`                | ✓     | ✓    | ✓      | ✓   | ✓   |
| `apply_rope1`               | ✓     | ✓    | ✓      | ✓   | ✓   |
| `apply_rope_split_half`     | ✓     | ✓    | ✓      | ✓   | ✓   |
| `apply_rope_split_half1`    | ✓     | ✓    | ✓      | ✓   | ✓   |
| `rms_rope`                  | ✓     | ✓    | ✓      | ✓   | ✓   |
| `rms_rope1`                 | ✓     | ✓    | ✓      | ✓   | ✓   |
| `rms_rope_split_half`       | ✓     | ✓    | ✓      | ✓   | ✓   |
| `rms_rope_split_half1`      | ✓     | ✓    | ✓      | ✓   | ✓   |
| `quantize_int8_rowwise`     | ✓     | ✓    | ✓      | ✓   | ✓   |
| `quantize_int8_tensorwise`  | ✓     | ✓    |        | ✓   | ✓   |
| `quantize_and_rotate_rowwise` | ✓   | ✓    | ✓      | ✓   | ✓*  |
| `quantize_int8_convrot_weight` | ✓  | ✓    |        | ✓   |     |
| `dequantize_int8_simple`    | ✓     | ✓    |        | ✓   | ✓   |
| `dequantize_int8_simple_dtype` | ✓  | ✓    |        | ✓   | ✓   |
| `dequantize_int8_convrot_weight_dtype` | ✓ | ✓ |    | ✓   |     |
| `int8_linear`               | ✓     | ✓    | ✓      | ✓   | ✓*  |
| `gemv_awq_w4a16`            | ✓     | ✓    |        | ✓   |     |
| `quantize_svdquant_w4a4`    | ✓     | ✓    |        | ✓   |     |
| `scaled_mm_svdquant_w4a4`   | ✓     | ✓    |        | ✓   |     |
| `convrot_w4a4_linear`       | ✓     | ✓    |        | ✓   | ✓*  |
| `quantize_convrot_w4a4_weight` | ✓  | ✓    |        | ✓   |     |
| `dequantize_convrot_w4a4_weight` | ✓ | ✓   |        | ✓   |     |

Each of the eight rope entries also has an in-place form (`apply_rope_`,
`rms_rope_split_half1_`, ...) with the same backend coverage as the row above.

\* Ascend support for `quantize_and_rotate_rowwise` requires
`torch_npu.npu_rotate_quant`; `int8_linear` requires
`torch_npu.npu_dynamic_quant` and `torch_npu.npu_quant_matmul`, while
`convrot_w4a4_linear` requires only `torch_npu.npu_quant_matmul`.
These capabilities are registered only when the
corresponding operator is available.

## Huawei Ascend backend

Ascend `convrot_w4a4_linear` requires `npu_quant_matmul`, not RotateQuant.
It preserves the caller's FP32/FP16/BF16 precision during rotation and signed
INT4 quantization. The packed codes are unpacked to INT8 for NPU INT32
accumulation, then cast and scaled in eager's order. This is not a zero-copy
packed A4W4 kernel: unpacking has a memory/runtime cost. It avoids silently
rounding FP32 inputs to BF16 or narrowing FP32/BF16 results through FP16.
It retains eager's intermediate arithmetic, including FP16 range limitations;
it does not promise higher precision than the reference. Unsupported calls
continue to use the existing eager fallback.

The optional `ascend` backend uses torch-npu operators on Huawei Ascend NPU
hardware. It is registered only when torch-npu and an Ascend device are
available, so comfy-kitchen keeps importing normally on CPU, CUDA, HIP, and XPU
installations. The PyTorch device type remains `npu`, as defined by torch-npu.

The backend supports:

- `quantize_int8_rowwise` through `torch_npu.npu_dynamic_quant`
- `quantize_int8_tensorwise` through Ascend reduction operators and
  `torch_npu.npu_quantize`
- `dequantize_int8_simple` and `dequantize_int8_simple_dtype` on device
- `quantize_and_rotate_rowwise` through `torch_npu.npu_rotate_quant`, when
  available
- `int8_linear` through `torch_npu.npu_quant_matmul`, when available; ConvRot
  uses `torch_npu.npu_rotate_quant` when supported and otherwise keeps the
  separate rotation and dynamic-quantization path
- interleaved and split-half RoPE through `torch_npu.npu_rotary_mul`
- RMS-RoPE through `torch_npu.npu_rms_norm` followed by
  `torch_npu.npu_rotary_mul`

Unsupported dtypes and stochastic rounding continue through another capable
backend instead of copying tensors to the CPU.

The RoPE implementations support four-dimensional BNSD and BSND tensors,
including packed-QKV views whose final dimension is contiguous. Unsupported
broadcast patterns and layouts continue through another capable backend.

RoPE capability detection is independent of quantization: it requires a
`npu_rotary_mul` schema with `rotary_mode`; RMS-RoPE additionally requires
`npu_rms_norm`. The quantization capability checks for `npu_quantize` with
`div_mode`. Missing optional operators do not disable unrelated capabilities.
INT8 linear accepts the current input RMSNorm and residual arguments through
the shared device-side helpers; those operations are not fused into its GEMM.

## HIP backend (AMD Vega / RDNA 1 / RDNA2 / RDNA3 / RDNA3.5 / RDNA4)

The `hip` backend implements the quantized paths with its own kernels: native
WMMA on RDNA3/RDNA3.5/RDNA4 and a software 16x16 tile policy on Vega, RDNA1 and
RDNA2. Every quantized matmul (fp8, int8, int4) is compiled from the sources in
`comfy_kitchen/backends/hip/` and never reaches hipBLAS/hipBLASLt. The only
exception is the unquantized fp16 path: on GPUs without native WMMA, and for
shapes its kernel declines, `fp16_linear` and `fp16_conv3d` run through torch
(rocBLAS and MIOpen) in bounded chunks; see
[Vega, RDNA1 and RDNA2](#vega-rdna1-and-rdna2-pre-wmma-gpus).

Both rope kernels address their inputs through the tensor's own strides, so a q/k
pair permuted or sliced out of a packed qkv is read where it lies rather than
copied contiguous first. The in-place entries (`apply_rope_`, `rms_rope_` and
their split-half and single-tensor siblings) rotate a strided view in place for
the same reason: each thread owns one element pair and loads both before storing
either.

`int8_linear(input_act=...)` folds the activation into the fused ConvRot
quantizer's load, so an MLP's `linear(act(proj(x)))` never writes act's output to
HBM just to read it straight back.

What a GPU gets depends on whether it has matrix cores:

| Generation | gfx targets                 | Matrix cores | What runs                               |
|------------|-----------------------------|--------------|-----------------------------------------|
| RDNA4      | `gfx1200`, `gfx1201`        | WMMA + fp8   | All HIP-supported kernels, fp8 native   |
| RDNA3.5    | `gfx1150`-`gfx1153`         | WMMA, no fp8 | All HIP-supported kernels; fp8 widened  |
| RDNA3      | `gfx1100`-`gfx1103`         | WMMA, no fp8 | All HIP-supported kernels; fp8 widened  |
| RDNA2      | `gfx1030`-`gfx1036`         | `sdot4`      | Software tile GEMMs and attention; fp16 GEMM/conv3d via chunked rocBLAS/MIOpen |
| RDNA1      | `gfx1010`-`gfx1012`         | vector ALU   | Software tile GEMMs and attention; fp16 GEMM/conv3d via chunked rocBLAS/MIOpen |
| Vega (APU) | `gfx90c`                    | vector ALU   | Software tile GEMMs and attention (wave64); fp16 GEMM/conv3d via chunked rocBLAS/MIOpen |
| Vega       | `gfx900`, `gfx906`          | vector ALU   | Software tile GEMMs and attention (wave64); fp16 GEMM/conv3d via chunked rocBLAS/MIOpen |

fp8, int8 and int4 share one byte-addressed tile kernel (`gemm_wmma.h`). RDNA3
and RDNA4 spread a WMMA operand across the wave differently and RDNA3 has no fp8
WMMA (it widens to bf16, which is exact), so each has its own set of `Mma`
policies in `mma.h`; the tile kernel itself is shared.

On pre-WMMA devices, each logical 32-lane wave partition computes the same
16x16 accumulator layout with row broadcasts and vector dot products. RDNA2
uses native packed `sdot4`; RDNA1 and Vega use compiler-generated arithmetic.
NA3D, Sage INT8 attention and Sol attention use the same fragment contract and
run on either policy. They are supported on the validated Vega and RDNA1 targets,
but do not have matrix-core throughput. Flash decode does not use this tile path
and is unavailable on those legacy architectures; it remains limited to its
native-WMMA/BF16 hardware envelope.

### Vega, RDNA1 and RDNA2 (pre-WMMA GPUs)

These GPUs have no matrix cores, so the backend runs the same 16x16 tile
contract as RDNA3/4 in software (see `mma.h`). Each lane owns one output column
and broadcasts the A row it needs; RDNA2 (`gfx103x`) accumulates int8 with its
packed `sdot4` instruction, RDNA1 (`gfx101x`) and Vega use plain integer and
float arithmetic, and Vega (`gfx90x`) keeps logical wave32 tiles on its physical
wave64 with a dedicated 64-lane broadcast. Because the fragment contract is
shared, the tile kernels, epilogues and attention algorithms are the same
source on every target.

On these targets the backend supports:

| Area | Functions | How it runs |
|------|-----------|-------------|
| FP8 quantization | `quantize_per_tensor_fp8`, `dequantize_per_tensor_fp8`, `stochastic_rounding_fp8` | Elementwise kernels; fp8 is a storage format only |
| FP8 GEMM | `torch.nn.functional.linear` on `TensorCoreFP8Layout` tensors (via `scaled_mm_v2`) | Software tile, fp8 widened in registers; tensor-wise scales only |
| INT8 quantization | `quantize_int8_rowwise`, `quantize_int8_tensorwise`, `quantize_and_rotate_rowwise`, `quantize_int8_convrot_weight`, `dequantize_int8_*` | Elementwise and row-reduction kernels |
| INT8 GEMM | `int8_linear` (incl. ConvRot, `input_act`, fused RMSNorm and residual) | Software tile (`sdot4` on RDNA2) |
| INT4 / W4A8 / W6A8 | `convrot_w4a4_linear`, `quantize/dequantize_convrot_w4a4_weight`, `quantize_svdquant_w4a4`, `scaled_mm_svdquant_w4a4`, `gemv_awq_w4a16`, `w4a8_int8_linear` | Software tile; packed codes decoded in registers |
| FP16 GEMM / conv | `fp16_linear`, `fp16_conv3d` | torch (rocBLAS / MIOpen) in 16 MiB chunks, see below |
| Normalization | `adaln`, `rms_adaln`, `group_norm_silu_pad3d` | Native kernels, same as RDNA3/4 |
| RoPE | all `apply_rope*` and `rms_rope*` entries, including in-place | Native strided kernels, same as RDNA3/4 |
| Attention | `na3d`, `na2d`, `sol_attn`, Sage INT8 attention (`int8_attention`) | Software tile; same memory behavior as on RDNA3/4 |

Not available on these targets, so these fall through to another backend (or,
for decode helpers, report unavailable through their `*_is_available()` check):

- Flash decode attention (`flash_attention_decode_is_available()` is false):
  RDNA2 and older have no bf16 arithmetic, and the kernel is written around it.
- GatedDeltaNet fused decode (`gated_delta_decode_is_available()` is false),
  for the same reason.
- The native fp16 GEMM and conv3d kernels. Without matrix cores, the software
  fp16 tile ran 7x to 15x slower than rocBLAS/MIOpen, so `fp16_linear` and
  `fp16_conv3d` use torch there instead. The calls are chunked so their scratch
  stays near 16 MiB: MIOpen's im2col workspace for an unchunked conv3d is about
  30x the output, and a pinned host weight is staged one row block at a time
  instead of being copied to VRAM whole. On `gfx1010` and `gfx90c`, 16 MiB
  chunks run within 8% of 64 MiB ones. The fused bias and residual epilogue is
  kept.
- NVFP4 and MXFP8, as on every HIP target.

Software tiles cost throughput compared with WMMA, so an RDNA2 card will not
reach RDNA3 speeds on the same kernels. The benefit on these GPUs comes from
memory footprint and from avoiding eager's slow paths, not from matrix-core
speed.

#### When to use `hip` instead of `eager` on these GPUs

Dispatch prefers `hip` automatically when it is registered, so this is mainly
about whether to build the extension for an older card and when not to force
`backend="eager"`:

- **INT8 models.** `torch._int_mm` is not usable on `gfx90x`/`gfx10xx`, so
  eager's `int8_linear` widens both int8 operands to fp32 in 1024-wide K chunks
  and runs an fp32 matmul. That costs 4x the operand memory and fp32 GEMM
  throughput. The HIP kernel reads int8 directly (with `sdot4` on RDNA2) and
  fuses activation quantization, ConvRot rotation, the input activation or
  RMSNorm, the dequant scales, bias and residual into one launch.
- **4-bit and FP8 checkpoints.** Eager's AWQ W4A16, ConvRot W4A4 and SVDQuant
  W4A4 paths unpack the packed 4-bit weight into a full-width float tensor
  before calling `matmul`, which temporarily needs about 4x the layer's
  quantized size. The HIP kernels unpack in registers, so a quantized model that
  barely fits in 4-8 GB of VRAM keeps fitting while it runs. FP8 layers
  likewise run straight from the fp8 weight through the software tile.
- **Weights larger than VRAM.** `hip.offload_weight` keeps a linear weight in
  mapped pinned host memory, and the GEMM kernels read it over PCIe (or from
  the APU's shared memory on `gfx90c`) without a VRAM copy. Eager can only
  copy the weight to the device first. This matters most on APUs and 4 GB
  cards such as the RX 5500 XT or RX 6500 XT.
- **Long-sequence video and image attention.** Eager `sol_attn` materializes a
  dense fp32 `(B, H, T, T)` score tensor, and eager `na3d` stacks copies of K
  and V for each window geometry to feed batched SDPA calls. The HIP kernels
  keep scores in registers with an online softmax and stage only small tiles in
  LDS (see below), so sequence lengths that run out of memory on eager can run
  on HIP. Sage INT8 attention has no eager implementation at all.
- **Fused elementwise ops.** AdaLN, RMS-AdaLN, RMS-RoPE, GroupNorm+SiLU+pad
  and the quantizers are each one kernel instead of a chain of torch ops, which
  saves memory traffic on bandwidth-limited cards. The RoPE kernels also rotate
  q/k slices of a packed qkv in place instead of copying them.
- **Video VAEs in fp16.** `fp16_conv3d` bounds MIOpen's workspace to 16 MiB
  windows. A direct `torch.nn.functional.conv3d` on a large video latent can
  allocate an im2col buffer many times the size of its output.

Eager is still the better choice when:

- The inputs are on the CPU, or the op has no HIP implementation (NVFP4,
  MXFP8, flash decode, GatedDeltaNet decode on these targets).
- The shapes are outside a kernel's domain, such as K not a multiple of 16,
  swizzled or non-tensor-wise fp8 scales, or head dimensions a kernel does not
  support. Dispatch falls back automatically, so no action is needed.
- You need a reference to check a HIP result against. The eager backend is
  the implementation the HIP tests compare with.
- Sequences are short. For Sol attention the packed Q/K/V carriers and routing
  workspace can cost more than eager's dense scores.

Set `COMFY_KITCHEN_DISABLE_HIP=1`, or pass `backend="eager"` per call, to
compare the two on your own workload.

### Legacy attention memory behavior

The software-tile policy changes throughput, not the fused attention algorithms'
storage strategy. Sage INT8 attention quantizes Q/K/V into packed INT8 carriers,
uses an online softmax, and stages only K/V tiles in LDS; it does not materialize
a full attention-score matrix. NA3D likewise keeps its score, softmax, and output
accumulators in registers, with only small V/probability tiles in LDS.

Sol attention allocates a caller-owned packed workspace for quantized Q/K/V,
pooled block summaries, routing state, and softmax partials. This avoids eager
Sol attention's dense FP32 `(B, H, T, T)` score tensor, which is especially
important at long sequence lengths. The route index is still block-quadratic
(`O(B * H * ceil(T / 64)^2)`), so the memory reduction is substantial rather
than strictly linear; `token_aug` adds further routing workspace. At short
sequences, the packed carriers and workspace can outweigh the benefit.

Large linear weights can stay in mapped pinned host memory instead of consuming
VRAM:

```python
from comfy_kitchen.backends import hip

host_weight = hip.offload_weight(weight)
output = hip.int8_linear(x, host_weight, weight_scale)
```

Activations, outputs and scales remain on the executing GPU. ROCm and the kernel
driver choose whether pinned pages use ordinary system RAM or an APU GTT aperture;
HIP does not expose a portable API to force Vega GTT placement or a peer-SDMA
route. Pageable CPU tensors are rejected rather than dereferenced by a kernel.

A request outside a kernel's domain (swizzled operands, scaling other than
tensor-wise, a K that is not a multiple of 16) falls back to torch or eager.
NVFP4 and MXFP8 stay on eager everywhere: RDNA has neither fp4 WMMA nor
microscaling hardware. Set `COMFY_KITCHEN_DISABLE_HIP=1` to remove the backend
from dispatch.

### Building

On a ROCm-only host, the backend is selected automatically when CUDA's `nvcc`
is absent. Both a system ROCm install and the pip `rocm-sdk` layout (which a
ROCm PyTorch build already pulls in) are detected, so on Linux and Windows alike
the usual build is:

```bash
pip install .
```

No environment variables, `CC`/`CXX` override or Visual Studio developer shell
are needed: the ROCm clang builds C, C++ and HIP alike and locates the MSVC
toolchain itself. CMake >= 3.26 and Ninja are required (Windows only ships a
Visual Studio generator, which has no HIP language support). On Windows the
Microsoft C++ build tools and Windows SDK must be installed, since clang links
against them and CMake compiles a resource file with the SDK's `rc.exe`, which
the build locates itself rather than expecting on `PATH`. Use the Visual Studio
2022 v143 toolset; newer MSVC toolsets are not yet reliable with ROCm.

When CUDA and ROCm toolchains are both installed, the source build defaults to
CUDA only. This avoids compiling an unused multi-architecture HIP binary on an
NVIDIA workstation. Request a combined build explicitly:

```bash
COMFY_KITCHEN_BUILD_HIP=1 pip install .
```

Architectures default to the validated GPUs the build machine can see, or to
every target in the backend's architecture manifest when it can see none (a CI
box), which is what the wheels carry. Detection reads the visible devices
through PyTorch, so under PEP 517 build isolation (a plain `pip install .`) it
sees nothing and falls back to the full target list; set `COMFY_HIP_ARCHS`, or
pass `--no-build-isolation`, to build for the local GPU instead. Building for
one target is much faster:

```bash
COMFY_HIP_ARCHS=gfx1201 pip install .
```

```powershell
$env:COMFY_HIP_ARCHS = "gfx1201"; pip install .
```

`PYTORCH_ROCM_ARCH` and `GPU_ARCHS` are honoured too. When the build machine sees
AMD GPUs but none is in the architecture manifest (Vega, RDNA1-4; CDNA has MFMA,
not WMMA, and is not covered), the extension is
skipped rather than built (seeing no GPU at all falls back to the full target list
above instead);
`COMFY_KITCHEN_BUILD_HIP=1` requests HIP explicitly (and makes an unsupported
visible AMD GPU a hard error), while `COMFY_KITCHEN_BUILD_NO_HIP=1` suppresses
the backend entirely.

Architecture overrides are exact and fail closed. A compiler-recognized target
that is not in the manifest is rejected until its device and WMMA policies have
been reviewed and added.

Both extensions are built against the Python limited API on 3.12+, so a wheel
carrying CUDA and HIP side by side keeps its `abi3` tag. At runtime only the
extension matching PyTorch's CUDA or ROCm runtime is loaded.


## Quantized Tensors

The library provides `QuantizedTensor`, a `torch.Tensor` subclass that transparently intercepts PyTorch operations and dispatches them to optimized quantized kernels when available.

| Layout                 | Format       | HW Requirement  | Description                             |
|------------------------|--------------|-----------------|----------------------------------------|
| `TensorCoreFP8Layout`  | FP8 E4M3     | SM ≥ 8.9 (Ada)  | Per-tensor scaling, 1:1 element mapping |
| `TensorCoreNVFP4Layout`| NVFP4 E2M1   | SM ≥ 10.0 (Blackwell) | Block quantization with 16-element blocks |
| `TensorCoreMXFP8Layout`| MXFP8 E4M3   | SM ≥ 10.0 (Blackwell) | Block quantization with 32-element blocks, E8M0 scales |

```python
from comfy_kitchen.tensor import QuantizedTensor, TensorCoreFP8Layout, TensorCoreNVFP4Layout

# Quantize a tensor
x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
qt = QuantizedTensor.from_float(x, TensorCoreFP8Layout)

# Operations dispatch to optimized kernels automatically
output = torch.nn.functional.linear(qt, weight_qt)

# Dequantize back to float
dq = qt.dequantize()
```


## Installation

### From PyPI

```bash
# Install default (Linux/Windows/MacOS)
pip install comfy-kitchen

# Install with CUBLAS for NVFP4 (+Blackwell)
pip install comfy-kitchen[cublas]
```

### Package Variants

- **CUDA wheels**: Linux x86_64 and Windows x64
- **Pure Python wheel**: Any platform, eager and triton backends only

Wheels are built for Python 3.10, 3.11, and 3.12+ (using Stable ABI for 3.12+).

### From Source

```bash
# Standard installation with CUDA support
pip install .

# Development installation
pip install -e ".[dev]"

# For faster rebuilds during development (skip build isolation)
pip install -e . --no-build-isolation -v
```

#### Build Options

These options require using `setup.py` directly (not `pip install`):

| Option | Command | Description | Default                                                                     |
|--------|---------|-------------|-----------------------------------------------------------------------------|
| `--no-cuda` | `python setup.py bdist_wheel --no-cuda` | Disable CUDA; without `--hip`, build a CPU-only wheel | Enabled (build with CUDA)                                                   |
| `--hip` | `python setup.py bdist_wheel --hip` | Add HIP explicitly (including to a CUDA build) | Auto only when CUDA is unavailable                                          |
| `--no-hip` | `python setup.py bdist_wheel --no-hip` | Disable HIP | Disabled                                                                    |
| `--hip-archs=...` | `python setup.py build_ext --hip-archs="gfx1200;gfx1201"` | HIP architectures to build for | Visible supported AMD GPUs, otherwise all supported targets                 |
| `--cuda-archs=...` | `python setup.py build_ext --cuda-archs="80;89"` | CUDA architectures to build for | `75-virtual;80;89;90a;100f;120f` (Linux), `75-virtual;80;89;120f` (Windows) |
| `--debug-build` | `python setup.py build_ext --debug-build` | Build in debug mode with symbols | Disabled (Release)                                                          |
| `--lineinfo` | `python setup.py build_ext --lineinfo` | Enable NVCC line info for profiling | Disabled                                                                    |

```bash
# Build CPU-only wheel (pure Python, no CUDA required)
python setup.py bdist_wheel --no-cuda

# Build with custom CUDA architectures
python setup.py build_ext --cuda-archs="80;89" bdist_wheel

# Debug build with line info for profiling
python setup.py build_ext --debug-build --lineinfo bdist_wheel
```



### Requirements

- **Python**: ≥3.10
- **PyTorch**: ≥2.7.0
- **CUDA Runtime** (for CUDA wheels): ≥13.0
  - Pre-built wheels require NVIDIA Driver r580+
  - Building from source requires CUDA Toolkit ≥12.8 and `CUDA_HOME` environment variable
- **nanobind**: ≥2.0.0 (for building from source)
- **CMake**: ≥3.26 (for building from source; the abi3 modules need FindPython's `Development.SABIModule`)

## Quick Start

```python
import comfy_kitchen as ck
import torch

# Automatic backend selection (hip -> cuda -> triton -> eager)
x = torch.randn(100, 100, device="cuda")
scale = torch.tensor([1.0], device="cuda")
result = ck.quantize_per_tensor_fp8(x, scale)

# Check which backends are available
print(ck.list_backends())

# Force a specific backend
result = ck.quantize_per_tensor_fp8(x, scale, backend="eager")

# Temporarily use a different backend
with ck.use_backend("triton"):
    result = ck.quantize_per_tensor_fp8(x, scale)
```

## Backend System

The library supports multiple backends:
- **eager**: Pure PyTorch implementation
- **cuda**: Custom CUDA C kernels (CUDA only)
- **hip**: Custom HIP kernels (native WMMA on RDNA3/3.5/4; software tiles on Vega, RDNA1 and RDNA2)
- **triton**: Triton JIT-compiled kernels

### Automatic Backend Selection

When you call a function, the registry selects the best backend by checking **constraints** in priority order (`hip` → `cuda` → `triton` → `eager`):

```python
# Backend is selected automatically based on input constraints
result = ck.quantize_per_tensor_fp8(x, scale)

# On CPU tensors → falls back to eager (only backend supporting CPU)
# On CUDA tensors → uses cuda or triton (higher priority)
```

### Constraint System

Each backend declares constraints for its functions:

| Constraint | Description |
|------------|-------------|
| **Device** | Which device types are supported |
| **Dtype** | Allowed input/output dtypes per parameter |
| **Shape** | Shape requirements (e.g., 2D tensors, dimensions divisible by 16) |
| **Compute Capability** | Minimum GPU architecture (e.g., SM 8.0 for FP8, SM 10.0 for NVFP4) |

The registry validates inputs against these constraints **before** calling the backend—no try/except fallback patterns. If no backend can handle the inputs, a `NoCapableBackendError` is raised with details.

```python
# Debug logging to see backend selection
import logging
logging.getLogger("comfy_kitchen.dispatch").setLevel(logging.DEBUG)
```


## Testing

Run the test suite with pytest:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_backends.py

# Run with verbose output
pytest -v

# Run specific test
pytest tests/test_backends.py::TestBackendSystem::test_list_backends
```
