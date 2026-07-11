# Comfy Kitchen

Fast kernel library for Diffusion inference with multiple compute backends.

## Backend Capabilities Matrix

| Function                    | eager | cuda | triton | hip |
|-----------------------------|-------|------|--------|-----|
| `quantize_per_tensor_fp8`   | ✓     | ✓    | ✓      | ✓   |
| `dequantize_per_tensor_fp8` | ✓     | ✓    | ✓      | ✓   |
| `stochastic_rounding_fp8`   | ✓     | ✓    |        | ✓   |
| `quantize_nvfp4`            | ✓     | ✓    | ✓      |     |
| `dequantize_nvfp4`          | ✓     | ✓    | ✓      |     |
| `scaled_mm_nvfp4`           | ✓     | ✓    |        |     |
| `quantize_mxfp8`            | ✓     | ✓    | ✓      |     |
| `dequantize_mxfp8`          | ✓     |      |        |     |
| `scaled_mm_mxfp8`           | ✓     |      |        |     |
| `adaln`                     | ✓     | ✓    | ✓      | ✓   |
| `apply_rope`                | ✓     | ✓    | ✓      | ✓   |
| `apply_rope1`               | ✓     | ✓    | ✓      | ✓   |
| `apply_rope_split_half`     | ✓     | ✓    | ✓      | ✓   |
| `apply_rope_split_half1`    | ✓     | ✓    | ✓      | ✓   |
| `quantize_int8_rowwise`     | ✓     | ✓    | ✓      | ✓   |
| `quantize_int8_tensorwise`  | ✓     | ✓    |        | ✓   |
| `quantize_and_rotate_rowwise` | ✓   | ✓    | ✓      | ✓   |
| `quantize_int8_convrot_weight` | ✓  | ✓    |        | ✓   |
| `int8_linear`               | ✓     |      | ✓      | ✓   |
| `gemv_awq_w4a16`            | ✓     | ✓    |        | ✓   |
| `quantize_svdquant_w4a4`    | ✓     | ✓    |        | ✓   |
| `scaled_mm_svdquant_w4a4`   | ✓     | ✓    |        | ✓   |
| `convrot_w4a4_linear`       | ✓     | ✓    |        | ✓   |
| `quantize_convrot_w4a4_weight` | ✓  | ✓    |        | ✓   |
| `dequantize_convrot_w4a4_weight` | ✓ | ✓   |        | ✓   |

## HIP backend (AMD RDNA4 / gfx12)

The `hip` backend implements the quantized paths with WMMA kernels written
against the `v_wmma_*_w32_gfx12` intrinsics. It does not link or call
hipBLAS/hipBLASLt; every matmul is compiled from the sources in
`comfy_kitchen/backends/hip/`.

These intrinsics are gfx12-only, so the backend does not register on any other
architecture (including RDNA3/gfx11) and dispatch falls through to triton/eager
there.

| Op | Instruction |
|----|-------------|
| fp8 `scaled_mm`           | `v_wmma_f32_16x16x16_fp8_fp8` |
| `int8_linear`             | `v_wmma_i32_16x16x16_iu8` |
| `convrot_w4a4_linear`     | `v_wmma_i32_16x16x32_iu4` |
| `scaled_mm_svdquant_w4a4` | `v_wmma_i32_16x16x32_iu4`, group-scaled, fused LoRA-up |
| `gemv_awq_w4a16`          | scalar; bandwidth bound, no operand reuse for a tile |

fp8, int8 and int4 share one tile kernel (`gemm_wmma.h`): all three hold 8 bytes
per lane per WMMA, so a K-step is 16 bytes of a row regardless of element type,
and the tile loop is byte-addressed.

`torch._scaled_mm` and `torch._int_mm`, the calls that route to hipBLASLt, are
not reached for any request the backend accepts; `tests/test_hip_wmma.py` traps
both and exercises the fp8, int8, int4, AWQ and SVDQuant paths. A request outside
a kernel's domain still falls back to torch or eager, as described below. NVFP4
and MXFP8 remain on eager throughout: RDNA4 has neither fp4 WMMA nor microscaling
hardware.

ROCm's fp8 `scaled_mm` is competitive on RDNA4 and is faster than the WMMA kernel
on square, deep-K and small-M shapes, while the WMMA kernel is faster at large N.
fp8 is nevertheless routed to WMMA on every call the kernel supports, with no
shape gate, to keep the backend free of BLAS calls. Calls outside its domain
(swizzled operands, scaling other than tensor-wise, a K that is not a multiple of
16) fall back to torch. Set `COMFY_KITCHEN_DISABLE_HIP=1` to remove the backend
from dispatch.

### Building

The backend is built whenever a ROCm toolchain is found. Both a system ROCm
install and the pip `rocm-sdk` layout (which a ROCm PyTorch build already pulls
in) are detected automatically, so on Linux and Windows alike the build is:

```bash
pip install .
```

No environment variables, no `CC`/`CXX` override and no Visual Studio developer
shell are required: the ROCm clang is used for C, C++ and HIP alike, and it
locates the MSVC toolchain by itself. CMake >= 3.21 is required, and Ninja is
used for the HIP extension because CMake's default Visual Studio generator on
Windows does not support the HIP language (both are declared as build
dependencies). On Windows the Microsoft C++ build tools and Windows SDK must be
installed, since clang links against them.

The target architectures default to the gfx12 GPUs the build machine can see.
When it cannot see any -- a CI or cross-build box -- they default to
`gfx1200;gfx1201`. To select them explicitly, set `COMFY_HIP_ARCHS`
(`PYTORCH_ROCM_ARCH` and `GPU_ARCHS` are also honoured):

```bash
COMFY_HIP_ARCHS=gfx1201 pip install .
```

The kernels are gfx12-only, so the extension is skipped rather than built when
the visible AMD GPUs are all of another architecture. Two more environment
variables control that: `COMFY_KITCHEN_BUILD_HIP=1` turns a missing ROCm
toolchain (or a non-gfx12 GPU) into an error instead of a skip, and
`COMFY_KITCHEN_BUILD_NO_HIP=1` suppresses the backend entirely.


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
| `--no-cuda` | `python setup.py bdist_wheel --no-cuda` | Build CPU-only wheel (`py3-none-any`) | Enabled (build with CUDA)                                                   |
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
- **PyTorch**: ≥2.5.0
- **CUDA Runtime** (for CUDA wheels): ≥13.0
  - Pre-built wheels require NVIDIA Driver r580+
  - Building from source requires CUDA Toolkit ≥12.8 and `CUDA_HOME` environment variable
- **nanobind**: ≥2.0.0 (for building from source)
- **CMake**: ≥3.18 (for building from source; ≥3.21 for the HIP backend)

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
- **hip**: Custom HIP WMMA kernels (AMD RDNA4 / gfx12 only)
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
