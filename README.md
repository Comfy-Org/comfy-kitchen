# Comfy Kitchen

Fast kernel library for Diffusion inference with multiple compute backends.

## Backend Capabilities Matrix

| Function                    | eager | cuda | triton | rocm |
|-----------------------------|-------|------|--------|------|
| `quantize_per_tensor_fp8`   | ✓     | ✓    | ✓      | ✓    |
| `dequantize_per_tensor_fp8` | ✓     | ✓    | ✓      | ✓    |
| `quantize_nvfp4`            | ✓     | ✓    | ✓      | ✓    |
| `dequantize_nvfp4`          | ✓     | ✓    |        | ✓    |
| `scaled_mm_nvfp4`           | ✓     | ✓    |        | ✓ ¹  |
| `quantize_mxfp8`            | ✓     | ✓    | ✓      | ✓    |
| `dequantize_mxfp8`          | ✓     |      |        | ✓    |
| `scaled_mm_mxfp8`           | ✓     |      |        | ✓ ²  |
| `apply_rope`                | ✓     | ✓    | ✓      | ✓    |
| `apply_rope1`               | ✓     | ✓    | ✓      | ✓    |

> ¹ AMD RDNA hardware lacks native FP4. `scaled_mm_nvfp4` dequantises to BF16 then runs `torch.mm`.
>
> ² Uses `torch._scaled_mm` → hipBLASLt FP8 GEMM on RDNA3 (gfx1100+) and RDNA4 (gfx1200+).
> Falls back to dequant + `torch.mm` on older hardware. TensorWise scaling is used as an
> approximation until PyTorch ROCm exposes block-scaled hipBLASLt directly.


## Quantized Tensors

The library provides `QuantizedTensor`, a `torch.Tensor` subclass that transparently intercepts PyTorch operations and dispatches them to optimized quantized kernels when available.

| Layout                  | Format     | HW Requirement        | Description                                        |
|-------------------------|------------|-----------------------|----------------------------------------------------|
| `TensorCoreFP8Layout`   | FP8 E4M3   | SM ≥ 8.9 (Ada) / RDNA3+ | Per-tensor scaling, 1:1 element mapping          |
| `TensorCoreNVFP4Layout` | NVFP4 E2M1 | SM ≥ 10.0 (Blackwell) | Block quantization with 16-element blocks          |
| `TensorCoreMXFP8Layout` | MXFP8 E4M3 | SM ≥ 10.0 (Blackwell) / RDNA3+ | Block quantization with 32-element blocks, E8M0 scales |

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

> Note: If you are on a system with a non-UTF-8 locale, builds may fail with a `UnicodeDecodeError`. Set `PYTHONUTF8=1` in your environment.

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

### AMD ROCm

The ROCm backend is pure Python and requires no compilation. Install the
ROCm build of PyTorch for your platform, then install comfy-kitchen normally —
everything is detected and configured automatically, with no extra setup needed.

**Supported hardware:** RDNA3 (RX 7000 series, gfx1100+), RDNA4 (RX 9000 series,
gfx1200+), and CDNA3 (MI300X, gfx940+). Older AMD GPUs are supported with
eager fallback for all ops.

### From Source

```bash
# Standard installation with CUDA support
pip install .

# Development installation
pip install -e ".[dev]"

# For faster rebuilds during development (skip build isolation)
pip install -e . --no-build-isolation -v

# Installation with ROCm support
pip install . --no-build-isolation -v
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

# Build with ROCm
python setup.py bdist_wheel

```



### Requirements

- **Python**: ≥3.10
- **PyTorch**: ≥2.5.0
- **CUDA Runtime** (for CUDA wheels): ≥13.0
  - Pre-built wheels require NVIDIA Driver r580+
  - Building from source requires CUDA Toolkit ≥12.8 and `CUDA_HOME` environment variable
- **ROCm PyTorch** (for ROCm backend)
  - No additional compilation or ROCm stack installation required on Windows
- **nanobind**: ≥2.0.0 (for building from source)
- **CMake**: ≥3.18 (for building from source)

## Quick Start

```python
import comfy_kitchen as ck
import torch

# Automatic backend selection (cuda -> rocm -> triton -> eager)
x = torch.randn(100, 100, device="cuda")
scale = torch.tensor([1.0], device="cuda")
result = ck.quantize_per_tensor_fp8(x, scale)

# Check which backends are available
print(ck.list_backends())

# Force a specific backend
result = ck.quantize_per_tensor_fp8(x, scale, backend="eager")

# Temporarily use a different backend
with ck.use_backend("rocm"):
    result = ck.quantize_per_tensor_fp8(x, scale)
```

## Backend System

The library supports multiple backends:

- **eager**: Pure PyTorch implementation, works on any device
- **cuda**: Custom CUDA C kernels (NVIDIA GPUs)
- **triton**: Triton JIT-compiled kernels (NVIDIA and AMD)
- **rocm**: Pure-Python AMD backend using hipBLASLt via `torch._scaled_mm`

### Automatic Backend Selection

When you call a function, the registry selects the best backend by checking **constraints** in priority order (`cuda` → `rocm` → `triton` → `eager`):

```python
# Backend is selected automatically based on input constraints
result = ck.quantize_per_tensor_fp8(x, scale)

# On CPU tensors → falls back to eager (only backend supporting CPU)
# On CUDA tensors (NVIDIA) → uses cuda or triton (higher priority)
# On CUDA tensors (AMD/ROCm) → uses rocm
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
