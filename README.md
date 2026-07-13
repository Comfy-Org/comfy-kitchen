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

## HIP backend (AMD RDNA2 / RDNA3 / RDNA3.5 / RDNA4)

The `hip` backend implements the quantized paths with its own kernels: WMMA
matrix-core GEMMs on RDNA3/RDNA4, and non-WMMA kernels (quantizers, RoPE, AdaLN,
the AWQ GEMV) that also run on RDNA2. It does not link or call hipBLAS/hipBLASLt;
every matmul is compiled from the sources in `comfy_kitchen/backends/hip/`.

What a GPU gets depends on whether it has matrix cores:

| Generation | gfx targets                 | Matrix cores | What runs                               |
|------------|-----------------------------|--------------|-----------------------------------------|
| RDNA4      | `gfx1200`, `gfx1201`        | WMMA + fp8   | All HIP-supported kernels, fp8 native   |
| RDNA3.5    | `gfx1150`-`gfx1153`         | WMMA, no fp8 | All HIP-supported kernels; fp8 widened  |
| RDNA3      | `gfx1100`-`gfx1103`         | WMMA, no fp8 | All HIP-supported kernels; fp8 widened  |
| RDNA2      | `gfx1030`-`gfx1036`         | none         | Non-WMMA kernels incl. AWQ GEMV; WMMA GEMMs decline |

fp8, int8 and int4 share one byte-addressed tile kernel (`gemm_wmma.h`). RDNA3
and RDNA4 spread a WMMA operand across the wave differently and RDNA3 has no fp8
WMMA (it widens to bf16, which is exact), so each has its own set of `Mma`
policies in `mma.h`; the tile kernel itself is shared.

RDNA2 has no matrix cores. It runs the kernels that do not need them (RoPE,
AdaLN, the quantizers, stochastic rounding, the AWQ GEMV) and does not advertise
the GEMMs, which fall through to triton/eager. In a process with a mix of GPUs
the capability set is the intersection, since kernels launch on the tensor's own
device.

A request outside a kernel's domain (swizzled operands, scaling other than
tensor-wise, a K that is not a multiple of 16) falls back to torch or eager.
NVFP4 and MXFP8 stay on eager everywhere: RDNA has neither fp4 WMMA nor
microscaling hardware. Set `COMFY_KITCHEN_DISABLE_HIP=1` to remove the backend
from dispatch.

### Building

The backend is built whenever a ROCm toolchain is found. Both a system ROCm
install and the pip `rocm-sdk` layout (which a ROCm PyTorch build already pulls
in) are detected automatically, so on Linux and Windows alike the build is:

```bash
pip install .
```

No environment variables, `CC`/`CXX` override or Visual Studio developer shell
are needed: the ROCm clang builds C, C++ and HIP alike and locates the MSVC
toolchain itself. CMake >= 3.26 and Ninja are required (Windows only ships a
Visual Studio generator, which has no HIP language support). On Windows the
Microsoft C++ build tools and Windows SDK must be installed, since clang links
against them.

Architectures default to the supported GPUs the build machine can see, or to
every RDNA2/3/3.5/4 target ROCm supports when it can see none (a CI box), which
is what the wheels carry. Detection reads the visible devices through PyTorch, so
under PEP 517 build isolation (a plain `pip install .`) it sees nothing and falls
back to the full target list; set `COMFY_HIP_ARCHS`, or pass
`--no-build-isolation`, to build for the local GPU instead. Building for one
target is much faster:

```bash
COMFY_HIP_ARCHS=gfx1201 pip install .
```

```powershell
$env:COMFY_HIP_ARCHS = "gfx1201"; pip install .
```

`PYTORCH_ROCM_ARCH` and `GPU_ARCHS` are honoured too. When the build machine sees
AMD GPUs but none is RDNA2/3/3.5/4 (CDNA has MFMA, not WMMA), the extension is
skipped rather than built (seeing no GPU at all falls back to the full target list
above instead);
`COMFY_KITCHEN_BUILD_HIP=1` makes that a hard error instead, and
`COMFY_KITCHEN_BUILD_NO_HIP=1` suppresses the backend entirely.

Both extensions are built against the Python limited API on 3.12+, so a wheel
carrying CUDA and HIP side by side keeps its `abi3` tag. Each backend withdraws
itself at import when its runtime is absent.


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
- **hip**: Custom HIP kernels (WMMA GEMMs on RDNA3/3.5/4; non-WMMA kernels also on RDNA2)
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
