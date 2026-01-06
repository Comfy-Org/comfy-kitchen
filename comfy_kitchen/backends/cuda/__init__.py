# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import contextlib
import importlib.util
import os
import sys

import torch

__all__ = [
    "apply_rope",
    "apply_rope1",
    "dequantize_nvfp4",
    "dequantize_per_tensor_fp8",
    "quantize_nvfp4",
    "quantize_per_tensor_fp8",
    "scaled_mm_nvfp4",
]


_dll_handle = None
try:
    import nvidia.cu13

    def find_lib_dir(start_dir, lib_pattern):
        for root, _dirs, files in os.walk(start_dir):
            for file in files:
                if lib_pattern in file:
                    return root
        return None

    nvidia_cu13_path = nvidia.cu13.__path__[0]

    if sys.platform == "win32":
        lib_dir = find_lib_dir(nvidia_cu13_path, "cublasLt64")
        if lib_dir:
            _dll_handle = os.add_dll_directory(lib_dir)
    else:
        lib_dir = find_lib_dir(nvidia_cu13_path, "libcublasLt.so")
        if lib_dir:
            import ctypes
            for filename in os.listdir(lib_dir):
                if "cublasLt" in filename and ".so" in filename:
                    with contextlib.suppress(Exception):
                        ctypes.CDLL(os.path.join(lib_dir, filename), mode=ctypes.RTLD_GLOBAL)
except Exception:
    pass  # nvidia.cu13 not installed or path doesn't exist


# Load _C extension using importlib to avoid circular import issues on Windows
try:
    _C = None  # type: ignore
    _module_path = os.path.join(os.path.dirname(__file__), "_C.abi3.pyd" if sys.platform == "win32" else "_C.abi3.so")

    if os.path.exists(_module_path):
        _spec = importlib.util.spec_from_file_location(
            "comfy_kitchen.backends.cuda._C", _module_path
        )
        if _spec and _spec.loader:
            _C = importlib.util.module_from_spec(_spec)
            sys.modules["comfy_kitchen.backends.cuda._C"] = _C
            _spec.loader.exec_module(_C)
            _EXT_AVAILABLE = True
            _EXT_ERROR = None
        else:
            _EXT_AVAILABLE = False
            _EXT_ERROR = f"Could not create module spec for {_module_path}"
    else:
        _EXT_AVAILABLE = False
        _EXT_ERROR = f"Extension file not found: {_module_path}"
except ImportError as e:
    _EXT_AVAILABLE = False
    _EXT_ERROR = str(e)
    _C = None  # type: ignore
except Exception as e:
    _EXT_AVAILABLE = False
    _EXT_ERROR = f"Failed to load extension: {e}"
    _C = None  # type: ignore

from comfy_kitchen.backends.eager.quantization import DTYPE_TO_CODE  # noqa: E402
from comfy_kitchen.float_utils import roundup  # noqa: E402

_cublas_workspace: torch.Tensor | None = None


def get_cublas_workspace_size_bytes() -> int:
    """Return 32 MiB if using hopper, 4 MiB for all other architectures."""
    if torch.cuda.get_device_properties(torch.cuda.current_device()).major >= 9:
        return 33_554_432
    return 4_194_304


def get_cublas_workspace() -> torch.Tensor:
    """Returns workspace for cublas."""
    global _cublas_workspace
    if _cublas_workspace is None:
        _cublas_workspace = torch.empty(
            get_cublas_workspace_size_bytes(), dtype=torch.uint8, device="cuda"
        )
    return _cublas_workspace


def _wrap_for_dlpack(tensor: torch.Tensor):
    """Export tensor via DLPack without cross-stream sync.

    Works around PyTorch issue where __dlpack__(stream=None) syncs with
    the default stream, breaking CUDA graph capture on non-default streams.
    See: https://github.com/pytorch/pytorch/pull/163242

    Returns a PyCapsule containing the DLTensor that nanobind can import.
    """
    # stream=-1 tells PyTorch to skip synchronization (DLPack spec)
    return tensor.__dlpack__(stream=-1)


def quantize_per_tensor_fp8(
    x: torch.Tensor, scale: torch.Tensor, output_type: torch.dtype = torch.float8_e4m3fn
) -> torch.Tensor:
    input_dtype_code = DTYPE_TO_CODE[x.dtype]
    output_dtype_code = DTYPE_TO_CODE[output_type]

    if not x.is_contiguous():
        x = x.contiguous()

    result_uint8 = torch.empty(x.shape, dtype=torch.uint8, device=x.device)

    numel = x.numel()
    stream_ptr = torch.cuda.current_stream(x.device).cuda_stream
    _C.quantize_per_tensor_fp8(
        _wrap_for_dlpack(x),
        _wrap_for_dlpack(scale),
        _wrap_for_dlpack(result_uint8),
        input_dtype_code,
        output_dtype_code,
        numel,
        stream_ptr,
    )

    return result_uint8.view(output_type)


def dequantize_per_tensor_fp8(
    x: torch.Tensor, scale: torch.Tensor, output_type: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    assert scale.numel() == 1, "Scale must be a scalar tensor"

    input_dtype_code = DTYPE_TO_CODE[x.dtype]
    output_dtype_code = DTYPE_TO_CODE[output_type]

    result = torch.empty(x.shape, dtype=output_type, device=x.device)
    numel = x.numel()
    stream_ptr = torch.cuda.current_stream(x.device).cuda_stream

    _C.dequantize_per_tensor_fp8(
        _wrap_for_dlpack(x.view(torch.uint8)),
        _wrap_for_dlpack(scale),
        _wrap_for_dlpack(result),
        input_dtype_code,
        output_dtype_code,
        numel,
        stream_ptr,
    )

    return result


def quantize_nvfp4(
    x: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    epsilon: float = 0.0,
    pad_16x: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    # CUDA backend: uses cuBLAS tiled layout for block scales
    assert x.is_contiguous(), "Input tensor must be contiguous"

    orig_rows, orig_cols = x.shape
    if pad_16x:
        num_rows = roundup(orig_rows, 16)
        num_cols = roundup(orig_cols, 16)
    else:
        num_rows, num_cols = orig_rows, orig_cols
        assert num_rows % 16 == 0, f"num_rows must be divisible by 16, got {num_rows}"
        assert num_cols % 16 == 0, f"num_cols must be divisible by 16, got {num_cols}"

    # Allocate output tensors
    # FP4: 2 values per uint8, so output is half the column size
    qx = torch.empty((num_rows, num_cols // 2), device=x.device, dtype=torch.uint8, memory_format=torch.contiguous_format)

    # Block scales: cuBLAS tiled layout
    # One scale per 16-element block, with tiling pattern
    # Allocate as uint8 for DLPack compatibility (nanobind doesn't handle float8 well)
    # Initialize to zero to avoid garbage in padded regions
    scale_rows = roundup(num_rows, 128)
    scale_cols = roundup(num_cols // 16, 4)
    sx_uint8 = torch.zeros((scale_rows, scale_cols), device=x.device, dtype=torch.uint8)

    # Reshape scalar to 1D for nanobind compatibility
    if per_tensor_scale.dim() == 0:
        per_tensor_scale = per_tensor_scale.reshape(1)

    stream_ptr = torch.cuda.current_stream(x.device).cuda_stream
    _C.quantize_nvfp4(
        _wrap_for_dlpack(x),
        _wrap_for_dlpack(per_tensor_scale),
        _wrap_for_dlpack(qx),
        _wrap_for_dlpack(sx_uint8),
        epsilon,
        pad_16x,
        stream_ptr,
    )

    # View uint8 scales as float8_e4m3fn before returning
    sx = sx_uint8.view(torch.float8_e4m3fn)

    return qx, sx


def dequantize_nvfp4(
    qx: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    block_scales: torch.Tensor,
    output_type: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    assert qx.is_contiguous(), "Input tensor must be contiguous"

    num_rows, num_cols_packed = qx.shape
    num_cols = num_cols_packed * 2  # Each uint8 contains 2 FP4 values

    output = torch.empty((num_rows, num_cols), device=qx.device, dtype=output_type)

    # Reshape scalar to 1D for nanobind compatibility
    if per_tensor_scale.dim() == 0:
        per_tensor_scale = per_tensor_scale.reshape(1)

    block_scales_uint8 = block_scales.view(torch.uint8)
    output_dtype_code = DTYPE_TO_CODE[output_type]
    stream_ptr = torch.cuda.current_stream(qx.device).cuda_stream

    _C.dequantize_nvfp4(
        _wrap_for_dlpack(qx),
        _wrap_for_dlpack(per_tensor_scale),
        _wrap_for_dlpack(block_scales_uint8),
        _wrap_for_dlpack(output),
        output_dtype_code,
        stream_ptr,
    )

    return output


def scaled_mm_nvfp4(
    a: torch.Tensor,
    b: torch.Tensor,
    tensor_scale_a: torch.Tensor,
    tensor_scale_b: torch.Tensor,
    block_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    bias: torch.Tensor = None,
    out_dtype: torch.dtype = None,
    alpha: torch.Tensor = None,
) -> torch.Tensor:
    # CUDA backend: cuBLAS FP4 GEMM (TN layout, K%32==0, N%8==0 required)
    # Scale layout: (RoundUp(M/N, 128), RoundUp(K//16, 4)) in swizzled format
    # See: https://docs.nvidia.com/cuda/cublas/index.html?highlight=fp4#d-block-quantization
    accumulate: bool = False
    block_length: int = 16

    # Convert Parameters to Tensors for nanobind compatibility (do this early)
    if isinstance(tensor_scale_a, torch.nn.Parameter):
        tensor_scale_a = tensor_scale_a.data
    if isinstance(tensor_scale_b, torch.nn.Parameter):
        tensor_scale_b = tensor_scale_b.data

    if alpha is None:
        alpha = tensor_scale_a * tensor_scale_b
    elif isinstance(alpha, torch.nn.Parameter):
        alpha = alpha.data

    # Ensure alpha is float32 and 1D with shape [1] for nanobind compatibility
    if alpha.dtype != torch.float32:
        alpha = alpha.to(torch.float32)
    if alpha.dim() == 0:
        alpha = alpha.reshape(1)

    # Convert remaining Parameters to Tensors for nanobind compatibility
    if isinstance(a, torch.nn.Parameter):
        a = a.data
    if isinstance(b, torch.nn.Parameter):
        b = b.data
    if isinstance(block_scale_a, torch.nn.Parameter):
        block_scale_a = block_scale_a.data
    if isinstance(block_scale_b, torch.nn.Parameter):
        block_scale_b = block_scale_b.data

    m, k_a = a.shape
    n, k_b = b.shape
    assert k_a == k_b, "Matrix dimensions do not match"

    # k is the number of FP4 elements in a row of a and b
    k = 2 * k_a  # 2 FP4 in 1 uint8 container

    out = torch.empty(m, n, dtype=out_dtype, device=a.device)

    # N must be aligned for cuBLAS (K alignment covered by constraint DivisibleBy(16))
    assert n % 8 == 0, "B tensor must have 8 alignment in N dimension"

    # Check scale layout
    assert block_scale_a.dtype == block_scale_b.dtype, "A and B scale dtype must match"

    if block_scale_a.dtype == torch.float8_e8m0fnu:
        # MXFP4: scales are E8M0, and stored in torch.uint8
        assert block_length == 32, "MXFP4 only supports block length 32"
        raise ValueError("MXFP4 is not supported yet for cuBLAS in CUDA 12.9")
    elif block_scale_a.dtype == torch.float8_e4m3fn:
        # NVFP4: scales are E4M3, and stored in torch.float8_e4m3fn
        assert block_length == 16, "NVFP4 only supports block length 16"
        assert alpha is not None, "alpha must be provided for NVFP4"
        assert alpha.dtype == torch.float32, "alpha must be float32"
        assert alpha.numel() == 1, "alpha must be a scalar"
    else:
        raise ValueError(f"Unsupported scale dtype: {block_scale_a.dtype}")

    roundup_m = roundup(m, 128)
    roundup_n = roundup(n, 128)
    # k is multiple of 32, so k / block_length is integer,
    roundup_sk = roundup(k // block_length, 4)

    assert block_scale_a.dim() == 2, "Invalid A scale shape"
    assert block_scale_a.size() == (roundup_m, roundup_sk), "Invalid A scale shape"

    assert block_scale_b.dim() == 2, "Invalid B scale shape"
    assert block_scale_b.size() == (roundup_n, roundup_sk), "Invalid B scale shape"

    if bias is None:
        bias = torch.Tensor()
    else:
        assert bias.dtype in (
            torch.float16,
            torch.bfloat16,
        ), "Only fp16 and bfloat16 bias are supported."

    # NVFP4/MXFP4 in sm100 supports TN layout only
    _transa, _transb = True, False

    # View float8 scales as uint8 for passing to C++
    block_scale_b_uint8 = block_scale_b.view(torch.uint8)
    block_scale_a_uint8 = block_scale_a.view(torch.uint8)

    out_dtype_code = DTYPE_TO_CODE[out_dtype]

    stream_ptr = torch.cuda.current_stream(a.device).cuda_stream

    # Handle empty bias
    if bias is None or bias.numel() == 0:
        bias = torch.empty(0, device=a.device, dtype=torch.float16)
    else:
        # Convert Parameter to Tensor for nanobind compatibility
        if isinstance(bias, torch.nn.Parameter):
            bias = bias.data

    _C.cublas_gemm_blockwise_fp4(
        _wrap_for_dlpack(b),
        _wrap_for_dlpack(block_scale_b_uint8),
        _wrap_for_dlpack(a),
        _wrap_for_dlpack(block_scale_a_uint8),
        _wrap_for_dlpack(out),
        out_dtype_code,
        _wrap_for_dlpack(bias),
        _wrap_for_dlpack(get_cublas_workspace()),
        accumulate,
        _wrap_for_dlpack(alpha),
        stream_ptr,
    )

    return out


def apply_rope1(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    if not x.is_contiguous():
        x = x.contiguous()
    if not freqs_cis.is_contiguous():
        freqs_cis = freqs_cis.contiguous()

    x_out = torch.empty_like(x)
    stream_ptr = torch.cuda.current_stream(x.device).cuda_stream

    _C.apply_rope(
        _wrap_for_dlpack(x),
        _wrap_for_dlpack(freqs_cis),
        _wrap_for_dlpack(x_out),
        stream_ptr,
    )

    return x_out


def apply_rope(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    xq_out = apply_rope1(xq, freqs_cis)
    xk_out = apply_rope1(xk, freqs_cis)
    return xq_out, xk_out


def _build_constraints() -> dict:
    from comfy_kitchen.constraints import (
        DivisibleBy,
        ExactDims,
        FunctionConstraints,
        ParamConstraint,
    )

    cuda_devices = frozenset({"cuda"})

    return {
        "quantize_per_tensor_fp8": FunctionConstraints(
            params={
                "x": ParamConstraint(
                    dtypes=frozenset({torch.float32, torch.float16, torch.bfloat16}),
                ),
                "scale": ParamConstraint(
                    dtypes=frozenset({torch.float32}),
                ),
                "output_type": ParamConstraint(
                    dtypes=frozenset({torch.float8_e4m3fn, torch.float8_e5m2}),
                ),
            },
            default_devices=cuda_devices,
        ),
        "dequantize_per_tensor_fp8": FunctionConstraints(
            params={
                "x": ParamConstraint(
                    dtypes=frozenset({torch.float8_e4m3fn, torch.float8_e5m2}),
                ),
                "scale": ParamConstraint(
                    dtypes=frozenset({torch.float32}),
                ),
                "output_type": ParamConstraint(
                    dtypes=frozenset({torch.float32, torch.float16, torch.bfloat16}),
                ),
            },
            default_devices=cuda_devices,
        ),
        "quantize_nvfp4": FunctionConstraints(
            params={
                "x": ParamConstraint(
                    dtypes=frozenset({torch.float32, torch.float16, torch.bfloat16}),
                    shape_rules=(ExactDims(2),),
                ),
                "per_tensor_scale": ParamConstraint(
                    dtypes=frozenset({torch.float32}),
                ),
            },
            default_devices=cuda_devices,
        ),
        "dequantize_nvfp4": FunctionConstraints(
            params={
                "qx": ParamConstraint(
                    dtypes=frozenset({torch.uint8}),
                    shape_rules=(ExactDims(2), DivisibleBy(dim=0, factor=16)),
                ),
                "per_tensor_scale": ParamConstraint(
                    dtypes=frozenset({torch.float32}),
                ),
                "block_scales": ParamConstraint(
                    dtypes=frozenset({torch.float8_e4m3fn}),
                ),
                "output_type": ParamConstraint(
                    dtypes=frozenset({torch.float32, torch.float16, torch.bfloat16}),
                ),
            },
            default_devices=cuda_devices,
        ),
        "scaled_mm_nvfp4": FunctionConstraints(
            params={
                "a": ParamConstraint(
                    dtypes=frozenset({torch.uint8}),
                    shape_rules=(ExactDims(2), DivisibleBy(dim=1, factor=16)),
                ),
                "b": ParamConstraint(
                    dtypes=frozenset({torch.uint8}),
                    shape_rules=(ExactDims(2), DivisibleBy(dim=1, factor=16)),
                ),
                "tensor_scale_a": ParamConstraint(
                    dtypes=frozenset({torch.float32}),
                ),
                "tensor_scale_b": ParamConstraint(
                    dtypes=frozenset({torch.float32}),
                ),
                "block_scale_a": ParamConstraint(
                    dtypes=frozenset({torch.float8_e4m3fn}),
                ),
                "block_scale_b": ParamConstraint(
                    dtypes=frozenset({torch.float8_e4m3fn}),
                ),
                "out_dtype": ParamConstraint(
                    dtypes=frozenset({torch.float16, torch.bfloat16}),
                ),
            },
            default_devices=cuda_devices,
            min_compute_capability=(10, 0),
        ),
        "apply_rope1": FunctionConstraints(
            params={
                "x": ParamConstraint(
                    dtypes=frozenset({torch.float16, torch.bfloat16}),
                    shape_rules=(ExactDims(4),),
                ),
                "freqs_cis": ParamConstraint(
                    dtypes=frozenset({torch.float32}),
                    shape_rules=(ExactDims(6),),
                ),
            },
            default_devices=cuda_devices,
        ),
        "apply_rope": FunctionConstraints(
            params={
                "xq": ParamConstraint(
                    dtypes=frozenset({torch.float16, torch.bfloat16}),
                    shape_rules=(ExactDims(4),),
                ),
                "xk": ParamConstraint(
                    dtypes=frozenset({torch.float16, torch.bfloat16}),
                    shape_rules=(ExactDims(4),),
                ),
                "freqs_cis": ParamConstraint(
                    dtypes=frozenset({torch.float32}),
                    shape_rules=(ExactDims(6),),
                ),
            },
            default_devices=cuda_devices,
        ),
    }


def _register():
    """Register CUDA backend with the global registry."""
    from comfy_kitchen.registry import registry

    if not _EXT_AVAILABLE:
        registry.mark_unavailable("cuda", _EXT_ERROR)
        return

    if not torch.cuda.is_available():
        registry.mark_unavailable("cuda", "CUDA not available on this system")
        return

    registry.register(
        name="cuda",
        module=__import__(__name__, fromlist=__all__),
        capabilities=_build_constraints(),
    )


_register()
