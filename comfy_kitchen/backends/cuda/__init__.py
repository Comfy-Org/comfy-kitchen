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
import ctypes
import importlib.util
import os
import sys

import torch

__all__ = [
    "adaln",
    "apply_rope",
    "apply_rope1",
    "apply_rope_split_half",
    "apply_rope_split_half1",
    "dequantize_nvfp4",
    "dequantize_per_tensor_fp8",
    "dequantize_int8_simple",
    "dequantize_int8_simple_dtype",
    "dequantize_int8_convrot_weight",
    "dequantize_int8_convrot_weight_dtype",
    "int8_linear",
    "quantize_int8_tensorwise",
    "quantize_int8_rowwise",
    "quantize_int8_convrot_weight",
    "quantize_int8_rowwise_convrot64",
    "quantize_and_rotate_rowwise",
    "gemv_awq_w4a16",
    "quantize_mxfp8",
    "quantize_nvfp4",
    "quantize_per_tensor_fp8",
    "quantize_svdquant_w4a4",
    "scaled_mm_nvfp4",
    "scaled_mm_svdquant_w4a4",
    "stochastic_rounding_fp8",
]


_dll_handle = None
try:
    try:
        import nvidia.cu13
        nvidia_cu13_path = nvidia.cu13.__path__[0]
    except Exception:
        nvidia_cu13_path = torch.__path__[0]

    def find_lib_dir(start_dir, lib_pattern):
        for root, _dirs, files in os.walk(start_dir):
            for file in files:
                if lib_pattern in file:
                    return root
        return None

    if sys.platform == "win32":
        lib_dir = find_lib_dir(nvidia_cu13_path, "cublasLt64")
        if lib_dir:
            _dll_handle = os.add_dll_directory(lib_dir)
    else:
        lib_dir = find_lib_dir(nvidia_cu13_path, "libcublasLt.so")
        if lib_dir:
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

    if not os.path.exists(_module_path):
        ext = '.pyd' if sys.platform == 'win32' else '.so'
        directory = os.path.dirname(__file__)
        for filename in os.listdir(directory):
            if filename.startswith('_C.') and filename.endswith(ext):
                _module_path = os.path.join(directory, filename)

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

from comfy_kitchen.backends._modulation import adaln_prep_modulation  # noqa: E402
from comfy_kitchen.backends.eager.quantization import (  # noqa: E402
    DTYPE_CODE_TO_DTYPE,
    DTYPE_TO_CODE,
)
from comfy_kitchen.backends.eager.quantization import (  # noqa: E402
    dequantize_int8_simple as eager_dequantize_int8_simple,
)
from comfy_kitchen.backends.eager.quantization import (  # noqa: E402
    quantize_int8_tensorwise as eager_quantize_int8_tensorwise,
)
from comfy_kitchen.constraints import (  # noqa: E402
    DivisibleBy,
    ExactDims,
    FunctionConstraints,
    MinDims,
    ParamConstraint,
)
from comfy_kitchen.float_utils import roundup  # noqa: E402
from comfy_kitchen.registry import registry  # noqa: E402
from comfy_kitchen.tensor.int8_utils import (  # noqa: E402
    _build_hadamard,
    _rotate_activation,
    _rotate_weight,
)

_CUBLASLT_AVAILABLE = _EXT_AVAILABLE and getattr(_C, "HAS_CUBLASLT", False)
_cublas_workspaces: dict[int, torch.Tensor] = {}
_empty_cuda_tensors: dict[tuple[str, int | None, torch.dtype], torch.Tensor] = {}
_int8_quant_scratch: dict[tuple[str, int | None, int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}
_int8_gemm_int32_scratch: dict[tuple[str, int | None, int, int, int], torch.Tensor] = {}
_turing_device_cache: dict[int, bool] = {}
_cutlass_int8_device_cache: dict[int, bool] = {}
_shared_memory_per_block_cache: dict[int, int] = {}


def _cuda_device_is_turing(device_index: int) -> bool:
    cached = _turing_device_cache.get(device_index)
    if cached is not None:
        return cached
    is_turing = torch.cuda.get_device_capability(device_index) == (7, 5)
    _turing_device_cache[device_index] = is_turing
    return is_turing


def _cuda_device_supports_cutlass_int8_dequant(tensor: torch.Tensor) -> bool:
    if not tensor.is_cuda:
        return False
    device_index = tensor.get_device()
    cached = _cutlass_int8_device_cache.get(device_index)
    if cached is not None:
        return cached
    major, _minor = torch.cuda.get_device_capability(device_index)
    supported = major >= 8
    _cutlass_int8_device_cache[device_index] = supported
    return supported


def _cublas_int8_n_alignment(tensor: torch.Tensor) -> int:
    # Turing cuBLASLt INT8 rejects some skinny-N shapes, e.g. N=17.
    return 32 if tensor.is_cuda and _cuda_device_is_turing(tensor.get_device()) else 8


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _pad_2d_cols(x: torch.Tensor, padded_cols: int) -> torch.Tensor:
    if x.shape[1] == padded_cols:
        return x
    padding = torch.zeros((x.shape[0], padded_cols - x.shape[1]), dtype=x.dtype, device=x.device)
    return torch.cat((x, padding), dim=1).contiguous()


def _pad_2d_rows(x: torch.Tensor, padded_rows: int) -> torch.Tensor:
    if x.shape[0] == padded_rows:
        return x
    padding = torch.zeros((padded_rows - x.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)
    return torch.cat((x, padding), dim=0).contiguous()


def _pad_1d(x: torch.Tensor, padded_size: int) -> torch.Tensor:
    if x.numel() == padded_size:
        return x
    padding = torch.zeros((padded_size - x.numel(),), dtype=x.dtype, device=x.device)
    return torch.cat((x.reshape(-1), padding), dim=0).contiguous()


def _convrot_fused_shared_memory_fits(x: torch.Tensor, k: int, group_size: int) -> bool:
    if not x.is_cuda or group_size != 256:
        return True
    device_index = x.get_device()
    max_shared = _shared_memory_per_block_cache.get(device_index)
    if max_shared is None:
        props = torch.cuda.get_device_properties(device_index)
        max_shared = getattr(props, "shared_memory_per_block_optin", props.shared_memory_per_block)
        _shared_memory_per_block_cache[device_index] = max_shared
    # The fused convrot rowwise kernel stages the rotated row plus scratch in
    # shared memory. For group_size=256 the CUDA kernel requests this amount.
    requested_shared = (k + 2048) * 4
    return requested_shared <= max_shared


def _should_use_convrot_fused_kernel(x: torch.Tensor, k: int, group_size: int) -> bool:
    return (
        group_size == 256
        and k % 256 == 0
        and k <= _CONVROT_FUSED_MAX_K
        and (k <= 5120 or k >= 8192)
        and _convrot_fused_shared_memory_fits(x, k, group_size)
    )


def _should_use_convrot_dequant_kernel(x: torch.Tensor, k: int, group_size: int) -> bool:
    # Dequant rotates each 256-wide group independently, so it does not need the
    # whole row staged in shared memory like ConvRot quantization does.
    return group_size == 256 and k % 256 == 0 and k <= _CONVROT_FUSED_MAX_K


def get_cublas_workspace_size_bytes() -> int:
    """Return 32 MiB if using hopper, 4 MiB for all other architectures."""
    if torch.cuda.get_device_properties(torch.cuda.current_device()).major >= 9:
        return 33_554_432
    return 4_194_304


def get_cublas_workspace() -> torch.Tensor:
    """Returns workspace for cublas."""
    device_index = torch.cuda.current_device()
    workspace = _cublas_workspaces.get(device_index)
    if workspace is None:
        workspace = torch.empty(
            get_cublas_workspace_size_bytes(),
            dtype=torch.uint8,
            device=device_index,
        )
        _cublas_workspaces[device_index] = workspace
    return workspace


def _empty_cuda_tensor(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (device.type, device.index, dtype)
    empty = _empty_cuda_tensors.get(key)
    if empty is None:
        empty = torch.empty(0, dtype=dtype, device=device)
        _empty_cuda_tensors[key] = empty
    return empty


def _int8_quant_scratch_tensors(
    device: torch.device, stream_ptr: int, num_rows: int, num_cols: int
) -> tuple[torch.Tensor, torch.Tensor]:
    # Internal activation quantization scratch. Key by stream so queued kernels
    # on different streams cannot overwrite each other's temporary q/scale data.
    key = (device.type, device.index, stream_ptr, num_rows, num_cols)
    scratch = _int8_quant_scratch.get(key)
    if scratch is None:
        scratch = (
            torch.empty((num_rows, num_cols), dtype=torch.int8, device=device),
            torch.empty((num_rows, 1), dtype=torch.float32, device=device),
        )
        _int8_quant_scratch[key] = scratch
    return scratch


def _int8_gemm_int32_scratch_tensor(
    device: torch.device, stream_ptr: int, num_rows: int, num_cols: int
) -> torch.Tensor:
    key = (device.type, device.index, stream_ptr, num_rows, num_cols)
    scratch = _int8_gemm_int32_scratch.get(key)
    if scratch is None:
        scratch = torch.empty((num_rows, num_cols), dtype=torch.int32, device=device)
        _int8_gemm_int32_scratch[key] = scratch
    return scratch


def _int8_weight_scale_arg(weight_scale: torch.Tensor, device: torch.device) -> torch.Tensor:
    if weight_scale.device == device and weight_scale.dtype == torch.float32 and weight_scale.is_contiguous():
        return weight_scale
    return weight_scale.to(device=device, dtype=torch.float32).reshape(-1).contiguous()


def _prefer_cublas_int8_fallback(m: int, n: int, k: int) -> bool:
    cutlass_n_le_k_exception = (
        (n == k and n <= 2560)
        or (m >= 1024 and n == 2560 and k == 6912)
    )
    return m > 1 and (
        (n <= k and not cutlass_n_le_k_exception)
        or (m <= 512 and k >= 4096 and n > k and n <= 3 * k)
    )


def _wrap_for_dlpack(tensor: torch.Tensor):
    """Export tensor via DLPack without cross-stream sync.

    Works around PyTorch issue where __dlpack__(stream=None) syncs with
    the default stream, breaking CUDA graph capture on non-default streams.
    See: https://github.com/pytorch/pytorch/pull/163242

    Detaches first so nn.Parameter (requires_grad=True) inputs like bias
    export without PyTorch's gradient-tracking refusal.

    Returns a PyCapsule containing the DLTensor that nanobind can import.
    """
    # stream=-1 tells PyTorch to skip synchronization (DLPack spec)
    if tensor.requires_grad:
        tensor = tensor.detach()
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


def stochastic_rounding_fp8(
    x: torch.Tensor,
    rng: torch.Tensor,
    output_type: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    output_dtype_code = DTYPE_TO_CODE[output_type]

    if not x.is_contiguous():
        x = x.contiguous()
    if not rng.is_contiguous():
        rng = rng.contiguous()

    stream_ptr = torch.cuda.current_stream(x.device).cuda_stream
    _C.stochastic_round_fp8(
        _wrap_for_dlpack(rng),
        _wrap_for_dlpack(x),
        output_dtype_code,
        x.numel(),
        stream_ptr,
    )

    return rng.view(output_type)


def quantize_nvfp4(
    x: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    epsilon: float = 0.0,
    pad_16x: bool = False,
    hi_first: bool = True,
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
        hi_first,
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
    hi_first: bool = True,
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
        hi_first,
        stream_ptr,
    )

    return output


def _int8_rng(
    shape: torch.Size | tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    seed: int | None,
) -> torch.Tensor:
    if seed is None or seed <= 0:
        return torch.empty((0,), dtype=dtype, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.rand(shape, dtype=dtype, device=device, generator=generator)


def quantize_int8_rowwise(
    x: torch.Tensor,
    stochastic_rounding: int | None = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to INT8 with per-row scales (for activations)."""
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    q_2d = torch.empty_like(x_2d, dtype=torch.int8)
    scales_2d = torch.empty((x_2d.shape[0], 1), dtype=torch.float32, device=x.device)
    rng = _int8_rng(q_2d.shape, x.device, x_2d.dtype, stochastic_rounding)
    stream_ptr = torch.cuda.current_stream(x.device).cuda_stream

    _C.quantize_int8_rowwise(
        _wrap_for_dlpack(x_2d),
        _wrap_for_dlpack(q_2d),
        _wrap_for_dlpack(scales_2d),
        _wrap_for_dlpack(rng),
        stochastic_rounding is not None and stochastic_rounding > 0,
        stream_ptr,
    )

    return q_2d.reshape(orig_shape), scales_2d.reshape(*orig_shape[:-1], 1)


def quantize_int8_rowwise_convrot(
    x_2d: torch.Tensor,
    group_size: int,
    stochastic_rounding: int | None = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused online ConvRot rotation + per-row INT8 quantization (single kernel).

    Avoids materializing the rotated bf16 activation in global memory. Expects a
    contiguous 2D [M, K] input with K divisible by group_size (256 only).
    """
    q_2d = torch.empty_like(x_2d, dtype=torch.int8)
    scales_2d = torch.empty((x_2d.shape[0], 1), dtype=torch.float32, device=x_2d.device)
    rng = _int8_rng(q_2d.shape, x_2d.device, x_2d.dtype, stochastic_rounding)
    stream_ptr = torch.cuda.current_stream(x_2d.device).cuda_stream

    _C.quantize_int8_rowwise_convrot(
        _wrap_for_dlpack(x_2d),
        _wrap_for_dlpack(q_2d),
        _wrap_for_dlpack(scales_2d),
        _wrap_for_dlpack(rng),
        group_size,
        stochastic_rounding is not None and stochastic_rounding > 0,
        stream_ptr,
    )

    return q_2d, scales_2d


def rotate_int8_convrot_weight(weight_2d: torch.Tensor, group_size: int) -> torch.Tensor:
    """ConvRot weight rotation using the CUDA FHT kernel."""
    output = torch.empty_like(weight_2d)
    stream_ptr = torch.cuda.current_stream(weight_2d.device).cuda_stream
    _C.rotate_int8_convrot_weight(
        _wrap_for_dlpack(weight_2d),
        _wrap_for_dlpack(output),
        group_size,
        stream_ptr,
    )
    return output


def quantize_int8_convrot_staged(
    weight_2d: torch.Tensor,
    group_size: int,
    stochastic_rounding: int | None = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ConvRot rotation with partial absmax followed by INT8 quantization."""
    n_groups = weight_2d.shape[-1] // group_size
    rotated = torch.empty_like(weight_2d)
    partial_absmax = torch.empty((weight_2d.shape[0], n_groups), dtype=torch.float32, device=weight_2d.device)
    q_2d = torch.empty_like(weight_2d, dtype=torch.int8)
    scales_2d = torch.empty((weight_2d.shape[0], 1), dtype=torch.float32, device=weight_2d.device)
    rng = _int8_rng(q_2d.shape, weight_2d.device, rotated.dtype, stochastic_rounding)
    stream_ptr = torch.cuda.current_stream(weight_2d.device).cuda_stream
    _C.quantize_int8_convrot_staged(
        _wrap_for_dlpack(weight_2d),
        _wrap_for_dlpack(rotated),
        _wrap_for_dlpack(partial_absmax),
        _wrap_for_dlpack(q_2d),
        _wrap_for_dlpack(scales_2d),
        _wrap_for_dlpack(rng),
        group_size,
        stochastic_rounding is not None and stochastic_rounding > 0,
        stream_ptr,
    )
    return q_2d, scales_2d


def quantize_int8_rowwise_convrot64(
    weight_2d: torch.Tensor,
    group_size: int,
    stochastic_rounding: int | None = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused ConvRot row-wise INT8 quantization using 64-lane groups."""
    q_2d = torch.empty_like(weight_2d, dtype=torch.int8)
    scales_2d = torch.empty((weight_2d.shape[0], 1), dtype=torch.float32, device=weight_2d.device)
    rng = _int8_rng(q_2d.shape, weight_2d.device, weight_2d.dtype, stochastic_rounding)
    stream_ptr = torch.cuda.current_stream(weight_2d.device).cuda_stream
    _C.quantize_int8_rowwise_convrot64(
        _wrap_for_dlpack(weight_2d),
        _wrap_for_dlpack(q_2d),
        _wrap_for_dlpack(scales_2d),
        _wrap_for_dlpack(rng),
        group_size,
        stochastic_rounding is not None and stochastic_rounding > 0,
        stream_ptr,
    )
    return q_2d, scales_2d


# The fused kernel holds the whole rotated row ((K + tmp) * 4 bytes) in shared
# memory. It uses a narrow 256-thread block for small K and a wide 1024-thread
# block for large K to keep enough warps resident under the single-block-per-SM
# regime. Cap K so the shared-memory request stays within the opt-in limit;
# larger rows fall back to the rotate-matmul + rowwise-quant path.
_CONVROT_FUSED_MAX_K = 16384

# Set COMFY_KITCHEN_DISABLE_CUTLASS=1 to force the cuBLAS int8 GEMM + separate
# dequant path (for benchmarking against the CUTLASS fused kernel).
_DISABLE_CUTLASS_INT8 = os.environ.get("COMFY_KITCHEN_DISABLE_CUTLASS", "0") == "1"


def quantize_int8_tensorwise(
    x: torch.Tensor,
    scale: torch.Tensor | float | str | None = None,
    stochastic_rounding: int | None = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to INT8 with a single tensor-wise scale."""
    return eager_quantize_int8_tensorwise(x, scale=scale, stochastic_rounding=stochastic_rounding)


def dequantize_int8_simple(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize INT8 tensor with scale."""
    return dequantize_int8_simple_dtype(q, scale, DTYPE_TO_CODE[torch.float32])


def dequantize_int8_simple_dtype(q: torch.Tensor, scale: torch.Tensor, output_dtype_code: int) -> torch.Tensor:
    """Dequantize INT8 tensor with scale into the requested floating output dtype."""
    if not q.is_contiguous():
        q = q.contiguous()

    scale_mode = -1
    inner_dim = q.shape[-1] if q.dim() > 0 else 1
    if scale.numel() == 1:
        scale_mode = 0
    elif tuple(scale.shape) == tuple(q.shape):
        scale_mode = 1
    elif (
        q.dim() > 0
        and scale.dim() == q.dim()
        and tuple(scale.shape[:-1]) == tuple(q.shape[:-1])
        and scale.shape[-1] == 1
    ):
        scale_mode = 2

    if scale_mode < 0:
        return eager_dequantize_int8_simple(q, scale).to(DTYPE_CODE_TO_DTYPE[output_dtype_code])

    scale = scale.to(device=q.device, dtype=torch.float32).contiguous()
    output_dtype = DTYPE_CODE_TO_DTYPE[output_dtype_code]
    output = torch.empty(q.shape, dtype=output_dtype, device=q.device)
    stream_ptr = torch.cuda.current_stream(q.device).cuda_stream
    _C.dequantize_int8_simple(
        _wrap_for_dlpack(q),
        _wrap_for_dlpack(scale.reshape(-1)),
        _wrap_for_dlpack(output),
        inner_dim,
        scale_mode,
        stream_ptr,
    )
    return output


def quantize_and_rotate_rowwise(
    x: torch.Tensor,
    h: torch.Tensor,
    group_size: int,
    stochastic_rounding: int | None = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Online activation rotation followed by CUDA row-wise quantization."""
    x_rot = _rotate_activation(x, h, group_size)
    return quantize_int8_rowwise(x_rot, stochastic_rounding=stochastic_rounding)


def quantize_int8_convrot_weight(
    weight: torch.Tensor,
    group_size: int,
    stochastic_rounding: int | None = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Offline ConvRot weight rotation followed by row-wise INT8 quantization.

    Uses the fused ConvRot CUDA kernel when it matches the linear path's launch
    heuristics, otherwise falls back to explicit rotation plus row-wise quantize.
    """
    if weight.dim() != 2:
        raise ValueError("ConvRot INT8 weight quantization expects a 2D weight tensor")

    weight_2d = weight.contiguous()
    k = weight_2d.shape[-1]
    if (
        group_size == 256
        and k % 256 == 0
        and 256 <= k <= _CONVROT_FUSED_MAX_K
    ):
        return quantize_int8_rowwise_convrot64(weight_2d, group_size, stochastic_rounding=stochastic_rounding)

    if (
        group_size == 256
        and k % 256 == 0
        and 1024 <= k < 8192
    ):
        return quantize_int8_convrot_staged(weight_2d, group_size, stochastic_rounding=stochastic_rounding)

    if _should_use_convrot_fused_kernel(weight_2d, k, group_size):
        return quantize_int8_rowwise_convrot(weight_2d, group_size, stochastic_rounding=stochastic_rounding)

    if group_size == 256 and k % 256 == 0:
        return quantize_int8_convrot_staged(weight_2d, group_size, stochastic_rounding=stochastic_rounding)

    h = _build_hadamard(group_size, device=weight_2d.device, dtype=weight_2d.dtype)
    return quantize_int8_rowwise(_rotate_weight(weight_2d, h, group_size), stochastic_rounding=stochastic_rounding)


def dequantize_int8_convrot_weight(q: torch.Tensor, scale: torch.Tensor, group_size: int) -> torch.Tensor:
    """Dequantize ConvRot INT8 weights and rotate them back to the original basis."""
    return dequantize_int8_convrot_weight_dtype(q, scale, group_size, DTYPE_TO_CODE[torch.float32])


def dequantize_int8_convrot_weight_dtype(
    q: torch.Tensor, scale: torch.Tensor, group_size: int, output_dtype_code: int
) -> torch.Tensor:
    """Dequantize ConvRot INT8 weights and rotate them back into the requested dtype."""
    if q.dim() != 2:
        raise ValueError("ConvRot INT8 weight dequantization expects a 2D q tensor")

    q_2d = q.contiguous()
    k = q_2d.shape[-1]
    output_dtype = DTYPE_CODE_TO_DTYPE[output_dtype_code]
    if _should_use_convrot_dequant_kernel(q_2d, k, group_size):
        scale_arg = scale.to(device=q_2d.device, dtype=torch.float32).reshape(-1).contiguous()
        output = torch.empty(q_2d.shape, dtype=output_dtype, device=q_2d.device)
        stream_ptr = torch.cuda.current_stream(q_2d.device).cuda_stream
        _C.dequantize_int8_convrot_weight(
            _wrap_for_dlpack(q_2d),
            _wrap_for_dlpack(scale_arg),
            _wrap_for_dlpack(output),
            group_size,
            stream_ptr,
        )
        return output

    h = _build_hadamard(group_size, device=q_2d.device, dtype=torch.float32)
    return _rotate_weight(dequantize_int8_simple(q_2d, scale), h, group_size).to(output_dtype)


def int8_gemv_dequant(
    x_qdata: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Single-row INT8 GEMV with fused dequantization."""
    out = torch.empty((1, weight.shape[0]), dtype=out_dtype, device=x_qdata.device)
    bias_arg = bias if bias is not None else _empty_cuda_tensor(x_qdata.device, out_dtype)
    if bias is not None and (bias.device != x_qdata.device or bias.dtype != out_dtype or not bias.is_contiguous()):
        bias_arg = bias.to(device=x_qdata.device, dtype=out_dtype).contiguous()
    stream_ptr = torch.cuda.current_stream(x_qdata.device).cuda_stream
    _C.int8_gemv_dequant(
        _wrap_for_dlpack(x_qdata),
        _wrap_for_dlpack(weight),
        _wrap_for_dlpack(x_scale),
        _wrap_for_dlpack(weight_scale),
        _wrap_for_dlpack(bias_arg),
        _wrap_for_dlpack(out),
        DTYPE_TO_CODE[out_dtype],
        stream_ptr,
    )
    return out


def quantize_mxfp8(
    x: torch.Tensor,
    pad_32x: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous(), "Input tensor must be contiguous"

    orig_rows, orig_cols = x.shape
    if pad_32x:
        num_rows = roundup(orig_rows, 32)
        num_cols = roundup(orig_cols, 32)
    else:
        num_rows, num_cols = orig_rows, orig_cols
        assert num_rows % 32 == 0, f"num_rows must be divisible by 32, got {num_rows}"
        assert num_cols % 32 == 0, f"num_cols must be divisible by 32, got {num_cols}"

    qx = torch.empty((num_rows, num_cols), device=x.device, dtype=torch.float8_e4m3fn, memory_format=torch.contiguous_format)

    scale_rows = roundup(num_rows, 128)
    scale_cols = roundup(num_cols // 32, 4)
    sx_uint8 = torch.zeros((scale_rows, scale_cols), device=x.device, dtype=torch.uint8)

    stream_ptr = torch.cuda.current_stream(x.device).cuda_stream
    _C.quantize_mxfp8(
        _wrap_for_dlpack(x),
        _wrap_for_dlpack(qx),
        _wrap_for_dlpack(sx_uint8),
        pad_32x,
        stream_ptr,
    )

    # View uint8 scales as float8_e8m0fnu before returning
    sx = sx_uint8.view(torch.float8_e8m0fnu)

    return qx, sx


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


def int8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor = None,
    out_dtype: torch.dtype = None,
    convrot: bool = False,
    convrot_groupsize: int = 256,
) -> torch.Tensor:
    orig_shape = x.shape
    x_2d = x if x.dim() == 2 and x.is_contiguous() else x.reshape(-1, x.shape[-1]).contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    stream_ptr = torch.cuda.current_stream(x.device).cuda_stream
    q_scratch = None
    scale_scratch = None

    m, k = x_2d.shape
    n, k_w = weight.shape
    assert k == k_w, "Input and weight inner dimensions must match"

    out_dtype = out_dtype or x.dtype
    output_dtype_code = DTYPE_TO_CODE[out_dtype]
    is_2d_output = len(orig_shape) == 2
    convrot_m1_supported = (
        m == 1
        and convrot
        and convrot_groupsize == 256
        and k % 256 == 0
        and 256 <= k <= _CONVROT_FUSED_MAX_K
    )
    nonconvrot_m1_supported = (
        m == 1
        and not convrot
        and k % 4 == 0
        and (k <= 2560 or (k == 6144 and n <= 128))
    )
    if convrot_m1_supported or nonconvrot_m1_supported:
        q_scratch, scale_scratch = _int8_quant_scratch_tensors(x.device, stream_ptr, 1, k)
        weight_scale = _int8_weight_scale_arg(weight_scale, x.device)
        out = torch.empty((1, n), dtype=out_dtype, device=x.device)
        bias_arg = bias if bias is not None else _empty_cuda_tensor(x.device, out_dtype)
        if bias is not None and (bias.device != x.device or bias.dtype != out_dtype or not bias.is_contiguous()):
            bias_arg = bias.to(device=x.device, dtype=out_dtype).contiguous()
        _C.int8_linear_m1(
            _wrap_for_dlpack(x_2d),
            _wrap_for_dlpack(q_scratch),
            _wrap_for_dlpack(scale_scratch),
            _wrap_for_dlpack(weight),
            _wrap_for_dlpack(weight_scale),
            _wrap_for_dlpack(bias_arg),
            _wrap_for_dlpack(out),
            output_dtype_code,
            convrot,
            convrot_groupsize,
            stream_ptr,
        )
        return out if is_2d_output else out.reshape(*orig_shape[:-1], n)

    # cuBLAS INT8 GEMM requires row-wise quantized activations and tensor-wise quantized weights.
    if convrot:
        k = x_2d.shape[-1]
        # Fused wins for small K (narrow block) and large K (wide block); the
        # 5120 < K < 8192 band loses to the rotate-matmul path on both, so skip
        # it. (Real model hidden dims avoid that band anyway.)
        if (
            convrot_groupsize == 256
            and k % 256 == 0
            and 256 <= k <= _CONVROT_FUSED_MAX_K
        ):
            q_scratch, scale_scratch = _int8_quant_scratch_tensors(x.device, stream_ptr, m, k)
            _C.quantize_int8_rowwise_convrot64(
                _wrap_for_dlpack(x_2d),
                _wrap_for_dlpack(q_scratch),
                _wrap_for_dlpack(scale_scratch),
                _wrap_for_dlpack(_int8_rng((0,), x.device, x_2d.dtype, 0)),
                convrot_groupsize,
                False,
                stream_ptr,
            )
            x_qdata, x_scale = q_scratch, scale_scratch
        elif _should_use_convrot_fused_kernel(x_2d, k, convrot_groupsize):
            # Fused single-kernel rotation + row-wise quant (no bf16 HBM round-trip).
            x_qdata, x_scale = quantize_int8_rowwise_convrot(x_2d, convrot_groupsize)
        else:
            # Fallback: standalone rotation matmul, then row-wise quant.
            h = _build_hadamard(convrot_groupsize, device=x_2d.device, dtype=x_2d.dtype)
            x_qdata, x_scale = quantize_and_rotate_rowwise(x_2d, h, convrot_groupsize)
    else:
        q_scratch, scale_scratch = _int8_quant_scratch_tensors(x.device, stream_ptr, m, k)
        _C.quantize_int8_rowwise(
            _wrap_for_dlpack(x_2d),
            _wrap_for_dlpack(q_scratch),
            _wrap_for_dlpack(scale_scratch),
            _wrap_for_dlpack(_int8_rng((0,), x.device, x_2d.dtype, 0)),
            False,
            stream_ptr,
        )
        x_qdata, x_scale = q_scratch, scale_scratch

    if m == 1 and k % 4 == 0:
        weight_scale = _int8_weight_scale_arg(weight_scale, x.device)
        out = torch.empty((1, n), dtype=out_dtype, device=x.device)
        bias_arg = bias if bias is not None else _empty_cuda_tensor(x.device, out_dtype)
        if bias is not None and (bias.device != x.device or bias.dtype != out_dtype or not bias.is_contiguous()):
            bias_arg = bias.to(device=x.device, dtype=out_dtype).contiguous()
        _C.int8_gemv_dequant(
            _wrap_for_dlpack(x_qdata),
            _wrap_for_dlpack(weight),
            _wrap_for_dlpack(x_scale),
            _wrap_for_dlpack(weight_scale),
            _wrap_for_dlpack(bias_arg),
            _wrap_for_dlpack(out),
            output_dtype_code,
            stream_ptr,
        )
        return out if is_2d_output else out.reshape(*orig_shape[:-1], n)

    out = torch.empty((m, n), dtype=out_dtype, device=x.device)
    weight_scale = _int8_weight_scale_arg(weight_scale, x.device)
    bias_arg = bias if bias is not None else _empty_cuda_tensor(x.device, out_dtype)
    if bias is not None and (bias.device != x.device or bias.dtype != out_dtype or not bias.is_contiguous()):
        bias_arg = bias.to(device=x.device, dtype=out_dtype).contiguous()

    used_cutlass = False
    prefer_cublas_fallback = _prefer_cublas_int8_fallback(m, n, k)
    if (
        not prefer_cublas_fallback
        and not _DISABLE_CUTLASS_INT8
        and _cuda_device_supports_cutlass_int8_dequant(x_qdata)
    ):
        ws_cutlass = weight_scale if weight_scale.numel() == n else weight_scale.expand(n).contiguous()
        bias_f32 = bias_arg.to(torch.float32).contiguous() if bias is not None else bias_arg
        used_cutlass = _C.cutlass_int8_dequant(
            _wrap_for_dlpack(x_qdata),
            _wrap_for_dlpack(weight),
            _wrap_for_dlpack(x_scale),
            _wrap_for_dlpack(ws_cutlass),
            _wrap_for_dlpack(bias_f32),
            _wrap_for_dlpack(out),
            output_dtype_code,
            stream_ptr,
        )
    if not used_cutlass:
        # Fallback: cuBLAS int8 GEMM (int32) + separate dequant kernel.
        use_turing_padding = x_qdata.is_cuda and _cuda_device_is_turing(x_qdata.get_device())
        if use_turing_padding:
            padded_k = _round_up(k, 16)
            padded_n = _round_up(n, _cublas_int8_n_alignment(x_qdata))
            cublas_x = _pad_2d_cols(x_qdata, padded_k)
            cublas_weight = _pad_2d_rows(_pad_2d_cols(weight, padded_k), padded_n)
        else:
            padded_n = n
            cublas_x = x_qdata
            cublas_weight = weight

        out_int32 = _int8_gemm_int32_scratch_tensor(x.device, stream_ptr, m, padded_n)
        _C.cublas_gemm_int8(
            _wrap_for_dlpack(cublas_x),
            _wrap_for_dlpack(cublas_weight),
            _wrap_for_dlpack(out_int32),
            _wrap_for_dlpack(get_cublas_workspace()),
            stream_ptr,
        )
        if padded_n != n:
            out_int32 = out_int32[:, :n].contiguous()
        _C.dequantize_int8_linear(
            _wrap_for_dlpack(out_int32),
            _wrap_for_dlpack(x_scale),
            _wrap_for_dlpack(weight_scale),
            _wrap_for_dlpack(bias_arg),
            _wrap_for_dlpack(out),
            output_dtype_code,
            stream_ptr,
        )

    return out if is_2d_output else out.reshape(*orig_shape[:-1], n)


def adaln(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    orig_shape = x.shape
    d = x.shape[-1]
    n = x.numel() // d

    x_flat = x.reshape(n, d)
    if not x_flat.is_contiguous():
        x_flat = x_flat.contiguous()

    scale_flat, scale_group = adaln_prep_modulation(scale, x, n, d)
    shift_flat, shift_group = adaln_prep_modulation(shift, x, n, d)

    out_flat = torch.empty_like(x_flat)
    dtype_code = DTYPE_TO_CODE[x.dtype]
    stream_ptr = torch.cuda.current_stream(x.device).cuda_stream

    _C.adaln(
        _wrap_for_dlpack(x_flat),
        _wrap_for_dlpack(scale_flat),
        _wrap_for_dlpack(shift_flat),
        _wrap_for_dlpack(out_flat),
        n,
        d,
        scale_group,
        shift_group,
        eps,
        dtype_code,
        stream_ptr,
    )

    return out_flat.reshape(orig_shape)


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
        None,  # xk
        None,  # xk_out
        stream_ptr,
        False,
    )

    return x_out


def apply_rope(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if xq.shape != xk.shape:  # TODO: fix cuda apply_rope to not need this?
        return apply_rope1(xq, freqs_cis), apply_rope1(xk, freqs_cis)

    if not xq.is_contiguous():
        xq = xq.contiguous()
    if not xk.is_contiguous():
        xk = xk.contiguous()
    if not freqs_cis.is_contiguous():
        freqs_cis = freqs_cis.contiguous()

    xq_out = torch.empty_like(xq)
    xk_out = torch.empty_like(xk)
    stream_ptr = torch.cuda.current_stream(xq.device).cuda_stream

    _C.apply_rope(
        _wrap_for_dlpack(xq),
        _wrap_for_dlpack(freqs_cis),
        _wrap_for_dlpack(xq_out),
        _wrap_for_dlpack(xk),
        _wrap_for_dlpack(xk_out),
        stream_ptr,
        False,
    )

    return xq_out, xk_out


def apply_rope_split_half1(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
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
        None,
        None,
        stream_ptr,
        True,
    )

    return x_out


def apply_rope_split_half(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if xq.shape != xk.shape:
        return apply_rope_split_half1(xq, freqs_cis), apply_rope_split_half1(xk, freqs_cis)

    if not xq.is_contiguous():
        xq = xq.contiguous()
    if not xk.is_contiguous():
        xk = xk.contiguous()
    if not freqs_cis.is_contiguous():
        freqs_cis = freqs_cis.contiguous()

    xq_out = torch.empty_like(xq)
    xk_out = torch.empty_like(xk)
    stream_ptr = torch.cuda.current_stream(xq.device).cuda_stream

    _C.apply_rope(
        _wrap_for_dlpack(xq),
        _wrap_for_dlpack(freqs_cis),
        _wrap_for_dlpack(xq_out),
        _wrap_for_dlpack(xk),
        _wrap_for_dlpack(xk_out),
        stream_ptr,
        True,
    )

    return xq_out, xk_out


# ---------------------------------------------------------------------------
# SVDQuant W4A4 — native kitchen kernels (int4 MMA GEMM with per-group dequant).
# ---------------------------------------------------------------------------

_SVDQUANT_W4A4_GROUP_SIZE = 64
_SVDQUANT_W4A4_BLOCK_N = 128
_SVDQUANT_WORKSPACE_CACHE: dict[tuple[object, ...], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _is_svdquant_tile_packed_weight(wgt: torch.Tensor) -> bool:
    return wgt.dim() == 4


def _svdquant_out_features_from_weight(wgt: torch.Tensor) -> int:
    if _is_svdquant_tile_packed_weight(wgt):
        return int(wgt.shape[0]) * _SVDQUANT_W4A4_BLOCK_N
    return int(wgt.shape[0])


def _natural_lora_up_from_tile_packed(lora_up: torch.Tensor) -> torch.Tensor:
    """Return a natural (N, R) view/copy for tile-packed proj_up.

    Converter layout is (N/128, R, 128). cuBLAS addmm_ wants a natural
    row-major (N, R) tensor. Cache the one-time reorder on the source tensor so
    the runtime path keeps cuBLAS speed without paying a per-forward permute.
    """
    if lora_up.dim() != 3:
        return lora_up
    cached = getattr(lora_up, "_ck_natural_lora_up", None)
    if (
        cached is not None
        and cached.device == lora_up.device
        and cached.dtype == lora_up.dtype
        and cached.shape == (lora_up.shape[0] * _SVDQUANT_W4A4_BLOCK_N, lora_up.shape[1])
    ):
        return cached
    natural = lora_up.permute(0, 2, 1).reshape(
        lora_up.shape[0] * _SVDQUANT_W4A4_BLOCK_N, lora_up.shape[1],
    ).contiguous()
    lora_up._ck_natural_lora_up = natural
    return natural


def quantize_svdquant_w4a4(
    x: torch.Tensor,
    smooth: torch.Tensor,
    lora_down: torch.Tensor,
    pad_size: int = 256,
    act_unsigned: bool = False,
    lora_x: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SVDQuant W4A4 activation quantize + smooth + LoRA-down (CUDA).

    Kitchen-native layouts:
      x         (M, K)       bf16/fp16  main-path input (pre-shifted if unsigned)
      smooth    (K,)         bf16/fp16
      lora_down (K, R)       bf16/fp16  natural row-major
      lora_x    (M, K)       bf16/fp16  pre-shift x for LoRA (defaults to x)
      q_x       (M_pad, K/2) int8       two int4 per byte
      ascales   (K/G, M_pad) bf16/fp16  per-row per-group
      lora_act  (M_pad, R)   bf16/fp16 by default (same dtype as x)

    act_unsigned=True selects scale=max/15 + clamp [0,15] (u4 bit patterns for
    downstream u4.s4 MMA). Caller must ensure x is non-negative — shift is a
    model-topology concern kept out of the kernel API. Pass lora_x=raw_x when
    x was pre-shifted (LoRA operates on pre-quantization, pre-shift activations).
    """
    assert x.dim() == 2 and x.is_contiguous(), "x must be 2D contiguous"
    m, k = x.shape
    r = lora_down.shape[1]
    m_pad = roundup(m, pad_size)
    g = _SVDQUANT_W4A4_GROUP_SIZE
    assert k % g == 0, f"K={k} must be divisible by group_size={g}"

    lora_act_dtype = torch.float32 if os.getenv(
        "COMFY_KITCHEN_SVDQUANT_LORA_ACT_FP32", ""
    ).lower() in {"1", "true", "yes", "on"} else x.dtype
    stream_ptr = torch.cuda.current_stream(x.device).cuda_stream
    reuse_workspace = os.getenv(
        "COMFY_KITCHEN_SVDQUANT_REUSE_WORKSPACE", "1"
    ).lower() not in {"0", "false", "no", "off"}
    if reuse_workspace:
        device_index = x.device.index if x.device.index is not None else torch.cuda.current_device()
        key = (device_index, int(stream_ptr), x.dtype, lora_act_dtype, m_pad, k, r)
        cached = _SVDQUANT_WORKSPACE_CACHE.get(key)
        if cached is None:
            cached = (
                torch.empty(m_pad, k // 2, dtype=torch.uint8, device=x.device),
                torch.empty(k // g, m_pad, dtype=x.dtype, device=x.device),
                torch.empty(m_pad, r, dtype=lora_act_dtype, device=x.device),
            )
            _SVDQUANT_WORKSPACE_CACHE[key] = cached
        q_x, ascales, lora_act = cached
    else:
        q_x = torch.empty(m_pad, k // 2, dtype=torch.uint8, device=x.device)
        ascales = torch.empty(k // g, m_pad, dtype=x.dtype, device=x.device)
        lora_act = torch.empty(m_pad, r, dtype=lora_act_dtype, device=x.device)

    _C.svdquant_quantize_w4a4(
        _wrap_for_dlpack(x),
        _wrap_for_dlpack(smooth),
        _wrap_for_dlpack(lora_down),
        _wrap_for_dlpack(q_x),
        _wrap_for_dlpack(ascales),
        _wrap_for_dlpack(lora_act),
        bool(act_unsigned),
        stream_ptr,
    )

    # LoRA-down uses un-shifted, un-smoothed x (SVDQuant invariant). Pure bf16
    # matmul returns bf16/fp16, which is exactly what scaled_mm's Python
    # epilogue consumes. Keeping this buffer 16-bit avoids a per-layer fp32
    # allocation and a later fp32->bf16/fp16 cast without changing CUDA-path
    # numerics. Set COMFY_KITCHEN_SVDQUANT_LORA_ACT_FP32=1 to force the older
    # fp32 staging buffer if a quality regression is suspected.
    lora_src = lora_x if lora_x is not None else x
    if m > 0:
        lora_act_rows = lora_act[:m]
        if lora_act_rows.dtype == lora_src.dtype and lora_act_rows.is_contiguous():
            torch.mm(lora_src, lora_down, out=lora_act_rows)
        else:
            lora_act_rows.copy_(lora_src @ lora_down, non_blocking=True)
    if m_pad > m:
        lora_act[m:].zero_()
    return q_x.view(torch.int8), ascales, lora_act


def scaled_mm_svdquant_w4a4(
    act: torch.Tensor,
    wgt: torch.Tensor,
    ascales: torch.Tensor,
    wscales: torch.Tensor,
    lora_act_in: torch.Tensor,
    lora_up: torch.Tensor,
    bias: torch.Tensor | None = None,
    act_unsigned: bool = False,
) -> torch.Tensor:
    """SVDQuant W4A4 int4 GEMM + LoRA-up + bias (CUDA).

    int4 MMA + per-group dequant + bias run in one kernel. By default, when
    lora_act_in already has the output dtype and proj_up's layout matches the
    weight layout, LoRA-up is fused into the CUDA epilogue with bf16/fp16
    tensor-core MMA. Set COMFY_KITCHEN_SVDQUANT_FUSE_LORA_UP=0 to force the
    older Python/cuBLAS addmm_ epilogue for comparison.

    Kitchen-native layouts:
      act         (M, K/2)   int8       two int4 per byte (signed or unsigned)
      wgt         (N, K/2)   int8       signed int4 weight, natural row-major
        or        (N/128, K/64, 32, 128) int8 kitchen_tile_packed_w4a4
      ascales     (K/G, M)   bf16/fp16  per-row per-group
      wscales     (K/G, N)   bf16/fp16  per-col per-group
        or        (N/128, K/G, 128) bf16/fp16 for tile-packed wgt
      lora_act_in (M, R)     bf16/fp16 or fp32
      lora_up     (N, R) or (N/128, R, 128) bf16/fp16
      bias        (N,) or (N/128, 128) bf16/fp16  (optional)
      out         (M, N)     bf16/fp16  (= lora_up.dtype)

    act_unsigned: if True, A fragments go through u4.s4 MMA instead of s4.s4.
    Set COMFY_KITCHEN_SVDQUANT_FAST_ACCUM=1 to use the experimental packed
    half2/bfloat162 accumulator instead of the default fp32 accumulator.
    """
    m = act.shape[0]
    n = _svdquant_out_features_from_weight(wgt)
    out = torch.empty(m, n, dtype=lora_up.dtype, device=act.device)
    empty = torch.empty(0, dtype=lora_up.dtype, device=act.device)
    fast_accum = os.getenv("COMFY_KITCHEN_SVDQUANT_FAST_ACCUM", "").lower() in {
        "1", "true", "yes", "on",
    }
    shared_scale_env = os.getenv("COMFY_KITCHEN_SVDQUANT_SHARED_SCALE")
    if shared_scale_env is None:
        shared_scale = _is_svdquant_tile_packed_weight(wgt)
    else:
        shared_scale = shared_scale_env.lower() in {"1", "true", "yes", "on"}
    lora_up_layout_matches_wgt = (
        _is_svdquant_tile_packed_weight(wgt) == (lora_up.dim() == 3)
    )
    fuse_lora_env = os.getenv("COMFY_KITCHEN_SVDQUANT_FUSE_LORA_UP")
    fuse_lora = (
        lora_act_in.dtype == out.dtype
        and lora_up_layout_matches_wgt
        and (fuse_lora_env is None or fuse_lora_env.lower() in {"1", "true", "yes", "on"})
    )

    stream_ptr = torch.cuda.current_stream(act.device).cuda_stream
    _C.svdquant_scaled_mm_w4a4(
        _wrap_for_dlpack(act.view(torch.uint8)),
        _wrap_for_dlpack(wgt.view(torch.uint8)),
        _wrap_for_dlpack(ascales),
        _wrap_for_dlpack(wscales),
        _wrap_for_dlpack(lora_act_in),
        _wrap_for_dlpack(lora_up),
        _wrap_for_dlpack(bias if bias is not None else empty),
        _wrap_for_dlpack(out),
        act_unsigned,
        fast_accum,
        shared_scale,
        fuse_lora,
        stream_ptr,
    )

    if not fuse_lora:
        # LoRA-up via bf16/fp16 addmm_. CUDA quantize normally returns lora_act
        # in out.dtype, so the common unfused comparison path avoids a cast.
        lora_bf16 = lora_act_in if lora_act_in.dtype == out.dtype else lora_act_in.to(out.dtype)
        lora_up_mm = _natural_lora_up_from_tile_packed(lora_up)
        out.addmm_(lora_bf16, lora_up_mm.t())
    return out


# Above this M the fused MMA kernel falls behind cuBLAS bf16 GEMM on Blackwell
# (cuBLAS approaches peak; the kernel here is single-thread-per-N-row in the
# dequant pass and lacks cp.async pipelining). Empirically the crossover sits
# near M=256 on RTX 5090 / Qwen-Image-Edit shapes (M=256: 1.6x vs eager,
# M=512: 0.88x). Above the limit we route to a CUDA-side dequant +
# torch.matmul (cuBLAS) path. Future tuning of the MMA kernel will raise
# this limit and eventually remove the fallback.
_AWQ_W4A16_MMA_M_LIMIT = 256


def _awq_w4a16_dequant_then_matmul(
    x: torch.Tensor,
    qweight: torch.Tensor,
    wscales: torch.Tensor,
    wzeros: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Large-M fallback: dequantize qweight to bf16 then bf16 cuBLAS matmul.

    qweight (N, K/2) int8 packed uint4 → W (N, K) bf16 via group dequant, then
    `out = x @ W.T`. Same algebra as eager.gemv_awq_w4a16 — used for M values
    where the in-kitchen MMA kernel is slower than cuBLAS bf16 GEMM.
    """
    n, k_half = qweight.shape
    k = k_half * 2
    g = group_size
    compute_dtype = wscales.dtype

    x32 = qweight.to(torch.int32)
    lo = (x32 & 0xF).to(torch.int8)
    hi = ((x32 >> 4) & 0xF).to(torch.int8)
    nibbles = torch.stack([lo, hi], dim=-1).reshape(n, k).to(compute_dtype)
    w = (
        (nibbles.view(n, k // g, g) - 8.0) * wscales.t().unsqueeze(-1)
        + wzeros.t().unsqueeze(-1)
    ).view(n, k)
    return x.matmul(w.t())


def gemv_awq_w4a16(
    x: torch.Tensor,
    qweight: torch.Tensor,
    wscales: torch.Tensor,
    wzeros: torch.Tensor,
    bias: torch.Tensor | None = None,
    group_size: int = 64,
) -> torch.Tensor:
    """AWQ W4A16 matmul: int4 weight @ fp activation (CUDA, kitchen-native).

    Tiered routing:
      M ≤ 8                    naive 1-thread-per-output kernel (GEMV-style)
      8 < M ≤ 512              fused int4 x bf16/fp16 MMA kernel — dequant
                               into shmem, mma.m16n8k16.f32 along K, no
                               intermediate bf16 W workspace
      M > 512                  dequant + cuBLAS bf16 matmul fallback

    bias is applied externally (`out.add_`), mirroring
    scaled_mm_svdquant_w4a4's epilogue contract.

    Layouts (match comfy_kitchen.backends.eager.awq):
      x        (M, K)   bf16/fp16  row-major activation. Leading dims are
                                   flattened.
      qweight  (N, K/2) int8       two unsigned int4 per byte
      wscales  (K/G, N) bf16/fp16  per-group, per-output-col scale
      wzeros   (K/G, N) bf16/fp16  per-group, per-output-col zero
      bias     (N,)     bf16/fp16  optional (= wscales.dtype)
      out      (..., N) bf16/fp16  same dtype as wscales
    """
    orig_shape = x.shape
    x2d = x.reshape(-1, orig_shape[-1])
    m, k = x2d.shape
    n = qweight.shape[0]
    if k % group_size != 0:
        raise ValueError(f"K={k} not divisible by group_size={group_size}")
    if qweight.shape[1] * 2 != k:
        raise ValueError(f"qweight K//2={qweight.shape[1]} inconsistent with x K={k}")

    if m > _AWQ_W4A16_MMA_M_LIMIT:
        out2d = _awq_w4a16_dequant_then_matmul(
            x2d.contiguous().to(wscales.dtype), qweight, wscales, wzeros, group_size,
        )
    else:
        out2d = torch.empty(m, n, dtype=wscales.dtype, device=x.device)
        stream_ptr = torch.cuda.current_stream(x.device).cuda_stream
        _C.awq_w4a16(
            _wrap_for_dlpack(x2d.contiguous().to(wscales.dtype)),
            _wrap_for_dlpack(qweight.view(torch.uint8)),
            _wrap_for_dlpack(wscales),
            _wrap_for_dlpack(wzeros),
            _wrap_for_dlpack(out2d),
            group_size,
            stream_ptr,
        )

    if bias is not None:
        out2d.add_(bias)

    return out2d.reshape(*orig_shape[:-1], n)


def _build_constraints() -> dict:
    cuda_devices = frozenset({"cuda"})

    constraints = {
        "adaln": FunctionConstraints(
            params={
                "x": ParamConstraint(
                    dtypes=frozenset({torch.float32, torch.float16, torch.bfloat16}),
                ),
                "scale": ParamConstraint(
                    dtypes=frozenset({torch.float32, torch.float16, torch.bfloat16}),
                ),
                "shift": ParamConstraint(
                    dtypes=frozenset({torch.float32, torch.float16, torch.bfloat16}),
                ),
            },
            default_devices=cuda_devices,
        ),
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
        "stochastic_rounding_fp8": FunctionConstraints(
            params={
                "x": ParamConstraint(
                    dtypes=frozenset({torch.float32, torch.float16, torch.bfloat16}),
                ),
                "rng": ParamConstraint(dtypes=frozenset({torch.uint8})),
                "output_type": ParamConstraint(
                    dtypes=frozenset({torch.float8_e4m3fn, torch.float8_e5m2}),
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
        "quantize_mxfp8": FunctionConstraints(
            params={
                "x": ParamConstraint(
                    dtypes=frozenset({torch.float32, torch.float16, torch.bfloat16}),
                    shape_rules=(ExactDims(2),),
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
        "apply_rope1": FunctionConstraints(
            params={
                "x": ParamConstraint(
                    dtypes=frozenset({torch.float16, torch.bfloat16}),
                    shape_rules=(ExactDims(4),),
                ),
                "freqs_cis": ParamConstraint(
                    dtypes=frozenset({torch.float32, torch.float16, torch.bfloat16}),
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
                    dtypes=frozenset({torch.float32, torch.float16, torch.bfloat16}),
                    shape_rules=(ExactDims(6),),
                ),
            },
            default_devices=cuda_devices,
        ),
        "quantize_int8_tensorwise": FunctionConstraints(
            params={
                "x": ParamConstraint(
                    dtypes=frozenset({torch.float32, torch.float16, torch.bfloat16}),
                ),
                "scale": ParamConstraint(
                    dtypes=frozenset({torch.float32, torch.float16, torch.bfloat16, float, str}),
                ),
                "stochastic_rounding": ParamConstraint(dtypes=frozenset({int})),
            },
            default_devices=cuda_devices,
        ),
        "quantize_int8_rowwise": FunctionConstraints(
            params={
                "x": ParamConstraint(
                    dtypes=frozenset({torch.float32, torch.float16, torch.bfloat16}),
                ),
                "stochastic_rounding": ParamConstraint(dtypes=frozenset({int})),
            },
            default_devices=cuda_devices,
        ),
        "quantize_and_rotate_rowwise": FunctionConstraints(
            params={
                "x": ParamConstraint(
                    dtypes=frozenset({torch.float32, torch.float16, torch.bfloat16}),
                ),
                "H": ParamConstraint(
                    dtypes=frozenset({torch.float32, torch.float16, torch.bfloat16}),
                ),
                "group_size": ParamConstraint(dtypes=frozenset({int})),
                "stochastic_rounding": ParamConstraint(dtypes=frozenset({int})),
            },
            default_devices=cuda_devices,
        ),
        "quantize_int8_convrot_weight": FunctionConstraints(
            params={
                "weight": ParamConstraint(
                    dtypes=frozenset({torch.float32, torch.float16, torch.bfloat16}),
                    shape_rules=(ExactDims(2),),
                ),
                "group_size": ParamConstraint(dtypes=frozenset({int})),
                "stochastic_rounding": ParamConstraint(dtypes=frozenset({int})),
            },
            default_devices=cuda_devices,
        ),
        "dequantize_int8_convrot_weight": FunctionConstraints(
            params={
                "q": ParamConstraint(
                    dtypes=frozenset({torch.int8}),
                    shape_rules=(ExactDims(2),),
                ),
                "scale": ParamConstraint(
                    dtypes=frozenset({torch.float32, torch.float16, torch.bfloat16}),
                ),
                "group_size": ParamConstraint(dtypes=frozenset({int})),
            },
            default_devices=cuda_devices,
        ),
        "dequantize_int8_convrot_weight_dtype": FunctionConstraints(
            params={
                "q": ParamConstraint(
                    dtypes=frozenset({torch.int8}),
                    shape_rules=(ExactDims(2),),
                ),
                "scale": ParamConstraint(
                    dtypes=frozenset({torch.float32, torch.float16, torch.bfloat16}),
                ),
                "group_size": ParamConstraint(dtypes=frozenset({int})),
                "output_dtype_code": ParamConstraint(dtypes=frozenset({int})),
            },
            default_devices=cuda_devices,
        ),
        "dequantize_int8_simple": FunctionConstraints(
            params={
                "q": ParamConstraint(
                    dtypes=frozenset({torch.int8}),
                ),
                "scale": ParamConstraint(
                    dtypes=frozenset({torch.float32, torch.float16, torch.bfloat16}),
                ),
            },
            default_devices=cuda_devices,
        ),
        "dequantize_int8_simple_dtype": FunctionConstraints(
            params={
                "q": ParamConstraint(
                    dtypes=frozenset({torch.int8}),
                ),
                "scale": ParamConstraint(
                    dtypes=frozenset({torch.float32, torch.float16, torch.bfloat16}),
                ),
                "output_dtype_code": ParamConstraint(dtypes=frozenset({int})),
            },
            default_devices=cuda_devices,
        ),
        "apply_rope_split_half1": FunctionConstraints(
            params={
                "x": ParamConstraint(
                    dtypes=frozenset({torch.float16, torch.bfloat16}),
                    shape_rules=(ExactDims(4),),
                ),
                "freqs_cis": ParamConstraint(
                    dtypes=frozenset({torch.float32, torch.float16, torch.bfloat16}),
                    shape_rules=(ExactDims(6),),
                ),
            },
            default_devices=cuda_devices,
        ),
        "apply_rope_split_half": FunctionConstraints(
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
                    dtypes=frozenset({torch.float32, torch.float16, torch.bfloat16}),
                    shape_rules=(ExactDims(6),),
                ),
            },
            default_devices=cuda_devices,
        ),
        "quantize_svdquant_w4a4": FunctionConstraints(
            params={
                "x": ParamConstraint(
                    dtypes=frozenset({torch.float16, torch.bfloat16}),
                    shape_rules=(ExactDims(2), DivisibleBy(dim=1, factor=64)),
                ),
                "smooth": ParamConstraint(
                    dtypes=frozenset({torch.float16, torch.bfloat16}),
                ),
                "lora_down": ParamConstraint(
                    dtypes=frozenset({torch.float16, torch.bfloat16}),
                    shape_rules=(ExactDims(2),),
                ),
            },
            default_devices=cuda_devices,
            min_compute_capability=(8, 0),
        ),
        "scaled_mm_svdquant_w4a4": FunctionConstraints(
            params={
                "act": ParamConstraint(dtypes=frozenset({torch.int8}), shape_rules=(ExactDims(2),)),
                "wgt": ParamConstraint(dtypes=frozenset({torch.int8})),
                "ascales": ParamConstraint(dtypes=frozenset({torch.float16, torch.bfloat16})),
                "wscales": ParamConstraint(dtypes=frozenset({torch.float16, torch.bfloat16})),
                "lora_act_in": ParamConstraint(
                    dtypes=frozenset({torch.float32, torch.float16, torch.bfloat16})
                ),
                "lora_up": ParamConstraint(dtypes=frozenset({torch.float16, torch.bfloat16})),
            },
            default_devices=cuda_devices,
            min_compute_capability=(8, 0),
        ),
        "gemv_awq_w4a16": FunctionConstraints(
            params={
                "x": ParamConstraint(
                    dtypes=frozenset({torch.float16, torch.bfloat16}),
                ),
                "qweight": ParamConstraint(
                    dtypes=frozenset({torch.int8}),
                    shape_rules=(ExactDims(2),),
                ),
                "wscales": ParamConstraint(
                    dtypes=frozenset({torch.float16, torch.bfloat16}),
                    shape_rules=(ExactDims(2),),
                ),
                "wzeros": ParamConstraint(
                    dtypes=frozenset({torch.float16, torch.bfloat16}),
                    shape_rules=(ExactDims(2),),
                ),
            },
            default_devices=cuda_devices,
            min_compute_capability=(8, 0),
        ),
    }

    if _CUBLASLT_AVAILABLE:
        constraints["int8_linear"] = FunctionConstraints(
            params={
                "x": ParamConstraint(
                    dtypes=frozenset({torch.float32, torch.float16, torch.bfloat16}),
                    shape_rules=(MinDims(2),),
                ),
                "weight": ParamConstraint(
                    dtypes=frozenset({torch.int8}),
                    shape_rules=(ExactDims(2),),
                ),
                "weight_scale": ParamConstraint(
                    dtypes=frozenset({torch.float32}),
                ),
                "bias": ParamConstraint(
                    dtypes=frozenset({torch.float32, torch.float16, torch.bfloat16}),
                ),
                "out_dtype": ParamConstraint(
                    dtypes=frozenset({torch.float32, torch.float16, torch.bfloat16}),
                ),
                "convrot": ParamConstraint(dtypes=frozenset({bool})),
                "convrot_groupsize": ParamConstraint(dtypes=frozenset({int})),
            },
            default_devices=cuda_devices,
            min_compute_capability=(7, 5),
        )
        constraints["scaled_mm_nvfp4"] = FunctionConstraints(
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
        )

    return constraints


def _register():
    """Register CUDA backend with the global registry."""
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
