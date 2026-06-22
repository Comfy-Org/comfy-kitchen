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
    "apply_rope_split_half",
    "apply_rope_split_half1",
    "dequantize_nvfp4",
    "dequantize_per_tensor_fp8",
    "int8_linear",
    "quantize_int8_rowwise",
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

from comfy_kitchen.backends.eager.quantization import DTYPE_TO_CODE  # noqa: E402
from comfy_kitchen.float_utils import roundup  # noqa: E402

_CUBLASLT_AVAILABLE = _EXT_AVAILABLE and getattr(_C, "HAS_CUBLASLT", False)
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


def quantize_int8_rowwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to INT8 with per-row scales (for activations)."""
    from comfy_kitchen.backends.eager.quantization import quantize_int8_rowwise as eager_quantize
    return eager_quantize(x)


def quantize_and_rotate_rowwise(x: torch.Tensor, H: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused online activation rotation + row-wise quantization."""
    from comfy_kitchen.backends.eager.quantization import quantize_and_rotate_rowwise as eager_quantize_rotate
    return eager_quantize_rotate(x, H, group_size)


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
    if convrot:
        from comfy_kitchen.tensor.int8 import _build_hadamard, _rotate_activation
        H = _build_hadamard(convrot_groupsize, device=x.device, dtype=x.dtype)
        x = _rotate_activation(x, H, convrot_groupsize)

    # cuBLAS INT8 GEMM requires row-wise quantized activations and tensor-wise quantized weights
    x_qdata, x_scale = torch.ops.comfy_kitchen.quantize_int8_rowwise(x)

    m, k = x.shape
    n, k_w = weight.shape
    assert k == k_w, "Input and weight inner dimensions must match"

    # cuBLAS INT8 GEMM outputs int32
    out_int32 = torch.empty((m, n), dtype=torch.int32, device=x.device)
    stream_ptr = torch.cuda.current_stream(x.device).cuda_stream

    _C.cublas_gemm_int8(
        _wrap_for_dlpack(x_qdata),
        _wrap_for_dlpack(weight),
        _wrap_for_dlpack(out_int32),
        _wrap_for_dlpack(get_cublas_workspace()),
        stream_ptr,
    )

    # Dequantize/Rescale using eager-friendly operations
    weight_scale = weight_scale.view(-1)
    out = out_int32.float() * (x_scale * weight_scale)

    if bias is not None:
        out += bias.float()

    out_dtype = out_dtype or x.dtype
    return out.to(out_dtype)


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
    reuse_workspace: bool = False,
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
    reuse_workspace = reuse_workspace and os.getenv(
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
    from comfy_kitchen.constraints import (
        DivisibleBy,
        ExactDims,
        FunctionConstraints,
        ParamConstraint,
    )

    cuda_devices = frozenset({"cuda"})

    constraints = {
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
        "int8_linear": FunctionConstraints(
            params={
                "x": ParamConstraint(
                    dtypes=frozenset({torch.float32, torch.float16, torch.bfloat16}),
                    shape_rules=(ExactDims(2),),
                ),
                "weight": ParamConstraint(
                    dtypes=frozenset({torch.int8}),
                    shape_rules=(ExactDims(2),),
                ),
                "weight_scale": ParamConstraint(
                    dtypes=frozenset({torch.float32}),
                ),
                "out_dtype": ParamConstraint(
                    dtypes=frozenset({torch.float32, torch.float16, torch.bfloat16}),
                ),
                "convrot": ParamConstraint(dtypes=frozenset({bool})),
                "convrot_groupsize": ParamConstraint(dtypes=frozenset({int})),
            },
            default_devices=cuda_devices,
            min_compute_capability=(7, 5),
        ),
        "quantize_int8_tensorwise": FunctionConstraints(
            params={
                "x": ParamConstraint(
                    dtypes=frozenset({torch.float32, torch.float16, torch.bfloat16}),
                ),
            },
            default_devices=cuda_devices,
        ),
        "quantize_int8_rowwise": FunctionConstraints(
            params={
                "x": ParamConstraint(
                    dtypes=frozenset({torch.float32, torch.float16, torch.bfloat16}),
                ),
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
