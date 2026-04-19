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
    "quantize_mxfp8",
    "quantize_nvfp4",
    "quantize_per_tensor_fp8",
    "quantize_svdquant_w4a4",
    "scaled_mm_nvfp4",
    "scaled_mm_svdquant_w4a4",
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
    )

    return xq_out, xk_out


# ---------------------------------------------------------------------------
# SVDQuant W4A4 — native kitchen kernels (int4 MMA GEMM with per-group dequant).
# ---------------------------------------------------------------------------

_SVDQUANT_W4A4_GROUP_SIZE = 64


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
      lora_act  (M_pad, R)   fp32

    act_unsigned=True selects scale=max/15 + clamp [0,15] (u4 bit patterns for
    downstream u4.s4 MMA). Caller must ensure x is non-negative — shift is a
    model-topology concern kept out of the kernel API. Pass lora_x=raw_x when
    x was pre-shifted (LoRA operates on pre-quantization, pre-shift activations).
    """
    assert x.dim() == 2 and x.is_contiguous(), "x must be 2D contiguous"
    M, K = x.shape
    R = lora_down.shape[1]
    M_pad = roundup(M, pad_size)
    G = _SVDQUANT_W4A4_GROUP_SIZE
    assert K % G == 0, f"K={K} must be divisible by group_size={G}"

    q_x = torch.empty(M_pad, K // 2, dtype=torch.uint8, device=x.device)
    ascales = torch.empty(K // G, M_pad, dtype=x.dtype, device=x.device)
    lora_act = torch.empty(M_pad, R, dtype=torch.float32, device=x.device)

    stream_ptr = torch.cuda.current_stream(x.device).cuda_stream
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
    # matmul: torch's bf16 @ bf16 returns a bf16 tensor, which we then upcast to
    # the fp32 lora_act buffer. This is NOT the same as nunchaku's 16-bit-operand
    # + fp32-accumulate LoRA path (src/kernels/zgemm/lora.cuh:164/284) — the
    # bf16 output is effectively a rounded approximation to the fp32-accumulated
    # result (~8e-3 abs error vs true fp32). Accepts lora_x to let callers keep
    # LoRA on the raw pre-quantization activation when x has been pre-shifted.
    lora_src = lora_x if lora_x is not None else x
    if M > 0:
        lora_act[:M].copy_(lora_src @ lora_down, non_blocking=True)
    if M_pad > M:
        lora_act[M:].zero_()
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

    int4 MMA + per-group dequant in one kernel; LoRA-up + bias are applied in
    Python via cuBLAS bf16 addmm_ (see the body below — faster than in-kernel
    fusion for R=96 and similar small ranks).

    Kitchen-native layouts:
      act         (M, K/2)   int8       two int4 per byte (signed or unsigned)
      wgt         (N, K/2)   int8       signed int4 weight, natural row-major
      ascales     (K/G, M)   bf16/fp16  per-row per-group
      wscales     (K/G, N)   bf16/fp16  per-col per-group
      lora_act_in (M, R)     fp32
      lora_up     (N, R)     bf16/fp16
      bias        (N,)       bf16/fp16  (optional)
      out         (M, N)     bf16/fp16  (= lora_up.dtype)

    act_unsigned: if True, A fragments go through u4.s4 MMA instead of s4.s4.
    """
    M = act.shape[0]
    N = wgt.shape[0]
    out = torch.empty(M, N, dtype=lora_up.dtype, device=act.device)
    empty = torch.empty(0, dtype=lora_up.dtype, device=act.device)

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
        stream_ptr,
    )

    # LoRA-up + bias via bf16 addmm_. lora_act_in is fp32 but we drop it to bf16
    # here before the matmul so `out` stays bf16/fp16 throughout — this avoids
    # allocating a fp32 temporary, at the cost of the fp32 precision carried by
    # lora_act_in. The (fp32 out_of_quantize) -> (bf16 truncate) -> (bf16 matmul)
    # -> (bf16 addmm into out) chain is lossier than nunchaku's 16-bit-operand +
    # fp32-accumulate LoRA (src/kernels/zgemm/lora.cuh:284); kitchen accepts that
    # tradeoff for the memory/launch-count win. If that ever shows up in model
    # quality, replace this pair of matmuls with a dedicated cuBLASLt helper
    # (16-bit operand, fp32 accumulate) rather than Python-level .float() casts,
    # which would undo the memory/perf gain that motivated the bf16 path.
    lora_bf16 = lora_act_in.to(out.dtype)
    out.addmm_(lora_bf16, lora_up.t())
    if bias is not None:
        out.add_(bias)
    return out


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
                "wgt": ParamConstraint(dtypes=frozenset({torch.int8}), shape_rules=(ExactDims(2),)),
                "ascales": ParamConstraint(dtypes=frozenset({torch.float16, torch.bfloat16})),
                "wscales": ParamConstraint(dtypes=frozenset({torch.float16, torch.bfloat16})),
                "lora_act_in": ParamConstraint(dtypes=frozenset({torch.float32})),
                "lora_up": ParamConstraint(dtypes=frozenset({torch.float16, torch.bfloat16})),
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
