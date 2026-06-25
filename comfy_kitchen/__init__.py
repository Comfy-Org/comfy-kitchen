import torch

from .backends import cuda as _cuda_backend  # noqa: F401

# Import backends to trigger auto-registration
from .backends import eager as _eager_backend  # noqa: F401
from .backends import triton as _triton_backend  # noqa: F401
from .backends.eager.quantization import DTYPE_TO_CODE
from .backends.eager.quantization import mm_int8 as _mm_int8
from .exceptions import (
    BackendError,
    BackendNotFoundError,
    BackendNotImplementedError,
    NoCapableBackendError,
)
from .float_utils import from_blocked, swap_nibbles, to_blocked

# Import registry and exceptions
from .registry import registry

__version__ = "0.1.0"

__all__ = [
    # Normalization
    "adaln",
    # Quantization / dequantization
    "quantize_per_tensor_fp8",
    "dequantize_per_tensor_fp8",
    "quantize_nvfp4",
    "dequantize_nvfp4",
    "quantize_mxfp8",
    "dequantize_mxfp8",
    "quantize_svdquant_w4a4",
    "quantize_int8_rowwise",
    "quantize_int8_tensorwise",
    "dequantize_int8_simple",
    # Fused matmul
    "scaled_mm_nvfp4",
    "scaled_mm_mxfp8",
    "scaled_mm_svdquant_w4a4",
    "gemv_awq_w4a16",
    "int8_linear",
    # Positional encoding
    "apply_rope",
    "apply_rope1",
    "apply_rope_split_half",
    "apply_rope_split_half1",
    # Utilities
    "swap_nibbles",
    "to_blocked",
    "from_blocked",
    # Backend configuration
    "list_backends",
    "set_backend_priority",
    "enable_backend",
    "disable_backend",
    "stochastic_rounding_fp8",
    "use_backend",
    # Exceptions
    "BackendError",
    "BackendNotFoundError",
    "BackendNotImplementedError",
    "NoCapableBackendError",
]


# =============================================================================
# Public API Functions
# =============================================================================


def adaln(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fused Adaptive Layer Normalization: layernorm(x) * (1 + scale) + shift.

    Args:
        x: Input tensor of any shape (..., D)
        scale: Modulation scale, broadcastable to x's shape
        shift: Modulation shift, broadcastable to x's shape
        eps: LayerNorm epsilon

    Returns:
        Normalized and modulated tensor with the same shape as x
    """
    return torch.ops.comfy_kitchen.adaln(x, scale, shift, eps)


def quantize_per_tensor_fp8(
    x: torch.Tensor,
    scale: torch.Tensor,
    output_type: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    """Quantize tensor to FP8 format with per-tensor scaling.

    Args:
        x: Input tensor
        scale: Scale tensor (scalar)
        output_type: FP8 dtype (float8_e4m3fn or float8_e5m2)

    Returns:
        Quantized FP8 tensor
    """
    dtype_code = DTYPE_TO_CODE[output_type]
    return torch.ops.comfy_kitchen.quantize_fp8(x, scale, dtype_code)


def dequantize_per_tensor_fp8(
    x: torch.Tensor,
    scale: torch.Tensor,
    output_type: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize tensor from FP8 format with per-tensor scaling.

    Args:
        x: Input FP8 tensor (float8_e4m3fn or float8_e5m2)
        scale: Scale tensor (scalar)
        output_type: Target dtype (float32, float16, or bfloat16)

    Returns:
        Dequantized tensor in specified output format
    """
    dtype_code = DTYPE_TO_CODE[output_type]
    return torch.ops.comfy_kitchen.dequantize_fp8(x, scale, dtype_code)


def stochastic_rounding_fp8(
    x: torch.Tensor,
    rng: torch.Tensor,
    output_type: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    """Stochastically round tensor to FP8 format.

    Args:
        x: Input tensor
        rng: Random uint8 tensor with the same shape as x
        output_type: FP8 dtype (float8_e4m3fn or float8_e5m2)

    Returns:
        Stochastically rounded FP8 tensor
    """
    kwargs = {"x": x, "rng": rng, "output_type": output_type}
    impl = registry.get_implementation("stochastic_rounding_fp8", kwargs=kwargs)
    return impl(**kwargs)


def quantize_nvfp4(
    x: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    epsilon: float = 0.0,
    pad_16x: bool = False,
    hi_first: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to NVFP4 format with block-wise scaling.

    Args:
        x: Input tensor (2D)
        per_tensor_scale: Global scale factor
        epsilon: Epsilon for numerical stability
        pad_16x: If True, implicit zero-padding is applied to make dimensions divisible by 16
        hi_first: Nibble packing order. If True (default), the even-indexed element
                  is stored in the high nibble of each packed byte. If False, the
                  even-indexed element is stored in the low nibble.

    Returns:
        Tuple of (quantized_tensor, block_scales)
    """
    return torch.ops.comfy_kitchen.quantize_nvfp4(x, per_tensor_scale, epsilon, pad_16x, hi_first)


def dequantize_nvfp4(
    qx: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    block_scales: torch.Tensor,
    output_type: torch.dtype = torch.bfloat16,
    hi_first: bool = True,
) -> torch.Tensor:
    """Dequantize tensor from NVFP4 format with block-wise scaling.

    Args:
        qx: Quantized FP4 tensor (packed as uint8)
        per_tensor_scale: Global scale factor
        block_scales: Block scales in swizzled layout (float8_e4m3fn)
        output_type: Target output dtype (float32, float16, or bfloat16)
        hi_first: Nibble packing order. Must match the packing order used
                  during quantization. If True (default), the even-indexed
                  element is in the high nibble.

    Returns:
        Dequantized tensor in specified output format
    """
    dtype_code = DTYPE_TO_CODE[output_type]
    return torch.ops.comfy_kitchen.dequantize_nvfp4(qx, per_tensor_scale, block_scales, dtype_code, hi_first)


def scaled_mm_nvfp4(
    a: torch.Tensor,
    b: torch.Tensor,
    tensor_scale_a: torch.Tensor,
    tensor_scale_b: torch.Tensor,
    block_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    alpha: torch.Tensor | None = None,
) -> torch.Tensor:
    """Matrix multiplication with NVFP4 quantized inputs.

    Computes: y = (a @ b.T) * (tensor_scale_a * tensor_scale_b) + bias

    Args:
        a: Quantized matrix A (M, K//2) in uint8 format
        b: Quantized matrix B (N, K//2) in uint8 format
        tensor_scale_a: Global scale for A
        tensor_scale_b: Global scale for B
        block_scale_a: Block-wise scales for A
        block_scale_b: Block-wise scales for B
        bias: Optional bias vector
        out_dtype: Output dtype (defaults to bfloat16)
        alpha: Output scale (tensor_scale_a * tensor_scale_b)

    Returns:
        Result tensor of shape (M, N)
    """
    if out_dtype is None:
        out_dtype = torch.bfloat16
    dtype_code = DTYPE_TO_CODE[out_dtype]
    return torch.ops.comfy_kitchen.scaled_mm_nvfp4(
        a, b, tensor_scale_a, tensor_scale_b,
        block_scale_a, block_scale_b, bias, dtype_code, alpha
    )


def quantize_mxfp8(
    x: torch.Tensor,
    pad_32x: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to MXFP8 format with block-wise E8M0 scaling.

    MXFP8 uses block size 32 with power-of-2 (E8M0) block scales.

    Args:
        x: Input tensor (2D, shape M x K, K must be divisible by 32)
        pad_32x: If True, pad dimensions to be divisible by 32

    Returns:
        Tuple of (quantized_fp8_tensor, block_scales_e8m0)
        - quantized_fp8_tensor: FP8 E4M3 data of shape (M, K)
        - block_scales_e8m0: E8M0 scales in swizzled layout
    """
    return torch.ops.comfy_kitchen.quantize_mxfp8(x, pad_32x)


def dequantize_mxfp8(
    qx: torch.Tensor,
    block_scales: torch.Tensor,
    output_type: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize tensor from MXFP8 format.

    Args:
        qx: Quantized FP8 tensor (float8_e4m3fn)
        block_scales: E8M0 block scales in swizzled layout (float8_e8m0fnu)
        output_type: Target output dtype (float32, float16, or bfloat16)

    Returns:
        Dequantized tensor in specified output format
    """
    dtype_code = DTYPE_TO_CODE[output_type]
    return torch.ops.comfy_kitchen.dequantize_mxfp8(qx, block_scales, dtype_code)


def scaled_mm_mxfp8(
    a: torch.Tensor,
    b: torch.Tensor,
    block_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Matrix multiplication with MXFP8 quantized inputs.

    Computes: y = a @ b.T + bias

    Args:
        a: Quantized FP8 matrix A (M, K)
        b: Quantized FP8 matrix B (N, K)
        block_scale_a: E8M0 block scales for A in swizzled layout
        block_scale_b: E8M0 block scales for B in swizzled layout
        bias: Optional bias vector
        out_dtype: Output dtype (defaults to bfloat16)

    Returns:
        Result tensor of shape (M, N)
    """
    if out_dtype is None:
        out_dtype = torch.bfloat16
    dtype_code = DTYPE_TO_CODE[out_dtype]
    return torch.ops.comfy_kitchen.scaled_mm_mxfp8(
        a, b, block_scale_a, block_scale_b, bias, dtype_code
    )


def quantize_svdquant_w4a4(
    x: torch.Tensor,
    smooth: torch.Tensor,
    lora_down: torch.Tensor,
    pad_size: int = 256,
    act_unsigned: bool = False,
    lora_x: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize activations to int4 with smoothing + LoRA down projection.

    Args:
        x: (M, K) bf16/fp16 main-path input (caller pre-shifts if unsigned path).
        smooth: (K,) smoothing factor applied before quantization.
        lora_down: (K, R) low-rank down projection weight.
        pad_size: pad M to multiple of this value (default 256).
        act_unsigned: if True, quantize into uint4 [0, 15] (scale=max/15) for u4
            MMA downstream. Caller must ensure x is non-negative — the shift
            constant is a model-topology concern, not part of this op.
        lora_x: (M, K) optional pre-shift activation for LoRA. Defaults to x.
            Pass raw (un-shifted) x when x has been pre-shifted for unsigned path.

    Returns:
        (quantized_x uint8 [M_pad, K//2], ascales [K//64, M_pad], lora_act [M_pad, R])

    Note: eager returns fp32 lora_act as a high-precision reference. The CUDA
    backend returns lora_act in x.dtype because the runtime epilogue consumes it
    as bf16/fp16; this avoids an otherwise redundant cast/allocation.
    """
    return torch.ops.comfy_kitchen.quantize_svdquant_w4a4(
        x, smooth, lora_down, pad_size, act_unsigned, lora_x,
    )


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
    """SVDQuant W4A4 int4 GEMM + LoRA-up + bias.

    Computes out = int4_matmul(act, wgt, ascales, wscales) + lora_act_in @ lora_up^T + bias.
    The CUDA backend performs int4 MMA + per-group dequant + bias in one
    kernel and, when lora_act_in/proj_up layout and dtype allow it, fuses
    LoRA-up into the same writeback epilogue with bf16/fp16 tensor-core MMA.
    Unsupported combinations fall back to the wrapper's bf16/fp16 addmm_ path.

    Args:
        act: (M, K//2) uint8 packed activations from quantize_svdquant_w4a4.
        wgt: (N, K//2) int8 packed weights (natural row-major), or backend
            specific tile-packed storage.
        ascales: (K//64, M) activation scales.
        wscales: (K//64, N) weight scales.
        lora_act_in: (M, R) LoRA activations from quantize step.
        lora_up: (N, R) LoRA up projection weight, or matching tile-packed
            storage for tile-packed weights.
        bias: optional (N,) bias.
        act_unsigned: if True, activations are interpreted as unsigned [0,15] by
            u4.s4 MMA (for post-GELU+shift fc2). Caller pre-shifts.

    Returns:
        (M, N) output tensor (same dtype as lora_up).
    """
    return torch.ops.comfy_kitchen.scaled_mm_svdquant_w4a4(
        act, wgt, ascales, wscales, lora_act_in, lora_up, bias, act_unsigned
    )


def gemv_awq_w4a16(
    x: torch.Tensor,
    qweight: torch.Tensor,
    wscales: torch.Tensor,
    wzeros: torch.Tensor,
    bias: torch.Tensor | None = None,
    group_size: int = 64,
) -> torch.Tensor:
    """AWQ W4A16 quantized GEMV (for modulation-style layers called with small batch).

    Args:
        x: (..., K) bf16/fp16 input.
        qweight: (N//4, K//2) int32 packed weight.
        wscales: (K//group_size, N) per-group scales.
        wzeros: (K//group_size, N) per-group zero points.
        bias: optional (N,) bias.
        group_size: quantization group size.

    Returns:
        (..., N) output tensor.
    """
    return torch.ops.comfy_kitchen.gemv_awq_w4a16(
        x, qweight, wscales, wzeros, bias, group_size
    )


def apply_rope(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Rotary Position Embedding (RoPE) to query and key tensors.

    Interleaved layout: pair k uses adjacent elements [2k, 2k+1].

    Args:
        xq: Query tensor
        xk: Key tensor
        freqs_cis: Precomputed frequency tensor

    Returns:
        Tuple of (transformed_query, transformed_key)
    """
    return torch.ops.comfy_kitchen.apply_rope(xq, xk, freqs_cis)


def apply_rope1(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """Apply Rotary Position Embedding (RoPE) to a single tensor.

    Interleaved layout: pair k uses adjacent elements [2k, 2k+1].

    Args:
        x: Input tensor
        freqs_cis: Precomputed frequency tensor

    Returns:
        Transformed tensor
    """
    return torch.ops.comfy_kitchen.apply_rope1(x, freqs_cis)


def apply_rope_split_half(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Rotary Position Embedding (RoPE) to query and key tensors.

    Split-half layout: pair k uses elements [k] and [k + head_dim//2].
    Matches the formula:
        t_ = t.reshape(*t.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2).to(freqs.dtype)
        t_out = freqs[..., 0] * t_[..., 0] + freqs[..., 1] * t_[..., 1]
        t_out.movedim(-1, -2).reshape(*t.shape).type_as(t)

    Args:
        xq: Query tensor
        xk: Key tensor
        freqs_cis: Precomputed frequency tensor shape (..., head_dim//2, 2, 2)

    Returns:
        Tuple of (transformed_query, transformed_key)
    """
    return torch.ops.comfy_kitchen.apply_rope_split_half(xq, xk, freqs_cis)


def apply_rope_split_half1(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """Apply Rotary Position Embedding (RoPE) to a single tensor.

    Split-half layout: pair k uses elements [k] and [k + head_dim//2].
    Matches the formula:
        t_ = t.reshape(*t.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2).to(freqs.dtype)
        t_out = freqs[..., 0] * t_[..., 0] + freqs[..., 1] * t_[..., 1]
        t_out.movedim(-1, -2).reshape(*t.shape).type_as(t)

    Args:
        x: Input tensor
        freqs_cis: Precomputed frequency tensor shape (..., head_dim//2, 2, 2)

    Returns:
        Transformed tensor
    """
    return torch.ops.comfy_kitchen.apply_rope_split_half1(x, freqs_cis)


def quantize_int8_tensorwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to INT8 with single tensorwise scale."""
    return torch.ops.comfy_kitchen.quantize_int8_tensorwise(x)


def quantize_int8_rowwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to INT8 with per-row scales."""
    return torch.ops.comfy_kitchen.quantize_int8_rowwise(x)


def dequantize_int8_simple(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize INT8 tensor with scale."""
    return torch.ops.comfy_kitchen.dequantize_int8_simple(q, scale)


def mm_int8(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """INT8 matrix multiplication: C[M,N] = A[M,K] @ B[K,N]."""
    return _mm_int8(a, b)


def int8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    convrot: bool = False,
    convrot_groupsize: int = 256,
) -> torch.Tensor:
    """INT8 linear layer dynamically quantized.

    Args:
        x: Input tensor.
        weight: INT8 weight tensor.
        weight_scale: Scalar weight scale.
        bias: Optional bias.
        out_dtype: Output dtype.
        convrot: If True, apply online activation rotation.
        convrot_groupsize: Group size for Hadamard rotation.

    Returns:
        Result tensor.
    """
    if out_dtype is None:
        out_dtype = torch.bfloat16
    kwargs = {
        "x": x,
        "weight": weight,
        "weight_scale": weight_scale,
        "bias": bias,
        "out_dtype": out_dtype,
        "convrot": convrot,
        "convrot_groupsize": convrot_groupsize,
    }
    impl = registry.get_implementation("int8_linear", kwargs=kwargs)
    return impl(**kwargs)


# =============================================================================
# Backend Configuration
# =============================================================================


def set_backend_priority(priority: list[str]) -> None:
    """Set the priority order for backend selection.

    Args:
        priority: List of backend names in order of preference
                 Example: ["cuda", "eager"] to prefer CUDA over Torch
    """
    registry.set_priority(priority)


def disable_backend(name: str) -> None:
    """Disable a backend, preventing its use.

    Args:
        name: Backend name to disable ("eager", "cuda", or "triton")
    """
    registry.disable(name)


def enable_backend(name: str) -> None:
    """Re-enable a previously disabled backend.

    Args:
        name: Backend name to enable ("eager", "cuda", or "triton")
    """
    registry.enable(name)


def list_backends() -> dict:
    """Get status information for all backends.

    Returns:
        Dictionary mapping backend names to their status:
        {
            "backend_name": {
                "available": bool,
                "disabled": bool,
                "unavailable_reason": str or None,
                "capabilities": list[str]
            }
        }
    """
    return registry.list_backends()


def use_backend(name: str):
    """Context manager to temporarily use a specific backend.

    Args:
        name: Backend name to use within the context

    Example:
        with comfy_kitchen.use_backend("eager"):
            result = comfy_kitchen.quantize_per_tensor_fp8(x, scale)
    """
    return registry.use_backend(name)
