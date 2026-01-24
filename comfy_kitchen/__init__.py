import torch

from .backends import cuda as _cuda_backend  # noqa: F401

# Import backends to trigger auto-registration
from .backends import eager as _eager_backend  # noqa: F401
from .backends import triton as _triton_backend  # noqa: F401
from .backends.eager.quantization import DTYPE_TO_CODE
from .exceptions import (
    BackendError,
    BackendNotFoundError,
    BackendNotImplementedError,
    NoCapableBackendError,
)

# Import registry and exceptions
from .registry import registry

__version__ = "0.1.0"

__all__ = [
    # Exceptions
    "BackendError",
    "BackendNotFoundError",
    "BackendNotImplementedError",
    "NoCapableBackendError",
    # Core functions
    "apply_rope",
    "apply_rope1",
    "dequantize_mxfp8",
    "dequantize_nvfp4",
    "dequantize_per_tensor_fp8",
    # Backend configuration
    "disable_backend",
    "enable_backend",
    "list_backends",
    "quantize_mxfp8",
    "quantize_nvfp4",
    "quantize_per_tensor_fp8",
    "scaled_mm_mxfp8",
    "scaled_mm_nvfp4",
    "set_backend_priority",
    "use_backend",
]


# =============================================================================
# Public API Functions
# =============================================================================


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


def quantize_nvfp4(
    x: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    epsilon: float = 0.0,
    pad_16x: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to NVFP4 format with block-wise scaling.

    Args:
        x: Input tensor (2D)
        per_tensor_scale: Global scale factor
        epsilon: Epsilon for numerical stability
        pad_16x: If True, implicit zero-padding is applied to make dimensions divisible by 16

    Returns:
        Tuple of (quantized_tensor, block_scales)
    """
    return torch.ops.comfy_kitchen.quantize_nvfp4(x, per_tensor_scale, epsilon, pad_16x)


def dequantize_nvfp4(
    qx: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    block_scales: torch.Tensor,
    output_type: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize tensor from NVFP4 format with block-wise scaling.

    Args:
        qx: Quantized FP4 tensor (packed as uint8)
        per_tensor_scale: Global scale factor
        block_scales: Block scales in swizzled layout (float8_e4m3fn)
        output_type: Target output dtype (float32, float16, or bfloat16)

    Returns:
        Dequantized tensor in specified output format
    """
    dtype_code = DTYPE_TO_CODE[output_type]
    return torch.ops.comfy_kitchen.dequantize_nvfp4(qx, per_tensor_scale, block_scales, dtype_code)


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


def apply_rope(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Rotary Position Embedding (RoPE) to query and key tensors.

    Args:
        xq: Query tensor
        xk: Key tensor
        freqs_cis: Precomputed frequency tensor

    Returns:
        Tuple of (transformed_query, transformed_key)
    """
    return torch.ops.comfy_kitchen.apply_rope(xq, xk, freqs_cis)


def apply_rope1(
    x: torch.Tensor, freqs_cis: torch.Tensor
) -> torch.Tensor:
    """Apply Rotary Position Embedding (RoPE) to a single tensor.

    Args:
        x: Input tensor
        freqs_cis: Precomputed frequency tensor

    Returns:
        Transformed tensor
    """
    return torch.ops.comfy_kitchen.apply_rope1(x, freqs_cis)


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
