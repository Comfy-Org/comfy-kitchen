from typing import Optional

import torch
from packaging import version

_TORCH_VERSION = version.parse(torch.__version__.split("+")[0])  # Remove git hash suffix
TORCH_2_10 = version.parse("2.10.0")
_HAS_SCALED_MM_V2 = hasattr(torch.nn.functional, "scaled_mm")

if _HAS_SCALED_MM_V2:
    from torch.nn.functional import ScalingType, SwizzleType
else:
    # Dummy types for older PyTorch versions
    class ScalingType:
        TensorWise = "TensorWise"
        BlockWise1x16 = "BlockWise1x16"
        BlockWise1x32 = "BlockWise1x32"

    class SwizzleType:
        NO_SWIZZLE = "NO_SWIZZLE"
        SWIZZLE_32_4_4 = "SWIZZLE_32_4_4"


def has_scaled_mm_v2() -> bool:
    return _HAS_SCALED_MM_V2


_FP8_E4M3 = torch.float8_e4m3fn


def _hip_fp8_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype | None,
    scale_recipe_a,
    scale_recipe_b,
    swizzle_a,
    swizzle_b,
):
    """Route an fp8 matmul to the gfx12 WMMA kernel.

    Returns None when the HIP backend is unavailable or the call is outside the
    kernel's domain (non-e4m3 operands, non-tensor-wise scaling, swizzled
    operands, K not a multiple of 16, unsupported output dtype), in which case
    the caller falls back to torch.
    """
    from .registry import registry

    if not registry.is_available("hip"):
        return None
    # Availability is process-wide, but the kernel launches on one device and takes
    # raw pointers: every operand has to live on the input's own GPU.
    if input.device.type != "cuda":
        return None
    others = [weight, scale_a, scale_b] + ([] if bias is None else [bias])
    if any(not torch.is_tensor(t) or t.device != input.device for t in others):
        return None
    # The kernel reads both operands row-major.
    if any(s not in (None, SwizzleType.NO_SWIZZLE) for s in (swizzle_a, swizzle_b)):
        return None
    if input.dtype is not _FP8_E4M3 or weight.dtype is not _FP8_E4M3:
        return None
    if input.dim() != 2 or weight.dim() != 2:
        return None
    if scale_recipe_a != ScalingType.TensorWise or scale_recipe_b != ScalingType.TensorWise:
        return None
    if isinstance(scale_a, list) or isinstance(scale_b, list):
        return None
    if scale_a.numel() != 1 or scale_b.numel() != 1:
        return None
    # The WMMA K-step reads 16 bytes of a row at a time.
    if input.shape[1] % 16 != 0:
        return None
    if out_dtype is not None and out_dtype not in (
        torch.bfloat16, torch.float16, torch.float32
    ):
        return None
    # The epilogue indexes bias[col] with a single dtype code.
    if bias is not None and (
        bias.dim() != 1
        or bias.numel() != weight.shape[1]
        or bias.dtype not in (torch.bfloat16, torch.float16, torch.float32)
    ):
        return None

    from .backends import hip

    return hip.scaled_mm_fp8(
        input, weight, scale_a, scale_b, bias,
        out_dtype if out_dtype is not None else torch.bfloat16,
    )


def scaled_mm_v2(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    scale_recipe_a = ScalingType.TensorWise,
    scale_recipe_b = ScalingType.TensorWise,
    swizzle_a: Optional['SwizzleType'] = SwizzleType.NO_SWIZZLE,
    swizzle_b: Optional['SwizzleType'] = SwizzleType.NO_SWIZZLE,
) -> torch.Tensor:

    out = _hip_fp8_gemm(
        input, weight, scale_a, scale_b, bias, out_dtype, scale_recipe_a, scale_recipe_b,
        swizzle_a, swizzle_b,
    )
    if out is not None:
        return out

    if has_scaled_mm_v2():
        return torch.nn.functional.scaled_mm(
            input,
            weight,
            scale_a=scale_a,
            scale_recipe_a=scale_recipe_a,
            scale_b=scale_b,
            scale_recipe_b=scale_recipe_b,
            swizzle_a=swizzle_a,
            swizzle_b=swizzle_b,
            bias=bias,
            output_dtype=out_dtype,
            use_fast_accum=False
        )
    else:
        add_bias_separate = False
        alpha = None

        if isinstance(scale_a, list):
            scale_a_for_mm, tensor_scale_a = scale_a
            scale_b_for_mm, tensor_scale_b = scale_b
            alpha = tensor_scale_a * tensor_scale_b
            add_bias_separate = bias is not None
        else:
            scale_a_for_mm = scale_a
            scale_b_for_mm = scale_b

        output = torch._scaled_mm(
            input,
            weight,
            scale_a=scale_a_for_mm,
            scale_b=scale_b_for_mm,
            out_dtype=out_dtype,
            bias = None if add_bias_separate else bias
        )

        # Handle tuple return
        if isinstance(output, tuple):
            output = output[0]
        if alpha is not None:
            output = output * alpha.to(output.dtype)
        if add_bias_separate:
            output = output + bias

        return output

# Version info for debugging
def get_pytorch_version_info() -> dict[str, str | bool]:
    """Get PyTorch version information for debugging.

    Returns:
        Dictionary with version info and feature flags
    """
    return {
        "torch_version": torch.__version__,
        "parsed_version": str(_TORCH_VERSION),
        "has_scaled_mm_v2": has_scaled_mm_v2(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    }
