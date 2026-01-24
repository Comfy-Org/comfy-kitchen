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
