__all__ = [
    "adaln",
    "apply_rope",
    "apply_rope1",
    "apply_rope_split_half",
    "apply_rope_split_half1",
    "rms_rope",
    "rms_rope1",
    "rms_rope_split_half",
    "rms_rope_split_half1",
    "dequantize_nvfp4",
    "dequantize_per_tensor_fp8",
    "quantize_mxfp8",
    "quantize_nvfp4",
    "quantize_per_tensor_fp8",
    "int8_linear",
    "quantize_int8_rowwise",
    "quantize_and_rotate_rowwise",
]

# Try to import triton and register if available
import torch

from comfy_kitchen.constraints import (
    ExactDims,
    FunctionConstraints,
    ParamConstraint,
)
from comfy_kitchen.registry import registry

_TRITON_AVAILABLE = True
_TRITON_ERROR = None

try:
    import triton  # noqa: F401
    from comfy_kitchen.backends.eager.quantization import (
        quantize_and_rotate_rowwise as _eager_quantize_and_rotate_rowwise,
    )
    from comfy_kitchen.backends.eager.quantization import (
        quantize_int8_rowwise as _eager_quantize_int8_rowwise,
    )

    from .adaln import adaln
    from .quantization import (
        dequantize_nvfp4,
        dequantize_per_tensor_fp8,
        int8_linear,
        quantize_mxfp8,
        quantize_nvfp4,
        quantize_per_tensor_fp8,
    )
    from .quantization import (
        triton_quantize_and_rotate_rowwise as _triton_quantize_and_rotate_rowwise,
    )
    from .quantization import (
        triton_quantize_rowwise as _triton_quantize_int8_rowwise,
    )
    from .rope import apply_rope, apply_rope1, apply_rope_split_half, apply_rope_split_half1
    from .rms_rope import rms_rope, rms_rope1, rms_rope_split_half, rms_rope_split_half1
except ImportError as e:
    _TRITON_AVAILABLE = False
    _TRITON_ERROR = f"ImportError: {e!s}"


if _TRITON_AVAILABLE:
    def quantize_int8_rowwise(
        x: torch.Tensor,
        stochastic_rounding: int | None = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if stochastic_rounding is not None and stochastic_rounding > 0:
            return _eager_quantize_int8_rowwise(x, stochastic_rounding=stochastic_rounding)
        return _triton_quantize_int8_rowwise(x)

    def quantize_and_rotate_rowwise(
        x: torch.Tensor,
        h: torch.Tensor,
        group_size: int,
        stochastic_rounding: int | None = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if stochastic_rounding is not None and stochastic_rounding > 0:
            return _eager_quantize_and_rotate_rowwise(
                x, h, group_size, stochastic_rounding=stochastic_rounding
            )
        return _triton_quantize_and_rotate_rowwise(x, h, group_size)


def _build_constraints() -> dict:
    cuda_devices = frozenset({"cuda"})
    triton_devices = frozenset({"cuda", "xpu"})
    standard_floats = frozenset({torch.float32, torch.float16, torch.bfloat16})

    # Skip on ROCm — the AMD backend has no `r` register and would crash
    # with "error: couldn't allocate input reg for constraint 'r'".
    sm_specific = {} if getattr(torch.version, "hip", None) else {
        # Uses inline PTX: cvt.rn.f16x2.e2m1x2 (SM100/Blackwell instruction)
        "dequantize_nvfp4": FunctionConstraints(
            params={
                "qx": ParamConstraint(
                    dtypes=frozenset({torch.uint8}),
                    shape_rules=(ExactDims(2),),
                ),
                "per_tensor_scale": ParamConstraint(dtypes=frozenset({torch.float32})),
                "block_scales": ParamConstraint(
                    dtypes=frozenset({torch.float8_e4m3fn}),
                ),
                "output_type": ParamConstraint(dtypes=standard_floats),
            },
            default_devices=cuda_devices,
            min_compute_capability=(10, 0),  # SM100 required for cvt.rn.f16x2.e2m1x2
        ),
    }

    return {
        "adaln": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=standard_floats),
                "scale": ParamConstraint(dtypes=standard_floats),
                "shift": ParamConstraint(dtypes=standard_floats),
            },
            default_devices=triton_devices,
        ),
        "quantize_per_tensor_fp8": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=standard_floats),
                "scale": ParamConstraint(dtypes=frozenset({torch.float32})),
                "output_type": ParamConstraint(
                    dtypes=frozenset({torch.float8_e4m3fn, torch.float8_e5m2}),
                ),
            },
            default_devices=triton_devices,
        ),
        "dequantize_per_tensor_fp8": FunctionConstraints(
            params={
                "x": ParamConstraint(
                    dtypes=frozenset({torch.float8_e4m3fn, torch.float8_e5m2}),
                ),
                "scale": ParamConstraint(dtypes=frozenset({torch.float32})),
                "output_type": ParamConstraint(dtypes=standard_floats),
            },
            default_devices=triton_devices,
        ),
        "quantize_nvfp4": FunctionConstraints(
            params={
                "x": ParamConstraint(
                    dtypes=standard_floats,
                    shape_rules=(ExactDims(2),),
                ),
                "per_tensor_scale": ParamConstraint(dtypes=frozenset({torch.float32})),
            },
            default_devices=cuda_devices,
        ),
        "quantize_mxfp8": FunctionConstraints(
            params={
                "x": ParamConstraint(
                    dtypes=standard_floats,
                    shape_rules=(ExactDims(2),),
                ),
            },
            default_devices=cuda_devices,
        ),
        **sm_specific,
        "apply_rope1": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=standard_floats),
                "freqs_cis": ParamConstraint(dtypes=standard_floats),
            },
            default_devices=triton_devices,
        ),
        "apply_rope": FunctionConstraints(
            params={
                "xq": ParamConstraint(dtypes=standard_floats),
                "xk": ParamConstraint(dtypes=standard_floats),
                "freqs_cis": ParamConstraint(dtypes=standard_floats),
            },
            default_devices=triton_devices,
        ),
        "rms_rope1": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=standard_floats),
                "freqs_cis": ParamConstraint(dtypes=standard_floats),
                "scale": ParamConstraint(dtypes=standard_floats),
            },
            default_devices=triton_devices,
        ),
        "rms_rope": FunctionConstraints(
            params={
                "q": ParamConstraint(dtypes=standard_floats),
                "k": ParamConstraint(dtypes=standard_floats),
                "freqs_cis": ParamConstraint(dtypes=standard_floats),
                "q_scale": ParamConstraint(dtypes=standard_floats),
                "k_scale": ParamConstraint(dtypes=standard_floats),
            },
            default_devices=triton_devices,
        ),
        "int8_linear": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=standard_floats),
                "weight": ParamConstraint(dtypes=frozenset({torch.int8})),
                "weight_scale": ParamConstraint(dtypes=standard_floats),
                "out_dtype": ParamConstraint(dtypes=standard_floats),
                "convrot": ParamConstraint(dtypes=frozenset({bool})),
                "convrot_groupsize": ParamConstraint(dtypes=frozenset({int})),
            },
            default_devices=triton_devices,
            min_compute_capability=(8, 0),  # Required for Triton INT8 dot
        ),
        "quantize_int8_rowwise": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=standard_floats),
                "stochastic_rounding": ParamConstraint(dtypes=frozenset({int})),
            },
            default_devices=triton_devices,
        ),
        "quantize_and_rotate_rowwise": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=standard_floats),
                "H": ParamConstraint(dtypes=standard_floats),
                "group_size": ParamConstraint(dtypes=frozenset({int})),
                "stochastic_rounding": ParamConstraint(dtypes=frozenset({int})),
            },
            default_devices=triton_devices,
        ),
        "apply_rope_split_half1": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=standard_floats),
                "freqs_cis": ParamConstraint(dtypes=standard_floats),
            },
            default_devices=triton_devices,
        ),
        "apply_rope_split_half": FunctionConstraints(
            params={
                "xq": ParamConstraint(dtypes=standard_floats),
                "xk": ParamConstraint(dtypes=standard_floats),
                "freqs_cis": ParamConstraint(dtypes=standard_floats),
            },
            default_devices=triton_devices,
        ),
        "rms_rope_split_half1": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=standard_floats),
                "freqs_cis": ParamConstraint(dtypes=standard_floats),
                "scale": ParamConstraint(dtypes=standard_floats),
            },
            default_devices=triton_devices,
        ),
        "rms_rope_split_half": FunctionConstraints(
            params={
                "q": ParamConstraint(dtypes=standard_floats),
                "k": ParamConstraint(dtypes=standard_floats),
                "freqs_cis": ParamConstraint(dtypes=standard_floats),
                "q_scale": ParamConstraint(dtypes=standard_floats),
                "k_scale": ParamConstraint(dtypes=standard_floats),
            },
            default_devices=triton_devices,
        ),
    }


def _register():
    if not _TRITON_AVAILABLE:
        registry.mark_unavailable("triton", _TRITON_ERROR or "Triton not available")
        return

    has_cuda = torch.cuda.is_available()
    has_xpu = hasattr(torch, "xpu") and torch.xpu.is_available()

    if not has_cuda and not has_xpu:
        registry.mark_unavailable("triton", "Neither CUDA nor XPU available on this system")
        return

    registry.register(
        name="triton",
        module=__import__(__name__, fromlist=__all__),
        capabilities=_build_constraints(),
    )


_register()
