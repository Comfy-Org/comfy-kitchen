__all__ = [
    "apply_rope",
    "apply_rope1",
    "dequantize_mxfp8",
    "dequantize_nvfp4",
    "dequantize_per_tensor_fp8",
    "quantize_mxfp8",
    "quantize_nvfp4",
    "quantize_per_tensor_fp8",
    "scaled_mm_mxfp8",
    "scaled_mm_nvfp4",
]

from .quantization import (
    dequantize_mxfp8,
    dequantize_nvfp4,
    dequantize_per_tensor_fp8,
    quantize_mxfp8,
    quantize_nvfp4,
    quantize_per_tensor_fp8,
    scaled_mm_mxfp8,
    scaled_mm_nvfp4,
)
from .rope import apply_rope, apply_rope1


def _build_constraints() -> dict:
    import torch

    from comfy_kitchen.constraints import (
        ExactDims,
        FunctionConstraints,
        ParamConstraint,
    )

    all_devices = frozenset({"cpu", "cuda", "mps", "xpu", "hpu", "meta", "*"})
    standard_floats = frozenset({torch.float32, torch.float16, torch.bfloat16})

    return {
        "quantize_per_tensor_fp8": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=standard_floats),
                "scale": ParamConstraint(dtypes=frozenset({torch.float32})),
                "output_type": ParamConstraint(
                    dtypes=frozenset({torch.float8_e4m3fn, torch.float8_e5m2}),
                ),
            },
            default_devices=all_devices,
        ),
        "dequantize_per_tensor_fp8": FunctionConstraints(
            params={
                "x": ParamConstraint(
                    dtypes=frozenset({torch.float8_e4m3fn, torch.float8_e5m2}),
                ),
                "scale": ParamConstraint(dtypes=standard_floats),
                "output_type": ParamConstraint(dtypes=standard_floats),
            },
            default_devices=all_devices,
        ),
        "quantize_nvfp4": FunctionConstraints(
            params={
                "x": ParamConstraint(
                    dtypes=standard_floats,
                    shape_rules=(ExactDims(2),),
                ),
                "per_tensor_scale": ParamConstraint(dtypes=frozenset({torch.float32})),
            },
            default_devices=all_devices,
        ),
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
            default_devices=all_devices,
        ),
        "scaled_mm_nvfp4": FunctionConstraints(
            params={
                "a": ParamConstraint(
                    dtypes=frozenset({torch.uint8}),
                    shape_rules=(ExactDims(2),),
                ),
                "b": ParamConstraint(
                    dtypes=frozenset({torch.uint8}),
                    shape_rules=(ExactDims(2),),
                ),
                "tensor_scale_a": ParamConstraint(dtypes=frozenset({torch.float32})),
                "tensor_scale_b": ParamConstraint(dtypes=frozenset({torch.float32})),
                "block_scale_a": ParamConstraint(
                    dtypes=frozenset({torch.float8_e4m3fn}),
                ),
                "block_scale_b": ParamConstraint(
                    dtypes=frozenset({torch.float8_e4m3fn}),
                ),
                "out_dtype": ParamConstraint(dtypes=standard_floats),
            },
            default_devices=all_devices,
        ),
        "quantize_mxfp8": FunctionConstraints(
            params={
                "x": ParamConstraint(
                    dtypes=standard_floats,
                    shape_rules=(ExactDims(2),),
                ),
            },
            default_devices=all_devices,
        ),
        "dequantize_mxfp8": FunctionConstraints(
            params={
                "qx": ParamConstraint(
                    dtypes=frozenset({torch.float8_e4m3fn}),
                    shape_rules=(ExactDims(2),),
                ),
                "block_scales": ParamConstraint(
                    dtypes=frozenset({torch.float8_e8m0fnu}),
                ),
                "output_type": ParamConstraint(dtypes=standard_floats),
            },
            default_devices=all_devices,
        ),
        "scaled_mm_mxfp8": FunctionConstraints(
            params={
                "a": ParamConstraint(
                    dtypes=frozenset({torch.float8_e4m3fn}),
                    shape_rules=(ExactDims(2),),
                ),
                "b": ParamConstraint(
                    dtypes=frozenset({torch.float8_e4m3fn}),
                    shape_rules=(ExactDims(2),),
                ),
                "block_scale_a": ParamConstraint(
                    dtypes=frozenset({torch.float8_e8m0fnu}),
                ),
                "block_scale_b": ParamConstraint(
                    dtypes=frozenset({torch.float8_e8m0fnu}),
                ),
                "out_dtype": ParamConstraint(dtypes=standard_floats),
            },
            default_devices=all_devices,
        ),
        "apply_rope1": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=standard_floats),
                "freqs_cis": ParamConstraint(dtypes=standard_floats),
            },
            default_devices=all_devices,
        ),
        "apply_rope": FunctionConstraints(
            params={
                "xq": ParamConstraint(dtypes=standard_floats),
                "xk": ParamConstraint(dtypes=standard_floats),
                "freqs_cis": ParamConstraint(dtypes=standard_floats),
            },
            default_devices=all_devices,
        ),
    }


def _register():
    from comfy_kitchen.registry import registry

    registry.register(
        name="eager",
        module=__import__(__name__, fromlist=__all__),
        capabilities=_build_constraints(),
    )


_register()
