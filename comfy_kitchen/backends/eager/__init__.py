__all__ = [
    "adaln",
    "apply_rope",
    "apply_rope1",
    "apply_rope_split_half",
    "apply_rope_split_half1",
    "dequantize_mxfp8",
    "dequantize_nvfp4",
    "dequantize_per_tensor_fp8",
    "dequantize_int8_simple",
    "dequantize_int8_simple_dtype",
    "dequantize_int8_convrot_weight",
    "dequantize_int8_convrot_weight_dtype",
    "gemv_awq_w4a16",
    "quantize_mxfp8",
    "quantize_nvfp4",
    "quantize_per_tensor_fp8",
    "quantize_int8_rowwise",
    "quantize_int8_convrot_weight",
    "quantize_and_rotate_rowwise",
    "quantize_int8_tensorwise",
    "quantize_svdquant_w4a4",
    "scaled_mm_mxfp8",
    "scaled_mm_nvfp4",
    "scaled_mm_svdquant_w4a4",
    "stochastic_rounding_fp8",
    "int8_linear",
]

import torch

from comfy_kitchen.constraints import (
    ExactDims,
    FunctionConstraints,
    ParamConstraint,
)
from comfy_kitchen.registry import registry

from .adaln import adaln
from .awq import gemv_awq_w4a16
from .quantization import (
    dequantize_int8_convrot_weight,
    dequantize_int8_convrot_weight_dtype,
    dequantize_int8_simple,
    dequantize_int8_simple_dtype,
    dequantize_mxfp8,
    dequantize_nvfp4,
    dequantize_per_tensor_fp8,
    int8_linear,
    quantize_and_rotate_rowwise,
    quantize_int8_convrot_weight,
    quantize_int8_rowwise,
    quantize_int8_tensorwise,
    quantize_mxfp8,
    quantize_nvfp4,
    quantize_per_tensor_fp8,
    scaled_mm_mxfp8,
    scaled_mm_nvfp4,
    stochastic_rounding_fp8,
)
from .rope import apply_rope, apply_rope1, apply_rope_split_half, apply_rope_split_half1
from .svdquant import quantize_svdquant_w4a4, scaled_mm_svdquant_w4a4


def _build_constraints() -> dict:
    all_devices = frozenset({"cpu", "cuda", "mps", "xpu", "hpu", "meta", "*"})
    standard_floats = frozenset({torch.float32, torch.float16, torch.bfloat16})

    out = {
        "adaln": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=standard_floats),
                "scale": ParamConstraint(dtypes=standard_floats),
                "shift": ParamConstraint(dtypes=standard_floats),
            },
            default_devices=all_devices,
        ),
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
        "stochastic_rounding_fp8": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=standard_floats),
                "rng": ParamConstraint(dtypes=frozenset({torch.uint8})),
                "output_type": ParamConstraint(
                    dtypes=frozenset({torch.float8_e4m3fn, torch.float8_e5m2}),
                ),
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
        "apply_rope_split_half1": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=standard_floats),
                "freqs_cis": ParamConstraint(dtypes=standard_floats),
            },
            default_devices=all_devices,
        ),
        "apply_rope_split_half": FunctionConstraints(
            params={
                "xq": ParamConstraint(dtypes=standard_floats),
                "xk": ParamConstraint(dtypes=standard_floats),
                "freqs_cis": ParamConstraint(dtypes=standard_floats),
            },
            default_devices=all_devices,
        ),
        "quantize_svdquant_w4a4": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=standard_floats, shape_rules=(ExactDims(2),)),
                "smooth": ParamConstraint(dtypes=standard_floats, shape_rules=(ExactDims(1),)),
                "lora_down": ParamConstraint(
                    dtypes=standard_floats, shape_rules=(ExactDims(2),),
                ),
            },
            default_devices=all_devices,
        ),
        "scaled_mm_svdquant_w4a4": FunctionConstraints(
            params={
                "act": ParamConstraint(
                    dtypes=frozenset({torch.int8, torch.uint8}),
                    shape_rules=(ExactDims(2),),
                ),
                "wgt": ParamConstraint(
                    dtypes=frozenset({torch.int8}),
                ),
                "ascales": ParamConstraint(dtypes=standard_floats, shape_rules=(ExactDims(2),)),
                "wscales": ParamConstraint(dtypes=standard_floats),
                "lora_act_in": ParamConstraint(
                    dtypes=standard_floats,
                    shape_rules=(ExactDims(2),),
                ),
                "lora_up": ParamConstraint(dtypes=standard_floats),
            },
            default_devices=all_devices,
        ),
        "gemv_awq_w4a16": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=standard_floats),
                "qweight": ParamConstraint(
                    dtypes=frozenset({torch.int8}),
                    shape_rules=(ExactDims(2),),
                ),
                "wscales": ParamConstraint(dtypes=standard_floats, shape_rules=(ExactDims(2),)),
                "wzeros": ParamConstraint(dtypes=standard_floats, shape_rules=(ExactDims(2),)),
            },
            default_devices=all_devices,
        ),
    }

    out["quantize_int8_tensorwise"] = FunctionConstraints(
        params={
            "x": ParamConstraint(dtypes=standard_floats),
        },
        default_devices=all_devices,
    )
    out["quantize_int8_rowwise"] = FunctionConstraints(
        params={
            "x": ParamConstraint(dtypes=standard_floats),
        },
        default_devices=all_devices,
    )
    out["quantize_and_rotate_rowwise"] = FunctionConstraints(
        params={
            "x": ParamConstraint(dtypes=standard_floats),
            "H": ParamConstraint(dtypes=standard_floats),
            "group_size": ParamConstraint(dtypes=frozenset({int})),
        },
        default_devices=all_devices,
    )
    out["quantize_int8_convrot_weight"] = FunctionConstraints(
        params={
            "weight": ParamConstraint(dtypes=standard_floats, shape_rules=(ExactDims(2),)),
            "group_size": ParamConstraint(dtypes=frozenset({int})),
        },
        default_devices=all_devices,
    )
    out["dequantize_int8_convrot_weight"] = FunctionConstraints(
        params={
            "q": ParamConstraint(dtypes=frozenset({torch.int8}), shape_rules=(ExactDims(2),)),
            "scale": ParamConstraint(dtypes=standard_floats),
            "group_size": ParamConstraint(dtypes=frozenset({int})),
        },
        default_devices=all_devices,
    )
    out["dequantize_int8_convrot_weight_dtype"] = FunctionConstraints(
        params={
            "q": ParamConstraint(dtypes=frozenset({torch.int8}), shape_rules=(ExactDims(2),)),
            "scale": ParamConstraint(dtypes=standard_floats),
            "group_size": ParamConstraint(dtypes=frozenset({int})),
            "output_dtype_code": ParamConstraint(dtypes=frozenset({int})),
        },
        default_devices=all_devices,
    )
    out["dequantize_int8_simple"] = FunctionConstraints(
        params={
            "q": ParamConstraint(dtypes=frozenset({torch.int8})),
            "scale": ParamConstraint(dtypes=standard_floats),
        },
        default_devices=all_devices,
    )
    out["dequantize_int8_simple_dtype"] = FunctionConstraints(
        params={
            "q": ParamConstraint(dtypes=frozenset({torch.int8})),
            "scale": ParamConstraint(dtypes=standard_floats),
            "output_dtype_code": ParamConstraint(dtypes=frozenset({int})),
        },
        default_devices=all_devices,
    )
    out["int8_linear"] = FunctionConstraints(
        params={
            "x": ParamConstraint(dtypes=standard_floats),
            "weight": ParamConstraint(dtypes=frozenset({torch.int8})),
            "weight_scale": ParamConstraint(dtypes=standard_floats),
            "bias": ParamConstraint(dtypes=standard_floats),
            "convrot": ParamConstraint(dtypes=frozenset({bool})),
            "convrot_groupsize": ParamConstraint(dtypes=frozenset({int})),
        },
        default_devices=all_devices,
    )

    if hasattr(torch, "float8_e8m0fnu"):
        out["quantize_mxfp8"] = FunctionConstraints(
                params={
                    "x": ParamConstraint(
                        dtypes=standard_floats,
                        shape_rules=(ExactDims(2),),
                    ),
                },
                default_devices=all_devices)

        out["dequantize_mxfp8"] = FunctionConstraints(
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
                default_devices=all_devices)

        out["scaled_mm_mxfp8"] = FunctionConstraints(
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
                default_devices=all_devices)

    return out


def _register():
    registry.register(
        name="eager",
        module=__import__(__name__, fromlist=__all__),
        capabilities=_build_constraints(),
    )


_register()
