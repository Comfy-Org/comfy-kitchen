__all__ = [
    "adaln",
    "fp16_conv3d",
    "group_norm_silu_pad3d",
    "na3d",
    "sol_attn",
    "draft_attention",
    "rms_adaln",
    "apply_rope",
    "apply_rope_",
    "apply_rope1",
    "apply_rope1_",
    "apply_rope_split_half",
    "apply_rope_split_half_",
    "apply_rope_split_half1",
    "apply_rope_split_half1_",
    "rms_rope",
    "rms_rope_",
    "rms_rope1",
    "rms_rope1_",
    "rms_rope_split_half",
    "rms_rope_split_half_",
    "rms_rope_split_half1",
    "rms_rope_split_half1_",
    "dequantize_mxfp8",
    "dequantize_nvfp4",
    "dequantize_per_tensor_fp8",
    "dequantize_int8_simple",
    "dequantize_int8_simple_dtype",
    "dequantize_int8_convrot_weight",
    "dequantize_int8_convrot_weight_dtype",
    "dequantize_int8_embedding",
    "dequantize_convrot_w4a4_weight",
    "dequantize_w4a8_int8_weight",
    "gemv_awq_w4a16",
    "convrot_w4a4_linear",
    "prepare_int4_weight_for_int8_linear",
    "quantize_mxfp8",
    "quantize_nvfp4",
    "quantize_per_tensor_fp8",
    "quantize_convrot_w4a4_weight",
    "quantize_w4a8_int8_weight",
    "quantize_int8_rowwise",
    "quantize_int8_convrot_weight",
    "rotate_int8_convrot_weight",
    "quantize_and_rotate_rowwise",
    "quantize_int8_tensorwise",
    "quantize_svdquant_w4a4",
    "scaled_mm_mxfp8",
    "scaled_mm_nvfp4",
    "scaled_mm_svdquant_w4a4",
    "stochastic_rounding_fp8",
    "fp16_linear",
    "int8_linear",
    "w4a8_int8_linear",
]

import torch

from comfy_kitchen.constraints import (
    ExactDims,
    FunctionConstraints,
    ParamConstraint,
    draft_attention_common_call_rule,
    na3d_common_call_rule,
    sol_attn_common_call_rule,
)
from comfy_kitchen.registry import registry

from .adaln import adaln, rms_adaln
from .draft import draft_attention
from .awq import gemv_awq_w4a16
from .conv3d import fp16_conv3d
from .convrot_w4a4 import (
    convrot_w4a4_linear,
    dequantize_convrot_w4a4_weight,
    prepare_int4_weight_for_int8_linear,
    quantize_convrot_w4a4_weight,
)
from .group_norm_pad3d import group_norm_silu_pad3d
from .na import na3d
from .quantization import (
    dequantize_int8_convrot_weight,
    dequantize_int8_convrot_weight_dtype,
    dequantize_int8_embedding,
    dequantize_int8_simple,
    dequantize_int8_simple_dtype,
    dequantize_mxfp8,
    dequantize_nvfp4,
    dequantize_per_tensor_fp8,
    fp16_linear,
    int8_linear,
    quantize_and_rotate_rowwise,
    quantize_int8_convrot_weight,
    quantize_int8_rowwise,
    quantize_int8_tensorwise,
    quantize_mxfp8,
    quantize_nvfp4,
    quantize_per_tensor_fp8,
    rotate_int8_convrot_weight,
    scaled_mm_mxfp8,
    scaled_mm_nvfp4,
    stochastic_rounding_fp8,
)
from .rope import (
    apply_rope,
    apply_rope1,
    apply_rope1_,
    apply_rope_,
    apply_rope_split_half,
    apply_rope_split_half1,
    apply_rope_split_half1_,
    apply_rope_split_half_,
    rms_rope,
    rms_rope1,
    rms_rope1_,
    rms_rope_,
    rms_rope_split_half,
    rms_rope_split_half1,
    rms_rope_split_half1_,
    rms_rope_split_half_,
)
from .sol_attn import sol_attn
from .svdquant import quantize_svdquant_w4a4, scaled_mm_svdquant_w4a4
from .w4a8_int8 import (
    dequantize_w4a8_int8_weight,
    quantize_w4a8_int8_weight,
    w4a8_int8_linear,
)


def _build_constraints() -> dict:
    all_devices = frozenset({"cpu", "cuda", "mps", "xpu", "hpu", "meta", "*"})
    standard_floats = frozenset({torch.float32, torch.float16, torch.bfloat16})
    scale_values = frozenset({torch.float32, torch.float16, torch.bfloat16, float, str})

    out = {
        "adaln": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=standard_floats),
                "scale": ParamConstraint(dtypes=standard_floats),
                "shift": ParamConstraint(dtypes=standard_floats),
            },
            default_devices=all_devices,
        ),
        "rms_adaln": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=standard_floats),
                "scale": ParamConstraint(dtypes=standard_floats),
                "shift": ParamConstraint(dtypes=standard_floats),
            },
            default_devices=all_devices,
        ),
        "fp16_conv3d": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=standard_floats, shape_rules=(ExactDims(5),)),
                "weight": ParamConstraint(dtypes=standard_floats, shape_rules=(ExactDims(5),)),
                "bias": ParamConstraint(dtypes=standard_floats | {type(None)}),
                "residual": ParamConstraint(dtypes=standard_floats | {type(None)}),
            },
            default_devices=all_devices,
        ),
        "group_norm_silu_pad3d": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=standard_floats, shape_rules=(ExactDims(5),)),
                "weight": ParamConstraint(dtypes=standard_floats | {type(None)}),
                "bias": ParamConstraint(dtypes=standard_floats | {type(None)}),
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
        "rms_rope": FunctionConstraints(
            params={
                "q": ParamConstraint(
                    dtypes=standard_floats, shape_rules=(ExactDims(4),),
                ),
                "k": ParamConstraint(
                    dtypes=standard_floats, shape_rules=(ExactDims(4),),
                ),
                "freqs_cis": ParamConstraint(
                    dtypes=standard_floats, shape_rules=(ExactDims(6),),
                ),
                "q_scale": ParamConstraint(
                    dtypes=standard_floats, shape_rules=(ExactDims(1),),
                ),
                "k_scale": ParamConstraint(
                    dtypes=standard_floats, shape_rules=(ExactDims(1),),
                ),
            },
            default_devices=all_devices,
        ),
        "rms_rope1": FunctionConstraints(
            params={
                "x": ParamConstraint(
                    dtypes=standard_floats, shape_rules=(ExactDims(4),),
                ),
                "freqs_cis": ParamConstraint(
                    dtypes=standard_floats, shape_rules=(ExactDims(6),),
                ),
                "scale": ParamConstraint(
                    dtypes=standard_floats, shape_rules=(ExactDims(1),),
                ),
            },
            default_devices=all_devices,
        ),
        "rms_rope_split_half": FunctionConstraints(
            params={
                "q": ParamConstraint(
                    dtypes=standard_floats, shape_rules=(ExactDims(4),),
                ),
                "k": ParamConstraint(
                    dtypes=standard_floats, shape_rules=(ExactDims(4),),
                ),
                "freqs_cis": ParamConstraint(
                    dtypes=standard_floats, shape_rules=(ExactDims(6),),
                ),
                "q_scale": ParamConstraint(
                    dtypes=standard_floats, shape_rules=(ExactDims(1),),
                ),
                "k_scale": ParamConstraint(
                    dtypes=standard_floats, shape_rules=(ExactDims(1),),
                ),
            },
            default_devices=all_devices,
        ),
        "rms_rope_split_half1": FunctionConstraints(
            params={
                "x": ParamConstraint(
                    dtypes=standard_floats, shape_rules=(ExactDims(4),),
                ),
                "freqs_cis": ParamConstraint(
                    dtypes=standard_floats, shape_rules=(ExactDims(6),),
                ),
                "scale": ParamConstraint(
                    dtypes=standard_floats, shape_rules=(ExactDims(1),),
                ),
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
        "quantize_convrot_w4a4_weight": FunctionConstraints(
            params={
                "weight": ParamConstraint(dtypes=standard_floats, shape_rules=(ExactDims(2),)),
                "convrot_groupsize": ParamConstraint(dtypes=frozenset({int})),
                "quant_group_size": ParamConstraint(dtypes=frozenset({int})),
                "stochastic_rounding": ParamConstraint(dtypes=frozenset({int})),
            },
            default_devices=all_devices,
        ),
        "dequantize_convrot_w4a4_weight": FunctionConstraints(
            params={
                "qdata": ParamConstraint(dtypes=frozenset({torch.int8}), shape_rules=(ExactDims(2),)),
                "scales": ParamConstraint(dtypes=standard_floats, shape_rules=(ExactDims(1),)),
                "convrot_groupsize": ParamConstraint(dtypes=frozenset({int})),
                "quant_group_size": ParamConstraint(dtypes=frozenset({int})),
                "output_dtype": ParamConstraint(dtypes=standard_floats),
            },
            default_devices=all_devices,
        ),
        "convrot_w4a4_linear": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=standard_floats),
                "qweight": ParamConstraint(dtypes=frozenset({torch.int8}), shape_rules=(ExactDims(2),)),
                "wscales": ParamConstraint(dtypes=standard_floats, shape_rules=(ExactDims(1),)),
                "bias": ParamConstraint(dtypes=standard_floats),
                "convrot_groupsize": ParamConstraint(dtypes=frozenset({int})),
                "quant_group_size": ParamConstraint(dtypes=frozenset({int})),
                "linear_dtype": ParamConstraint(dtypes=frozenset({str})),
            },
            default_devices=all_devices,
        ),
        "quantize_w4a8_int8_weight": FunctionConstraints(
            params={
                "weight": ParamConstraint(dtypes=standard_floats, shape_rules=(ExactDims(2),)),
                "group_size": ParamConstraint(dtypes=frozenset({int})),
                "convrot_groupsize": ParamConstraint(dtypes=frozenset({int})),
                "symmetric": ParamConstraint(dtypes=frozenset({bool})),
                "scale_dtype": ParamConstraint(dtypes=frozenset({torch.float8_e4m3fn, torch.float32})),
                "codebook": ParamConstraint(dtypes=frozenset({bool})),
            },
            default_devices=all_devices,
        ),
        "dequantize_w4a8_int8_weight": FunctionConstraints(
            params={
                "qdata": ParamConstraint(dtypes=frozenset({torch.int8}), shape_rules=(ExactDims(2),)),
                "s_rel": ParamConstraint(dtypes=frozenset({torch.float8_e4m3fn, torch.float32}), shape_rules=(ExactDims(2),)),
                "s_channel": ParamConstraint(dtypes=frozenset({torch.float32}), shape_rules=(ExactDims(1),)),
                "codebook": ParamConstraint(dtypes=frozenset({torch.float32}), shape_rules=(ExactDims(1),)),
                "correction": ParamConstraint(dtypes=standard_floats, shape_rules=(ExactDims(2),)),
                "group_size": ParamConstraint(dtypes=frozenset({int})),
                "convrot_groupsize": ParamConstraint(dtypes=frozenset({int})),
                "output_dtype": ParamConstraint(dtypes=standard_floats),
            },
            default_devices=all_devices,
        ),
        "w4a8_int8_linear": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=standard_floats),
                "qdata": ParamConstraint(dtypes=frozenset({torch.int8}), shape_rules=(ExactDims(2),)),
                "s_rel": ParamConstraint(dtypes=frozenset({torch.float8_e4m3fn, torch.float32}), shape_rules=(ExactDims(2),)),
                "s_channel": ParamConstraint(dtypes=frozenset({torch.float32}), shape_rules=(ExactDims(1),)),
                "codebook": ParamConstraint(dtypes=frozenset({torch.float32}), shape_rules=(ExactDims(1),)),
                "correction": ParamConstraint(dtypes=standard_floats, shape_rules=(ExactDims(2),)),
                "bias": ParamConstraint(dtypes=standard_floats),
                "group_size": ParamConstraint(dtypes=frozenset({int})),
                "convrot_groupsize": ParamConstraint(dtypes=frozenset({int})),
                "out_dtype": ParamConstraint(dtypes=standard_floats),
            },
            default_devices=all_devices,
        ),
        "prepare_int4_weight_for_int8_linear": FunctionConstraints(
            params={
                "weight": ParamConstraint(dtypes=frozenset({torch.int8}), shape_rules=(ExactDims(2),)),
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
            "scale": ParamConstraint(dtypes=scale_values),
            "stochastic_rounding": ParamConstraint(dtypes=frozenset({int})),
        },
        default_devices=all_devices,
    )
    out["quantize_int8_rowwise"] = FunctionConstraints(
        params={
            "x": ParamConstraint(dtypes=standard_floats),
            "stochastic_rounding": ParamConstraint(dtypes=frozenset({int})),
        },
        default_devices=all_devices,
    )
    out["quantize_and_rotate_rowwise"] = FunctionConstraints(
        params={
            "x": ParamConstraint(dtypes=standard_floats),
            "H": ParamConstraint(dtypes=standard_floats),
            "group_size": ParamConstraint(dtypes=frozenset({int})),
            "stochastic_rounding": ParamConstraint(dtypes=frozenset({int})),
        },
        default_devices=all_devices,
    )
    out["quantize_int8_convrot_weight"] = FunctionConstraints(
        params={
            "weight": ParamConstraint(dtypes=standard_floats, shape_rules=(ExactDims(2),)),
            "group_size": ParamConstraint(dtypes=frozenset({int})),
            "stochastic_rounding": ParamConstraint(dtypes=frozenset({int})),
        },
        default_devices=all_devices,
    )
    out["rotate_int8_convrot_weight"] = FunctionConstraints(
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
    out["dequantize_int8_embedding"] = FunctionConstraints(
        params={
            "q": ParamConstraint(dtypes=frozenset({torch.int8}), shape_rules=(ExactDims(2),)),
            "scale": ParamConstraint(dtypes=standard_floats),
            "indices": ParamConstraint(dtypes=frozenset({torch.int64, torch.int32})),
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
    out["fp16_linear"] = FunctionConstraints(
        params={
            "x": ParamConstraint(dtypes=standard_floats),
            "weight": ParamConstraint(dtypes=standard_floats),
            "bias": ParamConstraint(dtypes=standard_floats),
            "residual": ParamConstraint(dtypes=standard_floats),
            "residual_scale": ParamConstraint(dtypes=standard_floats),
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

    for inplace_name, functional_name in {
        "apply_rope_": "apply_rope",
        "apply_rope1_": "apply_rope1",
        "apply_rope_split_half_": "apply_rope_split_half",
        "apply_rope_split_half1_": "apply_rope_split_half1",
        "rms_rope_": "rms_rope",
        "rms_rope1_": "rms_rope1",
        "rms_rope_split_half_": "rms_rope_split_half",
        "rms_rope_split_half1_": "rms_rope_split_half1",
    }.items():
        out[inplace_name] = out[functional_name]
    out["draft_attention"] = FunctionConstraints(
        params={
            "q": ParamConstraint(
                dtypes=frozenset({torch.float16, torch.bfloat16}),
                shape_rules=(ExactDims(4),),
            ),
            "k": ParamConstraint(
                dtypes=frozenset({torch.float16, torch.bfloat16}),
                shape_rules=(ExactDims(4),),
            ),
            "v": ParamConstraint(
                dtypes=frozenset({torch.float16, torch.bfloat16}),
                shape_rules=(ExactDims(4),),
            ),
        },
        # Full-precision reference on every device: CPU, accelerators without a
        # native implementation, and GPUs without compiled kernels. On supported
        # GPUs the registry still prefers the native kernels; the reference only
        # runs there under an explicit use_backend("eager") override.
        default_devices=all_devices,
        call_rules=(draft_attention_common_call_rule,),
    )
    out["sol_attn"] = FunctionConstraints(
        params={
            "q": ParamConstraint(dtypes=standard_floats, shape_rules=(ExactDims(4),)),
            "k": ParamConstraint(dtypes=standard_floats, shape_rules=(ExactDims(4),)),
            "v": ParamConstraint(dtypes=standard_floats, shape_rules=(ExactDims(4),)),
        },
        default_devices=all_devices,
        call_rules=(sol_attn_common_call_rule,),
    )
    out["na3d"] = FunctionConstraints(
        params={
            "q": ParamConstraint(dtypes=standard_floats, shape_rules=(ExactDims(6),)),
            "k": ParamConstraint(dtypes=standard_floats, shape_rules=(ExactDims(6),)),
            "v": ParamConstraint(dtypes=standard_floats, shape_rules=(ExactDims(6),)),
        },
        default_devices=all_devices,
        call_rules=(na3d_common_call_rule,),
    )
    return out


def _register():
    registry.register(
        name="eager",
        module=__import__(__name__, fromlist=__all__),
        capabilities=_build_constraints(),
    )


_register()
