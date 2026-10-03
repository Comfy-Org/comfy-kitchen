# SPDX-License-Identifier: Apache-2.0
"""Metal (MPS) backend for Apple GPUs.

Apple GPUs before M5 have no fast integer matrix path, so this backend is
weight-only: integer weights stay resident in memory, are decoded to fp16 by a
single-pass Metal kernel, and feed the stock fp16 GEMM. Activations are not
quantized. ConvRot checkpoints store the weight already rotated, so the
activation is rotated online (cheap, group-wise) instead of un-rotating the
weight on every call.

MPS can hold fp8 tensors but has no fp8 arithmetic. W4A8 dequantize decodes the fp8
group-scale bytes in Metal; W4A8 quantize (LoRA requantize) runs eager on CPU.
"""
__all__ = [
    "dequantize_w4a8_int8_weight",
    "int8_linear",
    "quantize_w4a8_int8_weight",
    "w4a8_int8_linear",
]

import torch

from comfy_kitchen.constraints import ExactDims, FunctionConstraints, ParamConstraint
from comfy_kitchen.registry import registry

from .linear import int8_linear
from .w4a8 import dequantize_w4a8_int8_weight, quantize_w4a8_int8_weight, w4a8_int8_linear


def _build_constraints() -> dict:
    mps = frozenset({"mps"})
    floats = frozenset({torch.float32, torch.float16, torch.bfloat16})
    group_scales = frozenset({torch.float8_e4m3fn, torch.float32})
    packed_w4a8 = {
        "qdata": ParamConstraint(dtypes=frozenset({torch.int8}), shape_rules=(ExactDims(2),)),
        "s_rel": ParamConstraint(dtypes=group_scales, shape_rules=(ExactDims(2),)),
        "s_channel": ParamConstraint(dtypes=frozenset({torch.float32}), shape_rules=(ExactDims(1),)),
        "codebook": ParamConstraint(dtypes=frozenset({torch.float32}), shape_rules=(ExactDims(1),)),
        "correction": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(2),)),
        "group_size": ParamConstraint(dtypes=frozenset({int})),
        "convrot_groupsize": ParamConstraint(dtypes=frozenset({int})),
    }
    return {
        "int8_linear": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=floats),
                "weight": ParamConstraint(dtypes=frozenset({torch.int8})),
                "weight_scale": ParamConstraint(dtypes=floats),
                "bias": ParamConstraint(dtypes=floats),
                "convrot": ParamConstraint(dtypes=frozenset({bool})),
                "convrot_groupsize": ParamConstraint(dtypes=frozenset({int})),
            },
            default_devices=mps,
        ),
        "w4a8_int8_linear": FunctionConstraints(
            params={
                **packed_w4a8,
                "x": ParamConstraint(dtypes=floats),
                "bias": ParamConstraint(dtypes=floats),
                "out_dtype": ParamConstraint(dtypes=floats),
            },
            default_devices=mps,
        ),
        "dequantize_w4a8_int8_weight": FunctionConstraints(
            params={**packed_w4a8, "output_dtype": ParamConstraint(dtypes=floats)},
            default_devices=mps,
        ),
        "quantize_w4a8_int8_weight": FunctionConstraints(
            params={
                "weight": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(2),)),
                "group_size": ParamConstraint(dtypes=frozenset({int})),
                "convrot_groupsize": ParamConstraint(dtypes=frozenset({int})),
                "symmetric": ParamConstraint(dtypes=frozenset({bool})),
                "scale_dtype": ParamConstraint(dtypes=group_scales),
                "codebook": ParamConstraint(dtypes=frozenset({bool})),
            },
            default_devices=mps,
        ),
    }


def _register():
    if not (torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")):
        registry.mark_unavailable("mps", "MPS or torch.mps.compile_shader not available")
        return
    registry.register(
        name="mps",
        module=__import__(__name__, fromlist=__all__),
        capabilities=_build_constraints(),
    )


_register()
