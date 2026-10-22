# SPDX-License-Identifier: Apache-2.0
"""Metal (MPS) backend for Apple GPUs.

On MPS the eager RoPE and AdaLN implementations run as chains of small ops
(a norm, casts, broadcasts, multiply-adds), each a separate kernel launch and a
full pass over memory. This backend runs each call as one Metal kernel compiled
with ``torch.mps.compile_shader``: every RoPE variant (interleaved and
split-half, with or without the RMSNorm, partial ``rot_dim``, in place) and
``adaln`` / ``rms_adaln``, in float32, float16 and bfloat16. Calls the kernels
do not take (other devices, dtypes or layouts) are declined by the call rules
and run on eager.
"""
__all__ = [
    "adaln",
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
]

import torch

from comfy_kitchen.constraints import FunctionConstraints, ParamConstraint
from comfy_kitchen.registry import registry

from .norm import adaln, rms_adaln, validate_adaln
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
    validate_apply_rope,
    validate_apply_rope1,
    validate_rms_rope,
    validate_rms_rope1,
    validate_rms_rope_split_half,
)


def _build_constraints() -> dict:
    mps = frozenset({"mps"})
    floats = frozenset({torch.float32, torch.float16, torch.bfloat16})
    freqs = {"freqs_cis": ParamConstraint(dtypes=frozenset({torch.float32}))}

    def rope(names, rule, scales=()):
        params = {name: ParamConstraint(dtypes=floats) for name in names}
        params.update({name: ParamConstraint(dtypes=floats) for name in scales[:1]})
        params.update({name: ParamConstraint(dtypes=floats | {type(None)}) for name in scales[1:]})
        return FunctionConstraints(params={**params, **freqs}, default_devices=mps, call_rules=(rule,))

    norm = FunctionConstraints(
        params={name: ParamConstraint(dtypes=floats) for name in ("x", "scale", "shift")},
        default_devices=mps,
        call_rules=(validate_adaln,),
    )
    out = {
        "adaln": norm,
        "rms_adaln": norm,
        "apply_rope1": rope(("x",), validate_apply_rope1),
        "apply_rope_split_half1": rope(("x",), validate_apply_rope1),
        "apply_rope": rope(("xq", "xk"), validate_apply_rope),
        "apply_rope_split_half": rope(("xq", "xk"), validate_apply_rope),
        "rms_rope1": rope(("x",), validate_rms_rope1, ("scale",)),
        "rms_rope_split_half1": rope(("x",), validate_rms_rope1, ("scale",)),
        "rms_rope": rope(("q", "k"), validate_rms_rope, ("q_scale", "k_scale")),
        "rms_rope_split_half": rope(("q", "k"), validate_rms_rope_split_half, ("q_scale", "k_scale")),
    }
    for name in [n for n in out if "rope" in n]:
        out[name + "_"] = out[name]
    return out


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
