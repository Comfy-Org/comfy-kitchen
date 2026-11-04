# SPDX-License-Identifier: Apache-2.0
"""Metal (MPS) backend for Apple GPUs.

On MPS the eager AdaLN implementation runs as a chain of small ops (a norm,
casts, broadcasts, multiply-adds), each a separate kernel launch and a full
pass over memory. This backend runs ``adaln`` / ``rms_adaln`` as one Metal
kernel per tensor, compiled with ``torch.mps.compile_shader``, in float32,
float16 and bfloat16. Calls the kernel does not take (other devices, dtypes or
layouts) are declined by the call rule and run on eager.
"""
__all__ = [
    "adaln",
    "rms_adaln",
]

import torch

from comfy_kitchen.constraints import FunctionConstraints, ParamConstraint
from comfy_kitchen.registry import registry

from .norm import adaln, rms_adaln, validate_adaln


def _build_constraints() -> dict:
    mps = frozenset({"mps"})
    floats = frozenset({torch.float32, torch.float16, torch.bfloat16})

    norm = FunctionConstraints(
        params={name: ParamConstraint(dtypes=floats) for name in ("x", "scale", "shift")},
        default_devices=mps,
        call_rules=(validate_adaln,),
    )
    return {"adaln": norm, "rms_adaln": norm}


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
