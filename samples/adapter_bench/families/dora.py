"""DoRA epilogue on a merged weight: W + s(((W + ΔW) d / ||W_row||) - W)."""

import torch

import comfy_kitchen as ck

from .. import sweeps as sw
from ..arms import comfy as comfy_code
from ..core.case import Arm, Case
from .common import comfy, compile_arm, rnd

FAMILY = "dora"
SWEEPS = ("dim",)
STRENGTH = 0.8


def _ref_for(dtype):
    """The definition in fp64, with the eps of the weight's own dtype as ComfyUI adds it."""
    eps = torch.finfo(dtype).eps

    def ref(weight, lora_diff, dora_scale, strength):
        calc = weight + lora_diff
        norm = weight.reshape(weight.shape[0], -1).norm(dim=1, keepdim=True) + eps
        return weight + strength * (calc * (dora_scale / norm) - weight)

    return ref


def _kitchen(backend):
    def call(weight, lora_diff, dora_scale, strength):
        return ck.adapters.weight_decompose(
            weight, dora_scale.view(-1), strength, True, weight, weight, lora_diff
        )

    def run(weight, lora_diff, dora_scale, strength):
        if backend is None:
            return call(weight, lora_diff, dora_scale, strength)
        with ck.use_backend(backend):
            return call(weight, lora_diff, dora_scale, strength)

    return run


def _case(sweep, x_val, label, n, dtype, arms):
    weight = rnd(n, n, dtype=dtype)
    inputs = {
        "weight": weight,
        "lora_diff": rnd(n, n, dtype=dtype, scale=0.02),
        "dora_scale": weight.float().norm(dim=1, keepdim=True)
        * (1 + 0.05 * rnd(n, 1, dtype=torch.float32)),
        "strength": STRENGTH,
    }
    built = {
        "comfy": lambda: comfy(
            comfy_code.weight_decompose,
            note="ComfyUI base.weight_decompose(), output axis",
        ),
        "compile": lambda: compile_arm(comfy_code.weight_decompose, inputs),
        "eager": lambda: Arm(
            _kitchen("eager"), note="comfy_kitchen eager backend, delta=ΔW"
        ),
        "kitchen": lambda: Arm(_kitchen(None), note="comfy_kitchen.adapters, delta=ΔW"),
    }
    e = weight.element_size()
    return Case(
        family=FAMILY,
        op="weight_decompose",
        sweep=sweep,
        x=x_val,
        case=label,
        params={"dim": n, "strength": STRENGTH},
        inputs=inputs,
        grad_names=("lora_diff", "dora_scale"),
        ref=_ref_for(dtype),
        arms={k: f() for k, f in built.items() if k in arms},
        logical_bytes=3 * n * n * e,
        logical_flops=4 * n * n,
    )


def cases(sweep, dtype, quick, arms):
    if sweep == "dim":
        for d in sw.thin(sw.DIMS, quick):
            yield lambda d=d: _case(sweep, d, f"{d}x{d}", d, dtype, arms)
