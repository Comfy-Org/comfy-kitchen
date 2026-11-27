"""LoKr bypass (base + s x kron(w1, w2)ᵀ, the product never built) and merge."""

import torch

from .. import sweeps as sw
from ..arms import comfy as comfy_code
from ..arms import lycoris as lycoris_kernel
from ..core.case import Arm, Case
from ..core.device import capability
from .common import comfy, compile_arm, eager, kitchen, rnd

FAMILY = "lokr"
SWEEPS = ("tokens", "batch", "dim", "factor", "merge_dim", "merge_factor")
SCALE = 0.5


def _bypass_ref(x, w1, w2, base, scale):
    return base + scale * x @ torch.kron(w1, w2).t()


def _merge_ref(weight, w1, w2, scale):
    return weight + scale * torch.kron(w1, w2)


def _factors(n, factor, dtype):
    a, c = sw.factorization(n, factor)
    b, d = sw.factorization(n, factor)
    return rnd(a, b, dtype=dtype, scale=0.3), rnd(c, d, dtype=dtype, scale=0.3)


def _lycoris_arm(inputs):
    w1, w2 = inputs["w1"], inputs["w2"]
    # LyCORIS's kernel is a tl.dot kernel: it runs where Triton reaches tensor cores (sm_80+).
    cap = capability()
    if cap is None or cap < (8, 0):
        return None
    if not lycoris_kernel.supported(*w1.shape, *w2.shape):
        return None
    try:
        run, plan = lycoris_kernel.tuned_bypass(**inputs)
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:160]}"

        def failed(**_):
            raise RuntimeError(err)

        return Arm(failed, backward=False, note="LyCORIS triton kernel")
    return Arm(
        run,
        backward=False,
        note=f"LyCORIS triton kernel, plan (BT, warps, stages)={plan}",
    )


def _bypass_case(sweep, x_val, label, lead, n, factor, dtype, arms):
    t = 1
    for v in lead:
        t *= v
    w1, w2 = _factors(n, factor, dtype)
    inputs = {
        "x": rnd(*lead, n, dtype=dtype),
        "w1": w1,
        "w2": w2,
        "base": rnd(*lead, n, dtype=dtype),
        "scale": SCALE,
    }
    adapt = lambda x, w1, w2, base, scale: (x, w1, w2, scale, base)  # noqa: E731
    built = {
        "comfy": lambda: comfy(comfy_code.lokr_bypass),
        "compile": lambda: compile_arm(comfy_code.lokr_bypass, inputs),
        "eager": lambda: eager("lokr_bypass", adapt),
        "kitchen": lambda: kitchen("lokr_bypass", adapt),
        "lycoris": lambda: _lycoris_arm(inputs),
    }
    made = {k: f() for k, f in built.items() if k in arms}
    (a, b), (c, d) = w1.shape, w2.shape
    e = inputs["x"].element_size()
    return Case(
        family=FAMILY,
        op="lokr_bypass",
        sweep=sweep,
        x=x_val,
        case=label,
        params={
            "tokens": t,
            "dim": n,
            "factor": factor,
            "w1": [a, b],
            "w2": [c, d],
            "lead": list(lead),
        },
        inputs=inputs,
        grad_names=("x", "w1", "w2"),
        ref=_bypass_ref,
        arms={k: v for k, v in made.items() if v is not None},
        logical_bytes=(3 * t * n + a * b + c * d) * e,
        logical_flops=2 * t * (b * d * c + c * b * a),
    )


def _merge_case(sweep, x_val, label, n, factor, dtype, arms):
    w1, w2 = _factors(n, factor, dtype)
    inputs = {"weight": rnd(n, n, dtype=dtype), "w1": w1, "w2": w2, "scale": SCALE}
    adapt = lambda weight, w1, w2, scale: (weight, w1, w2, scale)  # noqa: E731
    built = {
        "comfy": lambda: comfy(comfy_code.lokr_merge),
        "compile": lambda: compile_arm(comfy_code.lokr_merge, inputs),
        "eager": lambda: eager("lokr_merge", adapt),
        "kitchen": lambda: kitchen("lokr_merge", adapt),
    }
    e = inputs["weight"].element_size()
    return Case(
        family=FAMILY,
        op="lokr_merge",
        sweep=sweep,
        x=x_val,
        case=label,
        params={"dim": n, "factor": factor, "w1": list(w1.shape), "w2": list(w2.shape)},
        inputs=inputs,
        grad_names=("w1", "w2"),
        ref=_merge_ref,
        arms={k: f() for k, f in built.items() if k in arms},
        logical_bytes=2 * n * n * e,
        logical_flops=n * n,
    )


def _fx(factor):
    """Plot position of a factor: balanced (-1) sits left of the explicit factors."""
    return 2 if factor < 0 else factor


def cases(sweep, dtype, quick, arms):
    n, t, f = sw.FIXED_DIM, sw.FIXED_TOKENS, sw.FIXED_FACTOR
    if sweep == "tokens":
        for tok in sw.thin(sw.TOKENS, quick):
            yield lambda tok=tok: _bypass_case(
                sweep, tok, f"{tok} tok", (tok,), n, f, dtype, arms
            )
    elif sweep == "batch":
        for b in sw.thin(sw.BATCH, quick):
            yield lambda b=b: _bypass_case(
                sweep, b, f"batch {b} x {t}", (b, t), n, f, dtype, arms
            )
    elif sweep == "dim":
        for d in sw.thin(sw.DIMS, quick):
            yield lambda d=d: _bypass_case(
                sweep, d, f"{d}x{d}", (t,), d, f, dtype, arms
            )
    elif sweep == "factor":
        for ff in sw.thin(sw.FACTORS, quick):
            label = "balanced" if ff < 0 else f"factor {ff}"
            yield lambda ff=ff, label=label: _bypass_case(
                sweep, _fx(ff), label, (t,), n, ff, dtype, arms
            )
    elif sweep == "merge_dim":
        for d in sw.thin(sw.DIMS, quick):
            yield lambda d=d: _merge_case(sweep, d, f"{d}x{d}", d, f, dtype, arms)
    elif sweep == "merge_factor":
        for ff in sw.thin(sw.FACTORS, quick):
            label = "balanced" if ff < 0 else f"factor {ff}"
            yield lambda ff=ff, label=label: _merge_case(
                sweep, _fx(ff), label, n, ff, dtype, arms
            )
