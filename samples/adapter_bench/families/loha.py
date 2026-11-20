"""LoHa bypass (base + x ((A1B1)⊙(A2B2) s)ᵀ) and merge (W + s (A1B1)⊙(A2B2))."""

from .. import sweeps as sw
from ..arms import comfy as comfy_code
from ..core.case import Case
from .common import comfy, compile_arm, eager, kitchen, rnd

FAMILY = "loha"
SWEEPS = ("tokens", "batch", "dim", "rank", "merge_dim", "merge_rank")
SCALE = 0.5


def _delta(w1a, w1b, w2a, w2b):
    return (w1a @ w1b) * (w2a @ w2b)


def _bypass_ref(x, w1a, w1b, w2a, w2b, base, scale):
    return base + x @ (scale * _delta(w1a, w1b, w2a, w2b)).t()


def _merge_ref(weight, w1a, w1b, w2a, w2b, scale):
    return weight + scale * _delta(w1a, w1b, w2a, w2b)


def _factors(n, r, dtype):
    return {
        "w1a": rnd(n, r, dtype=dtype, scale=0.3),
        "w1b": rnd(r, n, dtype=dtype, scale=0.3),
        "w2a": rnd(n, r, dtype=dtype, scale=0.3),
        "w2b": rnd(r, n, dtype=dtype, scale=0.3),
    }


def _bypass_case(sweep, x_val, label, lead, n, r, dtype, arms):
    t = 1
    for v in lead:
        t *= v
    inputs = {
        "x": rnd(*lead, n, dtype=dtype),
        **_factors(n, r, dtype),
        "base": rnd(*lead, n, dtype=dtype),
        "scale": SCALE,
    }

    def adapt(x, w1a, w1b, w2a, w2b, base, scale):
        return x, w1a, w1b, w2a, w2b, scale, base

    built = {
        "comfy": lambda: comfy(comfy_code.loha_bypass),
        "compile": lambda: compile_arm(comfy_code.loha_bypass, inputs),
        "eager": lambda: eager("loha_bypass", adapt),
        "kitchen": lambda: kitchen("loha_bypass", adapt),
    }
    e = inputs["x"].element_size()
    return Case(
        family=FAMILY,
        op="loha_bypass",
        sweep=sweep,
        x=x_val,
        case=label,
        params={"tokens": t, "dim": n, "rank": r, "lead": list(lead)},
        inputs=inputs,
        grad_names=("x", "w1a", "w1b", "w2a", "w2b"),
        ref=_bypass_ref,
        arms={k: f() for k, f in built.items() if k in arms},
        logical_bytes=(3 * t * n + 4 * n * r) * e,
        logical_flops=2 * t * n * n + 4 * n * n * r,
    )


def _merge_case(sweep, x_val, label, n, r, dtype, arms):
    inputs = {"weight": rnd(n, n, dtype=dtype), **_factors(n, r, dtype), "scale": SCALE}

    def adapt(weight, w1a, w1b, w2a, w2b, scale):
        return weight, w1a, w1b, w2a, w2b, scale

    built = {
        "comfy": lambda: comfy(comfy_code.loha_merge),
        "compile": lambda: compile_arm(comfy_code.loha_merge, inputs),
        "eager": lambda: eager("loha_merge", adapt),
        "kitchen": lambda: kitchen("loha_merge", adapt),
    }
    e = inputs["weight"].element_size()
    return Case(
        family=FAMILY,
        op="loha_merge",
        sweep=sweep,
        x=x_val,
        case=label,
        params={"dim": n, "rank": r},
        inputs=inputs,
        grad_names=("w1a", "w1b", "w2a", "w2b"),
        ref=_merge_ref,
        arms={k: f() for k, f in built.items() if k in arms},
        logical_bytes=(2 * n * n + 4 * n * r) * e,
        logical_flops=4 * n * n * r,
    )


def cases(sweep, dtype, quick, arms):
    n, r, t = sw.FIXED_DIM, sw.FIXED_RANK, sw.FIXED_TOKENS
    if sweep == "tokens":
        for tok in sw.thin(sw.TOKENS, quick):
            yield lambda tok=tok: _bypass_case(
                sweep, tok, f"{tok} tok", (tok,), n, r, dtype, arms
            )
    elif sweep == "batch":
        for b in sw.thin(sw.BATCH, quick):
            yield lambda b=b: _bypass_case(
                sweep, b, f"batch {b} x {t}", (b, t), n, r, dtype, arms
            )
    elif sweep == "dim":
        for d in sw.thin(sw.DIMS, quick):
            yield lambda d=d: _bypass_case(
                sweep, d, f"{d}x{d}", (t,), d, r, dtype, arms
            )
    elif sweep == "rank":
        for rr in sw.thin(sw.RANKS, quick):
            yield lambda rr=rr: _bypass_case(
                sweep, rr, f"r{rr}", (t,), n, rr, dtype, arms
            )
    elif sweep == "merge_dim":
        for d in sw.thin(sw.DIMS, quick):
            yield lambda d=d: _merge_case(sweep, d, f"{d}x{d}", d, r, dtype, arms)
    elif sweep == "merge_rank":
        for rr in sw.thin(sw.RANKS, quick):
            yield lambda rr=rr: _merge_case(sweep, rr, f"r{rr}", n, rr, dtype, arms)
