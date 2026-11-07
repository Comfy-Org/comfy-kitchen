"""LoRA bypass (y = base + s*(x Dᵀ)Uᵀ) and merge (W + s*U D)."""

from .. import sweeps as sw
from ..arms import comfy as comfy_code
from ..core.case import Case
from .common import comfy, compile_arm, eager, kitchen, rnd

FAMILY = "lora"
SWEEPS = ("tokens", "batch", "dim", "rank", "merge_dim", "merge_rank")
SCALE = 0.5


def _bypass_ref(x, up, down, base, scale):
    return base + scale * (x @ down.t()) @ up.t()


def _merge_ref(weight, up, down, scale):
    return weight + scale * (up @ down)


def _bypass_case(sweep, x_val, label, lead, n, r, dtype, arms):
    t = 1
    for v in lead:
        t *= v
    inputs = {
        "x": rnd(*lead, n, dtype=dtype),
        "up": rnd(n, r, dtype=dtype, scale=0.1),
        "down": rnd(r, n, dtype=dtype, scale=0.1),
        "base": rnd(*lead, n, dtype=dtype),
        "scale": SCALE,
    }
    adapt = lambda x, up, down, base, scale: (x, up, down, scale, base)  # noqa: E731
    built = {
        "comfy": lambda: comfy(comfy_code.lora_bypass),
        "compile": lambda: compile_arm(comfy_code.lora_bypass, inputs),
        "eager": lambda: eager("lora_bypass", adapt),
        "kitchen": lambda: kitchen("lora_bypass", adapt),
    }
    e = inputs["x"].element_size()
    return Case(
        family=FAMILY,
        op="lora_bypass",
        sweep=sweep,
        x=x_val,
        case=label,
        params={"tokens": t, "dim": n, "rank": r, "lead": list(lead)},
        inputs=inputs,
        grad_names=("x", "up", "down"),
        ref=_bypass_ref,
        arms={k: f() for k, f in built.items() if k in arms},
        logical_bytes=(3 * t * n + 2 * n * r) * e,
        logical_flops=2 * t * r * 2 * n,
    )


def _merge_case(sweep, x_val, label, n, r, dtype, arms):
    inputs = {
        "weight": rnd(n, n, dtype=dtype),
        "up": rnd(n, r, dtype=dtype, scale=0.1),
        "down": rnd(r, n, dtype=dtype, scale=0.1),
        "scale": SCALE,
    }
    adapt = lambda weight, up, down, scale: (weight, up, down, scale)  # noqa: E731
    built = {
        "comfy": lambda: comfy(comfy_code.lora_merge),
        "compile": lambda: compile_arm(comfy_code.lora_merge, inputs),
        "eager": lambda: eager("lora_merge", adapt),
        "kitchen": lambda: kitchen("lora_merge", adapt),
    }
    e = inputs["weight"].element_size()
    return Case(
        family=FAMILY,
        op="lora_merge",
        sweep=sweep,
        x=x_val,
        case=label,
        params={"dim": n, "rank": r},
        inputs=inputs,
        grad_names=("up", "down"),
        ref=_merge_ref,
        arms={k: f() for k, f in built.items() if k in arms},
        logical_bytes=(2 * n * n + 2 * n * r) * e,
        logical_flops=2 * n * n * r,
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
