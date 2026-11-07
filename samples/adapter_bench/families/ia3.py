"""IA3 channel scale: y = x * (alpha + gamma w[c]). ComfyUI has no IA3 adapter;
the baseline arm "torch" is the plain broadcast."""

from .. import sweeps as sw
from ..arms import comfy as comfy_code
from ..core.case import Case
from .common import comfy, compile_arm, eager, kitchen, rnd

FAMILY = "ia3"
SWEEPS = ("tokens", "dim")


def _ref(x, w, alpha, gamma):
    return x * (alpha + gamma * w)


def _case(sweep, x_val, label, t, n, dtype, arms):
    inputs = {
        "x": rnd(t, n, dtype=dtype),
        "w": rnd(n, dtype=dtype, scale=0.1),
        "alpha": 1.0,
        "gamma": 1.0,
    }
    adapt = lambda x, w, alpha, gamma: (x, w, -1, alpha, gamma)  # noqa: E731
    built = {
        "torch": lambda: comfy(
            comfy_code.channel_scale, note="plain torch (ComfyUI has no IA3)"
        ),
        "compile": lambda: compile_arm(comfy_code.channel_scale, inputs),
        "eager": lambda: eager("channel_scale", adapt),
        "kitchen": lambda: kitchen("channel_scale", adapt),
    }
    e = inputs["x"].element_size()
    return Case(
        family=FAMILY,
        op="channel_scale",
        sweep=sweep,
        x=x_val,
        case=label,
        params={"tokens": t, "dim": n},
        inputs=inputs,
        grad_names=("x", "w"),
        ref=_ref,
        arms={
            k: f()
            for k, f in built.items()
            if k in arms or (k == "torch" and "comfy" in arms)
        },
        logical_bytes=2 * t * n * e,
        logical_flops=2 * t * n,
        ulp_mode="elementwise",
    )


def cases(sweep, dtype, quick, arms):
    n, t = sw.FIXED_DIM, sw.FIXED_TOKENS
    if sweep == "tokens":
        for tok in sw.thin(sw.TOKENS, quick):
            yield lambda tok=tok: _case(sweep, tok, f"{tok} tok", tok, n, dtype, arms)
    elif sweep == "dim":
        for d in sw.thin(sw.DIMS, quick):
            yield lambda d=d: _case(sweep, d, f"{d} ch", t, d, dtype, arms)
