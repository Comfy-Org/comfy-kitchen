"""OFT: a block-diagonal Cayley rotation applied to activations (bypass g())
or to a weight (merge), out = rescale * Rᵀ y per block."""

import torch

from .. import sweeps as sw
from ..arms import comfy as comfy_code
from ..core.case import Case
from .common import comfy, compile_arm, eager, kitchen, rnd

FAMILY = "oft"
SWEEPS = (
    "tokens",
    "batch",
    "dim",
    "block",
    "conv_spatial",
    "conv_batch",
    "merge_dim",
    "merge_block",
)
MULT = 1.0
CONSTRAINT = 0.0


def _rotation64(blocks):
    q = blocks - blocks.transpose(1, 2)
    eye = torch.eye(blocks.shape[1], dtype=blocks.dtype, device=blocks.device)
    return (eye + q) @ torch.linalg.inv(eye - q)


def _apply_last(r, y):
    k, s, _ = r.shape
    yb = y.reshape(*y.shape[:-1], k, s)
    return torch.einsum("knm,...kn->...km", r, yb).reshape(y.shape)


def _act_ref(y, blocks, rescale, multiplier, constraint, conv):
    r = _rotation64(blocks)
    if conv:
        return (_apply_last(r, y.movedim(1, -1)) * rescale.view(-1)).movedim(-1, 1)
    return _apply_last(r, y) * rescale.view(-1)


def _weight_ref(weight, blocks, strength, constraint):
    r = _rotation64(blocks)
    k, s, _ = r.shape
    return torch.einsum("knm,kni->kmi", r, weight.reshape(k, s, -1)).reshape(
        weight.shape
    )


def _act_case(sweep, x_val, label, shape, conv, s, dtype, arms):
    channels = shape[1] if conv else shape[-1]
    k = channels // s
    inputs = {
        "y": rnd(*shape, dtype=dtype),
        "blocks": rnd(k, s, s, dtype=torch.float32, scale=0.05),
        "rescale": 1.0 + rnd(channels, 1, dtype=torch.float32, scale=0.1),
        "multiplier": MULT,
        "constraint": CONSTRAINT,
        "conv": conv,
    }
    axis = 1 if conv else -1

    def adapt(y, blocks, rescale, multiplier, constraint, conv):
        return blocks, y, rescale, constraint, multiplier, False, axis

    fixed_note = "ComfyUI OFTDiff.g() with the inverse's dtype matched (as shipped it raises in bf16/fp16)"
    built = {
        "comfy": lambda: comfy(
            comfy_code.oft_act, note="ComfyUI OFTDiff.g() as shipped"
        ),
        "comfy_fixed": lambda: comfy(comfy_code.oft_act_fixed, note=fixed_note),
        "compile": lambda: compile_arm(comfy_code.oft_act_fixed, inputs),
        "eager": lambda: eager("oft_apply", adapt),
        "kitchen": lambda: kitchen("oft_apply", adapt),
    }
    n = inputs["y"].numel()
    e = inputs["y"].element_size()
    return Case(
        family=FAMILY,
        op="oft_conv" if conv else "oft_act",
        sweep=sweep,
        x=x_val,
        case=label,
        params={"shape": list(shape), "channels": channels, "block": s},
        inputs=inputs,
        grad_names=("y", "blocks"),
        ref=_act_ref,
        arms={
            kk: f()
            for kk, f in built.items()
            if kk in arms or (kk == "comfy_fixed" and "comfy" in arms)
        },
        logical_bytes=2 * n * e + k * s * s * 4,
        logical_flops=2 * n * s,
    )


def _weight_case(sweep, x_val, label, n, s, dtype, arms):
    inputs = {
        "weight": rnd(n, n, dtype=dtype),
        "blocks": rnd(n // s, s, s, dtype=torch.float32, scale=0.05),
        "strength": MULT,
        "constraint": CONSTRAINT,
    }

    def adapt(weight, blocks, strength, constraint):
        return blocks, weight, None, constraint, strength, False, 0

    built = {
        "comfy": lambda: comfy(
            comfy_code.oft_weight, note="ComfyUI OFTAdapter.calculate_weight()"
        ),
        "compile": lambda: compile_arm(comfy_code.oft_weight, inputs),
        "eager": lambda: eager("oft_apply", adapt),
        "kitchen": lambda: kitchen("oft_apply", adapt),
    }
    e = inputs["weight"].element_size()
    return Case(
        family=FAMILY,
        op="oft_weight",
        sweep=sweep,
        x=x_val,
        case=label,
        params={"dim": n, "block": s},
        inputs=inputs,
        grad_names=("blocks",),
        ref=_weight_ref,
        arms={kk: f() for kk, f in built.items() if kk in arms},
        logical_bytes=2 * n * n * e,
        logical_flops=2 * n * n * s,
    )


def cases(sweep, dtype, quick, arms):
    n, t, s = sw.FIXED_DIM, sw.FIXED_TOKENS, sw.FIXED_BLOCK
    cc = sw.CONV_CHANNELS
    if sweep == "tokens":
        for tok in sw.thin(sw.TOKENS, quick):
            yield lambda tok=tok: _act_case(
                sweep, tok, f"{tok} tok", (tok, n), False, s, dtype, arms
            )
    elif sweep == "batch":
        for b in sw.thin(sw.BATCH, quick):
            yield lambda b=b: _act_case(
                sweep, b, f"batch {b} x {t}", (b, t, n), False, s, dtype, arms
            )
    elif sweep == "dim":
        for d in sw.thin(sw.DIMS, quick):
            yield lambda d=d: _act_case(
                sweep, d, f"{d} ch", (t, d), False, s, dtype, arms
            )
    elif sweep == "block":
        for bs in sw.thin(sw.BLOCKS, quick):
            yield lambda bs=bs: _act_case(
                sweep, bs, f"block {bs}", (t, n), False, bs, dtype, arms
            )
    elif sweep == "conv_spatial":
        for hw in sw.thin(sw.SPATIAL, quick):
            yield lambda hw=hw: _act_case(
                sweep, hw, f"2x{cc}x{hw}x{hw}", (2, cc, hw, hw), True, s, dtype, arms
            )
    elif sweep == "conv_batch":
        for b in sw.thin(sw.BATCH, quick):
            yield lambda b=b: _act_case(
                sweep, b, f"{b}x{cc}x64x64", (b, cc, 64, 64), True, s, dtype, arms
            )
    elif sweep == "merge_dim":
        for d in sw.thin(sw.DIMS, quick):
            yield lambda d=d: _weight_case(sweep, d, f"{d}x{d}", d, s, dtype, arms)
    elif sweep == "merge_block":
        for bs in sw.thin(sw.BLOCKS, quick):
            yield lambda bs=bs: _weight_case(
                sweep, bs, f"block {bs}", n, bs, dtype, arms
            )
