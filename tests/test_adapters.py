# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Weight-adapter ops against an fp64 reference written from the definitions.

One case per op x dtype x kernel backend (tilelang, triton), forward and
backward. The reference never calls the eager backend: each family is
restated in float64 (block-diagonal matrices for OFT, an explicit kron for
LoKr), and its gradients come from autograd on that restatement.
"""

import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.registry import registry

try:
    from comfy_kitchen.backends.triton.adapters.tuning import set_mode
except ImportError:
    set_mode = None

ADAPTERS = ck.adapters
_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
_BACKENDS = ["tilelang", "triton"]
# (max-abs, frobenius) relative error bounds per compute dtype.
_TOL = {
    torch.float32: (5e-4, 2e-4),
    torch.float16: (2e-2, 4e-3),
    torch.bfloat16: (5e-2, 2e-2),
}


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


DEVICE = _device()


@pytest.fixture(autouse=True, scope="module")
def _safe_configs():
    """Every kernel runs its smallest legal config, untimed."""
    if set_mode is not None:
        set_mode("safe")
    yield
    if set_mode is not None:
        set_mode("on")


def _require(backend, name, kwargs):
    if not registry.is_available(backend):
        pytest.skip(f"{backend} not available")
    if not registry.validate_backend_for_call(backend, name, kwargs).success:
        pytest.skip(f"{backend} does not take {name} with these inputs")


def _check(got, ref, dtype, name):
    assert got.shape == ref.shape, f"{name}: shape {got.shape} != {ref.shape}"
    got, ref = got.double(), ref.double()
    assert ref.abs().max() > 0, f"{name}: reference is all zero, the check is vacuous"
    max_err = ((got - ref).abs().max() / ref.abs().max()).item()
    fro_err = ((got - ref).norm() / ref.norm()).item()
    tmax, tfro = _TOL[dtype]
    assert (
        max_err <= tmax and fro_err <= tfro
    ), f"{name}: max-rel {max_err:.2e} (<= {tmax}), fro-rel {fro_err:.2e} (<= {tfro})"


def _rand(*shape, dtype=torch.float32, scale=1.0):
    return (torch.randn(*shape, device=DEVICE, dtype=torch.float32) * scale).to(dtype)


def _compare(backend, op, inputs, reference, dtype, tag, out_dtype=None):
    """op(*leaves) under backend vs reference(*fp64 leaves), output and every grad."""
    leaves = [t.detach().clone().requires_grad_(True) for t in inputs]
    with ck.use_backend(backend):
        out = op(*leaves)
    up = torch.randn_like(out)
    got = torch.autograd.grad(out, leaves, up)
    ref_leaves = [t.detach().double().requires_grad_(True) for t in inputs]
    ref = reference(*ref_leaves)
    want = torch.autograd.grad(ref, ref_leaves, up.double())
    tag = f"{tag}[{backend}/{dtype}]"
    assert out.dtype == (out_dtype or dtype), f"{tag}: out dtype {out.dtype}"
    _check(out, ref, dtype, tag)
    for k, (g, w, leaf) in enumerate(zip(got, want, leaves, strict=True)):
        assert g.dtype == leaf.dtype, f"{tag}: grad {k} dtype {g.dtype} != {leaf.dtype}"
        _check(g, w, dtype, f"{tag} grad {k}")


def _rounded(t, dtype):
    """t rounded to dtype in value, identity in gradient (factors are cast to x's dtype)."""
    return t.detach().to(dtype).double() + (t - t.detach())


def _kron(w1, w2):
    a, b = w1.shape
    c, d = w2.shape
    return (w1[:, None, :, None] * w2[None, :, None, :]).reshape(a * c, b * d)


def _ref_oft(blocks, x, rescale, constraint, multiplier, shift, axis):
    q = blocks - blocks.transpose(-1, -2)
    norm = q.flatten().norm() + 1e-8
    if constraint > 0 and norm > constraint:
        q = q * (constraint / norm)
    eye = torch.eye(blocks.shape[-1], dtype=blocks.dtype, device=blocks.device)
    r = torch.linalg.solve((eye - q).transpose(-1, -2), (eye + q).transpose(-1, -2))
    r = multiplier * r.transpose(-1, -2) + (1 - multiplier) * eye
    full = torch.block_diag(*[blk.t() for blk in r])
    xc = x.movedim(axis, -1)
    out = (xc @ full.t()) * rescale.reshape(-1)
    if shift:
        out = out - xc
    return out.movedim(-1, axis)


# Shapes off every 16-multiple, so masked edges and padding are exercised.
_LEAD, _IN, _OUT, _RANK = (3, 13), 200, 150, 12
_A, _B, _C, _D = 16, 16, 40, 36


def _lowrank(family, dtype):
    if family == "lora":
        return [
            _rand(_OUT, _RANK, dtype=dtype, scale=0.2),
            _rand(_RANK, _IN, dtype=dtype, scale=0.2),
        ]
    shapes = ((_OUT, _RANK), (_RANK, _IN), (_OUT, _RANK), (_RANK, _IN))
    return [_rand(*s, dtype=dtype, scale=0.4) for s in shapes]


def _delta(family, factors):
    if family == "lora":
        return factors[0] @ factors[1]
    if family == "loha":
        a1, b1, a2, b2 = factors
        return (a1 @ b1) * (a2 @ b2)
    return _kron(*factors)


_ARGS = {
    "lora": ("up", "down"),
    "loha": ("w1_up", "w1_down", "w2_up", "w2_down"),
    "lokr": ("w1", "w2"),
}


def _factors(family, dtype):
    if family == "lokr":
        return [
            _rand(_A, _B, dtype=dtype, scale=0.3),
            _rand(_C, _D, dtype=dtype, scale=0.3),
        ]
    return _lowrank(family, dtype)


def _shape(family):
    """(in, out) of the adapted layer."""
    return (_B * _D, _A * _C) if family == "lokr" else (_IN, _OUT)


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("family", ["lora", "loha", "lokr"])
def test_bypass(family, dtype, backend):
    torch.manual_seed(0)
    i, o = _shape(family)
    x = _rand(*_LEAD, i, dtype=dtype)
    factors = _factors(family, dtype)
    base = _rand(*_LEAD, o, dtype=dtype)
    _require(
        backend,
        f"{family}_bypass",
        {"x": x, "base": base, **dict(zip(_ARGS[family], factors, strict=True))},
    )
    n = len(factors)
    fn = getattr(ADAPTERS, f"{family}_bypass")

    def op(x, *rest):
        return fn(x, *rest[:n], 0.7, rest[n])

    def reference(x, *rest):
        delta = _delta(family, [_rounded(f, dtype) for f in rest[:n]])
        return rest[n] + 0.7 * (x @ delta.t())

    _compare(backend, op, [x, *factors, base], reference, dtype, f"{family}_bypass")


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("family", ["lora", "loha", "lokr"])
def test_merge(family, dtype, backend):
    torch.manual_seed(0)
    i, o = _shape(family)
    weight = _rand(o, i, dtype=dtype)
    factors = _factors(family, dtype)
    _require(
        backend,
        f"{family}_merge",
        {"weight": weight, **dict(zip(_ARGS[family], factors, strict=True))},
    )
    fn = getattr(ADAPTERS, f"{family}_merge")
    _compare(
        backend,
        lambda w, *f: fn(w, *f, 0.8),
        [weight, *factors],
        lambda w, *f: w + 0.8 * _delta(family, f),
        dtype,
        f"{family}_merge",
    )


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_oft(dtype, backend):
    """Blocks of 12 on channels-last activations and blocks of 6 (the fma layout's range) on conv activations."""
    torch.manual_seed(0)
    for k, s, shape, axis in ((8, 12, (2, 50, 96), -1), (16, 6, (2, 96, 5, 7), 1)):
        x = _rand(*shape, dtype=dtype)
        blocks = _rand(k, s, s, scale=0.05)
        rescale = 1 + 0.1 * _rand(k * s)
        _require(backend, "oft_apply", {"blocks": blocks, "x": x, "rescale": rescale})
        args = (0.05, 0.6, True, axis)
        _compare(
            backend,
            lambda b, x, r, args=args: ADAPTERS.oft_apply(b, x, r, *args),
            [blocks, x, rescale],
            lambda b, x, r, args=args: _ref_oft(b, x, r, *args),
            dtype,
            f"oft_apply s={s}",
        )


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_weight_decompose(dtype, backend):
    """ComfyUI's output-axis form (delta, base and norm are W) and an input-axis form (given base, norm of M)."""
    torch.manual_seed(0)
    w = _rand(_OUT, _IN, dtype=dtype)
    delta = _rand(_OUT, _IN, dtype=dtype, scale=0.05)
    d_out = (_rand(_OUT).abs() + 0.5).to(dtype)
    d_in = (_rand(_IN).abs() + 0.5).to(dtype)
    base = _rand(_OUT, _IN, dtype=dtype)
    _require(
        backend,
        "weight_decompose",
        {"weight": w, "dora_scale": d_out, "base": base, "delta": delta},
    )
    eps = torch.finfo(dtype).eps

    def comfy_ref(w, d, dl):
        n = w.norm(dim=1) + eps
        return w + 0.7 * ((w + dl) * (d / n)[:, None] - w)

    _compare(
        backend,
        lambda w, d, dl: ADAPTERS.weight_decompose(w, d, 0.7, True, w, w, dl),
        [w, d_out, delta],
        comfy_ref,
        dtype,
        "weight_decompose output axis",
    )

    def input_axis_ref(w, d, b):
        n = w.norm(dim=0) + eps
        return b + 0.7 * (w * (d / n)[None, :] - b)

    _compare(
        backend,
        lambda w, d, b: ADAPTERS.weight_decompose(w, d, 0.7, False, b),
        [w, d_in, base],
        input_axis_ref,
        dtype,
        "weight_decompose input axis",
    )


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_channel_scale(dtype, backend):
    torch.manual_seed(0)
    x = _rand(4, 77, 320, dtype=dtype)
    w = _rand(320, dtype=dtype, scale=0.5)
    _require(backend, "channel_scale", {"x": x, "w": w})
    _compare(
        backend,
        lambda x, w: ADAPTERS.channel_scale(x, w, -1, 1.0, 0.8),
        [x, w],
        lambda x, w: x * (1.0 + 0.8 * w),
        dtype,
        "channel_scale",
    )


def _lora_kwargs(dtype):
    return {
        "x": _rand(8, 64, dtype=dtype),
        "up": _rand(32, 4, dtype=dtype),
        "down": _rand(4, 64, dtype=dtype),
        "scale": 1.0,
        "base": None,
    }


def test_kernel_backend_selected():
    """fp16 reaches tilelang where installed, else triton; bf16 reaches triton, eager below sm_80."""
    if DEVICE == "cpu" or set_mode is None:
        pytest.skip("no kernel device")
    pre_ampere = DEVICE == "cuda" and torch.cuda.get_device_capability() < (8, 0)
    want16 = (
        "tilelang"
        if DEVICE == "cuda" and registry.is_available("tilelang")
        else "triton"
    )
    want_bf16 = "eager" if pre_ampere else "triton"
    assert (
        registry.get_capable_backend("lora_bypass", _lora_kwargs(torch.float16))
        == want16
    )
    assert (
        registry.get_capable_backend("lora_bypass", _lora_kwargs(torch.bfloat16))
        == want_bf16
    )
