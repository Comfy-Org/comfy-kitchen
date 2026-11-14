# SPDX-License-Identifier: Apache-2.0
"""mps backend: RoPE and RMSNorm+RoPE against eager.

The reference is the eager op in float32 on the CPU. A Metal result passes when it
is no further from that reference than eager running on MPS in the same dtype
(MPS's own bf16/fp16 rms_norm is slightly less precise than the CPU's), with a
small floor for cases where both are exact.
"""
import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.backends import eager
from comfy_kitchen.exceptions import NoCapableBackendError
from comfy_kitchen.registry import registry

pytestmark = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and registry.is_available("mps")),
    reason="needs an Apple GPU with the mps backend registered",
)

DEV = "mps"
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
FLOOR = {torch.float32: 2e-5, torch.float16: 2e-3, torch.bfloat16: 1.6e-2}
SHAPES = [(2, 45, 24, 128), (1, 17, 4, 96)]


def freqs_for(lead, pairs):
    ang = torch.rand(*lead, pairs, device=DEV) * 6.28
    return torch.stack(
        [torch.stack([ang.cos(), -ang.sin()], -1), torch.stack([ang.sin(), ang.cos()], -1)], -2)


LAYOUTS = ["BNHD", "BHND", "packed-qkv k slice", "per-batch freqs", "2-D"]


def make_case(layout, b, n, h, d, dtype):
    """(tensor, freqs leading shape) for the layouts models use."""
    if layout == "BNHD":
        return torch.randn(b, n, h, d, device=DEV, dtype=dtype) * 2, (1, n, 1)
    if layout == "BHND":
        return torch.randn(b, h, n, d, device=DEV, dtype=dtype) * 2, (1, 1, n)
    if layout == "packed-qkv k slice":
        qkv = torch.randn(b, n, 3 * h * d, device=DEV, dtype=dtype) * 2
        return qkv.chunk(3, dim=-1)[1].view(b, n, h, d), (1, n, 1)
    if layout == "per-batch freqs":
        return torch.randn(b, n, h, d, device=DEV, dtype=dtype) * 2, (b, n, 1)
    return torch.randn(n, d, device=DEV, dtype=dtype) * 2, (n,)


def assert_close_to_reference(got, stock, ref):
    ref = ref.float()
    err = (got.float().cpu() - ref).abs().max().item()
    stock_err = (stock.float().cpu() - ref).abs().max().item()
    floor = FLOOR[got.dtype] * max(1.0, ref.abs().max().item())
    assert got.shape == stock.shape and got.dtype == stock.dtype
    assert err <= max(stock_err * 1.05, floor), f"kernel {err:.3g} vs eager on mps {stock_err:.3g}"


CASES = [
    pytest.param(dtype, shape, layout, id=f"{str(dtype)[6:]}-{layout}-D{shape[-1]}")
    for dtype in DTYPES
    for shape in SHAPES
    for layout in LAYOUTS
]


@pytest.mark.parametrize("split_half", [False, True], ids=["interleaved", "split_half"])
@pytest.mark.parametrize(("dtype", "shape", "layout"), CASES)
def test_rope_matches_eager(dtype, shape, layout, split_half):
    torch.manual_seed(0)
    x, lead = make_case(layout, *shape, dtype)
    fr = freqs_for(lead, x.shape[-1] // 2)
    w = (torch.rand(x.shape[-1], device=DEV) + 0.5).to(x.dtype)
    plain, plain_ck = (
        (eager.apply_rope_split_half1, ck.apply_rope_split_half1) if split_half
        else (eager.apply_rope1, ck.apply_rope1))
    rms, rms_ck = (
        (eager.rms_rope_split_half1, ck.rms_rope_split_half1) if split_half
        else (eager.rms_rope1, ck.rms_rope1))
    plain_op = "apply_rope_split_half1" if split_half else "apply_rope1"
    assert registry.get_capable_backend(plain_op, {"x": x, "freqs_cis": fr}) == "mps"
    rms_op = "rms_rope_split_half1" if split_half else "rms_rope1"
    assert registry.get_capable_backend(
        rms_op, {"x": x, "freqs_cis": fr, "scale": w, "epsilon": 1e-6}
    ) == "mps"
    assert_close_to_reference(plain_ck(x, fr), plain(x, fr), plain(x.float().cpu(), fr.cpu()))
    assert_close_to_reference(rms_ck(x, fr, w, 1e-6), rms(x, fr, w, 1e-6),
                              rms(x.float().cpu(), fr.cpu(), w.float().cpu(), 1e-6))


@pytest.mark.parametrize("split_half", [False, True], ids=["interleaved", "split_half"])
def test_identity_frequencies_preserve_each_component_exactly(split_half):
    """cos=1, sin=0 must return the input bit-for-bit, not just within the floor."""
    torch.manual_seed(0)
    b, n, h, d = 2, 5, 3, 16
    x = (torch.rand(b, n, h, d, device=DEV, dtype=torch.float32) - 0.5) * 0.02
    fr = torch.zeros(1, n, 1, d // 2, 2, 2, device=DEV, dtype=torch.float32)
    fr[..., 0, 0] = 1.0
    fr[..., 1, 1] = 1.0
    op = ck.apply_rope_split_half1 if split_half else ck.apply_rope1
    assert torch.equal(op(x, fr), x)


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d)[6:])
def test_packed_qkv_in_place_leaves_v_alone(dtype):
    torch.manual_seed(0)
    b, n, h, d = 2, 21, 6, 128
    qkv = torch.randn(b, n, 3 * h * d, device=DEV, dtype=dtype)
    fr = freqs_for((1, n, 1), d // 2)
    w = (torch.rand(d, device=DEV) + 0.5).to(dtype)
    for inplace, outplace, extra in [
        (ck.apply_rope_, ck.apply_rope, ()),
        (ck.apply_rope_split_half_, ck.apply_rope_split_half, ()),
        (ck.rms_rope_, ck.rms_rope, (w, None, 1e-6)),
        (ck.rms_rope_split_half_, ck.rms_rope_split_half, (w, None, 1e-6)),
    ]:
        buf = qkv.clone()
        q, k, v = (t.view(b, n, h, d) for t in buf.chunk(3, dim=-1))
        v_before = v.clone()
        q0, k0 = (t.view(b, n, h, d) for t in qkv.chunk(3, dim=-1)[:2])
        expect = outplace(q0, k0, fr, *extra)
        inplace(q, k, fr, *extra)
        assert torch.equal(q, expect[0]) and torch.equal(k, expect[1])
        assert torch.equal(v, v_before)


@pytest.mark.parametrize("split_half", [False, True], ids=["interleaved", "split_half"])
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d)[6:])
def test_paired_rms_rope_uses_each_scale(dtype, split_half):
    torch.manual_seed(0)
    b, n, h, d = 2, 9, 4, 64
    q = torch.randn(b, n, h, d, device=DEV, dtype=dtype)
    k = torch.randn_like(q)
    fr = freqs_for((1, n, 1), d // 2)
    q_scale = (torch.rand(d, device=DEV) + 0.5).to(dtype)
    k_scale = (torch.rand(d, device=DEV) + 0.5).to(dtype)
    op = ck.rms_rope_split_half if split_half else ck.rms_rope
    eager_op = eager.rms_rope_split_half if split_half else eager.rms_rope
    op_name = "rms_rope_split_half" if split_half else "rms_rope"
    call = {"q": q, "k": k, "freqs_cis": fr, "q_scale": q_scale, "k_scale": k_scale,
            "epsilon": 1e-6}
    assert registry.get_capable_backend(op_name, call) == "mps"
    got = op(q, k, fr, q_scale, k_scale, 1e-6)
    stock = eager_op(q, k, fr, q_scale, k_scale, 1e-6)
    ref = eager_op(q.float().cpu(), k.float().cpu(), fr.cpu(), q_scale.float().cpu(),
                   k_scale.float().cpu(), 1e-6)
    for g, s, r in zip(got, stock, ref, strict=True):
        assert_close_to_reference(g, s, r)


@pytest.mark.parametrize("split_half", [False, True], ids=["interleaved", "split_half"])
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d)[6:])
def test_paired_rms_rope_allows_different_head_counts(dtype, split_half):
    torch.manual_seed(0)
    b, n, d = 2, 9, 64
    q = torch.randn(b, n, 4, d, device=DEV, dtype=dtype)
    k = torch.randn(b, n, 2, d, device=DEV, dtype=dtype)
    fr = freqs_for((1, n, 1), d // 2)
    q_scale = (torch.rand(d, device=DEV) + 0.5).to(dtype)
    k_scale = (torch.rand(d, device=DEV) + 0.5).to(dtype)
    op = ck.rms_rope_split_half if split_half else ck.rms_rope
    eager_op = eager.rms_rope_split_half if split_half else eager.rms_rope
    op_name = "rms_rope_split_half" if split_half else "rms_rope"
    call = {"q": q, "k": k, "freqs_cis": fr, "q_scale": q_scale, "k_scale": k_scale,
            "epsilon": 1e-6}
    assert registry.get_capable_backend(op_name, call) == "mps"
    got = op(q, k, fr, q_scale, k_scale, 1e-6)
    stock = eager_op(q, k, fr, q_scale, k_scale, 1e-6)
    ref = eager_op(q.float().cpu(), k.float().cpu(), fr.cpu(), q_scale.float().cpu(),
                   k_scale.float().cpu(), 1e-6)
    for g, s, r in zip(got, stock, ref, strict=True):
        assert_close_to_reference(g, s, r)


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d)[6:])
def test_partial_rotary(dtype):
    torch.manual_seed(0)
    q = torch.randn(1, 40, 8, 128, device=DEV, dtype=dtype)
    k = torch.randn_like(q)
    fr = freqs_for((1, 40, 1), 32)
    w = (torch.rand(128, device=DEV) + 0.5).to(dtype)
    got = ck.rms_rope_split_half(q, k, fr, w, None, 1e-6, rot_dim=64)
    stock = eager.rms_rope_split_half(q, k, fr, w, None, 1e-6, rot_dim=64)
    ref = eager.rms_rope_split_half(q.float().cpu(), k.float().cpu(), fr.cpu(), w.float().cpu(), None,
                                    1e-6, rot_dim=64)
    for g, s, r in zip(got, stock, ref, strict=True):
        assert_close_to_reference(g, s, r)


def test_unsupported_calls_fall_back_to_eager():
    x = torch.randn(1, 8, 2, 64, device=DEV, dtype=torch.bfloat16)
    fr = freqs_for((1, 8, 1), 32)
    declined = {
        "odd head dim": {"x": torch.randn(1, 8, 2, 7, device=DEV), "freqs_cis": freqs_for((1, 8, 1), 3)},
        "freqs pair count mismatch": {"x": x, "freqs_cis": freqs_for((1, 8, 1), 16)},
        "non-contiguous last dim": {"x": x.transpose(-1, -2).contiguous().transpose(-1, -2),
                                    "freqs_cis": fr},
        "float16 freqs": {"x": x, "freqs_cis": fr.half()},
    }
    for name, kwargs in declined.items():
        assert registry.get_capable_backend("apply_rope1", kwargs) == "eager", name
    assert registry.get_capable_backend("apply_rope1", {"x": x.cpu(), "freqs_cis": fr.cpu()}) == "eager"
    xt = x.transpose(-1, -2).contiguous().transpose(-1, -2)
    assert torch.equal(ck.apply_rope1(xt, fr), eager.apply_rope1(xt, fr))


def test_missing_scale_is_left_to_eager():
    # eager reads scale=None as an RMSNorm without weight; the kernel would skip the norm
    x = torch.randn(1, 8, 2, 64, device=DEV, dtype=torch.bfloat16)
    fr = freqs_for((1, 8, 1), 32)
    w = torch.ones(64, device=DEV, dtype=torch.bfloat16)
    kwargs = {"x": x, "freqs_cis": fr, "scale": None, "epsilon": 1e-6}
    assert registry.get_capable_backend("rms_rope1", kwargs) == "eager"
    pair = {"q": x, "k": x, "freqs_cis": fr, "q_scale": None, "k_scale": w, "epsilon": 1e-6}
    assert registry.get_capable_backend("rms_rope", pair) == "eager"
    assert registry.get_capable_backend("rms_rope_split_half", pair) == "eager"
    pair = {"q": x, "k": x, "freqs_cis": fr, "q_scale": w, "k_scale": None, "epsilon": 1e-6}
    assert registry.get_capable_backend("rms_rope", pair) == "mps"
    assert registry.get_capable_backend("rms_rope_split_half", pair) == "mps"


@pytest.mark.parametrize(
    "shape", [(1, 64), (64, 1), (1, 1, 64)], ids=lambda s: "x".join(map(str, s)))
def test_non_1d_scale_is_rejected_like_eager(shape):
    # rope() flattens the scale, but eager needs exactly [head_dim], where it is
    # rejected by the backend constraint. A broadcast view must not start working
    # only because the call landed on MPS.
    x = torch.randn(1, 8, 2, 64, device=DEV, dtype=torch.bfloat16)
    fr = freqs_for((1, 8, 1), 32)
    w = torch.ones(64, device=DEV, dtype=torch.bfloat16).view(shape)
    with pytest.raises(NoCapableBackendError):
        registry.get_capable_backend(
            "rms_rope1", {"x": x, "freqs_cis": fr, "scale": w, "epsilon": 1e-6})
    with pytest.raises(NoCapableBackendError):
        ck.rms_rope1(x, fr, w, 1e-6)


def test_shader_compile_failure_falls_back_to_eager(monkeypatch):
    """A failed compile is cached: two calls land on eager after one attempt."""
    from comfy_kitchen.backends.mps import kernels

    real_compile = torch.mps.compile_shader
    attempts = []

    def counting_compile(source):
        attempts.append(source)
        return real_compile(source)

    monkeypatch.setattr(kernels, "_lib", None)
    monkeypatch.setattr(kernels, "_compile_error", None)
    monkeypatch.setattr(kernels, "_SRC", "this is not Metal")
    monkeypatch.setattr(torch.mps, "compile_shader", counting_compile)
    x = torch.randn(1, 8, 2, 64, device=DEV, dtype=torch.bfloat16)
    fr = freqs_for((1, 8, 1), 32)
    expected = eager.apply_rope1(x, fr)
    for _ in range(2):
        assert registry.get_capable_backend("apply_rope1", {"x": x, "freqs_cis": fr}) == "eager"
        assert torch.equal(ck.apply_rope1(x, fr), expected)
    assert len(attempts) == 1
