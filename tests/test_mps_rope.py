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
from comfy_kitchen.registry import registry

pytestmark = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and registry.is_available("mps")),
    reason="needs an Apple GPU with the mps backend registered",
)

DEV = "mps"
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
FLOOR = {torch.float32: 2e-5, torch.float16: 2e-3, torch.bfloat16: 1.6e-2}
SHAPES = [(1, 45, 24, 128), (2, 33, 8, 64), (1, 17, 4, 96), (2, 9, 2, 256)]


def freqs_for(lead, pairs):
    ang = torch.rand(*lead, pairs, device=DEV) * 6.28
    return torch.stack(
        [torch.stack([ang.cos(), -ang.sin()], -1), torch.stack([ang.sin(), ang.cos()], -1)], -2)


def layouts(b, n, h, d, dtype):
    """(name, tensor, freqs leading shape) for the layouts models use."""
    yield "BNHD", torch.randn(b, n, h, d, device=DEV, dtype=dtype) * 2, (1, n, 1)
    yield "BHND", torch.randn(b, h, n, d, device=DEV, dtype=dtype) * 2, (1, 1, n)
    qkv = torch.randn(b, n, 3 * h * d, device=DEV, dtype=dtype) * 2
    yield "packed-qkv k slice", qkv.chunk(3, dim=-1)[1].view(b, n, h, d), (1, n, 1)
    yield "per-batch freqs", torch.randn(b, n, h, d, device=DEV, dtype=dtype) * 2, (b, n, 1)
    yield "2-D", torch.randn(n, d, device=DEV, dtype=dtype) * 2, (n,)


def assert_close_to_reference(got, stock, ref):
    ref = ref.float()
    err = (got.float().cpu() - ref).abs().max().item()
    stock_err = (stock.float().cpu() - ref).abs().max().item()
    floor = FLOOR[got.dtype] * max(1.0, ref.abs().max().item())
    assert got.shape == stock.shape and got.dtype == stock.dtype
    assert err <= max(stock_err * 1.05, floor), f"kernel {err:.3g} vs eager on mps {stock_err:.3g}"


def cases():
    for dtype in DTYPES:
        for shape in SHAPES:
            for name, x, lead in layouts(*shape, dtype):
                yield pytest.param(x, lead, id=f"{str(dtype)[6:]}-{name}-D{shape[-1]}")


@pytest.mark.parametrize("split_half", [False, True], ids=["interleaved", "split_half"])
@pytest.mark.parametrize(("x", "lead"), list(cases()))
def test_rope_matches_eager(x, lead, split_half):
    torch.manual_seed(0)
    fr = freqs_for(lead, x.shape[-1] // 2)
    w = (torch.rand(x.shape[-1], device=DEV) + 0.5).to(x.dtype)
    plain, plain_ck = (
        (eager.apply_rope_split_half1, ck.apply_rope_split_half1) if split_half
        else (eager.apply_rope1, ck.apply_rope1))
    rms, rms_ck = (
        (eager.rms_rope_split_half1, ck.rms_rope_split_half1) if split_half
        else (eager.rms_rope1, ck.rms_rope1))
    assert registry.get_capable_backend("apply_rope1", {"x": x, "freqs_cis": fr}) == "mps"
    assert_close_to_reference(plain_ck(x, fr), plain(x, fr), plain(x.float().cpu(), fr.cpu()))
    assert_close_to_reference(rms_ck(x, fr, w, 1e-6), rms(x, fr, w, 1e-6),
                              rms(x.float().cpu(), fr.cpu(), w.float().cpu(), 1e-6))


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d)[6:])
@pytest.mark.parametrize("split_half", [False, True], ids=["interleaved", "split_half"])
def test_in_place_matches_out_of_place(dtype, split_half):
    torch.manual_seed(0)
    x = torch.randn(2, 21, 6, 128, device=DEV, dtype=dtype)
    fr = freqs_for((1, 21, 1), 64)
    w = (torch.rand(128, device=DEV) + 0.5).to(dtype)
    pairs = [
        (ck.apply_rope_split_half1_ if split_half else ck.apply_rope1_,
         ck.apply_rope_split_half1 if split_half else ck.apply_rope1, ()),
        (ck.rms_rope_split_half1_ if split_half else ck.rms_rope1_,
         ck.rms_rope_split_half1 if split_half else ck.rms_rope1, (w, 1e-6)),
    ]
    for inplace, outplace, extra in pairs:
        y = x.clone()
        expect = outplace(x, fr, *extra)
        inplace(y, fr, *extra)
        assert torch.equal(y, expect)


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


def test_shader_compile_failure_falls_back_to_eager(monkeypatch):
    from comfy_kitchen.backends.mps import kernels

    attempts = 0
    compile_shader = torch.mps.compile_shader

    def counting_compile(src):
        nonlocal attempts
        attempts += 1
        return compile_shader(src)

    monkeypatch.setattr(kernels, "_lib", None)
    monkeypatch.setattr(kernels, "_compile_error", None)
    monkeypatch.setattr(kernels, "_SRC", "this is not Metal")
    monkeypatch.setattr(torch.mps, "compile_shader", counting_compile)
    x = torch.randn(1, 8, 2, 64, device=DEV, dtype=torch.bfloat16)
    fr = freqs_for((1, 8, 1), 32)
    for _ in range(2):  # the failure is cached, not retried and raised
        assert registry.get_capable_backend("apply_rope1", {"x": x, "freqs_cis": fr}) == "eager"
        assert torch.equal(ck.apply_rope1(x, fr), eager.apply_rope1(x, fr))
    ax = torch.randn(1, 8, 64, device=DEV, dtype=torch.bfloat16)
    sc = torch.ones(64, device=DEV, dtype=torch.bfloat16)
    sh = torch.zeros(64, device=DEV, dtype=torch.bfloat16)
    assert registry.get_capable_backend("adaln", {"x": ax, "scale": sc, "shift": sh}) == "eager"
    assert torch.equal(ck.adaln(ax, sc, sh), eager.adaln(ax, sc, sh))
    assert attempts == 1
    assert "compilation failed" in kernels._compile_error
