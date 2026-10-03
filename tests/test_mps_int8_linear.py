"""Weight-only INT8 linear on Apple GPUs (mps backend)."""
import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.backends._activations import apply_input_act
from comfy_kitchen.tensor.int8_utils import _build_hadamard

pytestmark = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and ck.list_backends().get("mps", {}).get("available")),
    reason="mps backend not available",
)

G = 256


def _make(n, k, rowscale, convrot):
    torch.manual_seed(0)
    w = torch.randn(n, k) * 0.02
    if convrot:
        h = _build_hadamard(G)
        w = (w.reshape(n, k // G, G) @ h.T).reshape(n, k)
    s = (w.abs().amax(1, keepdim=True) / 127) if rowscale else (w.abs().max() / 127).reshape(1)
    q = torch.round(w / s).clamp(-127, 127).to(torch.int8)
    wd = q.float() * s.reshape(-1, 1)
    if convrot:
        wd = (wd.reshape(n, k // G, G) @ _build_hadamard(G)).reshape(n, k)
    return q, s, wd


def _rel(a, b):
    return ((a - b).norm() / b.norm()).item()


@pytest.mark.parametrize("dtype,tol", [(torch.float32, 2e-4), (torch.float16, 1e-3), (torch.bfloat16, 6e-3)])
@pytest.mark.parametrize("convrot", [True, False])
@pytest.mark.parametrize("rowscale", [True, False])
@pytest.mark.parametrize("act", [None, "gelu_tanh", "swiglu", "rms_norm"])
def test_matches_float_oracle(dtype, tol, convrot, rowscale, act):
    n, k, m = 384, 1024, 37
    q, s, wd = _make(n, k, rowscale, convrot)
    x = torch.randn(2, m, k * 2 if act == "swiglu" else k)
    b = torch.randn(n)
    aw = torch.randn(k) if act == "rms_norm" else None
    res, rs = torch.randn(2, m, n), torch.rand(n)
    exact = torch.addcmul(res, torch.nn.functional.linear(apply_input_act(x, act, aw, 1e-6), wd, b), rs)
    d = "mps"
    out = ck.int8_linear(
        x.to(d, dtype), q.to(d), s.to(d), b.to(d, dtype), dtype,
        convrot=convrot, convrot_groupsize=G, input_act=act,
        input_act_weight=None if aw is None else aw.to(d), input_act_eps=1e-6,
        residual=res.to(d, dtype), residual_scale=rs.to(d, dtype),
    )
    assert out.dtype == dtype and out.device.type == "mps"
    assert _rel(out.float().cpu(), exact) < tol


def test_dispatch_prefers_mps_on_mps_and_eager_on_cpu():
    q, s, _ = _make(256, 512, True, True)
    x = torch.randn(4, 512)
    assert ck.registry.get_implementation(
        "int8_linear", kwargs={"x": x.to("mps"), "weight": q.to("mps"), "weight_scale": s.to("mps")}
    ).__module__.endswith("backends.mps.linear")
    assert ".eager." in ck.registry.get_implementation(
        "int8_linear", kwargs={"x": x, "weight": q, "weight_scale": s}
    ).__module__


def test_chunked_decode_matches_unchunked(monkeypatch):
    from comfy_kitchen.backends.mps import linear
    q, s, _ = _make(512, 1024, True, True)
    x = torch.randn(9, 1024, device="mps", dtype=torch.float16)
    a = linear.int8_linear(x, q.to("mps"), s.to("mps"), out_dtype=torch.float16, convrot=True)
    monkeypatch.setattr(linear, "_MAX_DECODE_BYTES", 2 * 1024 * 100)
    b = linear.int8_linear(x, q.to("mps"), s.to("mps"), out_dtype=torch.float16, convrot=True)
    assert torch.equal(a.cpu(), b.cpu())


def test_transposed_decode_matches_plain(monkeypatch):
    from comfy_kitchen.backends.mps import linear
    q, s, _ = _make(384, 1024, True, True)
    x = torch.randn(9, 1024, device="mps", dtype=torch.float16)
    a = linear.int8_linear(x, q.to("mps"), s.to("mps"), out_dtype=torch.float16, convrot=True)
    monkeypatch.setattr(linear, "_TRANSPOSED_DECODE_MIN_K", 1)
    b = linear.int8_linear(x, q.to("mps"), s.to("mps"), out_dtype=torch.float16, convrot=True)
    assert torch.allclose(a.cpu().float(), b.cpu().float(), rtol=2e-3, atol=2e-3)
    monkeypatch.setattr(linear, "_MAX_DECODE_BYTES", 2 * 1024 * 100)
    c = linear.int8_linear(x, q.to("mps"), s.to("mps"), out_dtype=torch.float16, convrot=True)
    assert torch.equal(b.cpu(), c.cpu())
