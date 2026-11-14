# SPDX-License-Identifier: Apache-2.0
"""ck.input_act: the standalone, dispatched form of the linear input activations."""
import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.backends._activations import apply_input_act


@pytest.mark.parametrize("act", ["gelu_tanh", "swiglu", "situ", "rms_norm"])
def test_eager_matches_apply_input_act(act):
    torch.manual_seed(0)
    x = torch.randn(3, 5, 16)
    weight = torch.rand(16) if act == "rms_norm" else None
    with ck.use_backend("eager"):
        got = ck.input_act(x, act, weight, 1e-6)
    assert torch.equal(got, apply_input_act(x, act, weight, 1e-6))


@pytest.mark.parametrize("act", [None, "none"])
def test_no_activation_returns_input(act):
    x = torch.randn(4, 8)
    assert ck.input_act(x, act) is x


@pytest.mark.parametrize(("act", "width"), [("gelu_tanh", 16), ("swiglu", 8), ("situ", 8)])
def test_compiles_with_the_right_output_shape(act, width):
    x = torch.randn(2, 16)
    fn = torch.compile(lambda t: ck.input_act(t, act), fullgraph=True)
    out = fn(x)
    assert out.shape == (2, width)
    assert torch.equal(out, apply_input_act(x, act))


def test_gradients_flow_through_the_eager_path():
    x = torch.randn(2, 8, dtype=torch.double, requires_grad=True)
    w = torch.rand(8, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x, w: ck.input_act(x, "rms_norm", w, 1e-6), (x, w))

    for act in ("gelu_tanh", "swiglu", "situ"):
        y = torch.randn(2, 16, dtype=torch.double, requires_grad=True)
        assert torch.autograd.gradcheck(lambda y, act=act: ck.input_act(y, act), (y,))


def _forward_reference(t, act, weight=None):
    """Out-of-place activation used as the forward-mode tangent reference."""
    if act == "gelu_tanh":
        return torch.nn.functional.gelu(t, approximate="tanh")
    if act == "swiglu":
        gate, up = t.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * up
    if act == "situ":
        hidden, gate = t.chunk(2, dim=-1)
        gated = gate.div(4.0).tanh().mul(4.0) * torch.sigmoid(gate)
        return hidden.div(25.0).tanh().mul(25.0) * gated
    return t * torch.rsqrt(t.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * weight


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("act", ["gelu_tanh", "swiglu", "situ", "rms_norm"])
def test_forward_mode_tangent_survives_no_grad(act, device):
    """A dual input keeps its tangent even when reverse-mode recording is off."""
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("needs an Apple GPU")
    torch.manual_seed(0)
    # MPS has no float64; the CPU run keeps the tighter double tolerance. The MPS
    # case is what catches a fallback to F.rms_norm, whose weighted forward-AD
    # path raises there.
    dtype = torch.float32 if device == "mps" else torch.double
    x = torch.randn(2, 16, dtype=dtype, device=device)
    tangent_in = torch.randn_like(x)
    weight = torch.rand(16, dtype=dtype, device=device) * 0.5 + 0.5
    kwargs = {"act_weight": weight, "act_eps": 1e-6} if act == "rms_norm" else {}
    with torch.no_grad():
        _, tangent = torch.func.jvp(lambda t: ck.input_act(t, act, **kwargs), (x,), (tangent_in,))
        _, expected = torch.func.jvp(
            lambda t: _forward_reference(t, act, weight), (x,), (tangent_in,))
    assert tangent.abs().max() > 0
    torch.testing.assert_close(tangent, expected)


def test_weight_gradient_flows_without_input_grad():
    x = torch.randn(2, 8, dtype=torch.double)
    w = torch.rand(8, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(lambda w: ck.input_act(x, "rms_norm", w, 1e-6), (w,))


def test_transposed_input_keeps_the_eager_layout():
    """The fake kernel must not claim contiguous strides for a transposed input."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    x = torch.randn(3, 2, 16).transpose(0, 1)
    for act in ("gelu_tanh", "swiglu", "situ"):
        with ck.use_backend("eager"):
            real = ck.input_act(x, act)
        with FakeTensorMode():
            fake = torch.ops.comfy_kitchen.input_act(
                torch.empty_strided(x.shape, x.stride(), dtype=x.dtype), act, None, 0.0
            )
        assert fake.shape == real.shape
        assert fake.stride() == real.stride()

        # A compiled downstream reshape must see the same layout as eager.
        compiled = torch.compile(lambda t, act=act: ck.input_act(t, act).reshape(-1), fullgraph=True)
        assert torch.equal(compiled(x), apply_input_act(x, act).reshape(-1))


@pytest.mark.parametrize("act", ["swiglu", "situ"])
def test_gated_activations_reject_an_odd_last_dim(act):
    with pytest.raises(ValueError, match="needs an even last dim"):
        ck.input_act(torch.randn(2, 3), act)


def test_op_disabled_returns_fresh_values():
    x = torch.randn(4, 8)
    out = torch.ops.comfy_kitchen.input_act(x, "none", None, 0.0)
    torch.testing.assert_close(out, x)
    assert out.data_ptr() != x.data_ptr()


def test_compiles_rms_norm_with_a_weight():
    x = torch.randn(2, 16)
    weight = torch.rand(16)
    fn = torch.compile(lambda t: ck.input_act(t, "rms_norm", weight, 1e-6), fullgraph=True)
    out = fn(x)
    assert torch.equal(out, apply_input_act(x, "rms_norm", weight, 1e-6))


@pytest.mark.parametrize("act", ["swiglu", "situ"])
@pytest.mark.parametrize("width", [3, 5])
def test_op_rejects_an_odd_last_dim(act, width):
    with pytest.raises(ValueError, match="needs an even last dim"):
        torch.ops.comfy_kitchen.input_act(torch.randn(2, width), act, None, 0.0)


@pytest.mark.parametrize("act", ["gelu_tanh", "rms_norm"])
def test_float64_takes_the_eager_path(act):
    x = torch.randn(2, 8, dtype=torch.float64)
    weight = torch.rand(8, dtype=torch.float64) if act == "rms_norm" else None
    out = ck.input_act(x, act, weight, 1e-6)
    assert out.dtype == torch.float64
    assert torch.equal(out, apply_input_act(x, act, weight, 1e-6))
