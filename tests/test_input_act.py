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
