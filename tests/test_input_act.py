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
