# SPDX-License-Identifier: Apache-2.0
"""mps backend: the gated input activations against eager (see test_mps_rope for the criterion)."""
import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.backends._activations import apply_input_act
from comfy_kitchen.registry import registry

pytestmark = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and registry.is_available("mps")),
    reason="needs an Apple GPU with the mps backend registered",
)

DEV = "mps"
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
FLOOR = {torch.float32: 2e-5, torch.float16: 2e-3, torch.bfloat16: 1.6e-2}
SHAPES = [(1, 4141, 16384), (3, 5, 2, 18)]


def assert_close_to_reference(got, stock, ref):
    ref = ref.float()
    err = (got.float().cpu() - ref).abs().max().item()
    stock_err = (stock.float().cpu() - ref).abs().max().item()
    floor = FLOOR[got.dtype] * max(1.0, ref.abs().max().item())
    assert got.shape == stock.shape and got.dtype == stock.dtype
    assert err <= max(stock_err * 1.05, floor), f"kernel {err:.3g} vs eager on mps {stock_err:.3g}"


@pytest.mark.parametrize("act", ["swiglu", "situ"])
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d)[6:])
def test_matches_eager(act, shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(*shape, device=DEV, dtype=dtype) * 4
    assert registry.get_capable_backend("input_act", {"x": x, "input_act": act}) == "mps"
    assert_close_to_reference(ck.input_act(x, act), apply_input_act(x, act),
                              apply_input_act(x.float().cpu(), act))


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d)[6:])
def test_strided_rows(dtype):
    torch.manual_seed(0)
    x = torch.randn(9, 128, device=DEV, dtype=dtype)[:, :64]
    assert not x.is_contiguous()
    assert_close_to_reference(ck.input_act(x, "swiglu"), apply_input_act(x, "swiglu"),
                              apply_input_act(x.float().cpu(), "swiglu"))


def test_other_activations_stay_on_eager():
    x = torch.randn(4, 64, device=DEV, dtype=torch.bfloat16)
    for act in ("gelu_tanh", "rms_norm"):
        assert registry.get_capable_backend("input_act", {"x": x, "input_act": act}) == "eager", act
    odd = torch.randn(4, 63, device=DEV)
    assert registry.get_capable_backend("input_act", {"x": odd, "input_act": "swiglu"}) == "eager"
    w = torch.rand(64, device=DEV, dtype=torch.bfloat16)
    assert torch.equal(ck.input_act(x, "rms_norm", w, 1e-6), apply_input_act(x, "rms_norm", w, 1e-6))
