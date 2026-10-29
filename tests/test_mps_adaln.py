# SPDX-License-Identifier: Apache-2.0
"""mps backend: adaln and rms_adaln against eager (see test_mps_rope for the criterion)."""
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
SHAPES = [
    ((1, 4141, 3072), (1, 1, 3072)),
    ((3, 50, 200), (200,)),
    ((2, 9, 1000), (2, 9, 1000)),
]


@pytest.mark.parametrize("name", ["adaln", "rms_adaln"])
@pytest.mark.parametrize(("shape", "mod_shape"), SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d)[6:])
def test_matches_eager(name, shape, mod_shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(*shape, device=DEV, dtype=dtype) * 3 + 0.5
    scale = torch.randn(*mod_shape, device=DEV, dtype=dtype) * 0.2
    shift = torch.randn(*mod_shape, device=DEV, dtype=dtype) * 0.2
    kwargs = {"x": x, "scale": scale, "shift": shift}
    assert registry.get_capable_backend(name, kwargs) == "mps"
    got = getattr(ck, name)(x, scale, shift, 1e-6)
    stock = getattr(eager, name)(x, scale, shift, 1e-6)
    ref = getattr(eager, name)(x.float().cpu(), scale.float().cpu(), shift.float().cpu(), 1e-6)
    err = (got.float().cpu() - ref).abs().max().item()
    stock_err = (stock.float().cpu() - ref).abs().max().item()
    floor = FLOOR[dtype] * max(1.0, ref.abs().max().item())
    assert got.shape == stock.shape and got.dtype == stock.dtype
    assert err <= max(stock_err * 1.05, floor), f"kernel {err:.3g} vs eager on mps {stock_err:.3g}"


def test_mixed_dtypes_fall_back_to_eager():
    x = torch.randn(2, 8, 64, device=DEV, dtype=torch.bfloat16)
    zeros = torch.zeros(64, device=DEV)
    assert registry.get_capable_backend("adaln", {"x": x, "scale": zeros, "shift": zeros}) == "eager"
    assert torch.equal(ck.adaln(x, zeros, zeros), eager.adaln(x, zeros, zeros))
