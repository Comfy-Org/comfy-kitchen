import pytest
import torch
import comfy_kitchen as ck


@pytest.mark.parametrize(
    "m,k,n,act", [(3, 256, 256, None), (4097, 256, 256, None), (4097, 256, 256, "swiglu")]
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_composed(m, k, n, act, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA")
    torch.manual_seed(42)
    x = torch.randn(m, k * (2 if act else 1), device=device, dtype=torch.bfloat16)
    w = torch.randint(-127, 128, (n, k), device=device, dtype=torch.int8)
    ws = torch.rand(n, device=device) * 0.01
    gate = torch.randn(3, n, device=device, dtype=torch.bfloat16)
    rows = torch.arange(m, device=device, dtype=torch.int32) % 3
    residual = torch.randn(m, n, device=device, dtype=torch.bfloat16)
    out = ck.int8_linear_indexed_gate(x, w, ws, gate, rows, residual, input_act=act)
    branch = ck.int8_linear(x, w, ws, convrot=True, input_act=act)
    ref = torch.addcmul(residual, branch, gate[rows.long()])
    assert torch.equal(out, ref)
    if device == "cuda":
        fn = torch.compile(ck.int8_linear_indexed_gate, fullgraph=True)
        assert torch.equal(fn(x, w, ws, gate, rows, residual, input_act=act), ref)
