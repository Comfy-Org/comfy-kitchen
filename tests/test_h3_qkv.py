# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.h3_qkv import _fallback, _op

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def operands(m):
    torch.manual_seed(324)
    a = torch.randint(-127, 128, (m, 5376), dtype=torch.int8, device="cuda")
    w = torch.randint(-127, 128, (21504, 5376), dtype=torch.int8, device="cuda")
    xs = torch.rand(m, 1, device="cuda") * 0.001
    ws = torch.rand(21504, 1, device="cuda") * 0.001
    angles = torch.rand(1, m, 1, 48, device="cuda")
    c, s = angles.cos(), angles.sin()
    rope = torch.stack((c, -s, s, c), dim=-1).reshape(1, m, 1, 48, 2, 2).to(torch.bfloat16)
    qw = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    kw = torch.randn_like(qw)
    return a, w, xs, ws, rope, qw, kw


@pytest.mark.parametrize("m", [37, 257, 1025, 4097, 8192, 14850])
def test_preparation_matches_native(m, record_property):
    args = operands(m)
    expected = _fallback(*args, 1e-5)
    actual = _op(*args, 1e-5)
    for name, out, ref in zip(("q", "k", "v", "qs", "ks", "vs"), actual, expected, strict=True):
        mismatch = (out != ref).sum().item()
        record_property(name + "_mismatch", mismatch)
        assert torch.equal(out, ref), (
            name,
            mismatch,
            (out.float() - ref.float()).abs().max().item(),
        )
    packed = ck.h3_qkv_prequantize(*args)
    out = ck.int8_attention_from_prequantized(packed)
    reference = ck.sage_attention.PrequantizedInt8Attention(
        *expected,
        128,
        torch.bfloat16,
        128**-0.5,
        128 if m > 1024 else 64,
        None,
    )
    assert torch.equal(out, ck.int8_attention_from_prequantized(reference))


def test_graph_and_compile():
    args = operands(8192)
    expected = _op(*args, 1e-5)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _op(*args, 1e-5)
    graph.replay()
    for a, b in zip(out, expected, strict=True):
        assert torch.equal(a, b)
    fn = torch.compile(_op, fullgraph=True)
    for a, b in zip(fn(*args, 1e-5), expected, strict=True):
        assert torch.equal(a, b)


@pytest.mark.parametrize("magnitude", [0.0, 1e-8, 1.0, 100.0])
def test_magnitudes_and_inputs_unchanged(magnitude):
    args = operands(8193)
    args[2].mul_(magnitude)
    before = [t.clone() for t in args]
    expected = _fallback(*args, 1e-5)
    actual = _op(*args, 1e-5)
    for a, b in zip(actual, expected, strict=True):
        assert torch.equal(a, b)
    for a, b in zip(args, before, strict=True):
        assert torch.equal(a, b)


def test_stream_and_strided_fallback():
    args = list(operands(8193))
    expected = _fallback(*args, 1e-5)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual = _op(*args, 1e-5)
    torch.cuda.current_stream().wait_stream(stream)
    for a, b in zip(actual, expected, strict=True):
        assert torch.equal(a, b)
    padded = torch.empty((8193, 5377), device="cuda", dtype=torch.int8)
    padded[:, :5376] = args[0]
    args[0] = padded[:, :5376]
    for a, b in zip(_op(*args, 1e-5), expected, strict=True):
        assert torch.equal(a, b)


@pytest.mark.parametrize("eps", [0.0, -1.0, 1e-40, 1e40, float("nan"), float("inf")])
def test_invalid_epsilon(eps):
    with pytest.raises(ValueError, match="eps"):
        _op(*operands(37), eps)


def test_invalid_rank():
    args = list(operands(37))
    args[0] = args[0].flatten()
    with pytest.raises(ValueError, match="matrix"):
        _op(*args, 1e-5)
