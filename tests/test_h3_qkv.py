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


def test_declined_native_releases_scratch(monkeypatch):
    from comfy_kitchen import h3_qkv
    from comfy_kitchen.backends import cuda

    if torch.cuda.get_device_capability() != (12, 0) or cuda._C is None:
        pytest.skip("SM120 extension required")
    args = operands(8193)
    before = torch.cuda.memory_allocated()
    fallback = h3_qkv._fallback

    def check(*a):
        assert torch.cuda.memory_allocated() == before
        return fallback(*a)

    monkeypatch.setattr(cuda._C, "h3_qkv_quant", lambda *a: False)
    monkeypatch.setattr(h3_qkv, "_fallback", check)
    h3_qkv._op(*args, 1e-5)


@pytest.mark.parametrize("m", [37, 1025, 8193])
def test_no_cutlass_matches_projection(monkeypatch, m):
    from comfy_kitchen.backends import cuda

    if cuda._C is None:
        pytest.skip("CUDA extension required for the independent reference")
    args = operands(m)
    expected = _fallback(*args, 1e-5)
    monkeypatch.setattr(cuda._C, "cutlass_int8_dequant", lambda *a: False)
    actual = _fallback(*args, 1e-5)
    for out, ref in zip(actual, expected, strict=True):
        assert torch.equal(out, ref)


def test_strided_large_sequence_memory():
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 native memory regression")
    args = list(operands(87142))
    expected = _op(*args, 1e-5)
    padded = torch.empty((87142, 5377), device="cuda", dtype=torch.int8)
    padded[:, :5376] = args[0]
    before_input = args[0]
    args[0] = padded[:, :5376]
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    actual = _op(*args, 1e-5)
    torch.cuda.synchronize()
    # Native outputs, BF16 scratch and the compact input copy fit in 7 GiB.
    # The old fallback needed several 7.5 GB INT32/FP32 projection planes.
    assert torch.cuda.max_memory_allocated() - before < 7 * 1024**3
    for out, ref in zip(actual, expected, strict=True):
        assert torch.equal(out, ref)
    assert torch.equal(args[0], before_input)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_fallback_with_different_current_device():
    with torch.cuda.device(1):
        args = operands(37)
        expected = _fallback(*args, 1e-5)
    with torch.cuda.device(0):
        actual = _fallback(*args, 1e-5)
        assert torch.cuda.current_device() == 0
    for out, ref in zip(actual, expected, strict=True):
        assert torch.equal(out, ref)
