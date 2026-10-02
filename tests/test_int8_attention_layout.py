# SPDX-License-Identifier: Apache-2.0
"""Output projection layout must change storage only, never attention values."""

import pytest
import torch

import comfy_kitchen as ck


@pytest.mark.parametrize("layout", ["bshd", "BSH", "", None])
def test_invalid_output_layout(layout):
    with pytest.raises(ValueError, match="output_layout"):
        ck.int8_attention_from_prequantized(None, output_layout=layout)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "shape,masked",
    [
        ((1, 4, 2, 1, 65, 64), False),
        ((2, 4, 2, 193, 257, 96), False),
        ((2, 4, 2, 257, 1153, 128), True),
        ((1, 4, 2, 129, 513, 256), False),
        ((2, 4, 2, 8193, 9217, 128), False),
    ],
)
def test_output_layout_values_stream_graph(shape, masked, dtype):
    if not ck.sage_attention.is_available():
        pytest.skip("native INT8 attention required")
    b, h, hk, nq, nk, d = shape
    torch.manual_seed(2026)
    q = torch.randn(b, h, nq, d, device="cuda", dtype=dtype)
    k = torch.randn(b, hk, nk, d, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    mask = None
    if masked:
        mask = torch.arange(nk, device="cuda").reshape(1, 1, 1, -1) >= 31
    packed = ck.prequantize_int8_attention(q, k, v, attn_mask=mask)
    expected = ck.int8_attention_from_prequantized(packed).transpose(1, 2).contiguous()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual = ck.int8_attention_from_prequantized(packed, output_layout="BSHD")
    torch.cuda.current_stream().wait_stream(stream)
    assert actual.shape == (b, nq, h, d)
    assert actual.dtype == dtype
    assert actual.is_contiguous()
    assert torch.equal(actual, expected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = ck.int8_attention_from_prequantized(packed, output_layout="BSHD")
    graph.replay()
    assert torch.equal(captured, expected)
