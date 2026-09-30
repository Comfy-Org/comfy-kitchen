# SPDX-License-Identifier: Apache-2.0
"""SM120 TMA lifetime, stream and graph regressions.

The A/B benchmark compares separate baseline/PR builds byte for byte.
"""

import pytest
import torch

import comfy_kitchen as ck
from tests.conftest import cuda_backend_available

pytestmark = pytest.mark.skipif(
    not cuda_backend_available(), reason="compiled CUDA backend required"
)


@pytest.mark.parametrize("shape", [(1, 8, 2, 8192, 8193), (2, 4, 2, 8197, 9217)])
@pytest.mark.parametrize("scale", [None, 0.0, -(128**-0.5)])
def test_tma_repeat_stream_graph(shape, scale):
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 required")
    b, h, hk, lq, lk = shape
    torch.manual_seed(42)
    q = torch.randn(b, h, lq, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, hk, lk, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    packed = ck.prequantize_int8_attention(q, k, v, scale=scale)
    expected = ck.int8_attention_from_prequantized(packed)
    for _ in range(3):
        assert torch.equal(ck.int8_attention_from_prequantized(packed), expected)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual = ck.int8_attention_from_prequantized(packed)
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.equal(actual, expected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = ck.int8_attention_from_prequantized(packed)
    graph.replay()
    assert torch.equal(captured, expected)
