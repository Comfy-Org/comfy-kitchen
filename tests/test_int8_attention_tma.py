# SPDX-License-Identifier: Apache-2.0
"""SM120 TMA accuracy, lifetime, stream and graph regressions.

The positive-scale A/B benchmark compares separate baseline/PR builds byte for
byte. Nonpositive scales are checked against independent FP32 math SDPA.
"""

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

import comfy_kitchen as ck
from tests.conftest import cuda_backend_available

pytestmark = pytest.mark.skipif(
    not cuda_backend_available(), reason="compiled CUDA backend required"
)


def _nrmse(actual, expected):
    error = (actual.float() - expected.float()).square().mean().sqrt()
    magnitude = expected.float().square().mean().sqrt()
    return (error / magnitude).item()


def _sdpa_reference(q, k, v, scale):
    """Use independent FP32 math SDPA, with bounded score-matrix memory."""
    repeats = q.shape[1] // k.shape[1]
    k = k.float().repeat_interleave(repeats, dim=1)
    v = v.float().repeat_interleave(repeats, dim=1)
    # Force math: optimized SDPA can produce NaNs for zero/negative scales on
    # the tested Torch/CUDA stack. Only queries are chunked; all keys remain
    # visible to each query, so this preserves unmasked attention semantics.
    with sdpa_kernel(SDPBackend.MATH):
        return torch.cat(
            [
                torch.nn.functional.scaled_dot_product_attention(chunk, k, v, scale=scale)
                for chunk in q.float().split(512, dim=2)
            ],
            dim=2,
        )


@pytest.mark.parametrize(
    "shape", [(1, 8, 2, 8192, 8193), (2, 4, 2, 8197, 9217), (1, 4, 4, 8192, 8192)]
)
@pytest.mark.parametrize("scale", [None, 0.0, -(128**-0.5)])
@pytest.mark.parametrize("layout", ["BHSD", "BSHD"])
def test_tma_repeat_stream_graph(shape, scale, layout, record_property):
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 required")
    b, h, hk, lq, lk = shape
    torch.manual_seed(42)
    q = torch.randn(b, h, lq, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, hk, lk, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    packed = ck.prequantize_int8_attention(q, k, v, scale=scale)
    expected = ck.int8_attention_from_prequantized(packed, output_layout=layout)
    reference = _sdpa_reference(q, k, v, scale)
    if layout == "BSHD":
        reference = reference.transpose(1, 2)
    assert torch.isfinite(reference).all()
    assert torch.isfinite(expected).all()
    nrmse = _nrmse(expected, reference)
    record_property("sdpa_nrmse", nrmse)
    assert nrmse < 0.03
    for _ in range(3):
        assert torch.equal(
            ck.int8_attention_from_prequantized(packed, output_layout=layout), expected
        )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual = ck.int8_attention_from_prequantized(packed, output_layout=layout)
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.equal(actual, expected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = ck.int8_attention_from_prequantized(packed, output_layout=layout)
    graph.replay()
    assert torch.equal(captured, expected)
