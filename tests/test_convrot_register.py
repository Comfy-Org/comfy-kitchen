# SPDX-License-Identifier: Apache-2.0
"""Regression tests against the retained generic CUDA ConvRot implementation."""

import pytest
import torch

from tests.conftest import cuda_backend_available

pytestmark = pytest.mark.skipif(
    not cuda_backend_available(), reason="compiled CUDA backend required"
)


def _compare(x):
    from comfy_kitchen.backends.cuda import quantize_int8_rowwise_convrot64

    # A two-byte storage offset keeps the generic scalar-load implementation.
    # Both inputs are contiguous and hold exactly the same BF16 bit patterns.
    storage = torch.empty(x.numel() + 1, dtype=x.dtype, device=x.device)
    generic = storage[1:].view_as(x)
    generic.copy_(x)
    expected = quantize_int8_rowwise_convrot64(generic, 256)
    actual = quantize_int8_rowwise_convrot64(x, 256)
    for a, b in zip(actual, expected, strict=True):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))


@pytest.mark.parametrize("k", [5376, 7168])
@pytest.mark.parametrize("m", [1, 17, 1024, 65536])
def test_register_matches_generic(m, k):
    torch.manual_seed(42)
    _compare(torch.randn(m, k, device="cuda", dtype=torch.bfloat16))


@pytest.mark.parametrize("k", [5376, 7168])
@pytest.mark.parametrize("kind", ["zeros", "negative_zero", "tiny", "huge", "nan", "inf", "bits"])
def test_register_special_values(k, kind):
    x = torch.randn(17, k, device="cuda", dtype=torch.bfloat16)
    if kind == "zeros":
        x.zero_()
    elif kind == "negative_zero":
        x.fill_(-0.0)
    elif kind == "tiny":
        x.mul_(1e-35)
    elif kind == "huge":
        x.mul_(1e38)
    elif kind == "nan":
        x[:, ::7] = float("nan")
    elif kind == "inf":
        x[:, ::7] = float("inf")
        x[:, 1::7] = -float("inf")
    else:
        x = torch.randint(-32768, 32768, (17, k), device="cuda", dtype=torch.int16).view(
            torch.bfloat16
        )
    _compare(x)


def test_register_nondefault_stream_and_graph():
    from comfy_kitchen.backends.cuda import quantize_int8_rowwise_convrot64

    x = torch.randn(37, 7168, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _compare(x)
        expected = quantize_int8_rowwise_convrot64(x, 256)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = quantize_int8_rowwise_convrot64(x, 256)
    graph.replay()
    for a, b in zip(actual, expected, strict=True):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
