# SPDX-License-Identifier: Apache-2.0
"""Changing V's reduction block size must preserve codes, scales and padding."""

import pytest
import torch

from comfy_kitchen.backends import cuda

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0)
    or not cuda._EXT_AVAILABLE,
    reason="requires the SM120 CUDA extension",
)


def quantize(v):
    b, h, n, d = v.shape
    padded = (n + 127) // 128 * 128
    out = torch.empty((b, h, d, padded), dtype=torch.int8, device=v.device)
    scale = torch.empty((b, h, d), dtype=torch.float32, device=v.device)
    cuda._C._quant_v_int8(
        *(cuda._wrap_for_dlpack(t) for t in (v, out, scale)),
        padded, {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}[v.dtype],
        torch.cuda.current_stream(v.device).cuda_stream,
    )
    return out, scale


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("length", [12288, 14851, 32700])
@pytest.mark.parametrize("layout", ["BHND", "BNHD", "QKV"])
def test_large_v_matches_independent_heads(dtype, length, layout):
    torch.manual_seed(53)
    if layout == "BHND":
        v = torch.randn(1, 56, length, 128, device="cuda", dtype=dtype)
    elif layout == "BNHD":
        v = torch.randn(1, length, 56, 128, device="cuda", dtype=dtype).transpose(1, 2)
    else:
        v = torch.randn(1, length, 3, 56, 128, device="cuda", dtype=dtype)[:, :, 2].transpose(1, 2)
    actual, scales = quantize(v)
    # Each head has its own scale. One-head calls retain the old schedule.
    for head in range(v.shape[1]):
        expected, expected_scale = quantize(v[:, head:head + 1])
        assert torch.equal(actual[:, head:head + 1], expected)
        assert torch.equal(scales[:, head:head + 1].view(torch.int32), expected_scale.view(torch.int32))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_large_v_extremes_and_graph(dtype):
    v = torch.zeros((1, 56, 12289, 128), device="cuda", dtype=dtype)
    v[:, :, 0] = -0.0
    v[:, :, 17, 0] = torch.finfo(dtype).max
    v[:, :, 1025, 1] = -torch.finfo(dtype).max
    v[:, :, 8191, 2] = float("inf")
    v[:, :, 8192, 3] = float("nan")
    v[:, :, 12288, 4] = torch.finfo(dtype).tiny
    expected, scales = quantize(v)
    for head in range(v.shape[1]):
        part, part_scales = quantize(v[:, head:head + 1])
        assert torch.equal(expected[:, head:head + 1], part)
        assert torch.equal(scales[:, head:head + 1].view(torch.int32), part_scales.view(torch.int32))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        quantize(v)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual, actual_scale = quantize(v)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(actual, expected)
    assert torch.equal(actual_scale.view(torch.int32), scales.view(torch.int32))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("batch,heads,length,dim", [
    (1, 32, 16383, 128), (1, 32, 16384, 128), (2, 16, 16384, 128),
    (1, 56, 12287, 128), (1, 56, 12288, 128), (1, 32, 16384, 64),
])
def test_v_schedule_boundaries(dtype, batch, heads, length, dim):
    torch.manual_seed(59)
    v = torch.randn(batch, heads, length, dim, device="cuda", dtype=dtype)
    actual, scales = quantize(v)
    for head in range(heads):
        expected, expected_scales = quantize(v[:, head:head + 1])
        assert torch.equal(actual[:, head:head + 1], expected)
        assert torch.equal(scales[:, head:head + 1].view(torch.int32), expected_scales.view(torch.int32))


def test_v_schedule_after_device_switch():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two visible CUDA devices")
    for device in [0, 1, 0]:
        with torch.cuda.device(device):
            v = torch.randn(1, 56, 12288, 128, device="cuda", dtype=torch.bfloat16)
            actual, scales = quantize(v)
            expected, expected_scale = quantize(v[:, :1])
            assert torch.equal(actual[:, :1], expected)
            assert torch.equal(scales[:, :1].view(torch.int32), expected_scale.view(torch.int32))
