# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.indexed_norm import _fallback


def operands(device, m=131, k=5376, groups=5, dtype=torch.float32):
    gen = torch.Generator(device=device).manual_seed(2341)
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16, generator=gen)
    w = torch.randn(k, device=device, dtype=torch.bfloat16, generator=gen)
    shift = torch.randn(groups, k, device=device, dtype=dtype, generator=gen)
    scale = torch.randn(groups, k, device=device, dtype=dtype, generator=gen)
    rows = torch.randint(-1, groups + 1, (m,), device=device, dtype=torch.int32, generator=gen)
    if m > 2:
        rows[:2] = torch.tensor([-2147483648, 2147483647], device=device, dtype=torch.int32)
    return x, w, shift, scale, rows


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("m,k,groups", [(0, 5376, 3), (1, 5376, 0), (131, 5376, 5), (7, 256, 3)])
def test_reference(device, dtype, m, k, groups):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    args = operands(device, m, k, groups, dtype)
    before = [x.clone() for x in args]
    actual = ck.indexed_norm_convrot(*args)
    if m:
        expected = _fallback(*args, 1e-5)
        for out, ref in zip(actual, expected, strict=True):
            assert torch.equal(out, ref)
    assert actual[0].shape == (m, k)
    assert actual[1].shape == (m, 1)
    assert all(x.is_contiguous() for x in actual)
    for tensor, saved in zip(args, before, strict=True):
        assert torch.equal(tensor, saved)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("magnitude", [0.0, 1e-20, 1.0, 1e10])
@pytest.mark.parametrize("strided", [False, True])
def test_cuda_extremes_and_strides(magnitude, strided):
    args = list(operands("cuda", m=33))
    args[0] *= magnitude
    if strided:
        for i in (0, 2, 3):
            original = args[i]
            padded = torch.empty(
                (original.shape[0], original.shape[1] + 4),
                dtype=original.dtype,
                device=original.device,
            )
            args[i] = padded[:, : original.shape[1]]
            args[i].copy_(original)
    actual = ck.indexed_norm_convrot(*args, eps=1e-5)
    expected = _fallback(*args, 1e-5)
    for out, ref in zip(actual, expected, strict=True):
        assert torch.equal(out, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_stream_graph():
    from comfy_kitchen.backends import cuda

    if torch.cuda.get_device_capability() != (12, 0) or cuda._C is None:
        pytest.skip("SM120 extension required")
    args = operands("cuda")
    expected = _fallback(*args, 1e-5)
    q, qs = (torch.empty_like(x) for x in expected)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        assert cuda._C.indexed_norm_convrot(
            *(cuda._wrap_for_dlpack(x) for x in (*args, q, qs)),
            1e-5,
            stream.cuda_stream,
        )
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.equal(q, expected[0])
    assert torch.equal(qs, expected[1])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = ck.indexed_norm_convrot(*args)
    graph.replay()
    for out, ref in zip(actual, expected, strict=True):
        assert torch.equal(out, ref)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_inductor(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    args = operands(device, m=7)
    fn = torch.compile(ck.indexed_norm_convrot, fullgraph=True)
    for out, ref in zip(fn(*args), ck.indexed_norm_convrot(*args), strict=True):
        assert torch.equal(out, ref)


@pytest.mark.parametrize("eps", [0.0, -1.0, float("inf"), float("nan"), 1e-40])
def test_invalid_epsilon(eps):
    with pytest.raises(ValueError, match="eps"):
        ck.indexed_norm_convrot(*operands("cpu", m=1), eps=eps)


def test_noncontiguous_cpu():
    args = list(operands("cpu", m=7, k=256))
    args[0] = args[0].T.contiguous().T
    for out, ref in zip(ck.indexed_norm_convrot(*args), _fallback(*args, 1e-5), strict=True):
        assert out.is_contiguous()
        assert torch.equal(out, ref)
