# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.indexed_gate import _fallback


def operands(device, m=37, n=256, k=128, groups=3):
    gen = torch.Generator(device=device).manual_seed(2026)
    return (
        torch.randint(-128, 128, (m, k), device=device, dtype=torch.int8, generator=gen),
        torch.randint(-128, 128, (n, k), device=device, dtype=torch.int8, generator=gen),
        torch.rand(m, device=device, generator=gen) * 0.01,
        torch.rand(n, device=device, generator=gen) * 0.01,
        torch.randn(groups, n, device=device, dtype=torch.bfloat16, generator=gen),
        torch.randint(-1, groups + 1, (m,), device=device, dtype=torch.int32, generator=gen),
        torch.randn(m, n, device=device, dtype=torch.bfloat16, generator=gen),
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "shape", [(37, 256, 128), (128, 512, 256), (7, 13, 17), (0, 16, 32), (4, 16, 0)]
)
def test_indexed_gate(device, shape):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    args = operands(device, *shape)
    out = ck.int8_gemm_indexed_gate(*args)
    expected = _fallback(*args)
    assert torch.equal(out, expected)
    # Independent integer oracle verifies GEMM and BF16 rounding separately.
    a, b, xs, ws, gate, indices, residual = [x.cpu() for x in args]
    acc = (a.to(torch.int64) @ b.to(torch.int64).T).float()
    branch = ((acc * xs[:, None]) * ws + 0.0).to(torch.bfloat16)
    selected = torch.zeros_like(residual)
    for row, index in enumerate(indices):
        if 0 <= index < len(gate):
            selected[row] = gate[index]
    oracle = torch.addcmul(residual, branch, selected)
    assert torch.equal(out.cpu(), oracle)


@pytest.mark.parametrize("groups", [0, 1, 5])
def test_invalid_indices_and_empty_gate(groups):
    args = operands("cpu", groups=groups)
    assert torch.equal(ck.int8_gemm_indexed_gate(*args), _fallback(*args))


def test_noncontiguous():
    args = list(operands("cpu"))
    for i in (0, 1, 4, 6):
        args[i] = args[i].T.contiguous().T
    assert torch.equal(ck.int8_gemm_indexed_gate(*args), _fallback(*args))


@pytest.mark.parametrize(
    "index,change,match",
    [
        (0, lambda x: x.float(), "int8"),
        (2, lambda x: x.half(), "float32"),
        (4, lambda x: x.float(), "bfloat16"),
        (5, lambda x: x.long(), "int32"),
        (6, lambda x: x[:, :1], "residual"),
    ],
)
def test_validation(index, change, match):
    args = list(operands("cpu"))
    args[index] = change(args[index])
    with pytest.raises(ValueError, match=match):
        ck.int8_gemm_indexed_gate(*args)


def test_compile():
    args = operands("cpu")
    compiled = torch.compile(ck.int8_gemm_indexed_gate, backend="eager", fullgraph=True)
    assert torch.equal(compiled(*args), ck.int8_gemm_indexed_gate(*args))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_inductor_noncontiguous_residual(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    args = list(operands(device))
    args[6] = args[6].T.contiguous().T
    assert not args[6].is_contiguous()
    before = [x.clone() for x in args]

    def consume(*inputs):
        result = ck.int8_gemm_indexed_gate(*inputs)
        return result, result.float().sum(dim=1)

    # Exercise Inductor's extern-output stride guard and a stride-sensitive
    # consumer; backend="eager" cannot detect a fake/real layout mismatch.
    compiled = torch.compile(consume, fullgraph=True)
    out, reduced = compiled(*args)
    expected = _fallback(*args)
    eager = ck.int8_gemm_indexed_gate(*args)
    assert eager.is_contiguous()
    assert out.is_contiguous()
    assert torch.equal(out, expected)
    torch.testing.assert_close(reduced, expected.float().sum(dim=1))
    for original, saved in zip(args, before, strict=True):
        assert torch.equal(original, saved)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_stream_graph_and_fused_path():
    from comfy_kitchen.backends import cuda

    if cuda._C is None or not hasattr(cuda._C, "cutlass_int8_indexed_gate"):
        pytest.skip("new CUDA extension required")
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 specialization")
    args = operands("cuda", 4096, 512, 256)
    expected = _fallback(*args)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        raw = torch.empty_like(args[-1])
        used = cuda._C.cutlass_int8_indexed_gate(
            *(cuda._wrap_for_dlpack(x) for x in (*args, raw)), stream.cuda_stream
        )
        assert used
        out = ck.int8_gemm_indexed_gate(*args)
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.equal(raw, expected)
    assert torch.equal(out, expected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = ck.int8_gemm_indexed_gate(*args)
    graph.replay()
    assert torch.equal(captured, expected)
