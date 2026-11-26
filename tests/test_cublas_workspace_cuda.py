from contextlib import nullcontext

import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.backends import cuda
from tests.conftest import requires_cuda_backend

pytestmark = requires_cuda_backend


@pytest.fixture(params=["int8", "nvfp4"])
def gemm(monkeypatch, request):
    torch.manual_seed(694)
    monkeypatch.setattr(cuda, "_DISABLE_CUTLASS_INT8", True)
    monkeypatch.setattr(cuda, "_cublas_workspaces", {})
    ck.set_allocation_context(None)
    if request.param == "int8":
        x = torch.randn(8, 512, dtype=torch.float16, device="cuda")
        weight = torch.randint(-32, 32, (512, 512), dtype=torch.int8, device="cuda")
        scale = torch.full((512,), 0.01, device="cuda")
        def operation():
            return cuda.int8_linear(x, weight, scale)
    else:
        if torch.cuda.get_device_capability()[0] < 10:
            pytest.skip("NVFP4 requires Blackwell")
        x = torch.randn(16, 512, dtype=torch.float16, device="cuda")
        weight = torch.randn(512, 512, dtype=torch.float16, device="cuda")
        scale = torch.ones(1, device="cuda")
        xq, xs = cuda.quantize_nvfp4(x, scale)
        wq, ws = cuda.quantize_nvfp4(weight, scale)
        def operation():
            return cuda.scaled_mm_nvfp4(xq, wq, scale, scale, xs, ws, out_dtype=torch.float16)
    yield operation
    torch.cuda.synchronize()
    ck.set_allocation_context(None)


@pytest.mark.parametrize("prime_eager", [False, True])
def test_ephemeral_workspace_capture_reset_recapture(gemm, prime_eager):
    reference = gemm()
    eager = cuda.get_cublas_workspace() if prime_eager else None
    if not prime_eager:
        cuda._cublas_workspaces.clear()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    ck.set_allocation_context(nullcontext())
    for _ in range(3):
        with torch.cuda.stream(stream):
            gemm()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = gemm()
        with torch.cuda.stream(stream):
            for _ in range(4):
                graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(result, reference)
        assert cuda._cublas_workspaces == ({0: eager} if prime_eager else {})
        graph.reset()
        del result, graph
    ck.set_allocation_context(None)
    if prime_eager:
        assert cuda.get_cublas_workspace() is eager
