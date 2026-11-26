import gc
import weakref

import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen import allocation
from comfy_kitchen.backends import cuda


@pytest.fixture
def allocations(monkeypatch):
    sizes = []

    def empty(size, dtype, device):
        assert size == 4096 and dtype == torch.uint8 and device in (0, 1)
        sizes.append((device, size))
        return torch.empty_like(template)

    template = torch.empty(1, dtype=torch.uint8)
    monkeypatch.setattr(cuda, "_cublas_workspaces", {})
    monkeypatch.setattr(allocation, "_context", None)
    monkeypatch.setattr(cuda, "get_cublas_workspace_size_bytes", lambda: 4096)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch, "empty", empty)
    return sizes


class Pause:
    def __enter__(self):
        pytest.fail("workspace allocation must not pause recording")

    def __exit__(self, *args):
        pytest.fail("workspace allocation must not pause recording")


def test_eager_workspace_is_cached_per_device(allocations, monkeypatch):
    first = cuda.get_cublas_workspace()
    for _ in range(10):
        assert cuda.get_cublas_workspace() is first
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    second = cuda.get_cublas_workspace()
    assert second is not first
    assert cuda.get_cublas_workspace() is second
    assert allocations == [(0, 4096), (1, 4096)]


@pytest.mark.parametrize("prime_eager", [False, True])
def test_context_makes_workspace_ephemeral_without_pausing(allocations, prime_eager):
    eager = cuda.get_cublas_workspace() if prime_eager else None
    ck.set_allocation_context(Pause())
    assert allocation.has_allocation_context()
    assert isinstance(allocation.allocation_context(), Pause)
    first = cuda.get_cublas_workspace()
    second = cuda.get_cublas_workspace()
    assert first is not second and first is not eager and second is not eager
    references = [weakref.ref(first), weakref.ref(second)]
    del first, second
    gc.collect()
    assert all(reference() is None for reference in references)
    assert cuda._cublas_workspaces == ({0: eager} if prime_eager else {})
    ck.set_allocation_context(None)
    assert not allocation.has_allocation_context()
    with allocation.allocation_context():
        pass
    restored = cuda.get_cublas_workspace()
    if prime_eager:
        assert restored is eager
    assert cuda.get_cublas_workspace() is restored
    assert len(allocations) == 3
