import gc
import weakref

import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.backends import cuda


class Pause:
    def __enter__(self):
        pytest.fail("workspace allocation must not pause recording")

    def __exit__(self, *args):
        pytest.fail("workspace allocation must not pause recording")


@pytest.mark.parametrize("registered", [False, True])
def test_workspace_is_call_local(monkeypatch, registered):
    template = torch.empty(1, dtype=torch.uint8)
    allocations = []

    def empty(size, dtype, device):
        assert size == 4096 and dtype == torch.uint8 and device == 0
        allocations.append(size)
        return torch.empty_like(template)

    monkeypatch.setattr(cuda, "get_cublas_workspace_size_bytes", lambda: 4096)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch, "empty", empty)
    ck.set_allocation_context(Pause() if registered else None)
    try:
        first = cuda.get_cublas_workspace()
        second = cuda.get_cublas_workspace()
        assert first is not second
        references = [weakref.ref(first), weakref.ref(second)]
        del first, second
        gc.collect()
        assert all(reference() is None for reference in references)
        assert allocations == [4096, 4096]
    finally:
        ck.set_allocation_context(None)
