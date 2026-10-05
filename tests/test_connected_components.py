import pytest
import torch

import comfy_kitchen as ck

from .conftest import requires_cuda_backend


def _reference(edges, n):
    # min-label propagation to the smallest index in each component
    label = torch.arange(n, device=edges.device)
    a, b = edges[:, 0].long(), edges[:, 1].long()
    while True:
        low = torch.minimum(label[a], label[b])
        new = label.clone()
        new.scatter_reduce_(0, a, low, "amin")
        new.scatter_reduce_(0, b, low, "amin")
        new = new[new]
        if torch.equal(new, label):
            return label
        label = new


@pytest.mark.cuda
@requires_cuda_backend
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(("n", "n_edges"), [(1, 0), (10, 4), (100000, 60000), (100000, 300000)])
def test_random_graph(n, n_edges, dtype):
    gen = torch.Generator(device="cuda").manual_seed(n + n_edges)
    edges = torch.randint(0, n, (n_edges, 2), device="cuda", generator=gen, dtype=dtype)
    assert torch.equal(ck.connected_components(edges, n), _reference(edges, n))


@pytest.mark.cuda
@requires_cuda_backend
def test_long_chain():
    # one shuffled path through all nodes: deep trees, many racing hooks
    n = 200000
    order = torch.randperm(n, device="cuda")
    edges = torch.stack([order[:-1], order[1:]], 1)
    assert torch.equal(
        ck.connected_components(edges, n), torch.zeros(n, dtype=torch.int64, device="cuda")
    )


@pytest.mark.cuda
@requires_cuda_backend
def test_star_and_isolated():
    n = 50000
    hub = 31337
    leaves = torch.arange(n, device="cuda")[::2]
    edges = torch.stack([leaves, torch.full_like(leaves, hub)], 1)
    labels = ck.connected_components(edges, n)
    expected = torch.arange(n, device="cuda")
    expected[::2] = 0  # node 0 is a star leaf
    expected[hub] = 0
    assert torch.equal(labels, expected)
