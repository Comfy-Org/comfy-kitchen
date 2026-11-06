import pytest
import torch

import comfy_kitchen as ck

from .conftest import requires_gpu_backend, requires_hip_backend


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
@requires_gpu_backend
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(("n", "n_edges"), [(1, 0), (10, 4), (100000, 60000), (100000, 300000)])
def test_random_graph(n, n_edges, dtype):
    gen = torch.Generator(device="cuda").manual_seed(n + n_edges)
    edges = torch.randint(0, n, (n_edges, 2), device="cuda", generator=gen, dtype=dtype)
    assert torch.equal(ck.connected_components(edges, n), _reference(edges, n))


@pytest.mark.cuda
@requires_gpu_backend
def test_long_chain():
    # one shuffled path through all nodes: deep trees, many racing hooks
    n = 200000
    order = torch.randperm(n, device="cuda", generator=torch.Generator(device="cuda").manual_seed(0))
    edges = torch.stack([order[:-1], order[1:]], 1)
    assert torch.equal(
        ck.connected_components(edges, n), torch.zeros(n, dtype=torch.int64, device="cuda")
    )


@pytest.mark.cuda
@requires_gpu_backend
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


@pytest.mark.cuda
@requires_gpu_backend
def test_rejects_bad_input():
    edges = torch.tensor([[0, 1], [1, 2]], device="cuda")
    for n in (None, -1, 2.5):
        with pytest.raises(ValueError, match="num_nodes"):
            ck.connected_components(edges, n)
    with pytest.raises(ValueError, match="edges must lie"):
        ck.connected_components(edges, 2)
    with pytest.raises(ValueError, match="edges must lie"):
        ck.connected_components(torch.tensor([[0, -1]], device="cuda"), 3)
    with pytest.raises(ValueError, match="shape"):
        ck.connected_components(torch.zeros(4, 3, dtype=torch.long, device="cuda"), 3)
    assert ck.connected_components(torch.empty(0, 2, dtype=torch.long, device="cuda"), 0).shape == (0,)


@pytest.mark.cuda
@requires_hip_backend
def test_hip_binding_rejects_undersized_and_wrong_dtype_buffers():
    from comfy_kitchen.backends import hip

    w, launch = hip._dl, hip._C.connected_components
    edges = torch.tensor([[0, 1]], dtype=torch.int32).cuda()
    parent = torch.arange(2, dtype=torch.int32).cuda()
    labels = torch.empty(2, dtype=torch.int64).cuda()
    stream = torch.cuda.current_stream().cuda_stream
    launch(w(edges), 1, False, 2, w(parent), w(labels), stream)
    assert labels.tolist() == [0, 0]
    with pytest.raises(RuntimeError, match="labels has 1 elements"):
        launch(w(edges), 1, False, 2, w(parent), w(labels[:1]), stream)
    with pytest.raises(RuntimeError, match="parent has 1 elements"):
        launch(w(edges), 1, False, 2, w(parent[:1]), w(labels), stream)
    with pytest.raises(RuntimeError, match="edges has 2 elements"):
        launch(w(edges), 2, False, 2, w(parent), w(labels), stream)
    with pytest.raises(RuntimeError, match="edges has the wrong dtype"):
        launch(w(edges), 1, True, 2, w(parent), w(labels), stream)
    with pytest.raises(RuntimeError, match="labels has the wrong dtype"):
        launch(w(edges), 1, False, 2, w(parent), w(parent), stream)
