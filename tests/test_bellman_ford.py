import pytest
import torch

import comfy_kitchen as ck

from .conftest import requires_cuda_backend


def _reference(tail, head, cost, init, iters):
    """Jacobi sweeps in torch over an arc list, checked every 16; parent ties go to the largest arc id."""
    n = len(init)
    ids = torch.arange(len(tail), device=tail.device)
    dist = init.clone()
    parent = torch.full((n,), -1, dtype=torch.long, device=tail.device)
    steps = max(1, (n - 1).bit_length())
    for it in range(iters):
        cand = dist[tail] + cost
        new = dist.scatter_reduce(0, head, cand, "amin")
        better = new < dist - 1e-12
        arc = torch.full_like(parent, -1).scatter_reduce_(0, head, torch.where(better[head] & (cand == new[head]), ids, -1), "amax")
        parent = torch.where(better, arc, parent)
        dist = new
        if (it + 1) % 16 and it + 1 < iters:
            continue
        if not bool(better.any()):
            break
        root = torch.where(parent >= 0, tail[parent.clamp_min(0)], torch.arange(n, device=tail.device))
        for _ in range(steps):
            root = root[root]
        on = torch.zeros(n, dtype=torch.bool, device=tail.device)
        on[root] = True
        on &= parent >= 0
        if bool(on.any()):
            return dist, parent, on
    return dist, parent, torch.zeros(n, dtype=torch.bool, device=tail.device)


def _graph(n, m, gen, shift):
    tail = torch.randint(0, n, (m,), generator=gen)
    head = torch.randint(0, n, (m,), generator=gen)
    cost = torch.randint(-3, 12, (m,), generator=gen).double() * 0.25 - shift
    cost[torch.rand(m, generator=gen) < 0.05] = torch.inf
    return tail, head, cost


def _run(tail, head, cost, init, iters):
    ref_dist, ref_parent, ref_cycle = _reference(tail, head, cost, init, iters)
    order = torch.argsort(head, stable=True)
    indptr = torch.cat([torch.zeros(1, dtype=torch.long), torch.bincount(head, minlength=len(init)).cumsum(0)])
    dist = init.cuda()
    parent, on_cycle = ck.bellman_ford(indptr.cuda(), tail[order].cuda(), cost[order].cuda(), dist, iters)
    parent = torch.where(parent >= 0, order.cuda()[parent.clamp_min(0)], -1).cpu()
    assert torch.equal(on_cycle.cpu(), ref_cycle)
    assert torch.equal(parent, ref_parent)
    assert torch.equal(dist.cpu(), ref_dist)
    return bool(on_cycle.any())


@requires_cuda_backend
@pytest.mark.parametrize("seed", range(6))
def test_matches_jacobi_reference(seed):
    gen = torch.Generator().manual_seed(seed)
    n = 300
    tail, head, cost = _graph(n, 1500, gen, shift=0.6 if seed % 2 else 0.0)
    init = torch.zeros(n, dtype=torch.float64)
    init[torch.rand(n, generator=gen) < 0.2] = torch.inf
    _run(tail, head, cost, init, n + 1)
    _run(tail, head, cost, init, 7)                               # stops at max_iters


@requires_cuda_backend
def test_finds_cycle_and_converges():
    gen = torch.Generator().manual_seed(3)
    n = 500
    tail, head, cost = _graph(n, 2000, gen, shift=0.0)
    cost = cost.abs()                                             # no negative cycle
    assert not _run(tail, head, cost, torch.zeros(n, dtype=torch.float64), n + 1)
    tail = torch.cat([tail, torch.tensor([10, 11, 12])])
    head = torch.cat([head, torch.tensor([11, 12, 10])])
    cost = torch.cat([cost, torch.tensor([-1.0, -1.0, 0.5], dtype=torch.float64)])
    assert _run(tail, head, cost, torch.zeros(n, dtype=torch.float64), n + 1)


@requires_cuda_backend
def test_rejects_bad_index():
    w = torch.zeros(2, dtype=torch.float64, device="cuda")
    with pytest.raises(ValueError):
        ck.bellman_ford(torch.tensor([0, 1, 2], device="cuda"), torch.tensor([0, 2], device="cuda"), w, w.clone(), 4)
