import heapq
import math

import pytest
import torch

import comfy_kitchen as ck

from .conftest import requires_cuda_backend


def _reference(indptr, dst, weight, dist, front, bound, active, blocked=None, stop=None, end=None, closer=None,
               reach=None):
    """Dijkstra per row under the same rules; returns the distances and the final bounds."""
    rows, n = dist.shape
    blocked, stop = ([False] * n if t is None else t.tolist() for t in (blocked, stop))
    end, closer = ([-1] * rows if t is None else t.tolist() for t in (end, closer))
    reach = [math.inf] * rows if reach is None else reach.tolist()
    indptr, dst, weight, active, out, bounds = (t.tolist() for t in (indptr, dst, weight, active, dist, bound))
    starts = [[] for _ in range(rows)]
    for item in front.tolist():
        starts[item // n].append(item % n)
    for r in range(rows):
        d = out[r]
        heap = [(d[v], v) for v in set(starts[r])]
        heapq.heapify(heap)
        while heap:
            dv, v = heapq.heappop(heap)
            if dv > d[v] or dv > reach[r] or v == end[r]:
                continue
            for a in range(indptr[v], indptr[v + 1]):
                to, c = dst[a], dv + weight[a]
                if math.isinf(c) or not active[to]:
                    continue
                if to == closer[r]:
                    bounds[r] = min(bounds[r], dv)
                    continue
                target = to == end[r] or stop[to]
                if blocked[to] and not target:
                    continue
                if c < d[to]:
                    d[to] = c
                    if target:
                        bounds[r] = min(bounds[r], c)
                    else:
                        heapq.heappush(heap, (c, to))
    return torch.tensor(out, dtype=torch.float64), torch.tensor(bounds, dtype=torch.float64)


def _random_graph(n, max_deg, gen, ties=False):
    deg = torch.randint(0, max_deg + 1, (n,), generator=gen)
    indptr = torch.cat([torch.zeros(1, dtype=torch.long), deg.cumsum(0)])
    dst = torch.randint(0, n, (int(deg.sum()),), generator=gen)
    if ties:
        weight = torch.randint(1, 4, dst.shape, generator=gen).double()
    else:
        weight = torch.rand(dst.shape, generator=gen, dtype=torch.float64) * 2 + 0.01
    weight[torch.rand(dst.shape, generator=gen) < 0.05] = torch.inf
    return indptr, dst, weight


def _run(n, rows, gen, sparse=False, ties=False, delta=0.5, max_slots=2 ** 30, use=(), previous=False):
    """Searches from a start node per row (some dense rows from two) with the options named in use, against _reference;
    previous starts dense rows from a finite earlier field."""
    indptr, dst, weight = _random_graph(n, 6, gen, ties)
    starts = torch.randint(0, n, (rows,), generator=gen)
    extra = torch.randint(0, n, (rows,), generator=gen)
    front = torch.arange(rows) * n + starts
    if not sparse:
        front = torch.unique(torch.cat([front, (torch.arange(rows) * n + extra)[::2]]))
    dist = torch.rand(rows, n, generator=gen, dtype=torch.float64) * 4 if previous else torch.full((rows, n), torch.inf, dtype=torch.float64)
    dist.view(-1)[front] = 0
    bound = torch.full((rows,), torch.inf, dtype=torch.float64)
    bound[1::3] = 5.0
    make = {"blocked": lambda: torch.rand(n, generator=gen) < 0.1,
            "stop": lambda: torch.rand(n, generator=gen) < 0.03,
            "end": lambda: torch.where(torch.rand(rows, generator=gen) < 0.7, torch.randint(0, n, (rows,), generator=gen), -1),
            "closer": lambda: starts.clone(),
            "reach": lambda: torch.where(torch.rand(rows, generator=gen) < 0.5, 3.0, torch.inf).double()}
    opts = {"active": torch.rand(n, generator=gen) > 0.1} | {name: make[name]() for name in use}
    ref, ref_bound = _reference(indptr, dst, weight, dist, front, bound, **opts)
    graph, b, opts = (indptr.cuda(), dst.cuda(), weight.cuda()), bound.cuda(), {k: v.cuda() for k, v in opts.items()}
    if sparse:
        keys, vals = ck.shortest_paths_sparse(*graph, starts.cuda(), b, delta, n, max_slots=max_slots, **opts)
        keys, vals = keys.cpu(), vals.cpu()
        assert bool((keys[1:] > keys[:-1]).all())
        got = torch.full((rows * n,), torch.inf, dtype=torch.float64)
        got[keys] = vals
        got = got.view(rows, n)
    else:
        got = dist.cuda()
        ck.shortest_paths(*graph, got, front.cuda(), b, delta, **opts)
        got = got.cpu()
    assert torch.equal(b.cpu(), ref_bound)
    within = ref <= ref_bound[:, None]
    assert torch.equal(got[within], ref[within])
    assert bool((got[~within] > ref_bound[:, None].expand_as(got)[~within]).all())


@requires_cuda_backend
@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("delta", [0.05, 1.0, 1e9])
def test_plain(ties, delta):
    _run(400, 9, torch.Generator().manual_seed(1), ties=ties, delta=delta)


@requires_cuda_backend
@pytest.mark.parametrize("seed", range(4))
def test_options(seed):
    gen = torch.Generator().manual_seed(seed)
    _run(300, 12, gen, ties=seed % 2 == 1, use=("end", "blocked"))
    _run(300, 12, gen, ties=seed % 2 == 1, use=("closer",))
    _run(400, 30, gen, sparse=True, use=("end", "blocked", "reach"))
    _run(400, 30, gen, sparse=True, use=("stop", "reach"))
    _run(400, 30, gen, sparse=True, use=("closer", "reach"))


@requires_cuda_backend
def test_lowers_previous_field():
    """A finite earlier field is only lowered, and nodes are expanded only once lowered."""
    _run(500, 4, torch.Generator().manual_seed(7), delta=0.3, previous=True)


@requires_cuda_backend
def test_rejects_bad_input():
    indptr = torch.tensor([0, 1, 2], device="cuda")
    dst = torch.tensor([1, 0], device="cuda")
    weight = torch.ones(2, dtype=torch.float64, device="cuda")
    dist = torch.tensor([[0.0, torch.inf]], dtype=torch.float64, device="cuda")
    bound = torch.full((1,), torch.inf, dtype=torch.float64, device="cuda")
    with pytest.raises(ValueError):                                   # checked where the search reaches it
        ck.shortest_paths(indptr, torch.tensor([1, 2], device="cuda"), weight, dist.clone(), torch.tensor([0], device="cuda"), bound, 1.0)
    with pytest.raises(ValueError):
        ck.shortest_paths(indptr, dst, weight, dist, torch.tensor([2], device="cuda"), bound, 1.0)
    with pytest.raises(ValueError):
        ck.shortest_paths(indptr, dst, weight, dist, torch.tensor([0], device="cuda"), bound, 0.0)


@requires_cuda_backend
def test_sparse_table_grows():
    """One search over a graph larger than the first table: it runs again with more slots."""
    _run(20000, 1, torch.Generator().manual_seed(5), sparse=True)


@requires_cuda_backend
def test_sparse_search_too_large():
    indptr, dst, weight = _random_graph(20000, 6, torch.Generator().manual_seed(7))
    with pytest.raises(ValueError, match="max_slots"):
        ck.shortest_paths_sparse(indptr.cuda(), dst.cuda(), weight.cuda(), torch.tensor([0], device="cuda"),
                                 torch.full((1,), torch.inf, dtype=torch.float64, device="cuda"), 0.5, 20000, max_slots=4096)


@requires_cuda_backend
def test_sparse_rows_split():
    """Searches that together overflow max_slots run in smaller groups of rows with the same result."""
    _run(2000, 40, torch.Generator().manual_seed(6), sparse=True, max_slots=4096, use=("end", "reach"))
