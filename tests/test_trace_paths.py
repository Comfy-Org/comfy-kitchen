import pytest
import torch

import comfy_kitchen as ck

from .conftest import requires_cuda_backend


def _reference(src, dst, w, dist, starts, ends, reach, stop):
    """Walk back per row in Python: smallest tail distance, then largest tail id."""
    n = dist.shape[1]
    into = [[] for _ in range(n)]
    for a, (u, v) in enumerate(zip(src.tolist(), dst.tolist(), strict=True)):
        into[v].append((u, w[a].item()))
    out = []
    for r, (s, e) in enumerate(zip(starts.tolist(), ends.tolist(), strict=True)):
        if e < 0:
            out.append(None)
            continue
        path, cur = [e], e
        while cur != s:
            dv = dist[r, cur].item()
            if dv == float("inf"):
                path = None
                break
            best, pick = float("inf"), -1
            for u, wa in into[cur]:
                du = dist[r, u].item()
                if du + wa != dv or du > reach[r] or (stop[u] and u != s):
                    continue
                if du < best or (du == best and u > pick):
                    best, pick = du, u
            if pick < 0:
                path = None
                break
            cur = pick
            path.append(cur)
        out.append(path[::-1] if path else None)
    return out


@requires_cuda_backend
@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("sparse", [False, True])
def test_matches_reference(seed, sparse):
    gen = torch.Generator().manual_seed(seed)
    n, m, rows = 400, 2400, 24
    src = torch.randint(0, n, (m,), generator=gen)
    dst = torch.randint(0, n, (m,), generator=gen)
    w = (torch.randint(1, 4, (m,), generator=gen).double() if seed % 2 else torch.rand(m, generator=gen, dtype=torch.float64) + 0.05)
    order = torch.argsort(src, stable=True)
    src, dst, w = src[order], dst[order], w[order]
    out_ptr = torch.cat([torch.zeros(1, dtype=torch.long), torch.bincount(src, minlength=n).cumsum(0)])
    starts = torch.randint(0, n, (rows,), generator=gen)
    d = torch.full((rows, n), torch.inf, dtype=torch.float64, device="cuda")
    d[torch.arange(rows), starts] = 0
    ck.shortest_paths(out_ptr.cuda(), dst.cuda(), w.cuda(), d, (torch.arange(rows) * n + starts).cuda(),
                      torch.full((rows,), torch.inf, dtype=torch.float64, device="cuda"), 0.5)
    d = d.cpu()
    ends = torch.randint(0, n, (rows,), generator=gen)
    ends[::5] = -1
    reach = torch.where(torch.rand(rows, generator=gen) < 0.3, 2.0, torch.inf).double()
    stop = torch.rand(n, generator=gen) < 0.05
    by_head = torch.argsort(dst, stable=True)
    in_ptr = torch.cat([torch.zeros(1, dtype=torch.long), torch.bincount(dst, minlength=n).cumsum(0)])
    ref = _reference(src, dst, w, d, starts, ends, reach.tolist(), stop.tolist())
    if sparse:
        keys = torch.nonzero(torch.isfinite(d.view(-1)))[:, 0]
        field = (keys.cuda(), d.view(-1)[keys].cuda())
    else:
        field = d.cuda()
    paths, lengths = ck.trace_paths(in_ptr.cuda(), src[by_head].cuda(), w[by_head].cuda(), starts.cuda(), ends.cuda(),
                                    field, n, reach=reach.cuda(), stop=stop.cuda())
    paths, lengths = paths.cpu(), lengths.cpu()
    at = torch.cumsum(lengths, 0) - lengths
    for r, p in enumerate(ref):
        got = paths[at[r]:at[r] + lengths[r]].tolist() if lengths[r] else None
        assert got == p, r
    assert sum(p is not None for p in ref) > len(ref) // 2


@requires_cuda_backend
def test_start_is_end():
    in_ptr = torch.tensor([0, 1, 2], device="cuda")
    tail = torch.tensor([1, 0], device="cuda")
    w = torch.ones(2, dtype=torch.float64, device="cuda")
    d = torch.tensor([[0.0, 1.0]], dtype=torch.float64, device="cuda")
    paths, lengths = ck.trace_paths(in_ptr, tail, w, torch.tensor([0], device="cuda"), torch.tensor([0], device="cuda"), d, 2)
    assert lengths.tolist() == [1] and paths.tolist() == [0]
    paths, lengths = ck.trace_paths(in_ptr, tail, w, torch.tensor([0], device="cuda"), torch.tensor([1], device="cuda"), d, 2)
    assert lengths.tolist() == [2] and paths.tolist() == [0, 1]
