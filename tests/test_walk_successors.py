import pytest
import torch

import comfy_kitchen as ck

from .conftest import requires_cuda_backend


def _reference(succ, starts, stop, active, twin):
    out = []
    for s in starts.tolist():
        seen, path, cur = set(), [s], s
        while True:
            seen.add(cur)
            if twin[cur] >= 0 and active[twin[cur]]:
                seen.add(twin[cur])
            nxt = succ[cur]
            if nxt < 0 or not active[nxt] or nxt in seen:
                path = None
                break
            cur = nxt
            path.append(cur)
            if stop[cur]:
                break
        out.append(path)
    return out


@requires_cuda_backend
@pytest.mark.parametrize("seed", range(4))
def test_matches_reference(seed):
    gen = torch.Generator().manual_seed(seed)
    n = 3000
    succ = torch.randint(0, n, (n,), generator=gen)
    succ[torch.rand(n, generator=gen) < 0.02] = -1
    stop = torch.rand(n, generator=gen) < 0.02
    active = torch.rand(n, generator=gen) > 0.03
    twin = torch.arange(n) ^ 2
    starts = torch.randint(0, n, (200,), generator=gen)
    ref = _reference(succ.tolist(), starts, stop.tolist(), active.tolist(), twin.tolist())
    paths, lengths = ck.walk_successors(succ.cuda(), starts.cuda(), stop.cuda(), active=active.cuda(), twin=twin.cuda())
    paths, lengths = paths.cpu(), lengths.cpu()
    for r, p in enumerate(ref):
        assert (paths[r, :lengths[r]].tolist() if lengths[r] else None) == p
    assert sum(p is not None for p in ref) > 5
    assert int(lengths.max()) == paths.shape[1]

