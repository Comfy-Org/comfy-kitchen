import pytest
import torch

import comfy_kitchen as ck

from .conftest import requires_cuda_backend


def _reference(nbr, cap, s_cap, t_cap, relabel_every=64):
    """Synchronous push-relabel in torch; returns the nodes that cannot reach the sink once no node can push."""
    n = nbr.shape[0]
    device = nbr.device
    big = n + 2
    valid = nbr >= 0
    j = nbr.clamp(min=0)
    rev = (nbr[j] == torch.arange(n, device=device)[:, None, None]).int().argmax(-1)
    tol = 1e-12 * float(cap.max()) if cap.numel() else 0.0
    r = torch.where(valid, cap, torch.zeros_like(cap))
    rt = t_cap.clone()
    e = s_cap.clone()

    def global_relabel():
        h = torch.full((n,), big, dtype=torch.long, device=device)
        frontier = rt > tol
        h[frontier] = 1
        d = 1
        while bool(frontier.any()):
            d += 1
            frontier = ((r > tol) & frontier[j]).any(1) & (h == big)
            h[frontier] = d
        return h

    h = global_relabel()
    it = 0
    while bool(((e > tol) & (h < big)).any()):
        amt = torch.where((e > tol) & (h == 1) & (rt > tol), torch.minimum(e, rt), torch.zeros_like(e))
        e = e - amt
        rt = rt - amt
        for k in range(nbr.shape[1]):
            jk = j[:, k]
            adm = (e > tol) & (h < big) & (r[:, k] > tol) & (h == h[jk] + 1)
            idx = adm.nonzero().squeeze(1)
            amt = torch.minimum(e[idx], r[idx, k])
            e[idx] -= amt
            r[idx, k] -= amt
            r[jk[idx], rev[idx, k]] += amt
            e.index_add_(0, jk[idx], amt)
        stuck = (e > tol) & (h < big)
        if bool(stuck.any()):
            hmin = torch.where(r > tol, h[j], big).amin(1)
            hmin = torch.where(rt > tol, torch.zeros_like(hmin), hmin)
            h = torch.where(stuck, torch.maximum(h, (hmin + 1).clamp(max=big)), h)
        it += 1
        if it % relabel_every == 0:
            h = global_relabel()
    return global_relabel() >= big


def _grid_graph(nx, ny, nz, device, gen):
    """6-connected grid as (N, 6) slots with random symmetric capacities."""
    idx = torch.arange(nx * ny * nz, device=device).view(nx, ny, nz)
    nbr = torch.full((nx * ny * nz, 6), -1, dtype=torch.int32, device=device)
    cap = torch.zeros((nx * ny * nz, 6), device=device)
    pad = torch.nn.functional.pad(idx, (1, 1, 1, 1, 1, 1), value=-1)
    shifts = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    for k, (dx, dy, dz) in enumerate(shifts):
        nbr[:, k] = pad[1 + dx:1 + dx + nx, 1 + dy:1 + dy + ny, 1 + dz:1 + dz + nz].reshape(-1)
    # symmetric capacities: draw per node, take the larger index's value for both directions
    w = torch.rand(nx * ny * nz, 6, device=device, generator=gen) + 0.05
    for k in range(6):
        other = nbr[:, k].long()
        ok = other >= 0
        pair = torch.where(torch.arange(nx * ny * nz, device=device) < other, w[:, k], w[other.clamp(min=0), k ^ 1])
        cap[:, k] = torch.where(ok, pair, torch.zeros_like(pair))
    return nbr, cap


def _check(nbr, cap, s_cap, t_cap):
    out = ck.min_cut(nbr, cap, s_cap, t_cap)
    ref = _reference(nbr, cap, s_cap, t_cap)
    assert out.dtype == torch.bool and out.shape == ref.shape
    assert torch.equal(out, ref)


@pytest.mark.cuda
@requires_cuda_backend
@pytest.mark.parametrize("shape", [(1, 1, 1), (4, 4, 4), (24, 24, 24), (40, 40, 40)])
def test_grid(shape):
    gen = torch.Generator(device="cuda").manual_seed(sum(shape))
    nbr, cap = _grid_graph(*shape, "cuda", gen)
    n = nbr.shape[0]
    # a source blob on one face and sinks on the opposite face, plus scattered terminals
    s_cap = torch.zeros(n, device="cuda")
    t_cap = torch.zeros(n, device="cuda")
    s_cap[: n // max(shape[0], 1)] = torch.rand(n // max(shape[0], 1), device="cuda", generator=gen) * 4
    t_cap[-(n // max(shape[0], 1)):] = torch.rand(n // max(shape[0], 1), device="cuda", generator=gen) * 4
    scatter = torch.rand(n, device="cuda", generator=gen)
    s_cap[scatter < 0.02] += 2.0
    t_cap[scatter > 0.98] += 2.0
    _check(nbr, cap, s_cap, t_cap)


@pytest.mark.cuda
@requires_cuda_backend
def test_terminals_everywhere():
    # every node touches both terminals: the cut is decided per node by the terminal capacities
    gen = torch.Generator(device="cuda").manual_seed(7)
    nbr, cap = _grid_graph(12, 12, 12, "cuda", gen)
    n = nbr.shape[0]
    s_cap = torch.rand(n, device="cuda", generator=gen) * 0.1
    t_cap = torch.rand(n, device="cuda", generator=gen) * 0.1
    _check(nbr, cap, s_cap, t_cap)


@pytest.mark.cuda
@requires_cuda_backend
def test_no_terminals_and_isolated():
    gen = torch.Generator(device="cuda").manual_seed(3)
    nbr, cap = _grid_graph(6, 6, 6, "cuda", gen)
    n = nbr.shape[0]
    zero = torch.zeros(n, device="cuda")
    assert bool(ck.min_cut(nbr, cap, zero, zero).all())   # nothing reaches the sink
    # isolated nodes keep only their terminal decision
    nbr_iso = torch.full((5, 4), -1, dtype=torch.int32, device="cuda")
    cap_iso = torch.zeros((5, 4), device="cuda")
    s = torch.tensor([1.0, 0.0, 2.0, 0.0, 0.5], device="cuda")
    t = torch.tensor([0.0, 1.0, 0.5, 0.0, 2.0], device="cuda")
    _check(nbr_iso, cap_iso, s, t)


@pytest.mark.cuda
@requires_cuda_backend
def test_int64_neighbours_and_repeatable():
    gen = torch.Generator(device="cuda").manual_seed(11)
    nbr, cap = _grid_graph(20, 20, 20, "cuda", gen)
    n = nbr.shape[0]
    s_cap = torch.rand(n, device="cuda", generator=gen) * (torch.rand(n, device="cuda", generator=gen) < 0.05)
    t_cap = torch.rand(n, device="cuda", generator=gen) * (torch.rand(n, device="cuda", generator=gen) < 0.05)
    first = ck.min_cut(nbr.long(), cap, s_cap, t_cap)
    assert torch.equal(first, _reference(nbr, cap, s_cap, t_cap))
    for _ in range(3):
        assert torch.equal(ck.min_cut(nbr, cap, s_cap, t_cap), first)


@pytest.mark.cuda
@requires_cuda_backend
def test_no_neighbour_slots():
    # terminal edges only: a node stays on the source side unless its sink capacity is larger
    gen = torch.Generator(device="cuda").manual_seed(5)
    s_cap = torch.rand(1000, device="cuda", generator=gen)
    t_cap = torch.rand(1000, device="cuda", generator=gen)
    out = ck.min_cut(torch.empty(1000, 0, dtype=torch.int32, device="cuda"), torch.empty(1000, 0, device="cuda"), s_cap, t_cap)
    assert torch.equal(out, s_cap >= t_cap)


@pytest.mark.cuda
@requires_cuda_backend
def test_rejects_bad_input():
    gen = torch.Generator(device="cuda").manual_seed(9)
    nbr, cap = _grid_graph(4, 4, 4, "cuda", gen)
    n = nbr.shape[0]
    zero = torch.zeros(n, device="cuda")
    with pytest.raises(ValueError, match="128"):
        ck.min_cut(torch.full((n, 129), -1, dtype=torch.int32, device="cuda"), torch.zeros(n, 129, device="cuda"), zero, zero)
    with pytest.raises(ValueError, match="cap"):
        ck.min_cut(nbr, cap[:, :5], zero, zero)
    with pytest.raises(ValueError, match="t_cap"):
        ck.min_cut(nbr, cap, zero, zero[:-1])
    bad = nbr.clone()
    bad[0, 0] = n
    with pytest.raises(ValueError, match="nbr must lie"):
        ck.min_cut(bad, cap, zero, zero)
