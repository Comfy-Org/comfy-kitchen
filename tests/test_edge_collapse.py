import math

import pytest
import torch

import comfy_kitchen as ck

from .conftest import requires_cuda_backend


def _sub(u, v):
    return (u[0] - v[0], u[1] - v[1], u[2] - v[2])


def _cross(u, v):
    return (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])


def _dot(u, v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def _reference(vertices, faces, edges, positions, cos_threshold):
    vs = vertices.double().cpu().tolist()
    fs = faces.cpu().tolist()
    fan = [[] for _ in range(len(vs))]
    for i, t in enumerate(fs):
        for x in t:
            fan[x].append(i)
    flips, skinny, link_ok = [], [], []
    for (a, b), p in zip(edges.cpu().tolist(), positions.double().cpu().tolist(), strict=True):
        n_flip, n_moved, sk = 0, 0, 0.0
        for v, other in ((a, b), (b, a)):
            for f in fan[v]:
                t = fs[f]
                if other in t:
                    continue
                old = [vs[x] for x in t]
                new = list(old)
                new[t.index(v)] = p
                n_old = _cross(_sub(old[1], old[0]), _sub(old[2], old[0]))
                n_new = _cross(_sub(new[1], new[0]), _sub(new[2], new[0]))
                len_old, len_new = math.sqrt(_dot(n_old, n_old)), math.sqrt(_dot(n_new, n_new))
                sq_old = sum(_dot(d, d) for d in (_sub(old[1], old[0]), _sub(old[2], old[0]), _sub(old[2], old[1])))
                sq = sum(_dot(d, d) for d in (_sub(new[1], new[0]), _sub(new[2], new[0]), _sub(new[2], new[1])))
                usable = len_old > 1e-6 * sq_old and len_new > 1e-6 * sq
                if usable and _dot(n_old, n_new) < cos_threshold * len_old * len_new:
                    n_flip += 1
                shape = 2 * math.sqrt(3) * len_new / max(sq, 1e-20)
                sk += 1 - min(max(shape, 0.0), 1.0)
                n_moved += 1
        on_edge = sum(b in fs[f] for f in fan[a])
        common = ({x for f in fan[a] for x in fs[f]} & {x for f in fan[b] for x in fs[f]}) - {a, b}
        flips.append(n_flip)
        skinny.append(sk / n_moved if n_moved else 0.0)
        link_ok.append(len(common) <= on_edge)
    return flips, skinny, link_ok


def _grid(n, gen, merges=0):
    # noisy height field; merges weld vertex pairs two apart, which makes the link condition fail
    ij = torch.stack(torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij"), -1).reshape(-1, 2).float()
    vertices = torch.cat([ij / n, 0.02 * torch.randn(n * n, 1, generator=gen)], 1)
    q = (torch.arange(n - 1)[:, None] * n + torch.arange(n - 1)[None, :]).reshape(-1)
    faces = torch.cat([torch.stack([q, q + 1, q + n + 1], 1), torch.stack([q, q + n + 1, q + n], 1)])
    if merges:
        remap = torch.arange(n * n)
        src = torch.randperm(n * n, generator=gen)[:merges]
        dst = (src + 2).clamp(max=n * n - 1)
        remap[src] = dst
        faces = remap[faces]
        faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 2] != faces[:, 0])]
    return vertices, faces


def _edges(faces):
    e = torch.sort(torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), 1).values
    return torch.unique(e, dim=0)


def _check(vertices, faces, edges, positions, cos_threshold=0.0):
    flips, skinny, link_ok = ck.edge_collapse_checks(
        vertices.cuda(), faces.cuda(), edges.cuda(), positions.cuda(), cos_threshold
    )
    ref_flips, ref_skinny, ref_link = _reference(vertices, faces, edges, positions, cos_threshold)
    assert flips.tolist() == ref_flips
    torch.testing.assert_close(skinny.cpu(), torch.tensor(ref_skinny, dtype=torch.float32), atol=1e-5, rtol=1e-4)
    assert link_ok.tolist() == ref_link
    return flips, link_ok


@pytest.mark.cuda
@requires_cuda_backend
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("merges", [0, 40])
@pytest.mark.parametrize("scale", [1.0, 1e-4])
def test_grid(index_dtype, merges, scale):
    gen = torch.Generator().manual_seed(merges)
    vertices, faces = _grid(40, gen, merges)
    edges = _edges(faces)
    # offset midpoints so some collapses fold faces over, at any scale
    positions = (vertices[edges].mean(1) + 0.03 * torch.randn(edges.shape[0], 3, generator=gen)) * scale
    vertices = vertices * scale
    flips, link_ok = _check(vertices, faces.to(index_dtype), edges.to(index_dtype), positions)
    assert (flips > 0).any() and (flips == 0).any()
    assert (~link_ok).any() == (merges > 0)


@pytest.mark.cuda
@requires_cuda_backend
def test_high_valence_fan():
    # cone apex with 200 faces: no valence cap
    k = 200
    angle = torch.arange(k) * (2 * math.pi / k)
    vertices = torch.cat([torch.zeros(1, 3), torch.stack([angle.cos(), angle.sin(), torch.full((k,), -0.5)], 1)])
    rim = torch.arange(1, k + 1)
    faces = torch.stack([torch.zeros(k, dtype=torch.long), rim, rim % k + 1], 1)
    edges = _edges(faces)
    positions = vertices[edges].mean(1)
    _check(vertices, faces, edges, positions, cos_threshold=0.5)


@pytest.mark.cuda
@requires_cuda_backend
def test_no_edges():
    vertices, faces = _grid(4, torch.Generator().manual_seed(0))
    flips, skinny, link_ok = ck.edge_collapse_checks(
        vertices.cuda(), faces.cuda(), torch.empty(0, 2, dtype=torch.long, device="cuda"),
        torch.empty(0, 3, device="cuda"),
    )
    assert flips.shape == skinny.shape == link_ok.shape == (0,)


@pytest.mark.cuda
@requires_cuda_backend
def test_deterministic():
    # the fans are filled with atomics, in varying order; the outputs must not depend on it
    gen = torch.Generator().manual_seed(3)
    vertices, faces = _grid(120, gen, 200)
    edges = _edges(faces)
    positions = vertices[edges].mean(1) + 0.03 * torch.randn(edges.shape[0], 3, generator=gen)
    args = (vertices.cuda(), faces.cuda(), edges.cuda(), positions.cuda(), 0.5)
    first = ck.edge_collapse_checks(*args)
    for _ in range(5):
        for a, b in zip(first, ck.edge_collapse_checks(*args), strict=True):
            assert torch.equal(a, b)
