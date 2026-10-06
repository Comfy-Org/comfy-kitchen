import pytest
import torch

import comfy_kitchen as ck

from .conftest import requires_cuda_backend

_FACES = torch.tensor([[1, 3, 2], [0, 2, 3], [0, 3, 1], [0, 1, 2]])


def _signed_volumes(verts, tets):
    a, b, c, d = (verts[tets[:, i]] for i in range(4))
    return ((b - a) * torch.cross(c - a, d - a, dim=-1)).sum(-1) / 6


def _check_valid(points, verts, tets, nbr):
    n = points.shape[0]
    tets, nbr = tets.long(), nbr.long()
    assert torch.allclose(verts[:n], points.double(), rtol=0, atol=1e-9 * max(1.0, float(points.abs().max())))
    vol = _signed_volumes(verts, tets)
    assert (vol > 0).all(), "inverted or flat tet"

    # the tets tile the bounding tet exactly
    c = verts[n:]
    box = abs(float(((c[1] - c[0]) * torch.cross(c[2] - c[0], c[3] - c[0], dim=-1)).sum())) / 6
    assert float(vol.sum()) == pytest.approx(box, rel=1e-9)

    # neighbours point back across the same 3 vertices; only the 4 outer faces have none
    t, k = (nbr >= 0).nonzero(as_tuple=True)
    other = nbr[t, k]
    assert ((nbr[other] == t[:, None]).sum(1) == 1).all()
    face = torch.sort(
        tets[t][torch.arange(t.numel(), device=t.device)[:, None], _FACES.to(t.device)[k]], 1
    ).values
    assert (tets[other][:, :, None] == face[:, None, :]).any(1).all()
    assert int((nbr < 0).sum()) == 4

    used = torch.zeros(n + 4, dtype=torch.bool, device=tets.device)
    used[tets.reshape(-1)] = True
    assert used.all()


def _non_delaunay_fraction(verts, tets, nbr):
    tets, nbr = tets.long(), nbr.long()
    t, k = (nbr >= 0).nonzero(as_tuple=True)
    other = nbr[t, k]
    face = tets[t][torch.arange(t.numel(), device=t.device)[:, None], _FACES.to(t.device)[k]]
    apex = tets[other][~(tets[other][:, :, None] == face[:, None, :]).any(2)]
    p = verts[tets[t]] - verts[apex][:, None, :]
    lifted = torch.cat([p, (p * p).sum(-1, keepdim=True)], -1)
    # > 0: apex inside the circumsphere
    inside = -torch.linalg.det(lifted)
    scale = p.abs().amax((1, 2)) ** 5
    return float((inside > 1e-9 * scale).float().mean())


@pytest.mark.cuda
@requires_cuda_backend
@pytest.mark.parametrize("n", [1, 5, 2000, 50000])
def test_uniform(n):
    gen = torch.Generator(device="cuda").manual_seed(n)
    points = torch.rand(n, 3, device="cuda", generator=gen)
    verts, tets, nbr = ck.delaunay3d(points)
    _check_valid(points, verts, tets, nbr)
    assert _non_delaunay_fraction(verts, tets, nbr) < 1e-3


@pytest.mark.cuda
@requires_cuda_backend
def test_matches_scipy():
    scipy_spatial = pytest.importorskip("scipy.spatial")
    gen = torch.Generator(device="cuda").manual_seed(0)
    points = torch.rand(3000, 3, device="cuda", generator=gen, dtype=torch.float64)
    tets = ck.delaunay3d(points)[1]
    ours = tets.long()[~(tets >= points.shape[0]).any(1)]
    ours = {tuple(sorted(t)) for t in ours.tolist()}
    ref = {
        tuple(sorted(t)) for t in scipy_spatial.Delaunay(points.cpu().numpy()).simplices.tolist()
    }
    # the bounding corners only change tets along the convex hull
    assert len(ours & ref) / len(ref) > 0.99


@pytest.mark.cuda
@requires_cuda_backend
def test_degenerate_input_stays_valid():
    # regular grid: cospherical cells and many coplanar points
    axis = torch.linspace(0, 1, 12, device="cuda")
    points = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
    verts, tets, nbr = ck.delaunay3d(points)
    _check_valid(points, verts, tets, nbr)


@pytest.mark.cuda
@requires_cuda_backend
@pytest.mark.parametrize("offset", [1e6, 1e7])
def test_far_from_origin(offset):
    # the joggle must exceed rounding at the coordinates' magnitude, not just their spread
    axis = torch.linspace(0, 1, 12, device="cuda", dtype=torch.float64)
    points = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3) + offset
    verts, tets, nbr = ck.delaunay3d(points)
    _check_valid(points, verts, tets, nbr)


@pytest.mark.cuda
@requires_cuda_backend
def test_empty_input():
    with pytest.raises(ValueError):
        ck.delaunay3d(torch.empty((0, 3), device="cuda"))


@pytest.mark.cuda
@requires_cuda_backend
def test_shell_points():
    # two close concentric spheres, like an offset surface sampled from both sides
    gen = torch.Generator(device="cuda").manual_seed(1)
    s = torch.nn.functional.normalize(torch.randn(20000, 3, device="cuda", generator=gen), dim=-1)
    points = torch.cat([s[:10000], s[10000:] * 0.99])
    verts, tets, nbr = ck.delaunay3d(points)
    _check_valid(points, verts, tets, nbr)
    assert _non_delaunay_fraction(verts, tets, nbr) < 2e-2


@pytest.mark.cuda
@requires_cuda_backend
def test_rejects_bad_input():
    with pytest.raises(ValueError, match="points"):
        ck.delaunay3d(torch.rand(10, 2, device="cuda", dtype=torch.float64))
