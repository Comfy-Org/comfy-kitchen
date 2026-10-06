import pytest
import torch

import comfy_kitchen as ck

from .conftest import requires_cuda_backend


def _closest_on_triangles(p, tris):
    # float64 reference, valid for degenerate triangles: nearest edge point or inside plane projection
    p, tris = p.double(), tris.double()
    v = [tris[..., k, :] for k in range(3)]
    best, best_d = None, None
    for k in range(3):
        a, ab = v[k], v[(k + 1) % 3] - v[k]
        s = (((p - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-300)).clamp(0, 1)
        c = a + s[..., None] * ab
        d = ((p - c) ** 2).sum(-1)
        if best is None:
            best, best_d = c, d
        else:
            take = d < best_d
            best, best_d = torch.where(take[..., None], c, best), torch.where(take, d, best_d)
    n = torch.cross(v[1] - v[0], v[2] - v[0], dim=-1)
    n2 = (n * n).sum(-1)
    q = p - (((p - v[0]) * n).sum(-1) / n2.clamp_min(1e-300))[..., None] * n
    inside = n2 > 0
    for k in range(3):
        inside = inside & ((torch.cross(v[(k + 1) % 3] - v[k], q - v[k], dim=-1) * n).sum(-1) >= 0)
    return torch.where(inside[..., None], q, best)


def _brute_force(points, tris):
    dist = []
    for p in points.split(64):
        dist.append(
            (_closest_on_triangles(p[:, None], tris[None]) - p[:, None].double()).norm(dim=-1).min(1).values
        )
    return torch.cat(dist).float()


def _random_mesh(n_tris, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    centre = torch.rand(n_tris, 1, 3, device="cuda", generator=gen)
    return centre + (torch.rand(n_tris, 3, 3, device="cuda", generator=gen) - 0.5) * 0.1


@pytest.mark.cuda
@requires_cuda_backend
@pytest.mark.parametrize("n_tris", [1, 2, 7, 1000, 20000])
def test_matches_brute_force(n_tris):
    tris = _random_mesh(n_tris, n_tris)
    gen = torch.Generator(device="cuda").manual_seed(1)
    points = torch.rand(2000, 3, device="cuda", generator=gen) * 1.4 - 0.2
    dist, closest, face = ck.closest_point_on_mesh(ck.mesh_bvh(tris), points)
    ref_dist = _brute_force(points, tris)
    torch.testing.assert_close(dist, ref_dist, rtol=1e-5, atol=1e-6)
    # the reported point is on the reported triangle and at the reported distance
    on_face = _closest_on_triangles(points, tris[face]).float()
    torch.testing.assert_close(closest, on_face, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close((closest - points).norm(dim=-1), dist, rtol=1e-5, atol=1e-6)


@pytest.mark.cuda
@requires_cuda_backend
def test_large_triangles():
    # a huge triangle among many small ones: nearest-centroid shortcuts get it wrong
    tris = torch.cat(
        [
            _random_mesh(5000, 3) * 0.2,
            torch.tensor([[[0, 0, 1], [1, 0, 1], [0, 1, 1]]], device="cuda"),
        ]
    )
    gen = torch.Generator(device="cuda").manual_seed(2)
    points = torch.rand(3000, 3, device="cuda", generator=gen) * torch.tensor([1.0, 1.0, 1.2], device="cuda")
    dist, _, _ = ck.closest_point_on_mesh(ck.mesh_bvh(tris.float()), points)
    torch.testing.assert_close(dist, _brute_force(points, tris.float()), rtol=1e-5, atol=1e-6)


@pytest.mark.cuda
@requires_cuda_backend
def test_degenerate_triangles():
    # collinear triangles and slivers down to 1e-6 wide, where float32 region tests cancel out
    gen = torch.Generator(device="cuda").manual_seed(4)
    a = torch.rand(3000, 3, device="cuda", generator=gen)
    b = torch.rand(3000, 3, device="cuda", generator=gen)
    t = torch.rand(3000, 1, device="cuda", generator=gen)
    side = torch.randn(3000, 3, device="cuda", generator=gen)
    width = torch.tensor([0.0, 1e-6, 1e-4, 1e-2], device="cuda").repeat(750)[:, None]
    tris = torch.stack([a, b, a + t * (b - a) + width * side], 1)
    points = torch.rand(3000, 3, device="cuda", generator=gen)
    dist, _, _ = ck.closest_point_on_mesh(ck.mesh_bvh(tris), points)
    torch.testing.assert_close(dist, _brute_force(points, tris), rtol=1e-5, atol=1e-6)


@pytest.mark.cuda
@requires_cuda_backend
@pytest.mark.parametrize("max_dist", [0.05, 0.0517, 0.0731, 0.1234])
def test_max_dist(max_dist):
    tris = _random_mesh(500, 5)
    gen = torch.Generator(device="cuda").manual_seed(6)
    points = torch.rand(4000, 3, device="cuda", generator=gen) * 3 - 1
    full, _, _ = ck.closest_point_on_mesh(ck.mesh_bvh(tris), points)
    dist, closest, face = ck.closest_point_on_mesh(ck.mesh_bvh(tris), points, max_dist=max_dist)
    # the kernel compares squared distances, `full` is a rounded square root: skip points within rounding of the limit
    clear = (full - max_dist).abs() > 1e-5
    miss = (full >= max_dist) & clear
    hit = (full < max_dist) & clear
    assert not bool(((face < 0) & hit).any()) and not bool(((face >= 0) & miss).any())
    assert torch.all(dist[miss] == torch.tensor(max_dist)) and torch.equal(closest[miss], points[miss])
    torch.testing.assert_close(dist[hit], full[hit])
