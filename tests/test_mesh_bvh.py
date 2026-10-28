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


@pytest.mark.cuda
@requires_cuda_backend
def test_negative_max_dist():
    tris = _random_mesh(200, 7)
    points = tris.mean(1)  # on the mesh, so any non-negative cutoff would find them
    dist, closest, face = ck.closest_point_on_mesh(ck.mesh_bvh(tris), points, max_dist=-1.0)
    assert torch.all(dist == -1.0) and torch.equal(closest, points) and torch.all(face == -1)


@pytest.mark.cuda
@requires_cuda_backend
def test_rejects_bad_input():
    tris = _random_mesh(20, 8)
    for bad in (tris[:2, :2], torch.cat([tris, tris[:, :1]], 1)):
        with pytest.raises(ValueError, match="triangles"):
            ck.mesh_bvh(bad)
    bvh = ck.mesh_bvh(tris)
    with pytest.raises(ValueError, match="points"):
        ck.closest_point_on_mesh(bvh, torch.zeros(5, 2, device="cuda"))
    with pytest.raises(ValueError, match="bvh is on"):
        ck.closest_point_on_mesh(bvh._replace(triangles=bvh.triangles.cpu()), torch.zeros(5, 3, device="cuda"))


def _ray_hits(origins, directions, tris, tmin, tmax, cull_backface=False):
    # float64 reference by plane intersection and in-plane edge distances, over broadcast (..., 3) rays and
    # (..., 3, 3) triangles. Returns hit t (inf on miss) and the hit's distance inside the nearest edge (< 0: outside).
    o, d, tr = origins.double(), directions.double(), tris.double()
    a, b, c = tr[..., 0, :], tr[..., 1, :], tr[..., 2, :]
    n = torch.cross(b - a, c - a, dim=-1)
    denom = (n * d).sum(-1)
    t = ((a - o) * n).sum(-1) / denom
    p = o + t[..., None] * d
    nn = n.norm(dim=-1)
    margin = torch.stack(
        [(torch.cross(v1 - v0, p - v0, dim=-1) * n).sum(-1) / (nn * (v1 - v0).norm(dim=-1)) for v0, v1 in ((a, b), (b, c), (c, a))],
        -1,
    ).amin(-1)
    ok = (denom != 0) & (margin >= 0) & (t > tmin) & (t < tmax)
    if cull_backface:
        ok = ok & (denom < 0)
    return torch.where(ok, t, torch.inf), torch.where((t > tmin) & (t < tmax), margin, torch.inf)


def _brute_force_rays(origins, directions, tris, tmin=0.0, tmax=float("inf"), cull_backface=False):
    # nearest t per ray, and whether some in-range triangle passes within 1e-5 of the ray (float32 may go either way)
    best, ambiguous = [], []
    for o, d in zip(origins.split(64), directions.split(64), strict=True):
        t, margin = _ray_hits(o[:, None], d[:, None], tris[None], tmin, tmax, cull_backface)
        best.append(t.min(1).values)
        ambiguous.append((margin.abs() < 1e-5).any(1))
    return torch.cat(best), torch.cat(ambiguous)


def _random_rays(n, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    origins = torch.rand(n, 3, device="cuda", generator=gen) * 1.4 - 0.2
    directions = torch.randn(n, 3, device="cuda", generator=gen)
    # unnormalised: t is in units of the direction's length
    return origins, directions * torch.rand(n, 1, device="cuda", generator=gen).mul(3).exp2()


def _check_rays(origins, directions, tris, t, face, tmin=0.0, tmax=float("inf"), cull_backface=False):
    ref, ambiguous = _brute_force_rays(origins, directions, tris, tmin, tmax, cull_backface)
    clear = ~ambiguous
    assert float(clear.float().mean()) > 0.95
    hit = clear & torch.isfinite(ref)
    miss = clear & ~torch.isfinite(ref)
    assert bool((face[miss] == -1).all()) and bool((t[miss] == tmax).all())
    assert bool((face[hit] >= 0).all())
    torch.testing.assert_close(t[hit], ref[hit].float(), rtol=1e-4, atol=1e-6)
    # the reported triangle is hit at the reported t
    on_face = _ray_hits(origins[hit], directions[hit], tris[face[hit]], tmin, tmax, cull_backface)[0]
    torch.testing.assert_close(t[hit], on_face.float(), rtol=1e-4, atol=1e-6)


@pytest.mark.cuda
@requires_cuda_backend
@pytest.mark.parametrize("n_tris", [1, 2, 7, 1000, 20000])
def test_ray_matches_brute_force(n_tris):
    tris = _random_mesh(n_tris, n_tris)
    origins, directions = _random_rays(3000, 11)
    t, face = ck.closest_hit_on_mesh(ck.mesh_bvh(tris), origins, directions)
    _check_rays(origins, directions, tris, t, face)


@pytest.mark.cuda
@requires_cuda_backend
@pytest.mark.parametrize(("tmin", "tmax"), [(0.05, float("inf")), (0.0, 0.1), (-0.3, 0.07)])
def test_ray_interval(tmin, tmax):
    tris = _random_mesh(2000, 12)
    origins, directions = _random_rays(3000, 13)
    t, face = ck.closest_hit_on_mesh(ck.mesh_bvh(tris), origins, directions, tmin=tmin, tmax=tmax)
    _check_rays(origins, directions, tris, t, face, tmin, tmax)


@pytest.mark.cuda
@requires_cuda_backend
@pytest.mark.parametrize("flip", [False, True])
def test_ray_cull_backface(flip):
    tris = _random_mesh(2000, 14)
    tris = tris.flip(1) if flip else tris  # reversed winding culls the other side
    origins, directions = _random_rays(3000, 15)
    t, face = ck.closest_hit_on_mesh(ck.mesh_bvh(tris), origins, directions, cull_backface=True)
    _check_rays(origins, directions, tris, t, face, cull_backface=True)


@pytest.mark.cuda
@requires_cuda_backend
def test_ray_closed_mesh_has_no_leaks():
    # a cube surface of 6 x 2 x 32^2 triangles: random rays from inside, and rays from the centre through every
    # grid vertex and edge midpoint (axis-parallel ones start on box planes), must all hit
    s = torch.linspace(-1, 1, 33, device="cuda")
    gy, gx = torch.meshgrid(s, s, indexing="ij")
    quad = torch.stack([gx, gy], -1)
    a, b, c, d = quad[:-1, :-1], quad[:-1, 1:], quad[1:, 1:], quad[1:, :-1]
    face2d = torch.cat([torch.stack([a, b, c], -2), torch.stack([a, c, d], -2)]).reshape(-1, 3, 2)
    ones = torch.ones_like(face2d[..., :1])
    faces = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            pts = torch.cat([face2d, ones * sign], -1)
            faces.append(pts.roll(axis + 1, dims=-1))
    tris = torch.cat(faces).contiguous()
    gen = torch.Generator(device="cuda").manual_seed(16)
    targets = torch.cat([tris.reshape(-1, 3), (tris + tris.roll(1, 1)).reshape(-1, 3) / 2])
    directions = torch.cat([torch.randn(200000, 3, device="cuda", generator=gen), targets])
    origins = torch.cat([(torch.rand(200000, 3, device="cuda", generator=gen) - 0.5) * 0.5, torch.zeros_like(targets)])
    t, face = ck.closest_hit_on_mesh(ck.mesh_bvh(tris), origins, directions)
    assert bool((face >= 0).all())
    p = origins + t[:, None] * directions
    torch.testing.assert_close(p.abs().amax(-1), torch.ones_like(t), rtol=1e-5, atol=1e-5)


@pytest.mark.cuda
@requires_cuda_backend
def test_ray_rejects_bad_input():
    tris = _random_mesh(20, 17)
    bvh = ck.mesh_bvh(tris)
    o = torch.zeros(5, 3, device="cuda")
    with pytest.raises(ValueError, match="origins"):
        ck.closest_hit_on_mesh(bvh, o[:, :2], o)
    with pytest.raises(ValueError, match="directions"):
        ck.closest_hit_on_mesh(bvh, o, o[:4])
    with pytest.raises(ValueError, match="bvh is on"):
        ck.closest_hit_on_mesh(bvh._replace(triangles=bvh.triangles.cpu()), o, o)


def _bumpy_sphere(subdivisions, seed):
    # closed, curved, star-shaped mesh: a subdivided icosahedron with jittered vertex radii
    p = (1 + 5**0.5) / 2
    verts = torch.tensor(
        [[-1, p, 0], [1, p, 0], [-1, -p, 0], [1, -p, 0], [0, -1, p], [0, 1, p], [0, -1, -p], [0, 1, -p],
         [p, 0, -1], [p, 0, 1], [-p, 0, -1], [-p, 0, 1]], dtype=torch.float64)
    faces = torch.tensor(
        [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6],
         [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10],
         [8, 6, 7], [9, 8, 1]])
    for _ in range(subdivisions):
        edges = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]).sort(1).values
        uniq, inv = torch.unique(edges, dim=0, return_inverse=True)
        mid = len(verts) + inv.view(3, -1)
        verts = torch.cat([verts, (verts[uniq[:, 0]] + verts[uniq[:, 1]]) / 2])
        a, b, c = faces.T
        faces = torch.cat([torch.stack(f, 1) for f in ((a, mid[0], mid[2]), (b, mid[1], mid[0]), (c, mid[2], mid[1]),
                                                       (mid[0], mid[1], mid[2]))])
    gen = torch.Generator().manual_seed(seed)
    verts = verts / verts.norm(dim=1, keepdim=True) * (1 + 0.1 * torch.rand(len(verts), 1, generator=gen, dtype=torch.float64))
    return verts.float().cuda(), faces.cuda()


@pytest.mark.cuda
@requires_cuda_backend
def test_ray_watertight_at_vertices_and_edges():
    # from inside a closed curved mesh, rays aimed exactly at every vertex and shared-edge midpoint must all hit
    verts, faces = _bumpy_sphere(5, 18)
    tris = verts[faces].contiguous()
    gen = torch.Generator(device="cuda").manual_seed(19)
    edges = torch.unique(torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]).sort(1).values, dim=0)
    targets = torch.cat([verts, (verts[edges[:, 0]] + verts[edges[:, 1]]) / 2])
    bvh = ck.mesh_bvh(tris)
    for origin in (torch.zeros(3, device="cuda"), torch.rand(3, device="cuda", generator=gen) * 0.6 - 0.3):
        origins = origin.expand(targets.shape[0], 3).contiguous()
        _, face = ck.closest_hit_on_mesh(bvh, origins, targets - origins)
        assert int((face < 0).sum()) == 0
