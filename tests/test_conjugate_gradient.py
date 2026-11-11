import pytest
import torch

import comfy_kitchen as ck

from .conftest import requires_cuda_backend


def _laplacian(n, m, gen):
    """A random weighted graph Laplacian plus a positive diagonal, as CSR (and dense)."""
    a = torch.randint(0, n, (m,), generator=gen)
    b = torch.randint(0, n, (m,), generator=gen)
    keep = a != b
    a, b = a[keep], b[keep]
    w = torch.rand(len(a), generator=gen, dtype=torch.float64) + 0.1
    dense = torch.zeros(n, n, dtype=torch.float64)
    dense.index_put_((a, b), -w, accumulate=True)
    dense.index_put_((b, a), -w, accumulate=True)
    dense += torch.diag(-dense.sum(1) + torch.rand(n, generator=gen, dtype=torch.float64) * 0.5 + 0.01)
    nz = torch.nonzero(dense)
    indptr = torch.cat([torch.zeros(1, dtype=torch.long), torch.bincount(nz[:, 0], minlength=n).cumsum(0)])
    return indptr, nz[:, 1], dense[nz[:, 0], nz[:, 1]], dense


@requires_cuda_backend
@pytest.mark.parametrize("k", [1, 2])
def test_solves_and_repeats(k):
    gen = torch.Generator().manual_seed(k)
    n = 3000
    indptr, cols, vals, dense = _laplacian(n, 12000, gen)
    b = torch.randn((n, k) if k > 1 else (n,), generator=gen, dtype=torch.float64)
    ref = torch.linalg.solve(dense, b)
    runs = []
    for _ in range(2):
        x = torch.zeros_like(b).cuda()
        ck.conjugate_gradient(indptr.cuda(), cols.cuda(), vals.cuda(), b.cuda(), x, 5000, 1e-12)
        runs.append(x.cpu())
    assert torch.equal(runs[0], runs[1])
    assert torch.allclose(runs[0], ref, rtol=1e-8, atol=1e-8)


@requires_cuda_backend
def test_starts_from_x_and_stops_at_max_iters():
    gen = torch.Generator().manual_seed(4)
    indptr, cols, vals, dense = _laplacian(500, 2000, gen)
    b = torch.randn(500, generator=gen, dtype=torch.float64)
    x = torch.linalg.solve(dense, b).cuda()
    start = x.clone()
    ck.conjugate_gradient(indptr.cuda(), cols.cuda(), vals.cuda(), b.cuda(), x, 100, 1e-6)
    assert torch.equal(x, start)                                  # already within tol: no step taken
    y = torch.zeros(500, dtype=torch.float64, device="cuda")
    ck.conjugate_gradient(indptr.cuda(), cols.cuda(), vals.cuda(), b.cuda(), y, 3, 1e-12)
    assert not torch.allclose(y.cpu(), x.cpu(), atol=1e-6)


@requires_cuda_backend
def test_rejects_bad_index():
    indptr = torch.tensor([0, 1, 2], device="cuda")
    vals = torch.ones(2, dtype=torch.float64, device="cuda")
    b = torch.ones(2, dtype=torch.float64, device="cuda")
    with pytest.raises(ValueError):
        ck.conjugate_gradient(indptr, torch.tensor([0, 2], device="cuda"), vals, b, torch.zeros_like(b), 10, 1e-9)
