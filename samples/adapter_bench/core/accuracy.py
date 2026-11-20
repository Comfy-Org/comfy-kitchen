"""Error against an fp64 reference: relative L2 and max error in ULP of the dtype."""

import torch


def rel_err(got: torch.Tensor, ref64: torch.Tensor) -> float:
    """||got - ref|| / ||ref||, in fp64."""
    ref = ref64.double()
    return ((got.double() - ref).norm() / ref.norm().clamp_min(1e-300)).item()


def ulp_err(
    got: torch.Tensor, ref64: torch.Tensor, dtype: torch.dtype, mode: str = "rms"
) -> float:
    """Max error in units of last place of ``dtype``; about 0.5 is a correctly rounded result.

    rms scales by the reference's RMS magnitude (GEMMs and reductions, where a
    near-zero output is cancellation); elementwise by each element's own
    magnitude (pointwise ops).
    """
    eps = torch.finfo(dtype).eps
    tiny = torch.finfo(dtype).tiny
    g, r = got.detach().double(), ref64.detach().double()
    if mode == "rms":
        scale = r.pow(2).mean().sqrt().clamp_min(tiny)
    else:
        scale = r.abs().clamp_min(tiny)
    return ((g - r).abs() / (scale * eps)).max().item()
