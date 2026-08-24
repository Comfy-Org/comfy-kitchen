from __future__ import annotations

import torch

from .backends import cuda as _cuda_backend


def is_available(device: torch.device | int | None = None) -> bool:
    """Return whether the fused gated-delta decode kernel is available."""
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None):
        return False
    if (
        not _cuda_backend._EXT_AVAILABLE
        or _cuda_backend._C is None
        or not hasattr(_cuda_backend._C, "gated_delta_decode")
    ):
        return False
    return True


def gated_delta_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    state: torch.Tensor,
    snapshots: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run S decode steps of the gated delta rule in one kernel.

    q/k: [B, S, H, DK] fp32 (normalized, q pre-scaled); v: [B, S, H, DV] fp32;
    beta/g: [B, S, H] fp32; state: [B, H, DK, DV] fp32, updated in place;
    snapshots: [>=S-1, B, H, DK, DV] fp32, written for steps 0..S-2.
    Returns out [B, S, H, DV] fp32. Falls back by raising if the launch is
    rejected (unsupported dims), so callers can keep an eager path.
    """
    if not is_available(q.device):
        raise RuntimeError("gated_delta_decode requires the CUDA extension")
    out = torch.empty_like(v)
    wrap = _cuda_backend._wrap_for_dlpack
    ok = _cuda_backend._C.gated_delta_decode(
        wrap(q), wrap(k), wrap(v), wrap(beta), wrap(g), wrap(state), wrap(out),
        wrap(snapshots) if snapshots is not None else None,
        torch.cuda.current_stream(q.device).cuda_stream,
    )
    if not ok:
        raise RuntimeError("gated_delta_decode launch rejected")
    return out
