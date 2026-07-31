import torch


def trim_rope_freqs(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Trim excess sequence frequencies to match apply_rope's input."""
    if (
        x.ndim > 2
        and freqs_cis.ndim > 2
        and x.shape[2] != 1
        and freqs_cis.shape[2] > x.shape[2]
    ):
        return freqs_cis[:, :, : x.shape[2]]
    return freqs_cis


def _storage_bounds(x: torch.Tensor) -> tuple[int, int]:
    lo = hi = x.storage_offset()
    for size, stride in zip(x.shape, x.stride(), strict=True):
        extent = (size - 1) * stride
        lo += min(0, extent)
        hi += max(0, extent)
    element_size = x.element_size()
    return lo * element_size, (hi + 1) * element_size - 1


def _interleaved_disjoint(x: torch.Tensor, y: torch.Tensor) -> bool:
    """True for same-layout views whose rows sit side by side within a shared
    period — the packed-QKV case, where q and k are strided slices of one fused
    projection: bounds overlap but the element sets never do."""
    if x.shape != y.shape or x.stride() != y.stride():
        return False
    strides = x.stride()
    if not strides or strides[-1] != 1:
        return False
    # fold the contiguous suffix into one row extent
    extent = 1
    i = len(strides)
    while i > 0 and strides[i - 1] == extent:
        extent *= x.shape[i - 1]
        i -= 1
    outer = [s for size, s in zip(x.shape[:i], strides[:i], strict=True) if size > 1]
    if not outer:
        return False
    period = min(outer)
    if period <= 0 or any(s % period != 0 for s in outer):
        # index combinations of non-multiple strides can land inside the
        # collision window; only the hierarchical layout is provably disjoint
        return False
    delta = abs(x.storage_offset() - y.storage_offset())
    return delta >= extent and delta + extent <= period


def _tensors_overlap(x: torch.Tensor, y: torch.Tensor) -> bool:
    if x.numel() == 0 or y.numel() == 0 or x.device != y.device:
        return False
    if x.untyped_storage().data_ptr() != y.untyped_storage().data_ptr():
        return False
    x0, x1 = _storage_bounds(x)
    y0, y1 = _storage_bounds(y)
    if max(x0, y0) > min(x1, y1):
        return False
    return not _interleaved_disjoint(x, y)


def detect_rms_rope_bnhd(
    x: torch.Tensor, freqs_cis: torch.Tensor, rot_dim: int = 0
) -> bool | None:
    if x.ndim != 4 or freqs_cis.ndim != 6:
        return None

    x_shape = x.shape
    freqs_shape = freqs_cis.shape
    head_dim = x_shape[3]
    rot = rot_dim if rot_dim else head_dim
    if (
        head_dim == 0
        or head_dim % 2 != 0
        or rot % 2 != 0
        or rot > head_dim
        or freqs_shape[3:] != (rot // 2, 2, 2)
        or freqs_shape[0] not in (1, x_shape[0])
    ):
        return None
    if freqs_shape[1] == 1 and freqs_shape[2] in (1, x_shape[2]):
        return False
    if freqs_shape[1] in (1, x_shape[1]) and freqs_shape[2] == 1:
        return True
    return None


def check_rope_inplace(
    *xs: torch.Tensor, readonly: tuple[torch.Tensor, ...] = ()
) -> None:
    for x in (*xs, *readonly):
        if x.requires_grad:
            raise RuntimeError(
                "in-place RoPE operations are inference-only and do not support autograd"
            )

    for x in xs:
        required = 1
        dimensions = sorted(
            (abs(stride), size)
            for size, stride in zip(x.shape, x.stride(), strict=True)
            if size > 1
        )
        for stride, size in dimensions:
            if stride < required:
                raise ValueError(
                    "in-place RoPE requires views without internal overlap"
                )
            required = stride * size

    if len(xs) == 2 and _tensors_overlap(xs[0], xs[1]):
        raise ValueError(
            "paired in-place RoPE requires non-overlapping input storage"
        )
    for x in xs:
        for source in readonly:
            if _tensors_overlap(x, source):
                raise ValueError(
                    "in-place RoPE inputs must not overlap frequencies or scales"
                )
