import torch


def _storage_bounds(x: torch.Tensor) -> tuple[int, int]:
    lo = hi = x.storage_offset()
    for size, stride in zip(x.shape, x.stride(), strict=True):
        extent = (size - 1) * stride
        lo += min(0, extent)
        hi += max(0, extent)
    element_size = x.element_size()
    return lo * element_size, (hi + 1) * element_size - 1


def _tensors_overlap(x: torch.Tensor, y: torch.Tensor) -> bool:
    if x.numel() == 0 or y.numel() == 0 or x.device != y.device:
        return False
    if x.untyped_storage().data_ptr() != y.untyped_storage().data_ptr():
        return False
    x0, x1 = _storage_bounds(x)
    y0, y1 = _storage_bounds(y)
    return max(x0, y0) <= min(x1, y1)


def detect_rms_rope_bnhd(x: torch.Tensor, freqs_cis: torch.Tensor) -> bool | None:
    if x.ndim != 4 or freqs_cis.ndim != 6:
        return None

    x_shape = x.shape
    freqs_shape = freqs_cis.shape
    head_dim = x_shape[3]
    if (
        head_dim == 0
        or head_dim % 2 != 0
        or freqs_shape[0] not in (1, x_shape[0])
        or freqs_shape[3:] != (head_dim // 2, 2, 2)
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
