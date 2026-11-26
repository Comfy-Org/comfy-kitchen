"""Regression coverage for native GroupNorm workspace bounds."""

import pytest
import torch

from comfy_kitchen.backends import cuda
from tests.conftest import requires_cuda_backend

pytestmark = requires_cuda_backend


@pytest.mark.parametrize("fused", [False, True])
def test_group_norm_native_rejects_undersized_workspace(fused):
    """Binding validation must agree with the kernel's adaptive chunk count."""
    b, c, t, h, w, groups = 1, 128, 1, 16, 16, 32
    x = torch.randn(b, c, t, h, w, device="cuda", dtype=torch.float16)
    x = x.contiguous(memory_format=torch.channels_last_3d)
    chunks = cuda._group_norm_chunks(b, c, t, h, w) if hasattr(cuda, "_group_norm_chunks") else 1
    required = 2 * b * t * (chunks * c + (groups if fused else 0))
    # Backing storage contains the actual writes even if validation is broken.
    backing = torch.full((required + 16,), 19.0, device="cuda", dtype=torch.float32)
    workspace = backing[: required - 1]
    wrap = cuda._wrap_for_dlpack
    stream = torch.cuda.current_stream().cuda_stream
    with pytest.raises(ValueError, match="workspace too small"):
        if fused:
            weight = torch.ones(c, device="cuda", dtype=x.dtype)
            bias = torch.zeros_like(weight)
            out = torch.empty_like(x)
            empty = torch.empty(0, device="cuda", dtype=torch.float32)
            cuda._C.group_norm_silu_pad3d(
                wrap(x),
                wrap(weight),
                wrap(bias),
                wrap(out),
                wrap(workspace),
                wrap(empty),
                b,
                c,
                t,
                h,
                w,
                groups,
                1e-6,
                0,
                0,
                0,
                0,
                0,
                False,
                False,
                w * c,
                cuda.DTYPE_TO_CODE[x.dtype],
                stream,
            )
        else:
            stats = torch.empty(b * t, groups, 2, device="cuda", dtype=torch.float32)
            cuda._C.group_norm_stats3d(
                wrap(x),
                wrap(workspace),
                wrap(stats),
                b,
                c,
                t,
                h,
                w,
                groups,
                1e-6,
                w * c,
                cuda.DTYPE_TO_CODE[x.dtype],
                stream,
            )
        torch.cuda.synchronize()
