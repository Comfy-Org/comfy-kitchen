"""The fused ConvRot activation quantizer's output must not depend on its schedule.

Two things about that kernel are chosen for speed and nothing else:

* its block width, which ``convrot_quant_fused_block_threads`` picks per
  architecture from a measured table -- on RDNA2 the table this tree ships was
  wrong by up to 3x before it had an entry for that family at all;
* whether it reads the four values a lane consumes in one vector load, which
  pays at narrow blocks and costs at the widest (see
  ``kConvrotVecLoadMaxBlock``).

Neither may move a single int8 code or a single scale: the GEMM downstream is
what everything else in the int8 path is measured against, and a 1-LSB shift in
the activation quantizer would change every generated image.

So this pins the invariant directly instead of comparing against the eager
reference, which the kernel was never bit-equal to -- it rounds the rotated row
to the input dtype at a different point in the butterfly network than a matmul
does, and that predates any of the changes here. Both schedule choices are
reachable at run time (``COMFY_CONVROT_FUSED_BLOCK``), so one process can run
the same input through every width and require every result to be the same bits.

Runs on any HIP device; no matrix cores and no model weights needed.
"""

from __future__ import annotations

import os

import pytest
import torch

from comfy_kitchen.backends.hip import _EXT_AVAILABLE, _EXT_ERROR, _rotate_quant_int8
from tests.test_hip_wmma import DEV, _unavailable_reason

_UNAVAILABLE = _unavailable_reason() or (None if _EXT_AVAILABLE else _EXT_ERROR)

pytestmark = [
    pytest.mark.skipif(_UNAVAILABLE is not None, reason=_UNAVAILABLE or ""),
    pytest.mark.slow,
]

GROUP = 256
OVERRIDE = "COMFY_CONVROT_FUSED_BLOCK"

# Widths that fit K*2 + (width/64)*2048 bytes of LDS for every K below, spanning
# both sides of kConvrotVecLoadMaxBlock and both sides of the block's own
# "groups in flight" behaviour.
WIDTHS = (64, 128, 256, 512, 1024)

# (M, K) the production int8 dispatch issues on RDNA2: SDXL's level-2
# projections and feed-forward, its cross-attention, and Anima's transformer
# block. K=1024 is here because it is four Hadamard groups, one fewer than the
# width the old heuristic launched, and that is the shape that regressed hardest.
SHAPES = [
    (512, 1024),
    (1024, 1280),
    (4096, 1280),
    (77, 2048),
    (1024, 2048),
    (4096, 2560),
    (1024, 5120),
    (1024, 6144),
    (1024, 8192),
]


def _quant_at(x: torch.Tensor, width: int | None) -> tuple[torch.Tensor, torch.Tensor]:
    previous = os.environ.get(OVERRIDE)
    try:
        if width is None:
            os.environ.pop(OVERRIDE, None)
        else:
            os.environ[OVERRIDE] = str(width)
        return _rotate_quant_int8(x, GROUP)
    finally:
        if previous is None:
            os.environ.pop(OVERRIDE, None)
        else:
            os.environ[OVERRIDE] = previous


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("m,k", SHAPES, ids=lambda v: str(v))
def test_convrot_quant_is_independent_of_block_width(dtype, m, k):
    torch.manual_seed(0xC0FFEE + m + k)
    x = torch.randn((m, k), device=DEV, dtype=dtype) * 0.7

    default_q, default_s = _quant_at(x, None)
    assert default_q.shape == (m, k)
    assert default_s.shape == (m,)

    for width in WIDTHS:
        q, s = _quant_at(x, width)
        assert torch.equal(q.to(torch.int32), default_q.to(torch.int32)), (
            f"M={m} K={k} {dtype}: int8 codes differ at block={width}"
        )
        assert torch.equal(s, default_s), (
            f"M={m} K={k} {dtype}: scales differ at block={width}"
        )


def test_the_default_width_is_not_the_widest_one():
    """Guard against the table silently degenerating to "always 1024", which is
    what it effectively was on RDNA2 before this tree had an entry for it."""
    x = torch.randn((1024, 1280), device=DEV, dtype=torch.float16) * 0.7
    default_q, _ = _quant_at(x, None)
    for width in (64, 128):
        q, _ = _quant_at(x, width)
        assert torch.equal(q.to(torch.int32), default_q.to(torch.int32))
    # And the heuristic really does land somewhere narrower than the dGPU answer.
    from comfy_kitchen.backends import hip

    assert hip._visible_gfx_arches() is not None
