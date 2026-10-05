"""The ConvRot activation quantizer's output must not depend on its schedule.

Two things about the fused kernel are chosen for speed and nothing else:

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

The int4 activation quantizer adds a third schedule choice, and the largest one:
whether it runs ``convrot_quant_fused_kernel`` at all or the legacy
``convrot_quant_kernel``. ``COMFY_CONVROT_PACK4_FUSED=0`` selects the legacy one,
so the same input can be put through both and required to come out identical.
That was long assumed impossible -- the fused kernel scales the rotation by 0.5
per butterfly stage where the legacy one multiplies by ``rsqrt(G)`` once at the
end -- and the assumption is what kept every W4A4 activation on the slower
kernel. These tests are the measurement that overturned it.

One caution the int4 tests carry with them: a knob that fails to switch paths
produces a perfect bit-exactness result, because both arms then run the same
kernel. Only ``test_convrot_quant_int4_knob_reaches_a_different_kernel`` (marked
``performance``) can see that, by timing.

Runs on any HIP device; no matrix cores and no model weights needed.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

from comfy_kitchen.backends.hip import (
    _C,
    _EXT_AVAILABLE,
    _EXT_ERROR,
    _dl,
    _rotate_quant_int8,
    _stream,
)
from tests.test_hip_wmma import DEV, _unavailable_reason

_UNAVAILABLE = _unavailable_reason() or (None if _EXT_AVAILABLE else _EXT_ERROR)

pytestmark = [
    pytest.mark.skipif(_UNAVAILABLE is not None, reason=_UNAVAILABLE or ""),
    pytest.mark.slow,
]


def test_the_loaded_extension_is_the_one_this_checkout_built():
    """Record which ``_C.abi3.pyd`` the bit-exactness checks below actually ran.

    The pairing matters because ``test_hip_build_provenance.py`` can only report the
    problem, not prevent it: when a stale ``.pyd`` shadows the installed one, the
    HIP module here fails to import its extension and every test in *this* file skips
    silently. That skip is indistinguishable from "this box has no HIP device", which
    is how the stale-binary case reads in CI. Printing the resolved path turns the
    skip into something a reader can check.

    See that file for how the shadowing happens and what it cost.
    """
    import comfy_kitchen.backends.hip as hip_pkg

    ext = getattr(hip_pkg._C, "__file__", None)
    print(f"\n  comfy_kitchen: {hip_pkg.__file__}")
    print(f"  extension:     {ext}")
    print(f"  _EXT_AVAILABLE: {hip_pkg._EXT_AVAILABLE}")
    assert hip_pkg._EXT_AVAILABLE, "the HIP extension failed to load; see test_hip_build_provenance.py"

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


PACK4_KNOB = "COMFY_CONVROT_PACK4_FUSED"
PACK4_GROUP = 256
# The int4 activation quantizer runs on the fused kernel when the rotated row
# fits in LDS, and on convrot_quant_kernel otherwise. Those are different
# kernels writing the same bytes, so which one ran is a schedule choice like
# any other -- but a schedule choice that is currently only reachable through a
# measurement knob, and an earlier revision of that knob was dead code (the
# gate read `PACK_INT4 || knob`, and PACK_INT4 is a compile-time constant, so
# `||` short-circuited it). A dead knob means an A/B that compares the fused
# kernel against itself and reports "bit-identical", which is exactly the
# result that looks like success. So: pinned here, from the outside, where no
# such bug can hide.
#
# (M, K) pairs: Anima's and SDXL's real activation shapes, plus the degenerate
# rows that exercise absmax and the clamp rather than the butterfly -- all-zero
# (rowmax floors at 1e-10), a single large outlier (everything else rounds to
# zero), and values far below bf16's smallest normal.
PACK4_SHAPES = [
    (512, 1024),
    (1024, 2048),
    (4096, 1280),
    (1024, 5120),
    (256, 8192),
]


def _pack4_quant_at(
    x: torch.Tensor, fused: bool, width: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    previous_knob = os.environ.get(PACK4_KNOB)
    previous_width = os.environ.get(OVERRIDE)
    try:
        os.environ[PACK4_KNOB] = "1" if fused else "0"
        if width is None:
            os.environ.pop(OVERRIDE, None)
        else:
            os.environ[OVERRIDE] = str(width)
        m, k = x.shape
        q = torch.empty((m, k // 2), dtype=torch.int8, device=x.device)
        s = torch.empty((m,), dtype=torch.float32, device=x.device)
        with torch.cuda.device(x.device):
            _C.convrot_quant_int4(_dl(x), _dl(q), _dl(s), m, k, PACK4_GROUP, _stream(x))
        return q, s
    finally:
        for name, value in ((PACK4_KNOB, previous_knob), (OVERRIDE, previous_width)):
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _pack4_row(kind: str, m: int, k: int, dtype: torch.dtype) -> torch.Tensor:
    if kind == "zeros":
        return torch.zeros((m, k), device=DEV, dtype=dtype)
    if kind == "spike":
        # One huge value sets the row absmax, so every other element rounds to
        # 0 or +-1 -- the regime where rintf's ties-to-even decides the code.
        t = torch.randn((m, k), device=DEV, dtype=torch.float32) * 1e-3
        t[0, 0] = 30000.0
        return t.to(dtype)
    if kind == "tiny":
        t = torch.randn((m, k), device=DEV, dtype=torch.float32) * 1e-30
        return t.to(dtype)
    raise ValueError(kind)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("m,k", PACK4_SHAPES, ids=lambda v: str(v))
def test_convrot_quant_int4_agrees_between_fused_and_legacy(dtype, m, k):
    """The fused and legacy int4 activation quantizers must produce the same bits.

    They normalize the rotation differently -- the fused kernel scales by 0.5 at
    each butterfly stage, the legacy one multiplies by rsqrt(G) once at the end --
    which is why this was long assumed to make them disagree. It does not:
    multiplying by 0.5 is exact in IEEE-754 and fp32 addition is homogeneous under
    a power-of-two scale, so the two groupings reach the same value, and both then
    round through the input dtype before taking the absmax. But that is the
    argument; this is the measurement, and it is the thing that would catch the
    argument being wrong.
    """
    torch.manual_seed(0xBEEF + m + k)
    x = torch.randn((m, k), device=DEV, dtype=dtype) * 0.7

    fused_q, fused_s = _pack4_quant_at(x, fused=True)
    legacy_q, legacy_s = _pack4_quant_at(x, fused=False)

    assert fused_q.shape == (m, k // 2)
    assert legacy_q.shape == (m, k // 2)
    # int8 compares signed, so widen before comparing: a nibble that differs by
    # bit 7 alone would hide behind an int8 inequality.
    assert torch.equal(fused_q.to(torch.int16), legacy_q.to(torch.int16)), (
        f"M={m} K={k} {dtype}: int4 activation codes differ between fused and legacy"
    )
    assert torch.equal(fused_s, legacy_s), (
        f"M={m} K={k} {dtype}: int4 activation scales differ between fused and legacy"
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("kind", ["zeros", "spike", "tiny"])
def test_convrot_quant_int4_agrees_on_degenerate_rows(dtype, kind):
    """Same invariant on the rows that stress absmax and the clamp, not the FHT."""
    m, k = 512, 2048
    x = _pack4_row(kind, m, k, dtype)
    fused_q, fused_s = _pack4_quant_at(x, fused=True)
    legacy_q, legacy_s = _pack4_quant_at(x, fused=False)
    assert torch.equal(fused_q.to(torch.int16), legacy_q.to(torch.int16)), (
        f"{kind} {dtype}: int4 codes differ between fused and legacy"
    )
    assert torch.equal(fused_s, legacy_s), f"{kind} {dtype}: scales differ"


@pytest.mark.parametrize("width", WIDTHS)
def test_convrot_quant_int4_fused_agrees_across_block_widths(width):
    """The fused int4 path is schedule-invariant too, once block width can reach it."""
    m, k = 1024, 2048
    torch.manual_seed(0x5EED + width)
    x = torch.randn((m, k), device=DEV, dtype=torch.float16) * 0.7
    default_q, default_s = _pack4_quant_at(x, fused=True)
    q, s = _pack4_quant_at(x, fused=True, width=width)
    assert torch.equal(q.to(torch.int16), default_q.to(torch.int16)), (
        f"int4 codes differ at fused block={width}"
    )
    assert torch.equal(s, default_s), f"int4 scales differ at fused block={width}"


@pytest.mark.performance
def test_convrot_quant_int4_knob_reaches_a_different_kernel():
    """Guard the knob itself, not just the numbers it produces.

    A knob that does not switch paths yields a perfect bit-exactness result and a
    1.00x speed ratio, which reads as confirmation rather than as a broken
    experiment -- and that is exactly what happened once already, when the gate read
    ``PACK_INT4 || knob`` and the compile-time constant short-circuited the knob.
    Every other test in this file would have passed, happily, against a dead knob.

    Timing is the only discriminator reachable from Python (the two kernels are
    bit-exact by design, and the shared-memory request that tells them apart is not
    exposed), but **not a two-arm comparison**. Four arms are used instead, so that
    every claim is a ratio between arms and no absolute time enters into it:

    ======  ==========================  ==========================================
    arm     setting                     what it establishes
    ======  ==========================  ==========================================
    A       default                     the fused kernel, whatever it costs
    B       default + FUSED_BLOCK=1024  a width only the *fused* kernel consults
    C       PACK4_FUSED=0               the legacy kernel
    D       PACK4_FUSED=0 + 1024        legacy must ignore that width
    ======  ==========================  ==========================================

    ``C/A`` large says the knob changes the kernel. ``D/A`` large says arm A really
    is the fused path (it honours an override that is fused-only). ``B/C`` near 1
    says arm C really is the legacy path, which is the half a two-arm test cannot
    check: C could differ from A for any reason at all and still pass.

    A two-arm version of this test was tried first and **read 0.739 ms against
    0.739 ms** on this shape -- equal, on a shape where the two kernels differ by
    2.2x, with both arms' sample curves converging point-for-point. The cause was not
    in this file at all: a stale ``_C.abi3.pyd`` left *beside the sources* by
    ``ab_kernel.bat`` was shadowing the installed extension for every pytest run rooted
    at this checkout, so the binary under test predated the kernel that defines this
    knob. ``test_hip_build_provenance.py`` now guards that, and the file-level reason
    the guard lives there rather than here is that a skip mark on this module would
    skip the guard along with everything it is warning about.

    Both forms are kept in this file's history for the same reason the four-arm form
    exists: the failure looked exactly like success, and only a measurement that
    distinguishes "same" from "different" can be trusted to notice.

    Marked ``performance`` because it is the one assertion here that is not purely a
    statement about arithmetic; the bit-exactness tests run unconditionally and are
    what actually pin the invariant.
    """
    import statistics
    import time

    m, k = 2048, 2048
    x = torch.randn((m, k), device=DEV, dtype=torch.bfloat16)

    def burst(knob: str | None, block: int | None, reps: int = 10) -> float:
        t0 = time.perf_counter()
        for _ in range(reps):
            _pack4_quant_at(x, fused=(knob is None), width=block)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1e3 / reps

    arms_spec = {
        "A": (None, None),
        "B": (None, 1024),
        "C": ("0", None),
        "D": ("0", 1024),
    }
    for knob, block in arms_spec.values():
        burst(knob, block)  # warm the clocks before anything is recorded

    samples: dict[str, list[float]] = {name: [] for name in arms_spec}
    for _ in range(5):
        for name, (knob, block) in arms_spec.items():
            samples[name].append(burst(knob, block))
    med = {name: statistics.median(v) for name, v in samples.items()}
    a, b, c, d = (med["A"], med["B"], med["C"], med["D"])
    detail = ", ".join(f"{n}={med[n]:.3f}" for n in "ABCD")

    # Measured on the 6-WGP 780M: C/A ~2.16x, D/A ~2.27x, B/C ~1.16x. Thresholds
    # are set well inside those, and the "ignored" bound is loose because it is the
    # one ratio whose true value is 1.
    assert c / a > 1.3, f"knob did not switch kernels ({detail})"
    assert d / a > 1.3, f"arm A ignored the fused-only block override ({detail})"
    assert abs(b / c - 1.0) < 0.25, f"legacy honoured a fused-only override ({detail})"


def test_convrot_quant_int4_agrees_at_every_k_the_kernel_accepts():
    """Sweep K across the whole range the int4 quantizer supports, not a chosen few.

    This replaces an earlier version of this test that assumed there was a K the
    fused kernel could not fit but the legacy one could, and fell back to it. There
    is not: the legacy kernel stages the entire rotated row in LDS and needs *more*
    of it than the fused kernel does, so past a point both refuse (``convrot: K=32768
    does not fit in LDS``). The launcher's ``if constexpr (!PACK_INT4)`` guard around
    the int8-only global spill therefore does not open a new fallback -- it keeps
    the existing behaviour, which is to decline the fused launch and go on to the
    legacy kernel, exactly as before this kernel was taught about PACK_INT4.

    So the invariant worth pinning is the reachable one: at every K the launcher
    accepts, the two kernels agree bit for bit, and K past the limit raises rather
    than silently writing an unwritten buffer.
    """
    from comfy_kitchen.backends.hip import DTYPE_TO_CODE

    m = 64
    dtype = torch.bfloat16
    code = DTYPE_TO_CODE[dtype]
    accepted = 0
    for k in range(256, 65536, 256):
        x = torch.randn((m, k), device=DEV, dtype=dtype) * 0.7
        try:
            fused_q, fused_s = _pack4_quant_at(x, fused=True)
        except RuntimeError as exc:
            # The refusal has to be the same one the legacy path gives, and it has
            # to name the K: a silent acceptance would mean a half-written buffer.
            assert "does not fit in LDS" in str(exc), f"K={k}: unexpected failure {exc}"
            try:
                _pack4_quant_at(x, fused=False)
            except RuntimeError as legacy_exc:
                assert str(legacy_exc) == str(exc), (
                    f"K={k}: fused and legacy disagree about whether this fits\n"
                    f"  fused:  {exc}\n  legacy: {legacy_exc}"
                )
            else:
                raise AssertionError(
                    f"K={k}: fused refused ({exc}) but legacy accepted it"
                )
            break
        legacy_q, legacy_s = _pack4_quant_at(x, fused=False)
        assert torch.equal(fused_q.to(torch.int16), legacy_q.to(torch.int16)), (
            f"K={k}: int4 codes differ between fused and legacy"
        )
        assert torch.equal(fused_s, legacy_s), f"K={k}: scales differ"
        accepted += 1

    assert accepted >= 8, (
        f"only {accepted} K values were accepted before the limit; the sweep proved "
        f"almost nothing"
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
