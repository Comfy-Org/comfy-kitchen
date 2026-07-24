"""HIP dispatch and gating logic. Needs no GPU and no compiled extension.

These cover which architectures the backend registers for and which ops it
advertises on each. They are deliberately not gated on registry.is_available("hip"):
on a CPU-only runner the kernel suite in test_hip_wmma.py skips in full, and these
routing rules would otherwise go untested behind a green tick.
"""
import pytest

from comfy_kitchen.backends import hip as hip_backend


@pytest.mark.parametrize(
    "arches",
    [
        [],
        ["gfx90a"],   # CDNA: MFMA, not WMMA
        ["gfx1010"],  # RDNA1: neither matrix cores nor the dot-product paths
        ["gfx1201", "gfx90a"],
    ],
)
def test_hip_declines_unsupported_arch(arches):
    """The backend registers for RDNA2/3/4 and nothing else."""
    assert hip_backend._unsupported_arch_reason(arches) is not None


@pytest.mark.parametrize(
    "arches",
    [["gfx1201", "gfx1200"], ["gfx1100"], ["gfx1151"], ["gfx1030"], ["gfx1200", "gfx1030"]],
)
def test_hip_accepts_rdna2_through_rdna4(arches):
    assert hip_backend._unsupported_arch_reason(arches) is None


def test_hip_declines_when_an_arch_cannot_be_read():
    """A device whose architecture is unknown cannot be shown to be supported."""
    assert hip_backend._unsupported_arch_reason([None]) is not None
    assert hip_backend._unsupported_arch_reason(["gfx1200", None]) is not None


@pytest.mark.parametrize(
    ("arches", "expected"),
    [
        (["gfx1200"], True),
        (["gfx1201", "gfx1100"], True),
        (["gfx1151"], True),
        (["gfx1030"], False),             # RDNA2 has no matrix cores
        (["gfx1200", "gfx1030"], False),  # kernels launch on the tensor's own device
        ([None], False),
    ],
)
def test_hip_wmma_capability(arches, expected):
    """Only an all-matrix-core process may advertise the GEMMs."""
    assert hip_backend._has_wmma(arches) is expected


def test_hip_drops_gemms_without_matrix_cores():
    """RDNA2 keeps the elementwise kernels and hands the GEMMs back to triton/eager."""
    with_wmma = hip_backend._build_constraints(has_wmma=True)
    without = hip_backend._build_constraints(has_wmma=False)

    # Every WMMA-only op must be advertised with matrix cores; an intersection
    # would pass while any single one was missing from the constraints.
    assert set(with_wmma) >= hip_backend._WMMA_ONLY_OPS
    assert not (hip_backend._WMMA_ONLY_OPS & set(without))
    # The elementwise kernels need no matrix cores and must survive.
    for op in ("apply_rope", "adaln", "quantize_per_tensor_fp8", "gemv_awq_w4a16"):
        assert op in without
