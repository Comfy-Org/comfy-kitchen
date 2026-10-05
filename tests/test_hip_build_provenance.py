"""Which ``_C.abi3.pyd`` is this session actually testing?

Kept in its own file with **no skip marks on purpose**. The failure it guards against
presents as a silent skip: when a stale extension shadows the installed one,
``comfy_kitchen.backends.hip`` fails to import its extension, ``_EXT_AVAILABLE`` goes
False, and every HIP test in the suite skips -- a result indistinguishable from "this
machine has no HIP device". A test that can only run when the thing it is warning
about is absent cannot be the thing that reports it.

How the shadowing happens
-------------------------
``rebuild_ck.bat`` installs into site-packages, and the ninja build also leaves a copy
at ``build/lib.../comfy_kitchen/backends/hip/_C.abi3.pyd``. A copy can additionally end
up *beside the sources* at ``comfy_kitchen/backends/hip/_C.abi3.pyd``. ``.gitignore``
lists both ``build/`` and ``*.pyd``, so neither shows up in ``git status`` and both
survive across builds -- the stale one indefinitely.

pytest inserts the rootdir at the front of ``sys.path``. So running the suite from the
source checkout makes ``import comfy_kitchen`` resolve to the source tree rather than
site-packages, and the extension that gets loaded is whatever was left there.

What it cost
------------
Adding ``COMFY_CONVROT_PACK4_FUSED`` (the knob that switches the PACK_INT4 activation
quantizer between the fused and legacy kernels) appeared to be a **dead knob**: all four
timing arms measured the same, which is exactly what a knob that changes nothing looks
like. The bit-exactness tests passed throughout -- they would have, since both arms
were running one kernel.

Everything cheaper had been measured and ruled out first: env-write cost (3.66 us, too
small to matter), LDS limits (identical in both processes), the ``torch.cuda.device``
context manager, the ``finally`` restore of the environment, cold clocks (excluded by
alternating arm order with per-arm warmup), ``conftest.py``, ``pytest.ini``, and import
order. What settled it was listing the two candidate binaries:

    57,369,600 bytes  01:47:36   comfy_kitchen/backends/hip/_C.abi3.pyd   (stale)
    59,084,800 bytes  12:20:50   site-packages/.../_C.abi3.pyd            (current)

A 1.7 MB difference, about the size of the PACK_INT4 template instantiations the change
added. Deleting the stale copy made the same test file pass 42/42.

Two lessons worth keeping, both of which cost more than the change being measured:

* A skip is not a pass. "42 skipped" is a result that has to be explained, and here the
  explanation was sitting in a build directory that ``.gitignore`` was hiding.
* Reproduce the anomaly before theorising about it. Copying the identical test file to
  a different working directory -- where it passed -- converted a mystery into a single
  variable in under a minute, after twenty minutes of hypotheses about env vars, shared
  memory and import side effects.
"""

from __future__ import annotations

import os
import sys

import pytest

SITE_MARKERS = ("site-packages", "dist-packages")


def _imported_hip_pkg():
    try:
        import comfy_kitchen.backends.hip as hip_pkg
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"comfy_kitchen.backends.hip does not import here: {exc}")
    return hip_pkg


def test_hip_extension_is_not_shadowed_by_a_stale_source_tree_copy():
    """The loaded extension must be the installed one, not a build left in the tree."""
    hip_pkg = _imported_hip_pkg()

    module_dir = os.path.normcase(os.path.dirname(os.path.abspath(hip_pkg.__file__)))
    in_site = any(marker in module_dir.lower() for marker in SITE_MARKERS)
    assert in_site, (
        f"comfy_kitchen.backends.hip resolved to {hip_pkg.__file__}, which is not under "
        f"site-packages/dist-packages. pytest put the rootdir on sys.path, so a "
        f"_C.abi3.pyd left beside the sources is shadowing the installed build and every "
        f"HIP test in this session is measuring that older binary. Delete "
        f"comfy_kitchen/backends/hip/_C.abi3.pyd (gitignored, so git status will not "
        f"show it) and re-run from outside the checkout."
    )


def test_the_extension_actually_loaded():
    """A shadowing stale copy also shows up as a failed extension import.

    Asserted separately from the path check because the two failures mean different
    things: the path check catches "wrong comfy_kitchen", this one catches "right
    comfy_kitchen, missing or unloadable .pyd". Both otherwise degrade into a skip.
    """
    hip_pkg = _imported_hip_pkg()
    assert hip_pkg._EXT_AVAILABLE, (
        f"the HIP extension did not load: {hip_pkg._EXT_ERROR!r}. If this machine "
        f"genuinely has no HIP device the other tests here would skip too, so this "
        f"assertion is what distinguishes 'no hardware' from 'stale shadowing build'."
    )
    ext_file = getattr(hip_pkg._C, "__file__", None)
    assert ext_file, "the HIP extension reports no __file__, so its age cannot be checked"


def test_no_extension_sits_beside_the_sources():
    """Catch the stale copy at its source, before it can shadow anything.

    Only meaningful when the checkout is on disk, which it is whenever these tests run
    from the repository. Reported rather than skipped when absent, so that a green run
    documents the check instead of quietly not making it.
    """
    repo_src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "comfy_kitchen")
    stray = os.path.join(repo_src, "backends", "hip", "_C.abi3.pyd")
    if os.path.exists(stray):
        pytest.fail(
            f"{stray} exists. It is gitignored, so it survives rebuilds, and it shadows "
            f"the installed extension for any pytest run rooted at this checkout. "
            f"Delete it; rebuild_ck.bat only needs to write to build/lib and "
            f"site-packages."
        )
