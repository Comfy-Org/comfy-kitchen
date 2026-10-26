# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Overlap-safe device memmove against a bounce-copy reference."""

import pytest
import torch

import comfy_kitchen as ck

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

CHUNK = 65536


@pytest.mark.parametrize("nbytes", [16, CHUNK, 3 * CHUNK + 16, 40 * CHUNK])
@pytest.mark.parametrize("shift", [16, -16, 1024, -1024, CHUNK, -CHUNK, 3 * CHUNK + 16, -3 * CHUNK - 16, 20 * CHUNK])
def test_memmove_overlapping(nbytes, shift):
    """Moving down or up by less than a chunk, a whole chunk, several chunks or past the end
    leaves the bytes the reference bounce copy would."""
    total = nbytes + abs(shift) + 32
    buf = torch.randint(0, 256, (total,), dtype=torch.uint8, device="cuda")
    ref = buf.clone()
    src = 16 + max(-shift, 0)
    dst = src + shift
    ref[dst:dst + nbytes] = buf[src:src + nbytes].clone()
    ck.memmove(buf[dst:dst + nbytes], buf[src:src + nbytes])
    torch.cuda.synchronize()
    assert torch.equal(buf, ref)


def test_memmove_views_other_dtypes():
    """Callers hand over the typed views they hold; the bytes move all the same."""
    buf = torch.randn(4096 + 64, dtype=torch.float32, device="cuda")
    ref = buf.clone()
    ref[:4096] = buf[64:].clone()
    ck.memmove(buf[:4096].view(64, 64), buf[64:].view(64, 64))
    torch.cuda.synchronize()
    assert torch.equal(buf, ref)
