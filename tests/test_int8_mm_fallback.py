# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The eager INT8 matmul must survive ROCm builds where torch._int_mm is unsupported."""

import pytest
import torch

from comfy_kitchen.backends.eager import quantization as q

HIPBLAS_ERROR = (
    "CUDA error: HIPBLAS_STATUS_INVALID_VALUE when calling `hipblasLtMatmulAlgoGetHeuristic(...)`"
)


@pytest.fixture(autouse=True)
def _reset_flag(monkeypatch):
    monkeypatch.setattr(q, "_int_mm_unsupported", False)


def _operands():
    g = torch.Generator().manual_seed(0)
    a = torch.randint(-127, 128, (40, 64), generator=g, dtype=torch.int8)
    b = torch.randint(-127, 128, (64, 48), generator=g, dtype=torch.int8)
    return a, b


def _patch_int_mm(monkeypatch, exc):
    def boom(*_args, **_kwargs):
        raise exc

    monkeypatch.setattr(torch, "_int_mm", boom)
    monkeypatch.setattr(torch, "int8_mm", boom, raising=False)


def test_falls_back_on_hipblas_error(monkeypatch):
    a, b = _operands()
    monkeypatch.setattr(torch.version, "hip", "7.0", raising=False)
    _patch_int_mm(monkeypatch, RuntimeError(HIPBLAS_ERROR))

    out = q._int8_matmul_accumulate(a, b)

    assert out.dtype == torch.int32
    assert torch.equal(out, a.int() @ b.int())
    assert q._int_mm_unsupported is True


def test_other_errors_still_raise(monkeypatch):
    a, b = _operands()
    monkeypatch.setattr(torch.version, "hip", "7.0", raising=False)
    _patch_int_mm(monkeypatch, RuntimeError("CUDA out of memory"))

    with pytest.raises(RuntimeError, match="out of memory"):
        q._int8_matmul_accumulate(a, b)
    assert q._int_mm_unsupported is False


def test_hipblas_error_not_swallowed_off_rocm(monkeypatch):
    a, b = _operands()
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    _patch_int_mm(monkeypatch, RuntimeError(HIPBLAS_ERROR))

    with pytest.raises(RuntimeError, match="HIPBLAS"):
        q._int8_matmul_accumulate(a, b)
