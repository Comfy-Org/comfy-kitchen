# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""W4A8 fused ConvRot+requant kernel vs the eager quantizer."""

import pytest
import torch

from comfy_kitchen.backends import cuda as cuda_backend
from comfy_kitchen.backends.eager import w4a8_int8 as eager_w4a8
from tests.conftest import requires_cuda_backend

pytestmark = [pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"), requires_cuda_backend]


def rel_l2(got, ref):
    got, ref = got.float(), ref.float()
    return ((got - ref).norm() / ref.norm().clamp(min=1e-9)).item()


@pytest.fixture
def weight(seed):
    return torch.randn(384, 1024, device="cuda", dtype=torch.bfloat16) * 0.02


def test_fused_requant_matches_eager(weight, monkeypatch):
    """The default 4-bit quantize runs the fused kernel and lands on the eager quantizer's
    result up to fp32 rounding differences."""
    if not cuda_backend._WXA8_FUSED_QUANT:
        pytest.skip("fused requant not built")
    q, s, c, _, cb = eager_w4a8.quantize_w4a8_int8_weight(weight, bits=4)
    monkeypatch.setattr(cuda_backend, "_quantize_w4a8_chunked", lambda *a, **k: pytest.fail("eager path used"))
    qf, sf, cf, _, cbf = cuda_backend.quantize_w4a8_int8_weight(weight, bits=4, codebook_tensor=cb)
    assert torch.allclose(cf, c, rtol=1e-5, atol=0)
    e = rel_l2(eager_w4a8.dequantize_w4a8_int8_weight(q, s, c, codebook=cb, output_dtype=torch.float32), weight)
    ef = rel_l2(eager_w4a8.dequantize_w4a8_int8_weight(qf, sf, cf, codebook=cbf, output_dtype=torch.float32), weight)
    assert abs(ef - e) < 1e-3 * e
    grid = eager_w4a8._dequant_int4_grouped_to_int8(q, s, cb, 16)
    grid_f = eager_w4a8._dequant_int4_grouped_to_int8(qf, sf, cbf, 16)
    assert (grid_f != grid).float().mean().item() < 1e-3


def test_fused_requant_stochastic_rounding(weight):
    if not cuda_backend._WXA8_FUSED_QUANT:
        pytest.skip("fused requant not built")
    q, s, c, _, cb = cuda_backend.quantize_w4a8_int8_weight(weight, bits=4, stochastic_rounding=5)
    q2, _, _, _, _ = cuda_backend.quantize_w4a8_int8_weight(weight, bits=4, codebook_tensor=cb, stochastic_rounding=5)
    assert torch.equal(q, q2)  # seeded, deterministic
    assert rel_l2(cuda_backend.dequantize_w4a8_int8_weight(q, s, c, codebook=cb, output_dtype=torch.float32), weight) < 0.12
