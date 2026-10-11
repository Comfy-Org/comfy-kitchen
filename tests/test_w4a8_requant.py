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


def fixed_codebook():
    return torch.tensor(eager_w4a8._FIXED_LUT, device="cuda", dtype=torch.float32)


def assert_quantization_matches_reference(weight, got, codebook):
    expected = eager_w4a8.quantize_w4a8_int8_weight(weight, codebook_tensor=codebook)
    q, s, c, correction, cb = got
    qe, se, ce, _, _ = expected
    assert correction is None and torch.equal(cb, codebook)
    assert torch.allclose(c, ce, rtol=1e-5, atol=1e-8)
    grid = eager_w4a8._dequant_int4_grouped_to_int8(q, s, cb, 16)
    reference_grid = eager_w4a8._dequant_int4_grouped_to_int8(qe, se, cb, 16)
    assert (grid != reference_grid).float().mean().item() < 1e-3
    reconstructed = cuda_backend.dequantize_w4a8_int8_weight(q, s, c, codebook=cb, output_dtype=torch.float32)
    assert rel_l2(reconstructed, weight) < 0.12


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("k", [256, 512, 768, 1024, 1280, 1536, 2048])
def test_short_row_fusion_matches_reference(dtype, k, seed, monkeypatch):
    """The smaller thread blocks must still rotate and quantize every group."""
    if not cuda_backend._WXA8_FUSED_QUANT:
        pytest.skip("fused requant not built")
    weight = torch.randn(384, k, device="cuda", dtype=dtype) * 0.02
    cb = fixed_codebook()
    monkeypatch.setattr(cuda_backend, "_quantize_w4a8_staged", lambda *a, **kw: pytest.fail("staged path used"))
    monkeypatch.setattr(cuda_backend, "_quantize_w4a8_chunked", lambda *a, **kw: pytest.fail("eager path used"))
    got = cuda_backend.quantize_w4a8_int8_weight(weight, codebook_tensor=cb)
    assert_quantization_matches_reference(weight, got, cb)


@pytest.mark.parametrize("dtype,k", [
    (torch.float32, 1024),
    (torch.float32, 51200),
    (torch.float32, 188160),
    (torch.float16, 188160),
    (torch.bfloat16, 188160),
])
def test_staged_fallback_handles_fp32_and_wide_strided_weights(dtype, k, seed, monkeypatch):
    """These inputs cannot use the new fused kernel, but fit the staged CUDA kernel."""
    if not cuda_backend._W4A8_STAGED_QUANT:
        pytest.skip("staged requant not built")
    weight = (torch.randn(5, k * 2, device="cuda", dtype=dtype) * 0.02)[:, ::2]
    cb = fixed_codebook()
    # Exercise multiple row chunks, including a final partial chunk.
    monkeypatch.setattr(cuda_backend, "_QUANT_ROW_ELEM_BUDGET", k * 2)
    monkeypatch.setattr(cuda_backend, "_quantize_w4a8_chunked", lambda *a, **kw: pytest.fail("eager fallback used"))
    got = cuda_backend.quantize_w4a8_int8_weight(weight, codebook_tensor=cb)
    assert_quantization_matches_reference(weight, got, cb)


def test_fused_decline_uses_staged_cuda(weight, monkeypatch):
    if not cuda_backend._W4A8_STAGED_QUANT or not cuda_backend._WXA8_FUSED_QUANT:
        pytest.skip("fused and staged requant not built")
    cb = fixed_codebook()
    monkeypatch.setattr(cuda_backend._C, "quantize_wxa8_convrot_fused", lambda *a, **kw: False)
    monkeypatch.setattr(cuda_backend, "_quantize_w4a8_chunked", lambda *a, **kw: pytest.fail("eager fallback used"))
    got = cuda_backend.quantize_w4a8_int8_weight(weight, codebook_tensor=cb)
    assert_quantization_matches_reference(weight, got, cb)


@pytest.mark.parametrize("unavailable", ["binding", "shared_memory"])
def test_eager_fallback_remains_available(weight, unavailable, monkeypatch):
    cb = fixed_codebook()
    monkeypatch.setattr(cuda_backend, "_fused_quantize_wxa8", lambda *a, **kw: None)
    if unavailable == "binding":
        monkeypatch.setattr(cuda_backend, "_W4A8_STAGED_QUANT", False)
    else:
        monkeypatch.setattr(cuda_backend, "_W4A8_STAGED_MAX_K", 0)
    monkeypatch.setattr(cuda_backend, "_quantize_w4a8_staged", lambda *a, **kw: pytest.fail("staged fallback used"))
    got = cuda_backend.quantize_w4a8_int8_weight(weight, codebook_tensor=cb)
    assert_quantization_matches_reference(weight, got, cb)


@pytest.mark.parametrize("stochastic_rounding", [0, 5])
@pytest.mark.parametrize("dtype,k", [(torch.float32, 1024), (torch.bfloat16, 188160)])
def test_staged_requant_is_deterministic_and_graph_capturable(dtype, k, stochastic_rounding, seed, monkeypatch):
    if not cuda_backend._W4A8_STAGED_QUANT:
        pytest.skip("staged requant not built")
    weight = torch.randn(5, k, device="cuda", dtype=dtype) * 0.02
    cb = fixed_codebook()
    monkeypatch.setattr(cuda_backend, "_QUANT_ROW_ELEM_BUDGET", k * 2)
    monkeypatch.setattr(cuda_backend, "_quantize_w4a8_chunked", lambda *a, **kw: pytest.fail("eager fallback used"))
    kwargs = {"codebook_tensor": cb, "stochastic_rounding": stochastic_rounding}
    expected = cuda_backend.quantize_w4a8_int8_weight(weight, **kwargs)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cuda_backend.quantize_w4a8_int8_weight(weight, **kwargs)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = cuda_backend.quantize_w4a8_int8_weight(weight, **kwargs)
    graph.replay()
    torch.cuda.synchronize()
    for actual, reference in zip(captured[:3], expected[:3], strict=True):
        assert torch.equal(actual.view(torch.uint8), reference.view(torch.uint8))
    q, s, c, _, _ = captured
    reconstructed = cuda_backend.dequantize_w4a8_int8_weight(q, s, c, codebook=cb, output_dtype=torch.float32)
    assert rel_l2(reconstructed, weight) < 0.12
