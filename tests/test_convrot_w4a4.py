"""Tests for ConvRot W4A4 int4 tensor-core layout."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as functional

import comfy_kitchen as ck
from comfy_kitchen.backends import cuda as cuda_backend
from comfy_kitchen.backends.eager.svdquant import _unpack_int4_row_major
from comfy_kitchen.tensor import QuantizedTensor
from comfy_kitchen.tensor.convrot_w4a4 import (
    convrot_w4a4_linear,
    dequantize_convrot_w4a4_weight,
    quantize_convrot_w4a4_weight,
)


def test_convrot_w4a4_weight_quantize_contract(seed):
    w = torch.randn(16, 256, dtype=torch.float32)

    q, scale = quantize_convrot_w4a4_weight(w)
    q_values = _unpack_int4_row_major(q)
    dq = dequantize_convrot_w4a4_weight(q, scale, output_dtype=w.dtype)

    assert q.shape == (16, 128)
    assert scale.shape == (16,)
    assert q_values.min().item() >= -7
    assert q_values.max().item() <= 7
    assert dq.shape == w.shape

    rel_err = (w - dq).abs() / (w.abs().max() + 1e-8)
    assert rel_err.mean().item() < 0.04


def test_convrot_w4a4_stochastic_rounding_eager(seed):
    torch.manual_seed(1234)
    w = torch.randn(16, 256, dtype=torch.float32)

    with ck.registry.use_backend("eager"):
        q1, scale1 = quantize_convrot_w4a4_weight(w, stochastic_rounding=123)
        q2, scale2 = quantize_convrot_w4a4_weight(w, stochastic_rounding=123)
        q3, scale3 = quantize_convrot_w4a4_weight(w, stochastic_rounding=124)

    assert torch.equal(q1, q2)
    assert not torch.equal(q1, q3)
    assert torch.equal(scale1, scale2)
    assert torch.equal(scale1, scale3)


def test_convrot_w4a4_linear_eager_matches_dequantized_weight(seed):
    x = torch.randn(5, 256, dtype=torch.float32)
    w = torch.randn(17, 256, dtype=torch.float32)
    bias = torch.randn(17, dtype=torch.float32)
    q, scale = quantize_convrot_w4a4_weight(w)

    with ck.registry.use_backend("eager"):
        out = convrot_w4a4_linear(x, q, scale, bias=bias)

    ref = functional.linear(x, w, bias)
    rel_err = (out - ref).abs() / (ref.abs().max() + 1e-8)
    assert rel_err.mean().item() < 0.08


def test_convrot_w4a4_layout_linear_mm_addmm(seed):
    x = torch.randn(5, 256, dtype=torch.float32)
    w = torch.randn(17, 256, dtype=torch.float32)
    bias = torch.randn(17, dtype=torch.float32)
    qt = QuantizedTensor.from_float(w, "TensorCoreConvRotW4A4Layout")

    out_linear = functional.linear(x, qt, bias)
    out_mm = torch.mm(x, qt.t())
    out_addmm = torch.addmm(bias, x, qt.t())

    assert out_linear.shape == (5, 17)
    assert out_mm.shape == (5, 17)
    assert out_addmm.shape == (5, 17)
    assert torch.allclose(out_linear, out_mm + bias, rtol=1e-5, atol=1e-5)
    assert torch.allclose(out_linear, out_addmm, rtol=1e-5, atol=1e-5)


def test_convrot_w4a4_layout_records_linear_dtype(seed):
    w = torch.randn(16, 256, dtype=torch.float32)
    qt = QuantizedTensor.from_float(w, "TensorCoreConvRotW4A4Layout", linear_dtype="int8")

    assert qt._params.linear_dtype == "int8"
    assert qt._params.convrot_groupsize == 256


def test_convrot_w4a4_rejects_bad_groups(seed):
    w = torch.randn(16, 250, dtype=torch.float32)
    with pytest.raises(ValueError, match="not divisible by convrot_groupsize"):
        quantize_convrot_w4a4_weight(w)


def test_convrot_w4a4_cuda_smoke(seed):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    qt = QuantizedTensor.from_float(w, "TensorCoreConvRotW4A4Layout")

    dq = qt.dequantize()
    out = functional.linear(x, qt, bias)
    assert dq.shape == w.shape
    assert dq.device.type == "cuda"
    assert dq.dtype == torch.bfloat16
    assert out.shape == (64, 128)
    assert out.dtype == torch.bfloat16


def test_convrot_w4a4_cuda_no_bias_large_m(seed):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    x = torch.randn(1152, 256, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    qt = QuantizedTensor.from_float(w, "TensorCoreConvRotW4A4Layout")

    out = functional.linear(x, qt)
    assert out.shape == (1152, 128)
    assert out.dtype == torch.bfloat16


def test_convrot_w4a4_cuda_large_k_quantize_matches_reference(seed):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    x = torch.randn(2, 15360, device="cuda", dtype=torch.bfloat16)
    q_cuda, scale_cuda = cuda_backend.quantize_int4_rowwise_convrot64(x, 256)
    q_values = _unpack_int4_row_major(q_cuda)
    dq = cuda_backend.dequantize_convrot_w4a4_weight(q_cuda, scale_cuda.reshape(-1), output_dtype=torch.float32)

    assert q_cuda.shape == (2, 7680)
    assert scale_cuda.shape == (2, 1)
    assert q_values.min().item() >= -7
    assert q_values.max().item() <= 7
    assert dq.shape == x.shape
    rel_err = (x.float() - dq).abs() / (x.float().abs().max() + 1e-8)
    assert rel_err.mean().item() < 0.04


@pytest.mark.parametrize("convrot_groupsize", [16, 64, 256])
def test_convrot_w4a4_cuda_quantize_clamps_overflow_scale(convrot_groupsize):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    x = torch.empty(2, 256, device="cuda", dtype=torch.float16)
    pattern = torch.tensor([65504.0, 65504.0, 65504.0, -65504.0], device="cuda", dtype=torch.float16)
    x.copy_(pattern.repeat(x.numel() // pattern.numel()).reshape_as(x))

    q, scale = cuda_backend.quantize_int4_rowwise_convrot64(x, convrot_groupsize)
    q_values = _unpack_int4_row_major(q)

    assert q.shape == (2, 128)
    assert scale.shape == (2, 1)
    assert torch.isfinite(scale).all()
    assert q_values.min().item() >= -7
    assert q_values.max().item() <= 7


def test_convrot_w4a4_stochastic_rounding_cuda(seed):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch.manual_seed(1234)
    w = torch.randn(16, 256, device="cuda", dtype=torch.bfloat16)

    with ck.registry.use_backend("cuda"):
        q1, scale1 = quantize_convrot_w4a4_weight(w, stochastic_rounding=123)
        q2, scale2 = quantize_convrot_w4a4_weight(w, stochastic_rounding=123)
        q3, scale3 = quantize_convrot_w4a4_weight(w, stochastic_rounding=124)

    assert torch.equal(q1, q2)
    assert not torch.equal(q1, q3)
    assert torch.equal(scale1, scale2)
    assert torch.equal(scale1, scale3)
