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

from .conftest import skip_without_cuda_extension


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


@pytest.mark.parametrize(
    ("m", "k", "expected"),
    [
        (1, 65536, (65536 + 8 * 2 * 256) * 4),
        (2, 256, (256 + 1 * 2 * 256) * 4),
        (2, 2560, (2560 + 10 * 2 * 256) * 4),
        (2, 6144, (6144 + 12 * 2 * 256) * 4),
        (2, 15360, (15360 + 16 * 2 * 256) * 4),
    ],
)
def test_convrot_int8_fused_shared_memory_bytes(m, k, expected):
    assert cuda_backend._convrot_int8_fused_shared_memory_bytes(m, k) == expected


@pytest.mark.parametrize(
    ("m", "k", "group_size", "dtype_size", "expected"),
    [
        (2, 4096, 16, 2, (4096 + 32 * 2 * 16) * 2),
        (2, 8192, 16, 2, (8192 + 128 * 2 * 16) * 2),
        (2, 4096, 64, 2, (4096 + 8 * 2 * 64) * 2),
        (2, 8192, 64, 2, (8192 + 32 * 2 * 64) * 2),
        (2, 4096, 256, 2, (4096 + 4 * 2 * 256) * 2),
        (1, 65536, 256, 2, (65536 + 8 * 2 * 256) * 2),
        (2, 15360, 256, 2, (15360 + 10 * 1 * 256) * 2),
        (2, 65536, 256, 2, (65536 + 16 * 2 * 256) * 2),
        (2, 65536, 256, 4, (65536 + 16 * 2 * 256) * 4),
    ],
)
def test_convrot_int4_fused_shared_memory_bytes(m, k, group_size, dtype_size, expected):
    assert cuda_backend._convrot_int4_fused_shared_memory_bytes(m, k, group_size, dtype_size) == expected


@pytest.mark.parametrize(
    ("m", "n", "k", "expected"),
    [
        (1, 5120, 5120, True),
        (512, 5120, 5120, False),
        (64, 11520, 3840, True),
        (64, 12288, 3072, False),
        (65, 6144, 6144, False),
        (65, 9216, 3072, False),
        (77, 5120, 5120, True),
        (77, 6144, 6144, False),
        (128, 6144, 6144, True),
        (512, 1536, 1536, False),
        (64, 15360, 256, True),
        (512, 2048, 1024, False),
        (1, 55296, 6144, False),
    ],
)
def test_prefer_legacy_int4_kernel_ada(monkeypatch, m, n, k, expected):
    device_index = 0
    monkeypatch.setitem(cuda_backend._device_multiprocessor_count_cache, device_index, 142)

    assert cuda_backend._prefer_legacy_int4_kernel(m, n, k, device_index) is expected


def test_prefer_legacy_int4_kernel_scales_for_ampere(monkeypatch):
    device_index = 0
    monkeypatch.setitem(cuda_backend._device_multiprocessor_count_cache, device_index, 108)

    assert cuda_backend._prefer_legacy_int4_kernel(64, 8192, 4096, device_index)
    assert not cuda_backend._prefer_legacy_int4_kernel(64, 10240, 4096, device_index)


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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("m", [64, 128, 256, 512])
def test_turing_int4_tensor_core_gemm_matches_int8_fallback(seed, dtype, with_bias, m):
    skip_without_cuda_extension()
    if torch.cuda.get_device_capability()[0] < 7:
        pytest.skip("INT4 tensor cores require SM75 or newer")
    if not hasattr(cuda_backend._C, "cutlass_turing_int4_dequant"):
        pytest.skip("CUDA extension was built without the Turing INT4 kernel")

    n, k = 128, 256
    x = torch.randn(m, k, device="cuda", dtype=dtype)
    weight = torch.randn(n, k, device="cuda", dtype=dtype)
    bias = torch.randn(n, device="cuda", dtype=dtype) if with_bias else None
    x_qdata, x_scale = cuda_backend.quantize_int4_rowwise(x)
    weight_qdata, weight_scale = cuda_backend.quantize_int4_rowwise(weight)

    actual = cuda_backend._int4_linear_turing(
        x_qdata,
        weight_qdata,
        x_scale.reshape(-1),
        weight_scale.reshape(-1),
        bias,
        dtype,
    )
    expected = cuda_backend._int4_linear_via_int8(
        x_qdata,
        weight_qdata,
        x_scale.reshape(-1),
        weight_scale.reshape(-1),
        bias,
        dtype,
    )

    assert actual is not None
    tolerance = 2 * torch.finfo(dtype).eps
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)

    if torch.cuda.get_device_capability()[0] >= 8:
        sm80_output = cuda_backend.int4_linear(
            x_qdata,
            weight_qdata,
            x_scale.reshape(-1),
            weight_scale.reshape(-1),
            bias,
            dtype,
        )
        torch.testing.assert_close(actual, sm80_output, rtol=tolerance, atol=tolerance)


def test_int4_linear_routes_turing_to_native_kernel(seed, monkeypatch):
    skip_without_cuda_extension()
    if not hasattr(cuda_backend._C, "cutlass_turing_int4_dequant"):
        pytest.skip("CUDA extension was built without the Turing INT4 kernel")

    m, n, k = 128, 128, 256
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    x_qdata, x_scale = cuda_backend.quantize_int4_rowwise(x)
    weight_qdata, weight_scale = cuda_backend.quantize_int4_rowwise(weight)
    expected = cuda_backend._int4_linear_turing(
        x_qdata,
        weight_qdata,
        x_scale.reshape(-1),
        weight_scale.reshape(-1),
        None,
        torch.bfloat16,
    )

    monkeypatch.setattr(cuda_backend, "_cuda_device_is_turing", lambda _device_index: True)
    actual = cuda_backend.int4_linear(
        x_qdata,
        weight_qdata,
        x_scale,
        weight_scale,
        None,
        torch.bfloat16,
    )

    assert expected is not None
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("m, expected_calls", [(8, 0), (128, 1)])
def test_convrot_w4a4_routes_turing_to_native_kernel(seed, monkeypatch, m, expected_calls):
    skip_without_cuda_extension()
    if torch.cuda.get_device_capability() < (7, 5):
        pytest.skip("INT4 tensor cores require SM75 or newer")
    if not hasattr(cuda_backend._C, "cutlass_turing_int4_dequant"):
        pytest.skip("CUDA extension was built without the Turing INT4 kernel")

    n, k = 128, 256
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    qweight, weight_scale = cuda_backend.quantize_convrot_w4a4_weight(weight)
    original = cuda_backend._int4_linear_turing
    calls = []

    def record_call(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(cuda_backend, "_cuda_device_is_turing", lambda _device_index: True)
    monkeypatch.setattr(cuda_backend, "_int4_linear_turing", record_call)
    output = cuda_backend.convrot_w4a4_linear(x, qweight, weight_scale)

    assert output.shape == (m, n)
    assert len(calls) == expected_calls


def test_convrot_cuda_shared_memory_fit_matches_device_limit():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    x_int8 = torch.empty((2, 65536), device="cuda", dtype=torch.float16)
    max_shared = cuda_backend._max_dynamic_shared_memory_per_block(x_int8)
    int8_requested = cuda_backend._convrot_int8_fused_shared_memory_bytes(x_int8.shape[0], x_int8.shape[1])

    assert cuda_backend._convrot_fused_shared_memory_fits(x_int8, x_int8.shape[1], 256) == (
        int8_requested < max_shared
    )

    x_int4 = torch.empty((2, 65536), device="cuda", dtype=torch.float16)
    int4_requested = cuda_backend._convrot_int4_fused_shared_memory_bytes(
        x_int4.shape[0],
        x_int4.shape[1],
        256,
        x_int4.element_size(),
    )

    assert cuda_backend._convrot_int4_fused_shared_memory_fits(x_int4, x_int4.shape[1], 256) == (
        int4_requested < max_shared
    )


@pytest.mark.parametrize("linear_dtype", ["int4", "int8"])
@pytest.mark.parametrize("convrot_groupsize", [16, 64, 256])
def test_convrot_w4a4_cuda_linear_handles_large_fp16_activations(seed, linear_dtype, convrot_groupsize):
    skip_without_cuda_extension()

    torch.manual_seed(789)
    x = torch.empty(8, 256, device="cuda", dtype=torch.float16)
    pattern = torch.tensor([65504.0, 65504.0, 65504.0, -65504.0], device="cuda", dtype=torch.float16)
    x.copy_(pattern.repeat(x.numel() // pattern.numel()).reshape_as(x))
    w = (torch.randn(32, 256, device="cuda", dtype=torch.float16) * 1.0e-4).contiguous()
    bias = torch.zeros(32, device="cuda", dtype=torch.float16)
    qt = QuantizedTensor.from_float(
        w,
        "TensorCoreConvRotW4A4Layout",
        convrot_groupsize=convrot_groupsize,
        linear_dtype=linear_dtype,
    )

    out = functional.linear(x, qt, bias)
    qact, scale = cuda_backend.quantize_int4_rowwise_convrot64(x, convrot_groupsize)
    q_values = _unpack_int4_row_major(qact)

    assert out.shape == (8, 32)
    assert out.dtype == torch.float16
    assert torch.isfinite(out).all()
    assert torch.isfinite(scale).all()
    assert q_values.min().item() >= -7
    assert q_values.max().item() <= 7


@pytest.mark.parametrize("linear_dtype", ["int4", "int8"])
@pytest.mark.parametrize(("m", "k"), [(128, 15360), (1152, 6912)])
def test_convrot_w4a4_cuda_linear_handles_large_fp16_activation_shapes(seed, linear_dtype, m, k):
    skip_without_cuda_extension()

    torch.manual_seed(790)
    x = torch.empty(m, k, device="cuda", dtype=torch.float16)
    pattern = torch.tensor([65504.0, 65504.0, 65504.0, -65504.0], device="cuda", dtype=torch.float16)
    x.copy_(pattern.repeat(x.numel() // pattern.numel()).reshape_as(x))
    w = (torch.randn(64, k, device="cuda", dtype=torch.float16) * 1.0e-5).contiguous()
    qt = QuantizedTensor.from_float(
        w,
        "TensorCoreConvRotW4A4Layout",
        convrot_groupsize=256,
        linear_dtype=linear_dtype,
    )

    out = functional.linear(x, qt)
    qact, scale = cuda_backend.quantize_int4_rowwise_convrot64(x, 256)
    q_values = _unpack_int4_row_major(qact)

    assert out.shape == (m, 64)
    assert out.dtype == torch.float16
    assert torch.isfinite(out).all()
    assert torch.isfinite(scale).all()
    assert q_values.min().item() >= -7
    assert q_values.max().item() <= 7


@pytest.mark.parametrize("linear_dtype", ["int4", "int8"])
@pytest.mark.parametrize(("m", "k", "n"), [(4192, 6144, 1536), (4192, 16384, 512)])
def test_convrot_w4a4_cuda_linear_handles_big_activation_tensors(seed, linear_dtype, m, k, n):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch.manual_seed(791)
    x = torch.randn(m, k, device="cuda", dtype=torch.float16)
    w = torch.randn(n, k, device="cuda", dtype=torch.float16)
    qt = QuantizedTensor.from_float(
        w,
        "TensorCoreConvRotW4A4Layout",
        convrot_groupsize=256,
        linear_dtype=linear_dtype,
    )

    out = functional.linear(x, qt)

    assert out.shape == (m, n)
    assert out.dtype == torch.float16
    assert torch.isfinite(out).all()


def test_convrot_w4a4_cuda_large_k_quantize_matches_reference(seed):
    skip_without_cuda_extension()

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
    skip_without_cuda_extension()

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
    skip_without_cuda_extension()

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
