# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for INT8 block-wise quantization."""

import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.backends import cuda
from comfy_kitchen.tensor import TensorWiseINT8Layout

from .conftest import (
    assert_values_close,
    get_capable_backends,
)


def test_cuda_int8_cublas_turing_n_alignment(monkeypatch):
    """CUDA cuBLAS fallback pads Turing skinny N to 32."""
    from comfy_kitchen.backends import cuda

    calls = []

    def fake_get_device_capability(device_index):
        calls.append(device_index)
        return (7, 5)

    cuda._turing_device_cache.clear()
    monkeypatch.setattr(torch.cuda, "get_device_capability", fake_get_device_capability)

    class FakeCudaTensor:
        is_cuda = True

        def get_device(self):
            return 0

    tensor = FakeCudaTensor()
    assert cuda._cublas_int8_n_alignment(tensor) == 32
    assert cuda._round_up(17, cuda._cublas_int8_n_alignment(tensor)) == 32
    assert calls == [0]


def test_eager_int8_matmul_turing_n_alignment(monkeypatch):
    """Eager torch INT8 matmul pads Turing skinny N to 32."""
    from comfy_kitchen.backends.eager import quantization

    calls = []

    def fake_get_device_capability(device_index):
        calls.append(device_index)
        return (7, 5)

    quantization._turing_device_cache.clear()
    monkeypatch.setattr(torch.cuda, "get_device_capability", fake_get_device_capability)

    class FakeCudaTensor:
        is_cuda = True

        def get_device(self):
            return 0

    tensor = FakeCudaTensor()
    assert quantization._int8_mm_n_alignment(tensor) == 32
    assert quantization._round_up(17, quantization._int8_mm_n_alignment(tensor)) == 32
    assert calls == [0]


def test_cuda_int8_linear_does_not_retain_scratch_tensors():
    """CUDA INT8 linear uses per-call temporaries instead of retained scratch caches."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    x = torch.randn(16, 128, device="cuda", dtype=torch.bfloat16)
    weight = torch.randint(-128, 127, (64, 128), device="cuda", dtype=torch.int8)
    weight_scale = torch.ones((64, 1), device="cuda", dtype=torch.float32)

    out = cuda.int8_linear(x, weight, weight_scale, out_dtype=torch.bfloat16)

    assert out.shape == (16, 64)
    assert not hasattr(cuda, "_int8_quant_scratch")
    assert not hasattr(cuda, "_int8_gemm_int32_scratch")
    assert not hasattr(cuda, "_int8_quant_scratch_tensors")
    assert not hasattr(cuda, "_int8_gemm_int32_scratch_tensor")


def test_turing_fused_int8_shape_selection():
    assert cuda._prefer_turing_fused_int8(128, 4096, 4096)
    assert cuda._prefer_turing_fused_int8(512, 2048, 1024)
    assert cuda._prefer_turing_fused_int8(1024, 4096, 2048)
    assert not cuda._prefer_turing_fused_int8(512, 4096, 4096)
    assert not cuda._prefer_turing_fused_int8(1024, 2048, 4096)


@pytest.mark.parametrize("m", [64, 128, 512, 1024])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("scalar_weight_scale", [False, True])
def test_turing_int8_fused_gemm_matches_reference(
    seed,
    m,
    dtype,
    with_bias,
    scalar_weight_scale,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not hasattr(cuda._C, "cutlass_turing_int8_dequant"):
        pytest.skip("CUDA extension was built without the Turing INT8 kernel")

    n, k = 128, 256
    activation = torch.randint(-127, 128, (m, k), device="cuda", dtype=torch.int8)
    weight = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
    activation_scale = torch.rand(m, device="cuda", dtype=torch.float32)
    weight_scale_shape = () if scalar_weight_scale else (n,)
    weight_scale = torch.rand(weight_scale_shape, device="cuda", dtype=torch.float32)
    bias = torch.randn(n, device="cuda", dtype=dtype) if with_bias else None

    actual = cuda._int8_linear_turing_quantized(
        activation,
        weight,
        activation_scale,
        weight_scale,
        bias,
        dtype,
    )
    accumulator = activation.cpu().to(torch.int32) @ weight.cpu().to(torch.int32).T
    expected = accumulator.to(device="cuda", dtype=torch.float32)
    expected *= activation_scale.reshape(-1, 1)
    expected *= weight_scale.reshape(1, -1)
    if bias is not None:
        expected += bias.float()
    expected = expected.to(dtype)

    assert actual is not None
    tolerance = 2 * torch.finfo(dtype).eps
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


def test_int8_linear_routes_turing_to_fused_kernel(seed, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not hasattr(cuda._C, "cutlass_turing_int8_dequant"):
        pytest.skip("CUDA extension was built without the Turing INT8 kernel")

    x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    weight = torch.randint(-127, 128, (128, 256), device="cuda", dtype=torch.int8)
    weight_scale = torch.rand(128, device="cuda", dtype=torch.float32)
    original = cuda._int8_linear_turing_quantized
    calls = []

    def record_call(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(cuda, "_cuda_device_is_turing", lambda _device_index: True)
    monkeypatch.setattr(cuda, "_int8_linear_turing_quantized", record_call)
    output = cuda.int8_linear(x, weight, weight_scale, out_dtype=torch.bfloat16)

    assert output.shape == (128, 128)
    assert calls == [True]


# =============================================================================
# INT8 Quantization Tests
# =============================================================================


def test_eager_int8_stochastic_rounding_tensorwise(seed):
    """Eager INT8 stochastic rounding is seeded and unbiased for half values."""
    weight = torch.full((4096,), 0.5, dtype=torch.float32)
    scale = torch.tensor(1.0, dtype=torch.float32)

    with ck.registry.use_backend("eager"):
        q1, params = TensorWiseINT8Layout.quantize(weight, scale=scale, stochastic_rounding=123)
        q2, _ = TensorWiseINT8Layout.quantize(weight, scale=scale, stochastic_rounding=123)
        q3, _ = TensorWiseINT8Layout.quantize(weight, scale=scale, stochastic_rounding=124)

    assert q1.dtype == torch.int8
    assert params.scale.item() == 1.0
    assert torch.equal(q1, q2)
    assert not torch.equal(q1, q3)
    assert 0.45 < q1.float().mean().item() < 0.55


def test_cuda_int8_stochastic_rounding_seeded(seed):
    """CUDA stochastic INT8 rounding is seeded and unbiased."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    x = torch.full((4096,), 0.5, device="cuda", dtype=torch.float16)
    x[0] = 127.0

    with ck.registry.use_backend("cuda"):
        q1, scale1 = ck.quantize_int8_rowwise(x, stochastic_rounding=123)
        q2, scale2 = ck.quantize_int8_rowwise(x, stochastic_rounding=123)
        q3, scale3 = ck.quantize_int8_rowwise(x, stochastic_rounding=124)

    assert torch.equal(q1, q2)
    assert not torch.equal(q1, q3)
    assert torch.equal(scale1, scale2)
    assert torch.equal(scale1, scale3)
    assert scale1.item() == 1.0
    assert 0.45 < q1[1:].float().mean().item() < 0.55


def test_eager_int8_stochastic_rounding_convrot(seed):
    """ConvRot INT8 quantization supports seeded stochastic rounding."""
    torch.manual_seed(1234)
    weight = torch.randn(4, 256, dtype=torch.float32)

    with ck.registry.use_backend("eager"):
        q1, params = TensorWiseINT8Layout.quantize(
            weight,
            per_channel=True,
            convrot=True,
            convrot_groupsize=256,
            stochastic_rounding=123,
        )
        q2, _ = TensorWiseINT8Layout.quantize(
            weight,
            per_channel=True,
            convrot=True,
            convrot_groupsize=256,
            stochastic_rounding=123,
        )
        q3, _ = TensorWiseINT8Layout.quantize(
            weight,
            per_channel=True,
            convrot=True,
            convrot_groupsize=256,
            stochastic_rounding=124,
        )

    assert q1.dtype == torch.int8
    assert params.convrot
    assert params.scale.shape == (4, 1)
    assert torch.equal(q1, q2)
    assert not torch.equal(q1, q3)


class TestTensorWiseINT8Layout:
    """Tests for TensorWiseINT8Layout quantized tensor format."""

    @pytest.fixture(autouse=True)
    def cuda_only(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for TensorWiseINT8Layout tests")

    def test_weight_quantize_shape_dtype(self, seed):
        """Weight path: output INT8, scalar scale, shape preserved."""
        from comfy_kitchen.tensor import QuantizedTensor

        w = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(w, "TensorWiseINT8Layout")

        assert qt._qdata.dtype == torch.int8
        assert qt._qdata.shape == w.shape
        assert qt._params.scale.numel() == 1
        assert qt._params.scale.dtype == torch.float32

    def test_activation_quantize_shape_dtype(self, seed):
        """Activation path (is_weight=False): per-row scales [..., 1]."""
        from comfy_kitchen.tensor import TensorWiseINT8Layout

        x = torch.randn(32, 128, device="cuda", dtype=torch.float16)
        qdata, params = TensorWiseINT8Layout.quantize(x, is_weight=False)

        assert qdata.dtype == torch.int8
        assert qdata.shape == x.shape
        assert params.scale.shape == (32, 1)

    def test_weight_dequantize_dtype(self, seed):
        """Dequantize restores original dtype."""
        from comfy_kitchen.tensor import QuantizedTensor

        for dtype in (torch.float16, torch.bfloat16):
            w = torch.randn(64, 128, device="cuda", dtype=dtype)
            qt = QuantizedTensor.from_float(w, "TensorWiseINT8Layout")
            dq = qt.dequantize()
            assert dq.dtype == dtype
            assert dq.shape == w.shape

    def test_weight_roundtrip_error(self, seed):
        """Roundtrip error stays within INT8 quantization tolerance."""
        from comfy_kitchen.tensor import QuantizedTensor

        w = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(w, "TensorWiseINT8Layout")
        dq = qt.dequantize()

        rel_err = (w.float() - dq.float()).abs() / (w.float().abs().max() + 1e-8)
        assert rel_err.mean().item() < 0.02, f"Mean relative error too high: {rel_err.mean():.4f}"

    def test_state_dict_tensors_keys(self, seed):
        """state_dict_tensors returns '' and '_scale' keys."""
        from comfy_kitchen.tensor import QuantizedTensor, TensorWiseINT8Layout

        w = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(w, "TensorWiseINT8Layout")
        sd = TensorWiseINT8Layout.state_dict_tensors(qt._qdata, qt._params)

        assert set(sd.keys()) == {"", "_scale"}
        assert sd[""].dtype == torch.int8
        assert sd["_scale"].numel() == 1

    def test_supports_fast_matmul(self):
        """supports_fast_matmul returns True on CUDA SM >= 7.5."""
        from comfy_kitchen.tensor import TensorWiseINT8Layout

        result = TensorWiseINT8Layout.supports_fast_matmul()
        assert isinstance(result, bool)
        sm = torch.cuda.get_device_capability()
        if sm >= (7, 5):
            assert result is True

    def test_linear_dispatch(self, seed):
        """aten.linear dispatch fires and produces correct shape/dtype."""
        from comfy_kitchen.tensor import QuantizedTensor

        x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        qt_w = QuantizedTensor.from_float(w, "TensorWiseINT8Layout")

        out = torch.nn.functional.linear(x, qt_w)

        assert out.shape == (4, 64)
        assert out.dtype == torch.bfloat16

    def test_linear_dispatch_uses_activation_dtype(self, seed):
        """aten.linear defaults output dtype to runtime activation dtype."""
        from comfy_kitchen.tensor import QuantizedTensor

        x = torch.randn(4, 128, device="cuda", dtype=torch.float16)
        w = torch.randn(64, 128, device="cuda", dtype=torch.float32)
        bias = torch.randn(64, device="cuda", dtype=torch.float32)
        qt_w = QuantizedTensor.from_float(w, "TensorWiseINT8Layout")

        out = torch.nn.functional.linear(x, qt_w, bias)

        assert out.shape == (4, 64)
        assert out.dtype == torch.float16

    def test_mm_dispatch(self, seed):
        """aten.mm dispatch fires and produces correct shape."""
        from comfy_kitchen.tensor import QuantizedTensor

        # mm: A [M,K] @ B [K,N] — store B as [K,N] so quantize/dequantize preserves shape
        a = torch.randn(8, 128, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
        qt_b = QuantizedTensor.from_float(b, "TensorWiseINT8Layout")

        out = torch.mm(a, qt_b)
        assert out.shape == (8, 64)

    def test_mm_dispatch_uses_activation_dtype(self, seed):
        """aten.mm defaults output dtype to runtime activation dtype."""
        from comfy_kitchen.tensor import QuantizedTensor

        a = torch.randn(8, 128, device="cuda", dtype=torch.float16)
        b = torch.randn(128, 64, device="cuda", dtype=torch.float32)
        qt_b = QuantizedTensor.from_float(b, "TensorWiseINT8Layout")

        out = torch.mm(a, qt_b)

        assert out.shape == (8, 64)
        assert out.dtype == torch.float16

    def test_addmm_dispatch(self, seed):
        """aten.addmm dispatch fires and produces correct shape/dtype."""
        from comfy_kitchen.tensor import QuantizedTensor

        x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        bias = torch.randn(64, device="cuda", dtype=torch.bfloat16)
        qt_w = QuantizedTensor.from_float(w, "TensorWiseINT8Layout")

        out = torch.nn.functional.linear(x, qt_w, bias)

        assert out.shape == (4, 64)
        assert out.dtype == torch.bfloat16

    def test_addmm_dispatch_uses_activation_dtype(self, seed):
        """aten.addmm defaults output dtype to runtime activation dtype."""
        from comfy_kitchen.tensor import QuantizedTensor

        bias = torch.randn(64, device="cuda", dtype=torch.float32)
        x = torch.randn(4, 128, device="cuda", dtype=torch.float16)
        w = torch.randn(128, 64, device="cuda", dtype=torch.float32)
        qt_w = QuantizedTensor.from_float(w, "TensorWiseINT8Layout")

        out = torch.addmm(bias, x, qt_w)

        assert out.shape == (4, 64)
        assert out.dtype == torch.float16

    @pytest.mark.parametrize("backend", get_capable_backends("int8_linear", "cuda"))
    def test_int8_linear_correctness(self, seed, backend):
        """Check int8_linear parity across all capable backends."""
        import comfy_kitchen as ck
        from comfy_kitchen.backends.eager.quantization import quantize_int8_tensorwise

        x = torch.randn(128, 256, device="cuda", dtype=torch.float16)
        w = torch.randn(64, 256, device="cuda", dtype=torch.float16)
        bias = torch.randn(64, device="cuda", dtype=torch.float16)

        w_int8, w_scale = quantize_int8_tensorwise(w)

        with ck.registry.use_backend("eager"):
            ref_out = ck.int8_linear(x, w_int8, w_scale, bias=bias, out_dtype=torch.float16)

        with ck.registry.use_backend(backend):
            out = ck.int8_linear(x, w_int8, w_scale, bias=bias, out_dtype=torch.float16)

        # cuBLAS INT8 GEMM output compared to eager may have slight differences due to rounding
        # However, eager vs triton vs cuda should be very close.
        assert_values_close(out, ref_out, rtol=1e-2, atol=1e-2, name=f"int8_linear_{backend}", max_mismatch_ratio=0.01)

    def test_int8_linear_cuda_single_row_gemv(self, seed):
        """CUDA int8_linear uses the single-row GEMV path correctly."""
        import comfy_kitchen as ck
        from comfy_kitchen.backends.eager.quantization import quantize_int8_tensorwise

        x = torch.randn(1, 512, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(384, 512, device="cuda", dtype=torch.bfloat16)
        bias = torch.randn(384, device="cuda", dtype=torch.bfloat16)
        w_int8, w_scale = quantize_int8_tensorwise(w)

        with ck.registry.use_backend("eager"):
            ref_out = ck.int8_linear(x, w_int8, w_scale, bias=bias, out_dtype=torch.bfloat16)

        with ck.registry.use_backend("cuda"):
            out = ck.int8_linear(x, w_int8, w_scale, bias=bias, out_dtype=torch.bfloat16)

        assert out.shape == (1, 384)
        assert out.dtype == torch.bfloat16
        assert_values_close(out, ref_out, rtol=1e-2, atol=1e-2, name="int8_linear_cuda_single_row_gemv", max_mismatch_ratio=0.01)

    def test_public_api_quantize_tensorwise(self, seed):
        """comfy_kitchen.quantize_int8_tensorwise op is reachable."""
        import comfy_kitchen as ck

        x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        q, scale = ck.quantize_int8_tensorwise(x)

        assert q.dtype == torch.int8
        assert q.shape == x.shape
        assert scale.numel() == 1

    def test_public_api_quantize_rowwise(self, seed):
        """comfy_kitchen.quantize_int8_rowwise op is reachable."""
        import comfy_kitchen as ck

        x = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
        q, scale = ck.quantize_int8_rowwise(x)

        assert q.dtype == torch.int8
        assert q.shape == x.shape
        assert scale.shape == (32, 1)

    def test_public_api_dequantize_simple(self, seed):
        """comfy_kitchen.dequantize_int8_simple op is reachable."""
        import comfy_kitchen as ck

        x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
        q, scale = ck.quantize_int8_tensorwise(x)
        dq = ck.dequantize_int8_simple(q, scale)

        assert dq.dtype == torch.float32
        assert dq.shape == x.shape

    @pytest.mark.parametrize("backend", get_capable_backends("dequantize_int8_simple", "cuda"))
    def test_dequantize_simple_backend_correctness(self, seed, backend):
        """CUDA INT8 dequantize matches eager for scalar and rowwise scales."""
        import comfy_kitchen as ck

        x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
        q_scalar, scale_scalar = ck.quantize_int8_tensorwise(x)
        q_row, scale_row = ck.quantize_int8_rowwise(x)

        with ck.registry.use_backend("eager"):
            ref_scalar = ck.dequantize_int8_simple(q_scalar, scale_scalar)
            ref_row = ck.dequantize_int8_simple(q_row, scale_row)

        with ck.registry.use_backend(backend):
            out_scalar = ck.dequantize_int8_simple(q_scalar, scale_scalar)
            out_row = ck.dequantize_int8_simple(q_row, scale_row)

        assert_values_close(out_scalar, ref_scalar, rtol=0, atol=0, name=f"dequant_scalar_{backend}")
        assert_values_close(out_row, ref_row, rtol=0, atol=0, name=f"dequant_rowwise_{backend}")

    def test_dequantize_direct_output_dtype_matches_final_cast(self, seed):
        """Direct fp16/bf16 dequant output matches the prior float32-then-cast behavior."""
        import comfy_kitchen as ck

        x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)

        with ck.registry.use_backend("cuda"):
            q_row, scale_row = ck.quantize_int8_rowwise(x)
            ref_row = torch.ops.comfy_kitchen.dequantize_int8_simple(q_row, scale_row)
            q_conv, scale_conv = torch.ops.comfy_kitchen.quantize_int8_convrot_weight(x, 256)
            ref_conv = torch.ops.comfy_kitchen.dequantize_int8_convrot_weight(q_conv, scale_conv, 256)

            for dtype, code in ((torch.float16, 1), (torch.bfloat16, 2)):
                out_row = torch.ops.comfy_kitchen.dequantize_int8_simple_dtype(q_row, scale_row, code)
                out_conv = torch.ops.comfy_kitchen.dequantize_int8_convrot_weight_dtype(q_conv, scale_conv, 256, code)

                assert out_row.dtype == dtype
                assert out_conv.dtype == dtype
                assert torch.equal(out_row, ref_row.to(dtype))
                assert torch.equal(out_conv, ref_conv.to(dtype))

    def test_public_api_int8_linear(self, seed):
        """comfy_kitchen.int8_linear op is reachable."""
        import comfy_kitchen as ck
        from comfy_kitchen.backends.eager.quantization import quantize_int8_tensorwise

        x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        w_int8, w_scale = quantize_int8_tensorwise(w)

        out = ck.int8_linear(x, w_int8, w_scale)

        assert out.shape == (4, 64)
        assert out.dtype == torch.bfloat16

    def test_convrot_hadamard_properties(self):
        """Verify _build_hadamard constructs correct orthogonal, symmetric matrix."""
        from comfy_kitchen.tensor.int8_utils import _build_hadamard

        # Test valid sizes
        for size in [4, 16, 64, 256]:
            h = _build_hadamard(size, device="cuda", dtype=torch.float32)
            assert h.shape == (size, size)
            # Check symmetry: H^T = H
            assert torch.allclose(h, h.T, atol=1e-5)
            # Check orthogonality: H^T @ H = I
            identity = torch.eye(size, device="cuda", dtype=torch.float32)
            assert torch.allclose(torch.matmul(h.T, h), identity, atol=1e-4)

        # Test invalid sizes
        for size in [2, 8, 32, 128, 500]:
            with pytest.raises(ValueError, match="Regular Hadamard size must be a power of 4"):
                _build_hadamard(size, device="cuda")

    def test_convrot_param_validation(self):
        """Verify parameter combinations for convrot raise expected ValueErrors."""
        from comfy_kitchen.tensor import TensorWiseINT8Layout

        w = torch.randn(64, 256, device="cuda", dtype=torch.float16)

        # 1. convrot with is_weight=False -> ValueError
        with pytest.raises(ValueError, match="convrot is only supported when is_weight is True"):
            TensorWiseINT8Layout.quantize(w, is_weight=False, convrot=True)

        # 2. convrot with per_channel=False -> ValueError
        with pytest.raises(ValueError, match="convrot is only supported when per_channel is True"):
            TensorWiseINT8Layout.quantize(w, is_weight=True, per_channel=False, convrot=True)

    def test_convrot_weight_roundtrip(self, seed):
        """Verify weight roundtrip (quantize -> dequantize) with convrot=True preserves values."""
        from comfy_kitchen.tensor import QuantizedTensor

        w = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        # Using default convrot_groupsize=256
        qt = QuantizedTensor.from_float(
            w, "TensorWiseINT8Layout", per_channel=True, convrot=True, convrot_groupsize=256
        )

        assert qt._params.convrot is True
        assert qt._params.convrot_groupsize == 256

        dq = qt.dequantize()
        assert dq.dtype == torch.bfloat16
        assert dq.shape == w.shape

        # Roundtrip error should stay within expected INT8 quantization limits
        rel_err = (w.float() - dq.float()).abs() / (w.float().abs().max() + 1e-8)
        assert rel_err.mean().item() < 0.02

    def test_convrot_weight_quantize_cuda_roundtrip(self, seed):
        """CUDA ConvRot weight quantization preserves the weight after inverse rotation."""
        from comfy_kitchen.backends import cuda as cuda_backend
        from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_weight

        w = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        h = _build_hadamard(256, device=w.device, dtype=w.dtype)

        with torch.cuda.device(w.device):
            q_cuda, scale_cuda = cuda_backend.quantize_int8_convrot_weight(w, 256)

        dq_rotated = q_cuda.float() * scale_cuda
        dq = _rotate_weight(dq_rotated, h.to(dtype=torch.float32), 256).to(w.dtype)

        rel_err = (w.float() - dq.float()).abs() / (w.float().abs().max() + 1e-8)
        assert rel_err.mean().item() < 0.02

    def test_convrot_weight_dequantize_torch_op_roundtrip(self, seed):
        """The ConvRot dequant torch op preserves weights within INT8 tolerance."""
        from comfy_kitchen.tensor import QuantizedTensor

        w = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(
            w, "TensorWiseINT8Layout", per_channel=True, convrot=True, convrot_groupsize=256
        )

        dq = torch.ops.comfy_kitchen.dequantize_int8_convrot_weight(
            qt._qdata, qt._params.scale, qt._params.convrot_groupsize
        ).to(w.dtype)

        rel_err = (w.float() - dq.float()).abs() / (w.float().abs().max() + 1e-8)
        assert rel_err.mean().item() < 0.02

    def test_convrot_divisibility(self, seed):
        """Verify error when channels are not divisible by convrot_groupsize."""
        from comfy_kitchen.tensor import QuantizedTensor

        # in_features (250) not divisible by 256
        w = torch.randn(64, 250, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="not divisible by group_size"):
            QuantizedTensor.from_float(
                w, "TensorWiseINT8Layout", per_channel=True, convrot=True, convrot_groupsize=256
            )

    def test_convrot_linear_mm_addmm_dispatch(self, seed):
        """Verify linear, mm, and addmm dispatch with convrot=True works and is highly accurate."""
        from comfy_kitchen.tensor import QuantizedTensor

        # Shapes must be compatible with group_size 64 (which is a valid power of 4)
        group_size = 64
        x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        bias = torch.randn(64, device="cuda", dtype=torch.bfloat16)

        # Baseline: normal INT8 Quantization (no rotation)
        qt_w_normal = QuantizedTensor.from_float(
            w, "TensorWiseINT8Layout", per_channel=True, convrot=False
        )
        out_linear_normal = torch.nn.functional.linear(x, qt_w_normal, bias)

        # Active rotation version
        qt_w_rot = QuantizedTensor.from_float(
            w, "TensorWiseINT8Layout", per_channel=True, convrot=True, convrot_groupsize=group_size
        )
        out_linear_rot = torch.nn.functional.linear(x, qt_w_rot, bias)

        # Output shapes & dtypes should match perfectly
        assert out_linear_rot.shape == out_linear_normal.shape
        assert out_linear_rot.dtype == out_linear_normal.dtype

        # Result with and without ConvRot should be extremely close (it is mathematically equivalent under exact math)
        # allowing for expected tiny differences in quantization noise.
        rel_err_linear = (out_linear_rot.float() - out_linear_normal.float()).abs() / (out_linear_normal.float().abs().max() + 1e-8)
        assert rel_err_linear.mean().item() < 0.02

        # Test mm dispatch through the common linear decomposition shape:
        # a [M, K] @ weight.t() [K, N].
        x_mm = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
        w_mm = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        qt_w_mm_rot = QuantizedTensor.from_float(
            w_mm, "TensorWiseINT8Layout", per_channel=True, convrot=True, convrot_groupsize=group_size
        )
        out_mm_rot = torch.mm(x_mm, qt_w_mm_rot.t())
        assert out_mm_rot.shape == (4, 64)
        assert out_mm_rot.dtype == torch.bfloat16

        # Test addmm dispatch
        # addmm: bias + a [M, K] @ weight.t() [K, N]
        bias_mm = torch.randn(64, device="cuda", dtype=torch.bfloat16)
        out_addmm_rot = torch.addmm(bias_mm, x_mm, qt_w_mm_rot.t())
        assert out_addmm_rot.shape == (4, 64)
        assert out_addmm_rot.dtype == torch.bfloat16

    def test_convrot_triton_fused_correctness(self, seed):
        """Verify that fused Triton ConvRot+Quantization matches the eager baseline."""
        import comfy_kitchen as ck
        from comfy_kitchen.tensor import QuantizedTensor

        group_size = 64
        x = torch.randn(32, 128, device="cuda", dtype=torch.float16)
        w = torch.randn(64, 128, device="cuda", dtype=torch.float16)
        bias = torch.randn(64, device="cuda", dtype=torch.float16)

        # Quantize weight with convrot
        qt_w = QuantizedTensor.from_float(
            w, "TensorWiseINT8Layout", per_channel=True, convrot=True, convrot_groupsize=group_size
        )
        weight_qdata, weight_scale = qt_w._qdata, qt_w._params.scale

        # Run with Eager backend
        with ck.registry.use_backend("eager"):
            out_eager = ck.int8_linear(
                x, weight_qdata, weight_scale, bias=bias, out_dtype=torch.float16,
                convrot=True, convrot_groupsize=group_size
            )

        # Run with Triton backend
        with ck.registry.use_backend("triton"):
            out_triton = ck.int8_linear(
                x, weight_qdata, weight_scale, bias=bias, out_dtype=torch.float16,
                convrot=True, convrot_groupsize=group_size
            )

        # Triton and Eager outputs must be extremely close
        assert_values_close(out_triton, out_eager, rtol=1.0e-1, atol=1.0e-1, name="convrot_triton_vs_eager", max_mismatch_ratio=0.02)





class TestTensorWisePublicAPI:
    @pytest.fixture
    def seed(self):
        torch.manual_seed(42)

    def test_public_api_quantize_tensorwise(self, seed, device):
        """comfy_kitchen.quantize_int8_tensorwise op is reachable."""
        import torch

        import comfy_kitchen as ck

        x = torch.randn(64, 128, device=device, dtype=torch.bfloat16)
        q, scale = ck.quantize_int8_tensorwise(x)

        assert q.dtype == torch.int8
        assert q.shape == x.shape
        assert scale.numel() == 1

    def test_public_api_quantize_rowwise(self, seed, device):
        """comfy_kitchen.quantize_int8_rowwise op is reachable."""
        import torch

        import comfy_kitchen as ck

        x = torch.randn(32, 128, device=device, dtype=torch.bfloat16)
        q, scale = ck.quantize_int8_rowwise(x)

        assert q.dtype == torch.int8
        assert q.shape == x.shape
        assert scale.shape == (32, 1)

    def test_public_api_dequantize_simple(self, seed, device):
        """comfy_kitchen.dequantize_int8_simple op is reachable."""
        import torch

        import comfy_kitchen as ck

        x = torch.randn(32, 64, device=device, dtype=torch.bfloat16)
        q, scale = ck.quantize_int8_tensorwise(x)
        dq = ck.dequantize_int8_simple(q, scale)

        assert dq.dtype == torch.float32
        assert dq.shape == x.shape

    def test_public_api_int8_linear(self, seed, device):
        """comfy_kitchen.int8_linear op is reachable."""
        import torch

        import comfy_kitchen as ck
        from comfy_kitchen.backends.eager.quantization import quantize_int8_tensorwise

        x = torch.randn(4, 128, device=device, dtype=torch.bfloat16)
        w = torch.randn(64, 128, device=device, dtype=torch.bfloat16)
        w_int8, w_scale = quantize_int8_tensorwise(w)

        out = ck.int8_linear(x, w_int8, w_scale)

        assert out.shape == (4, 64)
        assert out.dtype == torch.bfloat16

    def test_eager_int8_linear_single_row(self, seed, device):
        """Eager int8_linear supports single-row batches."""
        import torch

        import comfy_kitchen as ck
        from comfy_kitchen.backends.eager.quantization import quantize_int8_tensorwise

        x = torch.randn(1, 128, device=device, dtype=torch.bfloat16)
        w = torch.randn(64, 128, device=device, dtype=torch.bfloat16)
        w_int8, w_scale = quantize_int8_tensorwise(w)

        with ck.registry.use_backend("eager"):
            out = ck.int8_linear(x, w_int8, w_scale)

        assert out.shape == (1, 64)
        assert out.dtype == torch.bfloat16

    def test_eager_int8_linear_pads_k_to_int8_mm_tile(self, seed, device):
        """Eager int8_linear pads K to int8 matmul's tile size."""
        import torch

        import comfy_kitchen as ck
        from comfy_kitchen.backends.eager.quantization import quantize_int8_tensorwise

        x = torch.randn(17, 12, device=device, dtype=torch.bfloat16)
        w = torch.randn(64, 12, device=device, dtype=torch.bfloat16)
        w_int8, w_scale = quantize_int8_tensorwise(w)

        with ck.registry.use_backend("eager"):
            out = ck.int8_linear(x, w_int8, w_scale)

        assert out.shape == (17, 64)
        assert out.dtype == torch.bfloat16

    def test_eager_int8_linear_pads_n_to_int8_mm_tile(self, seed, device):
        """Eager int8_linear pads N to int8 matmul's tile size."""
        import torch

        import comfy_kitchen as ck
        from comfy_kitchen.backends.eager.quantization import quantize_int8_tensorwise

        x = torch.randn(17, 16, device=device, dtype=torch.bfloat16)
        w = torch.randn(1, 16, device=device, dtype=torch.bfloat16)
        w_int8, w_scale = quantize_int8_tensorwise(w)

        with ck.registry.use_backend("eager"):
            out = ck.int8_linear(x, w_int8, w_scale)

        assert out.shape == (17, 1)
        assert out.dtype == torch.bfloat16
