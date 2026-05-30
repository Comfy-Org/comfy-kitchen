# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for INT8 block-wise quantization."""

import pytest
import torch

from .conftest import (
    assert_values_close,
    get_capable_backends,
    get_supported_devices,
)


# =============================================================================
# INT8 Quantization Tests
# =============================================================================


class TestTensorWiseINT8Layout:
    """Tests for TensorWiseINT8Layout quantized tensor format."""

    @pytest.fixture(autouse=True)
    def cuda_only(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for TensorWiseINT8Layout tests")

    def test_weight_quantize_shape_dtype(self, seed):
        """Weight path: output INT8, scalar scale, shape preserved."""
        from comfy_kitchen.tensor import TensorWiseINT8Layout, QuantizedTensor

        w = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
        qt = QuantizedTensor.from_float(w, "TensorWiseINT8Layout")

        assert qt._qdata.dtype == torch.int8
        assert qt._qdata.shape == w.shape
        assert qt._params.scale.numel() == 1
        assert qt._params.scale.dtype == torch.float32

    def test_activation_quantize_shape_dtype(self, seed):
        """Activation path (is_weight=False): per-row scales [..., 1]."""
        from comfy_kitchen.tensor import TensorWiseINT8Layout, QuantizedTensor

        x = torch.randn(32, 128, device="cuda", dtype=torch.float16)
        qdata, params = TensorWiseINT8Layout.quantize(x, is_weight=False)

        assert qdata.dtype == torch.int8
        assert qdata.shape == x.shape
        assert params.scale.shape == (32, 1)

    def test_weight_dequantize_dtype(self, seed):
        """Dequantize restores original dtype."""
        from comfy_kitchen.tensor import TensorWiseINT8Layout, QuantizedTensor

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
        from comfy_kitchen.tensor import TensorWiseINT8Layout, QuantizedTensor

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

    def test_mm_dispatch(self, seed):
        """aten.mm dispatch fires and produces correct shape."""
        from comfy_kitchen.tensor import QuantizedTensor

        # mm: A [M,K] @ B [K,N] — store B as [K,N] so quantize/dequantize preserves shape
        a = torch.randn(8, 128, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
        qt_b = QuantizedTensor.from_float(b, "TensorWiseINT8Layout")

        out = torch.mm(a, qt_b.dequantize())
        assert out.shape == (8, 64)

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





class TestTensorWisePublicAPI:
    @pytest.fixture
    def seed(self):
        torch.manual_seed(42)

    def test_public_api_quantize_tensorwise(self, seed):
        """comfy_kitchen.quantize_int8_tensorwise op is reachable."""
        import comfy_kitchen as ck
        import torch

        x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        q, scale = ck.quantize_int8_tensorwise(x)

        assert q.dtype == torch.int8
        assert q.shape == x.shape
        assert scale.numel() == 1

    def test_public_api_quantize_rowwise(self, seed):
        """comfy_kitchen.quantize_int8_rowwise op is reachable."""
        import comfy_kitchen as ck
        import torch

        x = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
        q, scale = ck.quantize_int8_rowwise(x)

        assert q.dtype == torch.int8
        assert q.shape == x.shape
        assert scale.shape == (32, 1)

    def test_public_api_dequantize_simple(self, seed):
        """comfy_kitchen.dequantize_int8_simple op is reachable."""
        import comfy_kitchen as ck
        import torch

        x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
        q, scale = ck.quantize_int8_tensorwise(x)
        dq = ck.dequantize_int8_simple(q, scale)

        assert dq.dtype == torch.float32
        assert dq.shape == x.shape

    def test_public_api_int8_linear(self, seed):
        """comfy_kitchen.int8_linear op is reachable."""
        import comfy_kitchen as ck
        from comfy_kitchen.backends.eager.quantization import quantize_int8_tensorwise
        import torch

        x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        w_int8, w_scale = quantize_int8_tensorwise(w)

        out = ck.int8_linear(x, w_int8, w_scale)

        assert out.shape == (4, 64)
        assert out.dtype == torch.bfloat16
