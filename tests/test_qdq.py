import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.float_utils import (
    F4_E2M1_MAX,
    F8_E4M3_MAX,
    fp4_x2_to_f32,
)

from .conftest import (
    ConstraintAwareTestInputs,
    assert_values_close,
    get_capable_backends,
    get_supported_devices,
)

# =============================================================================
# FP8 Quantization Tests
# =============================================================================


class TestQuantizePerTensorFP8:
    """FP8 quantization tests parametrized by constraints."""

    @pytest.fixture
    def capable_backends(self, device):
        """Get backends capable of running on current device."""
        backends = get_capable_backends("quantize_per_tensor_fp8", device)
        if not backends:
            pytest.skip(f"No backend supports quantize_per_tensor_fp8 on {device}")
        return backends

    @pytest.mark.parametrize("m,k", [
        (1000, 1000),
        (1024, 2048),
        (2048, 4096),
    ])
    def test_quantize_fp8_all_backends(self, capable_backends, device, seed, m, k):
        """Test FP8 quantization across all capable backends."""
        for backend_name in capable_backends:
            inputs = ConstraintAwareTestInputs("quantize_per_tensor_fp8", backend_name, device)
            x = inputs.tensor("x", shape=(m, k))
            scale = inputs.scalar("scale", 1.0)

            with ck.use_backend(backend_name):
                result = ck.quantize_per_tensor_fp8(x, scale)

            assert result.shape == x.shape
            assert result.dtype == torch.float8_e4m3fn
            assert result.device == x.device

    @pytest.mark.parametrize("m,k", [(512, 512)])
    def test_quantize_fp8_cross_backend_consistency(self, capable_backends, device, seed, m, k):
        """Test that all backends produce consistent results."""
        if len(capable_backends) < 2:
            pytest.skip("Need at least 2 backends for cross-validation")

        inputs = ConstraintAwareTestInputs("quantize_per_tensor_fp8", capable_backends[0], device)
        x = inputs.tensor("x", shape=(m, k))
        scale = inputs.scalar("scale", 1.0)

        results = {}
        for backend_name in capable_backends:
            with ck.use_backend(backend_name):
                results[backend_name] = ck.quantize_per_tensor_fp8(x, scale)

        # Compare all results against first
        ref_backend = capable_backends[0]
        ref_result = results[ref_backend]

        for backend_name, result in results.items():
            if backend_name != ref_backend:
                assert torch.allclose(
                    ref_result.to(torch.float32),
                    result.to(torch.float32),
                    rtol=1e-4, atol=1e-4
                ), f"{backend_name} differs from {ref_backend}"

    def test_quantize_fp8_cpu_fallback(self, seed):
        """Test that eager backend handles CPU tensors."""
        if "cpu" not in get_supported_devices("quantize_per_tensor_fp8"):
            pytest.skip("No backend supports CPU for quantize_per_tensor_fp8")

        cpu_backends = get_capable_backends("quantize_per_tensor_fp8", "cpu")
        assert "eager" in cpu_backends, "Eager should support CPU"

        x = torch.randn(100, 100, dtype=torch.float32, device="cpu")
        scale = torch.tensor([1.0], dtype=torch.float32, device="cpu")

        with ck.use_backend("eager"):
            result = ck.quantize_per_tensor_fp8(x, scale)

        assert result.device.type == "cpu"
        assert result.dtype == torch.float8_e4m3fn


class TestDequantizePerTensorFP8:
    """FP8 dequantization tests."""

    @pytest.fixture
    def capable_backends(self, device):
        backends = get_capable_backends("dequantize_per_tensor_fp8", device)
        if not backends:
            pytest.skip(f"No backend supports dequantize_per_tensor_fp8 on {device}")
        return backends

    @pytest.mark.parametrize("m,k", [(1024, 2048)])
    @pytest.mark.parametrize("output_dtype", [torch.float16, torch.bfloat16])
    def test_dequantize_fp8_all_backends(
        self, capable_backends, device, seed, m, k, output_dtype
    ):
        """Test FP8 dequantization across all capable backends."""
        # First quantize using eager
        x_fp16 = torch.randn(m, k, device=device, dtype=torch.float16)
        scale = torch.tensor([1.0], device=device)

        with ck.use_backend("eager"):
            x_fp8 = ck.quantize_per_tensor_fp8(x_fp16, scale)

        for backend_name in capable_backends:
            with ck.use_backend(backend_name):
                result = ck.dequantize_per_tensor_fp8(x_fp8, scale, output_type=output_dtype)

            assert result.shape == x_fp8.shape
            assert result.dtype == output_dtype
            assert result.device == x_fp8.device


# =============================================================================
# NVFP4 Quantization Tests
# =============================================================================


class TestQuantizeNVFP4:
    """NVFP4 quantization tests."""

    @pytest.fixture
    def capable_backends(self, device):
        backends = get_capable_backends("quantize_nvfp4", device)
        if not backends:
            pytest.skip(f"No backend supports quantize_nvfp4 on {device}")
        return backends

    @pytest.mark.parametrize("m,k", [
        (1024, 2048),
        (512, 1024),
        (129, 128),   # Edge case: odd rows requiring padding
        (33, 65),     # Edge case: both dimensions odd
    ])
    def test_quantize_nvfp4_all_backends(self, capable_backends, device, seed, m, k):
        """Test NVFP4 quantization across all capable backends with accuracy testing."""
        if "eager" not in capable_backends:
            pytest.skip("Need eager backend as reference")

        # Create test input
        x = torch.randn(m, k, device=device, dtype=torch.bfloat16) * 4
        scale = torch.max(torch.abs(x)) / (F8_E4M3_MAX * F4_E2M1_MAX)
        scale = scale.to(torch.float32)
        needs_padding = (m % 16 != 0) or (k % 16 != 0)

        with ck.use_backend("eager"):
            ref_qx, ref_sx = ck.quantize_nvfp4(x, scale, pad_16x=needs_padding)

        for backend_name in capable_backends:
            with ck.use_backend(backend_name):
                qx, sx = ck.quantize_nvfp4(x, scale, pad_16x=needs_padding)

                # Check basic properties
                assert qx.dtype == torch.uint8
                assert sx.dtype == torch.float8_e4m3fn

                assert_values_close(
                    sx.to(torch.float32),
                    ref_sx.to(torch.float32),
                    rtol=1e-5,
                    atol=1e-3,
                    name=f"scales ({backend_name} vs eager)"
                )

                qx_f32 = fp4_x2_to_f32(qx)
                ref_qx_f32 = fp4_x2_to_f32(ref_qx)
                assert_values_close(
                    qx_f32,
                    ref_qx_f32,
                    rtol=1e-2,
                    atol=2.0,
                    name=f"quantized data ({backend_name} vs eager)"
                )

    def test_quantize_nvfp4_cpu_fallback(self, seed):
        """Test that eager backend handles CPU tensors for NVFP4."""
        if "cpu" not in get_supported_devices("quantize_nvfp4"):
            pytest.skip("No backend supports CPU for quantize_nvfp4")

        x = torch.randn(64, 128, dtype=torch.bfloat16, device="cpu") * 4
        scale = torch.max(torch.abs(x)) / (F8_E4M3_MAX * F4_E2M1_MAX)
        scale = scale.to(torch.float32)

        with ck.use_backend("eager"):
            qx, sx = ck.quantize_nvfp4(x, scale)

        assert qx.device.type == "cpu"
        assert sx.device.type == "cpu"


class TestDequantizeNVFP4:
    """NVFP4 dequantization tests."""

    @pytest.fixture
    def capable_backends(self, device):
        backends = get_capable_backends("dequantize_nvfp4", device)
        if not backends:
            pytest.skip(f"No backend supports dequantize_nvfp4 on {device}")
        return backends

    @pytest.mark.parametrize("m,k", [
        (1024, 2048),
        (512, 4096),
        (129, 128),  # Edge case with padding
    ])
    @pytest.mark.parametrize("output_dtype", [torch.float16, torch.bfloat16])
    def test_dequantize_nvfp4_all_backends(
        self, capable_backends, device, seed, m, k, output_dtype
    ):
        """Test NVFP4 dequantization across all capable backends with accuracy testing."""
        if "eager" not in capable_backends:
            pytest.skip("Need eager backend as reference")

        x = torch.randn(m, k, device=device, dtype=torch.bfloat16) * 4
        scale = torch.max(torch.abs(x)) / (F8_E4M3_MAX * F4_E2M1_MAX)
        scale = scale.to(torch.float32)
        needs_padding = (m % 16 != 0) or (k % 16 != 0)

        # Quantize with eager
        with ck.use_backend("eager"):
            qx, sx = ck.quantize_nvfp4(x, scale, pad_16x=needs_padding)
            ref_result = ck.dequantize_nvfp4(qx, scale, sx, output_type=output_dtype)
            # Unpad if needed
            ref_result = ref_result[:m, :k]

        for backend_name in capable_backends:
            with ck.use_backend(backend_name):
                result = ck.dequantize_nvfp4(qx, scale, sx, output_type=output_dtype)
                result = result[:m, :k]  # Unpad if needed

            assert result.shape == (m, k)
            assert result.dtype == output_dtype
            assert result.device == x.device

            assert_values_close(
                result,
                ref_result,
                rtol=1e-3,
                atol=1e-2,
                name=f"dequantized output ({backend_name} vs eager)"
            )


# =============================================================================
# MXFP8 Quantization Tests
# =============================================================================


class TestQuantizeMXFP8:
    """MXFP8 quantization tests."""

    @pytest.fixture
    def capable_backends(self, device):
        backends = get_capable_backends("quantize_mxfp8", device)
        if not backends:
            pytest.skip(f"No backend supports quantize_mxfp8 on {device}")
        return backends

    @pytest.mark.parametrize("m,k", [
        (1024, 2048),
        (512, 1024),
        (128, 256),
        (65, 96),  # Edge case: odd rows requiring padding
    ])
    def test_quantize_mxfp8_all_backends(self, capable_backends, device, seed, m, k):
        """Test MXFP8 quantization across all capable backends."""
        if "eager" not in capable_backends:
            pytest.skip("Need eager backend as reference")

        # Create test input
        x = torch.randn(m, k, device=device, dtype=torch.bfloat16) * 10
        needs_padding = (m % 32 != 0) or (k % 32 != 0)

        with ck.use_backend("eager"):
            ref_qx, ref_sx = ck.quantize_mxfp8(x, pad_32x=needs_padding)

        for backend_name in capable_backends:
            with ck.use_backend(backend_name):
                qx, sx = ck.quantize_mxfp8(x, pad_32x=needs_padding)

                # Check basic properties
                assert qx.dtype == torch.float8_e4m3fn, f"{backend_name}: expected float8_e4m3fn, got {qx.dtype}"
                assert sx.dtype == torch.float8_e8m0fnu, f"{backend_name}: expected float8_e8m0fnu, got {sx.dtype}"
                assert qx.shape == ref_qx.shape, f"{backend_name}: qx shape mismatch"
                assert sx.shape == ref_sx.shape, f"{backend_name}: sx shape mismatch"

                # Compare scales (E8M0 should match exactly or be very close)
                assert_values_close(
                    sx.view(torch.uint8).to(torch.float32),
                    ref_sx.view(torch.uint8).to(torch.float32),
                    rtol=0.0,
                    atol=1.0,  # Allow 1 ULP difference in exponent
                    name=f"scales ({backend_name} vs eager)"
                )

                # Compare quantized data
                assert_values_close(
                    qx.to(torch.float32),
                    ref_qx.to(torch.float32),
                    rtol=1e-3,
                    atol=1e-3,
                    name=f"quantized data ({backend_name} vs eager)"
                )


class TestScaledMMNVFP4:
    """NVFP4 matrix multiplication tests."""

    @pytest.fixture
    def capable_backends(self, device):
        # Exclude eager from scaled_mm tests - it uses torch._scaled_mm which
        # has different matrix semantics than the cuBLAS-based implementations
        backends = get_capable_backends("scaled_mm_nvfp4", device)
        backends = [b for b in backends if b != "eager"]
        if not backends:
            pytest.skip(f"No optimized backend supports scaled_mm_nvfp4 on {device}")
        return backends

    @torch.no_grad()
    @pytest.mark.parametrize("m,k,n", [
        (1024, 2048, 4096),
        (512, 1024, 2048),
    ])
    def test_scaled_mm_nvfp4_all_backends(
        self, capable_backends, device, seed, m, k, n
    ):
        """Test NVFP4 matmul across all capable backends."""
        x = torch.randn(m, k, device=device, dtype=torch.bfloat16).contiguous()
        w = torch.randn(n, k, device=device, dtype=torch.bfloat16).contiguous()
        bias = torch.randn(n, device=device, dtype=torch.bfloat16).contiguous()

        x_absmax = torch.max(torch.abs(x)).float()
        w_absmax = torch.max(torch.abs(w)).float()

        lp_max = 448.0 * 6.0
        tensor_scale_x = x_absmax / lp_max
        tensor_scale_w = w_absmax / lp_max

        for backend_name in capable_backends:
            with ck.use_backend(backend_name):
                qx, sx = ck.quantize_nvfp4(x, tensor_scale_x)
                qw, sw = ck.quantize_nvfp4(w, tensor_scale_w)

            with ck.use_backend("eager"):
                _x, _s = ck.quantize_nvfp4(x, tensor_scale_x)
                x_fake = ck.dequantize_nvfp4(_x, tensor_scale_x, _s, output_type=x.dtype)
                _w, _s = ck.quantize_nvfp4(w, tensor_scale_w)
                w_fake = ck.dequantize_nvfp4(_w, tensor_scale_w, _s, output_type=w.dtype)
            out_hp = torch.nn.functional.linear(x, w, bias=bias)
            out_fake = torch.nn.functional.linear(x_fake, w_fake, bias=bias)

            with ck.use_backend(backend_name):
                out = ck.scaled_mm_nvfp4(
                    qx, qw,
                    tensor_scale_x, tensor_scale_w,
                    sx, sw,
                    bias,
                    out_dtype=x.dtype,
                )

            delta_hp = torch.abs(out - out_hp).mean()
            delta_hp_fake = torch.abs(out_fake - out_hp).mean()
            assert delta_hp < delta_hp_fake + 1e-1, f"Backend {backend_name} failed"
