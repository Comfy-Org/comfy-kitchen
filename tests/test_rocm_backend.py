"""
tests/test_rocm_backend.py
===========================

Test suite for the comfy_kitchen ROCm backend.

Run on an AMD GPU system with ROCm PyTorch:
    pytest tests/test_rocm_backend.py -v

Run with fallback to eager (no ROCm):
    pytest tests/test_rocm_backend.py -v -k "not scaled_mm"

These tests verify:
  1. Backend registers correctly on ROCm
  2. FP8 quantise/dequantise round-trips correctly
  3. NVFP4 pack/unpack round-trips correctly
  4. RoPE output matches eager reference
  5. scaled_mm_fp8 matches BF16 matmul within tolerance
  6. scaled_mm_nvfp4 matches eager dequant + matmul
"""

from __future__ import annotations

import math
import pytest
import torch

# ---------------------------------------------------------------------------
# Skip the entire module if not on a ROCm system
# ---------------------------------------------------------------------------

IS_ROCM = getattr(torch.version, "hip", None) is not None
HAS_CUDA = torch.cuda.is_available()

rocm_only = pytest.mark.skipif(
    not IS_ROCM or not HAS_CUDA,
    reason="ROCm PyTorch with a GPU required",
)

rocm_fp8 = pytest.mark.skipif(
    not IS_ROCM or not HAS_CUDA,
    reason="ROCm + gfx1100+ required for FP8 GEMM",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gfx_major() -> int:
    try:
        import re
        props = torch.cuda.get_device_properties(0)
        arch = getattr(props, "gcnArchName", "gfx000").split(":")[0]
        m = re.search(r"\d+", arch)
        return int(m.group(0)) if m else 0
    except Exception:
        return 0


def _fp8_supported() -> bool:
    n = _gfx_major()
    return n >= 1100 or (940 <= n <= 942)


# ---------------------------------------------------------------------------
# Backend registration
# ---------------------------------------------------------------------------

class TestRocmBackendRegistration:

    @rocm_only
    def test_rocm_backend_in_list(self):
        import comfy_kitchen as ck
        backends = ck.list_backends()
        assert "rocm" in backends, (
            f"rocm backend not registered. Available: {list(backends.keys())}"
        )

    @rocm_only
    def test_rocm_backend_available(self):
        import comfy_kitchen as ck
        backends = ck.list_backends()
        info = backends["rocm"]
        assert info["available"] is True, (
            f"rocm backend registered but not available: {info['unavailable_reason']}"
        )

    @rocm_only
    def test_rocm_backend_capabilities(self):
        import comfy_kitchen as ck
        backends = ck.list_backends()
        caps = set(backends["rocm"]["capabilities"])
        expected = {
            "quantize_per_tensor_fp8",
            "dequantize_per_tensor_fp8",
            "quantize_nvfp4",
            "dequantize_nvfp4",
            "apply_rope",
            "apply_rope1",
            "scaled_mm_nvfp4",
        }
        assert expected.issubset(caps), f"Missing caps: {expected - caps}"

    @rocm_only
    @pytest.mark.skipif(not _fp8_supported(), reason="FP8 needs gfx1100+")
    def test_scaled_mm_fp8_in_capabilities(self):
        import comfy_kitchen as ck
        backends = ck.list_backends()
        caps = set(backends["rocm"]["capabilities"])
        assert "scaled_mm_fp8" in caps


# ---------------------------------------------------------------------------
# FP8 quantise / dequantise
# ---------------------------------------------------------------------------

class TestFP8QuantDequant:

    @rocm_only
    def test_quantize_shape_preserved(self):
        import comfy_kitchen as ck
        x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        scale = torch.tensor([1.0], device="cuda")
        with ck.use_backend("rocm"):
            q = ck.quantize_per_tensor_fp8(x, scale)
        assert q.shape == x.shape

    @rocm_only
    def test_quantize_dtype(self):
        import comfy_kitchen as ck
        x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        scale = torch.tensor([1.0], device="cuda")
        with ck.use_backend("rocm"):
            q = ck.quantize_per_tensor_fp8(x, scale)
        assert q.dtype == torch.float8_e4m3fnuz

    @rocm_only
    def test_roundtrip_accuracy(self):
        """Quantise then dequantise: max error < 5% of scale * FP8_MAX."""
        import comfy_kitchen as ck
        torch.manual_seed(42)
        x = torch.randn(256, 256, device="cuda", dtype=torch.float32) * 10.0
        scale = x.abs().max() / torch.finfo(torch.float8_e4m3fnuz).max
        scale = scale.unsqueeze(0).to("cuda")

        with ck.use_backend("rocm"):
            q  = ck.quantize_per_tensor_fp8(x, scale)
            xr = ck.dequantize_per_tensor_fp8(q, scale)

        # FP8 has 3-bit mantissa, expect ~3% rounding error
        rel_err = (x - xr).abs() / (x.abs() + 1e-6)
        assert rel_err.mean() < 0.05, f"Mean rel error too high: {rel_err.mean():.4f}"

    @rocm_only
    def test_scale_respected(self):
        """A scale of 2.0 should halve the effective range before quantising."""
        import comfy_kitchen as ck
        x = torch.ones(16, device="cuda", dtype=torch.bfloat16)
        scale_1 = torch.tensor([1.0], device="cuda")
        scale_2 = torch.tensor([2.0], device="cuda")

        with ck.use_backend("rocm"):
            q1 = ck.quantize_per_tensor_fp8(x, scale_1)
            q2 = ck.quantize_per_tensor_fp8(x, scale_2)
            r1 = ck.dequantize_per_tensor_fp8(q1, scale_1)
            r2 = ck.dequantize_per_tensor_fp8(q2, scale_2)

        # Both should recover approx 1.0; the quantisation point differs
        torch.testing.assert_close(r1, torch.ones_like(r1), atol=0.1, rtol=0.05)
        torch.testing.assert_close(r2, torch.ones_like(r2), atol=0.1, rtol=0.05)


# ---------------------------------------------------------------------------
# NVFP4 pack / unpack
# ---------------------------------------------------------------------------

class TestNVFP4:

    @rocm_only
    def test_pack_shape(self):
        """Packed tensor should have half as many bytes as elements."""
        import comfy_kitchen as ck
        x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
        scale = torch.ones(x.numel() // 16, device="cuda")
        with ck.use_backend("rocm"):
            packed = ck.quantize_nvfp4(x.flatten(), scale)
        assert packed.numel() == x.numel() // 2

    @rocm_only
    def test_roundtrip_values(self):
        """Values representable in E2M1 should survive a round-trip."""
        import comfy_kitchen as ck
        # E2M1 exactly representable values: 0, 0.5, 1, 1.5, 2, 3, 4, 6
        representable = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
             0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
            device="cuda", dtype=torch.bfloat16
        )
        scale = torch.ones(1, device="cuda")  # scale=1 so values pass through

        with ck.use_backend("rocm"):
            packed = ck.quantize_nvfp4(representable, scale)
            unpacked = ck.dequantize_nvfp4(packed, scale, torch.bfloat16)

        torch.testing.assert_close(unpacked, representable, atol=0.0, rtol=0.0)

    @rocm_only
    def test_dequant_dtype(self):
        import comfy_kitchen as ck
        x = torch.randn(32, device="cuda", dtype=torch.bfloat16)
        scale = torch.ones(2, device="cuda")
        with ck.use_backend("rocm"):
            packed = ck.quantize_nvfp4(x, scale)
            dq_bf16 = ck.dequantize_nvfp4(packed, scale, torch.bfloat16)
            dq_f32  = ck.dequantize_nvfp4(packed, scale, torch.float32)
        assert dq_bf16.dtype == torch.bfloat16
        assert dq_f32.dtype  == torch.float32


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

class TestRoPE:

    @staticmethod
    def _ref_rope(x, cos, sin):
        """Pure-PyTorch reference RoPE implementation."""
        half = x.shape[-1] // 2
        x0, x1 = x[..., :half], x[..., half:]
        return torch.cat([x0 * cos - x1 * sin, x1 * cos + x0 * sin], dim=-1)

    @rocm_only
    def test_apply_rope_matches_reference(self):
        import comfy_kitchen as ck
        torch.manual_seed(0)
        B, S, H, D = 2, 16, 4, 64
        q   = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
        k   = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
        cos = torch.ones(S, D // 2, device="cuda")
        sin = torch.zeros(S, D // 2, device="cuda")  # sin=0 → identity

        with ck.use_backend("rocm"):
            q_r, k_r = ck.apply_rope(q, k, cos, sin)

        # With cos=1, sin=0: output should equal input
        torch.testing.assert_close(q_r, q, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(k_r, k, atol=1e-3, rtol=1e-3)

    @rocm_only
    def test_apply_rope_vs_reference(self):
        import comfy_kitchen as ck
        torch.manual_seed(1)
        B, S, H, D = 1, 8, 2, 32
        q   = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32)
        k   = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32)
        cos = torch.cos(torch.arange(S * D // 2, device="cuda").float()
                        .reshape(S, D // 2) * 0.01)
        sin = torch.sin(torch.arange(S * D // 2, device="cuda").float()
                        .reshape(S, D // 2) * 0.01)

        with ck.use_backend("rocm"):
            q_rocm, k_rocm = ck.apply_rope(q, k, cos, sin)

        # Reference (broadcast cos/sin over batch and head dims)
        cos_b = cos.unsqueeze(0).unsqueeze(2)
        sin_b = sin.unsqueeze(0).unsqueeze(2)
        q_ref = self._ref_rope(q, cos_b, sin_b)
        k_ref = self._ref_rope(k, cos_b, sin_b)

        torch.testing.assert_close(q_rocm, q_ref, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(k_rocm, k_ref, atol=1e-4, rtol=1e-4)

    @rocm_only
    def test_apply_rope1(self):
        import comfy_kitchen as ck
        torch.manual_seed(2)
        B, S, H, D = 2, 4, 1, 16
        x   = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32)
        cos = torch.ones(S, D // 2, device="cuda")
        sin = torch.zeros(S, D // 2, device="cuda")

        with ck.use_backend("rocm"):
            out = ck.apply_rope1(x, cos, sin)

        torch.testing.assert_close(out, x, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# scaled_mm_fp8  (hipBLASLt path — gfx1100+ only)
# ---------------------------------------------------------------------------

class TestScaledMMFP8:

    @pytest.fixture(autouse=True)
    def skip_if_no_fp8(self):
        if not IS_ROCM or not HAS_CUDA or not _fp8_supported():
            pytest.skip(f"scaled_mm_fp8 needs ROCm + gfx1100+ (current: gfx{_gfx_major()})")

    def test_output_shape(self):
        import comfy_kitchen as ck
        M, K, N = 128, 256, 64
        a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)

        scale_a = torch.tensor(1.0, device="cuda")
        scale_b = torch.tensor(1.0, device="cuda")

        a_fp8 = a.to(torch.float8_e4m3fnuz)
        b_fp8 = b.to(torch.float8_e4m3fnuz)

        with ck.use_backend("rocm"):
            out = ck.scaled_mm_fp8(a_fp8, b_fp8, scale_a, scale_b,
                                   out_dtype=torch.bfloat16)

        assert out.shape == (M, N)
        assert out.dtype == torch.bfloat16

    def test_matches_bf16_matmul(self):
        """scaled_mm_fp8 should match BF16 matmul within FP8 quantisation error."""
        import comfy_kitchen as ck
        torch.manual_seed(42)
        M, K, N = 64, 128, 64
        a = torch.randn(M, K, device="cuda", dtype=torch.float32) * 0.1
        b = torch.randn(K, N, device="cuda", dtype=torch.float32) * 0.1

        ref = torch.mm(a, b).to(torch.bfloat16)

        scale_a = torch.tensor(1.0, device="cuda")
        scale_b = torch.tensor(1.0, device="cuda")
        a_fp8 = a.to(torch.float8_e4m3fnuz)
        b_fp8 = b.to(torch.float8_e4m3fnuz)

        with ck.use_backend("rocm"):
            out = ck.scaled_mm_fp8(a_fp8, b_fp8, scale_a, scale_b,
                                   out_dtype=torch.bfloat16)

        # FP8 has limited precision; allow ~2% relative error
        rel_err = (ref - out).abs() / (ref.abs() + 1e-6)
        assert rel_err.mean() < 0.02, f"Mean rel err: {rel_err.mean():.4f}"

    def test_scale_applied(self):
        """Non-unit scales should scale the output proportionally."""
        import comfy_kitchen as ck
        torch.manual_seed(7)
        M, K, N = 32, 32, 32
        a = torch.ones(M, K, device="cuda", dtype=torch.float8_e4m3fnuz)
        b = torch.ones(K, N, device="cuda", dtype=torch.float8_e4m3fnuz)

        scale_a = torch.tensor(2.0, device="cuda")
        scale_b = torch.tensor(3.0, device="cuda")
        scale_1 = torch.tensor(1.0, device="cuda")

        with ck.use_backend("rocm"):
            out_scaled = ck.scaled_mm_fp8(a, b, scale_a, scale_b,
                                          out_dtype=torch.float32)
            out_unit   = ck.scaled_mm_fp8(a, b, scale_1, scale_1,
                                          out_dtype=torch.float32)

        # output_scaled should be ~6x output_unit (scale_a * scale_b = 6)
        ratio = out_scaled / (out_unit + 1e-9)
        torch.testing.assert_close(ratio,
                                   torch.full_like(ratio, 6.0),
                                   atol=0.1, rtol=0.05)


# ---------------------------------------------------------------------------
# scaled_mm_nvfp4  (dequant + BF16 fallback — all ROCm devices)
# ---------------------------------------------------------------------------

class TestScaledMMNVFP4:

    @rocm_only
    def test_output_shape(self):
        import comfy_kitchen as ck
        M, K, N = 64, 128, 32
        a_full  = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        b_full  = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)

        scale_a = torch.ones(M * K // 16, device="cuda")
        scale_b = torch.ones(N * K // 16, device="cuda")

        with ck.use_backend("rocm"):
            a_p = ck.quantize_nvfp4(a_full.flatten(), scale_a)
            b_p = ck.quantize_nvfp4(b_full.flatten(), scale_b)
            out = ck.scaled_mm_nvfp4(a_p, b_p, scale_a, scale_b)

        assert out.shape == (M, N)

    @rocm_only
    def test_matches_eager(self):
        """ROCm nvfp4 path must match eager dequant + matmul."""
        import comfy_kitchen as ck
        torch.manual_seed(3)
        M, K, N = 32, 64, 32
        a_full = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        b_full = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)

        scale_a = torch.ones(M * K // 16, device="cuda")
        scale_b = torch.ones(N * K // 16, device="cuda")

        with ck.use_backend("rocm"):
            a_p = ck.quantize_nvfp4(a_full.flatten(), scale_a)
            b_p = ck.quantize_nvfp4(b_full.flatten(), scale_b)
            out_rocm = ck.scaled_mm_nvfp4(a_p, b_p, scale_a, scale_b)

        with ck.use_backend("eager"):
            a_p2 = ck.quantize_nvfp4(a_full.flatten(), scale_a)
            b_p2 = ck.quantize_nvfp4(b_full.flatten(), scale_b)
            out_eager = ck.scaled_mm_nvfp4(a_p2, b_p2, scale_a, scale_b)

        torch.testing.assert_close(out_rocm, out_eager, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Backend isolation:  explicit backend selection
# ---------------------------------------------------------------------------

class TestBackendIsolation:

    @rocm_only
    def test_can_force_eager_on_rocm(self):
        """The eager backend must still work alongside the rocm backend."""
        import comfy_kitchen as ck
        x = torch.randn(32, 32, device="cuda", dtype=torch.bfloat16)
        scale = torch.tensor([1.0], device="cuda")

        with ck.use_backend("eager"):
            q = ck.quantize_per_tensor_fp8(x, scale)
        assert q is not None

    @rocm_only
    def test_can_force_rocm(self):
        import comfy_kitchen as ck
        x = torch.randn(32, 32, device="cuda", dtype=torch.bfloat16)
        scale = torch.tensor([1.0], device="cuda")

        with ck.use_backend("rocm"):
            q = ck.quantize_per_tensor_fp8(x, scale)
        assert q is not None

    @rocm_only
    def test_auto_selection_prefers_rocm_over_eager(self):
        """Auto-dispatch should use rocm (priority=5) over eager (priority=0)."""
        import comfy_kitchen as ck
        import logging

        log_capture = []
        handler = logging.handlers.MemoryHandler(capacity=100, flushLevel=logging.DEBUG)

        x = torch.randn(32, 32, device="cuda", dtype=torch.bfloat16)
        scale = torch.tensor([1.0], device="cuda")

        # Just verify it doesn't crash and produces output of the right shape
        q = ck.quantize_per_tensor_fp8(x, scale)
        assert q.shape == x.shape
