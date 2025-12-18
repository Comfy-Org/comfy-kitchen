import pytest
import torch

import comfy_kitchen as ck

from .conftest import assert_values_close, get_capable_backends


class TestApplyRope:
    """RoPE (Rotary Position Embedding) tests."""

    @pytest.fixture
    def capable_backends(self, device):
        backends = get_capable_backends("apply_rope", device)
        if not backends:
            pytest.skip(f"No backend supports apply_rope on {device}")
        return backends

    @pytest.fixture
    def capable_backends_rope1(self, device):
        backends = get_capable_backends("apply_rope1", device)
        if not backends:
            pytest.skip(f"No backend supports apply_rope1 on {device}")
        return backends

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("config_name,config", [
        ("FLUX", (1, 24, 4352, 128)),
        ("LTX", (2, 32, 4996, 64)),
    ])
    def test_apply_rope_all_backends(self, capable_backends, device, seed, dtype, config_name, config):
        """Test RoPE across all capable backends."""
        b, h, n, d = config
        xq = torch.randn(b, h, n, d, dtype=dtype, device=device)
        xk = torch.randn(b, h, n, d, dtype=dtype, device=device)
        freqs_cis = torch.randn(b, 1, n, d // 2, 2, 2, dtype=torch.float32, device=device)

        for backend_name in capable_backends:
            with ck.use_backend(backend_name):
                xq_out, xk_out = ck.apply_rope(xq, xk, freqs_cis)

            assert xq_out.shape == xq.shape, f"[{backend_name}] xq shape mismatch"
            assert xk_out.shape == xk.shape, f"[{backend_name}] xk shape mismatch"
            assert xq_out.dtype == xq.dtype, f"[{backend_name}] xq dtype mismatch"
            assert xk_out.dtype == xk.dtype, f"[{backend_name}] xk dtype mismatch"
            assert xq_out.device == xq.device

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_apply_rope_cross_backend_consistency(self, capable_backends, device, seed, dtype):
        """Test that all backends produce consistent RoPE results."""
        if len(capable_backends) < 2:
            pytest.skip("Need at least 2 backends for cross-validation")

        b, h, n, d = 1, 8, 256, 64
        xq = torch.randn(b, h, n, d, dtype=dtype, device=device)
        xk = torch.randn(b, h, n, d, dtype=dtype, device=device)
        freqs_cis = torch.randn(b, 1, n, d // 2, 2, 2, dtype=torch.float32, device=device)

        results = {}
        for backend_name in capable_backends:
            with ck.use_backend(backend_name):
                xq_out, xk_out = ck.apply_rope(xq, xk, freqs_cis)
                results[backend_name] = (xq_out, xk_out)

        # Compare all against first
        ref_backend = capable_backends[0]
        ref_xq, ref_xk = results[ref_backend]

        for backend_name, (xq_out, xk_out) in results.items():
            if backend_name != ref_backend:
                assert_values_close(
                    xq_out, ref_xq,
                    rtol=1e-3, atol=1e-3,
                    name=f"xq ({backend_name} vs {ref_backend})"
                )
                assert_values_close(
                    xk_out, ref_xk,
                    rtol=1e-3, atol=1e-3,
                    name=f"xk ({backend_name} vs {ref_backend})"
                )

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_apply_rope1_all_backends(self, capable_backends_rope1, device, seed, dtype):
        """Test apply_rope1 (single tensor) across all capable backends."""
        b, h, n, d = 1, 8, 256, 64
        x = torch.randn(b, h, n, d, dtype=dtype, device=device)
        freqs_cis = torch.randn(b, 1, n, d // 2, 2, 2, dtype=torch.float32, device=device)

        for backend_name in capable_backends_rope1:
            with ck.use_backend(backend_name):
                x_out = ck.apply_rope1(x, freqs_cis)

            assert x_out.shape == x.shape, f"[{backend_name}] shape mismatch"
            assert x_out.dtype == x.dtype, f"[{backend_name}] dtype mismatch"
            assert x_out.device == x.device

    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    def test_apply_rope1_matches_apply_rope(self, capable_backends_rope1, device, seed, dtype):
        """Test that apply_rope1 matches apply_rope when called on individual tensors."""
        b, h, n, d = 1, 8, 256, 64
        xq = torch.randn(b, h, n, d, dtype=dtype, device=device)
        xk = torch.randn(b, h, n, d, dtype=dtype, device=device)
        freqs_cis = torch.randn(b, 1, n, d // 2, 2, 2, dtype=torch.float32, device=device)

        for backend_name in capable_backends_rope1:
            with ck.use_backend(backend_name):
                # Using apply_rope
                xq_rope, xk_rope = ck.apply_rope(xq, xk, freqs_cis)
                # Using apply_rope1 individually
                xq_rope1 = ck.apply_rope1(xq, freqs_cis)
                xk_rope1 = ck.apply_rope1(xk, freqs_cis)

            assert_values_close(xq_rope, xq_rope1, rtol=1e-4, atol=1e-4, name=f"xq [{backend_name}]")
            assert_values_close(xk_rope, xk_rope1, rtol=1e-4, atol=1e-4, name=f"xk [{backend_name}]")
