import pytest
import torch

import comfy_kitchen as ck

from .conftest import assert_values_close, get_capable_backends


class TestApplyRope:
    """RoPE (Rotary Position Embedding) tests."""
    @pytest.mark.parametrize("op_name", ["apply_rope", "apply_rope1"])
    @pytest.mark.parametrize("backend", ["cuda", "triton", "eager"])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
    @pytest.mark.parametrize("config_name,layout,config", [
        ("FLUX", "BHND", (1, 24, 4352, 128)),
        ("LTX", "BHND", (2, 32, 4996, 64)),
        ("ZIMAGE", "BNHD", (1, 4096, 30, 128)),
    ], ids=lambda cfg: f"{cfg[0]}")
    def test_rope_ops(self, op_name, backend, device, seed, dtype, config_name, layout, config):
        """Test RoPE operations (apply_rope and apply_rope1) for a specific backend."""
        backends = get_capable_backends(op_name, device)
        if backend not in backends:
            pytest.skip(f"{backend} does not support {op_name} on {device}")

        if layout == "BHND":
            b, h, n, d = config
            x_shape = (b, h, n, d)
            freqs_shape = (b, 1, n, d // 2, 2, 2)  # broadcast over heads
        else:  # BNHD
            b, n, h, d = config
            x_shape = (b, n, h, d)
            freqs_shape = (1, n, 1, d // 2, 2, 2)  # broadcast over batch and heads

        freqs_cis = torch.randn(freqs_shape, dtype=torch.float32, device=device)

        # Run operation based on type
        if op_name == "apply_rope":
            xq = torch.randn(x_shape, dtype=dtype, device=device)
            xk = torch.randn(x_shape, dtype=dtype, device=device)

            with ck.use_backend(backend):
                xq_out, xk_out = ck.apply_rope(xq, xk, freqs_cis)

            # Compare against eager reference
            ref_xq = None
            ref_xk = None
            if backend != "eager":
                with ck.use_backend("eager"):
                    ref_xq, ref_xk = ck.apply_rope(xq, xk, freqs_cis)
            self._validate(xq, xq_out, layout, dtype, config_name, backend, ref_xq)
            self._validate(xk, xk_out, layout, dtype, config_name, backend, ref_xk)

        else:  # apply_rope1
            x = torch.randn(x_shape, dtype=dtype, device=device)

            with ck.use_backend(backend):
                x_out = ck.apply_rope1(x, freqs_cis)

            ref_x = None
            if backend != "eager":
                with ck.use_backend("eager"):
                    ref_x = ck.apply_rope1(x, freqs_cis)
            self._validate(x, x_out, layout, dtype, config_name, backend, ref_x)

    def _validate(self, x, x_out, layout, dtype, config_name, backend, ref_x=None):
        assert x_out.shape == x.shape, f"{layout} shape mismatch"
        assert x_out.dtype == x.dtype, f"{layout} dtype mismatch"
        assert x_out.device == x.device
        if ref_x is not None:
            max_mismatch = 0.001 if dtype == torch.bfloat16 else 0.0
            assert_values_close(
                x_out, ref_x, rtol=1e-3, atol=1e-3, max_mismatch_ratio=max_mismatch,
                name=f"{config_name} {layout} x ({backend} vs eager)"
            )
