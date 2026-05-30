import pytest
import torch

import comfy_kitchen as ck

from .conftest import assert_values_close, get_capable_backends


def _reference_apply_rope_split_half(t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Exact Python reference for the split-half RoPE formula."""
    t_ = t.reshape(*t.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2).to(freqs.dtype)
    t_out = freqs[..., 0] * t_[..., 0] + freqs[..., 1] * t_[..., 1]
    return t_out.movedim(-1, -2).reshape(*t.shape).type_as(t)


class TestApplyRope:
    """RoPE (Rotary Position Embedding) tests."""
    @pytest.mark.parametrize("op_name", ["apply_rope", "apply_rope1"])
    @pytest.mark.parametrize("backend", ["cuda", "triton", "eager"])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
    @pytest.mark.parametrize("freqs_dtype", [torch.float32, torch.float16, torch.bfloat16], ids=["freqs_fp32", "freqs_fp16", "freqs_bf16"])
    @pytest.mark.parametrize("config_name,layout,config", [
        ("FLUX", "BHND", (1, 24, 4352, 128)),
        ("LTX", "BHND", (2, 32, 4996, 64)),
        ("ZIMAGE", "BNHD", (1, 4096, 30, 128)),
    ], ids=lambda cfg: f"{cfg[0]}")
    def test_rope_ops(self, op_name, backend, device, seed, dtype, freqs_dtype, config_name, layout, config):
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

        freqs_cis = torch.randn(freqs_shape, dtype=freqs_dtype, device=device)

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
            self._validate(xq, xq_out, layout, dtype, freqs_dtype, config_name, backend, ref_xq)
            self._validate(xk, xk_out, layout, dtype, freqs_dtype, config_name, backend, ref_xk)

        else:  # apply_rope1
            x = torch.randn(x_shape, dtype=dtype, device=device)

            with ck.use_backend(backend):
                x_out = ck.apply_rope1(x, freqs_cis)

            ref_x = None
            if backend != "eager":
                with ck.use_backend("eager"):
                    ref_x = ck.apply_rope1(x, freqs_cis)
            self._validate(x, x_out, layout, dtype, freqs_dtype, config_name, backend, ref_x)

    def _validate(self, x, x_out, layout, dtype, freqs_dtype, config_name, backend, ref_x=None):
        assert x_out.shape == x.shape, f"{layout} shape mismatch"
        assert x_out.dtype == x.dtype, f"{layout} dtype mismatch"
        assert x_out.device == x.device

        rtol, atol = 1e-3, 1e-3
        max_mismatch = 0

        if ref_x is not None:
            # Different order of operations between eager (column-wise) and triton/cuda (row-wise)
            # causes ULP rounding differences in reduced precision.
            # - bfloat16: 7-bit mantissa (~0.008 precision) → ~25% values affected
            # - float16:  10-bit mantissa (~0.001 precision) → ~5% values affected
            # - float32:  23-bit mantissa → expect perfect or near-perfect match
            if freqs_dtype == torch.bfloat16:
                max_mismatch = 0.25  # 25% for bf16 freqs
            elif freqs_dtype == torch.float16 or dtype == torch.bfloat16:
                max_mismatch = 0.05  # 5% for fp16 freqs or bf16 inputs
            else:
                max_mismatch = 1e-5  # Very strict for fp32 freqs (0.001%)
            assert_values_close(
                x_out, ref_x, rtol=rtol, atol=atol, max_mismatch_ratio=max_mismatch,
                name=f"{config_name} {layout} x ({backend} vs eager, freqs={freqs_dtype})"
            )


class TestApplyRopeSplitHalf:
    """Tests for apply_rope_split_half and apply_rope_split_half1."""

    @pytest.mark.parametrize("op_name", ["apply_rope_split_half", "apply_rope_split_half1"])
    @pytest.mark.parametrize("backend", ["cuda", "triton", "eager"])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
    @pytest.mark.parametrize("freqs_dtype", [torch.float32, torch.float16, torch.bfloat16], ids=["freqs_fp32", "freqs_fp16", "freqs_bf16"])
    @pytest.mark.parametrize("config_name,layout,config", [
        ("WAN", "BNHD", (2, 12288, 16, 128)),
        ("FLUX", "BHND", (1, 24, 4352, 128)),
        ("LTX", "BHND", (2, 32, 4996, 64)),
    ], ids=lambda cfg: f"{cfg[0]}")
    def test_split_half_vs_reference(self, op_name, backend, device, seed, dtype, freqs_dtype, config_name, layout, config):
        """Verify split-half backends match the Python reference formula."""
        backends = get_capable_backends(op_name, device)
        if backend not in backends:
            pytest.skip(f"{backend} does not support {op_name} on {device}")

        if layout == "BHND":
            b, h, n, d = config
            x_shape = (b, h, n, d)
            freqs_shape = (b, 1, n, d // 2, 2, 2)
        else:  # BNHD
            b, n, h, d = config
            x_shape = (b, n, h, d)
            freqs_shape = (1, n, 1, d // 2, 2, 2)

        freqs_cis = torch.randn(freqs_shape, dtype=freqs_dtype, device=device)

        if op_name == "apply_rope_split_half":
            xq = torch.randn(x_shape, dtype=dtype, device=device)
            xk = torch.randn(x_shape, dtype=dtype, device=device)

            with ck.use_backend(backend):
                xq_out, xk_out = ck.apply_rope_split_half(xq, xk, freqs_cis)

            ref_xq = _reference_apply_rope_split_half(xq, freqs_cis)
            ref_xk = _reference_apply_rope_split_half(xk, freqs_cis)

            self._validate(xq, xq_out, ref_xq, layout, dtype, freqs_dtype, config_name, backend)
            self._validate(xk, xk_out, ref_xk, layout, dtype, freqs_dtype, config_name, backend)
        else:  # apply_rope_split_half1
            x = torch.randn(x_shape, dtype=dtype, device=device)

            with ck.use_backend(backend):
                x_out = ck.apply_rope_split_half1(x, freqs_cis)

            ref_x = _reference_apply_rope_split_half(x, freqs_cis)
            self._validate(x, x_out, ref_x, layout, dtype, freqs_dtype, config_name, backend)

    @pytest.mark.parametrize("op_name", ["apply_rope_split_half", "apply_rope_split_half1"])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
    @pytest.mark.parametrize("freqs_dtype", [torch.float32, torch.float16, torch.bfloat16], ids=["freqs_fp32", "freqs_fp16", "freqs_bf16"])
    def test_split_half_cross_backend(self, op_name, device, seed, dtype, freqs_dtype):
        """Verify all available backends produce the same result for split-half RoPE."""
        if device != "cuda":
            pytest.skip("cross-backend test requires CUDA")

        backends = get_capable_backends(op_name, device)
        if len(backends) < 2:
            pytest.skip(f"need ≥2 backends for cross-backend test, got {backends}")

        # Use the WAN shape from the user's spec
        b, n, h, d = 2, 256, 16, 128
        x_shape = (b, n, h, d)
        freqs_shape = (1, n, 1, d // 2, 2, 2)

        freqs_cis = torch.randn(freqs_shape, dtype=freqs_dtype, device=device)

        results = {}
        if op_name == "apply_rope_split_half":
            xq = torch.randn(x_shape, dtype=dtype, device=device)
            xk = torch.randn(x_shape, dtype=dtype, device=device)
            for be in backends:
                with ck.use_backend(be):
                    results[be] = ck.apply_rope_split_half(xq, xk, freqs_cis)
            ref_be = "eager"
            ref_xq, ref_xk = results[ref_be]
            for be, (out_xq, out_xk) in results.items():
                if be == ref_be:
                    continue
                mm = _max_mismatch(freqs_dtype, dtype)
                assert_values_close(out_xq, ref_xq, rtol=1e-3, atol=1e-3, max_mismatch_ratio=mm,
                                    name=f"apply_rope_split_half xq ({be} vs eager)")
                assert_values_close(out_xk, ref_xk, rtol=1e-3, atol=1e-3, max_mismatch_ratio=mm,
                                    name=f"apply_rope_split_half xk ({be} vs eager)")
        else:
            x = torch.randn(x_shape, dtype=dtype, device=device)
            for be in backends:
                with ck.use_backend(be):
                    results[be] = ck.apply_rope_split_half1(x, freqs_cis)
            ref_be = "eager"
            ref_x = results[ref_be]
            for be, out_x in results.items():
                if be == ref_be:
                    continue
                mm = _max_mismatch(freqs_dtype, dtype)
                assert_values_close(out_x, ref_x, rtol=1e-3, atol=1e-3, max_mismatch_ratio=mm,
                                    name=f"apply_rope_split_half1 ({be} vs eager)")

    def _validate(self, x, x_out, ref, layout, dtype, freqs_dtype, config_name, backend):
        assert x_out.shape == x.shape, f"{config_name} {layout} shape mismatch"
        assert x_out.dtype == x.dtype, f"{config_name} {layout} dtype mismatch"
        assert x_out.device == x.device

        mm = _max_mismatch(freqs_dtype, dtype)
        assert_values_close(
            x_out, ref, rtol=1e-3, atol=1e-3, max_mismatch_ratio=mm,
            name=f"{config_name} {layout} ({backend} vs reference, freqs={freqs_dtype})"
        )


def _max_mismatch(freqs_dtype, dtype):
    if freqs_dtype == torch.bfloat16:
        return 0.25
    if freqs_dtype == torch.float16 or dtype == torch.bfloat16:
        return 0.05
    return 1e-5
