# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.nn import functional

import comfy_kitchen as ck
from tests.conftest import assert_values_close, get_capable_backends

# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------

def _ref_adaln(x, scale, shift, eps=1e-6):
    return functional.layer_norm(x, x.shape[-1:], eps=eps) * (1 + scale) + shift


def _ref_rms_adaln(x, scale, shift, eps=1e-6):
    return functional.rms_norm(x, x.shape[-1:], eps=eps) * (1 + scale) + shift


# ---------------------------------------------------------------------------
# Parametrize helpers
# ---------------------------------------------------------------------------

_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
_SHAPES = [
    (2, 16, 64),    # small
    (2, 256, 768),  # bert-like
    (1, 64, 3072),  # flux-like
]


def _scale_shift(shape, dtype, device):
    """Return scale and shift tensors shaped (B, 1, D) — as produced by modulation."""
    batch, _, dim = shape
    scale = torch.randn(batch, 1, dim, dtype=dtype, device=device) * 0.1
    shift = torch.randn(batch, 1, dim, dtype=dtype, device=device) * 0.1
    return scale, shift


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAdaLN:
    @pytest.mark.parametrize("backend", ["cuda", "hip", "triton", "eager"])
    @pytest.mark.parametrize("dtype", _DTYPES)
    @pytest.mark.parametrize("shape", _SHAPES)
    def test_forward(self, backend, dtype, shape, seed, cuda_available):
        device = "cuda" if cuda_available else "cpu"
        capable = get_capable_backends("adaln", device)

        if backend not in capable:
            pytest.skip(f"backend '{backend}' not capable for adaln on {device}")

        x = torch.randn(shape, dtype=dtype, device=device)
        scale, shift = _scale_shift(shape, dtype, device)

        with ck.use_backend(backend):
            out = ck.adaln(x, scale, shift, 1e-6)

        if backend == "eager":
            ref = _ref_adaln(x, scale, shift, 1e-6)
        else:
            ref = _ref_adaln(
                x.float(), scale.float(), shift.float(), 1e-6
            ).to(dtype)

        rtol = 1e-2 if dtype != torch.float32 else 1e-4
        atol = 1e-2 if dtype != torch.float32 else 1e-5

        assert out.shape == x.shape
        assert out.dtype == dtype
        assert_values_close(out.float(), ref.float(), rtol=rtol, atol=atol,
                            name=f"adaln[{backend}/{dtype}]")

    @pytest.mark.parametrize("backend", ["cuda", "hip", "triton"])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_vs_eager(self, backend, dtype, seed, cuda_available):
        """Non-eager backends must match the appropriate dtype reference."""
        device = "cuda" if cuda_available else "cpu"
        capable = get_capable_backends("adaln", device)

        if backend not in capable:
            pytest.skip(f"backend '{backend}' not capable for adaln on {device}")
        if "eager" not in capable:
            pytest.skip("eager backend not capable (reference unavailable)")

        shape = (2, 64, 768)
        x = torch.randn(shape, dtype=dtype, device=device)
        scale, shift = _scale_shift(shape, dtype, device)

        if dtype is torch.bfloat16:
            ref = _ref_adaln(
                x.float(), scale.float(), shift.float(), 1e-6
            ).to(dtype)
        else:
            with ck.use_backend("eager"):
                ref = ck.adaln(x, scale, shift, 1e-6)

        with ck.use_backend(backend):
            out = ck.adaln(x, scale, shift, 1e-6)

        assert_values_close(out.float(), ref.float(), rtol=1e-2, atol=1e-2,
                            name=f"adaln[{backend} vs eager/{dtype}]")

    def test_broadcast_scale_shift(self, seed, cuda_available):
        """Scale/shift of shape (B, 1, D) broadcast correctly over N tokens."""
        device = "cuda" if cuda_available else "cpu"
        capable = get_capable_backends("adaln", device)
        if not capable:
            pytest.skip("no capable backend available")

        shape = (2, 32, 128)
        batch, _, dim = shape
        dtype = torch.float32
        x = torch.randn(shape, dtype=dtype, device=device)
        scale = torch.randn(batch, 1, dim, dtype=dtype, device=device)
        shift = torch.randn(batch, 1, dim, dtype=dtype, device=device)

        out = ck.adaln(x, scale, shift, 1e-6)
        ref = _ref_adaln(x, scale, shift, 1e-6)

        assert out.shape == shape
        assert_values_close(out, ref, rtol=1e-4, atol=1e-5,
                            name="adaln[broadcast scale/shift]")

    def test_same_scale_shift(self, seed, cuda_available):
        """scale=0, shift=0 should return plain layer norm."""
        device = "cuda" if cuda_available else "cpu"
        capable = get_capable_backends("adaln", device)
        if not capable:
            pytest.skip("no capable backend available")

        shape = (1, 8, 64)
        x = torch.randn(shape, device=device)
        scale = torch.zeros_like(x)
        shift = torch.zeros_like(x)

        out = ck.adaln(x, scale, shift, 1e-6)
        ref = functional.layer_norm(x, x.shape[-1:], eps=1e-6)

        assert_values_close(out, ref, rtol=1e-4, atol=1e-5, name="adaln[no-op scale/shift]")

    @pytest.mark.parametrize("backend", ["cuda", "hip", "triton", "eager"])
    def test_contiguous_output(self, backend, cuda_available):
        """Output must always be contiguous."""
        device = "cuda" if cuda_available else "cpu"
        capable = get_capable_backends("adaln", device)
        if backend not in capable:
            pytest.skip(f"backend '{backend}' not capable")

        x = torch.randn(4, 16, 64, device=device)
        scale = torch.zeros(4, 1, 64, device=device)
        shift = torch.zeros(4, 1, 64, device=device)

        with ck.use_backend(backend):
            out = ck.adaln(x, scale, shift)
        assert out.is_contiguous()


class TestRMSAdaLN:
    @pytest.mark.parametrize("backend", ["cuda", "hip", "triton", "eager"])
    @pytest.mark.parametrize("dtype", _DTYPES)
    @pytest.mark.parametrize("shape", _SHAPES)
    def test_forward(self, backend, dtype, shape, seed, cuda_available):
        device = "cuda" if cuda_available else "cpu"
        capable = get_capable_backends("rms_adaln", device)

        if backend not in capable:
            pytest.skip(f"backend '{backend}' not capable for rms_adaln on {device}")

        x = torch.randn(shape, dtype=dtype, device=device)
        scale, shift = _scale_shift(shape, dtype, device)

        with ck.use_backend(backend):
            out = ck.rms_adaln(x, scale, shift, 1e-6)

        if backend == "eager":
            ref = _ref_rms_adaln(x, scale, shift, 1e-6)
        else:
            ref = _ref_rms_adaln(
                x.float(), scale.float(), shift.float(), 1e-6
            ).to(dtype)

        rtol = 1e-2 if dtype != torch.float32 else 1e-4
        atol = 1e-2 if dtype != torch.float32 else 1e-5

        assert out.shape == x.shape
        assert out.dtype == dtype
        assert_values_close(out.float(), ref.float(), rtol=rtol, atol=atol,
                            name=f"rms_adaln[{backend}/{dtype}]")

    @pytest.mark.parametrize("backend", ["cuda", "hip", "triton", "eager"])
    @pytest.mark.parametrize("batch", [1, 2, 3])
    def test_per_sample_modulation_broadcast(self, backend, batch, seed, cuda_available):
        """scale/shift of shape (B, 1, D) must broadcast per sample, not per row.

        Regression guard: a modulo-based row->modulation mapping agrees with the
        correct block mapping only when B == 1, so it passes single-sample tests
        and silently corrupts batched (e.g. CFG) runs.
        """
        device = "cuda" if cuda_available else "cpu"
        capable = get_capable_backends("rms_adaln", device)
        if backend not in capable:
            pytest.skip(f"backend '{backend}' not capable for rms_adaln on {device}")

        tokens, dim = 37, 128  # token count deliberately not a multiple of batch
        x = torch.randn(batch, tokens, dim, dtype=torch.float32, device=device)
        scale = torch.randn(batch, 1, dim, dtype=torch.float32, device=device)
        shift = torch.randn(batch, 1, dim, dtype=torch.float32, device=device)

        with ck.use_backend(backend):
            out = ck.rms_adaln(x, scale, shift, 1e-6)
        ref = _ref_rms_adaln(x, scale, shift, 1e-6)

        assert_values_close(out, ref, rtol=1e-4, atol=1e-5,
                            name=f"rms_adaln[{backend}/per-sample B={batch}]")

    @pytest.mark.parametrize("backend", ["cuda", "hip", "triton", "eager"])
    def test_per_token_modulation(self, backend, seed, cuda_available):
        """Fully materialized per-token scale/shift (no broadcast) must also work."""
        device = "cuda" if cuda_available else "cpu"
        capable = get_capable_backends("rms_adaln", device)
        if backend not in capable:
            pytest.skip(f"backend '{backend}' not capable for rms_adaln on {device}")

        shape = (2, 48, 256)
        x = torch.randn(shape, dtype=torch.float32, device=device)
        scale = torch.randn(shape, dtype=torch.float32, device=device)
        shift = torch.randn(shape, dtype=torch.float32, device=device)

        with ck.use_backend(backend):
            out = ck.rms_adaln(x, scale, shift, 1e-6)
        ref = _ref_rms_adaln(x, scale, shift, 1e-6)

        assert_values_close(out, ref, rtol=1e-4, atol=1e-5,
                            name=f"rms_adaln[{backend}/per-token]")

    @pytest.mark.parametrize("backend", ["cuda", "hip", "triton", "eager"])
    @pytest.mark.parametrize("dim", [1, 31, 129, 4096])
    def test_odd_and_large_dims(self, backend, dim, seed, cuda_available):
        """Non-vectorizable and large hidden dims take the scalar tail / block loop."""
        device = "cuda" if cuda_available else "cpu"
        capable = get_capable_backends("rms_adaln", device)
        if backend not in capable:
            pytest.skip(f"backend '{backend}' not capable for rms_adaln on {device}")

        x = torch.randn(2, 8, dim, dtype=torch.float32, device=device)
        scale = torch.randn(2, 1, dim, dtype=torch.float32, device=device)
        shift = torch.randn(2, 1, dim, dtype=torch.float32, device=device)

        with ck.use_backend(backend):
            out = ck.rms_adaln(x, scale, shift, 1e-6)
        ref = _ref_rms_adaln(x, scale, shift, 1e-6)

        assert_values_close(out, ref, rtol=1e-4, atol=1e-5,
                            name=f"rms_adaln[{backend}/D={dim}]")

    @pytest.mark.parametrize("backend", ["cuda", "hip", "triton", "eager"])
    @pytest.mark.parametrize("rows", [512, 2048])
    def test_warp_and_block_kernel_paths(self, backend, rows, seed, cuda_available):
        """The CUDA backend switches kernels at N == 1024; cover both sides."""
        device = "cuda" if cuda_available else "cpu"
        capable = get_capable_backends("rms_adaln", device)
        if backend not in capable:
            pytest.skip(f"backend '{backend}' not capable for rms_adaln on {device}")

        x = torch.randn(1, rows, 320, dtype=torch.float32, device=device)
        scale = torch.randn(1, 1, 320, dtype=torch.float32, device=device)
        shift = torch.randn(1, 1, 320, dtype=torch.float32, device=device)

        with ck.use_backend(backend):
            out = ck.rms_adaln(x, scale, shift, 1e-6)
        ref = _ref_rms_adaln(x, scale, shift, 1e-6)

        assert_values_close(out, ref, rtol=1e-4, atol=1e-5,
                            name=f"rms_adaln[{backend}/N={rows}]")

    def test_differs_from_layernorm_adaln(self, seed, cuda_available):
        """rms_adaln must not silently be adaln: a non-zero-mean row differs."""
        device = "cuda" if cuda_available else "cpu"
        if not get_capable_backends("rms_adaln", device):
            pytest.skip("no capable backend available")

        x = torch.randn(1, 8, 64, dtype=torch.float32, device=device) + 5.0
        scale = torch.zeros(1, 1, 64, dtype=torch.float32, device=device)
        shift = torch.zeros(1, 1, 64, dtype=torch.float32, device=device)

        rms_out = ck.rms_adaln(x, scale, shift, 1e-6)
        ln_out = ck.adaln(x, scale, shift, 1e-6)

        assert not torch.allclose(rms_out, ln_out, rtol=1e-3, atol=1e-3)
        assert_values_close(rms_out, _ref_rms_adaln(x, scale, shift, 1e-6),
                            rtol=1e-4, atol=1e-5, name="rms_adaln[vs adaln]")

    @pytest.mark.parametrize("backend", ["cuda", "hip", "triton", "eager"])
    def test_contiguous_output(self, backend, cuda_available):
        device = "cuda" if cuda_available else "cpu"
        capable = get_capable_backends("rms_adaln", device)
        if backend not in capable:
            pytest.skip(f"backend '{backend}' not capable")

        x = torch.randn(4, 16, 64, device=device)
        scale = torch.zeros(4, 1, 64, device=device)
        shift = torch.zeros(4, 1, 64, device=device)

        with ck.use_backend(backend):
            out = ck.rms_adaln(x, scale, shift)
        assert out.is_contiguous()

    def test_non_contiguous_input(self, seed, cuda_available):
        """A transposed (non-contiguous) input must still normalize over the last dim."""
        device = "cuda" if cuda_available else "cpu"
        if not get_capable_backends("rms_adaln", device):
            pytest.skip("no capable backend available")

        base = torch.randn(2, 128, 64, dtype=torch.float32, device=device)
        x = base.transpose(1, 2)  # (2, 64, 128), non-contiguous
        scale = torch.randn(2, 1, 128, dtype=torch.float32, device=device)
        shift = torch.randn(2, 1, 128, dtype=torch.float32, device=device)

        out = ck.rms_adaln(x, scale, shift, 1e-6)
        ref = _ref_rms_adaln(x, scale, shift, 1e-6)

        assert out.shape == x.shape
        assert_values_close(out, ref, rtol=1e-4, atol=1e-5,
                            name="rms_adaln[non-contiguous]")
