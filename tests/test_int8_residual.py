# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""int8_linear(residual=...) — the pre-norm block's addcmul fused into the GEMM
epilogue."""

import pytest
import torch

import comfy_kitchen as ck
from tests.conftest import get_capable_backends, rel_err

_GROUP = 256


def _linear_ref(x, weight, wscale, bias, **kwargs):
    return ck.int8_linear(x, weight, wscale, bias, torch.float16,
                          convrot=True, convrot_groupsize=_GROUP, **kwargs)


class TestResidualEpilogue:
    # shapes chosen to hit different tile configs of the fused int8 GEMM,
    # including the stream-K swizzles (the H3 decoder's w2 lands on one)
    @pytest.mark.parametrize(
        "m,k,n",
        [
            (1797, 2048, 2048),
            (1797, 8192, 2048),
            (1797, 2048, 6144),
            (512, 4096, 512),
            (37, 2048, 256),
        ],
    )
    @pytest.mark.parametrize("with_bias", [True, False])  # nullptr bias broadcasts 0
    def test_matches_eager_addcmul(self, m, k, n, with_bias, seed, cuda_available):
        if not cuda_available:
            pytest.skip("CUDA required")

        x = torch.randn(m, k, dtype=torch.float16, device="cuda")
        weight = torch.randint(-127, 127, (n, k), dtype=torch.int8, device="cuda")
        wscale = torch.tensor(0.01, dtype=torch.float32, device="cuda")
        bias = torch.randn(n, dtype=torch.float16, device="cuda") if with_bias else None
        resid = torch.randn(m, n, dtype=torch.float16, device="cuda")
        rscale = torch.randn(n, dtype=torch.float16, device="cuda")

        plain = _linear_ref(x, weight, wscale, bias)
        ref = torch.addcmul(resid, plain, rscale)
        got = _linear_ref(x, weight, wscale, bias, residual=resid, residual_scale=rscale)

        assert got.shape == ref.shape
        # The GEMM and bias rounding are identical; only the final multiply/add
        # runs in float32 instead of fp16, so differences are rounding-level.
        rel = rel_err(got, ref)
        assert rel < 1e-2, f"rel={rel:.3e}"

    @pytest.mark.parametrize("backend", ["cuda", "hip", "triton", "eager"])
    def test_backends_agree(self, backend, seed, cuda_available):
        device = "cuda" if cuda_available else "cpu"
        if backend not in get_capable_backends("int8_linear", device):
            pytest.skip(f"backend '{backend}' not capable")

        x = torch.randn(256, 2048, dtype=torch.float16, device=device)
        weight = torch.randint(-127, 127, (512, 2048), dtype=torch.int8, device=device)
        wscale = torch.tensor(0.01, dtype=torch.float32, device=device)
        bias = torch.randn(512, dtype=torch.float16, device=device)
        resid = torch.randn(256, 512, dtype=torch.float16, device=device)
        rscale = torch.randn(512, dtype=torch.float16, device=device)

        with ck.use_backend(backend):
            plain = _linear_ref(x, weight, wscale, bias)
            ref = torch.addcmul(resid, plain, rscale)
            got = _linear_ref(x, weight, wscale, bias, residual=resid, residual_scale=rscale)
        rel = rel_err(got, ref)
        assert rel < 1e-2, f"{backend}: rel={rel:.3e}"

    def test_residual_requires_scale(self, cuda_available):
        if not cuda_available:
            pytest.skip("CUDA required")
        x = torch.randn(4, 2048, dtype=torch.float16, device="cuda")
        weight = torch.randint(-127, 127, (256, 2048), dtype=torch.int8, device="cuda")
        wscale = torch.tensor(0.01, dtype=torch.float32, device="cuda")
        resid = torch.randn(4, 256, dtype=torch.float16, device="cuda")
        with pytest.raises(ValueError, match="residual"):
            ck.int8_linear(x, weight, wscale, None, torch.float16, residual=resid)

    def test_residual_dtype_mismatch_falls_back(self, seed, cuda_available):
        """A residual whose dtype differs from out_dtype must take the eager
        route (keeping out_dtype), not raise from the fused binding or promote."""
        if not cuda_available:
            pytest.skip("CUDA required")

        x = torch.randn(512, 2048, dtype=torch.float16, device="cuda")
        weight = torch.randint(-127, 127, (512, 2048), dtype=torch.int8, device="cuda")
        wscale = torch.tensor(0.01, dtype=torch.float32, device="cuda")
        bias = torch.randn(512, dtype=torch.float16, device="cuda")
        resid = torch.randn(512, 512, dtype=torch.bfloat16, device="cuda")
        rscale = torch.randn(512, dtype=torch.bfloat16, device="cuda")

        got = ck.int8_linear(x, weight, wscale, bias, torch.float16,
                             convrot=True, convrot_groupsize=_GROUP,
                             residual=resid, residual_scale=rscale)
        assert got.dtype == torch.float16
        plain = _linear_ref(x, weight, wscale, bias)
        ref = torch.addcmul(resid.half(), plain, rscale.half())
        rel = rel_err(got, ref)
        assert rel < 1e-2, f"rel={rel:.3e}"
