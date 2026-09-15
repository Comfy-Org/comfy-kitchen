# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""fp16_linear — CUTLASS fp16-accumulate GEMM with an optional fused residual.

The CUDA path accumulates in fp16 (the allow_fp16_accumulation numerics), so
comparisons against the fp32-accumulate torch reference use a tolerance that
grows with K: rounding error compounds roughly with sqrt(K) * 2^-11.
"""


import pytest
import torch

import comfy_kitchen as ck
from tests.conftest import cuda_backend_available, fp16_accum_tol, rel_err


class TestFp16Linear:
    # covers the identity-swizzle configs and the deep-K stream-K path
    @pytest.mark.parametrize(
        "m,n,k",
        [
            (1797, 6144, 2048),
            (1797, 2048, 2048),
            (1797, 16384, 2048),
            (1797, 2048, 8192),
            (512, 512, 512),
            (37, 264, 128),
            (37, 264, 0),   # zero-K linear is the bias broadcast; the kernel declines it
        ],
    )
    @pytest.mark.parametrize("with_bias", [True, False])
    def test_matches_fp32_accum_reference(self, m, n, k, with_bias, seed, cuda_available):
        if not cuda_available:
            pytest.skip("CUDA required")

        x = torch.randn(m, k, dtype=torch.float16, device="cuda")
        w = torch.randn(n, k, dtype=torch.float16, device="cuda") * 0.02
        b = torch.randn(n, dtype=torch.float16, device="cuda") if with_bias else None

        ref = torch.nn.functional.linear(x, w, b).float()
        got = ck.fp16_linear(x, w, b).float()
        rel = rel_err(got, ref)
        assert rel < fp16_accum_tol(k), f"rel={rel:.4f} tol={fp16_accum_tol(k):.4f}"

    @pytest.mark.parametrize("m,n,k", [(1797, 2048, 2048), (1797, 2048, 8192), (512, 512, 512)])
    @pytest.mark.parametrize("with_bias", [True, False])
    def test_residual_matches_eager_addcmul(self, m, n, k, with_bias, seed, cuda_available):
        if not cuda_available:
            pytest.skip("CUDA required")

        x = torch.randn(m, k, dtype=torch.float16, device="cuda")
        w = torch.randn(n, k, dtype=torch.float16, device="cuda") * 0.02
        b = torch.randn(n, dtype=torch.float16, device="cuda") if with_bias else None
        resid = torch.randn(m, n, dtype=torch.float16, device="cuda")
        rscale = torch.randn(n, dtype=torch.float16, device="cuda")

        # same fp16-accum GEMM either way; only the epilogue rounding differs
        plain = ck.fp16_linear(x, w, b)
        ref = torch.addcmul(resid, plain, rscale).float()
        got = ck.fp16_linear(x, w, b, residual=resid, residual_scale=rscale).float()
        rel = rel_err(got, ref)
        assert rel < 1e-2, f"rel={rel:.3e}"

    def test_3d_input(self, seed, cuda_available):
        if not cuda_available:
            pytest.skip("CUDA required")

        x = torch.randn(2, 512, 2048, dtype=torch.float16, device="cuda")
        w = torch.randn(1024, 2048, dtype=torch.float16, device="cuda") * 0.02
        b = torch.randn(1024, dtype=torch.float16, device="cuda")
        resid = torch.randn(2, 512, 1024, dtype=torch.float16, device="cuda")
        rscale = torch.randn(1024, dtype=torch.float16, device="cuda")

        got = ck.fp16_linear(x, w, b, residual=resid, residual_scale=rscale)
        assert got.shape == (2, 512, 1024)
        ref = torch.addcmul(resid, torch.nn.functional.linear(x, w, b), rscale).float()
        rel = rel_err(got, ref)
        assert rel < fp16_accum_tol(2048), f"rel={rel:.4f}"

    @pytest.mark.parametrize("case", ["unaligned_k", "misaligned_input"])
    def test_unservable_inputs_fall_back(self, case, seed, cuda_available):
        """K % 8 != 0 and a 16-byte-misaligned view cannot run the fused kernel
        (the latter would fault asynchronously) but must still be correct."""
        if not cuda_available:
            pytest.skip("CUDA required")

        if case == "unaligned_k":
            x = torch.randn(64, 132, dtype=torch.float16, device="cuda")
        else:
            base = torch.randn(64 * 2048 + 4, dtype=torch.float16, device="cuda")
            x = base[4:4 + 64 * 2048].reshape(64, 2048)
            assert x.data_ptr() % 16 == 8
        w = torch.randn(96, x.shape[1], dtype=torch.float16, device="cuda") * 0.02
        got = ck.fp16_linear(x, w, None).float()
        torch.cuda.synchronize()
        ref = torch.nn.functional.linear(x, w, None).float()
        assert ((got - ref).abs().max()).item() < 1e-3  # the fallback is torch's own linear

    def test_residual_requires_scale(self, cuda_available):
        if not cuda_available:
            pytest.skip("CUDA required")
        x = torch.randn(4, 128, dtype=torch.float16, device="cuda")
        w = torch.randn(64, 128, dtype=torch.float16, device="cuda")
        resid = torch.randn(4, 64, dtype=torch.float16, device="cuda")
        with pytest.raises(ValueError, match="residual"):
            ck.fp16_linear(x, w, None, residual=resid)

    @pytest.mark.parametrize(
        "m,n,k,served",
        [
            (1797, 6144, 2048, True),   # decoder qkv: 360 plain tiles
            (1797, 2048, 8192, True),   # decoder w2: stream-K, 120 tiles
            (7188, 16384, 2048, True),  # four batched decoder tiles: above the old 4096-row gate
            (1024, 2048, 2048, False),  # 64 plain tiles: cuBLAS split-K is ~2x faster
            (64, 1024, 4096, False),    # 4 tiles: ~10x slower than cuBLAS
            (64, 2048, 8192, False),    # stream-K but only 16 tiles
        ],
    )
    def test_small_launches_are_declined(self, m, n, k, served, seed, cuda_available):
        """The kernel declines launches too small to fill the GPU so the caller
        runs cuBLAS; the public op stays correct either way."""
        if not cuda_backend_available():
            pytest.skip("compiled CUDA backend required")
        from comfy_kitchen.backends import cuda as cuda_backend

        x = torch.randn(m, k, dtype=torch.float16, device="cuda")
        w = torch.randn(n, k, dtype=torch.float16, device="cuda") * 0.02
        out = torch.empty(m, n, dtype=torch.float16, device="cuda")
        wrap = cuda_backend._wrap_for_dlpack
        ok = cuda_backend._C.cutlass_fp16_linear(
            wrap(x), wrap(w), wrap(cuda_backend._empty_cuda_tensor(x.device, torch.float16)), wrap(out),
            torch.cuda.current_stream().cuda_stream)
        assert ok == served
        got = ck.fp16_linear(x, w, None).float()
        ref = torch.nn.functional.linear(x, w, None).float()
        assert rel_err(got, ref) < fp16_accum_tol(k)
