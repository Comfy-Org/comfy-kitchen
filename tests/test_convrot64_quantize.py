# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the fused ConvRot rotate + row-wise INT8 quantize CUDA kernel."""

import pytest
import torch

from comfy_kitchen.backends import cuda
from comfy_kitchen.tensor.int8_utils import _build_hadamard

from .conftest import assert_values_close

GROUP_SIZE = 256


@pytest.fixture(autouse=True)
def cuda_only():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for quantize_int8_rowwise_convrot64 tests")


class TestConvrot64QuantizeShapes:
    """Shape/dtype coverage across the launcher's special cases and branches."""

    @pytest.mark.parametrize(
        "k", [256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 3072, 3584, 4096, 6144, 8192]
    )
    @pytest.mark.parametrize("m", [1, 4, 63, 9216])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    def test_shape_dtype_matrix(self, k, m, dtype):
        """Every (M, K, dtype) combination returns valid shape/dtype/range output."""
        torch.manual_seed(m * 100_000 + k)
        x = torch.randn(m, k, device="cuda", dtype=dtype)

        q, scale = cuda.quantize_int8_rowwise_convrot64(x, GROUP_SIZE)

        assert q.shape == (m, k)
        assert q.dtype == torch.int8
        assert scale.shape == (m, 1)
        assert scale.dtype == torch.float32
        assert q.min() >= -128
        assert q.max() <= 127
        assert torch.isfinite(scale).all()
        assert (scale > 0).all()

    def test_k_not_divisible_by_group_size_raises(self):
        x = torch.randn(4, 300, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(RuntimeError, match="divisible by 256"):
            cuda.quantize_int8_rowwise_convrot64(x, GROUP_SIZE)


class TestConvrot64QuantizeDeterminism:
    """Non-stochastic quantize is a pure function -- two calls must match bitwise."""

    @pytest.mark.parametrize("k", [768, 1024, 1280, 1792, 2048, 2304, 3072, 3584, 4096, 8192])
    def test_two_calls_bitwise_equal(self, k):
        torch.manual_seed(0)
        x = torch.randn(512, k, device="cuda", dtype=torch.bfloat16)

        q1, s1 = cuda.quantize_int8_rowwise_convrot64(x, GROUP_SIZE)
        q2, s2 = cuda.quantize_int8_rowwise_convrot64(x, GROUP_SIZE)

        assert torch.equal(q1, q2)
        assert torch.equal(s1, s2)


class TestConvrot64QuantizeStochasticRounding:
    """Stochastic path (seeded) must be deterministic and produce valid output."""

    @pytest.mark.parametrize("k", [768, 1024, 1280, 1792, 2048, 3072])
    def test_seeded_calls_bitwise_equal_and_valid(self, k):
        torch.manual_seed(0)
        x = torch.randn(512, k, device="cuda", dtype=torch.bfloat16)

        q1, s1 = cuda.quantize_int8_rowwise_convrot64(x, GROUP_SIZE, stochastic_rounding=424242)
        q2, s2 = cuda.quantize_int8_rowwise_convrot64(x, GROUP_SIZE, stochastic_rounding=424242)

        assert torch.equal(q1, q2)
        assert torch.equal(s1, s2)
        assert q1.min() >= -128
        assert q1.max() <= 127
        assert torch.isfinite(s1).all()
        assert (s1 > 0).all()


class TestConvrot64QuantizeScaleFloor:
    """An all-zero row's scale must floor at 1e-30, not collapse to 0."""

    @pytest.mark.parametrize("k", [1024, 3072, 4096])
    def test_all_zero_row_floors_scale(self, k):
        x = torch.zeros(8, k, device="cuda", dtype=torch.bfloat16)

        q, scale = cuda.quantize_int8_rowwise_convrot64(x, GROUP_SIZE)

        assert torch.equal(scale, torch.full_like(scale, 1.0e-30))
        assert torch.equal(q, torch.zeros_like(q))

    @pytest.mark.parametrize("k", [1024, 3072, 4096])
    def test_all_zero_row_among_normal_rows(self, k):
        """A degenerate row's scale must not affect any other row's."""
        torch.manual_seed(1)
        x = torch.randn(8, k, device="cuda", dtype=torch.bfloat16)
        x[3, :] = 0.0

        q, scale = cuda.quantize_int8_rowwise_convrot64(x, GROUP_SIZE)

        # float32-vs-float32 compare, not .item(): avoids a float64 literal mismatch.
        assert torch.equal(scale[3], torch.full_like(scale[3], 1.0e-30))
        assert torch.equal(q[3], torch.zeros(k, dtype=torch.int8, device="cuda"))
        assert (scale[torch.arange(8) != 3] > 1.0e-30).all()


class TestConvrot64QuantizeOverflowClamp:
    """A near-dtype-max row (NaN mid-butterfly) must still yield finite output."""

    @pytest.mark.parametrize("k", [1024, 3072, 4096])
    def test_near_dtype_max_row_stays_finite(self, k):
        bf16_max = 3.38953139e38
        x = torch.randn(8, k, device="cuda", dtype=torch.bfloat16)
        x[0, :] = bf16_max * 0.99
        x[1, :] = -bf16_max * 0.99

        q, scale = cuda.quantize_int8_rowwise_convrot64(x, GROUP_SIZE)

        assert torch.isfinite(scale).all()
        assert (scale > 0).all()
        assert torch.isfinite(q.float()).all()
        assert q.min() >= -128
        assert q.max() <= 127


class TestConvrot64QuantizeEagerReference:
    """CUDA kernel vs the eager (dense-matmul rotation) reference, within tolerance."""

    @pytest.mark.parametrize("k", [1024, 2048, 3072, 3584, 4096, 8192])
    def test_cuda_matches_eager_within_tolerance(self, k):
        torch.manual_seed(3)
        x = torch.randn(32, k, device="cuda", dtype=torch.bfloat16)

        q_cuda, scale_cuda = cuda.quantize_int8_rowwise_convrot64(x, GROUP_SIZE)

        h = _build_hadamard(GROUP_SIZE, device=x.device, dtype=x.dtype)
        from comfy_kitchen.backends.eager.quantization import quantize_and_rotate_rowwise

        q_eager, scale_eager = quantize_and_rotate_rowwise(x, h, GROUP_SIZE)

        assert_values_close(
            scale_cuda, scale_eager, rtol=1.0e-2, atol=1.0e-4, name="convrot64_scale"
        )
        # Rounding-boundary drift under a different summation order: ~2-4% of codes, max +/-2.
        max_code_diff = (q_cuda.to(torch.int32) - q_eager.to(torch.int32)).abs().max().item()
        n_mismatch = (q_cuda != q_eager).sum().item()
        mismatch_ratio = n_mismatch / q_cuda.numel()
        assert max_code_diff <= 3, (
            f"max int8 code diff {max_code_diff} too large vs eager reference"
        )
        assert mismatch_ratio <= 0.06, (
            f"{mismatch_ratio:.4%} of int8 codes differ from eager reference"
        )
