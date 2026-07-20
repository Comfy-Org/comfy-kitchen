# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Direct coverage for comfy_kitchen.backends.cuda.quantize_int8_rowwise_convrot64
(the fused online-ConvRot-rotation + row-wise-INT8-quantize CUDA kernel).

This kernel previously had no direct test coverage (it was only exercised
indirectly through int8_linear / TensorWiseINT8Layout convrot weight-quantize
tests). Added alongside the kernel launch-config fix (right-sized thread count
for K <= 2048, chunked two-pass for 2048 < K <= 4096, original always-1024-
thread single-pass kernel unchanged for K > 4096 -- see
launch_quantize_int8_rowwise_convrot64_kernel's docstring in int8_linear.cu for
why the chunked kernel is deliberately NOT used beyond K=4096) to pin down its
documented contract: rotation math, scale computation, rounding, and dtype
dispatch.
"""

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
    """Shape/dtype coverage across the launcher's special cases, the
    right-sized generic branch (K <= 2048, single-pass), the chunked branch
    (2048 < K <= 4096, two-pass), and the K > 4096 branch (original
    always-1024-thread single-pass kernel, unchanged -- the chunked kernel
    regressed ~8% at K=8192/M=9216 on an RTX 3090 due to pass 2's re-read no
    longer reliably hitting L2 at that row count, so it is gated off beyond
    K=4096; see int8_linear.cu's launcher docstring)."""

    @pytest.mark.parametrize(
        "k", [256, 512, 1024, 1536, 2048, 2304, 2560, 3072, 3584, 4096, 6144, 8192]
    )
    @pytest.mark.parametrize("m", [1, 4, 63, 9216])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    def test_shape_dtype_matrix(self, k, m, dtype):
        """Every (M, K, dtype) combination returns correctly-shaped, finite,
        in-range int8 output and a positive finite per-row scale. Covers the
        M==1 single-row-GEMV passthrough (untouched by the launch-config
        change), the num_cols in {256, 2560, 6144} special cases (also
        untouched), the right-sized single-pass branch (K in {512, 1024,
        1536, 2048}), the chunked two-pass branch (K in {2304, 3072, 3584,
        4096} -- 2304/256=9, 3072/256=12, and 3584/256=14 groups are NOT
        multiples of the chunked kernel's 8-groups-per-chunk width, so these
        three specifically exercise the partial-last-chunk path where some of
        the 8 "sub" lanes in the final chunk iteration are masked
        `active=false` (2304 is the most extreme case: only 1 of 8 lanes
        active in the second chunk), unlike 4096's exact 2-chunks-of-8-groups
        split), and K=8192 (back to the original always-1024-thread
        single-pass kernel, byte-identical dispatch to
        stock -- deliberately NOT chunked, see the class docstring)."""
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
    """Rotation + quantization is a pure function of the input values (no
    randomness in the non-stochastic path) -- two calls on the same input
    must be bitwise identical, including across the resident/chunked boundary
    at K=2048/2304, the chunked partial-last-chunk cases at K=3072/3584, and
    the chunked/original-single-pass boundary at K=4096/4352 (the latter
    tested implicitly via K=4096 vs K=8192 here)."""

    @pytest.mark.parametrize("k", [1024, 2048, 2304, 3072, 3584, 4096, 8192])
    def test_two_calls_bitwise_equal(self, k):
        torch.manual_seed(0)
        x = torch.randn(512, k, device="cuda", dtype=torch.bfloat16)

        q1, s1 = cuda.quantize_int8_rowwise_convrot64(x, GROUP_SIZE)
        q2, s2 = cuda.quantize_int8_rowwise_convrot64(x, GROUP_SIZE)

        assert torch.equal(q1, q2)
        assert torch.equal(s1, s2)


class TestConvrot64QuantizeScaleFloor:
    """An all-zero row rotates to all-zero (H4 is linear), so its abs-max is
    0 and the scale must floor at 1e-30 (guards div-by-zero on dequant/re-use)
    rather than collapsing to 0 itself."""

    @pytest.mark.parametrize("k", [1024, 3072, 4096])
    def test_all_zero_row_floors_scale(self, k):
        x = torch.zeros(8, k, device="cuda", dtype=torch.bfloat16)

        q, scale = cuda.quantize_int8_rowwise_convrot64(x, GROUP_SIZE)

        assert torch.equal(scale, torch.full_like(scale, 1.0e-30))
        assert torch.equal(q, torch.zeros_like(q))

    @pytest.mark.parametrize("k", [1024, 3072, 4096])
    def test_all_zero_row_among_normal_rows(self, k):
        """A single all-zero row's degenerate scale must not affect any other
        row's (rows are independent -- one block per row)."""
        torch.manual_seed(1)
        x = torch.randn(8, k, device="cuda", dtype=torch.bfloat16)
        x[3, :] = 0.0

        q, scale = cuda.quantize_int8_rowwise_convrot64(x, GROUP_SIZE)

        # Compare as float32 tensors (not scale[3].item() == 1e-30 as Python
        # floats): .item() upconverts to float64 first, and float32(1e-30)
        # upcast to float64 is not exactly the float64 literal 1e-30, so that
        # comparison would spuriously fail despite the kernel producing
        # exactly its own "1.0e-30f" floor literal.
        assert torch.equal(scale[3], torch.full_like(scale[3], 1.0e-30))
        assert torch.equal(q[3], torch.zeros(k, dtype=torch.int8, device="cuda"))
        assert (scale[torch.arange(8) != 3] > 1.0e-30).all()


class TestConvrot64QuantizeOverflowClamp:
    """A row whose magnitude is close enough to the input dtype's finite max
    that intermediate Hadamard-butterfly sums overflow to +/-Inf (and a later
    stage's Inf - Inf produces NaN) must still produce a finite, in-range
    scale and int8 output -- this is the PR #66 clamp
    (finite_absmax_for_int8_scale) this change does not touch, exercised here
    for the K shapes/launch configs this change DOES touch."""

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
    """CUDA kernel vs the eager backend's reference implementation
    (rotate-then-quantize as two separate ops, dense H @ x matmul for the
    rotation). NOT compared bitwise: the eager path builds the full
    normalized Hadamard matrix via torch.kron and rotates with a dense
    matmul, which sums the same terms in a different order (and in the
    dense-matmul epilogue's own accumulation, not the CUDA kernel's explicit
    4-stage radix-4 butterfly with a 0.5x scale per stage) than the CUDA
    kernel's fast Hadamard transform -- floating-point addition is not
    associative, so a different summation order can differ in the last few
    bits even though both are mathematically the same rotation. Compared
    with a tolerance loose enough to absorb that (not the rotation's
    correctness, which is exact up to associativity) while still catching a
    genuinely wrong rotation/quantization.
    """

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
        # Individual int8 codes can land on either side of a rounding boundary
        # under a different summation order; measured empirically (see the PR
        # description) at ~2-4% of codes differing by at most +/-2, stable
        # across K -- allow comfortable headroom above that while still
        # catching a genuinely wrong rotation/quantization (which would push
        # both numbers far higher, not shift them by a couple percent).
        max_code_diff = (q_cuda.to(torch.int32) - q_eager.to(torch.int32)).abs().max().item()
        n_mismatch = (q_cuda != q_eager).sum().item()
        mismatch_ratio = n_mismatch / q_cuda.numel()
        assert max_code_diff <= 3, (
            f"max int8 code diff {max_code_diff} too large vs eager reference"
        )
        assert mismatch_ratio <= 0.06, (
            f"{mismatch_ratio:.4%} of int8 codes differ from eager reference"
        )
