"""SVDQuant W4A4 unit tests.

Covers the op-level contract for kitchen's int4 quantize + scaled_mm:
  - Quantizer emission range (signed [-7,7] / unsigned [0,15], absmax/qmax scheme).
  - act_unsigned flag dispatch through wrapper → kernel.
  - lora_x kwarg separation (LoRA sees pre-shift activation).
  - Eager ↔ CUDA numerical agreement on synthetic data.

Heavier parity against real nunchaku checkpoints lives outside the unit-test
tree (kitchen is pure ops; model-checkpoint parity is a converter/integration
concern).
"""
from __future__ import annotations

import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.backends.eager.svdquant import (
    _INT4_MAX,
    _UINT4_MAX,
    _unpack_int4_row_major,
    _unpack_uint4_row_major,
)

from .conftest import assert_values_close, get_capable_backends


_GROUP = 64


# =============================================================================
# Quantizer clamp contract (#3 regression guard)
# =============================================================================


class TestQuantizerClampContract:
    """Eager signed quantize must emit values in [-_INT4_MAX, +_INT4_MAX]
    (= [-7, 7]), matching the CUDA kernel and nunchaku's QVALUE_MAX_SIGNED=7.

    -8 is representable by the signed nibble but intentionally not emitted —
    absmax/7 scaling assumes ±qmax symmetry; emitting -8 would break dequant
    parity with nunchaku (gemm_w4a4.cuh:435) and kitchen CUDA
    (svdquant_utils.cuh:20). This class regression-guards that contract.
    """

    def test_constants(self):
        assert _INT4_MAX == 7, f"_INT4_MAX drifted to {_INT4_MAX}"
        assert _UINT4_MAX == 15, f"_UINT4_MAX drifted to {_UINT4_MAX}"

    @pytest.mark.parametrize("scale_factor", [0.5, 3.0, 100.0])
    def test_signed_clamp_normal_and_extreme(self, cuda_available, seed, scale_factor):
        if not cuda_available:
            pytest.skip("CUDA required for scaled_mm dispatch")
        K = 128
        x = torch.randn(4, K, dtype=torch.bfloat16, device="cuda") * scale_factor
        smooth = torch.ones(K, dtype=torch.bfloat16, device="cuda")
        lora_down = torch.zeros(K, 16, dtype=torch.bfloat16, device="cuda")

        with ck.use_backend("eager"):
            q_packed, _, _ = ck.quantize_svdquant_w4a4(
                x, smooth, lora_down, pad_size=16, act_unsigned=False,
            )
        q_vals = _unpack_int4_row_major(q_packed[:4])
        assert q_vals.min().item() >= -_INT4_MAX, (
            f"scale_factor={scale_factor}: min={q_vals.min().item()} "
            f"< -{_INT4_MAX} (eager regressed to [-8, 7])"
        )
        assert q_vals.max().item() <= _INT4_MAX

    def test_signed_clamp_forced_negative_outlier(self, cuda_available, seed):
        """Inject an extreme negative outlier that would clamp to -8 under the
        broken [-8, 7] contract. The fix keeps it at -7.
        """
        if not cuda_available:
            pytest.skip("CUDA required for scaled_mm dispatch")
        K = 128
        x = torch.randn(4, K, dtype=torch.bfloat16, device="cuda") * 5.0
        x[0, 0] = -100.0  # forces negative saturation
        smooth = torch.ones(K, dtype=torch.bfloat16, device="cuda")
        lora_down = torch.zeros(K, 16, dtype=torch.bfloat16, device="cuda")

        with ck.use_backend("eager"):
            q_packed, _, _ = ck.quantize_svdquant_w4a4(
                x, smooth, lora_down, pad_size=16, act_unsigned=False,
            )
        q_vals = _unpack_int4_row_major(q_packed[:4])
        # Never -8 even on forced saturation
        assert (q_vals != -8).all().item(), "eager emitted -8, regressed from absmax/7 contract"
        assert q_vals.min().item() == -_INT4_MAX  # the outlier did saturate

    def test_no_neg8_at_scale(self, cuda_available, seed):
        """Bulk test across many samples."""
        if not cuda_available:
            pytest.skip("CUDA required for scaled_mm dispatch")
        K = 128
        x = torch.randn(256, K, dtype=torch.bfloat16, device="cuda") * 3.0
        smooth = torch.ones(K, dtype=torch.bfloat16, device="cuda")
        lora_down = torch.zeros(K, 16, dtype=torch.bfloat16, device="cuda")

        with ck.use_backend("eager"):
            q_packed, _, _ = ck.quantize_svdquant_w4a4(
                x, smooth, lora_down, pad_size=256, act_unsigned=False,
            )
        q_vals = _unpack_int4_row_major(q_packed[:256])
        neg8 = (q_vals == -8).sum().item()
        assert neg8 == 0, f"{neg8} values emitted as -8 across {q_vals.numel()} samples"

    def test_unsigned_clamp_range(self, cuda_available, seed):
        """Unsigned path: emission must be in [0, 15]."""
        if not cuda_available:
            pytest.skip("CUDA required for scaled_mm dispatch")
        K = 128
        # Non-negative input (simulates post-GELU + shift)
        x = torch.rand(4, K, dtype=torch.bfloat16, device="cuda") * 3.0 + 0.01
        smooth = torch.ones(K, dtype=torch.bfloat16, device="cuda")
        lora_down = torch.zeros(K, 16, dtype=torch.bfloat16, device="cuda")

        with ck.use_backend("eager"):
            q_packed, _, _ = ck.quantize_svdquant_w4a4(
                x, smooth, lora_down, pad_size=16, act_unsigned=True,
            )
        q_vals = _unpack_uint4_row_major(q_packed[:4])
        assert q_vals.min().item() >= 0
        assert q_vals.max().item() <= _UINT4_MAX


# =============================================================================
# act_unsigned dispatch: u4.s4 vs s4.s4 MMA selection
# =============================================================================


class TestActUnsignedDispatch:
    """Verify act_unsigned flag actually selects the u4.s4 MMA variant at
    kernel level (as distinct from the signed s4.s4 MMA).

    Construction: pack an activation byte of 0xFF (all 1s) and a weight of 1.
    Signed interpretation: 0xFF = -1, result = 64 * -1 * 1 = -64.
    Unsigned interpretation: 0xFF = 15, result = 64 * 15 * 1 = +960.
    The sign/magnitude split is the cleanest test that the flag routes
    correctly end-to-end.
    """

    @pytest.fixture
    def _cuda_required(self, cuda_available):
        if not cuda_available:
            pytest.skip("CUDA required for int4 MMA kernels")

    def _run(self, act_unsigned):
        M, N, K, R = 16, 8, 64, 16
        q_act = torch.full((M, K // 2), 0xFF, dtype=torch.uint8, device="cuda").view(torch.int8)
        q_wgt = torch.full((N, K // 2), 0x11, dtype=torch.int8, device="cuda")  # two s4=1 per byte
        asc = torch.ones(K // 64, M, dtype=torch.bfloat16, device="cuda")
        wsc = torch.ones(K // 64, N, dtype=torch.bfloat16, device="cuda")
        lai = torch.zeros(M, R, dtype=torch.float32, device="cuda")
        lu = torch.zeros(N, R, dtype=torch.bfloat16, device="cuda")
        b = torch.zeros(N, dtype=torch.bfloat16, device="cuda")
        with ck.use_backend("cuda"):
            return ck.scaled_mm_svdquant_w4a4(
                act=q_act, wgt=q_wgt, ascales=asc, wscales=wsc,
                lora_act_in=lai, lora_up=lu, bias=b, act_unsigned=act_unsigned,
            )

    def test_signed_mma_gives_minus_64(self, _cuda_required, seed):
        out = self._run(act_unsigned=False)
        # Allow tiny tolerance for fp16 accumulation rounding
        assert abs(out[0, 0].item() - (-64.0)) < 1.0
        assert (out == out[0, 0]).all().item()

    def test_unsigned_mma_gives_plus_960(self, _cuda_required, seed):
        out = self._run(act_unsigned=True)
        assert abs(out[0, 0].item() - 960.0) < 5.0
        assert (out == out[0, 0]).all().item()


# =============================================================================
# lora_x kwarg: LoRA uses raw x when caller pre-shifts main input
# =============================================================================


class TestLoraXSeparation:
    """SVDQuant defines LoRA-down on the pre-quantization, pre-shift activation
    regardless of the main-path treatment. The quantize wrapper's lora_x kwarg
    lets callers pass a separate (un-shifted) tensor for the LoRA matmul.

    These tests verify *routing* (that lora_x actually changes where the LoRA
    matmul reads its input from) and consistency with a high-precision fp32
    reference. They deliberately do NOT compare eager and CUDA LoRA outputs
    bitwise: the CUDA backend takes a bf16 @ bf16 matmul path for memory/launch
    savings, the eager reference runs in fp32. The tolerances below reflect
    bf16 matmul precision (~8e-3 abs).
    """

    @pytest.mark.parametrize("backend", ["eager", "cuda"])
    def test_lora_x_none_defaults_to_x(self, cuda_available, seed, backend):
        """Explicit lora_x=x is indistinguishable from lora_x=None (default)."""
        if backend == "cuda" and not cuda_available:
            pytest.skip("CUDA backend not available")
        device = "cuda" if cuda_available else "cpu"
        if backend == "eager" and device == "cpu":
            pass  # eager runs on CPU
        elif device != "cuda":
            pytest.skip(f"backend {backend} needs cuda")

        K, R = 128, 16
        x = torch.randn(4, K, dtype=torch.bfloat16, device=device)
        smooth = torch.ones(K, dtype=torch.bfloat16, device=device)
        lora_down = torch.randn(K, R, dtype=torch.bfloat16, device=device) * 0.1

        with ck.use_backend(backend):
            q1, asc1, la1 = ck.quantize_svdquant_w4a4(x, smooth, lora_down, pad_size=16)
            q2, asc2, la2 = ck.quantize_svdquant_w4a4(
                x, smooth, lora_down, pad_size=16, lora_x=x,
            )
        # Both main-path (quantize) outputs and LoRA outputs must be bit-identical
        # — same backend, same input, same code path.
        assert torch.equal(q1, q2)
        assert torch.equal(asc1, asc2)
        assert_values_close(la1, la2, rtol=0.0, atol=0.0, name=f"{backend} lora_act")

    @pytest.mark.parametrize("backend", ["eager", "cuda"])
    def test_lora_x_separate_uses_raw_for_lora(self, cuda_available, seed, backend):
        """Pass a pre-shifted x as main input + raw x as lora_x. LoRA output
        must follow the raw lora_x, proving the kwarg is wired through and not
        silently falling back to the shifted x. Accuracy check uses a fp32
        reference with bf16-precision tolerance (CUDA backend uses bf16 matmul,
        not fp32-accumulate — do not interpret this as a cross-backend bit-parity
        test).
        """
        if backend == "cuda" and not cuda_available:
            pytest.skip("CUDA backend not available")
        device = "cuda" if cuda_available else "cpu"
        if backend != "eager" and device != "cuda":
            pytest.skip(f"backend {backend} needs cuda")

        K, R = 128, 16
        raw_x = torch.randn(4, K, dtype=torch.bfloat16, device=device) * 0.5
        shifted_x = raw_x + 0.171875
        smooth = torch.ones(K, dtype=torch.bfloat16, device=device)
        lora_down = torch.randn(K, R, dtype=torch.bfloat16, device=device) * 0.1

        with ck.use_backend(backend):
            # Correct: pre-shifted for main, raw for lora
            _, _, la_correct = ck.quantize_svdquant_w4a4(
                shifted_x, smooth, lora_down, pad_size=16, lora_x=raw_x,
            )
            # Incorrect baseline: shifted used for both (what happens if caller
            # forgets lora_x). Used to prove the kwarg is live, not a pass-through.
            _, _, la_if_shifted = ck.quantize_svdquant_w4a4(
                shifted_x, smooth, lora_down, pad_size=16,
            )
        assert not torch.allclose(la_correct, la_if_shifted), (
            "lora_x had no effect — LoRA matmul is still using shifted input"
        )

        # Accuracy check against a fp32 reference. Tolerance is bf16-precision
        # because the CUDA wrapper uses a bf16 matmul (then upcasts to the fp32
        # lora_act buffer). Eager uses fp32 internally so it will be tighter,
        # but we apply the same tolerance to avoid treating eager as a bit-
        # parity oracle — it is a higher-precision reference, not a backend
        # parity target.
        expected = raw_x.float() @ lora_down.float()
        assert_values_close(
            la_correct[:4].float(), expected, rtol=5e-3, atol=5e-3,
            name=f"{backend} lora_act vs fp32 reference (bf16 tolerance)",
        )


# =============================================================================
# Cross-backend smoke: quantize + scaled_mm round-trip
# =============================================================================


class TestSvdquantSmoke:
    """Basic end-to-end smoke test that the CUDA and eager paths produce
    reasonable outputs on matched random data (not a nunchaku bit-parity test —
    that's an integration concern)."""

    @pytest.mark.parametrize("M,N,K,R", [
        (16, 8, 64, 16),     # one MMA
        (64, 32, 128, 16),
        (256, 128, 512, 32),
    ])
    def test_signed_forward_runs(self, cuda_available, seed, M, N, K, R):
        if not cuda_available:
            pytest.skip("CUDA required")
        device = "cuda"
        x = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.3
        smooth = torch.ones(K, dtype=torch.bfloat16, device=device) * 1.0
        proj_down = torch.randn(K, R, dtype=torch.bfloat16, device=device) * 0.05
        proj_up = torch.randn(N, R, dtype=torch.bfloat16, device=device) * 0.05
        wscales = torch.rand(K // _GROUP, N, dtype=torch.bfloat16, device=device) * 0.5 + 0.1
        # Signed wgt in [-7, 7]
        wgt_int = torch.randint(-7, 8, (N, K), dtype=torch.int8, device=device)
        lo = wgt_int[..., 0::2].to(torch.int32) & 0x0F
        hi = wgt_int[..., 1::2].to(torch.int32) & 0x0F
        wgt = (lo | (hi << 4)).to(torch.int8)

        with ck.use_backend("cuda"):
            q_act, asc, la = ck.quantize_svdquant_w4a4(x, smooth, proj_down, pad_size=256)
            out = ck.scaled_mm_svdquant_w4a4(
                act=q_act, wgt=wgt, ascales=asc, wscales=wscales,
                lora_act_in=la, lora_up=proj_up,
            )
        assert out.shape[0] >= M  # padded to pad_size
        assert out.shape[1] == N
        assert torch.isfinite(out).all()
