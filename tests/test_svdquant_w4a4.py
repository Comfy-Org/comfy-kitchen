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
import torch.nn.functional as functional

import comfy_kitchen as ck
from comfy_kitchen.backends.eager.svdquant import (
    _INT4_MAX,
    _UINT4_MAX,
    _unpack_int4_row_major,
    _unpack_uint4_row_major,
)
from comfy_kitchen.tensor import (
    QuantizedTensor,
    TensorCoreSVDQuantW4A4Layout,
    svdquant_w4a4_can_share_quant,
    svdquant_w4a4_fuse_linear_weights,
    svdquant_w4a4_fused_grouped_linear,
    svdquant_w4a4_grouped_linear,
)
from comfy_kitchen.tensor.svdquant_w4a4 import _cuda_quantize_pad_size

from .conftest import assert_values_close

_GROUP = 64
_TILE_BN = 128
_TILE_INTERLEAVE = 4


def test_svdquant_lora_down_transposed_gemm_policy():
    from comfy_kitchen.backends.cuda import _svdquant_use_transposed_lora_down

    assert not _svdquant_use_transposed_lora_down(1, 32)
    assert _svdquant_use_transposed_lora_down(16, 16)
    assert not _svdquant_use_transposed_lora_down(24, 16)
    assert _svdquant_use_transposed_lora_down(15, 32)
    assert not _svdquant_use_transposed_lora_down(16, 32)
    assert _svdquant_use_transposed_lora_down(4, 64)
    assert not _svdquant_use_transposed_lora_down(8, 64)
    assert _svdquant_use_transposed_lora_down(3, 128)
    assert not _svdquant_use_transposed_lora_down(4, 128)
    assert not _svdquant_use_transposed_lora_down(2, 0)


def test_svdquant_grouped_fuse_cache_is_enabled_by_default(monkeypatch):
    import comfy_kitchen.tensor.svdquant_w4a4 as svd_mod

    monkeypatch.delenv("COMFY_KITCHEN_SVDQUANT_GROUPED_FUSE", raising=False)
    assert svd_mod._svdquant_grouped_fuse_enabled()
    monkeypatch.setenv("COMFY_KITCHEN_SVDQUANT_GROUPED_FUSE", "0")
    assert not svd_mod._svdquant_grouped_fuse_enabled()


def _pack_tile_packed_weight(w_int4: torch.Tensor) -> torch.Tensor:
    """Pack dense signed-int4 (N, K) values into kitchen_tile_packed_w4a4."""
    n, k = w_int4.shape
    if n % _TILE_BN != 0:
        raise ValueError(f"N={n} must be divisible by {_TILE_BN}")
    if k % _GROUP != 0:
        raise ValueError(f"K={k} must be divisible by {_GROUP}")
    w = w_int4.view(
        n // _TILE_BN, _TILE_BN // _TILE_INTERLEAVE, _TILE_INTERLEAVE, k // _GROUP, _GROUP,
    ).permute(0, 3, 1, 2, 4).contiguous()
    lo = w[..., 0::2].to(torch.int32) & 0x0F
    hi = (w[..., 1::2].to(torch.int32) & 0x0F) << 4
    return (lo | hi).to(torch.int8).view(
        n // _TILE_BN, k // _GROUP, _TILE_BN // _TILE_INTERLEAVE,
        _TILE_INTERLEAVE * _GROUP // 2,
    )


def _pack_n_interleaved(t: torch.Tensor) -> torch.Tensor:
    """Pack natural (N, ...) tensors into (N/128, ..., 128)."""
    n = t.shape[0]
    if n % _TILE_BN != 0:
        raise ValueError(f"N={n} must be divisible by {_TILE_BN}")
    return t.view(n // _TILE_BN, _TILE_BN, *t.shape[1:]).movedim(1, -1).contiguous()


def _pack_natural_int4(w_int4: torch.Tensor) -> torch.Tensor:
    lo = w_int4[..., 0::2].to(torch.int32) & 0x0F
    hi = (w_int4[..., 1::2].to(torch.int32) & 0x0F) << 4
    return (lo | hi).to(torch.int8)


def _make_svdquant_qtensor(
    *,
    n: int,
    k: int,
    r: int,
    smooth: torch.Tensor,
    proj_down: torch.Tensor,
    dtype: torch.dtype,
    device: str,
) -> tuple[QuantizedTensor, torch.Tensor]:
    wgt_int = torch.randint(-7, 8, (n, k), dtype=torch.int8, device=device)
    wgt = _pack_natural_int4(wgt_int)
    wscales = torch.rand(k // _GROUP, n, dtype=dtype, device=device) * 0.5 + 0.1
    proj_up = torch.randn(n, r, dtype=dtype, device=device) * 0.05
    bias = torch.randn(n, dtype=dtype, device=device) * 0.01
    params = TensorCoreSVDQuantW4A4Layout.Params(
        scale=wscales,
        orig_dtype=dtype,
        orig_shape=(n, k),
        proj_down=proj_down,
        proj_up=proj_up,
        smooth_factor=smooth,
    )
    return QuantizedTensor(wgt, "TensorCoreSVDQuantW4A4Layout", params), bias


# =============================================================================
# Runtime split-QKV grouping
# =============================================================================


class TestGroupedSplitQKV:
    def test_grouped_linear_matches_individual_eager(self, seed):
        m0, m1, n, k, r = 2, 7, 128, 128, 16
        dtype = torch.float32
        device = "cpu"
        x = torch.randn(m0, m1, k, dtype=dtype, device=device) * 0.3
        smooth = torch.rand(k, dtype=dtype, device=device) * 0.5 + 0.75
        proj_down = torch.randn(k, r, dtype=dtype, device=device) * 0.05

        weights = []
        biases = []
        for _ in range(3):
            weight, bias = _make_svdquant_qtensor(
                n=n,
                k=k,
                r=r,
                smooth=smooth.clone(),
                proj_down=proj_down.clone(),
                dtype=dtype,
                device=device,
            )
            weights.append(weight)
            biases.append(bias)

        assert not svdquant_w4a4_can_share_quant(weights, validate=False)
        assert svdquant_w4a4_can_share_quant(weights, validate=True)

        with ck.use_backend("eager"):
            grouped = svdquant_w4a4_grouped_linear(
                x, weights, biases, validate_shared_quant=True,
            )
            expected = tuple(
                functional.linear(x, weight, bias)
                for weight, bias in zip(weights, biases, strict=True)
            )

        assert len(grouped) == 3
        for idx, (actual, ref) in enumerate(zip(grouped, expected, strict=True)):
            assert_values_close(actual, ref, rtol=0.0, atol=0.0, name=f"grouped qkv {idx}")

    def test_grouped_linear_can_trust_prevalidated_copies(self, seed):
        n, k, r = 128, 128, 16
        dtype = torch.float32
        device = "cpu"
        smooth = torch.rand(k, dtype=dtype, device=device) * 0.5 + 0.75
        proj_down = torch.randn(k, r, dtype=dtype, device=device) * 0.05
        weights = [
            _make_svdquant_qtensor(
                n=n,
                k=k,
                r=r,
                smooth=smooth.clone(),
                proj_down=proj_down.clone(),
                dtype=dtype,
                device=device,
            )[0]
            for _ in range(3)
        ]

        assert not svdquant_w4a4_can_share_quant(weights, validate=False)
        assert svdquant_w4a4_can_share_quant(weights, validate=True)
        assert svdquant_w4a4_can_share_quant(weights, trust=True)

    def test_runtime_fused_qkv_matches_individual_eager(self, seed):
        m0, m1, n, k, r = 2, 5, 128, 128, 16
        dtype = torch.float32
        device = "cpu"
        x = torch.randn(m0, m1, k, dtype=dtype, device=device) * 0.3
        smooth = torch.rand(k, dtype=dtype, device=device) * 0.5 + 0.75
        proj_down = torch.randn(k, r, dtype=dtype, device=device) * 0.05

        weights = []
        biases = []
        for _ in range(3):
            weight, bias = _make_svdquant_qtensor(
                n=n,
                k=k,
                r=r,
                smooth=smooth.clone(),
                proj_down=proj_down.clone(),
                dtype=dtype,
                device=device,
            )
            weights.append(weight)
            biases.append(bias)

        fused_weight, fused_bias, splits = svdquant_w4a4_fuse_linear_weights(
            weights, biases, validate_shared_quant=True,
        )
        assert splits == (n, n, n)
        assert fused_weight.shape == (3 * n, k)
        assert fused_bias.shape == (3 * n,)

        with ck.use_backend("eager"):
            fused = svdquant_w4a4_fused_grouped_linear(x, fused_weight, fused_bias, splits)
            expected = tuple(
                functional.linear(x, weight, bias)
                for weight, bias in zip(weights, biases, strict=True)
            )

        for idx, (actual, ref) in enumerate(zip(fused, expected, strict=True)):
            assert_values_close(actual, ref, rtol=0.0, atol=0.0, name=f"fused qkv {idx}")

    def test_grouped_linear_auto_fuses_cuda_and_reuses_cache(self, cuda_available, seed, monkeypatch):
        if not cuda_available:
            pytest.skip("CUDA required")

        import comfy_kitchen.tensor.svdquant_w4a4 as svd_mod
        from comfy_kitchen.backends import cuda as cuda_backend

        if not getattr(cuda_backend, "_EXT_AVAILABLE", False):
            pytest.skip("CUDA extension required")

        m, n, k, r = 8, 128, 128, 16
        dtype = torch.bfloat16
        device = "cuda"
        x = torch.randn(m, k, dtype=dtype, device=device) * 0.3
        smooth = torch.rand(k, dtype=dtype, device=device) * 0.5 + 0.75
        proj_down = torch.randn(k, r, dtype=dtype, device=device) * 0.05

        weights = []
        biases = []
        for _ in range(3):
            weight, bias = _make_svdquant_qtensor(
                n=n,
                k=k,
                r=r,
                smooth=smooth,
                proj_down=proj_down,
                dtype=dtype,
                device=device,
            )
            weights.append(weight)
            biases.append(bias)

        monkeypatch.setenv("COMFY_KITCHEN_SVDQUANT_GROUPED_FUSE", "0")
        expected = svdquant_w4a4_grouped_linear(x, weights, biases, assume_shared_quant=True)

        calls = {"fuse": 0}
        orig_fuse = svd_mod.svdquant_w4a4_fuse_linear_weights

        def spy_fuse(*args, **kwargs):
            calls["fuse"] += 1
            return orig_fuse(*args, **kwargs)

        monkeypatch.setattr(svd_mod, "svdquant_w4a4_fuse_linear_weights", spy_fuse)
        monkeypatch.setenv("COMFY_KITCHEN_SVDQUANT_GROUPED_FUSE", "1")

        actual = svdquant_w4a4_grouped_linear(x, weights, biases, assume_shared_quant=True)
        again = svdquant_w4a4_grouped_linear(x, weights, biases, assume_shared_quant=True)

        assert calls == {"fuse": 1}
        assert getattr(weights[0], "_ck_svdquant_fused_group_cache", None) is not None
        for idx, (got, ref) in enumerate(zip(actual, expected, strict=True)):
            assert_values_close(got, ref, rtol=0.0, atol=0.0, name=f"auto-fused grouped {idx}")
        for idx, (got, ref) in enumerate(zip(again, expected, strict=True)):
            assert_values_close(got, ref, rtol=0.0, atol=0.0, name=f"cached auto-fused grouped {idx}")

        new_biases = [bias.clone() for bias in biases]
        new_biases[0].add_(0.125)
        monkeypatch.setenv("COMFY_KITCHEN_SVDQUANT_GROUPED_FUSE", "0")
        expected_after_replacement = svdquant_w4a4_grouped_linear(
            x, weights, new_biases, assume_shared_quant=True,
        )
        monkeypatch.setenv("COMFY_KITCHEN_SVDQUANT_GROUPED_FUSE", "1")
        after_replacement = svdquant_w4a4_grouped_linear(
            x, weights, new_biases, assume_shared_quant=True,
        )

        assert calls == {"fuse": 2}
        for idx, (got, ref) in enumerate(zip(after_replacement, expected_after_replacement, strict=True)):
            assert_values_close(got, ref, rtol=0.0, atol=0.0, name=f"replaced auto-fused grouped {idx}")


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
        k = 128
        x = torch.randn(4, k, dtype=torch.bfloat16, device="cuda") * scale_factor
        smooth = torch.ones(k, dtype=torch.bfloat16, device="cuda")
        lora_down = torch.zeros(k, 16, dtype=torch.bfloat16, device="cuda")

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
        k = 128
        x = torch.randn(4, k, dtype=torch.bfloat16, device="cuda") * 5.0
        x[0, 0] = -100.0  # forces negative saturation
        smooth = torch.ones(k, dtype=torch.bfloat16, device="cuda")
        lora_down = torch.zeros(k, 16, dtype=torch.bfloat16, device="cuda")

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
        k = 128
        x = torch.randn(256, k, dtype=torch.bfloat16, device="cuda") * 3.0
        smooth = torch.ones(k, dtype=torch.bfloat16, device="cuda")
        lora_down = torch.zeros(k, 16, dtype=torch.bfloat16, device="cuda")

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
        k = 128
        # Non-negative input (simulates post-GELU + shift)
        x = torch.rand(4, k, dtype=torch.bfloat16, device="cuda") * 3.0 + 0.01
        smooth = torch.ones(k, dtype=torch.bfloat16, device="cuda")
        lora_down = torch.zeros(k, 16, dtype=torch.bfloat16, device="cuda")

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
        m, n, k, r = 16, 8, 64, 16
        q_act = torch.full((m, k // 2), 0xFF, dtype=torch.uint8, device="cuda").view(torch.int8)
        q_wgt = torch.full((n, k // 2), 0x11, dtype=torch.int8, device="cuda")  # two s4=1 per byte
        asc = torch.ones(k // 64, m, dtype=torch.bfloat16, device="cuda")
        wsc = torch.ones(k // 64, n, dtype=torch.bfloat16, device="cuda")
        lai = torch.zeros(m, r, dtype=torch.float32, device="cuda")
        lu = torch.zeros(n, r, dtype=torch.bfloat16, device="cuda")
        b = torch.zeros(n, dtype=torch.bfloat16, device="cuda")
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

        k, r = 128, 16
        x = torch.randn(4, k, dtype=torch.bfloat16, device=device)
        smooth = torch.ones(k, dtype=torch.bfloat16, device=device)
        lora_down = torch.randn(k, r, dtype=torch.bfloat16, device=device) * 0.1

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

        k, r = 128, 16
        raw_x = torch.randn(4, k, dtype=torch.bfloat16, device=device) * 0.5
        shifted_x = raw_x + 0.171875
        smooth = torch.ones(k, dtype=torch.bfloat16, device=device)
        lora_down = torch.randn(k, r, dtype=torch.bfloat16, device=device) * 0.1

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

    @pytest.mark.parametrize("m,n,k,r", [
        (16, 8, 64, 16),     # one MMA
        (64, 32, 128, 16),
        (256, 128, 512, 32),
    ])
    def test_signed_forward_runs(self, cuda_available, seed, m, n, k, r):
        if not cuda_available:
            pytest.skip("CUDA required")
        device = "cuda"
        x = torch.randn(m, k, dtype=torch.bfloat16, device=device) * 0.3
        smooth = torch.ones(k, dtype=torch.bfloat16, device=device) * 1.0
        proj_down = torch.randn(k, r, dtype=torch.bfloat16, device=device) * 0.05
        proj_up = torch.randn(n, r, dtype=torch.bfloat16, device=device) * 0.05
        wscales = torch.rand(k // _GROUP, n, dtype=torch.bfloat16, device=device) * 0.5 + 0.1
        # Signed wgt in [-7, 7]
        wgt_int = torch.randint(-7, 8, (n, k), dtype=torch.int8, device=device)
        lo = wgt_int[..., 0::2].to(torch.int32) & 0x0F
        hi = wgt_int[..., 1::2].to(torch.int32) & 0x0F
        wgt = (lo | (hi << 4)).to(torch.int8)

        with ck.use_backend("cuda"):
            q_act, asc, la = ck.quantize_svdquant_w4a4(x, smooth, proj_down, pad_size=256)
            out = ck.scaled_mm_svdquant_w4a4(
                act=q_act, wgt=wgt, ascales=asc, wscales=wscales,
                lora_act_in=la, lora_up=proj_up,
            )
        assert out.shape[0] >= m  # padded to pad_size
        assert out.shape[1] == n
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("fast_accum", [False, True])
    def test_tile_packed_matches_natural_cuda(self, cuda_available, seed, monkeypatch, fast_accum):
        """Tile-packed storage should be a layout-only change vs natural CUDA."""
        if not cuda_available:
            pytest.skip("CUDA required")
        if fast_accum:
            monkeypatch.setenv("COMFY_KITCHEN_SVDQUANT_FAST_ACCUM", "1")
        else:
            monkeypatch.delenv("COMFY_KITCHEN_SVDQUANT_FAST_ACCUM", raising=False)

        m, n, k, r = 64, 256, 256, 32
        device = "cuda"
        x = torch.randn(m, k, dtype=torch.bfloat16, device=device) * 0.3
        smooth = torch.rand(k, dtype=torch.bfloat16, device=device) * 0.5 + 0.75
        proj_down = torch.randn(k, r, dtype=torch.bfloat16, device=device) * 0.05
        proj_up = torch.randn(n, r, dtype=torch.bfloat16, device=device) * 0.05
        bias = torch.randn(n, dtype=torch.bfloat16, device=device) * 0.01
        wscales = torch.rand(k // _GROUP, n, dtype=torch.bfloat16, device=device) * 0.5 + 0.1

        wgt_int = torch.randint(-7, 8, (n, k), dtype=torch.int8, device=device)
        lo = wgt_int[..., 0::2].to(torch.int32) & 0x0F
        hi = (wgt_int[..., 1::2].to(torch.int32) & 0x0F) << 4
        wgt_natural = (lo | hi).to(torch.int8)
        wgt_tile = _pack_tile_packed_weight(wgt_int)
        wscales_tile = _pack_n_interleaved(wscales.t().contiguous())
        proj_up_tile = _pack_n_interleaved(proj_up)

        with ck.use_backend("cuda"):
            q_act, asc, la = ck.quantize_svdquant_w4a4(x, smooth, proj_down, pad_size=64)
            out_natural = ck.scaled_mm_svdquant_w4a4(
                act=q_act, wgt=wgt_natural, ascales=asc, wscales=wscales,
                lora_act_in=la, lora_up=proj_up, bias=bias,
            )
            out_tile = ck.scaled_mm_svdquant_w4a4(
                act=q_act, wgt=wgt_tile, ascales=asc, wscales=wscales_tile,
                lora_act_in=la, lora_up=proj_up_tile, bias=bias,
            )

        assert out_tile.shape == out_natural.shape
        assert_values_close(out_tile, out_natural, rtol=0.0, atol=0.0, name="tile-packed vs natural")

    @pytest.mark.parametrize("layout", ["natural", "tile"])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_fused_lora_epilogue_matches_unfused_cuda(self, cuda_available, seed, monkeypatch, layout, dtype):
        """Fused LoRA-up should match the old cuBLAS addmm_ epilogue to output dtype precision."""
        if not cuda_available:
            pytest.skip("CUDA required")

        m, n, k, r = 64, 256, 256, 32
        device = "cuda"
        q_act_i = torch.randint(-7, 8, (m, k), dtype=torch.int8, device=device)
        wgt_i = torch.randint(-7, 8, (n, k), dtype=torch.int8, device=device)
        q_act = _pack_natural_int4(q_act_i)
        wgt = _pack_natural_int4(wgt_i)
        asc = torch.rand(k // _GROUP, m, dtype=dtype, device=device) * 0.05 + 0.005
        wscales = torch.rand(k // _GROUP, n, dtype=dtype, device=device) * 0.05 + 0.005
        lora_act = torch.randn(m, r, dtype=dtype, device=device) * 0.1
        lora_up = torch.randn(n, r, dtype=dtype, device=device) * 0.1
        bias = torch.randn(n, dtype=dtype, device=device) * 0.01

        if layout == "tile":
            wgt = _pack_tile_packed_weight(wgt_i)
            wscales = _pack_n_interleaved(wscales.t().contiguous())
            lora_up = _pack_n_interleaved(lora_up)

        with ck.use_backend("cuda"):
            monkeypatch.setenv("COMFY_KITCHEN_SVDQUANT_FUSE_LORA_UP", "0")
            unfused = ck.scaled_mm_svdquant_w4a4(
                q_act, wgt, asc, wscales, lora_act, lora_up, bias,
            )
            monkeypatch.setenv("COMFY_KITCHEN_SVDQUANT_FUSE_LORA_UP", "1")
            fused = ck.scaled_mm_svdquant_w4a4(
                q_act, wgt, asc, wscales, lora_act, lora_up, bias,
            )

        atol = 1e-2 if dtype is torch.bfloat16 else 1.2e-3
        assert_values_close(
            fused.float(), unfused.float(), rtol=0.0, atol=atol,
            name=f"fused LoRA epilogue vs unfused ({layout}, {dtype})",
        )

    def test_rank0_lora_quantize_returns_empty_activation(self, cuda_available, seed):
        if not cuda_available:
            pytest.skip("CUDA required")

        m, k, r = 13, 256, 0
        dtype = torch.bfloat16
        device = "cuda"
        x = torch.randn(m, k, dtype=dtype, device=device) * 0.1
        smooth = torch.rand(k, dtype=dtype, device=device) * 0.1 + 0.9
        lora_down = torch.empty(k, r, dtype=dtype, device=device)

        with ck.use_backend("cuda"):
            q_act, ascales, lora_act = ck.quantize_svdquant_w4a4(
                x, smooth, lora_down, pad_size=64,
            )

        assert q_act.shape == (64, k // 2)
        assert ascales.shape == (k // _GROUP, 64)
        assert lora_act.shape == (64, 0)

    @pytest.mark.parametrize("layout", ["natural", "tile"])
    def test_rank0_lora_skips_unfused_addmm(self, cuda_available, seed, monkeypatch, layout):
        """Rank-0 LoRA is a no-op and should match the fused kernel path."""
        if not cuda_available:
            pytest.skip("CUDA required")

        m, n, k, r = 64, 256, 256, 0
        dtype = torch.bfloat16
        device = "cuda"
        q_act_i = torch.randint(-7, 8, (m, k), dtype=torch.int8, device=device)
        wgt_i = torch.randint(-7, 8, (n, k), dtype=torch.int8, device=device)
        q_act = _pack_natural_int4(q_act_i)
        wgt = _pack_natural_int4(wgt_i)
        asc = torch.rand(k // _GROUP, m, dtype=dtype, device=device) * 0.05 + 0.005
        wscales = torch.rand(k // _GROUP, n, dtype=dtype, device=device) * 0.05 + 0.005
        lora_act = torch.empty(m, r, dtype=dtype, device=device)
        lora_up = torch.empty(n, r, dtype=dtype, device=device)
        bias = torch.randn(n, dtype=dtype, device=device) * 0.01

        if layout == "tile":
            wgt = _pack_tile_packed_weight(wgt_i)
            wscales = _pack_n_interleaved(wscales.t().contiguous())
            lora_up = _pack_n_interleaved(lora_up)

        with ck.use_backend("cuda"):
            monkeypatch.setenv("COMFY_KITCHEN_SVDQUANT_FUSE_LORA_UP", "0")
            unfused = ck.scaled_mm_svdquant_w4a4(
                q_act, wgt, asc, wscales, lora_act, lora_up, bias,
            )
            monkeypatch.setenv("COMFY_KITCHEN_SVDQUANT_FUSE_LORA_UP", "1")
            fused = ck.scaled_mm_svdquant_w4a4(
                q_act, wgt, asc, wscales, lora_act, lora_up, bias,
            )

        assert torch.equal(unfused, fused)

    def test_quantized_tensor_direct_cuda_survives_disabled_registry(self, cuda_available, seed, monkeypatch):
        """ComfyUI may globally disable comfy_kitchen's CUDA registry entry on
        older PyTorch CUDA wheels. SVDQuant QuantizedTensor forward should still
        call the locally built CUDA extension directly when it is available.
        """
        if not cuda_available:
            pytest.skip("CUDA required")

        from comfy_kitchen.backends import cuda as cuda_backend

        if not getattr(cuda_backend, "_EXT_AVAILABLE", False):
            pytest.skip("CUDA extension required")

        m, n, k, r = 32, 128, 128, 16
        device = "cuda"
        dtype = torch.bfloat16
        x = torch.randn(m, k, dtype=dtype, device=device) * 0.3
        smooth = torch.rand(k, dtype=dtype, device=device) * 0.5 + 0.75
        proj_down = torch.randn(k, r, dtype=dtype, device=device) * 0.05
        proj_up = torch.randn(n, r, dtype=dtype, device=device) * 0.05
        bias = torch.randn(n, dtype=dtype, device=device) * 0.01
        wscales = torch.rand(k // _GROUP, n, dtype=dtype, device=device) * 0.5 + 0.1
        wgt_int = torch.randint(-7, 8, (n, k), dtype=torch.int8, device=device)
        lo = wgt_int[..., 0::2].to(torch.int32) & 0x0F
        hi = (wgt_int[..., 1::2].to(torch.int32) & 0x0F) << 4
        wgt = (lo | hi).to(torch.int8)
        params = TensorCoreSVDQuantW4A4Layout.Params(
            scale=wscales,
            orig_dtype=dtype,
            orig_shape=(n, k),
            proj_down=proj_down,
            proj_up=proj_up,
            smooth_factor=smooth,
        )
        qt = QuantizedTensor(wgt, "TensorCoreSVDQuantW4A4Layout", params)

        calls = {"quantize": 0, "scaled_mm": 0}
        pad_sizes = []
        orig_quantize = cuda_backend.quantize_svdquant_w4a4
        orig_scaled_mm = cuda_backend.scaled_mm_svdquant_w4a4

        def spy_quantize(*args, **kwargs):
            calls["quantize"] += 1
            pad_sizes.append(kwargs.get("pad_size"))
            return orig_quantize(*args, **kwargs)

        def spy_scaled_mm(*args, **kwargs):
            calls["scaled_mm"] += 1
            return orig_scaled_mm(*args, **kwargs)

        monkeypatch.setattr(cuda_backend, "quantize_svdquant_w4a4", spy_quantize)
        monkeypatch.setattr(cuda_backend, "scaled_mm_svdquant_w4a4", spy_scaled_mm)

        cuda_was_disabled = ck.list_backends()["cuda"]["disabled"]
        ck.disable_backend("cuda")
        try:
            out = functional.linear(x, qt, bias)
        finally:
            if cuda_was_disabled:
                ck.disable_backend("cuda")
            else:
                ck.enable_backend("cuda")

        assert out.shape == (m, n)
        assert torch.isfinite(out).all()
        assert calls == {"quantize": 1, "scaled_mm": 1}
        assert pad_sizes == [4]

        calls.update({"quantize": 0, "scaled_mm": 0})
        pad_sizes.clear()
        with ck.use_backend("eager"):
            eager_out = functional.linear(x, qt, bias)

        assert eager_out.shape == (m, n)
        assert torch.isfinite(eager_out).all()
        assert calls == {"quantize": 0, "scaled_mm": 0}
        assert pad_sizes == []


def test_svdquant_direct_cuda_pad_policy():
    assert [_cuda_quantize_pad_size(m) for m in [0, 1, 2, 3, 32, 128, 129, 132, 256, 257, 640, 641]] == [
        1, 1, 2, 4, 4, 4, 256, 4, 4, 64, 64, 4,
    ]


def test_svdquant_lora_up_internal_cache_reuses_by_identity():
    from comfy_kitchen.backends.cuda import _natural_lora_up_from_tile_packed

    tile = torch.arange(2 * 3 * _TILE_BN, dtype=torch.float32).reshape(2, 3, _TILE_BN)
    natural = _natural_lora_up_from_tile_packed(tile)
    assert torch.equal(natural, tile.permute(0, 2, 1).reshape(2 * _TILE_BN, 3))
    assert _natural_lora_up_from_tile_packed(tile).data_ptr() == natural.data_ptr()


def test_svdquant_cache_helpers_accept_inference_tensors():
    from comfy_kitchen.backends.cuda import _natural_lora_up_from_tile_packed

    with torch.inference_mode():
        tile = torch.arange(2 * 3 * _TILE_BN, dtype=torch.float32).reshape(2, 3, _TILE_BN)

    natural = _natural_lora_up_from_tile_packed(tile)
    assert torch.equal(natural, tile.permute(0, 2, 1).reshape(2 * _TILE_BN, 3))
    assert _natural_lora_up_from_tile_packed(tile).data_ptr() == natural.data_ptr()


def test_svdquant_lora_up_auto_fuse_policy(cuda_available, seed, monkeypatch):
    if not cuda_available:
        pytest.skip("CUDA required")

    from comfy_kitchen.backends import cuda as cuda_backend

    if not getattr(cuda_backend, "_EXT_AVAILABLE", False):
        pytest.skip("CUDA extension required")

    dtype = torch.bfloat16
    device = "cuda"
    k = 64
    r = 16
    seen = []
    orig_scaled_mm = cuda_backend._C.svdquant_scaled_mm_w4a4

    def spy_scaled_mm(*args, **kwargs):
        seen.append(bool(args[-2]))
        return orig_scaled_mm(*args, **kwargs)

    monkeypatch.setattr(cuda_backend._C, "svdquant_scaled_mm_w4a4", spy_scaled_mm)

    def run(m: int, n: int):
        q_act = torch.randint(-7, 8, (m, k // 2), dtype=torch.int8, device=device)
        wgt = torch.randint(-7, 8, (n, k // 2), dtype=torch.int8, device=device)
        ascales = torch.rand(k // _GROUP, m, dtype=dtype, device=device) * 0.01 + 0.001
        wscales = torch.rand(k // _GROUP, n, dtype=dtype, device=device) * 0.01 + 0.001
        lora_act = torch.randn(m, r, dtype=dtype, device=device) * 0.01
        lora_up = torch.randn(n, r, dtype=dtype, device=device) * 0.01
        return cuda_backend.scaled_mm_svdquant_w4a4(
            q_act, wgt, ascales, wscales, lora_act, lora_up,
        )

    monkeypatch.delenv("COMFY_KITCHEN_SVDQUANT_FUSE_LORA_UP", raising=False)
    run(32, 4096)
    run(32, 12288)
    run(16, 12288)
    monkeypatch.setenv("COMFY_KITCHEN_SVDQUANT_FUSE_LORA_UP", "0")
    run(32, 4096)
    monkeypatch.setenv("COMFY_KITCHEN_SVDQUANT_FUSE_LORA_UP", "1")
    run(32, 12288)

    assert seen == [True, False, True, False, True]
