# SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.nn import functional

import comfy_kitchen as ck
from tests.conftest import get_capable_backends

FP8 = torch.float8_e4m3fn
ACTS = ["swiglu", "swiglu_hidden_gate", "gelu_tanh"]
IS_HIP = bool(getattr(torch.version, "hip", None))

# linear2 input scales stored in Flux.2 Klein fp8 checkpoints, plus 1.0: what
# ComfyUI passes when a checkpoint has no input_scale.
SCALES = [0.3589285612106323, 0.1573660671710968, 1.735714316368103, 1.0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _act(h, act):
    """act(h) as the model modules compute it (Flux SiLUActivation, yak_mlp, nn.GELU)."""
    if act == "swiglu":
        gate, up = h.chunk(2, dim=-1)
        return functional.silu(gate) * up
    if act == "swiglu_hidden_gate":
        hidden, gate = h.chunk(2, dim=-1)
        return functional.silu(gate) * hidden
    return functional.gelu(h, approximate="tanh")


def _unfused(x, h, scale, act, quant_backend=None):
    """The unfused sequence: act, cat, then quantize_per_tensor_fp8 (optionally forced backend)."""
    cat = torch.cat((x, _act(h, act)), dim=-1)
    if quant_backend is None:
        return ck.quantize_per_tensor_fp8(cat, scale, FP8)
    with ck.use_backend(quant_backend):
        return ck.quantize_per_tensor_fp8(cat, scale, FP8)


def _h_cols(act, dm):
    return dm if act == "gelu_tanh" else 2 * dm


def _inputs(lead, dx, dm, act, dtype, device, spread=3.0):
    """x and h with a heavy tail (some values saturate fp8 at the test scales);
    h is a strided column slice of a wider buffer, like linear1's MLP part."""
    x = (torch.randn(*lead, dx, device=device) * 0.7).to(dtype)
    buf = torch.randn(*lead, 64 + _h_cols(act, dm), device=device) * spread
    spikes = torch.rand(buf.shape, device=device) < 1e-3
    buf = torch.where(spikes, buf * 40, buf).to(dtype)
    return x, buf[..., 64:]


def _scale(value, device):
    return torch.tensor(value, dtype=torch.float32, device=device)


def _mismatch(a, b, keep=None):
    """Number of differing bytes (optionally only where keep is True)."""
    differ = a.view(torch.uint8) != b.view(torch.uint8)
    return int((differ if keep is None else differ & keep).sum())


def _require(backend, device="cuda"):
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    if backend not in get_capable_backends("act_cat_quantize_fp8", device):
        pytest.skip(f"backend '{backend}' not capable for act_cat_quantize_fp8 on {device}")


def _triton_refusal():
    """Why triton's call rule refuses a plain bf16 call on this GPU (e.g. a ROCm arch
    other than gfx12), or None where triton serves it."""
    x = torch.zeros(2, 8, device="cuda", dtype=torch.bfloat16)
    h = torch.zeros(2, 16, device="cuda", dtype=torch.bfloat16)
    kwargs = _kwargs(x, h, _scale(1.0, "cuda"))
    valid = ck.registry.validate_backend_for_call("triton", "act_cat_quantize_fp8", kwargs)
    return None if valid.success else valid.failure_reason


def _require_served():
    """triton is available and its call rule serves this GPU."""
    _require("triton")
    reason = _triton_refusal()
    if reason is not None:
        pytest.skip(f"triton does not serve act_cat_quantize_fp8 here: {reason}")


def _native_quant_backend():
    """The compiled per-tensor quantizer the triton op reproduces: cuda, or hip under ROCm."""
    name = "hip" if IS_HIP else "cuda"
    if name not in get_capable_backends("quantize_per_tensor_fp8", "cuda"):
        pytest.skip(f"compiled {name} backend required")
    return name


def _exact_reference(x, h, scale, act):
    """(reference, positions to compare) for a bit-exact check of the triton op.

    With the compiled quantizer (cuda / hip) available: that quantizer. Under ROCm the
    HIP quantizer keeps a NaN's sign, and the sign of a NaN the activation computes
    from a NaN / inf input follows Triton's rules there, not PyTorch's: those
    positions are skipped. Without a compiled quantizer, under CUDA only: kitchen's
    triton quantizer, which divides the same way but clamps NaN to +max where the
    CUDA kernel gives -max, so NaN positions are skipped.
    """
    native = "hip" if IS_HIP else "cuda"
    pre = torch.cat((x, _act(h, act)), dim=-1)
    if native in get_capable_backends("quantize_per_tensor_fp8", "cuda"):
        ref = _unfused(x, h, scale, act, native)
        keep = torch.ones(ref.shape, dtype=torch.bool, device=ref.device)
        if IS_HIP:
            keep[..., x.shape[-1]:] = ~torch.isnan(pre[..., x.shape[-1]:])
        return ref, keep
    if IS_HIP:
        pytest.skip("compiled hip backend required")
    return _unfused(x, h, scale, act, "triton"), ~torch.isnan(pre)


def _kwargs(x, h, scale, act="swiglu", output_type=FP8):
    return {"x": x, "h": h, "scale": scale, "act": act, "output_type": output_type}


def _triton(x, h, scale, act):
    kwargs = _kwargs(x, h, scale, act)
    assert ck.registry.validate_backend_for_call("triton", "act_cat_quantize_fp8", kwargs).success
    with ck.use_backend("triton"):
        return ck.act_cat_quantize_fp8(x, h, scale, act)


# (leading dims, Dx, Dm): Dx and Dm not multiples of the 1024-column tile, rows
# not a multiple of the 2-row tile, no leading dims, and Flux's widths.
_SHAPES = [
    ((37,), 200, 300),
    ((2, 45), 512, 1024),
    ((), 96, 160),
    ((2, 64), 3072, 9216),
]


# ---------------------------------------------------------------------------
# Triton: bit-exact against the unfused sequence
# ---------------------------------------------------------------------------


class TestActCatQuantizeFp8Triton:
    @pytest.mark.parametrize("act", ACTS)
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("shape", _SHAPES)
    @pytest.mark.parametrize("scale", SCALES)
    def test_matches_native_quantizer(self, act, dtype, shape, scale, seed):
        """Bit-identical to act + cat + the compiled CUDA (HIP) quantizer, with NaN and inf
        in x (copied as they are) and saturating values in both parts."""
        _require_served()
        native = _native_quant_backend()
        lead, dx, dm = shape
        x, h = _inputs(lead, dx, dm, act, dtype, "cuda")
        specials = [float("nan"), float("inf"), -float("inf"), -0.0]
        x.view(-1)[:4] = torch.tensor(specials, dtype=dtype)
        st = _scale(scale, "cuda")

        out = _triton(x, h, st, act)
        ref = _unfused(x, h, st, act, native)

        assert out.shape == (*lead, dx + dm)
        assert out.dtype == FP8
        assert out.is_contiguous()
        assert _mismatch(out, ref) == 0

    @pytest.mark.parametrize("act", ACTS)
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("shape", _SHAPES)
    @pytest.mark.parametrize("scale", SCALES)
    def test_matches_triton_quantizer(self, act, dtype, shape, scale, seed):
        """Against act + cat + kitchen's triton quantizer, finite inputs. Under CUDA both
        divide the same way, so the bytes are equal; under ROCm the op follows the HIP
        quantizer's fast division, which rounds a few values one fp8 step apart."""
        _require_served()
        lead, dx, dm = shape
        x, h = _inputs(lead, dx, dm, act, dtype, "cuda")
        st = _scale(scale, "cuda")

        out = _triton(x, h, st, act)
        ref = _unfused(x, h, st, act, "triton")

        assert out.shape == (*lead, dx + dm)
        if not IS_HIP:
            assert _mismatch(out, ref) == 0
        else:
            step = (out.view(torch.uint8).int() - ref.view(torch.uint8).int()).abs()
            assert int(step.max()) <= 1
            assert _mismatch(out, ref) <= max(1, out.numel() // 1000)

    @pytest.mark.parametrize("act", ACTS)
    def test_every_bf16_value(self, act, seed):
        """Every bf16 bit pattern in both inputs (x copied, h activated), three scales."""
        _require_served()
        vals = torch.arange(-32768, 32768, dtype=torch.int32, device="cuda")
        vals = vals.to(torch.int16).view(torch.bfloat16).reshape(128, 512)
        other = (torch.randn(128, 512, device="cuda") * 4).to(torch.bfloat16)
        if act == "swiglu":
            h = torch.cat((vals.roll(37, 0), other), dim=1)
        elif act == "swiglu_hidden_gate":
            h = torch.cat((other, vals.roll(37, 0)), dim=1)
        else:
            h = vals.roll(37, 0)
        for scale in (0.3589285612106323, 1.0, 1e-3):
            st = _scale(scale, "cuda")
            out = _triton(vals, h, st, act)
            ref, keep = _exact_reference(vals, h, st, act)
            assert _mismatch(out, ref, keep) == 0

    @pytest.mark.parametrize("scale", [0.3589285612106323, 1.0])
    def test_flux2_klein_block(self, scale, seed):
        """Flux.2 Klein single block: attn (B, N, 3072) and h = the MLP slice (B, N, 18432)
        of linear1's (B, N, 3 * 3072 + 18432) bf16 output, swiglu."""
        _require_served()
        b, n, d, dm = 2, 300, 3072, 9216
        lin1 = (torch.randn(b, n, 3 * d + 2 * dm, device="cuda") * 2.5).to(torch.bfloat16)
        attn = (torch.randn(b, n, d, device="cuda") * 0.7).to(torch.bfloat16)
        mlp = lin1[..., 3 * d:]
        st = _scale(scale, "cuda")

        out = _triton(attn, mlp, st, "swiglu")
        assert out.shape == (b, n, d + dm)
        ref, keep = _exact_reference(attn, mlp, st, "swiglu")
        assert _mismatch(out, ref, keep) == 0

    def test_non_contiguous_x(self, seed):
        """x with strided rows (a column slice) is read in place."""
        _require_served()
        x = torch.randn(130, 1024, device="cuda", dtype=torch.bfloat16)[:, 100:612]
        _, h = _inputs((130,), 1, 384, "swiglu", torch.bfloat16, "cuda")
        st = _scale(0.25, "cuda")
        out = _triton(x, h, st, "swiglu")
        ref, keep = _exact_reference(x.contiguous(), h, st, "swiglu")
        assert _mismatch(out, ref, keep) == 0

    def test_zero_rows(self):
        _require_served()
        x = torch.empty(3, 0, 64, device="cuda", dtype=torch.bfloat16)
        h = torch.empty(3, 0, 128, device="cuda", dtype=torch.bfloat16)
        out = _triton(x, h, _scale(1.0, "cuda"), "swiglu")
        assert out.shape == (3, 0, 128)
        assert out.dtype == FP8


# ---------------------------------------------------------------------------
# Eager reference and fallback
# ---------------------------------------------------------------------------


class TestActCatQuantizeFp8Eager:
    @pytest.mark.parametrize("act", ACTS)
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_eager_is_unfused_sequence(self, act, dtype, device, seed):
        """Eager = act + cat + quantize_per_tensor_fp8, on every device."""
        if device == "cpu" and dtype != torch.float32:
            pytest.skip("half-precision act on CPU not checked")
        x, h = _inputs((3, 50), 96, 160, act, dtype, device)
        st = _scale(0.3589285612106323, device)
        with ck.use_backend("eager"):
            out = ck.act_cat_quantize_fp8(x, h, st, act)
        ref = _unfused(x, h, st, act, "eager")
        assert out.shape == (3, 50, 96 + 160)
        assert out.dtype == FP8
        assert _mismatch(out, ref) == 0

    def test_e5m2_output(self, device, seed):
        x, h = _inputs((40,), 64, 64, "swiglu", torch.float32, device)
        st = _scale(0.5, device)
        out = ck.act_cat_quantize_fp8(x, h, st, "swiglu", torch.float8_e5m2)
        cat = torch.cat((x, _act(h, "swiglu")), dim=-1)
        ref = ck.quantize_per_tensor_fp8(cat, st, torch.float8_e5m2)
        assert out.dtype == torch.float8_e5m2
        assert _mismatch(out, ref) == 0

    @pytest.mark.parametrize(
        "case",
        ["fp32", "mixed_dtype", "e5m2", "scale_cpu", "scale_two_elements", "lead_mismatch"],
    )
    def test_fallback(self, case, seed):
        """Calls the triton kernel does not serve are refused there and run the
        unfused sequence (with the default quantizer dispatch) instead."""
        _require("triton")
        dtype = torch.float32 if case == "fp32" else torch.bfloat16
        x, h = _inputs((2, 40), 128, 256, "swiglu", dtype, "cuda")
        st = _scale(0.5, "cuda")
        output_type = FP8
        if case == "mixed_dtype":
            h = h.half()
        elif case == "e5m2":
            output_type = torch.float8_e5m2
        elif case == "scale_cpu":
            st = st.cpu()
        elif case == "scale_two_elements":
            st = torch.tensor([0.5, 0.5], device="cuda")
        elif case == "lead_mismatch":
            h = h.reshape(80, -1)
        kwargs = _kwargs(x, h, st, "swiglu", output_type)
        valid = ck.registry.validate_backend_for_call("triton", "act_cat_quantize_fp8", kwargs)
        assert not valid.success
        if case in ("scale_cpu", "scale_two_elements", "lead_mismatch"):
            return  # refused by triton; the fallback then behaves as the unfused calls do
        out = ck.act_cat_quantize_fp8(x, h, st, "swiglu", output_type)
        cat = torch.cat((x, _act(h, "swiglu")), dim=-1)
        assert _mismatch(out, ck.quantize_per_tensor_fp8(cat, st, output_type)) == 0

    @pytest.mark.parametrize("act", ACTS)
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_unserved_gpu_runs_unfused(self, act, dtype, seed):
        """Where triton's call rule refuses the GPU itself (a ROCm arch other than
        gfx12), the public op runs eager and returns what the unfused calls return."""
        _require("triton")
        if _triton_refusal() is None:
            pytest.skip("triton serves act_cat_quantize_fp8 on this GPU")
        x, h = _inputs((2, 45), 512, 1024, act, dtype, "cuda")
        st = _scale(0.3589285612106323, "cuda")
        kwargs = _kwargs(x, h, st, act)
        assert ck.registry.get_capable_backend("act_cat_quantize_fp8", kwargs) == "eager"
        out = ck.act_cat_quantize_fp8(x, h, st, act)
        assert _mismatch(out, _unfused(x, h, st, act)) == 0

    def test_invalid_act(self, device):
        x = torch.zeros(4, 8, device=device)
        h = torch.zeros(4, 16, device=device)
        st = _scale(1.0, device)
        with pytest.raises(ValueError, match="unsupported act"), ck.use_backend("eager"):
            ck.act_cat_quantize_fp8(x, h, st, "relu")
        with pytest.raises(ValueError, match="even number"), ck.use_backend("eager"):
            ck.act_cat_quantize_fp8(x, h[:, :15], st, "swiglu")

    @pytest.mark.parametrize("act", ACTS)
    def test_fake_shape(self, act):
        x = torch.empty(2, 7, 96, device="meta", dtype=torch.bfloat16)
        h = torch.empty(2, 7, 320, device="meta", dtype=torch.bfloat16)
        st = torch.empty((), device="meta", dtype=torch.float32)
        out = torch.ops.comfy_kitchen.act_cat_quantize_fp8(x, h, st, act, 5)
        width = 96 + (320 if act == "gelu_tanh" else 160)
        assert out.shape == (2, 7, width)
        assert out.dtype == FP8
