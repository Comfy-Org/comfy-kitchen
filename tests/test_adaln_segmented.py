# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""adaln_segmented / gated_residual_segmented -- reference-token (index_timestep_zero)
modulation of Qwen-Image, Krea2 and Flux / Flux.2.

The eager backend is the sequence ComfyUI runs today (Flux apply_mod with modulation_dims,
Krea2 SingleStreamBlock.forward), so it is compared to copies of those for equality. The
Triton kernels keep the chain in fp32 and round once, so they are compared to eager with a
tolerance and to an fp32 reference.
"""

import importlib

import pytest
import torch
from torch.nn import functional

import comfy_kitchen as ck
from comfy_kitchen.registry import registry

eager_backend = importlib.import_module("comfy_kitchen.backends.eager.adaln_segmented")


# ComfyUI fb2b26f, comfy/ldm/flux/layers.py apply_mod
def _flux_apply_mod(tensor, m_mult, m_add=None, modulation_dims=None):
    for d in modulation_dims:
        tensor[:, d[0] : d[1]] *= m_mult[:, d[2] : d[2] + 1]
        if m_add is not None:
            tensor[:, d[0] : d[1]] += m_add[:, d[2] : d[2] + 1]
    return tensor


# ComfyUI fb2b26f, comfy/ldm/krea2/model.py RMSNorm.forward + SingleStreamBlock.forward
def _krea2_mod(x, norm_scale, scale, shift, bs, split, eps):
    weight = norm_scale.to(torch.float32) + 1.0
    pre = functional.rms_norm(x.float(), (x.shape[-1],), weight=weight, eps=eps).to(x.dtype)
    pre[:, :split].mul_(1 + scale[:bs]).add_(shift[:bs])
    pre[:, split:].mul_(1 + scale[bs:]).add_(shift[bs:])
    return pre


def _krea2_gate(x, attn, gate, bs, split):
    attn[:, :split].mul_(gate[:bs])
    attn[:, split:].mul_(gate[bs:])
    return x + attn


# ComfyUI fb2b26f, comfy/ldm/qwen_image/model.py _modulate / _apply_gate
def _qwen_mod(x, shift, scale, split):
    b = shift.size(0) // 2
    reg = torch.addcmul(shift[:b].unsqueeze(1), x[:, :split], 1 + scale[:b].unsqueeze(1))
    zero = torch.addcmul(shift[b:].unsqueeze(1), x[:, split:], 1 + scale[b:].unsqueeze(1))
    return torch.cat((reg, zero), dim=1)


def _qwen_gate(x, y, gate, split):
    return y + torch.cat((x[:, :split] * gate[0], x[:, split:] * gate[1]), dim=1)


def _reference(x, scale, shift, split, eps=1e-6, rms_weight=None, rms_weight_offset=0.0):
    xf = x.float()
    if rms_weight is None:
        xn = functional.layer_norm(xf, xf.shape[-1:], eps=eps)
    else:
        w = rms_weight.float() + rms_weight_offset
        xn = functional.rms_norm(xf, xf.shape[-1:], weight=w, eps=eps)
    seg = (torch.arange(x.shape[1], device=x.device) >= split).long()
    sc = scale.float()[seg].transpose(0, 1)
    sh = shift.float()[seg].transpose(0, 1)
    return xn * (1 + sc) + sh


def _gate_reference(residual, x, gate, split):
    seg = (torch.arange(x.shape[1], device=x.device) >= split).long()
    return residual.float() + x.float() * gate.float()[seg].transpose(0, 1)


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def _flux_modulation(batch, d, chunks, dtype, device):
    # Modulation.forward of a [B, 2, C] vec: chunks of [B, 2, chunks * D]
    out = torch.randn(batch, 2, chunks * d, dtype=dtype, device=device) * 0.5
    return out.chunk(chunks, dim=-1)


def _split_rows(t):
    # [2B, ..., D] modulation of Qwen-Image / Krea2 -> the [2, B, D] view the op takes
    return t.reshape(t.shape[0], t.shape[-1]).unflatten(0, (2, -1))


class TestSegmentedEager:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_adaln_matches_flux_apply_mod(self, dtype):
        x = torch.randn(2, 23, 64, dtype=dtype)
        shift, scale, _ = _flux_modulation(2, 64, 3, dtype, "cpu")
        dims = [(0, 9, 0), (9, 23, 1)]
        want = _flux_apply_mod(functional.layer_norm(x, (64,), eps=1e-6), 1 + scale, shift, dims)
        with ck.use_backend("eager"):
            got = ck.adaln_segmented(x, scale.transpose(0, 1), shift.transpose(0, 1), 9)
        assert torch.equal(got, want)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_adaln_matches_krea2(self, dtype):
        bs, d = 2, 64
        x = torch.randn(bs, 17, d, dtype=dtype)
        norm_scale = (torch.randn(d) * 0.1).to(dtype)
        mod = torch.randn(2 * bs, 1, 6 * d, dtype=dtype).chunk(6, dim=-1)
        scale, shift = mod[0], mod[1]
        want = _krea2_mod(x, norm_scale, scale, shift, bs, 5, 1e-5)
        with ck.use_backend("eager"):
            got = ck.adaln_segmented(
                x, _split_rows(scale), _split_rows(shift), 5, 1e-5, norm_scale, 1.0
            )
        assert torch.equal(got, want)

    def test_adaln_qwen_addcmul_within_one_rounding(self):
        # Qwen-Image's addcmul rounds x * (1 + scale) + shift once; the reference rounds twice
        x = torch.randn(1, 40, 96, dtype=torch.bfloat16)
        mod = torch.randn(2, 3 * 96, dtype=torch.bfloat16)
        shift, scale, _ = mod.chunk(3, dim=-1)
        want = _qwen_mod(functional.layer_norm(x, (96,), eps=1e-6), shift, scale, 13)
        with ck.use_backend("eager"):
            got = ck.adaln_segmented(x, _split_rows(scale), _split_rows(shift), 13)
        torch.testing.assert_close(got, want, rtol=1e-2, atol=1e-2)
        ref = _reference(x, _split_rows(scale), _split_rows(shift), 13)
        assert _rel(got, ref) < 1e-2

    @pytest.mark.parametrize("split", [0, 6, 17])
    def test_gate_matches_qwen_krea2_flux(self, split):
        bs, d = 2, 64
        x = torch.randn(bs, 17, d, dtype=torch.bfloat16)
        residual = torch.randn(bs, 17, d, dtype=torch.bfloat16)
        gate = torch.randn(2 * bs, 1, d, dtype=torch.bfloat16)
        g = _split_rows(gate)
        with ck.use_backend("eager"):
            got = ck.gated_residual_segmented(residual, x, g, split)
        assert torch.equal(got, _qwen_gate(x, residual, (gate[:bs], gate[bs:]), split))
        assert torch.equal(got, _krea2_gate(residual, x.clone(), gate, bs, split))
        flux = residual.clone()
        flux += _flux_apply_mod(x.clone(), g.transpose(0, 1), None, [(0, split, 0), (split, 17, 1)])
        assert torch.equal(got, flux)

    def test_gate_in_place(self):
        x = torch.randn(1, 12, 32)
        residual = torch.randn(1, 12, 32)
        gate = torch.randn(2, 1, 32)
        want = ck.gated_residual_segmented(residual, x, gate, 4)
        with ck.use_backend("eager"):
            out = ck.gated_residual_segmented_(residual, x, gate, 4)
        assert out is residual
        assert torch.equal(residual, want)

    def test_rejects_bad_arguments(self):
        x = torch.randn(1, 8, 16)
        good = torch.randn(2, 1, 16)
        with pytest.raises(ValueError):
            ck.adaln_segmented(x, torch.randn(1, 1, 16), good, 4)  # one segment row
        with pytest.raises(ValueError):
            ck.adaln_segmented(x, good, torch.randn(2, 3, 16), 4)  # batch 3 vs 1
        with pytest.raises(ValueError):
            ck.adaln_segmented(x, good, good, 9)  # split > L
        with pytest.raises(ValueError):
            ck.adaln_segmented(x, good, good, 4, rms_weight=torch.ones(8))
        with pytest.raises(ValueError):
            ck.adaln_segmented(x[0], good, good, 4)  # not [B, L, D]
        with pytest.raises(ValueError):
            ck.gated_residual_segmented(torch.randn(1, 8, 8), x, good, 4)


# Captured shapes (bf16), tokens scaled down 8x for CI; D, the split position relative to
# L and the modulation layouts are kept. Full sizes: Krea2 (1, 8414, 6144) split 4270,
# Qwen-Image (1, 12336, 3072) split 4096, Flux.2 Klein 9B double block step 1
# (1, 12168, 4096) split 4056, single block (1, 12680, 4096) split 4568, and after the KV
# cache step (1, 4056, 4096) / (1, 4568, 4096) with split == L.
_CASES = {
    "krea2": (1052, 534, 6144),
    "qwen_image": (1542, 512, 3072),
    "flux2_double": (1521, 507, 4096),
    "flux2_single": (1585, 571, 4096),
    "flux2_double_kv": (507, 507, 4096),
}


def _case_inputs(case, dtype, device="cuda"):
    seq, split, d = _CASES[case]
    x = torch.randn(1, seq, d, dtype=dtype, device=device) * 2 + 0.3
    if case == "krea2":
        mod = torch.randn(2, 1, 6 * d, dtype=dtype, device=device).chunk(6, dim=-1)
        scale, shift, gate = _split_rows(mod[0]), _split_rows(mod[1]), _split_rows(mod[2])
        rms_weight = torch.randn(d, dtype=dtype, device=device) * 0.1
    elif case == "qwen_image":
        shift, scale, gate = torch.randn(2, 3 * d, dtype=dtype, device=device).chunk(3, dim=-1)
        scale, shift, gate = _split_rows(scale), _split_rows(shift), _split_rows(gate)
        rms_weight = None
    else:
        shift, scale, gate = _flux_modulation(1, d, 3, dtype, device)
        scale, shift, gate = scale.transpose(0, 1), shift.transpose(0, 1), gate.transpose(0, 1)
        rms_weight = None
    return x, scale, shift, gate, split, rms_weight


def _tol(dtype):
    return {torch.bfloat16: 1e-2, torch.float16: 2e-3, torch.float32: 1e-5}[dtype]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
class TestSegmentedTriton:
    @pytest.fixture(autouse=True)
    def _triton(self):
        pytest.importorskip("triton")
        if not registry.is_available("triton"):
            pytest.skip("triton backend not available")

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("case", list(_CASES))
    def test_adaln_matches_eager(self, case, dtype):
        x, scale, shift, _, split, w = _case_inputs(case, dtype)
        off = 1.0 if w is not None else 0.0
        kwargs = {
            "x": x,
            "scale": scale,
            "shift": shift,
            "split": split,
            "eps": 1e-6,
            "rms_weight": w,
            "rms_weight_offset": off,
        }
        assert registry.validate_backend_for_call("triton", "adaln_segmented", kwargs).success
        with ck.use_backend("triton"):
            got = ck.adaln_segmented(x, scale, shift, split, 1e-6, w, off)
        ref = eager_backend.adaln_segmented(x, scale, shift, split, 1e-6, w, off)
        assert got.shape == x.shape and got.dtype == dtype and got.is_contiguous()
        assert _rel(got, ref) < _tol(dtype)
        assert _rel(got, _reference(x, scale, shift, split, 1e-6, w, off)) < _tol(dtype)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("case", list(_CASES))
    def test_gate_matches_eager(self, case, dtype):
        x, _, _, gate, split, _ = _case_inputs(case, dtype)
        residual = torch.randn_like(x) * 4
        kwargs = {"residual": residual, "x": x, "gate": gate, "split": split}
        assert registry.validate_backend_for_call(
            "triton", "gated_residual_segmented", kwargs
        ).success
        assert registry.validate_backend_for_call(
            "triton", "gated_residual_segmented_", kwargs
        ).success
        with ck.use_backend("triton"):
            got = ck.gated_residual_segmented(residual, x, gate, split)
            inplace = residual.clone()
            ck.gated_residual_segmented_(inplace, x, gate, split)
        ref = eager_backend.gated_residual_segmented(residual, x, gate, split)
        assert got.shape == x.shape and got.dtype == dtype
        assert _rel(got, ref) < _tol(dtype)
        assert _rel(got, _gate_reference(residual, x, gate, split)) < _tol(dtype)
        assert torch.equal(inplace, got)

    @pytest.mark.parametrize("split", [0, 1, 37, 63, 64])
    @pytest.mark.parametrize("mod_batch", [1, 3])
    def test_split_and_batch(self, split, mod_batch):
        # every split position incl. empty segments, batch 3 with per-batch or shared rows
        x = torch.randn(3, 64, 1000, dtype=torch.bfloat16, device="cuda")
        scale = torch.randn(2, mod_batch, 1000, dtype=torch.bfloat16, device="cuda")
        shift = torch.randn(2, mod_batch, 1000, dtype=torch.bfloat16, device="cuda")
        residual = torch.randn_like(x)
        with ck.use_backend("triton"):
            got = ck.adaln_segmented(x, scale, shift, split)
            gated = ck.gated_residual_segmented(residual, x, scale, split)
        assert _rel(got, _reference(x, scale, shift, split)) < 1e-2
        assert _rel(gated, _gate_reference(residual, x, scale, split)) < 1e-2

    def test_strided_inputs(self):
        # x / residual as token slices of larger buffers (read in place), x with a non-unit
        # D stride (copied), and modulation with a non-unit D stride (copied)
        d = 512
        big = torch.randn(2, 80, d, dtype=torch.bfloat16, device="cuda")
        x_tok = big[:, 7:71]
        x_t = torch.randn(2, d, 64, dtype=torch.bfloat16, device="cuda").transpose(1, 2)
        scale = torch.randn(2, 2, 2 * d, dtype=torch.bfloat16, device="cuda")[..., ::2]
        shift = torch.randn(2, 2, d, dtype=torch.bfloat16, device="cuda")
        for x in (x_tok, x_t):
            kwargs = {"x": x, "scale": scale, "shift": shift, "split": 20, "eps": 1e-6}
            assert registry.validate_backend_for_call("triton", "adaln_segmented", kwargs).success
            with ck.use_backend("triton"):
                got = ck.adaln_segmented(x, scale, shift, 20)
            assert _rel(got, eager_backend.adaln_segmented(x, scale, shift, 20)) < 1e-2

        buf = torch.randn(2, 80, d, dtype=torch.bfloat16, device="cuda")
        before = buf.clone()
        residual = buf[:, 7:71]
        with ck.use_backend("triton"):
            want = ck.gated_residual_segmented(residual, x_t, scale, 20)
            ck.gated_residual_segmented_(residual, x_t, scale, 20)
        assert torch.equal(buf[:, 7:71], want)
        assert torch.equal(buf[:, :7], before[:, :7]) and torch.equal(buf[:, 71:], before[:, 71:])
        assert _rel(want, _gate_reference(before[:, 7:71], x_t, scale, 20)) < 1e-2

    def test_unsupported_inputs_go_to_eager(self):
        def cuda(*shape, dtype=torch.bfloat16):
            return torch.randn(shape, dtype=dtype, device="cuda")

        mods = cuda(2, 1, 64)
        adaln_cases = [
            {"x": cuda(1, 0, 64), "scale": mods, "shift": mods},  # empty
            {"x": cuda(1, 4, 8320), "scale": cuda(2, 1, 8320), "shift": cuda(2, 1, 8320)},  # D
            {"x": cuda(1, 8, 64), "scale": mods.float(), "shift": mods},  # mixed dtype
            {"x": mods[:, :1].cpu(), "scale": mods.cpu(), "shift": mods.cpu()},  # cpu tensors
        ]
        for kwargs in adaln_cases:
            kwargs.update(split=0, eps=1e-6, rms_weight=None, rms_weight_offset=0.0)
            assert registry.get_capable_backend("adaln_segmented", kwargs) == "eager"
            with ck.use_backend("triton"):
                got = ck.adaln_segmented(**kwargs)
            assert got.shape == kwargs["x"].shape

        # an in-place residual without unit stride on D cannot be written by the kernel
        residual = cuda(1, 64, 8).transpose(1, 2)
        x = cuda(1, 8, 64)
        kwargs = {"residual": residual, "x": x, "gate": mods, "split": 3}
        assert registry.validate_backend_for_call(
            "triton", "gated_residual_segmented", kwargs
        ).success
        assert registry.get_capable_backend("gated_residual_segmented_", kwargs) == "eager"
        want = _gate_reference(residual, x, mods, 3).to(residual.dtype)
        with ck.use_backend("triton"):
            ck.gated_residual_segmented_(residual, x, mods, 3)
        assert _rel(residual, want) < 1e-2
