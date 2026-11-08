# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""gated_residual_segmented -- reference-token (index_timestep_zero) gated residual of
Qwen-Image, Krea2 and Flux / Flux.2.

The eager backend is the sequence ComfyUI runs today (Flux apply_mod with modulation_dims,
Krea2 SingleStreamBlock.forward, Qwen-Image _apply_gate), so it is compared to copies of
those for equality. The Triton kernel keeps the chain in fp32 and rounds once, so it is
compared to eager with a tolerance and to an fp32 reference.
"""

import importlib

import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.registry import registry

eager_backend = importlib.import_module("comfy_kitchen.backends.eager.gated_residual_segmented")


# ComfyUI fb2b26f, comfy/ldm/flux/layers.py apply_mod
def _flux_apply_mod(tensor, m_mult, m_add=None, modulation_dims=None):
    for d in modulation_dims:
        tensor[:, d[0] : d[1]] *= m_mult[:, d[2] : d[2] + 1]
        if m_add is not None:
            tensor[:, d[0] : d[1]] += m_add[:, d[2] : d[2] + 1]
    return tensor


# ComfyUI fb2b26f, comfy/ldm/krea2/model.py SingleStreamBlock.forward
def _krea2_gate(x, attn, gate, bs, split):
    attn[:, :split].mul_(gate[:bs])
    attn[:, split:].mul_(gate[bs:])
    return x + attn


# ComfyUI fb2b26f, comfy/ldm/qwen_image/model.py _apply_gate
def _qwen_gate(x, y, gate, split):
    return y + torch.cat((x[:, :split] * gate[0], x[:, split:] * gate[1]), dim=1)


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

    @pytest.mark.parametrize("op", ["gated_residual_segmented", "gated_residual_segmented_"])
    @pytest.mark.parametrize(
        "case",
        ["residual shape", "split -1", "split > L", "gate rows", "gate batch", "gate D", "2-D x"],
    )
    def test_rejects_bad_arguments(self, op, case):
        x = torch.randn(1, 8, 16)
        residual = torch.randn(1, 8, 16)
        gate = torch.randn(2, 1, 16)
        split = 4
        if case == "residual shape":
            residual = torch.randn(1, 8, 8)
        elif case == "split -1":
            split = -1
        elif case == "split > L":
            split = 9
        elif case == "gate rows":
            gate = torch.randn(3, 1, 16)
        elif case == "gate batch":
            gate = torch.randn(2, 2, 16)
        elif case == "gate D":
            gate = torch.randn(2, 1, 15)
        else:
            x, residual = x[0], residual[0]
        with pytest.raises(ValueError):
            getattr(ck, op)(residual, x, gate, split)

    def test_inplace_rejects_overlapping_residual(self):
        x = torch.randn(1, 8, 16)
        gate = torch.randn(2, 1, 16)
        residual = torch.randn(1, 1, 16).expand(1, 8, 16)
        with pytest.raises(ValueError, match="internal overlap"):
            ck.gated_residual_segmented_(residual, x, gate, 4)

    @pytest.mark.parametrize("other", ["x", "gate"])
    def test_inplace_rejects_x_or_gate_overlapping_residual(self, other):
        buf = torch.randn(2 * 8 * 16)
        residual = buf[: 8 * 16].view(1, 8, 16)
        x = torch.randn(1, 8, 16)
        gate = torch.randn(2, 1, 16)
        if other == "x":
            x = buf[16 : 16 + 8 * 16].view(1, 8, 16)
        else:
            gate = buf[: 2 * 16].view(2, 1, 16)
        with pytest.raises(ValueError, match=f"{other} must not overlap"):
            ck.gated_residual_segmented_(residual, x, gate, 4)

    def test_inplace_rejects_x_of_another_dtype_in_residual_storage(self):
        # same shape and element strides, rows interleaved by element offset, but x is bf16 and
        # residual fp32, so the bytes overlap
        buf = torch.zeros(64)
        residual = buf.view(1, 4, 16)[..., :8]
        x = buf.view(torch.bfloat16).as_strided((1, 4, 8), (64, 16, 1), 8)
        with pytest.raises(ValueError, match="x must not overlap"):
            ck.gated_residual_segmented_(residual, x, torch.randn(2, 1, 8), 2)

    def test_inplace_accepts_disjoint_views_of_one_buffer(self):
        buf = torch.randn(3 * 8 * 16)
        residual = buf[: 8 * 16].view(1, 8, 16)
        x = buf[8 * 16 : 2 * 8 * 16].view(1, 8, 16)
        gate = buf[2 * 8 * 16 : 2 * 8 * 16 + 2 * 16].view(2, 1, 16)
        expected = eager_backend.gated_residual_segmented(residual, x, gate, 4)
        ck.gated_residual_segmented_(residual, x, gate, 4)
        assert torch.equal(residual, expected)


# Captured shapes (bf16), tokens scaled down 8x for CI; D, the split position relative to
# L and the gate layouts are kept. Full sizes: Krea2 (1, 8414, 6144) split 4270,
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
        gate = _split_rows(mod[2])
    elif case == "qwen_image":
        _, _, gate = torch.randn(2, 3 * d, dtype=dtype, device=device).chunk(3, dim=-1)
        gate = _split_rows(gate)
    else:
        _, _, gate = _flux_modulation(1, d, 3, dtype, device)
        gate = gate.transpose(0, 1)
    return x, gate, split


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
    def test_gate_matches_eager(self, case, dtype):
        x, gate, split = _case_inputs(case, dtype)
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
        gate = torch.randn(2, mod_batch, 1000, dtype=torch.bfloat16, device="cuda")
        residual = torch.randn_like(x)
        kwargs = {"residual": residual, "x": x, "gate": gate, "split": split}
        assert registry.get_capable_backend("gated_residual_segmented", kwargs) == "triton"
        with ck.use_backend("triton"):
            gated = ck.gated_residual_segmented(residual, x, gate, split)
        assert _rel(gated, _gate_reference(residual, x, gate, split)) < 1e-2

    @pytest.mark.parametrize("layout", ["flux", "qwen_krea2"])
    def test_batched_model_layouts(self, layout):
        # B = 2 with the per-batch gate rows the models pass as views (batch stride != D)
        b, d = 2, 512
        x = torch.randn(b, 64, d, dtype=torch.bfloat16, device="cuda")
        residual = torch.randn_like(x)
        if layout == "flux":
            gate = _flux_modulation(b, d, 3, torch.bfloat16, "cuda")[2].transpose(0, 1)
        else:
            mod = torch.randn(2 * b, 3 * d, dtype=torch.bfloat16, device="cuda") * 0.5
            gate = _split_rows(mod.chunk(3, dim=-1)[2])
        assert gate.shape == (2, b, d) and gate.stride(1) != d
        kwargs = {"residual": residual, "x": x, "gate": gate, "split": 20}
        assert registry.get_capable_backend("gated_residual_segmented", kwargs) == "triton"
        assert registry.get_capable_backend("gated_residual_segmented_", kwargs) == "triton"
        want = _gate_reference(residual, x, gate, 20)
        with ck.use_backend("triton"):
            got = ck.gated_residual_segmented(residual, x, gate, 20)
            inplace = residual.clone()
            ck.gated_residual_segmented_(inplace, x, gate, 20)
        assert _rel(got, want) < 1e-2
        assert torch.equal(inplace, got)

    def test_strided_inputs(self):
        # residual as a token slice of a larger buffer (written in place), x with a non-unit
        # D stride (copied), and a gate with a non-unit D stride (copied)
        d = 512
        x_t = torch.randn(2, d, 64, dtype=torch.bfloat16, device="cuda").transpose(1, 2)
        gate = torch.randn(2, 2, 2 * d, dtype=torch.bfloat16, device="cuda")[..., ::2]
        buf = torch.randn(2, 80, d, dtype=torch.bfloat16, device="cuda")
        before = buf.clone()
        residual = buf[:, 7:71]
        kwargs = {"residual": residual, "x": x_t, "gate": gate, "split": 20}
        assert registry.get_capable_backend("gated_residual_segmented", kwargs) == "triton"
        assert registry.get_capable_backend("gated_residual_segmented_", kwargs) == "triton"
        with ck.use_backend("triton"):
            want = ck.gated_residual_segmented(residual, x_t, gate, 20)
            ck.gated_residual_segmented_(residual, x_t, gate, 20)
        assert torch.equal(buf[:, 7:71], want)
        assert torch.equal(buf[:, :7], before[:, :7]) and torch.equal(buf[:, 71:], before[:, 71:])
        assert _rel(want, _gate_reference(before[:, 7:71], x_t, gate, 20)) < 1e-2

    def test_unsupported_inputs_go_to_eager(self):
        def cuda(*shape, dtype=torch.bfloat16):
            return torch.randn(shape, dtype=dtype, device="cuda")

        # an in-place residual without unit stride on D cannot be written by the kernel
        mods = cuda(2, 1, 64)
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
