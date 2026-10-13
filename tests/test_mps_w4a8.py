# SPDX-License-Identifier: Apache-2.0
"""MPS weight-only W4A8 / W6A8 linear, dequantize and requantize against eager on CPU.

The MPS path quantizes no activations, so its oracle is the float linear over eager's
dequantized weight, not eager's INT8 GEMM. The decoded fp16 weight itself must be
bit-identical to eager's rotated-basis decode cast to fp16.
"""

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

import comfy_kitchen as ck
from comfy_kitchen.backends.eager import w4a8_int8 as eager_w4a8

pytestmark = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")),
    reason="MPS with torch.mps.compile_shader required",
)

N, K = 192, 512

# (bits, quantize kwargs): 4-bit codebook, 4-bit uniform, 6-bit, 4-bit asymmetric (correction)
FORMATS = {
    "w4_codebook": {"bits": 4},
    "w4_uniform": {"bits": 4, "codebook": False},
    "w6": {"bits": 6},
    "w4_asym": {"bits": 4, "symmetric": False},
}


def mps_w4a8():
    from comfy_kitchen.backends.mps import w4a8

    return w4a8


def quantize(fmt, group_size=16, convrot_groupsize=256, scale_dtype=torch.float8_e4m3fn, n=N, k=K):
    gen = torch.Generator().manual_seed(0)
    weight = (torch.randn(n, k, generator=gen) * 0.02).to(torch.bfloat16)
    return eager_w4a8.quantize_w4a8_int8_weight(
        weight,
        group_size=group_size,
        convrot_groupsize=convrot_groupsize,
        scale_dtype=scale_dtype,
        **FORMATS[fmt],
    )


def to_mps(*tensors):
    return [None if t is None else t.to("mps") for t in tensors]


def assert_close_to_oracle(got, x, packed, bias, group_size, convrot_groupsize):
    """fp16 GEMM over the decoded weight vs the fp32 linear over eager's dequantized weight.
    Allows fp16 compute error plus one rounding to the output dtype; eager's INT8-activation
    GEMM sits near 8e-3 relative L2, so a path that quantized activations would fail here."""
    q, s_rel, s_channel, correction, codebook = packed
    weight = eager_w4a8.dequantize_w4a8_int8_weight(
        q, s_rel, s_channel, codebook, correction, group_size, convrot_groupsize, torch.float32
    )
    ref = F.linear(x.float(), weight, None if bias is None else bias.float())
    out_rounding = torch.finfo(got.dtype).eps / 2
    got = got.float().cpu()
    scale = ref.abs().max().item()
    assert (got - ref).abs().max().item() <= (3e-3 + 2 * out_rounding) * scale
    assert ((got - ref).norm() / ref.norm()).item() < 1.5e-3 + out_rounding


def normalized_reference(grid_fp32, ch_exp, dtype):
    """Eager's fp32 rotated-basis weight divided by 2^ch_exp, rounded once to ``dtype``."""
    return torch.ldexp(grid_fp32.double(), -ch_exp.double()[:, None]).to(dtype)


class TestDecode:
    @pytest.mark.parametrize("fmt", list(FORMATS))
    @pytest.mark.parametrize("scale_dtype", [torch.float8_e4m3fn, torch.float32])
    @pytest.mark.parametrize("out_dtype", [torch.float16, torch.float32])
    def test_normalized_decode_is_eager_scaled_by_a_power_of_two(self, fmt, scale_dtype, out_dtype):
        """The GEMM's weight: eager's fp32 rotated-basis decode (int8 grid * s_channel
        [+ correction]) divided by 2^ch_exp, |w'| <= 1, one rounding to the compute dtype.
        Row-major and transposed kernels write the same bits."""
        bits = FORMATS[fmt]["bits"]
        group_size = 32 if bits == 6 else 16
        q, s_rel, s_channel, correction, codebook = quantize(fmt, group_size, scale_dtype=scale_dtype)
        grid = eager_w4a8._dequant_int4_grouped_to_int8(q, s_rel, codebook, group_size)
        ref32 = eager_w4a8._dequantize_w4a8_int8_weight_from_int8(
            grid, s_channel, correction, group_size, torch.float32
        )
        mod = mps_w4a8()
        levels = codebook.to("mps") if codebook is not None else mod._uniform_levels(bits, torch.device("mps"))
        corr = None if correction is None else correction.float().to("mps")
        corr_max = None if corr is None else corr.abs().amax(0)
        ch = torch.empty(N, dtype=torch.int32, device="mps")
        common = (q.to("mps"), s_rel.to("mps"), s_channel.to("mps"), levels, corr, K, bits, group_size)
        got = mod._decode_w4a8(*common, 0, N, out_dtype, ch, corr_max).cpu()
        f = ch.cpu()
        scaled = torch.ldexp(ref32.double(), -f.double()[:, None])
        assert scaled.abs().max() <= 1 and scaled.abs().amax(1).min() > 2.0 ** -3
        assert torch.equal(got, normalized_reference(ref32, f, out_dtype))
        trans = mod._decode_w4a8(*common, 0, N, out_dtype, ch, corr_max, transposed=True).cpu()
        assert torch.equal(trans, got.T)

    @pytest.mark.parametrize("bits,codebook", [(4, True), (4, False), (6, False)])
    @pytest.mark.parametrize("correction", [False, True])
    def test_random_bytes_round_like_eager(self, bits, codebook, correction):
        """Arbitrary packed bytes and every finite e4m3 scale byte. Uniform levels times
        power-of-two scales land on .5 ties and on -0: eager rounds ties to even and its int8
        cast drops the sign of -0, so the kernel must do both. Rows 0..3 tail the 32-row
        transposed tile (n = 67)."""
        if bits == 6 and codebook:
            pytest.skip("6-bit has no codebook")
        gen = torch.Generator().manual_seed(1)
        n, k, gs = 67, 512, 16
        q = torch.randint(-128, 128, (n, k * bits // 8), generator=gen, dtype=torch.int8)
        s_rel = torch.randint(0, 0x7F, (n, k // gs), generator=gen, dtype=torch.uint8).view(torch.float8_e4m3fn)
        s_channel = torch.rand(n, generator=gen) * 1e-3 + 1e-6
        s_channel[3] = 0  # zero channel: the correction alone sets the exponent
        table = torch.sort(torch.randn(16, generator=gen))[0] if codebook else None
        corr = (torch.randn(k // gs, n, generator=gen) * 1e-2).to(torch.bfloat16) if correction else None
        grid = eager_w4a8._dequant_int4_grouped_to_int8(q, s_rel, table, gs)
        ref32 = eager_w4a8._dequantize_w4a8_int8_weight_from_int8(grid, s_channel, corr, gs, torch.float32)
        mod = mps_w4a8()
        levels = table.to("mps") if codebook else mod._uniform_levels(bits, torch.device("mps"))
        corr_m = None if corr is None else corr.float().to("mps")
        common = (q.to("mps"), s_rel.to("mps"), s_channel.to("mps"), levels, corr_m, k, bits, gs)
        assert torch.equal(mod._decode_w4a8(*common, 0, n).cpu().view(torch.int32), ref32.view(torch.int32))
        ch = torch.empty(n, dtype=torch.int32, device="mps")
        corr_max = None if corr_m is None else corr_m.abs().amax(0)
        for row0 in (0, 5):
            got = mod._decode_w4a8(*common, row0, n - row0, torch.float16, ch, corr_max).cpu()
            assert torch.equal(got, normalized_reference(ref32[row0:], ch.cpu()[row0:], torch.float16))
            trans = mod._decode_w4a8(*common, row0, n - row0, torch.float16, ch, corr_max, transposed=True)
            assert torch.equal(trans.cpu(), got.T)

    def test_index_math_is_64_bit(self):
        """Finding 5: for N=131073, K=131072, group 4, row 131072's first group-scale index is
        2^32, which 32-bit math wraps to 0 (row zero's scales). The kernels compute every
        global index through these helpers; this runs them on the GPU without the 64 GB tensor."""
        mod = mps_w4a8()
        lib = torch.mps.compile_shader(mod._SHADERS + """
        kernel void index_probe(device ulong* out, uint i [[thread_position_in_grid]]) {
            out[0] = w4_scale_index(131072, 131072, 4, 0);
            out[1] = w4_corr_index(32767, 131073, 131072);
            out[2] = w4_row_offset(131072, 131072, 6);
            out[3] = w4_scale_index(65536, 1u << 20, 16, 3);
        }
        """)
        out = torch.zeros(4, dtype=torch.int64, device="mps")
        lib.index_probe(out, threads=1)
        assert out.cpu().tolist() == [
            131072 * 32768, 32767 * 131073 + 131072, 131072 * 131072 * 6 // 8, 65536 * 65536 + 3,
        ]


class TestLinear:
    @pytest.mark.parametrize("fmt", list(FORMATS))
    @pytest.mark.parametrize("with_bias", [False, True])
    @pytest.mark.parametrize("convrot_groupsize", [16, 256])
    def test_matches_float_oracle(self, fmt, with_bias, convrot_groupsize):
        group_size = 32 if fmt == "w6" else 16
        packed = quantize(fmt, group_size, convrot_groupsize)
        gen = torch.Generator().manual_seed(2)
        x = torch.randn(2, 7, K, generator=gen).to(torch.bfloat16)
        bias = torch.randn(N, generator=gen).to(torch.bfloat16) if with_bias else None
        q, s_rel, s_channel, correction, codebook = to_mps(*packed)
        got = ck.w4a8_int8_linear(
            x.to("mps"), q, s_rel, s_channel, codebook=codebook, correction=correction,
            bias=None if bias is None else bias.to("mps"),
            group_size=group_size, convrot_groupsize=convrot_groupsize, out_dtype=torch.bfloat16,
        )
        assert got.shape == (2, 7, N) and got.dtype == torch.bfloat16 and got.device.type == "mps"
        assert_close_to_oracle(got, x, packed, bias, group_size, convrot_groupsize)

    @pytest.mark.parametrize("x_dtype", [torch.float16, torch.float32])
    @pytest.mark.parametrize("group_size", [4, 64])
    def test_input_dtypes_and_group_sizes(self, x_dtype, group_size):
        packed = quantize("w4_codebook", group_size, 64, scale_dtype=torch.float32)
        x = torch.randn(5, K, generator=torch.Generator().manual_seed(3)).to(x_dtype)
        got = mps_w4a8().w4a8_int8_linear(
            x.to("mps"), *to_mps(*packed[:3]), codebook=packed[4].to("mps"),
            group_size=group_size, convrot_groupsize=64, out_dtype=x_dtype,
        )
        assert got.dtype == x_dtype
        assert_close_to_oracle(got, x, packed, None, group_size, 64)

    @pytest.mark.parametrize("fmt", ["w4_codebook", "w6", "w4_asym"])
    def test_chunked_decode_matches_single_pass(self, fmt, monkeypatch):
        """A decode budget of 48 rows (50 rounded to the epilogue's 4-column grid) runs 4 chunks;
        the transposed decode (wide-K layout) runs the same chunks."""
        mod = mps_w4a8()
        group_size = 32 if fmt == "w6" else 16
        packed = quantize(fmt, group_size)
        x = torch.randn(9, K, generator=torch.Generator().manual_seed(4)).to(torch.float16).to("mps")
        q, s_rel, s_channel, correction, codebook = to_mps(*packed)
        kwargs = {"codebook": codebook, "correction": correction, "group_size": group_size, "out_dtype": torch.float16}
        from comfy_kitchen.backends.mps import linear

        whole = mod.w4a8_int8_linear(x, q, s_rel, s_channel, **kwargs)
        monkeypatch.setattr(linear, "_MAX_DECODE_BYTES", 50 * 2 * K)
        chunked = mod.w4a8_int8_linear(x, q, s_rel, s_channel, **kwargs)
        assert torch.equal(chunked, whole)
        monkeypatch.setattr(linear, "_TRANSPOSED_DECODE_MIN_K", 1)
        transposed = mod.w4a8_int8_linear(x, q, s_rel, s_channel, **kwargs)
        assert_close_to_oracle(transposed, x.cpu(), packed, None, group_size, 256)

    def test_dispatch_selects_mps(self):
        from comfy_kitchen.registry import registry

        q, s_rel, s_channel, _, codebook = to_mps(*quantize("w4_codebook"))
        impl = registry.get_implementation(
            "w4a8_int8_linear",
            kwargs={
                "x": torch.randn(1, K, device="mps"), "qdata": q, "s_rel": s_rel, "s_channel": s_channel,
                "codebook": codebook, "correction": None, "bias": None, "group_size": 16,
                "convrot_groupsize": 256, "out_dtype": torch.bfloat16,
            },
        )
        assert impl is mps_w4a8().w4a8_int8_linear

    def test_asymmetric_uses_the_exact_stored_weight_not_eagers_rounded_one(self):
        # Deliberate precision contract: eager's asymmetric path rounds the weight to x's dtype
        # (bf16 1.00390625 -> 1) before F.linear, giving 0 here. The mps path computes from
        # the stored weight's exact value, which gives 1 (review 2026-10-09).
        k = 256
        q = torch.full((1, k // 2), 0x99, dtype=torch.uint8).view(torch.int8)
        s_rel, s_ch = torch.ones(1, k // 16), torch.tensor([2.0 ** -8])
        corr = torch.ones(16, 1, dtype=torch.bfloat16)
        bias = torch.tensor([-256.0], dtype=torch.bfloat16)
        x = torch.ones(1, k, dtype=torch.bfloat16)
        kwargs = {"codebook": None, "group_size": 16, "convrot_groupsize": 256, "out_dtype": torch.float32}
        eager = eager_w4a8.w4a8_int8_linear(x, q, s_rel, s_ch, correction=corr, bias=bias, **kwargs)
        mps = mps_w4a8().w4a8_int8_linear(*to_mps(x, q, s_rel, s_ch), correction=corr.to("mps"),
                                          bias=bias.to("mps"), **kwargs)
        assert eager.item() == 0.0 and mps.cpu().item() == 1.0

    def test_rejects_mismatched_k(self):
        q, s_rel, s_channel, _, codebook = to_mps(*quantize("w4_codebook"))
        with pytest.raises(ValueError, match="does not match"):
            mps_w4a8().w4a8_int8_linear(torch.randn(3, K + 16, device="mps"), q, s_rel, s_channel, codebook)


def mps_scale_like_comfyui(s_rel):
    """How ComfyUI lands a checkpoint's fp8 group scale on the GPU: safetensors bytes as
    uint8, moved to the device, then viewed back as fp8. MPS holds fp8 storage but cannot
    convert it, so any `.float()` on this tensor raises."""
    if s_rel.dtype != torch.float8_e4m3fn:
        return s_rel.to("mps")
    return s_rel.view(torch.uint8).to("mps").view(torch.float8_e4m3fn)


def ulp_close(got, ref):
    """Equal up to one rounding in the output dtype (the rotation GEMM may sum in another order)."""
    eps = torch.finfo(ref.dtype).eps
    return ((got.float() - ref.float()).abs() <= eps * ref.float().abs() + 1e-30).all().item()


class TestDequantize:
    @pytest.mark.parametrize("fmt", list(FORMATS))
    @pytest.mark.parametrize("scale_dtype", [torch.float8_e4m3fn, torch.float32])
    def test_fp32_decode_is_bit_identical_to_eager(self, fmt, scale_dtype):
        """The fp32 kernel variant reproduces eager's rotated-basis fp32 weight exactly."""
        bits = FORMATS[fmt]["bits"]
        group_size = 32 if bits == 6 else 16
        q, s_rel, s_channel, correction, codebook = quantize(fmt, group_size, scale_dtype=scale_dtype)
        grid = eager_w4a8._dequant_int4_grouped_to_int8(q, s_rel, codebook, group_size)
        ref = eager_w4a8._dequantize_w4a8_int8_weight_from_int8(
            grid, s_channel, correction, group_size, torch.float32
        )
        mod = mps_w4a8()
        levels = codebook.to("mps") if codebook is not None else mod._uniform_levels(bits, torch.device("mps"))
        got = mod._decode_w4a8(
            q.to("mps"), mps_scale_like_comfyui(s_rel), s_channel.to("mps"), levels,
            None if correction is None else correction.float().to("mps"),
            K, bits, group_size, 0, N, torch.float32,
        ).cpu()
        assert torch.equal(got.view(torch.int32), ref.view(torch.int32))

    @pytest.mark.parametrize("fmt", list(FORMATS))
    @pytest.mark.parametrize("convrot_groupsize", [16, 256])
    @pytest.mark.parametrize("output_dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_matches_eager_on_cpu(self, fmt, convrot_groupsize, output_dtype):
        """Physical (un-rotated) weight, through the public dispatcher with ComfyUI's fp8 scale.
        The rotated-basis decode is exact (test above); the rotation back is a GEMM in the
        output dtype. MPS rounds that GEMM correctly (== fp32 GEMM then one rounding); CPU's
        native fp16/bf16 GEMM is one ulp off in ~1e-7 / ~1e-6 of elements at full-layer size.
        This data has no such element in fp16, so fp16 is checked bitwise; bf16/fp32 allow
        one output ulp."""
        from comfy_kitchen.tensor.w4a8_int8 import dequantize_w4a8_int8_weight

        group_size = 32 if fmt == "w6" else 16
        q, s_rel, s_channel, correction, codebook = quantize(fmt, group_size, convrot_groupsize)
        ref = eager_w4a8.dequantize_w4a8_int8_weight(
            q, s_rel, s_channel, codebook, correction, group_size, convrot_groupsize, output_dtype
        )
        got = dequantize_w4a8_int8_weight(
            q.to("mps"), mps_scale_like_comfyui(s_rel), s_channel.to("mps"),
            *to_mps(codebook, correction), group_size, convrot_groupsize, output_dtype,
        )
        assert got.device.type == "mps" and got.dtype == output_dtype and got.shape == (N, K)
        got = got.cpu()
        if output_dtype == torch.float16:
            assert torch.equal(got.view(torch.int16), ref.view(torch.int16))
        else:
            assert ulp_close(got, ref)

    def test_chunked_matches_single_pass(self, monkeypatch):
        mod = mps_w4a8()
        q, s_rel, s_channel, _, _ = to_mps(*quantize("w6", 32))
        whole = mod.dequantize_w4a8_int8_weight(q, s_rel, s_channel, group_size=32, output_dtype=torch.float16)
        monkeypatch.setattr(mod, "_MAX_DECODE_BYTES", 50 * 4 * K)
        chunked = mod.dequantize_w4a8_int8_weight(q, s_rel, s_channel, group_size=32, output_dtype=torch.float16)
        assert torch.equal(chunked, whole)

    def test_dispatch_selects_mps_and_cpu_stays_eager(self):
        from comfy_kitchen.registry import registry

        q, s_rel, s_channel, _, codebook = quantize("w4_codebook")
        kwargs = {
            "qdata": q, "s_rel": s_rel, "s_channel": s_channel, "codebook": codebook, "correction": None,
            "group_size": 16, "convrot_groupsize": 256, "output_dtype": torch.bfloat16,
        }
        assert registry.get_implementation("dequantize_w4a8_int8_weight", kwargs=kwargs) is (
            eager_w4a8.dequantize_w4a8_int8_weight
        )
        mps_kwargs = {**kwargs, **dict(zip(("qdata", "s_rel", "s_channel", "codebook"), to_mps(q, s_rel, s_channel, codebook), strict=False))}
        assert registry.get_implementation("dequantize_w4a8_int8_weight", kwargs=mps_kwargs) is (
            mps_w4a8().dequantize_w4a8_int8_weight
        )


class TestRequantize:
    @pytest.mark.parametrize("fmt", list(FORMATS))
    @pytest.mark.parametrize("stochastic_rounding", [0, 1234])
    def test_quantize_on_mps_equals_cpu(self, fmt, stochastic_rounding):
        """MPS quantize is eager-on-CPU moved back: identical packed tensors, eager's dtypes
        (fp8 scales stay fp8 storage on MPS), and an MPS decode equal to the CPU decode."""
        from comfy_kitchen.tensor.w4a8_int8 import dequantize_w4a8_int8_weight, quantize_w4a8_int8_weight

        group_size = 32 if fmt == "w6" else 16
        weight = (torch.randn(N, K, generator=torch.Generator().manual_seed(5)) * 0.02).to(torch.float16)
        kwargs = dict(group_size=group_size, stochastic_rounding=stochastic_rounding, **FORMATS[fmt])
        ref = eager_w4a8.quantize_w4a8_int8_weight(weight, **kwargs)
        got = quantize_w4a8_int8_weight(weight.to("mps"), **kwargs)
        for r, g in zip(ref, got, strict=True):
            assert (r is None) == (g is None)
            if r is not None:
                assert g.device.type == "mps" and g.dtype == r.dtype
                assert torch.equal(g.cpu().view(torch.uint8), r.view(torch.uint8))

        q, s_rel, s_channel, correction, codebook = ref
        deq_ref = eager_w4a8.dequantize_w4a8_int8_weight(
            q, s_rel, s_channel, codebook, correction, group_size, 256, torch.float16
        )
        q, s_rel, s_channel, correction, codebook = got
        deq = dequantize_w4a8_int8_weight(q, s_rel, s_channel, codebook, correction, group_size, 256, torch.float16)
        assert torch.equal(deq.cpu(), deq_ref)

    @pytest.mark.parametrize("fmt", ["w4_codebook", "w4_uniform", "w6", "w4_asym"])
    def test_comfyui_lora_flow_matches_cpu(self, fmt):
        """ComfyUI's LoRA patch, end to end: a QuantizedTensor loaded on the GPU with the
        checkpoint's fp8 scale, cast to the LoRA compute dtype, dequantized, plus a low-rank
        delta, requantized with "recalculate" and a seed, then run as a linear. The same flow
        on CPU must produce the same packed weight, and the MPS linear must match the float
        linear over that CPU weight."""
        from comfy_kitchen.tensor import AsymW4A8Int8Layout, QuantizedTensor

        group_size = 32 if fmt == "w6" else 16
        q, s_rel, s_channel, correction, codebook = quantize(fmt, group_size)

        def loaded(device):
            params = AsymW4A8Int8Layout.Params(
                scale=s_rel.to(device) if device == "cpu" else mps_scale_like_comfyui(s_rel),
                s_channel=s_channel.to(device),
                correction=None if correction is None else correction.to(device),
                codebook=None if codebook is None else codebook.to(device),
                group_size=group_size, convrot_groupsize=256,
                orig_dtype=torch.bfloat16, orig_shape=(N, K),
            )
            return QuantizedTensor(q.to(device), "AsymW4A8Int8Layout", params)

        gen = torch.Generator().manual_seed(6)
        lora_a = torch.randn(8, K, generator=gen) * 0.02
        lora_b = torch.randn(N, 8, generator=gen) * 0.02

        def patch(device):
            qt = loaded(device).to(torch.float16)  # patch_weight_to_device: lora_compute_dtype
            w = qt.dequantize()
            w = w + (lora_b.to(device) @ lora_a.to(device)).to(w.dtype)
            return qt.requantize_from_float(w, scale="recalculate", stochastic_rounding=77, inplace_ops=True)

        cpu_qt, mps_qt = patch("cpu"), patch("mps")
        cpu_parts = AsymW4A8Int8Layout.get_plain_tensors(cpu_qt)
        mps_parts = AsymW4A8Int8Layout.get_plain_tensors(mps_qt)
        assert mps_qt._qdata.device.type == "mps"
        assert mps_qt._params.scale.dtype == s_rel.dtype  # the next requantize reads it back
        for c, m in zip(cpu_parts, mps_parts, strict=True):
            assert (c is None) == (m is None)
            if c is not None:
                assert torch.equal(m.cpu().view(torch.uint8), c.view(torch.uint8))

        x = torch.randn(3, K, generator=gen).to(torch.float16)
        got = F.linear(x.to("mps"), mps_qt)
        assert got.device.type == "mps" and got.dtype == torch.float16
        qd, sr, sc, corr, cb = cpu_parts
        assert_close_to_oracle(got, x, (qd, sr, sc, corr, cb), None, group_size, 256)


class TestRangeAndSemantics:
    """Review 2026-10-02 findings 3 and 6 on the W4A8 path, and the shared range machinery."""

    @staticmethod
    def uniform_asym(n, k, code_byte, corr_value, gs=16):
        q = torch.full((n, k // 2), code_byte, dtype=torch.uint8).view(torch.int8)
        return q, torch.zeros(n, k // gs), torch.ones(n), torch.full((k // gs, n), corr_value)

    def test_f3_asymmetric_bias_is_added_before_rounding(self):
        # Codes 0x88 are level 0, so the weight is the unit correction: x . w = 255.75, which
        # bf16 rounds to 256; eager adds bias -256 inside F.linear and returns -0.25.
        q, s_rel, s_ch, corr = self.uniform_asym(1, 256, 0x88, 1.0)
        x = torch.ones(1, 256)
        x[0, -1] = 0.75
        x, bias = x.bfloat16(), torch.tensor([-256.0]).bfloat16()
        want = eager_w4a8.w4a8_int8_linear(x, q, s_rel, s_ch, None, corr, bias, 16, 256, torch.bfloat16)
        assert want.item() == -0.25
        got = ck.w4a8_int8_linear(x.to("mps"), *to_mps(q, s_rel, s_ch), None, corr.to("mps"), bias.to("mps"),
                                  16, 256, torch.bfloat16)
        assert got.cpu().item() == -0.25

    def test_asymmetric_bias_keeps_input_precision(self):
        # Re-review finding 1: fp32 x and bias, bf16 out. The linear part is exactly 1; eager
        # adds the fp32 bias -1.001 inside F.linear and returns ~-0.001. Casting bias to bf16
        # first (-1) would return 0.
        q, s_rel, s_ch, corr = self.uniform_asym(1, 256, 0x88, 1 / 256)
        x, bias = torch.ones(1, 256), torch.tensor([-1.001])
        want = eager_w4a8.w4a8_int8_linear(x, q, s_rel, s_ch, None, corr, bias, 16, 256, torch.bfloat16)
        got = ck.w4a8_int8_linear(x.to("mps"), *to_mps(q, s_rel, s_ch), None, corr.to("mps"), bias.to("mps"),
                                  16, 256, torch.bfloat16)
        assert want.item() == pytest.approx(-0.00099945, rel=1e-3)
        assert torch.equal(got.cpu(), want)

    def test_symmetric_bias_rounds_first_like_eager(self):
        # Symmetric W4A8 is eager's int8_linear: round to out_dtype, then add bias. Codes 0x99
        # (level 1) * s_rel 64 * s_channel 2^-6 decode to 1 and x rotates to a single 255.75
        # lane, so x . w = 255.75: eager rounds it to 256 and the bias cancels it to 0.
        from comfy_kitchen.tensor.int8_utils import _build_hadamard

        q = torch.full((1, 128), 0x99, dtype=torch.uint8).view(torch.int8)
        s_rel, s_ch = torch.full((1, 16), 64.0), torch.tensor([2.0 ** -6])
        x = torch.zeros(1, 256)
        x[0, 0] = 255.75
        x = x @ _build_hadamard(256).T  # the kernel's rotation maps it back to [255.75, 0, ...]
        bias = torch.tensor([-256.0]).bfloat16()
        want = eager_w4a8.w4a8_int8_linear(x, q, s_rel, s_ch, None, None, bias, 16, 256, torch.bfloat16)
        got = ck.w4a8_int8_linear(x.to("mps"), *to_mps(q, s_rel, s_ch), None, None, bias.to("mps"),
                                  16, 256, torch.bfloat16)
        assert want.item() == 0.0
        assert torch.equal(got.cpu(), want)

    @staticmethod
    def float64_linear(x, q, s_rel, s_ch, corr, gs=16, rot=256):
        """The exact result: eager's int8 grid, then channel scale, correction and the
        rotation back in float64, so no fp32 step can flush or overflow."""
        from comfy_kitchen.backends.eager.quantization import rotate_int8_convrot_weight

        grid = eager_w4a8._dequant_int4_grouped_to_int8(q, s_rel, None, gs).double()
        w = grid * s_ch.double()[:, None]
        if corr is not None:
            w = w + corr.double().t().repeat_interleave(gs, dim=1)
        return x.double() @ rotate_int8_convrot_weight(w, rot).T

    @pytest.mark.parametrize("sc_exp,x_exp", [(-149, 120), (124, -120)])
    @pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16])
    def test_extreme_channel_scales_decode_exactly(self, sc_exp, x_exp, x_dtype):
        # Re-review finding 2: grid 64 times s_channel 2^-149 flushes to zero and times 2^124
        # overflows fp32 unless the scale is normalized before the multiply.
        q = torch.full((2, 128), 0x99, dtype=torch.uint8).view(torch.int8)
        s_rel = torch.full((2, 16), 64.0)
        s_ch = torch.tensor([2.0 ** sc_exp, 1e-3])
        x = torch.full((1, 256), 2.0 ** x_exp).to(x_dtype)
        want = self.float64_linear(x, q, s_rel, s_ch, None)
        got = mps_w4a8().w4a8_int8_linear(x.to("mps"), *to_mps(q, s_rel, s_ch), None, None, None,
                                          16, 256, torch.float32).cpu().double()
        assert torch.isfinite(got).all() and (got != 0).all()
        assert torch.allclose(got, want, rtol=1e-6 if x_dtype == torch.float32 else 2e-3)  # bf16 x: fp16 GEMM
        if sc_exp == -149:
            assert got[0, 0].item() == 2.0 ** -15
        else:
            assert got[0, 0].item() == 262144.0

    def test_wider_than_fp16_range_computes_in_fp32(self):
        gen = torch.Generator().manual_seed(14)
        k = 65536 + 1024
        q = torch.randint(-128, 128, (4, k // 2), generator=gen, dtype=torch.int8)
        s_rel = torch.rand(4, k // 16, generator=gen) * 8 + 1
        s_ch = torch.rand(4, generator=gen) * 1e-3
        corr = torch.randn(k // 16, 4, generator=gen) * 1e-3
        x = (torch.randn(2, k, generator=gen) * 100).bfloat16()
        want = self.float64_linear(x, q, s_rel, s_ch, corr)
        got = ck.w4a8_int8_linear(x.to("mps"), *to_mps(q, s_rel, s_ch), None, corr.to("mps"), None,
                                  16, 256, torch.float32).cpu().double()
        assert ((got - want).norm() / want.norm()).item() < 1e-5

    @pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("asym", [False, True])
    def test_f6_large_channel_scale_and_correction_stay_finite(self, x_dtype, asym):
        # int8 grid 64 * s_channel 1e4 = 6.4e5 overflowed the fp16 decode even for fp32 compute.
        n, k = 8, 256
        q = torch.full((n, k // 2), 0x99, dtype=torch.uint8).view(torch.int8)  # level 1 (uniform)
        s_rel = torch.full((n, k // 16), 64.0)
        s_ch = torch.logspace(-3, 6, n)
        corr = torch.logspace(2, 7, k // 16 * n).reshape(k // 16, n) if asym else None
        if asym:
            s_ch[2] = 0
        x = (torch.randn(3, k, generator=torch.Generator().manual_seed(8)) * 0.01).to(x_dtype)
        weight = eager_w4a8.dequantize_w4a8_int8_weight(q, s_rel, s_ch, None, corr, 16, 256, torch.float64)
        want = x.double() @ weight.T
        got = mps_w4a8().w4a8_int8_linear(x.to("mps"), *to_mps(q, s_rel, s_ch), None,
                                          None if corr is None else corr.to("mps"), None, 16, 256, torch.float32)
        got = got.cpu().double()
        assert torch.isfinite(got).all()
        tol = 1e-5 if x_dtype == torch.float32 else 2e-3
        assert ((got - want).norm() / want.norm()).item() < tol

    @pytest.mark.parametrize("x_shape,bias_shape,out_shape", [((K,), (1, N), (N,)), ((0, K), (N,), (0, N)),
                                                              ((2, 3, K), (N, 1), (2, 3, N))])
    def test_shapes_follow_eager(self, x_shape, bias_shape, out_shape):
        packed = quantize("w4_codebook")
        q, s_rel, s_channel, _, codebook = packed
        x = torch.randn(x_shape, generator=torch.Generator().manual_seed(9)).bfloat16()
        bias = torch.randn(bias_shape).bfloat16()
        got = ck.w4a8_int8_linear(x.to("mps"), *to_mps(q, s_rel, s_channel), codebook.to("mps"), None,
                                  bias.to("mps"), 16, 256, torch.bfloat16)
        assert tuple(got.shape) == out_shape
        if got.numel():  # eager raises on an empty batch (torch.cat of no chunks)
            want = eager_w4a8.w4a8_int8_linear(x, q, s_rel, s_channel, codebook, None, bias, 16, 256, torch.bfloat16)
            assert tuple(want.shape) == out_shape
            assert_close_to_oracle(got.reshape(-1, N), x.reshape(-1, K), packed, bias.reshape(-1), 16, 256)

    @pytest.mark.parametrize("out_dtype", [torch.float16, torch.float32])
    def test_dequantize_keeps_eagers_contract_at_large_scale(self, out_dtype):
        """Dequantize is eager's fp32 decode cast to the output dtype, then rotated back; past
        fp16's range both produce the same non-finite values in fp16 and exact values in fp32.
        Only the linear path normalizes."""
        from comfy_kitchen.tensor.w4a8_int8 import dequantize_w4a8_int8_weight

        gen = torch.Generator().manual_seed(4)
        q = torch.randint(-128, 128, (4, 128), generator=gen, dtype=torch.int8)
        s_rel, s_ch = torch.full((4, 16), 8.0), torch.tensor([1e-3, 1.0, 1e3, 1e4])
        ref = eager_w4a8.dequantize_w4a8_int8_weight(q, s_rel, s_ch, None, None, 16, 256, out_dtype)
        got = dequantize_w4a8_int8_weight(*to_mps(q, s_rel, s_ch), None, None, 16, 256, out_dtype).cpu()
        finite = torch.isfinite(ref)
        assert torch.equal(torch.isfinite(got), finite)
        assert finite.all() == (out_dtype == torch.float32)
        rows = finite.all(1)  # the rotation back sums in another order: compare at the row's scale
        err = (got[rows] - ref[rows]).float().abs().amax(1)
        assert (err <= 4 * torch.finfo(out_dtype).eps * ref[rows].float().abs().amax(1)).all()


class TestLora:
    """Unmerged LoRA: the result is the float linear over eager's dequantized (physical-basis)
    weight plus ``scale * up @ down``, the weight a merge would have produced before requantizing."""

    @staticmethod
    def lora(rank, down_std=0.05, up_std=0.05, n=N, k=K):
        gen = torch.Generator().manual_seed(5)
        return torch.randn(rank, k, generator=gen) * down_std, torch.randn(n, rank, generator=gen) * up_std

    @staticmethod
    def oracle(x, packed, group_size, down, up, scale, bias=None):
        q, s_rel, s_channel, correction, codebook = packed
        w = eager_w4a8.dequantize_w4a8_int8_weight(
            q, s_rel, s_channel, codebook, correction, group_size, 256, torch.float32
        ).double() + scale * up.double() @ down.double()
        out = x.double() @ w.T
        return out if bias is None else out + bias.double()

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("rank", [16, 64])
    @pytest.mark.parametrize("with_bias", [False, True])
    @pytest.mark.parametrize("fmt", ["w4_codebook", "w6", "w4_asym"])
    def test_matches_merged_oracle(self, dtype, rank, with_bias, fmt):
        group_size = 32 if fmt == "w6" else 16
        packed = quantize(fmt, group_size)
        down, up = self.lora(rank)
        gen = torch.Generator().manual_seed(6)
        x = torch.randn(2, 7, K, generator=gen).to(dtype)
        bias = torch.randn(N, generator=gen).to(dtype) if with_bias else None
        ref = self.oracle(x, packed, group_size, down, up, 0.6, bias)
        base = self.oracle(x, packed, group_size, down, up, 0.0, bias)
        assert ((base - ref).norm() / ref.norm()).item() > 0.1  # the LoRA term matters
        q, s_rel, s_channel, correction, codebook = to_mps(*packed)
        got = ck.w4a8_int8_linear(
            x.to("mps"), q, s_rel, s_channel, codebook=codebook, correction=correction,
            bias=None if bias is None else bias.to("mps"), group_size=group_size, out_dtype=dtype,
            lora_down=down.to("mps", dtype), lora_up=up.to("mps", dtype), lora_scale=0.6,
        )
        assert got.shape == (2, 7, N) and got.dtype == dtype
        out_rounding = torch.finfo(dtype).eps / 2
        got = got.cpu().double()
        assert (got - ref).abs().max().item() <= (3e-3 + 2 * out_rounding) * ref.abs().max().item()
        assert ((got - ref).norm() / ref.norm()).item() < 1.5e-3 + out_rounding

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_large_activations_stay_finite_and_exact(self, dtype):
        """x ~ 2e4 against a down weight of ~4: the low-rank activation passes fp16's range by
        far; per-row accuracy holds for rows 1e6 apart in magnitude."""
        packed = quantize("w6", 32)
        down, up = self.lora(64, down_std=4.0, up_std=1e-3)
        x = torch.randn(5, K, generator=torch.Generator().manual_seed(7)) * 2.0e4
        x[2] *= 1e-6
        x = x.to(dtype)
        ref = self.oracle(x, packed, 32, down, up, 1.0)
        q, s_rel, s_channel, correction, codebook = to_mps(*packed)
        got = mps_w4a8().w4a8_int8_linear(
            x.to("mps"), q, s_rel, s_channel, codebook, correction, group_size=32, out_dtype=dtype,
            lora_down=down.to("mps", dtype), lora_up=up.to("mps", dtype),
        ).cpu().double()
        assert torch.isfinite(got).all()
        tol = 6e-3 if dtype == torch.bfloat16 else 1.5e-3
        for r in range(5):
            assert ((got[r] - ref[r]).norm() / ref[r].norm()).item() < tol

    @pytest.mark.parametrize("n,rank", [(67, 13), (190, 5), (200, 64)])
    @pytest.mark.parametrize("transposed", [False, True])
    @pytest.mark.parametrize("chunked", [False, True])
    @pytest.mark.parametrize("fmt", ["w6", "w4_asym"])
    def test_chunked_transposed_and_tails(self, n, rank, transposed, chunked, fmt, monkeypatch):
        """Ranks off the 8-row tile (13, 5) and columns off the 4- and 64-column grids (67, 190)
        through the transposed decode (row stride N + R8) and through several chunks, where
        later chunks read a compact copy of the first chunk's low-rank columns."""
        from comfy_kitchen.backends.mps import linear

        if transposed:
            monkeypatch.setattr(linear, "_TRANSPOSED_DECODE_MIN_K", 1)
        if chunked:
            monkeypatch.setattr(linear, "_MAX_DECODE_BYTES", 48 * 2 * K)  # 48-row chunks
        group_size = 32 if fmt == "w6" else 16
        packed = quantize(fmt, group_size, n=n)
        down, up = self.lora(rank, n=n)
        x = torch.randn(11, K, generator=torch.Generator().manual_seed(8)).to(torch.float16)
        bias = torch.randn(n, generator=torch.Generator().manual_seed(9)).to(torch.float16)
        ref = self.oracle(x, packed, group_size, down, up, 0.75, bias)
        q, s_rel, s_channel, correction, codebook = to_mps(*packed)
        got = mps_w4a8().w4a8_int8_linear(
            x.to("mps"), q, s_rel, s_channel, codebook, correction, bias.to("mps"), group_size,
            out_dtype=torch.float16, lora_down=down.to("mps"), lora_up=up.to("mps"), lora_scale=0.75,
        ).cpu().double()
        assert ((got - ref).norm() / ref.norm()).item() < 1.5e-3

    def test_shape_mismatch_raises(self):
        packed = to_mps(*quantize("w6", 32))
        with pytest.raises(ValueError, match="lora"):
            mps_w4a8().w4a8_int8_linear(
                torch.randn(3, K, device="mps"), *packed[:3], None, None, group_size=32,
                lora_down=torch.zeros(8, K - 4, device="mps"), lora_up=torch.zeros(N, 8, device="mps"),
            )
