# SPDX-License-Identifier: Apache-2.0
"""MPS weight-only W4A8 / W6A8 linear against the eager decode.

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


class TestDecode:
    @pytest.mark.parametrize("fmt", list(FORMATS))
    @pytest.mark.parametrize("scale_dtype", [torch.float8_e4m3fn, torch.float32])
    def test_decode_is_bit_identical_to_eager(self, fmt, scale_dtype):
        """Rotated-basis fp16 weight == eager's int8-grid decode * s_channel (+ correction) in fp16.
        The original-basis weight is not compared bitwise: eager rotates it back in the output
        dtype, so its last bit depends on the GEMM that does that."""
        bits = FORMATS[fmt]["bits"]
        group_size = 32 if bits == 6 else 16
        q, s_rel, s_channel, correction, codebook = quantize(fmt, group_size, scale_dtype=scale_dtype)
        grid = eager_w4a8._dequant_int4_grouped_to_int8(q, s_rel, codebook, group_size)
        ref = eager_w4a8._dequantize_w4a8_int8_weight_from_int8(
            grid, s_channel, correction, group_size, torch.float16
        )
        mod = mps_w4a8()
        levels = codebook.to("mps") if codebook is not None else mod._uniform_levels(bits, torch.device("mps"))
        got = mod._decode_w4a8(
            q.to("mps"), s_rel.to("mps"), s_channel.to("mps"), levels,
            None if correction is None else correction.float().to("mps"),
            K, bits, group_size, 0, N,
        ).cpu()
        assert torch.equal(got.view(torch.int16), ref.view(torch.int16))

    @pytest.mark.parametrize("bits,codebook", [(4, True), (4, False), (6, False)])
    @pytest.mark.parametrize("correction", [False, True])
    def test_random_bytes_round_like_eager(self, bits, codebook, correction):
        """Arbitrary packed bytes and every finite e4m3 scale byte. Uniform levels times
        power-of-two scales land on .5 ties and on -0: eager rounds ties to even and its int8
        cast drops the sign of -0, so the kernel must do both."""
        if bits == 6 and codebook:
            pytest.skip("6-bit has no codebook")
        gen = torch.Generator().manual_seed(1)
        n, k, gs = 64, 512, 16
        q = torch.randint(-128, 128, (n, k * bits // 8), generator=gen, dtype=torch.int8)
        s_rel = torch.randint(0, 0x7F, (n, k // gs), generator=gen, dtype=torch.uint8).view(torch.float8_e4m3fn)
        s_channel = torch.rand(n, generator=gen) * 1e-3 + 1e-6
        table = torch.sort(torch.randn(16, generator=gen))[0] if codebook else None
        corr = (torch.randn(k // gs, n, generator=gen) * 1e-2).to(torch.bfloat16) if correction else None
        grid = eager_w4a8._dequant_int4_grouped_to_int8(q, s_rel, table, gs)
        ref = eager_w4a8._dequantize_w4a8_int8_weight_from_int8(grid, s_channel, corr, gs, torch.float16)
        mod = mps_w4a8()
        levels = table.to("mps") if codebook else mod._uniform_levels(bits, torch.device("mps"))
        got = mod._decode_w4a8(
            q.to("mps"), s_rel.to("mps"), s_channel.to("mps"), levels,
            None if corr is None else corr.float().to("mps"), k, bits, gs, 0, n,
        ).cpu()
        assert torch.equal(got.view(torch.int16), ref.view(torch.int16))


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
        """A decode budget of 50 rows (not a divisor of N) runs 4 chunks with a short tail."""
        mod = mps_w4a8()
        group_size = 32 if fmt == "w6" else 16
        packed = quantize(fmt, group_size)
        x = torch.randn(9, K, generator=torch.Generator().manual_seed(4)).to(torch.float16).to("mps")
        q, s_rel, s_channel, correction, codebook = to_mps(*packed)
        kwargs = {"codebook": codebook, "correction": correction, "group_size": group_size, "out_dtype": torch.float16}
        whole = mod.w4a8_int8_linear(x, q, s_rel, s_channel, **kwargs)
        monkeypatch.setattr(mod, "_MAX_DECODE_BYTES", 50 * 2 * K)
        chunked = mod.w4a8_int8_linear(x, q, s_rel, s_channel, **kwargs)
        assert torch.equal(chunked, whole)

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

    def test_rejects_mismatched_k(self):
        q, s_rel, s_channel, _, codebook = to_mps(*quantize("w4_codebook"))
        with pytest.raises(ValueError, match="does not match"):
            mps_w4a8().w4a8_int8_linear(torch.randn(3, K + 16, device="mps"), q, s_rel, s_channel, codebook)
