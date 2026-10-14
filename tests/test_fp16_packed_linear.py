# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""fp16_packed_linear: fp16 weights, activations scaled per row into fp16, packed fp16
products folded into fp32 every 16. Compared against the fp64 GEMM of the inputs."""

import pytest
import torch

import comfy_kitchen as ck


def _nrmse(actual, expected):
    actual, expected = actual.double(), expected.double()
    return ((actual - expected).norm() / expected.norm()).item()


def _reference(x, w, b):
    return torch.nn.functional.linear(x.double(), w.double(), None if b is None else b.double())


@pytest.mark.parametrize("m,n,k", [(9, 40, 64), (100, 129, 72), (300, 1000, 5376), (2048, 512, 1024)])
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("with_bias", [False, True])
def test_matches_fp64_gemm(m, n, k, x_dtype, with_bias, cuda_available):
    if not cuda_available:
        pytest.skip("CUDA required")
    torch.manual_seed(0)
    x = torch.randn(m, k, device="cuda").to(x_dtype)
    w = (torch.randn(n, k, device="cuda") * 0.05).half()
    b = torch.randn(n, device="cuda") if with_bias else None

    out = ck.fp16_packed_linear(x, w, b, out_dtype=torch.float32)

    assert out.dtype == torch.float32 and out.shape == (m, n)
    assert _nrmse(out, _reference(x, w, b)) < 2e-3


def test_rows_of_any_range_stay_finite_and_accurate(cuda_available):
    if not cuda_available:
        pytest.skip("CUDA required")
    torch.manual_seed(1)
    x = torch.randn(64, 2048, device="cuda")
    x[0] *= 1e6
    x[1] *= 1e-6
    x[2] = 0
    x[3, :10] = 3e4
    w = (torch.randn(256, 2048, device="cuda") * 20).half()  # large weights tighten the scale
    out = ck.fp16_packed_linear(x, w, out_dtype=torch.float32)
    expected = _reference(x, w, None)

    assert torch.isfinite(out).all()
    assert torch.count_nonzero(out[2]) == 0
    for row in (0, 1, 3, 4):
        assert _nrmse(out[row], expected[row]) < 2e-3


def test_small_weights_keep_scaled_rows_in_fp16_range(cuda_available):
    if not cuda_available:
        pytest.skip("CUDA required")
    torch.manual_seed(2)
    x = torch.randn(64, 2048, device="cuda")
    x[0] *= 1e6
    w = (torch.randn(256, 2048, device="cuda") * 1e-4).half()  # stage sums alone would scale rows past fp16
    out = ck.fp16_packed_linear(x, w, out_dtype=torch.float32)
    expected = _reference(x, w, None)

    assert torch.isfinite(out).all()
    for row in (0, 1):
        assert _nrmse(out[row], expected[row]) < 2e-3


@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_output_dtypes_and_given_weight_amax(out_dtype, cuda_available):
    if not cuda_available:
        pytest.skip("CUDA required")
    torch.manual_seed(2)
    x = torch.randn(2, 50, 64, device="cuda")
    w = torch.randn(48, 64, device="cuda").half()
    amax = torch.linalg.vector_norm(w, float("inf"), dtype=torch.float32)
    out = ck.fp16_packed_linear(x, w, out_dtype=out_dtype, weight_amax=amax)
    assert out.shape == (2, 50, 48) and out.dtype == out_dtype
    assert _nrmse(out, _reference(x, w, None)) < 1e-2


def test_accelerated_only_on_simt_gemm_devices(cuda_available):
    if not cuda_available:
        pytest.skip("CUDA required")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] if torch.version.hip else None
    assert ck.fp16_packed_linear_is_accelerated(torch.device("cuda", 0)) == (arch == "gfx1010")
    assert not ck.fp16_packed_linear_is_accelerated(torch.device("cpu"))


@pytest.mark.parametrize("input_act", [None, "rms_norm", "swiglu", "gelu_tanh"])
@pytest.mark.parametrize("x_dtype", [torch.float16, torch.float32])
def test_int8_convrot_linear_on_packed_gemm(input_act, x_dtype, cuda_available):
    """On SIMT GEMM devices int8_linear runs ConvRot INT8 weights on the packed fp16
    GEMM against fp16 activations, so only the weights' int8 rounding remains."""
    if not cuda_available:
        pytest.skip("CUDA required")
    if not ck.fp16_packed_linear_is_accelerated(torch.device("cuda", 0)):
        pytest.skip("packed fp16 GEMM device required")
    from comfy_kitchen.backends import amd
    from comfy_kitchen.backends._activations import apply_input_act

    _, hip = amd.backend_for(torch.device("cuda", 0))

    torch.manual_seed(3)
    m, n, k = 300, 384, 1024
    width = 2 if input_act == "swiglu" else 1
    x = torch.randn(m, k * width, device="cuda").to(x_dtype)
    act_weight = (torch.rand(k, device="cuda") + 0.5).to(x_dtype) if input_act == "rms_norm" else None
    wq, ws = hip.quantize_int8_convrot_weight(torch.randn(n, k, device="cuda") * 0.05, 256)
    b = torch.randn(n, device="cuda")

    out = hip.int8_linear(x, wq, ws, b, torch.float32, convrot=True, input_act=input_act,
                          input_act_weight=act_weight, input_act_eps=1e-6)

    w = hip.dequantize_int8_convrot_weight_dtype(wq, ws, 256, 0)
    expected = _reference(apply_input_act(x.double(), input_act,
                                          None if act_weight is None else act_weight.double(), 1e-6),
                          w, b)
    assert _nrmse(out, expected) < 2e-3


class TestQ8_0Linear:
    """q8_0_linear: a GGUF Q8_0 weight given as raw blocks. rdna1 reads the codes in its tiled
    packed GEMM and applies the block scales at each stage's fold; small M falls back to the
    dequantized fp16 GEMM. Either way the result matches the dequantized weight's product."""

    @staticmethod
    def _q8_0(n, k, device):
        scales = (torch.rand(n, k // 32, device=device) * 0.02 + 1e-3).half()
        codes = torch.randint(-127, 128, (n, k // 32, 32), dtype=torch.int8, device=device)
        raw = torch.cat([scales.view(torch.uint8).reshape(n, k // 32, 2), codes.view(torch.uint8)], dim=2)
        weight = (codes.float() * scales.float()[..., None]).reshape(n, k)
        return raw.reshape(n, -1).contiguous(), weight

    @pytest.mark.parametrize("m,k,n", [(4224, 1024, 768), (300, 640, 512), (130, 2048, 1024), (64, 512, 256), (9, 256, 128)])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("with_bias", [False, True])
    def test_matches_dequantized(self, m, k, n, dtype, with_bias):
        if not torch.cuda.is_available():
            pytest.skip("CUDA/HIP required")
        torch.manual_seed(0)
        raw, weight = self._q8_0(n, k, "cuda")
        x = (torch.randn(m, k, device="cuda") * 4).to(dtype)
        bias = torch.randn(n, device="cuda") if with_bias else None
        ref = x.double() @ weight.double().T + (0 if bias is None else bias.double())
        out = ck.q8_0_linear(x, raw, n, bias)
        assert out.shape == (m, n) and out.dtype == dtype
        assert (out.double() - ref).norm() / ref.norm() < (2e-3 if dtype != torch.bfloat16 else 6e-3)

    def test_eager_dequantize_matches_blocks(self):
        from comfy_kitchen.backends.eager.quantization import dequantize_q8_0

        raw, weight = self._q8_0(16, 64, "cpu")
        assert torch.equal(dequantize_q8_0(raw, 16, 64, torch.float32), weight)


class TestQ4KLinear:
    """q4_k_linear: a GGUF Q4_K weight given as raw super-blocks. rdna1 reads centred codes in
    its tiled packed GEMM and applies each sub-block's scale and offset at its stage's fold;
    small M falls back to the dequantized fp16 GEMM."""

    @staticmethod
    def _q4_k(n, k, device):
        g = torch.Generator().manual_seed(0)
        raw = torch.randint(0, 256, (n, k // 256, 144), dtype=torch.uint8, generator=g)
        # d and dmin of a realistic magnitude (raw random bits would hit inf/nan halves)
        d = (torch.rand(n, k // 256, generator=g) * 2e-3 + 1e-4).half()
        dmin = (torch.rand(n, k // 256, generator=g) * 2e-3).half()
        raw[..., 0:2] = d.view(torch.uint8).reshape(n, k // 256, 2)
        raw[..., 2:4] = dmin.view(torch.uint8).reshape(n, k // 256, 2)
        return raw.reshape(n, -1).contiguous().to(device)

    @pytest.mark.parametrize("m,k,n", [(4224, 1024, 768), (300, 768, 512), (130, 2048, 1024), (64, 512, 256), (9, 256, 128)])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("with_bias", [False, True])
    def test_matches_dequantized(self, m, k, n, dtype, with_bias):
        if not torch.cuda.is_available():
            pytest.skip("CUDA/HIP required")
        from comfy_kitchen.backends.eager.quantization import dequantize_q4_k

        torch.manual_seed(0)
        raw = self._q4_k(n, k, "cuda")
        weight = dequantize_q4_k(raw, n, k, torch.float32)
        x = (torch.randn(m, k, device="cuda") * 4).to(dtype)
        bias = torch.randn(n, device="cuda") if with_bias else None
        ref = x.double() @ weight.double().T + (0 if bias is None else bias.double())
        out = ck.q4_k_linear(x, raw, n, bias)
        assert out.shape == (m, n) and out.dtype == dtype
        assert (out.double() - ref).norm() / ref.norm() < (2e-3 if dtype != torch.bfloat16 else 6e-3)

    def test_eager_dequantize_matches_gguf(self):
        gguf = pytest.importorskip("gguf")
        from comfy_kitchen.backends.eager.quantization import dequantize_q4_k

        raw = self._q4_k(8, 512, "cpu")
        ref = gguf.quants.dequantize(raw.numpy(), gguf.GGMLQuantizationType.Q4_K)
        out = dequantize_q4_k(raw, 8, 512, torch.float32)
        assert torch.allclose(out, torch.from_numpy(ref).float(), rtol=0, atol=1e-6)
