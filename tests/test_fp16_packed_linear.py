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
