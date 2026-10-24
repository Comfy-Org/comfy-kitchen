# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the FP16-tensor-core INT8 fallback (``int8_linear_fp16``).

Volta (sm_70) has no INT8 tensor core, so ``torch._int_mm`` emulates INT8 with
DP4A and ends up slower than the device's FP16 tensor cores. The fallback
dequantizes the INT8 weights to FP16 on the fly (memory footprint unchanged)
and runs the GEMM on FP16, keeping the activation on the INT8 quantization grid
so the numerics stay on par with the INT8 path.
"""

import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.backends.eager import quantization as eager_quant

from .conftest import rel_err


def _int8_tensorwise_weight(n: int, k: int, device, generator):
    """Random INT8 weights plus a per-output-channel scale."""
    weight = torch.randint(
        -127, 128, (n, k), dtype=torch.int8, device=device, generator=generator
    )
    weight_scale = (
        torch.rand(n, 1, device=device, generator=generator) * 0.02 + 0.001
    ).to(torch.float32)
    return weight, weight_scale


def _reference_fp32(x, weight, weight_scale, convrot, group_size):
    """Exact reference: un-rotate the weight, then a float32 matmul."""
    w = weight.float()
    if convrot:
        from comfy_kitchen.tensor.int8_utils import _build_hadamard

        h = _build_hadamard(group_size, device=x.device, dtype=torch.float32)
        n_groups = w.shape[1] // group_size
        w = torch.matmul(w.reshape(w.shape[0], n_groups, group_size), h).reshape(w.shape)
    w = w * weight_scale.reshape(-1, 1).float()
    return torch.matmul(x.float(), w.t())


@pytest.mark.parametrize("convrot", [False, True])
@pytest.mark.parametrize("group_size", [256])
def test_int8_linear_fp16_matches_int8_path(convrot, group_size):
    """The FP16 fallback must stay on the INT8 quantization grid.

    Both paths quantize the activation to INT8 with the same grid, so their
    results agree up to the accumulation precision of the two GEMMs.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(0)

    n, k, m = 512, 4096, 256
    weight, weight_scale = _int8_tensorwise_weight(n, k, device, generator)
    x = torch.randn(m, k, device=device, dtype=torch.float16, generator=generator) * 0.5

    int8_out = eager_quant.int8_linear(
        x, weight, weight_scale, out_dtype=torch.float16, convrot=convrot,
        convrot_groupsize=group_size,
    )
    fp16_out = eager_quant.int8_linear_fp16(
        x, weight, weight_scale, out_dtype=torch.float16, convrot=convrot,
        convrot_groupsize=group_size,
    )

    assert fp16_out.shape == int8_out.shape
    assert fp16_out.dtype == int8_out.dtype
    # Same quantization grid -> the two agree to within the GEMM accumulation
    # error. The INT8 path accumulates in int32 and scales afterwards, the FP16
    # path accumulates in FP16, so a small drift is expected.
    assert rel_err(fp16_out, int8_out) < 0.02


@pytest.mark.parametrize("convrot", [False, True])
def test_int8_linear_fp16_matches_float32_reference(convrot):
    """The fallback must be no further from the float32 truth than INT8 is."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(1)

    n, k, m = 512, 4096, 256
    weight, weight_scale = _int8_tensorwise_weight(n, k, device, generator)
    x = torch.randn(m, k, device=device, dtype=torch.float16, generator=generator) * 0.5

    ref = _reference_fp32(x, weight, weight_scale, convrot, 256)
    int8_out = eager_quant.int8_linear(
        x, weight, weight_scale, out_dtype=torch.float32, convrot=convrot,
    )
    fp16_out = eager_quant.int8_linear_fp16(
        x, weight, weight_scale, out_dtype=torch.float32, convrot=convrot,
    )

    int8_err = rel_err(int8_out, ref)
    fp16_err = rel_err(fp16_out, ref)
    # Both sit on the INT8 activation quantization grid, so their distance to
    # the float32 reference is comparable; the fallback must not be worse.
    assert fp16_err < int8_err * 1.5 + 1e-3


def test_int8_linear_fp16_supports_bias_and_residual():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(2)

    n, k, m = 256, 4096, 128
    weight, weight_scale = _int8_tensorwise_weight(n, k, device, generator)
    x = torch.randn(m, k, device=device, dtype=torch.float16, generator=generator) * 0.5
    bias = torch.randn(n, device=device, dtype=torch.float16, generator=generator)
    residual = torch.randn(m, n, device=device, dtype=torch.float16, generator=generator)
    residual_scale = torch.randn(n, device=device, dtype=torch.float16, generator=generator)

    out = eager_quant.int8_linear_fp16(
        x, weight, weight_scale, bias=bias, out_dtype=torch.float16,
        residual=residual, residual_scale=residual_scale,
    )
    assert out.shape == (m, n)
    assert torch.isfinite(out).all()


def test_int8_linear_fp16_scalar_weight_scale():
    """A scalar (per-tensor) weight scale must be accepted."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(3)

    n, k, m = 128, 4096, 64
    weight, weight_scale = _int8_tensorwise_weight(n, k, device, generator)
    scalar_scale = weight_scale.mean().reshape(1)

    out = eager_quant.int8_linear_fp16(
        x := torch.randn(m, k, device=device, dtype=torch.float16, generator=generator) * 0.5,
        weight, scalar_scale, out_dtype=torch.float16,
    )
    assert out.shape == (m, n)
    assert x.shape == (m, k)


def test_int8_linear_fp16_rejects_shape_mismatch():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(4)

    weight, weight_scale = _int8_tensorwise_weight(64, 4096, device, generator)
    x = torch.randn(8, 2048, device=device, dtype=torch.float16, generator=generator)

    with pytest.raises(ValueError, match="inner dimensions must match"):
        eager_quant.int8_linear_fp16(x, weight, weight_scale, out_dtype=torch.float16)


def test_int8_linear_fp16_is_registered_as_a_capability():
    """The new entry point must be reachable through the public dispatch."""
    backends = ck.list_backends()
    assert "int8_linear_fp16" in backends["eager"]["capabilities"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(5)

    n, k, m = 64, 4096, 32
    weight, weight_scale = _int8_tensorwise_weight(n, k, device, generator)
    x = torch.randn(m, k, device=device, dtype=torch.float16, generator=generator) * 0.5

    out = ck.int8_linear_fp16(x, weight, weight_scale, out_dtype=torch.float16)
    assert out.shape == (m, n)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_int8_linear_delegates_to_fp16_without_int8_tensor_core(monkeypatch):
    """On a device without INT8 tensor cores, ``int8_linear`` takes the FP16 path."""
    monkeypatch.setattr(
        eager_quant, "_cuda_device_has_int8_tensor_core", lambda _index: False
    )

    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(6)

    n, k, m = 256, 4096, 128
    weight, weight_scale = _int8_tensorwise_weight(n, k, device, generator)
    x = torch.randn(m, k, device=device, dtype=torch.float16, generator=generator) * 0.5

    delegated = eager_quant.int8_linear(x, weight, weight_scale, out_dtype=torch.float16)
    direct = eager_quant.int8_linear_fp16(
        x, weight, weight_scale, out_dtype=torch.float16
    )
    assert torch.equal(delegated, direct)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_int8_linear_keeps_int8_path_with_int8_tensor_core(monkeypatch):
    """Devices that do have INT8 tensor cores keep the existing INT8 path."""
    monkeypatch.setattr(
        eager_quant, "_cuda_device_has_int8_tensor_core", lambda _index: True
    )

    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(7)

    n, k, m = 256, 4096, 128
    weight, weight_scale = _int8_tensorwise_weight(n, k, device, generator)
    x = torch.randn(m, k, device=device, dtype=torch.float16, generator=generator) * 0.5

    int8_out = eager_quant.int8_linear(x, weight, weight_scale, out_dtype=torch.float16)
    fp16_out = eager_quant.int8_linear_fp16(
        x, weight, weight_scale, out_dtype=torch.float16
    )
    # The INT8 path accumulates in int32, the FP16 path in FP16, so they differ
    # slightly -- but the INT8 path must not be the FP16 one.
    assert not torch.equal(int8_out, fp16_out)
