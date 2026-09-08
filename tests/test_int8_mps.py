# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Apple MPS regression tests for fused and fallback INT8 linear paths."""

import platform

import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.backends._activations import apply_input_act
from comfy_kitchen.backends.eager import quantization as eager_quantization
from comfy_kitchen.backends.eager.quantization import (
    _dequantized_int8_linear,
    quantize_int8_rowwise,
)
from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation


def _has_metal_4_mpp() -> bool:
    version = platform.mac_ver()[0]
    if not version:
        return False
    major, minor, *_ = (int(part) for part in version.split("."))
    return (major, minor) >= (26, 2) and hasattr(torch.mps, "compile_shader")


def _reference_int8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    convrot: bool,
    convrot_groupsize: int,
) -> torch.Tensor:
    if convrot:
        h = _build_hadamard(convrot_groupsize, device=x.device, dtype=x.dtype)
        x = _rotate_activation(x, h, convrot_groupsize)
    original_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    quantized, row_scale = quantize_int8_rowwise(x_2d)
    accum = torch.matmul(
        quantized.cpu().to(torch.int32),
        weight.cpu().to(torch.int32).T,
    ).to(x.device)
    result = (
        accum.float()
        * (row_scale.float() * weight_scale.to(x.device, torch.float32).reshape(1, -1))
    ).to(torch.bfloat16)
    if bias is not None:
        result = result + bias.to(x.device, torch.bfloat16).reshape(1, -1)
    return result.reshape(*original_shape[:-1], weight.shape[0])


def test_dequantized_int8_linear_supports_channel_scales_and_bias(seed):
    x = torch.randn(3, 16, dtype=torch.bfloat16)
    weight = torch.randint(-127, 128, (7, 16), dtype=torch.int8)
    weight_scale = torch.rand(7, dtype=torch.float32)
    bias = torch.randn(7, dtype=torch.bfloat16)

    actual = _dequantized_int8_linear(
        x,
        weight,
        weight_scale,
        bias,
        torch.bfloat16,
        False,
        16,
    )
    expected_weight = weight.to(torch.bfloat16)
    expected_weight.mul_(weight_scale.to(torch.bfloat16).reshape(-1, 1))
    expected = torch.nn.functional.linear(x, expected_weight, bias)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(
    not torch.backends.mps.is_available() or not _has_metal_4_mpp(),
    reason="Metal 4 MPP on MPS required",
)
@pytest.mark.parametrize("scalar_scale", [False, True])
@pytest.mark.parametrize("convrot", [False, True])
def test_eager_int8_linear_mps_uses_fused_kernel(seed, monkeypatch, scalar_scale, convrot):
    device = torch.device("mps")
    group_size = 16
    x = torch.randn(4, 32, device=device, dtype=torch.bfloat16)
    weight = torch.randint(-127, 128, (12, 32), device=device, dtype=torch.int8)
    scale_shape = () if scalar_scale else (12,)
    weight_scale = torch.rand(scale_shape, device=device, dtype=torch.float32)
    bias = torch.randn(12, device=device, dtype=torch.bfloat16)

    def unexpected_fallback(*_args, **_kwargs):
        raise AssertionError("Metal 4 MPS path must not widen the INT8 weight")

    monkeypatch.setattr(eager_quantization, "_dequantized_int8_linear", unexpected_fallback)

    with ck.registry.use_backend("eager"):
        actual = ck.int8_linear(
            x,
            weight,
            weight_scale,
            bias=bias,
            out_dtype=torch.bfloat16,
            convrot=convrot,
            convrot_groupsize=group_size,
        )

    expected = _reference_int8_linear(
        x,
        weight,
        weight_scale.reshape(-1),
        bias,
        convrot,
        group_size,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.device.type == "mps"


@pytest.mark.skipif(
    not torch.backends.mps.is_available() or not _has_metal_4_mpp(),
    reason="Metal 4 MPP on MPS required",
)
def test_eager_int8_linear_mps_fuses_after_swiglu(seed, monkeypatch):
    device = torch.device("mps")
    x = torch.randn(3, 32, device=device, dtype=torch.bfloat16)
    weight = torch.randint(-127, 128, (8, 16), device=device, dtype=torch.int8)
    weight_scale = torch.rand(8, device=device, dtype=torch.float32)

    def unexpected_fallback(*_args, **_kwargs):
        raise AssertionError("Metal 4 MPS path must not widen the INT8 weight")

    monkeypatch.setattr(eager_quantization, "_dequantized_int8_linear", unexpected_fallback)

    with ck.registry.use_backend("eager"):
        actual = ck.int8_linear(
            x,
            weight,
            weight_scale,
            out_dtype=torch.bfloat16,
            input_act="swiglu",
        )

    activated = apply_input_act(x, "swiglu")
    expected = _reference_int8_linear(
        activated,
        weight,
        weight_scale,
        None,
        False,
        256,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
