# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Apple MPS regression tests for the eager INT8 linear fallback."""

import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.backends._activations import apply_input_act
from comfy_kitchen.backends.eager.quantization import _dequantized_int8_linear


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


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS required")
@pytest.mark.parametrize("scalar_scale", [False, True])
@pytest.mark.parametrize("convrot", [False, True])
def test_eager_int8_linear_mps_avoids_int_mm(seed, monkeypatch, scalar_scale, convrot):
    device = torch.device("mps")
    group_size = 16
    x = torch.randn(4, 32, device=device, dtype=torch.bfloat16)
    weight = torch.randint(-127, 128, (12, 32), device=device, dtype=torch.int8)
    scale_shape = () if scalar_scale else (12,)
    weight_scale = torch.rand(scale_shape, device=device, dtype=torch.float32)
    bias = torch.randn(12, device=device, dtype=torch.bfloat16)

    def unexpected_int_mm(*_args, **_kwargs):
        raise AssertionError("MPS fallback must not call torch int8_mm")

    monkeypatch.setattr(torch, "_int_mm", unexpected_int_mm)
    monkeypatch.setattr(torch, "int8_mm", unexpected_int_mm, raising=False)

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

    expected = _dequantized_int8_linear(
        x,
        weight,
        weight_scale.reshape(-1),
        bias,
        torch.bfloat16,
        convrot,
        group_size,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.device.type == "mps"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS required")
def test_eager_int8_linear_mps_applies_swiglu_before_shape_check(seed, monkeypatch):
    device = torch.device("mps")
    x = torch.randn(3, 32, device=device, dtype=torch.bfloat16)
    weight = torch.randint(-127, 128, (8, 16), device=device, dtype=torch.int8)
    weight_scale = torch.rand(8, device=device, dtype=torch.float32)

    def unexpected_int_mm(*_args, **_kwargs):
        raise AssertionError("MPS fallback must not call torch int8_mm")

    monkeypatch.setattr(torch, "_int_mm", unexpected_int_mm)
    monkeypatch.setattr(torch, "int8_mm", unexpected_int_mm, raising=False)

    with ck.registry.use_backend("eager"):
        actual = ck.int8_linear(
            x,
            weight,
            weight_scale,
            out_dtype=torch.bfloat16,
            input_act="swiglu",
        )

    activated = apply_input_act(x, "swiglu")
    expected = _dequantized_int8_linear(
        activated,
        weight,
        weight_scale,
        None,
        torch.bfloat16,
        False,
        256,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
