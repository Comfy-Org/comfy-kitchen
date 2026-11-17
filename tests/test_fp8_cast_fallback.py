# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""float8 decode on devices without float8 casts (e.g. MPS).

Such devices can store and bit-view float8 tensors but raise ``Undefined type
Float8_e4m3fn`` on any cast. The fallback is exercised on CPU by forcing the probe to
report no cast support, and must match the native cast bit for bit.
"""

import pytest
import torch

from comfy_kitchen import float_utils
from comfy_kitchen.backends.eager import w4a8_int8 as eager_w4a8

FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


@pytest.fixture
def no_fp8_cast(monkeypatch):
    monkeypatch.setattr(float_utils, "supports_fp8_cast", lambda device: False)
    monkeypatch.setattr(float_utils, "_FP8_DECODE_LUTS", {})


def all_bit_patterns(dtype):
    return torch.arange(256, dtype=torch.uint8).view(dtype).reshape(16, 16)


def test_probe_reports_cpu_support_and_caches():
    float_utils._FP8_CAST_SUPPORT.pop("cpu", None)
    assert float_utils.supports_fp8_cast("cpu") is True
    assert float_utils._FP8_CAST_SUPPORT["cpu"] is True
    assert float_utils.supports_fp8_cast(torch.device("cpu")) is True


def test_non_fp8_input_is_a_plain_cast(no_fp8_cast):
    x = torch.randn(4, 8, dtype=torch.bfloat16)
    assert torch.equal(float_utils.fp8_to_float(x), x.float())
    assert not float_utils._FP8_DECODE_LUTS


@pytest.mark.parametrize("fp8_dtype", FP8_DTYPES)
@pytest.mark.parametrize("out_dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_fallback_matches_native_cast_for_every_bit_pattern(no_fp8_cast, fp8_dtype, out_dtype):
    x = all_bit_patterns(fp8_dtype)
    got = float_utils.fp8_to_float(x, out_dtype)
    ref = x.to(out_dtype)
    assert got.dtype == out_dtype and got.shape == x.shape
    assert torch.equal(got.isnan(), ref.isnan())
    assert torch.equal(got.nan_to_num(0.0), ref.nan_to_num(0.0))
    assert len(float_utils._FP8_DECODE_LUTS) == 1


@pytest.mark.parametrize("bits", [4, 6])
def test_w4a8_dequantize_without_fp8_cast(monkeypatch, seed, bits):
    """Group scales are stored as float8_e4m3fn; decoding them must not need a float8 cast."""
    weight = torch.randn(64, 512, dtype=torch.float32) * 0.02
    q, s_rel, s_channel, correction, codebook = eager_w4a8.quantize_w4a8_int8_weight(
        weight, bits=bits
    )
    assert s_rel.dtype == torch.float8_e4m3fn

    def dequantize():
        return eager_w4a8.dequantize_w4a8_int8_weight(
            q, s_rel, s_channel, codebook=codebook, correction=correction,
            output_dtype=torch.float32,
        )

    ref = dequantize()
    monkeypatch.setattr(float_utils, "supports_fp8_cast", lambda device: False)
    monkeypatch.setattr(float_utils, "_FP8_DECODE_LUTS", {})
    got = dequantize()
    assert float_utils._FP8_DECODE_LUTS, "fallback decode was not used"
    assert torch.equal(got, ref)


def test_w4a8_scale_search_without_fp8_cast(monkeypatch, seed):
    weight = torch.randn(64, 512, dtype=torch.float32) * 0.02
    ref = eager_w4a8.quantize_w4a8_int8_weight(weight, bits=6, scale_search=True)
    monkeypatch.setattr(float_utils, "supports_fp8_cast", lambda device: False)
    monkeypatch.setattr(float_utils, "_FP8_DECODE_LUTS", {})
    got = eager_w4a8.quantize_w4a8_int8_weight(weight, bits=6, scale_search=True)
    assert torch.equal(got[0], ref[0])
    assert torch.equal(got[1].view(torch.uint8), ref[1].view(torch.uint8))
