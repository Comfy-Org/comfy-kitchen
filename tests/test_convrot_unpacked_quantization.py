"""Ensure unpacked activation quantization preserves the packed reference."""

import pytest
import torch

from comfy_kitchen.backends.eager.convrot_w4a4 import (
    _quantize_signed_int4_rowwise_unpacked,
    _round_int4,
    quantize_signed_int4_rowwise,
)
from comfy_kitchen.backends.eager.svdquant import (
    _INT4_MAX,
    _pack_int4_row_major,
    _unpack_int4_row_major,
)


def _packed_reference(x, stochastic_rounding):
    # Pre-refactor implementation, independent of the new helper.
    rows, _ = x.shape
    absmax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    scales = absmax / _INT4_MAX
    q = _round_int4(x / scales, stochastic_rounding=stochastic_rounding)
    return _pack_int4_row_major(q), scales.reshape(rows).to(torch.float32)


@pytest.fixture(params=["cpu", "npu"])
def quant_device(request):
    if request.param == "npu":
        npu = getattr(torch, "npu", None)
        if npu is None or not npu.is_available():
            pytest.skip("Huawei Ascend device required")
        npu.set_device("npu:0")
    return request.param


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seed", [None, 0, 123])
@pytest.mark.parametrize("layout", ["contiguous", "strided", "transposed"])
def test_unpacked_quantization_matches_original(quant_device, dtype, seed, layout):
    torch.manual_seed(42)
    x = torch.randn(7, 512, device=quant_device, dtype=dtype)
    if layout == "strided":
        x = x[:, ::2]
    elif layout == "transposed":
        x = x.t().contiguous().t()
    before = x.clone()
    expected_packed, expected_scale = _packed_reference(x, seed)
    codes, scale = _quantize_signed_int4_rowwise_unpacked(x, seed)
    packed, packed_scale = quantize_signed_int4_rowwise(x, seed)

    assert codes.dtype == torch.int8
    assert codes.shape == x.shape
    assert scale.dtype == torch.float32
    torch.testing.assert_close(codes, _unpack_int4_row_major(expected_packed), rtol=0, atol=0)
    torch.testing.assert_close(scale, expected_scale, rtol=0, atol=0)
    torch.testing.assert_close(packed, expected_packed, rtol=0, atol=0)
    torch.testing.assert_close(packed_scale, expected_scale, rtol=0, atol=0)
    torch.testing.assert_close(x, before, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seed", [0, 123])
def test_unpacked_quantization_edge_rows(quant_device, dtype, seed):
    x = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1e-12, -1e-12, 1e-6, -1e-6, 0, 0, 0, 0],
            [-7, -2.5001, -2.5, -2.4999, 2.4999, 2.5, 2.5001, 7],
            [float("nan"), 0, 1, -1, 0, 0, 0, 0],
            [float("inf"), -float("inf"), 1, -1, 0, 0, 0, 0],
            [torch.finfo(dtype).max, -torch.finfo(dtype).max, 1, -1, 0, 0, 0, 0],
        ],
        device=quant_device,
        dtype=dtype,
    )
    expected_packed, expected_scale = _packed_reference(x, seed)
    codes, scale = _quantize_signed_int4_rowwise_unpacked(x, seed)
    packed, packed_scale = quantize_signed_int4_rowwise(x, seed)
    torch.testing.assert_close(codes, _unpack_int4_row_major(expected_packed), rtol=0, atol=0)
    torch.testing.assert_close(scale, expected_scale, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(packed, expected_packed, rtol=0, atol=0)
    torch.testing.assert_close(packed_scale, expected_scale, rtol=0, atol=0, equal_nan=True)


def test_packed_quantization_still_rejects_odd_width():
    with pytest.raises(ValueError, match="last dim must be even"):
        quantize_signed_int4_rowwise(torch.ones(2, 3))


def test_unpacked_quantization_empty_rows():
    x = torch.empty(0, 256)
    codes, scale = _quantize_signed_int4_rowwise_unpacked(x)
    packed, packed_scale = quantize_signed_int4_rowwise(x)
    assert codes.shape == (0, 256)
    assert packed.shape == (0, 128)
    assert scale.shape == packed_scale.shape == (0,)
