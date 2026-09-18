import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.exceptions import NoCapableBackendError
from comfy_kitchen.registry import registry

from .conftest import get_supported_devices

torch_npu = pytest.importorskip("torch_npu")

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="Huawei Ascend device required"
)


@pytest.fixture
def ascend_device():
    torch.npu.set_device("npu:0")
    return torch.device("npu:0")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_quantize_int8_rowwise_matches_eager(ascend_device, dtype):
    x = torch.randn(17, 257, device=ascend_device, dtype=dtype)
    x[0].zero_()

    with ck.use_backend("eager"):
        expected_q, expected_scale = ck.quantize_int8_rowwise(x)
    with ck.use_backend("ascend"):
        actual_q, actual_scale = ck.quantize_int8_rowwise(x)

    assert actual_q.device.type == "npu"
    assert actual_scale.device.type == "npu"
    assert actual_q.dtype == torch.int8
    assert actual_scale.dtype == torch.float32
    assert actual_q.shape == x.shape
    assert actual_scale.shape == (*x.shape[:-1], 1)
    torch.testing.assert_close(actual_q.float(), expected_q.float(), rtol=0, atol=1)
    torch.testing.assert_close(actual_scale, expected_scale, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_quantize_int8_tensorwise_recalculate_matches_eager(ascend_device, dtype):
    x = torch.randn(7, 11, 13, device=ascend_device, dtype=dtype)

    with ck.use_backend("eager"):
        expected_q, expected_scale = ck.quantize_int8_tensorwise(x, scale="recalculate")
    with ck.use_backend("ascend"):
        actual_q, actual_scale = ck.quantize_int8_tensorwise(x, scale="recalculate")

    assert actual_q.device.type == "npu"
    assert actual_scale.device.type == "npu"
    assert actual_q.shape == x.shape
    assert actual_scale.shape == expected_scale.shape == torch.Size([])
    torch.testing.assert_close(actual_q.float(), expected_q.float(), rtol=0, atol=1)
    torch.testing.assert_close(actual_scale, expected_scale, rtol=1e-3, atol=1e-6)


def test_quantize_int8_tensorwise_uses_provided_scale(ascend_device):
    x = torch.randn(5, 9, device=ascend_device, dtype=torch.bfloat16)
    scale = torch.tensor(0.125, device=ascend_device)

    with ck.use_backend("eager"):
        expected_q, expected_scale = ck.quantize_int8_tensorwise(x, scale=scale)
    with ck.use_backend("ascend"):
        actual_q, actual_scale = ck.quantize_int8_tensorwise(x, scale=scale)

    torch.testing.assert_close(actual_q, expected_q)
    torch.testing.assert_close(actual_scale, expected_scale)


def test_dequantize_int8_stays_on_ascend(ascend_device):
    x = torch.randn(19, 67, device=ascend_device, dtype=torch.bfloat16)
    with ck.use_backend("ascend"):
        q, scale = ck.quantize_int8_rowwise(x)
        output = ck.dequantize_int8_simple(q, scale)

    assert output.device.type == "npu"
    assert output.dtype == torch.float32
    torch.testing.assert_close(output, q.float() * scale)


@pytest.mark.parametrize(
    "dtype_code,dtype", [(0, torch.float32), (1, torch.float16), (2, torch.bfloat16)]
)
def test_dequantize_int8_dtype_stays_on_ascend(ascend_device, dtype_code, dtype):
    x = torch.randn(11, 31, device=ascend_device, dtype=torch.bfloat16)
    with ck.use_backend("ascend"):
        q, scale = ck.quantize_int8_rowwise(x)
        output = torch.ops.comfy_kitchen.dequantize_int8_simple_dtype(q, scale, dtype_code)

    assert output.device.type == "npu"
    assert output.dtype == dtype
    torch.testing.assert_close(output, (q.float() * scale).to(dtype))


def test_ascend_is_selected_automatically(ascend_device):
    x = torch.randn(4, 32, device=ascend_device, dtype=torch.bfloat16)
    selected = registry.get_capable_backend(
        "quantize_int8_rowwise", {"x": x, "stochastic_rounding": 0}
    )
    assert selected == "ascend"


def test_get_supported_devices_includes_torch_npu_device_type():
    assert "npu" in get_supported_devices("quantize_int8_rowwise")


def test_ascend_declines_unsupported_calls(ascend_device):
    fp32 = torch.randn(4, 32, device=ascend_device, dtype=torch.float32)
    with pytest.raises(NoCapableBackendError):
        registry.get_implementation(
            "quantize_int8_rowwise",
            backend="ascend",
            kwargs={"x": fp32, "stochastic_rounding": 0},
        )

    bf16 = fp32.bfloat16()
    with pytest.raises(NoCapableBackendError):
        registry.get_implementation(
            "quantize_int8_rowwise",
            backend="ascend",
            kwargs={"x": bf16, "stochastic_rounding": 123},
        )


def test_unsupported_dtype_uses_device_side_eager_fallback(ascend_device):
    x = torch.randn(4, 32, device=ascend_device, dtype=torch.float32)
    selected = registry.get_capable_backend(
        "quantize_int8_rowwise", {"x": x, "stochastic_rounding": 0}
    )
    assert selected == "eager"

    q, scale = ck.quantize_int8_rowwise(x)
    assert q.device.type == "npu"
    assert scale.device.type == "npu"
