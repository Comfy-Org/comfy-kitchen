import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.backends import ascend as ascend_backend
from comfy_kitchen.backends.eager.quantization import (
    quantize_and_rotate_rowwise as eager_quantize_and_rotate_rowwise,
)
from comfy_kitchen.exceptions import BackendNotImplementedError, NoCapableBackendError
from comfy_kitchen.registry import registry

from .conftest import get_supported_devices

torch_npu = pytest.importorskip("torch_npu")
requires_npu_quant_matmul = pytest.mark.skipif(
    not hasattr(torch_npu, "npu_quant_matmul"),
    reason="torch-npu with npu_quant_matmul is required",
)
requires_npu_rotate_quant = pytest.mark.skipif(
    not hasattr(torch_npu, "npu_rotate_quant"),
    reason="torch-npu with npu_rotate_quant is required",
)

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


@requires_npu_rotate_quant
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("group_size", [16, 64, 256])
def test_quantize_and_rotate_rowwise_matches_separate_ops(ascend_device, dtype, group_size):
    from comfy_kitchen.tensor.int8_utils import _build_hadamard

    x = torch.randn(2, 3, 512, device=ascend_device, dtype=dtype)
    x[0, 0].zero_()
    h = _build_hadamard(group_size, device=ascend_device, dtype=dtype)

    expected_q, expected_scale = eager_quantize_and_rotate_rowwise(x, h, group_size)
    actual_q, actual_scale = ascend_backend.quantize_and_rotate_rowwise(x, h, group_size)

    assert actual_q.shape == x.shape
    assert actual_q.dtype == torch.int8
    assert actual_q.device.type == "npu"
    assert actual_scale.shape == (*x.shape[:-1], 1)
    assert actual_scale.dtype == torch.float32
    assert actual_scale.device.type == "npu"
    torch.testing.assert_close(actual_q.float(), expected_q.float(), rtol=0, atol=1)
    torch.testing.assert_close(actual_scale, expected_scale, rtol=1e-3, atol=1e-6)
    assert torch.all(actual_scale > 0)


@requires_npu_rotate_quant
def test_quantize_and_rotate_rowwise_handles_noncontiguous_input(ascend_device):
    x = torch.randn(5, 128, 2, device=ascend_device, dtype=torch.float16)[..., 0]
    h = torch.randn(16, 16, device=ascend_device, dtype=x.dtype) / 4
    assert not x.is_contiguous()

    expected_q, expected_scale = eager_quantize_and_rotate_rowwise(x, h, 16)
    actual_q, actual_scale = ascend_backend.quantize_and_rotate_rowwise(x, h, 16)

    torch.testing.assert_close(actual_q.float(), expected_q.float(), rtol=0, atol=1)
    torch.testing.assert_close(actual_scale, expected_scale, rtol=1e-3, atol=1e-6)


@requires_npu_rotate_quant
def test_rotate_quant_backend_selection_and_declines(ascend_device):
    from comfy_kitchen.tensor.int8_utils import _build_hadamard

    x = torch.randn(4, 256, device=ascend_device, dtype=torch.bfloat16)
    h = _build_hadamard(64, device=ascend_device, dtype=x.dtype)
    call = {"x": x, "H": h, "group_size": 64, "stochastic_rounding": 0}
    assert registry.get_capable_backend("quantize_and_rotate_rowwise", call) == "ascend"

    too_small = dict(call, x=x[:, :64])
    assert registry.get_capable_backend("quantize_and_rotate_rowwise", too_small) == "eager"

    stochastic = dict(call, stochastic_rounding=123)
    assert registry.get_capable_backend("quantize_and_rotate_rowwise", stochastic) == "eager"

    wrong_shape = dict(call, H=h[:16, :16])
    assert registry.get_capable_backend("quantize_and_rotate_rowwise", wrong_shape) == "eager"


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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_quantize_int8_tensorwise_handles_zero_scale(ascend_device, dtype):
    x = torch.tensor([-1.0, 0.0, 1.0], device=ascend_device, dtype=dtype)
    scale = torch.tensor(0.0, device=ascend_device)

    with ck.use_backend("eager"):
        expected_q, expected_scale = ck.quantize_int8_tensorwise(x, scale=scale)
    with ck.use_backend("ascend"):
        actual_q, actual_scale = ck.quantize_int8_tensorwise(x, scale=scale)

    torch.testing.assert_close(actual_q, expected_q)
    torch.testing.assert_close(actual_scale, expected_scale)
    assert actual_scale.item() == 0.0


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

    linear_call = {
        "x": x,
        "weight": torch.ones(16, 32, device=ascend_device, dtype=torch.int8),
        "weight_scale": torch.ones(16, device=ascend_device),
        "out_dtype": torch.bfloat16,
    }
    if not hasattr(torch_npu, "npu_quant_matmul"):
        with pytest.raises(BackendNotImplementedError):
            registry.get_implementation("int8_linear", backend="ascend", kwargs=linear_call)
        return

    selected = registry.get_capable_backend("int8_linear", linear_call)
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


def _int8_linear_reference(x, weight, weight_scale, bias=None):
    quantized_x, activation_scale = torch_npu.npu_dynamic_quant(x)
    output = torch.matmul(quantized_x.float(), weight.t().float())
    output *= activation_scale.reshape(-1, 1)
    output *= weight_scale.reshape(1, -1)
    if bias is not None:
        output += bias.reshape(1, -1)
    return output


@requires_npu_quant_matmul
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("scalar_weight_scale", [False, True])
def test_int8_linear_matches_reference(ascend_device, dtype, with_bias, scalar_weight_scale):
    m, n, k = 17, 64, 128
    x = torch.randn(m, k, device=ascend_device, dtype=dtype)
    weight = torch.randint(-127, 128, (n, k), device=ascend_device, dtype=torch.int8)
    scale_shape = (1,) if scalar_weight_scale else (n,)
    weight_scale = torch.rand(scale_shape, device=ascend_device, dtype=torch.float32) / 127
    bias = torch.randn(n, device=ascend_device, dtype=torch.float32) if with_bias else None

    expected = _int8_linear_reference(x, weight, weight_scale, bias).to(dtype)
    with ck.use_backend("ascend"):
        actual = ck.int8_linear(x, weight, weight_scale, bias=bias, out_dtype=dtype)

    assert actual.device.type == "npu"
    assert actual.shape == (m, n)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@requires_npu_quant_matmul
def test_int8_linear_supports_batched_input(ascend_device):
    x = torch.randn(2, 3, 128, device=ascend_device, dtype=torch.bfloat16)
    weight = torch.randint(-127, 128, (64, 128), device=ascend_device, dtype=torch.int8)
    weight_scale = torch.rand(64, 1, device=ascend_device, dtype=torch.float32) / 127

    with ck.use_backend("ascend"):
        output = ck.int8_linear(x, weight, weight_scale)

    assert output.shape == (2, 3, 64)
    assert output.device.type == "npu"


@requires_npu_quant_matmul
@pytest.mark.parametrize("input_act", [None, "gelu_tanh", "swiglu"])
def test_convrot_int8_linear_matches_reference(ascend_device, input_act):
    from comfy_kitchen.backends._activations import apply_input_act
    from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation

    m, n, k = 9, 64, 128
    raw_k = k * 2 if input_act == "swiglu" else k
    x = torch.randn(m, raw_k, device=ascend_device, dtype=torch.bfloat16)
    weight = torch.randint(-127, 128, (n, k), device=ascend_device, dtype=torch.int8)
    weight_scale = torch.rand(n, device=ascend_device, dtype=torch.float32) / 127
    bias = torch.randn(n, device=ascend_device, dtype=torch.bfloat16)

    activated = apply_input_act(x, input_act)
    hadamard = _build_hadamard(64, device=ascend_device, dtype=x.dtype)
    rotated = _rotate_activation(activated, hadamard, 64)
    expected = _int8_linear_reference(rotated, weight, weight_scale, bias).bfloat16()

    with ck.use_backend("ascend"):
        actual = ck.int8_linear(
            x,
            weight,
            weight_scale,
            bias=bias,
            convrot=True,
            convrot_groupsize=64,
            input_act=input_act,
        )

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@requires_npu_quant_matmul
@requires_npu_rotate_quant
def test_convrot_int8_linear_uses_rotate_quant(ascend_device, monkeypatch):
    calls = 0
    original = torch_npu.npu_rotate_quant

    def counted_rotate_quant(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_npu, "npu_rotate_quant", counted_rotate_quant)
    x = torch.randn(7, 256, device=ascend_device, dtype=torch.bfloat16)
    weight = torch.randint(-127, 128, (64, 256), device=ascend_device, dtype=torch.int8)
    weight_scale = torch.rand(64, device=ascend_device, dtype=torch.float32) / 127

    with ck.use_backend("ascend"):
        ck.int8_linear(
            x,
            weight,
            weight_scale,
            convrot=True,
            convrot_groupsize=64,
        )

    assert calls == 1


@requires_npu_quant_matmul
def test_convrot_int8_linear_falls_back_when_rotate_quant_is_unavailable(
    ascend_device, monkeypatch
):
    def unexpected_rotate_quant(*args, **kwargs):
        raise AssertionError("npu_rotate_quant should not be called")

    monkeypatch.setattr(ascend_backend, "_ASCEND_ROTATE_QUANT_AVAILABLE", False)
    if hasattr(torch_npu, "npu_rotate_quant"):
        monkeypatch.setattr(torch_npu, "npu_rotate_quant", unexpected_rotate_quant)

    x = torch.randn(7, 256, device=ascend_device, dtype=torch.bfloat16)
    weight = torch.randint(-127, 128, (64, 256), device=ascend_device, dtype=torch.int8)
    weight_scale = torch.rand(64, device=ascend_device, dtype=torch.float32) / 127

    with ck.use_backend("ascend"):
        output = ck.int8_linear(
            x,
            weight,
            weight_scale,
            convrot=True,
            convrot_groupsize=64,
        )

    assert output.shape == (7, 64)
    assert output.device.type == "npu"


@requires_npu_quant_matmul
@requires_npu_rotate_quant
def test_convrot_int8_linear_uses_separate_path_for_small_groups(ascend_device, monkeypatch):
    def unexpected_rotate_quant(*args, **kwargs):
        raise AssertionError("unsupported shapes must use the separate NPU path")

    monkeypatch.setattr(torch_npu, "npu_rotate_quant", unexpected_rotate_quant)
    x = torch.randn(7, 64, device=ascend_device, dtype=torch.bfloat16)
    weight = torch.randint(-127, 128, (64, 64), device=ascend_device, dtype=torch.int8)
    weight_scale = torch.rand(64, device=ascend_device, dtype=torch.float32) / 127

    with ck.use_backend("ascend"):
        output = ck.int8_linear(
            x,
            weight,
            weight_scale,
            convrot=True,
            convrot_groupsize=4,
        )

    assert output.shape == (7, 64)


@requires_npu_quant_matmul
@pytest.mark.parametrize(
    "kwargs,failed_param",
    [
        ({"weight_scale_size": 3}, "weight_scale"),
        ({"bias_size": 3}, "bias"),
        ({"weight_k": 64}, "weight"),
        ({"convrot": True, "convrot_groupsize": 32}, "convrot_groupsize"),
        ({"input_act": "silu"}, "input_act"),
    ],
)
def test_int8_linear_declines_unsupported_contracts(ascend_device, kwargs, failed_param):
    x = torch.randn(4, 128, device=ascend_device, dtype=torch.bfloat16)
    weight = torch.ones(64, kwargs.get("weight_k", 128), device=ascend_device, dtype=torch.int8)
    weight_scale = torch.ones(
        kwargs.get("weight_scale_size", 64), device=ascend_device, dtype=torch.float32
    )
    bias = torch.ones(kwargs.get("bias_size", 64), device=ascend_device, dtype=torch.bfloat16)
    call = {
        "x": x,
        "weight": weight,
        "weight_scale": weight_scale,
        "bias": bias,
        "out_dtype": torch.bfloat16,
        "convrot": kwargs.get("convrot", False),
        "convrot_groupsize": kwargs.get("convrot_groupsize", 64),
        "input_act": kwargs.get("input_act"),
    }

    result = registry.validate_backend_for_call("ascend", "int8_linear", call)
    assert not result.success
    assert result.failed_param == failed_param
