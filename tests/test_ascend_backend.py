import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.backends import ascend as ascend_backend
from comfy_kitchen.backends.eager.convrot_w4a4 import quantize_signed_int4_rowwise
from comfy_kitchen.backends.eager.quantization import (
    quantize_and_rotate_rowwise as eager_quantize_and_rotate_rowwise,
)
from comfy_kitchen.backends.eager.svdquant import _pack_int4_row_major, _unpack_int4_row_major
from comfy_kitchen.exceptions import BackendNotImplementedError, NoCapableBackendError
from comfy_kitchen.registry import registry
from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation

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
requires_ascend_w4a4 = pytest.mark.skipif(
    not hasattr(torch_npu, "npu_quant_matmul"),
    reason="torch-npu with quant matmul is required",
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

    too_wide = dict(
        call,
        x=torch.randn(4, 16384, device=ascend_device, dtype=x.dtype),
    )
    assert registry.get_capable_backend("quantize_and_rotate_rowwise", too_wide) == "eager"

    stochastic = dict(call, stochastic_rounding=123)
    assert registry.get_capable_backend("quantize_and_rotate_rowwise", stochastic) == "eager"

    wrong_shape = dict(call, H=h[:16, :16])
    assert registry.get_capable_backend("quantize_and_rotate_rowwise", wrong_shape) == "eager"


@requires_ascend_w4a4
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_convrot_w4a4_linear_matches_eager(ascend_device, dtype):
    torch.manual_seed(123)
    x = torch.randn(17, 256, device=ascend_device, dtype=dtype) * 0.25
    x[0].zero_()
    weight = torch.randn(128, 256, device=ascend_device, dtype=dtype) * 0.25
    bias = torch.randn(128, device=ascend_device, dtype=dtype) * 0.1

    with ck.use_backend("eager"):
        qweight, wscales = ck.quantize_convrot_w4a4_weight(
            weight,
            convrot_groupsize=64,
        )
        expected = ck.convrot_w4a4_linear(
            x,
            qweight,
            wscales,
            bias=bias,
            convrot_groupsize=64,
        )
    with ck.use_backend("ascend"):
        actual = ck.convrot_w4a4_linear(
            x,
            qweight,
            wscales,
            bias=bias,
            convrot_groupsize=64,
        )

    assert actual.shape == expected.shape
    assert actual.dtype == dtype
    assert actual.device.type == "npu"
    torch.testing.assert_close(actual[0], bias, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@requires_ascend_w4a4
def test_convrot_w4a4_linear_preserves_codes_and_accumulator(ascend_device, monkeypatch):
    calls = []
    original_quant_matmul = torch_npu.npu_quant_matmul
    torch.manual_seed(123)
    qweight = torch.randint(-128, 128, (128, 128), device=ascend_device, dtype=torch.int8)
    original_weight = qweight.clone()
    x = torch.randn(7, 256, device=ascend_device, dtype=torch.float32)
    h = _build_hadamard(64, device=x.device, dtype=x.dtype)
    qref, _ = quantize_signed_int4_rowwise(_rotate_activation(x, h, 64))

    def counted_quant_matmul(x1, x2, scale, **kwargs):
        assert x1.dtype == x2.dtype == torch.int8
        assert kwargs == {"output_dtype": torch.int32}
        torch.testing.assert_close(x1, _unpack_int4_row_major(qref), rtol=0, atol=0)
        torch.testing.assert_close(x2, _unpack_int4_row_major(qweight).t(), rtol=0, atol=0)
        torch.testing.assert_close(scale, torch.ones_like(scale), rtol=0, atol=0)
        out = original_quant_matmul(x1, x2, scale, **kwargs)
        # Independent integer oracle, covering signed nibbles including -8.
        torch.testing.assert_close(
            out.cpu().long(), x1.cpu().long() @ x2.cpu().long(), rtol=0, atol=0
        )
        calls.append(True)
        return out

    monkeypatch.delattr(torch_npu, "npu_rotate_quant", raising=False)
    monkeypatch.setattr(torch_npu, "npu_quant_matmul", counted_quant_matmul)
    wscales = torch.rand(128, device=ascend_device, dtype=torch.float32) / 7

    with ck.use_backend("ascend"):
        output = ck.convrot_w4a4_linear(
            x,
            qweight,
            wscales,
            convrot_groupsize=64,
        )

    assert output.shape == (7, 128)
    assert calls == [True]
    torch.testing.assert_close(qweight, original_weight, rtol=0, atol=0)


@requires_ascend_w4a4
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("input_features", [6144, 16384])
def test_convrot_w4a4_linear_real_widths(ascend_device, monkeypatch, dtype, input_features):
    torch.manual_seed(321)
    x = torch.randn(2, input_features, device=ascend_device, dtype=dtype) * 0.1
    weight = torch.randn(128, input_features, device=ascend_device, dtype=dtype) * 0.1

    with ck.use_backend("eager"):
        qweight, wscales = ck.quantize_convrot_w4a4_weight(weight)
        expected = ck.convrot_w4a4_linear(x, qweight, wscales)

    def unexpected_fused_call(*args, **kwargs):
        raise AssertionError("fused rotate-quant must not be used above its K limit")

    monkeypatch.setattr(torch_npu, "npu_rotate_quant", unexpected_fused_call, raising=False)
    with ck.use_backend("ascend"):
        actual = ck.convrot_w4a4_linear(x, qweight, wscales)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@requires_ascend_w4a4
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_convrot_w4a4_linear_preserves_large_finite_output(ascend_device, dtype):
    x = torch.full((16, 256), 1e6, device=ascend_device, dtype=dtype)
    weight = _pack_int4_row_major(torch.ones((16, 256), device=ascend_device, dtype=torch.int8))
    scale = torch.ones(16, device=ascend_device)
    with ck.use_backend("eager"):
        expected = ck.convrot_w4a4_linear(x, weight, scale)
    with ck.use_backend("ascend"):
        actual = ck.convrot_w4a4_linear(x, weight, scale)
    assert actual.isfinite().all()
    assert actual.abs().max() > torch.finfo(torch.float16).max
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@requires_ascend_w4a4
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("storage_dtype", [torch.int8, torch.uint8])
def test_convrot_w4a4_linear_strided_batched_inputs(ascend_device, dtype, storage_dtype):
    torch.manual_seed(42)
    x = torch.randn(2, 5, 512, device=ascend_device, dtype=dtype)[..., ::2]
    weight = torch.randint(-128, 128, (32, 256), device=ascend_device, dtype=torch.int8)[:, ::2].to(
        storage_dtype
    )
    scale = torch.rand(64, device=ascend_device)[::2]
    bias = torch.randn(64, device=ascend_device, dtype=dtype)[::2]
    with ck.use_backend("eager"):
        expected = ck.convrot_w4a4_linear(x, weight, scale, bias)
    with ck.use_backend("ascend"):
        actual = ck.convrot_w4a4_linear(x, weight, scale, bias)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_convrot_w4a4_registration_does_not_require_rotate_quant(monkeypatch):
    monkeypatch.setattr(ascend_backend, "_ASCEND_ROTATE_QUANT_AVAILABLE", False)
    monkeypatch.setattr(ascend_backend, "_ASCEND_QUANT_MATMUL_AVAILABLE", True)
    assert "convrot_w4a4_linear" in ascend_backend._build_constraints()
    monkeypatch.setattr(ascend_backend, "_ASCEND_QUANT_MATMUL_AVAILABLE", False)
    assert "convrot_w4a4_linear" not in ascend_backend._build_constraints()


@requires_ascend_w4a4
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_convrot_w4a4_linear_tiny_and_nonfinite_rows(ascend_device, dtype):
    x = torch.zeros(4, 256, device=ascend_device, dtype=dtype)
    x[1].fill_(1e-6)
    x[2, 0] = float("inf")
    x[3, 0] = float("nan")
    weight = _pack_int4_row_major(torch.ones((16, 256), device=ascend_device, dtype=torch.int8))
    scale = torch.ones(16, device=ascend_device)
    with ck.use_backend("eager"):
        expected = ck.convrot_w4a4_linear(x, weight, scale)
    with ck.use_backend("ascend"):
        actual = ck.convrot_w4a4_linear(x, weight, scale)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    assert actual[0].isfinite().all()


def test_convrot_w4a4_declines_int32_accumulator_overflow():
    width = (ascend_backend._INT4_MAX_ACCUMULATION_FEATURES // 256 + 1) * 256
    result = ascend_backend._validate_convrot_w4a4_linear(
        {
            "x": torch.empty((1, width), device="meta"),
            "qweight": torch.empty((8, width // 2), device="meta", dtype=torch.int8),
        }
    )
    assert not result.success
    assert result.failed_param == "x"


@requires_ascend_w4a4
def test_convrot_w4a4_backend_selection_and_declines(ascend_device):
    x = torch.randn(4, 256, device=ascend_device, dtype=torch.bfloat16)
    qweight = torch.randint(-128, 128, (128, 128), device=ascend_device, dtype=torch.int8)
    wscales = torch.rand(128, device=ascend_device, dtype=torch.float32)
    call = {
        "x": x,
        "qweight": qweight,
        "wscales": wscales,
        "bias": None,
        "convrot_groupsize": 64,
        "quant_group_size": 64,
        "linear_dtype": "int4",
    }
    assert registry.get_capable_backend("convrot_w4a4_linear", call) == "ascend"

    assert (
        registry.get_capable_backend("convrot_w4a4_linear", dict(call, linear_dtype="int8"))
        == "eager"
    )
    assert (
        registry.get_capable_backend("convrot_w4a4_linear", dict(call, quant_group_size=32))
        == "eager"
    )
    assert (
        registry.get_capable_backend("convrot_w4a4_linear", dict(call, qweight=qweight[:127]))
        == "eager"
    )


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
def test_convrot_int8_linear_uses_separate_path_above_fused_limit(ascend_device, monkeypatch):
    def unexpected_rotate_quant(*args, **kwargs):
        raise AssertionError("feature widths above the fused limit must use the separate path")

    if hasattr(torch_npu, "npu_rotate_quant"):
        monkeypatch.setattr(torch_npu, "npu_rotate_quant", unexpected_rotate_quant)
    input_features = 16384
    x = torch.randn(2, input_features, device=ascend_device, dtype=torch.bfloat16)
    weight = torch.randint(
        -127,
        128,
        (64, input_features),
        device=ascend_device,
        dtype=torch.int8,
    )
    weight_scale = torch.rand(64, device=ascend_device, dtype=torch.float32) / 127

    with ck.use_backend("ascend"):
        output = ck.int8_linear(
            x,
            weight,
            weight_scale,
            convrot=True,
            convrot_groupsize=256,
        )

    assert output.shape == (2, 64)


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
