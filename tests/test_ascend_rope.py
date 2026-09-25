from types import SimpleNamespace

import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.backends import ascend as ascend_backend
from comfy_kitchen.backends.eager import rope as eager_rope
from comfy_kitchen.registry import registry

torch_npu = pytest.importorskip("torch_npu")

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="Huawei Ascend device required"
)


@pytest.fixture
def ascend_device():
    torch.npu.set_device("npu:0")
    return torch.device("npu:0")


@pytest.mark.skipif(
    not ascend_backend._ASCEND_QUANT_AVAILABLE,
    reason="compatible Ascend quantization operators required",
)
def test_required_quantization_parameter_is_available():
    assert ascend_backend._operator_has_parameter(torch_npu.npu_quantize, "div_mode")


@pytest.mark.skipif(
    not ascend_backend._ASCEND_ROPE_AVAILABLE,
    reason="compatible Ascend rotary operator required",
)
def test_required_rotary_parameter_is_available():
    assert ascend_backend._operator_has_parameter(torch_npu.npu_rotary_mul, "rotary_mode")


def test_operator_parameter_detection_handles_older_schemas():
    arguments = [SimpleNamespace(name="input"), SimpleNamespace(name="r1")]
    operator = SimpleNamespace(
        default=SimpleNamespace(_schema=SimpleNamespace(arguments=arguments))
    )

    assert ascend_backend._operator_has_parameter(operator, "r1")
    assert not ascend_backend._operator_has_parameter(operator, "rotary_mode")
    assert not ascend_backend._operator_has_parameter(object(), "rotary_mode")


def _shapes(layout, head_dim=64, *, batch=1):
    if layout == "bsnd":
        return (
            (batch, 7, 3, head_dim),
            (1, 7, 1, head_dim // 2, 2, 2),
        )
    return (
        (batch, 3, 7, head_dim),
        (1, 1, 7, head_dim // 2, 2, 2),
    )


def _packed_views(shape, device, dtype):
    storage = torch.randn(*shape[:-1], shape[-1] * 3, device=device, dtype=dtype)
    head_dim = shape[-1]
    return (
        storage,
        storage[..., :head_dim],
        storage[..., head_dim : 2 * head_dim],
        storage[..., 2 * head_dim :],
    )


def _assert_close(actual, expected):
    torch.testing.assert_close(actual.float(), expected.float(), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("layout", ["bsnd", "bnsd"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("split_half", [False, True])
@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("inplace", [False, True])
def test_apply_rope_matches_eager_for_packed_views(
    ascend_device, layout, dtype, split_half, paired, inplace
):
    shape, freqs_shape = _shapes(layout)
    storage, q, k, untouched = _packed_views(shape, ascend_device, dtype)
    del storage
    untouched_before = untouched.clone()
    q_before = q.clone()
    k_before = k.clone()
    freqs = torch.randn(freqs_shape, device=ascend_device, dtype=torch.float32)

    stem = "apply_rope_split_half" if split_half else "apply_rope"
    op_name = stem if paired else f"{stem}1"
    if inplace:
        op_name += "_"

    reference_op = getattr(eager_rope, stem if paired else f"{stem}1")
    expected = reference_op(q_before, k_before, freqs) if paired else reference_op(q_before, freqs)

    q_ptr = q.data_ptr()
    k_ptr = k.data_ptr()
    with ck.use_backend("ascend"):
        actual = getattr(ck, op_name)(q, k, freqs) if paired else getattr(ck, op_name)(q, freqs)

    if paired:
        for result, expected_result in zip(actual, expected, strict=True):
            _assert_close(result, expected_result)
        if inplace:
            assert actual[0].data_ptr() == q_ptr
            assert actual[1].data_ptr() == k_ptr
    else:
        _assert_close(actual, expected)
        if inplace:
            assert actual.data_ptr() == q_ptr
    if not inplace:
        torch.testing.assert_close(q, q_before)
        torch.testing.assert_close(k, k_before)
    elif not paired:
        torch.testing.assert_close(k, k_before)
    torch.testing.assert_close(untouched, untouched_before)


@pytest.mark.parametrize("layout", ["bsnd", "bnsd"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("split_half", [False, True])
@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("inplace", [False, True])
def test_rms_rope_matches_eager_for_packed_views(
    ascend_device, layout, dtype, split_half, paired, inplace
):
    shape, freqs_shape = _shapes(layout)
    storage, q, k, untouched = _packed_views(shape, ascend_device, dtype)
    del storage
    untouched_before = untouched.clone()
    q_before = q.clone()
    k_before = k.clone()
    freqs = torch.randn(freqs_shape, device=ascend_device, dtype=torch.float32)
    q_scale = torch.randn(shape[-1], device=ascend_device, dtype=torch.float32)
    k_scale = torch.randn(shape[-1], device=ascend_device, dtype=torch.float32)

    stem = "rms_rope_split_half" if split_half else "rms_rope"
    op_name = stem if paired else f"{stem}1"
    if inplace:
        op_name += "_"

    reference_op = getattr(eager_rope, stem if paired else f"{stem}1")
    expected = (
        reference_op(q_before, k_before, freqs, q_scale, k_scale)
        if paired
        else reference_op(q_before, freqs, q_scale)
    )

    q_ptr = q.data_ptr()
    k_ptr = k.data_ptr()
    with ck.use_backend("ascend"):
        actual = (
            getattr(ck, op_name)(q, k, freqs, q_scale, k_scale)
            if paired
            else getattr(ck, op_name)(q, freqs, q_scale)
        )

    if paired:
        for result, expected_result in zip(actual, expected, strict=True):
            _assert_close(result, expected_result)
        if inplace:
            assert actual[0].data_ptr() == q_ptr
            assert actual[1].data_ptr() == k_ptr
    else:
        _assert_close(actual, expected)
        if inplace:
            assert actual.data_ptr() == q_ptr
    if not inplace:
        torch.testing.assert_close(q, q_before)
        torch.testing.assert_close(k, k_before)
    elif not paired:
        torch.testing.assert_close(k, k_before)
    torch.testing.assert_close(untouched, untouched_before)


@pytest.mark.parametrize("inplace", [False, True])
def test_rms_rope_split_half_partial_rotary_matches_eager(ascend_device, inplace):
    shape = (1, 11, 5, 128)
    rot_dim = 64
    freqs_shape = (1, 11, 1, rot_dim // 2, 2, 2)
    q = torch.randn(shape, device=ascend_device, dtype=torch.bfloat16)
    k = torch.randn(shape, device=ascend_device, dtype=torch.bfloat16)
    freqs = torch.randn(freqs_shape, device=ascend_device, dtype=torch.float32)
    q_scale = torch.randn(shape[-1], device=ascend_device, dtype=torch.float32)
    k_scale = torch.randn(shape[-1], device=ascend_device, dtype=torch.float32)
    expected = eager_rope.rms_rope_split_half(q, k, freqs, q_scale, k_scale, rot_dim=rot_dim)

    op_name = "rms_rope_split_half_" if inplace else "rms_rope_split_half"
    with ck.use_backend("ascend"):
        actual = getattr(ck, op_name)(q, k, freqs, q_scale, k_scale, rot_dim=rot_dim)

    for result, expected_result in zip(actual, expected, strict=True):
        _assert_close(result, expected_result)


def test_apply_rope_supports_gqa_head_counts(ascend_device):
    q = torch.randn(1, 13, 16, 64, device=ascend_device, dtype=torch.bfloat16)
    k = torch.randn(1, 13, 4, 64, device=ascend_device, dtype=torch.bfloat16)
    freqs = torch.randn(1, 13, 1, 32, 2, 2, device=ascend_device, dtype=torch.float32)

    with ck.use_backend("ascend"):
        actual = ck.apply_rope(q, k, freqs)
    expected = eager_rope.apply_rope(q, k, freqs)
    for result, expected_result in zip(actual, expected, strict=True):
        _assert_close(result, expected_result)


@pytest.mark.parametrize(
    "input_dtype,freqs_dtype",
    [
        (torch.float16, torch.float16),
        (torch.float16, torch.bfloat16),
        (torch.bfloat16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
    ],
)
@pytest.mark.parametrize("split_half", [False, True])
def test_apply_rope_preserves_eager_frequency_dtype_semantics(
    ascend_device, input_dtype, freqs_dtype, split_half
):
    x = torch.randn(1, 7, 3, 64, device=ascend_device, dtype=input_dtype)
    freqs = torch.randn(1, 7, 1, 32, 2, 2, device=ascend_device, dtype=freqs_dtype)
    op_name = "apply_rope_split_half1" if split_half else "apply_rope1"

    with ck.use_backend("ascend"):
        actual = getattr(ck, op_name)(x, freqs)
    expected = getattr(eager_rope, op_name)(x, freqs)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=5e-2, atol=4e-2)


def test_ascend_inplace_rope_rejects_overlapping_inputs(ascend_device):
    q = torch.randn(1, 7, 3, 64, device=ascend_device, dtype=torch.bfloat16)
    k = q[..., :]
    freqs = torch.randn(1, 7, 1, 32, 2, 2, device=ascend_device, dtype=torch.float32)

    with ck.use_backend("ascend"), pytest.raises(ValueError, match="non-overlapping"):
        ck.apply_rope_(q, k, freqs)


@pytest.mark.parametrize(
    "case,failed_param",
    [
        ("interleaved_batch_freqs", "freqs_cis"),
        ("strided_last_dim", "x"),
        ("oversized_rotary_dim", "x"),
        ("bad_scale", "scale"),
    ],
)
def test_ascend_rope_declines_unsupported_contracts(ascend_device, case, failed_param):
    x = torch.randn(1, 7, 3, 64, device=ascend_device, dtype=torch.bfloat16)
    freqs = torch.randn(1, 7, 1, 32, 2, 2, device=ascend_device, dtype=torch.float32)
    op_name = "apply_rope1"
    call = {"x": x, "freqs_cis": freqs}

    if case == "interleaved_batch_freqs":
        x = x.expand(2, -1, -1, -1)
        freqs = freqs.expand(2, -1, -1, -1, -1, -1)
        call = {"x": x, "freqs_cis": freqs}
    elif case == "strided_last_dim":
        wide = torch.randn(1, 7, 3, 128, device=ascend_device, dtype=torch.bfloat16)
        call["x"] = wide[..., ::2]
    elif case == "oversized_rotary_dim":
        call["x"] = torch.randn(1, 2, 1, 896, device=ascend_device, dtype=torch.bfloat16)
        call["freqs_cis"] = torch.randn(
            1,
            2,
            1,
            448,
            2,
            2,
            device=ascend_device,
            dtype=torch.float32,
        )
    else:
        op_name = "rms_rope1"
        call["scale"] = torch.ones(32, device=ascend_device, dtype=torch.float32)

    result = registry.validate_backend_for_call("ascend", op_name, call)
    assert not result.success
    assert result.failed_param == failed_param


def test_ascend_rope_is_selected_automatically(ascend_device):
    x = torch.randn(1, 7, 3, 64, device=ascend_device, dtype=torch.bfloat16)
    freqs = torch.randn(1, 7, 1, 32, 2, 2, device=ascend_device, dtype=torch.float32)
    assert registry.get_capable_backend("apply_rope1", {"x": x, "freqs_cis": freqs}) == "ascend"
