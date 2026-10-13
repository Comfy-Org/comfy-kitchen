"""Empty contractions finish before allocating shader or decode operands."""
import pytest
import torch

from comfy_kitchen.backends.mps import _build_constraints, linear
from comfy_kitchen.constraints import validate_function_call


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(0,), (2, 0), (2, 3, 0), (0, 0), (2, 0, 0)])
def test_zero_inner_dimension_matches_linear_with_bias_and_residual(dtype, shape):
    x = torch.empty(shape, dtype=dtype)
    bias = torch.tensor([1, -2, 3], dtype=dtype)
    residual = torch.full((*shape[:-1], 3), 2, dtype=dtype)
    scale = torch.tensor(0.5, dtype=dtype)

    def no_decode(*args):
        pytest.fail("empty contraction must not decode weights")

    actual = linear.weight_only_linear(
        x, 3, 0, no_decode, bias=bias, out_dtype=dtype, group=None,
        residual=residual, residual_scale=scale,
    )
    expected = torch.nn.functional.linear(x, torch.empty(3, 0, dtype=dtype), bias)
    expected = torch.addcmul(residual, expected, scale)
    assert actual.shape == expected.shape and actual.dtype == dtype
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_int8_admits_supported_output_dtype(dtype):
    assert validate_function_call(_build_constraints()["int8_linear"], {"out_dtype": dtype}).success


@pytest.mark.parametrize("dtype", [torch.int8, torch.float64, torch.bool])
def test_int8_rejects_unsupported_output_dtype_before_dispatch(dtype):
    result = validate_function_call(_build_constraints()["int8_linear"], {"out_dtype": dtype})
    assert not result.success and result.failed_param == "out_dtype"
