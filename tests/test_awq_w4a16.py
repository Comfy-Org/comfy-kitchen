from __future__ import annotations

import pytest
import torch
import torch.nn.functional as functional

import comfy_kitchen as ck
from comfy_kitchen.backends import cuda as cuda_backend
from comfy_kitchen.backends.eager.awq import gemv_awq_w4a16 as eager_awq_w4a16
from comfy_kitchen.tensor import QuantizedTensor, TensorCoreAWQW4A16Layout

_AWQ_EAGER_ATOL = 8e-3


def _make_awq_case(*, m: int = 192, n: int = 128, k: int = 128, group_size: int = 64):
    dtype = torch.bfloat16
    device = "cuda"
    x = torch.randn(m, k, dtype=dtype, device=device) * 0.1
    qweight = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=device).view(torch.int8)
    wscales = torch.rand(k // group_size, n, dtype=dtype, device=device) * 0.05 + 0.005
    wzeros = torch.randn(k // group_size, n, dtype=dtype, device=device) * 0.01
    bias = torch.randn(n, dtype=dtype, device=device) * 0.01
    return x, qweight, wscales, wzeros, bias, group_size


@pytest.fixture
def cuda_required(cuda_available):
    if not cuda_available:
        pytest.skip("CUDA required")
    if not getattr(cuda_backend, "_EXT_AVAILABLE", False):
        pytest.skip("CUDA extension required")


def test_awq_dequant_cache_default_is_disabled_and_opt_in_reuses(cuda_required, monkeypatch, seed):
    x, qweight, wscales, wzeros, bias, group_size = _make_awq_case()

    monkeypatch.setenv("COMFY_KITCHEN_AWQ_CACHE_DEQUANT", "0")
    uncached = cuda_backend.gemv_awq_w4a16(x, qweight, wscales, wzeros, bias, group_size)
    assert getattr(qweight, "_ck_awq_dequant_cache", None) is None

    monkeypatch.delenv("COMFY_KITCHEN_AWQ_CACHE_DEQUANT", raising=False)
    default_uncached = cuda_backend.gemv_awq_w4a16(x, qweight, wscales, wzeros, bias, group_size)
    assert getattr(qweight, "_ck_awq_dequant_cache", None) is None
    assert torch.equal(default_uncached, uncached)

    monkeypatch.setenv("COMFY_KITCHEN_AWQ_CACHE_DEQUANT", "1")
    cached_first = cuda_backend.gemv_awq_w4a16(x, qweight, wscales, wzeros, bias, group_size)
    cache_obj = getattr(qweight, "_ck_awq_dequant_cache", None)
    cached_second = cuda_backend.gemv_awq_w4a16(x, qweight, wscales, wzeros, bias, group_size)

    assert cache_obj is not None
    assert getattr(qweight, "_ck_awq_dequant_cache", None) is cache_obj
    assert torch.equal(cached_first, cached_second)
    assert torch.equal(cached_first, uncached)


def test_awq_cuda_dequant_kernel_is_registered_and_matches_reference(cuda_required, seed):
    _, qweight, wscales, wzeros, _, group_size = _make_awq_case(n=32, k=128)

    assert hasattr(cuda_backend._C, "awq_dequant_w4a16")
    actual = torch.empty(
        qweight.shape[0], qweight.shape[1] * 2,
        dtype=wscales.dtype, device=qweight.device,
    )
    cuda_backend._C.awq_dequant_w4a16(
        cuda_backend._wrap_for_dlpack(qweight.view(torch.uint8)),
        cuda_backend._wrap_for_dlpack(wscales),
        cuda_backend._wrap_for_dlpack(wzeros),
        cuda_backend._wrap_for_dlpack(actual),
        group_size,
        torch.cuda.current_stream(qweight.device).cuda_stream,
    )

    n, k_half = qweight.shape
    k = k_half * 2
    q16 = qweight.view(torch.uint8).to(torch.int16)
    lo = q16 & 0xF
    hi = (q16 >> 4) & 0xF
    nibbles = torch.stack([lo, hi], dim=-1).reshape(n, k).float()
    expected = (
        (nibbles.view(n, k // group_size, group_size) - 8.0)
        * wscales.float().t().unsqueeze(-1)
        + wzeros.float().t().unsqueeze(-1)
    ).view(n, k).to(wscales.dtype)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_awq_dequant_cache_invalidates_when_scale_tensor_replaced(cuda_required, monkeypatch, seed):
    x, qweight, wscales, wzeros, bias, group_size = _make_awq_case()

    monkeypatch.setenv("COMFY_KITCHEN_AWQ_CACHE_DEQUANT", "1")
    before = cuda_backend.gemv_awq_w4a16(x, qweight, wscales, wzeros, bias, group_size)
    old_cache = getattr(qweight, "_ck_awq_dequant_cache", None)

    new_scales = wscales + 0.01
    after = cuda_backend.gemv_awq_w4a16(x, qweight, new_scales, wzeros, bias, group_size)
    eager_after = eager_awq_w4a16(x, qweight, new_scales, wzeros, bias, group_size)

    assert old_cache is not None
    assert getattr(qweight, "_ck_awq_dequant_cache", None) is not old_cache
    assert not torch.equal(before, after)
    torch.testing.assert_close(after.float(), eager_after.float(), rtol=0.0, atol=_AWQ_EAGER_ATOL)


def test_awq_dequant_cache_invalidates_when_zero_tensor_replaced(cuda_required, monkeypatch, seed):
    x, qweight, wscales, wzeros, bias, group_size = _make_awq_case()

    monkeypatch.setenv("COMFY_KITCHEN_AWQ_CACHE_DEQUANT", "1")
    before = cuda_backend.gemv_awq_w4a16(x, qweight, wscales, wzeros, bias, group_size)
    old_cache = getattr(qweight, "_ck_awq_dequant_cache", None)

    new_zeros = wzeros + 0.01
    after = cuda_backend.gemv_awq_w4a16(x, qweight, wscales, new_zeros, bias, group_size)
    eager_after = eager_awq_w4a16(x, qweight, wscales, new_zeros, bias, group_size)

    assert old_cache is not None
    assert getattr(qweight, "_ck_awq_dequant_cache", None) is not old_cache
    assert not torch.equal(before, after)
    torch.testing.assert_close(after.float(), eager_after.float(), rtol=0.0, atol=_AWQ_EAGER_ATOL)


def test_awq_small_m_reuses_existing_dequant_cache(cuda_required, monkeypatch, seed):
    x_large, qweight, wscales, wzeros, bias, group_size = _make_awq_case(m=192)
    x_small = x_large[:8].contiguous()

    monkeypatch.setenv("COMFY_KITCHEN_AWQ_CACHE_DEQUANT", "1")
    _ = cuda_backend.gemv_awq_w4a16(x_large, qweight, wscales, wzeros, bias, group_size)
    cache_obj = getattr(qweight, "_ck_awq_dequant_cache", None)

    actual = cuda_backend.gemv_awq_w4a16(x_small, qweight, wscales, wzeros, bias, group_size)
    expected = eager_awq_w4a16(x_small, qweight, wscales, wzeros, bias, group_size)

    assert cache_obj is not None
    assert getattr(qweight, "_ck_awq_dequant_cache", None) is cache_obj
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.0, atol=_AWQ_EAGER_ATOL)


def test_awq_small_m_uses_temporary_dequant_by_default(cuda_required, monkeypatch, seed):
    x, qweight, wscales, wzeros, bias, group_size = _make_awq_case(m=8)

    monkeypatch.delenv("COMFY_KITCHEN_AWQ_CACHE_DEQUANT", raising=False)
    monkeypatch.delenv("COMFY_KITCHEN_AWQ_SMALL_M_CACHE_CALLS", raising=False)
    first = cuda_backend.gemv_awq_w4a16(x, qweight, wscales, wzeros, bias, group_size)
    cache_obj = getattr(qweight, "_ck_awq_dequant_cache", None)
    second = cuda_backend.gemv_awq_w4a16(x, qweight, wscales, wzeros, bias, group_size)
    expected = eager_awq_w4a16(x, qweight, wscales, wzeros, bias, group_size)

    assert cache_obj is None
    assert getattr(qweight, "_ck_awq_dequant_cache", None) is None
    torch.testing.assert_close(first.float(), expected.float(), rtol=0.0, atol=_AWQ_EAGER_ATOL)
    torch.testing.assert_close(second.float(), expected.float(), rtol=0.0, atol=_AWQ_EAGER_ATOL)


def test_awq_small_m_cache_trigger_can_be_disabled(cuda_required, monkeypatch, seed):
    x, qweight, wscales, wzeros, bias, group_size = _make_awq_case(m=8)

    monkeypatch.delenv("COMFY_KITCHEN_AWQ_CACHE_DEQUANT", raising=False)
    monkeypatch.setenv("COMFY_KITCHEN_AWQ_SMALL_M_CACHE_CALLS", "0")
    for _ in range(4):
        cuda_backend.gemv_awq_w4a16(x, qweight, wscales, wzeros, bias, group_size)

    assert getattr(qweight, "_ck_awq_dequant_cache", None) is None


def test_awq_cache_disabled_uses_temporary_dequant_for_small_m(cuda_required, monkeypatch, seed):
    x, qweight, wscales, wzeros, bias, group_size = _make_awq_case(m=8)
    calls = {"dequant_mm": 0}
    orig_dequant_mm = cuda_backend._awq_w4a16_dequant_then_matmul_with_optional_bias

    def spy_dequant_mm(*args, **kwargs):
        calls["dequant_mm"] += 1
        return orig_dequant_mm(*args, **kwargs)

    monkeypatch.setattr(cuda_backend, "_awq_w4a16_dequant_then_matmul_with_optional_bias", spy_dequant_mm)
    monkeypatch.setenv("COMFY_KITCHEN_AWQ_CACHE_DEQUANT", "0")
    monkeypatch.delenv("COMFY_KITCHEN_AWQ_FORCE_FUSED_SMALL_M", raising=False)

    actual = cuda_backend.gemv_awq_w4a16(x, qweight, wscales, wzeros, bias, group_size)
    expected = eager_awq_w4a16(x, qweight, wscales, wzeros, bias, group_size)

    assert calls == {"dequant_mm": 1}
    assert getattr(qweight, "_ck_awq_dequant_cache", None) is None
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.0, atol=_AWQ_EAGER_ATOL)

    calls["dequant_mm"] = 0
    monkeypatch.setenv("COMFY_KITCHEN_AWQ_FORCE_FUSED_SMALL_M", "1")
    forced = cuda_backend.gemv_awq_w4a16(x, qweight, wscales, wzeros, bias, group_size)

    assert calls == {"dequant_mm": 0}
    torch.testing.assert_close(forced.float(), expected.float(), rtol=0.0, atol=_AWQ_EAGER_ATOL)


def test_awq_quantized_tensor_linear_reuses_dequant_cache(cuda_required, monkeypatch, seed):
    x, qweight, wscales, wzeros, bias, group_size = _make_awq_case(m=192)
    params = TensorCoreAWQW4A16Layout.Params(
        scale=wscales,
        zeros=wzeros,
        orig_dtype=wscales.dtype,
        orig_shape=(qweight.shape[0], qweight.shape[1] * 2),
        group_size=group_size,
    )
    weight = QuantizedTensor(qweight, "TensorCoreAWQW4A16Layout", params)

    monkeypatch.setenv("COMFY_KITCHEN_AWQ_CACHE_DEQUANT", "1")
    first = functional.linear(x, weight, bias)
    cache_obj = getattr(qweight, "_ck_awq_dequant_cache", None)
    second = functional.linear(x, weight, bias)
    expected = eager_awq_w4a16(x, qweight, wscales, wzeros, bias, group_size)

    assert cache_obj is not None
    assert getattr(qweight, "_ck_awq_dequant_cache", None) is cache_obj
    assert torch.equal(first, second)
    torch.testing.assert_close(first.float(), expected.float(), rtol=0.0, atol=_AWQ_EAGER_ATOL)


def test_awq_direct_cuda_respects_eager_backend_override(cuda_required, monkeypatch, seed):
    x, qweight, wscales, wzeros, bias, group_size = _make_awq_case(m=192)
    params = TensorCoreAWQW4A16Layout.Params(
        scale=wscales,
        zeros=wzeros,
        orig_dtype=wscales.dtype,
        orig_shape=(qweight.shape[0], qweight.shape[1] * 2),
        group_size=group_size,
    )
    weight = QuantizedTensor(qweight, "TensorCoreAWQW4A16Layout", params)

    monkeypatch.delenv("COMFY_KITCHEN_AWQ_DIRECT_CUDA", raising=False)
    monkeypatch.delenv("COMFY_KITCHEN_AWQ_CACHE_DEQUANT", raising=False)
    with ck.use_backend("eager"):
        actual = functional.linear(x, weight, bias)
    expected = eager_awq_w4a16(x, qweight, wscales, wzeros, bias, group_size)

    assert getattr(qweight, "_ck_awq_dequant_cache", None) is None
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.0, atol=0.0)
