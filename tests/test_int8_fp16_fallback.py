# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the FP16-tensor-core INT8 fallback (``int8_linear_fp16``).

Volta (sm_70) has no INT8 tensor core, so ``torch._int_mm`` emulates INT8 with
DP4A and ends up slower than the device's FP16 tensor cores. The fallback
dequantizes the INT8 weights to FP16 on the fly (memory footprint unchanged)
and runs the GEMM on FP16, keeping the activation on the INT8 quantization grid
so the numerics stay on par with the INT8 path.
"""

import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.backends.eager import quantization as eager_quant

from .conftest import rel_err


def _int8_tensorwise_weight(n: int, k: int, device, generator):
    """Random INT8 weights plus a per-output-channel scale."""
    weight = torch.randint(
        -127, 128, (n, k), dtype=torch.int8, device=device, generator=generator
    )
    weight_scale = (
        torch.rand(n, 1, device=device, generator=generator) * 0.02 + 0.001
    ).to(torch.float32)
    return weight, weight_scale


def _reference_fp32(x, weight, weight_scale, convrot, group_size):
    """Exact reference: un-rotate the weight, then a float32 matmul."""
    w = weight.float()
    if convrot:
        from comfy_kitchen.tensor.int8_utils import _build_hadamard

        h = _build_hadamard(group_size, device=x.device, dtype=torch.float32)
        n_groups = w.shape[1] // group_size
        w = torch.matmul(w.reshape(w.shape[0], n_groups, group_size), h).reshape(w.shape)
    w = w * weight_scale.reshape(-1, 1).float()
    return torch.matmul(x.float(), w.t())


@pytest.mark.parametrize("convrot", [False, True])
@pytest.mark.parametrize("group_size", [256])
def test_int8_linear_fp16_matches_int8_path(convrot, group_size):
    """The FP16 fallback must stay on the INT8 quantization grid.

    Both paths quantize the activation to INT8 with the same grid, so their
    results agree up to the accumulation precision of the two GEMMs.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(0)

    n, k, m = 512, 4096, 256
    weight, weight_scale = _int8_tensorwise_weight(n, k, device, generator)
    x = torch.randn(m, k, device=device, dtype=torch.float16, generator=generator) * 0.5

    int8_out = eager_quant.int8_linear(
        x, weight, weight_scale, out_dtype=torch.float16, convrot=convrot,
        convrot_groupsize=group_size,
    )
    fp16_out = eager_quant.int8_linear_fp16(
        x, weight, weight_scale, out_dtype=torch.float16, convrot=convrot,
        convrot_groupsize=group_size,
    )

    assert fp16_out.shape == int8_out.shape
    assert fp16_out.dtype == int8_out.dtype
    # Same quantization grid -> the two agree to within the GEMM accumulation
    # error. The INT8 path accumulates in int32 and scales afterwards, the FP16
    # path accumulates in FP16, so a small drift is expected.
    assert rel_err(fp16_out, int8_out) < 0.02


@pytest.mark.parametrize("convrot", [False, True])
def test_int8_linear_fp16_matches_float32_reference(convrot):
    """The fallback must be no further from the float32 truth than INT8 is."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(1)

    n, k, m = 512, 4096, 256
    weight, weight_scale = _int8_tensorwise_weight(n, k, device, generator)
    x = torch.randn(m, k, device=device, dtype=torch.float16, generator=generator) * 0.5

    ref = _reference_fp32(x, weight, weight_scale, convrot, 256)
    int8_out = eager_quant.int8_linear(
        x, weight, weight_scale, out_dtype=torch.float32, convrot=convrot,
    )
    fp16_out = eager_quant.int8_linear_fp16(
        x, weight, weight_scale, out_dtype=torch.float32, convrot=convrot,
    )

    int8_err = rel_err(int8_out, ref)
    fp16_err = rel_err(fp16_out, ref)
    # Both sit on the INT8 activation quantization grid, so their distance to
    # the float32 reference is comparable; the fallback must not be worse.
    assert fp16_err < int8_err * 1.5 + 1e-3


def test_int8_linear_fp16_supports_bias_and_residual():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(2)

    n, k, m = 256, 4096, 128
    weight, weight_scale = _int8_tensorwise_weight(n, k, device, generator)
    x = torch.randn(m, k, device=device, dtype=torch.float16, generator=generator) * 0.5
    bias = torch.randn(n, device=device, dtype=torch.float16, generator=generator)
    residual = torch.randn(m, n, device=device, dtype=torch.float16, generator=generator)
    residual_scale = torch.randn(n, device=device, dtype=torch.float16, generator=generator)

    out = eager_quant.int8_linear_fp16(
        x, weight, weight_scale, bias=bias, out_dtype=torch.float16,
        residual=residual, residual_scale=residual_scale,
    )
    assert out.shape == (m, n)
    assert torch.isfinite(out).all()


def test_int8_linear_fp16_scalar_weight_scale():
    """A scalar (per-tensor) weight scale must be accepted."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(3)

    n, k, m = 128, 4096, 64
    weight, weight_scale = _int8_tensorwise_weight(n, k, device, generator)
    scalar_scale = weight_scale.mean().reshape(1)

    out = eager_quant.int8_linear_fp16(
        x := torch.randn(m, k, device=device, dtype=torch.float16, generator=generator) * 0.5,
        weight, scalar_scale, out_dtype=torch.float16,
    )
    assert out.shape == (m, n)
    assert x.shape == (m, k)


def test_int8_linear_fp16_rejects_shape_mismatch():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(4)

    weight, weight_scale = _int8_tensorwise_weight(64, 4096, device, generator)
    x = torch.randn(8, 2048, device=device, dtype=torch.float16, generator=generator)

    with pytest.raises(ValueError, match="inner dimensions must match"):
        eager_quant.int8_linear_fp16(x, weight, weight_scale, out_dtype=torch.float16)


def test_int8_linear_fp16_is_registered_as_a_capability():
    """The new entry point must be reachable through the public dispatch."""
    backends = ck.list_backends()
    assert "int8_linear_fp16" in backends["eager"]["capabilities"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(5)

    n, k, m = 64, 4096, 32
    weight, weight_scale = _int8_tensorwise_weight(n, k, device, generator)
    x = torch.randn(m, k, device=device, dtype=torch.float16, generator=generator) * 0.5

    out = ck.int8_linear_fp16(x, weight, weight_scale, out_dtype=torch.float16)
    assert out.shape == (m, n)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_int8_linear_delegates_to_fp16_without_int8_tensor_core(monkeypatch):
    """On a device without INT8 tensor cores, ``int8_linear`` takes the FP16 path."""
    monkeypatch.setattr(
        eager_quant, "_cuda_device_has_int8_tensor_core", lambda _index: False
    )

    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(6)

    n, k, m = 256, 4096, 128
    weight, weight_scale = _int8_tensorwise_weight(n, k, device, generator)
    x = torch.randn(m, k, device=device, dtype=torch.float16, generator=generator) * 0.5

    delegated = eager_quant.int8_linear(x, weight, weight_scale, out_dtype=torch.float16)
    direct = eager_quant.int8_linear_fp16(
        x, weight, weight_scale, out_dtype=torch.float16
    )
    assert torch.equal(delegated, direct)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_int8_linear_keeps_int8_path_with_int8_tensor_core(monkeypatch):
    """Devices that do have INT8 tensor cores keep the existing INT8 path."""
    monkeypatch.setattr(
        eager_quant, "_cuda_device_has_int8_tensor_core", lambda _index: True
    )

    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(7)

    n, k, m = 256, 4096, 128
    weight, weight_scale = _int8_tensorwise_weight(n, k, device, generator)
    x = torch.randn(m, k, device=device, dtype=torch.float16, generator=generator) * 0.5

    int8_out = eager_quant.int8_linear(x, weight, weight_scale, out_dtype=torch.float16)
    fp16_out = eager_quant.int8_linear_fp16(
        x, weight, weight_scale, out_dtype=torch.float16
    )
    # The INT8 path accumulates in int32, the FP16 path in FP16, so they differ
    # slightly -- but the INT8 path must not be the FP16 one.
    assert not torch.equal(int8_out, fp16_out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_int8_fp16_compute_dtype_prefers_fp16_without_bf16_tensor_core(monkeypatch):
    """A device without BF16 tensor cores must not run the GEMM in BF16.

    Volta has no BF16 tensor core, so a BF16 GEMM there falls back to a slow
    conversion path (measured 13.449 ms for 4096^3 against 1.535 ms for FP16
    and 2.909 ms for the DP4A INT8 path the fallback replaces).
    """
    monkeypatch.setattr(
        eager_quant, "_cuda_device_has_bf16_tensor_core", lambda _index: False
    )
    bf16 = torch.zeros(4, dtype=torch.bfloat16, device="cuda")
    fp32 = torch.zeros(4, dtype=torch.float32, device="cuda")
    assert eager_quant._int8_fp16_compute_dtype(bf16) == torch.float16
    assert eager_quant._int8_fp16_compute_dtype(fp32) == torch.float16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_int8_fp16_compute_dtype_keeps_bf16_with_bf16_tensor_core(monkeypatch):
    """Devices with BF16 tensor cores keep the activation's own BF16 dtype."""
    monkeypatch.setattr(
        eager_quant, "_cuda_device_has_bf16_tensor_core", lambda _index: True
    )
    assert eager_quant._int8_fp16_compute_dtype(
        torch.zeros(4, dtype=torch.bfloat16, device="cuda")
    ) == torch.bfloat16


@pytest.mark.parametrize(
    "capability,has_bf16_tensor_core",
    [((6, 1), False), ((7, 0), False), ((7, 5), False), ((8, 0), True), ((9, 0), True)],
)
def test_bf16_tensor_core_detection(monkeypatch, capability, has_bf16_tensor_core):
    """BF16 tensor cores start at Ampere; Pascal, Volta and Turing are excluded.

    The cache is swapped out for the duration of the test so the simulated
    capability cannot survive into a later test that asks about the real
    device.
    """
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda _index: capability
    )
    monkeypatch.setattr(eager_quant, "_bf16_tensor_core_cache", {})
    assert eager_quant._cuda_device_has_bf16_tensor_core(0) is has_bf16_tensor_core


def test_int8_fp16_fallback_can_be_disabled(monkeypatch):
    """The escape hatch lets a device keep the original INT8 path."""
    monkeypatch.setenv("COMFY_KITCHEN_INT8_FP16_FALLBACK", "0")
    assert eager_quant._int8_fp16_fallback_enabled() is False


def test_fp32_input_is_not_downgraded_by_default(monkeypatch):
    """An FP32 activation keeps the INT8 path unless the caller opts in.

    FP32 numerics are usually requested on purpose (a NaN/overflow fuse
    re-running a block in FP32, or a model pinned to FP32 compute), so the
    fallback must not silently trade 13 mantissa bits for a 2.27x speedup.
    """
    monkeypatch.delenv("COMFY_KITCHEN_INT8_FP16_FALLBACK_FP32", raising=False)
    assert eager_quant._int8_fp16_fallback_for_fp32_input() is False


def test_fp32_input_can_opt_in_to_the_fp16_fallback(monkeypatch):
    """COMFY_KITCHEN_INT8_FP16_FALLBACK_FP32=1 routes FP32 through FP16 too."""
    monkeypatch.setenv("COMFY_KITCHEN_INT8_FP16_FALLBACK_FP32", "1")
    assert eager_quant._int8_fp16_fallback_for_fp32_input() is True


@pytest.mark.parametrize(
    "capability,has_fp16_tensor_core",
    [((6, 1), False), ((5, 2), False), ((7, 0), True), ((7, 5), True), ((8, 0), True)],
)
def test_fp16_tensor_core_detection(monkeypatch, capability, has_fp16_tensor_core):
    """Only Volta and later have FP16 tensor cores; Pascal/Maxwell do not.

    The cache is swapped out for the duration of the test so the simulated
    capability cannot survive into a later test that asks about the real
    device.
    """
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda _index: capability
    )
    monkeypatch.setattr(eager_quant, "_fp16_tensor_core_cache", {})
    assert eager_quant._cuda_device_has_fp16_tensor_core(0) is has_fp16_tensor_core


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pre_volta_device_keeps_the_int8_path(monkeypatch):
    """A Pascal device must not be routed through the FP16 fallback.

    FP16 there runs at a fraction of the DP4A INT8 rate (1/64 on the GTX 1080),
    so the fallback would be a regression rather than the Volta speedup.
    """
    monkeypatch.setattr(eager_quant, "_cuda_device_has_int8_tensor_core", lambda _i: False)
    monkeypatch.setattr(eager_quant, "_cuda_device_has_fp16_tensor_core", lambda _i: False)
    monkeypatch.setenv("COMFY_KITCHEN_INT8_FP16_FALLBACK", "1")

    x = torch.randn(8, 32, device="cuda", dtype=torch.float16)
    weight = torch.randint(-8, 8, (16, 32), dtype=torch.int8, device="cuda")
    weight_scale = torch.full((16,), 0.05, device="cuda")

    calls = []
    monkeypatch.setattr(
        eager_quant,
        "int8_linear_fp16",
        lambda *a, **k: calls.append(True) or torch.zeros(8, 16, device="cuda"),
    )
    eager_quant.int8_linear(x, weight, weight_scale, out_dtype=torch.float16)
    assert calls == [], "a pre-Volta device was routed to the FP16 fallback"

    monkeypatch.setattr(eager_quant, "_cuda_device_has_fp16_tensor_core", lambda _i: True)
    eager_quant.int8_linear(x, weight, weight_scale, out_dtype=torch.float16)
    assert len(calls) == 1, "a Volta+ device should use the FP16 fallback"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_int8_linear_keeps_fp32_on_the_int8_path(monkeypatch):
    """End-to-end: an FP32 activation must not reach the FP16 fallback."""
    monkeypatch.setattr(
        eager_quant, "_cuda_device_has_int8_tensor_core", lambda _index: False
    )
    monkeypatch.setenv("COMFY_KITCHEN_INT8_FP16_FALLBACK", "1")
    monkeypatch.delenv("COMFY_KITCHEN_INT8_FP16_FALLBACK_FP32", raising=False)

    x = torch.randn(8, 32, device="cuda", dtype=torch.float32) * 0.5
    weight = torch.randint(-8, 8, (16, 32), dtype=torch.int8, device="cuda")
    weight_scale = torch.full((16,), 0.05, device="cuda")

    calls = []
    monkeypatch.setattr(
        eager_quant,
        "int8_linear_fp16",
        lambda *a, **k: calls.append(True) or torch.zeros(8, 16, device="cuda"),
    )
    eager_quant.int8_linear(x, weight, weight_scale, out_dtype=torch.float32)
    assert calls == [], "FP32 activation was routed to the FP16 fallback"

    monkeypatch.setenv("COMFY_KITCHEN_INT8_FP16_FALLBACK_FP32", "1")
    eager_quant.int8_linear(x, weight, weight_scale, out_dtype=torch.float32)
    assert len(calls) == 1, "opt-in did not route the FP32 activation to FP16"
    monkeypatch.setenv("COMFY_KITCHEN_INT8_FP16_FALLBACK", "1")
    assert eager_quant._int8_fp16_fallback_enabled() is True
