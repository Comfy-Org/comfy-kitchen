# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``int8_linear`` on devices without an INT8 GEMM (``aten::_int_mm``).

PyTorch has no ``_int_mm`` kernel for MPS, so the eager ``int8_linear`` raised
``NotImplementedError`` at the first quantized linear of any INT8 checkpoint on
Apple Silicon. It reaches the kernel through two doors: the ``QuantizedTensor``
dispatch handlers (``F.linear(x, qt)``) and direct calls with a fused input
activation (ComfyUI's ``linear_input_act``). Both must fall back to a float GEMM
over the stored INT8 values on such devices — ConvRot rotating the activations
exactly as the INT8 path does, and the weight scale applied to the output.
"""

import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.backends._activations import apply_input_act
from comfy_kitchen.backends.eager import quantization
from comfy_kitchen.tensor import QuantizedTensor

from .conftest import assert_values_close

MPS_AVAILABLE = torch.backends.mps.is_available()
N, K = 256, 512

# bf16 comparisons: the fallback keeps the INT8 weight values exact and applies the scale to
# the output, while the reference multiplies bf16-rounded dequantized weights — different
# roundings whose element-wise difference is ~2**-9 * sqrt(K) * |x| * |w| (std ~0.06 here,
# while |y| ~ sqrt(K) ~ 22). atol=0.5 is ~8 sigma; the tight float32 test below pins the math.
BF16_TOL = dict(rtol=2e-2, atol=0.5)


def _quantized_weight(convrot, seed, device="cpu"):
    torch.manual_seed(seed)
    w = torch.randn(N, K, device=device, dtype=torch.bfloat16)
    with ck.registry.use_backend("eager"):
        if convrot:
            # ConvRot checkpoints (e.g. MiniMax H3) are per-channel + rotated.
            return QuantizedTensor.from_float(
                w, "TensorWiseINT8Layout", per_channel=True, convrot=True, convrot_groupsize=256
            )
        return QuantizedTensor.from_float(w, "TensorWiseINT8Layout")


def _to_device(qt, device):
    params = qt._params
    moved = {
        name: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for name, v in ((n, getattr(params, n)) for n in params.__dataclass_fields__)
    }
    return QuantizedTensor(qt._qdata.to(device), qt._layout_cls, type(params)(**moved))


def _forbid_int_mm(monkeypatch):
    monkeypatch.setattr(quantization, "_device_has_int8_mm", lambda device_type: False)

    def no_int_mm(*args, **kwargs):
        raise AssertionError("torch._int_mm must not be called on the fallback path")

    monkeypatch.setattr(torch, "_int_mm", no_int_mm)


def test_device_has_int8_mm_trusts_cuda_and_cpu():
    assert quantization._device_has_int8_mm("cpu")
    assert quantization._device_has_int8_mm("cuda")


@pytest.mark.parametrize("convrot", [False, True])
def test_dispatch_path_falls_back_without_int_mm(monkeypatch, convrot):
    """``F.linear(x, qt)`` / ``mm`` / ``addmm`` must not touch ``_int_mm``."""
    _forbid_int_mm(monkeypatch)
    qt = _quantized_weight(convrot, seed=0)
    x = torch.randn(8, K, dtype=torch.bfloat16)
    bias = torch.randn(N, dtype=torch.bfloat16)
    w = qt.dequantize()

    out = torch.nn.functional.linear(x, qt, bias)
    assert out.dtype == torch.bfloat16
    assert_values_close(
        out.float(), torch.nn.functional.linear(x, w, bias).float(), **BF16_TOL
    )
    assert_values_close(
        torch.mm(x, qt.t()).float(), torch.mm(x, w.t()).float(), **BF16_TOL
    )
    assert_values_close(
        torch.addmm(bias, x, qt.t()).float(),
        torch.addmm(bias, x, w.t()).float(),
        **BF16_TOL,
    )


@pytest.mark.parametrize("convrot", [False, True])
@pytest.mark.parametrize("input_act", [None, "swiglu"])
def test_direct_call_falls_back_without_int_mm(monkeypatch, convrot, input_act):
    """``ck.int8_linear(..., input_act=...)`` — the door ComfyUI's ``linear_input_act`` uses."""
    _forbid_int_mm(monkeypatch)
    qt = _quantized_weight(convrot, seed=1)
    in_features = 2 * K if input_act == "swiglu" else K
    x = torch.randn(8, in_features, dtype=torch.bfloat16)
    bias = torch.randn(N, dtype=torch.bfloat16)

    out = ck.int8_linear(
        x,
        qt._qdata,
        qt._params.scale,
        bias,
        torch.bfloat16,
        convrot=convrot,
        convrot_groupsize=256,
        input_act=input_act,
    )
    ref = torch.nn.functional.linear(apply_input_act(x, input_act), qt.dequantize(), bias)
    assert out.dtype == torch.bfloat16
    assert_values_close(out.float(), ref.float(), **BF16_TOL)


def test_fallback_math_is_exact_in_float32(monkeypatch):
    """In float32 the fallback equals dequantize-then-GEMM to round-off: rotating the
    activations instead of un-rotating the weights is the same product (H is orthogonal)."""
    _forbid_int_mm(monkeypatch)
    qt = _quantized_weight(convrot=True, seed=5)
    x = torch.randn(8, K, dtype=torch.float32)
    w_ref = quantization.dequantize_int8_convrot_weight(
        qt._qdata, qt._params.scale.reshape(-1, 1), 256
    )
    out = ck.int8_linear(
        x, qt._qdata, qt._params.scale, None, torch.float32, convrot=True, convrot_groupsize=256
    )
    assert_values_close(out, torch.nn.functional.linear(x, w_ref), rtol=1e-3, atol=1e-3)


def test_fallback_preserves_input_rank(monkeypatch):
    """A 1-D input must yield a 1-D output (and 3-D stays 3-D), matching the native path."""
    _forbid_int_mm(monkeypatch)
    qt = _quantized_weight(convrot=False, seed=6)
    bias = torch.randn(N, dtype=torch.bfloat16)
    out1 = ck.int8_linear(
        torch.randn(K, dtype=torch.bfloat16), qt._qdata, qt._params.scale, bias, torch.bfloat16
    )
    out3 = ck.int8_linear(
        torch.randn(2, 8, K, dtype=torch.bfloat16), qt._qdata, qt._params.scale, None, torch.bfloat16
    )
    assert out1.shape == (N,)
    assert out3.shape == (2, 8, N)


@pytest.mark.parametrize("convrot", [False, True])
def test_fallback_chunks_match_single_pass(monkeypatch, convrot):
    """A tiny chunk budget (forcing many weight slices) must give the same answer."""
    _forbid_int_mm(monkeypatch)
    qt = _quantized_weight(convrot, seed=7)
    x = torch.randn(8, K, dtype=torch.bfloat16)
    bias = torch.randn(N, dtype=torch.bfloat16)

    def run():
        return ck.int8_linear(
            x, qt._qdata, qt._params.scale, bias, torch.bfloat16,
            convrot=convrot, convrot_groupsize=256,
        )

    single = run()
    monkeypatch.setattr(quantization, "_FALLBACK_CHUNK_BYTES", K * 2 * 7)  # ~7 rows per slice
    chunked = run()
    torch.testing.assert_close(chunked, single, rtol=0, atol=0)


def test_native_path_untouched_when_int_mm_exists(monkeypatch):
    """CPU has ``_int_mm``: the fallback must not engage there."""
    calls = []
    real = torch._int_mm

    def counting_int_mm(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(torch, "_int_mm", counting_int_mm)
    qt = _quantized_weight(convrot=False, seed=2)
    torch.nn.functional.linear(torch.randn(8, K, dtype=torch.bfloat16), qt)
    assert calls, "expected the native INT8 GEMM on CPU"


@pytest.mark.skipif(not MPS_AVAILABLE, reason="MPS device required")
@pytest.mark.parametrize("convrot", [False, True])
def test_dispatch_path_runs_on_mps(convrot):
    qt_cpu = _quantized_weight(convrot, seed=3)
    x = torch.randn(8, K, dtype=torch.bfloat16)

    out = torch.nn.functional.linear(x.to("mps"), _to_device(qt_cpu, "mps"))
    ref = torch.nn.functional.linear(x, qt_cpu.dequantize())
    assert out.device.type == "mps" and out.dtype == torch.bfloat16
    assert_values_close(out.float().cpu(), ref.float(), **BF16_TOL)


@pytest.mark.skipif(not MPS_AVAILABLE, reason="MPS device required")
def test_direct_call_with_swiglu_runs_on_mps():
    """The MiniMax H3 MLP down-projection shape: ConvRot weight, fused SwiGLU input."""
    qt_cpu = _quantized_weight(convrot=True, seed=4)
    x = torch.randn(8, 2 * K, dtype=torch.bfloat16)
    qt = _to_device(qt_cpu, "mps")

    out = ck.int8_linear(
        x.to("mps"),
        qt._qdata,
        qt._params.scale,
        None,
        torch.bfloat16,
        convrot=True,
        convrot_groupsize=256,
        input_act="swiglu",
    )
    ref = torch.nn.functional.linear(apply_input_act(x, "swiglu"), qt_cpu.dequantize())
    assert out.device.type == "mps" and out.dtype == torch.bfloat16
    assert_values_close(out.float().cpu(), ref.float(), **BF16_TOL)
