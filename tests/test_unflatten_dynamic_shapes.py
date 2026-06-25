"""Regression tests: QuantizedTensor.__tensor_unflatten__ must honor outer_size/outer_stride.

PyTorch's subclass fakeification (torch._subclasses.meta_utils) asserts that a tensor subclass
rebuilt via __tensor_unflatten__ reports exactly the outer_size/outer_stride it was given. Under
torch.compile with dynamic shapes these are (possibly symbolic, e.g. a fused activation row dim
s31*s81) and differ from the static orig_shape captured at quantize time. Ignoring them raises:

    AssertionError: Expected return value from ...__tensor_unflatten__() to have shape
    equal to (s31, 2560), but got: torch.Size([s31*s81, 2560])

which crashed mxfp8 models under torch.compile. All tests except TestUnflattenRealCompileCUDA are
CPU-only and use eager-quantizable real layouts.
"""
import dataclasses

import pytest
import torch

from comfy_kitchen.tensor import QuantizedTensor, get_layout_class
from comfy_kitchen.tensor.base import register_layout_class

# INT8 weight-quant and FP8 both have CPU code paths -> usable as real layouts on CPU.
CPU_LAYOUTS = [
    pytest.param("TensorWiseINT8Layout", id="int8"),
    pytest.param("TensorCoreFP8Layout", id="fp8"),
]


def _require(layout_cls):
    try:
        get_layout_class(layout_cls)
    except KeyError:
        pytest.skip(f"{layout_cls} not registered in this build")


def _flatten(qt):
    names, ctx = qt.__tensor_flatten__()
    return {n: getattr(qt, n) for n in names}, ctx


class TestUnflattenOuterSizeCPU:
    """Synthetic + real-layout coverage of the outer_size/outer_stride threading (CPU-only)."""

    @dataclasses.dataclass(frozen=True)
    class _DummyParams:
        scale: torch.Tensor
        orig_dtype: torch.dtype
        orig_shape: tuple

        def _tensor_fields(self):
            return ["scale"]

    class _DummyLayout:
        pass

    def _dummy_qt(self):
        register_layout_class("Dummy", self._DummyLayout)
        return QuantizedTensor(
            torch.zeros(8, 16, dtype=torch.uint8),
            "Dummy",
            self._DummyParams(scale=torch.ones(8, 1), orig_dtype=torch.bfloat16, orig_shape=(8, 16)),
        )

    def test_honors_concrete_outer_size_and_stride(self):
        inner, ctx = _flatten(self._dummy_qt())
        outer_size, outer_stride = (5, 16), (16, 1)
        rebuilt = QuantizedTensor.__tensor_unflatten__(inner, ctx, outer_size, outer_stride)
        assert tuple(rebuilt.shape) == outer_size
        assert tuple(rebuilt.stride()) == outer_stride
        # orig_shape must follow outer_size so layout slice-back stays consistent.
        assert tuple(rebuilt._params.orig_shape) == outer_size

    def test_scalar_placeholder_keeps_orig_shape(self):
        # comfy.memory_management.interpret_gathered_like calls __tensor_unflatten__(actuals, ctx, 0, 0)
        # meaning "keep the stored shape"; a scalar 0 must NOT be treated as a size.
        inner, ctx = _flatten(self._dummy_qt())
        rebuilt = QuantizedTensor.__tensor_unflatten__(inner, ctx, 0, 0)
        assert tuple(rebuilt.shape) == (8, 16)

    @pytest.mark.parametrize("layout_cls", CPU_LAYOUTS)
    def test_eager_roundtrip_placeholder(self, layout_cls):
        _require(layout_cls)
        torch.manual_seed(0)
        w = torch.randn(48, 96, dtype=torch.float32)
        qt = QuantizedTensor.from_float(w, layout_cls)
        inner, ctx = _flatten(qt)
        rebuilt = QuantizedTensor.__tensor_unflatten__(inner, ctx, 0, 0)
        assert tuple(rebuilt.shape) == tuple(qt.shape)
        assert rebuilt._layout_cls == qt._layout_cls
        assert torch.equal(rebuilt.dequantize(), qt.dequantize())


class TestUnflattenSymbolicShapesCPU:
    """The production trigger: rebuild a REAL-layout QuantizedTensor under FakeTensorMode with a
    COMPOUND symbolic outer_size (s_b*s_s), exactly what automatic-dynamic produces. Drives the real
    torch._subclasses.meta_utils path. FAILS before the fix (rebuilt reports static shape), passes
    after. CPU-only."""

    @pytest.mark.parametrize("layout_cls", CPU_LAYOUTS)
    def test_unflatten_compound_symbolic_outer_size(self, layout_cls):
        _require(layout_cls)
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        torch.manual_seed(0)
        w = torch.randn(64, 128, dtype=torch.float32)
        qt = QuantizedTensor.from_float(w, layout_cls)
        _, ctx = qt.__tensor_flatten__()
        cols = qt.shape[1]

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)
        with fake_mode:
            # Harvest two backed dynamic symbols (like s31, s81) from a dynamic helper tensor.
            helper = torch.randn(2, 3)
            torch._dynamo.maybe_mark_dynamic(helper, 0)
            torch._dynamo.maybe_mark_dynamic(helper, 1)
            fh = fake_mode.from_tensor(helper, symbolic_context=None)
            compound = fh.shape[0] * fh.shape[1]  # e.g. s26*s49, like production s31*s81

            inner = {n: fake_mode.from_tensor(getattr(qt, n)) for n in qt.__tensor_flatten__()[0]}
            outer_size = (compound, cols)
            outer_stride = (cols, 1)

            rebuilt = QuantizedTensor.__tensor_unflatten__(inner, ctx, outer_size, outer_stride)

            # This is exactly the assertion torch._subclasses.meta_utils makes internally.
            assert tuple(rebuilt.shape) == tuple(outer_size)
            assert tuple(rebuilt.stride()) == tuple(outer_stride)
            assert tuple(rebuilt._params.orig_shape) == tuple(outer_size)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestUnflattenRealCompileCUDA:
    """End-to-end production reproduction: torch.compile over a real-layout model run across TWO
    different batch shapes forces automatic-dynamic, which fakeifies the QuantizedTensor with a
    symbolic outer_size. Pre-fix this raises the meta_utils AssertionError. Requires a GPU whose
    SM supports the layout's fast matmul (Blackwell SM12.0)."""

    @pytest.mark.parametrize("layout_cls", ["TensorWiseINT8Layout", "TensorCoreFP8Layout"])
    def test_compile_dynamic_two_shapes(self, layout_cls):
        from comfy_kitchen.tensor import get_layout_class

        lc = get_layout_class(layout_cls)
        if not lc.supports_fast_matmul():
            pytest.skip(f"{layout_cls} fast matmul unsupported on this GPU")

        torch.manual_seed(0)
        w = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
        qw = QuantizedTensor.from_float(w, layout_cls)

        class M(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.w = weight

            def forward(self, x):
                # Runtime-quantize the activation -> the subclass carries the dynamic row dim.
                xq = QuantizedTensor.from_float(x, layout_cls)
                return torch.nn.functional.linear(xq, self.w)

        cm = torch.compile(M(qw), dynamic=None)  # automatic-dynamic on 2nd shape
        with torch.no_grad():
            o1 = cm(torch.randn(8, 64, device="cuda", dtype=torch.bfloat16))
            o2 = cm(torch.randn(16, 64, device="cuda", dtype=torch.bfloat16))  # recompile -> symbolic
        assert o1.shape == (8, 128)
        assert o2.shape == (16, 128)
