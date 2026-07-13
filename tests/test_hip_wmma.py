"""Correctness tests for the HIP backend.

Skipped unless the backend registered, which requires an RDNA2/3/4 device and the
compiled extension. The GEMM tests need matrix cores on top of that, so they are
skipped on RDNA2, which has none; see needs_wmma.
"""
import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.registry import registry

pytestmark = pytest.mark.skipif(
    not registry.is_available("hip"), reason="HIP backend not available"
)


def _has_wmma() -> bool:
    # Only the backend's absence should read as "no WMMA"; a registered backend
    # that raises here is a real failure and must surface rather than turn every
    # needs_wmma test into a green skip.
    if not registry.is_available("hip"):
        return False
    from comfy_kitchen.backends import hip as hip_backend

    return hip_backend.has_wmma()


HAS_WMMA = _has_wmma()

# The GEMMs compile on RDNA2 but trap: it has no matrix cores, and the backend
# does not advertise them there.
needs_wmma = pytest.mark.skipif(
    not HAS_WMMA, reason="GEMM kernels need matrix cores (RDNA3/RDNA4)"
)

DEV = "cuda"


@pytest.fixture
def hip():
    from comfy_kitchen.backends import hip as hip_backend

    return hip_backend


# Covers each tile path: GEMV (M <= 8), 64x64, 128x128 and 256x128 (K > N), plus
# sizes that are not multiples of the macro tile.
GEMM_SHAPES = [
    (1, 256, 256),
    (8, 512, 256),
    (17, 256, 512),
    (128, 512, 256),
    (333, 1152, 1152),
    (512, 256, 1024),
    (1024, 2048, 512),
]


@pytest.mark.parametrize(("m", "n", "k"), GEMM_SHAPES)
@pytest.mark.parametrize("bias", [False, True])
@needs_wmma
def test_scaled_mm_fp8_matches_reference(hip, m, n, k, bias):
    torch.manual_seed(0)
    a = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)
    b = torch.randn(n, k, device=DEV, dtype=torch.bfloat16)
    sa = (a.abs().max() / 448).float()
    sb = (b.abs().max() / 448).float()
    aq = (a / sa).to(torch.float8_e4m3fn)
    bq = (b / sb).to(torch.float8_e4m3fn)
    bias_t = torch.randn(n, device=DEV, dtype=torch.bfloat16) if bias else None

    out = hip.scaled_mm_fp8(aq, bq.t(), sa, sb, bias_t, torch.bfloat16)

    ref = (aq.float() * sa) @ (bq.float() * sb).t()
    if bias:
        ref += bias_t.float()

    assert out.shape == (m, n)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(("m", "n", "k"), GEMM_SHAPES)
@pytest.mark.parametrize("per_channel", [False, True])
@needs_wmma
def test_int8_linear_matches_eager(m, n, k, per_channel):
    torch.manual_seed(0)
    x = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)
    w = torch.randn(n, k, device=DEV, dtype=torch.bfloat16)

    wq, ws = ck.quantize_int8_rowwise(w)
    ws = ws.reshape(-1) if per_channel else ws.reshape(-1)[:1]
    bias = torch.randn(n, device=DEV, dtype=torch.bfloat16)

    with ck.use_backend("hip"):
        out = ck.int8_linear(x, wq, ws, bias, torch.bfloat16)
    with ck.use_backend("eager"):
        ref = ck.int8_linear(x, wq, ws, bias, torch.bfloat16)

    assert out.shape == (m, n)
    # Both paths quantize x per row and differ only in the rounding of the
    # activation quantizer, so compare with an int8-sized tolerance.
    scale = ref.float().abs().max().item()
    assert (out.float() - ref.float()).abs().max().item() < 0.05 * scale


@needs_wmma
def test_int8_linear_convrot_matches_eager():
    torch.manual_seed(0)
    m, n, k = 256, 512, 512
    x = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)
    w = torch.randn(n, k, device=DEV, dtype=torch.bfloat16)
    wq, ws = ck.quantize_int8_rowwise(w)

    with ck.use_backend("hip"):
        out = ck.int8_linear(x, wq, ws.reshape(-1), None, torch.bfloat16, convrot=True,
                             convrot_groupsize=256)
    with ck.use_backend("eager"):
        ref = ck.int8_linear(x, wq, ws.reshape(-1), None, torch.bfloat16, convrot=True,
                             convrot_groupsize=256)

    scale = ref.float().abs().max().item()
    assert (out.float() - ref.float()).abs().max().item() < 0.05 * scale


@pytest.mark.parametrize("group_size", [16, 64, 256])
def test_convrot_w4a4_weight_quant_close_to_eager(hip, group_size):
    """The fused rotation runs in fp32; eager rotates in the weight dtype.

    Codes may therefore differ by one level, but no more.
    """
    torch.manual_seed(0)
    from comfy_kitchen.backends.eager.convrot_w4a4 import _unpack_int4_row_major

    w = torch.randn(256, 1024, device=DEV, dtype=torch.bfloat16)

    qw_h, ws_h = hip.quantize_convrot_w4a4_weight(w, group_size)
    with ck.use_backend("eager"):
        qw_e, ws_e = ck.quantize_convrot_w4a4_weight(w, group_size)

    assert qw_h.shape == qw_e.shape
    codes_h = _unpack_int4_row_major(qw_h).int()
    codes_e = _unpack_int4_row_major(qw_e).int()

    delta = (codes_h - codes_e).abs()
    assert delta.max().item() <= 1
    assert (delta > 0).float().mean().item() < 0.02

    torch.testing.assert_close(ws_h, ws_e, rtol=1e-2, atol=0)


@pytest.mark.parametrize(("m", "n", "k"), [(128, 512, 512), (512, 256, 1024), (333, 512, 512)])
@needs_wmma
def test_convrot_w4a4_linear_matches_eager(hip, m, n, k):
    torch.manual_seed(0)
    x = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)
    w = torch.randn(n, k, device=DEV, dtype=torch.bfloat16)

    qw, ws = hip.quantize_convrot_w4a4_weight(w, 256)
    out = hip.convrot_w4a4_linear(x, qw, ws, None, 256)

    with ck.use_backend("eager"):
        qw_e, ws_e = ck.quantize_convrot_w4a4_weight(w, 256)
        ref = ck.convrot_w4a4_linear(x, qw_e, ws_e, None, 256)

    assert out.shape == (m, n)
    # W4A4 has 15 levels per operand, so agreement is coarse: this asserts the
    # kernel computes the same function, not that it rounds identically.
    scale = ref.float().abs().max().item()
    assert (out.float() - ref.float()).abs().max().item() < 0.25 * scale


def test_convrot_w4a4_roundtrip(hip):
    torch.manual_seed(0)
    w = torch.randn(128, 512, device=DEV, dtype=torch.bfloat16)

    qw, ws = hip.quantize_convrot_w4a4_weight(w, 256)
    deq = hip.dequantize_convrot_w4a4_weight(qw, ws, 256, output_dtype=torch.float32)

    assert deq.shape == w.shape
    rel = (deq - w.float()).norm() / w.float().norm()
    assert rel < 0.2  # int4 weight fidelity


def test_quantize_int8_rowwise_matches_eager():
    torch.manual_seed(0)
    x = torch.randn(64, 512, device=DEV, dtype=torch.bfloat16)

    with ck.use_backend("hip"):
        q, s = ck.quantize_int8_rowwise(x)
    with ck.use_backend("eager"):
        qe, se = ck.quantize_int8_rowwise(x)

    torch.testing.assert_close(s.float(), se.float(), rtol=1e-2, atol=0)
    assert (q.int() - qe.int()).abs().max().item() <= 1


def test_quantize_int8_tensorwise_matches_eager():
    torch.manual_seed(0)
    x = torch.randn(64, 512, device=DEV, dtype=torch.bfloat16)

    with ck.use_backend("hip"):
        q, s = ck.quantize_int8_tensorwise(x)
    with ck.use_backend("eager"):
        qe, se = ck.quantize_int8_tensorwise(x)

    torch.testing.assert_close(s.float().reshape(()), se.float().reshape(()), rtol=1e-2, atol=0)
    assert (q.int() - qe.int()).abs().max().item() <= 1


def test_fp8_quantize_dequantize_roundtrip():
    torch.manual_seed(0)
    x = torch.randn(128, 256, device=DEV, dtype=torch.bfloat16)
    scale = (x.abs().max() / 448).float()

    with ck.use_backend("hip"):
        q = ck.quantize_per_tensor_fp8(x, scale)
        deq = ck.dequantize_per_tensor_fp8(q, scale, torch.bfloat16)

    assert q.dtype == torch.float8_e4m3fn
    rel = (deq.float() - x.float()).norm() / x.float().norm()
    assert rel < 0.1


@pytest.mark.parametrize(
    ("shape", "mshape"),
    [((2, 4096, 3072), (2, 1, 3072)), ((4, 256, 1152), (4, 1, 1152)), ((8, 128), (8, 128))],
)
def test_adaln_matches_eager(shape, mshape):
    torch.manual_seed(0)
    x = torch.randn(*shape, device=DEV, dtype=torch.bfloat16)
    scale = torch.randn(*mshape, device=DEV, dtype=torch.bfloat16)
    shift = torch.randn(*mshape, device=DEV, dtype=torch.bfloat16)

    with ck.use_backend("hip"):
        out = ck.adaln(x, scale, shift)
    with ck.use_backend("eager"):
        ref = ck.adaln(x, scale, shift)

    # Both results are bf16 and round apart by roughly a ULP regardless of
    # correctness, so comparing them to each other measures only that rounding.
    # Score both against an fp32 reference and require the kernel to be no worse
    # than eager; it reduces in fp32 throughout.
    xf = x.float()
    truth = torch.nn.functional.layer_norm(xf, xf.shape[-1:], eps=1e-6)
    truth = truth * (1 + scale.float()) + shift.float()

    err_hip = (out.float() - truth).norm() / truth.norm()
    err_eager = (ref.float() - truth).norm() / truth.norm()

    assert err_hip < 1e-2
    assert err_hip <= err_eager * 1.1


@pytest.mark.parametrize("split_half", [False, True])
@pytest.mark.parametrize("freqs_dtype", [torch.float32, torch.bfloat16])
def test_rope_matches_eager(split_half, freqs_dtype):
    torch.manual_seed(0)
    batch, heads, seq, dim = 2, 8, 128, 64
    xq = torch.randn(batch, heads, seq, dim, device=DEV, dtype=torch.bfloat16)
    xk = torch.randn(batch, heads, seq, dim, device=DEV, dtype=torch.bfloat16)
    freqs = torch.randn(1, 1, seq, dim // 2, 2, 2, device=DEV, dtype=freqs_dtype)

    pair = ck.apply_rope_split_half if split_half else ck.apply_rope
    with ck.use_backend("hip"):
        q, k = pair(xq, xk, freqs)
    with ck.use_backend("eager"):
        qr, kr = pair(xq, xk, freqs)

    torch.testing.assert_close(q.float(), qr.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(k.float(), kr.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(("m", "n", "k"), [(1, 512, 512), (8, 1024, 1024), (64, 1152, 1152)])
def test_gemv_awq_w4a16_matches_eager(m, n, k):
    torch.manual_seed(0)
    g = 64
    x = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)
    qw = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=DEV).view(torch.int8)
    ws = torch.randn(k // g, n, device=DEV, dtype=torch.bfloat16).abs() * 0.01
    wz = torch.randn(k // g, n, device=DEV, dtype=torch.bfloat16) * 0.01
    bias = torch.randn(n, device=DEV, dtype=torch.bfloat16)

    with ck.use_backend("hip"):
        out = ck.gemv_awq_w4a16(x, qw, ws, wz, bias, g)
    with ck.use_backend("eager"):
        ref = ck.gemv_awq_w4a16(x, qw, ws, wz, bias, g)

    assert out.shape == (m, n)
    # Identical dequant math on both sides; only the accumulation order differs.
    rel = (out.float() - ref.float()).norm() / ref.float().norm()
    assert rel < 1e-2


@pytest.mark.parametrize("act_unsigned", [False, True])
@pytest.mark.parametrize(("m", "n", "k", "r"), [(256, 1024, 1024, 32), (333, 512, 512, 16)])
@needs_wmma
def test_svdquant_w4a4_beats_eager_against_fp32_truth(m, n, k, r, act_unsigned):
    """HIP quantizes in fp32 where eager quantizes in bf16, so the two do not
    match bitwise. Score both against an fp32 simulation of the same W4A4 math
    and require HIP to be no worse.
    """
    torch.manual_seed(0)
    from comfy_kitchen.backends.eager.svdquant import (
        _unpack_int4_row_major,
        _unpack_uint4_row_major,
    )

    x = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)
    if act_unsigned:
        x = x.abs()  # the caller pre-shifts for the unsigned path
    smooth = torch.rand(k, device=DEV, dtype=torch.bfloat16) + 0.5
    lora_down = torch.randn(k, r, device=DEV, dtype=torch.bfloat16) * 0.05
    lora_up = torch.randn(n, r, device=DEV, dtype=torch.bfloat16) * 0.05
    wgt = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=DEV).view(torch.int8)
    wscales = torch.randn(k // 64, n, device=DEV, dtype=torch.bfloat16).abs() * 0.02
    bias = torch.randn(n, device=DEV, dtype=torch.bfloat16)

    qmax = 15.0 if act_unsigned else 7.0
    qmin = 0.0 if act_unsigned else -7.0

    xs = x.float() / smooth.float()
    groups = xs.view(m, k // 64, 64)
    scales = groups.abs().amax(-1).clamp(min=1e-10) / qmax
    q_true = (groups / scales.unsqueeze(-1)).round().clamp(qmin, qmax)
    act_fp = (q_true * scales.unsqueeze(-1)).view(m, k)

    w_int = _unpack_int4_row_major(wgt).float().view(n, k // 64, 64)
    w_fp = (w_int * wscales.t().float().unsqueeze(-1)).view(n, k)

    truth = act_fp @ w_fp.t()
    truth += (x.float() @ lora_down.float()) @ lora_up.float().t()
    truth += bias.float()

    with ck.use_backend("hip"):
        q, asc, la = ck.quantize_svdquant_w4a4(x, smooth, lora_down, 256, act_unsigned)
        out = ck.scaled_mm_svdquant_w4a4(q, wgt, asc, wscales, la, lora_up, bias, act_unsigned)
    with ck.use_backend("eager"):
        qe, asce, lae = ck.quantize_svdquant_w4a4(x, smooth, lora_down, 256, act_unsigned)
        ref = ck.scaled_mm_svdquant_w4a4(qe, wgt, asce, wscales, lae, lora_up, bias, act_unsigned)

    assert q.shape == qe.shape
    assert asc.shape == asce.shape

    err_hip = (out[:m].float() - truth).norm() / truth.norm()
    err_eager = (ref[:m].float() - truth).norm() / truth.norm()

    assert err_hip < 2e-2
    assert err_hip <= err_eager * 1.1

    # Activation codes must land on the fp32 grid to within one level; a larger
    # deviation indicates the group scaling or the packing is wrong.
    unpack = _unpack_uint4_row_major if act_unsigned else _unpack_int4_row_major
    codes = unpack(q[:m]).int()
    assert (codes - q_true.view(m, k).int()).abs().max() <= 1


@needs_wmma
def test_no_hipblaslt_on_the_quantized_paths(monkeypatch):
    """The quantized paths must not reach hipBLASLt or fall back to a plain matmul.

    torch._scaled_mm and torch._int_mm are the entry points that route to hipBLASLt,
    and an eager fallback would reach BLAS through torch.matmul/mm/bmm. Trap all of
    them so a silent fall-through cannot pass with an empty record.
    """
    from comfy_kitchen.tensor import QuantizedTensor

    called = []
    monkeypatch.setattr(torch, "_scaled_mm", lambda *a, **k: called.append("_scaled_mm"))
    monkeypatch.setattr(torch, "_int_mm", lambda *a, **k: called.append("_int_mm"))
    monkeypatch.setattr(torch, "matmul", lambda *a, **k: called.append("matmul"))
    monkeypatch.setattr(torch, "mm", lambda *a, **k: called.append("mm"))
    monkeypatch.setattr(torch, "bmm", lambda *a, **k: called.append("bmm"))
    if hasattr(torch.nn.functional, "scaled_mm"):
        monkeypatch.setattr(
            torch.nn.functional, "scaled_mm", lambda *a, **k: called.append("F.scaled_mm")
        )

    torch.manual_seed(0)
    x = torch.randn(512, 1024, device=DEV, dtype=torch.bfloat16)
    w = torch.randn(2048, 1024, device=DEV, dtype=torch.bfloat16)

    # fp8 linear via QuantizedTensor
    xq = QuantizedTensor.from_float(x, "TensorCoreFP8Layout")
    wq = QuantizedTensor.from_float(w, "TensorCoreFP8Layout")
    out = torch.nn.functional.linear(xq, wq)
    assert out.shape == (512, 2048)

    # int8 linear
    w8, ws = ck.quantize_int8_rowwise(w)
    ck.int8_linear(x, w8, ws.reshape(-1), None, torch.bfloat16)

    # int4 ConvRot
    from comfy_kitchen.backends import hip as hip_backend

    qw, wsc = hip_backend.quantize_convrot_w4a4_weight(w, 256)
    hip_backend.convrot_w4a4_linear(x, qw, wsc, None, 256)

    # AWQ W4A16 and SVDQuant W4A4 reach BLAS through eager's torch.matmul
    qawq = torch.randint(0, 256, (2048, 512), dtype=torch.uint8, device=DEV).view(torch.int8)
    sc = torch.ones(1024 // 64, 2048, device=DEV, dtype=torch.bfloat16)
    ck.gemv_awq_w4a16(x, qawq, sc, sc, None, 64)

    smooth = torch.ones(1024, device=DEV, dtype=torch.bfloat16)
    ld = torch.randn(1024, 32, device=DEV, dtype=torch.bfloat16) * 0.05
    lu = torch.randn(2048, 32, device=DEV, dtype=torch.bfloat16) * 0.05
    q, asc, la = ck.quantize_svdquant_w4a4(x, smooth, ld)
    ck.scaled_mm_svdquant_w4a4(q, qawq, asc, sc, la, lu, None)

    assert called == [], f"quantized path fell through to hipBLASLt: {called}"


def test_hip_registers_on_this_device():
    """The backend registered for whatever supported AMD device is running the suite."""
    from comfy_kitchen.backends import hip as hip_backend

    arch = hip_backend._gfx_arch()
    assert arch is not None and arch.startswith(hip_backend._ARCH_SUPPORTED)
    assert hip_backend.is_available()
    # has_wmma() is the intersection over every visible device, while arch names
    # only this one; derive the expectation from the same device set. The GEMM
    # tests below skip without it.
    arches = hip_backend._visible_gfx_arches()
    assert hip_backend.has_wmma() == hip_backend._has_wmma(arches)


# The kernels are compiled with -ffast-math, under which isnan() folds to false
# and the finite clamps drop a NaN operand. These pin the encoding of the values
# that exercise those paths.
FP8_EDGE_VALUES = [
    float("nan"), -float("nan"), float("inf"), -float("inf"),
    1e30, -1e30, 448.0, -448.0, 0.0, -0.0, 1.0, 1e-9,
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("out_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_fp8_quantize_edge_values_match_eager(dtype, out_dtype):
    x = torch.tensor(FP8_EDGE_VALUES, device=DEV, dtype=dtype)
    scale = torch.tensor(1.0, device=DEV)

    with ck.use_backend("hip"):
        q = ck.quantize_per_tensor_fp8(x, scale, out_dtype)
    with ck.use_backend("eager"):
        ref = ck.quantize_per_tensor_fp8(x, scale, out_dtype)

    assert torch.equal(q.view(torch.uint8), ref.view(torch.uint8))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_stochastic_rounding_fp8_edge_values_match_eager(dtype):
    x = torch.tensor(FP8_EDGE_VALUES, device=DEV, dtype=dtype)
    rng = torch.randint(0, 256, (x.numel(),), dtype=torch.uint8, device=DEV)

    with ck.use_backend("hip"):
        q = ck.stochastic_rounding_fp8(x.clone(), rng.clone(), torch.float8_e4m3fn)
    with ck.use_backend("eager"):
        ref = ck.stochastic_rounding_fp8(x.clone(), rng.clone(), torch.float8_e4m3fn)

    finite = ~torch.isnan(x.float())
    assert torch.equal(q.view(torch.uint8)[finite], ref.view(torch.uint8)[finite])
    # eager drops the sign of a NaN here while the kernel keeps it; both are NaN.
    assert torch.isnan(q.float()[~finite]).all()


# A zero-length dimension makes the grid zero-dimensional, which HIP rejects with
# hipErrorInvalidConfiguration rather than treating as a no-op.
@needs_wmma
def test_empty_inputs_do_not_launch_zero_grids(hip):
    k, n = 128, 64
    empty = torch.empty(0, k, device=DEV, dtype=torch.bfloat16)
    w = torch.randn(n, k, device=DEV, dtype=torch.bfloat16) / 8
    scale = torch.tensor(1.0, device=DEV)

    assert hip.quantize_per_tensor_fp8(empty, scale).shape == (0, k)
    assert hip.quantize_int8_rowwise(empty)[0].shape == (0, k)
    assert hip.quantize_int8_tensorwise(empty)[0].shape == (0, k)
    assert hip.scaled_mm_fp8(
        empty.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn).t(),
        scale, scale, None, torch.bfloat16,
    ).shape == (0, n)
    assert hip.int8_linear(empty, *hip.quantize_int8_rowwise(w)).shape == (0, n)

    cw, cs = hip.quantize_convrot_w4a4_weight(w, 64)
    assert hip.convrot_w4a4_linear(empty, cw, cs, None, 64).shape == (0, n)

    adaln_mod = torch.randn(1, k, device=DEV, dtype=torch.bfloat16)
    assert hip.adaln(empty, adaln_mod, adaln_mod).shape == (0, k)
    torch.cuda.synchronize()


@needs_wmma
def test_scaled_mm_v2_declines_a_bias_it_cannot_index():
    """The epilogue indexes bias[col] with one dtype code; anything else falls back."""
    from comfy_kitchen.scaled_mm_v2 import ScalingType, SwizzleType, _hip_fp8_gemm

    a = (torch.randn(64, 128, device=DEV) / 8).to(torch.float8_e4m3fn)
    b = (torch.randn(128, 128, device=DEV) / 8).to(torch.float8_e4m3fn)
    scale = torch.tensor(1.0, device=DEV)
    tw, no = ScalingType.TensorWise, SwizzleType.NO_SWIZZLE

    def routes_to_hip(bias):
        out = _hip_fp8_gemm(a, b, scale, scale, bias, torch.bfloat16, tw, tw, no, no)
        return out is not None

    assert routes_to_hip(None)
    assert routes_to_hip(torch.randn(128, device=DEV, dtype=torch.bfloat16))
    assert not routes_to_hip(torch.randn(8, device=DEV, dtype=torch.bfloat16))
    assert not routes_to_hip(torch.randn(1, 128, device=DEV, dtype=torch.bfloat16))
    assert not routes_to_hip(torch.randn(128, device=DEV, dtype=torch.float64))


def test_rope_binding_rejects_mismatched_operands(hip):
    """The kernel addresses xk and both outputs with xq's strides and dtype code, and
    writes xk_out whenever xk is present."""
    xq = torch.randn(2, 4, 64, 64, device=DEV, dtype=torch.bfloat16)
    freqs = torch.randn(1, 1, 64, 32, 2, 2, device=DEV, dtype=torch.float32)
    xq_out = torch.empty_like(xq)
    xk = torch.empty_like(xq)
    xk_out = torch.empty_like(xq)
    dl, stream = hip._dl, hip._stream(xq)

    def call(xk_a, xq_out_a, xk_out_a):
        hip._C.apply_rope(
            dl(xq), None if xk_a is None else dl(xk_a), dl(freqs),
            dl(xq_out_a), None if xk_out_a is None else dl(xk_out_a), False, stream,
        )

    with pytest.raises(RuntimeError, match="together or not at all"):
        call(xk, xq_out, None)
    with pytest.raises(RuntimeError, match="together or not at all"):
        call(None, xq_out, xk_out)
    with pytest.raises(RuntimeError, match="xq's dtype"):
        call(xk.to(torch.float16), xq_out, xk_out)
    with pytest.raises(RuntimeError, match="shape and strides"):
        call(None, torch.empty(2, 4, 64, 32, device=DEV, dtype=torch.bfloat16), None)

    call(xk, xq_out, xk_out)
    torch.cuda.synchronize()


@needs_wmma
def test_convrot_rejects_k_not_divisible_by_the_group(hip):
    """The kernel rotates K/G groups but reads back all K, so a partial group
    would quantize uninitialized LDS."""
    x = torch.randn(4, 192, device=DEV, dtype=torch.bfloat16)
    q = torch.zeros(4, 96, dtype=torch.int8, device=DEV)
    scales = torch.zeros(4, dtype=torch.float32, device=DEV)

    with pytest.raises(RuntimeError, match="divisible"):
        hip._C.convrot_quant_int4(
            hip._dl(x), hip._dl(q), hip._dl(scales), 4, 192, 256, hip._stream(x)
        )

    # The wrapper declines rather than reaching the kernel; a divisible K still runs.
    with pytest.raises(ValueError, match="divisible"):
        hip.quantize_and_rotate_rowwise(x, torch.eye(256, device=DEV, dtype=torch.bfloat16), 256)
    qw, ws = hip.quantize_convrot_w4a4_weight(
        torch.randn(64, 192, device=DEV, dtype=torch.bfloat16), 64
    )
    assert torch.isfinite(hip.convrot_w4a4_linear(x, qw, ws, None, 64)).all()


@needs_wmma
def test_bias_shape_is_validated(hip):
    """The epilogue indexes bias[col], so a mis-shaped bias reads out of bounds."""
    x = torch.randn(8, 128, device=DEV, dtype=torch.bfloat16)
    wq, ws = hip.quantize_int8_rowwise(torch.randn(64, 128, device=DEV, dtype=torch.bfloat16))

    for bad in (torch.randn(32, device=DEV), torch.randn(1, 64, device=DEV)):
        with pytest.raises(ValueError, match="bias must be 1D"):
            hip.int8_linear(x, wq, ws.reshape(-1), bad.bfloat16(), torch.bfloat16)

    # A bias on another device is moved to the launch device, not passed as-is.
    out = hip.int8_linear(x, wq, ws.reshape(-1), torch.randn(64).bfloat16(), torch.bfloat16)
    assert out.shape == (8, 64)


@needs_wmma
def test_exported_gemms_reject_unaligned_k(hip):
    """The tile loader and the small-M GEMV read a row 16 bytes at a time, so an
    unaligned K reads past the end of it. The registry constraints enforce this for
    dispatched calls; the exported helpers are reachable directly."""
    scale = torch.tensor(1.0, device=DEV)

    a = (torch.randn(4, 24, device=DEV) / 8).to(torch.float8_e4m3fn)
    b = (torch.randn(32, 24, device=DEV) / 8).to(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="divisible by 16"):
        hip.scaled_mm_fp8(a, b.t(), scale, scale, None, torch.bfloat16)

    x = torch.randn(4, 24, device=DEV, dtype=torch.bfloat16)
    wq, ws = hip.quantize_int8_rowwise(torch.randn(32, 24, device=DEV, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="divisible by 16"):
        hip.int8_linear(x, wq, ws.reshape(-1), None, torch.bfloat16)

    # int4 packs two per byte, so the packed row needs K/2 to be a whole 16-byte chunk.
    xc = torch.randn(4, 48, device=DEV, dtype=torch.bfloat16)
    qw = torch.zeros(32, 24, dtype=torch.int8, device=DEV)
    with pytest.raises(ValueError, match="divisible by 32"):
        hip.convrot_w4a4_linear(xc, qw, torch.ones(32, device=DEV), None, 16)

    # The native launcher refuses too, so a C++ caller cannot slip past.
    out = torch.empty(4, 32, dtype=torch.bfloat16, device=DEV)
    with pytest.raises(RuntimeError, match="multiple of 16"):
        hip._C.scaled_mm_fp8(
            hip._dl(a.view(torch.uint8)), hip._dl(b.contiguous().view(torch.uint8)),
            hip._dl(out), hip._dl(torch.ones(1, device=DEV)), hip._dl(torch.ones(1, device=DEV)),
            None, 4, 32, 24, 2, hip._stream(a),
        )


def test_fp8_scale_must_be_a_single_element(hip):
    """The kernels read one float off a raw pointer on the launch stream."""
    x = torch.randn(64, device=DEV, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="single per-tensor scale"):
        hip.quantize_per_tensor_fp8(x, torch.ones(2, device=DEV))

    # A scale on another device is moved to the launch device, not passed as-is.
    assert hip.quantize_per_tensor_fp8(x, torch.tensor(1.0)).shape == (64,)


@needs_wmma
def test_svdquant_rejects_partial_scale_group(hip):
    """One scale per 64-element group: a partial trailing group has no scale."""
    act = torch.zeros(64, 48, dtype=torch.int8, device=DEV)  # K = 96, not a multiple of 64
    wgt = torch.zeros(32, 48, dtype=torch.int8, device=DEV)
    ascales = torch.ones(1, 64, device=DEV, dtype=torch.bfloat16)
    wscales = torch.ones(1, 32, device=DEV, dtype=torch.bfloat16)
    lora_act = torch.zeros(64, 8, device=DEV, dtype=torch.float32)
    lora_up = torch.zeros(32, 8, device=DEV, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="not divisible by group_size"):
        hip.scaled_mm_svdquant_w4a4(act, wgt, ascales, wscales, lora_act, lora_up)


@needs_wmma
def test_scaled_mm_fp8_validates_bias(hip):
    """scaled_mm_fp8 is public, so it enforces the same bias contract as the rest."""
    a = (torch.randn(64, 128, device=DEV) / 8).to(torch.float8_e4m3fn)
    b = (torch.randn(128, 128, device=DEV) / 8).to(torch.float8_e4m3fn)
    scale = torch.tensor(1.0, device=DEV)

    for bad in (
        torch.randn(8, device=DEV, dtype=torch.bfloat16),
        torch.randn(1, 128, device=DEV, dtype=torch.bfloat16),
    ):
        with pytest.raises(ValueError, match="bias must be 1D"):
            hip.scaled_mm_fp8(a, b.t(), scale, scale, bad, torch.bfloat16)

    with pytest.raises(ValueError, match="bias dtype"):
        hip.scaled_mm_fp8(
            a, b.t(), scale, scale, torch.randn(128, device=DEV, dtype=torch.float64),
            torch.bfloat16,
        )

    # A bias on another device is moved to the launch device, not passed as-is.
    out = hip.scaled_mm_fp8(
        a, b.t(), scale, scale, torch.randn(128, dtype=torch.bfloat16), torch.bfloat16
    )
    assert out.shape == (64, 128)


def test_gemv_awq_validates_its_memory_contract(hip):
    """The inner loop decodes eight weights at a time and rescales the chunk once."""
    k, n, g = 128, 64, 64
    x = torch.randn(1, k, device=DEV, dtype=torch.bfloat16)
    qw = torch.randint(-8, 8, (n, k // 2), device=DEV, dtype=torch.int8)
    ws = torch.randn(k // g, n, device=DEV, dtype=torch.bfloat16).abs() + 0.1
    wz = torch.zeros(k // g, n, device=DEV, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="multiple of 8"):
        hip.gemv_awq_w4a16(x, qw, ws, wz, None, 12)
    with pytest.raises(ValueError, match="wscales must have shape"):
        hip.gemv_awq_w4a16(x, qw, ws.reshape(-1), wz, None, g)
    with pytest.raises(ValueError, match="bias must be 1D"):
        hip.gemv_awq_w4a16(x, qw, ws, wz, torch.randn(8, device=DEV, dtype=torch.bfloat16), g)

    assert hip.gemv_awq_w4a16(x, qw, ws, wz, None, g).shape == (1, n)


@needs_wmma
def test_svdquant_validates_its_operands(hip):
    """The kernels get M, N, K and R but no tensor bounds of their own."""
    m, k, n, r = 64, 256, 128, 16
    x = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)
    smooth = torch.randn(k, device=DEV, dtype=torch.bfloat16).abs() + 0.1
    lora_down = torch.randn(k, r, device=DEV, dtype=torch.bfloat16) / 8
    lora_up = torch.randn(n, r, device=DEV, dtype=torch.bfloat16) / 8
    wgt = torch.randint(-8, 8, (n, k // 2), device=DEV, dtype=torch.int8)
    wscales = torch.randn(k // 64, n, device=DEV, dtype=torch.bfloat16).abs() + 0.1

    with pytest.raises(ValueError, match="smooth must have shape"):
        hip.quantize_svdquant_w4a4(x, smooth[:32], lora_down)
    with pytest.raises(ValueError, match="lora_down must have shape"):
        hip.quantize_svdquant_w4a4(x, smooth, lora_down[:64])
    with pytest.raises(ValueError, match="lora_x must have shape"):
        hip.quantize_svdquant_w4a4(x, smooth, lora_down, lora_x=x[:8])

    q, ascales, lora_act = hip.quantize_svdquant_w4a4(x, smooth, lora_down)
    with pytest.raises(ValueError, match="wscales must have shape"):
        hip.scaled_mm_svdquant_w4a4(q, wgt, ascales, wscales.reshape(-1), lora_act, lora_up)
    with pytest.raises(ValueError, match="lora_up must have shape"):
        hip.scaled_mm_svdquant_w4a4(q, wgt, ascales, wscales, lora_act, lora_up[:8])

    out = hip.scaled_mm_svdquant_w4a4(q, wgt, ascales, wscales, lora_act, lora_up)
    assert torch.isfinite(out).all()


def test_stochastic_rounding_rejects_non_contiguous_rng(hip):
    """The kernel writes the result into rng, which a copy would silently discard."""
    x = torch.randn(8, 16, device=DEV, dtype=torch.bfloat16)
    rng = torch.randint(0, 256, (8, 32), dtype=torch.uint8, device=DEV)[:, ::2]

    with pytest.raises(ValueError, match="contiguous"):
        hip.stochastic_rounding_fp8(x, rng, torch.float8_e4m3fn)

    rng = torch.randint(0, 256, (8, 16), dtype=torch.uint8, device=DEV)
    out = hip.stochastic_rounding_fp8(x, rng, torch.float8_e4m3fn)
    assert out.data_ptr() == rng.data_ptr()


def test_fp8_nan_survives_dequantize(hip):
    """pack_fp8 emits the NaN encoding, so unpack_fp8 has to decode it back."""
    x = torch.tensor([float("nan"), 1.0, -448.0, 0.0], device=DEV, dtype=torch.float32)
    scale = torch.tensor(1.0, device=DEV)

    q = hip.quantize_per_tensor_fp8(x, scale)
    deq = hip.dequantize_per_tensor_fp8(q, scale, torch.float32)

    assert deq[0].isnan()
    torch.testing.assert_close(deq[1:], x[1:])


@pytest.mark.parametrize("offset", [0.0, 1e3, 1e4])
def test_adaln_variance_survives_a_large_row_offset(hip, offset):
    """E[x^2] - mean^2 cancels catastrophically once the offset dwarfs the spread."""
    torch.manual_seed(0)
    x = torch.randn(4, 2048, device=DEV, dtype=torch.float32) + offset
    zeros = torch.zeros(4, 2048, device=DEV, dtype=torch.float32)

    out = hip.adaln(x, zeros, zeros)

    xf = x.double()
    ref = (xf - xf.mean(-1, keepdim=True)) * (
        xf.var(-1, unbiased=False, keepdim=True) + 1e-6
    ).rsqrt()
    assert (out.double() - ref).abs().max().item() < 1e-2


# K % 128 == 64 leaves the last tile's second group entirely in the padding, and
# its scales do not exist.
@pytest.mark.parametrize("k", [192, 320, 576, 1088])
@needs_wmma
def test_svdquant_tail_group_stays_in_bounds(hip, k):
    torch.manual_seed(0)
    m, n, r = 64, 128, 16
    x = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)
    smooth = torch.randn(k, device=DEV, dtype=torch.bfloat16).abs() + 0.1
    lora_down = torch.randn(k, r, device=DEV, dtype=torch.bfloat16) / 8
    lora_up = torch.randn(n, r, device=DEV, dtype=torch.bfloat16) / 8
    wq = torch.randint(-8, 8, (n, k // 2), device=DEV, dtype=torch.int8)
    wscales = torch.randn(k // 64, n, device=DEV, dtype=torch.bfloat16).abs() + 0.1

    q, ascales, lora_act = hip.quantize_svdquant_w4a4(x, smooth, lora_down, pad_size=256)
    out = hip.scaled_mm_svdquant_w4a4(q, wq, ascales, wscales, lora_act, lora_up)
    torch.cuda.synchronize()

    assert out.shape[1] == n
    assert torch.isfinite(out).all()


def test_rope_splits_mismatched_xk_dtype(hip):
    """One dtype code is passed for both tensors, so a differing xk must not share it."""
    xq = torch.randn(2, 4, 64, 64, device=DEV, dtype=torch.bfloat16)
    xk = xq.to(torch.float16)
    freqs = torch.randn(1, 1, 64, 32, 2, 2, device=DEV, dtype=torch.float32)

    q, k = hip.apply_rope(xq, xk, freqs)

    assert q.dtype == torch.bfloat16
    assert k.dtype == torch.float16
    torch.testing.assert_close(q.float(), hip.apply_rope1(xq, freqs).float())
    torch.testing.assert_close(k.float(), hip.apply_rope1(xk, freqs).float())


@pytest.mark.parametrize(
    ("shape", "why"),
    [
        ((1, 1, 64, 16, 2, 2), "head_dim/2 too small"),
        ((1, 1, 64, 32, 3, 2), "rotation dim is not 2"),
        ((3, 1, 64, 32, 2, 2), "leading dim neither 1 nor the batch"),
    ],
)
def test_rope_rejects_malformed_freqs(hip, shape, why):
    """The kernel indexes the trailing dims blind, so the binding has to check them."""
    xq = torch.randn(2, 4, 64, 64, device=DEV, dtype=torch.bfloat16)
    freqs = torch.randn(*shape, device=DEV, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="freqs_cis"):
        hip.apply_rope1(xq, freqs)


def test_convrot_falls_back_to_eager_past_the_lds_bound(hip):
    """The fused rotation stages a whole row in LDS, so a large K does not fit.

    The launch does not fail cleanly past the budget, so the wrapper has to route
    those shapes to eager rather than reach the kernel.
    """
    max_k = hip._C.convrot_max_k()
    assert max_k > 0

    k = (max_k + 1024) & ~63  # past the bound, still a multiple of the group size
    x = torch.randn(2, k, device=DEV, dtype=torch.bfloat16)
    h = torch.eye(64, device=DEV, dtype=torch.bfloat16)

    q, scales = hip.quantize_and_rotate_rowwise(x, h, 64)
    torch.cuda.synchronize()
    assert q.shape == (2, k)
    assert torch.isfinite(scales).all()

    # The launcher itself still refuses, so a C++ caller cannot slip past.
    qb = torch.zeros(2, k, dtype=torch.int8, device=DEV)
    sb = torch.zeros(2, dtype=torch.float32, device=DEV)
    with pytest.raises(RuntimeError, match="LDS"):
        hip._C.quantize_int8_convrot(
            hip._dl(x), hip._dl(qb), hip._dl(sb), 2, k, 64, hip._stream(x)
        )


@pytest.mark.parametrize("group_size", [0, 32, 512])
def test_convrot_launcher_rejects_unsupported_group_size(hip, group_size):
    """The fused rotation only handles G in {16, 64, 256}; the dispatch wrappers
    fall back to eager, so guard the launcher the C++ callers see."""
    x = torch.randn(8, 128, device=DEV, dtype=torch.bfloat16)
    q = torch.zeros(8, 64, dtype=torch.int8, device=DEV)
    scales = torch.zeros(8, dtype=torch.float32, device=DEV)

    with pytest.raises(RuntimeError, match="group_size"):
        hip._C.convrot_quant_int4(
            hip._dl(x), hip._dl(q), hip._dl(scales), 8, 128, group_size, hip._stream(x)
        )
