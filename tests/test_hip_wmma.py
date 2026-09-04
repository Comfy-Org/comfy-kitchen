"""Correctness tests for the HIP backend.

Skipped unless the backend registered, which requires an RDNA2/3/4 device and the
compiled extension. The GEMM tests need matrix cores on top of that, so they are
skipped on RDNA2, which has none; see needs_wmma.
"""
import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.backends.eager import w4a8_int8 as eager_w4a8
from comfy_kitchen.backends.eager.convrot_w4a4 import _unpack_int4_row_major
from comfy_kitchen.backends.eager.quantization import (
    quantize_and_rotate_rowwise as eager_quantize_and_rotate_rowwise,
)
from comfy_kitchen.backends.eager.quantization import (
    rotate_int8_convrot_weight as eager_rotate_int8_convrot_weight,
)
from comfy_kitchen.constraints import validate_function_call
from comfy_kitchen.registry import registry
from comfy_kitchen.tensor import AsymW4A8Int8Layout, QuantizedTensor
from comfy_kitchen.tensor.int8_utils import _build_hadamard


def _unavailable_reason() -> str | None:
    """None when the backend is usable, else the registry's reason for withdrawing it.

    Reporting it keeps a broken build from reading as unsupported hardware.
    """
    if registry.is_available("hip"):
        return None
    reason = registry.list_backends().get("hip", {}).get("unavailable_reason")
    return reason or "HIP backend not available"


_UNAVAILABLE = _unavailable_reason()

pytestmark = pytest.mark.skipif(_UNAVAILABLE is not None, reason=_UNAVAILABLE or "")


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
# does not advertise them there. An absent backend reports why instead.
needs_wmma = pytest.mark.skipif(
    not HAS_WMMA, reason=_UNAVAILABLE or "GEMM kernels need matrix cores (RDNA3/RDNA4)"
)

DEV = "cuda"


@pytest.fixture
def hip():
    from comfy_kitchen.backends import hip as hip_backend

    return hip_backend


# Covers each tile path on 16-48 WGP parts: GEMV (M <= 8), skinny (M or N <= 64),
# and 64x64/128x128 at both K depths including both deep-K warp grids. K=2064/4112
# are multiples of 16 but not of BKB=128, to hit its K tail.
GEMM_SHAPES = [
    (1, 256, 256),
    (8, 512, 256),
    (17, 256, 512),
    (64, 512, 2064),
    (128, 512, 256),
    (256, 48, 512),
    (300, 300, 2064),
    (333, 1152, 1152),
    (512, 256, 1024),
    (512, 512, 4112),
    (1024, 512, 4112),
    (1024, 1024, 4112),
    (1024, 2048, 512),
    (2048, 2048, 4112),
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


@needs_wmma
@pytest.mark.parametrize("k", [3840, 10240, 32768])
def test_int8_linear_convrot_large_k_matches_eager(k):
    """G=256 INT8 convrot: fused (3840, 10240), global spill (32768) vs eager."""
    torch.manual_seed(k)
    m, n = 64, 256
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


def _offset_copy(t: torch.Tensor) -> torch.Tensor:
    """A contiguous copy of ``t`` deliberately based off a 16-byte boundary."""
    flat = t.reshape(-1)
    storage = torch.empty(flat.numel() + 16, dtype=t.dtype, device=t.device)
    view = storage[1:1 + flat.numel()].view_as(t)
    view.copy_(t)
    assert view.is_contiguous() and view.data_ptr() % 16 != 0
    return view


@needs_wmma
@pytest.mark.parametrize("m", [4, 256], ids=["gemv", "wmma"])
def test_int8_linear_realigns_an_offset_weight(m):
    """A contiguous weight can still start off a 16-byte boundary.

    contiguous() is a no-op on such a slice, and the row loads are uint4, so the
    operand has to be moved to a fresh allocation first.
    """
    torch.manual_seed(0)
    n, k = 64, 256
    aligned = torch.randint(-127, 127, (n, k), dtype=torch.int8, device=DEV)
    offset = _offset_copy(aligned)

    x = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)
    ws = torch.full((n,), 0.01, dtype=torch.float32, device=DEV)

    with ck.use_backend("hip"):
        out_aligned = ck.int8_linear(x, aligned, ws, None, torch.bfloat16)
        out_offset = ck.int8_linear(x, offset, ws, None, torch.bfloat16)
    assert torch.equal(out_offset, out_aligned)


@needs_wmma
@pytest.mark.parametrize("m", [4, 256], ids=["gemv", "wmma"])
def test_scaled_mm_fp8_realigns_offset_operands(hip, m):
    """Same trap on the fp8 entry, where both operands come straight from the caller."""
    torch.manual_seed(0)
    n, k = 64, 256
    a = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)
    b = torch.randn(n, k, device=DEV, dtype=torch.bfloat16)
    sa = (a.abs().max() / 448).float()
    sb = (b.abs().max() / 448).float()
    aq = (a / sa).to(torch.float8_e4m3fn)
    bq = (b / sb).to(torch.float8_e4m3fn)

    ref = hip.scaled_mm_fp8(aq, bq.t(), sa, sb, None, torch.bfloat16)
    got = hip.scaled_mm_fp8(
        _offset_copy(aq), _offset_copy(bq).t(), sa, sb, None, torch.bfloat16
    )
    assert torch.equal(got, ref)


def _gelu_tanh(x):
    return torch.nn.functional.gelu(x, approximate="tanh")


# One shape per route: only the fused ConvRot quantizer absorbs the activation.
@needs_wmma
@pytest.mark.parametrize(
    ("tag", "shape", "kwargs"),
    [
        ("fused convrot", (256, 1024), {"convrot": True, "convrot_groupsize": 256}),
        ("no convrot", (256, 1024), {"convrot": False}),
        ("K not a multiple of the group", (256, 512 + 16), {"convrot": False}),
        ("m == 1", (1, 1024), {"convrot": True, "convrot_groupsize": 256}),
    ],
)
def test_int8_linear_input_act_matches_the_eager_chain(tag, shape, kwargs):
    """int8_linear(x, input_act=a) == int8_linear(a(x)) on every route."""
    torch.manual_seed(0)
    m, k = shape
    n = 256
    x = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)
    w = torch.randn(n, k, device=DEV, dtype=torch.bfloat16)
    wq, ws = ck.quantize_int8_rowwise(w)
    ws = ws.reshape(-1)

    with ck.use_backend("hip"):
        ref = ck.int8_linear(_gelu_tanh(x), wq, ws, None, torch.bfloat16, **kwargs)
        out = ck.int8_linear(x, wq, ws, None, torch.bfloat16,
                             input_act="gelu_tanh", **kwargs)

    # An elementwise tolerance is meaningless for an int8 GEMM: one LSB in the
    # quantized activation shifts the whole accumulated sum.
    scale = ref.float().abs().max().item()
    assert (out.float() - ref.float()).abs().max().item() < 0.05 * scale, tag


def _swiglu(x):
    gate, up = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * up


# swiglu is the gated pair: the input row is [gate | up] and the quantized row is
# half as wide, so the shapes below give the raw (2 * K) width.
@needs_wmma
@pytest.mark.parametrize(
    ("tag", "shape", "kwargs"),
    [
        ("fused convrot", (256, 2 * 1024), {"convrot": True, "convrot_groupsize": 256}),
        ("no convrot", (256, 2 * 1024), {"convrot": False}),
        ("m == 1", (1, 2 * 1024), {"convrot": True, "convrot_groupsize": 256}),
    ],
)
def test_int8_linear_swiglu_matches_the_eager_chain(tag, shape, kwargs):
    """int8_linear(x, input_act="swiglu") == int8_linear(swiglu(x)) on every route."""
    torch.manual_seed(0)
    m, k2 = shape
    k = k2 // 2
    n = 256
    x = torch.randn(m, k2, device=DEV, dtype=torch.bfloat16)
    w = torch.randn(n, k, device=DEV, dtype=torch.bfloat16)
    wq, ws = ck.quantize_int8_rowwise(w)
    ws = ws.reshape(-1)

    with ck.use_backend("hip"):
        ref = ck.int8_linear(_swiglu(x), wq, ws, None, torch.bfloat16, **kwargs)
        out = ck.int8_linear(x, wq, ws, None, torch.bfloat16,
                             input_act="swiglu", **kwargs)

    assert out.shape == (m, n), tag
    scale = ref.float().abs().max().item()
    assert (out.float() - ref.float()).abs().max().item() < 0.05 * scale, tag


@needs_wmma
def test_swiglu_quantizer_is_at_least_as_accurate_as_the_chain(hip):
    """Fusing the gate into the rotation's load must not lose accuracy against
    materializing silu(gate) * up in bf16 first."""
    torch.manual_seed(0)
    m, k, group = 512, 1024, 256
    x = torch.randn(m, 2 * k, device=DEV, dtype=torch.bfloat16) * 2.0

    xd = x.double()
    activated = torch.nn.functional.silu(xd[:, :k]) * xd[:, k:]
    mat = _build_hadamard(group, device=x.device, dtype=torch.float64)
    rotated = (activated.reshape(-1, k // group, group) @ mat).reshape(-1, k)
    scale = (rotated.abs().amax(-1, keepdim=True) / 127.0).clamp(min=1e-30)
    exact = (rotated / scale).round().clamp(-128, 127)

    chain_q, _ = hip._rotate_quant_int8(_swiglu(x).contiguous(), group)
    fused_q, fused_s = hip._rotate_quant_int8(x, group, "swiglu")

    err_chain = (chain_q.double() - exact).abs().mean().item()
    err_fused = (fused_q.double() - exact).abs().mean().item()
    assert err_fused <= max(err_chain * 1.05, 1e-4), (
        f"fused ({err_fused:.6f}) less accurate than chain ({err_chain:.6f})"
    )
    assert fused_q.shape == (m, k)
    assert fused_s.shape == (m,)


@needs_wmma
def test_int8_linear_input_act_none_is_the_identity():
    """None and "none" must be bit-identical to omitting the argument."""
    torch.manual_seed(0)
    x = torch.randn(128, 1024, device=DEV, dtype=torch.bfloat16)
    w = torch.randn(256, 1024, device=DEV, dtype=torch.bfloat16)
    wq, ws = ck.quantize_int8_rowwise(w)
    ws = ws.reshape(-1)

    with ck.use_backend("hip"):
        outs = [
            ck.int8_linear(x, wq, ws, None, torch.bfloat16, convrot=True,
                           convrot_groupsize=256, **extra)
            for extra in ({}, {"input_act": None}, {"input_act": "none"})
        ]
    assert torch.equal(outs[0], outs[1])
    assert torch.equal(outs[0], outs[2])


@needs_wmma
@pytest.mark.parametrize("convrot", [False, True])
def test_int8_linear_rejects_an_unknown_input_act(convrot):
    """Every route must reject the same way, before it picks a quantizer."""
    x = torch.randn(128, 1024, device=DEV, dtype=torch.bfloat16)
    wq = torch.randint(-127, 127, (256, 1024), dtype=torch.int8, device=DEV)
    ws = torch.full((256,), 0.01, dtype=torch.float32, device=DEV)

    with ck.use_backend("hip"), pytest.raises(ValueError, match="unsupported input_act"):
        ck.int8_linear(x, wq, ws, None, torch.bfloat16, input_act="silu",
                       convrot=convrot, convrot_groupsize=256)


def test_convrot_quantizer_folds_the_activation_in(hip):
    """Folding gelu into the rotation must not be worse than gelu-then-rotate.

    It is normally better: the eager chain rounds gelu's output back to the input
    dtype before the quantizer sees it, while the fold stays in fp32 throughout.
    """
    torch.manual_seed(0)
    group = 256
    x = torch.randn(512, 1024, device=DEV, dtype=torch.bfloat16) * 2.0
    h = _build_hadamard(group, device=DEV, dtype=torch.bfloat16)

    # fp64 truth, nothing rounded in between. The fused kernel synthesizes this
    # same regular Hadamard from radix-4 butterflies.
    xd = x.double()
    gd = 0.5 * xd * (1 + torch.tanh(0.7978845608028654 * (xd + 0.044715 * xd**3)))
    mat = _build_hadamard(group, device=DEV, dtype=torch.float64)
    rot = (gd.reshape(-1, x.shape[-1] // group, group) @ mat).reshape_as(gd)
    truth = (rot / (rot.abs().amax(-1, keepdim=True) / 127.0).clamp(min=1e-30)).round()

    chain_q, _ = hip.quantize_and_rotate_rowwise(_gelu_tanh(x).contiguous(), h, group)
    fused_q, fused_s = hip._rotate_quant_int8(x, group, "gelu_tanh")

    err_chain = (chain_q.double() - truth).abs().mean().item()
    err_fused = (fused_q.double() - truth).abs().mean().item()
    assert err_fused <= max(err_chain * 1.05, 1e-4), (err_fused, err_chain)
    assert fused_q.shape == x.shape
    assert fused_s.shape == (x.shape[0],)


@pytest.mark.parametrize("group_size", [16, 256])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_convrot_int8_preserves_input_dtype_precision(hip, dtype, group_size):
    """The fused row buffer must round to the input dtype before absmax, like legacy."""
    k = group_size if group_size == 256 else group_size * 4
    idx1 = group_size if group_size == 16 else group_size // 4
    norm = (1.0 / float(group_size)) ** 0.5
    x = torch.zeros(2, k, device=DEV, dtype=dtype)
    x[0, 0] = 1.001
    x[1, idx1] = 3.141
    h = _build_hadamard(group_size, device=DEV, dtype=dtype)

    q_hip, scales_hip = hip.quantize_and_rotate_rowwise(x, h, group_size)
    q_eager, scales_eager = eager_quantize_and_rotate_rowwise(x, h, group_size)

    values = torch.stack((x[0, 0], x[1, idx1]))
    expected_rowmax = (values.float() * norm).to(dtype).float().abs()
    torch.testing.assert_close(scales_hip.reshape(-1) * 127.0, expected_rowmax, rtol=1e-6, atol=0)
    torch.testing.assert_close(scales_hip, scales_eager, rtol=1e-6, atol=0)
    assert (q_hip.int() - q_eager.int()).abs().max().item() <= 1


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_convrot_int4_preserves_input_dtype_precision(hip, dtype):
    """The W4A4 quantizer shares the dtype-correct fused rotation buffer."""
    group = 16
    weight = torch.zeros(2, 64, device=DEV, dtype=dtype)
    weight[0, 0] = 1.001
    weight[1, 16] = 3.141

    q_hip, scales_hip = hip.quantize_convrot_w4a4_weight(weight, group)
    with ck.use_backend("eager"):
        q_eager, scales_eager = ck.quantize_convrot_w4a4_weight(weight, group)

    assert torch.equal(q_hip, q_eager)
    values = torch.stack((weight[0, 0], weight[1, 16]))
    expected_rowmax = (values.float() * 0.25).to(dtype).float().abs()
    torch.testing.assert_close(scales_hip * 7.0, expected_rowmax, rtol=1e-6, atol=0)
    torch.testing.assert_close(scales_hip, scales_eager, rtol=2e-3, atol=0)


@pytest.mark.parametrize("group_size", [16, 64, 256])
def test_convrot_w4a4_weight_quant_close_to_eager(hip, group_size):
    """The fused rotation runs in fp32; eager rotates in the weight dtype.

    Codes may therefore differ by one level, but no more.
    """
    torch.manual_seed(0)

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


# ---------------------------------------------------------------------------
# Grouped W4A8 over the INT8 GEMM
# ---------------------------------------------------------------------------

def _quantize_w4a8(n, k, **kwargs):
    w = torch.randn(n, k, device=DEV, dtype=torch.bfloat16) * 0.02
    return w, eager_w4a8.quantize_w4a8_int8_weight(w, **kwargs)


@pytest.mark.parametrize("group_size", [4, 8, 16, 32, 64])
@pytest.mark.parametrize("scale_dtype", [torch.float8_e4m3fn, torch.float32])
@pytest.mark.parametrize("codebook", [False, True])
def test_w4a8_int4_decode_is_bit_exact(hip, group_size, scale_dtype, codebook):
    """The decode rounds one value at a time, so it must agree with eager exactly."""
    torch.manual_seed(0)
    _, (qdata, s_rel, _, _, cb) = _quantize_w4a8(
        128, 512, group_size=group_size, scale_dtype=scale_dtype, codebook=codebook
    )

    out = hip._dequant_int4_grouped_to_int8(qdata, s_rel, cb, group_size)
    ref = eager_w4a8._dequant_int4_grouped_to_int8(qdata, s_rel, cb, group_size)

    assert out.shape == ref.shape and out.dtype == torch.int8
    assert torch.equal(out, ref)


@pytest.mark.parametrize("symmetric", [True, False])
def test_w4a8_dequantize_weight_matches_eager(hip, symmetric):
    torch.manual_seed(0)
    w, (qdata, s_rel, s_channel, correction, cb) = _quantize_w4a8(
        128, 512, symmetric=symmetric, codebook=symmetric
    )
    kwargs = {"codebook": cb, "correction": correction, "output_dtype": torch.bfloat16}

    out = hip.dequantize_w4a8_int8_weight(qdata, s_rel, s_channel, **kwargs)
    ref = eager_w4a8.dequantize_w4a8_int8_weight(qdata, s_rel, s_channel, **kwargs)

    torch.testing.assert_close(out.float(), ref.float(), rtol=0, atol=0)
    rel = (out.float() - w.float()).norm() / w.float().norm()
    assert rel < 0.2  # int4 weight fidelity


@pytest.mark.parametrize(("m", "n", "k"), [(1, 256, 512), (17, 512, 256), (256, 1024, 512)])
@pytest.mark.parametrize("bias", [False, True])
@needs_wmma
def test_w4a8_linear_matches_eager(hip, m, n, k, bias):
    torch.manual_seed(0)
    _, (qdata, s_rel, s_channel, correction, cb) = _quantize_w4a8(n, k)
    x = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)
    bias_t = torch.randn(n, device=DEV, dtype=torch.bfloat16) if bias else None
    kwargs = {
        "codebook": cb,
        "correction": correction,
        "bias": bias_t,
        "out_dtype": torch.bfloat16,
    }

    out = hip.w4a8_int8_linear(x, qdata, s_rel, s_channel, **kwargs)
    ref = eager_w4a8.w4a8_int8_linear(x, qdata, s_rel, s_channel, **kwargs)

    assert out.shape == (m, n)
    # Both decode the weight identically and differ only in how the activation
    # quantizer rounds, so compare with an int8-sized tolerance.
    scale = ref.float().abs().max().item()
    assert (out.float() - ref.float()).abs().max().item() < 0.05 * scale


@pytest.mark.parametrize(("n", "chunk_cols"), [(1024, 256), (1280, 512), (1152, 384)])
@pytest.mark.parametrize("m", [4, 192], ids=["gemv", "wmma"])
@needs_wmma
def test_w4a8_linear_chunking_does_not_change_the_result(hip, monkeypatch, n, chunk_cols, m):
    """Each chunk writes an N-column slice of the output through ldc, so a chunk
    boundary (including a short trailing one) must not disturb its neighbours."""
    torch.manual_seed(0)
    k = 512
    _, (qdata, s_rel, s_channel, _, cb) = _quantize_w4a8(n, k)
    x = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)
    bias = torch.randn(n, device=DEV, dtype=torch.bfloat16)

    monkeypatch.setattr(hip, "_w4a8_chunk_cols", lambda *_: n)
    one_pass = hip.w4a8_int8_linear(x, qdata, s_rel, s_channel, codebook=cb, bias=bias)
    monkeypatch.setattr(hip, "_w4a8_chunk_cols", lambda *_: chunk_cols)
    chunked = hip.w4a8_int8_linear(x, qdata, s_rel, s_channel, codebook=cb, bias=bias)

    assert chunked.shape == (m, n)
    assert torch.equal(chunked, one_pass)


def test_w4a8_chunk_cols_stays_within_its_bounds(hip):
    n, k, dev = 12288, 3072, torch.device(DEV, torch.cuda.current_device())
    cols = hip._w4a8_chunk_cols(1, n, k, dev)
    assert cols < n
    assert cols % 128 == 0
    assert hip._W4A8_MIN_CHUNK_COLS <= cols <= hip._W4A8_MAX_CHUNK_COLS
    # Enough rows that the GEMM, not the decode, sets the cost.
    assert hip._w4a8_chunk_cols(hip._W4A8_CHUNK_MAX_ROWS + 1, n, k, dev) == n
    # A weight no wider than one chunk is decoded in one pass either way.
    assert hip._w4a8_chunk_cols(1, cols // 2, k, dev) == cols // 2
    # A deep K keeps the floor rather than shrinking the GEMM's N grid, and a
    # shallow one keeps the cap rather than widening a chunk pointlessly.
    assert hip._w4a8_chunk_cols(1, n, 1 << 20, dev) == hip._W4A8_MIN_CHUNK_COLS
    assert hip._w4a8_chunk_cols(1, 1 << 20, 16, dev) == hip._W4A8_MAX_CHUNK_COLS


@pytest.mark.parametrize("m", [4, 192], ids=["gemv", "wmma"])
@needs_wmma
def test_w4a8_realigns_an_offset_qdata(hip, m):
    """The decode reads packed weights as uint2, so qdata has to start aligned;
    contiguous() is a no-op on a slice that does not."""
    torch.manual_seed(0)
    n, k = 256, 512
    _, (qdata, s_rel, s_channel, _, cb) = _quantize_w4a8(n, k)
    offset = _offset_copy(qdata)
    x = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)

    assert torch.equal(
        hip._dequant_int4_grouped_to_int8(offset, s_rel, cb, 16),
        hip._dequant_int4_grouped_to_int8(qdata, s_rel, cb, 16),
    )
    assert torch.equal(
        hip.w4a8_int8_linear(x, offset, s_rel, s_channel, codebook=cb),
        hip.w4a8_int8_linear(x, qdata, s_rel, s_channel, codebook=cb),
    )


@needs_wmma
def test_w4a8_consumes_a_row_chunked_quantize(hip):
    """The shared packer rotates and packs in row chunks and pins one codebook across
    them. HIP owns no part of that, so what it must keep proving is that it decodes
    whatever comes out, including a table reused across a requantize."""
    torch.manual_seed(0)
    k = 2048
    n = eager_w4a8._QUANT_ROW_ELEM_BUDGET // k + 512  # over the packer's row budget
    w = torch.randn(n, k, device=DEV, dtype=torch.bfloat16) * 0.02
    qdata, s_rel, _, _, cb = eager_w4a8.quantize_w4a8_int8_weight(w)
    assert cb is not None

    assert torch.equal(
        hip._dequant_int4_grouped_to_int8(qdata, s_rel, cb, 16),
        eager_w4a8._dequant_int4_grouped_to_int8(qdata, s_rel, cb, 16),
    )

    # A LoRA-style reload pins the earlier table instead of choosing a new one.
    updated = w + torch.randn_like(w) * 0.002
    packed = eager_w4a8.quantize_w4a8_int8_weight(updated, codebook_tensor=cb)
    x = torch.randn(64, k, device=DEV, dtype=torch.bfloat16)
    out = hip.w4a8_int8_linear(x, packed[0], packed[1], packed[2], codebook=packed[4])
    ref = eager_w4a8.w4a8_int8_linear(x, packed[0], packed[1], packed[2], codebook=packed[4])

    scale = ref.float().abs().max().item()
    assert (out.float() - ref.float()).abs().max().item() < 0.05 * scale


# ---------------------------------------------------------------------------
# Fused W4A8 requantize
# ---------------------------------------------------------------------------


def _requant_inputs(n, k, dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    w = torch.randn(n, k, device=DEV, dtype=dtype) * 0.02
    cb = eager_w4a8._decide_codebook(w, eager_rotate_int8_convrot_weight, 16, 256)
    return w, cb


def _requires_fused_requant(hip, w):
    """The wrapper falls back to the eager packer when the device's LDS budget declines
    this K, which would leave a test exercising eager under the kernel's name."""
    assert hip._requant_supported(w.shape[1], w.device, w.dtype)


def _assert_same_packed(out, ref):
    """Bit equality across the packed tuple, reaching the fp8 scales through uint8."""
    for got, want in zip(out, ref, strict=True):
        if want is None:
            assert got is None
        elif want.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            assert torch.equal(got.view(torch.uint8), want.view(torch.uint8))
        else:
            assert torch.equal(got, want)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(("n", "k"), [(64, 256), (33, 512), (128, 1024), (96, 4096)])
def test_w4a8_fused_quantize_matches_eager(hip, dtype, n, k):
    """The kernel repeats eager's arithmetic on the same rotated values, so it has to
    reproduce eager's codes, not merely land near them.

    The one thing it cannot reproduce is the summation order of eager's 16-wide
    reductions, which is torch's, not a sequential accumulate. That moves an ALS group
    scale by an ulp, and an ulp is visible only where the scale sat on an e4m3 rounding
    boundary -- so a handful of groups shift one e4m3 code, and the weights in those
    groups shift at most one int4 level. Anything wider than that is a real divergence.
    """
    w, cb = _requant_inputs(n, k, dtype)
    _requires_fused_requant(hip, w)
    ref = eager_w4a8.quantize_w4a8_int8_weight(w, 16, convrot_groupsize=256, codebook_tensor=cb)
    out = hip.quantize_w4a8_int8_weight(w, 16, convrot_groupsize=256, codebook_tensor=cb)

    # Adjacency is the real assertion; the counts are a second guard against a
    # systematic divergence, which would flip a large fraction rather than a few.
    # Measured over these cases: at most 1 group and 0 weights differ, and the
    # smallest shapes differ nowhere, so the floor keeps their bound a handful wide
    # instead of exactly one.
    srel_delta = (out[1].view(torch.uint8).int() - ref[1].view(torch.uint8).int()).abs()
    assert srel_delta.max() <= 1
    assert srel_delta.count_nonzero() <= max(8, srel_delta.numel() // 1000)

    code_delta = (
        _unpack_int4_row_major(out[0]).int() - _unpack_int4_row_major(ref[0]).int()
    ).abs()
    assert code_delta.max() <= 1
    assert code_delta.count_nonzero() <= max(8, code_delta.numel() // 10000)

    torch.testing.assert_close(out[2], ref[2], rtol=1e-6, atol=0)
    assert out[3] is None and ref[3] is None

    # ...and the decode is no less faithful to the weight than eager's.
    def rel(packed):
        decoded = ck.dequantize_w4a8_int8_weight(
            packed[0], packed[1], packed[2], packed[4], None, 16, 256, torch.float32
        )
        return ((decoded - w.float()).norm() / w.float().norm()).item()

    assert rel(out) <= rel(ref) * 1.01


def test_w4a8_fused_quantize_does_not_depend_on_the_row_block(hip, monkeypatch):
    """Rows are rotated a block at a time to cap peak memory. Every output is per-row
    and the kernel reduces only within a row, so the packed result must be the same
    whether a row block covers the tensor or a ragged tail closes it.

    The stochastic draw is the documented exception. The kernel keys it on the element
    index within its launch and the caller shifts the seed by the block's first row, so
    the draw is reproducible for a given block size and deliberately decorrelated
    across blocks. Both other backends do this; pinned here so a "fix" that keys the
    draw globally has to change a test that says why.

    The rotation ahead of the kernel is an eager matmul, and its tiling follows the
    row count, so a row comes out of it up to an ulp apart between blockings on some
    cards. Every blocking is handed the same rotated rows, which leaves the kernel as
    the only thing the comparison can fail on; the stub checks the rows it is asked to
    rotate, so the caller's slicing is still under test.
    """
    n, k = 700, 512
    w, cb = _requant_inputs(n, k)
    _requires_fused_requant(hip, w)
    kwargs = {"convrot_groupsize": 256, "codebook_tensor": cb}

    rotated = eager_rotate_int8_convrot_weight(w, 256)
    cursor = [0]

    def rotate_rows(block, groupsize):
        r0, r1 = cursor[0], cursor[0] + block.shape[0]
        assert groupsize == 256 and torch.equal(block, w[r0:r1])
        cursor[0] = r1
        return rotated[r0:r1]

    monkeypatch.setattr(hip._eager, "rotate_int8_convrot_weight", rotate_rows)

    def quantize(budget, **extra):
        monkeypatch.setattr(hip, "_QUANT_ROW_ELEM_BUDGET", budget)
        cursor[0] = 0
        return hip.quantize_w4a8_int8_weight(w, 16, **kwargs, **extra)

    whole = quantize(n * k)
    chunked = quantize(128 * k)  # 5 blocks + a 60-row tail

    assert torch.equal(whole[0], chunked[0])
    assert torch.equal(whole[1].view(torch.uint8), chunked[1].view(torch.uint8))
    assert torch.equal(whole[2], chunked[2])

    assert not torch.equal(quantize(n * k, stochastic_rounding=7)[0],
                           quantize(128 * k, stochastic_rounding=7)[0])


@pytest.mark.parametrize(
    ("tag", "kwargs"),
    [
        ("asymmetric", {"symmetric": False}),
        ("uniform", {"codebook": False}),
        ("fp32_scale", {"scale_dtype": torch.float32}),
        ("group_size_32", {"group_size": 32}),
    ],
)
def test_w4a8_fused_quantize_declines_off_the_default_layout(hip, tag, kwargs, monkeypatch):
    """The kernel implements one layout. Everything else has to reach the shared
    packer unchanged, not a near-miss fast path.

    The fused entry is stubbed rather than inferred from output equality: the two
    paths happen to agree to an ulp today, so equality alone would keep passing if
    the gate ever let one of these layouts through.
    """
    w, _ = _requant_inputs(64, 512)
    kwargs = {"convrot_groupsize": 256, **kwargs}

    def _refuse(*_args, **_kwargs):
        raise AssertionError(f"fused path taken for {tag}")

    monkeypatch.setattr(hip, "_fused_quantize_w4a8", _refuse)
    ref = eager_w4a8.quantize_w4a8_int8_weight(w, **kwargs)
    out = hip.quantize_w4a8_int8_weight(w, **kwargs)

    _assert_same_packed(out, ref)


def test_w4a8_fused_quantize_declines_a_row_wider_than_lds(hip, monkeypatch):
    """Group scales sit in LDS, so K is bounded per device. Past the budget the
    wrapper must fall back instead of launching something that cannot fit.

    Declining the budget and consulting it are two claims: the fused entry is
    stubbed so the second one is asserted rather than inferred from a shape the two
    paths happen to agree on.
    """
    w, cb = _requant_inputs(8, 512)
    monkeypatch.setattr(hip, "_w4a8_requant_max_k", {})
    monkeypatch.setattr(hip._C, "w4a8_requant_max_k", lambda: 256)
    assert not hip._requant_supported(512, w.device, w.dtype)

    def _refuse(*_args, **_kwargs):
        raise AssertionError("fused path taken for a row wider than the LDS budget")

    monkeypatch.setattr(hip, "_fused_quantize_w4a8", _refuse)
    ref = eager_w4a8.quantize_w4a8_int8_weight(w, 16, convrot_groupsize=256, codebook_tensor=cb)
    out = hip.quantize_w4a8_int8_weight(w, 16, convrot_groupsize=256, codebook_tensor=cb)
    _assert_same_packed(out, ref)


def test_w4a8_fused_quantize_stochastic_rounding_is_seeded(hip):
    """A seed has to reproduce, and two seeds have to disagree, or the draw is not
    doing the job it was added for."""
    w, cb = _requant_inputs(128, 512)
    _requires_fused_requant(hip, w)
    kwargs = {"convrot_groupsize": 256, "codebook_tensor": cb}
    a = hip.quantize_w4a8_int8_weight(w, 16, stochastic_rounding=7, **kwargs)[0]
    b = hip.quantize_w4a8_int8_weight(w, 16, stochastic_rounding=7, **kwargs)[0]
    c = hip.quantize_w4a8_int8_weight(w, 16, stochastic_rounding=8, **kwargs)[0]

    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_w4a8_fused_quantize_stochastic_rounding_is_unbiased(hip):
    """Round-to-nearest carries a fixed per-weight bias: the same weight always lands
    on the same code, so a merged delta smaller than the int4 step rounds away every
    time. The seeded draw is unbiased instead, so averaging decodes over seeds closes
    on the original weight where a single nearest decode cannot."""
    w, cb = _requant_inputs(128, 512)
    _requires_fused_requant(hip, w)
    kwargs = {"convrot_groupsize": 256, "codebook_tensor": cb}
    reference = w.float()

    def decode(packed):
        return ck.dequantize_w4a8_int8_weight(
            packed[0], packed[1], packed[2], packed[4], None, 16, 256, torch.float32
        )

    def rel(decoded):
        return ((decoded - reference).norm() / reference.norm()).item()

    nearest = decode(hip.quantize_w4a8_int8_weight(w, 16, **kwargs))
    draws = torch.zeros_like(reference)
    seeds = 32
    for seed in range(1, seeds + 1):
        draws += decode(hip.quantize_w4a8_int8_weight(w, 16, stochastic_rounding=seed, **kwargs))

    assert rel(draws / seeds) < 0.6 * rel(nearest)


@needs_wmma
def test_w4a8_linear_asymmetric_matches_eager(hip):
    """The zero-point correction has no INT8 epilogue; that layout runs off the
    dequantized weight, which the eager path does too."""
    torch.manual_seed(0)
    _, (qdata, s_rel, s_channel, correction, cb) = _quantize_w4a8(
        128, 512, symmetric=False, codebook=False
    )
    assert correction is not None
    x = torch.randn(64, 512, device=DEV, dtype=torch.bfloat16)
    kwargs = {"codebook": cb, "correction": correction, "out_dtype": torch.bfloat16}

    out = hip.w4a8_int8_linear(x, qdata, s_rel, s_channel, **kwargs)
    ref = eager_w4a8.w4a8_int8_linear(x, qdata, s_rel, s_channel, **kwargs)

    torch.testing.assert_close(out.float(), ref.float(), rtol=0, atol=0)


@needs_wmma
def test_w4a8_linear_dispatches_to_hip():
    torch.manual_seed(0)
    w = torch.randn(256, 512, device=DEV, dtype=torch.bfloat16) * 0.02
    qdata, params = AsymW4A8Int8Layout.quantize(w)
    x = torch.randn(32, 512, device=DEV, dtype=torch.bfloat16)

    impl = registry.get_implementation(
        "w4a8_int8_linear",
        kwargs={
            "x": x,
            "qdata": qdata,
            "s_rel": params.scale,
            "s_channel": params.s_channel,
            "codebook": params.codebook,
            "correction": params.correction,
            "bias": None,
            "group_size": params.group_size,
            "convrot_groupsize": params.convrot_groupsize,
            "out_dtype": torch.bfloat16,
        },
    )
    assert impl.__module__ == "comfy_kitchen.backends.hip"

    out = torch.nn.functional.linear(x, QuantizedTensor(qdata, "AsymW4A8Int8Layout", params))
    rel = (out.float() - (x @ w.t()).float()).norm() / (x @ w.t()).float().norm()
    assert out.shape == (32, 256)
    assert rel < 0.2


# 0 and 3 are caught by the positivity and divisibility checks. 2 and 12 divide
# their K and only the vector layout rejects them: 2 is under the four scales a
# vector loads, 12 neither divides 16 nor is a multiple of it.
@pytest.mark.parametrize(("group_size", "k"), [(0, 256), (3, 256), (2, 256), (12, 48)])
def test_w4a8_decode_launcher_rejects_an_unsupported_group_size(hip, group_size, k):
    """Each vector loads four group scales, so the wrappers only reach the kernel
    with a group of 4, 8, 16 or a multiple of 16; guard the C++ callers too."""
    qdata = torch.zeros(8, k // 2, dtype=torch.int8, device=DEV)
    s_rel = torch.ones(8, k // max(group_size, 1), dtype=torch.float32, device=DEV)
    out = torch.zeros(8, k, dtype=torch.int8, device=DEV)

    with pytest.raises(RuntimeError, match="group_size"):
        hip._C.dequant_int4_grouped_to_int8(
            hip._dl(qdata), hip._dl(s_rel), 0, None, hip._dl(out), 8, k, group_size,
            hip._stream(qdata),
        )


def test_quantize_int8_rowwise_matches_eager():
    torch.manual_seed(0)
    x = torch.randn(64, 512, device=DEV, dtype=torch.bfloat16)

    with ck.use_backend("hip"):
        q, s = ck.quantize_int8_rowwise(x)
    with ck.use_backend("eager"):
        qe, se = ck.quantize_int8_rowwise(x)

    torch.testing.assert_close(s.float(), se.float(), rtol=1e-2, atol=0)
    assert (q.int() - qe.int()).abs().max().item() <= 1


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("k", [8, 9], ids=["vector-width", "ragged"])
def test_quantize_int8_rowwise_offset_matches_aligned(dtype, k):
    torch.manual_seed(k)
    aligned = torch.randn(17, k, device=DEV, dtype=dtype)
    offset = _offset_copy(aligned)

    with ck.use_backend("hip"):
        aligned_q, aligned_scale = ck.quantize_int8_rowwise(aligned)
        offset_q, offset_scale = ck.quantize_int8_rowwise(offset)

    assert torch.equal(offset_q, aligned_q)
    assert torch.equal(offset_scale, aligned_scale)


def test_quantize_int8_tensorwise_matches_eager():
    torch.manual_seed(0)
    x = torch.randn(64, 512, device=DEV, dtype=torch.bfloat16)

    with ck.use_backend("hip"):
        q, s = ck.quantize_int8_tensorwise(x)
    with ck.use_backend("eager"):
        qe, se = ck.quantize_int8_tensorwise(x)

    torch.testing.assert_close(s.float().reshape(()), se.float().reshape(()), rtol=1e-2, atol=0)
    assert (q.int() - qe.int()).abs().max().item() <= 1


@pytest.mark.parametrize(
    ("scale_kind", "scale_shape"),
    [
        ("scalar", ()),
        ("elementwise", (3, 4, 32)),
        ("rowwise", (3, 4, 1)),
    ],
)
@pytest.mark.parametrize(
    ("output_dtype_code", "output_dtype"),
    [(0, torch.float32), (1, torch.float16), (2, torch.bfloat16)],
)
def test_dequantize_int8_simple_dtype_matches_eager(
    hip, scale_kind, scale_shape, output_dtype_code, output_dtype
):
    torch.manual_seed(0)
    q = torch.randint(-128, 128, (3, 4, 32), dtype=torch.int8, device=DEV)
    scale = torch.rand(scale_shape, dtype=torch.bfloat16, device=DEV) * 0.02

    out = hip.dequantize_int8_simple_dtype(q, scale, output_dtype_code)
    ref = (q.float() * scale).to(output_dtype)

    assert out.dtype == output_dtype
    assert torch.equal(out, ref), scale_kind


@pytest.mark.parametrize("group_size", [16, 64, 256])
@pytest.mark.parametrize("scale_kind", ["scalar", "rowwise"])
@pytest.mark.parametrize(
    ("output_dtype_code", "output_dtype"),
    [(0, torch.float32), (1, torch.float16), (2, torch.bfloat16)],
)
def test_dequantize_int8_convrot_weight_dtype_matches_eager(
    hip, group_size, scale_kind, output_dtype_code, output_dtype
):
    torch.manual_seed(0)
    q = torch.randint(
        -128, 128, (7, group_size * 2), dtype=torch.int8, device=DEV
    )
    scale_shape = () if scale_kind == "scalar" else (7, 1)
    scale = torch.rand(scale_shape, dtype=torch.float16, device=DEV) * 0.02

    out = hip.dequantize_int8_convrot_weight_dtype(
        q, scale, group_size, output_dtype_code
    )
    with ck.use_backend("eager"):
        ref = torch.ops.comfy_kitchen.dequantize_int8_convrot_weight_dtype(
            q, scale, group_size, output_dtype_code
        )

    assert out.dtype == output_dtype
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-4, atol=2e-4)


def test_int8_dtype_dequant_custom_ops_do_not_fall_back_to_eager(hip, monkeypatch):
    def unexpected_eager(*args, **kwargs):
        raise AssertionError("supported INT8 dequantization fell back to eager")

    monkeypatch.setattr(
        hip._eager, "dequantize_int8_simple_dtype", unexpected_eager
    )
    monkeypatch.setattr(
        hip._eager, "dequantize_int8_convrot_weight_dtype", unexpected_eager
    )

    q = torch.randint(-128, 128, (5, 256), dtype=torch.int8, device=DEV)
    scale = torch.rand(5, 1, dtype=torch.float32, device=DEV) * 0.02
    with ck.use_backend("hip"):
        simple = torch.ops.comfy_kitchen.dequantize_int8_simple_dtype(q, scale, 2)
        convrot = torch.ops.comfy_kitchen.dequantize_int8_convrot_weight_dtype(
            q, scale, 256, 2
        )

    assert simple.shape == q.shape
    assert convrot.shape == q.shape
    torch.cuda.synchronize()


def test_int8_dtype_dequant_empty_inputs_do_not_launch_zero_grids(hip):
    q = torch.empty(0, 256, dtype=torch.int8, device=DEV)
    scale = torch.empty(0, 1, dtype=torch.float32, device=DEV)

    simple = hip.dequantize_int8_simple_dtype(q, scale, 2)
    convrot = hip.dequantize_int8_convrot_weight_dtype(q, scale, 256, 2)

    assert simple.shape == q.shape
    assert convrot.shape == q.shape
    torch.cuda.synchronize()


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


@pytest.mark.parametrize(
    ("shape", "mshape"),
    [((2, 4096, 3072), (2, 1, 3072)), ((4, 256, 1152), (4, 1, 1152)), ((8, 128), (8, 128))],
)
def test_rms_adaln_matches_eager(shape, mshape):
    """Same kernel as adaln with the mean pinned to zero. Scored the same way."""
    torch.manual_seed(0)
    x = torch.randn(*shape, device=DEV, dtype=torch.bfloat16)
    scale = torch.randn(*mshape, device=DEV, dtype=torch.bfloat16)
    shift = torch.randn(*mshape, device=DEV, dtype=torch.bfloat16)

    with ck.use_backend("hip"):
        out = ck.rms_adaln(x, scale, shift)
    with ck.use_backend("eager"):
        ref = ck.rms_adaln(x, scale, shift)

    xf = x.float()
    truth = torch.nn.functional.rms_norm(xf, xf.shape[-1:], eps=1e-6)
    truth = truth * (1 + scale.float()) + shift.float()

    err_hip = (out.float() - truth).norm() / truth.norm()
    err_eager = (ref.float() - truth).norm() / truth.norm()

    assert err_hip < 1e-2
    assert err_hip <= err_eager * 1.1


def test_rms_adaln_is_not_adaln(hip):
    """A non-zero-mean row separates the two statistics.

    An rms_adaln silently running the LayerNorm path would pass every
    tolerance test above on zero-mean random input.
    """
    torch.manual_seed(0)
    x = torch.randn(2, 32, 256, device=DEV, dtype=torch.float32) + 5.0
    zero = torch.zeros(2, 1, 256, device=DEV, dtype=torch.float32)

    rms = hip.rms_adaln(x, zero, zero, 1e-6)
    ln = hip.adaln(x, zero, zero, 1e-6)

    assert not torch.allclose(rms, ln, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(
        rms, torch.nn.functional.rms_norm(x, x.shape[-1:], eps=1e-6), rtol=1e-4, atol=1e-5
    )


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


# (in-place name, functional sibling, takes a q/k pair, takes a norm weight)
INPLACE_ROPE_OPS = [
    ("apply_rope_", "apply_rope", True, False),
    ("apply_rope1_", "apply_rope1", False, False),
    ("apply_rope_split_half_", "apply_rope_split_half", True, False),
    ("apply_rope_split_half1_", "apply_rope_split_half1", False, False),
    ("rms_rope_", "rms_rope", True, True),
    ("rms_rope1_", "rms_rope1", False, True),
    ("rms_rope_split_half_", "rms_rope_split_half", True, True),
    ("rms_rope_split_half1_", "rms_rope_split_half1", False, True),
]


@pytest.mark.parametrize(
    ("inplace_name", "functional_name", "paired", "has_scale"),
    INPLACE_ROPE_OPS,
    ids=[case[0] for case in INPLACE_ROPE_OPS],
)
def test_rope_inplace_rotates_a_strided_view_where_it_lies(
    hip, inplace_name, functional_name, paired, has_scale
):
    """The in-place entries write through the caller's view, not a copy.

    The result must land on the original storage with the original strides, and
    match the functional path bit for bit.
    """
    torch.manual_seed(0)
    seq, dim = 3, 64
    # A last-dim stride of 2 is exactly what the functional path would copy away.
    q = torch.randn(2, 4, seq, dim * 2, device=DEV, dtype=torch.float16)[..., ::2]
    k = torch.randn(2, 4, seq, dim * 2, device=DEV, dtype=torch.float16)[..., ::2]
    freqs = torch.randn(1, 1, seq, dim // 2, 2, 2, device=DEV, dtype=torch.float32)
    scale = (torch.randn(dim, device=DEV, dtype=torch.float32),) if has_scale else ()

    functional = getattr(hip, functional_name)
    inplace = getattr(hip, inplace_name)
    operands = (q, k) if paired else (q,)
    expected = functional(*(t.clone() for t in operands), freqs, *scale)
    expected = expected if paired else (expected,)

    pointers = tuple(t.data_ptr() for t in operands)
    strides = tuple(t.stride() for t in operands)
    result = inplace(*operands, freqs, *scale)
    result = result if paired else (result,)

    assert tuple(t.data_ptr() for t in result) == pointers
    assert tuple(t.stride() for t in result) == strides
    for got, want in zip(result, expected, strict=True):
        assert torch.equal(got, want)


@pytest.mark.parametrize(
    "inplace_name", [case[0] for case in INPLACE_ROPE_OPS if not case[2]]
)
def test_rope_inplace_rejects_autograd(hip, inplace_name):
    """In-place is inference-only; a grad-tracking operand must raise."""
    x = torch.randn(1, 1, 1, 64, device=DEV, dtype=torch.float16, requires_grad=True)
    freqs = torch.randn(1, 1, 1, 32, 2, 2, device=DEV, dtype=torch.float32)
    extra = (
        (torch.randn(64, device=DEV, dtype=torch.float32),)
        if inplace_name.startswith("rms_")
        else ()
    )

    with pytest.raises(RuntimeError, match="inference-only"):
        getattr(hip, inplace_name)(x, freqs, *extra)


@pytest.mark.parametrize("split_half", [False, True])
def test_rope_inplace_splits_a_pair_the_kernel_cannot_share(hip, split_half):
    """One stride set covers both operands, so a mismatched pair goes one at a
    time. Out of place they would be densified into agreement instead."""
    torch.manual_seed(0)
    seq, dim = 5, 64
    q = torch.randn(2, 4, seq, dim, device=DEV, dtype=torch.float16)
    k = torch.randn(2, 4, seq, dim * 2, device=DEV, dtype=torch.float16)[..., ::2]
    assert hip._effective_strides(q) != hip._effective_strides(k)
    freqs = torch.randn(1, 1, seq, dim // 2, 2, 2, device=DEV, dtype=torch.float32)

    one = hip.apply_rope_split_half1 if split_half else hip.apply_rope1
    pair_ = hip.apply_rope_split_half_ if split_half else hip.apply_rope_
    expected = (one(q.clone(), freqs), one(k.clone(), freqs))

    pointers = (q.data_ptr(), k.data_ptr())
    out_q, out_k = pair_(q, k, freqs)

    assert (out_q.data_ptr(), out_k.data_ptr()) == pointers
    assert torch.equal(out_q, expected[0])
    assert torch.equal(out_k, expected[1])


# BHND puts the sequence on axis 2 and BNHD on axis 1; head_dim covers the pair
# count landing above, on and below the block width.
RMS_ROPE_LAYOUTS = [
    ((2, 8, 128, 128), (1, 1, 128, 64, 2, 2)),   # BHND, pairs == block
    ((2, 128, 8, 64), (1, 128, 1, 32, 2, 2)),    # BNHD, pairs < block
    ((1, 3, 11, 160), (1, 1, 11, 80, 2, 2)),     # head_dim neither 64 nor 128
]


@pytest.mark.parametrize("split_half", [False, True])
@pytest.mark.parametrize("freqs_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(("shape", "freqs_shape"), RMS_ROPE_LAYOUTS, ids=["BHND", "BNHD", "D160"])
def test_rms_rope_matches_eager(split_half, freqs_dtype, shape, freqs_shape):
    torch.manual_seed(0)
    head_dim = shape[-1]
    q = torch.randn(shape, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(shape, device=DEV, dtype=torch.bfloat16)
    freqs = torch.randn(freqs_shape, device=DEV, dtype=freqs_dtype)
    q_scale = torch.randn(head_dim, device=DEV, dtype=torch.float32)
    k_scale = torch.randn(head_dim, device=DEV, dtype=torch.float32)

    pair = ck.rms_rope_split_half if split_half else ck.rms_rope
    one = ck.rms_rope_split_half1 if split_half else ck.rms_rope1
    with ck.use_backend("hip"):
        out_q, out_k = pair(q, k, freqs, q_scale, k_scale)
        out_one = one(q, freqs, q_scale)
    with ck.use_backend("eager"):
        ref_q, ref_k = pair(q, k, freqs, q_scale, k_scale)

    torch.testing.assert_close(out_q.float(), ref_q.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(out_k.float(), ref_k.float(), rtol=2e-2, atol=2e-2)
    # The fused and single-tensor entries are the same launch with k dropped.
    assert torch.equal(out_one, out_q)


def _partial_rotary_reference(x, freqs, scale, epsilon, rot_dim):
    """Norm over the full head_dim, split-half rotation over the first rot_dim."""
    x_float = x.float()
    rrms = torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + epsilon)
    x_norm = (x_float * rrms * scale.float()).to(x.dtype).float()
    half = rot_dim // 2
    x1, x2 = x_norm[..., :half], x_norm[..., half:rot_dim]
    f = freqs.float()
    o1 = f[..., 0, 0] * x1 + f[..., 0, 1] * x2
    o2 = f[..., 1, 0] * x1 + f[..., 1, 1] * x2
    return torch.cat([o1, o2, x_norm[..., rot_dim:]], dim=-1).to(x.dtype)


@pytest.mark.parametrize("rot_dim", [32, 64, 96])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_rms_rope_partial_rotary(hip, rot_dim, dtype):
    """rot_dim rotates a head-dim prefix; the tail is normalized but not rotated."""
    torch.manual_seed(0)
    s, h, d = 333, 8, 128
    q = torch.randn(1, s, h, d, device=DEV, dtype=dtype)
    k = torch.randn(1, s, h, d, device=DEV, dtype=dtype)
    freqs = torch.randn(1, s, 1, rot_dim // 2, 2, 2, device=DEV, dtype=torch.float32)
    q_scale = torch.randn(d, device=DEV, dtype=torch.float32)
    k_scale = torch.randn(d, device=DEV, dtype=torch.float32)

    out_q, out_k = hip.rms_rope_split_half(q, k, freqs, q_scale, k_scale, rot_dim=rot_dim)

    for out, x, scale in ((out_q, q, q_scale), (out_k, k, k_scale)):
        ref = _partial_rotary_reference(x, freqs, scale, 1e-6, rot_dim)
        torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)


def test_rms_rope_partial_rotary_with_an_odd_tail(hip, monkeypatch):
    """Only the rotated prefix has to be even. A 65-wide row with rot_dim=64 leaves
    one norm-only element, which the kernel handles, so it must not go to eager."""
    monkeypatch.setattr(
        hip._eager, "rms_rope_split_half",
        lambda *a, **kw: pytest.fail("an odd head_dim with an even rot_dim fell back"),
    )
    torch.manual_seed(0)
    s, h, d, rot_dim = 64, 4, 65, 64
    q = torch.randn(1, s, h, d, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(1, s, h, d, device=DEV, dtype=torch.bfloat16)
    freqs = torch.randn(1, s, 1, rot_dim // 2, 2, 2, device=DEV, dtype=torch.float32)
    scale = torch.randn(d, device=DEV, dtype=torch.float32)

    out_q, out_k = hip.rms_rope_split_half(q, k, freqs, scale, scale, rot_dim=rot_dim)

    for out, x in ((out_q, q), (out_k, k)):
        ref = _partial_rotary_reference(x, freqs, scale, 1e-6, rot_dim)
        torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)


def test_rms_rope_full_rot_dim_matches_the_default(hip):
    """rot_dim == head_dim must be bit-identical to omitting it."""
    torch.manual_seed(0)
    s, h, d = 128, 4, 64
    q = torch.randn(1, s, h, d, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(1, s, h, d, device=DEV, dtype=torch.bfloat16)
    freqs = torch.randn(1, s, 1, d // 2, 2, 2, device=DEV, dtype=torch.float32)
    scale = torch.randn(d, device=DEV, dtype=torch.float32)

    full = hip.rms_rope_split_half(q, k, freqs, scale, scale, rot_dim=d)
    default = hip.rms_rope_split_half(q, k, freqs, scale, scale)

    assert torch.equal(full[0], default[0])
    assert torch.equal(full[1], default[1])


def test_rms_rope_partial_rotary_in_place_on_a_qkv_slice(hip):
    """The MiniMax layout: q and k are strided slices of one packed projection."""
    torch.manual_seed(0)
    s, h, d, rot_dim = 96, 6, 128, 96
    qkv = torch.randn(s, 3 * h * d, device=DEV, dtype=torch.bfloat16)
    qkv_ref = qkv.clone()
    freqs = torch.randn(1, s, 1, rot_dim // 2, 2, 2, device=DEV, dtype=torch.float32)
    scale = torch.randn(d, device=DEV, dtype=torch.float32)

    q, k, v = (t.view(1, s, h, d) for t in qkv.split(h * d, dim=-1))
    q_ref, k_ref, v_ref = (t.view(1, s, h, d) for t in qkv_ref.split(h * d, dim=-1))

    hip.rms_rope_split_half_(q, k, freqs, scale, scale, rot_dim=rot_dim)
    expect_q, expect_k = hip.rms_rope_split_half(
        q_ref, k_ref, freqs, scale, scale, rot_dim=rot_dim)

    assert torch.equal(q, expect_q)
    assert torch.equal(k, expect_k)
    assert torch.equal(v, v_ref), "v must not be touched by in-place q/k rope"


@pytest.mark.parametrize("rot_dim", [33, 66], ids=["odd", "over head_dim"])
def test_rms_rope_rejects_an_unusable_rot_dim(hip, rot_dim):
    """The rotation pairs (i, i + rot_dim/2) inside head_dim, so an odd prefix or
    one wider than the row has no kernel."""
    q = torch.randn(1, 8, 2, 64, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(1, 8, 2, 64, device=DEV, dtype=torch.bfloat16)
    freqs = torch.randn(1, 8, 1, rot_dim // 2, 2, 2, device=DEV, dtype=torch.float32)
    scale = torch.randn(64, device=DEV, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="rot_dim"):
        hip.rms_rope_split_half(q, k, freqs, scale, scale, rot_dim=rot_dim)


def test_rms_rope_forwards_rot_dim_to_the_eager_fallback(hip, monkeypatch):
    """A weight the kernel cannot index goes back to eager, which must still be
    told the rotation is partial: dropping rot_dim there rotates the whole row."""
    q = torch.randn(1, 8, 2, 64, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(1, 8, 2, 64, device=DEV, dtype=torch.bfloat16)
    freqs = torch.randn(1, 8, 1, 24, 2, 2, device=DEV, dtype=torch.float32)
    scale = torch.randn(1, 64, device=DEV, dtype=torch.float32)  # 2D: no kernel

    seen = {}
    monkeypatch.setattr(
        hip._eager, "rms_rope_split_half",
        lambda *a, **kw: seen.update(kw) or (torch.zeros_like(q), torch.zeros_like(k)),
    )
    hip.rms_rope_split_half(q, k, freqs, scale, scale, rot_dim=48)

    assert seen.get("rot_dim") == 48, "eager fallback lost rot_dim"


def test_rms_rope_partial_rotary_with_mismatched_q_k_heads(hip):
    """GQA splits the pair into two single-tensor launches; both need rot_dim."""
    torch.manual_seed(0)
    s, d, rot_dim = 65, 128, 96
    q = torch.randn(1, s, 8, d, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(1, s, 2, d, device=DEV, dtype=torch.bfloat16)
    freqs = torch.randn(1, s, 1, rot_dim // 2, 2, 2, device=DEV, dtype=torch.float32)
    q_scale = torch.randn(d, device=DEV, dtype=torch.float32)
    k_scale = torch.randn(d, device=DEV, dtype=torch.float32)

    out_q, out_k = hip.rms_rope_split_half(q, k, freqs, q_scale, k_scale, rot_dim=rot_dim)

    for out, x, scale in ((out_q, q, q_scale), (out_k, k, k_scale)):
        ref = _partial_rotary_reference(x, freqs, scale, 1e-6, rot_dim)
        torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("split_half", [False, True])
def test_rms_rope_reads_a_qkv_slice_in_place(hip, split_half):
    """A q/k pair sliced out of a packed qkv must match the copy bit for bit."""
    torch.manual_seed(0)
    b, s, h, d = 2, 96, 6, 128
    qkv = torch.randn(b, s, 3, h, d, device=DEV, dtype=torch.bfloat16)
    q, k, _ = qkv.unbind(dim=2)
    assert not q.is_contiguous()
    freqs = torch.randn(b, s, 1, d // 2, 2, 2, device=DEV, dtype=torch.float32)
    scale = torch.randn(d, device=DEV, dtype=torch.bfloat16)

    fn = hip.rms_rope_split_half if split_half else hip.rms_rope
    out_q, out_k = fn(q, k, freqs, scale, scale)
    ref_q, ref_k = fn(q.contiguous(), k.contiguous(), freqs, scale, scale)

    assert out_q.is_contiguous() and out_k.is_contiguous()
    assert torch.equal(out_q, ref_q)
    assert torch.equal(out_k, ref_k)


def test_rms_rope_takes_non_contiguous_freqs(hip):
    """freqs is indexed through its own strides, so a sliced table needs no copy."""
    torch.manual_seed(0)
    b, s, h, d = 2, 64, 4, 64
    x = torch.randn(b, s, h, d, device=DEV, dtype=torch.bfloat16)
    table = torch.randn(b, 4 * s, 1, d // 2, 2, 2, device=DEV, dtype=torch.float32)
    freqs = table[:, :s]
    assert not freqs.is_contiguous()
    scale = torch.randn(d, device=DEV, dtype=torch.bfloat16)

    assert torch.equal(
        hip.rms_rope1(x, freqs, scale), hip.rms_rope1(x, freqs.contiguous(), scale)
    )


def test_rms_rope_copies_when_q_and_k_disagree_on_layout(hip):
    """One stride set covers both inputs, so a mixed pair has to be normalized first."""
    torch.manual_seed(0)
    b, s, h, d = 1, 64, 4, 64
    q = torch.randn(b, s, 3, h, d, device=DEV, dtype=torch.bfloat16).unbind(dim=2)[0]
    k = torch.randn(b, s, h, d, device=DEV, dtype=torch.bfloat16)
    freqs = torch.randn(b, s, 1, d // 2, 2, 2, device=DEV, dtype=torch.float32)
    scale = torch.randn(d, device=DEV, dtype=torch.bfloat16)

    out_q, out_k = hip.rms_rope(q, k, freqs, scale, scale)

    torch.testing.assert_close(out_q.float(), hip.rms_rope1(q, freqs, scale).float())
    torch.testing.assert_close(out_k.float(), hip.rms_rope1(k, freqs, scale).float())


def test_rms_rope_splits_mismatched_operands(hip):
    """One dtype code covers both inputs and one more both weights."""
    q = torch.randn(2, 4, 64, 64, device=DEV, dtype=torch.bfloat16)
    k = q.to(torch.float16)
    freqs = torch.randn(1, 1, 64, 32, 2, 2, device=DEV, dtype=torch.float32)
    scale = torch.randn(64, device=DEV, dtype=torch.float32)

    out_q, out_k = hip.rms_rope(q, k, freqs, scale)

    assert out_q.dtype == torch.bfloat16
    assert out_k.dtype == torch.float16
    torch.testing.assert_close(out_q.float(), hip.rms_rope1(q, freqs, scale).float())
    torch.testing.assert_close(out_k.float(), hip.rms_rope1(k, freqs, scale).float())


def test_rms_rope_hands_back_a_weight_it_cannot_index(hip, monkeypatch):
    """The kernel reads one weight entry per head_dim; eager broadcasts anything else."""
    q = torch.randn(2, 4, 64, 64, device=DEV, dtype=torch.bfloat16)
    freqs = torch.randn(1, 1, 64, 32, 2, 2, device=DEV, dtype=torch.float32)
    scale = torch.randn(1, 64, device=DEV, dtype=torch.float32)

    called = []
    monkeypatch.setattr(
        hip._eager, "rms_rope1", lambda *a, **kw: called.append(a) or torch.zeros_like(q)
    )
    hip.rms_rope1(q, freqs, scale)

    assert called, "a 2D weight has no kernel and must go back to eager"


def test_rms_rope_rejects_a_short_weight(hip):
    """The kernel indexes the weight up to head_dim off a raw pointer."""
    q = torch.randn(2, 4, 64, 64, device=DEV, dtype=torch.bfloat16)
    freqs = torch.randn(1, 1, 64, 32, 2, 2, device=DEV, dtype=torch.float32)
    q_out = torch.empty_like(q)

    with pytest.raises(RuntimeError, match="q_scale"):
        hip._C.rms_rope(
            q.__dlpack__(stream=-1), None, freqs.__dlpack__(stream=-1),
            torch.randn(32, device=DEV).__dlpack__(stream=-1), None,
            q_out.__dlpack__(stream=-1), None,
            1e-6, False, torch.cuda.current_stream(q.device).cuda_stream, 0,
        )


def test_operands_that_require_grad_are_exportable(hip):
    """Weights arrive as nn.Parameter, and neither no_grad nor inference_mode
    clears requires_grad, which __dlpack__ refuses."""
    torch.manual_seed(0)
    d = 64
    x = torch.randn(1, 4, 8, d, device=DEV, dtype=torch.bfloat16)
    freqs = torch.randn(1, 1, 8, d // 2, 2, 2, device=DEV, dtype=torch.float32)
    scale = torch.nn.Parameter(torch.randn(d, device=DEV, dtype=torch.bfloat16))
    assert scale.requires_grad

    with torch.inference_mode():
        out = hip.rms_rope1(x, freqs, scale)
    torch.testing.assert_close(out.float(), hip.rms_rope1(x, freqs, scale.detach()).float())


@needs_wmma
def test_bias_that_requires_grad_is_exportable(hip):
    """The same export path carries every epilogue bias."""
    torch.manual_seed(0)
    m, n, k = 32, 128, 128
    x = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)
    w = torch.randint(-8, 8, (n, k), dtype=torch.int8, device=DEV)
    w_scale = torch.rand(n, device=DEV, dtype=torch.float32) * 0.01
    bias = torch.nn.Parameter(torch.randn(n, device=DEV, dtype=torch.bfloat16))

    with torch.inference_mode():
        out = hip.int8_linear(x, w, w_scale, bias)
    torch.testing.assert_close(
        out.float(), hip.int8_linear(x, w, w_scale, bias.detach()).float()
    )


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

    def trap(name):
        # Recording and raising: the raise names the entry point at the moment it
        # is reached, and the record still fails the assertion below if a caller
        # swallows the exception and falls back.
        def fn(*args, **kwargs):
            called.append(name)
            raise AssertionError(f"quantized path reached torch.{name}")

        return fn

    monkeypatch.setattr(torch, "_scaled_mm", trap("_scaled_mm"))
    monkeypatch.setattr(torch, "_int_mm", trap("_int_mm"))
    monkeypatch.setattr(torch, "matmul", trap("matmul"))
    monkeypatch.setattr(torch, "mm", trap("mm"))
    monkeypatch.setattr(torch, "bmm", trap("bmm"))
    if hasattr(torch.nn.functional, "scaled_mm"):
        monkeypatch.setattr(torch.nn.functional, "scaled_mm", trap("F.scaled_mm"))

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
    assert arch in hip_backend._ARCH_SUPPORTED
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


@pytest.mark.parametrize("out_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("numel", [15, 16, 17, 1 << 20])
def test_stochastic_rounding_fp8_vector_and_scalar_paths_agree(hip, dtype, numel, out_dtype):
    """An offset view runs the scalar fallback, which must give the same bits as
    the vectorized path. The sizes straddle kVecElems to cover the chunk tail, and
    the two fp8 formats take different constants through the rounding."""
    torch.manual_seed(numel)
    x = torch.randn(numel, device=DEV, dtype=dtype) * 10
    rng = torch.randint(0, 256, (numel,), dtype=torch.uint8, device=DEV)

    aligned = hip.stochastic_rounding_fp8(x.clone(), rng.clone(), out_dtype)
    offset = hip.stochastic_rounding_fp8(_offset_copy(x), _offset_copy(rng), out_dtype)

    assert x.data_ptr() % 16 == 0 and rng.data_ptr() % 16 == 0
    assert torch.equal(aligned.view(torch.uint8), offset.view(torch.uint8))


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
    with pytest.raises(RuntimeError, match="the input's dtype"):
        call(xk.to(torch.float16), xq_out, xk_out)
    with pytest.raises(RuntimeError, match="the input's shape"):
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


@pytest.mark.parametrize("split_half", [False, True])
def test_rope_reads_a_permuted_qkv_in_place(hip, split_half):
    """A q/k pair permuted out of a packed qkv must match the copy bit for bit."""
    torch.manual_seed(0)
    b, seq, heads, dim = 2, 96, 6, 128
    qkv = torch.randn(b, seq, 3 * heads * dim, device=DEV, dtype=torch.bfloat16)
    xq, xk, _ = qkv.view(b, seq, 3, heads, dim).permute(2, 0, 3, 1, 4)
    assert not xq.is_contiguous() and xq.stride() == xk.stride()
    freqs = torch.randn(1, 1, seq, dim // 2, 2, 2, device=DEV, dtype=torch.float32)

    pair = hip.apply_rope_split_half if split_half else hip.apply_rope
    out_q, out_k = pair(xq, xk, freqs)
    ref_q, ref_k = pair(xq.contiguous(), xk.contiguous(), freqs)

    assert out_q.is_contiguous() and out_k.is_contiguous()
    assert torch.equal(out_q, ref_q)
    assert torch.equal(out_k, ref_k)


def test_rope_takes_non_contiguous_freqs(hip):
    """freqs is indexed through its own strides, so a sliced table needs no copy."""
    torch.manual_seed(0)
    b, seq, heads, dim = 2, 64, 4, 64
    x = torch.randn(b, heads, seq, dim, device=DEV, dtype=torch.bfloat16)
    table = torch.randn(b, 1, 4 * seq, dim // 2, 2, 2, device=DEV, dtype=torch.float32)
    freqs = table[:, :, :seq]
    assert not freqs.is_contiguous()

    assert torch.equal(
        hip.apply_rope1(x, freqs), hip.apply_rope1(x, freqs.contiguous())
    )


def test_rope_copies_a_row_that_is_not_dense(hip):
    """A last axis with a stride would scatter every load, so it is made contiguous."""
    torch.manual_seed(0)
    b, seq, heads, dim = 1, 64, 4, 64
    x = torch.randn(b, heads, dim, seq, device=DEV, dtype=torch.bfloat16).transpose(-1, -2)
    assert x.stride(-1) != 1
    freqs = torch.randn(1, 1, seq, dim // 2, 2, 2, device=DEV, dtype=torch.float32)

    out = hip.apply_rope1(x, freqs)

    assert out.is_contiguous()
    torch.testing.assert_close(out.float(), hip.apply_rope1(x.contiguous(), freqs).float())


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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_convrot_falls_back_to_eager_past_the_lds_bound(hip, dtype):
    """The fused rotation stages a whole row in LDS, so a large K does not fit.

    The launch does not fail cleanly past the budget, so the wrapper has to route
    those shapes to eager rather than reach the kernel.
    """
    max_k = hip._C.convrot_max_k(hip.DTYPE_TO_CODE[dtype])
    assert max_k > 0

    k = (max_k + 1024) & ~63  # past the bound, still a multiple of the group size
    x = torch.randn(2, k, device=DEV, dtype=dtype)
    h = torch.eye(64, device=DEV, dtype=dtype)

    q, scales = hip.quantize_and_rotate_rowwise(x, h, 64)
    torch.cuda.synchronize()
    assert q.shape == (2, k)
    assert torch.isfinite(scales).all()

    # The launcher itself still refuses, so a C++ caller cannot slip past.
    qb = torch.zeros(2, k, dtype=torch.int8, device=DEV)
    sb = torch.zeros(2, dtype=torch.float32, device=DEV)
    with pytest.raises(RuntimeError, match="LDS"):
        hip._C.quantize_int8_convrot(
            hip._dl(x), hip._dl(qb), hip._dl(sb), None, None, 2, k, 64, 0, hip._stream(x)
        )


def test_convrot_spill_rotated_dtype_must_match_x(hip):
    """spill_rotated is written as RowT derived from x; dtype must match."""
    m, k = 2, 256
    x = torch.randn(m, k, device=DEV, dtype=torch.float32)
    q = torch.zeros(m, k, dtype=torch.int8, device=DEV)
    scales = torch.zeros(m, dtype=torch.float32, device=DEV)
    spill_rotated = torch.empty(m, k, dtype=torch.bfloat16, device=DEV)
    spill_partials = torch.empty(m, k // 256, dtype=torch.float32, device=DEV)
    with pytest.raises(RuntimeError, match="spill_rotated dtype must match x"):
        hip._C.quantize_int8_convrot(
            hip._dl(x),
            hip._dl(q),
            hip._dl(scales),
            hip._dl(spill_rotated),
            hip._dl(spill_partials),
            m,
            k,
            256,
            0,
            hip._stream(x),
        )


@needs_wmma
def test_convrot_int8_needs_spill_probe(hip):
    in_code = hip.DTYPE_TO_CODE[torch.bfloat16]
    with torch.cuda.device(DEV):
        assert not hip._C.convrot_int8_needs_spill(4128, 10240, in_code)
        assert hip._C.convrot_int8_needs_spill(4128, 32768, in_code)


def test_convrot_lds_bound_accounts_for_row_dtype(hip):
    max_fp32 = hip._C.convrot_max_k(hip.DTYPE_TO_CODE[torch.float32])
    max_fp16 = hip._C.convrot_max_k(hip.DTYPE_TO_CODE[torch.float16])
    max_bf16 = hip._C.convrot_max_k(hip.DTYPE_TO_CODE[torch.bfloat16])

    assert 0 < max_fp32 < max_fp16
    assert max_fp16 == max_bf16
    assert hip._C.convrot_max_k(99) == 0


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


# ---------------------------------------------------------------------------
# Neighborhood attention
# ---------------------------------------------------------------------------


def _na_window(i, kernel, length, causal):
    if causal:
        return max(0, i - kernel + 1), i + 1
    kernel = min(kernel, length)
    s = min(max(i - kernel // 2, 0), length - kernel)
    return s, s + kernel


def _ref_na3d(q, k, v, kernel_size, is_causal, scale):
    """Per-query loop over the NATTEN window, accumulated and returned in fp32.

    Upstream's copy of this reference rounds each window result back to the input
    dtype. Keeping it in fp32 costs nothing and stops a fifth of the bf16 tolerance
    budget going on a rounding the reference never had to do.
    """
    b, t, h, w, nh, hd = q.shape
    if scale is None:
        scale = hd ** -0.5
    out = torch.empty(q.shape, device=q.device, dtype=torch.float32)
    for ti in range(t):
        t0, t1 = _na_window(ti, kernel_size[0], t, is_causal[0])
        for hi in range(h):
            h0, h1 = _na_window(hi, kernel_size[1], h, is_causal[1])
            for wi in range(w):
                w0, w1 = _na_window(wi, kernel_size[2], w, is_causal[2])
                kk = k[:, t0:t1, h0:h1, w0:w1].reshape(b, -1, nh, hd)
                vv = v[:, t0:t1, h0:h1, w0:w1].reshape(b, -1, nh, hd)
                s = torch.einsum("bnd,bknd->bnk", q[:, ti, hi, wi].float(), kk.float()) * scale
                a = torch.softmax(s, dim=-1)
                out[:, ti, hi, wi] = torch.einsum("bnk,bknd->bnd", a, vv.float())
    return out


NA_CASES = [
    ((1, 5, 9, 12, 2, 64), (3, 7, 7), (False, False, False)),
    ((1, 12, 13, 15, 2, 64), (11, 11, 11), (False, False, False)),
    ((2, 6, 8, 8, 4, 64), (5, 5, 5), (True, False, False)),
    ((1, 3, 6, 6, 2, 64), (5, 5, 5), (True, False, False)),    # kernel > dims, causal T
    ((1, 2, 5, 5, 2, 64), (3, 7, 7), (False, False, False)),   # kernel > dims -> clamp
    ((1, 1, 16, 16, 2, 64), (1, 5, 5), (False, False, False)),  # single frame (na2d shape)
    ((1, 4, 7, 40, 2, 32), (3, 5, 5), (False, False, False)),
    ((1, 2, 3, 65, 2, 64), (3, 3, 5), (False, False, False)),  # ragged tail query block
    ((1, 2, 3, 96, 1, 64), (1, 3, 33), (False, False, False)),  # window wider than a key tile
    ((1, 3, 4, 34, 2, 16), (3, 3, 5), (False, False, False)),
    ((1, 3, 4, 34, 2, 48), (3, 3, 5), (False, False, False)),  # head_dim not a power of 2
    ((1, 4, 6, 20, 2, 64), (3, 5, 7), (False, False, True)),   # causal W
    ((1, 4, 6, 20, 2, 64), (3, 5, 7), (False, True, False)),   # causal H
    ((1, 4, 6, 20, 2, 64), (3, 5, 7), (True, True, True)),
    ((2, 3, 4, 18, 3, 32), (3, 3, 5), (False, False, False)),  # batch and heads > 1
]


@pytest.mark.parametrize(("shape", "kernel", "causal"), NA_CASES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scale", [None, 1.0])
@needs_wmma
def test_na3d_matches_the_windowed_reference(hip, shape, kernel, causal, dtype, scale):
    torch.manual_seed(0)
    q = torch.randn(shape, device=DEV, dtype=dtype)
    k = torch.randn(shape, device=DEV, dtype=dtype)
    v = torch.randn(shape, device=DEV, dtype=dtype)

    out = hip.na3d(q, k, v, list(kernel), list(causal), scale)
    ref = _ref_na3d(q, k, v, kernel, causal, scale)

    assert out.shape == q.shape
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)


@needs_wmma
def test_na3d_reads_a_non_contiguous_input(hip):
    """The kernel indexes off bare pointers with contiguous strides, so a permuted
    view has to be materialised rather than read at its own strides."""
    torch.manual_seed(0)
    shape = (1, 4, 6, 20, 2, 64)
    kernel, causal = (3, 5, 5), (False, False, False)
    # A head slice, so the head stride no longer packs. All three operands get one:
    # the kernel reads each off a bare pointer, so a copy dropped from any of them
    # would leave that one read at packed strides.
    views = [
        torch.randn(1, 4, 6, 20, 4, 64, device=DEV, dtype=torch.bfloat16)[:, :, :, :, :2]
        for _ in range(3)
    ]
    assert all(x.shape == shape and not x.is_contiguous() for x in views)

    torch.testing.assert_close(
        hip.na3d(*views, list(kernel), list(causal), None).float(),
        hip.na3d(*(x.contiguous() for x in views), list(kernel), list(causal), None).float(),
        rtol=0,
        atol=0,
    )


@needs_wmma
def test_na3d_dispatches_to_hip(hip):
    """fp32 has no 16-bit matrix path and must not reach the kernel; the half types
    must, or the op silently needs triton at runtime."""
    constraints = hip._build_constraints(has_wmma=True)["na3d"]
    shape = (1, 4, 6, 20, 2, 64)
    for dtype, expected in ((torch.bfloat16, True), (torch.float16, True), (torch.float32, False)):
        x = torch.zeros(shape, device=DEV, dtype=dtype)
        kwargs = {"q": x, "k": x, "v": x, "kernel_size": [3, 5, 5], "is_causal": None}
        assert validate_function_call(constraints, kwargs).success is expected

    # head_dim is bounded by the WMMA K-step and the register-resident accumulators.
    for head_dim, expected in ((16, True), (48, True), (64, True), (80, False), (24, False)):
        x = torch.zeros(1, 2, 3, 8, 2, head_dim, device=DEV, dtype=torch.bfloat16)
        kwargs = {"q": x, "k": x, "v": x, "kernel_size": [1, 3, 3], "is_causal": None}
        assert validate_function_call(constraints, kwargs).success is expected

    # Both grid dims at their boundary, off a stride-0 view so the rejected shape
    # costs no memory. The reason is read back as well, or a shape some other rule
    # happens to decline would pass this for the wrong branch.
    base = torch.zeros(1, 1, 1, 1, 1, 64, device=DEV, dtype=torch.bfloat16)
    for fits, over in (
        ((1, 65535, 1, 1, 1, 64), (1, 65536, 1, 1, 1, 64)),  # blocks over T*H
        ((65535, 1, 1, 1, 1, 64), (65536, 1, 1, 1, 1, 64)),  # blocks over B*heads
    ):
        def call(shape):
            x = base.expand(shape)
            return validate_function_call(
                constraints,
                {"q": x, "k": x, "v": x, "kernel_size": [1, 1, 1], "is_causal": None},
            )

        assert call(fits).success
        assert call(over).failure_reason == "grid dims exceed HIP limits"


@needs_wmma
def test_na3d_launcher_rejects_a_mismatched_operand(hip):
    """_C is importable, so the size checks cannot be left to the Python layer: the
    kernel reads k and v with q's extents."""
    q = torch.zeros(1, 2, 3, 16, 2, 64, device=DEV, dtype=torch.bfloat16)
    short = torch.zeros(1, 2, 3, 8, 2, 64, device=DEV, dtype=torch.bfloat16)
    out = torch.empty_like(q)
    # Codes come from the shared table the wrapper dispatches on, so a change there
    # cannot leave this test asserting against a stale literal.
    extents = (1, 2, 3, 16, 2, 64, 1, 3, 3, 0, 0, 0, 0.125)
    stream = 0

    # Both operands are read with q's extents, so both slots are checked: a launcher
    # that validates one and forgets the other reads past the end of the other.
    for k_arg, v_arg in ((short, q), (q, short)):
        with pytest.raises(RuntimeError, match="needs at least"):
            hip._C.na3d(hip._dl(q), hip._dl(k_arg), hip._dl(v_arg), hip._dl(out),
                        *extents, hip.DTYPE_TO_CODE[torch.bfloat16], stream)
    with pytest.raises(RuntimeError, match="float16 or bfloat16"):
        hip._C.na3d(hip._dl(q.float()), hip._dl(q), hip._dl(q), hip._dl(out),
                    *extents, hip.DTYPE_TO_CODE[torch.float32], stream)
