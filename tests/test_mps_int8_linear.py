"""Weight-only INT8 linear on Apple GPUs (mps backend)."""
import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.backends._activations import apply_input_act
from comfy_kitchen.tensor.int8_utils import _build_hadamard

pytestmark = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and ck.list_backends().get("mps", {}).get("available")),
    reason="mps backend not available",
)

G = 256


def _make(n, k, rowscale, convrot):
    torch.manual_seed(0)
    w = torch.randn(n, k) * 0.02
    if convrot:
        h = _build_hadamard(G)
        w = (w.reshape(n, k // G, G) @ h.T).reshape(n, k)
    s = (w.abs().amax(1, keepdim=True) / 127) if rowscale else (w.abs().max() / 127).reshape(1)
    q = torch.round(w / s).clamp(-127, 127).to(torch.int8)
    wd = q.float() * s.reshape(-1, 1)
    if convrot:
        wd = (wd.reshape(n, k // G, G) @ _build_hadamard(G)).reshape(n, k)
    return q, s, wd


def _rel(a, b):
    return ((a - b).norm() / b.norm()).item()


@pytest.mark.parametrize("dtype,tol", [(torch.float32, 2e-4), (torch.float16, 1e-3), (torch.bfloat16, 6e-3)])
@pytest.mark.parametrize("convrot", [True, False])
@pytest.mark.parametrize("rowscale", [True, False])
@pytest.mark.parametrize("act", [None, "gelu_tanh", "swiglu", "rms_norm"])
def test_matches_float_oracle(dtype, tol, convrot, rowscale, act):
    n, k, m = 384, 1024, 37
    q, s, wd = _make(n, k, rowscale, convrot)
    x = torch.randn(2, m, k * 2 if act == "swiglu" else k)
    b = torch.randn(n)
    aw = torch.randn(k) if act == "rms_norm" else None
    res, rs = torch.randn(2, m, n), torch.rand(n)
    exact = torch.addcmul(res, torch.nn.functional.linear(apply_input_act(x, act, aw, 1e-6), wd, b), rs)
    d = "mps"
    out = ck.int8_linear(
        x.to(d, dtype), q.to(d), s.to(d), b.to(d, dtype), dtype,
        convrot=convrot, convrot_groupsize=G, input_act=act,
        input_act_weight=None if aw is None else aw.to(d), input_act_eps=1e-6,
        residual=res.to(d, dtype), residual_scale=rs.to(d, dtype),
    )
    assert out.dtype == dtype and out.device.type == "mps"
    assert _rel(out.float().cpu(), exact) < tol


def test_dispatch_prefers_mps_on_mps_and_eager_on_cpu():
    q, s, _ = _make(256, 512, True, True)
    x = torch.randn(4, 512)
    assert ck.registry.get_implementation(
        "int8_linear", kwargs={"x": x.to("mps"), "weight": q.to("mps"), "weight_scale": s.to("mps")}
    ).__module__.endswith("backends.mps.linear")
    assert ".eager." in ck.registry.get_implementation(
        "int8_linear", kwargs={"x": x, "weight": q, "weight_scale": s}
    ).__module__


def test_chunked_decode_matches_unchunked(monkeypatch):
    from comfy_kitchen.backends.mps import linear
    q, s, _ = _make(512, 1024, True, True)
    x = torch.randn(9, 1024, device="mps", dtype=torch.float16)
    a = linear.int8_linear(x, q.to("mps"), s.to("mps"), out_dtype=torch.float16, convrot=True)
    monkeypatch.setattr(linear, "_MAX_DECODE_BYTES", 2 * 1024 * 100)
    b = linear.int8_linear(x, q.to("mps"), s.to("mps"), out_dtype=torch.float16, convrot=True)
    assert torch.equal(a.cpu(), b.cpu())


def test_transposed_decode_matches_plain(monkeypatch):
    from comfy_kitchen.backends.mps import linear
    q, s, _ = _make(384, 1024, True, True)
    x = torch.randn(9, 1024, device="mps", dtype=torch.float16)
    a = linear.int8_linear(x, q.to("mps"), s.to("mps"), out_dtype=torch.float16, convrot=True)
    monkeypatch.setattr(linear, "_TRANSPOSED_DECODE_MIN_K", 1)
    b = linear.int8_linear(x, q.to("mps"), s.to("mps"), out_dtype=torch.float16, convrot=True)
    assert torch.allclose(a.cpu().float(), b.cpu().float(), rtol=2e-3, atol=2e-3)
    monkeypatch.setattr(linear, "_MAX_DECODE_BYTES", 2 * 1024 * 100)
    c = linear.int8_linear(x, q.to("mps"), s.to("mps"), out_dtype=torch.float16, convrot=True)
    assert torch.equal(b.cpu(), c.cpu())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_large_activations_do_not_overflow_fp16(dtype):
    # Text-encoder states reach ~1.6e4; one GEMM in fp16 would overflow 65504.
    q, s, wd = _make(384, 1024, True, True)
    x = torch.randn(5, 1024) * 2.0e4
    x[2] *= 1e-6  # tiny rows must keep their precision next to huge ones
    exact = torch.nn.functional.linear(x.to(dtype).float(), wd)
    out = ck.int8_linear(x.to("mps", dtype), q.to("mps"), s.to("mps"), None, dtype, convrot=True)
    o = out.float().cpu()
    assert torch.isfinite(o).all()
    tol = 6e-3 if dtype == torch.bfloat16 else 1e-3
    for r in range(5):
        assert _rel(o[r], exact[r]) < tol


# --- Range safety and eager semantics (review 2026-10-02, findings 1, 2, 4, 6, 7) -----------

D = "mps"


def _one_hot_weight(n, k, value, scale):
    w = torch.zeros(n, k, dtype=torch.int8)
    w[:, 0] = value
    return w, torch.tensor([scale])


def _oracle(x, q, s, bias=None, convrot=False):
    """float64 CPU linear over the exact dequantized weight (rotated basis when convrot)."""
    x = x.cpu().double()
    w = q.double() * s.double().reshape(-1, 1)
    if convrot:
        h = _build_hadamard(G).double()
        x = (x.reshape(-1, x.shape[-1] // G, G) @ h).reshape(x.shape)
    out = x @ w.T
    return out if bias is None else out + bias.cpu().double().reshape(-1)


def test_f1_fp16_input_is_range_scaled():
    # fp16 inputs used to skip scaling: 8 x 1e4 overflowed fp16 inside the GEMM.
    q = torch.ones(1, 8, dtype=torch.int8)
    x = torch.full((1, 8), 10000.0)
    out = ck.int8_linear(x.to(D, torch.float16), q.to(D), torch.ones(1, device=D), None, torch.bfloat16)
    assert out.cpu().item() == torch.tensor(80000.0).bfloat16().item()


def test_f1_rotation_growth_is_range_scaled():
    # Rotation grows the max by sqrt(256) = 16: 1e4 * sign(H[:, 0]) rotates to 160000 in one lane.
    h = _build_hadamard(G)
    x = (10000 * torch.sign(h[:, 0])).reshape(1, G)
    q, s = _one_hot_weight(1, G, 1, 0.001)
    out = ck.int8_linear(x.to(D, torch.float16), q.to(D), s.to(D), None, torch.bfloat16, convrot=True)
    assert out.cpu().item() == pytest.approx(_oracle(x, q, s, convrot=True).item(), rel=4e-3)


def test_f1_dot_product_is_bounded():
    # Inputs of 1 with weights of 100 used to keep row_scale 1 and sum to 102400 in fp16.
    q = torch.full((1, 1024), 100, dtype=torch.int8)
    out = ck.int8_linear(torch.ones(1, 1024, device=D, dtype=torch.bfloat16), q.to(D),
                         torch.ones(1, device=D), None, torch.bfloat16)
    assert out.cpu().item() == 102400.0


@pytest.mark.parametrize("x0,value,scale", [(1e5, 1, 1e-3), (2.0 ** -26, 100, 1.0)])
def test_f2_fp16_output_restores_range_exactly(x0, value, scale):
    # The restore scale used to be cast to fp16 first: 2^17 overflowed, 2^-25 underflowed.
    x = torch.zeros(1, 8)
    x[0, 0] = x0
    x = x.bfloat16()
    q, s = _one_hot_weight(1, 8, value, scale)
    out = ck.int8_linear(x.to(D), q.to(D), s.to(D), None, torch.float16)
    want = _oracle(x, q, s).half()
    assert torch.isfinite(out).all() and out.item() != 0
    assert out.cpu().item() == pytest.approx(want.item(), rel=1e-3)


@pytest.mark.parametrize("dtype,x0", [(torch.bfloat16, 2e38), (torch.float32, 3.4e38), (torch.float32, -3.0e38)])
@pytest.mark.parametrize("compute", [None, "float16"])
def test_f4_near_max_inputs_stay_finite(dtype, x0, compute, monkeypatch):
    # A float32 scale of 2^128 used to overflow: 0 * inf = NaN.
    if compute:
        monkeypatch.setenv("COMFY_KITCHEN_MPS_COMPUTE_DTYPE", compute)
    x = torch.zeros(2, 256)
    x[0, :3] = x0
    x[1] = torch.randn(256)
    x = x.to(dtype)
    q, s = _one_hot_weight(4, 256, 1, 1e-3)
    q[:, 1] = -2
    out = ck.int8_linear(x.to(D), q.to(D), s.to(D), None, dtype, convrot=True)
    want = _oracle(x, q, s, convrot=True)
    got = out.cpu().double()
    assert torch.isfinite(got).all()
    for r in range(2):
        assert ((got[r] - want[r]).norm() / want[r].norm()).item() < 1e-2


def test_f6_large_weight_scale_decodes_without_overflow():
    # fp16 decode of int8 * 1e5 used to overflow even with fp32 compute.
    q = torch.ones(3, 8, dtype=torch.int8)
    x = torch.full((2, 8), 0.001 / 8)
    s = torch.tensor([1e5, 3e7, 1e-30])
    for dtype in (torch.float32, torch.bfloat16):
        out = ck.int8_linear(x.to(D, dtype), q.to(D), s.to(D), None, torch.float32)
        assert torch.allclose(out.cpu().double(), _oracle(x.to(dtype), q, s), rtol=1e-3)


@pytest.mark.parametrize("x_shape,bias_shape,out_shape", [
    ((64,), (1, 32), (32,)),
    ((3, 64), (32, 1), (3, 32)),
    ((2, 3, 64), (1,), (2, 3, 32)),
    ((64,), (32,), (32,)),
])
def test_f7_bias_is_flattened_like_eager(x_shape, bias_shape, out_shape):
    from comfy_kitchen.backends.eager.quantization import int8_linear as eager_int8_linear

    gen = torch.Generator().manual_seed(7)
    x = torch.randn(x_shape, generator=gen).bfloat16()
    q = torch.randint(-127, 128, (32, 64), generator=gen, dtype=torch.int8)
    s = torch.full((32,), 0.01)
    b = torch.randn(bias_shape, generator=gen).bfloat16()
    assert tuple(eager_int8_linear(x, q, s, b, torch.bfloat16).shape) == out_shape
    out = ck.int8_linear(x.to(D), q.to(D), s.to(D), b.to(D), torch.bfloat16)
    assert tuple(out.shape) == out_shape
    want = _oracle(x.reshape(-1, 64), q, s, b.reshape(-1)).reshape(out_shape)
    assert torch.allclose(out.cpu().double(), want, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("x,scale", [
    ([[1.0]], 255.75),                     # re-review finding 4: eager rounds 255.75 to 256 first
    ([[1.0] * 255 + [0.75]], 1.0),         # same cancellation across a 256-wide dot product
])
def test_bias_order_matches_eager(x, scale):
    """Eager INT8 rounds the scaled result to out_dtype, then adds bias in out_dtype. Both
    cases cancel against bias -256 and return exactly what eager returns (0, not -0.25)."""
    from comfy_kitchen.backends.eager.quantization import int8_linear as eager_int8_linear

    x = torch.tensor(x)
    q = torch.ones(1, x.shape[1], dtype=torch.int8)
    s = torch.tensor([scale])
    bias = torch.tensor([-256.0]).bfloat16()
    want = eager_int8_linear(x, q, s, bias, torch.bfloat16)
    got = ck.int8_linear(x.to(D), q.to(D), s.to(D), bias.to(D), torch.bfloat16)
    assert torch.equal(got.cpu(), want) and want.item() == 0.0


def test_restore_overflow_before_cancelling_bias_matches_eager():
    """Re-review finding 3: x = 2e38, weight 2, bias -x. The exact answer (~1.99e38) is
    finite, but x * w overflows fp32 before the bias can cancel it. Eager INT8 overflows the
    same way (its fp32 rescale gives inf, then adds bias), so the backend keeps eager's inf."""
    from comfy_kitchen.backends.eager.quantization import int8_linear as eager_int8_linear

    x = torch.tensor([[2e38]]).bfloat16()
    q, s = torch.tensor([[2]], dtype=torch.int8), torch.tensor([1.0])
    want = eager_int8_linear(x, q, s, -x.reshape(1), torch.bfloat16)
    got = ck.int8_linear(x.to(D), q.to(D), s.to(D), -x.reshape(1).to(D), torch.bfloat16)
    assert torch.isinf(want).all()
    assert torch.equal(got.cpu(), want)


def test_residual_cancels_exactly_like_eager_with_per_row_scales():
    """Integer activations with a 127 in every row make eager's int8 activation quantization
    exact, and power-of-two per-row weight scales keep every product exact, so the fp16 GEMM
    and eager agree to the bit. The residual is chosen to cancel the output exactly, which
    only works if bias rounding and residual order match eager's."""
    from comfy_kitchen.backends.eager.quantization import int8_linear as eager_int8_linear

    gen = torch.Generator().manual_seed(12)
    x = torch.randint(-9, 10, (5, 8), generator=gen).float()
    x[:, 0] = 127
    q = torch.randint(-3, 4, (6, 8), generator=gen, dtype=torch.int8)
    s = 2.0 ** torch.randint(-6, 6, (6,), generator=gen).float()
    bias = (torch.randn(6, generator=gen) * 50).bfloat16()
    rs = (2.0 ** torch.randint(-2, 3, (6,), generator=gen).float()).bfloat16()
    lin = eager_int8_linear(x, q, s, bias, torch.bfloat16)
    res = -(lin.float() * rs.float()).bfloat16()
    want = eager_int8_linear(x, q, s, bias, torch.bfloat16, residual=res, residual_scale=rs)
    assert (want == 0).all()
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        args = (x.to(D, dtype), q.to(D), s.to(D), bias.to(D), torch.bfloat16)
        assert torch.equal(ck.int8_linear(*args).cpu(), lin)
        got = ck.int8_linear(*args, residual=res.to(D), residual_scale=rs.to(D))
        assert torch.equal(got.cpu(), want)


@pytest.mark.parametrize("convrot", [False, True])
def test_wider_than_fp16_range_computes_in_fp32(convrot):
    """K > 65504 could overflow even normalized fp16 dot products, so it falls back to fp32
    compute; 1e-5 relative error is beyond reach of an fp16 GEMM (~5e-4)."""
    from comfy_kitchen.backends.mps import linear

    k = 65536 + 1024
    assert linear._compute_dtype(torch.empty(0, dtype=torch.bfloat16), k) == torch.float32
    gen = torch.Generator().manual_seed(13)
    q = torch.randint(-127, 128, (8, k), generator=gen, dtype=torch.int8)
    s = torch.rand(8, generator=gen) * 1e-2
    x = (torch.randn(2, k, generator=gen) * 100).bfloat16()
    out = ck.int8_linear(x.to(D), q.to(D), s.to(D), None, torch.float32, convrot=convrot)
    assert _rel(out.cpu().double(), _oracle(x, q, s, convrot=convrot)) < 1e-5


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_fp16_output_from_every_input_dtype(dtype):
    q, s, wd = _make(96, 512, True, True)
    x = torch.randn(7, 512) * 300
    b = torch.randn(96)
    out = ck.int8_linear(x.to(D, dtype), q.to(D), s.to(D), b.to(D, torch.float16), torch.float16, convrot=True)
    exact = torch.nn.functional.linear(x.to(dtype).float(), wd, b.half().float())
    assert out.dtype == torch.float16
    assert _rel(out.float().cpu(), exact) < 1e-3


def test_fp16_output_overflows_only_when_the_result_does():
    q = torch.full((2, 16), 127, dtype=torch.int8)
    x = torch.full((1, 16), 64.0)
    out = ck.int8_linear(x.to(D, torch.bfloat16), q.to(D), torch.tensor([1.0, 0.25], device=D),
                         None, torch.float16)
    assert out.cpu().tolist() == [[float("inf"), 32512.0]]  # 130048 and 32512


def test_zero_rows_and_zero_weights():
    q, s, _ = _make(64, 256, True, True)
    x = torch.randn(4, 256)
    x[1] = 0
    b = torch.randn(64)
    out = ck.int8_linear(x.to(D, torch.bfloat16), q.to(D), s.to(D), b.to(D, torch.bfloat16),
                         torch.bfloat16, convrot=True).cpu()
    assert torch.equal(out[1], b.bfloat16())
    zero = ck.int8_linear(x.to(D, torch.bfloat16), torch.zeros_like(q).to(D), torch.zeros(1, device=D),
                          None, torch.bfloat16, convrot=True).cpu()
    assert torch.equal(zero, torch.zeros_like(zero))


@pytest.mark.parametrize("x_shape", [(0, 256), (2, 0, 256)])
def test_empty_batch(x_shape):
    q, s, _ = _make(32, 256, True, False)
    x = torch.randn(x_shape, device=D, dtype=torch.bfloat16)
    res = torch.randn(*x_shape[:-1], 32, device=D, dtype=torch.bfloat16)
    out = ck.int8_linear(x, q.to(D), s.to(D), torch.randn(32, device=D), torch.bfloat16,
                         residual=res, residual_scale=torch.ones(32, device=D))
    assert out.shape == (*x_shape[:-1], 32) and out.dtype == torch.bfloat16


def test_1d_input():
    q, s, _ = _make(48, 256, True, True)
    x = torch.randn(256)
    out = ck.int8_linear(x.to(D), q.to(D), s.to(D), None, torch.float32, convrot=True)
    assert out.shape == (48,)
    assert torch.equal(out, ck.int8_linear(x[None].to(D), q.to(D), s.to(D), None, torch.float32, convrot=True)[0])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_subnormal_rows_keep_their_value(dtype):
    # Apple GPUs flush fp32 subnormals in arithmetic; exponents are applied on raw bits.
    gen = torch.Generator().manual_seed(3)
    q = torch.randint(-127, 128, (16, 256), generator=gen, dtype=torch.int8)
    s = torch.full((16,), 0.5)
    x = torch.randn(3, 256, generator=gen)
    x[0] *= 1e-39  # subnormal inputs (bf16 and fp32)
    x[1] *= 1e-41 if dtype == torch.float32 else 1e-39 * 2.0 ** -6  # deeper; bf16 bottoms out at 9e-41
    x = x.to(dtype)
    for convrot in (False, True):
        out = ck.int8_linear(x.to(D), q.to(D), s.to(D), None, dtype, convrot=convrot).cpu().double()
        want = _oracle(x, q, s, convrot=convrot)
        for r in range(3):
            assert (out[r] != 0).any()
            tol = 1e-2 if r < 1 or dtype == torch.bfloat16 else 3e-2  # few mantissa bits at 1e-42
            assert ((out[r] - want[r]).norm() / want[r].norm()).item() < tol


def test_non_finite_inputs_propagate():
    q, s, _ = _make(8, 256, True, True)
    x = torch.randn(3, 256)
    x[0, 5] = float("inf")
    x[1, 7] = float("nan")
    out = ck.int8_linear(x.to(D, torch.bfloat16), q.to(D), s.to(D), None, torch.bfloat16, convrot=True).cpu()
    assert not torch.isfinite(out[0]).all() and torch.isnan(out[1]).all() and torch.isfinite(out[2]).all()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float16])
def test_strided_and_offset_operands_match_contiguous(dtype):
    gen = torch.Generator().manual_seed(11)
    q_big = torch.randint(-127, 128, (70, 260), generator=gen, dtype=torch.int8).to(D)
    s_big = (torch.rand(70, generator=gen) * 0.01).to(D)
    x_big = (torch.randn(2, 9, 300, generator=gen) * 50).to(D, dtype)
    b_big = torch.randn(80, generator=gen).to(D, dtype)
    res_big = torch.randn(2, 9, 100, generator=gen).to(D, dtype)
    rs_big = torch.rand(70, generator=gen).to(D, dtype)
    q, s = q_big[3:67, 2:258], s_big[3:67]       # offset rows and columns: non-contiguous
    b, rs = b_big[5:69], rs_big[1:65]            # storage offsets
    res = res_big[..., 7:71]                     # strided rows
    for x in (x_big[..., 3:259], x_big[..., 4:260], x_big[..., :256].transpose(0, 1)):
        args = {"out_dtype": dtype, "convrot": True, "residual_scale": rs,
                "residual": res if x.shape[0] == 2 else res.transpose(0, 1)}
        got = ck.int8_linear(x, q, s, b, **args)
        want = ck.int8_linear(x.contiguous(), q.contiguous(), s.contiguous(), b.contiguous(),
                              **{**args, "residual": args["residual"].contiguous(), "residual_scale": rs.contiguous()})
        assert torch.equal(got.cpu(), want.cpu())


def test_residual_broadcast_and_scalar_scale_follow_eager():
    from comfy_kitchen.backends._activations import apply_residual

    q, s, wd = _make(32, 256, True, False)
    x = torch.randn(2, 5, 256)
    lin = torch.nn.functional.linear(x, wd).bfloat16()
    for res, rs in ((torch.randn(32), torch.rand(32)), (torch.randn(2, 5, 32), torch.tensor(0.5)),
                    (torch.randn(1, 5, 32), torch.rand(1, 1, 32))):
        want = apply_residual(lin, res.bfloat16(), rs.bfloat16()).float()
        got = ck.int8_linear(x.to(D, torch.bfloat16), q.to(D), s.to(D), None, torch.bfloat16,
                             residual=res.to(D, torch.bfloat16), residual_scale=rs.to(D)).float().cpu()
        assert got.shape == want.shape
        assert _rel(got, want) < 1e-2


@pytest.mark.parametrize("k", [4096, 260])
def test_wide_and_unaligned_groups(k):
    # Rotation groups past the fused kernel (4096) rotate in fp32 through torch; K % 4 != 0
    # takes the scalar kernels.
    gen = torch.Generator().manual_seed(5)
    q = torch.randint(-127, 128, (24, k), generator=gen, dtype=torch.int8)
    s = torch.rand(24, generator=gen) * 1e-2
    x = torch.randn(3, k, generator=gen) * 1e3
    convrot = k == 4096
    if convrot:
        h = _build_hadamard(k).double()
        want = (x.double() @ h) @ (q.double() * s.double()[:, None]).T
    else:
        want = _oracle(x, q, s)
    out = ck.int8_linear(x.to(D), q.to(D), s.to(D), None, torch.float32,
                         convrot=convrot, convrot_groupsize=k)
    assert _rel(out.cpu().double(), want) < 1e-4


@pytest.mark.parametrize("n,k,row0", [(70, 300, 0), (70, 300, 5), (64, 512, 32), (37, 258, 3)])
@pytest.mark.parametrize("cdtype", [torch.float16, torch.float32])
def test_decoded_weight_bits_plain_and_transposed(n, k, row0, cdtype):
    """Every decode kernel (plain, 4-wide, transposed tile, transposed 4-wide) writes
    (int8 * scale) / 2^ch_exp, with |w'| < 1, rounded once to the compute dtype."""
    from comfy_kitchen.backends.mps import linear

    gen = torch.Generator().manual_seed(9)
    q = torch.randint(-128, 128, (n, k), generator=gen, dtype=torch.int8)
    for s in (torch.randn(n, generator=gen) * 10.0 ** torch.randint(-30, 30, (n,), generator=gen),
              torch.tensor([3.7e-3])):
        rows = n - row0
        ch = torch.full((n,), -999, dtype=torch.int32, device=D)
        plain = linear.decode_int8(q.to(D), s.to(D), row0, rows, cdtype, False, ch).cpu()
        ch_plain = ch.cpu().clone()
        ch.fill_(-999)
        trans = linear.decode_int8(q.to(D), s.to(D), row0, rows, cdtype, True, ch).cpu()
        assert torch.equal(ch.cpu(), ch_plain)
        assert (ch_plain[:row0] == -999).all() and (ch_plain[row0:] != -999).all()
        f = ch_plain[row0:].double()
        want = torch.ldexp(q[row0:].double() * s.reshape(-1, 1).double().expand(n, 1)[row0:], -f[:, None])
        assert want.abs().max() < 1
        assert torch.equal(plain, want.to(cdtype))
        assert torch.equal(trans, plain.T)


def test_bf16_and_fp16_stores_match_torch_casts():
    """The epilogue writes bf16 as ushort with round-to-nearest-even: bit-exact with torch."""
    from comfy_kitchen.backends.mps import linear

    lib = torch.mps.compile_shader(linear.METAL_COMMON + """
    kernel void store_probe(device ushort* bf, device half* hf, device const uint* x,
                            uint i [[thread_position_in_grid]]) {
        store_b(bf, i, x[i]);
        store_b(hf, i, x[i]);
    }
    """)
    gen = torch.Generator().manual_seed(1)
    bits = torch.randint(-2**31, 2**31 - 1, (1 << 16,), generator=gen, dtype=torch.int32)
    edge = torch.tensor([0.0, -0.0, 1.0, 1.00390625, 1.01171875, 3.3895314e38, 3.4028235e38,
                         float("inf"), -float("inf"), 1e-40, -1e-45, 65504.0, 65519.0, 65520.0,
                         2.0 ** -24, 2.0 ** -25, 3 * 2.0 ** -26, 6.1e-5])
    x = torch.cat([bits.view(torch.float32), edge,
                   (torch.randn(4096, generator=gen) * 2.0 ** torch.randint(-140, 120, (4096,), generator=gen))])
    xm = x.to(D)
    bf = torch.empty(x.numel(), dtype=torch.bfloat16, device=D)
    hf = torch.empty(x.numel(), dtype=torch.float16, device=D)
    lib.store_probe(bf, hf, xm.view(torch.int32))
    finite = ~torch.isnan(x)
    want_bf = x.bfloat16()
    assert torch.equal(bf.cpu()[finite].view(torch.int16), want_bf[finite].view(torch.int16))
    assert torch.equal(bf.cpu()[finite].view(torch.int16), xm.bfloat16().cpu()[finite].view(torch.int16))
    assert torch.isnan(bf.cpu()[~finite]).all()
    normal = finite & ((x.abs() >= 2.0 ** -126) | (x == 0))  # the GPU flushes fp32 subnormal inputs
    assert torch.equal(hf.cpu()[normal].view(torch.int16), x[normal].half().view(torch.int16))


def test_ldexp_bits_rounds_like_float64():
    from comfy_kitchen.backends.mps import linear

    lib = torch.mps.compile_shader(linear.METAL_COMMON + """
    kernel void ldexp_probe(device uint* out, device const uint* x, device const int* s,
                            uint i [[thread_position_in_grid]]) {
        out[i] = ldexp_bits(x[i], s[i]);
    }
    """)
    gen = torch.Generator().manual_seed(2)
    n = 1 << 16
    x = torch.randn(n, generator=gen) * 2.0 ** torch.randint(-149, 127, (n,), generator=gen)
    s = torch.randint(-300, 300, (n,), generator=gen, dtype=torch.int32)
    s[: n // 2] = torch.randint(-40, 40, (n // 2,), generator=gen, dtype=torch.int32)
    out = torch.empty(n, dtype=torch.int32, device=D)
    lib.ldexp_probe(out, x.to(D).view(torch.int32), s.to(D))
    want = torch.ldexp(x.double(), s.double()).float()  # one rounding, ties to even
    assert torch.equal(out.cpu().view(torch.float32), want)
