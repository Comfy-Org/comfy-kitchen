"""MFMA int8_linear on wave64 CDNA: bit-identical to the eager backend."""
import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.backends import eager
from comfy_kitchen.tensor import QuantizedTensor


def _skip_unless_cdna():
    if not getattr(torch.version, "hip", None) or not torch.cuda.is_available():
        pytest.skip("requires HIP")
    from comfy_kitchen.backends import hip

    if hip._CDNA is None or not hip._is_cdna(torch.device("cuda")):
        pytest.skip("requires a CDNA device and the _cdna extension")
    return hip


@pytest.mark.parametrize("m", [1, 7, 31, 128, 129, 1000, 4100])
@pytest.mark.parametrize("n,k", [(64, 16), (1000, 1024), (1536, 1536), (4096, 768)])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("per_channel", [False, True])
def test_cdna_int8_linear_matches_eager(m, n, k, bias, per_channel):
    hip = _skip_unless_cdna()
    torch.manual_seed(m * 7 + n + k)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 3
    x[0, : k // 2] = 0  # a row whose scale comes from half the row
    if m > 2:
        x[2] = 0  # an all-zero row takes the clamped minimum scale
    w = torch.randint(-127, 128, (n, k), dtype=torch.int8, device="cuda")
    ws = torch.rand(n if per_channel else 1, device="cuda") * 0.01
    b = torch.randn(n, device="cuda", dtype=torch.bfloat16) if bias else None
    got = hip.int8_linear(x, w, ws, b)
    want = eager.int8_linear(x, w, ws, b)
    assert got.dtype == want.dtype == torch.bfloat16
    assert torch.equal(got, want)


@pytest.mark.parametrize("groupsize", [64, 256])
def test_cdna_int8_linear_convrot_and_residual_match_eager(groupsize):
    hip = _skip_unless_cdna()
    torch.manual_seed(groupsize)
    x = torch.randn(2, 333, 1536, device="cuda", dtype=torch.bfloat16)
    w = torch.randint(-127, 128, (1024, 1536), dtype=torch.int8, device="cuda")
    ws = torch.rand(1024, device="cuda") * 0.01
    b = torch.randn(1024, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(2, 333, 1024, device="cuda", dtype=torch.bfloat16)
    rscale = torch.randn(1024, device="cuda", dtype=torch.bfloat16)
    kwargs = {"convrot": True, "convrot_groupsize": groupsize, "residual": residual, "residual_scale": rscale}
    got = hip.int8_linear(x, w, ws, b, **kwargs)
    want = eager.int8_linear(x, w, ws, b, **kwargs)
    assert got.shape == (2, 333, 1024)
    assert torch.equal(got, want)


@pytest.mark.parametrize("input_act", ["gelu_tanh", "swiglu", "rms_norm"])
def test_cdna_int8_linear_input_act_matches_eager(input_act):
    hip = _skip_unless_cdna()
    torch.manual_seed(5)
    width = 2048 if input_act == "swiglu" else 1024
    x = torch.randn(3, 50, width, device="cuda", dtype=torch.bfloat16)
    w = torch.randint(-127, 128, (768, 1024), dtype=torch.int8, device="cuda")
    ws = torch.rand(768, device="cuda") * 0.01
    kwargs = {"convrot": True, "input_act": input_act}
    if input_act == "rms_norm":
        kwargs.update(input_act_weight=torch.randn(1024, device="cuda", dtype=torch.bfloat16), input_act_eps=1e-6)
    assert torch.equal(hip.int8_linear(x, w, ws, **kwargs), eager.int8_linear(x, w, ws, **kwargs))


def test_cdna_int8_linear_noncontiguous_input_matches_eager():
    hip = _skip_unless_cdna()
    torch.manual_seed(9)
    x = torch.randn(1536, 300, device="cuda", dtype=torch.bfloat16).t()  # [300, 1536], column-major
    w = torch.randint(-127, 128, (512, 1536), dtype=torch.int8, device="cuda")
    ws = torch.rand(512, device="cuda") * 0.01
    for convrot in (False, True):
        assert torch.equal(hip.int8_linear(x, w, ws, convrot=convrot), eager.int8_linear(x, w, ws, convrot=convrot))


def test_cdna_int8_linear_nonfinite_rows_match_eager():
    """A NaN or inf in a row makes eager's row scale non-finite; the output row must match, NaNs included."""
    hip = _skip_unless_cdna()
    torch.manual_seed(13)
    x = torch.randn(6, 256, device="cuda", dtype=torch.bfloat16)
    x[1, 7] = float("nan")
    x[3, 0] = float("inf")
    x[4, 9] = float("-inf")
    w = torch.randint(-127, 128, (64, 256), dtype=torch.int8, device="cuda")
    ws = torch.rand(64, device="cuda") * 0.01
    b = torch.randn(64, device="cuda", dtype=torch.bfloat16)
    got, want = hip.int8_linear(x, w, ws, b), eager.int8_linear(x, w, ws, b)
    assert torch.equal(got.isnan(), want.isnan())
    assert torch.equal(got.nan_to_num(), want.nan_to_num())


def test_cdna_int8_linear_runs_on_the_operands_device():
    """With another device current, the kernel must still launch on the operands' device."""
    hip = _skip_unless_cdna()
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two devices")
    torch.manual_seed(17)
    x = torch.randn(40, 512, device="cuda:1", dtype=torch.bfloat16)
    w = torch.randint(-127, 128, (256, 512), dtype=torch.int8, device="cuda:1")
    ws = torch.rand(256, device="cuda:1") * 0.01
    with torch.cuda.device(0):
        got = hip.int8_linear(x, w, ws, convrot=True)
    assert got.device == x.device
    assert torch.equal(got, eager.int8_linear(x, w, ws, convrot=True))


def test_cdna_int8_linear_unaligned_weight_matches_eager():
    hip = _skip_unless_cdna()
    torch.manual_seed(3)
    x = torch.randn(65, 256, device="cuda", dtype=torch.bfloat16)
    storage = torch.randint(-127, 128, (128 * 256 + 8,), dtype=torch.int8, device="cuda")
    w = storage[8:].view(128, 256)  # 8 bytes off a 16-byte boundary
    ws = torch.rand(128, device="cuda") * 0.01
    assert torch.equal(hip.int8_linear(x, w, ws), eager.int8_linear(x, w, ws))


def test_cdna_int8_linear_empty_input():
    hip = _skip_unless_cdna()
    x = torch.empty(0, 64, device="cuda", dtype=torch.bfloat16)
    w = torch.zeros(32, 64, dtype=torch.int8, device="cuda")
    assert hip.int8_linear(x, w, torch.ones(1, device="cuda")).shape == (0, 32)


def test_quantized_linear_dispatches_to_cdna(monkeypatch):
    """F.linear on an INT8 QuantizedTensor reaches the MFMA kernel and matches eager."""
    hip = _skip_unless_cdna()
    if "int8_linear" not in ck.list_backends()["hip"]["capabilities"]:
        pytest.skip("int8_linear is not advertised (mixed device set)")
    torch.manual_seed(11)
    weight = QuantizedTensor.from_float(torch.randn(512, 1024, device="cuda", dtype=torch.bfloat16),
                                        "TensorWiseINT8Layout", per_channel=True, convrot=True)
    x = torch.randn(3, 77, 1024, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(512, device="cuda", dtype=torch.bfloat16)

    calls = []
    kernel = hip._CDNA.int8_linear_mfma
    monkeypatch.setattr(hip._CDNA, "int8_linear_mfma", lambda *a: calls.append(1) or kernel(*a), raising=False)
    got = torch.nn.functional.linear(x, weight, bias)
    assert calls
    with ck.use_backend("eager"):
        want = torch.nn.functional.linear(x, weight, bias)
    assert torch.equal(got, want)
