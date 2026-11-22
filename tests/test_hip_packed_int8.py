import pytest
import torch

from comfy_kitchen.backends import hip
from comfy_kitchen.tensor import QuantizedTensor, TensorWiseINT8Layout
from .conftest import skip_unless_gfx12_wmma


@pytest.mark.parametrize('m', [1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 31, 32, 33, 64, 127, 128, 129, 256, 511, 512, 1024, 1025])
@pytest.mark.parametrize('n', [512, 1280])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize('bias', [False, True])
@pytest.mark.parametrize('scale_stride', [0, 1])
def test_packed_int8_exact(m, n, dtype, bias, scale_stride):
    skip_unless_gfx12_wmma()
    torch.manual_seed(211)
    k = 1792
    a = torch.randint(-127, 128, ((m + 127) // 128 * 128, k), dtype=torch.int8, device='cuda')
    at = a.reshape(-1, 128, k // 128, 128).transpose(1, 2).contiguous()
    a = a[:m].contiguous()
    b = torch.randint(-127, 128, (n, k), dtype=torch.int8, device='cuda')
    bt = b.reshape(n // 256, 256, k // 64, 64).transpose(1, 2).contiguous()
    sa = torch.rand(m, device='cuda') * .01
    sb = torch.rand(n if scale_stride else 1, device='cuda') * .01
    bias = torch.randn(n, device='cuda', dtype=dtype) if bias else None
    reference = torch.empty((m, n), device='cuda', dtype=dtype)
    output = torch.full_like(reference, float('nan'))
    dl, code, stream = hip._dl, hip.DTYPE_TO_CODE[dtype], hip._stream(a)
    hip._C.int8_gemm(dl(a), dl(b), dl(reference), dl(sa), dl(sb), scale_stride,
                     dl(bias) if bias is not None else None, m, n, k, code, stream)
    hip._C.int8_gemm_a_tiled128(dl(at), dl(bt), dl(output), dl(sa), dl(sb), scale_stride,
                                dl(bias) if bias is not None else None, m, n, k, code, stream, 1)
    assert torch.equal(output, reference)
    # Both kernels share the build flags. Check the rounding order independently.
    dot = (a[0].int() * b[:16].int()).sum(dim=1).float()
    expected = dot * sa[0] * sb[:16]
    if bias is not None:
        expected = expected + bias[:16].float()
    assert torch.equal(output[0, :16], expected.to(dtype))


def test_packed_linear_decompositions_do_not_untile(monkeypatch):
    skip_unless_gfx12_wmma()
    torch.manual_seed(7)
    weight = QuantizedTensor.from_float(torch.randn(512, 1792, device='cuda', dtype=torch.bfloat16),
                                       'TensorWiseINT8Layout', per_channel=True, convrot=True)
    storage = weight._qdata.clone().view(torch.uint8).flatten()
    resident = QuantizedTensor(storage.view(torch.int8).view_as(weight._qdata), 'TensorWiseINT8Layout', weight._params)
    tiled = resident.optimize_for_gemm(storage)
    assert tiled._params.tiled
    assert TensorWiseINT8Layout._fusion_operand(tiled) is None
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        x = torch.randn(9, 1792, device='cuda', dtype=dtype)
        bias = torch.randn(512, device='cuda', dtype=dtype)
        reference = torch.nn.functional.linear(x, weight, bias)

        def no_copy(*args):
            raise AssertionError('packed linear tried to untile its weight')

        with monkeypatch.context() as patch:
            patch.setattr(TensorWiseINT8Layout, 'get_plain_tensors', no_copy)
            assert torch.equal(torch.nn.functional.linear(x, tiled, bias), reference)
            assert torch.equal(torch.addmm(bias, x, tiled.t()), reference)


@pytest.mark.parametrize(("per_channel", "convrot"), [(False, False), (True, False), (True, True)])
def test_packed_embedding_reads_logical_rows(per_channel, convrot):
    skip_unless_gfx12_wmma()
    torch.manual_seed(211)
    weight = QuantizedTensor.from_float(
        torch.randn(512, 512, device="cuda", dtype=torch.bfloat16),
        "TensorWiseINT8Layout", per_channel=per_channel, convrot=convrot,
    )
    storage = weight._qdata.clone().view(torch.uint8).flatten()
    resident = QuantizedTensor(storage.view(torch.int8).view_as(weight._qdata), "TensorWiseINT8Layout", weight.params)
    tiled = resident.optimize_for_gemm(storage)
    assert tiled.params.tiled
    rows = torch.tensor([[511, 255, 0], [256, 255, 1]], device="cuda", dtype=torch.int32).t()
    for indices in (rows, rows[0, 0], rows[:0]):
        expected = TensorWiseINT8Layout.dequantize_embedding(weight._qdata, weight.params, indices)
        actual = TensorWiseINT8Layout.dequantize_embedding(tiled._qdata, tiled.params, indices)
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("shape", [(2, 3, 128), (0, 128), (256, 0)])
def test_gemm_optimization_skips_ineligible_shapes(shape):
    skip_unless_gfx12_wmma()
    weight = QuantizedTensor.from_float(
        torch.randn(shape, device="cuda", dtype=torch.bfloat16), "TensorWiseINT8Layout", scale=1.0,
    )
    storage = weight._qdata.view(torch.uint8).flatten()
    original = storage.clone()
    assert weight.optimize_for_gemm(storage) is weight
    assert torch.equal(storage, original)
