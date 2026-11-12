"""Integer-reference checks for large rowwise HIP GEMMs and their fallbacks."""

import pytest
import torch

import comfy_kitchen as ck

pytestmark = pytest.mark.skipif(
    not torch.version.hip or not ck.list_backends().get("hip", {}).get("available", False),
    reason="compiled HIP backend required",
)


@pytest.mark.parametrize(
    "m,n,k,scalar,bias,dtype",
    [
        (2048, 21504, 5376, False, False, torch.bfloat16),
        (2049, 28672, 5376, False, False, torch.bfloat16),
        (4097, 5376, 7168, False, False, torch.bfloat16),
        (2304, 5376, 14336, False, False, torch.bfloat16),
        (2047, 5376, 7168, False, False, torch.bfloat16),
        (8193, 5376, 7168, False, False, torch.bfloat16),
        (2049, 4112, 4112, False, False, torch.bfloat16),
        (2049, 5376, 7168, True, False, torch.bfloat16),
        (2049, 5376, 7168, False, True, torch.bfloat16),
        (2049, 5376, 7168, False, False, torch.float16),
        (2049, 5376, 7168, False, False, torch.float32),
    ],
)
def test_large_rowwise_gemm(m, n, k, scalar, bias, dtype):
    from comfy_kitchen.backends import hip

    generator = torch.Generator(device="cuda").manual_seed(42)
    a = torch.randint(-128, 128, (m, k), device="cuda", dtype=torch.int8, generator=generator)
    b = torch.randint(-128, 128, (n, k), device="cuda", dtype=torch.int8, generator=generator)
    sa = torch.rand(m, device="cuda", generator=generator) / 256
    sb = torch.rand(1 if scalar else n, device="cuda", generator=generator) / 256
    offset = torch.rand(n, device="cuda", generator=generator).to(dtype) if bias else None
    storage = torch.full((m * n + 16,), -123, device="cuda", dtype=dtype)
    output = storage[:m * n].view(m, n)
    code = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}[dtype]
    hip._C.int8_gemm(a, b, output, sa, sb, 0 if scalar else 1, offset, m, n, k, code, torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    # Cover tile boundaries, the final partial M tile, and the last N group.
    rows = sorted({0, 127, 128, 255, 256, 511, m - 1})
    cols = sorted({0, 127, 128, 255, 256, n - 129, n - 128, n - 1})
    integer = a[rows].cpu().long() @ b[cols].cpu().long().T
    expected = integer.float() * sa[rows].cpu()[:, None]
    expected = expected * (sb.cpu() if scalar else sb[cols].cpu())[None, :]
    if bias:
        expected = expected + offset[cols].cpu().float()[None, :]
    torch.testing.assert_close(output[rows][:, cols].cpu(), expected.to(dtype), rtol=0, atol=0)
    assert torch.all(storage[m * n:] == -123), "GEMM wrote past the output"
