# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""INT4 VALU GEMM must agree with the INT8 one over the same numbers.

v_dot8_i32_i4 computes eight signed 4x4-bit products per instruction against
v_dot4_i32_i8's four, at the same instruction rate on gfx10. That makes the int4
tile worth roughly twice the int8 one, but only if the packing is right: the two
kernels have to produce the same result when handed the same values, so every
shape below is run through both with the int8 operands packed two per byte
exactly as ops/convrot_w4a4.hip's quantized weights are.

The reference is fp64 over the unpacked values; the tolerance is fp16 output
epsilon rather than exact, because that is what the epilogue can represent.
"""
import pytest
import torch

pytest.importorskip("comfy_kitchen")

from comfy_kitchen.backends import hip as hip_backend

_C = getattr(hip_backend, "_C", None)

pytestmark = pytest.mark.skipif(
    not hip_backend.is_available() or not hip_backend.rdna2_is_available(),
    reason="needs the HIP extension on an RDNA2 device (gfx103x)",
)

# (M, K, N). The big three are the shapes the Anima transformer block issues; the
# small ones are there to catch tile-boundary and single-wave mistakes, which is
# where a wrong chunk offset shows up first.
SHAPES = [
    (128, 128, 128),
    (256, 512, 256),
    (512, 1024, 2048),
    (77, 2048, 1280),
    (1024, 1024, 1024),
    (1024, 2048, 2048),
    (4096, 2048, 2048),
    (2048, 6144, 1024),
    (9216, 2048, 2048),
    (9216, 2048, 8192),
    (9216, 8192, 2048),
]


def pack_int4(x: torch.Tensor) -> torch.Tensor:
    """[N, K] int8 in [-8, 7] -> [N, K/2] int8, low nibble = even k.

    .to(torch.int8) rather than a cast, so the high nibble's bit 7 cannot be
    read back as a sign bit.
    """
    lo = x[:, 0::2] & 0xF
    hi = x[:, 1::2] & 0xF
    return (lo | (hi << 4)).to(torch.int8)


@pytest.mark.parametrize("m,k,n", SHAPES)
def test_int4_matches_int8_and_fp64(m, k, n):
    dev = torch.device("cuda:0")
    code = hip_backend.DTYPE_TO_CODE[torch.float16]
    g = torch.Generator(device=dev).manual_seed(m * 7 + k * 13 + n)

    x8 = torch.randint(-8, 8, (m, k), device=dev, dtype=torch.int8, generator=g)
    w8 = torch.randint(-8, 8, (n, k), device=dev, dtype=torch.int8, generator=g)
    x4 = pack_int4(x8).contiguous()
    w4 = pack_int4(w8).contiguous()
    sa = torch.full((m,), 1.0, device=dev, dtype=torch.float32)
    sw = torch.full((n,), 1.0, device=dev, dtype=torch.float32)

    out4 = torch.empty((m, n), device=dev, dtype=torch.float16)
    out8 = torch.empty((m, n), device=dev, dtype=torch.float16)
    stream = hip_backend._stream(x8)

    assert hasattr(_C, "int4_gemm"), "int4_gemm is not bound in this build"
    _C.int4_gemm(hip_backend._dl(x4), hip_backend._dl(w4), hip_backend._dl(out4),
                 hip_backend._dl(sa), hip_backend._dl(sw), 1, None, m, n, k, code, stream)
    _C.int8_gemm(hip_backend._dl(x8), hip_backend._dl(w8), hip_backend._dl(out8),
                 hip_backend._dl(sa), hip_backend._dl(sw), 1, None, m, n, k, code, stream)
    torch.cuda.synchronize()

    ref = x8.double() @ w8.double().t()
    tol = 5e-3 * (ref.abs().max().item() + 1)
    for name, out in (("int4", out4), ("int8", out8)):
        bad = (out.double() - ref).abs().gt(tol).sum().item()
        assert bad == 0, f"{name} gemm wrong on {bad}/{m * n} entries at {m}x{k}x{n}"


def test_int4_rejects_k_that_is_not_a_multiple_of_32():
    """The tile reads 16 bytes at a time, which is 32 k-values."""
    dev = torch.device("cuda:0")
    x4 = torch.zeros((8, 16), device=dev, dtype=torch.int8)
    sa = torch.ones(8, device=dev, dtype=torch.float32)
    sw = torch.ones(8, device=dev, dtype=torch.float32)
    out = torch.empty((8, 8), device=dev, dtype=torch.float16)
    with pytest.raises(RuntimeError, match=r"K must be a multiple of 32, got 16"):
        _C.int4_gemm(hip_backend._dl(x4), hip_backend._dl(x4), hip_backend._dl(out),
                     hip_backend._dl(sa), hip_backend._dl(sw), 1, None,
                     8, 8, 16, hip_backend.DTYPE_TO_CODE[torch.float16],
                     hip_backend._stream(x4))
