# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import comfy_kitchen as ck
from tests.conftest import assert_values_close, get_capable_backends

# ---------------------------------------------------------------------------
# Reference implementation: direct per-query loop over NATTEN window semantics
# ---------------------------------------------------------------------------


def _window(i, kernel, length, causal):
    if causal:
        return max(0, i - kernel + 1), i + 1
    kernel = min(kernel, length)
    s = min(max(i - kernel // 2, 0), length - kernel)
    return s, s + kernel


def _ref_na3d(q, k, v, kernel_size, is_causal, scale):
    b, t, h, w, nh, hd = q.shape
    if scale is None:
        scale = hd ** -0.5
    out = torch.empty_like(v)
    for ti in range(t):
        t0, t1 = _window(ti, kernel_size[0], t, is_causal[0])
        for hi in range(h):
            h0, h1 = _window(hi, kernel_size[1], h, is_causal[1])
            for wi in range(w):
                w0, w1 = _window(wi, kernel_size[2], w, is_causal[2])
                kk = k[:, t0:t1, h0:h1, w0:w1].reshape(b, -1, nh, hd)
                vv = v[:, t0:t1, h0:h1, w0:w1].reshape(b, -1, nh, hd)
                qq = q[:, ti, hi, wi]  # [b, nh, hd]
                s = torch.einsum("bnd,bknd->bnk", qq.float(), kk.float()) * scale
                a = torch.softmax(s, dim=-1)
                out[:, ti, hi, wi] = torch.einsum("bnk,bknd->bnd", a, vv.float()).to(v.dtype)
    return out


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

CASES = [
    # (shape (B,T,H,W,NH,HD), kernel, is_causal)
    ((1, 5, 9, 12, 2, 64), (3, 7, 7), (False, False, False)),
    ((1, 12, 13, 15, 2, 64), (11, 11, 11), (False, False, False)),
    ((2, 6, 8, 8, 4, 64), (5, 5, 5), (True, False, False)),   # causal T
    ((1, 3, 6, 6, 2, 64), (5, 5, 5), (True, False, False)),   # kernel > dims, causal T
    ((1, 2, 5, 5, 2, 64), (3, 7, 7), (False, False, False)),  # kernel > dims -> clamp
    ((1, 1, 16, 16, 2, 64), (1, 5, 5), (False, False, False)),  # single frame (na2d shape)
    ((1, 4, 7, 40, 2, 32), (3, 5, 5), (False, False, False)),  # hd 32, elongated W
    # W past one CUDA query block and not a multiple of it: ragged tail block
    ((1, 2, 3, 65, 2, 64), (3, 3, 5), (False, False, False)),
    ((1, 2, 3, 96, 1, 64), (1, 3, 33), (False, False, False)),  # kw > CUDA BK=32
    ((1, 3, 4, 34, 2, 16), (3, 3, 5), (False, False, False)),   # hd 16
    ((1, 3, 4, 34, 2, 48), (3, 3, 5), (False, False, False)),   # hd 48 (not a power of 2)
    ((1, 4, 6, 20, 2, 64), (3, 5, 7), (False, False, True)),    # causal W
    ((1, 4, 6, 20, 2, 64), (3, 5, 7), (False, True, False)),    # causal H
    ((1, 4, 6, 20, 2, 64), (3, 5, 7), (True, True, True)),      # causal on every axis
    ((2, 3, 4, 18, 3, 32), (3, 3, 5), (False, False, False)),   # batch and heads > 1
]


def _tolerances(dtype):
    if dtype == torch.float32:
        return 2e-4, 2e-5
    return 2e-2, 2e-2


@pytest.mark.parametrize("shape,kernel,causal", CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scale", [None, 1.0])
def test_na3d_backends(shape, kernel, causal, dtype, scale):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and dtype == torch.float16:
        pytest.skip("fp16 SDPA is not supported on the CPU backend")
    backends = get_capable_backends("na3d", device)
    assert backends, "no backend for na3d"

    torch.manual_seed(0)
    q = torch.randn(shape, device=device, dtype=dtype)
    k = torch.randn(shape, device=device, dtype=dtype)
    v = torch.randn(shape, device=device, dtype=dtype)
    ref = _ref_na3d(q, k, v, kernel, causal, scale)

    rtol, atol = _tolerances(dtype)
    for backend in backends:
        with ck.use_backend(backend):
            out = ck.na3d(q, k, v, list(kernel), list(causal), scale)
        assert_values_close(out.float(), ref.float(), rtol, atol, name=f"na3d[{backend}]")


def test_na2d_matches_na3d_single_frame():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1)
    q = torch.randn(1, 16, 16, 2, 64, device=device)
    k = torch.randn(1, 16, 16, 2, 64, device=device)
    v = torch.randn(1, 16, 16, 2, 64, device=device)
    out2d = ck.na2d(q, k, v, [5, 5])
    ref = _ref_na3d(q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1),
                    (1, 5, 5), (False, False, False), None).squeeze(1)
    assert_values_close(out2d.float(), ref.float(), 2e-4, 2e-5, name="na2d")


@pytest.mark.parametrize("shape,kernel", [
    ((1, 8, 24, 24, 4, 64), (3, 7, 7)),
    ((1, 2, 3, 70, 2, 64), (3, 3, 7)),   # W spanning several CUDA query blocks
])
def test_na3d_backend_agreement(shape, kernel):
    """Every capable backend must agree, not just the first two."""
    if not torch.cuda.is_available():
        pytest.skip("cuda only")
    backends = get_capable_backends("na3d", "cuda")
    if len(backends) < 2:
        pytest.skip("single backend")
    torch.manual_seed(2)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    outs = {}
    for backend in backends:
        with ck.use_backend(backend):
            outs[backend] = ck.na3d(q, k, v, list(kernel), [False, False, False], 1.0)
    ref_name = next(iter(outs))
    for name, out in outs.items():
        if name == ref_name:
            continue
        assert_values_close(out.float(), outs[ref_name].float(), 2e-2, 2e-2,
                            name=f"{ref_name}-vs-{name}")


# ---------------------------------------------------------------------------
# Input validation -- a mismatched k/v is an out-of-bounds read, so these raise
# ---------------------------------------------------------------------------


def _qkv(device, shape=(1, 2, 3, 8, 2, 64), dtype=torch.bfloat16):
    return tuple(torch.randn(shape, device=device, dtype=dtype) for _ in range(3))


@pytest.mark.parametrize("bad", ["k_shape", "v_shape", "k_dtype", "v_dtype"])
def test_na3d_rejects_mismatched_qkv(bad):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    q, k, v = _qkv(device, dtype=dtype)
    if bad == "k_shape":
        k = torch.randn(1, 2, 3, 4, 2, 64, device=device, dtype=dtype)
    elif bad == "v_shape":
        v = torch.randn(1, 2, 3, 4, 2, 64, device=device, dtype=dtype)
    elif bad == "k_dtype":
        k = k.float()
    else:
        v = v.float()
    with pytest.raises(Exception, match=r"(?i)same (shape|dtype)|no backend"):
        ck.na3d(q, k, v, [3, 3, 3], [False, False, False], None)


@pytest.mark.parametrize("kernel,causal", [
    ([3, 3], [False, False, False]),
    ([3, 3, 3, 3], [False, False, False]),
    ([3, 3, 3], [False, False]),
])
def test_na3d_rejects_bad_axis_lists(kernel, causal):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q, k, v = _qkv(device)
    with pytest.raises(Exception, match=r"(?i)3 elements|no backend"):
        ck.na3d(q, k, v, kernel, causal, None)


@pytest.mark.parametrize("kernel,causal", [([5, 5, 5], [False, False]), ([5], [False, False])])
def test_na2d_rejects_bad_axis_lists(kernel, causal):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    shape = (1, 8, 8, 2, 64)
    q, k, v = (torch.randn(shape, device=device, dtype=torch.bfloat16) for _ in range(3))
    with pytest.raises(ValueError, match="2 elements"):
        ck.na2d(q, k, v, kernel, causal, None)
