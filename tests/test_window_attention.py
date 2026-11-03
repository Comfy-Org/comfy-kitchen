# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.exceptions import NoCapableBackendError
from tests.conftest import assert_values_close, get_capable_backends

# ---------------------------------------------------------------------------
# Reference implementation: direct per-query loop over the window semantics
# ---------------------------------------------------------------------------


def _axis(i, kernel, length, clamp):
    if clamp:
        kernel = min(kernel, length)
        s = min(max(i - kernel // 2, 0), length - kernel)
        return s, s + kernel
    return max(i - kernel // 2, 0), min(i - kernel // 2 + kernel, length)


def _ref_window_attention(q, k, v, windows, temporal_window, clamp, scale):
    b, tq, h, w, nh, hd = q.shape
    tk, kv_heads = k.shape[1], k.shape[4]
    windows = windows * tk if len(windows) == 1 else windows
    temporal_window = tk if temporal_window is None else temporal_window
    if scale is None:
        scale = hd ** -0.5
    k = k.repeat_interleave(nh // kv_heads, dim=4)
    v = v.repeat_interleave(nh // kv_heads, dim=4)
    out = torch.empty_like(q)
    for t in range(tq):
        last = t + tk - tq
        planes = range(max(0, last - temporal_window + 1), last + 1)
        for y in range(h):
            for x in range(w):
                ks, vs = [], []
                for p in planes:
                    y0, y1 = _axis(y, windows[p][0], h, clamp)
                    x0, x1 = _axis(x, windows[p][1], w, clamp)
                    ks.append(k[:, p, y0:y1, x0:x1].reshape(b, -1, nh, hd))
                    vs.append(v[:, p, y0:y1, x0:x1].reshape(b, -1, nh, hd))
                kk, vv = torch.cat(ks, dim=1), torch.cat(vs, dim=1)
                s = torch.einsum("bnd,bknd->bnk", q[:, t, y, x].float(), kk.float()) * scale
                a = torch.softmax(s, dim=-1)
                out[:, t, y, x] = torch.einsum("bnk,bknd->bnd", a, vv.float()).to(q.dtype)
    return out


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

CASES = [
    # (q shape (B,Tq,H,W,NH,HD), key planes Tk, kv heads, windows, temporal_window, clamp)
    ((1, 5, 9, 20, 4, 128), 5, 4, [(6, 10)], 4, True),         # causal planes, every query plane
    ((1, 3, 9, 20, 2, 128), 6, 2, [(6, 10)], 4, True),         # last 3 of 6 planes (cached keys)
    ((1, 1, 12, 40, 8, 64), 4, 2, [(7, 7), (7, 7), (4, 4), (4, 4)], None, False),  # per-plane windows, GQA
    ((1, 1, 12, 40, 8, 64), 3, 2, [(7, 7), (7, 7), (4, 4)], None, False),
    ((1, 1, 12, 40, 8, 64), 1, 2, [(9, 9)], None, False),
    ((1, 1, 6, 10, 2, 64), 2, 2, [(9, 13)], None, False),      # cut-off window larger than the frame
    ((1, 1, 6, 10, 2, 64), 2, 2, [(9, 13)], None, True),       # clamped window larger than the frame
    ((1, 2, 24, 50, 2, 64), 2, 2, [(22, 40)], 2, True),        # even windows
    ((2, 1, 3, 70, 2, 32), 2, 1, [(3, 9)], None, False),       # W over several query blocks, ragged tail
    ((1, 2, 4, 34, 2, 48), 2, 2, [(3, 5)], 1, True),           # hd 48, own plane only
    ((1, 2, 5, 33, 4, 16), 3, 1, [(3, 3), (5, 5), (3, 7)], 2, False),  # hd 16, MQA
    ((1, 2, 6, 70, 2, 128), 4, 2, [(5, 40)], 3, True),        # hd 128 with four warps along W (wide window)
    ((1, 1, 9, 20, 8, 128), 3, 2, [(5, 7), (5, 7), (3, 5)], None, False),  # hd 128 on the grouped-head patch kernel
]


def _tolerances(dtype):
    if dtype == torch.float32:
        return 2e-4, 2e-5
    return 2e-2, 2e-2


def _inputs(shape, tk, kv_heads, dtype, device, seed=0):
    torch.manual_seed(seed)
    b, _, h, w, _, hd = shape
    q = torch.randn(shape, device=device, dtype=dtype)
    k = torch.randn((b, tk, h, w, kv_heads, hd), device=device, dtype=dtype)
    v = torch.randn((b, tk, h, w, kv_heads, hd), device=device, dtype=dtype)
    return q, k, v


@pytest.mark.parametrize("shape,tk,kv_heads,windows,temporal_window,clamp", CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_window_attention_backends(shape, tk, kv_heads, windows, temporal_window, clamp, dtype):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and dtype == torch.float16:
        pytest.skip("fp16 SDPA is not supported on the CPU backend")
    backends = get_capable_backends("window_attention", device)
    assert backends, "no backend for window_attention"

    q, k, v = _inputs(shape, tk, kv_heads, dtype, device)
    ref = _ref_window_attention(q, k, v, windows, temporal_window, clamp, None)

    rtol, atol = _tolerances(dtype)
    for backend in backends:
        with ck.use_backend(backend):
            out = ck.window_attention(q, k, v, windows, temporal_window, clamp)
        assert_values_close(out.float(), ref.float(), rtol, atol, name=f"window_attention[{backend}]")


def test_window_attention_matches_na3d():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q, k, v = _inputs((1, 6, 9, 20, 2, 64), 6, 2, torch.bfloat16, device, seed=1)
    ref = ck.na3d(q, k, v, [3, 5, 7], [True, False, False], None)
    out = ck.window_attention(q, k, v, (5, 7), temporal_window=3, clamp=True)
    assert_values_close(out.float(), ref.float(), 2e-2, 2e-2, name="window_attention-vs-na3d")


# ---------------------------------------------------------------------------
# Input validation -- a mismatched k/v is an out-of-bounds read, so these raise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad,match", [
    ("k_width", "batch, height, width and head_dim"),
    ("v_shape", "same shape as k"),
    ("heads", "do not divide"),
    ("planes", "more than the"),
    ("windows", "one window for every key plane"),
])
def test_window_attention_rejects_mismatches(bad, match):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q, k, v = _inputs((1, 2, 4, 8, 4, 64), 3, 2, torch.bfloat16, device)
    windows = [(3, 3)]
    if bad == "k_width":
        k = k[:, :, :, :6].contiguous()
        v = v[:, :, :, :6].contiguous()
    elif bad == "v_shape":
        v = v[:, :2].contiguous()
    elif bad == "heads":
        k = torch.cat((k, k[:, :, :, :, :1]), dim=4)
        v = torch.cat((v, v[:, :, :, :, :1]), dim=4)
    elif bad == "planes":
        k, v = k[:, :1].contiguous(), v[:, :1].contiguous()
    else:
        windows = [(3, 3), (5, 5)]
    with pytest.raises(NoCapableBackendError, match=match):
        ck.window_attention(q, k, v, windows)


def test_window_attention_rejects_non_positive_window():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q, k, v = _inputs((1, 1, 4, 8, 2, 64), 1, 2, torch.bfloat16, device)
    with pytest.raises(NoCapableBackendError, match="must be positive"):
        ck.window_attention(q, k, v, (0, 3))
