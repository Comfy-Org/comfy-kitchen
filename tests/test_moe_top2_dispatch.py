# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""moe_top2_dispatch -- HiDream MoE top-2 routing and expert-major dispatch.

The eager backend is the sequence ComfyUI's HiDream MoE runs today (MoEGate's softmax and
topk, then repeat_interleave and the per-expert boolean masks of MOEFeedForwardSwiGLU), so it
must match the stock code exactly, and the MoE forward rebuilt on the op (experts on slices of
x_sorted, index_select by slot) must give the stock output. The Triton kernels round the
softmax once to the logits dtype in the order of the PyTorch softmax kernel and copy rows, so
everything is compared exactly, with the top-2 pair of each row in canonical (expert index)
order; the kernel's own pair order (highest first, the smaller index first on ties) is checked
separately.
"""

import contextlib
import importlib

import pytest
import torch
from torch.nn import functional

import comfy_kitchen as ck
from comfy_kitchen.backends import eager as eager_backend
from comfy_kitchen.registry import registry

OP = "moe_top2_dispatch"


def _stock_gate(hidden_states, gate_weight, top_k=2):
    # comfy/ldm/hidream/model.py MoEGate.forward (ComfyUI fb2b26f), softmax scoring,
    # norm_topk_prob False; cast_to of the gate weight -> .to
    _bsz, _seq_len, h = hidden_states.shape
    hidden_states = hidden_states.view(-1, h)
    logits = functional.linear(
        hidden_states, gate_weight.to(dtype=hidden_states.dtype, device=hidden_states.device), None
    )
    scores = logits.softmax(dim=-1)
    topk_weight, topk_idx = torch.topk(scores, k=top_k, dim=-1, sorted=False)
    return topk_idx, topk_weight


def _swiglu(x, w1, w2, w3):
    # FeedForwardSwiGLU.forward
    return functional.linear(
        functional.silu(functional.linear(x, w1)) * functional.linear(x, w3), w2
    )


def _stock_moe(x, gate_weight, experts, shared):
    # MOEFeedForwardSwiGLU.forward, "if True" branch
    wtype = x.dtype
    identity = x
    orig_shape = x.shape
    topk_idx, topk_weight = _stock_gate(x, gate_weight)
    x = x.view(-1, x.shape[-1])
    flat_topk_idx = topk_idx.view(-1)
    x = x.repeat_interleave(2, dim=0)
    y = torch.empty_like(x, dtype=wtype)
    for i, expert in enumerate(experts):
        y[flat_topk_idx == i] = _swiglu(x[flat_topk_idx == i], *expert).to(dtype=wtype)
    y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
    y = y.view(*orig_shape).to(dtype=wtype)
    return y + _swiglu(identity, *shared)


def _kitchen_moe(x, gate_weight, experts, shared):
    # the same forward on the op: one host read of the counts, each expert on its slice of
    # x_sorted, the outputs back to token-major order with index_select(0, slot)
    wtype = x.dtype
    identity = x
    orig_shape = x.shape
    hidden_states = x.view(-1, x.shape[-1])
    logits = functional.linear(hidden_states, gate_weight.to(dtype=wtype, device=x.device), None)
    topk_weight, _, x_sorted, slot, counts = ck.moe_top2_dispatch(hidden_states, logits)
    out = torch.empty_like(x_sorted)
    start = 0
    for expert, n in zip(experts, counts.tolist(), strict=True):
        if n:
            out[start : start + n] = _swiglu(x_sorted[start : start + n], *expert).to(dtype=wtype)
        start += n
    y = out.index_select(0, slot)
    y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
    y = y.view(*orig_shape).to(dtype=wtype)
    return y + _swiglu(identity, *shared)


def _moe_weights(dim, hidden, n_experts, dtype, device="cpu"):
    g = torch.Generator(device="cpu").manual_seed(7)

    def r(o, i):
        return (torch.randn(o, i, generator=g) * 0.05).to(device=device, dtype=dtype)

    experts = [(r(hidden, dim), r(dim, hidden), r(hidden, dim)) for _ in range(n_experts)]
    shared = (r(hidden // 2, dim), r(dim, hidden // 2), r(hidden // 2, dim))
    return experts, shared


def _canonical(weight, idx, slot):
    # the top-2 pair of every row ordered by expert index (the pair order does not change
    # x_sorted or the MoE output)
    order = idx.argsort(dim=1, stable=True)
    return weight.gather(1, order), idx.gather(1, order), slot.view(-1, 2).gather(1, order)


def _assert_same_dispatch(got, want):
    gw, gi, gs = _canonical(got[0], got[1], got[3])
    ww, wi, ws = _canonical(want[0], want[1], want[3])
    assert torch.equal(gi, wi)
    assert torch.equal(gw, ww)
    assert torch.equal(gs, ws)
    assert torch.equal(got[2], want[2])
    assert torch.equal(got[4], want[4])


def _stable_top2(logits):
    # highest score first, the smaller expert index first on ties
    scores = logits.softmax(dim=-1).float()
    return torch.sort(-scores, dim=-1, stable=True).indices[:, :2]


def _router(n, d, n_experts=4, *, device="cpu", dtype=torch.bfloat16, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(n, d, generator=g).to(device=device, dtype=dtype)
    gate = (torch.randn(n_experts, d, generator=g) * 0.02).to(device=device, dtype=dtype)
    return x, functional.linear(x, gate)


def _non_finite_router(n, d, *, device="cpu", seed=0):
    # rows with a NaN, a +inf, two +inf or only -inf logits (4 NaN scores after the softmax)
    # among finite rows, in the first and in a later 32-row block; the row with one finite
    # logit and three -inf logits stays finite
    x, logits = _router(n, d, device=device, seed=seed)
    nan, inf = float("nan"), float("inf")
    bad = {
        2: [1.0, nan, 0.0, 2.0],
        5: [inf, 0.0, 1.0, 2.0],
        40: [-inf, -inf, -inf, -inf],
        41: [nan, nan, nan, nan],
        42: [inf, inf, 0.0, 0.0],
    }
    for row, values in bad.items():
        logits[row] = torch.tensor(values, dtype=logits.dtype, device=device)
    logits[43] = torch.tensor([0.0, -inf, -inf, -inf], dtype=logits.dtype, device=device)
    return x, logits, sorted(bad)


def _assert_dispatch_consistent(x, out, n_experts=4):
    # two distinct experts per token, counts that match the indices, and slots that are a
    # permutation of the rows of x_sorted, each holding its token row
    _, idx, x_sorted, slot, counts = out
    n = x.shape[0]
    assert bool((idx[:, 0] != idx[:, 1]).all())
    assert torch.equal(counts.cpu(), torch.bincount(idx.view(-1).cpu(), minlength=n_experts))
    assert int(counts.sum()) == 2 * n
    assert torch.equal(slot.sort().values.cpu(), torch.arange(2 * n))
    assert torch.equal(x_sorted.index_select(0, slot), x.repeat_interleave(2, dim=0))


def _call_rule():
    triton_backend = importlib.import_module("comfy_kitchen.backends.triton")
    (rule,) = triton_backend._build_constraints()[OP].call_rules
    return rule


class TestMoeTop2DispatchEager:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("n_experts", [4, 8])
    @pytest.mark.parametrize("n", [1, 7, 64])
    def test_matches_stock_gate_and_masks(self, n, n_experts, dtype):
        g = torch.Generator(device="cpu").manual_seed(n)
        x3 = torch.randn(1, n, 32, generator=g).to(dtype)
        gate = (torch.randn(n_experts, 32, generator=g) * 0.5).to(dtype)
        topk_idx, topk_weight = _stock_gate(x3, gate)
        x = x3.view(-1, 32)
        with ck.use_backend("eager"):
            weight, idx, x_sorted, slot, counts = ck.moe_top2_dispatch(
                x, functional.linear(x, gate)
            )
        assert torch.equal(idx, topk_idx) and torch.equal(weight, topk_weight)
        flat = topk_idx.view(-1)
        xr = x.repeat_interleave(2, dim=0)
        assert torch.equal(x_sorted, torch.cat([xr[flat == i] for i in range(n_experts)]))
        assert torch.equal(counts, torch.bincount(flat, minlength=n_experts))
        assert counts.dtype == torch.int64 and slot.dtype == torch.int64
        assert torch.equal(x_sorted.index_select(0, slot), xr)
        assert torch.equal(slot.sort().values, torch.arange(2 * n))

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_moe_forward_matches_stock(self, dtype):
        # HiDream sizing scaled down: dim 64, routed hidden 96, 4 experts, top 2
        x = torch.randn(2, 40, 64).to(dtype)
        gate = (torch.randn(4, 64) * 0.5).to(dtype)
        experts, shared = _moe_weights(64, 96, 4, dtype)
        with ck.use_backend("eager"):
            got = _kitchen_moe(x, gate, experts, shared)
        assert torch.equal(got, _stock_moe(x, gate, experts, shared))

    def test_expert_without_tokens(self):
        # an expert no row picks gets an empty slice
        x = torch.randn(9, 16)
        logits = torch.tensor([[3.0, 2.0, 0.0, -1.0]]).repeat(9, 1)
        with ck.use_backend("eager"):
            _, _, x_sorted, slot, counts = ck.moe_top2_dispatch(x, logits)
        assert counts.tolist() == [9, 9, 0, 0]
        assert torch.equal(x_sorted.index_select(0, slot), x.repeat_interleave(2, dim=0))

    def test_strided_inputs(self):
        base = torch.randn(30, 48)
        x = base[:, 8:40]
        logits = torch.randn(30, 10)[:, 3:7]
        assert not x.is_contiguous() and not logits.is_contiguous()
        with ck.use_backend("eager"):
            got = ck.moe_top2_dispatch(x, logits)
            want = ck.moe_top2_dispatch(x.contiguous(), logits.contiguous())
        for g, w in zip(got, want, strict=True):
            assert torch.equal(g, w)

    def test_empty_tokens(self):
        with ck.use_backend("eager"):
            out = ck.moe_top2_dispatch(torch.randn(0, 16), torch.randn(0, 4))
        assert [tuple(t.shape) for t in out] == [(0, 2), (0, 2), (0, 16), (0,), (4,)]
        assert out[4].tolist() == [0, 0, 0, 0]

    def test_non_finite_rows(self):
        # the stock softmax + topk: 4 NaN scores, NaN weights, experts 0 and 1
        x, logits, bad = _non_finite_router(70, 16)
        with ck.use_backend("eager"):
            out = ck.moe_top2_dispatch(x, logits)
        want_weight, want_idx = torch.topk(logits.softmax(dim=-1), k=2, dim=-1, sorted=False)
        assert torch.equal(out[1], want_idx)
        torch.testing.assert_close(out[0], want_weight, rtol=0, atol=0, equal_nan=True)
        assert bool(out[0][bad].isnan().all())
        assert not bool(out[0][43].isnan().any())
        _assert_dispatch_consistent(x, out)

    @pytest.mark.parametrize(
        ("x_shape", "logits_shape"),
        [
            ((2, 8, 16), (16, 4)),  # 3D x
            ((16,), (16, 4)),  # 1D x
            ((8, 16), (8, 1, 4)),  # 3D logits
            ((8, 16), (7, 4)),  # token count
            ((8, 16), (8, 1)),  # fewer experts than top 2
        ],
    )
    def test_rejects_bad_inputs(self, x_shape, logits_shape):
        with pytest.raises(ValueError):
            ck.moe_top2_dispatch(torch.randn(x_shape), torch.randn(logits_shape))

    def test_fake_shape(self):
        x = torch.empty(10, 24, device="meta", dtype=torch.bfloat16)
        logits = torch.empty(10, 4, device="meta", dtype=torch.bfloat16)
        weight, idx, x_sorted, slot, counts = ck.moe_top2_dispatch(x, logits)
        assert weight.shape == (10, 2) and weight.dtype == torch.bfloat16
        assert idx.shape == (10, 2) and idx.dtype == torch.int64
        assert x_sorted.shape == (20, 24) and x_sorted.dtype == torch.bfloat16
        assert slot.shape == (20,) and slot.dtype == torch.int64
        assert counts.shape == (4,) and counts.dtype == torch.int64

    def test_cpu_goes_to_eager(self):
        x, logits = _router(8, 16)
        assert registry.get_capable_backend(OP, {"x": x, "logits": logits}) == "eager"


class TestMoeTop2DispatchCallRule:
    # the Triton call rule only looks at shapes and strides, so it runs on meta tensors
    @pytest.mark.parametrize(
        ("n", "d", "n_experts", "ok"),
        [
            (8192, 2560, 4, True),  # HiDream-I1 double block, batch 2
            (4480, 2560, 4, True),  # single block, batch 1
            (1, 8, 4, True),
            (8, 16, 8, False),  # 8 experts
            (8, 16, 2, False),
            (0, 16, 4, False),  # empty
            (8, 0, 4, False),  # empty
            (2**14, 1, 4, True),
            (2**14 + 1, 1, 4, False),  # over the token cap
        ],
    )
    def test_call_rule(self, n, d, n_experts, ok):
        x = torch.empty(n, d, device="meta", dtype=torch.bfloat16)
        logits = torch.empty(n, n_experts, device="meta", dtype=torch.bfloat16)
        assert _call_rule()({"x": x, "logits": logits}).success == ok

    def test_call_rule_last_dim_stride(self):
        x = torch.empty(8, 32, device="meta", dtype=torch.bfloat16)
        logits = torch.empty(8, 4, device="meta", dtype=torch.bfloat16)
        assert _call_rule()({"x": x[:, ::2], "logits": logits}).success is False
        assert _call_rule()({"x": x, "logits": logits.t().contiguous().t()}).success is False
        assert _call_rule()({"x": x[:, 8:24], "logits": logits}).success  # row-strided view

    def test_bf16_only(self):
        triton_backend = importlib.import_module("comfy_kitchen.backends.triton")
        params = triton_backend._build_constraints()[OP].params
        for name in ("x", "logits"):
            assert params[name].dtypes == frozenset({torch.bfloat16})


# Token counts and widths around the kernels' 32-row blocks and 256-column copy steps, with
# the HiDream-I1 width (2560) among them; the token counts of a real call are 4096 to 8960.
_SHAPES = [(1, 2560), (31, 2560), (32, 2560), (33, 2560), (300, 2560), (1000, 64), (517, 520)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
class TestMoeTop2DispatchTriton:
    @pytest.fixture(autouse=True)
    def _triton(self):
        pytest.importorskip("triton")
        if not registry.is_available("triton"):
            pytest.skip("triton backend not available")

    @staticmethod
    @contextlib.contextmanager
    def _served(expected):
        # the backend of the implementation the registry returns for each call in the block
        # (get_implementation wrapped), so the check is on the call that ran, not on a query
        served = []
        get_implementation = registry.get_implementation

        def recording(func_name, backend=None, kwargs=None):
            impl = get_implementation(func_name, backend, kwargs)
            if func_name == OP:
                served.append(impl.__module__.rsplit(".", 2)[-2])
            return impl

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(registry, "get_implementation", recording)
            yield
        assert served and all(b == expected for b in served), served

    @pytest.mark.parametrize(("n", "d"), _SHAPES)
    def test_matches_eager(self, n, d):
        x, logits = _router(n, d, device="cuda", seed=n)
        kwargs = {"x": x, "logits": logits}
        assert registry.validate_backend_for_call("triton", OP, kwargs).success
        assert registry.get_capable_backend(OP, kwargs) == "triton"
        with ck.use_backend("triton"):
            got = ck.moe_top2_dispatch(x, logits)
        want = eager_backend.moe_top2_dispatch(**kwargs)
        _assert_same_dispatch(got, want)
        assert got[0].dtype == torch.bfloat16 and got[2].dtype == torch.bfloat16
        assert torch.equal(got[1], _stable_top2(logits))
        assert torch.equal(got[2].index_select(0, got[3]), x.repeat_interleave(2, dim=0))

    def test_ties(self):
        # equal scores: the smaller expert index wins, as torch.topk(k=2, sorted=False)
        logits = torch.tensor(
            [[1, 1, 0, 0], [0, 1, 1, 1], [2, 2, 2, 2], [0, 0, 0, 1], [3, 1, 3, 1], [0, 0, 1, 1]],
            dtype=torch.bfloat16,
            device="cuda",
        )
        x = torch.randn(6, 64, device="cuda").to(torch.bfloat16)
        with ck.use_backend("triton"):
            weight, idx, x_sorted, slot, _ = ck.moe_top2_dispatch(x, logits)
        assert idx.tolist() == [[0, 1], [1, 2], [0, 1], [3, 0], [0, 2], [2, 3]]
        assert torch.equal(weight, logits.softmax(dim=-1).gather(1, idx))
        assert torch.equal(x_sorted.index_select(0, slot), x.repeat_interleave(2, dim=0))

    @pytest.mark.parametrize("layout", ["x_rows_view", "logits_view", "batch_view"])
    def test_layouts(self, layout):
        x, logits = _router(300, 520, device="cuda", seed=3)
        if layout == "x_rows_view":
            # columns of a wider tensor: row stride 600, unit column stride
            wide = torch.zeros(300, 600, device="cuda", dtype=torch.bfloat16)
            wide[:, 40:560] = x
            x = wide[:, 40:560]
        elif layout == "logits_view":
            wide = torch.zeros(300, 10, device="cuda", dtype=torch.bfloat16)
            wide[:, 3:7] = logits
            logits = wide[:, 3:7]
        else:
            # [B, S, D].view(-1, D), as MoEGate does
            x = x.view(2, 150, 520).view(-1, 520)
        kwargs = {"x": x, "logits": logits}
        assert registry.get_capable_backend(OP, kwargs) == "triton"
        with ck.use_backend("triton"):
            got = ck.moe_top2_dispatch(x, logits)
        _assert_same_dispatch(got, eager_backend.moe_top2_dispatch(**kwargs))

    def test_non_finite_rows(self):
        # a row whose scores are all NaN gets two distinct experts, as torch.topk gives, and
        # does not move the slots of the other rows
        x, logits, bad = _non_finite_router(70, 64, device="cuda", seed=5)
        kwargs = {"x": x, "logits": logits}
        assert registry.get_capable_backend(OP, kwargs) == "triton"
        with ck.use_backend("triton"):
            got = ck.moe_top2_dispatch(x, logits)
        want = eager_backend.moe_top2_dispatch(**kwargs)
        gw, gi, gs = _canonical(got[0], got[1], got[3])
        ww, wi, ws = _canonical(want[0], want[1], want[3])
        assert torch.equal(gi, wi)
        torch.testing.assert_close(gw, ww, rtol=0, atol=0, equal_nan=True)
        assert torch.equal(gs, ws)
        assert torch.equal(got[2], want[2])
        assert torch.equal(got[4], want[4])
        assert got[1][bad].tolist() == [[0, 1]] * len(bad)
        assert bool(got[0][bad].isnan().all())
        _assert_dispatch_consistent(x, got)

    def test_moe_forward_matches_stock(self):
        # the whole MoE on the op against the stock forward; the expert GEMMs run on slices
        # of x_sorted instead of masked copies, so allow for a different GEMM kernel choice
        x = torch.randn(2, 300, 256, device="cuda").to(torch.bfloat16)
        gate = (torch.randn(4, 256) * 0.02).to(device="cuda", dtype=torch.bfloat16)
        experts, shared = _moe_weights(256, 384, 4, torch.bfloat16, "cuda")
        with ck.use_backend("triton"):
            got = _kitchen_moe(x, gate, experts, shared)
        ref = _stock_moe(x, gate, experts, shared)
        rel = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
        assert rel < 1e-2

    @pytest.mark.parametrize(
        ("n", "d"),
        [(8192, 2560), (8960, 2560), (4096, 2560), (4480, 2560)],
        ids=["double_batch2", "single_batch2", "double_batch1", "single_batch1"],
    )
    def test_real_shape_served(self, n, d):
        # HiDream-I1 at 1024x1024: 4096 image tokens per sample, plus 3 x 128 text tokens in a
        # single block; batch 2 is cond + uncond
        x, logits = _router(n, d, device="cuda", seed=11)
        kwargs = {"x": x, "logits": logits}
        assert registry.get_capable_backend(OP, kwargs) == "triton"
        with ck.use_backend("triton"):
            got = ck.moe_top2_dispatch(x, logits)
        _assert_same_dispatch(got, eager_backend.moe_top2_dispatch(**kwargs))
        assert int(got[4].sum()) == 2 * n

    def test_token_cap(self):
        # the largest accepted token count is served by triton; one more token by eager
        x, logits = _router(2**14, 8, device="cuda", seed=5)
        kwargs = {"x": x, "logits": logits}
        with self._served("triton"), ck.use_backend("triton"):
            got = ck.moe_top2_dispatch(x, logits)
        _assert_same_dispatch(got, eager_backend.moe_top2_dispatch(**kwargs))
        x, logits = _router(2**14 + 1, 8, device="cuda", seed=5)
        kwargs = {"x": x, "logits": logits}
        assert not registry.validate_backend_for_call("triton", OP, kwargs).success
        with self._served("eager"), ck.use_backend("triton"):
            got = ck.moe_top2_dispatch(x, logits)
        _assert_same_dispatch(got, eager_backend.moe_top2_dispatch(**kwargs))

    def test_empty_input_goes_to_eager(self):
        x = torch.empty(0, 64, device="cuda", dtype=torch.bfloat16)
        logits = torch.empty(0, 4, device="cuda", dtype=torch.bfloat16)
        kwargs = {"x": x, "logits": logits}
        assert not registry.validate_backend_for_call("triton", OP, kwargs).success
        with ck.use_backend("triton"):
            out = ck.moe_top2_dispatch(x, logits)
        assert out[2].shape == (0, 64) and out[4].tolist() == [0, 0, 0, 0]

    @pytest.mark.parametrize("case", ["float16", "float32", "8_experts", "x_col_step2", "cpu"])
    def test_unsupported_inputs_go_to_eager(self, case):
        if case == "8_experts":
            x, logits = _router(64, 32, 8, device="cuda")
        elif case == "x_col_step2":
            x, logits = _router(64, 64, device="cuda")
            x = x[:, ::2]
            logits = functional.linear(x, torch.randn(4, 32, device="cuda").to(x.dtype))
        elif case == "cpu":
            x, logits = _router(64, 32)
        else:
            x, logits = _router(64, 32, device="cuda", dtype=getattr(torch, case))
        kwargs = {"x": x, "logits": logits}
        assert not registry.validate_backend_for_call("triton", OP, kwargs).success
        assert registry.get_capable_backend(OP, kwargs) == "eager"
        with ck.use_backend("triton"):
            got = ck.moe_top2_dispatch(x, logits)
        for g, w in zip(got, eager_backend.moe_top2_dispatch(**kwargs), strict=True):
            assert torch.equal(g, w)
