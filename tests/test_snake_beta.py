# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""snake_beta / snake_beta_with_sum -- Oobleck audio VAE SnakeBeta, optionally after the
ResidualUnit add.

The eager backend is the sequence ComfyUI's Oobleck decoder runs today (ResidualUnit's
``x + res``, then SnakeBeta.forward with alpha_logscale=True), so it must match the stock code
exactly. The Triton kernel computes in fp32 and rounds once, so its activation is compared to
eager and to an fp32 reference with a tolerance; the returned sum must be bit-equal to the eager
add. The decoder passes contiguous fp16 tensors (conv outputs and the fp16 parameters); other
dtypes and layouts must go to eager.
"""

import contextlib
import importlib

import pytest
import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode

import comfy_kitchen as ck
from comfy_kitchen.backends import eager as eager_backend
from comfy_kitchen.registry import registry

OP = "snake_beta"
OP_SUM = "snake_beta_with_sum"


def snake_beta(x, alpha, beta):
    # comfy/ldm/audio/autoencoder.py snake_beta (ComfyUI fb2b26f)
    return x + (1.0 / (beta + 0.000000001)) * pow(torch.sin(x * alpha), 2)


class _StockSnakeBeta(nn.Module):
    # comfy/ldm/audio/autoencoder.py SnakeBeta (ComfyUI fb2b26f), parameters set by the test
    def __init__(self, alpha, beta, alpha_logscale=True):
        super().__init__()
        self.alpha_logscale = alpha_logscale
        self.alpha = nn.Parameter(alpha.clone(), requires_grad=False)
        self.beta = nn.Parameter(beta.clone(), requires_grad=False)
        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1).to(x.device)  # line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1).to(x.device)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = snake_beta(x, alpha, beta)

        return x


def _stock(x, alpha, beta, residual=None):
    # ResidualUnit.forward's 'return x + res', then the next SnakeBeta
    s = x if residual is None else x + residual
    return s, _StockSnakeBeta(alpha, beta)(s)


def _params(c, *, device="cpu", dtype=torch.float32, scale=0.5):
    # log-scale alpha / beta, as SnakeBeta stores them
    alpha = (torch.randn(c, device=device) * scale).to(dtype)
    beta = (torch.randn(c, device=device) * scale).to(dtype)
    return alpha, beta


class _ResidualUnit(nn.Module):
    # comfy/ldm/audio/autoencoder.py ResidualUnit(use_snake=True) with plain Conv1d in place of
    # WNConv1d (weight norm only changes how the conv weight is stored)
    def __init__(self, channels, dilation, dtype):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.layers = nn.Sequential(
            _StockSnakeBeta(*_params(channels, dtype=dtype)),
            nn.Conv1d(channels, channels, kernel_size=7, dilation=dilation, padding=padding),
            _StockSnakeBeta(*_params(channels, dtype=dtype)),
            nn.Conv1d(channels, channels, kernel_size=1),
        ).to(dtype)

    def forward(self, x):
        res = x
        x = self.layers(x)
        return x + res


def _decoder_block_tail(units, final_snake, x):
    # DecoderBlock's three ResidualUnits followed by the next SnakeBeta (the next block's first
    # activation or the decoder's final one), as ComfyUI runs it
    for unit in units:
        x = unit(x)
    return final_snake(x)


def _decoder_block_tail_kitchen(units, final_snake, x):
    # the same walk through the ops: every ResidualUnit add goes into the SnakeBeta that follows
    # it; snake_beta_with_sum when the sum is also the next unit's residual, snake_beta when it
    # only feeds the final SnakeBeta
    first = units[0].layers
    h = ck.snake_beta(x, first[0].alpha, first[0].beta)
    res = x
    for i, unit in enumerate(units):
        layers = unit.layers
        y = layers[3](ck.snake_beta(layers[1](h), layers[2].alpha, layers[2].beta))
        if i + 1 < len(units):
            nxt = units[i + 1].layers[0]
            res, h = ck.snake_beta_with_sum(y, nxt.alpha, nxt.beta, res)
        else:
            h = ck.snake_beta(y, final_snake.alpha, final_snake.beta, res)
    return h


def _reference32(x, alpha, beta, residual=None):
    r = None if residual is None else residual.float()
    return _stock(x.float(), alpha.float(), beta.float(), r)


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def _call_rule():
    triton_backend = importlib.import_module("comfy_kitchen.backends.triton")
    (rule,) = triton_backend._build_constraints()[OP].call_rules
    return rule


class TestSnakeBetaEager:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("form", ["snake", "add_snake", "add_snake_sum"])
    def test_matches_stock(self, form, dtype):
        x = torch.randn(2, 16, 37).to(dtype)
        alpha, beta = _params(16, dtype=dtype)
        residual = None if form == "snake" else torch.randn(2, 16, 37).to(dtype)
        s_ref, ref = _stock(x, alpha, beta, residual)
        with ck.use_backend("eager"):
            if form == "add_snake_sum":
                s, out = ck.snake_beta_with_sum(x, alpha, beta, residual)
                assert torch.equal(s, s_ref)
            else:
                out = ck.snake_beta(x, alpha, beta, residual)
        assert out.dtype == dtype
        assert torch.equal(out, ref)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_decoder_block_walk_matches_stock(self, dtype):
        # three ResidualUnits and the following SnakeBeta, stock modules vs the op walk
        torch.manual_seed(0)
        units = [_ResidualUnit(8, d, dtype) for d in (1, 3, 9)]
        final_snake = _StockSnakeBeta(*_params(8, dtype=dtype))
        x = torch.randn(1, 8, 64).to(dtype)
        with torch.no_grad():
            ref = _decoder_block_tail(units, final_snake, x)
            with ck.use_backend("eager"):
                got = _decoder_block_tail_kitchen(units, final_snake, x)
        assert torch.equal(got, ref)

    def test_mixed_dtypes_promote_like_stock(self):
        # fp32 parameters with an fp16 x (or residual) give fp32, as in the stock module
        x = torch.randn(1, 8, 20).half()
        alpha, beta = _params(8)
        residual = torch.randn(1, 8, 20)
        for res in (None, residual):
            out = ck.snake_beta(x, alpha, beta, res)
            assert out.dtype == torch.float32
            assert torch.equal(out, _stock(x, alpha, beta, res)[1])
        s, out = ck.snake_beta_with_sum(x, alpha.half(), beta.half(), residual)
        s_ref, ref = _stock(x, alpha.half(), beta.half(), residual)
        assert s.dtype == s_ref.dtype == torch.float32
        assert torch.equal(s, s_ref) and torch.equal(out, ref)

    def test_strided_inputs(self):
        base = torch.randn(2, 16, 40)
        x = base[:, :, ::2]
        residual = base.transpose(1, 2).contiguous().transpose(1, 2)[:, :, 1::2]
        alpha, beta = _params(32)
        alpha, beta = alpha[::2], beta[::2]
        assert not x.is_contiguous() and not residual.is_contiguous()
        assert not alpha.is_contiguous()
        s, out = ck.snake_beta_with_sum(x, alpha, beta, residual)
        s_ref, ref = _stock(x, alpha, beta, residual)
        assert torch.equal(s, s_ref) and torch.equal(out, ref)
        assert torch.equal(ck.snake_beta(x, alpha, beta), _stock(x, alpha, beta)[1])

    @pytest.mark.parametrize(
        ("x_shape", "p_shape", "res_shape"),
        [
            ((16, 37), (16,), None),  # 2D x
            ((1, 2, 16, 37), (16,), None),  # 4D x
            ((1, 16, 37), (8,), None),  # parameter size
            ((1, 16, 37), (1, 16, 1), None),  # parameters already lined up
            ((1, 16, 37), (16,), (1, 16, 36)),  # residual length
            ((1, 16, 37), (16,), (1, 16, 1)),  # broadcast residual
        ],
    )
    def test_rejects_bad_inputs(self, x_shape, p_shape, res_shape):
        x = torch.randn(x_shape)
        alpha, beta = torch.randn(p_shape), torch.randn(p_shape)
        residual = None if res_shape is None else torch.randn(res_shape)
        with pytest.raises(ValueError):
            ck.snake_beta(x, alpha, beta, residual)
        with pytest.raises(ValueError):
            ck.snake_beta_with_sum(x, alpha, beta, torch.randn(res_shape or (1, 16, 36)))

    def test_fake_shape_and_dtype(self):
        def meta(shape, dtype=torch.float16):
            return torch.empty(shape, device="meta", dtype=dtype)

        x, p = meta((2, 8, 5)), meta((8,))
        out = ck.snake_beta(x, p, p)
        assert out.shape == (2, 8, 5) and out.dtype == torch.float16
        assert ck.snake_beta(x, p, p, meta((2, 8, 5), torch.float32)).dtype == torch.float32
        assert ck.snake_beta(x, meta((8,), torch.float32), p).dtype == torch.float32
        s, out = ck.snake_beta_with_sum(x, p, meta((8,), torch.float32), x)
        assert s.shape == out.shape == (2, 8, 5)
        assert s.dtype == torch.float16 and out.dtype == torch.float32
        # the fake's dtypes are eager's for every mix of the three inputs
        for xd in (torch.float16, torch.bfloat16, torch.float32):
            for pd in (torch.float16, torch.bfloat16, torch.float32):
                for rd in (torch.float16, torch.float32):
                    xc = torch.randn(1, 8, 5).to(xd)
                    pc = torch.randn(8).to(pd)
                    rc = torch.randn(1, 8, 5).to(rd)
                    s_ref, ref = eager_backend.snake_beta_with_sum(xc, pc, pc, rc)
                    s, out = ck.snake_beta_with_sum(
                        meta((1, 8, 5), xd), meta((8,), pd), meta((8,), pd), meta((1, 8, 5), rd)
                    )
                    assert (s.dtype, out.dtype) == (s_ref.dtype, ref.dtype)

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required"),
            ),
        ],
    )
    @pytest.mark.parametrize("view", ["transposed", "step2"])
    def test_fake_strides_non_contiguous(self, view, device):
        # The real outputs must have the fake's strides also when eager serves a non-contiguous
        # x, or torch.compile asserts on the stride. opcheck compares strides on GPU tensors
        # only, so the strides are also compared directly.
        if view == "transposed":
            x = torch.randn(2, 40, 16, device=device).half().transpose(1, 2)
        else:
            x = torch.randn(2, 16, 80, device=device).half()[:, :, ::2]
        residual = torch.randn(2, 40, 16, device=device).half().transpose(1, 2)
        alpha, beta = _params(16, device=device, dtype=torch.float16)
        assert not x.is_contiguous()
        cases = [
            (torch.ops.comfy_kitchen.snake_beta.default, (x, alpha, beta, None)),
            (torch.ops.comfy_kitchen.snake_beta.default, (x, alpha, beta, residual)),
            (torch.ops.comfy_kitchen.snake_beta_with_sum.default, (x, alpha, beta, residual)),
        ]
        for op, args in cases:
            torch.library.opcheck(op, args, test_utils=("test_faketensor",))
            real = op(*args)
            with FakeTensorMode() as mode:
                fake = op(*(None if a is None else mode.from_tensor(a) for a in args))
            real = real if isinstance(real, tuple) else (real,)
            fake = fake if isinstance(fake, tuple) else (fake,)
            assert [t.stride() for t in real] == [t.stride() for t in fake]

    def test_cpu_goes_to_eager(self):
        x = torch.randn(1, 8, 20, dtype=torch.float16)
        alpha, beta = _params(8, dtype=torch.float16)
        kwargs = {"x": x, "alpha": alpha, "beta": beta, "residual": None}
        assert registry.get_capable_backend(OP, kwargs) == "eager"
        kwargs["residual"] = x.clone()
        assert registry.get_capable_backend(OP_SUM, kwargs) == "eager"


class TestSnakeBetaCallRule:
    # the Triton call rule only looks at shapes, layouts and devices, so it runs on meta tensors
    @staticmethod
    def _kwargs(x_shape, *, residual=False, x_view=None, p_view=False):
        x = torch.empty(x_shape, device="meta", dtype=torch.float16)
        if x_view == "step2":
            x = torch.empty(*x_shape[:2], 2 * x_shape[2], device="meta", dtype=torch.float16)
            x = x[:, :, ::2]
        elif x_view == "transposed":
            x = torch.empty(x_shape[0], x_shape[2], x_shape[1], device="meta", dtype=torch.float16)
            x = x.transpose(1, 2)
        c = x_shape[1]
        p = torch.empty(2 * c if p_view else c, device="meta", dtype=torch.float16)
        p = p[::2] if p_view else p
        res = torch.empty(x_shape, device="meta", dtype=torch.float16) if residual else None
        return {"x": x, "alpha": p, "beta": p, "residual": res}

    @pytest.mark.parametrize(
        ("x_shape", "kw", "ok"),
        [
            ((1, 2048, 3000), {}, True),
            ((1, 128, 5760000), {"residual": True}, True),
            ((1, 65535, 4), {}, True),  # 65535 rows
            ((1, 65536, 4), {}, False),  # 65536 rows (grid)
            ((2, 32768, 4), {"residual": True}, False),
            ((0, 8, 4), {}, False),  # empty
            ((1, 8, 0), {}, False),  # empty
            ((1, 8, 64), {"x_view": "step2"}, False),
            ((1, 8, 64), {"x_view": "transposed"}, False),
            ((1, 8, 64), {"p_view": True}, False),
        ],
    )
    def test_call_rule(self, x_shape, kw, ok):
        assert _call_rule()(self._kwargs(x_shape, **kw)).success == ok

    def test_call_rule_strided_residual(self):
        kwargs = self._kwargs((1, 8, 64), residual=True)
        kwargs["residual"] = torch.empty(1, 8, 128, device="meta", dtype=torch.float16)[..., ::2]
        assert not _call_rule()(kwargs).success

    def test_fp16_only(self):
        triton_backend = importlib.import_module("comfy_kitchen.backends.triton")
        constraints = triton_backend._build_constraints()
        for op in (OP, OP_SUM):
            params = constraints[op].params
            for name in ("x", "alpha", "beta", "residual"):
                assert params[name].dtypes == frozenset({torch.float16})


# The Oobleck decoder calls of an ACE-Step 1.5 decode (120 s, batch 1, fp16): C as captured, T
# scaled down (captured T in the comment). "snake" = a SnakeBeta alone, "sum" = ResidualUnit add +
# SnakeBeta returning the sum (the sum is the next unit's residual), "add" = ResidualUnit add +
# SnakeBeta without the sum (the next block's / the final SnakeBeta).
# (form, C, T)
_SHAPES = [
    ("snake", 2048, 3000),  # T 3000, the first DecoderBlock's SnakeBeta
    ("snake", 1024, 3000),  # T 30000
    ("snake", 512, 4500),  # T 180000
    ("snake", 256, 9000),  # T 720000
    ("snake", 128, 18000),  # T 2880000
    ("snake", 128, 36000),  # T 5760000, the final SnakeBeta
    ("sum", 1024, 3000),
    ("add", 1024, 3000),
    ("sum", 512, 4500),
    ("add", 512, 4500),
    ("sum", 256, 9000),
    ("add", 256, 9000),
    ("sum", 128, 18000),
    ("add", 128, 18000),
    ("sum", 128, 36000),
    ("add", 128, 36000),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
class TestSnakeBetaTriton:
    @pytest.fixture(autouse=True)
    def _triton(self):
        pytest.importorskip("triton")
        if not registry.is_available("triton"):
            pytest.skip("triton backend not available")

    @staticmethod
    @contextlib.contextmanager
    def _served(expected, op=OP):
        # the backend of the implementation the registry returns for each call in the block
        # (get_implementation wrapped), so the check is on the call that ran, not on a query
        served = []
        get_implementation = registry.get_implementation

        def recording(func_name, backend=None, kwargs=None):
            impl = get_implementation(func_name, backend, kwargs)
            if func_name == op:
                served.append(impl.__module__.rsplit(".", 2)[-2])
            return impl

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(registry, "get_implementation", recording)
            yield
        assert served and all(b == expected for b in served), served

    @staticmethod
    def _inputs(c, t, residual, b=1, dtype=torch.float16):
        x = torch.randn(b, c, t, device="cuda").to(dtype)
        alpha, beta = _params(c, device="cuda", dtype=dtype)
        res = torch.randn(b, c, t, device="cuda").to(dtype) if residual else None
        return x, alpha, beta, res

    @staticmethod
    def _check_close(got, ref, ref32):
        # one rounding: never further from the fp32 reference than the eager chain, max and mean
        assert _rel(got, ref) < 2e-3
        assert _rel(got, ref32) < 2e-3
        err, err_eager = (got.float() - ref32).abs(), (ref.float() - ref32).abs()
        assert err.max() <= err_eager.max()
        assert err.mean() <= err_eager.mean()

    @pytest.mark.parametrize(("form", "c", "t"), _SHAPES)
    def test_matches_eager(self, form, c, t):
        x, alpha, beta, res = self._inputs(c, t, form != "snake")
        kwargs = {"x": x, "alpha": alpha, "beta": beta, "residual": res}
        op = OP_SUM if form == "sum" else OP
        assert registry.validate_backend_for_call("triton", op, kwargs).success
        _, ref32 = _reference32(x, alpha, beta, res)
        if form == "sum":
            with self._served("triton", op), ck.use_backend("triton"):
                s, got = ck.snake_beta_with_sum(x, alpha, beta, res)
            s_ref, ref = eager_backend.snake_beta_with_sum(**kwargs)
            assert s.dtype == torch.float16 and s.is_contiguous()
            assert torch.equal(s, s_ref)
        else:
            with self._served("triton", op), ck.use_backend("triton"):
                got = ck.snake_beta(x, alpha, beta, res)
            ref = eager_backend.snake_beta(**kwargs)
        assert got.shape == x.shape and got.dtype == torch.float16 and got.is_contiguous()
        self._check_close(got, ref, ref32)

    def test_sum_and_single_outputs_agree(self):
        # the activation is the same with and without the stored sum
        x, alpha, beta, res = self._inputs(256, 5000, True)
        with self._served("triton", OP), self._served("triton", OP_SUM), ck.use_backend("triton"):
            s, out_sum = ck.snake_beta_with_sum(x, alpha, beta, res)
            out = ck.snake_beta(x, alpha, beta, res)
        assert torch.equal(out, out_sum)
        assert torch.equal(s, x + res)

    @pytest.mark.parametrize(
        ("b", "c", "t"),
        [
            (2, 96, 2049),  # batch 2, T one past a block
            (1, 7, 1),  # one step
            (3, 5, 2047),  # T one short of a block
            (1, 160, 4096),  # T a multiple of the block
        ],
    )
    def test_odd_sizes(self, b, c, t):
        x, alpha, beta, res = self._inputs(c, t, True, b=b)
        kwargs = {"x": x, "alpha": alpha, "beta": beta, "residual": res}
        with self._served("triton", OP), self._served("triton", OP_SUM), ck.use_backend("triton"):
            got = ck.snake_beta(x, alpha, beta)
            s, got_add = ck.snake_beta_with_sum(x, alpha, beta, res)
        assert _rel(got, eager_backend.snake_beta(x, alpha, beta)) < 2e-3
        s_ref, ref = eager_backend.snake_beta_with_sum(**kwargs)
        assert torch.equal(s, s_ref)
        assert _rel(got_add, ref) < 2e-3

    def test_real_shape_served(self):
        # the second DecoderBlock's SnakeBeta call at its captured size
        x, alpha, beta, res = self._inputs(1024, 30000, True)
        kwargs = {"x": x, "alpha": alpha, "beta": beta, "residual": res}
        with self._served("triton", OP_SUM), ck.use_backend("triton"):
            s, got = ck.snake_beta_with_sum(x, alpha, beta, res)
        s_ref, ref = eager_backend.snake_beta_with_sum(**kwargs)
        assert torch.equal(s, s_ref)
        self._check_close(got, ref, _reference32(x, alpha, beta, res)[1])

    def test_long_row_offsets(self):
        # T past 2**31: the tile start no longer fits in int32. Needs two 4 GiB fp16 buffers;
        # only the first and the last n steps are filled and checked.
        t = 2**31 + 4096
        n = 8192
        if torch.cuda.mem_get_info()[0] < 9 * 2**30:
            pytest.skip("needs 9 GiB of free GPU memory")
        x = got = None
        try:
            x = torch.empty(1, 1, t, device="cuda", dtype=torch.float16)
            x[..., :n].normal_()
            x[..., -n:].normal_()
            alpha, beta = _params(1, device="cuda", dtype=torch.float16)
            with self._served("triton"), ck.use_backend("triton"):
                got = ck.snake_beta(x, alpha, beta)
            torch.cuda.synchronize()
            for sl in (slice(0, n), slice(t - n, t)):
                ref = eager_backend.snake_beta(x[..., sl], alpha, beta)
                assert _rel(got[..., sl], ref) < 2e-3
        finally:
            del x, got
            torch.cuda.empty_cache()

    def test_empty_input_goes_to_eager(self):
        x, alpha, beta, _ = self._inputs(8, 0, False)
        kwargs = {"x": x, "alpha": alpha, "beta": beta, "residual": None}
        assert not registry.validate_backend_for_call("triton", OP, kwargs).success
        with ck.use_backend("triton"):
            got = ck.snake_beta(x, alpha, beta)
        assert got.shape == (1, 8, 0) and got.dtype == torch.float16

    @pytest.mark.parametrize(
        "case",
        ["bf16", "fp32", "fp32_params", "fp32_residual", "x_step2", "x_transposed", "res_view"],
    )
    def test_unsupported_inputs_go_to_eager(self, case):
        x, alpha, beta, res = self._inputs(64, 300, True)
        if case == "bf16":
            x, alpha, beta, res = (t.to(torch.bfloat16) for t in (x, alpha, beta, res))
        elif case == "fp32":
            x, alpha, beta, res = (t.float() for t in (x, alpha, beta, res))
        elif case == "fp32_params":
            alpha, beta = alpha.float(), beta.float()
        elif case == "fp32_residual":
            res = res.float()
        elif case == "x_step2":
            x = torch.randn(1, 64, 600, device="cuda").half()[:, :, ::2]
        elif case == "x_transposed":
            x = torch.randn(1, 300, 64, device="cuda").half().transpose(1, 2)
        else:
            res = torch.randn(1, 64, 301, device="cuda").half()[:, :, 1:]
        kwargs = {"x": x, "alpha": alpha, "beta": beta, "residual": res}
        assert not registry.validate_backend_for_call("triton", OP_SUM, kwargs).success
        assert registry.get_capable_backend(OP_SUM, kwargs) == "eager"
        with ck.use_backend("triton"):
            s, got = ck.snake_beta_with_sum(x, alpha, beta, res)
            got_add = ck.snake_beta(x, alpha, beta, res)
        s_ref, ref = _stock(x, alpha, beta, res)
        assert torch.equal(s, s_ref) and torch.equal(got, ref) and torch.equal(got_add, ref)

    def test_cpu_parameters_go_to_eager(self):
        # SnakeBeta.forward moves its parameters to x's device; the kernel needs them there
        x, alpha, beta, _ = self._inputs(64, 300, False)
        alpha, beta = alpha.cpu(), beta.cpu()
        kwargs = {"x": x, "alpha": alpha, "beta": beta, "residual": None}
        assert registry.get_capable_backend(OP, kwargs) == "eager"
        with ck.use_backend("triton"):
            got = ck.snake_beta(x, alpha, beta)
        assert torch.equal(got, _stock(x, alpha, beta)[1])
