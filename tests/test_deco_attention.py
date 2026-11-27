# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch.nn import functional

import comfy_kitchen as ck
from comfy_kitchen.deco_attention import _sampling_constants, _single_writer


requires_deco = pytest.mark.skipif(
    not ck.deco_attention_is_available(), reason="requires native DeCo kernels"
)


@pytest.mark.parametrize("mode", ["dense", "mask", "causal", "gqa", "dropout"])
def test_sdpa_fallback(mode):
    torch.manual_seed(9)
    q = torch.randn(2, 4, 11, 16, requires_grad=True)
    k = torch.randn(2, 2 if mode == "gqa" else 4, 13, 16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    kwargs = {"scale": 0.2}
    if mode == "mask":
        kwargs["attn_mask"] = torch.randn(11, 13)
    elif mode == "causal":
        kwargs["is_causal"] = True
    elif mode == "gqa":
        kwargs["enable_gqa"] = True
    elif mode == "dropout":
        kwargs["dropout_p"] = 0.3
    torch.manual_seed(17)
    got = ck.deco_attention(q, k, v, **kwargs)
    torch.manual_seed(17)
    ref = functional.scaled_dot_product_attention(q, k, v, **kwargs)
    torch.testing.assert_close(got, ref, rtol=0, atol=0)
    for a, b in zip(
        torch.autograd.grad(got.sum(), (q, k, v)),
        torch.autograd.grad(ref.sum(), (q, k, v)),
        strict=True,
    ):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def _expected_rows(tags, order):
    count = max(64, tags.numel() // 256)
    tags, positions = tags.tolist(), order.tolist()
    image = [i for i in positions if tags[i] == 0]
    other = [i for i in positions if tags[i] > 0]
    image = image[: count - min(count // 2, len(other))]
    draw = image + other[: count - len(image)]
    return order.new_tensor(((draw or [0]) * count)[:count])


@pytest.mark.cuda
@requires_deco
@pytest.mark.parametrize("dtype", [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64])
def test_tag_draw(dtype):
    from comfy_kitchen.backends.cuda import _C, _wrap_for_dlpack

    length, heads = 4096, 8
    device = torch.device("cuda", torch.cuda.current_device())
    order, rotation = _sampling_constants(length, device, True)
    q = torch.randn(1, heads, length, 128, dtype=torch.bfloat16, device=device)
    out = q.transpose(1, 2).contiguous()
    ws = torch.empty(
        _C.deco_attention_workspace_size(length, heads, device.index),
        device=device,
        dtype=torch.uint8,
    )
    writer = _single_writer(torch.cuda.get_device_name(device), length, heads)
    stream = torch.cuda.current_stream().cuda_stream
    storage = torch.zeros(2 * length, dtype=dtype, device=device)
    mixed = storage[::2]
    mixed[::3] = 1 << 42 if dtype == torch.int64 else 3
    fixtures = [mixed, mixed[:1].expand(length)]
    if dtype != torch.uint8:
        sparse = torch.full_like(mixed, -1)
        fixtures.append(sparse.clone())
        sparse[:11] = 0
        sparse[-5:] = 1
        fixtures.append(sparse)
    for tags in fixtures:
        rows = _expected_rows(tags, order)
        _C.deco_attention_forward(
            *(_wrap_for_dlpack(t) for t in (q, q, q, rows, rotation, out, ws)), writer, stream, None
        )
        assert torch.equal(ck.deco_attention(q, q, q, token_tags=tags), out.transpose(1, 2))


def test_tag_validation_and_cpu_availability():
    assert not ck.deco_attention_is_available("cpu")
    q = torch.zeros(1, 2, 5, 16)
    for tags in (
        torch.zeros(5),
        torch.zeros(4, dtype=torch.int32),
        torch.zeros(5, dtype=torch.int32).to_sparse(),
    ):
        with pytest.raises(ValueError, match="token_tags"):
            ck.deco_attention(q, q, q, token_tags=tags)


def test_rocm_device_with_matching_capability(monkeypatch):
    import importlib

    module = importlib.import_module("comfy_kitchen.deco_attention")
    monkeypatch.setattr(torch.version, "hip", "7.0")
    monkeypatch.setattr(module, "_ARCHITECTURES", ((12, 0),))
    monkeypatch.setattr(module.registry, "is_available", lambda backend: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    assert not ck.deco_attention_is_available("cuda")


def test_writer_selection():
    assert not _single_writer("NVIDIA RTX PRO 2000 Blackwell", 4352, 24)
    assert _single_writer("NVIDIA RTX PRO 2000 Blackwell", 22563, 56)
    assert not _single_writer("NVIDIA GeForce RTX 5080", 16481, 56)
    assert not _single_writer("NVIDIA GB10", 32768, 1)
    assert _single_writer("NVIDIA RTX PRO 6000 Blackwell", 16481, 56)


@pytest.mark.cuda
@requires_deco
def test_guards_on_native_geometry(monkeypatch):
    # Only one condition changes at a time, starting from an eligible geometry.
    q = torch.empty(1, 8, 4096, 128, device="cuda", dtype=torch.bfloat16)
    monkeypatch.setattr(
        functional, "scaled_dot_product_attention", lambda query, *args, **kw: query
    )
    for kw in (
        {"is_causal": True},
        {"attn_mask": q.new_ones(1)},
        {"scale": 0.2},
        {"dropout_p": 0.1},
        {"enable_gqa": True},
    ):
        assert ck.deco_attention(q, q, q, **kw) is q
    assert ck.deco_attention(q, q[:, :, :-1], q[:, :, :-1]) is q
    assert ck.deco_attention(q, q, q.detach().requires_grad_()) is q
    assert ck.deco_attention(q, q, q[..., :1].expand_as(q)) is q
    with torch.autocast("cuda", dtype=torch.float16):
        assert ck.deco_attention(q, q, q) is q
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(q, torch.ones_like(q))
        assert ck.deco_attention(dual, q, q) is dual
    ck.disable_backend("cuda")
    assert ck.deco_attention(q, q, q) is q


@pytest.mark.cuda
@requires_deco
def test_deco_native_layout_capture_and_compile():
    from comfy_kitchen.backends.cuda import _C

    device = torch.cuda.current_device()
    assert _C.deco_attention_workspace_size(4096, 8, device) > 0
    assert _C.deco_attention_workspace_size(151, 20, device) == 0
    g = torch.Generator(device="cuda").manual_seed(37)
    x = torch.randn(1, 4096, 8, 3, 128, generator=g, device="cuda", dtype=torch.bfloat16)
    q, k, v = (t.transpose(1, 2) for t in x.unbind(3))
    tags = torch.zeros((1, 4096), dtype=torch.int32, device="cuda")
    rng = torch.cuda.get_rng_state()
    out = ck.deco_attention(q, k, v, token_tags=tags)
    assert torch.equal(rng, torch.cuda.get_rng_state())
    assert out.shape == q.shape and out.transpose(1, 2).is_contiguous()
    contiguous = ck.deco_attention(q.contiguous(), k.contiguous(), v.contiguous(), token_tags=tags)
    assert torch.equal(out, contiguous)
    lazy_zero = torch._efficientzerotensor(v.shape, dtype=v.dtype, device=v.device)
    assert torch.equal(ck.deco_attention(q, k, lazy_zero, token_tags=tags), torch.zeros_like(out))
    lazy_negative = torch._neg_view(v)
    assert torch.equal(
        ck.deco_attention(q, k, lazy_negative, token_tags=tags),
        ck.deco_attention(q, k, -v, token_tags=tags),
    )
    ref = functional.scaled_dot_product_attention(q[:, :, ::256].float(), k.float(), v.float())
    assert (
        functional.cosine_similarity(out[:, :, ::256].float().flatten(), ref.flatten(), dim=0)
        > 0.95
    )
    assert torch.equal(
        torch.compile(ck.deco_attention, fullgraph=True, backend="eager")(q, k, v, token_tags=tags),
        out,
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay = ck.deco_attention(q, k, v, token_tags=tags)
    tags[:, ::3] = 1
    k.neg_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(replay, ck.deco_attention(q, k, v, token_tags=tags))
    assert not torch.equal(out, replay)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU SDPA")
@pytest.mark.parametrize("shape", [(1, 20, 151, 128), (151, 20, 12, 128)])
def test_short_trace_shapes_use_sdpa(shape):
    q, k, v = (torch.randn(shape, dtype=torch.bfloat16, device="cuda") for _ in range(3))
    ref = functional.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(ck.deco_attention(q, k, v), ref, rtol=0, atol=0)


@pytest.mark.cuda
@requires_deco
def test_native_calls_on_separate_streams():
    shape = (1, 8, 4096, 128)
    q, k, v = (torch.randn(shape, dtype=torch.bfloat16, device="cuda") for _ in range(3))
    inputs = [(q, k, v), (v, q, k)]
    expected = [ck.deco_attention(*tensors) for tensors in inputs]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    actual = []
    for stream, tensors in zip(streams, inputs, strict=True):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            actual.append(ck.deco_attention(*tensors))
    for stream, out, ref in zip(streams, actual, expected, strict=True):
        torch.cuda.current_stream().wait_stream(stream)
        torch.testing.assert_close(out, ref, rtol=0, atol=0)


@pytest.mark.cuda
@requires_deco
def test_dynamic_compile_routes_adjacent_lengths():
    compiled = torch.compile(ck.deco_attention, fullgraph=True, dynamic=True, backend="eager")
    x = torch.randn(1, 8, 4097, 128, device="cuda", dtype=torch.bfloat16)
    for sequence in (4096, 4097, 4096):
        q = x[:, :, :sequence]
        expected = (
            ck.deco_attention(q, q, q)
            if sequence == 4096
            else functional.scaled_dot_product_attention(q, q, q)
        )
        assert torch.equal(compiled(q, q, q), expected)
