# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch.nn import functional

import comfy_kitchen as ck
from comfy_kitchen.deco_attention import _single_writer


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


def _tagged_rows(tags: torch.Tensor, priority: torch.Tensor) -> torch.Tensor:
    """Stratified deterministic draw; padding is negative and an empty draw repeats row zero."""
    count = max(64, tags.numel() // 256)
    groups = torch.stack((tags == 0, tags > 0))
    rows = torch.where(groups, priority, 1 << 30).topk(count, largest=False).indices
    video, other = groups.sum(dim=1, keepdim=True).unbind()
    take_video = torch.minimum(video, count - other.clamp(max=count // 2))
    drawn = take_video + torch.minimum(other, count - take_video)
    slot = torch.arange(count, device=tags.device) % drawn.clamp(min=1)
    index = torch.where(slot < take_video, slot, slot - take_video + count)
    return rows.flatten().gather(0, index).masked_fill(drawn == 0, 0).to(torch.int32)


@pytest.mark.cuda
@pytest.mark.skipif(not ck.deco_attention_is_available(), reason="requires native DeCo kernels")
@pytest.mark.parametrize("dtype", [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64])
def test_tag_draw(dtype):
    from comfy_kitchen.backends.cuda import _C, _wrap_for_dlpack

    length = 4149
    device = torch.device("cuda", torch.cuda.current_device())
    priority = torch.randperm(length, device=device).to(torch.int32)
    order = priority.argsort().to(torch.int32)
    ws = torch.empty(4 * ((length + 255) // 256), device=device, dtype=torch.uint8)
    rows = torch.empty(64, device=device, dtype=torch.int32)
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
        _C.deco_attention_sample_rows(
            *(_wrap_for_dlpack(t) for t in (tags, order, rows, ws)), stream
        )
        assert torch.equal(rows, _tagged_rows(tags, priority))


def test_tag_validation_and_cpu_availability():
    assert not ck.deco_attention_is_available("cpu")
    q = torch.zeros(1, 2, 5, 16)
    for tags in (torch.zeros(5), torch.zeros(4, dtype=torch.int32)):
        with pytest.raises(ValueError, match="token_tags"):
            ck.deco_attention(q, q, q, token_tags=tags)


def test_writer_selection():
    assert not _single_writer("NVIDIA RTX PRO 2000 Blackwell", 4352, 24)
    assert _single_writer("NVIDIA RTX PRO 2000 Blackwell", 22563, 56)
    assert not _single_writer("NVIDIA GeForce RTX 5080", 16481, 56)
    assert not _single_writer("NVIDIA GB10", 32768, 1)
    assert _single_writer("NVIDIA RTX PRO 6000 Blackwell", 16481, 56)


@pytest.mark.cuda
@pytest.mark.skipif(not ck.deco_attention_is_available(), reason="requires native DeCo kernels")
def test_deco_native_layout_capture_and_compile():
    from comfy_kitchen.backends.cuda import _C

    device = torch.cuda.current_device()
    assert _C.deco_attention_supported(4096, 8, device)
    assert not _C.deco_attention_supported(151, 20, device)
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
@pytest.mark.skipif(not ck.deco_attention_is_available(), reason="requires native DeCo kernels")
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
@pytest.mark.skipif(not ck.deco_attention_is_available(), reason="requires native DeCo kernels")
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
