# SPDX-License-Identifier: Apache-2.0

import gc
import weakref

import pytest
import torch

import comfy_kitchen as ck
import comfy_kitchen.sage_attention as sage_attention_module

_CUDA_READY = torch.cuda.is_available() and ck.int8_attention_is_available()
requires_int8_attention = pytest.mark.skipif(
    not _CUDA_READY,
    reason="requires the CUDA extension on an INT8-attention-capable GPU",
)


def _qkv(batch, q_heads, kv_heads, q_length, kv_length, head_dim, dtype=torch.bfloat16):
    q = torch.randn(batch, q_length, q_heads, head_dim, device="cuda", dtype=dtype).transpose(1, 2)
    k = torch.randn(batch, kv_length, kv_heads, head_dim, device="cuda", dtype=dtype).transpose(
        1, 2
    )
    v = torch.randn(batch, kv_length, kv_heads, head_dim, device="cuda", dtype=dtype).transpose(
        1, 2
    )
    return q, k, v


def _nrmse(actual, expected):
    error = (actual.float() - expected.float()).square().mean().sqrt()
    magnitude = expected.float().square().mean().sqrt()
    return (error / magnitude).item()


def test_int8_attention_availability_is_bool():
    assert isinstance(ck.int8_attention_is_available(), bool)


@pytest.mark.parametrize(
    ("capability", "expected"),
    [
        ((7, 5), True),
        ((7, 0), False),
        ((8, 0), True),
        ((8, 7), True),
        ((11, 0), True),
    ],
)
def test_int8_attention_capability_dispatch(monkeypatch, capability, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: capability)
    monkeypatch.setattr(sage_attention_module._cuda_backend, "_EXT_AVAILABLE", True)
    assert sage_attention_module.is_available() is expected


@requires_int8_attention
def test_int8_attention_allocates_only_integer_8bit_scratch(monkeypatch):
    q, k, v = _qkv(1, 4, 4, 129, 129, 64)
    allocated_dtypes = []
    original_empty = torch.empty

    def recording_empty(*args, **kwargs):
        allocated_dtypes.append(kwargs.get("dtype"))
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", recording_empty)
    ck.int8_attention(q, k, v)

    assert allocated_dtypes.count(torch.int8) == 3
    assert torch.float8_e4m3fn not in allocated_dtypes


@requires_int8_attention
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("head_dim", [1, 64, 96, 128, 192, 256])
def test_int8_attention_matches_sdpa(dtype, head_dim):
    q, k, v = _qkv(1, 8, 8, 257, 257, head_dim, dtype)
    assert not q.is_contiguous()

    actual = ck.int8_attention(q, k, v)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    assert actual.shape == expected.shape
    assert actual.dtype == dtype
    assert torch.isfinite(actual).all()
    assert _nrmse(actual, expected) < 0.03


@pytest.mark.parametrize("option", ["softmax_dtype", "post_pv_dtype"])
def test_int8_attention_has_no_low_precision_options(option):
    with pytest.raises(TypeError):
        ck.int8_attention(None, None, None, **{option: "input"})

@requires_int8_attention
def test_int8_attention_gqa_and_unequal_lengths():
    q, k, v = _qkv(1, 16, 4, 191, 257, 128)
    actual = ck.int8_attention(q, k, v, scale=0.07)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q,
        k.repeat_interleave(4, dim=1),
        v.repeat_interleave(4, dim=1),
        scale=0.07,
    )

    assert actual.shape == (1, 16, 191, 128)
    assert _nrmse(actual, expected) < 0.03


@requires_int8_attention
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("mask_dtype", [torch.bool, torch.float16, torch.bfloat16])
def test_int8_attention_mask_gqa_broadcast_and_fully_masked_row(head_dim, mask_dtype):
    q, k, v = _qkv(1, 8, 2, 193, 257, head_dim)
    if mask_dtype == torch.bool:
        mask = torch.rand(1, 1, 193, 257, device="cuda") > 0.15
        mask[..., 7, :] = False
    else:
        mask = torch.zeros(1, 1, 193, 257, device="cuda", dtype=mask_dtype)
        mask[..., 220:] = -torch.inf
        mask[..., 7, :] = -torch.inf

    actual = ck.int8_attention(q, k, v, attn_mask=mask, convrot=True)
    baseline_mask = mask
    if mask.dtype != torch.bool and mask.dtype != q.dtype:
        baseline_mask = mask.to(q.dtype)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q,
        k.repeat_interleave(4, dim=1),
        v.repeat_interleave(4, dim=1),
        attn_mask=baseline_mask,
    )

    assert torch.count_nonzero(actual[..., 7, :]) == 0
    assert torch.isfinite(actual).all()
    assert _nrmse(actual, expected) < 0.03


@requires_int8_attention
@pytest.mark.parametrize(
    "mask_dtype", [torch.bool, torch.float16, torch.bfloat16, torch.float32]
)
def test_int8_attention_key_broadcast_mask(mask_dtype):
    q, k, v = _qkv(1, 8, 2, 193, 257, 64)
    if mask_dtype == torch.bool:
        mask = torch.rand(1, 1, 1, 257, device="cuda") > 0.15
    else:
        mask = torch.linspace(-1, 1, 257, device="cuda", dtype=mask_dtype).reshape(
            1, 1, 1, 257
        )
        mask[..., 240:] = -torch.inf

    actual = ck.int8_attention(q, k, v, attn_mask=mask, convrot=True)
    baseline_mask = mask
    if mask.dtype != torch.bool and mask.dtype != q.dtype:
        baseline_mask = mask.to(q.dtype)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q,
        k.repeat_interleave(4, dim=1),
        v.repeat_interleave(4, dim=1),
        attn_mask=baseline_mask,
    )

    assert torch.isfinite(actual).all()
    assert _nrmse(actual, expected) < 0.03


@requires_int8_attention
@pytest.mark.parametrize("mask_dtype", [torch.bool, torch.bfloat16])
def test_int8_attention_fully_masked_key_broadcast_is_zero(mask_dtype):
    q, k, v = _qkv(1, 4, 4, 129, 97, 64)
    if mask_dtype == torch.bool:
        mask = torch.zeros(1, 1, 1, 97, dtype=torch.bool, device="cuda")
    else:
        mask = torch.full(
            (1, 1, 1, 97), -torch.inf, dtype=mask_dtype, device="cuda"
        )

    actual = ck.int8_attention(q, k, v, attn_mask=mask, convrot=True)

    assert torch.count_nonzero(actual) == 0


@requires_int8_attention
@pytest.mark.parametrize(
    "heads,length,head_dim",
    [(4, 257, 64), (24, 1024, 128), (56, 4096, 128)],
)
def test_int8_attention_causal(heads, length, head_dim):
    q, k, v = _qkv(1, heads, heads, length, length, head_dim)
    actual = ck.int8_attention(q, k, v, is_causal=True)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert _nrmse(actual, expected) < 0.03


@requires_int8_attention
def test_int8_attention_smooth_k():
    q, k, v = _qkv(1, 8, 8, 257, 257, 128)
    k.add_(torch.linspace(-2, 2, 128, device="cuda", dtype=k.dtype))
    actual = ck.int8_attention(q, k, v, smooth_k=True)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    assert _nrmse(actual, expected) < 0.03


@requires_int8_attention
@pytest.mark.parametrize(
    "configuration",
    [
        {
            "q_length": 257,
            "kv_length": 257,
            "head_dim": 64,
            "dtype": torch.float16,
        },
        {"q_length": 193, "kv_length": 1281, "head_dim": 128, "convrot": True},
        {
            "q_length": 193,
            "kv_length": 257,
            "head_dim": 96,
            "smooth_k": True,
            "dtype": torch.float32,
        },
        {"q_length": 257, "kv_length": 257, "head_dim": 256, "is_causal": True},
    ],
)
def test_prequantized_attention_is_bitwise_identical_to_fused(configuration):
    torch.manual_seed(123)
    options = {
        key: configuration[key]
        for key in ("convrot", "smooth_k", "is_causal")
        if key in configuration
    }
    q, k, v = _qkv(
        1,
        8,
        2,
        configuration["q_length"],
        configuration["kv_length"],
        configuration["head_dim"],
        configuration.get("dtype", torch.bfloat16),
    )

    expected = ck.int8_attention(q, k, v, **options)
    quantized = ck.prequantize_int8_attention(q, k, v, **options)
    actual = ck.int8_attention_from_prequantized(quantized)

    assert torch.equal(actual, expected)


@requires_int8_attention
def test_prequantized_masked_attention_is_bitwise_identical_to_fused():
    q, k, v = _qkv(1, 8, 2, 193, 257, 128)
    mask = torch.linspace(-1, 1, 257, device="cuda", dtype=torch.float32).reshape(
        1, 1, 1, 257
    )
    mask[..., 240:] = -torch.inf

    expected = ck.int8_attention(q, k, v, attn_mask=mask, convrot=True)
    quantized = ck.prequantize_int8_attention(
        q,
        k,
        v,
        attn_mask=mask,
        convrot=True,
    )
    actual = ck.int8_attention_from_prequantized(quantized)

    assert torch.equal(actual, expected)


@requires_int8_attention
def test_prequantized_attention_releases_float_inputs_before_execution():
    q, k, v = _qkv(1, 8, 2, 513, 769, 128)
    expected = ck.int8_attention(q, k, v, convrot=True)
    input_references = tuple(weakref.ref(tensor) for tensor in (q, k, v))

    quantized = ck.prequantize_int8_attention(q, k, v, convrot=True)
    del q, k, v
    gc.collect()
    assert all(reference() is None for reference in input_references)

    # Force allocator reuse on the same stream before consuming the packed
    # tensors. This catches a split implementation that only appears correct
    # while its asynchronous quantization inputs remain allocated.
    allocator_churn = torch.empty(
        64 * 1024 * 1024,
        dtype=torch.uint8,
        device="cuda",
    )
    allocator_churn.fill_(0xA5)
    actual = ck.int8_attention_from_prequantized(quantized)

    assert torch.equal(actual, expected)


@requires_int8_attention
def test_int8_attention_torch_compile_fullgraph():
    q, k, v = _qkv(1, 4, 4, 129, 129, 64)
    compiled = torch.compile(
        lambda q_, k_, v_: ck.int8_attention(q_, k_, v_, convrot=True),
        backend="eager",
        fullgraph=True,
    )
    actual = compiled(q, k, v)
    expected = ck.int8_attention(q, k, v, convrot=True)
    torch.testing.assert_close(actual, expected)


@requires_int8_attention
def test_int8_attention_cuda_graph():
    q, k, v = _qkv(1, 4, 4, 129, 129, 64)
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        ck.int8_attention(q, k, v)
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = ck.int8_attention(q, k, v)
    graph.replay()
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    assert _nrmse(actual, expected) < 0.03


@requires_int8_attention
def test_convrot_improves_outlier_quality():
    torch.manual_seed(1)
    q, k, v = _qkv(1, 8, 8, 513, 513, 128)
    q[..., 0].mul_(12)
    k[..., 0].mul_(12)
    q.mul_(q.float().square().mean(-1, keepdim=True).rsqrt().to(q.dtype))
    k.mul_(k.float().square().mean(-1, keepdim=True).rsqrt().to(k.dtype))
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    baseline = ck.int8_attention(q, k, v)
    rotated = ck.int8_attention(q, k, v, convrot=True)

    assert _nrmse(rotated, expected) < _nrmse(baseline, expected) * 0.95


@requires_int8_attention
def test_causal_rejects_unequal_lengths():
    q, k, v = _qkv(1, 4, 4, 64, 96, 64)
    with pytest.raises(ValueError, match="equal q and k/v sequence lengths"):
        ck.int8_attention(q, k, v, is_causal=True)
