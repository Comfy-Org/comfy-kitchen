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
    reason="requires a CUDA or HIP extension on an INT8-attention-capable GPU",
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


def test_int8_attention_cta_k_selection():
    select = sage_attention_module._select_cta_k
    assert select(128, 1025, has_mask=False) == 128
    assert select(64, 1025, has_mask=False) == 64
    assert select(128, 1025, has_mask=True) == 64


def test_prequantized_attention_rejects_cpu_tensors():
    packed = sage_attention_module.PrequantizedInt8Attention(
        q=torch.empty(1, 1, 1, 64, dtype=torch.int8),
        k=torch.empty(1, 1, 1, 64, dtype=torch.int8),
        v=torch.empty(64, 64, dtype=torch.int8),
        q_scale=torch.empty(1, dtype=torch.float32),
        k_scale=torch.empty(1, dtype=torch.float32),
        v_scale=torch.empty(64, dtype=torch.float32),
        original_head_dim=64,
        input_dtype=torch.float16,
        attention_scale=0.125,
        cta_k=64,
        attn_mask=None,
    )

    with pytest.raises(ValueError, match="CUDA device"):
        ck.int8_attention_from_prequantized(packed)


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
    if getattr(torch.version, "hip", None):
        pytest.skip("compute capability does not gate the HIP path")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: capability)
    monkeypatch.setattr(sage_attention_module._cuda_backend, "_EXT_AVAILABLE", True)
    assert sage_attention_module.is_available() is expected


@pytest.mark.parametrize("has_wmma", [True, False])
def test_int8_attention_hip_dispatch_follows_matrix_cores(monkeypatch, has_wmma):
    """On ROCm the gate is matrix cores, not a compute capability.

    torch.cuda is the ROCm API there and reports an SM-shaped capability for a
    gfx part, so the CUDA test above would wave RDNA2 through to a kernel built
    on WMMA. RDNA2 has none and must decline.
    """
    if not getattr(torch.version, "hip", None):
        pytest.skip("requires a ROCm PyTorch runtime")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(sage_attention_module._hip_backend, "has_wmma", lambda: has_wmma)
    assert sage_attention_module.is_available() is has_wmma


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
    assert allocated_dtypes.count(torch.int32) == 1
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


@pytest.mark.parametrize(
    "option",
    [
        "softmax_dtype",
        "post_pv_dtype",
        "convrot",
        "stabilize_k",
        "smooth_k",
        "is_causal",
    ],
)
def test_int8_attention_rejects_removed_options(option):
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
@pytest.mark.parametrize("masked", [False, True])
def test_int8_attention_batch_two_direct_and_prequantized(masked):
    q, k, v = _qkv(2, 8, 2, 193, 257, 128)
    mask = None
    if masked:
        mask = torch.zeros(2, 1, 1, 257, device="cuda", dtype=torch.bfloat16)
        mask[0, ..., 240:] = -torch.inf
        mask[1, ..., :17] = -torch.inf

    actual = ck.int8_attention(q, k, v, attn_mask=mask)
    quantized = ck.prequantize_int8_attention(q, k, v, attn_mask=mask)
    prequantized = ck.int8_attention_from_prequantized(quantized)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q,
        k.repeat_interleave(4, dim=1),
        v.repeat_interleave(4, dim=1),
        attn_mask=mask,
    )

    assert torch.equal(prequantized, actual)
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

    actual = ck.int8_attention(q, k, v, attn_mask=mask)
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
@pytest.mark.parametrize("mask_dtype", [torch.bool, torch.float16, torch.bfloat16, torch.float32])
def test_int8_attention_key_broadcast_mask(mask_dtype):
    q, k, v = _qkv(1, 8, 2, 193, 257, 64)
    if mask_dtype == torch.bool:
        mask = torch.rand(1, 1, 1, 257, device="cuda") > 0.15
    else:
        mask = torch.linspace(-1, 1, 257, device="cuda", dtype=mask_dtype).reshape(1, 1, 1, 257)
        mask[..., 240:] = -torch.inf

    actual = ck.int8_attention(q, k, v, attn_mask=mask)
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
        mask = torch.full((1, 1, 1, 97), -torch.inf, dtype=mask_dtype, device="cuda")

    actual = ck.int8_attention(q, k, v, attn_mask=mask)

    assert torch.count_nonzero(actual) == 0


@requires_int8_attention
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("mask_dtype", [torch.bool, torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("key_only", [False, True])
def test_int8_attention_long_masked_sequence(head_dim, mask_dtype, key_only):
    """Cover multiple K tiles, a partial tail, GQA, and noncontiguous masks."""
    torch.manual_seed(177)
    batch, heads, q_length, kv_length = 2, 4, 137, 1153
    q, k, v = _qkv(batch, heads, 2, q_length, kv_length, head_dim)
    mask_rows = 1 if key_only else q_length
    bias = torch.randn(batch, heads, mask_rows, kv_length * 2, device="cuda")
    if mask_dtype == torch.bool:
        mask = bias > -0.5
        masked_value = False
    else:
        mask = bias.to(mask_dtype)
        masked_value = -torch.inf
    mask = mask[..., ::2]
    assert mask.stride(-1) == 2
    # All-masked initial tiles must not pollute later valid keys. Also cover
    # a fully masked head (or row), including the partial final tile.
    mask[..., :128] = masked_value
    mask[0, 1, 0, :] = masked_value
    mask[..., -17:] = masked_value

    actual = ck.int8_attention(q, k, v, attn_mask=mask)
    packed = ck.prequantize_int8_attention(q, k, v, attn_mask=mask)
    split = ck.int8_attention_from_prequantized(packed)
    baseline_mask = mask if mask_dtype == torch.bool else mask.to(q.dtype)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q,
        k.repeat_interleave(2, dim=1),
        v.repeat_interleave(2, dim=1),
        attn_mask=baseline_mask,
    )

    assert torch.equal(actual, split)
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual[0, 1] if key_only else actual[0, 1, 0]) == 0
    assert _nrmse(actual, expected) < 0.03


@requires_int8_attention
def test_int8_attention_stabilizes_large_common_key_component():
    torch.manual_seed(7)
    q, k, v = _qkv(1, 16, 16, 513, 513, 128)
    common_key = torch.randn(1, 16, 1, 128, device="cuda", dtype=torch.float32)
    common_key.mul_(40.0 / common_key.square().mean(-1, keepdim=True).sqrt())
    k.add_(common_key.to(k.dtype))
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    actual = ck.int8_attention(q, k, v)

    assert torch.isfinite(actual).all()
    assert _nrmse(actual, expected) < 0.03


@requires_int8_attention
def test_int8_attention_stabilization_is_deterministic():
    torch.manual_seed(11)
    q, k, v = _qkv(1, 8, 8, 257, 257, 128)

    first = ck.int8_attention(q, k, v)
    second = ck.int8_attention(q, k, v)

    assert torch.equal(first, second)


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
        {"q_length": 193, "kv_length": 1281, "head_dim": 128},
        {
            "q_length": 193,
            "kv_length": 257,
            "head_dim": 96,
            "dtype": torch.float32,
        },
        {
            "q_length": 193,
            "kv_length": 1281,
            "head_dim": 256,
        },
        {"q_length": 257, "kv_length": 257, "head_dim": 256},
    ],
)
def test_prequantized_attention_is_bitwise_identical_to_fused(configuration):
    torch.manual_seed(123)
    q, k, v = _qkv(
        1,
        8,
        2,
        configuration["q_length"],
        configuration["kv_length"],
        configuration["head_dim"],
        configuration.get("dtype", torch.bfloat16),
    )

    expected = ck.int8_attention(q, k, v)
    quantized = ck.prequantize_int8_attention(q, k, v)
    actual = ck.int8_attention_from_prequantized(quantized)

    assert torch.equal(actual, expected)


@requires_int8_attention
def test_prequantized_masked_attention_is_bitwise_identical_to_fused():
    q, k, v = _qkv(1, 8, 2, 193, 257, 128)
    mask = torch.linspace(-1, 1, 257, device="cuda", dtype=torch.float32).reshape(1, 1, 1, 257)
    mask[..., 240:] = -torch.inf

    expected = ck.int8_attention(q, k, v, attn_mask=mask)
    quantized = ck.prequantize_int8_attention(
        q,
        k,
        v,
        attn_mask=mask,
    )
    actual = ck.int8_attention_from_prequantized(quantized)

    assert torch.equal(actual, expected)


@requires_int8_attention
def test_prequantized_attention_releases_float_inputs_before_execution():
    q, k, v = _qkv(1, 8, 2, 513, 769, 128)
    expected = ck.int8_attention(q, k, v)
    input_references = tuple(weakref.ref(tensor) for tensor in (q, k, v))

    quantized = ck.prequantize_int8_attention(q, k, v)
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
        lambda q_, k_, v_: ck.int8_attention(q_, k_, v_),
        backend="eager",
        fullgraph=True,
    )
    actual = compiled(q, k, v)
    expected = ck.int8_attention(q, k, v)
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
def test_rotation_handles_outliers():
    torch.manual_seed(1)
    q, k, v = _qkv(1, 8, 8, 513, 513, 128)
    q[..., 0].mul_(12)
    k[..., 0].mul_(12)
    q.mul_(q.float().square().mean(-1, keepdim=True).rsqrt().to(q.dtype))
    k.mul_(k.float().square().mean(-1, keepdim=True).rsqrt().to(k.dtype))
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    actual = ck.int8_attention(q, k, v)

    assert _nrmse(actual, expected) < 0.03


@requires_int8_attention
@pytest.mark.parametrize("scale", [None, 0.0, -(128**-0.5)])
def test_int8_attention_long_sequence_and_partial_tile(scale):
    torch.manual_seed(31)
    q, k, v = _qkv(1, 4, 4, 129, 8193, 128)
    actual = ck.int8_attention(q, k, v, scale=scale)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)

    assert torch.isfinite(actual).all()
    assert _nrmse(actual, expected) < 0.03


@requires_int8_attention
def test_int8_attention_long_sequence_preserves_constant_values():
    torch.manual_seed(32)
    q, k, v = _qkv(1, 4, 4, 129, 8193, 128)
    constant = torch.linspace(-3, 3, 128, device="cuda", dtype=v.dtype)
    constant[0] = 0
    v.copy_(constant)

    actual = ck.int8_attention(q, k, v)

    # Backends that sum probabilities before U8 rounding can introduce a small
    # normalization error; zero-valued channels must still remain exactly zero.
    torch.testing.assert_close(actual, constant.expand_as(actual), rtol=0.005, atol=0)


@requires_int8_attention
def test_int8_attention_rescales_across_large_increases_in_logits():
    torch.manual_seed(33)
    q, k, v = _qkv(1, 4, 4, 129, 1025, 128)
    direction = q[:, :, :1].clone()
    q.copy_(direction)
    steps = (torch.arange(1025, device="cuda") // 64).to(torch.float32) * 8
    k.add_((direction.float() * steps[:, None] / (128**0.5)).to(k.dtype))

    actual = ck.int8_attention(q, k, v)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    assert torch.isfinite(actual).all()
    assert _nrmse(actual, expected) < 0.03


@requires_int8_attention
def test_int8_attention_accepts_dlpack_normalized_batch_stride():
    """A size-one extent carries no address, so its stride must not be policed.

    PyTorch rewrites the stride of any size-one dimension to 1 on the way out
    through DLPack (ATen/DLConvertor.cpp, gh-83069), so a batch-one attention
    input arrives with stride 1 no matter what the caller built.
    """
    torch.manual_seed(0)
    length, heads, head_dim = 372, 8, 128
    packed = [
        torch.randn(length, heads, head_dim, dtype=torch.bfloat16, device="cuda") for _ in range(3)
    ]
    reported = [t.transpose(0, 1).unsqueeze(0) for t in packed]
    normalized = [
        torch.as_strided(t, (1, heads, length, head_dim), (1, head_dim, heads * head_dim, 1))
        for t in packed
    ]
    assert normalized[0].stride(0) == 1
    assert torch.equal(reported[0], normalized[0])

    expected = ck.int8_attention_from_prequantized(ck.prequantize_int8_attention(*reported))
    actual = ck.int8_attention_from_prequantized(ck.prequantize_int8_attention(*normalized))

    assert torch.equal(actual, expected)


@requires_int8_attention
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("bias_kind", ["fade", "constant_tiles", "fully_masked"])
@pytest.mark.parametrize("head_dim", [32, 64, 96, 128, 160, 256])
def test_int8_attention_prepared_key_bias(dtype, bias_kind, head_dim):
    torch.manual_seed(178)
    q, k, v = _qkv(2, 4, 2, 129, 33601, head_dim, dtype)
    keys = torch.arange(k.shape[2], device="cuda", dtype=torch.float32)
    if bias_kind == "fade":
        phase = ((keys / (k.shape[2] - 1) - 0.6) / 0.4).clamp(0, 1)
        bias = ((1 + (phase * torch.pi).cos()) * 0.5).clamp_min(1e-4).log()
    elif bias_kind == "constant_tiles":
        # Different constants must affect the relative weights of tiles.
        bias = ((keys // 128) % 7 - 3) * 2
        bias[:128] = -torch.inf
        bias[-129:] = -torch.inf
    else:
        bias = torch.full_like(keys, -torch.inf)
    mask = bias.to(dtype).reshape(1, 1, 1, -1).expand(2, 4, q.shape[2], -1)

    packed = ck.prequantize_int8_attention(q, k, v, attn_mask=mask)
    actual = ck.int8_attention_from_prequantized(packed)
    direct = ck.int8_attention(q, k, v, attn_mask=mask)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q,
        k.repeat_interleave(2, dim=1),
        v.repeat_interleave(2, dim=1),
        attn_mask=mask,
    )

    if not getattr(torch.version, "hip", None):
        assert packed.cta_k == (64 if head_dim <= 64 else 128)
        assert packed.attn_mask.shape == (1, 1, 33928)
    else:
        assert packed.cta_k == 64
        assert packed.attn_mask.shape == (1, 1, 34192)
    assert torch.equal(actual, direct)
    assert torch.isfinite(actual).all()
    if bias_kind == "fully_masked":
        assert torch.count_nonzero(actual) == 0
    else:
        assert _nrmse(actual, expected) < 0.03


@requires_int8_attention
@pytest.mark.parametrize("head_dim", range(1, 257))
def test_int8_attention_prepared_mask_all_head_dimensions(head_dim):
    torch.manual_seed(179)
    q, k, v = _qkv(2, 4, 2, 137, 1153, head_dim)
    mask = torch.zeros(1, 1, 1, k.shape[2], dtype=q.dtype, device="cuda")
    mask[..., 512:] = torch.linspace(-1, -4, k.shape[2] - 512, device="cuda")
    mask[..., :128] = -torch.inf
    packed = ck.prequantize_int8_attention(q, k, v, attn_mask=mask)
    actual = ck.int8_attention_from_prequantized(packed)
    direct = ck.int8_attention(q, k, v, attn_mask=mask)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q,
        k.repeat_interleave(2, dim=1),
        v.repeat_interleave(2, dim=1),
        attn_mask=mask,
    )

    assert packed.attn_mask.ndim == 3
    assert actual.shape == q.shape
    assert torch.isfinite(actual).all()
    assert torch.equal(actual, direct)
    assert _nrmse(actual, expected) < 0.03


@requires_int8_attention
def test_int8_attention_prepared_key_mask_compile_and_graph():
    q, k, v = _qkv(1, 4, 2, 129, 1153, 128)
    mask = torch.linspace(-3, 0, k.shape[2], device="cuda").reshape(1, 1, 1, -1)
    compiled = torch.compile(ck.int8_attention, backend="eager", fullgraph=True)
    expected = ck.int8_attention(q, k, v, attn_mask=mask)
    torch.testing.assert_close(compiled(q, k, v, attn_mask=mask), expected)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        ck.int8_attention(q, k, v, attn_mask=mask)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = ck.int8_attention(q, k, v, attn_mask=mask)
    graph.replay()
    torch.testing.assert_close(actual, expected)
    # Preparation must run again on replay, rather than cache stale mask data.
    mask.fill_(-torch.inf)
    graph.replay()
    assert torch.count_nonzero(actual) == 0


@requires_int8_attention
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_int8_attention_prepared_mask_nonfinite_entries(dtype):
    q, k, v = _qkv(1, 4, 2, 65, 1153, 128)
    mask = torch.zeros(1, 1, 1, 1153, device="cuda", dtype=dtype)
    mask[..., :128] = torch.nan
    mask[..., 256:384] = torch.inf
    mask[..., -128:] = -torch.inf
    sanitized = torch.where(torch.isfinite(mask), mask, -torch.inf)
    actual = ck.int8_attention(q, k, v, attn_mask=mask)
    expected = ck.int8_attention(q, k, v, attn_mask=sanitized)
    assert torch.isfinite(actual).all()
    assert torch.equal(actual, expected)


@requires_int8_attention
def test_int8_attention_prepared_mask_releases_original():
    q, k, v = _qkv(1, 4, 2, 65, 1153, 128)
    mask = torch.zeros(1, 1, 1, k.shape[2], device="cuda")
    mask_ref = weakref.ref(mask)
    packed = ck.prequantize_int8_attention(q, k, v, attn_mask=mask)
    expected = ck.int8_attention(q, k, v, attn_mask=mask)

    # The prequantized snapshot owns its prepared buffer, not the input mask.
    mask.fill_(-torch.inf)
    del mask
    gc.collect()
    assert mask_ref() is None
    actual = ck.int8_attention_from_prequantized(packed)
    assert torch.count_nonzero(actual) > 0
    assert torch.equal(actual, expected)


@requires_int8_attention
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("mask_shape", [(2, 1), (1, 4), (2, 4)])
def test_int8_attention_prepared_bias_batch_head_strides(head_dim, mask_shape):
    """Compact batch/head rows must keep independent tile biases and empty rows."""
    torch.manual_seed(180)
    q, k, v = _qkv(2, 4, 2, 129, 1153, head_dim)
    mask_batch, mask_heads = mask_shape
    storage = torch.empty(mask_heads, mask_batch, 1, 2306, device="cuda")
    mask = storage.transpose(0, 1)[..., ::2]
    keys = torch.arange(1153, device="cuda")
    for batch in range(mask_batch):
        for head in range(mask_heads):
            mask[batch, head, 0] = ((keys // 64 + batch + head) % 5 - 2) * 3
    mask[0, 0] = -torch.inf
    # Skip full tiles, then visit a mixed tile containing only the final key.
    mask[-1, -1, :, :-1] = -torch.inf
    packed = ck.prequantize_int8_attention(q, k, v, attn_mask=mask)
    assert packed.attn_mask.ndim == 3
    assert packed.attn_mask.shape[:2] == mask_shape
    actual = ck.int8_attention_from_prequantized(packed)
    direct = ck.int8_attention(q, k, v, attn_mask=mask)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.float(),
        k.float().repeat_interleave(2, dim=1),
        v.float().repeat_interleave(2, dim=1),
        attn_mask=mask,
    )
    assert torch.equal(actual, direct)
    assert torch.isfinite(actual).all()
    assert _nrmse(actual, expected) < 0.03
    assert torch.count_nonzero(actual.masked_select((expected == 0).all(-1, keepdim=True))) == 0


@requires_int8_attention
@pytest.mark.parametrize("scale", [0.0, -0.125])
@pytest.mark.parametrize("mask_dtype", [torch.bool, torch.float32])
def test_int8_attention_key_mask_nonpositive_scale(scale, mask_dtype):
    torch.manual_seed(181)
    q, k, v = _qkv(1, 4, 2, 129, 1153, 128)
    mask = torch.arange(1153, device="cuda").reshape(1, 1, 1, -1) >= 256
    if mask_dtype != torch.bool:
        mask = torch.where(mask, torch.linspace(-3, 2, 1153, device="cuda"), -torch.inf)
    packed = ck.prequantize_int8_attention(q, k, v, scale=scale, attn_mask=mask)
    assert packed.attn_mask.ndim == (3 if torch.version.hip else 4)
    actual = ck.int8_attention_from_prequantized(packed)
    direct = ck.int8_attention(q, k, v, scale=scale, attn_mask=mask)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.float(),
        k.float().repeat_interleave(2, dim=1),
        v.float().repeat_interleave(2, dim=1),
        scale=scale,
        attn_mask=mask,
    )
    assert torch.equal(actual, direct)
    assert _nrmse(actual, expected) < 0.03


@requires_int8_attention
@pytest.mark.skipif(not torch.version.hip, reason="HIP fused short key masks")
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("q_length", [1, 129])
@pytest.mark.parametrize("kv_length", [65, 2048])
@pytest.mark.parametrize("mask_dtype", [torch.bool, torch.float32])
def test_hip_short_key_mask_matches_prequantized_snapshot(
    head_dim, q_length, kv_length, mask_dtype
):
    torch.manual_seed(182)
    q, k, v = _qkv(2, 4, 2, q_length, kv_length, head_dim)
    storage = torch.randn(2, 4, 1, kv_length * 2, device="cuda")
    if mask_dtype == torch.bool:
        storage = storage > 0
    mask = storage[..., ::2]
    masked_value = False if mask_dtype == torch.bool else -torch.inf
    mask[..., :64] = masked_value
    mask[0, 0] = masked_value
    if kv_length > 512:
        mask[..., 448:512] = masked_value

    packed = ck.prequantize_int8_attention(q, k, v, attn_mask=mask)
    expected = ck.int8_attention_from_prequantized(packed)
    actual = ck.int8_attention(q, k, v, attn_mask=mask)
    assert torch.equal(actual, expected)
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual[0, 0]) == 0


@requires_int8_attention
@pytest.mark.skipif(not torch.version.hip, reason="HIP prepared-mask binding")
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scale", [0.0, -0.125, 0.125, 1e-10])
@pytest.mark.parametrize("mask_shape", [(1, 1), (2, 1), (1, 4), (2, 4)])
@pytest.mark.parametrize("kv_length", [257, 4097])
def test_hip_compact_dense_bool_matches_float_bias(head_dim, dtype, scale, mask_shape, kv_length):
    torch.manual_seed(183)
    q, k, v = _qkv(2, 4, 2, 129, kv_length, head_dim, dtype)
    mask_batch, mask_heads = mask_shape
    storage = torch.rand(mask_heads, mask_batch, 129, kv_length * 2, device="cuda") > 0.15
    mask = storage.transpose(0, 1)[..., ::2]
    mask[..., 0, :] = False
    mask[..., -1, :] = False
    mask[..., -1, -1] = True
    bias = torch.where(mask, 0.0, -torch.inf)
    packed = ck.prequantize_int8_attention(q, k, v, scale=scale, attn_mask=mask)
    assert packed.attn_mask.dtype == torch.int32
    assert packed.attn_mask.shape[-1] == 32
    assert packed.attn_mask.shape[:2] == mask_shape
    expected = ck.int8_attention(q, k, v, scale=scale, attn_mask=bias)
    direct = ck.int8_attention(q, k, v, scale=scale, attn_mask=mask)
    # The compact representation must also retain the prequantized snapshot.
    mask.fill_(False)
    actual = ck.int8_attention_from_prequantized(packed)
    assert torch.equal(actual, expected)
    assert torch.equal(direct, expected)
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual[:, :, 0]) == 0


@requires_int8_attention
@pytest.mark.skipif(not torch.version.hip, reason="HIP prepared-mask binding")
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("mask_dtype", [torch.bool, torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("kv_length", [65, 257])
def test_hip_dense_mask_empty_tiles(head_dim, mask_dtype, kv_length):
    """Empty tiles must not poison other tiles or fully masked rows (#214)."""
    torch.manual_seed(214)
    q, k, v = _qkv(2, 4, 2, 17, kv_length, head_dim, torch.float16)
    keep = torch.ones(1, 1, 17, kv_length, device="cuda", dtype=torch.bool)
    keep[..., 0, :] = False
    keep[..., 1, -1] = False  # The final tile contains only this masked key.
    keep[..., 2:5, :] = False
    keep[..., 2, -1] = True  # Empty initial tiles, then a valid final key.
    keep[..., 3, 0] = True  # A valid first key, then empty tiles.
    keep[..., 4, 0] = True
    keep[..., 4, -1] = True  # Empty interior tiles between two valid keys.
    mask = keep if mask_dtype == torch.bool else torch.where(keep, 0.0, -torch.inf).to(mask_dtype)

    # Zero scale isolates mask/softmax handling from QK quantization error.
    packed = ck.prequantize_int8_attention(q, k, v, scale=0.0, attn_mask=mask)
    direct = ck.int8_attention(q, k, v, scale=0.0, attn_mask=mask)
    snapshot = ck.int8_attention_from_prequantized(packed)
    reference = torch.nn.functional.scaled_dot_product_attention(
        q.float(),
        k.float().repeat_interleave(2, dim=1),
        v.float().repeat_interleave(2, dim=1),
        attn_mask=keep,
        scale=0.0,
    )
    assert torch.isfinite(direct).all()
    assert torch.equal(direct, snapshot)
    assert torch.count_nonzero(direct[:, :, 0]) == 0
    assert _nrmse(direct, reference) < 0.02


@requires_int8_attention
@pytest.mark.skipif(not torch.version.hip, reason="HIP prepared-mask binding")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_hip_compact_dense_bool_rejects_additive_mask(dtype):
    hip = sage_attention_module._hip_backend
    mask = torch.zeros(1, 1, 17, 65, device="cuda", dtype=dtype)
    output = torch.full((1, 1, 2, 2, 32), 17, device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError, match="bit-packed preparation requires a Boolean mask"):
        hip._C.sage_prepare_dense_mask(
            hip._dl(mask), hip._dl(output), torch.cuda.current_stream().cuda_stream
        )
    assert torch.all(output == 17)


@requires_int8_attention
@pytest.mark.skipif(not torch.version.hip, reason="HIP prepared-mask binding")
@pytest.mark.parametrize("mask_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("strided", [False, True])
def test_hip_dense_16_preparation_preserves_finite_bits(mask_dtype, strided):
    hip = sage_attention_module._hip_backend
    torch.manual_seed(184)
    storage = torch.randn(2, 3, 17, 130, device="cuda", dtype=mask_dtype)
    mask = storage[..., ::2] if strided else storage[..., :65].contiguous()
    values = torch.tensor(
        [
            0.0,
            -0.0,
            torch.finfo(mask_dtype).tiny * torch.finfo(mask_dtype).eps,
            -torch.finfo(mask_dtype).tiny,
            torch.finfo(mask_dtype).max,
            -torch.finfo(mask_dtype).max,
            torch.inf,
            -torch.inf,
            torch.nan,
        ],
        device="cuda",
        dtype=mask_dtype,
    )
    mask[..., : values.numel()] = values
    prepared = torch.empty(2, 3, 2, 2, 1024, device="cuda", dtype=mask_dtype)
    hip._C.sage_prepare_dense_mask(hip._dl(mask), hip._dl(prepared), hip._stream(mask))
    decoded = prepared.reshape(2, 3, 2, 2, 4, 2, 16, 8).permute(0, 1, 2, 6, 3, 4, 5, 7)
    decoded = decoded.reshape(2, 3, 32, 128)
    expected = torch.full_like(decoded, -torch.inf)
    expected[..., :17, :65] = mask.masked_fill(~torch.isfinite(mask), -torch.inf)
    assert torch.equal(decoded.view(torch.int16), expected.view(torch.int16))


@requires_int8_attention
@pytest.mark.skipif(not torch.version.hip, reason="HIP prepared-mask binding")
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mask_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scale", [0.0, -0.125, 0.125, 1e-10])
@pytest.mark.parametrize("mask_shape", [(1, 1), (2, 4)])
@pytest.mark.parametrize("q_length", [129, 273])
def test_hip_dense_16_accuracy_and_snapshot(
    head_dim, dtype, mask_dtype, scale, mask_shape, q_length
):
    torch.manual_seed(185)
    q, k, v = _qkv(2, 4, 2, q_length, 257, head_dim, dtype)
    storage = torch.randn(*mask_shape, q_length, 514, device="cuda", dtype=mask_dtype)
    mask = storage[..., ::2]
    mask[..., 0, :] = -torch.inf
    mask[..., -1, :] = -torch.inf
    mask[..., -1, -1] = 60000.0 if mask_dtype == torch.float16 else 1e10
    packed = ck.prequantize_int8_attention(q, k, v, scale=scale, attn_mask=mask)
    assert packed.attn_mask.dtype == mask_dtype
    assert packed.attn_mask.shape[-1] == 1024
    reference = torch.nn.functional.scaled_dot_product_attention(
        q.float(),
        k.float().repeat_interleave(2, dim=1),
        v.float().repeat_interleave(2, dim=1),
        attn_mask=mask.float(),
        scale=scale,
    )
    baseline = ck.int8_attention(q, k, v, scale=scale, attn_mask=mask.float())
    direct = ck.int8_attention(q, k, v, scale=scale, attn_mask=mask)
    mask.fill_(-torch.inf)
    snapshot = ck.int8_attention_from_prequantized(packed)
    assert torch.equal(direct, snapshot)
    assert torch.isfinite(direct).all()
    assert torch.count_nonzero(direct[:, :, 0]) == 0
    assert _nrmse(direct, reference) <= _nrmse(baseline, reference) + 0.0002


@requires_int8_attention
@pytest.mark.skipif(not torch.version.hip, reason="HIP prepared-mask binding")
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mask_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scale", [0.0, -0.125, 0.125])
def test_hip_dense_16_preserves_constant_values(head_dim, dtype, mask_dtype, scale):
    torch.manual_seed(186)
    q, k, v = _qkv(1, 4, 2, 33, 1153, head_dim, dtype)
    v.fill_(1)
    v[..., 0] = 0
    v[..., 1] = -0.5
    mask = torch.randn(1, 1, 33, 1153, device="cuda", dtype=mask_dtype)
    mask[..., 0, :] = -torch.inf
    mask[..., 128:448] = -torch.inf
    actual = ck.int8_attention(q, k, v, scale=scale, attn_mask=mask)
    expected = v[:, :1, :1].expand_as(actual).clone()
    expected[:, :, 0] = 0
    assert torch.equal(actual, expected)


@requires_int8_attention
@pytest.mark.skipif(not torch.version.hip, reason="HIP fused dense masks")
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mask_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("lengths", [(2, 65), (256, 2048)])
@pytest.mark.parametrize("mask_shape", [(1, 1), (2, 1), (1, 4), (2, 4)])
def test_hip_fused_dense_mask_matches_snapshot(
    head_dim, dtype, mask_dtype, lengths, mask_shape, monkeypatch
):
    hip = sage_attention_module._hip_backend
    q_length, kv_length = lengths
    torch.manual_seed(187)
    q, k, v = _qkv(2, 4, 2, q_length, kv_length, head_dim, dtype)
    mask_batch, mask_heads = mask_shape
    storage = torch.randn(
        mask_heads, mask_batch, q_length, kv_length * 2, device="cuda", dtype=mask_dtype
    )
    mask = storage.transpose(0, 1)[..., ::2]
    values = torch.tensor(
        [0.0, -0.0, torch.finfo(mask_dtype).tiny * torch.finfo(mask_dtype).eps,
         -torch.finfo(mask_dtype).tiny, torch.inf, -torch.inf, torch.nan],
        device="cuda", dtype=mask_dtype,
    )
    mask[..., : values.numel()] = values
    mask[..., 0, :] = -torch.inf
    snapshot = ck.prequantize_int8_attention(q, k, v, attn_mask=mask)
    expected = ck.int8_attention_from_prequantized(snapshot)
    allocate = hip._sage_dense_mask_buffer
    buffers = []

    def capture_buffer(raw):
        packed = allocate(raw)
        buffers.append(packed)
        return packed

    def unexpected_preparation(*args):
        raise AssertionError("The fused call launched separate mask preparation")

    monkeypatch.setattr(hip, "_sage_dense_mask_buffer", capture_buffer)
    monkeypatch.setattr(hip._C, "sage_prepare_dense_mask", unexpected_preparation)
    actual = ck.int8_attention(q, k, v, attn_mask=mask)
    assert len(buffers) == 1
    assert torch.equal(buffers[0].view(torch.int16), snapshot.attn_mask.view(torch.int16))
    assert torch.equal(actual, expected)
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual[:, :, 0]) == 0


@requires_int8_attention
@pytest.mark.skipif(not torch.version.hip, reason="HIP fused dense-mask binding")
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("mask_dtype", [torch.float16, torch.bfloat16])
def test_hip_fused_dense_mask_graph_reads_updated_bias(head_dim, mask_dtype):
    torch.manual_seed(188)
    q, k, v = _qkv(1, 4, 2, 129, 1153, head_dim)
    mask = torch.randn(1, 1, 129, 2306, device="cuda", dtype=mask_dtype)[..., ::2]
    for _ in range(3):
        ck.int8_attention(q, k, v, scale=-0.125, attn_mask=mask)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = ck.int8_attention(q, k, v, scale=-0.125, attn_mask=mask)
    for empty in (True, False):
        if empty:
            mask.fill_(-torch.inf)
        else:
            mask.normal_()
            mask[..., 0, :] = -torch.inf
        snapshot = ck.prequantize_int8_attention(q, k, v, scale=-0.125, attn_mask=mask)
        expected = ck.int8_attention_from_prequantized(snapshot)
        graph.replay()
        assert torch.equal(actual, expected)
        assert torch.isfinite(actual).all()


@requires_int8_attention
@pytest.mark.skipif(not torch.version.hip, reason="HIP fused dense-mask binding")
@pytest.mark.parametrize(
    "invalid",
    ["input_dtype", "output_dtype", "heads", "width", "stride", "input_cpu",
     "output_cpu", "missing", "broadcast", "alignment"],
)
def test_hip_fused_dense_mask_rejects_invalid_buffers(invalid):
    hip = sage_attention_module._hip_backend
    q, k, v = _qkv(1, 4, 2, 17, 65, 64)
    buffers, anchors = hip._sage_buffers(q, k, 64)
    output = torch.empty(q.shape, device="cuda", dtype=q.dtype)
    raw = torch.zeros(1, 1, 17, 65, device="cuda", dtype=torch.bfloat16).expand(1, 4, 17, 65)
    prepared = torch.full((1, 1, 2, 2, 1024), 17, device="cuda", dtype=raw.dtype)
    if invalid == "input_dtype":
        raw = raw.float()
    elif invalid == "output_dtype":
        prepared = prepared.half()
    elif invalid == "heads":
        prepared = prepared.expand(1, 4, 2, 2, 1024).contiguous()
    elif invalid == "width":
        prepared = prepared[..., :-1]
    elif invalid == "stride":
        prepared = prepared.repeat_interleave(2, dim=-1)[..., ::2]
    elif invalid == "input_cpu":
        raw = raw.cpu()
    elif invalid == "output_cpu":
        prepared = prepared.cpu()
    elif invalid == "missing":
        prepared = None
    elif invalid == "broadcast":
        raw = raw[:, :, :1].expand_as(raw)
    else:
        prepared = torch.full((4097,), 17, device="cuda", dtype=raw.dtype)[1:].reshape_as(prepared)
    tensors = (q, k, v, output, buffers["q_int8"], buffers["q_scale"], buffers["k_int8"],
               buffers["k_scale"], buffers["v_int8"], buffers["v_scale"], anchors)
    with pytest.raises(RuntimeError, match="sage_sdpa"):
        hip._C.sage_sdpa(
            *(hip._dl(tensor) for tensor in tensors), 0.125, 64, 2, 2, hip._stream(q),
            None if prepared is None else hip._dl(prepared), hip._dl(raw),
        )
    if prepared is not None:
        assert torch.all(prepared == 17)


@requires_int8_attention
@pytest.mark.skipif(not torch.version.hip, reason="HIP prepared-mask binding")
@pytest.mark.parametrize("invalid", ["input_dtype", "output_dtype", "width", "stride", "cpu"])
def test_hip_prepare_dense_bool_rejects_invalid_buffers(invalid):
    hip = sage_attention_module._hip_backend
    mask = torch.ones(1, 1, 17, 65, device="cuda", dtype=torch.bool)
    output = torch.full((1, 1, 2, 2, 32), 17, device="cuda", dtype=torch.int32)
    if invalid == "input_dtype":
        mask = mask.to(torch.uint8)
    elif invalid == "output_dtype":
        output = torch.full((1, 1, 2, 2, 1024), 17, device="cuda", dtype=torch.int16)
    elif invalid == "width":
        output = output[..., :-1]
    elif invalid == "stride":
        output = output.repeat_interleave(2, dim=-1)[..., ::2]
    else:
        mask = mask.cpu()
        output = output.cpu()
    with pytest.raises(RuntimeError, match="sage_prepare_dense_mask"):
        hip._C.sage_prepare_dense_mask(
            hip._dl(mask), hip._dl(output), torch.cuda.current_stream().cuda_stream
        )
    assert torch.all(output == 17)


@requires_int8_attention
@pytest.mark.skipif(not torch.version.hip, reason="HIP prepared-mask binding")
@pytest.mark.parametrize("invalid", ["input_dtype", "output_dtype", "width", "stride", "cpu"])
def test_hip_prepare_key_mask_rejects_invalid_buffers(invalid):
    hip = sage_attention_module._hip_backend
    mask = torch.zeros(1, 1, 1, 1153, device="cuda")
    width = ((19 * 65 + 3) // 4) * 4
    output = torch.full((1, 1, width), 17.0, device="cuda")
    if invalid == "input_dtype":
        mask = mask.to(torch.uint8)
    elif invalid == "output_dtype":
        output = output.to(torch.float16)
    elif invalid == "width":
        output = output[..., :-1]
    elif invalid == "stride":
        output = torch.full((1, 1, width * 2), 17.0, device="cuda")[..., ::2]
    else:
        mask = mask.cpu()
        output = output.cpu()
    with pytest.raises(RuntimeError, match="sage_prepare_key_mask"):
        hip._C.sage_prepare_key_mask(
            hip._dl(mask), hip._dl(output), torch.cuda.current_stream().cuda_stream
        )
    assert torch.all(output == 17)


@requires_int8_attention
@pytest.mark.skipif(not torch.version.hip, reason="HIP prepared-mask binding")
@pytest.mark.parametrize("invalid", ["dtype", "width", "stride", "heads", "cpu"])
def test_hip_attention_rejects_invalid_prepared_mask(invalid):
    q, k, v = _qkv(1, 4, 2, 129, 1153, 128)
    mask = torch.zeros(1, 1, 1, 1153, device="cuda")
    packed = ck.prequantize_int8_attention(q, k, v, attn_mask=mask)
    mask = packed.attn_mask
    if invalid == "dtype":
        mask = mask.to(torch.float16)
    elif invalid == "width":
        mask = mask[..., :-1]
    elif invalid == "stride":
        mask = mask.repeat_interleave(2, dim=-1)[..., ::2]
    elif invalid == "heads":
        mask = mask.expand(1, 3, -1).contiguous()
    else:
        mask = mask.cpu()
    with pytest.raises(RuntimeError, match=r"prepared key mask|mask must be on q's ROCm device"):
        sage_attention_module._hip_backend.sage_int8_attend(
            packed.q,
            packed.k,
            packed.v,
            packed.q_scale,
            packed.k_scale,
            packed.v_scale,
            attention_scale=packed.attention_scale,
            attn_mask=mask,
            output_dtype=torch.bfloat16,
            cta_k=packed.cta_k,
        )
