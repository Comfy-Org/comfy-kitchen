# SPDX-License-Identifier: Apache-2.0

import gc
import weakref

import pytest
import torch

import comfy_kitchen as ck
import comfy_kitchen.sage_attention as sage_attention_module
from .conftest import (
    skip_unless_gfx12_wmma,
)

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


def test_int8_attention_cta_k_selection():
    select = sage_attention_module._select_cta_k
    assert select(128, 1025, has_mask=False) == 128
    assert select(64, 1025, has_mask=False) == 64
    assert select(128, 1025, has_mask=True) == 64


@pytest.mark.parametrize(
    ("scale", "expected"),
    [(0.5, 0.5), (2, 2.0), (torch.tensor(0.5), 0.5), (torch.tensor([0.5]), 0.5)],
)
def test_attention_scale_accepts_numeric_scalars(scale, expected):
    assert sage_attention_module._validate_attention_scale(scale, 1.0) == expected


@pytest.mark.parametrize(
    ("scale", "error"),
    [
        (torch.ones(2), ValueError),
        ("0.5", TypeError),
        (True, TypeError),
        (float("nan"), ValueError),
        (torch.tensor(float("inf")), ValueError),
    ],
)
def test_attention_scale_rejects_invalid_values(scale, error):
    with pytest.raises(error, match="scale"):
        sage_attention_module._validate_attention_scale(scale, 1.0)


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
    monkeypatch.setattr(
        sage_attention_module._hip_backend, "has_wmma", lambda: has_wmma
    )
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
        mask = torch.full(
            (1, 1, 1, 97), -torch.inf, dtype=mask_dtype, device="cuda"
        )

    actual = ck.int8_attention(q, k, v, attn_mask=mask)

    assert torch.count_nonzero(actual) == 0


@requires_int8_attention
def test_int8_attention_stabilizes_large_common_key_component():
    torch.manual_seed(7)
    q, k, v = _qkv(1, 16, 16, 513, 513, 128)
    common_key = torch.randn(
        1, 16, 1, 128, device="cuda", dtype=torch.float32
    )
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
def test_gfx12_bf16_attention_handles_long_rectangular_lengths():
    skip_unless_gfx12_wmma()

    torch.manual_seed(37)
    q = torch.randn(1, 1025, 4, 128, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    k = torch.randn(1, 1033, 4, 128, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    v = torch.randn(1, 1033, 4, 128, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    actual = ck.hip_attention(q, k, v)

    assert ck.hip_attention_is_supported(q, k, v)
    assert ck.hip_int8_attention_is_supported(q, k, v)
    assert actual.dtype is torch.bfloat16
    assert actual.shape == q.shape
    assert (actual.float() - expected.float()).abs().max() <= 0.00390625


@requires_int8_attention
def test_gfx12_bf16_full_k_tile_route_is_shape_generic():
    skip_unless_gfx12_wmma()

    torch.manual_seed(39)
    # Deliberately differs from every captured model contract: seven heads,
    # rectangular attention, and a one-row query tail. K/V is the only exact
    # tile requirement of this schedule.
    q = torch.randn(1, 7, 1025, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 7, 1056, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 7, 1056, 128, device="cuda", dtype=torch.bfloat16)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    actual = ck.hip_attention(q, k, v)

    assert ck.hip_attention_is_supported(q, k, v)
    assert ck.hip_int8_attention_is_supported(q, k, v)
    assert (actual.float() - expected.float()).abs().max() <= 0.00390625


@requires_int8_attention
def test_gfx12_short_attention_capability_is_dimensional():
    skip_unless_gfx12_wmma()

    def supported(batch, heads, length, *, dtype=torch.bfloat16):
        tensors = tuple(
            torch.empty(batch, heads, length, 128, device="cuda", dtype=dtype)
            for _ in range(3)
        )
        return ck.hip_attention_is_supported(*tensors)

    assert not supported(1, 4, 127)
    assert supported(1, 4, 128)
    assert supported(1, 56, 388)
    assert supported(1, 17, 1024)
    assert ck.hip_int8_attention_is_supported(
        *(
            torch.empty(1, 17, 1024, 128, device="cuda", dtype=torch.bfloat16)
            for _ in range(3)
        )
    )
    assert not supported(2, 4, 16)
    assert supported(64, 16, 16)
    assert not supported(1, 4, 160, dtype=torch.float16)
    unaligned = tuple(
        torch.empty(1, 4, 128, 129, device="cuda", dtype=torch.bfloat16)[..., 1:]
        for _ in range(3)
    )
    assert not ck.hip_attention_is_supported(*unaligned)
    long_fp16 = tuple(
        torch.empty(1, 4, 1024, 128, device="cuda", dtype=torch.float16)
        for _ in range(3)
    )
    assert ck.hip_int8_attention_is_supported(*long_fp16)
    assert not ck.hip_attention_is_supported(*long_fp16)


@requires_int8_attention
def test_gfx12_compact_batched_attention_matches_sdpa():
    skip_unless_gfx12_wmma()

    torch.manual_seed(41)
    q, k, v = (
        torch.randn(64, 16, 12, 128, device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    )
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    bf16 = ck.hip_attention(q, k, v)
    int8_mode = ck.hip_int8_attention(q, k, v)

    assert ck.hip_attention_is_supported(q, k, v)
    assert ck.hip_int8_attention_is_supported(q, k, v)
    assert torch.equal(int8_mode, bf16)
    assert (bf16.float() - expected.float()).abs().max() <= 0.00390625


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
    mask = torch.linspace(-1, 1, 257, device="cuda", dtype=torch.float32).reshape(
        1, 1, 1, 257
    )
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
def test_int8_attention_accepts_dlpack_normalized_batch_stride():
    """A size-one extent carries no address, so its stride must not be policed.

    PyTorch rewrites the stride of any size-one dimension to 1 on the way out
    through DLPack (ATen/DLConvertor.cpp, gh-83069), so a batch-one attention
    input arrives with stride 1 no matter what the caller built.
    """
    torch.manual_seed(0)
    length, heads, head_dim = 372, 8, 128
    packed = [
        torch.randn(length, heads, head_dim, dtype=torch.bfloat16, device="cuda")
        for _ in range(3)
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
