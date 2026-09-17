# SPDX-License-Identifier: Apache-2.0
"""Draft attention: reference, layout, routing, public-op, and native tests.

Three layers in order. First the architecture-independent reference contracts:
the ragged video layout, the routing budget, and the eager algorithm itself
(the semantic anchor). Then the public custom-op contracts: dispatch, device
portability, torch.compile, and native availability. Last, the native CUDA
regression tests that compare the SM89/SM120 executors against the reference
on real GPUs.
"""

import math

import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.draft import materialize_layout, resolve_options
from comfy_kitchen.backends.eager.draft import (
    draft_attention,
    draft_probability,
    route_counts,
    route_probability,
)
from comfy_kitchen.registry import NoCapableBackendError

# The eager reference is the semantic anchor the native executors are compared
# against in the GPU regression tests below.
reference = draft_attention


# --- Reference, layout, and routing contracts (architecture-independent) ---


@pytest.mark.parametrize("block", [64, 128])
@pytest.mark.parametrize("shape", [(1, 1, 1), (2, 7, 9), (3, 15, 26)])
def test_layout_exact_cover(block, shape):
    layout = materialize_layout("cpu", shape, block)
    tokens = shape[0] * shape[1] * shape[2]
    assert layout.indices.numel() == layout.counts.numel() * block
    assert layout.counts.sum().item() == tokens
    assert layout.slot_valid.sum().item() == tokens
    torch.testing.assert_close(
        layout.indices[layout.slot_valid].sort().values, torch.arange(tokens)
    )
    torch.testing.assert_close(layout.indices[layout.inverse], torch.arange(tokens))
    assert layout.logical_block == block


@pytest.mark.parametrize("block", [64, 128])
@pytest.mark.parametrize("prefix", [0, 1, 64, 65, 129])
def test_physical_k64_lengths(block, prefix):
    layout = materialize_layout("cpu", (2, 11, 17), block)
    # A logical block maps to one or two physical K64 stages whose lengths
    # never exceed 64; prefix tokens fill whole leading K64 stages.
    prefix_counts = torch.tensor(
        [min(64, prefix - start) for start in range(0, prefix, 64)], dtype=torch.int32
    )
    video_counts = layout.counts
    if block == 128:
        video_counts = torch.stack(
            (video_counts.clamp(max=64), (video_counts - 64).clamp(min=0, max=64)), dim=-1
        ).flatten()
    counts = torch.cat((prefix_counts, video_counts)).view(1, -1)
    assert counts.dtype == torch.int32
    assert counts.sum().item() == prefix + 2 * 11 * 17
    assert counts.min().item() >= 0 and counts.max().item() <= 64
    assert counts.shape == (1, (prefix + 63) // 64 + layout.counts.numel() * (block // 64))


@pytest.mark.parametrize("block", [64, 128])
def test_anchor_ids_match_boolean_matrix(block):
    layout = materialize_layout("cpu", (2, 16, 16), block, True)
    assert layout.anchor_count == layout.anchors.sum().item()
    torch.testing.assert_close(
        layout.anchor_ids.long().sort().values, layout.anchors.flatten().nonzero().flatten()
    )
    n = layout.blocks_per_frame
    assert not layout.anchors[:n, n:].any()
    assert not layout.anchors[n:, :n].any()
    assert layout.anchors.diag().all()


@pytest.mark.parametrize("block", [64, 128])
@pytest.mark.parametrize("anchors", [False, True])
def test_layout_matches_direct_construction(block, anchors):
    # The cached builder constructs index tables with vectorized torch ops; a
    # literal per-frame/per-block loop is the readable specification.
    from comfy_kitchen.draft import make_ragged_2d_partition

    frames, height, width = 3, 11, 13
    layout = materialize_layout("cpu", (frames, height, width), block, anchors)
    partition = make_ragged_2d_partition(height, width, block, include_adjacency=anchors)
    indices, live = [], []
    inverse = [0] * (frames * height * width)
    for frame in range(frames):
        for source_block in partition.blocks:
            for token in source_block:
                inverse[frame * height * width + token] = len(indices)
                indices.append(frame * height * width + token)
                live.append(True)
            indices.extend([0] * (block - len(source_block)))
            live.extend([False] * (block - len(source_block)))
    assert torch.equal(layout.indices, torch.tensor(indices, dtype=torch.int64))
    assert torch.equal(layout.slot_valid, torch.tensor(live, dtype=torch.bool))
    assert torch.equal(layout.inverse, torch.tensor(inverse, dtype=torch.int64))
    assert torch.equal(layout.counts, torch.tensor(partition.counts * frames, dtype=torch.int32))
    assert layout.blocks_per_frame == partition.block_count


def test_layout_cached_per_key():
    first = materialize_layout("cpu", (2, 5, 7), 64, False)
    again = materialize_layout("cpu", (2, 5, 7), 64, False)
    other = materialize_layout("cpu", (2, 5, 7), 128, False)
    assert first is again
    assert other is not first and other.logical_block == 128


@pytest.mark.parametrize(
    "ratios", [(0, 1, 0), (0, 0, 1), (1, 0, 0), (0.6, 0.3, 0.1), (0.75, 0.25, 0)]
)
@pytest.mark.parametrize("sparsity", [0, 0.5, 0.8, 0.99])
def test_route_counts_preserve_global_budget(ratios, sparsity):
    counts = route_counts(25, sparsity, *ratios)
    assert sum(counts) == max(1, math.floor((1 - sparsity) * 25 + 0.5))
    assert all(x >= 0 for x in counts)


def _phase_matrix(route):
    result = torch.zeros_like(route.block_ids)
    positions = torch.arange(result.shape[-1]).view(1, 1, 1, -1)
    n4, n8, n16 = (
        c.unsqueeze(-1) for c in (route.nvfp4_counts, route.middle_counts, route.fp16_counts)
    )
    codes = torch.where(
        positions < n4,
        1,
        torch.where(positions < n4 + n8, 2, torch.where(positions < n4 + n8 + n16, 3, 0)),
    )
    return result.scatter(-1, route.block_ids.long(), codes.to(result.dtype))


def test_stable_ties_assign_highest_precision_first():
    probability = torch.ones(1, 1, 4, 4, dtype=torch.float16)
    route = route_probability(
        probability, sparsity_ratio=0.5, nvfp4_ratio=0.25, middle_ratio=0.5, fp16_ratio=0.25
    )
    torch.testing.assert_close(
        _phase_matrix(route).flatten(),
        torch.tensor([3, 3, 2, 2, 2, 2, 1, 1] + [0] * 8, dtype=torch.int32),
    )


def test_anchors_replace_only_lowest_precision():
    probability = torch.arange(16, 0, -1, dtype=torch.float16).view(1, 1, 4, 4)
    anchors = torch.zeros(4, 4, dtype=torch.bool)
    anchors[-1, -1] = True
    route = route_probability(
        probability,
        sparsity_ratio=0.5,
        nvfp4_ratio=0.25,
        middle_ratio=0.5,
        fp16_ratio=0.25,
        anchors=anchors,
        anchor_count=1,
    )
    matrix = _phase_matrix(route).flatten()
    torch.testing.assert_close(matrix[:6], torch.tensor([3, 3, 2, 2, 2, 2], dtype=torch.int32))
    assert matrix[-1] == 1 and matrix[7] == 0 and matrix[6] == 1
    assert matrix.count_nonzero() == 8


def test_anchor_budget_rejected_before_routing():
    with pytest.raises(ValueError, match="lowest-precision budget"):
        route_probability(
            torch.ones(1, 1, 4, 4),
            sparsity_ratio=0.5,
            nvfp4_ratio=0.125,
            middle_ratio=0.875,
            anchors=torch.eye(4, dtype=torch.bool),
            anchor_count=4,
        )


@pytest.mark.parametrize("block", [64, 128])
@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("prefix", [0, 11])
def test_reference_dense_budget(block, dim, prefix):
    g = torch.Generator().manual_seed(17)
    q, k, v = (torch.randn(1, prefix + 2 * 7 * 9, 2, dim, generator=g).half() for _ in range(3))
    out = draft_attention(
        q,
        k,
        v,
        video_shape=(2, 7, 9),
        prefix_tokens=prefix,
        query_block_size=block,
        sparsity_ratio=0,
        int8_ratio=0.7,
        fp16_ratio=0.3,
    )
    expected = (
        torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2).float(), k.transpose(1, 2).float(), v.transpose(1, 2).float()
        )
        .transpose(1, 2)
        .half()
    )
    torch.testing.assert_close(out, expected, atol=0.002, rtol=0.002)
    assert out.is_contiguous()


@pytest.mark.parametrize("proxy", ["mean", "k_tail_r1", "k_tail_r2"])
def test_reference_proxy_finite(proxy):
    g = torch.Generator().manual_seed(12)
    q, k, v = (torch.randn(1, 130, 2, 128, generator=g).half() for _ in range(3))
    out = draft_attention(q, k, v, video_shape=(2, 5, 13), draftmap_proxy=proxy)
    assert out.shape == q.shape and out.dtype == q.dtype
    assert out.isfinite().all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"architecture": "sm89", "nvfp4_ratio": 1, "int8_ratio": 0},
        {"architecture": "sm120", "int8_ratio": 0.5, "mxfp8_ratio": 0.5},
        {"architecture": "sm120", "smooth_k": True},
        {"architecture": "sm120", "query_block_size": 128, "draftmap_proxy": "k_tail_r1"},
        {"architecture": "sm120", "diag_jensen": True, "maxpool_weight": 0.5},
    ],
)
def test_invalid_architecture_phase_combinations(kwargs):
    with pytest.raises(ValueError):
        resolve_options(**kwargs)


def test_draft_probability_jensen_matches_direct_formula():
    g = torch.Generator().manual_seed(42)
    q = torch.randn(1, 2, 4, 64, generator=g).half()
    k = torch.randn(1, 2, 4, 64, generator=g).half()
    qs, ks = q.float().square() + 0.2, k.float().square() + 0.3
    scores = q.float() @ k.float().transpose(-1, -2) / 8
    variance = (
        qs @ ks.transpose(-1, -2) - q.float().square() @ k.float().square().transpose(-1, -2)
    ).clamp_min(0) / 64
    expected = torch.softmax(scores + variance / 2, -1).half()
    torch.testing.assert_close(draft_probability(q, k, qs, ks), expected, rtol=0, atol=0)


# --- Public op and dispatch contracts ---


def test_public_registration():
    assert "draft_attention" in ck.__all__
    assert "draft_attention_is_available" in ck.__all__
    assert torch.ops.comfy_kitchen.draft_attention.default._schema.name == "comfy_kitchen::draft_attention"


@pytest.mark.parametrize("block", [64, 128])
@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("prefix", [0, 7])
def test_public_dense_cpu(block, dim, prefix):
    gen = torch.Generator().manual_seed(20260910)
    q, k, v = (torch.randn(1, 126 + prefix, 2, dim, generator=gen).half() for _ in range(3))
    result = ck.draft_attention(
        q,
        k,
        v,
        video_shape=(2, 7, 9),
        prefix_tokens=prefix,
        query_block_size=block,
        sparsity_ratio=0,
        int8_ratio=0.5,
        fp16_ratio=0.5,
    )
    expected = (
        torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2).float(), k.transpose(1, 2).float(), v.transpose(1, 2).float()
        )
        .transpose(1, 2)
        .half()
    )
    torch.testing.assert_close(result, expected, atol=0.002, rtol=0.002)
    assert result.is_contiguous()


@pytest.mark.parametrize(
    "options",
    [
        {"int8_ratio": 0.0, "nvfp4_ratio": 0.7, "mxfp8_ratio": 0.2, "fp16_ratio": 0.1},
        {"int8_ratio": 0.0, "fp16_ratio": 1.0},
        {"draftmap_proxy": "k_tail_r1"},
        {"draftmap_proxy": "k_tail_r2"},
        {"diag_jensen": True},
        {"maxpool_weight": 0.5},
        {"smooth_k": True},
    ],
)
def test_public_options_reach_reference(options):
    q = torch.randn(1, 130, 2, 128).half()
    actual = ck.draft_attention(q, q, q, video_shape=(2, 5, 13), **options)
    expected = reference(q, q, q, video_shape=(2, 5, 13), **options)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    "options",
    [
        {"query_block_size": 32},
        {"sparsity_ratio": 1.0},
        {"int8_ratio": -0.1},
        {"int8_ratio": 0.5, "mxfp8_ratio": 0.5},
        {"nvfp4_scales": (0.0, 1.0, 1.0)},
        {"diag_jensen": True, "maxpool_weight": 0.5},
        {"query_block_size": 128, "draftmap_proxy": "k_tail_r1"},
    ],
)
def test_public_invalid_options(options):
    q = torch.zeros(1, 64, 1, 128, dtype=torch.float16)
    with pytest.raises((NoCapableBackendError, ValueError)):
        ck.draft_attention(q, q, q, video_shape=(1, 8, 8), **options)


def test_public_invalid_geometry():
    q = torch.zeros(1, 64, 1, 128, dtype=torch.float16)
    with pytest.raises(NoCapableBackendError, match="video_shape"):
        ck.draft_attention(q, q, q, video_shape=(1, 8, 7))
    with pytest.raises(NoCapableBackendError, match="head_dim"):
        ck.draft_attention(q[..., :32], q[..., :32], q[..., :32], video_shape=(1, 8, 8))
    with pytest.raises(NoCapableBackendError, match="same shape"):
        ck.draft_attention(q, q[:, :-1], q, video_shape=(1, 8, 8))


def test_custom_op_fullgraph_cpu():
    q = torch.randn(1, 130, 2, 64).half()
    compiled = torch.compile(ck.draft_attention, backend="eager", fullgraph=True)
    result = compiled(
        q, q, q, video_shape=(2, 5, 13), query_block_size=128, int8_ratio=0.8, fp16_ratio=0.2
    )
    expected = ck.draft_attention(
        q, q, q, video_shape=(2, 5, 13), query_block_size=128, int8_ratio=0.8, fp16_ratio=0.2
    )
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_fake_op_output_metadata():
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        q = torch.empty(1, 130, 3, 128, dtype=torch.bfloat16)
        out = ck.draft_attention(q, q, q, video_shape=(2, 5, 13), query_block_size=128)
        assert out.shape == q.shape and out.dtype == q.dtype and out.device == q.device
        assert out.is_contiguous()


def test_reference_serves_non_cpu_devices():
    # The full-precision reference is device-portable like every other eager
    # implementation; meta tensors exercise the routing without a GPU.
    q = torch.empty(1, 130, 3, 128, dtype=torch.float16, device="meta")
    backend = ck.registry.get_capable_backend(
        "draft_attention",
        {"q": q, "k": q, "v": q, "video_shape": (2, 5, 13), "query_block_size": 64},
    )
    assert backend == "eager"


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize("dim", [64, 128])
def test_reference_runs_on_cuda(dim):
    q = torch.randn(1, 130, 3, dim, dtype=torch.float16, device="cuda")
    with ck.use_backend("eager"):
        out = ck.draft_attention(q, q, q, video_shape=(2, 5, 13), query_block_size=64)
    assert out.shape == q.shape and out.device.type == "cuda" and out.dtype == q.dtype


def test_native_availability_without_cuda():
    if not torch.cuda.is_available():
        assert not ck.draft_attention_is_available()


@pytest.mark.parametrize("capability", [(8, 8), (8, 9), (9, 0), (10, 0), (11, 8), (12, 0)])
def test_availability_requires_native_plan_and_architecture(monkeypatch, capability):
    from types import SimpleNamespace

    from comfy_kitchen.backends import cuda

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    monkeypatch.setattr(cuda, "_EXT_AVAILABLE", True)
    extension = SimpleNamespace(
        draft=object(),
        draft_plan=object(),
        draft_supports_arch=lambda arch: arch in (89, 90, 100, 118, 120),
    )
    monkeypatch.setattr(cuda, "_C", extension, raising=False)
    assert cuda.draft_attention_is_available() == (capability >= (8, 9))
    del extension.draft_plan
    assert not cuda.draft_attention_is_available()


# --- Native architecture regression (requires the built extension and a GPU) ---


native = pytest.mark.skipif(
    not ck.draft_attention_is_available(),
    reason="complete native Draft extension and an SM89-or-newer GPU required",
)


@pytest.mark.cuda
@native
@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("anchors_enabled", [False, True])
def test_native_sparse_pipeline_matches_reference(tied, anchors_enabled):
    generator = torch.Generator(device="cuda").manual_seed(71)
    q, k, v = (
        torch.randn(1, 4 * 9 * 15, 2, 128, dtype=torch.float16, device="cuda", generator=generator)
        * 0.25
        for _ in range(3)
    )
    if tied:
        q.zero_()
        k.zero_()
    options = {
        "video_shape": (4, 9, 15),
        "sparsity_ratio": 0.1,
        "int8_ratio": 0.0,
        "fp16_ratio": 1.0,
        "enable_anchors": anchors_enabled,
    }
    actual = ck.draft_attention(q, k, v, **options)
    expected = reference(q, k, v, **options)
    torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.02)


@pytest.mark.cuda
@native
@pytest.mark.parametrize("block", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("prefix", [0, 7])
@pytest.mark.parametrize("dim", [64, 128])
def test_native_dense_fp16_matches_reference(block, dtype, prefix, dim):
    gen = torch.Generator(device="cuda").manual_seed(12)
    q, k, v = (
        torch.randn(1, 130 + prefix, 2, dim, dtype=dtype, device="cuda", generator=gen)
        for _ in range(3)
    )
    options = {
        "video_shape": (2, 5, 13),
        "prefix_tokens": prefix,
        "query_block_size": block,
        "sparsity_ratio": 0.0,
        "int8_ratio": 0.0,
        "fp16_ratio": 1.0,
    }
    actual = ck.draft_attention(q, k, v, **options)
    expected = reference(q, k, v, **options)
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
    assert actual.is_contiguous()


@pytest.mark.cuda
@native
@pytest.mark.parametrize("block", [64, 128])
def test_native_nondefault_stream_and_noncontiguous_input(block):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        storage = [
            torch.randn(1, 133, 2, 256, device="cuda", dtype=torch.float16) for _ in range(3)
        ]
        q, k, v = [x[..., ::2] for x in storage]
        out = ck.draft_attention(
            q,
            k,
            v,
            video_shape=(2, 5, 13),
            prefix_tokens=3,
            query_block_size=block,
            sparsity_ratio=0.5,
        )
        expected_shape = q.shape
        del storage, q, k, v
    torch.cuda.current_stream().wait_stream(stream)
    assert out.shape == expected_shape and out.isfinite().all()


@pytest.mark.cuda
@native
@pytest.mark.parametrize("block", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_native_preserves_bhsd_input_storage(monkeypatch, block, dtype):
    from comfy_kitchen.backends import cuda

    q, k, v = [
        torch.randn(1, 2, 133, 128, device="cuda", dtype=dtype).transpose(1, 2) for _ in range(3)
    ]
    exported = []
    wrap = cuda._wrap_for_dlpack

    def record(tensor):
        exported.append(tensor.data_ptr())
        return wrap(tensor)

    monkeypatch.setattr(cuda, "_wrap_for_dlpack", record)
    options = {"video_shape": (2, 5, 13), "prefix_tokens": 3, "query_block_size": block}
    actual = ck.draft_attention(q, k, v, **options)
    assert exported[:3] == [x.data_ptr() for x in (q, k, v)]
    expected = ck.draft_attention(*(x.contiguous() for x in (q, k, v)), **options)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.cuda
@native
@pytest.mark.parametrize("block", [64, 128])
def test_native_plan_owns_no_device_storage(block):
    from comfy_kitchen.backends import cuda

    device = torch.cuda.current_device()
    before = torch.cuda.memory_allocated(device)
    plan = cuda._C.draft_plan(
        device,
        1,
        256,
        2,
        128,
        block,
        0,
        256 // block,
        0,
        False,
        0.5,
        (0.0, 1.0, 0.0, 0.0),
        -1,
        -1,
        0,
        False,
        0.0,
        False,
        False,
        (1.0, 1.0, 1.0),
    )
    assert plan.workspace_bytes > 0 and not plan.native_prefix
    assert torch.cuda.memory_allocated(device) == before
    # Stage-specific allocation/launch APIs are intentionally private C++ now.
    assert not hasattr(cuda._C, "sm89_h3_route_precision")
    assert not hasattr(cuda._C, "prepare_h3_sm120_operands")
