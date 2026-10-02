import gc
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from comfy_kitchen.backends import ascend
from comfy_kitchen.backends.ascend.weight_cache import W4A4WeightCache
from comfy_kitchen.backends.eager.svdquant import _unpack_int4_row_major
from comfy_kitchen.tensor.convrot_w4a4 import convrot_w4a4_linear


@pytest.mark.parametrize("budget", [-1, 1.5, True])
def test_invalid_budget(budget):
    with pytest.raises(ValueError):
        W4A4WeightCache(budget)


@pytest.mark.parametrize("inference", [False, True])
@pytest.mark.parametrize("packed_dtype", [torch.int8, torch.uint8])
def test_mutation_and_identity(inference, packed_dtype):
    with torch.inference_mode(inference):
        source = torch.randint(-128, 128, (4, 8), dtype=torch.int8).to(packed_dtype)
        with W4A4WeightCache(1024) as cache:
            first = cache._unpack(source)
            assert cache._unpack(source) is first
            assert cache.bytes_used == 96
            source.fill_(7)
            updated = cache._unpack(source)
            assert updated is not first
            assert torch.equal(updated, _unpack_int4_row_major(source))
            # .data writes need not bump the original tensor's version counter.
            source.data.fill_(17)
            assert torch.equal(cache._unpack(source), _unpack_int4_row_major(source))
            other = source.clone()
            assert cache._unpack(other) is not updated
            assert cache.bytes_used == 192
        assert cache.bytes_used == 0


def test_budget_does_not_thrash():
    first = torch.zeros(4, 8, dtype=torch.int8)
    second = torch.ones_like(first)
    with W4A4WeightCache(96) as cache:
        prepared = cache._unpack(first)
        cache._unpack(second)
        assert cache._unpack(first) is prepared
        assert cache.bytes_used == 96
        assert cache.hits == 1
        assert cache.misses == 2


def test_source_collection_releases_entry():
    source = torch.zeros(4, 8, dtype=torch.int8)
    with W4A4WeightCache(1024) as cache:
        cache._unpack(source)
        del source
        gc.collect()
        assert cache.bytes_used == 0


def test_exception_and_new_scope():
    cache = W4A4WeightCache(1024)
    source = torch.zeros(4, 8, dtype=torch.int8)
    with pytest.raises(LookupError), cache:
        cache._unpack(source)
        raise LookupError
    assert cache.bytes_used == 0
    with pytest.raises(RuntimeError):
        cache._unpack(source)
    with cache:
        cache._unpack(source)
        assert cache.hits == 0
        assert cache.misses == 1
        with pytest.raises(RuntimeError):
            cache.__enter__()


def test_view_offset_resize_and_storage_replacement():
    base = torch.arange(64, dtype=torch.int8).reshape(8, 8)
    source = base[2:6, ::2]
    with W4A4WeightCache(4096) as cache:
        first = cache._unpack(source)
        assert torch.equal(first, _unpack_int4_row_major(source))
        base[2, 0] = 99
        assert torch.equal(cache._unpack(source), _unpack_int4_row_major(source))
        source.set_(torch.ones(4, 4, dtype=torch.int8))
        assert torch.equal(cache._unpack(source), _unpack_int4_row_major(source))
        source.resize_(2, 8)
        assert torch.equal(cache._unpack(source), _unpack_int4_row_major(source))


def test_other_thread_bypasses_cache():
    source = torch.zeros(4, 8, dtype=torch.int8)
    with W4A4WeightCache(1024) as cache:
        first = cache._unpack(source)
        with ThreadPoolExecutor(max_workers=1) as pool:
            second = pool.submit(cache._unpack, source).result()
        assert second is not first
        assert torch.equal(second, first)
        assert cache.hits == 0


def test_zero_budget():
    source = torch.zeros(4, 8, dtype=torch.int8)
    with W4A4WeightCache(0) as cache:
        assert torch.equal(cache._unpack(source), _unpack_int4_row_major(source))
        assert cache.bytes_used == 0


def test_too_small_budget_and_empty_tensor():
    source = torch.zeros(4, 8, dtype=torch.int8)
    with W4A4WeightCache(95) as cache:
        cache._unpack(source)
        cache._unpack(source)
        assert cache.bytes_used == 0
        assert cache.hits == 0
    with W4A4WeightCache(1024) as cache:
        # Preserve the eager helper's error for ambiguous empty reshapes.
        empty = torch.empty(0, 8, dtype=torch.int8)
        with pytest.raises(RuntimeError, match="ambiguous"):
            _unpack_int4_row_major(empty)
        with pytest.raises(RuntimeError, match="ambiguous"):
            cache._unpack(empty)
        assert cache.bytes_used == 0


def test_replacing_view_and_clearing_entries():
    base = torch.zeros(8, 8, dtype=torch.int8)
    first, second = base[:4], base[4:]
    second.fill_(17)
    with W4A4WeightCache(1024) as cache:
        assert not torch.equal(cache._unpack(first), cache._unpack(second))
        cache.clear()
        assert cache.bytes_used == 0
        assert torch.equal(cache._unpack(second), _unpack_int4_row_major(second))


def test_npu_inference_tensor_and_stream_switch():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    with torch.inference_mode(), W4A4WeightCache(1024 * 1024) as cache:
        source = torch.zeros((32, 128), device="npu", dtype=torch.int8)
        assert source.is_inference()
        first = cache._unpack(source)
        assert cache._unpack(source) is first
        source.fill_(17)
        changed = cache._unpack(source)
        assert torch.equal(changed, _unpack_int4_row_major(source))
        current = torch.npu.current_stream()
        other = torch.npu.Stream()
        other.wait_stream(current)
        with torch.npu.stream(other):
            bypass = cache._unpack(source)
            assert bypass is not changed
            assert torch.equal(bypass, changed)
        current.wait_stream(other)
        assert cache._unpack(source) is changed


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_npu_linear_matches_and_detects_mutation(dtype):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    device = torch.device("npu:0")
    x = torch.randn(8, 256, device=device, dtype=dtype)
    weight = torch.randint(-128, 128, (32, 128), device=device, dtype=torch.int8)
    scale = torch.ones(32, device=device)
    bias = torch.randn(32, device=device, dtype=dtype)
    with torch.inference_mode(), W4A4WeightCache(1024 * 1024) as cache:
        for _ in range(3):
            expected = ascend.convrot_w4a4_linear(x, weight, scale, bias)
            actual = ascend.convrot_w4a4_linear(x, weight, scale, bias, weight_cache=cache)
            assert torch.equal(actual, expected)
        weight.fill_(17)
        assert torch.equal(
            ascend.convrot_w4a4_linear(x, weight, scale, bias, weight_cache=cache),
            ascend.convrot_w4a4_linear(x, weight, scale, bias),
        )
        # Only packed weights are cached, never scales, activations or bias.
        scale.mul_(0.5)
        bias.add_(1)
        x.mul_(0.5)
        assert torch.equal(
            ascend.convrot_w4a4_linear(x, weight, scale, bias, weight_cache=cache),
            ascend.convrot_w4a4_linear(x, weight, scale, bias),
        )


def test_public_dispatch_preserves_eager_selection():
    x = torch.randn(2, 256)
    weight = torch.randint(-128, 128, (32, 128), dtype=torch.int8)
    scale = torch.ones(32)
    with W4A4WeightCache(1024 * 1024) as cache:
        expected = convrot_w4a4_linear(x, weight, scale)
        actual = convrot_w4a4_linear(x, weight, scale, weight_cache=cache)
        assert torch.equal(expected, actual)
        assert cache.hits == cache.misses == 0


@pytest.mark.parametrize("packed_dtype", [torch.int8, torch.uint8])
@pytest.mark.parametrize("strided", [False, True])
def test_public_dispatch_npu_cache(packed_dtype, strided):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    x = torch.randn(2, 256, device="npu")
    storage = torch.randint(0, 256, (32, 256), device="npu", dtype=torch.int32).to(packed_dtype)
    weight = storage[:, ::2] if strided else storage[:, :128].contiguous()
    scale = torch.ones(32, device="npu")
    with W4A4WeightCache(1024 * 1024) as cache:
        expected = convrot_w4a4_linear(x, weight, scale)
        assert torch.equal(expected, convrot_w4a4_linear(x, weight, scale, weight_cache=cache))
        assert torch.equal(expected, convrot_w4a4_linear(x, weight, scale, weight_cache=cache))
        assert cache.hits == 1
        assert cache.misses == 1
        weight.fill_(17)
        expected = convrot_w4a4_linear(x, weight, scale)
        assert torch.equal(expected, convrot_w4a4_linear(x, weight, scale, weight_cache=cache))
        assert cache.misses == 2
