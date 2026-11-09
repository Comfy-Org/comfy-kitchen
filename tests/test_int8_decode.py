import pytest
import torch

from comfy_kitchen.int8_decode import Int8DecodeCache, is_available
from comfy_kitchen.sage_attention import PrequantizedInt8Attention, int8_attention_from_prequantized
import comfy_kitchen.int8_decode as decode


pytestmark = pytest.mark.skipif(not is_available(), reason="INT8 decode backend unavailable")


@pytest.mark.parametrize("batch,heads", [(1, 1), (2, 2)])
def test_incremental_flat_cache_matches_full_quantization(batch, heads):
    torch.manual_seed(7)
    capacity, window = 4096, 1536
    keys = torch.randn(batch, heads, 2059, 256, device="cuda", dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    cache = Int8DecodeCache(batch, heads, capacity, "cuda")
    reference = Int8DecodeCache(batch, heads, capacity, "cuda")
    ring_k = torch.empty(batch, heads, window, 256, device="cuda", dtype=torch.bfloat16)
    ring_v = torch.empty_like(ring_k)
    length = torch.tensor([509], device="cuda")
    cache.update(keys[:, :, :509].contiguous(), values[:, :, :509].contiguous(), length, initialize=True)
    for end in (512, 513, 1025, 1537, 2048, 2053, 2045, 2059):
        length.fill_(end)
        start = max(0, end - window)
        slots = torch.arange(start, end, device="cuda").remainder(window)
        ring_k.index_copy_(2, slots, keys[:, :, start:end])
        ring_v.index_copy_(2, slots, values[:, :, start:end])
        cache.update(ring_k, ring_v, length)
        reference.update(keys[:, :, :end].contiguous(), values[:, :, :end].contiguous(), length, initialize=True)
        chunks = (end + 511) // 512
        assert torch.equal(cache.key[:, :, :end], reference.key[:, :, :end])
        assert torch.equal(cache.value[:, :chunks * 512], reference.value[:, :chunks * 512])
        assert torch.equal(cache.key_scale[:, :, :chunks * 32], reference.key_scale[:, :, :chunks * 32])
        assert torch.equal(cache.value_scale[:, :chunks], reference.value_scale[:, :chunks])


@pytest.mark.parametrize("batch,kv_heads,groups,prefix,rows", [(1, 2, 4, 0, 129), (1, 4, 6, 509, 519), (2, 2, 4, 511, 1027)])
@pytest.mark.parametrize("mask_kind", ["additive", "bool", "broadcast", "none"])
def test_cached_prefill_matches_quantized_reference(batch, kv_heads, groups, prefix, rows, mask_kind):
    if torch.version.hip:
        pytest.skip("CUDA packing reference")
    torch.manual_seed(23)
    length = prefix + rows
    capacity = (length + 511) // 512 * 512
    heads = kv_heads * groups
    key = torch.randn(batch, kv_heads, length, 256, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    value[:, :, 512:] *= 4
    value[:, :, 1024:] *= 0.125
    cache = Int8DecodeCache(batch, kv_heads, capacity, "cuda")
    cache.update(key, value, torch.tensor([length], device="cuda"), initialize=True)
    query = torch.randn(batch, heads, rows, 256, device="cuda", dtype=torch.bfloat16)
    bias = torch.zeros(rows, length, device="cuda").masked_fill_(
        torch.arange(length, device="cuda").unsqueeze(0) > torch.arange(prefix, length, device="cuda").unsqueeze(1), float("-inf"))
    mask = bias
    if mask_kind == "bool":
        mask = bias.isfinite()
    elif mask_kind == "broadcast":
        mask = bias.view(1, 1, rows, length).expand(batch, heads, rows, length)
    elif mask_kind == "none":
        mask = None
        bias.zero_()
    qi = torch.empty_like(query, dtype=torch.int8)
    qs = torch.empty(batch, heads, (rows + 127) // 128 * 64, device="cuda")
    out = torch.empty(batch, rows, heads * 256, device="cuda", dtype=torch.bfloat16)

    def attend():
        decode.backend._C._int8_prefill(
            *map(decode.wrap_for_dlpack, (query, qi, qs, cache.key, cache.value, cache.key_scale, cache.value_scale, out)),
            decode.wrap_for_dlpack(mask) if mask is not None else None, length, torch.cuda.current_stream().cuda_stream)

    attend()
    q_index = torch.tensor([(r // 128) * 64 + (r % 128 // 16) * 8 + r % 8 for r in range(rows)], device="cuda")
    k_index = torch.tensor([(r // 64) * 4 + (r % 8 // 2) for r in range(length)], device="cuda")
    qf = qi.float() * qs[:, :, q_index].unsqueeze(-1)
    kf = cache.key[:, :, :length].float() * cache.key_scale[:, :, k_index].unsqueeze(-1)
    positions = torch.arange(length, device="cuda")
    w = positions % 16
    perm = (positions // 16 * 16) | (w & 1) | ((w >> 3 & 1) << 1) | ((w >> 1 & 1) << 2) | ((w >> 2 & 1) << 3)
    vf = cache.value.view(batch, kv_heads, 256, capacity)[:, :, :, perm].transpose(-1, -2).float()
    vf *= cache.value_scale[:, positions // 512].permute(0, 2, 1, 3)
    scores = torch.matmul(qf, kf.repeat_interleave(groups, 1).transpose(-1, -2)).mul_(0.0625).add_(bias)
    expected = torch.matmul(scores.softmax(-1), vf.repeat_interleave(groups, 1)).transpose(1, 2).reshape_as(out)
    torch.testing.assert_close(out.float(), expected, atol=0.035, rtol=0.035)
    assert torch.equal(cache.prefill(query, length, mask), out)

    cache.value_scale.copy_(cache.value_scale[:, :1].expand_as(cache.value_scale))
    attend()
    padded = (length + 63) // 64 * 64
    packed = PrequantizedInt8Attention(
        qi, cache.key[:, :, :length].contiguous(),
        cache.value.view(batch, kv_heads, 256, capacity)[:, :, :, :padded].contiguous().view(batch * kv_heads * 256, padded),
        qs, cache.key_scale[:, :, :(length + 63) // 64 * 4].contiguous(),
        cache.value_scale[:, 0].contiguous().flatten(), 256, torch.bfloat16, 0.0625, 64,
        bias.view(1, 1, rows, length).expand(batch, heads, rows, length) if mask is not None else None)
    standard = int8_attention_from_prequantized(packed).transpose(1, 2).reshape_as(out)
    assert torch.equal(standard, out)
