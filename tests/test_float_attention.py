import pytest
import torch

import comfy_kitchen as ck

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")

# Relative to the largest output: fp32 is computed in fp32, fp16 and bf16 round the result.
TOLERANCE = {torch.float32: 2e-5, torch.float16: 2e-3, torch.bfloat16: 1e-2}


def _reference(q, k, v, mask=None, scale=None, kv_lengths=None):
    groups = q.shape[1] // k.shape[1]
    k = k.double().repeat_interleave(groups, 1)
    v = v.double().repeat_interleave(groups, 1)
    scores = q.double() @ k.transpose(-1, -2) * (q.shape[-1] ** -0.5 if scale is None else scale)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf")) if mask.dtype == torch.bool else scores + mask.double()
    if kv_lengths is not None:
        keep = torch.arange(k.shape[2], device=q.device) < kv_lengths.view(-1, 1, 1, 1)
        scores = scores.masked_fill(~keep, float("-inf"))
    return (torch.softmax(scores, -1).nan_to_num(0.0) @ v)


def _assert_close(actual, expected, dtype):
    assert actual.dtype == dtype
    error = (actual.double() - expected).abs().max() / expected.abs().max().clamp_min(1e-6)
    assert error < TOLERANCE[dtype], f"relative error {error.item():.2e}"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize(("heads", "kv_heads"), [(8, 8), (16, 8), (32, 4)])
@pytest.mark.parametrize(("q_len", "kv_len"), [(1, 1), (1, 300), (7, 31), (100, 64), (333, 517)])
def test_float_attention_matches_reference(dtype, head_dim, heads, kv_heads, q_len, kv_len):
    q = torch.randn(2, heads, q_len, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(2, kv_heads, kv_len, head_dim, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    out = ck.float_attention(q, k, v)
    assert out.shape == q.shape
    _assert_close(out, _reference(q, k, v), dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_float_attention_strided_seq_first_layout(dtype):
    # [B, S, H, D] storage viewed as [B, H, S, D], as transformer blocks hand it over.
    q = torch.randn(2, 90, 16, 128, device="cuda", dtype=dtype).transpose(1, 2)
    k = torch.randn(2, 70, 8, 128, device="cuda", dtype=dtype).transpose(1, 2)
    v = torch.randn(2, 70, 8, 128, device="cuda", dtype=dtype).transpose(1, 2)
    _assert_close(ck.float_attention(q, k, v, scale=0.05), _reference(q, k, v, scale=0.05), dtype)


@pytest.mark.parametrize("kind", ["additive", "bool", "per_head"])
def test_float_attention_masks(kind):
    q = torch.randn(2, 16, 50, 64, device="cuda")
    k = torch.randn(2, 8, 77, 64, device="cuda")
    v = torch.randn_like(k)
    if kind == "additive":
        mask = torch.randn(1, 1, 50, 77, device="cuda")
        mask[..., ::3] = torch.finfo(torch.float32).min
    elif kind == "bool":
        mask = torch.rand(2, 1, 1, 77, device="cuda") > 0.4
        mask[..., 0] = True
    else:
        mask = torch.randn(2, 16, 50, 77, device="cuda", dtype=torch.float16).float()
    _assert_close(ck.float_attention(q, k, v, attn_mask=mask), _reference(q, k, v, mask), torch.float32)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_float_attention_kv_lengths_over_a_cache(dtype):
    # A fixed-capacity cache in [B, capacity, Hk, D] storage, as KV-cache decode keeps it.
    capacity = 2049
    k = torch.randn(3, capacity, 8, 128, device="cuda", dtype=dtype).transpose(1, 2)
    v = torch.randn(3, capacity, 8, 128, device="cuda", dtype=dtype).transpose(1, 2)
    q = torch.randn(3, 16, 1, 128, device="cuda", dtype=dtype)
    lengths = torch.tensor([1, 900, capacity], device="cuda", dtype=torch.int32)
    out = ck.float_attention(q, k, v, kv_lengths=lengths)
    _assert_close(out, _reference(q, k, v, kv_lengths=lengths), dtype)


def test_float_attention_row_without_keys_is_zero():
    q = torch.randn(2, 8, 1, 128, device="cuda")
    k = torch.randn(2, 8, 64, 128, device="cuda")
    out = ck.float_attention(q, k, torch.randn_like(k), kv_lengths=torch.tensor([0, 64], device="cuda", dtype=torch.int32))
    assert out[0].abs().max() == 0 and torch.isfinite(out).all()


@pytest.mark.skipif(not torch.cuda.is_available() or not ck.float_attention_is_available(
    torch.empty(1, 1, 1, 128, device="cuda"), torch.empty(1, 1, 1, 128, device="cuda")),
    reason="requires the native float attention kernel")
@pytest.mark.parametrize("num_splits", [1, 2, 7, 64])
def test_float_attention_split_counts_agree(num_splits):
    from comfy_kitchen.backends import rdna1

    q = torch.randn(1, 16, 1, 128, device="cuda")
    k = torch.randn(1, 8, 3000, 128, device="cuda")
    v = torch.randn_like(k)
    out = torch.empty(1, 1, 16, 128, device="cuda").transpose(1, 2)
    o_accum = torch.empty(num_splits, 1, 16, 1, 128, device="cuda")
    lse_accum = torch.empty(num_splits, 1, 16, 1, device="cuda")
    rdna1._C.flash_attention(rdna1._dl(q), rdna1._dl(k), rdna1._dl(v), None, None, rdna1._dl(out),
                             rdna1._dl(o_accum) if num_splits > 1 else None,
                             rdna1._dl(lse_accum) if num_splits > 1 else None,
                             num_splits, 128 ** -0.5, rdna1._stream(q))
    _assert_close(out, _reference(q, k, v), torch.float32)
