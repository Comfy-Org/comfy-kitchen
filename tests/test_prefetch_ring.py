import time

import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen import prefetch_ring

requires_ring = pytest.mark.skipif(not ck.prefetch_ring_is_available(), reason="prefetch ring requires SM90 or newer")

LOOKAHEAD = 1 << 20
# DeltaNet shape of the deferred-decode credit test
B, HV, HK, DK, DV, HD, KS = 2, 4, 2, 128, 128, 256, 4
C = 2 * HK * DK + HV * DV


def _run_step(regions, credits, consume):
    """Configure a ring over `regions` ((base, bytes) or (base, bytes, flags)), start one step, run
    `consume` and check the counters.

    The regions total less than LOOKAHEAD, so the issuers request everything up front and then
    wait for credit; without any they give up (100 ms) and count as stalled (all but the CTAs
    whose stride already carried them past the wrap-around end). Self-crediting regions are
    credited by the issuer itself, exactly once although its walk wraps past them again.
    """
    consume()   # first launches load modules, which would exceed the issuers' give-up time
    regions = [r if len(r) == 3 else (*r, 0) for r in regions]
    descriptors = torch.tensor(regions, dtype=torch.uint64, device="cuda")
    prefetch_ring.configure(descriptors, len(regions), LOOKAHEAD, credits=credits)
    prefetch_ring.start(descriptors.device)
    consume()
    torch.cuda.synchronize()   # the issuers run on a side stream
    total, consumed, stalled = prefetch_ring.counters()
    prefetch_ring.disable(descriptors.device)
    assert total == sum(bytes_ for _, bytes_, _ in regions)
    expected = sum(bytes_ for _, bytes_, flags in regions if flags & prefetch_ring.SELF_CREDIT or credits)
    assert consumed == expected
    assert (stalled == 0) == (expected == total)


def _self_credit(tensor):
    return (tensor.data_ptr(), tensor.numel() * tensor.element_size(), prefetch_ring.SELF_CREDIT)


@requires_ring
def test_self_credit_regions_need_no_consumer():
    # nothing reads these: the issuer credits them as it passes and the step completes
    tensors = [torch.empty(n, device="cuda", dtype=torch.uint8) for n in (16, 8192, 65536, 48, 256)]
    _run_step([_self_credit(t) for t in tensors], 0, lambda: None)


@requires_ring
def test_issuer_exits_when_the_next_step_resets():
    # step 1's issuer credits `scale` and then waits for credit on `unread` that never comes;
    # step 2 (with `unread` shrunk to nothing) resets the count underneath it. The stale issuer
    # must see the count fall and exit, not stall for 100 ms while step 2's issuer queues behind it.
    scale = torch.empty(65536, device="cuda", dtype=torch.uint8)
    unread = torch.empty(1 << 19, device="cuda", dtype=torch.uint8)
    descriptors = torch.tensor([_self_credit(scale), (unread.data_ptr(), unread.numel(), 0)], dtype=torch.uint64, device="cuda")
    prefetch_ring.configure(descriptors, 2, LOOKAHEAD)
    prefetch_ring.start(descriptors.device)
    time.sleep(0.02)
    descriptors[1, 1] = 0
    prefetch_ring.start(descriptors.device)
    total, consumed, stalled = prefetch_ring.counters()
    prefetch_ring.disable(descriptors.device)
    assert (total, consumed, stalled) == (scale.numel(), scale.numel(), 0)


@requires_ring
@pytest.mark.skipif(not ck.flash_attention_decode_gqa_is_available(), reason="requires the head_dim-256 GQA decode kernel")
@pytest.mark.parametrize("query_length", [1, 4])
@pytest.mark.parametrize("credits", [prefetch_ring.CREDIT_KV, 0])
def test_flash_decode_gqa_credits_live_kv_rows(query_length, credits):
    # one credit per (batch, kv head) of the K and V rows up to kv_lengths, regardless of the
    # group-head packing (query_length 1) or split count; nothing without the KV credit bit
    torch.manual_seed(0)
    batch, kv_heads, heads, capacity = 2, 2, 8, 512
    q = torch.randn(batch, heads, query_length, 256, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, kv_heads, capacity, 256, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    lengths = [100, 300]
    kv_lengths = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    row_bytes = 256 * k.element_size()
    regions = [(t[b, h].data_ptr(), lengths[b] * row_bytes) for t in (k, v) for b in range(batch) for h in range(kv_heads)]
    scale, bias = torch.empty(256, device="cuda", dtype=torch.bfloat16), torch.empty(2048, device="cuda", dtype=torch.bfloat16)
    regions = [_self_credit(scale), *regions, _self_credit(bias)]   # norm scale ahead, bias after: issuer-credited either side
    _run_step(regions, credits, lambda: ck.flash_attention_decode_gqa(q, k, v, kv_lengths))


@requires_ring
@pytest.mark.skipif(not ck.gated_delta_deferred_is_available(), reason="deferred DeltaNet decode kernels unavailable")
@pytest.mark.parametrize("credits", [prefetch_ring.CREDIT_DELTA, 0])
def test_gated_delta_deferred_credits_state(credits):
    torch.manual_seed(0)
    key_dim = HK * DK
    dtype = torch.bfloat16
    w = torch.randn(C, 1, KS, device="cuda", dtype=dtype) * 0.5
    b = torch.randn(C, device="cuda", dtype=dtype) * 0.1
    w_a = torch.randn(HV, HD, device="cuda", dtype=dtype) * 0.05
    w_b = torch.randn(HV, HD, device="cuda", dtype=dtype) * 0.05
    dt_bias = torch.randn(HV, device="cuda")
    g_decay = -torch.rand(HV, device="cuda") - 0.5
    norm_w = torch.rand(DV, device="cuda", dtype=dtype) + 0.5
    conv_state = torch.randn(B, C, KS - 1, device="cuda", dtype=dtype)
    state = torch.randn(B, HV, DK, DV, device="cuda") * 0.1
    qkv_buf, proj_buf, gates_buf, sumsq_buf = ck.gated_delta_deferred_buffers(B, C, HV, HK, dtype, torch.device("cuda"))
    ctl = torch.zeros((ck.gated_delta_ctl_ints,), dtype=torch.int32, device="cuda")
    ctl[2:10] = torch.arange(8, device="cuda", dtype=torch.int32)
    ctl[10:18] = torch.arange(-1, 7, device="cuda", dtype=torch.int32)
    proj = torch.randn(B, 1, C, device="cuda", dtype=dtype)
    x = torch.randn(B, 1, HD, device="cuda", dtype=dtype)
    z = torch.randn(B, 1, HV * DV, device="cuda", dtype=dtype)

    def consume():
        ck.deltanet_conv_step_deferred(proj, conv_state, w, b, proj_buf, qkv_buf, ctl)
        ck.gated_delta_decode_deferred(x, w_a, w_b, dt_bias, g_decay, state, key_dim, HK, DK ** -0.5,
                                       z, norm_w, 1e-6, qkv_buf, gates_buf, sumsq_buf, ctl)

    _run_step([(state.data_ptr(), state.numel() * state.element_size())], credits, consume)
