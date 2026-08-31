"""Sol-Attn sparse attention.

The CUDA backend runs INT8 internally, so tests assert cosine similarity (not
bitwise equality) against the full-precision eager reference, plus the
layout and validation invariants: batch > 1, sinks, ragged tails, strided
inputs, and the real model's constants (rot_dim, activation scales,
inference mode).
"""

import math

import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.backends import cuda as cuda_backend
from comfy_kitchen.backends.eager.sol_attn import sol_attn as sol_attn_eager
from comfy_kitchen.exceptions import NoCapableBackendError

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

HD = 128


def _qkv(b, t, h, seed=0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)

    def mk(s):
        return torch.randn(b, t, h, HD, device=device, dtype=torch.bfloat16,
                           generator=g) * s

    return mk(0.5), mk(0.5), mk(1.0)


def _bhnd_views(b, t, h, seed=3):
    """Native BHND tensors viewed as BTHD: last dim contiguous, T stride != H*D."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    views = tuple(torch.randn(b, h, t, HD, device="cuda", dtype=torch.bfloat16,
                              generator=g).mul_(0.5).transpose(1, 2) for _ in range(3))
    assert not views[0].is_contiguous() and views[0].stride(-1) == 1
    return views


def _cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (torch.dot(a, b) / (a.norm() * b.norm())).item()


def _dense(q, k, v):
    qq, kk, vv = (x.permute(0, 2, 1, 3).float() for x in (q, k, v))
    out = torch.nn.functional.scaled_dot_product_attention(qq, kk, vv, scale=HD ** -0.5)
    return out.permute(0, 2, 1, 3)


def _chunked_case(seed, rot, v_scale=1.0):
    """A qkv-projection tensor plus everything both attention paths need from
    it: the separate-rope reference inputs and the producer's chunk list."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    t, h, d = 4096 + 100, 4, HD          # last chunk is 100 tokens: ragged at the chunk AND 64-block level
    qkv = torch.randn(t, 3 * h * d, device="cuda", dtype=torch.bfloat16,
                      generator=g) * 0.5
    qkv[:, 2 * h * d:] *= v_scale
    freqs = torch.randn(1, t, 1, rot // 2, 2, 2, device="cuda", generator=g)
    qw = torch.randn(d, device="cuda", dtype=torch.bfloat16, generator=g)
    kw = torch.randn(d, device="cuda", dtype=torch.bfloat16, generator=g)
    q = qkv[:, :h * d].view(1, t, h, d).clone()
    k = qkv[:, h * d:2 * h * d].view(1, t, h, d).clone()
    v = qkv[:, 2 * h * d:].view(1, t, h, d)
    ck.rms_rope_split_half_(q, k, freqs, qw, kw, epsilon=1e-6, rot_dim=rot)
    return {"t": t, "h": h, "chunks": list(qkv.split(1024)), "freqs": freqs,
            "norm": (qw, kw), "q": q, "k": k, "v": v}


# 3137 leaves a 1-token tail; 1000 and 1088 are ragged too
@pytest.mark.parametrize("t", [256, 1024, 2048, 1000, 1088, 3137])
@pytest.mark.parametrize("tau", [0.5, 1.0, 2.0, 6.0])
def test_matches_eager_reference(t, tau):
    q, k, v = _qkv(1, t, 4)
    got = ck.sol_attn(q, k, v, tau=tau)
    assert torch.isfinite(got.float()).all()
    assert _cos(got, sol_attn_eager(q, k, v, tau=tau)) > 0.998


@pytest.mark.parametrize("t", [2048 + 1, 2048 + 4, 2048 + 32])
def test_ragged_tail_routes_like_the_reference(t):
    """The tail query block must route on the mean over its LIVE rows."""
    q, k, v = _qkv(1, t, 4, seed=7)
    tail = slice(t - (t % 64), t)
    got = ck.sol_attn(q, k, v, tau=1.4)
    ref = sol_attn_eager(q, k, v, tau=1.4)
    assert _cos(got[:, tail], ref[:, tail]) > 0.999


@pytest.mark.parametrize("b", [2, 3])
def test_batch(b):
    """Every batch must match the same input run alone."""
    q, k, v = _qkv(b, 1024, 4)
    got = ck.sol_attn(q, k, v, tau=1.4)
    for i in range(b):
        alone = ck.sol_attn(q[i:i + 1].contiguous(), k[i:i + 1].contiguous(),
                            v[i:i + 1].contiguous(), tau=1.4)
        assert _cos(got[i], alone[0]) > 0.9999


@pytest.mark.parametrize(
    "sink_blocks,sink_q",
    [([0, 2], [0, 0]), ([0, 0], [0, 2]), ([0, 2], [0, 2])],
)
def test_sinks(sink_blocks, sink_q):
    q, k, v = _qkv(1, 1024, 4)
    got = ck.sol_attn(q, k, v, tau=1.4, sink_blocks=sink_blocks, sink_q=sink_q)
    ref = sol_attn_eager(q, k, v, tau=1.4, sink_blocks=sink_blocks, sink_q=sink_q)
    assert _cos(got, ref) > 0.998


def test_sink_q_attends_everything():
    """A query block inside sink_q is exact over the whole sequence, so those
    rows must equal dense attention."""
    q, k, v = _qkv(1, 1024, 4)
    got = ck.sol_attn(q, k, v, tau=6.0, sink_q=[0, 1])
    ref = _dense(q, k, v)
    assert _cos(got[:, :64], ref[:, :64]) > 0.999


@pytest.mark.parametrize("b", [1, 2])
@pytest.mark.parametrize("select", [{"tau": 1.4}, {"topk_ratio": 0.2}])
def test_strided_inputs(b, select):
    """A BHND view (last dim contiguous) must match its contiguous copy exactly."""
    q, k, v = _bhnd_views(b, 1024, 4)
    got = ck.sol_attn(q, k, v, **select)
    ref = ck.sol_attn(q.contiguous(), k.contiguous(), v.contiguous(), **select)
    assert torch.equal(got, ref)


def test_rejects_noncontiguous_last_dim():
    """A strided last dim would read neighbouring channels rather than fail."""
    _q, k, v = _qkv(1, 256, 4)
    bad = torch.empty(1, 256, 4, HD * 2, device="cuda", dtype=torch.bfloat16)[..., ::2]
    assert bad.stride(-1) != 1
    with pytest.raises(ValueError, match="contiguous last dim"):
        cuda_backend.sol_attn(bad, k, v, tau=1.4)


def test_tau_direction():
    """Higher tau routes fewer blocks exactly, so the output moves away from
    dense attention; parity at each tau is test_matches_eager_reference."""
    q, k, v = _qkv(1, 2048, 8)
    ref = _dense(q, k, v)
    assert _cos(ck.sol_attn(q, k, v, tau=0.5), ref) > _cos(ck.sol_attn(q, k, v, tau=6.0), ref) + 5e-3


def test_output_strides_agree_across_backends():
    """register_fake, CUDA and eager must return the same layout."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    qh = torch.randn(1, 4, 1024, HD, device="cuda", dtype=torch.bfloat16)
    v = qh.transpose(1, 2)
    assert not v.is_contiguous()

    cuda_strides = ck.sol_attn(v, v, v, tau=1.4).stride()
    eager_strides = sol_attn_eager(v.float(), v.float(), v.float(), tau=1.4).stride()
    with FakeTensorMode():
        fv = torch.empty(v.shape, dtype=v.dtype, device=v.device)
        fake_strides = torch.ops.comfy_kitchen.sol_attn(
            fv, fv, fv, tau=1.4, scale=None, sink_blocks=[0, 0], sink_q=[0, 0],
            key_bias=None, topk_ratio=0.0, tail=True, block_len=None,
            coarse_gate=None).stride()
    assert cuda_strides == eager_strides == fake_strides


def test_unaligned_input_is_rejected():
    """An odd storage_offset would fault the 16 B staging loads."""
    n = 1 * 256 * 4 * HD
    base = torch.randn(n + 8, device="cuda", dtype=torch.bfloat16)
    bad = base[1:1 + n].view(1, 256, 4, HD)
    assert bad.stride(-1) == 1 and bad.data_ptr() % 16
    with pytest.raises(ValueError, match="16-byte aligned"):
        cuda_backend.sol_attn(bad, bad, bad, tau=1.4)


def test_misaligned_stride_is_rejected():
    """A padded-row layout (132-wide sliced to 128) misaligns the 16 B loads."""
    base = torch.randn(1, 256, 4, HD + 4, device="cuda", dtype=torch.bfloat16)
    bad = base[..., :HD]
    assert bad.stride(-1) == 1 and bad.data_ptr() % 16 == 0 and bad.stride(2) % 8
    with pytest.raises(ValueError, match="multiple of 8"):
        cuda_backend.sol_attn(bad, bad, bad, tau=1.4)


def test_eager_refuses_video_length_rather_than_oom():
    """The O(T^2) reference must refuse video length, not die in the allocator."""
    q, k, v = (torch.empty(1, 37296, 56, HD, device="meta", dtype=torch.float16)
               for _ in range(3))
    with pytest.raises(RuntimeError, match="O\\(T\\^2\\)"):
        sol_attn_eager(q, k, v, tau=1.4)


@pytest.mark.parametrize("sink", [[3], [0, 1, 2], [2, 1], [-5, 2]])
def test_bad_sink_range_is_rejected(sink):
    """Sinks are [start, end) pairs; bad shapes must fail validation."""
    q, k, v = _qkv(1, 256, 4)
    with pytest.raises(NoCapableBackendError):
        ck.sol_attn(q, k, v, tau=1.4, sink_blocks=sink)


def test_mismatched_dtype_is_rejected():
    """The call rule cross-checks k/v against q."""
    q, k, v = _qkv(1, 256, 4)
    with pytest.raises(NoCapableBackendError, match="dtype"):
        ck.sol_attn(q, k.half(), v, tau=1.4)


def test_head_dim_constraint():
    """head_dim 128 is baked into both backends."""
    q, k, v = (torch.randn(1, 256, 4, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    with pytest.raises(NoCapableBackendError, match="head_dim must be 128"):
        ck.sol_attn(q, k, v, tau=1.4)


def test_key_bias_matches_eager():
    """Per-key logit bias, honoured by the exact branch; biased blocks are sink-covered."""
    q, k, v = _qkv(1, 2048, 4)
    bias = torch.zeros(1, 2048, device="cuda")
    bias[:, -128:-64] = math.log(0.3)
    bias[:, -64:] = math.log(2.0)
    sinks = [2048 // 64 - 2, 2048 // 64]
    got = ck.sol_attn(q, k, v, tau=1.4, key_bias=bias, sink_blocks=sinks)
    ref = sol_attn_eager(q, k, v, tau=1.4, key_bias=bias, sink_blocks=sinks)
    assert _cos(got, ref) > 0.998
    # and the bias must actually do something
    plain = ck.sol_attn(q, k, v, tau=1.4, sink_blocks=sinks)
    assert not torch.equal(got, plain)


def test_key_bias_inf_masks_out_keys():
    """-inf bias must remove keys without poisoning the output."""
    q, k, v = _qkv(1, 1024, 4)
    bias = torch.zeros(1, 1024, device="cuda")
    bias[:, -32:] = float("-inf")
    sinks = [1024 // 64 - 1, 1024 // 64]
    got = ck.sol_attn(q, k, v, tau=1.4, key_bias=bias, sink_blocks=sinks)
    assert torch.isfinite(got.float()).all()
    ref = sol_attn_eager(q, k, v, tau=1.4, key_bias=bias, sink_blocks=sinks)
    assert _cos(got, ref) > 0.998


def test_key_bias_bad_shape_rejected():
    q, k, v = _qkv(1, 256, 4)
    with pytest.raises(ValueError, match="key_bias"):
        cuda_backend.sol_attn(q, k, v, tau=1.4,
                              key_bias=torch.zeros(1, 128, device="cuda"))


def test_key_bias_wrong_device_rejected():
    """A host tensor must be rejected before launch (it would poison the context)."""
    q, k, v = _qkv(1, 256, 4)
    with pytest.raises((ValueError, RuntimeError), match="key_bias"):
        ck.sol_attn(q, k, v, tau=1.4, key_bias=torch.zeros(256))


def test_direct_backend_validates_like_the_public_path():
    """The backend-direct entry runs the same shared rule as the registry."""
    q, k, v = _qkv(1, 512, 4)
    with pytest.raises(ValueError, match="bfloat16"):
        cuda_backend.sol_attn(q.half(), k.half(), v.half(), tau=1.4)
    with pytest.raises(ValueError, match="shape"):
        cuda_backend.sol_attn(q, k[:, :256].contiguous(), v, tau=1.4)


def test_sub_sm80_rejected_at_the_wrapper(monkeypatch):
    """Sub-sm_80 cubins are stubs returning uninitialised memory; the wrapper
    must check q.device itself since the registry gate caches one device."""
    q, k, v = _qkv(1, 256, 4)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (7, 5))
    with pytest.raises(RuntimeError, match="sm_80"):
        cuda_backend.sol_attn(q, k, v, tau=1.4)


@pytest.mark.parametrize("t", [4096, 1000, 3137])
def test_topk_matches_eager(t):
    """Top-k parity; int8 rounding can flip boundary blocks, hence the looser bar."""
    q, k, v = _qkv(1, t, 4)
    got = ck.sol_attn(q, k, v, topk_ratio=0.2)
    ref = sol_attn_eager(q, k, v, topk_ratio=0.2)
    assert _cos(got, ref) > 0.995
    # selection changes with the budget
    assert not torch.equal(got, ck.sol_attn(q, k, v, topk_ratio=0.5))


def test_topk_budget_direction():
    """A bigger top-k budget moves the reference toward dense attention."""
    q, k, v = _qkv(1, 4096, 2)
    ref = _dense(q, k, v)
    assert (_cos(sol_attn_eager(q, k, v, topk_ratio=0.05), ref) + 5e-3
            < _cos(sol_attn_eager(q, k, v, topk_ratio=0.6), ref))


def test_topk_keeps_sinks_exact():
    """Sinks ride on top of the top-k budget, exactly as in tau mode."""
    q, k, v = _qkv(1, 2048, 2)
    sinks = [0, 4]
    got = ck.sol_attn(q, k, v, topk_ratio=0.1, sink_blocks=sinks)
    ref = sol_attn_eager(q, k, v, topk_ratio=0.1, sink_blocks=sinks)
    assert _cos(got, ref) > 0.995


@pytest.mark.parametrize("t", [64, 40])
def test_topk_single_key_block(t):
    """n == 1 must not crash the k-th-score threshold; the diagonal makes
    the result dense in both backends."""
    q, k, v = _qkv(1, t, 2)
    got = ck.sol_attn(q, k, v, topk_ratio=0.2)
    assert _cos(got, _dense(q, k, v)) > 0.999
    assert _cos(sol_attn_eager(q, k, v, topk_ratio=0.2), _dense(q, k, v)) > 0.9999


def test_topk_ratio_validation():
    """The range lives in the shared rule, so every entry rejects it."""
    q, k, v = _qkv(1, 1024, 1)
    with pytest.raises(ValueError, match="topk_ratio"):
        cuda_backend.sol_attn(q, k, v, topk_ratio=1.5)
    with pytest.raises(NoCapableBackendError, match="topk_ratio"):
        ck.sol_attn(q, k, v, topk_ratio=1.5)


@pytest.mark.parametrize("rot", [64, 96])
def test_chunked_producer_matches_separate_rope(rot):
    """Chunked producer vs rms_rope_split_half_ + sol_attn. rot=96 is H3's real
    rot_dim (non-power-of-two lane offset); V is scaled to realistic size."""
    c = _chunked_case(seed=11, rot=rot, v_scale=0.02)
    ref = cuda_backend.sol_attn(c["q"], c["k"], c["v"], tau=1.4, sink_blocks=[0, 2])
    out1, km, vs = cuda_backend.sol_attn_chunked(
        c["chunks"], c["t"], c["h"], c["freqs"], c["norm"], tau=1.4, sink_blocks=[0, 2])
    out2, _, _ = cuda_backend.sol_attn_chunked(
        c["chunks"], c["t"], c["h"], c["freqs"], c["norm"], kmean=km, vscale=vs,
        tau=1.4, sink_blocks=[0, 2])
    assert _cos(out1, ref) > 0.995       # bootstrap self-measures, no blind scales
    assert _cos(out2, ref) > 0.995
    # ComfyUI runs under inference_mode: no ._version access anywhere
    with torch.inference_mode():
        out3, _, _ = cuda_backend.sol_attn_chunked(
            c["chunks"], c["t"], c["h"], c["freqs"].clone(), c["norm"], kmean=km, vscale=vs,
            tau=1.4, sink_blocks=[0, 2])
    assert _cos(out3, ref) > 0.995


def test_bindings_check_buffer_sizes():
    """The _C entries take bare pointers sized from integers; each must reject
    an undersized buffer itself, not rely on the Python wrappers."""
    from comfy_kitchen.backends.cuda import _C
    from comfy_kitchen.backends.cuda import _wrap_for_dlpack as w
    t, h = 256, 2
    q, k, v = _qkv(1, t, h)
    ws = torch.empty(_C.sol_attn_plan(1, t, h)["total"], dtype=torch.uint8, device="cuda")
    stream = torch.cuda.current_stream().cuda_stream
    args = (1, t, h, HD, 1.0, HD ** -0.5, 0, 0, 0, 0, stream)
    with pytest.raises(RuntimeError, match="out"):
        _C.sol_attn(w(q), w(k), w(v), w(q[:, :128].contiguous()), w(ws), *args)
    with pytest.raises(RuntimeError, match="workspace"):
        _C.sol_attn(w(q), w(k), w(v), w(torch.empty_like(q)), w(ws[:-1]), *args)
    with pytest.raises(RuntimeError, match="\\(B, T, H, D\\)"):
        _C.sol_attn(w(q.view(t, h, HD)), w(k), w(v), w(torch.empty_like(q)), w(ws), *args)
    # staging-load layout contract, checked at the binding too
    base = torch.randn(1, t, h, HD + 4, device="cuda", dtype=torch.bfloat16)   # row stride 132: not a multiple of 8
    off = torch.randn(t * h * HD + 8, device="cuda", dtype=torch.bfloat16)
    wide = torch.randn(1, t, h, 2 * HD, device="cuda", dtype=torch.bfloat16)
    bad_layouts = (wide[..., ::2], off[1:1 + t * h * HD].view(1, t, h, HD), base[..., :HD])
    for bad in bad_layouts:
        with pytest.raises(RuntimeError, match="16-byte aligned base"):
            _C.sol_attn(w(bad), w(k), w(v), w(torch.empty_like(q)), w(ws), *args)
    with pytest.raises(RuntimeError, match="out must be contiguous"):
        _C.sol_attn(w(q), w(k), w(v), w(torch.empty(1, h, t, HD, device="cuda", dtype=torch.bfloat16).transpose(1, 2)),
                    w(ws), *args)
    stats = torch.empty(h, HD, device="cuda")
    with pytest.raises(RuntimeError, match="out"):
        _C.sol_attn_core(w(ws), w(q[:, :128].contiguous()), w(stats), w(stats), w(stats),
                         1, t, h, 1.0, HD ** -0.5, 0, 0, 0, 0, stream)
    with pytest.raises(RuntimeError, match="qkv"):
        _C.sol_producer_chunk(w(ws), w(torch.empty(64, 3 * h * HD - 8, device="cuda", dtype=torch.bfloat16)),
                              w(torch.empty(t, 64, 2, device="cuda")), w(stats[0]), w(stats[0]),
                              w(stats), w(stats), 1e-6, 64, 0, 64, 1, t, h, stream)
    # producer metadata: rot_dim, batch, and the chunk range are checked before launch
    chunk, fab = torch.empty(64, 3 * h * HD, device="cuda", dtype=torch.bfloat16), torch.empty(t, 64, 2, device="cuda")

    def produce(rot=64, t0=0, m=64, batch=1):
        _C.sol_producer_chunk(w(ws), w(chunk), w(fab), w(stats[0]), w(stats[0]), w(stats), w(stats),
                              1e-6, rot, t0, m, batch, t, h, stream)
    with pytest.raises(RuntimeError, match="rot_dim"):
        produce(rot=12)
    with pytest.raises(RuntimeError, match="B=1"):
        produce(batch=2)
    for t0, m in ((-64, 64), (32, 64), (t, 64), (t - 32, 64)):
        with pytest.raises(RuntimeError, match="64-aligned"):
            produce(t0=t0, m=m)


def test_chunked_producer_validates():
    """Coverage, width and device are checked before any launch."""
    c = _chunked_case(seed=11, rot=64)
    with pytest.raises(ValueError, match="chunks cover"):
        cuda_backend.sol_attn_chunked(c["chunks"][:-1], c["t"], c["h"], c["freqs"], c["norm"])
    with pytest.raises(ValueError, match="chunks must be"):
        cuda_backend.sol_attn_chunked(
            [ch[:, :-8] for ch in c["chunks"]], c["t"], c["h"], c["freqs"], c["norm"])
    with pytest.raises(ValueError, match="topk_ratio"):
        cuda_backend.sol_attn_chunked(
            c["chunks"], c["t"], c["h"], c["freqs"], c["norm"], topk_ratio=2.0)
    with pytest.raises(ValueError, match="sink_blocks"):
        cuda_backend.sol_attn_chunked(
            c["chunks"], c["t"], c["h"], c["freqs"], c["norm"], sink_blocks=[3, 1])
    # rot 12: a lane's four channels would straddle rot/2 and need two partner lanes
    bad_freqs = torch.randn(1, c["t"], 1, 6, 2, 2, device="cuda")
    with pytest.raises(ValueError, match="rot_dim"):
        cuda_backend.sol_attn_chunked(c["chunks"], c["t"], c["h"], bad_freqs, c["norm"])


def test_chunked_zero_vscale_is_clamped():
    """A caller-supplied all-zero V scale must not poison the run (1/0 in the
    producer, 255/0 in route); it is clamped like the internal bootstrap."""
    c = _chunked_case(seed=11, rot=64)
    ref = cuda_backend.sol_attn(c["q"], c["k"], c["v"], tau=1.4)
    zeros = torch.zeros(c["h"], HD, device="cuda")
    out, km, vs = cuda_backend.sol_attn_chunked(
        c["chunks"], c["t"], c["h"], c["freqs"], c["norm"], kmean=zeros, vscale=zeros, tau=1.4)
    assert torch.isfinite(out.float()).all()          # that step's V is sign-quantized: finite is the bar
    out, _, _ = cuda_backend.sol_attn_chunked(
        c["chunks"], c["t"], c["h"], c["freqs"], c["norm"], kmean=km, vscale=vs, tau=1.4)
    assert _cos(out, ref) > 0.995                      # and its statistics recover the next step


def test_chunked_producer_topk():
    """Producer-path top-k (threshold from the workspace) vs the separate-rope path."""
    c = _chunked_case(seed=13, rot=64)
    ref = cuda_backend.sol_attn(c["q"], c["k"], c["v"], topk_ratio=0.2, sink_blocks=[0, 2])
    _, km, vs = cuda_backend.sol_attn_chunked(
        c["chunks"], c["t"], c["h"], c["freqs"], c["norm"], topk_ratio=0.2, sink_blocks=[0, 2])
    out, _, _ = cuda_backend.sol_attn_chunked(
        c["chunks"], c["t"], c["h"], c["freqs"], c["norm"], kmean=km, vscale=vs,
        topk_ratio=0.2, sink_blocks=[0, 2])
    assert _cos(out, ref) > 0.995


# ---- VSA-style pieces: padded tiles, no tail, gated coarse branch ----

def _padded_case(t=64 * 40, h=4, seed=21, b=1):
    """Random live-row counts per block (the last entry may exceed the ragged
    tail and gets clamped); dead rows hold garbage the kernel must ignore."""
    from comfy_kitchen.backends.eager.sol_attn import _block_lengths
    q, k, v = _qkv(b, t, h, seed=seed)
    g = torch.Generator(device="cuda").manual_seed(seed)
    n = (t + 63) // 64
    block_len = torch.randint(1, 65, (n,), device="cuda", generator=g).to(torch.int32)
    lengths = _block_lengths(t, n, "cuda", block_len)
    valid = (torch.arange(t, device="cuda") % 64) < lengths.repeat_interleave(64)[:t]
    return q, k, v, block_len, valid


def _cos_rows(a, b, valid):
    return _cos(a[:, valid], b[:, valid])


@pytest.mark.parametrize("t", [64 * 40, 1000])   # 1000: ragged tail, block_len[-1] clamped
@pytest.mark.parametrize("select", [{"tau": 1.4}, {"topk_ratio": 0.2}])
def test_block_len_matches_eager(t, select):
    """Zero-padded tiles: only the first block_len rows of each block are keys,
    and the pooled means use the live counts."""
    q, k, v, block_len, valid = _padded_case(t=t)
    got = ck.sol_attn(q, k, v, block_len=block_len, **select)
    ref = sol_attn_eager(q, k, v, block_len=block_len, **select)
    assert torch.isfinite(got[:, valid].float()).all()
    assert _cos_rows(got, ref, valid) > (0.995 if "topk_ratio" in select else 0.998)
    # dead rows really are excluded from keys, values and means: perturbing
    # them changes nothing on the live rows
    q2, k2, v2 = (x.clone() for x in (q, k, v))
    for x in (q2, k2, v2):
        x[:, ~valid] = torch.randn_like(x[:, ~valid]) * 3
    assert torch.equal(ck.sol_attn(q2, k2, v2, block_len=block_len, **select)[:, valid],
                       got[:, valid])


def _masked_dense(q, k, v, valid):
    qq, kk, vv = (x.permute(0, 2, 1, 3).float() for x in (q, k, v))
    mask = torch.zeros(1, 1, 1, q.shape[1], device="cuda")
    mask[..., ~valid] = float("-inf")
    out = torch.nn.functional.scaled_dot_product_attention(qq, kk, vv, attn_mask=mask, scale=HD ** -0.5)
    return out.permute(0, 2, 1, 3)


def test_no_tail_matches_eager():
    """tail=False: softmax over the routed blocks only (the SLA / VSA fine stage).
    Against the fp32 reference the bar is loose because without the tail every
    int8-vs-fp32 block-selection flip shows in full; with every block routed
    (sink_q over all query blocks) it must equal dense attention."""
    q, k, v = _qkv(1, 4096, 4)
    got = ck.sol_attn(q, k, v, topk_ratio=0.2, tail=False)
    assert _cos(got, sol_attn_eager(q, k, v, topk_ratio=0.2, tail=False)) > 0.99
    assert not torch.equal(got, ck.sol_attn(q, k, v, topk_ratio=0.2))
    q, k, v, block_len, valid = _padded_case(t=64 * 30)
    dense = ck.sol_attn(q, k, v, tau=1.4, tail=False, block_len=block_len, sink_q=[0, 30])
    assert _cos_rows(dense, _masked_dense(q, k, v, valid), valid) > 0.9999


def test_no_tail_sage_exact_ragged_and_strided():
    """The long-sequence exact branch must preserve arbitrary BTHD strides and
    mask the final partial KV tile after following its sparse route."""
    q, k, v = _bhnd_views(1, 4096 + 31, 2, seed=19)
    kw = {"topk_ratio": 0.2, "tail": False}
    got = ck.sol_attn(q, k, v, **kw)
    contiguous = ck.sol_attn(q.contiguous(), k.contiguous(), v.contiguous(), **kw)
    assert torch.equal(got, contiguous)
    assert _cos(got, sol_attn_eager(q, k, v, **kw)) > 0.99


def _coarse_reference(q, k, v, valid, block_len, gate, scale=HD ** -0.5):
    """Independent per-block loop: masked dense attention plus gate * coarse term."""
    from comfy_kitchen.backends.eager.sol_attn import _block_lengths
    t = q.shape[1]
    n = (t + 63) // 64
    lengths = _block_lengths(t, n, "cuda", block_len)
    x = [xx.float() * valid.view(1, -1, 1, 1) for xx in (q, k, v)]
    means = [torch.stack([xx[:, 64 * i:64 * i + 64].sum(1) / lengths[i] for i in range(n)], 1)
             for xx in x]                                            # 3 x (B, N, H, D)
    qm, km, vm = (m.permute(0, 2, 1, 3) for m in means)              # (B, H, N, D)
    oc = torch.softmax(qm @ km.transpose(-1, -2) * scale, -1) @ vm   # (B, H, N, D)
    out = _masked_dense(q, k, v, valid).float()
    for i in range(n):
        rows = slice(64 * i, min(64 * i + 64, t))
        out[:, rows] += gate[:, rows].float() * oc[:, :, i].unsqueeze(1)
    return out


def test_coarse_gate_matches_eager():
    """VSA's coarse branch: gate * softmax(q_mean k_mean^T) v_mean per block,
    checked with every block routed so selection cannot blur it, against both
    the shared eager path and an independent per-block reference."""
    t = 64 * 30 + 17                     # ragged tail
    q, k, v, block_len, valid = _padded_case(t=t)
    n = (t + 63) // 64
    gate = torch.randn(q.shape, device="cuda", dtype=torch.bfloat16) * 0.5
    kw = {"tau": 1.4, "tail": False, "block_len": block_len, "sink_q": [0, n], "coarse_gate": gate}
    got = ck.sol_attn(q, k, v, **kw)
    assert _cos_rows(got, sol_attn_eager(q, k, v, **kw), valid) > 0.9999
    assert _cos_rows(got, _coarse_reference(q, k, v, valid, block_len, gate), valid) > 0.9999
    assert _cos_rows(got, _masked_dense(q, k, v, valid), valid) < 0.999   # the gate did something


def test_coarse_gate_batch():
    """The [B*H] ordering of the coarse means: each batch must equal itself run alone."""
    q, k, v, block_len, valid = _padded_case(t=64 * 20, b=2, seed=8)
    gate = torch.randn(q.shape, device="cuda", dtype=torch.bfloat16) * 0.5
    kw = {"topk_ratio": 0.3, "tail": False, "block_len": block_len, "coarse_gate": gate}
    got = ck.sol_attn(q, k, v, **kw)
    for i in range(2):
        alone = ck.sol_attn(q[i:i + 1].contiguous(), k[i:i + 1].contiguous(), v[i:i + 1].contiguous(),
                            **{**kw, "coarse_gate": gate[i:i + 1].contiguous()})
        assert _cos_rows(got[i:i + 1], alone, valid) > 0.9999


def test_topk_budget_excludes_sinks():
    """Sink blocks are always exact, so the top-k budget ranks and counts only
    the other blocks; a sink range running past n counts only what exists."""
    from comfy_kitchen.backends.cuda import _topk_from_pooled
    from comfy_kitchen.backends.eager.sol_attn import _topk_count
    g = torch.Generator(device="cuda").manual_seed(2)
    n, bh, d = 32, 3, HD
    c8 = torch.randint(-127, 128, (bh, n, d), device="cuda", generator=g).float()
    csc = torch.rand(bh, n, device="cuda", generator=g) + 0.5
    kc = torch.randn(bh, n, d, device="cuda", generator=g)
    for sinks, n_eff in (((0, 16), 16), ((28, 40), 28), ((0, 0), 32)):
        thr = _topk_from_pooled(c8, csc, kc, 0.2, 1.0, sinks)
        # recompute the kernel-space scores independently
        ksc = (kc.abs().amax(-1, True) / 127.0).clamp_min(1e-12)
        s = torch.bmm(c8, torch.round(kc / ksc).clamp(-127, 127).transpose(1, 2))
        s = s * csc.unsqueeze(-1) * ksc.squeeze(-1).unsqueeze(-2)
        s[..., sinks[0]:sinks[1]] = float("-inf")
        kk = _topk_count(n_eff, 0.2)
        expect = s.topk(kk, dim=-1).values[..., -1]
        assert torch.allclose(thr, expect, rtol=2e-5), sinks
        assert (s >= thr.unsqueeze(-1)).sum(-1).eq(kk).all(), sinks   # exactly kk non-sink blocks kept


def test_topk_zero_budget_routes_nothing():
    """One selectable block after sink exclusion is a zero budget (the n-1
    clamp never routes everything): block 4 must go to the tail, so with
    tail=False query blocks 0..2 see only the sink keys."""
    q, k, v = _qkv(1, 320, 2)                              # n = 5
    kw = {"topk_ratio": 0.5, "sink_blocks": [0, 4], "tail": False}
    valid = torch.arange(320, device="cuda") < 256           # keys of blocks 0..3
    ref = _masked_dense(q, k, v, valid)
    assert _cos(sol_attn_eager(q, k, v, **kw)[:, :192], ref[:, :192]) > 0.9999
    assert _cos(ck.sol_attn(q, k, v, **kw)[:, :192], ref[:, :192]) > 0.999


def test_topk_ties_over_select():
    """A tied group straddling the budget must be kept whole. Blocks 8..31 are
    identical and score highest for every query block, so with k = 8 the
    boundary is always inside the group; CUDA must agree with eager."""
    q, k, v = _qkv(1, 64 * 32, 2)
    u = q.mean(dim=1, keepdim=True) * 640           # scores highest for every query block
    k = k.clone()
    k[:, 8 * 64:] = u
    kw = {"topk_ratio": 0.25, "tail": False}
    assert _cos(ck.sol_attn(q, k, v, **kw), sol_attn_eager(q, k, v, **kw)) > 0.99


def test_chunked_vsa_mode():
    """The producer path with block_len + no tail + coarse gate must match the
    direct path on the same post-rope q/k/v (means come from the workspace)."""
    c = _chunked_case(seed=17, rot=96)
    t, h = c["t"], c["h"]
    g = torch.Generator(device="cuda").manual_seed(5)
    block_len = torch.randint(32, 65, ((t + 63) // 64,), device="cuda", generator=g).to(torch.int32)
    block_len[-1] = min(int(block_len[-1]), (t - 1) % 64 + 1)   # ragged last block
    valid = (torch.arange(t, device="cuda") % 64) < block_len.repeat_interleave(64)[:t]
    gate = torch.randn(1, t, h, HD, device="cuda", dtype=torch.bfloat16) * 0.5
    kw = {"topk_ratio": 0.2, "tail": False, "block_len": block_len, "coarse_gate": gate,
          "sink_blocks": [0, 2]}
    ref = cuda_backend.sol_attn(c["q"], c["k"], c["v"], **kw)
    _, km, vs = cuda_backend.sol_attn_chunked(c["chunks"], t, h, c["freqs"], c["norm"], **kw)
    out, _, _ = cuda_backend.sol_attn_chunked(c["chunks"], t, h, c["freqs"], c["norm"],
                                              kmean=km, vscale=vs, **kw)
    assert torch.isfinite(out.float()).all()
    assert _cos_rows(out, ref, valid) > 0.995


def test_block_len_validation():
    """block_len / coarse_gate are checked by the shared rule (every entry) and
    again by the chunked entry, which has no q."""
    q, k, v = _qkv(1, 512, 4)
    bad_n = torch.ones(7, dtype=torch.int32, device="cuda")
    bad_dtype = torch.ones(8, dtype=torch.int64, device="cuda")
    bad_gate = torch.zeros(1, 256, 4, HD, device="cuda")
    for bad in ({"block_len": bad_n}, {"block_len": bad_dtype}, {"coarse_gate": bad_gate}):
        name = next(iter(bad))
        with pytest.raises(ValueError, match=name):
            cuda_backend.sol_attn(q, k, v, tau=1.4, **bad)
        with pytest.raises(NoCapableBackendError, match=name):
            ck.sol_attn(q, k, v, tau=1.4, **bad)
    c = _chunked_case(seed=11, rot=64)
    with pytest.raises(ValueError, match="block_len"):
        cuda_backend.sol_attn_chunked(c["chunks"], c["t"], c["h"], c["freqs"], c["norm"], block_len=bad_dtype)
