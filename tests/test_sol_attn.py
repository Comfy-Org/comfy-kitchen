"""Sol-Attn sparse attention.

The CUDA backend runs INT8 internally, so tests assert cosine similarity (not
bitwise equality) against the full-precision eager reference, plus the
invariants that have actually broken in development: batch > 1, sinks,
ragged tails, strided inputs, and the real model's constants (rot_dim,
activation scales, inference mode).
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
    t, h, d = 4096 + 128, 4, HD          # ragged tail across chunk boundary
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
@pytest.mark.parametrize("tau", [1.0, 2.0])
def test_matches_eager_reference(t, tau):
    q, k, v = _qkv(1, t, 4)
    got = ck.sol_attn(q, k, v, tau=tau)
    assert torch.isfinite(got.float()).all()
    assert _cos(got, sol_attn_eager(q, k, v, tau=tau)) > 0.998


@pytest.mark.parametrize("t", [2048 + 1, 2048 + 4, 2048 + 32])
def test_ragged_tail_routes_like_the_reference(t):
    """The TAIL query block must route on the mean over its LIVE rows: clamped
    dead rows once inflated the column mean 64x at a 1-token tail, routing the
    wrong blocks. Invisible in a whole-tensor cosine, so check the tail alone."""
    q, k, v = _qkv(1, t, 4, seed=7)
    tail = slice(t - (t % 64), t)
    got = ck.sol_attn(q, k, v, tau=1.4)
    ref = sol_attn_eager(q, k, v, tau=1.4)
    assert _cos(got[:, tail], ref[:, tail]) > 0.999


@pytest.mark.parametrize("b", [2, 3])
def test_batch(b):
    """Every batch must match the same input run alone -- a missing batch offset
    in one kernel silently made every batch read batch 0's values."""
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
    """Only the last dim must be contiguous, so a BHND view goes in as-is. The
    kernels take explicit strides and the top-k threshold pools through a
    dim-splitting view; nothing else in the suite would notice if either
    stopped honouring them."""
    q, k, v = _bhnd_views(b, 1024, 4)
    got = ck.sol_attn(q, k, v, **select)
    ref = ck.sol_attn(q.contiguous(), k.contiguous(), v.contiguous(), **select)
    assert torch.equal(got, ref)


def test_rejects_noncontiguous_last_dim():
    """The staging loads are 16 B wide, so a strided last dim would read
    neighbouring channels rather than fail."""
    _q, k, v = _qkv(1, 256, 4)
    bad = torch.empty(1, 256, 4, HD * 2, device="cuda", dtype=torch.bfloat16)[..., ::2]
    assert bad.stride(-1) != 1
    with pytest.raises(ValueError, match="contiguous last dim"):
        cuda_backend.sol_attn(bad, k, v, tau=1.4)


def test_tau_monotonicity():
    """Higher tau routes fewer blocks exactly, so it can only move away from
    dense attention."""
    q, k, v = _qkv(1, 2048, 8)
    ref = _dense(q, k, v)
    sims = [_cos(ck.sol_attn(q, k, v, tau=t), ref) for t in (0.5, 2.0, 6.0)]
    assert sims[0] >= sims[1] >= sims[2] - 1e-3


def test_output_strides_agree_across_backends():
    """register_fake, CUDA and eager must return the SAME layout: torch.compile
    plans downstream ops against the fake, which used to promise a BHND-viewed
    v's strides while both real implementations return contiguous."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    qh = torch.randn(1, 4, 1024, HD, device="cuda", dtype=torch.bfloat16)
    v = qh.transpose(1, 2)
    assert not v.is_contiguous()

    cuda_strides = ck.sol_attn(v, v, v, tau=1.4).stride()
    eager_strides = sol_attn_eager(v.float(), v.float(), v.float(), tau=1.4).stride()
    with FakeTensorMode():
        fv = torch.empty(v.shape, dtype=v.dtype, device=v.device)
        # kwargs, not positional: this call has broken on every parameter
        # addition when written positionally.
        fake_strides = torch.ops.comfy_kitchen.sol_attn(
            fv, fv, fv, tau=1.4, scale=None, sink_blocks=[0, 0], sink_q=[0, 0],
            key_bias=None,
            topk_ratio=0.0).stride()
    assert cuda_strides == eager_strides == fake_strides


def test_unaligned_input_is_rejected():
    """An odd storage_offset passes the contiguous-last-dim test but faults the
    16 B staging loads with a context-poisoning misaligned address."""
    n = 1 * 256 * 4 * HD
    base = torch.randn(n + 8, device="cuda", dtype=torch.bfloat16)
    bad = base[1:1 + n].view(1, 256, 4, HD)
    assert bad.stride(-1) == 1 and bad.data_ptr() % 16
    with pytest.raises(ValueError, match="16-byte aligned"):
        cuda_backend.sol_attn(bad, bad, bad, tau=1.4)


def test_misaligned_stride_is_rejected():
    """An aligned base is not enough: a padded-row layout (a 132-wide buffer
    sliced back to 128) puts rows at +264 B, misaligning the 16 B loads."""
    base = torch.randn(1, 256, 4, HD + 4, device="cuda", dtype=torch.bfloat16)
    bad = base[..., :HD]
    assert bad.stride(-1) == 1 and bad.data_ptr() % 16 == 0 and bad.stride(2) % 8
    with pytest.raises(ValueError, match="multiple of 8"):
        cuda_backend.sol_attn(bad, bad, bad, tau=1.4)


def test_eager_refuses_video_length_rather_than_oom():
    """fp16 lands on the O(T^2) reference; at video length it must say so
    instead of dying in the allocator."""
    q, k, v = (torch.empty(1, 37296, 56, HD, device="meta", dtype=torch.float16)
               for _ in range(3))
    with pytest.raises(RuntimeError, match="O\\(T\\^2\\)"):
        sol_attn_eager(q, k, v, tau=1.4)


@pytest.mark.parametrize("sink", [[3], [0, 1, 2], [2, 1], [-5, 2]])
def test_bad_sink_range_is_rejected(sink):
    """Sinks are [start, end) pairs; bad shapes must fail validation, not
    IndexError deep in a backend or get silently truncated."""
    q, k, v = _qkv(1, 256, 4)
    with pytest.raises(NoCapableBackendError):
        ck.sol_attn(q, k, v, tau=1.4, sink_blocks=sink)


def test_mismatched_dtype_is_rejected():
    """The call rule cross-checks k/v against q; without it a bf16 q with an
    fp16 k silently fell through to the dense reference."""
    q, k, v = _qkv(1, 256, 4)
    with pytest.raises(NoCapableBackendError, match="dtype"):
        ck.sol_attn(q, k.half(), v, tau=1.4)


def test_head_dim_constraint():
    """Both backends derive their layout from head_dim 128, so neither can take
    the call and the registry finds nothing capable."""
    q, k, v = (torch.randn(1, 256, 4, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    with pytest.raises(NoCapableBackendError, match="head_dim must be 128"):
        ck.sol_attn(q, k, v, tau=1.4)


def test_key_bias_matches_eager():
    """LTX-style guide-strength bias: per-key additive logit bias, honoured by
    the exact branch. Guide blocks must be sink-covered (the pooled tail cannot
    see per-token bias), which the node does automatically."""
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
    """w=0 (a hard spatial-mask hole) is log(0) = -inf; those keys must vanish
    without poisoning the output."""
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
    """A host tensor's data_ptr handed to the preprocess is an asynchronous
    illegal memory access that poisons the CUDA context on some LATER call --
    it must be rejected before launch, not discovered downstream."""
    q, k, v = _qkv(1, 256, 4)
    with pytest.raises((ValueError, RuntimeError), match="key_bias"):
        ck.sol_attn(q, k, v, tau=1.4, key_bias=torch.zeros(256))


def test_direct_backend_validates_like_the_public_path():
    """The backend-direct entry must run the same shared rule as the registry:
    fp16 once ran silently to plausible garbage (bytes reinterpreted as bf16)
    and a shorter k read out of bounds."""
    q, k, v = _qkv(1, 512, 4)
    with pytest.raises(ValueError, match="bfloat16"):
        cuda_backend.sol_attn(q.half(), k.half(), v.half(), tau=1.4)
    with pytest.raises(ValueError, match="shape"):
        cuda_backend.sol_attn(q, k[:, :256].contiguous(), v, tau=1.4)


def test_sub_sm80_rejected_at_the_wrapper(monkeypatch):
    """The sm_75 cubins in a full-arch build compile the guarded kernel bodies
    to a bare EXIT (verified in SASS), so a Turing launch would return
    UNINITIALIZED memory rather than fail. The registry gate reads (and
    caches) the CURRENT device's capability, so on a mixed-arch machine it can
    wave such a call through -- the wrapper must check q.device itself."""
    q, k, v = _qkv(1, 256, 4)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (7, 5))
    with pytest.raises(RuntimeError, match="sm_80"):
        cuda_backend.sol_attn(q, k, v, tau=1.4)


@pytest.mark.parametrize("t", [4096, 1000, 3137])
def test_topk_matches_eager(t):
    """SLA-style top-k selection: the CUDA path implements it as a per-block
    threshold at the (k+1)-th pooled score, so int8 rounding can flip blocks
    right at the boundary -- hence a slightly looser bar than tau parity.
    Ragged tails exercise the host-side threshold's tail-block pooling."""
    q, k, v = _qkv(1, t, 4)
    got = ck.sol_attn(q, k, v, topk_ratio=0.2)
    ref = sol_attn_eager(q, k, v, topk_ratio=0.2)
    assert _cos(got, ref) > 0.995
    # selection changes with the budget
    assert not torch.equal(got, ck.sol_attn(q, k, v, topk_ratio=0.5))


def test_topk_budget_moves_toward_dense():
    """A bigger top-k budget can only move the output toward dense attention."""
    q, k, v = _qkv(1, 4096, 2)
    ref = _dense(q, k, v)
    cs = [_cos(sol_attn_eager(q, k, v, topk_ratio=r), ref)
          for r in (0.05, 0.2, 0.6)]
    assert cs[0] < cs[1] < cs[2]


def test_topk_keeps_sinks_exact():
    """Sinks ride on top of the top-k budget, exactly as in tau mode."""
    q, k, v = _qkv(1, 2048, 2)
    sinks = [0, 4]
    got = ck.sol_attn(q, k, v, topk_ratio=0.1, sink_blocks=sinks)
    ref = sol_attn_eager(q, k, v, topk_ratio=0.1, sink_blocks=sinks)
    assert _cos(got, ref) > 0.995


def test_topk_ratio_validation():
    """The range lives in the shared rule, so every entry rejects it."""
    q, k, v = _qkv(1, 1024, 1)
    with pytest.raises(ValueError, match="topk_ratio"):
        cuda_backend.sol_attn(q, k, v, topk_ratio=1.5)
    with pytest.raises(NoCapableBackendError, match="topk_ratio"):
        ck.sol_attn(q, k, v, topk_ratio=1.5)


@pytest.mark.parametrize("rot", [64, 96])
def test_chunked_producer_matches_separate_rope(rot):
    """The chunked QKV producer (projection output consumed in 64-aligned
    chunks, carriers emitted directly, norm+rope fused, stale-stats centering)
    must match rms_rope_split_half_ + sol_attn. rot=96 is H3's real rot_dim:
    the rope partner shuffle must not assume a power-of-two lane offset. V is
    scaled down to realistic size: blind bootstrap scales quantized it to
    garbage (broke a real run)."""
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


def test_chunked_producer_validates():
    """Chunks reach the producer as bare pointers, so coverage, width and
    device are checked up front; the selection arguments run the shared rule."""
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


def test_chunked_producer_topk():
    """Producer-path top-k: the threshold comes from the workspace's own
    pooled outputs (post-rope, kernel-exact quantization) and must match the
    separate-rope top-k path. perm_d is not an involution -- the centroid is
    the side that gets unpermuted."""
    c = _chunked_case(seed=13, rot=64)
    ref = cuda_backend.sol_attn(c["q"], c["k"], c["v"], topk_ratio=0.2, sink_blocks=[0, 2])
    _, km, vs = cuda_backend.sol_attn_chunked(
        c["chunks"], c["t"], c["h"], c["freqs"], c["norm"], topk_ratio=0.2, sink_blocks=[0, 2])
    out, _, _ = cuda_backend.sol_attn_chunked(
        c["chunks"], c["t"], c["h"], c["freqs"], c["norm"], kmean=km, vscale=vs,
        topk_ratio=0.2, sink_blocks=[0, 2])
    assert _cos(out, ref) > 0.995
