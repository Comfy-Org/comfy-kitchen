import pytest
import torch

import comfy_kitchen as ck

from .conftest import assert_values_close, get_capable_backends, requires_cuda_backend


def _reference_rms_rope(x, freqs_cis, scale, epsilon, *, split_half=False):
    x_float = x.float()
    rrms = torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + epsilon)
    x_norm = (x_float * rrms * scale.float()).to(x.dtype).float()
    freqs = freqs_cis.float()
    if split_half:
        pairs = x_norm.reshape(*x.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2)
        out = freqs[..., 0] * pairs[..., 0] + freqs[..., 1] * pairs[..., 1]
        return out.movedim(-1, -2).reshape_as(x).to(x.dtype)

    pairs = x_norm.reshape(*x.shape[:-1], -1, 1, 2)
    out = freqs[..., 0] * pairs[..., 0] + freqs[..., 1] * pairs[..., 1]
    return out.reshape_as(x).to(x.dtype)


_INTERLEAVED_CONFIGS = [
    ("FLUX", "BHND", (1, 24, 4352, 128)),
    ("LTX", "BHND", (2, 32, 4996, 64)),
    ("ZIMAGE", "BNHD", (1, 4096, 30, 128)),
]

_SPLIT_HALF_CONFIGS = [
    ("WAN", "BNHD", (2, 12288, 16, 128)),
    ("FLUX", "BHND", (1, 24, 4352, 128)),
    ("LTX", "BHND", (2, 32, 4996, 64)),
]


def _shapes(layout, config):
    if layout == "BHND":
        batch, heads, seq_len, head_dim = config
        return (
            (batch, heads, seq_len, head_dim),
            (batch, 1, seq_len, head_dim // 2, 2, 2),
            head_dim,
        )

    batch, seq_len, heads, head_dim = config
    return (
        (batch, seq_len, heads, head_dim),
        (1, seq_len, 1, head_dim // 2, 2, 2),
        head_dim,
    )


def _run_backend(op_name, backend, args, monkeypatch):
    if backend == "cuda":
        from comfy_kitchen.backends.eager import rope as eager_rope

        def fail_fallback(*unused_args, **unused_kwargs):
            raise AssertionError(f"{op_name} unexpectedly used the eager fallback")

        with monkeypatch.context() as patch:
            patch.setattr(eager_rope, op_name, fail_fallback)
            with ck.use_backend(backend):
                return getattr(ck, op_name)(*args)

    with ck.use_backend(backend):
        return getattr(ck, op_name)(*args)


def _max_mismatch(freqs_dtype, dtype):
    if freqs_dtype == torch.bfloat16:
        return 0.35
    if freqs_dtype == torch.float16 or dtype == torch.bfloat16:
        return 0.055
    return 1e-5


class TestRMSRope:
    """Interleaved fused RMSNorm + RoPE functionality and correctness tests."""

    @pytest.mark.parametrize("op_name", ["rms_rope", "rms_rope1"])
    @pytest.mark.parametrize("backend", ["cuda", "triton", "eager"])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
    @pytest.mark.parametrize(
        "freqs_dtype",
        [torch.float32, torch.float16, torch.bfloat16],
        ids=["freqs_fp32", "freqs_fp16", "freqs_bf16"],
    )
    @pytest.mark.parametrize(
        "config_name,layout,config",
        _INTERLEAVED_CONFIGS,
        ids=["FLUX", "LTX", "ZIMAGE"],
    )
    def test_rms_rope_ops(
        self,
        op_name,
        backend,
        device,
        seed,
        dtype,
        freqs_dtype,
        config_name,
        layout,
        config,
        monkeypatch,
    ):
        backends = get_capable_backends(op_name, device)
        if backend not in backends:
            pytest.skip(f"{backend} does not support {op_name} on {device}")

        x_shape, freqs_shape, head_dim = _shapes(layout, config)
        freqs_cis = torch.randn(freqs_shape, dtype=freqs_dtype, device=device)
        q_scale = torch.randn(head_dim, dtype=torch.float32, device=device)

        if op_name == "rms_rope":
            q = torch.randn(x_shape, dtype=dtype, device=device)
            k = torch.randn(x_shape, dtype=dtype, device=device)
            k_scale = torch.randn(head_dim, dtype=torch.float32, device=device)
            q_out, k_out = _run_backend(
                op_name,
                backend,
                (q, k, freqs_cis, q_scale, k_scale),
                monkeypatch,
            )

            q_ref = _reference_rms_rope(q, freqs_cis, q_scale, 1e-6)
            k_ref = _reference_rms_rope(k, freqs_cis, k_scale, 1e-6)
            self._validate(q, q_out, layout, dtype, freqs_dtype, config_name, backend, q_ref)
            self._validate(k, k_out, layout, dtype, freqs_dtype, config_name, backend, k_ref)
        else:
            x = torch.randn(x_shape, dtype=dtype, device=device)
            x_out = _run_backend(
                op_name,
                backend,
                (x, freqs_cis, q_scale),
                monkeypatch,
            )

            x_ref = _reference_rms_rope(x, freqs_cis, q_scale, 1e-6)
            self._validate(x, x_out, layout, dtype, freqs_dtype, config_name, backend, x_ref)

    def _validate(self, x, x_out, layout, dtype, freqs_dtype, config_name, backend, ref):
        assert x_out.shape == x.shape, f"{layout} shape mismatch"
        assert x_out.dtype == x.dtype, f"{layout} dtype mismatch"
        assert x_out.device == x.device
        assert_values_close(
            x_out,
            ref,
            rtol=1e-3,
            atol=1e-3,
            max_mismatch_ratio=_max_mismatch(freqs_dtype, dtype),
            name=f"{config_name} {layout} x ({backend} vs reference, freqs={freqs_dtype})",
        )


class TestRMSRopeSplitHalf:
    """Split-half fused RMSNorm + RoPE functionality and correctness tests."""

    @pytest.mark.parametrize("op_name", ["rms_rope_split_half", "rms_rope_split_half1"])
    @pytest.mark.parametrize("backend", ["cuda", "triton", "eager"])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
    @pytest.mark.parametrize(
        "freqs_dtype",
        [torch.float32, torch.float16, torch.bfloat16],
        ids=["freqs_fp32", "freqs_fp16", "freqs_bf16"],
    )
    @pytest.mark.parametrize(
        "config_name,layout,config",
        _SPLIT_HALF_CONFIGS,
        ids=["WAN", "FLUX", "LTX"],
    )
    def test_rms_split_half_vs_reference(
        self,
        op_name,
        backend,
        device,
        seed,
        dtype,
        freqs_dtype,
        config_name,
        layout,
        config,
        monkeypatch,
    ):
        backends = get_capable_backends(op_name, device)
        if backend not in backends:
            pytest.skip(f"{backend} does not support {op_name} on {device}")

        x_shape, freqs_shape, head_dim = _shapes(layout, config)
        freqs_cis = torch.randn(freqs_shape, dtype=freqs_dtype, device=device)
        q_scale = torch.randn(head_dim, dtype=torch.float32, device=device)

        if op_name == "rms_rope_split_half":
            q = torch.randn(x_shape, dtype=dtype, device=device)
            k = torch.randn(x_shape, dtype=dtype, device=device)
            k_scale = torch.randn(head_dim, dtype=torch.float32, device=device)
            q_out, k_out = _run_backend(
                op_name,
                backend,
                (q, k, freqs_cis, q_scale, k_scale),
                monkeypatch,
            )
            q_ref = _reference_rms_rope(q, freqs_cis, q_scale, 1e-6, split_half=True)
            k_ref = _reference_rms_rope(k, freqs_cis, k_scale, 1e-6, split_half=True)
            self._validate(q, q_out, q_ref, layout, dtype, freqs_dtype, config_name, backend)
            self._validate(k, k_out, k_ref, layout, dtype, freqs_dtype, config_name, backend)
        else:
            x = torch.randn(x_shape, dtype=dtype, device=device)
            x_out = _run_backend(
                op_name,
                backend,
                (x, freqs_cis, q_scale),
                monkeypatch,
            )
            ref = _reference_rms_rope(x, freqs_cis, q_scale, 1e-6, split_half=True)
            self._validate(x, x_out, ref, layout, dtype, freqs_dtype, config_name, backend)

    def _validate(self, x, x_out, ref, layout, dtype, freqs_dtype, config_name, backend):
        assert x_out.shape == x.shape, f"{config_name} {layout} shape mismatch"
        assert x_out.dtype == x.dtype, f"{config_name} {layout} dtype mismatch"
        assert x_out.device == x.device
        assert_values_close(
            x_out,
            ref,
            rtol=1e-3,
            atol=1e-3,
            max_mismatch_ratio=_max_mismatch(freqs_dtype, dtype),
            name=(f"{config_name} {layout} ({backend} vs reference, freqs={freqs_dtype})"),
        )


@pytest.mark.cuda
@requires_cuda_backend
@pytest.mark.parametrize(
    "op_name",
    ["rms_rope", "rms_rope1", "rms_rope_split_half", "rms_rope_split_half1"],
)
@pytest.mark.parametrize("head_dim", [32, 96, 160])
def test_rms_rope_cuda_multiple_of_32(op_name, head_dim, monkeypatch):
    torch.manual_seed(7)
    q = torch.randn(1, 3, 11, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    freqs = torch.randn(1, 1, 11, head_dim // 2, 2, 2, device="cuda", dtype=torch.bfloat16)
    scale = torch.randn(head_dim, device="cuda", dtype=torch.float32)
    split_half = "split_half" in op_name

    if op_name.endswith("1"):
        q_out = _run_backend(op_name, "cuda", (q, freqs, scale), monkeypatch)
    else:
        q_out, k_out = _run_backend(op_name, "cuda", (q, k, freqs, scale), monkeypatch)

    q_ref = _reference_rms_rope(q, freqs, scale, 1e-6, split_half=split_half)
    assert_values_close(
        q_out,
        q_ref,
        rtol=1e-3,
        atol=1e-3,
        max_mismatch_ratio=0.25,
        name=f"{op_name} D={head_dim} q",
    )
    if not op_name.endswith("1"):
        k_ref = _reference_rms_rope(k, freqs, scale, 1e-6, split_half=split_half)
        assert_values_close(
            k_out,
            k_ref,
            rtol=1e-3,
            atol=1e-3,
            max_mismatch_ratio=0.25,
            name=f"{op_name} D={head_dim} k",
        )


@pytest.mark.parametrize("op_name", ["rms_rope1", "rms_rope1_"])
@pytest.mark.parametrize("backend", ["cuda", "triton", "eager"])
def test_rms_rope_nondefault_epsilon_and_broadcast(
    op_name, backend, device, monkeypatch
):
    if backend not in get_capable_backends(op_name, device):
        pytest.skip(f"{backend} does not support {op_name} on {device}")

    epsilon = 0.125
    x = torch.randn(2, 3, 5, 64, device=device, dtype=torch.float16)
    freqs = torch.randn(1, 1, 1, 32, 2, 2, device=device, dtype=torch.float32)
    scale = torch.randn(64, device=device, dtype=torch.float32)
    reference = _reference_rms_rope(x, freqs, scale, epsilon)
    pointer = x.data_ptr()
    actual = _run_backend(op_name, backend, (x, freqs, scale, epsilon), monkeypatch)

    if op_name.endswith("_"):
        assert actual.data_ptr() == pointer
    torch.testing.assert_close(actual, reference, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("backend", ["cuda", "triton", "eager"])
@pytest.mark.parametrize("split_half", [False, True])
@pytest.mark.parametrize("last_dim_strided", [False, True])
def test_rms_rope_strided_views(
    backend, split_half, last_dim_strided, device, monkeypatch
):
    op_name = "rms_rope_split_half1" if split_half else "rms_rope1"
    if backend not in get_capable_backends(op_name, device):
        pytest.skip(f"{backend} does not support {op_name} on {device}")
    head_dim = 64
    width = head_dim * (2 if last_dim_strided else 1)
    storage = torch.randn(2, 6, 5, width, device=device, dtype=torch.float16)
    x = storage[:, ::2, :, ::2] if last_dim_strided else storage[:, ::2]
    freq_storage = torch.randn(
        2, 1, 5, head_dim, 4, 4, device=device, dtype=torch.float32
    )
    freqs = freq_storage[:, :, :, ::2, ::2, ::2]
    scale_storage = torch.randn(head_dim * 2, device=device, dtype=torch.float32)
    scale = scale_storage[::2]
    reference = _reference_rms_rope(
        x, freqs, scale, 1e-6, split_half=split_half
    )
    actual = _run_backend(op_name, backend, (x, freqs, scale), monkeypatch)
    torch.testing.assert_close(actual, reference, rtol=2e-3, atol=2e-3)


def test_rms_rope_triton_expanded_input(device):
    if "triton" not in get_capable_backends("rms_rope1", device):
        pytest.skip(f"triton does not support rms_rope1 on {device}")

    x = torch.randn(1, 1, 1, 64, device=device, dtype=torch.float16).expand(
        2, 3, 5, 64
    )
    freqs = torch.randn(2, 1, 5, 32, 2, 2, device=device, dtype=torch.float32)
    scale = torch.randn(64, device=device, dtype=torch.float32)
    reference = _reference_rms_rope(x, freqs, scale, 1e-6)

    with ck.use_backend("triton"):
        actual = ck.rms_rope1(x, freqs, scale)

    torch.testing.assert_close(actual, reference, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize(
    "op_name",
    [
        "rms_rope",
        "rms_rope_",
        "rms_rope_split_half",
        "rms_rope_split_half_",
    ],
)
@pytest.mark.parametrize("backend", ["cuda", "triton"])
@pytest.mark.parametrize("layout", ["BHND", "BNHD"])
def test_rms_rope_gqa_different_qk_shapes(
    op_name, backend, layout, device, monkeypatch
):
    if backend not in get_capable_backends(op_name, device):
        pytest.skip(f"{backend} does not support {op_name} on {device}")

    if layout == "BHND":
        q_shape = (2, 8, 5, 64)
        k_shape = (2, 2, 5, 64)
        freqs_shape = (1, 1, 5, 32, 2, 2)
    else:
        q_shape = (2, 5, 8, 64)
        k_shape = (2, 5, 2, 64)
        freqs_shape = (1, 5, 1, 32, 2, 2)

    q = torch.randn(q_shape, device=device, dtype=torch.float16)
    k = torch.randn(k_shape, device=device, dtype=torch.float16)
    freqs = torch.randn(freqs_shape, device=device, dtype=torch.float32)
    q_scale = torch.randn(64, device=device, dtype=torch.float32)
    k_scale = torch.randn(64, device=device, dtype=torch.float32)
    split_half = "split_half" in op_name
    reference = (
        _reference_rms_rope(q, freqs, q_scale, 1e-6, split_half=split_half),
        _reference_rms_rope(k, freqs, k_scale, 1e-6, split_half=split_half),
    )

    pointers = (q.data_ptr(), k.data_ptr())
    actual = _run_backend(
        op_name, backend, (q, k, freqs, q_scale, k_scale), monkeypatch
    )

    assert actual[0].shape == q_shape
    assert actual[1].shape == k_shape
    if op_name.endswith("_"):
        assert (actual[0].data_ptr(), actual[1].data_ptr()) == pointers
    for result, expected in zip(actual, reference, strict=True):
        torch.testing.assert_close(result, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize(
    "op_name",
    [
        "rms_rope_",
        "rms_rope1_",
        "rms_rope_split_half_",
        "rms_rope_split_half1_",
    ],
)
@pytest.mark.parametrize("backend", ["cuda", "triton", "eager"])
def test_rms_rope_inplace_storage(op_name, backend, device, monkeypatch):
    if backend not in get_capable_backends(op_name, device):
        pytest.skip(f"{backend} does not support {op_name} on {device}")
    paired = not op_name.endswith("1_")
    functional_name = op_name[:-1]
    freqs = torch.randn(1, 1, 3, 32, 2, 2, device=device, dtype=torch.float32)
    q = torch.randn(2, 4, 3, 128, device=device, dtype=torch.float16)[..., ::2]
    k = torch.randn_like(q)
    scale = torch.randn(128, device=device, dtype=torch.float32)[::2]
    args = (q, k, freqs, scale) if paired else (q, freqs, scale)
    reference_args = tuple(a.clone() if torch.is_tensor(a) else a for a in args)
    with ck.use_backend("eager"):
        reference = getattr(ck, functional_name)(*reference_args)
    tensors = (q, k) if paired else (q,)
    pointers = tuple(x.data_ptr() for x in tensors)
    strides = tuple(x.stride() for x in tensors)
    result = _run_backend(op_name, backend, args, monkeypatch)
    outputs = result if paired else (result,)
    assert tuple(x.data_ptr() for x in outputs) == pointers
    assert tuple(x.stride() for x in outputs) == strides
    refs = reference if paired else (reference,)
    for actual, expected in zip(outputs, refs, strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


class TestPartialRotary:
    """rot_dim rotates a head-dim prefix in split-half pairs (i, i + rot_dim//2);
    the RMS norm always spans the full head_dim and the tail passes through."""

    _S, _H, _D, _ROT = 333, 8, 128, 96

    def _reference(self, x, freqs_cis, scale, epsilon, rot_dim):
        x_float = x.float()
        rrms = torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + epsilon)
        x_norm = (x_float * rrms * scale.float()).to(x.dtype).float()
        half = rot_dim // 2
        x1, x2 = x_norm[..., :half], x_norm[..., half:rot_dim]
        f = freqs_cis.float()
        o1 = f[..., 0, 0] * x1 + f[..., 0, 1] * x2
        o2 = f[..., 1, 0] * x1 + f[..., 1, 1] * x2
        return torch.cat([o1, o2, x_norm[..., rot_dim:]], dim=-1).to(x.dtype)

    def _inputs(self, device, dtype):
        q = torch.randn(1, self._S, self._H, self._D, dtype=dtype, device=device)
        k = torch.randn(1, self._S, self._H, self._D, dtype=dtype, device=device)
        freqs = torch.randn(1, self._S, 1, self._ROT // 2, 2, 2, dtype=torch.float32, device=device)
        q_scale = torch.randn(self._D, dtype=torch.float32, device=device)
        k_scale = torch.randn(self._D, dtype=torch.float32, device=device)
        return q, k, freqs, q_scale, k_scale

    @pytest.mark.parametrize("backend", ["cuda", "triton", "eager"])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
    def test_partial_vs_reference(self, backend, device, seed, dtype, monkeypatch):
        if backend not in get_capable_backends("rms_rope_split_half", device):
            pytest.skip(f"{backend} does not support rms_rope_split_half on {device}")
        q, k, freqs, q_scale, k_scale = self._inputs(device, dtype)
        if backend == "cuda":
            # the fused kernel must serve partial rotary itself, not fall back
            q_out, k_out = _run_backend(
                "rms_rope_split_half", backend,
                (q, k, freqs, q_scale, k_scale, 1e-6, self._ROT), monkeypatch)
        else:
            with ck.use_backend(backend):
                q_out, k_out = ck.rms_rope_split_half(
                    q, k, freqs, q_scale, k_scale, rot_dim=self._ROT)
        for name, out, x, scale in (("q", q_out, q, q_scale), ("k", k_out, k, k_scale)):
            ref = self._reference(x, freqs, scale, 1e-6, self._ROT)
            assert_values_close(out, ref, rtol=1e-3, atol=1e-3,
                                max_mismatch_ratio=_max_mismatch(torch.float32, dtype),
                                name=f"partial rotary {name} ({backend})")

    @pytest.mark.parametrize("backend", ["cuda", "eager"])
    def test_full_rot_dim_matches_default(self, backend, device, seed, monkeypatch):
        """rot_dim == head_dim must be bit-identical to the default full rotation."""
        if backend not in get_capable_backends("rms_rope_split_half", device):
            pytest.skip(f"{backend} does not support rms_rope_split_half on {device}")
        q, k, _, q_scale, k_scale = self._inputs(device, torch.bfloat16)
        freqs = torch.randn(1, self._S, 1, self._D // 2, 2, 2, dtype=torch.float32, device=device)
        with ck.use_backend(backend):
            full = ck.rms_rope_split_half(q, k, freqs, q_scale, k_scale, rot_dim=self._D)
            default = ck.rms_rope_split_half(q, k, freqs, q_scale, k_scale)
        assert torch.equal(full[0], default[0]) and torch.equal(full[1], default[1])

    @pytest.mark.parametrize("backend", ["cuda", "triton", "eager"])
    def test_inplace_packed_qkv(self, backend, device, seed, monkeypatch):
        """In-place on q/k slices of a fused QKV projection: interleaved views of
        one buffer, bounds overlap but the element sets are disjoint."""
        if backend not in get_capable_backends("rms_rope_split_half_", device):
            pytest.skip(f"{backend} does not support rms_rope_split_half_ on {device}")
        S, H, D, ROT = self._S, self._H, self._D, self._ROT
        qkv = torch.randn(S, 3 * H * D, dtype=torch.bfloat16, device=device)
        qkv_ref = qkv.clone()
        freqs = torch.randn(1, S, 1, ROT // 2, 2, 2, dtype=torch.float32, device=device)
        q_scale = torch.randn(D, dtype=torch.float32, device=device)
        k_scale = torch.randn(D, dtype=torch.float32, device=device)
        q, k, v = (t.view(1, S, H, D) for t in qkv.split(H * D, dim=-1))
        qr, kr, vr = (t.view(1, S, H, D) for t in qkv_ref.split(H * D, dim=-1))
        with ck.use_backend(backend):
            ck.rms_rope_split_half_(q, k, freqs, q_scale, k_scale, rot_dim=ROT)
            q_ref, k_ref = ck.rms_rope_split_half(qr, kr, freqs, q_scale, k_scale, rot_dim=ROT)
        assert torch.equal(q, q_ref) and torch.equal(k, k_ref)
        assert torch.equal(v, vr), "v must not be touched by in-place q/k rope"

    def test_inplace_rejects_true_overlap(self, device, seed):
        """Views that genuinely share elements must still be rejected."""
        buf = torch.randn(64, 256, dtype=torch.bfloat16, device=device)
        q = buf[:, :128].view(1, 64, 1, 128)
        k = buf[:, 64:192].view(1, 64, 1, 128)  # overlaps q's second half
        freqs = torch.randn(1, 64, 1, 64, 2, 2, dtype=torch.float32, device=device)
        scale = torch.randn(128, dtype=torch.float32, device=device)
        with pytest.raises(ValueError, match="non-overlapping"):
            ck.rms_rope_split_half_(q, k, freqs, scale, scale)

    def test_inplace_rejects_nonmultiple_strides(self, device, seed):
        """Same-layout views whose outer strides are not multiples of each other
        can collide through index combinations; the check stays conservative."""
        base = torch.randn(4096, dtype=torch.bfloat16, device=device)
        x = base.as_strided((3, 4, 20), (1000, 130, 1), 0).unsqueeze(0)
        y = base.as_strided((3, 4, 20), (1000, 130, 1), 40).unsqueeze(0)
        freqs = torch.randn(1, 3, 1, 10, 2, 2, dtype=torch.float32, device=device)
        scale = torch.randn(20, dtype=torch.float32, device=device)
        with pytest.raises(ValueError, match="non-overlapping"):
            ck.rms_rope_split_half_(x, y, freqs, scale, scale)
