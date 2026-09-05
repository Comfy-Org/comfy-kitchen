import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen._rope_utils import check_rope_inplace

from .conftest import assert_values_close, get_capable_backends


def _reference_apply_rope(
    t: torch.Tensor, freqs: torch.Tensor, *, split_half: bool = False
) -> torch.Tensor:
    if split_half:
        pairs = (
            t.reshape(*t.shape[:-1], 2, -1)
            .movedim(-2, -1)
            .unsqueeze(-2)
            .to(freqs.dtype)
        )
        out = freqs[..., 0] * pairs[..., 0] + freqs[..., 1] * pairs[..., 1]
        return out.movedim(-1, -2).reshape_as(t).type_as(t)

    pairs = t.to(freqs.dtype).reshape(*t.shape[:-1], -1, 1, 2)
    out = freqs[..., 0] * pairs[..., 0] + freqs[..., 1] * pairs[..., 1]
    return out.reshape_as(t).type_as(t)


class TestApplyRope:
    """RoPE (Rotary Position Embedding) tests."""
    @pytest.mark.parametrize("op_name", ["apply_rope", "apply_rope1"])
    @pytest.mark.parametrize("backend", ["cuda", "hip", "triton", "eager"])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
    @pytest.mark.parametrize("freqs_dtype", [torch.float32, torch.float16, torch.bfloat16], ids=["freqs_fp32", "freqs_fp16", "freqs_bf16"])
    @pytest.mark.parametrize("config_name,layout,config", [
        ("FLUX", "BHND", (1, 24, 4352, 128)),
        ("LTX", "BHND", (2, 32, 4996, 64)),
        ("ZIMAGE", "BNHD", (1, 4096, 30, 128)),
    ], ids=lambda cfg: f"{cfg[0]}")
    def test_rope_ops(self, op_name, backend, device, seed, dtype, freqs_dtype, config_name, layout, config):
        """Test RoPE operations (apply_rope and apply_rope1) for a specific backend."""
        backends = get_capable_backends(op_name, device)
        if backend not in backends:
            pytest.skip(f"{backend} does not support {op_name} on {device}")

        if layout == "BHND":
            b, h, n, d = config
            x_shape = (b, h, n, d)
            freqs_shape = (b, 1, n, d // 2, 2, 2)  # broadcast over heads
        else:  # BNHD
            b, n, h, d = config
            x_shape = (b, n, h, d)
            freqs_shape = (1, n, 1, d // 2, 2, 2)  # broadcast over batch and heads

        freqs_cis = torch.randn(freqs_shape, dtype=freqs_dtype, device=device)

        # Run operation based on type
        if op_name == "apply_rope":
            xq = torch.randn(x_shape, dtype=dtype, device=device)
            xk = torch.randn(x_shape, dtype=dtype, device=device)

            with ck.use_backend(backend):
                xq_out, xk_out = ck.apply_rope(xq, xk, freqs_cis)

            ref_xq = _reference_apply_rope(xq, freqs_cis)
            ref_xk = _reference_apply_rope(xk, freqs_cis)
            self._validate(xq, xq_out, layout, dtype, freqs_dtype, config_name, backend, ref_xq)
            self._validate(xk, xk_out, layout, dtype, freqs_dtype, config_name, backend, ref_xk)

        else:  # apply_rope1
            x = torch.randn(x_shape, dtype=dtype, device=device)

            with ck.use_backend(backend):
                x_out = ck.apply_rope1(x, freqs_cis)

            ref_x = _reference_apply_rope(x, freqs_cis)
            self._validate(x, x_out, layout, dtype, freqs_dtype, config_name, backend, ref_x)

    def _validate(self, x, x_out, layout, dtype, freqs_dtype, config_name, backend, ref_x):
        assert x_out.shape == x.shape, f"{layout} shape mismatch"
        assert x_out.dtype == x.dtype, f"{layout} dtype mismatch"
        assert x_out.device == x.device
        assert_values_close(
            x_out,
            ref_x,
            rtol=1e-3,
            atol=1e-3,
            max_mismatch_ratio=_max_mismatch(freqs_dtype, dtype),
            name=f"{config_name} {layout} x ({backend} vs reference, freqs={freqs_dtype})",
        )


class TestApplyRopeSplitHalf:
    """Tests for apply_rope_split_half and apply_rope_split_half1."""

    @pytest.mark.parametrize("op_name", ["apply_rope_split_half", "apply_rope_split_half1"])
    @pytest.mark.parametrize("backend", ["cuda", "hip", "triton", "eager"])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
    @pytest.mark.parametrize("freqs_dtype", [torch.float32, torch.float16, torch.bfloat16], ids=["freqs_fp32", "freqs_fp16", "freqs_bf16"])
    @pytest.mark.parametrize("config_name,layout,config", [
        ("WAN", "BNHD", (2, 12288, 16, 128)),
        ("FLUX", "BHND", (1, 24, 4352, 128)),
        ("LTX", "BHND", (2, 32, 4996, 64)),
    ], ids=lambda cfg: f"{cfg[0]}")
    def test_apply_split_half_vs_reference(self, op_name, backend, device, seed, dtype, freqs_dtype, config_name, layout, config):
        """Verify split-half backends match the Python reference formula."""
        backends = get_capable_backends(op_name, device)
        if backend not in backends:
            pytest.skip(f"{backend} does not support {op_name} on {device}")

        if layout == "BHND":
            b, h, n, d = config
            x_shape = (b, h, n, d)
            freqs_shape = (b, 1, n, d // 2, 2, 2)
        else:  # BNHD
            b, n, h, d = config
            x_shape = (b, n, h, d)
            freqs_shape = (1, n, 1, d // 2, 2, 2)

        freqs_cis = torch.randn(freqs_shape, dtype=freqs_dtype, device=device)

        if op_name == "apply_rope_split_half":
            xq = torch.randn(x_shape, dtype=dtype, device=device)
            xk = torch.randn(x_shape, dtype=dtype, device=device)

            with ck.use_backend(backend):
                xq_out, xk_out = ck.apply_rope_split_half(xq, xk, freqs_cis)

            ref_xq = _reference_apply_rope(xq, freqs_cis, split_half=True)
            ref_xk = _reference_apply_rope(xk, freqs_cis, split_half=True)

            self._validate(xq, xq_out, ref_xq, layout, dtype, freqs_dtype, config_name, backend)
            self._validate(xk, xk_out, ref_xk, layout, dtype, freqs_dtype, config_name, backend)
        else:  # apply_rope_split_half1
            x = torch.randn(x_shape, dtype=dtype, device=device)

            with ck.use_backend(backend):
                x_out = ck.apply_rope_split_half1(x, freqs_cis)

            ref_x = _reference_apply_rope(x, freqs_cis, split_half=True)
            self._validate(x, x_out, ref_x, layout, dtype, freqs_dtype, config_name, backend)

    def _validate(self, x, x_out, ref, layout, dtype, freqs_dtype, config_name, backend):
        assert x_out.shape == x.shape, f"{config_name} {layout} shape mismatch"
        assert x_out.dtype == x.dtype, f"{config_name} {layout} dtype mismatch"
        assert x_out.device == x.device

        mm = _max_mismatch(freqs_dtype, dtype)
        assert_values_close(
            x_out, ref, rtol=1e-3, atol=1e-3, max_mismatch_ratio=mm,
            name=f"{config_name} {layout} ({backend} vs reference, freqs={freqs_dtype})"
        )


def _max_mismatch(freqs_dtype, dtype):
    if freqs_dtype == torch.bfloat16:
        return 0.25
    if freqs_dtype == torch.float16 or dtype == torch.bfloat16:
        return 0.05
    return 1e-5


@pytest.mark.parametrize("backend", ["cuda", "hip", "triton", "eager"])
@pytest.mark.parametrize("split_half", [False, True])
def test_apply_rope_broadcasts_single_frequency(backend, split_half, device):
    op_name = "apply_rope_split_half1" if split_half else "apply_rope1"
    if backend not in get_capable_backends(op_name, device):
        pytest.skip(f"{backend} does not support {op_name} on {device}")

    x = torch.randn(2, 3, 5, 64, device=device, dtype=torch.float16)
    freqs = torch.randn(1, 1, 1, 32, 2, 2, device=device, dtype=torch.float32)
    reference = _reference_apply_rope(x, freqs, split_half=split_half)
    with ck.use_backend(backend):
        actual = getattr(ck, op_name)(x, freqs)
    torch.testing.assert_close(actual, reference, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("backend", ["cuda", "hip", "triton", "eager"])
@pytest.mark.parametrize(
    "q_seq_len,k_seq_len",
    [(3, 3), (3, 5)],
    ids=["shared-short-sequence", "short-query-long-key"],
)
def test_apply_rope_trims_excess_sequence_frequencies(
    backend, q_seq_len, k_seq_len, device
):
    if backend not in get_capable_backends("apply_rope", device):
        pytest.skip(f"{backend} does not support apply_rope on {device}")

    q = torch.randn(2, 4, q_seq_len, 64, device=device, dtype=torch.float16)
    k = torch.randn(2, 4, k_seq_len, 64, device=device, dtype=torch.float16)
    freqs = torch.randn(1, 1, 5, 32, 2, 2, device=device, dtype=torch.float32)
    reference = (
        _reference_apply_rope(q, freqs[:, :, :q_seq_len]),
        _reference_apply_rope(k, freqs[:, :, :k_seq_len]),
    )

    with ck.use_backend(backend):
        actual = ck.apply_rope(q, k, freqs)

    for result, expected in zip(actual, reference, strict=True):
        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("backend", ["cuda", "hip", "triton", "eager"])
@pytest.mark.parametrize("split_half", [False, True])
@pytest.mark.parametrize("last_dim_strided", [False, True])
def test_apply_rope_strided_views(backend, split_half, last_dim_strided, device):
    op_name = "apply_rope_split_half1" if split_half else "apply_rope1"
    if backend not in get_capable_backends(op_name, device):
        pytest.skip(f"{backend} does not support {op_name} on {device}")

    head_dim = 64
    width = head_dim * (2 if last_dim_strided else 1)
    x_storage = torch.randn(2, 6, 5, width, device=device, dtype=torch.float16)
    x = x_storage[:, ::2, :, ::2] if last_dim_strided else x_storage[:, ::2]
    freq_storage = torch.randn(
        2, 1, 5, head_dim, 4, 4, device=device, dtype=torch.float32
    )
    freqs = freq_storage[:, :, :, ::2, ::2, ::2]
    reference = _reference_apply_rope(x, freqs, split_half=split_half)
    with ck.use_backend(backend):
        actual = getattr(ck, op_name)(x, freqs)
    torch.testing.assert_close(actual, reference, rtol=1e-3, atol=1e-3)


def test_apply_rope_triton_expanded_input(device):
    if "triton" not in get_capable_backends("apply_rope1", device):
        pytest.skip(f"triton does not support apply_rope1 on {device}")

    x = torch.randn(1, 1, 1, 64, device=device, dtype=torch.float16).expand(
        2, 3, 5, 64
    )
    freqs = torch.randn(2, 1, 5, 32, 2, 2, device=device, dtype=torch.float32)
    reference = _reference_apply_rope(x, freqs)

    with ck.use_backend("triton"):
        actual = ck.apply_rope1(x, freqs)

    torch.testing.assert_close(actual, reference, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "op_name",
    [
        "apply_rope",
        "apply_rope_",
        "apply_rope_split_half",
        "apply_rope_split_half_",
    ],
)
@pytest.mark.parametrize("backend", ["cuda", "hip", "triton"])
@pytest.mark.parametrize("layout", ["BHND", "BNHD"])
def test_apply_rope_gqa_different_qk_shapes(op_name, backend, layout, device):
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
    split_half = "split_half" in op_name
    reference = (
        _reference_apply_rope(q, freqs, split_half=split_half),
        _reference_apply_rope(k, freqs, split_half=split_half),
    )

    pointers = (q.data_ptr(), k.data_ptr())
    with ck.use_backend(backend):
        actual = getattr(ck, op_name)(q, k, freqs)

    assert actual[0].shape == q_shape
    assert actual[1].shape == k_shape
    if op_name.endswith("_"):
        assert (actual[0].data_ptr(), actual[1].data_ptr()) == pointers
    for result, expected in zip(actual, reference, strict=True):
        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "op_name",
    [
        "apply_rope",
        "apply_rope_",
        "apply_rope_split_half",
        "apply_rope_split_half_",
    ],
)
@pytest.mark.parametrize("backend", ["cuda", "hip", "triton"])
def test_apply_rope_paired_different_strides(op_name, backend, device):
    if backend not in get_capable_backends(op_name, device):
        pytest.skip(f"{backend} does not support {op_name} on {device}")

    q = torch.randn(2, 4, 3, 64, device=device, dtype=torch.float16)
    k = torch.randn(2, 4, 3, 128, device=device, dtype=torch.float16)[..., ::2]
    assert q.shape == k.shape
    assert q.stride() != k.stride()
    freqs = torch.randn(1, 1, 3, 32, 2, 2, device=device, dtype=torch.float32)
    split_half = "split_half" in op_name
    reference = (
        _reference_apply_rope(q, freqs, split_half=split_half),
        _reference_apply_rope(k, freqs, split_half=split_half),
    )

    pointers = (q.data_ptr(), k.data_ptr())
    strides = (q.stride(), k.stride())
    with ck.use_backend(backend):
        actual = getattr(ck, op_name)(q, k, freqs)

    if op_name.endswith("_"):
        assert (actual[0].data_ptr(), actual[1].data_ptr()) == pointers
        assert (actual[0].stride(), actual[1].stride()) == strides
    for result, expected in zip(actual, reference, strict=True):
        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "op_name",
    [
        "apply_rope_",
        "apply_rope1_",
        "apply_rope_split_half_",
        "apply_rope_split_half1_",
    ],
)
@pytest.mark.parametrize("backend", ["cuda", "hip", "triton", "eager"])
def test_apply_rope_inplace_storage(op_name, backend, device):
    if backend not in get_capable_backends(op_name, device):
        pytest.skip(f"{backend} does not support {op_name} on {device}")
    paired = not op_name.endswith("1_")
    functional_name = op_name[:-1]
    freqs = torch.randn(1, 1, 3, 32, 2, 2, device=device, dtype=torch.float32)
    q = torch.randn(2, 4, 3, 128, device=device, dtype=torch.float16)[..., ::2]
    k = torch.randn_like(q)
    args = (q, k, freqs) if paired else (q, freqs)
    with ck.use_backend("eager"):
        reference = getattr(ck, functional_name)(*(tuple(a.clone() if torch.is_tensor(a) else a for a in args)))
    pointers = tuple(x.data_ptr() for x in ((q, k) if paired else (q,)))
    strides = tuple(x.stride() for x in ((q, k) if paired else (q,)))
    with ck.use_backend(backend):
        result = getattr(ck, op_name)(*args)
    outputs = result if paired else (result,)
    assert tuple(x.data_ptr() for x in outputs) == pointers
    assert tuple(x.stride() for x in outputs) == strides
    refs = reference if paired else (reference,)
    for actual, expected in zip(outputs, refs, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)



@pytest.mark.parametrize("op_name", ["apply_rope1_", "rms_rope1_"])
@pytest.mark.parametrize("backend", ["cuda", "hip", "triton", "eager"])
def test_inplace_backends_reject_autograd(op_name, backend, device):
    if backend not in get_capable_backends(op_name, device):
        pytest.skip(f"{backend} does not support {op_name} on {device}")

    x = torch.randn(
        1, 1, 1, 64, device=device, dtype=torch.float16, requires_grad=True
    )
    freqs = torch.randn(1, 1, 1, 32, 2, 2, device=device, dtype=torch.float32)
    args = (x, freqs)
    if op_name == "rms_rope1_":
        scale = torch.randn(64, device=device, dtype=torch.float32)
        args += (scale,)

    with ck.use_backend(backend), pytest.raises(RuntimeError, match="inference-only"):
        getattr(ck, op_name)(*args)


def test_check_rope_inplace_rejects_internal_overlap():
    x = torch.empty(1).expand(2)
    with pytest.raises(ValueError, match="internal overlap"):
        check_rope_inplace(x)


def test_check_rope_inplace_rejects_paired_overlap():
    storage = torch.empty(12)
    q = storage[:8]
    k = storage[4:]
    with pytest.raises(ValueError, match="non-overlapping input storage"):
        check_rope_inplace(q, k)


def test_check_rope_inplace_rejects_readonly_overlap():
    storage = torch.empty(12)
    x = storage[:8]
    readonly = storage[4:]
    with pytest.raises(ValueError, match="must not overlap"):
        check_rope_inplace(x, readonly=(readonly,))
