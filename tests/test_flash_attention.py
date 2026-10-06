import pytest
import torch

import comfy_kitchen as ck
import comfy_kitchen.flash_attention as flash_attention_module

requires_flash_decode = pytest.mark.skipif(
    not ck.flash_attention_decode_is_available(),
    reason="requires a supported CUDA or HIP flash decode kernel",
)

# The HIP binding takes plain ndarrays rather than device-typed ones, so it has
# to reject a host operand and an off-boundary base itself.
requires_hip_flash_decode = pytest.mark.skipif(
    not ck.flash_attention_decode_is_available() or not getattr(torch.version, "hip", None),
    reason="requires the HIP flash decode kernel",
)

HEAD_DIMS = [128, 256]


def _decode_operands(batch=2, capacity=256, kv_heads=2, groups=4, head_dim=128):
    heads = kv_heads * groups
    q = torch.randn(batch, 1, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, capacity, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    lengths = torch.full((batch,), capacity, device="cuda", dtype=torch.int32)
    return q, k, v, lengths


@requires_hip_flash_decode
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_flash_attention_decode_rejects_host_lengths(head_dim):
    q, k, v, lengths = _decode_operands(head_dim=head_dim)
    with pytest.raises(RuntimeError, match="must be ROCm device memory"):
        ck.flash_attention_decode(q, k, v, lengths.cpu())
    # The rejection must leave the context usable.
    assert torch.isfinite(ck.flash_attention_decode(q, k, v, lengths).float()).all()


@requires_hip_flash_decode
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_flash_attention_decode_rejects_pinned_host_memory(head_dim):
    # Pinned host memory reports kDLCUDAHost rather than kDLCPU, so a check for
    # "not the CPU device" would wave these host pointers through. The launch
    # goes through _C directly because the Python wrapper's stream lookup
    # rejects a host tensor before the binding sees it.
    from comfy_kitchen.backends import hip as hip_backend

    batch, capacity, kv_heads, groups = 2, 128, 2, 4
    pinned = [
        torch.randn(batch * groups, kv_heads, head_dim, dtype=torch.bfloat16).pin_memory(),
        torch.randn(batch, capacity, kv_heads, head_dim, dtype=torch.bfloat16).pin_memory(),
        torch.randn(batch, capacity, kv_heads, head_dim, dtype=torch.bfloat16).pin_memory(),
        torch.full((batch,), capacity, dtype=torch.int32).pin_memory(),
        torch.empty(batch * groups, kv_heads, head_dim, dtype=torch.bfloat16).pin_memory(),
        torch.empty(batch * kv_heads * groups, dtype=torch.float32).pin_memory(),
    ]
    assert pinned[0].__dlpack_device__()[0] != 1
    empty = pinned[-1][:0]
    args = [hip_backend._dl(t) for t in (*pinned, empty, empty)]
    with pytest.raises(RuntimeError, match="must be ROCm device memory"):
        hip_backend._C.flash_attention_decode(*args, 1, 0)


@requires_hip_flash_decode
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_flash_attention_decode_rejects_strided_lengths(head_dim):
    # Read linearly off the base pointer, so a strided view of the right size
    # would silently be taken as packed.
    q, k, v, lengths = _decode_operands(head_dim=head_dim)
    strided = torch.stack([lengths, torch.zeros_like(lengths)], dim=1).flatten()[::2]
    assert not strided.is_contiguous() and strided.numel() == lengths.numel()
    with pytest.raises(RuntimeError, match="must be contiguous"):
        ck.flash_attention_decode(q, k, v, strided)


@requires_hip_flash_decode
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_flash_attention_decode_rejects_misaligned_operand(head_dim):
    q, k, v, lengths = _decode_operands(head_dim=head_dim)
    elements = k.numel()
    storage = torch.randn(elements + 8, device="cuda", dtype=torch.bfloat16)
    misaligned = storage[1 : 1 + elements].view_as(k)
    assert misaligned.is_contiguous() and misaligned.data_ptr() % 8
    with pytest.raises(RuntimeError, match="8-byte aligned"):
        ck.flash_attention_decode(q, misaligned, v, lengths)


@requires_hip_flash_decode
def test_flash_attention_decode_256_with_eight_byte_alignment():
    q, k, v, lengths = _decode_operands(head_dim=256)
    storage = torch.randn(k.numel() + 4, device="cuda", dtype=torch.bfloat16)
    k = storage[4:].view_as(k)
    assert k.data_ptr() % 16 == 8
    actual = ck.flash_attention_decode(q, k, v, lengths)
    torch.testing.assert_close(actual, _reference(q, k, v, lengths), atol=2e-3, rtol=1e-2)



def _reference(q, k, v, lengths):
    groups = q.shape[2] // k.shape[2]
    outputs = []
    for batch, length in enumerate(lengths.tolist()):
        query = q[batch].transpose(0, 1).unsqueeze(0)
        key = k[batch, :length].transpose(0, 1).repeat_interleave(groups, dim=0).unsqueeze(0)
        value = v[batch, :length].transpose(0, 1).repeat_interleave(groups, dim=0).unsqueeze(0)
        output = torch.nn.functional.scaled_dot_product_attention(query, key, value)
        outputs.append(output.squeeze(0).transpose(0, 1))
    return torch.stack(outputs)


def test_flash_attention_decode_availability_is_bool():
    assert isinstance(ck.flash_attention_decode_is_available(), bool)


@pytest.mark.parametrize(
    ("capability", "has_kernel", "expected"),
    [
        ((7, 5), True, False),
        ((8, 0), True, True),
        ((9, 0), True, True),
        ((9, 0), False, False),
    ],
)
def test_flash_attention_decode_availability_checks_capability_and_kernel(
    monkeypatch, capability, has_kernel, expected
):
    if getattr(torch.version, "hip", None):
        pytest.skip("flash_attention_decode is CUDA-only")

    class Extension:
        pass

    extension = Extension()
    if has_kernel:
        extension.flash_attention_decode = object()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: capability)
    monkeypatch.setattr(flash_attention_module._cuda_backend, "_EXT_AVAILABLE", True)
    monkeypatch.setattr(flash_attention_module._cuda_backend, "_C", extension)

    assert flash_attention_module.is_available() is expected


@pytest.mark.parametrize("has_wmma", [True, False])
def test_flash_attention_decode_hip_gate_follows_bf16_arch(monkeypatch, has_wmma):
    """On ROCm the gate is the arch's bf16 support, not a compute capability.

    torch.cuda is the ROCm API there and reports an SM-shaped capability for a
    gfx part, so the CUDA test above would wave RDNA2 through. RDNA2 has no
    bf16, and a caller that drops to another dtype there arrives with a KV
    cache this kernel declines. WMMA marks the same line: gfx11 and newer.
    """
    if not getattr(torch.version, "hip", None):
        pytest.skip("requires a ROCm PyTorch runtime")
    if not flash_attention_module._hip_backend._EXT_AVAILABLE:
        pytest.skip("requires the built HIP extension")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        flash_attention_module._hip_backend, "has_wmma", lambda: has_wmma
    )
    assert flash_attention_module.is_available() is has_wmma


@requires_flash_decode
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize(("heads", "kv_heads"), [(4, 4), (8, 2), (24, 4), (32, 4)])
@pytest.mark.parametrize("capacity", [257, 32768])
@pytest.mark.parametrize("transposed_cache", [False, True])
def test_flash_attention_decode(head_dim, heads, kv_heads, capacity, transposed_cache):
    torch.manual_seed(0)
    q = torch.randn(3, 1, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(3, capacity, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    if transposed_cache:
        k = k.transpose(1, 2).contiguous().transpose(1, 2)
        v = v.transpose(1, 2).contiguous().transpose(1, 2)
    lengths = torch.tensor([1, 73, capacity], device="cuda", dtype=torch.int32)
    actual = ck.flash_attention_decode(q, k, v, lengths)
    torch.testing.assert_close(actual, _reference(q, k, v, lengths), atol=2e-3, rtol=1e-2)


@requires_flash_decode
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("num_splits", [1, 2, 5, 8, 16, 32])
def test_flash_attention_decode_split_counts(monkeypatch, num_splits, head_dim):
    # num_splits comes from a heuristic over multi_processor_count, so on a wide
    # enough GPU it returns 1 and the split accumulators, the combine pass and
    # its row decomposition never run. Pin it instead of hoping.
    monkeypatch.setattr(flash_attention_module, "_num_splits", lambda *_, **__: num_splits)
    torch.manual_seed(0)
    q = torch.randn(3, 1, 24, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(3, 257, 4, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    lengths = torch.tensor([1, 73, 257], device="cuda", dtype=torch.int32)
    actual = ck.flash_attention_decode(q, k, v, lengths)
    torch.testing.assert_close(actual, _reference(q, k, v, lengths), atol=2e-3, rtol=1e-2)


@requires_flash_decode
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_flash_attention_decode_cuda_graph_dynamic_lengths(head_dim):
    q = torch.randn(2, 1, 24, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 512, 4, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    lengths = torch.tensor([128, 512], device="cuda", dtype=torch.int32)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        ck.flash_attention_decode(q, k, v, lengths)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = ck.flash_attention_decode(q, k, v, lengths)
    for current_lengths in [[17, 333], [511, 1], [129, 257]]:
        lengths.copy_(torch.tensor(current_lengths, device="cuda", dtype=torch.int32))
        graph.replay()
        torch.testing.assert_close(actual, _reference(q, k, v, lengths), atol=2e-3, rtol=1e-2)


@requires_flash_decode
@pytest.mark.parametrize(
    ("q_dim", "k_dim", "v_dim"),
    [(64, 64, 64), (192, 192, 192), (256, 128, 128), (128, 256, 256), (256, 256, 128)],
)
def test_flash_attention_decode_rejects_invalid_dimensions(q_dim, k_dim, v_dim):
    q = torch.randn(1, 1, 24, q_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 257, 4, k_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 257, 4, v_dim, device="cuda", dtype=torch.bfloat16)
    lengths = torch.tensor([257], device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError, match=r"dimensions|k/v shape mismatch"):
        ck.flash_attention_decode(q, k, v, lengths)


requires_gqa_decode = pytest.mark.skipif(
    not ck.flash_attention_decode_gqa_is_available(),
    reason="requires the head_dim-256 GQA decode kernel",
)


@requires_gqa_decode
def test_flash_attention_decode_tree_merge_matches_full_softmax():
    # A verify tree's rows read the committed prefix through one non-causal decode pass and
    # fold their own ancestors in afterwards. Row r must then equal a plain softmax over the
    # prefix plus exactly the slots on its root-to-r path.
    torch.manual_seed(0)
    batch, kv_heads, groups, head_dim, capacity, prefix = 2, 2, 4, 256, 96, 37
    heads = kv_heads * groups
    parent = [-1, 0, 1, 2, 0, 1, 4, 4]
    rows = len(parent)
    key = torch.randn(batch, kv_heads, capacity, head_dim, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    q = torch.randn(batch, heads, rows, head_dim, device="cuda", dtype=torch.bfloat16)
    xk = torch.randn(batch, kv_heads, rows, head_dim, device="cuda", dtype=torch.bfloat16)
    xv = torch.randn_like(xk)
    key[:, :, prefix:prefix + rows] = xk
    value[:, :, prefix:prefix + rows] = xv

    def path(r):
        out = []
        while r >= 0:
            out.append(r)
            r = parent[r]
        return out[::-1]

    mask = torch.tensor([sum(1 << t for t in path(r)) for r in range(rows)], dtype=torch.int32, device="cuda")
    lengths = torch.full((batch,), prefix, device="cuda", dtype=torch.int32)
    out, lse = ck.flash_attention_decode_gqa(q, key, value, lengths, return_lse=True, causal=False)
    merged = torch.empty_like(out)
    ck.flash_attention_decode_tree_merge(out, lse, q, xk, xv, mask, merged)

    kf = key.repeat_interleave(groups, dim=1).float()
    vf = value.repeat_interleave(groups, dim=1).float()
    expected = torch.empty(batch, rows, heads * head_dim, device="cuda", dtype=torch.float32)
    for r in range(rows):
        cols = list(range(prefix)) + [prefix + t for t in path(r)]
        scores = (q[:, :, r].float().unsqueeze(2) * kf[:, :, cols]).sum(-1) * head_dim ** -0.5
        probs = scores.softmax(dim=-1)
        expected[:, r] = (probs.unsqueeze(-1) * vf[:, :, cols]).sum(2).reshape(batch, heads * head_dim)
    torch.testing.assert_close(merged.float(), expected, atol=3e-3, rtol=1e-2)


@requires_flash_decode
@pytest.mark.parametrize("num_splits", [2, 5, 32])
def test_flash_attention_decode_fused_combine_matches_two_launch(monkeypatch, num_splits):
    """The in-kernel split fold (last CTA per row) must reproduce the separate
    combine launch bit for bit: same warp-width reduction of the split maxima and
    sums, same split order in the weighted accumulation."""
    monkeypatch.setattr(flash_attention_module, "_num_splits", lambda *_, **__: num_splits)
    torch.manual_seed(0)
    q = torch.randn(3, 1, 8, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(3, 4097, 2, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    lengths = torch.tensor([1, 73, 4097], device="cuda", dtype=torch.int32)
    monkeypatch.setattr(flash_attention_module, "fused_combine", False)
    two_launch = ck.flash_attention_decode(q, k, v, lengths)
    monkeypatch.setattr(flash_attention_module, "fused_combine", True)
    fused = [ck.flash_attention_decode(q, k, v, lengths) for _ in range(3)]  # counters must rearm between calls
    for out in fused:
        assert torch.equal(out, two_launch)
    torch.testing.assert_close(two_launch, _reference(q, k, v, lengths), atol=2e-3, rtol=1e-2)


@requires_gqa_decode
@pytest.mark.parametrize("num_splits", [2, 5, 32])
@pytest.mark.parametrize("query_length", [1, 4])
def test_flash_attention_decode_gqa_fused_combine_matches_two_launch(monkeypatch, num_splits, query_length):
    monkeypatch.setattr(flash_attention_module, "_num_splits", lambda *_, **__: num_splits)
    torch.manual_seed(0)
    q = torch.randn(3, 8, query_length, 256, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(3, 2, 4097, 256, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    lengths = torch.tensor([query_length, 73, 4097], device="cuda", dtype=torch.int32)
    monkeypatch.setattr(flash_attention_module, "fused_combine", False)
    out_ref, lse_ref = ck.flash_attention_decode_gqa(q, k, v, lengths, return_lse=True)
    monkeypatch.setattr(flash_attention_module, "fused_combine", True)
    for _ in range(3):
        out, lse = ck.flash_attention_decode_gqa(q, k, v, lengths, return_lse=True)
        assert torch.equal(out, out_ref)
        assert torch.equal(lse, lse_ref)
