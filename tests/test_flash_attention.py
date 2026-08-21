import pytest
import torch

import comfy_kitchen as ck
import comfy_kitchen.flash_attention as flash_attention_module

requires_flash_decode = pytest.mark.skipif(
    not ck.flash_attention_decode_is_available(),
    reason="requires the CUDA extension on SM80 or newer",
)

# The HIP binding takes plain ndarrays rather than device-typed ones, so it has
# to reject a host operand and an off-boundary base itself.
requires_hip_flash_decode = pytest.mark.skipif(
    not ck.flash_attention_decode_is_available() or not getattr(torch.version, "hip", None),
    reason="requires the HIP flash decode kernel",
)


def _decode_operands(batch=2, capacity=256, kv_heads=2, groups=4):
    heads = kv_heads * groups
    q = torch.randn(batch, 1, heads, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, capacity, kv_heads, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    lengths = torch.full((batch,), capacity, device="cuda", dtype=torch.int32)
    return q, k, v, lengths


@requires_hip_flash_decode
def test_flash_attention_decode_rejects_host_lengths():
    q, k, v, lengths = _decode_operands()
    with pytest.raises(RuntimeError, match="must be ROCm device memory"):
        ck.flash_attention_decode(q, k, v, lengths.cpu())
    # The rejection must leave the context usable.
    assert torch.isfinite(ck.flash_attention_decode(q, k, v, lengths).float()).all()


@requires_hip_flash_decode
def test_flash_attention_decode_rejects_pinned_host_memory():
    # Pinned host memory reports kDLCUDAHost rather than kDLCPU, so a check for
    # "not the CPU device" would wave these host pointers through. The launch
    # goes through _C directly because the Python wrapper's stream lookup
    # rejects a host tensor before the binding sees it.
    from comfy_kitchen.backends import hip as hip_backend

    batch, capacity, kv_heads, groups = 2, 128, 2, 4
    pinned = [
        torch.randn(batch * groups, kv_heads, 128, dtype=torch.bfloat16).pin_memory(),
        torch.randn(batch, capacity, kv_heads, 128, dtype=torch.bfloat16).pin_memory(),
        torch.randn(batch, capacity, kv_heads, 128, dtype=torch.bfloat16).pin_memory(),
        torch.full((batch,), capacity, dtype=torch.int32).pin_memory(),
        torch.empty(batch * groups, kv_heads, 128, dtype=torch.bfloat16).pin_memory(),
        torch.empty(batch * kv_heads * groups, dtype=torch.float32).pin_memory(),
    ]
    assert pinned[0].__dlpack_device__()[0] != 1
    empty = pinned[-1][:0]
    args = [hip_backend._dl(t) for t in (*pinned, empty, empty)]
    with pytest.raises(RuntimeError, match="must be ROCm device memory"):
        hip_backend._C.flash_attention_decode(*args, 1, 0)


@requires_hip_flash_decode
def test_flash_attention_decode_rejects_strided_lengths():
    # Read linearly off the base pointer, so a strided view of the right size
    # would silently be taken as packed.
    q, k, v, lengths = _decode_operands()
    strided = torch.stack([lengths, torch.zeros_like(lengths)], dim=1).flatten()[::2]
    assert not strided.is_contiguous() and strided.numel() == lengths.numel()
    with pytest.raises(RuntimeError, match="must be contiguous"):
        ck.flash_attention_decode(q, k, v, strided)


@requires_hip_flash_decode
def test_flash_attention_decode_rejects_misaligned_operand():
    q, k, v, lengths = _decode_operands()
    elements = k.numel()
    storage = torch.randn(elements + 8, device="cuda", dtype=torch.bfloat16)
    misaligned = storage[1 : 1 + elements].view_as(k)
    assert misaligned.is_contiguous() and misaligned.data_ptr() % 8
    with pytest.raises(RuntimeError, match="8-byte aligned"):
        ck.flash_attention_decode(q, misaligned, v, lengths)



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
def test_flash_attention_decode():
    torch.manual_seed(0)
    q = torch.randn(3, 1, 8, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(3, 257, 2, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    lengths = torch.tensor([1, 73, 257], device="cuda", dtype=torch.int32)
    actual = ck.flash_attention_decode(q, k, v, lengths)
    torch.testing.assert_close(actual, _reference(q, k, v, lengths), atol=2e-3, rtol=1e-2)


@requires_flash_decode
@pytest.mark.parametrize("num_splits", [1, 2, 5, 32])
def test_flash_attention_decode_split_counts(monkeypatch, num_splits):
    # num_splits comes from a heuristic over multi_processor_count, so on a wide
    # enough GPU it returns 1 and the split accumulators, the combine pass and
    # its row decomposition never run. Pin it instead of hoping.
    monkeypatch.setattr(flash_attention_module, "_num_splits", lambda *_, **__: num_splits)
    torch.manual_seed(0)
    q = torch.randn(3, 1, 8, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(3, 257, 2, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    lengths = torch.tensor([1, 73, 257], device="cuda", dtype=torch.int32)
    actual = ck.flash_attention_decode(q, k, v, lengths)
    torch.testing.assert_close(actual, _reference(q, k, v, lengths), atol=2e-3, rtol=1e-2)


@requires_flash_decode
def test_flash_attention_decode_cuda_graph_dynamic_lengths():
    q = torch.randn(2, 1, 8, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 512, 2, 128, device="cuda", dtype=torch.bfloat16)
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
    lengths.copy_(torch.tensor([17, 333], device="cuda", dtype=torch.int32))
    graph.replay()
    torch.testing.assert_close(actual, _reference(q, k, v, lengths), atol=2e-3, rtol=1e-2)
