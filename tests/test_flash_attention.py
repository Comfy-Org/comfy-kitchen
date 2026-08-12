import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.backends import cuda as cuda_backend


requires_flash_decode = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not cuda_backend._EXT_AVAILABLE
    or torch.cuda.get_device_capability() < (8, 0),
    reason="requires the CUDA extension on SM80 or newer",
)


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
