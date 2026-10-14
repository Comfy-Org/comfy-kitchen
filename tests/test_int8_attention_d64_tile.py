# SPDX-License-Identifier: Apache-2.0
"""The SM120 D64 tile must preserve the existing per-thread Q scale groups."""

import pytest
import torch

import comfy_kitchen as ck

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0)
    or not ck.int8_attention_is_available(),
    reason="requires the SM120 CUDA attention kernel",
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch,length", [(1, 1024), (2, 1025), (4, 1797), (8, 1800), (4, 2048)])
def test_d64_query_tile_matches_split_heads(dtype, batch, length):
    torch.manual_seed(41)
    # Interleaved QKV and noncontiguous BHSD views match the VAE producer.
    qkv = torch.randn(batch, length, 32, 3, 64, device="cuda", dtype=dtype)
    q, k, v = [qkv[:, :, :, i].transpose(1, 2) for i in range(3)]
    actual = ck.int8_attention(q, k, v)
    # Sixteen heads use the original 128-query tile. Heads are independent;
    # splitting them preserves every quantization group and provides an oracle
    # without exposing a testing-only kernel selector in the public API.
    expected = torch.cat([
        ck.int8_attention(q[:, :16], k[:, :16], v[:, :16]),
        ck.int8_attention(q[:, 16:], k[:, 16:], v[:, 16:]),
    ], dim=1)
    assert torch.equal(actual.view(torch.uint8), expected.contiguous().view(torch.uint8))
