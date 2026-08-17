# SPDX-License-Identifier: Apache-2.0

import importlib
import math

import pytest
import torch

import comfy_kitchen as ck

sol_attention_module = importlib.import_module("comfy_kitchen.sol_attention")


requires_sol_attention = pytest.mark.skipif(
    not ck.sol_attention_is_available(),
    reason="requires the CUDA extension on SM80 or newer",
)


def _qkv(batch=1, heads=1, sequence=256):
    # Transposing a natural BNHD input also exercises the public API's friendly
    # handling of the strided BHND views produced by model code.
    shape = (batch, sequence, heads, 128)
    q = torch.randn(*shape, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    k = torch.randn(*shape, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    v = torch.randn(*shape, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    return q, k, v


def _nrmse(actual, expected):
    error = (actual.float() - expected.float()).square().mean().sqrt()
    magnitude = expected.float().square().mean().sqrt()
    return (error / magnitude).item()


def _sol_reference(q, k, v, *, tau=1.0, scale=None):
    """Compact mathematical reference for the exact/summary block routing."""
    batch, heads, sequence, dim = q.shape
    block = 64
    blocks = sequence // block
    scale = dim**-0.5 if scale is None else float(scale)

    q_blocks = q.float().reshape(batch, heads, blocks, block, dim)
    k_blocks = k.float().reshape(batch, heads, blocks, block, dim)
    v_blocks = v.float().reshape(batch, heads, blocks, block, dim)
    kc = k_blocks.mean(dim=3).to(torch.bfloat16).float()
    vc = v_blocks.sum(dim=3).to(torch.bfloat16).float()
    key_mean = kc.mean(dim=2)
    key_variance = (kc.square().mean(dim=2) - key_mean.square()).clamp_min(0)
    q_mean = q_blocks.mean(dim=3)
    threshold = scale * (q_mean * key_mean[:, :, None]).sum(dim=-1)
    threshold += tau * torch.sqrt(
        scale * scale * (q_mean.square() * key_variance[:, :, None]).sum(dim=-1) + 1e-6
    )

    output = torch.empty_like(q_blocks)
    block_ids = torch.arange(blocks, device=q.device)
    log_block = math.log(block)
    for batch_id in range(batch):
        for head_id in range(heads):
            for query_block in range(blocks):
                query = q_blocks[batch_id, head_id, query_block]
                proxy = query @ kc[batch_id, head_id].mT * scale
                exact = proxy.mean(dim=0) > threshold[batch_id, head_id, query_block]
                exact |= (block_ids - query_block).abs() <= 1

                logits = []
                values = []
                for key_block in range(blocks):
                    if exact[key_block]:
                        logits.append(query @ k_blocks[batch_id, head_id, key_block].mT * scale)
                        values.append(v_blocks[batch_id, head_id, key_block])
                    else:
                        # One pseudo-token is equivalent to 64 identical centroid
                        # logits in the denominator and the block-summed V numerator.
                        logits.append(proxy[:, key_block : key_block + 1] + log_block)
                        values.append(vc[batch_id, head_id, key_block : key_block + 1] / block)
                probabilities = torch.cat(logits, dim=1).softmax(dim=1)
                output[batch_id, head_id, query_block] = probabilities @ torch.cat(values, dim=0)
    return output.reshape_as(q).to(q.dtype)


def test_sol_attention_availability_is_bool():
    assert isinstance(ck.sol_attention_is_available(), bool)


@pytest.mark.parametrize(
    ("capability", "has_kernel", "expected"),
    [
        ((7, 5), True, False),
        ((8, 0), True, True),
        ((8, 9), True, True),
        ((9, 0), False, False),
    ],
)
def test_sol_attention_availability_checks_capability_and_kernel(
    monkeypatch, capability, has_kernel, expected
):
    if getattr(torch.version, "hip", None):
        pytest.skip("SOL attention is CUDA-only")

    class Extension:
        pass

    extension = Extension()
    if has_kernel:
        extension.sol_attention_bf16 = object()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: capability)
    monkeypatch.setattr(sol_attention_module._cuda_backend, "_EXT_AVAILABLE", True)
    monkeypatch.setattr(sol_attention_module._cuda_backend, "_C", extension)
    assert sol_attention_module.is_available() is expected


def test_sol_attention_rejects_cpu_inputs():
    q = torch.empty(1, 1, 64, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="same CUDA device"):
        ck.sol_attention(q, q, q)


@requires_sol_attention
def test_sol_attention_all_exact_matches_sdpa_with_custom_scale():
    torch.manual_seed(0)
    q, k, v = _qkv(batch=2, heads=2, sequence=128)
    actual = ck.sol_attention(q, k, v, tau=-1e6, scale=0.07)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=0.07)
    torch.testing.assert_close(actual, expected, atol=3e-3, rtol=1e-2)


@requires_sol_attention
def test_sol_attention_matches_block_sparse_reference():
    torch.manual_seed(1)
    q, k, v = _qkv(sequence=256)
    actual = ck.sol_attention(q, k, v)
    expected = _sol_reference(q, k, v)
    assert actual.shape == q.shape
    assert actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    assert _nrmse(actual, expected) < 0.005


@requires_sol_attention
@pytest.mark.parametrize(
    ("shape", "message"),
    [
        ((1, 1, 65, 128), "divisible by 64"),
        ((1, 1, 64, 64), "head_dim 128"),
    ],
)
def test_sol_attention_validates_kernel_shape_contract(shape, message):
    q = torch.empty(*shape, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=message):
        ck.sol_attention(q, q, q)


def test_sol_attention_rejects_nonfinite_parameters():
    q, k, v = _qkv(sequence=64)
    with pytest.raises(ValueError, match="tau must be finite"):
        ck.sol_attention(q, k, v, tau=math.inf)
    with pytest.raises(ValueError, match="scale must be finite"):
        ck.sol_attention(q, k, v, scale=math.nan)


def test_sol_attention_cuda_graph_capture():
    q, k, v = _qkv(sequence=256)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        ck.sol_attention(q, k, v)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = ck.sol_attention(q, k, v)
    graph.replay()
    assert torch.isfinite(output).all()
