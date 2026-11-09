# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bitwise recipe checks for the deterministic BF16 ConvRot specialization."""

import pytest
import torch

from comfy_kitchen.backends import cuda
from .conftest import requires_cuda_backend


@requires_cuda_backend
@pytest.mark.parametrize("rows", [0, 1, 8, 9, 17, 4352])
@pytest.mark.parametrize("offset", [0, 1])
def test_convrot_bf16_exact_rows_and_alignment(rows, offset):
    # The 64-lane implementation independently evaluates the same ordered FP32
    # recipe. Offset exercises the unaligned fallback; rows 8/9 cross dispatch.
    torch.manual_seed(8713)
    storage = torch.randn(rows * 4096 + offset, device="cuda", dtype=torch.bfloat16)
    x = storage[offset:].view(rows, 4096)
    q, scale = cuda.quantize_int8_rowwise_convrot(x, 256)
    expected_q, expected_scale = cuda.quantize_int8_rowwise_convrot64(x, 256)
    assert torch.equal(q, expected_q)
    assert torch.equal(scale.view(torch.int32), expected_scale.view(torch.int32))


@requires_cuda_backend
@pytest.mark.parametrize(
    "pattern",
    ["zero", "signed_zero", "constant", "outlier", "tiny", "huge", "maxfinite", "signed_max"],
)
def test_convrot_bf16_exact_extremes(pattern):
    torch.manual_seed(391)
    x = torch.randn(17, 4096, device="cuda", dtype=torch.bfloat16)
    if pattern == "zero":
        x.zero_()
    elif pattern == "signed_zero":
        x.zero_()
        x[:, 1::2] = -0.0
    elif pattern == "constant":
        x.fill_(0.25)
    elif pattern == "outlier":
        x[:, ::257] = 32
    elif pattern == "tiny":
        x.mul_(2.0**-110)
    elif pattern == "huge":
        x.mul_(2.0**100)
    elif pattern == "maxfinite":
        x.fill_(torch.finfo(x.dtype).max)
    else:
        x = x.sign() * torch.finfo(x.dtype).max
    q, scale = cuda.quantize_int8_rowwise_convrot(x, 256)
    expected_q, expected_scale = cuda.quantize_int8_rowwise_convrot64(x, 256)
    assert torch.equal(q, expected_q)
    assert torch.equal(scale.view(torch.int32), expected_scale.view(torch.int32))


@requires_cuda_backend
def test_convrot_bf16_graph_replay_changed_input():
    x = torch.randn(17, 4096, device="cuda", dtype=torch.bfloat16)
    # Warm the backend on a side stream as required by CUDA graph capture.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cuda.quantize_int8_rowwise_convrot(x, 256)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q, scale = cuda.quantize_int8_rowwise_convrot(x, 256)
        consumer_q, consumer_scale = q.clone(), scale.clone()
    x.neg_()
    graph.replay()
    expected_q, expected_scale = cuda.quantize_int8_rowwise_convrot64(x, 256)
    assert torch.equal(consumer_q, expected_q)
    assert torch.equal(consumer_scale.view(torch.int32), expected_scale.view(torch.int32))
