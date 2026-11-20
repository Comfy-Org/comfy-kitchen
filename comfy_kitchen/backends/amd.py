# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which AMD backend serves a device, for the callers that bypass the registry.

rdna1 serves gfx1010, built for that one target; every other device stays with the
hip backend. One process can see both kinds, so callers resolve per device. Import
only under a ROCm PyTorch build.
"""
import torch

from comfy_kitchen.registry import registry

from . import hip, rdna1


def backend_for(device: torch.device | None = None):
    """(registry name, module) of the AMD backend for ``device``, the current device
    when None. Off rdna1 this is the hip backend, registered or not, which is what
    upstream callers of the hip module expect."""
    if registry.is_available("rdna1") and rdna1.serves(device):
        return "rdna1", rdna1
    return "hip", hip


def sol_attn_chunked(qkv_chunks, t, h, rope_freqs, *args, **kwargs):
    """sol_attn_chunked on the backend serving ``rope_freqs``' device."""
    return backend_for(rope_freqs.device)[1].sol_attn_chunked(
        qkv_chunks, t, h, rope_freqs, *args, **kwargs)
