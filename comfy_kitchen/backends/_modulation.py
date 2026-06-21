# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for AdaLN-style modulation across compute backends.

Pure-torch, dependency-light (no triton/cuda imports) so both the CUDA and
Triton backends can reuse the exact same broadcast logic without drifting.
"""

import torch


def adaln_prep_modulation(t: torch.Tensor, x: torch.Tensor, n: int, d: int):
    """Flatten a scale/shift tensor to (R, D) distinct rows + a broadcast group.

    Returns (flat, group) where output row r reads vector ``r // group`` and
    ``group == N // R``. In AdaLN, scale/shift are broadcast over inner dims
    (tokens, or spatial H/W): e.g. x (B,1,H,W,D) with scale (B,1,1,1,D). Whenever
    the broadcast collapses to contiguous blocks of output rows — that is, every
    "kept" (non-broadcast) leading dim is outer to every broadcast dim — the
    distinct rows reshape to (R, D) and row r maps to r // (N // R). That avoids
    the expand+copy and the N*D of redundant read traffic. Broadcast patterns
    that don't form contiguous blocks fall back to materializing (group == 1).
    """
    if t.shape[-1] == d:
        lead_x = x.shape[:-1]
        lead_t = t.shape[:-1]
        # Left-pad scale's leading dims with 1s to align with x (PyTorch broadcast).
        pad = len(lead_x) - len(lead_t)
        if pad >= 0:
            lead_t = (1,) * pad + tuple(lead_t)
            groupable = True
            seen_broadcast = False
            for s_i, x_i in zip(lead_t, lead_x, strict=True):
                if s_i == x_i:
                    # A real "kept" dim must not sit inside a broadcast dim, or
                    # its rows wouldn't repeat in contiguous blocks.
                    if x_i != 1 and seen_broadcast:
                        groupable = False
                        break
                elif s_i == 1:
                    if x_i != 1:
                        seen_broadcast = True
                else:
                    groupable = False  # not a valid broadcast
                    break

            if groupable:
                rows = t.numel() // d
                if rows > 0 and n % rows == 0:
                    flat = t.reshape(rows, d)
                    if not flat.is_contiguous():
                        flat = flat.contiguous()
                    return flat, n // rows

    # General broadcast: materialize. Correct for any shape, just not the fast path.
    flat = t.expand_as(x).reshape(n, d)
    return (flat if flat.is_contiguous() else flat.contiguous()), 1
