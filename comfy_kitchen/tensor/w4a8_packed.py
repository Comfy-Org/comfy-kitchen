# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""W4A8/W6A8 packed-weight codec shared by the layout and its backends."""

from __future__ import annotations

import torch

# Lloyd-Max-optimal 16 levels for a group-normalized Gaussian. ConvRot makes every layer's
# rotated groups Gaussian, so this one table matches a per-tensor fit and skips the k-means.
_FIXED_LUT = (
    -0.980602, -0.794529, -0.638165, -0.500986, -0.377321, -0.263187, -0.155210, -0.050720,
    0.052541, 0.156985, 0.265284, 0.379533, 0.502636, 0.638953, 0.794876, 0.980671,
)


def _pack_codes(unsigned: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack unsigned int32 codes [N, K] into the int8 storage contract: a K/2-byte nibble
    plane (even col = low nibble), followed at 6 bits by a K/4-byte plane of each code's
    top 2 bits (col c: byte c//4, bit 2*(c%4))."""
    low = ((unsigned[:, 0::2] & 0xF) | ((unsigned[:, 1::2] & 0xF) << 4)).to(torch.int8)
    if bits == 4:
        return low.contiguous()
    hi = (unsigned[:, 0::4] >> 4) & 0x3  # slice first: each term touches K/4, not K
    for j in range(1, 4):
        hi |= ((unsigned[:, j::4] >> 4) & 0x3) << (2 * j)
    return torch.cat([low, hi.to(torch.int8)], dim=1).contiguous()


def _unpack_codes(qdata: torch.Tensor, k: int, bits: int) -> torch.Tensor:
    """Inverse of _pack_codes: int8 storage -> unsigned int32 codes [N, K]."""
    n = qdata.shape[0]
    packed = qdata.view(torch.uint8).to(torch.int32)  # uint8 view: no sign-extension mask
    low = packed[:, : k // 2]
    codes = torch.empty(n, k, dtype=torch.int32, device=qdata.device)
    codes[:, 0::2] = low & 0xF
    codes[:, 1::2] = (low >> 4) & 0xF
    if bits == 6:
        # high-plane byte b holds cols 4b..4b+3 at bits 0,2,4,6: one fused expansion
        shifts = torch.tensor([0, 2, 4, 6], device=qdata.device, dtype=torch.int32)
        hi = (packed[:, k // 2 :].unsqueeze(-1) >> shifts) & 0x3
        codes |= hi.reshape(n, k) << 4
    return codes
