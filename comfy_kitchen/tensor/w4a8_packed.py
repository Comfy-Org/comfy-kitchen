# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""W3A8/W4A8/W6A8 packed-weight codec shared by the layout and its backends."""

from __future__ import annotations

import torch

# Lloyd-Max-optimal 16 levels for a group-normalized Gaussian. ConvRot makes every layer's
# rotated groups Gaussian, so this one table matches a per-tensor fit and skips the k-means.
_FIXED_LUT = (
    -0.980602, -0.794529, -0.638165, -0.500986, -0.377321, -0.263187, -0.155210, -0.050720,
    0.052541, 0.156985, 0.265284, 0.379533, 0.502636, 0.638953, 0.794876, 0.980671,
)
# The same fit at 8 levels (symmetrized) for 3-bit storage.
_FIXED_LUT_3 = (
    -0.923229, -0.593622, -0.335697, -0.108977, 0.108977, 0.335697, 0.593622, 0.923229,
)


def default_w4a8_codebook(bits: int = 4) -> torch.Tensor:
    """The frozen Lloyd-Max codebook (the CUDA streamed decode kernel's built-in LUT)."""
    return torch.tensor(_FIXED_LUT_3 if bits == 3 else _FIXED_LUT, dtype=torch.float32)


def _pack_codes(unsigned: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack unsigned int32 codes [N, K] into the int8 storage contract. 4 and 6 bits: a
    K/2-byte nibble plane (even col = low nibble), followed at 6 bits by a K/4-byte plane of
    each code's top 2 bits (col c: byte c//4, bit 2*(c%4)). 3 bits: a K/4-byte plane of each
    code's low 2 bits (same col->bit map) then a K/8-byte plane of bit 2 (col c: byte c//8,
    bit c%8)."""
    if bits == 3:
        lo = unsigned[:, 0::4] & 0x3
        for j in range(1, 4):
            lo |= (unsigned[:, j::4] & 0x3) << (2 * j)
        hi = (unsigned[:, 0::8] >> 2) & 0x1
        for j in range(1, 8):
            hi |= ((unsigned[:, j::8] >> 2) & 0x1) << j
        return torch.cat([lo.to(torch.int8), hi.to(torch.int8)], dim=1).contiguous()
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
    if bits == 3:
        shifts2 = torch.tensor([0, 2, 4, 6], device=qdata.device, dtype=torch.int32)
        codes = ((packed[:, : k // 4].unsqueeze(-1) >> shifts2) & 0x3).reshape(n, k)
        shifts1 = torch.arange(8, device=qdata.device, dtype=torch.int32)
        codes |= ((packed[:, k // 4 :].unsqueeze(-1) >> shifts1) & 0x1).reshape(n, k) << 2
        return codes
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


def w4a8_mma_record_bytes(bits: int) -> int:
    """Bytes per 16-output x 32-K record: 512 codes plus 32 fp8 scale bytes (224 / 288 / 416)."""
    return 512 * bits // 8 + 32


def _mma_bits(packed_numel: int, n: int, k: int) -> int:
    for bits in (3, 4, 6):
        if packed_numel == n * k * w4a8_mma_record_bytes(bits) // 512:
            return bits
    raise ValueError(
        f"MMA packed weight must be a contiguous int8 [{n * k * 7 // 16}], "
        f"[{n * k * 9 // 16}] or [{n * k * 13 // 16}] tensor"
    )


def _fragment_major_inverse(fragments: torch.Tensor, cols: int) -> torch.Tensor:
    tiles, k_rows = fragments.shape[:2]
    half = cols // 2
    codes = torch.empty((tiles, k_rows, 16, cols), dtype=fragments.dtype, device=fragments.device)
    codes[:, :, :8, :half] = fragments[:, :, :, :, 0, :].reshape(tiles, k_rows, 8, half)
    codes[:, :, 8:, :half] = fragments[:, :, :, :, 1, :].reshape(tiles, k_rows, 8, half)
    codes[:, :, :8, half:] = fragments[:, :, :, :, 2, :].reshape(tiles, k_rows, 8, half)
    codes[:, :, 8:, half:] = fragments[:, :, :, :, 3, :].reshape(tiles, k_rows, 8, half)
    return codes


def unpack_w4a8_mma_weight(
    packed: torch.Tensor,
    n: int,
    k: int,
    scale_dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover canonical codes/scales for explicit dequantization and eager fallback."""
    if packed.dim() != 1 or packed.dtype != torch.int8 or not packed.is_contiguous():
        raise ValueError("MMA packed weight must be a contiguous 1D int8 tensor")
    if n % 16 != 0 or k % 256 != 0:
        raise ValueError("MMA packed weight requires N % 16 == 0 and K % 256 == 0")
    bits = _mma_bits(packed.numel(), n, k)
    record_bytes = w4a8_mma_record_bytes(bits)
    tiles, k_rows = n // 16, k // 32
    records = packed.view(torch.uint8).view(tiles, k_rows, record_bytes)
    code_bytes = record_bytes - 32
    if bits == 4:
        fragments = records[:, :, :code_bytes].view(tiles, k_rows, 8, 4, 4, 2)
        qdata = _fragment_major_inverse(fragments, 16).permute(0, 2, 1, 3).reshape(n, k // 2)
    elif bits == 3:
        # lane's 16 codes form one 48-bit value (code (j, c) at bit 12j + 3c): bytes 0-3 at
        # lane*4, bytes 4-5 at 128 + lane*2
        lane_bytes = torch.cat(
            (records[:, :, :128].reshape(tiles, k_rows, 32, 4), records[:, :, 128:code_bytes].reshape(tiles, k_rows, 32, 2)),
            dim=3,
        ).to(torch.int64)
        words = sum(lane_bytes[..., i] << (8 * i) for i in range(6))
        shifts = torch.arange(0, 48, 3, device=packed.device, dtype=torch.int64)
        fragments = ((words.unsqueeze(-1) >> shifts) & 0x7).view(tiles, k_rows, 8, 4, 4, 4)
        codes = _fragment_major_inverse(fragments, 32).permute(0, 2, 1, 3).reshape(n, k).to(torch.int32)
        qdata = _pack_codes(codes, bits).view(torch.uint8)
    else:
        lane_bytes = torch.cat(
            (records[:, :, :256].reshape(tiles, k_rows, 32, 8), records[:, :, 256:code_bytes].reshape(tiles, k_rows, 32, 4)),
            dim=3,
        ).to(torch.int32).view(tiles, k_rows, 8, 4, 4, 3)
        words = lane_bytes[..., 0] | lane_bytes[..., 1] << 8 | lane_bytes[..., 2] << 16
        shifts = torch.tensor([0, 6, 12, 18], device=packed.device, dtype=torch.int32)
        fragments = (words.unsqueeze(-1) >> shifts) & 0x3F
        codes = _fragment_major_inverse(fragments, 32).permute(0, 2, 1, 3).reshape(n, k)
        qdata = _pack_codes(codes, bits).view(torch.uint8)

    scale_fragments = records[:, :, code_bytes:].view(tiles, k_rows, 8, 4)
    scale_tiles = torch.empty((tiles, k_rows, 16, 2), dtype=torch.uint8, device=packed.device)
    scale_tiles[:, :, :8, 0] = scale_fragments[:, :, :, 0]
    scale_tiles[:, :, 8:, 0] = scale_fragments[:, :, :, 1]
    scale_tiles[:, :, :8, 1] = scale_fragments[:, :, :, 2]
    scale_tiles[:, :, 8:, 1] = scale_fragments[:, :, :, 3]
    scales = scale_tiles.permute(0, 2, 1, 3).reshape(n, k // 16)
    return qdata.view(torch.int8), scales.view(scale_dtype)
