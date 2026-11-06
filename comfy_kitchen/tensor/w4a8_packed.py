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


# Its 8-level (3-bit) twin, fit the same way; equal to a per-tensor fit on rotated weights.
_FIXED_LUT8 = (-0.923975, -0.594572, -0.336347, -0.109145, 0.109145, 0.336347, 0.594572, 0.923975)


def default_w4a8_codebook(levels: int = 16) -> torch.Tensor:
    """The frozen Lloyd-Max codebook (the CUDA streamed decode kernel's built-in LUT):
    16 levels for 4-bit codes, 8 for 3-bit."""
    if levels not in (8, 16):
        raise ValueError(f"codebook levels must be 8 or 16, got {levels}")
    return torch.tensor(_FIXED_LUT if levels == 16 else _FIXED_LUT8, dtype=torch.float32)


def _plane(unsigned: torch.Tensor, shift: int, width: int) -> torch.Tensor:
    """A bit plane of ``width`` bits per code taken from ``unsigned >> shift``, 8/width codes
    per byte (col c: byte c // (8/width), bit width*(c % (8/width)))."""
    per_byte = 8 // width
    mask = (1 << width) - 1
    plane = (unsigned[:, 0::per_byte] >> shift) & mask  # slice first: each term touches K/per_byte
    for j in range(1, per_byte):
        plane |= ((unsigned[:, j::per_byte] >> shift) & mask) << (width * j)
    return plane.to(torch.int8)


def _unplane(packed: torch.Tensor, n: int, k: int, width: int) -> torch.Tensor:
    """Inverse of _plane: int32 bytes [N, K*width/8] -> int32 field [N, K]."""
    shifts = torch.arange(0, 8, width, device=packed.device, dtype=torch.int32)
    return ((packed.unsqueeze(-1) >> shifts) & ((1 << width) - 1)).reshape(n, k)


def _pack_codes(unsigned: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack unsigned int32 codes [N, K] into the int8 storage contract of bit planes:
    4: nibble plane (K/2 bytes, even col = low nibble); 6: nibble plane + 2-bit plane of the
    top 2 bits (K/4 bytes); 5: nibble plane + 1-bit plane of the top bit (K/8 bytes);
    3: 2-bit plane of the low 2 bits (K/4 bytes) + 1-bit plane of the top bit (K/8 bytes)."""
    if bits == 3:
        return torch.cat([_plane(unsigned, 0, 2), _plane(unsigned, 2, 1)], dim=1).contiguous()
    low = _plane(unsigned, 0, 4)
    if bits == 4:
        return low.contiguous()
    if bits == 5:
        return torch.cat([low, _plane(unsigned, 4, 1)], dim=1).contiguous()
    if bits == 6:
        return torch.cat([low, _plane(unsigned, 4, 2)], dim=1).contiguous()
    raise ValueError(f"bits must be 3, 4, 5 or 6, got {bits}")


def _unpack_codes(qdata: torch.Tensor, k: int, bits: int) -> torch.Tensor:
    """Inverse of _pack_codes: int8 storage -> unsigned int32 codes [N, K]."""
    n = qdata.shape[0]
    packed = qdata.view(torch.uint8).to(torch.int32)  # uint8 view: no sign-extension mask
    if bits == 3:
        return _unplane(packed[:, : k // 4], n, k, 2) | (_unplane(packed[:, k // 4 :], n, k, 1) << 2)
    codes = _unplane(packed[:, : k // 2], n, k, 4)
    if bits == 5:
        codes |= _unplane(packed[:, k // 2 :], n, k, 1) << 4
    elif bits == 6:
        codes |= _unplane(packed[:, k // 2 :], n, k, 2) << 4
    return codes


def w4a8_mma_record_bytes(bits: int) -> int:
    """Bytes per 16-output x 32-K record: 512 codes plus 32 fp8 scale bytes (224 / 288 / 352 / 416)."""
    return 512 * bits // 8 + 32


def _mma_bits(packed_numel: int, n: int, k: int) -> int:
    for bits in (3, 4, 5, 6):
        if packed_numel == n * k * w4a8_mma_record_bytes(bits) // 512:
            return bits
    sizes = ", ".join(str(n * k * w4a8_mma_record_bytes(b) // 512) for b in (3, 4, 5, 6))
    raise ValueError(f"MMA packed weight must be a contiguous int8 tensor of {sizes} bytes (3/4/5/6-bit)")


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
    else:
        # lane's 16 codes at bit q*bits of a 6/10/12-byte string (LaneCodes in w4a8_gemm.cu):
        # plane A has 4 (3-bit) or 8 bytes per lane, plane B 2 or 4 (6-bit) more
        a_bytes, b_bytes = (4 if bits == 3 else 8), (4 if bits == 6 else 2)
        lane_bytes = torch.cat(
            (records[:, :, :32 * a_bytes].reshape(tiles, k_rows, 32, a_bytes),
             records[:, :, 32 * a_bytes:code_bytes].reshape(tiles, k_rows, 32, b_bytes),
             torch.zeros((tiles, k_rows, 32, 1), dtype=torch.uint8, device=packed.device)),
            dim=3,
        ).to(torch.int32)
        q = torch.arange(16, device=packed.device) * bits
        lo, hi = lane_bytes[..., q // 8], lane_bytes[..., q // 8 + 1]
        fragments = (((lo | hi << 8) >> (q % 8)) & ((1 << bits) - 1)).view(tiles, k_rows, 8, 4, 4, 4)
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
