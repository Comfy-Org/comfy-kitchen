# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""W3A8/W4A8/W5A8/W6A8 packed-weight codec shared by the layout and its backends."""

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
# Nested 3/4/5-bit tables for weights whose precision can be lowered in place: an 8-level
# Lloyd-Max fit, each cell Lloyd-split into two for 16 levels and again for 32, so the
# children of code p at one width are codes 2p and 2p+1 at the next, and dropping a code's
# low bit lands on the parent level. Stored as one [56] tensor: 8 | 16 | 32 entries.
_NESTED_LUT = (
    -0.923985, -0.586828, -0.326914, -0.104694, 0.104694, 0.326914, 0.586828, 0.923985,
    -0.985933, -0.824656, -0.670487, -0.523194, -0.392372, -0.272664, -0.160595, -0.053216,
    0.053216, 0.160595, 0.272664, 0.392372, 0.523194, 0.670487, 0.824656, 0.985933,
    -0.998096, -0.934925, -0.862538, -0.783983, -0.707216, -0.632095, -0.560187, -0.490820,
    -0.425224, -0.362703, -0.302748, -0.244903, -0.188777, -0.133984, -0.080025, -0.026632,
    0.026632, 0.080025, 0.133984, 0.188777, 0.244903, 0.302748, 0.362703, 0.425224,
    0.490820, 0.560187, 0.632095, 0.707216, 0.783983, 0.862538, 0.934925, 0.998096,
)
NESTED_CODEBOOK_SIZE = 56
_NESTED_OFFSETS = {3: 0, 4: 8, 5: 24}


def default_w4a8_codebook(bits: int = 4) -> torch.Tensor:
    """The frozen Lloyd-Max codebook (the CUDA streamed decode kernel's built-in LUT)."""
    return torch.tensor(_FIXED_LUT_3 if bits == 3 else _FIXED_LUT, dtype=torch.float32)


def nested_w4a8_codebook() -> torch.Tensor:
    """The nested 3/4/5-bit table as one [56] tensor."""
    return torch.tensor(_NESTED_LUT, dtype=torch.float32)


def is_nested_codebook(codebook: torch.Tensor | None) -> bool:
    return codebook is not None and codebook.numel() == NESTED_CODEBOOK_SIZE


def w4a8_level_table(codebook: torch.Tensor, bits: int) -> torch.Tensor:
    """The [2^bits] level table the codes of a weight index: the codebook itself, or the
    width's sub-table of a nested codebook."""
    if not is_nested_codebook(codebook):
        return codebook
    if bits not in _NESTED_OFFSETS:
        raise ValueError(f"nested codebook has no {bits}-bit table")
    start = _NESTED_OFFSETS[bits]
    return codebook[start : start + (1 << bits)]


def _pack_codes(unsigned: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack unsigned int32 codes [N, K] into the int8 storage contract. 4, 5 and 6 bits: a
    K/2-byte nibble plane (even col = low nibble), followed at 6 bits by a K/4-byte plane of
    each code's top 2 bits (col c: byte c//4, bit 2*(c%4)) and at 5 bits by a K/8-byte plane
    of bit 4 (col c: byte c//8, bit c%8). 3 bits: a K/4-byte plane of each code's low 2 bits
    (same col->bit map as the 6-bit high plane) then a K/8-byte plane of bit 2."""
    if bits == 3:
        lo = unsigned[:, 0::4] & 0x3
        for j in range(1, 4):
            lo |= (unsigned[:, j::4] & 0x3) << (2 * j)
        return torch.cat([lo.to(torch.int8), _bit_plane(unsigned, 2)], dim=1).contiguous()
    low = ((unsigned[:, 0::2] & 0xF) | ((unsigned[:, 1::2] & 0xF) << 4)).to(torch.int8)
    if bits == 4:
        return low.contiguous()
    if bits == 5:
        return torch.cat([low, _bit_plane(unsigned, 4)], dim=1).contiguous()
    hi = (unsigned[:, 0::4] >> 4) & 0x3  # slice first: each term touches K/4, not K
    for j in range(1, 4):
        hi |= ((unsigned[:, j::4] >> 4) & 0x3) << (2 * j)
    return torch.cat([low, hi.to(torch.int8)], dim=1).contiguous()


def _bit_plane(unsigned: torch.Tensor, bit: int) -> torch.Tensor:
    """K/8-byte plane of one code bit: col c -> byte c//8, bit c%8."""
    plane = (unsigned[:, 0::8] >> bit) & 0x1
    for j in range(1, 8):
        plane |= ((unsigned[:, j::8] >> bit) & 0x1) << j
    return plane.to(torch.int8)


def _unpack_codes(qdata: torch.Tensor, k: int, bits: int) -> torch.Tensor:
    """Inverse of _pack_codes: int8 storage -> unsigned int32 codes [N, K]."""
    n = qdata.shape[0]
    packed = qdata.view(torch.uint8).to(torch.int32)  # uint8 view: no sign-extension mask
    shifts1 = torch.arange(8, device=qdata.device, dtype=torch.int32)
    if bits == 3:
        shifts2 = torch.tensor([0, 2, 4, 6], device=qdata.device, dtype=torch.int32)
        codes = ((packed[:, : k // 4].unsqueeze(-1) >> shifts2) & 0x3).reshape(n, k)
        codes |= ((packed[:, k // 4 :].unsqueeze(-1) >> shifts1) & 0x1).reshape(n, k) << 2
        return codes
    low = packed[:, : k // 2]
    codes = torch.empty(n, k, dtype=torch.int32, device=qdata.device)
    codes[:, 0::2] = low & 0xF
    codes[:, 1::2] = (low >> 4) & 0xF
    if bits == 5:
        codes |= ((packed[:, k // 2 :].unsqueeze(-1) >> shifts1) & 0x1).reshape(n, k) << 4
    elif bits == 6:
        # high-plane byte b holds cols 4b..4b+3 at bits 0,2,4,6: one fused expansion
        shifts = torch.tensor([0, 2, 4, 6], device=qdata.device, dtype=torch.int32)
        hi = (packed[:, k // 2 :].unsqueeze(-1) >> shifts) & 0x3
        codes |= hi.reshape(n, k) << 4
    return codes


def w4a8_mma_record_bytes(bits: int) -> int:
    """Bytes per 16-output x 32-K record: 512 codes plus 32 fp8 scale bytes (224 / 288 / 416)."""
    return 512 * bits // 8 + 32


# A record's low-bit tier: one uint16 per lane, bit 4c + j = the low bit of the lane's
# code c of chunk j. The tiers live in their own tensor, so the record stream alone is
# the weight one bit narrower with the same scales.
LSB_TIER_BYTES = 64


def w4a8_mma_packed_numel(n: int, k: int, bits: int) -> int:
    """Bytes of a packed record stream of the given code width."""
    return n * k // 512 * w4a8_mma_record_bytes(bits)


def w4a8_mma_tier_numel(n: int, k: int) -> int:
    """Bytes of a record stream's LSB tier (one bit per weight)."""
    return n * k // 512 * LSB_TIER_BYTES


def _mma_bits(packed_numel: int, n: int, k: int) -> int:
    """Code width of a packed record stream (3, 4 or 6)."""
    for bits in (3, 4, 6):
        if packed_numel == w4a8_mma_packed_numel(n, k, bits):
            return bits
    expected = ", ".join(str(w4a8_mma_packed_numel(n, k, b)) for b in (3, 4, 6))
    raise ValueError(
        f"MMA packed weight must be a contiguous int8 tensor of {expected} bytes, got {packed_numel}"
    )


def _fragment_major_inverse(fragments: torch.Tensor, cols: int) -> torch.Tensor:
    """[tiles, k_rows, 8 lane rows, 4 lane cols, 4 chunks, cols/8] -> [tiles, k_rows, 16, cols]."""
    tiles, k_rows = fragments.shape[:2]
    half = cols // 2
    codes = torch.empty((tiles, k_rows, 16, cols), dtype=fragments.dtype, device=fragments.device)
    codes[:, :, :8, :half] = fragments[:, :, :, :, 0, :].reshape(tiles, k_rows, 8, half)
    codes[:, :, 8:, :half] = fragments[:, :, :, :, 1, :].reshape(tiles, k_rows, 8, half)
    codes[:, :, :8, half:] = fragments[:, :, :, :, 2, :].reshape(tiles, k_rows, 8, half)
    codes[:, :, 8:, half:] = fragments[:, :, :, :, 3, :].reshape(tiles, k_rows, 8, half)
    return codes


def _fragment_major(codes: torch.Tensor) -> torch.Tensor:
    """Inverse of _fragment_major_inverse at 32 columns: [tiles, k_rows, 16, 32] ->
    [tiles, k_rows, 8, 4, 4, 4], lane (r, q) chunk j holding row r + 8 (j & 1), cols
    16 (j >> 1) + 4q .. +3 (mma m16n8k32 fragment order)."""
    tiles, k_rows = codes.shape[:2]
    fragments = torch.empty((tiles, k_rows, 8, 4, 4, 4), dtype=codes.dtype, device=codes.device)
    fragments[:, :, :, :, 0, :] = codes[:, :, :8, :16].reshape(tiles, k_rows, 8, 4, 4)
    fragments[:, :, :, :, 1, :] = codes[:, :, 8:, :16].reshape(tiles, k_rows, 8, 4, 4)
    fragments[:, :, :, :, 2, :] = codes[:, :, :8, 16:].reshape(tiles, k_rows, 8, 4, 4)
    fragments[:, :, :, :, 3, :] = codes[:, :, 8:, 16:].reshape(tiles, k_rows, 8, 4, 4)
    return fragments


def _split_mma_stream(packed: torch.Tensor, n: int, k: int):
    """(record stream view [tiles, k_rows, record_bytes], stream code width)."""
    if packed.dim() != 1 or packed.dtype != torch.int8 or not packed.is_contiguous():
        raise ValueError("MMA packed weight must be a contiguous 1D int8 tensor")
    if n % 16 != 0 or k % 256 != 0:
        raise ValueError("MMA packed weight requires N % 16 == 0 and K % 256 == 0")
    bits = _mma_bits(packed.numel(), n, k)
    tiles, k_rows = n // 16, k // 32
    record_bytes = w4a8_mma_record_bytes(bits)
    return packed.view(torch.uint8).view(tiles, k_rows, record_bytes), bits


def _tier_bits(lsb_tier: torch.Tensor, tiles: int, k_rows: int) -> torch.Tensor:
    """1D uint8 tier -> [tiles, k_rows, 8, 4, 4 chunks, 4 codes] low bits."""
    if lsb_tier.dtype != torch.uint8 or lsb_tier.numel() != tiles * k_rows * LSB_TIER_BYTES:
        raise ValueError(f"LSB tier must be {tiles * k_rows * LSB_TIER_BYTES} uint8 bytes")
    tiers = lsb_tier.contiguous().view(torch.uint16).view(tiles, k_rows, 32)
    shifts = torch.arange(16, device=tiers.device, dtype=torch.int32).view(4, 4).t().reshape(16)
    bits = (tiers.to(torch.int32).unsqueeze(-1) >> shifts) & 1  # [..., 32, (j, c)]
    return bits.view(tiles, k_rows, 8, 4, 4, 4)


def unpack_w4a8_mma_weight(
    packed: torch.Tensor,
    n: int,
    k: int,
    scale_dtype: torch.dtype = torch.float8_e4m3fn,
    lsb_tier: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover canonical codes/scales for explicit dequantization and eager fallback.
    ``lsb_tier`` widens each code by its low bit."""
    records, bits = _split_mma_stream(packed, n, k)
    tiles, k_rows = records.shape[:2]
    code_bytes = w4a8_mma_record_bytes(bits) - 32
    if bits == 4:
        nibbles = records[:, :, :code_bytes].view(tiles, k_rows, 8, 4, 4, 2).to(torch.int32)
        fragments = torch.stack((nibbles & 0xF, nibbles >> 4), dim=-1).view(tiles, k_rows, 8, 4, 4, 4)
    elif bits == 3:
        # lane's 16 codes form one 48-bit value (code (j, c) at bit 12j + 3c): bytes 0-3 at
        # lane*4, bytes 4-5 at 128 + lane*2
        lane_bytes = torch.cat(
            (records[:, :, :128].reshape(tiles, k_rows, 32, 4), records[:, :, 128:code_bytes].reshape(tiles, k_rows, 32, 2)),
            dim=3,
        ).to(torch.int64)
        words = sum(lane_bytes[..., i] << (8 * i) for i in range(6))
        shifts = torch.arange(0, 48, 3, device=packed.device, dtype=torch.int64)
        fragments = ((words.unsqueeze(-1) >> shifts) & 0x7).view(tiles, k_rows, 8, 4, 4, 4).to(torch.int32)
    else:
        lane_bytes = torch.cat(
            (records[:, :, :256].reshape(tiles, k_rows, 32, 8), records[:, :, 256:code_bytes].reshape(tiles, k_rows, 32, 4)),
            dim=3,
        ).to(torch.int32).view(tiles, k_rows, 8, 4, 4, 3)
        words = lane_bytes[..., 0] | lane_bytes[..., 1] << 8 | lane_bytes[..., 2] << 16
        shifts = torch.tensor([0, 6, 12, 18], device=packed.device, dtype=torch.int32)
        fragments = (words.unsqueeze(-1) >> shifts) & 0x3F
    if lsb_tier is not None:
        fragments = (fragments << 1) | _tier_bits(lsb_tier, tiles, k_rows)
        bits += 1
    codes = _fragment_major_inverse(fragments, 32).permute(0, 2, 1, 3).reshape(n, k)
    qdata = _pack_codes(codes, bits)

    scale_fragments = records[:, :, code_bytes:].view(tiles, k_rows, 8, 4)
    scale_tiles = torch.empty((tiles, k_rows, 16, 2), dtype=torch.uint8, device=packed.device)
    scale_tiles[:, :, :8, 0] = scale_fragments[:, :, :, 0]
    scale_tiles[:, :, 8:, 0] = scale_fragments[:, :, :, 1]
    scale_tiles[:, :, :8, 1] = scale_fragments[:, :, :, 2]
    scale_tiles[:, :, 8:, 1] = scale_fragments[:, :, :, 3]
    scales = scale_tiles.permute(0, 2, 1, 3).reshape(n, k // 16)
    return qdata, scales.view(scale_dtype)


def pack_w4a8_mma_weight(
    qdata: torch.Tensor, s_rel: torch.Tensor, lsb_tier: bool = False
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Canonical codes [N, K*bits/8] and fp8 scales [N, K/16] -> the 1D int8 record stream
    the streamed decode reads (the inverse of unpack_w4a8_mma_weight). With ``lsb_tier``
    the stream holds each code's high bits and the 1D uint8 tier of low bits is returned
    with it."""
    n = qdata.shape[0]
    k = s_rel.shape[1] * 16
    bits = qdata.shape[1] * 8 // k
    if s_rel.dtype != torch.float8_e4m3fn or n % 16 or k % 256 or qdata.shape[1] * 8 != k * bits:
        raise ValueError("pack_w4a8_mma_weight needs fp8 g16 scales, N % 16 == 0 and K % 256 == 0")
    if bits not in ((4, 5) if lsb_tier else (3, 4, 6)):
        raise ValueError(f"no packed form for {bits}-bit codes with lsb_tier={lsb_tier}")
    tiles, k_rows = n // 16, k // 32
    codes = _unpack_codes(qdata, k, bits).view(tiles, 16, k_rows, 32).permute(0, 2, 1, 3)
    fragments = _fragment_major(codes)
    tier = None
    if lsb_tier:
        low = (fragments & 1).view(tiles, k_rows, 32, 16)  # (j, c) order
        shifts = torch.arange(16, device=qdata.device, dtype=torch.int32).view(4, 4).t().reshape(16)
        tier = (low << shifts).sum(-1).to(torch.int32)
        tier = torch.stack((tier & 0xFF, tier >> 8), dim=-1).to(torch.uint8).flatten()
        fragments = fragments >> 1
        bits -= 1
    code_bytes = w4a8_mma_record_bytes(bits) - 32
    if bits == 4:
        pairs = fragments.view(tiles, k_rows, 8, 4, 4, 2, 2)
        code_part = (pairs[..., 0] | (pairs[..., 1] << 4)).to(torch.uint8).reshape(tiles, k_rows, code_bytes)
    elif bits == 3:
        shifts = torch.arange(0, 48, 3, device=qdata.device, dtype=torch.int64)
        words = (fragments.to(torch.int64).view(tiles, k_rows, 32, 16) << shifts).sum(-1)
        lane_bytes = torch.stack([(words >> (8 * i)) & 0xFF for i in range(6)], dim=-1).to(torch.uint8)
        code_part = torch.cat(
            (lane_bytes[..., :4].reshape(tiles, k_rows, 128), lane_bytes[..., 4:].reshape(tiles, k_rows, 64)), dim=-1
        )
    else:
        shifts = torch.tensor([0, 6, 12, 18], device=qdata.device, dtype=torch.int32)
        words = (fragments.view(tiles, k_rows, 32, 4, 4) << shifts).sum(-1)  # 24-bit chunk words
        lane_bytes = torch.stack([(words >> (8 * i)) & 0xFF for i in range(3)], dim=-1).to(torch.uint8)
        lane_bytes = lane_bytes.reshape(tiles, k_rows, 32, 12)
        code_part = torch.cat(
            (lane_bytes[..., :8].reshape(tiles, k_rows, 256), lane_bytes[..., 8:].reshape(tiles, k_rows, 128)), dim=-1
        )
    scale_tiles = s_rel.view(torch.uint8).view(tiles, 16, k_rows, 2).permute(0, 2, 1, 3)
    scale_fragments = torch.stack(
        (scale_tiles[:, :, :8, 0], scale_tiles[:, :, 8:, 0], scale_tiles[:, :, :8, 1], scale_tiles[:, :, 8:, 1]), dim=-1
    ).reshape(tiles, k_rows, 32)
    stream = torch.cat((code_part, scale_fragments), dim=-1).flatten().view(torch.int8).contiguous()
    if tier is not None:
        return stream, tier
    return stream
