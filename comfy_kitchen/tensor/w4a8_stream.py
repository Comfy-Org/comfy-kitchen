# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Record-stream layout of a W4A8/W6A8 weight for the CUDA streamed decode kernel
(ops/w4a8_gemm.cu). Pure tensor relayout, shared by the tensor layout, the eager
backend and the CUDA/HIP/Triton backends; a leaf module so none of them import
each other for it."""

from __future__ import annotations

import torch

# Lloyd-Max-optimal 16 levels for a group-normalized Gaussian. ConvRot makes every layer's
# rotated groups Gaussian, so this one table matches a per-tensor fit and skips the k-means.
_FIXED_LUT = (
    -0.980602, -0.794529, -0.638165, -0.500986, -0.377321, -0.263187, -0.155210, -0.050720,
    0.052541, 0.156985, 0.265284, 0.379533, 0.502636, 0.638953, 0.794876, 0.980671,
)


def default_w4a8_codebook() -> torch.Tensor:
    """The frozen Lloyd-Max codebook (the CUDA streamed decode kernel's built-in LUT)."""
    return torch.tensor(_FIXED_LUT, dtype=torch.float32)


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


W4A8_MMA_PACK_ROWS = 8  # records per contiguous run (the kernel's kPackRows)
# The streamed kernel's grid is sized so tiles x splits fills about one wave of this
# many warps. A constant rather than the running GPU's SM count (170 x 20 on the
# RTX 5090) so the packed layout is a function of the shape alone.
_W4A8_MMA_WAVE_WARPS = 3400


def w4a8_mma_record_bytes(bits: int) -> int:
    """Bytes per 16-output x 32-K record: 512 codes plus 32 fp8 scale bytes (288 / 416)."""
    return 512 * bits // 8 + 32


def w4a8_mma_stream_rows(n: int, k: int) -> int:
    """Records per split each warp streams for an [N, K] weight, or 0 when the MMA
    layout does not apply: the largest of 32/16/8 dividing K/32 that still fills a
    wave, so small matrices (o_proj) do not run under-occupied."""
    if n % 16 != 0 or k % (W4A8_MMA_PACK_ROWS * 32) != 0:
        return 0
    k_rows, tiles = k // 32, n // 16
    rows = 32
    while rows > W4A8_MMA_PACK_ROWS and (k_rows % rows != 0 or tiles * (k_rows // rows) < _W4A8_MMA_WAVE_WARPS):
        rows //= 2
    return rows


def _mma_bits(packed_numel: int, n: int, k: int) -> int:
    for bits in (4, 6):
        if packed_numel == n * k * w4a8_mma_record_bytes(bits) // 512:
            return bits
    raise ValueError(f"MMA packed weight must be a contiguous int8 [{n * k * 9 // 16}] or [{n * k * 13 // 16}] tensor")


def _fragment_major(codes: torch.Tensor, cols: int) -> torch.Tensor:
    """[tiles, k_rows, 16 outputs, cols] -> [tiles, k_rows, 8 tokens, 4 groups, 4 fragments,
    cols/8]: the m16n8k32 A-fragment order (frag j = output half j & 1, K half j >> 1)."""
    tiles, k_rows = codes.shape[:2]
    half = cols // 2
    return torch.stack(
        (
            codes[:, :, :8, :half].reshape(tiles, k_rows, 8, 4, half // 4),
            codes[:, :, 8:, :half].reshape(tiles, k_rows, 8, 4, half // 4),
            codes[:, :, :8, half:].reshape(tiles, k_rows, 8, 4, half // 4),
            codes[:, :, 8:, half:].reshape(tiles, k_rows, 8, 4, half // 4),
        ),
        dim=4,
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


def pack_w4a8_mma_weight(qdata: torch.Tensor, s_rel: torch.Tensor, stream_rows: int) -> torch.Tensor:
    """Relayout [N, K*bits/8] codes + [N, K/16] fp8 group scales into the streamed MMA
    kernel's record stream (w4a8_gemm.cu): one record per 16-output x 32-K tile, the
    codes in mma m16n8k32 fragment order then 32 scale bytes, run-major
    [krow_in_split // 8][split][tile][8][record] with K/32/stream_rows splits, which the
    kernel reads front to back. At 4 bits a lane's 16 codes are 8 nibble bytes at
    lane*8; at 6 bits each 4-code fragment is 24 little-endian bits, the lane's first
    8 bytes at lane*8 and the last 4 at 256 + lane*4."""
    if qdata.dim() != 2 or qdata.dtype != torch.int8:
        raise ValueError("qdata must be a 2D int8 tensor")
    n, groups = s_rel.shape
    k = groups * 16
    bits = qdata.shape[1] * 8 // k if k else 0
    if qdata.shape[0] != n or bits not in (4, 6) or qdata.shape[1] * 8 != k * bits:
        raise ValueError("MMA packing requires group_size 16 scales and 4- or 6-bit codes")
    if n % 16 != 0 or k % (stream_rows * 32) != 0 or stream_rows % W4A8_MMA_PACK_ROWS != 0:
        raise ValueError("MMA packing requires N % 16 == 0 and K % (32 * stream_rows) == 0")
    tiles, k_rows = n // 16, k // 32
    if bits == 4:
        codes = qdata.view(tiles, 16, k_rows, 16).permute(0, 2, 1, 3)
        weight_bytes = _fragment_major(codes, 16).contiguous().view(tiles, k_rows, 256).view(torch.uint8)
    else:
        codes = _unpack_codes(qdata, k, bits).view(tiles, 16, k_rows, 32).permute(0, 2, 1, 3)
        fragments = _fragment_major(codes, 32)  # [..., 4 frag, 4 codes]
        words = fragments[..., 0] | fragments[..., 1] << 6 | fragments[..., 2] << 12 | fragments[..., 3] << 18
        lane_bytes = (
            torch.stack((words & 0xFF, (words >> 8) & 0xFF, (words >> 16) & 0xFF), dim=-1)
            .to(torch.uint8)
            .reshape(tiles, k_rows, 32, 12)
        )
        weight_bytes = torch.cat(
            (lane_bytes[..., :8].reshape(tiles, k_rows, 256), lane_bytes[..., 8:].reshape(tiles, k_rows, 128)),
            dim=2,
        )
    scale_tiles = s_rel.view(torch.uint8).view(tiles, 16, k_rows, 2).permute(0, 2, 1, 3)
    scale_bytes = torch.stack(
        (
            scale_tiles[:, :, :8, 0],
            scale_tiles[:, :, 8:, 0],
            scale_tiles[:, :, :8, 1],
            scale_tiles[:, :, 8:, 1],
        ),
        dim=3,
    ).contiguous().view(tiles, k_rows, 32)
    records = torch.cat((weight_bytes, scale_bytes), dim=2)
    splits, runs = k_rows // stream_rows, stream_rows // W4A8_MMA_PACK_ROWS
    return (
        records.view(tiles, splits, runs, W4A8_MMA_PACK_ROWS, w4a8_mma_record_bytes(bits))
        .permute(2, 1, 0, 3, 4)
        .contiguous()
        .view(torch.int8)
        .view(-1)
    )


def unpack_w4a8_mma_weight(
    packed: torch.Tensor,
    n: int,
    k: int,
    stream_rows: int,
    scale_dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of pack_w4a8_mma_weight: conventional qdata and scales for prefill,
    dequantization, and saving."""
    if packed.dim() != 1 or packed.dtype != torch.int8:
        raise ValueError("MMA packed weight must be a contiguous 1D int8 tensor")
    bits = _mma_bits(packed.numel(), n, k)
    record_bytes = w4a8_mma_record_bytes(bits)
    tiles, k_rows = n // 16, k // 32
    splits, runs = k_rows // stream_rows, stream_rows // W4A8_MMA_PACK_ROWS
    records = (
        packed.view(torch.uint8)
        .view(runs, splits, tiles, W4A8_MMA_PACK_ROWS, record_bytes)
        .permute(2, 1, 0, 3, 4)
        .reshape(tiles, k_rows, record_bytes)
    )
    code_bytes = record_bytes - 32
    if bits == 4:
        fragments = records[:, :, :code_bytes].view(tiles, k_rows, 8, 4, 4, 2)
        qdata = _fragment_major_inverse(fragments, 16).permute(0, 2, 1, 3).reshape(n, k // 2)
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
