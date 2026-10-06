# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Record-stream layout of a W4A8 weight for the CUDA streamed decode kernel
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



W4A8_MMA_PACK_ROWS = 8  # records per contiguous run (the kernel's kPackRows)
# The streamed kernel's grid is sized so tiles x splits fills about one wave of this
# many warps. A constant rather than the running GPU's SM count (170 x 20 on the
# RTX 5090) so the packed layout is a function of the shape alone.
_W4A8_MMA_WAVE_WARPS = 3400


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


def pack_w4a8_mma_weight(qdata: torch.Tensor, s_rel: torch.Tensor, stream_rows: int) -> torch.Tensor:
    """Relayout [N, K/2] int4 codes + [N, K/16] fp8 group scales into the streamed
    MMA kernel's record stream (w4a8_gemm.cu): 288-byte records per 16-output x 32-K
    tile, run-major [krow_in_split // 8][split][tile][8][288] with K/32/stream_rows
    splits, which the kernel reads front to back."""
    if qdata.dim() != 2 or qdata.dtype != torch.int8:
        raise ValueError("qdata must be a 2D int8 tensor")
    n, k_half = qdata.shape
    k = k_half * 2
    if n % 16 != 0 or k % (stream_rows * 32) != 0 or stream_rows % W4A8_MMA_PACK_ROWS != 0:
        raise ValueError("MMA packing requires N % 16 == 0 and K % (32 * stream_rows) == 0")
    if tuple(s_rel.shape) != (n, k // 16):
        raise ValueError("MMA packing requires group_size 16 scales")
    tiles, k_rows = n // 16, k // 32
    codes = qdata.view(tiles, 16, k_rows, 16).permute(0, 2, 1, 3)
    fragments = torch.stack(
        (
            codes[:, :, :8, :8].reshape(tiles, k_rows, 8, 4, 2),
            codes[:, :, 8:, :8].reshape(tiles, k_rows, 8, 4, 2),
            codes[:, :, :8, 8:].reshape(tiles, k_rows, 8, 4, 2),
            codes[:, :, 8:, 8:].reshape(tiles, k_rows, 8, 4, 2),
        ),
        dim=4,
    )
    weight_bytes = fragments.contiguous().view(tiles, k_rows, 256).view(torch.uint8)
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
        records.view(tiles, splits, runs, W4A8_MMA_PACK_ROWS, 288)
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
    if packed.dim() != 1 or packed.dtype != torch.int8 or packed.numel() != n * k * 9 // 16:
        raise ValueError(f"MMA packed weight must be a contiguous int8 [{n * k * 9 // 16}] tensor")
    tiles, k_rows = n // 16, k // 32
    splits, runs = k_rows // stream_rows, stream_rows // W4A8_MMA_PACK_ROWS
    records = (
        packed.view(torch.uint8)
        .view(runs, splits, tiles, W4A8_MMA_PACK_ROWS, 288)
        .permute(2, 1, 0, 3, 4)
        .reshape(tiles, k_rows, 288)
    )
    fragments = records[:, :, :256].view(tiles, k_rows, 8, 4, 4, 2)
    codes = torch.empty((tiles, k_rows, 16, 16), dtype=torch.uint8, device=packed.device)
    codes[:, :, :8, :8] = fragments[:, :, :, :, 0, :].reshape(tiles, k_rows, 8, 8)
    codes[:, :, 8:, :8] = fragments[:, :, :, :, 1, :].reshape(tiles, k_rows, 8, 8)
    codes[:, :, :8, 8:] = fragments[:, :, :, :, 2, :].reshape(tiles, k_rows, 8, 8)
    codes[:, :, 8:, 8:] = fragments[:, :, :, :, 3, :].reshape(tiles, k_rows, 8, 8)
    qdata = codes.permute(0, 2, 1, 3).reshape(n, k // 2)

    scale_fragments = records[:, :, 256:].view(tiles, k_rows, 8, 4)
    scale_tiles = torch.empty((tiles, k_rows, 16, 2), dtype=torch.uint8, device=packed.device)
    scale_tiles[:, :, :8, 0] = scale_fragments[:, :, :, 0]
    scale_tiles[:, :, 8:, 0] = scale_fragments[:, :, :, 1]
    scale_tiles[:, :, :8, 1] = scale_fragments[:, :, :, 2]
    scale_tiles[:, :, 8:, 1] = scale_fragments[:, :, :, 3]
    scales = scale_tiles.permute(0, 2, 1, 3).reshape(n, k // 16)
    return qdata.view(torch.int8), scales.view(scale_dtype)
