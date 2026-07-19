import torch

import triton
import triton.language as tl


def _detect_seq_axis(x: torch.Tensor, freqs_cis: torch.Tensor) -> int:
    if freqs_cis.shape[1] == 1 and freqs_cis.shape[2] in (1, x.shape[2]):
        return 2
    return 1


@triton.jit
def rms_rope_kernel(
    x_ptr,
    freqs_ptr,
    scale_ptr,
    out_ptr,
    head_dim,
    freqs_batch,
    stride_x_batch,
    stride_x_head,
    stride_x_seq,
    stride_x_dim,
    stride_freqs_batch,
    stride_freqs_seq,
    stride_freqs_dim,
    stride_freqs_rot,
    stride_freqs_pair,
    epsilon,
    compute_dtype: tl.constexpr,
    norm_dtype: tl.constexpr,
    block_size: tl.constexpr,
    split_half: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    seq_idx = tl.program_id(2)

    n_pairs = head_dim // 2
    offsets = tl.arange(0, block_size)
    mask = offsets < n_pairs

    x_offset = (batch_idx * stride_x_batch + head_idx * stride_x_head + seq_idx * stride_x_seq)
    x_base = x_ptr + x_offset
    out_base = out_ptr + x_offset

    full_offsets = tl.arange(0, block_size * 2)
    full_mask = full_offsets < head_dim

    x_full = tl.load(x_base + full_offsets * stride_x_dim, mask=full_mask, other=0.0).to(tl.float32)

    inv_rms = tl.math.rsqrt(tl.sum(x_full * x_full, axis=0) / head_dim + epsilon)

    if split_half:
        dim_idx_0 = offsets
        dim_idx_1 = offsets + n_pairs
    else:
        dim_idx_0 = offsets * 2
        dim_idx_1 = offsets * 2 + 1

    x_0 = tl.load(x_base + dim_idx_0 * stride_x_dim, mask=mask, other=0.0).to(tl.float32)
    x_1 = tl.load(x_base + dim_idx_1 * stride_x_dim, mask=mask, other=0.0).to(tl.float32)

    scale_0 = tl.load(scale_ptr + dim_idx_0, mask=mask, other=0.0).to(tl.float32)
    scale_1 = tl.load(scale_ptr + dim_idx_1, mask=mask, other=0.0).to(tl.float32)

    # Match RMSNorm output materialization before RoPE.
    x_0 = (x_0 * inv_rms * scale_0).to(norm_dtype).to(compute_dtype)
    x_1 = (x_1 * inv_rms * scale_1).to(norm_dtype).to(compute_dtype)

    freqs_batch_idx = tl.where(freqs_batch == 1, 0, batch_idx)
    freqs_base = (freqs_ptr +
                  freqs_batch_idx * stride_freqs_batch +
                  seq_idx * stride_freqs_seq +
                  offsets * stride_freqs_dim)

    freqs_00 = tl.load(freqs_base, mask=mask, other=0.0)
    freqs_01 = tl.load(freqs_base + stride_freqs_pair, mask=mask, other=0.0)
    freqs_10 = tl.load(freqs_base + stride_freqs_rot, mask=mask, other=0.0 )
    freqs_11 = tl.load(freqs_base + stride_freqs_rot + stride_freqs_pair, mask=mask, other=0.0)

    out_0 = freqs_00 * x_0 + freqs_01 * x_1
    out_1 = freqs_10 * x_0 + freqs_11 * x_1

    tl.store(out_base + dim_idx_0 * stride_x_dim, out_0, mask=mask)
    tl.store(out_base + dim_idx_1 * stride_x_dim, out_1, mask=mask)


def _rms_rope(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float,
    split_half: bool,
) -> torch.Tensor:
    if not x.is_contiguous():
        x = x.contiguous()

    if not freqs_cis.is_contiguous():
        freqs_cis = freqs_cis.contiguous()

    if not scale.is_contiguous():
        scale = scale.contiguous()

    batch, dim1, dim2, head_dim = x.shape
    freqs_batch = freqs_cis.shape[0]
    seq_axis = _detect_seq_axis(x, freqs_cis)

    if seq_axis == 1:
        num_heads = dim2
        seq_len = dim1
        stride_x_head = x.stride(2)
        stride_x_seq = x.stride(1)
        stride_freqs_seq = freqs_cis.stride(1)
    else:
        num_heads = dim1
        seq_len = dim2
        stride_x_head = x.stride(1)
        stride_x_seq = x.stride(2)
        stride_freqs_seq = freqs_cis.stride(2)

    dtype_map = {
        torch.float32: tl.float32,
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
    }

    out = torch.empty_like(x)
    block_size = triton.next_power_of_2(head_dim // 2)
    grid = (batch, num_heads, seq_len)

    rms_rope_kernel[grid](
        x,
        freqs_cis,
        scale,
        out,
        head_dim,
        freqs_batch,
        x.stride(0),
        stride_x_head,
        stride_x_seq,
        x.stride(3),
        freqs_cis.stride(0),
        stride_freqs_seq,
        freqs_cis.stride(3),
        freqs_cis.stride(4),
        freqs_cis.stride(5),
        epsilon,
        compute_dtype=dtype_map.get(freqs_cis.dtype, tl.float32),
        norm_dtype=dtype_map.get(x.dtype, tl.float32),
        block_size=block_size,
        split_half=split_half,
    )

    return out


def rms_rope1(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    return _rms_rope(x, freqs_cis, scale, epsilon, split_half=False)


def rms_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor | None = None,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    if k_scale is None:
        k_scale = q_scale

    return (
        _rms_rope(q, freqs_cis, q_scale, epsilon, split_half=False),
        _rms_rope(k, freqs_cis, k_scale, epsilon, split_half=False),
    )


def rms_rope_split_half1(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    return _rms_rope(x, freqs_cis, scale, epsilon, split_half=True)


def rms_rope_split_half(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor | None = None,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    if k_scale is None:
        k_scale = q_scale

    return (
        _rms_rope(q, freqs_cis, q_scale, epsilon, split_half=True),
        _rms_rope(k, freqs_cis, k_scale, epsilon, split_half=True),
    )
