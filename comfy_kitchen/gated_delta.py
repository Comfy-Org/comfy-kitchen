from __future__ import annotations

import torch

from .backends import cuda as _cuda_backend

if getattr(torch.version, "hip", None):
    from .backends import hip as _hip_backend
else:
    _hip_backend = None

_MAX_STEPS = 8
_device_optin: dict[int, int] = {}


def _fused_shmem_bytes(key_head_dim: int, value_head_dim: int) -> int:
    # fp32 state slice, q/k rows for every step, per-warp reduce scratch (mirrors the launcher)
    warps = max(value_head_dim, 128) // 32
    return (key_head_dim * value_head_dim + 2 * _MAX_STEPS * key_head_dim + 4 * _MAX_STEPS * warps) * 4


def is_available(device: torch.device | None = None, key_head_dim: int = 128, value_head_dim: int = 128) -> bool:
    """Return whether the fused DeltaNet decode kernels can run on this device for these head dims."""
    if device is not None and device.type != "cuda":
        return False
    if not torch.cuda.is_available():
        return False
    if _hip_backend is not None:
        # torch.cuda is the ROCm API here, so the CUDA extension test below would
        # wave AMD hardware through to an extension that never loaded. The HIP
        # backend answers for the process rather than for one device, the way
        # flash_attention.py asks it to: its arch gate is the intersection over
        # every visible device. It sizes its own shared memory, so there is no
        # opt-in budget to check.
        return _hip_backend.gated_delta_decode_is_available(key_head_dim, value_head_dim)
    ext = _cuda_backend._C if _cuda_backend._EXT_AVAILABLE else None
    if ext is None or not hasattr(ext, "gated_delta_decode_fused") or not hasattr(ext, "deltanet_conv_step"):
        return False
    if key_head_dim != 128 or value_head_dim % 32 != 0 or not 0 < value_head_dim <= 512:
        return False
    index = device.index if device is not None else None
    if index is None:
        index = torch.cuda.current_device()
    optin = _device_optin.get(index)
    if optin is None:
        props = torch.cuda.get_device_properties(index)
        optin = getattr(props, "shared_memory_per_block_optin", None)
        if optin is None:
            optin = 96 * 1024 if props.major >= 8 else 0
        _device_optin[index] = optin
    return _fused_shmem_bytes(key_head_dim, value_head_dim) <= optin


def gated_delta_decode_fused(
    mixed_qkv: torch.Tensor,
    x: torch.Tensor,
    w_a: torch.Tensor,
    w_b: torch.Tensor,
    dt_bias: torch.Tensor,
    g_decay: torch.Tensor,
    state: torch.Tensor,
    key_dim: int,
    num_key_heads: int,
    scale: float,
    z: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    snapshots: torch.Tensor | None = None,
) -> torch.Tensor:
    """S GatedDeltaNet decode steps from the conv output [B, C, S].

    The fp32 state [B, Hv, DK, DV] is updated in place. State, dt_bias, g_decay,
    and snapshots must be contiguous.
    """
    batch, _, seq = mixed_qkv.shape
    heads, key_dim_head, value_dim = state.shape[1], state.shape[2], state.shape[3]
    if not is_available(x.device, key_dim_head, value_dim):
        raise RuntimeError("gated_delta_decode_fused is unavailable for this device and head shape")
    out = torch.empty((batch, seq, heads, value_dim), dtype=x.dtype, device=x.device)
    if _hip_backend is not None:
        ok = _hip_backend.gated_delta_decode_fused(
            mixed_qkv.contiguous(), x.contiguous(), w_a.contiguous(), w_b.contiguous(),
            dt_bias, g_decay, state, out, snapshots,
            z.reshape(batch, seq, heads * value_dim).contiguous(), norm_weight.contiguous(),
            eps, key_dim, num_key_heads, scale,
        )
        if not ok:
            raise RuntimeError("gated_delta_decode_fused launch rejected")
        return out
    wrap = _cuda_backend._wrap_for_dlpack
    ok = _cuda_backend._C.gated_delta_decode_fused(
        wrap(mixed_qkv.contiguous()), wrap(x.contiguous()), wrap(w_a.contiguous()), wrap(w_b.contiguous()),
        wrap(dt_bias), wrap(g_decay), wrap(state), wrap(out),
        wrap(snapshots) if snapshots is not None else None,
        wrap(z.reshape(batch, seq, heads * value_dim).contiguous()), wrap(norm_weight.contiguous()), eps,
        key_dim, num_key_heads, scale,
        torch.cuda.current_stream(x.device).cuda_stream,
    )
    if not ok:
        raise RuntimeError("gated_delta_decode_fused launch rejected")
    return out


def deltanet_conv_step(
    proj: torch.Tensor,
    conv_state: torch.Tensor,
    conv_w: torch.Tensor,
    conv_b: torch.Tensor | None = None,
    snapshots: torch.Tensor | None = None,
) -> torch.Tensor:
    """Depthwise causal conv + silu over proj [B, S, C], returning [B, C, S].

    The conv_state [B, C, KS-1] is updated in place. Conv_state and snapshots
    must be contiguous.
    """
    if not is_available(proj.device):
        raise RuntimeError("deltanet_conv_step requires the CUDA or HIP extension")
    batch, seq, channels = proj.shape
    out = torch.empty((batch, channels, seq), dtype=proj.dtype, device=proj.device)
    if _hip_backend is not None:
        ok = _hip_backend.deltanet_conv_step(
            proj.contiguous(), conv_state, conv_w.reshape(channels, -1).contiguous(),
            conv_b.contiguous() if conv_b is not None else None, out, snapshots,
        )
        if not ok:
            raise RuntimeError("deltanet_conv_step launch rejected")
        return out
    wrap = _cuda_backend._wrap_for_dlpack
    ok = _cuda_backend._C.deltanet_conv_step(
        wrap(proj.contiguous()), wrap(conv_state), wrap(conv_w.reshape(channels, -1).contiguous()),
        wrap(conv_b.contiguous()) if conv_b is not None else None, wrap(out),
        wrap(snapshots) if snapshots is not None else None,
        torch.cuda.current_stream(proj.device).cuda_stream,
    )
    if not ok:
        raise RuntimeError("deltanet_conv_step launch rejected")
    return out
