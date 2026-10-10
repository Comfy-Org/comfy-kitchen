from __future__ import annotations

import torch

from .backends import cuda as _cuda_backend

if getattr(torch.version, "hip", None):
    from .backends import hip as _hip_backend
else:
    _hip_backend = None

_MAX_STEPS = 8
# Published wheels compile 75-virtual as the floor (see setup.py). Volta
# (sm_70) still reports enough opt-in shared memory for the fused decode
# budget, so the shmem check alone waves V100 through to a rejected launch.
_NATIVE_MINIMUM_CAPABILITY = (7, 5)
_device_optin: dict[int, int] = {}
SLOT_MAX = 8  # token slots per deferred side buffer; verify steps run S <= SLOT_MAX tokens
CTL_INTS = 2  # ctl = {pending, parity}


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
    if torch.cuda.get_device_capability(device) < _NATIVE_MINIMUM_CAPABILITY:
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


def deferred_is_available(device: torch.device | int | None = None, key_head_dim: int = 128, value_head_dim: int = 128) -> bool:
    """Return whether deltanet_conv_step_deferred and gated_delta_decode_deferred run their
    native kernels on this device: DK = DV = 128 on CUDA sm_90+ (the decode kernel runs one
    head per 4-block cluster) or on the HIP extension's supported GPUs. They otherwise compute
    the same result in torch."""
    if not torch.cuda.is_available():
        return False
    if device is not None and torch.device(device).type != "cuda":
        return False
    if key_head_dim != 128 or value_head_dim != 128:
        return False
    if _hip_backend is not None:
        return _hip_backend.gated_delta_deferred_is_available(key_head_dim, value_head_dim)
    ext = _cuda_backend._C if _cuda_backend._EXT_AVAILABLE else None
    if ext is None or not hasattr(ext, "gated_delta_decode_deferred") or not hasattr(ext, "deltanet_conv_deferred"):
        return False
    return torch.cuda.get_device_capability(device) >= (9, 0)


def deferred_buffers(batch: int, channels: int, heads: int, num_key_heads: int, dtype: torch.dtype, device: torch.device):
    """Allocate the side buffers of the deferred-commit decode: (qkv_buf, proj_buf, gates_buf, sumsq_buf).

    Each is double-buffered over the step parity and holds SLOT_MAX token slots.
    """
    qkv_buf = torch.empty((2, batch, channels, SLOT_MAX), dtype=dtype, device=device)
    proj_buf = torch.empty((2, batch, SLOT_MAX, channels), dtype=dtype, device=device)
    gates_buf = torch.empty((2, batch, SLOT_MAX, heads, 2), dtype=torch.float32, device=device)
    sumsq_buf = torch.empty((2, batch, SLOT_MAX, num_key_heads, 2), dtype=torch.float32, device=device)
    return qkv_buf, proj_buf, gates_buf, sumsq_buf


def deltanet_conv_step_deferred(
    proj: torch.Tensor,
    conv_state: torch.Tensor,
    conv_w: torch.Tensor,
    conv_b: torch.Tensor | None,
    proj_buf: torch.Tensor,
    qkv_buf: torch.Tensor,
    ctl: torch.Tensor,
) -> None:
    """Depthwise causal conv + silu over proj [B, S, C] into qkv_buf[ctl[1]] (stride SLOT_MAX).

    ctl is an int32 device vector {pending, parity}: the first `pending` tokens of
    proj_buf[1 - parity] are committed into conv_state before convolving the current
    chain. The current projections are saved to proj_buf[parity] for the next step.
    """
    channels = proj.shape[2]
    conv_w = conv_w.reshape(channels, -1).contiguous()
    if not deferred_is_available(proj.device):
        _conv_deferred_torch(proj, conv_state, conv_w, conv_b, proj_buf, qkv_buf, ctl)
        return
    if _hip_backend is not None:
        ok = _hip_backend.deltanet_conv_deferred(
            proj.contiguous(), proj_buf, conv_state, conv_w,
            conv_b.contiguous() if conv_b is not None else None, qkv_buf, ctl)
    else:
        wrap = _cuda_backend._wrap_for_dlpack
        ok = _cuda_backend._C.deltanet_conv_deferred(
            wrap(proj), wrap(proj_buf), wrap(conv_state), wrap(conv_w),
            wrap(conv_b.contiguous()) if conv_b is not None else None, wrap(qkv_buf), wrap(ctl),
            torch.cuda.current_stream(proj.device).cuda_stream,
        )
    if not ok:
        raise RuntimeError("deltanet_conv_step_deferred launch rejected")


def _slab(buf: torch.Tensor, parity: torch.Tensor) -> torch.Tensor:
    # buf[parity] for a 0-d device parity: a copy, so the caller writes back with index_copy_
    return buf.index_select(0, parity.view(1))[0]


def _conv_deferred_torch(proj, conv_state, conv_w, conv_b, proj_buf, qkv_buf, ctl):
    # the kernels' semantics in torch ops only: ctl stays on its device (graph-capturable)
    batch, seq, channels = proj.shape
    window = conv_state.shape[2]
    taps = conv_w.shape[1]
    ctl = ctl.long()  # index_copy_ wants int64 indices
    pending, parity = ctl[0], ctl[1]
    prev = _slab(proj_buf, 1 - parity).transpose(1, 2)  # [B, C, 8]
    # committed window = the last `window` of [conv_state, prev[0..pending)]
    combined = torch.cat([conv_state, prev], dim=2)
    committed = combined.index_select(2, pending + torch.arange(window, device=ctl.device))
    conv_state.copy_(committed)
    win = torch.cat([committed, proj.transpose(1, 2)], dim=2).float()  # [B, C, window + S]
    # tap d of row s: the token d steps back in the committed window followed by the chain
    rows = torch.arange(seq, dtype=ctl.dtype, device=ctl.device)
    src = window + rows.view(seq, 1) - torch.arange(taps, dtype=ctl.dtype, device=ctl.device)  # [S, taps] in distance order
    gathered = win.index_select(2, src.view(-1)).view(batch, channels, seq, taps)
    acc = (gathered * conv_w.float().flip(1).view(1, channels, 1, taps)).sum(-1)
    if conv_b is not None:
        acc = acc + conv_b.float().view(1, channels, 1)
    y = acc.to(proj.dtype).float()
    out = (y / (1.0 + torch.exp(-y))).to(proj.dtype)  # [B, C, S]
    qkv_buf.index_copy_(0, parity.view(1), torch.nn.functional.pad(out, (0, SLOT_MAX - seq)).unsqueeze(0))
    proj_buf.index_copy_(0, parity.view(1), torch.nn.functional.pad(proj, (0, 0, 0, SLOT_MAX - seq)).unsqueeze(0))


def gated_delta_decode_deferred(
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
    qkv_buf: torch.Tensor,
    gates_buf: torch.Tensor,
    sumsq_buf: torch.Tensor,
    ctl: torch.Tensor,
) -> torch.Tensor:
    """S GatedDeltaNet decode steps from qkv_buf[ctl[1]] written by deltanet_conv_step_deferred.

    Replays the first `ctl[0]` tokens of the previous step from the [1 - parity] side
    buffers, writes the committed fp32 state [B, Hv, DK, DV] in place, then returns the
    outputs of the S current chain tokens without committing them.
    State, dt_bias and g_decay must be contiguous.
    """
    batch, seq, _ = x.shape
    heads, key_dim_head, value_dim = state.shape[1], state.shape[2], state.shape[3]
    z = z.reshape(batch, seq, heads * value_dim)
    if not deferred_is_available(x.device, key_dim_head, value_dim):
        return _decode_deferred_torch(x, w_a, w_b, dt_bias, g_decay, state, key_dim, num_key_heads, scale,
                                      z, norm_weight, eps, qkv_buf, gates_buf, sumsq_buf, ctl)
    out = torch.empty((batch, seq, heads, value_dim), dtype=x.dtype, device=x.device)
    if _hip_backend is not None:
        ok = _hip_backend.gated_delta_decode_deferred(
            x.contiguous(), w_a.contiguous(), w_b.contiguous(), dt_bias, g_decay,
            qkv_buf, gates_buf, sumsq_buf, ctl, state, out, z, norm_weight.contiguous(), eps,
            key_dim, num_key_heads, scale)
    else:
        wrap = _cuda_backend._wrap_for_dlpack
        ok = _cuda_backend._C.gated_delta_decode_deferred(
            wrap(x.contiguous()), wrap(w_a.contiguous()), wrap(w_b.contiguous()),
            wrap(dt_bias), wrap(g_decay), wrap(qkv_buf), wrap(gates_buf), wrap(sumsq_buf), wrap(ctl),
            wrap(state), wrap(out), wrap(z), wrap(norm_weight.contiguous()), eps,
            key_dim, num_key_heads, scale,
            torch.cuda.current_stream(x.device).cuda_stream,
        )
    if not ok:
        raise RuntimeError("gated_delta_decode_deferred launch rejected")
    return out


def _decode_deferred_torch(x, w_a, w_b, dt_bias, g_decay, state, key_dim, num_key_heads, scale,
                           z, norm_weight, eps, qkv_buf, gates_buf, sumsq_buf, ctl):
    # the kernel's semantics in torch ops only, rounding where it rounds; ctl stays on its
    # device, so the pending replay runs all SLOT_MAX slots with the gates of the slots past
    # `pending` neutralized (decay 1, beta 0)
    batch, seq, _ = x.shape
    heads, key_dim_head, value_dim = state.shape[1], state.shape[2], state.shape[3]
    groups = heads // num_key_heads
    dtype = x.dtype
    ctl = ctl.long()  # index_copy_ wants int64 indices
    pending, parity = ctl[0], ctl[1]

    # gates of the current rows: bf16 projections and sigmoid, fp32 softplus and exp
    beta = torch.nn.functional.linear(x, w_b).sigmoid().float()  # [B, S, Hv]
    aa = torch.nn.functional.linear(x, w_a).float() + dt_bias
    decay = torch.exp(g_decay * torch.nn.functional.softplus(aa))
    cur = _slab(qkv_buf, parity)[:, :, :seq]  # [B, C, S]
    cur_q = cur[:, :key_dim].reshape(batch, num_key_heads, key_dim_head, seq).float()
    cur_k = cur[:, key_dim:2 * key_dim].reshape(batch, num_key_heads, key_dim_head, seq).float()
    cur_v = cur[:, 2 * key_dim:].reshape(batch, heads, value_dim, seq).float()
    cur_ss = torch.stack([cur_q.square().sum(2), cur_k.square().sum(2)], dim=-1)  # [B, Hk, S, 2]
    gates_buf.index_copy_(0, parity.view(1), torch.nn.functional.pad(
        torch.stack([decay, beta], dim=-1), (0, 0, 0, 0, 0, SLOT_MAX - seq)).unsqueeze(0))
    sumsq_buf.index_copy_(0, parity.view(1), torch.nn.functional.pad(
        cur_ss.transpose(1, 2), (0, 0, 0, 0, 0, SLOT_MAX - seq)).unsqueeze(0))

    # the previous step's rows; the slots past `pending` are zeroed (the kernels never read them)
    live = pending > torch.arange(SLOT_MAX, device=ctl.device)
    prev = torch.where(live.view(1, 1, SLOT_MAX), _slab(qkv_buf, 1 - parity), 0.0)
    prev_q = prev[:, :key_dim].reshape(batch, num_key_heads, key_dim_head, SLOT_MAX).float()
    prev_k = prev[:, key_dim:2 * key_dim].reshape(batch, num_key_heads, key_dim_head, SLOT_MAX).float()
    prev_v = prev[:, 2 * key_dim:].reshape(batch, heads, value_dim, SLOT_MAX).float()
    prev_gates = _slab(gates_buf, 1 - parity)  # [B, 8, Hv, 2]
    prev_ss = torch.where(live.view(1, 1, SLOT_MAX, 1),
                          _slab(sumsq_buf, 1 - parity).transpose(1, 2), 1.0)  # [B, Hk, 8, 2]
    prev_decay = torch.where(live.view(1, SLOT_MAX, 1), prev_gates[..., 0], 1.0)
    prev_beta = torch.where(live.view(1, SLOT_MAX, 1), prev_gates[..., 1], 0.0)

    def normalized(q, k, ss):
        n = ss.sqrt().clamp(min=1e-12).unsqueeze(2)  # [B, Hk, 1, T, 2]
        q = (q / n[..., 0] * scale).repeat_interleave(groups, dim=1)
        k = (k / n[..., 1]).repeat_interleave(groups, dim=1)
        return q, k

    st = state
    oc = torch.empty((batch, seq, heads, value_dim), dtype=torch.float32, device=x.device)

    def step(st, q, k, v, decay, beta):
        # q/k [B, Hv, DK], v [B, Hv, DV], decay/beta [B, Hv]; returns (new state, output)
        sgs = st * decay[:, :, None, None]
        kvm = torch.einsum("bhk,bhkv->bhv", k, sgs)
        delta = (v - kvm) * beta[:, :, None]
        n = sgs + k[:, :, :, None] * delta[:, :, None, :]
        return n, torch.einsum("bhk,bhkv->bhv", q, n)

    q, k = normalized(prev_q, prev_k, prev_ss)
    for i in range(SLOT_MAX):
        st, _ = step(st, q[..., i], k[..., i], prev_v[..., i], prev_decay[:, i], prev_beta[:, i])
    state.copy_(st)

    q, k = normalized(cur_q, cur_k, cur_ss)
    for s in range(seq):
        st, oc[:, s] = step(st, q[..., s], k[..., s], cur_v[..., s], decay[:, s], beta[:, s])

    # torch rms_norm (fp32 compute, one rounding) times a bf16 silu gate
    oc = oc.to(dtype).float()
    rstd = torch.rsqrt(oc.square().mean(-1, keepdim=True) + eps)
    y = (oc * rstd * norm_weight.float()).to(dtype)
    gate = torch.nn.functional.silu(z.view(batch, seq, heads, value_dim))
    return y * gate


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
