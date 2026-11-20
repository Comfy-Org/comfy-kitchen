# ruff: noqa: E741, N806, N812  -- ComfyUI's own names (F, I, I_w), kept so the copy reads against the source.
"""ComfyUI's own arithmetic, copied from comfy/weight_adapter for the linear case.

Bypass: comfy/weight_adapter/bypass.py runs g(f(x) + h(x, f(x))); base is f(x).
Each function below is the matching h()/g()/calculate_weight()/__call__ body
with the module attributes turned into arguments. calculate_weight's
`weight += ...` is written out of place: ModelPatcher hands it a fresh copy,
and autograd cannot differentiate an in-place add into a leaf.
"""

import torch
from torch.nn import functional as F


def lora_bypass(x, up, down, base, scale):
    """LoRAAdapter.h() (lora.py), then bypass g(base + h) with g the identity."""
    up = up.to(dtype=x.dtype)
    down = down.to(dtype=x.dtype)
    hidden = F.linear(x, down)
    out = F.linear(hidden, up)
    return base + out * scale


def lora_merge(weight, up, down, scale):
    """LoRAAdapter.calculate_weight(), intermediate_dtype=float32, no DoRA."""
    mat1 = up.to(torch.float32)
    mat2 = down.to(torch.float32)
    lora_diff = torch.mm(mat1.flatten(start_dim=1), mat2.flatten(start_dim=1)).reshape(
        weight.shape
    )
    return weight + (scale * lora_diff).type(weight.dtype)


def loha_bypass(x, w1a, w1b, w2a, w2b, base, scale):
    """LoHaAdapter.h() (loha.py), linear, no Tucker."""
    w1a = w1a.to(dtype=x.dtype)
    w1b = w1b.to(dtype=x.dtype)
    w2a = w2a.to(dtype=x.dtype)
    w2b = w2b.to(dtype=x.dtype)
    m1 = w1a @ w1b
    m2 = w2a @ w2b
    diff_weight = (m1 * m2) * scale
    return base + F.linear(x, diff_weight)


def loha_merge(weight, w1a, w1b, w2a, w2b, scale):
    """LoHaAdapter.calculate_weight(), intermediate_dtype=float32, no Tucker, no DoRA."""
    m1 = torch.mm(w1a.to(torch.float32), w1b.to(torch.float32))
    m2 = torch.mm(w2a.to(torch.float32), w2b.to(torch.float32))
    lora_diff = (m1 * m2).reshape(weight.shape)
    return weight + (scale * lora_diff).type(weight.dtype)


def lokr_bypass(x, w1, w2, base, scale):
    """LoKrAdapter.h() (lokr.py) linear branch; the scale is the multiplier."""
    uq = w1.size(1)
    h_in_group = x.reshape(*x.shape[:-1], uq, -1)
    hb = F.linear(h_in_group, w2)
    h_cross = hb.transpose(-1, -2)
    hc = F.linear(h_cross, w1)
    hc = hc.transpose(-1, -2)
    out = hc.reshape(*hc.shape[:-2], -1)
    return base + out * scale


def lokr_merge(weight, w1, w2, scale):
    """LoKrAdapter.calculate_weight(), intermediate_dtype=float32, no DoRA."""
    w1 = w1.to(torch.float32)
    w2 = w2.to(torch.float32)
    lora_diff = torch.kron(w1, w2).reshape(weight.shape)
    return weight + (scale * lora_diff).type(weight.dtype)


def _orthogonal(blocks, constraint, dtype):
    """OFTDiff._get_orthogonal_matrix(): R in the activations' dtype, inverse in fp32."""
    blocks = blocks.to(dtype=dtype)
    block_size = blocks.shape[1]
    I = torch.eye(block_size, device=blocks.device, dtype=dtype)
    q = blocks - blocks.transpose(1, 2)
    normed_q = q
    if constraint:
        q_norm = torch.norm(q) + 1e-8
        if q_norm > constraint:
            normed_q = q * constraint / q_norm
    r = (I + normed_q) @ (I - normed_q).float().inverse()
    return r.to(dtype)


def _orthogonal_fixed(blocks, constraint, dtype):
    """_orthogonal with the one change it needs to run in bf16/fp16: both factors
    of the product in fp32 (as shipped, a bf16 @ fp32 matmul raises)."""
    blocks = blocks.to(dtype=dtype)
    block_size = blocks.shape[1]
    I = torch.eye(block_size, device=blocks.device, dtype=dtype)
    q = blocks - blocks.transpose(1, 2)
    normed_q = q
    if constraint:
        q_norm = torch.norm(q) + 1e-8
        if q_norm > constraint:
            normed_q = q * constraint / q_norm
    r = (I + normed_q).float() @ (I - normed_q).float().inverse()
    return r.to(dtype)


def oft_act(y, blocks, rescale, multiplier, constraint, conv):
    """OFTDiff.g() (oft.py): the rotation applied to the base output."""
    return _oft_apply(
        _orthogonal(blocks, constraint, y.dtype), y, rescale, multiplier, conv
    )


def oft_act_fixed(y, blocks, rescale, multiplier, constraint, conv):
    """OFTDiff.g() with _orthogonal_fixed: ComfyUI's arithmetic, made to run."""
    return _oft_apply(
        _orthogonal_fixed(blocks, constraint, y.dtype), y, rescale, multiplier, conv
    )


def _oft_apply(r, y, rescale, multiplier, conv):
    """The rest of OFTDiff.g() after the rotation."""
    block_num, block_size, _ = r.shape
    I = torch.eye(block_size, device=y.device, dtype=y.dtype)
    r = r * multiplier + (1 - multiplier) * I
    if conv:
        y = y.transpose(1, -1)
    *batch_shape, out_features = y.shape
    y_blocked = y.reshape(*batch_shape, block_num, block_size)
    out_blocked = torch.einsum("k n m, ... k n -> ... k m", r, y_blocked)
    out = out_blocked.reshape(*batch_shape, out_features)
    if rescale is not None:
        rescale = rescale.to(device=y.device, dtype=y.dtype)
        out = out * rescale.view(-1)
    if conv:
        out = out.transpose(1, -1)
    return out


def oft_weight(weight, blocks, strength, constraint):
    """OFTAdapter.calculate_weight() (oft.py), intermediate_dtype=float32, no DoRA."""
    blocks = blocks.to(torch.float32)
    block_num, block_size, *_ = blocks.shape
    I = torch.eye(block_size, device=blocks.device, dtype=blocks.dtype)
    q = blocks - blocks.transpose(1, 2)
    normed_q = q
    if constraint > 0:
        q_norm = torch.norm(q) + 1e-8
        if q_norm > constraint:
            normed_q = q * constraint / q_norm
    r = (I + normed_q) @ (I - normed_q).float().inverse()
    r = r.to(weight)
    I_w = torch.eye(block_size, device=weight.device, dtype=weight.dtype)
    _, *shape = weight.shape
    lora_diff = torch.einsum(
        "k n m, k n ... -> k m ...",
        (r * strength) - strength * I_w,
        weight.view(block_num, block_size, *shape),
    ).view(-1, *shape)
    return weight + lora_diff.type(weight.dtype)


def weight_decompose(weight, lora_diff, dora_scale, strength):
    """comfy.weight_adapter.base.weight_decompose(), output-axis branch, alpha 1,
    intermediate_dtype float32, out of place."""
    dora_scale = dora_scale.to(torch.float32)
    weight_calc = weight + lora_diff.type(weight.dtype)
    weight_norm = (
        weight.reshape(weight.shape[0], -1)
        .norm(dim=1, keepdim=True)
        .reshape(weight.shape[0], *[1] * (weight.dim() - 1))
    )
    weight_norm = weight_norm + torch.finfo(weight.dtype).eps
    weight_calc = weight_calc * (dora_scale / weight_norm).type(weight.dtype)
    if strength != 1.0:
        return weight + strength * (weight_calc - weight)
    return weight_calc


def channel_scale(x, w, alpha, gamma):
    """IA3: ComfyUI has no IA3 adapter; plain torch broadcast over the last axis."""
    return x * (alpha + gamma * w)
