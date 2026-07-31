# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from comfy_kitchen._rope_utils import check_rope_inplace, trim_rope_freqs
from comfy_kitchen.registry import registry


def apply_rope1(x: torch.Tensor, freqs_cis: torch.Tensor):
    x_ = x.to(dtype=freqs_cis.dtype).reshape(*x.shape[:-1], -1, 1, 2)
    freqs_cis = trim_rope_freqs(x, freqs_cis)

    x_out = freqs_cis[..., 0] * x_[..., 0]
    x_out.addcmul_(freqs_cis[..., 1], x_[..., 1])
    return x_out.reshape(*x.shape).type_as(x)


def apply_rope(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    return apply_rope1(xq, freqs_cis), apply_rope1(xk, freqs_cis)


def apply_rope1_(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    check_rope_inplace(x, readonly=(freqs_cis,))
    x.copy_(apply_rope1(x, freqs_cis))
    return x


def apply_rope_(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    check_rope_inplace(xq, xk, readonly=(freqs_cis,))
    q_out, k_out = apply_rope(xq, xk, freqs_cis)
    xq.copy_(q_out)
    xk.copy_(k_out)
    return xq, xk


def apply_rope_split_half1(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    t_ = x.reshape(*x.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2).to(freqs_cis.dtype)
    t_out = freqs_cis[..., 0] * t_[..., 0] + freqs_cis[..., 1] * t_[..., 1]
    return t_out.movedim(-1, -2).reshape(*x.shape).type_as(x)


def apply_rope_split_half(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return apply_rope_split_half1(xq, freqs_cis), apply_rope_split_half1(xk, freqs_cis)


def apply_rope_split_half1_(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    check_rope_inplace(x, readonly=(freqs_cis,))
    x.copy_(apply_rope_split_half1(x, freqs_cis))
    return x


def apply_rope_split_half_(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    check_rope_inplace(xq, xk, readonly=(freqs_cis,))
    q_out, k_out = apply_rope_split_half(xq, xk, freqs_cis)
    xq.copy_(q_out)
    xk.copy_(k_out)
    return xq, xk


def _rms_rope1(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float,
    *,
    split_half: bool,
    rot_dim: int = 0,
) -> torch.Tensor:
    x_norm = torch.nn.functional.rms_norm(
        x,
        (x.shape[-1],),
        weight=scale,
        eps=epsilon,
    )
    if rot_dim and rot_dim != x.shape[-1]:
        # partial rotary: rotate the first rot_dim dims, pass the rest through
        rotated = x_norm[..., :rot_dim]
        if split_half:
            rotated = apply_rope_split_half1(rotated, freqs_cis)
        else:
            rotated = apply_rope1(rotated, freqs_cis)
        return torch.cat((rotated, x_norm[..., rot_dim:]), dim=-1)
    if split_half:
        return apply_rope_split_half1(x_norm, freqs_cis)
    return apply_rope1(x_norm, freqs_cis)


def rms_rope1(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    return _rms_rope1(x, freqs_cis, scale, epsilon, split_half=False)


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
        _rms_rope1(q, freqs_cis, q_scale, epsilon, split_half=False),
        _rms_rope1(k, freqs_cis, k_scale, epsilon, split_half=False),
    )


def rms_rope_split_half1(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    return _rms_rope1(x, freqs_cis, scale, epsilon, split_half=True)


def rms_rope_split_half(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor | None = None,
    epsilon: float = 1e-6,
    rot_dim: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if k_scale is None:
        k_scale = q_scale
    return (
        _rms_rope1(q, freqs_cis, q_scale, epsilon, split_half=True, rot_dim=rot_dim),
        _rms_rope1(k, freqs_cis, k_scale, epsilon, split_half=True, rot_dim=rot_dim),
    )


def rms_rope1_(
    x: torch.Tensor, freqs_cis: torch.Tensor, scale: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    check_rope_inplace(x, readonly=(freqs_cis, scale))
    x.copy_(rms_rope1(x, freqs_cis, scale, epsilon))
    return x


def rms_rope_(
    q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor,
    q_scale: torch.Tensor, k_scale: torch.Tensor | None = None,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    if k_scale is None:
        k_scale = q_scale
    check_rope_inplace(q, k, readonly=(freqs_cis, q_scale, k_scale))
    q_out, k_out = rms_rope(q, k, freqs_cis, q_scale, k_scale, epsilon)
    q.copy_(q_out)
    k.copy_(k_out)
    return q, k


def rms_rope_split_half1_(
    x: torch.Tensor, freqs_cis: torch.Tensor, scale: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    check_rope_inplace(x, readonly=(freqs_cis, scale))
    x.copy_(rms_rope_split_half1(x, freqs_cis, scale, epsilon))
    return x


def rms_rope_split_half_(
    q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor,
    q_scale: torch.Tensor, k_scale: torch.Tensor | None = None,
    epsilon: float = 1e-6, rot_dim: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if k_scale is None:
        k_scale = q_scale
    check_rope_inplace(q, k, readonly=(freqs_cis, q_scale, k_scale))
    q_out, k_out = rms_rope_split_half(q, k, freqs_cis, q_scale, k_scale, epsilon, rot_dim=rot_dim)
    q.copy_(q_out)
    k.copy_(k_out)
    return q, k


# =============================================================================
# torch.library Custom Op Definitions
# =============================================================================


@torch.library.custom_op("comfy_kitchen::apply_rope", mutates_args=())
def _op_apply_rope(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    kwargs = {"xq": xq, "xk": xk, "freqs_cis": freqs_cis}
    impl = registry.get_implementation("apply_rope", kwargs=kwargs)
    return impl(**kwargs)


@_op_apply_rope.register_fake
def _op_apply_rope_fake(xq, xk, freqs_cis):
    return torch.empty_like(xq), torch.empty_like(xk)


@torch.library.custom_op("comfy_kitchen::rms_rope", mutates_args=())
def _op_rms_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor | None = None,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    kwargs = {
        "q": q,
        "k": k,
        "freqs_cis": freqs_cis,
        "q_scale": q_scale,
        "k_scale": k_scale,
        "epsilon": epsilon,
    }
    impl = registry.get_implementation("rms_rope", kwargs=kwargs)
    return impl(**kwargs)


@_op_rms_rope.register_fake
def _op_rms_rope_fake(q, k, freqs_cis, q_scale, k_scale=None, epsilon=1e-6):
    return torch.empty_like(q), torch.empty_like(k)


@torch.library.custom_op("comfy_kitchen::rms_rope1", mutates_args=())
def _op_rms_rope1(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    kwargs = {
        "x": x,
        "freqs_cis": freqs_cis,
        "scale": scale,
        "epsilon": epsilon,
    }
    impl = registry.get_implementation("rms_rope1", kwargs=kwargs)
    return impl(**kwargs)


@_op_rms_rope1.register_fake
def _op_rms_rope1_fake(x, freqs_cis, scale, epsilon=1e-6):
    return torch.empty_like(x)


@torch.library.custom_op("comfy_kitchen::rms_rope_split_half", mutates_args=())
def _op_rms_rope_split_half(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor | None = None,
    epsilon: float = 1e-6,
    rot_dim: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    kwargs = {
        "q": q,
        "k": k,
        "freqs_cis": freqs_cis,
        "q_scale": q_scale,
        "k_scale": k_scale,
        "epsilon": epsilon,
        "rot_dim": rot_dim,
    }
    impl = registry.get_implementation("rms_rope_split_half", kwargs=kwargs)
    return impl(**kwargs)


@_op_rms_rope_split_half.register_fake
def _op_rms_rope_split_half_fake(
    q, k, freqs_cis, q_scale, k_scale=None, epsilon=1e-6, rot_dim=0
):
    return torch.empty_like(q), torch.empty_like(k)


@torch.library.custom_op("comfy_kitchen::rms_rope_split_half1", mutates_args=())
def _op_rms_rope_split_half1(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    kwargs = {
        "x": x,
        "freqs_cis": freqs_cis,
        "scale": scale,
        "epsilon": epsilon,
    }
    impl = registry.get_implementation("rms_rope_split_half1", kwargs=kwargs)
    return impl(**kwargs)


@_op_rms_rope_split_half1.register_fake
def _op_rms_rope_split_half1_fake(x, freqs_cis, scale, epsilon=1e-6):
    return torch.empty_like(x)


@torch.library.custom_op("comfy_kitchen::apply_rope_", mutates_args={"xq", "xk"})
def _op_apply_rope_(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> None:
    kwargs = {"xq": xq, "xk": xk, "freqs_cis": freqs_cis}
    registry.get_implementation("apply_rope_", kwargs=kwargs)(**kwargs)


@_op_apply_rope_.register_fake
def _op_apply_rope_fake_(xq, xk, freqs_cis):
    return None


@torch.library.custom_op("comfy_kitchen::apply_rope1_", mutates_args={"x"})
def _op_apply_rope1_(x: torch.Tensor, freqs_cis: torch.Tensor) -> None:
    kwargs = {"x": x, "freqs_cis": freqs_cis}
    registry.get_implementation("apply_rope1_", kwargs=kwargs)(**kwargs)


@_op_apply_rope1_.register_fake
def _op_apply_rope1_fake_(x, freqs_cis):
    return None


@torch.library.custom_op(
    "comfy_kitchen::apply_rope_split_half_", mutates_args={"xq", "xk"}
)
def _op_apply_rope_split_half_(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> None:
    kwargs = {"xq": xq, "xk": xk, "freqs_cis": freqs_cis}
    registry.get_implementation("apply_rope_split_half_", kwargs=kwargs)(**kwargs)


@_op_apply_rope_split_half_.register_fake
def _op_apply_rope_split_half_fake_(xq, xk, freqs_cis):
    return None


@torch.library.custom_op(
    "comfy_kitchen::apply_rope_split_half1_", mutates_args={"x"}
)
def _op_apply_rope_split_half1_(x: torch.Tensor, freqs_cis: torch.Tensor) -> None:
    kwargs = {"x": x, "freqs_cis": freqs_cis}
    registry.get_implementation("apply_rope_split_half1_", kwargs=kwargs)(**kwargs)


@_op_apply_rope_split_half1_.register_fake
def _op_apply_rope_split_half1_fake_(x, freqs_cis):
    return None


@torch.library.custom_op("comfy_kitchen::rms_rope_", mutates_args={"q", "k"})
def _op_rms_rope_(
    q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor,
    q_scale: torch.Tensor, k_scale: torch.Tensor | None = None,
    epsilon: float = 1e-6,
) -> None:
    kwargs = {
        "q": q, "k": k, "freqs_cis": freqs_cis, "q_scale": q_scale,
        "k_scale": k_scale, "epsilon": epsilon,
    }
    registry.get_implementation("rms_rope_", kwargs=kwargs)(**kwargs)


@_op_rms_rope_.register_fake
def _op_rms_rope_fake_(q, k, freqs_cis, q_scale, k_scale=None, epsilon=1e-6):
    return None


@torch.library.custom_op("comfy_kitchen::rms_rope1_", mutates_args={"x"})
def _op_rms_rope1_(
    x: torch.Tensor, freqs_cis: torch.Tensor, scale: torch.Tensor,
    epsilon: float = 1e-6,
) -> None:
    kwargs = {"x": x, "freqs_cis": freqs_cis, "scale": scale, "epsilon": epsilon}
    registry.get_implementation("rms_rope1_", kwargs=kwargs)(**kwargs)


@_op_rms_rope1_.register_fake
def _op_rms_rope1_fake_(x, freqs_cis, scale, epsilon=1e-6):
    return None


@torch.library.custom_op(
    "comfy_kitchen::rms_rope_split_half_", mutates_args={"q", "k"}
)
def _op_rms_rope_split_half_(
    q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor,
    q_scale: torch.Tensor, k_scale: torch.Tensor | None = None,
    epsilon: float = 1e-6, rot_dim: int = 0,
) -> None:
    kwargs = {
        "q": q, "k": k, "freqs_cis": freqs_cis, "q_scale": q_scale,
        "k_scale": k_scale, "epsilon": epsilon, "rot_dim": rot_dim,
    }
    registry.get_implementation("rms_rope_split_half_", kwargs=kwargs)(**kwargs)


@_op_rms_rope_split_half_.register_fake
def _op_rms_rope_split_half_fake_(
    q, k, freqs_cis, q_scale, k_scale=None, epsilon=1e-6, rot_dim=0
):
    return None


@torch.library.custom_op(
    "comfy_kitchen::rms_rope_split_half1_", mutates_args={"x"}
)
def _op_rms_rope_split_half1_(
    x: torch.Tensor, freqs_cis: torch.Tensor, scale: torch.Tensor,
    epsilon: float = 1e-6,
) -> None:
    kwargs = {"x": x, "freqs_cis": freqs_cis, "scale": scale, "epsilon": epsilon}
    registry.get_implementation("rms_rope_split_half1_", kwargs=kwargs)(**kwargs)


@_op_rms_rope_split_half1_.register_fake
def _op_rms_rope_split_half1_fake_(x, freqs_cis, scale, epsilon=1e-6):
    return None



@torch.library.custom_op("comfy_kitchen::apply_rope1", mutates_args=())
def _op_apply_rope1(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    kwargs = {"x": x, "freqs_cis": freqs_cis}
    impl = registry.get_implementation("apply_rope1", kwargs=kwargs)
    return impl(**kwargs)


@_op_apply_rope1.register_fake
def _op_apply_rope1_fake(x, freqs_cis):
    return torch.empty_like(x)


@torch.library.custom_op("comfy_kitchen::apply_rope_split_half", mutates_args=())
def _op_apply_rope_split_half(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    kwargs = {"xq": xq, "xk": xk, "freqs_cis": freqs_cis}
    impl = registry.get_implementation("apply_rope_split_half", kwargs=kwargs)
    return impl(**kwargs)


@_op_apply_rope_split_half.register_fake
def _op_apply_rope_split_half_fake(xq, xk, freqs_cis):
    return torch.empty_like(xq), torch.empty_like(xk)


@torch.library.custom_op("comfy_kitchen::apply_rope_split_half1", mutates_args=())
def _op_apply_rope_split_half1(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    kwargs = {"x": x, "freqs_cis": freqs_cis}
    impl = registry.get_implementation("apply_rope_split_half1", kwargs=kwargs)
    return impl(**kwargs)


@_op_apply_rope_split_half1.register_fake
def _op_apply_rope_split_half1_fake(x, freqs_cis):
    return torch.empty_like(x)
