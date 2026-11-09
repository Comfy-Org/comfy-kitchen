# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SnakeBeta activation (log-scale alpha / beta), optionally after a residual add.

The ``SnakeBeta`` of the Oobleck audio VAE (``comfy/ldm/audio/autoencoder.py``, BigVGAN's
activation with ``alpha_logscale=True``), for x ``[B, C, T]`` and the raw ``[C]`` parameters:

    a = exp(alpha)[None, :, None]
    b = exp(beta)[None, :, None]
    out = x + (1.0 / (b + 1e-9)) * sin(x * a) ** 2

In the decoder every ``ResidualUnit`` ends with ``x + res`` and that sum goes straight into the
next ``SnakeBeta`` (the next unit's first activation, the next block's activation or the final
one). ``residual`` covers that add: ``snake_beta(x, alpha, beta, residual)`` is
``SnakeBeta(x + residual)``, and ``snake_beta_with_sum`` also returns the sum, which the next
unit keeps as its own residual.
"""

import torch

from comfy_kitchen.registry import registry

_NO_DIV_BY_ZERO = 0.000000001  # SnakeBeta.no_div_by_zero


def _snake_beta_stock(x, alpha, beta):
    # autoencoder.py snake_beta
    return x + (1.0 / (beta + _NO_DIV_BY_ZERO)) * pow(torch.sin(x * alpha), 2)


def _snake_beta_forward(x, alpha, beta):
    # autoencoder.py SnakeBeta.forward with alpha_logscale=True
    alpha = alpha.unsqueeze(0).unsqueeze(-1).to(x.device)  # line up with x to [B, C, T]
    beta = beta.unsqueeze(0).unsqueeze(-1).to(x.device)
    alpha = torch.exp(alpha)
    beta = torch.exp(beta)
    x = _snake_beta_stock(x, alpha, beta)
    return x


def snake_beta(
    x: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference: ``ResidualUnit``'s ``x + res`` (when residual is given), then
    ``SnakeBeta.forward``, the stock ops in the stock order."""
    if residual is not None:
        x = x + residual
    return _snake_beta_forward(x, alpha, beta)


def snake_beta_with_sum(
    x: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference: ``s = x + residual`` (``ResidualUnit``), then ``SnakeBeta.forward(s)``;
    returns ``(s, out)``."""
    s = x + residual
    return s, _snake_beta_forward(s, alpha, beta)


def _check(x, alpha, beta, residual):
    name = "snake_beta"
    if x.dim() != 3:
        raise ValueError(f"{name}: x must be [B, C, T], got {tuple(x.shape)}")
    channels = x.shape[1]
    for pname, p in (("alpha", alpha), ("beta", beta)):
        if p.dim() != 1 or p.shape[0] != channels:
            raise ValueError(f"{name}: {pname} must be [C] = [{channels}], got {tuple(p.shape)}")
    if residual is not None:
        if residual.shape != x.shape:
            raise ValueError(
                f"{name}: residual must have x's shape {tuple(x.shape)}, got {tuple(residual.shape)}"
            )
        if residual.device != x.device:
            raise ValueError(f"{name}: residual must be on x's device")


def _sum_dtype(x, residual):
    return x.dtype if residual is None else torch.promote_types(x.dtype, residual.dtype)


def _out_dtype(s_dtype, alpha, beta):
    return torch.promote_types(torch.promote_types(s_dtype, alpha.dtype), beta.dtype)


@torch.library.custom_op("comfy_kitchen::snake_beta", mutates_args=())
def _op_snake_beta(
    x: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    _check(x, alpha, beta, residual)
    kwargs = {"x": x, "alpha": alpha, "beta": beta, "residual": residual}
    impl = registry.get_implementation("snake_beta", kwargs=kwargs)
    # Contiguous on every backend, like the fake below and the Triton output: eager keeps a
    # non-contiguous x's layout, which the fake would otherwise have to predict.
    return impl(**kwargs).contiguous()


@_op_snake_beta.register_fake
def _op_snake_beta_fake(x, alpha, beta, residual=None):
    _check(x, alpha, beta, residual)
    return x.new_empty(x.shape, dtype=_out_dtype(_sum_dtype(x, residual), alpha, beta))


@torch.library.custom_op("comfy_kitchen::snake_beta_with_sum", mutates_args=())
def _op_snake_beta_with_sum(
    x: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _check(x, alpha, beta, residual)
    kwargs = {"x": x, "alpha": alpha, "beta": beta, "residual": residual}
    impl = registry.get_implementation("snake_beta_with_sum", kwargs=kwargs)
    s, out = impl(**kwargs)
    return s.contiguous(), out.contiguous()  # see _op_snake_beta


@_op_snake_beta_with_sum.register_fake
def _op_snake_beta_with_sum_fake(x, alpha, beta, residual):
    _check(x, alpha, beta, residual)
    s_dtype = _sum_dtype(x, residual)
    return x.new_empty(x.shape, dtype=s_dtype), x.new_empty(
        x.shape, dtype=_out_dtype(s_dtype, alpha, beta)
    )
