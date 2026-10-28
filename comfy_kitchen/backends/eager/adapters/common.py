# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared pieces of the adapter custom ops: dispatch, dtype rules, empty grads."""

import torch
from torch import Tensor

from comfy_kitchen.registry import registry

_CHOICES: dict[tuple, object] = {}


def _signature(kwargs) -> tuple:
    """Dispatch cache key: each tensor's dtype, device and rank, every other argument's type."""
    return tuple(
        (k, v.dtype, v.device, v.dim()) if isinstance(v, Tensor) else (k, type(v))
        for k, v in kwargs.items()
    )


def dispatch(name: str, **kwargs):
    """Run the registry's backend for one adapter op, cached per argument signature and registry state."""
    key = (name, registry.selection_state(), _signature(kwargs))
    impl = _CHOICES.get(key)
    if impl is None:
        impl = registry.get_implementation(name, kwargs=kwargs)
        _CHOICES[key] = impl
    return impl(**kwargs)


def factor_dtype(*factors: Tensor) -> torch.dtype:
    """The dtype a weight rebuild runs in: the widest of its factors."""
    dtype = factors[0].dtype
    for factor in factors[1:]:
        dtype = torch.promote_types(dtype, factor.dtype)
    return dtype


def bypass_dtype(x: Tensor, base: Tensor | None) -> torch.dtype:
    """A bypass computes in x's dtype and returns base's when one is fused in."""
    return x.dtype if base is None else base.dtype


def add_base(y: Tensor, base: Tensor | None) -> Tensor:
    """base + y in base's dtype, or y without a base."""
    if base is None:
        return y
    return (base + y).to(base.dtype)


def absent(like: Tensor) -> Tensor:
    """The stand-in for an optional gradient the call had no input for."""
    return like.new_empty(0)


def present(t: Tensor) -> Tensor | None:
    """t, or None for an absent() stand-in."""
    return None if t.numel() == 0 else t
