# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch


def apply_rope1(x: torch.Tensor, freqs_cis: torch.Tensor):
    x_ = x.to(dtype=freqs_cis.dtype).reshape(*x.shape[:-1], -1, 1, 2)
    x_out = freqs_cis[..., 0] * x_[..., 0]
    x_out.addcmul_(freqs_cis[..., 1], x_[..., 1])
    return x_out.reshape(*x.shape).type_as(x)


def apply_rope(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    return apply_rope1(xq, freqs_cis), apply_rope1(xk, freqs_cis)


# =============================================================================
# torch.library Custom Op Definitions
# =============================================================================


@torch.library.custom_op("comfy_kitchen::apply_rope", mutates_args=())
def _op_apply_rope(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from comfy_kitchen.registry import registry

    kwargs = {"xq": xq, "xk": xk, "freqs_cis": freqs_cis}
    impl = registry.get_implementation("apply_rope", kwargs=kwargs)
    return impl(**kwargs)


@_op_apply_rope.register_fake
def _op_apply_rope_fake(xq, xk, freqs_cis):
    return torch.empty_like(xq), torch.empty_like(xk)


@torch.library.custom_op("comfy_kitchen::apply_rope1", mutates_args=())
def _op_apply_rope1(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    from comfy_kitchen.registry import registry

    kwargs = {"x": x, "freqs_cis": freqs_cis}
    impl = registry.get_implementation("apply_rope1", kwargs=kwargs)
    return impl(**kwargs)


@_op_apply_rope1.register_fake
def _op_apply_rope1_fake(x, freqs_cis):
    return torch.empty_like(x)
