# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
from torch import Tensor


def adaln(x: Tensor, scale: Tensor, shift: Tensor, eps: float = 1e-6) -> Tensor:
    """Fused AdaLN: layernorm(x, elementwise_affine=False) * (1 + scale) + shift"""
    normalized = F.layer_norm(x, x.shape[-1:], eps=eps)
    return normalized * (1 + scale) + shift


# =============================================================================
# torch.library Custom Op Definitions
# =============================================================================


@torch.library.custom_op("comfy_kitchen::adaln", mutates_args=())
def _op_adaln(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    from comfy_kitchen.registry import registry

    kwargs = {"x": x, "scale": scale, "shift": shift, "eps": eps}
    impl = registry.get_implementation("adaln", kwargs=kwargs)
    return impl(**kwargs)


@_op_adaln.register_fake
def _op_adaln_fake(x, scale, shift, eps):
    return torch.empty_like(x)
