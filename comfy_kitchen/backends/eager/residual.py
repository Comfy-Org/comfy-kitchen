# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn.functional as functional


def rms_gated_residual(
    activation: torch.Tensor,
    norm_weight: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor,
    eps: float = 1.0e-5,
) -> torch.Tensor:
    """Reference RMSNorm followed by the visible gate product and residual add."""
    normalized = functional.rms_norm(
        activation, (activation.shape[-1],), norm_weight, eps
    )
    return residual + gate * normalized
