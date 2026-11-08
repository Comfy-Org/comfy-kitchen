# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Conv bias add + LeakyReLU, in place on the output of a conv run without its bias.

FILM frame interpolation (``comfy_extras/frame_interpolation_models/film_net.py``)
``FilmConv2d.forward`` with ``activation=True``:

    x = self.conv(x)          # Conv2d with bias -> F.conv2d(x, weight, bias, ...)
    x = self.activation(x)    # nn.LeakyReLU(0.2)

On CUDA (cuDNN) and ROCm (MIOpen) ``at::convolution`` runs the convolution without the bias
and then adds it in place, ``output.add_(bias.reshape(1, -1, 1, 1))``, before LeakyReLU runs as
a third kernel. With the conv called with ``bias=None``, this op is the rest of that chain on
the conv output ``x`` ``[N, C, H, W]``:

    x.add_(bias.reshape(1, -1, 1, 1))
    x = leaky_relu(x, negative_slope)

written back into ``x``. The add is rounded to x's dtype before the activation reads it, as in
the stock chain. The stock ``nn.LeakyReLU`` is out of place; the conv output is a fresh tensor
nothing else holds, so writing the activation into it gives the same values without the
extra buffer.
"""

import torch
from torch.nn import functional

from comfy_kitchen.registry import registry


def conv_bias_leaky_relu_(
    x: torch.Tensor, bias: torch.Tensor, negative_slope: float = 0.2
) -> torch.Tensor:
    """Reference: the bias ``add_`` of ``at::convolution`` and ``F.leaky_relu``, in ComfyUI's
    order, written into ``x``."""
    x.add_(bias.reshape((1, -1) + (1,) * (x.dim() - 2)))
    functional.leaky_relu(x, negative_slope, inplace=True)
    return x


def _check(x, bias):
    name = "conv_bias_leaky_relu_"
    if x.dim() < 2:
        raise ValueError(f"{name}: x must be [N, C, ...], got {tuple(x.shape)}")
    if bias.dim() != 1 or bias.shape[0] != x.shape[1]:
        raise ValueError(f"{name}: bias must be [C] = [{x.shape[1]}], got {tuple(bias.shape)}")
    if bias.device != x.device:
        raise ValueError(f"{name}: bias must be on x's device")


@torch.library.custom_op("comfy_kitchen::conv_bias_leaky_relu_", mutates_args={"x"})
def _op_conv_bias_leaky_relu_(x: torch.Tensor, bias: torch.Tensor, negative_slope: float) -> None:
    _check(x, bias)
    kwargs = {"x": x, "bias": bias, "negative_slope": negative_slope}
    registry.get_implementation("conv_bias_leaky_relu_", kwargs=kwargs)(**kwargs)


@_op_conv_bias_leaky_relu_.register_fake
def _op_conv_bias_leaky_relu_fake_(x, bias, negative_slope):
    _check(x, bias)
    return None
