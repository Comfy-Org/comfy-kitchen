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

from comfy_kitchen._rope_utils import _tensors_overlap
from comfy_kitchen.registry import registry


def conv_bias_leaky_relu_(
    x: torch.Tensor, bias: torch.Tensor, negative_slope: float = 0.2
) -> torch.Tensor:
    """Reference: the bias ``add_`` of ``at::convolution`` and ``F.leaky_relu``, in ComfyUI's
    order, written into ``x``."""
    x.add_(bias.view((-1,) + (1,) * (x.dim() - 2)))
    functional.leaky_relu_(x, negative_slope)
    return x


def _check(x, bias):
    name = "conv_bias_leaky_relu_"
    if x.dim() < 2:
        raise ValueError(f"{name}: x must be [N, C, ...], got {tuple(x.shape)}")
    if bias.dim() != 1 or bias.shape[0] != x.shape[1]:
        raise ValueError(f"{name}: bias must be [C] = [{x.shape[1]}], got {tuple(bias.shape)}")
    if bias.device != x.device:
        raise ValueError(f"{name}: bias must be on x's device")


def _byte_range(t):
    # [first, last + 1) byte address the view can touch
    lo = hi = t.data_ptr()
    size_of = t.element_size()
    for size, stride in zip(t.shape, t.stride(), strict=True):
        extent = (size - 1) * stride * size_of
        if extent < 0:
            lo += extent
        else:
            hi += extent
    return lo, hi + size_of


def _check_overlap(x, bias):
    # every backend rejects a bias that shares memory with x (eager's add_ raises on a partial
    # overlap, the Triton kernel would read bias values it has already overwritten). The common
    # case (bias outside the bytes of x's storage) only reads a few attributes. x itself needs no
    # check: the Triton call rule takes only dense x, and add_ raises on an x with internal
    # overlap.
    storage = x.untyped_storage()
    s_first, b_first = storage.data_ptr(), bias.data_ptr()
    if b_first >= s_first + storage.nbytes():
        return
    size_of = bias.element_size()  # bias is 1-D (checked)
    if b_first + (bias.shape[0] - 1) * bias.stride(0) * size_of + size_of <= s_first:
        return
    if x.numel() == 0 or bias.numel() == 0:
        return
    x_first, x_end = _byte_range(x)
    b_first, b_end = _byte_range(bias)
    if b_end <= x_first or b_first >= x_end:
        return
    # the byte ranges meet. _tensors_overlap compares element offsets and returns False for
    # different storages (e.g. two torch.frombuffer wrappers of one buffer), so mixed element
    # sizes and different storages are treated as overlapping
    if (
        bias.element_size() != x.element_size()
        or bias.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
        or _tensors_overlap(bias, x)
    ):
        raise ValueError("conv_bias_leaky_relu_: bias must not overlap x")


def check_conv_bias_leaky_relu_args(x, bias):
    """The checks every backend's call goes through (ValueError on bad shapes, devices or a bias
    that overlaps x)."""
    _check(x, bias)
    _check_overlap(x, bias)


@torch.library.custom_op("comfy_kitchen::conv_bias_leaky_relu_", mutates_args={"x"})
def _op_conv_bias_leaky_relu_(x: torch.Tensor, bias: torch.Tensor, negative_slope: float) -> None:
    check_conv_bias_leaky_relu_args(x, bias)
    kwargs = {"x": x, "bias": bias, "negative_slope": negative_slope}
    registry.get_implementation("conv_bias_leaky_relu_", kwargs=kwargs)(**kwargs)


@_op_conv_bias_leaky_relu_.register_fake
def _op_conv_bias_leaky_relu_fake_(x, bias, negative_slope):
    _check(x, bias)
    return None
