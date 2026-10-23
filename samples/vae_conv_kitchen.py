"""Route the VAE decoder's convolutions to comfy-kitchen's fp16 conv3d.

The VAE decode is a third of an SDXL run (3214 ms of 9462 ms at 1024x1024,
8 steps) and 87.7% of that is convolution, on torch's conv2d -- which is MIOpen
here, and MIOpen has no optimized gfx1103 kernel for the grouped case. The
kitchen's own conv3d is 1.7x-2.8x faster on the shapes the SDXL decoder runs
(measured on the 6-WGP 780M, fp16, per call):

    x(1, 512, 128, 128) w(512, 512, 3, 3)    26.3 ms ->  9.4 ms
    x(1, 512, 256, 256) w(512, 512, 3, 3)    98.1 ms -> 39.4 ms
    x(1, 512, 512, 512) w(512, 512, 3, 3)   394.1 ms -> 180.2 ms
    x(1, 256, 1024, 1024) w(256, 256, 3, 3)  249.1 ms -> 148.6 ms
    x(1, 128, 1024, 1024) w(128, 128, 3, 3)  100.0 ms -> 38.6 ms

The last convolution of the decoder (128 channels to 3) is the exception: the
kernel's tile is 128 wide, so three output channels waste it, and it is 4x
slower there. The 1x1 convolutions of the mid block's attention are a wash and
stay on MIOpen.

The kitchen's kernel is a 3-D fp16 NDHWC conv. A 2-D convolution is the same
thing with a depth of one. The kernel reads the spatial border as zeros
(`zero_border=True`), so the input needs one NCHW into NDHWC copy and the
output one transpose back -- and that is what the remaining layout cost is.
Earlier the caller padded, which meant copying the whole activation into a
zero-filled buffer in addition to the layout convert; on these shapes the copy
landed in the scattered direction and the whole hook broke even.

Same-workflow A/B on the 6-WGP 780M, SDXL at 1024x1024, the VAE in fp16, 8
steps (2026-10): the decode's convolutions 1989 -> 1846 ms, the whole decode
2404 -> 2287 ms, the rest of the pipeline unchanged. The raw kernel advantage
is larger (1.7x-2.8x per call above) and the difference is the two layout
copies around each conv. The Anima/qwen VAE has only three eligible
convolutions and stays at par.

The VAE must be in fp16 for this (the kitchen's conv is fp16); ComfyUI's
--fp16-vae flag does that. In bf16 the convolutions stay on MIOpen.

Usage, from a ComfyUI custom node or an import-time hook::

    from vae_conv_kitchen import patch_vae_decode

    patch_vae_decode()
"""
from __future__ import annotations

import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.sd
import comfy_kitchen

logger = logging.getLogger(__name__)

CL3D = torch.channels_last_3d


def _kitchen_backend():
    """The HIP backend, for the transpose the output conversion needs."""
    from comfy_kitchen.backends import hip as hip_backend

    return hip_backend


class KitchenConv2d(nn.Module):
    """A 2-D convolution that runs on the kitchen's fp16 conv3d.

    The weight is converted once and kept; the activations change layout
    around the call, the input into channels_last_3d and padded with zeros
    (the kernel does no padding of its own), the output back to NCHW.
    """

    def __init__(self, conv: nn.Conv2d) -> None:
        super().__init__()
        self.conv = conv
        self._weight = None

    def _kitchen_weight(self):
        if self._weight is None:
            k, c, r, s = self.conv.weight.shape
            self._weight = self.conv.weight.view(k, c, 1, r, s).contiguous(memory_format=CL3D)
        return self._weight

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:  # noqa: ANN401
        conv = self.conv
        k, c, r, s = conv.weight.shape
        # The kernel reads the zero border itself, so the input is only converted
        # into channels_last_3d, not copied into a padded buffer.
        x_ndhwc = x.unsqueeze(2).contiguous(memory_format=CL3D)
        out = comfy_kitchen.fp16_conv3d(x_ndhwc, self._kitchen_weight(), conv.bias, None, 1,
                                        zero_border=True)
        return _kitchen_backend().ndhwc_to_nchw(out).squeeze(2)


def _eligible(conv: nn.Conv2d) -> bool:
    """Whether the kitchen's conv3d can serve this convolution.

    fp16 (the kernel is fp16 only), a 3x3 square kernel (the padding and the
    weight's view), dense, and an output channel count the kernel's 128-wide
    tile can fill.
    """
    w = conv.weight
    return (
        w.dtype == torch.float16
        and w.device.type == "cuda"
        and w.dim() == 4
        and w.shape[2] == 3 and w.shape[3] == 3
        and conv.groups == 1
        and conv.stride == (1, 1)
        and conv.dilation == (1, 1)
        and w.shape[0] % 8 == 0
        and w.shape[1] % 8 == 0
    )


def _patch_decoder(vae) -> int:
    """Replace the decoder's convolutions, once per VAE."""
    model = getattr(vae, "first_stage_model", None)
    decoder = getattr(model, "decoder", None)
    if decoder is None:
        if os.getenv("COMFY_KITCHEN_VAE_CONV_VERBOSE") == "1":
            print(f"  [kitchen conv] the VAE has no decoder: "
                  f"{[a for a in dir(vae) if 'stage' in a or 'model' in a]}")
        return 0
    if getattr(decoder, "_kitchen_conv_patched", False):
        return 0
    patched = 0
    for name, mod in list(decoder.named_modules()):
        if not isinstance(mod, nn.Conv2d):
            continue
        if not _eligible(mod) and os.getenv("COMFY_KITCHEN_VAE_CONV_VERBOSE") == "1" and patched == 0:
            w = mod.weight
            print(f"  [kitchen conv] {name} is not eligible: weight {tuple(w.shape)} "
                  f"{w.dtype} on {w.device}, groups {mod.groups}, stride {mod.stride}, "
                  f"dilation {mod.dilation}")
        if not _eligible(mod):
            continue
        parent = decoder
        parts = name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], KitchenConv2d(mod))
        patched += 1
    decoder._kitchen_conv_patched = True
    if patched:
        logger.info("kitchen conv: %d VAE convolutions", patched)
    return patched


def patch_vae_decode(verbose: bool = False) -> None:
    """Route the VAE's convolution-heavy decode through the kitchen's conv3d.

    The patch is on VAE.decode, so it runs when the VAE decodes and the
    decoder's convolutions are replaced the first time.
    """
    if getattr(comfy.sd.VAE, "_kitchen_conv_patched", False):
        return

    original = comfy.sd.VAE.decode

    def decode(self, samples_in, *args, **kwargs):  # noqa: ANN001 - ComfyUI's signature
        n = _patch_decoder(self)
        if verbose and n:
            print(f"  [kitchen conv] {n} VAE convolutions on the kitchen's conv3d")
        return original(self, samples_in, *args, **kwargs)

    comfy.sd.VAE.decode = decode
    comfy.sd.VAE._kitchen_conv_patched = True
    logger.info("kitchen conv: the VAE decode is routed through the kitchen's conv3d")


if os.getenv("COMFY_KITCHEN_VAE_CONV", "1") == "1":
    # Importing this module is the opt-in: a ComfyUI custom node is imported
    # at startup, and the patch is inert for a VAE that is not in fp16.
    patch_vae_decode(verbose=os.getenv("COMFY_KITCHEN_VAE_CONV_VERBOSE", "0") == "1")
