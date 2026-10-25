"""Route the VAE decoder's convolutions to comfy-kitchen's fp16 conv3d.

The VAE decode is ~15% of an SDXL run at 1024x1024 and ~85% of that is
convolution, on torch's conv2d -- which is MIOpen here. The kitchen's conv3d is a 3-D fp16
NDHWC conv (a 2-D conv is the same thing with a depth of one) that reads the
spatial border as zeros (`zero_border=True`), so the route pays one NCHW into
NDHWC copy, the kernel, and one transpose back. Those copies are fixed work
per element, while the kernel's advantage grows with the shape's FLOPs, so the
route only wins on large shapes. Measured on the 6-WGP 780M, fp16, per call
(2026-10, current driver):

    C K spatial        MIOpen NCHW  route    (MIOpen / route)
    512 512 512^2        372.7 ms   244.4    1.52x
    256 256 1024^2       255.9 ms   214.6    1.19x
    512 256 512^2        155.4 ms   124.3    1.25x
    256 128 1024^2       171.8 ms   117.9    1.46x
    256 256 768^2        140.0 ms   120.8    1.16x
    512 512 384^2        100.9 ms   126.2    0.80x   (below the gate)
    512 512 256^2         26.8 ms    57.4    0.47x   (below the gate)
    128 128 1024^2        33.5 ms    66.1    0.51x   (below the gate)

The two losses at 6.9e11 FLOPs against the win at the same 6.9e11 split on
the output width: MIOpen's 512-wide output tile runs at 6.9 TFLOPS on this
part and its 256-wide one at 5.0, while the kitchen kernel is steady at
7-8 TFLOPS. The wrapper therefore checks the FLOPs per call and hands the
small ones back to MIOpen: route past 1.2e12 FLOPs, and in the 6e11-1.2e12
band only for outputs of 256 channels or fewer. That makes the outcome
resolution-independent: at 1024x1024 the 512-channel and 1024-spatial stages
route, at 768x768 only the 256-channel 768-spatial one, and at 512x512 and
below nearly everything is below the gate. Earlier measurements with an older
driver had MIOpen slower on the same shapes and made the case for routing
them; the new table replaces that expectation.

The last convolution of the decoder (128 channels to 3) and the mid block's
1x1 convolutions never qualify and stay on MIOpen.

The diffusion model's convolutions are not in scope, and not for lack of trying:
on their own shapes (batch 2, 320-1280 channels at 128^2-32^2) MIOpen runs at
10-12 TFLOPS while the kernel reaches 6.4-8.8, so the route measured 0.49-0.87x
there even before its two layout copies.

The kernel is fp16-only. fp16 VAEs run as is. bf16 VAEs cast into fp16 fused
into the two layout copies (`copy_` converts dtype while it lays out, so the
NCHW into NDHWC convert takes the cast along, and the output casts back
during the transpose-out), and the cast back is a third pass the fp16 case
does not have. It works and is exact for in-range values -- the route's error
against an fp32 reference is smaller than MIOpen's own bf16 conv -- but it
eats the margin: 1.05x on the decode at 1024x1024 against 1.41x for a native
fp16 VAE, and ~0.96x at 768x768, so bf16 stays on MIOpen unless
COMFY_KITCHEN_VAE_CONV_BF16=1 is set.

Same-workflow A/B on the 6-WGP 780M, SDXL, the VAE in fp16, 8 steps
(2026-10, two runs a side): at 1024x1024 the decode goes 2515 -> 2217 ms
(its convolutions 2095 -> 1821), so 1.13x; at 768x768 it moves 912 -> 972 ms,
inside the pipeline's 7% run-to-run band and not a gain, though paired
in-process on the decode alone it is 1.01x. Below 768x768 nothing routes.

Usage, from a ComfyUI custom node or an import-time hook::

    from vae_conv_kitchen import patch_vae_decode

    patch_vae_decode()

The patch is on VAE.decode: every fp16/bf16 VAE that decodes gets its
eligible convolutions wrapped once. Without this module (or with
COMFY_KITCHEN_VAE_CONV=0) nothing changes.
"""

from __future__ import annotations

import logging
import os

import torch
import torch.nn as nn

import comfy.sd
import comfy_kitchen

logger = logging.getLogger(__name__)

CL3D = torch.channels_last_3d

# 2*9*C*K*H*W per call. Two thresholds: past the full one the kernel's
# throughput wins outright; in the band between them MIOpen is only slow for
# outputs at most 256 channels wide (its 512-wide tile is efficient), so those
# and only those still route. Every row of the table above is a measured
# shape; the band rules out (512,512)@384^2 (0.80x) and keeps
# (512,256)@512^2 (1.25x), (256,256)@768^2 (1.16x), (256,128)@1024^2 (1.46x).
ROUTE_MIN_FLOPS = 1.2e12
ROUTE_BAND_FLOPS = 6e11
ROUTE_BAND_MAX_K = 256

# The kernel wants fp16; bf16 weights cast into fp16 exactly.
_ROUTE_DTYPES = (torch.float16, torch.bfloat16)

# bf16 VAEs route only on request: the cast into and out of fp16 costs the
# kernel's margin (measured 1.05x on the decode at 1024^2 against 1.41x for a
# native fp16 VAE, and no gain at 768^2), so the default leaves bf16 on MIOpen.
_BF16_ROUTE = os.getenv("COMFY_KITCHEN_VAE_CONV_BF16", "0") == "1"


def _kitchen_backend():
    """The HIP backend, for the transpose the output conversion needs."""
    from comfy_kitchen.backends import hip as hip_backend

    return hip_backend


class KitchenConv2d(nn.Module):
    """A 2-D convolution that runs on the kitchen's fp16 conv3d.

    The weight is converted once and kept; the activations change layout
    around the call, the input into channels_last_3d (the kernel reads the
    zero border itself), the output back to NCHW. Shapes below the FLOPs gate
    and calls whose dtype the kernel does not serve fall through to MIOpen.
    """

    def __init__(self, conv: nn.Conv2d) -> None:
        super().__init__()
        self.conv = conv
        self._weight = None
        self._bias = None
        self._logged = False

    def _kitchen_weight(self):
        if self._weight is None:
            w = self.conv.weight
            self._bias = None if self.conv.bias is None else self.conv.bias.to(torch.float16)
            self._weight = (
                w.to(torch.float16)
                .view(w.shape[0], w.shape[1], 1, w.shape[2], w.shape[3])
                .contiguous(memory_format=CL3D)
            )
        return self._weight

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        conv = self.conv
        if x.dim() != 4 or x.dtype not in _ROUTE_DTYPES or x.dtype != conv.weight.dtype:
            self._log_first(x, None, "dtype")
            return conv(x, *args, **kwargs)
        n, c, h, w = x.shape
        k = conv.weight.shape[0]
        flops = 2 * 9 * c * k * n * h * w
        if flops < ROUTE_BAND_FLOPS or (flops < ROUTE_MIN_FLOPS and k > ROUTE_BAND_MAX_K):
            self._log_first(x, "below gate", "size")
            return conv(x, *args, **kwargs)
        w3, b3 = self._kitchen_weight(), self._bias
        backend = _kitchen_backend()
        if x.dtype == torch.float16:
            self._log_first(x, None, "routed")
            x_ndhwc = x.unsqueeze(2).contiguous(memory_format=CL3D)
            out = comfy_kitchen.fp16_conv3d(x_ndhwc, w3, b3, None, 1, zero_border=True)
            return backend.ndhwc_to_nchw(out).squeeze(2)
        self._log_first(x, None, "routed bf16")
        # bf16: cast fused into the layout convert in, and back out in the
        # transpose (copy_ converts dtype while it moves).
        x_fp16 = torch.empty(
            (n, c, 1, h, w), dtype=torch.float16, device=x.device, memory_format=CL3D
        )
        x_fp16.copy_(x.unsqueeze(2))
        out = comfy_kitchen.fp16_conv3d(x_fp16, w3, b3, None, 1, zero_border=True)
        o = backend.ndhwc_to_nchw(out)
        y = torch.empty((n, k, h, w), dtype=torch.bfloat16, device=x.device)
        y.copy_(o.squeeze(2))
        return y

    def _log_first(self, x: torch.Tensor, note: str | None, decision: str) -> None:
        """A verbose log of the first call's routing decision for this conv."""
        if self._logged or os.getenv("COMFY_KITCHEN_VAE_CONV_VERBOSE") != "1":
            return
        w = self.conv.weight
        extra = f" ({note})" if note else ""
        print(
            f"  [kitchen conv] {w.shape[0]:4d}<-{x.shape[1]:4d} "
            f"+"
            f"{tuple(x.shape)[2:]} {x.dtype}, {decision}{extra}"
        )
        self._logged = True


def _eligible(conv: nn.Conv2d) -> bool:
    """Whether the kitchen's conv3d can serve this convolution.

    fp16 or bf16 (the kernel is fp16, bf16 casts exactly), a 3x3 square kernel
    (the weight's view and the zero border), dense, stride one, and K and C
    the kernel's 128-wide tile serves; the FLOPs gate runs per call, since the
    spatial size it needs is only known there.
    """
    w = conv.weight
    return (
        w.dtype in _ROUTE_DTYPES
        and (w.dtype == torch.float16 or _BF16_ROUTE)
        and w.device.type == "cuda"
        and w.dim() == 4
        and w.shape[2] == 3
        and w.shape[3] == 3
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
            print(
                f"  [kitchen conv] the VAE has no decoder: "
                f"{[a for a in dir(vae) if 'stage' in a or 'model' in a]}"
            )
        return 0
    if getattr(decoder, "_kitchen_conv_patched", False):
        return 0
    patched = 0
    for name, mod in list(decoder.named_modules()):
        if not isinstance(mod, nn.Conv2d):
            continue
        if (
            not _eligible(mod)
            and os.getenv("COMFY_KITCHEN_VAE_CONV_VERBOSE") == "1"
            and patched == 0
        ):
            w = mod.weight
            print(
                f"  [kitchen conv] {name} is not eligible: weight {tuple(w.shape)} "
                f"{w.dtype} on {w.device}, groups {mod.groups}, stride {mod.stride}, "
                f"dilation {mod.dilation}"
            )
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
    if original is None:
        return

    def decode(self, samples_in, *args, **kwargs):
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
