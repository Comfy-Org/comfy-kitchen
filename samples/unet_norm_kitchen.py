"""Fuse the diffusion model's resnet GroupNorm + SiLU on the kitchen's kernel.

Every SDXL resnet block runs `norm` then `silu` on a contiguous NCHW
activation. torch does that in four streaming passes: GroupNorm reads x and
writes its output, SiLU reads that and writes again. Only the intermediate is
wasted -- the group statistics cannot be known before the group has been read
once -- so a fused kernel needs three passes (read for the statistics, read and
write for the apply). On this iGPU that is worth more than the pass count
suggests, because the pair runs at 19-36 GB/s while a single streaming kernel of
this shape reaches 92-100 GB/s. Measured per call on the 6-WGP 780M, fp16,
against `silu(group_norm(x))` (2026-10, current driver):

    C    H^2    x2     torch      fused    (torch / fused)
    320  128^2  21 MB   2.12 ms   1.01 ms   2.10x
    320   64^2   5 MB   0.72 ms   0.24 ms   2.99x
    640   64^2  10 MB   1.18 ms   0.47 ms   2.51x
    640   32^2   3 MB   0.26 ms   0.11 ms   2.31x
    1280  32^2   5 MB   0.45 ms   0.23 ms   2.01x
    2560  32^2  10 MB   0.82 ms   0.43 ms   1.90x
    1920  32^2   8 MB   0.60 ms   0.34 ms   1.79x
    1920  64^2  31 MB   2.34 ms   1.28 ms   1.84x
    1280  64^2  21 MB   1.59 ms   0.88 ms   1.80x
    960   64^2  16 MB   1.19 ms   0.66 ms   1.81x
    960  128^2  63 MB   4.59 ms   2.53 ms   1.81x
    640  128^2  42 MB   3.23 ms   1.70 ms   1.90x

Standalone that is 56.7 -> 27.2 ms over the 34 pairs of one SDXL forward (batch 2,
128x128 latent); run where the model runs them, with its other tensors live, the
stream is 47.4 -> 31.2 ms, because both sides lose some of their standalone
bandwidth to the forward's cache and TLB state.

A paired in-process A/B of the whole UNet forward (see benchmark/unet_norm_pair.py)
moves 2630 -> 2614 ms, so ~15 ms per forward, ~0.8% of a 1790 ms step, with a
spread of 2 ms across four runs a side.

The patch replaces the `forward` of each resnet block's GroupNorm with the
fused call and swaps the SiLU that followed it for a pass-through, so the
module tree, the parameter names and the state dict keys are unchanged. Only
`openaimodel.ResBlock` pairs are touched -- the SD/SDXL/most-2D-UNet family --
and the `out_layers` pair only when the block does not modulate between the
norm and the activation (`use_scale_shift_norm`), which would otherwise change
the result. The 12 GroupNorms that the transformer blocks run before `proj_in`
have no SiLU after them and are left alone, as are the LayerNorms.

Usage, from a ComfyUI custom node or an import-time hook::

    from unet_norm_kitchen import patch_unet_norms

    patch_unet_norms()

Without this module (or with COMFY_KITCHEN_UNET_NORM=0) nothing changes.
"""

from __future__ import annotations

import logging
import os
import types

import torch
import torch.nn as nn

import comfy.ldm.modules.diffusionmodules.openaimodel as openaimodel
import comfy.model_management
import comfy.ops

logger = logging.getLogger(__name__)

_DTYPES = (torch.float16, torch.bfloat16)

# Off switches the fused branch without unpatching, so a paired A/B can flip it
# inside one process (see benchmark/unet_norm_pair.py).
_ENABLED = True


def _backend():
    from comfy_kitchen.backends import hip as hip_backend

    return hip_backend


def set_fused(enabled: bool) -> None:
    """Turn the fused branch on or off in place, for paired measurement."""
    global _ENABLED
    _ENABLED = enabled


class _AbsorbedSilu(nn.Module):
    """The SiLU the fused norm already applied.

    Keeping the original module (instead of dropping it) lets `set_fused(False)`
    restore the stock path, and keeps the Sequential's later indices where the
    block's forward expects them.
    """

    def __init__(self, silu: nn.SiLU) -> None:
        super().__init__()
        self.silu = silu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x if _ENABLED else self.silu(x)


def _usable(norm: nn.GroupNorm, x: torch.Tensor) -> bool:
    """Whether this call can take the fused path.

    Everything the kernel insists on is checked here: a packed 4-D activation in
    the norm's own dtype and device, a 16-byte-aligned base for its vector
    loads, and no weight/bias patch functions that would have to be applied
    first.
    """
    w = norm.weight
    return (
        x.dim() == 4
        and x.dtype in _DTYPES
        and x.dtype == w.dtype
        and x.device == w.device
        and x.is_contiguous()
        and x.data_ptr() % 16 == 0
        and x.shape[1] == norm.num_channels
        and x.shape[0] * norm.num_groups <= 65535
        and not getattr(norm, "weight_function", None)
        and not getattr(norm, "bias_function", None)
    )


def _fuse_seq(seq: nn.Sequential, dm: nn.Module) -> bool:
    """Fuse `seq[0]` (GroupNorm) and `seq[1]` (SiLU) of one block."""
    norm, act = seq[0], seq[1]
    if not isinstance(norm, nn.GroupNorm) or type(act) is not nn.SiLU:
        return False
    if not norm.affine or norm.weight is None or norm.bias is None:
        return False
    if norm.weight.dtype not in _DTYPES or norm.bias.dtype != norm.weight.dtype:
        return False
    if norm.num_channels % norm.num_groups != 0:
        return False
    # Only a SiLU that belongs to this pair may be turned into a pass-through.
    if sum(1 for m in dm.modules() if m is act) != 1:
        return False

    original = norm.forward
    num_groups, eps = norm.num_groups, norm.eps

    def forward(self, x, *args, **kwargs):
        if _ENABLED and _usable(self, x):
            comfy.ops.run_every_op()
            return _backend().group_norm_silu_nchw(
                x, self.weight, self.bias, num_groups, eps
            )
        return original(x, *args, **kwargs)

    norm.forward = types.MethodType(forward, norm)
    norm._kitchen_norm = True
    seq[1] = _AbsorbedSilu(act)
    return True


def _patch_block(block: openaimodel.ResBlock, dm: nn.Module) -> int:
    patched = _fuse_seq(block.in_layers, dm)
    # With scale/shift norm the block multiplies and offsets the norm output
    # before the SiLU, so that pair is not `silu(group_norm(x))`.
    if not getattr(block, "use_scale_shift_norm", False):
        patched += _fuse_seq(block.out_layers, dm)
    return int(patched)


def _patch_model(model) -> int:
    dm = getattr(model, "diffusion_model", None)
    if dm is None or getattr(dm, "_kitchen_norm_patched", False):
        return 0
    patched = 0
    for block in dm.modules():
        if isinstance(block, openaimodel.ResBlock):
            patched += _patch_block(block, dm)
    dm._kitchen_norm_patched = True
    if patched:
        logger.info("kitchen norm: %d resnet GroupNorm+SiLU pairs fused", patched)
    return patched


def patch_unet_norms(verbose: bool = False) -> None:
    """Fuse the resnet norm/activation pairs of every diffusion model loaded.

    The patch hangs off `load_models_gpu`, so it applies when a model is moved
    onto the device and covers the checkpoint and diffusion-only load paths
    alike.
    """
    if getattr(comfy.model_management, "_kitchen_norm_patched", False):
        return
    if not _backend().group_norm_silu_nchw_is_available():
        logger.info("kitchen norm: this extension has no fused GroupNorm+SiLU")
        return

    original = comfy.model_management.load_models_gpu

    def load_models_gpu(models, *args, **kwargs):
        patched = 0
        for p in models if isinstance(models, (list, tuple)) else [models]:
            m = getattr(p, "model", None)
            if m is not None:
                patched += _patch_model(m)
        if patched and verbose:
            print(f"  [kitchen norm] {patched} resnet GroupNorm+SiLU pairs fused")
        return original(models, *args, **kwargs)

    comfy.model_management.load_models_gpu = load_models_gpu
    comfy.model_management._kitchen_norm_patched = True
    logger.info("kitchen norm: resnet GroupNorm+SiLU pairs route through the kitchen")


if os.getenv("COMFY_KITCHEN_UNET_NORM", "1") == "1":
    # Importing this module is the opt-in: a ComfyUI custom node is imported at
    # startup, and the patch is inert for a model whose dtype or layout the
    # kernel does not serve.
    patch_unet_norms(verbose=os.getenv("COMFY_KITCHEN_UNET_NORM_VERBOSE", "0") == "1")
