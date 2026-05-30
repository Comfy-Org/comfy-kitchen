# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# SVDQuant W4A4 (int4 weight, int4 activation, SVD low-rank correction) layout.

"""SVDQuant W4A4 quantization layout for tensor cores.

Each quantized linear stores:
  qweight:       (N, K // 2)  int8        packed W4 residual
  scale=wscales: (K // 64, N) bf16/fp16   per-group weight scales
  proj_down:     (K, R)       bf16/fp16   SVD down projection (V^T)
  proj_up:       (N, R)       bf16/fp16   SVD up projection (U)
  smooth_factor: (K,)         bf16/fp16   input-side smoothing

LoRA-style proj_down / proj_up recover the outlier-heavy singular directions
that pure 4-bit quantization cannot represent; the dispatched kernel fuses
activation quantization + low-rank correction + int4 matmul into a single call.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

import comfy_kitchen as ck

from .base import BaseLayoutParams, QuantizedLayout, dequantize_args, register_layout_op

if TYPE_CHECKING:
    from .base import QuantizedTensor

logger = logging.getLogger(__name__)

_INT4_GROUP_SIZE = 64
_TILE_PACKED_BLOCK_N = 128
_GELU_UNSIGNED_SHIFT = 0.171875
_FALSE_ENV_VALUES = {"0", "false", "no", "off"}


def _env_enabled(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in _FALSE_ENV_VALUES


def _registry_backend_override() -> str | None:
    return getattr(ck.registry._thread_local, "backend_override", None)


def _direct_cuda_backend(input_tensor: torch.Tensor, qdata: torch.Tensor):
    """Return the CUDA backend module when SVDQuant can safely bypass registry.

    ComfyUI currently disables comfy_kitchen's global CUDA backend on PyTorch
    CUDA < 13, but this SVDQuant extension is built locally and works on the
    cu128/RTX 5090 environment. The QuantizedTensor path can call the CUDA
    implementation directly while still respecting an explicit non-CUDA backend
    override such as ``with ck.use_backend("eager")``.
    """
    if not _env_enabled("COMFY_KITCHEN_SVDQUANT_DIRECT_CUDA", default=True):
        return None

    override = _registry_backend_override()
    if override is not None and override != "cuda":
        return None

    if not input_tensor.is_cuda or not qdata.is_cuda:
        return None
    if input_tensor.device != qdata.device:
        return None

    try:
        from comfy_kitchen.backends import cuda as cuda_backend
    except Exception:
        return None
    if not getattr(cuda_backend, "_EXT_AVAILABLE", False):
        return None
    return cuda_backend


def _is_svdquant_w4a4_qtensor(tensor: torch.Tensor) -> bool:
    from .base import QuantizedTensor

    return (
        isinstance(tensor, QuantizedTensor)
        and tensor._layout_cls == "TensorCoreSVDQuantW4A4Layout"
        and not bool(getattr(tensor._params, "transposed", False))
    )


def _same_tensor_metadata(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and a.device == b.device


def svdquant_w4a4_can_share_quant(
    weights: Sequence[torch.Tensor],
    *,
    validate: bool = False,
    trust: bool = False,
) -> bool:
    """Return True when split SVDQuant projections can reuse activation quantize.

    Qwen split Q/K/V checkpoints store separate weight tensors but share the
    exact same activation-side SVDQuant parameters: ``smooth_factor`` and
    ``proj_down``. When that invariant holds, the expensive activation int4
    quantization and LoRA-down matmul can run once and feed all projections.

    ``validate=True`` compares tensor values when pointers differ. Use it once
    at module setup or first forward, then cache the result at the caller. Pass
    ``trust=True`` only after such a cached validation, for temporary casted
    copies whose tensor pointers differ but whose source parameters were
    already proven identical.
    """
    if len(weights) == 0:
        return False
    if not all(_is_svdquant_w4a4_qtensor(weight) for weight in weights):
        return False

    base = weights[0]._params
    for weight in weights[1:]:
        params = weight._params
        if bool(params.act_unsigned) != bool(base.act_unsigned):
            return False
        if not _same_tensor_metadata(params.smooth_factor, base.smooth_factor):
            return False
        if not _same_tensor_metadata(params.proj_down, base.proj_down):
            return False
        if params.smooth_factor.data_ptr() == base.smooth_factor.data_ptr() and (
            params.proj_down.data_ptr() == base.proj_down.data_ptr()
        ):
            continue
        if trust:
            continue
        if not validate:
            return False
        if not torch.equal(params.smooth_factor, base.smooth_factor):
            return False
        if not torch.equal(params.proj_down, base.proj_down):
            return False
    return True


def _w4a4_forward_from_quantized_activation(
    q_x: torch.Tensor,
    ascales: torch.Tensor,
    lora_act: torch.Tensor,
    weight_qt: QuantizedTensor,
    bias: torch.Tensor | None,
    cuda_backend,
    *,
    m: int,
) -> tuple[torch.Tensor, int]:
    qdata, wscales, _smooth, _proj_down, proj_up = TensorCoreSVDQuantW4A4Layout.get_plain_tensors(weight_qt)
    act_unsigned = bool(getattr(weight_qt._params, "act_unsigned", False))
    if cuda_backend is not None:
        out = cuda_backend.scaled_mm_svdquant_w4a4(
            act=q_x, wgt=qdata, ascales=ascales, wscales=wscales,
            lora_act_in=lora_act, lora_up=proj_up, bias=bias,
            act_unsigned=act_unsigned,
        )
    else:
        out = ck.scaled_mm_svdquant_w4a4(
            act=q_x, wgt=qdata, ascales=ascales, wscales=wscales,
            lora_act_in=lora_act, lora_up=proj_up, bias=bias,
            act_unsigned=act_unsigned,
        )
    out_features = TensorCoreSVDQuantW4A4Layout.get_out_features_from_storage(qdata)
    return out[:m], out_features


def svdquant_w4a4_grouped_linear(
    input_tensor: torch.Tensor,
    weights: Sequence[torch.Tensor],
    biases: Sequence[torch.Tensor | None] | None = None,
    *,
    validate_shared_quant: bool = False,
    assume_shared_quant: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Run split SVDQuant linears while sharing activation quantize.

    This is intentionally a runtime grouping helper, not a fused-QKV storage
    format. It preserves split Q/K/V checkpoint tensors and only removes the
    repeated per-input work that is identical across those projections.
    """
    if len(weights) == 0:
        return ()
    if biases is None:
        biases = (None,) * len(weights)
    if len(biases) != len(weights):
        raise ValueError(f"got {len(weights)} weights but {len(biases)} biases")
    if not svdquant_w4a4_can_share_quant(
        weights, validate=validate_shared_quant, trust=assume_shared_quant,
    ):
        raise ValueError("SVDQuant weights do not share quantization parameters")

    first = weights[0]
    qdata, _wscales, smooth, proj_down, _proj_up = TensorCoreSVDQuantW4A4Layout.get_plain_tensors(first)
    act_unsigned = bool(getattr(first._params, "act_unsigned", False))

    orig_shape = input_tensor.shape
    x2d = input_tensor.reshape(-1, orig_shape[-1])
    m = x2d.shape[0]

    if act_unsigned:
        x_main = x2d + _GELU_UNSIGNED_SHIFT
        lora_x = x2d
    else:
        x_main = x2d
        lora_x = None

    cuda_backend = _direct_cuda_backend(x_main, qdata)
    if cuda_backend is not None:
        q_x, ascales, lora_act = cuda_backend.quantize_svdquant_w4a4(
            x_main,
            smooth=smooth,
            lora_down=proj_down,
            act_unsigned=act_unsigned,
            lora_x=lora_x,
            reuse_workspace=True,
        )
    else:
        q_x, ascales, lora_act = ck.quantize_svdquant_w4a4(
            x_main,
            smooth=smooth,
            lora_down=proj_down,
            act_unsigned=act_unsigned,
            lora_x=lora_x,
        )

    outputs = []
    for weight, bias in zip(weights, biases, strict=True):
        out2d, out_features = _w4a4_forward_from_quantized_activation(
            q_x, ascales, lora_act, weight, bias, cuda_backend, m=m,
        )
        outputs.append(out2d.reshape(*orig_shape[:-1], out_features))
    return tuple(outputs)


def _cat_svdquant_n_axis(tensors: Sequence[torch.Tensor], *, natural_dim: int) -> torch.Tensor:
    dim = tensors[0].dim()
    if any(t.dim() != dim for t in tensors):
        raise ValueError("cannot fuse mixed natural/tile-packed SVDQuant tensors")
    if dim in {3, 4}:
        return torch.cat(tuple(tensors), dim=0).contiguous()
    return torch.cat(tuple(tensors), dim=natural_dim).contiguous()


def svdquant_w4a4_fuse_linear_weights(
    weights: Sequence[torch.Tensor],
    biases: Sequence[torch.Tensor | None] | None = None,
    *,
    validate_shared_quant: bool = False,
    assume_shared_quant: bool = False,
) -> tuple[QuantizedTensor, torch.Tensor | None, tuple[int, ...]]:
    """Fuse split SVDQuant projections into a runtime-only wide projection.

    This does not change checkpoint storage. It concatenates the already loaded
    split weights along output-N so a caller can execute Q/K/V as one wider
    SVDQuant linear and split the output view afterwards.
    """
    from .base import QuantizedTensor

    if len(weights) == 0:
        raise ValueError("expected at least one weight")
    if biases is None:
        biases = (None,) * len(weights)
    if len(biases) != len(weights):
        raise ValueError(f"got {len(weights)} weights but {len(biases)} biases")
    if not svdquant_w4a4_can_share_quant(
        weights, validate=validate_shared_quant, trust=assume_shared_quant,
    ):
        raise ValueError("SVDQuant weights do not share quantization parameters")

    first = weights[0]
    first_params = first._params
    in_features = first_params.orig_shape[1]
    act_unsigned = bool(first_params.act_unsigned)
    out_features = []
    qdatas = []
    scales = []
    proj_ups = []
    for weight in weights:
        qdata, wscale, _smooth, _proj_down, proj_up = TensorCoreSVDQuantW4A4Layout.get_plain_tensors(weight)
        if weight._params.orig_shape[1] != in_features:
            raise ValueError("all fused SVDQuant weights must have the same input features")
        if bool(weight._params.act_unsigned) != act_unsigned:
            raise ValueError("all fused SVDQuant weights must have the same act_unsigned flag")
        out_features.append(TensorCoreSVDQuantW4A4Layout.get_out_features_from_storage(qdata))
        qdatas.append(qdata)
        scales.append(wscale)
        proj_ups.append(proj_up)

    fused_qdata = _cat_svdquant_n_axis(qdatas, natural_dim=0)
    fused_scale = _cat_svdquant_n_axis(scales, natural_dim=1)
    fused_proj_up = _cat_svdquant_n_axis(proj_ups, natural_dim=0)

    if all(bias is None for bias in biases):
        fused_bias = None
    else:
        bias_parts = []
        for bias, n in zip(biases, out_features, strict=True):
            if bias is None:
                bias = torch.zeros(n, dtype=fused_proj_up.dtype, device=fused_proj_up.device)
            bias_parts.append(bias)
        fused_bias = torch.cat(tuple(bias_parts), dim=0).contiguous()

    params = TensorCoreSVDQuantW4A4Layout.Params(
        scale=fused_scale,
        orig_dtype=first_params.orig_dtype,
        orig_shape=(sum(out_features), in_features),
        proj_down=first_params.proj_down,
        proj_up=fused_proj_up,
        smooth_factor=first_params.smooth_factor,
        act_unsigned=act_unsigned,
    )
    return (
        QuantizedTensor(fused_qdata, "TensorCoreSVDQuantW4A4Layout", params),
        fused_bias,
        tuple(out_features),
    )


def svdquant_w4a4_fused_grouped_linear(
    input_tensor: torch.Tensor,
    fused_weight: torch.Tensor,
    fused_bias: torch.Tensor | None,
    output_features: Sequence[int],
) -> tuple[torch.Tensor, ...]:
    """Execute a runtime-fused split projection and return split output views."""
    if not _is_svdquant_w4a4_qtensor(fused_weight):
        raise TypeError("fused_weight must be a TensorCoreSVDQuantW4A4Layout QuantizedTensor")
    out = _w4a4_forward(input_tensor, fused_weight, fused_bias)
    return tuple(torch.split(out, tuple(output_features), dim=-1))


class TensorCoreSVDQuantW4A4Layout(QuantizedLayout):
    """SVDQuant W4A4 weight quantization with low-rank correction.

    Note:
        Offline-quantized only — `quantize()` raises NotImplementedError because
        SVDQuant factorization requires calibration (smooth_factor, proj_down,
        proj_up) that must be computed from activation statistics. Use the
        DeepCompressor pipeline to produce the pre-quantized tensors.
    """

    # m16n8k64 int4 MMA requires SM >= 8.0 (Ampere). The kitchen CUDA kernels
    # in comfy_kitchen/backends/cuda/ops/{quantize,scaled_mm}_svdquant_w4a4.cu
    # gate the PTX body on __CUDA_ARCH__ >= 800 and raise at runtime on older
    # arches (see svdquant_utils.cuh::trap_pre_sm80).
    MIN_SM_VERSION = (8, 0)

    # Activation quantization is fused inside the kernel — do not pre-wrap
    # the input with QuantizedTensor.from_float(). Consumers (e.g. ComfyUI's
    # mixed_precision_ops.Linear) should read this flag before attempting to
    # quantize an incoming float activation.
    QUANTIZES_INPUT = False

    @dataclass(frozen=True)
    class Params(BaseLayoutParams):
        """SVDQuant W4A4 parameters.

        Inherits `scale` (= wscales), `orig_dtype`, `orig_shape` from
        BaseLayoutParams. Adds the three tensors that parameterize the
        low-rank correction and input smoothing, plus a logical-transpose flag
        used by the aten.t / aten.mm dispatch path.
        """
        proj_down: torch.Tensor
        proj_up: torch.Tensor
        smooth_factor: torch.Tensor
        act_unsigned: bool = False
        transposed: bool = False

        def _tensor_fields(self) -> list[str]:
            return ["scale", "proj_down", "proj_up", "smooth_factor"]

        def _validate_tensor_fields(self):
            # Unlike per-tensor scale layouts, wscales is per-group and stays
            # in the model compute dtype (bf16 / fp16) — do not coerce.
            return

    @classmethod
    def quantize(
        cls,
        tensor: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, Params]:
        raise NotImplementedError(
            "SVDQuant W4A4 requires offline calibration (DeepCompressor). "
            "Load pre-quantized tensors via `from_state_dict` instead."
        )

    @classmethod
    def dequantize(cls, qdata: torch.Tensor, params: Params) -> torch.Tensor:
        """Reconstruct the effective weight W_eff such that plain ``x @ W_eff.T + bias``
        reproduces the SVDQuant kernel output to bf16 precision.

        Uses the kitchen kernel itself with an identity input rather than
        reimplementing dequant in Python. Kitchen weight layout is natural
        row-major packed int4 ``(N, K/2)`` — the kernel reads it directly, so
        this path stays bit-exact with the actual compute path regardless of
        per-group scaling / LoRA composition details. Tile-packed storage is
        handled by the same kernel path and produces the same logical weight.
        """
        in_features = params.orig_shape[1]
        device = qdata.device
        dtype = params.orig_dtype

        eye = torch.eye(in_features, dtype=dtype, device=device)
        q_x, ascales, lora_act = ck.quantize_svdquant_w4a4(
            eye, smooth=params.smooth_factor, lora_down=params.proj_down,
        )
        w_eff = ck.scaled_mm_svdquant_w4a4(
            act=q_x, wgt=qdata, ascales=ascales, wscales=params.scale,
            lora_act_in=lora_act, lora_up=params.proj_up, bias=None,
        )[:in_features]
        return w_eff.t().contiguous()

    @classmethod
    def get_plain_tensors(
        cls, qtensor: QuantizedTensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        p = qtensor._params
        return qtensor._qdata, p.scale, p.smooth_factor, p.proj_down, p.proj_up

    @classmethod
    def get_out_features_from_storage(cls, qdata: torch.Tensor) -> int:
        if qdata.dim() == 4:
            return int(qdata.shape[0]) * _TILE_PACKED_BLOCK_N
        return int(qdata.shape[0])

    @classmethod
    def state_dict_tensors(cls, qdata: torch.Tensor, params: Params) -> dict[str, torch.Tensor]:
        """Serialization mapping.

        Suffixes compose onto the owning Parameter's key (typically `*.weight`),
        producing for example `transformer_blocks.0.attn.to_q.weight`,
        `...weight_scale`, `...weight_proj_down`, etc.
        """
        return {
            "": qdata,
            "_scale": params.scale,
            "_proj_down": params.proj_down,
            "_proj_up": params.proj_up,
            "_smooth_factor": params.smooth_factor,
        }


# ==================== Linear Dispatch ====================

def _w4a4_forward(
    input_tensor: torch.Tensor,
    weight_qt: QuantizedTensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Compute y = x @ W^T + bias via the int4 kernel.

    For layers flagged ``act_unsigned`` (nunchaku convention: post-GELU fc2),
    we apply the +0.171875 shift to the main-path activation here at the layer
    so it falls into the unsigned [0, 15] quantization grid. LoRA continues to
    use the raw un-shifted activation (SVDQuant invariant: LoRA is the residual
    between full-precision W and the int4 approximation, computed on the
    pre-quantization activation).

    Kernel API stays shift-free — shift is a Qwen/Flux model-topology constant,
    not a quantize-op parameter.
    """
    qdata, wscales, smooth, proj_down, proj_up = TensorCoreSVDQuantW4A4Layout.get_plain_tensors(weight_qt)
    act_unsigned = bool(getattr(weight_qt._params, "act_unsigned", False))

    orig_shape = input_tensor.shape
    x2d = input_tensor.reshape(-1, orig_shape[-1])
    m = x2d.shape[0]

    if act_unsigned:
        x_main = x2d + _GELU_UNSIGNED_SHIFT  # fed to quantize for unsigned grid
        lora_x = x2d                         # LoRA always sees raw x
    else:
        x_main = x2d
        lora_x = None                        # wrapper will fall back to x_main

    cuda_backend = _direct_cuda_backend(x_main, qdata)
    if cuda_backend is not None:
        q_x, ascales, lora_act = cuda_backend.quantize_svdquant_w4a4(
            x_main,
            smooth=smooth,
            lora_down=proj_down,
            act_unsigned=act_unsigned,
            lora_x=lora_x,
            reuse_workspace=True,
        )
        out = cuda_backend.scaled_mm_svdquant_w4a4(
            act=q_x, wgt=qdata, ascales=ascales, wscales=wscales,
            lora_act_in=lora_act, lora_up=proj_up, bias=bias,
            act_unsigned=act_unsigned,
        )
    else:
        q_x, ascales, lora_act = ck.quantize_svdquant_w4a4(
            x_main,
            smooth=smooth,
            lora_down=proj_down,
            act_unsigned=act_unsigned,
            lora_x=lora_x,
        )
        out = ck.scaled_mm_svdquant_w4a4(
            act=q_x, wgt=qdata, ascales=ascales, wscales=wscales,
            lora_act_in=lora_act, lora_up=proj_up, bias=bias,
            act_unsigned=act_unsigned,
        )
    out_features = TensorCoreSVDQuantW4A4Layout.get_out_features_from_storage(qdata)
    return out[:m].reshape(*orig_shape[:-1], out_features)


@register_layout_op(torch.ops.aten.t.default, TensorCoreSVDQuantW4A4Layout)
def _handle_w4a4_t(qt, args, kwargs):
    """Zero-copy logical transpose — flip the ``transposed`` flag.

    Lets ``F.linear(x, W)`` decompose into ``x @ W.t()`` without reordering any
    storage; ``mm`` / ``addmm`` handlers below unwind the flag.
    """
    import dataclasses

    from .base import QuantizedTensor

    input_tensor = args[0]
    if not isinstance(input_tensor, QuantizedTensor):
        return torch.ops.aten.t.default(*args, **kwargs)

    old = input_tensor._params
    new_params = dataclasses.replace(
        old,
        orig_shape=(old.orig_shape[1], old.orig_shape[0]),
        transposed=not old.transposed,
    )
    return QuantizedTensor(input_tensor._qdata, "TensorCoreSVDQuantW4A4Layout", new_params)


def _resolve_svdquant_rhs(rhs: QuantizedTensor) -> QuantizedTensor:
    """Return rhs unchanged if it is logically transposed (represents W^T)."""
    if not rhs._params.transposed:
        raise RuntimeError(
            "SVDQuant W4A4 GEMM expects the RHS to be W.T (stored W). "
            "Use F.linear(x, W) or mm(x, W.t())."
        )
    return rhs


@register_layout_op(torch.ops.aten.linear.default, TensorCoreSVDQuantW4A4Layout)
def _handle_w4a4_linear(qt, args, kwargs):
    """Direct F.linear(input, W, bias) → kitchen kernel."""
    from .base import QuantizedTensor

    input_tensor, weight = args[0], args[1]
    bias = args[2] if len(args) > 2 else None

    if not isinstance(weight, QuantizedTensor):
        return torch.nn.functional.linear(*dequantize_args((input_tensor, weight, bias)))
    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()
    if weight._params.transposed:
        return torch.nn.functional.linear(input_tensor, weight.dequantize(), bias)
    return _w4a4_forward(input_tensor, weight, bias)


@register_layout_op(torch.ops.aten.mm.default, TensorCoreSVDQuantW4A4Layout)
def _handle_w4a4_mm(qt, args, kwargs):
    """Handle ``mm(x, W.t())`` — the decomposition F.linear takes when the weight
    is a non-default tensor subclass.
    """
    from .base import QuantizedTensor

    a, b = args[0], args[1]
    if not isinstance(b, QuantizedTensor):
        return torch.mm(*dequantize_args((a, b)))
    if isinstance(a, QuantizedTensor):
        a = a.dequantize()
    b = _resolve_svdquant_rhs(b)
    return _w4a4_forward(a, b, bias=None)


@register_layout_op(torch.ops.aten.addmm.default, TensorCoreSVDQuantW4A4Layout)
def _handle_w4a4_addmm(qt, args, kwargs):
    """Handle ``addmm(bias, x, W.t())``."""
    from .base import QuantizedTensor

    bias, a, b = args[0], args[1], args[2]
    if not isinstance(b, QuantizedTensor):
        return torch.addmm(*dequantize_args((bias, a, b)))
    if isinstance(a, QuantizedTensor):
        a = a.dequantize()
    b = _resolve_svdquant_rhs(b)
    return _w4a4_forward(a, b, bias=bias)
