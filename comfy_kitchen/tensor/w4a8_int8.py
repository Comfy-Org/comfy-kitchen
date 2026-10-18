# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Grouped W4A8 / W6A8 weights executed through the shared INT8 linear operation.

Weights are ConvRot-rotated and quantized per ``group_size``. The default (``bits=4``)
uses a symmetric Lloyd-Max codebook and FP8 group scales; ``bits=6`` stores uniform
6-bit codes (no codebook) in a 3K/4-byte row, at ~3x lower weight error for 1.5x the
bytes. The bit depth is implied by the packed row width, so one layout class and one
checkpoint format cover both. The tensor layout owns only metadata and operator
routing; eager, CUDA, and Triton backends own the operation implementations.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch

from comfy_kitchen.exceptions import NoCapableBackendError
from comfy_kitchen.registry import registry

from .base import (
    BaseLayoutParams,
    QuantizedLayout,
    QuantizedTensor,
    dequantize_args,
    register_layout_op,
)
from .w4a8_packed import (
    _mma_bits,
    default_w4a8_codebook,
    unpack_w4a8_mma_weight,
)


def quantize_w4a8_int8_weight(
    weight: torch.Tensor,
    group_size: int = 16,
    convrot_groupsize: int = 256,
    symmetric: bool = True,
    scale_dtype: torch.dtype = torch.float8_e4m3fn,
    codebook: bool = True,
    codebook_tensor: torch.Tensor | None = None,
    stochastic_rounding: int = 0,
    bits: int = 4,
    scale_search: bool = True,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Rotate and prepare a floating weight for grouped W4A8 (or W6A8) storage.

    ``codebook_tensor`` reuses a previously decided table (e.g. on LoRA requantize) so the
    codebook decision -- kurtosis probe and any k-means -- is skipped. ``stochastic_rounding``
    > 0 seeds stochastic rounding of the level assignment so a merged LoRA below the int4 step
    is preserved instead of rounded away. ``bits=6`` selects uniform, codebook-free 6-bit
    storage; ``scale_search`` (6-bit only) runs the grid-aware group-scale search at quantize
    time, which the requantize path skips.
    """
    if scale_dtype not in (torch.float32, torch.float8_e4m3fn):
        raise ValueError(f"scale_dtype must be float32 or float8_e4m3fn, got {scale_dtype}")
    kwargs = {
        "weight": weight,
        "group_size": group_size,
        "convrot_groupsize": convrot_groupsize,
        "symmetric": symmetric,
        "scale_dtype": scale_dtype,
        "codebook": codebook,
        "codebook_tensor": codebook_tensor,
        "stochastic_rounding": stochastic_rounding,
        "bits": bits,
        "scale_search": scale_search,
    }
    impl = registry.get_implementation("quantize_w4a8_int8_weight", kwargs=kwargs)
    return impl(**kwargs)


def dequantize_w4a8_int8_weight(
    qdata: torch.Tensor,
    s_rel: torch.Tensor,
    s_channel: torch.Tensor,
    codebook: torch.Tensor | None = None,
    correction: torch.Tensor | None = None,
    group_size: int = 16,
    convrot_groupsize: int = 256,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize a packed W4A8 weight into its original basis."""
    kwargs = {
        "qdata": qdata,
        "s_rel": s_rel,
        "s_channel": s_channel,
        "codebook": codebook,
        "correction": correction,
        "group_size": group_size,
        "convrot_groupsize": convrot_groupsize,
        "output_dtype": output_dtype,
    }
    impl = registry.get_implementation("dequantize_w4a8_int8_weight", kwargs=kwargs)
    return impl(**kwargs)


def w4a8_int8_linear(
    x: torch.Tensor,
    qdata: torch.Tensor,
    s_rel: torch.Tensor,
    s_channel: torch.Tensor,
    codebook: torch.Tensor | None = None,
    correction: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    group_size: int = 16,
    convrot_groupsize: int = 256,
    out_dtype: torch.dtype = torch.bfloat16,
    mma_packed: bool = False,
    input_act: str | None = None,
    input_act_weight: torch.Tensor | None = None,
    input_act_eps: float = 0.0,
    residual: torch.Tensor | None = None,
    residual_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``x @ W.T + bias`` with the selected W4A8 backend.

    input_act ("swiglu" or "rms_norm", with input_act_weight/input_act_eps for the
    norm) is applied to x on the way into the activation quantizer; the CUDA
    decode path folds it into that kernel, every other path applies it eagerly
    with identical results. residual/residual_scale make the result
    ``residual + residual_scale * (x @ W.T + bias)`` (a pre-norm block's addcmul,
    with the linear rounded to out_dtype first), fused into the CUDA decode
    kernel's epilogue and applied after the GEMM everywhere else.
    """
    kwargs = {
        "x": x,
        "qdata": qdata,
        "s_rel": s_rel,
        "s_channel": s_channel,
        "codebook": codebook,
        "correction": correction,
        "bias": bias,
        "group_size": group_size,
        "convrot_groupsize": convrot_groupsize,
        "out_dtype": out_dtype,
        "input_act": input_act,
        "input_act_weight": input_act_weight,
        "input_act_eps": input_act_eps,
        "residual": residual,
        "residual_scale": residual_scale,
    }
    operation = "w4a8_packed_linear" if mma_packed else "w4a8_int8_linear"
    impl = registry.get_implementation(operation, kwargs=kwargs)
    return impl(**kwargs)


class AsymW4A8Int8Layout(QuantizedLayout):
    """Grouped W4A8 (or uniform W6A8) weights run through the selected INT8 backend."""

    MIN_SM_VERSION = (8, 0)
    QUANTIZES_INPUT = False

    @dataclass(frozen=True)
    class Params(BaseLayoutParams):
        # scale holds s_rel: per-group relative scale [N, K // group_size].
        s_channel: torch.Tensor | None = None
        correction: torch.Tensor | None = None
        codebook: torch.Tensor | None = None
        group_size: int = 16
        convrot_groupsize: int = 256
        # Packed qdata interleaves codes and group scales; scale is then empty.
        mma_packed: bool = False
        transposed: bool = False

        def _tensor_fields(self) -> list[str]:
            fields = ["scale", "s_channel"]
            if self.correction is not None:
                fields.append("correction")
            if self.codebook is not None:
                fields.append("codebook")
            return fields

        def _validate_tensor_fields(self):
            if len(self.orig_shape) != 2:
                raise ValueError(f"orig_shape must be 2D, got {self.orig_shape}")
            n, k = self.orig_shape
            if self.transposed:
                n, k = k, n
            if (
                self.group_size < 4
                or k % 16 != 0
                or k % self.group_size != 0
                or k % self.convrot_groupsize != 0
                or (16 % self.group_size != 0 and self.group_size % 16 != 0)
            ):
                raise ValueError(
                    f"K={k} must be divisible by 16, group_size={self.group_size}, and "
                    f"convrot_groupsize={self.convrot_groupsize}; group_size must be >=4 "
                    f"and divide 16 or be a multiple of 16"
                )
            groups = k // self.group_size
            expected_scale_shape = (0,) if self.mma_packed else (n, groups)
            if tuple(self.scale.shape) != expected_scale_shape:
                raise ValueError(
                    f"scale must have shape {expected_scale_shape}, got {tuple(self.scale.shape)}"
                )
            if self.s_channel is None or tuple(self.s_channel.shape) != (n,):
                actual_shape = None if self.s_channel is None else tuple(self.s_channel.shape)
                raise ValueError(f"s_channel must have shape {(n,)}, got {actual_shape}")
            if self.correction is not None and tuple(self.correction.shape) != (groups, n):
                raise ValueError(
                    f"correction must have shape {(groups, n)}, got {tuple(self.correction.shape)}"
                )
            if self.codebook is not None and tuple(self.codebook.shape) != (16,):
                raise ValueError(
                    f"codebook must have shape (16,), got {tuple(self.codebook.shape)}"
                )

    @classmethod
    def quantize(
        cls,
        tensor: torch.Tensor,
        group_size: int = 16,
        convrot_groupsize: int = 256,
        symmetric: bool = True,
        scale_dtype: torch.dtype = torch.float8_e4m3fn,
        codebook: bool = True,
        codebook_tensor: torch.Tensor | None = None,
        stochastic_rounding: int = 0,
        bits: int = 4,
        scale_search: bool = True,
        **kwargs,
    ) -> tuple[torch.Tensor, Params]:
        qdata, s_rel, s_channel, correction, codebook_tensor = quantize_w4a8_int8_weight(
            tensor,
            group_size=group_size,
            convrot_groupsize=convrot_groupsize,
            symmetric=symmetric,
            scale_dtype=scale_dtype,
            codebook=codebook,
            codebook_tensor=codebook_tensor,
            stochastic_rounding=stochastic_rounding,
            bits=bits,
            scale_search=scale_search,
        )
        params = cls.Params(
            scale=s_rel,
            s_channel=s_channel,
            correction=correction,
            codebook=codebook_tensor,
            orig_dtype=tensor.dtype,
            orig_shape=tuple(tensor.shape),
            group_size=group_size,
            convrot_groupsize=convrot_groupsize,
        )
        return qdata, params

    @classmethod
    def optimize_for_gemv(cls, weight: QuantizedTensor, storage: torch.Tensor) -> QuantizedTensor:
        """Pack resident codes/scales for small-M MMA. Prefill dequantizes the same
        records directly into INT8 chunks; neither path needs canonical weights."""
        qdata, params = weight._qdata, weight._params
        if (
            params.mma_packed
            or params.transposed
            or not qdata.is_cuda
            or torch.version.hip is not None
            or not registry.is_available("cuda")
            or torch.cuda.get_device_capability(qdata.device) < (8, 0)
        ):
            return weight
        n, k = params.orig_shape
        if params.codebook is None:
            # 6-bit codes are uniform; 4-bit without a codebook is not what the kernel decodes
            codebook_ok = qdata.shape[1] * 8 == k * 6
        else:
            codebook_ok = torch.equal(params.codebook.float().cpu(), default_w4a8_codebook())
        if (
            n % 16 != 0 or k % 256 != 0
            or params.group_size != 16
            or params.correction is not None
            or params.scale.dtype != torch.float8_e4m3fn
            or not codebook_ok
        ):
            return weight
        if storage.dtype != torch.uint8 or storage.ndim != 1 or not storage.is_contiguous() or storage.device != qdata.device:
            raise ValueError("resident storage must be a contiguous byte buffer on the weight's device")
        start = qdata.data_ptr() - storage.data_ptr()
        scale_start = params.scale.data_ptr() - storage.data_ptr()
        end = scale_start + params.scale.nbytes
        if start < 0 or scale_start < start + qdata.nbytes or end > storage.nbytes:
            return weight
        # Import only on the eligible CUDA path; the backend imports this layout.
        from comfy_kitchen.backends.cuda import pack_w4a8_mma_weight_

        dest = storage[start:start + qdata.nbytes + params.scale.nbytes].view(qdata.dtype)
        pack_w4a8_mma_weight_(dest, params.scale)
        params = dataclasses.replace(
            params, scale=params.scale.new_empty(0), mma_packed=True
        )
        return QuantizedTensor(dest, weight._layout_cls, params)

    @classmethod
    def dequantize(cls, qdata: torch.Tensor, params: Params) -> torch.Tensor:
        s_rel = params.scale
        if params.mma_packed:
            n, k = params.orig_shape[::-1] if params.transposed else params.orig_shape
            qdata, s_rel = unpack_w4a8_mma_weight(qdata, n, k)
        return dequantize_w4a8_int8_weight(
            qdata,
            s_rel,
            params.s_channel,
            codebook=params.codebook,
            correction=params.correction,
            group_size=params.group_size,
            convrot_groupsize=params.convrot_groupsize,
            output_dtype=params.orig_dtype,
        )

    @classmethod
    def get_plain_tensors(
        cls,
        qtensor: QuantizedTensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        params = qtensor._params
        return (
            qtensor._qdata,
            params.scale,
            params.s_channel,
            params.correction,
            params.codebook,
        )

    @classmethod
    def linear_input_act(cls, weight: QuantizedTensor, x: torch.Tensor, bias=None,
                         input_act=None, input_act_weight=None, input_act_eps=0.0,
                         residual=None, residual_scale=None):
        params = weight._params
        if params.transposed:
            return NotImplemented
        qdata, s_rel, s_channel, correction, codebook = cls.get_plain_tensors(weight)
        return w4a8_int8_linear(
            x, qdata, s_rel, s_channel, codebook=codebook, correction=correction, bias=bias,
            group_size=params.group_size, convrot_groupsize=params.convrot_groupsize,
            out_dtype=params.orig_dtype, mma_packed=params.mma_packed,
            input_act=input_act, input_act_weight=input_act_weight, input_act_eps=input_act_eps,
            residual=residual, residual_scale=residual_scale,
        )

    @classmethod
    def state_dict_tensors(cls, qdata: torch.Tensor, params: Params) -> dict[str, torch.Tensor]:
        if params.mma_packed:
            raise ValueError("GEMV-optimized weights are resident-only; save the canonical layout")
        out = {
            "": qdata,
            "_s_rel": params.scale,
            "_s_channel": params.s_channel,
        }
        if params.correction is not None:
            out["_correction"] = params.correction
        if params.codebook is not None:
            out["_codebook"] = params.codebook
        return out

    @classmethod
    def bits(cls, qtensor: QuantizedTensor) -> int:
        """Code width implied by the packed row: K/2 bytes at 4 bits, 3K/4 at 6."""
        from comfy_kitchen.backends.eager.w4a8_int8 import _w4a8_geometry

        params = qtensor._params
        if params.mma_packed:
            n, k = params.orig_shape
            return _mma_bits(qtensor._qdata.numel(), n, k)
        return _w4a8_geometry(qtensor._qdata, params.scale, params.group_size)[2]

    @classmethod
    def requantize_kwargs(cls, qtensor: QuantizedTensor) -> dict[str, object]:
        params = qtensor._params
        return {
            "group_size": params.group_size,
            "convrot_groupsize": params.convrot_groupsize,
            "symmetric": params.correction is None,
            "codebook": params.codebook is not None,
            "codebook_tensor": params.codebook,
            "scale_dtype": params.scale.dtype,
            "bits": cls.bits(qtensor),
            # requantize reruns per layer under VRAM offload; the search is quantize-time only
            "scale_search": False,
        }


def try_relayout_w4a8_weights(weights: tuple[QuantizedTensor, ...], storage: torch.Tensor):
    """Pack canonical W4A8/W6A8 weights resident in ``storage`` into one MMA weight.

    Codes must be ordered and disjoint inside ``storage``. The combined records
    start at the first weight's codes and are followed by the codebook (W4) and
    concatenated channel scales; every other byte of the source weights in
    ``storage`` may be overwritten. Returns the combined weight and one row-slice
    view per input, or NotImplemented before any write if the group is unsupported.
    """
    if storage.dtype != torch.uint8 or storage.ndim != 1 or not storage.is_contiguous():
        raise ValueError("resident storage must be a contiguous 1D uint8 buffer")
    first = weights[0]._params
    k = first.orig_shape[1]
    if torch.version.hip is not None or k <= 0 or k % 256 != 0:
        return NotImplemented
    bits = weights[0]._qdata.shape[-1] * 8 // k
    if bits not in (4, 6):
        return NotImplemented
    default_codebook, checked_codebooks = default_w4a8_codebook(), set()
    total_n, end = 0, None
    for weight in weights:
        qdata, params = weight._qdata, weight._params
        n = params.orig_shape[0]
        if (
            params.mma_packed
            or params.transposed
            or params.correction is not None
            or params.group_size != 16
            or params.convrot_groupsize != first.convrot_groupsize
            or params.orig_dtype != first.orig_dtype
            or params.orig_shape[1] != k
            or n % 16 != 0
            or qdata.dtype != torch.int8
            or tuple(qdata.shape) != (n, k * bits // 8)
            or not qdata.is_contiguous()
            or params.scale.dtype != torch.float8_e4m3fn
            or params.s_channel.dtype != torch.float32
            or params.scale.device != storage.device
            or params.s_channel.device != storage.device
            or (params.codebook is None) != (bits == 6)
        ):
            return NotImplemented
        if params.codebook is not None and id(params.codebook) not in checked_codebooks:
            if not torch.equal(params.codebook.float().cpu(), default_codebook):
                return NotImplemented
            checked_codebooks.add(id(params.codebook))
        offset = qdata.data_ptr() - storage.data_ptr()
        if (
            qdata.device != storage.device
            or offset < (0 if end is None else end)
            or offset + qdata.nbytes > storage.nbytes
        ):
            raise ValueError("resident weight codes must be ordered and disjoint inside storage")
        if end is None:
            start = offset
        end = offset + qdata.nbytes
        total_n += n
    records_end = start + total_n * k * (2 * bits + 1) // 16
    meta_start = records_end + (-(storage.data_ptr() + records_end)) % 16
    meta_end = meta_start + 4 * (total_n + (16 if bits == 4 else 0))
    if meta_end > storage.nbytes:
        return NotImplemented
    try:
        pack = registry.get_implementation("pack_w4a8_mma_weights_", kwargs={"storage": storage})
    except NoCapableBackendError:
        return NotImplemented
    if torch.cuda.get_device_capability(storage.device) < (8, 0):
        return NotImplemented

    # Save everything the relayout may overwrite before the first write; all of it
    # is ordered on the current stream with the backend's moves and pack.
    s_channel = torch.cat([w._params.s_channel for w in weights])
    codebook = first.codebook.to(storage.device, torch.float32, copy=True) if bits == 4 else None
    records = pack(storage, [w._qdata for w in weights], [w._params.scale for w in weights])
    meta = storage[meta_start:meta_end].view(torch.float32)
    if codebook is not None:
        codebook = meta[:16].copy_(codebook)
        meta = meta[16:]
    s_channel = meta.copy_(s_channel)

    params = dataclasses.replace(
        first, scale=first.scale.new_empty(0), s_channel=s_channel, codebook=codebook,
        orig_shape=(total_n, k), mma_packed=True,
    )
    views, row = [], 0
    for weight in weights:
        n = weight._params.orig_shape[0]
        lo, hi = row * k * (2 * bits + 1) // 16, (row + n) * k * (2 * bits + 1) // 16
        view_params = dataclasses.replace(params, s_channel=s_channel[row:row + n], orig_shape=(n, k))
        views.append(QuantizedTensor(records[lo:hi], "AsymW4A8Int8Layout", view_params))
        row += n
    return QuantizedTensor(records, "AsymW4A8Int8Layout", params), tuple(views)


def _is_w4a8(weight: object) -> bool:
    return isinstance(weight, QuantizedTensor) and weight._layout_cls == "AsymW4A8Int8Layout"


def _resolve_w4a8_rhs(rhs: QuantizedTensor) -> QuantizedTensor:
    if not rhs._params.transposed:
        raise RuntimeError("W4A8 GEMM expects RHS W.T. Use F.linear(x, W) or mm(x, W.t()).")
    return rhs


def _w4a8_int8_forward(
    input_tensor: torch.Tensor,
    weight: QuantizedTensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    qdata, s_rel, s_channel, correction, codebook = AsymW4A8Int8Layout.get_plain_tensors(weight)
    params = weight._params
    return w4a8_int8_linear(
        input_tensor,
        qdata,
        s_rel,
        s_channel,
        codebook=codebook,
        correction=correction,
        bias=bias,
        group_size=params.group_size,
        convrot_groupsize=params.convrot_groupsize,
        out_dtype=out_dtype,
        mma_packed=params.mma_packed,
    )


@register_layout_op(torch.ops.aten.t.default, AsymW4A8Int8Layout)
def _handle_w4a8int8_t(qt, args, kwargs):
    input_tensor = args[0]
    if not isinstance(input_tensor, QuantizedTensor):
        return torch.ops.aten.t.default(*args, **kwargs)
    old = input_tensor._params
    new_params = dataclasses.replace(
        old,
        orig_shape=(old.orig_shape[1], old.orig_shape[0]),
        transposed=not old.transposed,
    )
    return QuantizedTensor(input_tensor._qdata, "AsymW4A8Int8Layout", new_params)


@register_layout_op(torch.ops.aten.linear.default, AsymW4A8Int8Layout)
def _handle_w4a8int8_linear(qt, args, kwargs):
    input_tensor, weight = args[0], args[1]
    bias = args[2] if len(args) > 2 else None
    if not _is_w4a8(weight):
        return torch.nn.functional.linear(*dequantize_args((input_tensor, weight, bias)))
    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()
    if weight._params.transposed:
        return torch.nn.functional.linear(input_tensor, weight.dequantize(), bias)
    out_dtype = kwargs.get("out_dtype", weight._params.orig_dtype)
    return _w4a8_int8_forward(input_tensor, weight, bias, out_dtype)


@register_layout_op(torch.ops.aten.mm.default, AsymW4A8Int8Layout)
def _handle_w4a8int8_mm(qt, args, kwargs):
    input_tensor, weight = args[0], args[1]
    if not _is_w4a8(weight):
        return torch.mm(*dequantize_args((input_tensor, weight)))
    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()
    weight = _resolve_w4a8_rhs(weight)
    return _w4a8_int8_forward(
        input_tensor,
        weight,
        bias=None,
        out_dtype=weight._params.orig_dtype,
    )


@register_layout_op(torch.ops.aten.addmm.default, AsymW4A8Int8Layout)
def _handle_w4a8int8_addmm(qt, args, kwargs):
    bias, input_tensor, weight = args[0], args[1], args[2]
    scaled = kwargs.get("beta", 1) != 1 or kwargs.get("alpha", 1) != 1
    matrix_addend = isinstance(bias, torch.Tensor) and bias.dim() != 1
    if not _is_w4a8(weight) or scaled or matrix_addend:
        return torch.addmm(*dequantize_args(args), **dequantize_args(kwargs))
    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()
    weight = _resolve_w4a8_rhs(weight)
    return _w4a8_int8_forward(
        input_tensor,
        weight,
        bias=bias,
        out_dtype=weight._params.orig_dtype,
    )
